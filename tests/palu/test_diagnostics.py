import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, TrainingArguments

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/palu")]
from data.collators import DataCollatorForSupervisedDataset
from data.unlearn import ForgetRetainDataset
from trainer.unlearn.palu import PALU
from trainer.palu_diagnostics import (answer_prediction_positions, capture_pre_head, preserve_rng_and_modes,
    representation_distances, sha256_file, tokenizer_fingerprint)
from prepare_diagnostics import fixed_samples
from plot_history import LOSSES, aggregate, mean_interval, read_history, render
from run_experiment import build_commands, write_json
from test_experiment_protocol import settings, compose_training


def fixture():
    torch.manual_seed(24)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, pad_token_id=0, eos_token_id=2, attention_dropout=.15))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}, unk_token="[UNK]")),
        pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    forget = [{"input_ids": torch.tensor([1, 3 + i, 4, 5, 2]),
               "labels": torch.tensor([-100, -200, 4, -200, 2])} for i in range(4)]
    retain = [{"input_ids": torch.tensor([1, 6 + i, 4, 5, 2]),
               "labels": torch.tensor([-100, 6 + i, 4, 5, 2])} for i in range(3)]
    collator = DataCollatorForSupervisedDataset(tokenizer)
    return model, tokenizer, ForgetRetainDataset(forget, retain), collator


def cache_fixture(path, model, tokenizer, dataset, collator):
    teacher = copy.deepcopy(model).eval()
    samples = {}
    for split, count in (("forget", 3), ("retain", 2)):
        samples[split], _ = fixed_samples(getattr(dataset, split), collator, count, 42, tokenizer.eos_token_id)
        with torch.no_grad():
            for sample in samples[split]:
                hidden = teacher.model(input_ids=sample["input_ids"][None],
                    attention_mask=sample["attention_mask"][None], use_cache=False).last_hidden_state
                sample["reference_hidden"] = hidden[0, sample["prediction_index"]].clone()
    torch.save({"metadata": {"schema": 1, "position": "predict_first_answer",
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "architecture": {key: getattr(model.config, key) for key in ("model_type", "hidden_size", "vocab_size")}},
        "samples": samples}, path)
    return {"cache_path": str(path), "cache_sha256": sha256_file(path), "every": 1, "temperature": 1.}


class DiagnosticTests(unittest.TestCase):
    def test_first_answer_includes_common_tokens_and_handles_padding(self):
        labels = torch.tensor([[-100, -100, -200, 4, -100], [-100, -100, -100, -200, 4]])
        mask = torch.tensor([[1, 1, 1, 1, 0], [0, 1, 1, 1, 1]])
        ids = torch.tensor([[1, 3, 4, 5, 0], [0, 1, 3, 4, 5]])
        self.assertEqual(answer_prediction_positions(labels, mask, ids).tolist(), [1, 2])
        with self.assertRaisesRegex(ValueError, "No answer"):
            answer_prediction_positions(labels.fill_(-100), mask, ids)
        with self.assertRaisesRegex(ValueError, "preceding prompt"):
            answer_prediction_positions(torch.tensor([[4, 5]]), torch.ones(1, 2), torch.tensor([[4, 5]]))

    def test_kl_direction_feature_sum_and_temperature(self):
        ref, cur = torch.tensor([[0., 2., -2.], [1., 0., .5]]), torch.tensor([[-2., .5, 2.], [.4, -.3, .2]])
        kl, _ = representation_distances(ref, cur)
        expected = torch.nn.functional.kl_div(cur.log_softmax(-1), ref.log_softmax(-1), log_target=True, reduction="batchmean")
        torch.testing.assert_close(kl.mean(), expected)
        self.assertFalse(torch.allclose(kl, representation_distances(cur, ref)[0]))
        torch.testing.assert_close(representation_distances(ref, ref)[0], torch.zeros(2))
        with self.assertRaises(ValueError):
            representation_distances(ref, cur, 0)

    def test_capture_is_after_final_norm_and_before_head(self):
        model, _, dataset, collator = fixture()
        model.eval()
        batch = collator([dataset.forget[0]])
        position = answer_prediction_positions(batch["labels"], batch["attention_mask"], batch["input_ids"])
        with torch.no_grad(), capture_pre_head(model, [position]) as captured:
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        expected = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state[:, 0]
        torch.testing.assert_close(captured[0], expected)
        self.assertFalse(captured[0].requires_grad)

    def test_probe_preserves_rng_and_each_module_mode(self):
        module = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Dropout())
        module.train()
        module[1].eval()
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with preserve_rng_and_modes([module], torch.device("cpu")):
            self.assertFalse(module.training)
            random.random(), np.random.rand(), torch.rand(4)
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        torch.testing.assert_close(torch.get_rng_state(), before[2])
        self.assertTrue(module.training)
        self.assertFalse(module[1].training)

    def test_diagnostics_leave_optimizer_updates_identical_with_random_retain_and_dropout(self):
        initial, tokenizer, dataset, collator = fixture()
        states = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = cache_fixture(root / "reference.pt", initial, tokenizer, dataset, collator)
            options["every"] = 2
            for enabled in (False, True):
                model = copy.deepcopy(initial)
                args = TrainingArguments(output_dir=str(root / str(enabled)), use_cpu=True,
                    per_device_train_batch_size=1, gradient_accumulation_steps=2, max_steps=3,
                    learning_rate=1e-3, optim="adamw_torch", report_to=[], save_strategy="no",
                    eval_strategy="no", logging_strategy="no", remove_unused_columns=False,
                    disable_tqdm=True, seed=11, data_seed=11)
                trainer = PALU(model=model, args=args, train_dataset=dataset, data_collator=collator,
                    tokenizer=tokenizer, top_k=3, first_n=1, alpha=.2,
                    track_loss_components=True, diagnostics=options if enabled else None)
                trainer.train()
                states.append(copy.deepcopy(model.state_dict()))
                history = read_history(root / str(enabled) / "loss_history.jsonl")
                self.assertEqual(list(history), [1, 2, 3])
                self.assertTrue(all(row["microbatches_all_ranks"] == 2 for row in history.values()))
                if enabled:
                    probes = read_history(root / str(enabled) / "probe_history.jsonl")
                    self.assertEqual(list(probes), [0, 2, 3])
                    self.assertTrue(all(row["forget_count"] == 3 and row["retain_count"] == 2 for row in probes.values()))
                    self.assertAlmostEqual(probes[0]["forget_repr_kl"], 0., places=6)
                    records = (root / str(enabled) / "probe_samples.jsonl").read_text().splitlines()
                    self.assertEqual(len(records), 15)
                    self.assertTrue(all(row["included_in_trainer_runtime"] for row in probes.values()))
                self.assertTrue(all(parameter.grad is None for parameter in trainer.ref_model.parameters()))
            self.assertTrue(all(torch.equal(value, states[1][key]) for key, value in states[0].items()))

    def test_corrupt_cache_is_rejected(self):
        initial, tokenizer, dataset, collator = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = cache_fixture(root / "reference.pt", initial, tokenizer, dataset, collator)
            with (root / "reference.pt").open("ab") as stream:
                stream.write(b"changed")
            trainer = PALU(model=initial, tokenizer=tokenizer, train_dataset=dataset, data_collator=collator,
                args=TrainingArguments(output_dir=str(root / "run"), use_cpu=True, max_steps=1, report_to=[],
                    remove_unused_columns=False, disable_tqdm=True), diagnostics=options)
            with self.assertRaisesRegex(ValueError, "cache hash"):
                trainer.train()


class PlotTests(unittest.TestCase):
    def test_student_t_interval_and_no_invented_single_seed_ci(self):
        interval = mean_interval([1., 2., 3.])
        self.assertAlmostEqual(interval["mean"], 2.)
        self.assertAlmostEqual(interval["ci95_half_width"], 2.4841377, places=6)
        self.assertIsNone(mean_interval([1.])["ci95_half_width"])

    def test_missing_steps_use_only_observed_seeds_without_interpolation(self):
        histories = {0: {1: {"loss_total": 1.}, 2: {"loss_total": 2.}},
                     1: {1: {"loss_total": 3.}}}
        rows = aggregate(histories, [0, 1], "training", LOSSES[:1])
        self.assertEqual((rows[0]["mean"], rows[0]["n"], rows[0]["all_seeds_present"]), (2., 2, True))
        self.assertEqual((rows[1]["mean"], rows[1]["n"], rows[1]["all_seeds_present"]), (2., 1, False))
        self.assertIsNone(rows[1]["ci95_half_width"])

    def test_four_curves_export_png_pdf_and_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            for seed in (0, 1, 2):
                run = root / f"seed_{seed}"
                run.mkdir(parents=True)
                runs.append({"seed": seed, "output_dir": str(run)})
                rows = [{"step": step, "loss_total": 2 + seed * .1,
                    "loss_forget": 1 + seed * .1, "loss_retain_nll": .7, "loss_common_kl": .3,
                    "forget_repr_kl": .5 + seed * .1, "retain_repr_kl": .2} for step in (1, 2)]
                text = "".join(json.dumps(row) + "\n" for row in rows)
                (run / "loss_history.jsonl").write_text(text)
                (run / "probe_history.jsonl").write_text(text)
            write_json(root / "manifest.json", {"seeds": [0, 1, 2], "runs": runs})
            output = render(root)
            for name in ("loss_curves.png", "loss_curves.pdf", "loss_vs_retain_kl.png", "loss_vs_retain_kl.pdf", "curve_statistics.csv"):
                self.assertGreater((output / name).stat().st_size, 100)


class LauncherTests(unittest.TestCase):
    def test_bash_defaults_and_quoted_paths_compose_real_config(self):
        import os
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "中文 plan with spaces"
            env = {**os.environ, "PYTHON_BIN": sys.executable, "OUTPUT_DIR": str(output),
                   "DATASET_PATH": "/tmp/data with spaces", "RETAIN_MODEL_PATH": "/tmp/retain with spaces"}
            process = subprocess.run(["bash", str(ROOT / "scripts/palu/run.sh"), "--dry-run"],
                cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["seeds"], [0, 1, 2])
            self.assertFalse(manifest["settings"]["keep_model"])
            self.assertIn("reference_precompute_argv", manifest)
            args = settings()
            args.output, args.retain_model_path = output, "/tmp/retain with spaces"
            args.probe_every, args.probe_temperature = 5, 1.
            args.probe_cache_sha256 = "a" * 64
            config = compose_training(args, Path("/tmp/seed_42"))
            self.assertEqual(config["trainer"]["method_args"]["diagnostics"]["cache_sha256"], "a" * 64)
            self.assertEqual(config["trainer"]["method_args"]["diagnostics"]["cache_path"], str(output / "reference_cache.pt"))
            self.assertTrue(config["trainer"]["method_args"]["track_loss_components"])
            self.assertEqual(len(manifest["runs"]), 3)


if __name__ == "__main__":
    unittest.main()
