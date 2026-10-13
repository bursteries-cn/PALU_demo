import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from efficient_soup.search import efficient_search, deviation
from efficient_soup.weights import WeightMixer
from efficient_soup.proxy import extraction_strength, evaluate_es, prepare_split

spec = importlib.util.spec_from_file_location("soup_cli", ROOT / "scripts/palu/efficient_soup.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def fixture(root, tied=False):
    tokenizer_object = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "Question": 2, ":": 3,
        "Answer": 4, "alpha": 5, "beta": 6, "gamma": 7}, unk_token="[UNK]"))
    tokenizer_object.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object, unk_token="[UNK]",
                                       eos_token="[EOS]", pad_token="[EOS]")
    model = LlamaForCausalLM(LlamaConfig(vocab_size=8, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        eos_token_id=1, pad_token_id=1, tie_word_embeddings=tied))
    paths = [root / name for name in ("full", "seed0", "seed1")]
    for i, path in enumerate(paths):
        if i:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.003 * i)
        model.save_pretrained(path, safe_serialization=True, max_shard_size="1KB")
        tokenizer.save_pretrained(path)
    return model, tokenizer, paths


class EfficientSoupTests(unittest.TestCase):
    def test_real_palu_provenance_and_eval_configuration(self):
        from test_experiment_protocol import settings, compose_training
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / str(i) for i in range(2)]
            for i, path in enumerate(paths):
                path.mkdir()
                config = compose_training(settings(), path, i)
                (path / "resolved_config.json").write_text(json.dumps(config))
            records = cli.candidate_provenance(paths, original=settings().model_path, forget_split="forget05")
            self.assertEqual([r["seed"] for r in records], [0, 1])
            with self.assertRaisesRegex(ValueError, "initialization"):
                cli.candidate_provenance(paths, original="wrong/full")
            with self.assertRaisesRegex(ValueError, "Forget"):
                cli.candidate_provenance(paths, forget_split="forget10")
            with self.assertRaisesRegex(ValueError, "template"):
                cli.candidate_provenance(paths, template={"wrong": True})
            args = cli.parse_args(["--original", "unused", "--candidates", *map(str, paths),
                "--output", str(Path(directory) / "output"), "--dataset", "/tmp/data with spaces",
                "--retain-logs", "/tmp/retain.json", "--classifier-model", "/tmp/classifier", "--device", "cpu"])
            command = cli.full_eval_commands(args, {"soup": paths[0]})["soup"]
            with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
                cfg = compose(config_name="eval", overrides=command[2:])
                self.assertTrue(cfg.eval.tofu.overwrite)
                self.assertEqual(cfg.model.model_args.pretrained_model_name_or_path, str(paths[0]))
                self.assertEqual(cfg.model.model_args.device_map, "cpu")
                self.assertEqual(cfg.eval.tofu.metrics.forget_Q_A_gibberish.device, "cpu")
                self.assertEqual(cfg.tofu_dataset_path, "/tmp/data with spaces")

    def test_launcher_dry_run_keeps_both_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "plan with spaces"
            env = {**os.environ, "PYTHON_BIN": sys.executable, "SOUP_RUN_DIR": str(root)}
            execution = subprocess.run(["bash", str(ROOT / "scripts/palu/run_efficient_soup.sh"), "--dry-run"],
                                       env=env, capture_output=True, text=True)
            self.assertEqual(execution.returncode, 0, execution.stdout + execution.stderr)
            training = json.loads((root / "candidates/manifest.json").read_text())
            soup = json.loads((root / "soup_search/manifest.json").read_text())
            self.assertTrue(training["settings"]["keep_model"])
            self.assertEqual(training["seeds"], [0, 1])
            self.assertEqual(soup["status"], "planned")
            self.assertEqual(soup["settings"]["original"], training["settings"]["model_path"])

    def test_es_suffix_boundaries(self):
        labels = torch.tensor([1, 2, 3])
        for prediction, expected in (([1, 2, 3], 1), ([0, 2, 3], 2/3), ([1, 0, 3], 1/3),
                                     ([1, 2, 0], 0), ([0, 0, 0], 0)):
            self.assertAlmostEqual(extraction_strength(torch.tensor(prediction), labels), expected)
        with self.assertRaises(ValueError):
            extraction_strength(torch.tensor([]), torch.tensor([]))

    def test_teacher_forcing_matches_greedy_suffix(self):
        torch.manual_seed(7)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=8, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)).eval()
        ids = torch.tensor([[1, 3, 4, 2, 5]])
        predictions = model(ids).logits[0, 1:-1].argmax(-1)
        score = extraction_strength(predictions, ids[0, 2:])
        expected_prefix = 3
        for prefix in range(4):
            generated = ids[:, :2 + prefix].clone()
            for _ in range(3 - prefix):
                token = model(generated).logits[:, -1].argmax(-1, keepdim=True)
                generated = torch.cat((generated, token), dim=1)
            if torch.equal(generated, ids):
                expected_prefix = prefix
                break
        self.assertAlmostEqual(score, 1 - expected_prefix / 3)

    def test_search_budget_and_incumbent(self):
        seen = []
        def evaluate(weights, stage):
            seen.append(weights)
            return {"ds_es": (weights[0] - .4)**2 + (weights[1] - .4)**2}
        result = efficient_search(evaluate, depth=3)
        self.assertLessEqual(len(seen), 9)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertLessEqual(result["final"]["ds_es"], result["best_single"]["ds_es"])
        self.assertLessEqual(result["final"]["ds_es"], result["stage_one"]["ds_es"])
        self.assertTrue(all(abs(sum(w) - 1) < 1e-12 and min(w) >= 0 for w in seen))
        self.assertEqual(efficient_search(lambda w, s: {"ds_es": 1}, 2)["final"]["weights"], [1., 0., 0.])
        with self.assertRaises(ValueError):
            efficient_search(lambda w, s: {"ds_es": float("nan")})
        self.assertAlmostEqual(deviation(.3, .6), .5)

    def test_streaming_shards_tied_weights_and_no_accumulation(self):
        for tied in (False, True):
            with self.subTest(tied=tied), tempfile.TemporaryDirectory() as directory:
                model, _, paths = fixture(Path(directory), tied=tied)
                states = [LlamaForCausalLM.from_pretrained(p).state_dict() for p in paths]
                mixer = WeightMixer(model, paths)
                try:
                    for weights in ((.25, .25, .5), (0, 1, 0), (1, 0, 0)):
                        mixer.apply(weights)
                        for key, value in model.state_dict().items():
                            expected = sum(state[key].float() * w for state, w in zip(states, weights))
                            torch.testing.assert_close(value, expected)
                    with self.assertRaises(ValueError):
                        mixer.apply((1, 1, 1))
                finally:
                    mixer.close()

    def test_missing_weights_and_nonfloating_buffers_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / str(i) for i in range(3)]
            model = torch.nn.Module()
            model.register_buffer("counter", torch.tensor(1))
            for i, path in enumerate(paths):
                path.mkdir()
                save_file({"counter": torch.tensor(i)}, str(path / "model.safetensors"))
            mixer = WeightMixer(model, paths)
            try:
                with self.assertRaisesRegex(ValueError, "Nonfloating"):
                    mixer.apply((1, 0, 0))
            finally:
                mixer.close()
            (paths[1] / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "KEEP_MODEL"):
                WeightMixer(model, paths)

    def test_pad_equals_eos_is_attended(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
            def forward(self, input_ids, attention_mask, **kwargs):
                self.mask = attention_mask
                logits = torch.zeros(*input_ids.shape, 3)
                logits[:, :, 1] = 1
                return type("Output", (), {"logits": logits})
        model = Model()
        items = [{"input_ids": torch.tensor(x), "labels": torch.tensor([-100] + [1]*(len(x)-1)),
                  "attention_mask": torch.ones(len(x), dtype=torch.long)} for x in ([2, 1, 1], [2, 1])]
        result = evaluate_es(model, items, 2, 1)
        self.assertEqual(model.mask.tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(result["mean"], 1)

    def test_offline_cli_save_reload_and_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, tokenizer, paths = fixture(root)
            data = root / "data"
            data.mkdir()
            for split in ("forget05_perturbed", "retain_perturbed"):
                rows = [{"question": f"{split} {i}", "answer": "alpha beta gamma"} for i in range(6)]
                # Duplicate questions must never straddle search and held-out sets.
                rows.append(rows[0])
                (data / f"{split}.json").write_text(json.dumps(rows))
            output = root / "result"
            command = [sys.executable, str(ROOT / "scripts/palu/efficient_soup.py"), "--original", str(paths[0]),
                "--candidates", *map(str, paths[1:]), "--output", str(output), "--dataset", str(data),
                "--model", "Llama-2-7b-hf", "--proxy-questions", "2", "--depth", "2",
                "--device", "cpu", "--dtype", "float32", "--batch-size", "2"]
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            template = {"apply_chat_template": False, "user_start_tag": "Question: ", "user_end_tag": "\n",
                        "asst_start_tag": "Answer: ", "asst_end_tag": "\n\n"}
            with self.assertRaisesRegex(ValueError, "max_length"):
                prepare_split(str(data), "forget05_perturbed", tokenizer, template, 3, 2, 42)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertLessEqual(manifest["proxy_evaluations"], 7)
            for metadata in json.loads((output / "samples.json").read_text()).values():
                self.assertFalse(set(metadata["proxy_indices"]) & set(metadata["heldout_indices"]))
                self.assertEqual(0 in metadata["proxy_indices"], 6 in metadata["proxy_indices"])
            selection = json.loads((output / "selection.json").read_text())
            saved = LlamaForCausalLM.from_pretrained(output / "soup")
            mixer = WeightMixer(saved, paths)
            before = {k: v.clone() for k, v in saved.state_dict().items()}
            try:
                mixer.apply(selection["final"]["weights"])
                for key, value in saved.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            finally:
                mixer.close()
            # Existing outputs must remain untouched on accidental repeat.
            repeated = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "completed")


if __name__ == "__main__":
    unittest.main()
