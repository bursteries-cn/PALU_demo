"""CPU checks for diagnostics that must not alter the training trajectory."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/palu"), str(Path(__file__).parent)]
from seed_epoch_experiment import configured_run, protocol_config


class ConfigChecks(unittest.TestCase):
    def test_seed_protocol_preserves_objective_and_budget(self):
        source = {"trainer": {"handler": "PALU", "method_args": {"alpha": 2., "first_n": 3,
            "top_k": 5000}, "args": {"num_train_epochs": 10, "learning_rate": 2e-5,
            "per_device_train_batch_size": 8, "gradient_accumulation_steps": 4,
            "warmup_steps": 6, "seed": 0}}, "eval": {"tofu": {"metrics": {
                "forget_quality": {}, "model_utility": {}}}}}
        with tempfile.TemporaryDirectory() as temporary:
            result = configured_run(source, Path(temporary), 100)
            another = configured_run(source, Path(temporary) / "other", 101)
        self.assertEqual(result["trainer"]["method_args"]["alpha"], 2.)
        self.assertEqual(result["trainer"]["args"]["warmup_steps"], 6)
        self.assertEqual(result["trainer"]["args"]["num_train_epochs"], 10)
        self.assertEqual(result["trainer"]["args"]["data_seed"], 100)
        self.assertEqual(result["trainer"]["args"]["eval_strategy"], "epoch")
        self.assertEqual(result["trainer"]["args"]["report_to"], "none")
        self.assertEqual(source["trainer"]["args"]["seed"], 0)
        self.assertEqual(protocol_config(result), protocol_config(another))


HAS_TRAINING_STACK = all(importlib.util.find_spec(package) for package in (
    "torch", "transformers", "omegaconf", "tokenizers", "scipy"))


@unittest.skipUnless(HAS_TRAINING_STACK, "CPU training-stack dependencies are unavailable")
class TrajectoryChecks(unittest.TestCase):
    def test_epoch_evaluation_preserves_weights_rng_and_real_epoch_boundaries(self):
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from transformers import TrainingArguments
        from test_diagnostics import fixture
        from trainer.palu_epoch_eval import PALUEpochTrainer, target_counts
        from trainer.unlearn.palu import PALU

        class NoisyEvaluator:
            def __init__(self, reference_path):
                self.eval_cfg = OmegaConf.create({"metrics": {"forget_quality": {
                    "reference_logs": {"retain_model_logs": {"path": str(reference_path)}}}}})

            def evaluate(self, model, output_dir, **kwargs):
                # Deliberately perturb every RNG and modes to test restoration.
                model.eval()
                random.random(), np.random.rand(), torch.rand(4)
                scores = {"forget_truth_ratio": {"value_by_index": {"0": {"score": 1.}, "1": {"score": 2.}}}}
                Path(output_dir, "TOFU_EVAL.json").write_text(json.dumps(scores))
                return {"forget_quality": 1., "model_utility": .5}

        initial, tokenizer, dataset, collator = fixture()
        # Dataset indices already exist in real QADataset/QAwithLabelDataset.
        for split in ("forget", "retain"):
            for index, row in enumerate(getattr(dataset, split)):
                row["index"] = index
        states = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.json"
            reference.write_text(json.dumps({"forget_truth_ratio": {"value_by_index": {
                "0": {"score": 1.}, "1": {"score": 2.}}}}))
            for measured in (0, 1, 2):
                model = copy.deepcopy(initial)
                args = TrainingArguments(output_dir=str(root / str(measured)), use_cpu=True,
                    per_device_train_batch_size=1, gradient_accumulation_steps=2,
                    num_train_epochs=2, learning_rate=1e-3, optim="adamw_torch", report_to=[],
                    save_strategy="no", eval_strategy="epoch" if measured else "no",
                    eval_on_start=measured, logging_strategy="no", remove_unused_columns=False,
                    disable_tqdm=True, seed=11, data_seed=11, dataloader_num_workers=0)
                kwargs = dict(model=model, args=args, train_dataset=dataset, data_collator=collator,
                              tokenizer=tokenizer, top_k=3, first_n=1, alpha=.2,
                              track_loss_components=True)
                trainer = PALUEpochTrainer(**kwargs, evaluators={"tofu": NoisyEvaluator(reference)},
                    epoch_options={"eval_every_epochs": measured}) if measured else PALU(**kwargs)
                trainer.train()
                states.append(copy.deepcopy(model.state_dict()))
                if measured:
                    run = root / str(measured)
                    rows = [json.loads(line) for line in (run / "epoch_history.jsonl").read_text().splitlines()]
                    self.assertEqual([row["epoch_index"] for row in rows], [0, 1, 2] if measured == 1 else [0, 2])
                    self.assertEqual([row["step"] for row in rows], [0, 2, 4] if measured == 1 else [0, 4])
                    self.assertTrue(all(row["forget_count"] == 4 and row["retain_count"] == 3 for row in rows))
                    self.assertTrue(all(not row["partial_epoch"] for row in rows))
                    pairs = [json.loads(line) for line in (run / "training_pairs.jsonl").read_text().splitlines()]
                    self.assertEqual(len(pairs), 8)
                    self.assertEqual([sum(pair["epoch_index"] == epoch for pair in pairs) for epoch in (1, 2)], [4, 4])
                    self.assertEqual(len((run / "loss_history.jsonl").read_text().splitlines()), 4)
                    self.assertTrue(model.training)
                    self.assertFalse(trainer.ref_model.training)
                    self.assertEqual(target_counts([-100, 4, 5, -200, 6, 2], [-100, 4, 2], 1, 2), (2, 1, 2))
            for key, expected in states[0].items():
                torch.testing.assert_close(expected, states[1][key], rtol=0, atol=0)
                torch.testing.assert_close(expected, states[2][key], rtol=0, atol=0)

    def test_final_cutoff_is_recorded_as_actual_partial_epoch(self):
        import torch
        from omegaconf import OmegaConf
        from transformers import TrainingArguments
        from test_diagnostics import fixture
        from trainer.palu_epoch_eval import PALUEpochTrainer

        class Evaluator:
            def __init__(self, reference):
                self.eval_cfg = OmegaConf.create({"metrics": {"forget_quality": {
                    "reference_logs": {"retain_model_logs": {"path": str(reference)}}}}})

            def evaluate(self, model, output_dir, **kwargs):
                Path(output_dir, "TOFU_EVAL.json").write_text(json.dumps({
                    "forget_truth_ratio": {"value_by_index": {"0": {"score": 1.}}}}))
                return {"forget_quality": 1., "model_utility": .5}

        initial, tokenizer, dataset, collator = fixture()
        for split in ("forget", "retain"):
            for index, row in enumerate(getattr(dataset, split)):
                row["index"] = index
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.json"
            reference.write_text(json.dumps({"forget_truth_ratio": {"value_by_index": {"0": {"score": 1.}}}}))
            args = TrainingArguments(output_dir=str(root / "run"), use_cpu=True,
                per_device_train_batch_size=1, gradient_accumulation_steps=2, max_steps=3,
                learning_rate=1e-3, optim="adamw_torch", report_to=[], save_strategy="no",
                eval_strategy="epoch", eval_on_start=True, logging_strategy="no",
                remove_unused_columns=False, disable_tqdm=True, seed=11, data_seed=11)
            trainer = PALUEpochTrainer(model=initial, tokenizer=tokenizer, train_dataset=dataset,
                data_collator=collator, args=args, top_k=3, first_n=1, alpha=.2,
                evaluators={"tofu": Evaluator(reference)}, epoch_options={"eval_every_epochs": 3})
            trainer.train()
            trainer.evaluate()  # The post-train fallback must not duplicate the endpoint.
            rows = [json.loads(line) for line in (root / "run" / "epoch_history.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch_index"] for row in rows], [0, 2])
            self.assertEqual([row["step"] for row in rows], [0, 3])
            self.assertEqual(rows[-1]["actual_epoch_microbatches"], 2)
            self.assertEqual(rows[-1]["expected_epoch_microbatches"], 4)
            self.assertTrue(rows[-1]["partial_epoch"])
            self.assertEqual(rows[-1]["evaluation_kind"], "final_tail")
            pairs = [json.loads(line) for line in (root / "run" / "training_pairs.jsonl").read_text().splitlines()]
            prepared = [json.loads(line) for line in (root / "run" / "prepared_pairs.jsonl").read_text().splitlines()]
            self.assertEqual(len(pairs), 6)
            self.assertGreaterEqual(len(prepared), len(pairs))
            self.assertEqual(rows[-1]["training_examples_this_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
