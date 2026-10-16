"""Retention safety and interrupted-evaluation recovery without models or GPUs."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/palu"))
import checkpoint_retention as retention
from layout_loss_curve_experiment import storage_budget


class RetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        (self.run / "evaluation_workspace").mkdir()
        self.manifest = {"schedule_sha256": "test-inputs", "schedule_metadata": {
            "data_sizes": {"forget": 2, "retain": 3}, "fixed_dataset_sha256": "test-fixed",
            "fixed_target_counts": {"sensitive": 3, "common": 2, "retain": 6},
            "seed103_source_provenance": {"reference_sha256": "test-reference"}}, "checkpoints": []}
        self.samples = [
            {"split": "forget", "dataset_index": 0, "raw_sensitive_loss": 1.0, "sensitive_token_count": 1, "raw_common_kl": 0.2, "common_token_count": 1},
            {"split": "forget", "dataset_index": 1, "raw_sensitive_loss": 3.0, "sensitive_token_count": 2, "raw_common_kl": 0.4, "common_token_count": 1},
            *[{"split": "retain", "dataset_index": i, "raw_retain_nll": (i + 1) / 10,
               "retain_token_count": i + 1} for i in range(3)]]
        for step in (6, 60):
            folder = self.run / f"checkpoint-{step}"
            folder.mkdir()
            inventory = {}
            for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json"):
                data = b"CPU test weight stub, not a model"
                (folder / name).write_bytes(data)
                inventory[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            (folder / "config.json").write_text("{}")
            checkpoint = {"step": step, "checkpoint": str(folder), "parameter_sha256": f"test-weights-{step}", "files": inventory}
            retention.atomic_json(folder / "checkpoint_provenance.json", checkpoint)
            self.manifest["checkpoints"].append(checkpoint)
        self.checkpoint = self.manifest["checkpoints"][0]

    def inputs(self, step=6):
        source = self.root / f"source-{step}"
        source.mkdir(exist_ok=True)
        epoch = {"step": step, "fixed_dataset_sha256": "test-fixed", "reference_sha256": "test-reference",
                 "checkpoint_parameter_sha256": f"test-weights-{step}", "training_schedule_sha256": "test-inputs",
                 "forget_quality": 0.5, "model_utility": 0.7, "ks_d": 0.1,
                 "raw_sensitive_loss": 7 / 3, "raw_common_kl": 0.3, "raw_retain_nll": 1.4 / 6}
        sizes = {"forget_truth_ratio": 2, "forget_Q_A_PARA_Prob": 2, "forget_Q_A_PERT_Prob": 2}
        for prefix, size, suffixes in (
            ("retain", 400, ("Q_A_Prob", "Q_A_PARA_Prob", "Q_A_PERT_Prob", "Q_A_ROUGE", "Truth_Ratio")),
            ("ra", 100, ("Q_A_Prob", "Q_A_PERT_Prob", "Q_A_Prob_normalised", "Q_A_ROUGE", "Truth_Ratio")),
            ("wf", 117, ("Q_A_Prob", "Q_A_PERT_Prob", "Q_A_Prob_normalised", "Q_A_ROUGE", "Truth_Ratio"))):
            sizes.update({prefix + "_" + suffix: size for suffix in suffixes})
        native = {key: {"value_by_index": {str(i): {"score": 0.1} for i in range(size)}} for key, size in sizes.items()}
        native.update({key: {"agg_value": epoch[key]} for key in ("forget_quality", "model_utility")})
        retention.atomic_json(source / "TOFU_EVAL.json", native)
        fixed = self.root / f"samples-{step}.jsonl"
        fixed.write_text("".join(json.dumps(dict(row, step=step)) + "\n" for row in self.samples))
        return epoch, source, fixed

    def commit(self):
        return retention.commit_observation(self.run, self.manifest, self.checkpoint, *self.inputs())

    def test_delete_only_weights_after_complete_evaluation(self):
        row = self.commit()
        retention.rebuild_aggregates(self.run, [row])
        result = retention.prune_intermediate(self.run, self.manifest, self.checkpoint)
        self.assertEqual(result["status"], "deleted")
        folder = self.run / "checkpoint-6"
        self.assertFalse(list(folder.glob("*.safetensors")))
        self.assertFalse((folder / "model.safetensors.index.json").exists())
        self.assertTrue((folder / "evals/TOFU_EVAL.json").is_file())
        self.assertTrue((folder / "config.json").is_file())
        self.assertEqual(len((self.run / "fixed_loss_samples.jsonl").read_text().splitlines()), 5)
        self.assertTrue((folder / "checkpoint_provenance.json").is_file())

    def test_final_and_external_initial_full_are_never_deleted(self):
        final = self.manifest["checkpoints"][-1]
        retention.prune_intermediate(self.run, self.manifest, final)
        self.assertTrue((self.run / "checkpoint-60/model-00001-of-00002.safetensors").exists())
        outside = self.root / "Full.safetensors"
        outside.write_bytes(b"protected Full")
        retention.prune_intermediate(self.run, self.manifest, {"step": 0, "checkpoint": str(outside)})
        self.assertTrue(outside.exists())

    def test_no_receipt_prevents_deletion(self):
        with self.assertRaisesRegex(ValueError, "No validated"):
            retention.prune_intermediate(self.run, self.manifest, self.checkpoint)
        self.assertTrue((self.run / "checkpoint-6/model-00001-of-00002.safetensors").exists())

    def test_incomplete_fixed_samples_prevent_commit_and_deletion(self):
        epoch, source, fixed = self.inputs()
        fixed.write_text("\n".join(fixed.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "Incomplete fixed"):
            retention.commit_observation(self.run, self.manifest, self.checkpoint, epoch, source, fixed)
        self.assertIsNone(retention.read_receipt(self.run, self.manifest, self.checkpoint))
        self.assertTrue((self.run / "checkpoint-6/model-00001-of-00002.safetensors").exists())

    def test_missing_native_per_example_metric_prevents_commit(self):
        epoch, source, fixed = self.inputs()
        native = json.loads((source / "TOFU_EVAL.json").read_text())
        del native["wf_Q_A_ROUGE"]
        retention.atomic_json(source / "TOFU_EVAL.json", native)
        with self.assertRaisesRegex(ValueError, "Incomplete native"):
            retention.commit_observation(self.run, self.manifest, self.checkpoint, epoch, source, fixed)

    def test_tampered_artifact_prevents_deletion(self):
        self.commit()
        (self.run / "checkpoint-6/evals/TOFU_EVAL.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            retention.prune_intermediate(self.run, self.manifest, self.checkpoint)
        self.assertTrue((self.run / "checkpoint-6/model-00001-of-00002.safetensors").exists())

    def test_recovery_uses_committed_observations_without_deleted_weights(self):
        row = self.commit()
        retention.prune_intermediate(self.run, self.manifest, self.checkpoint)
        shutil.rmtree(self.root / "source-6")
        for name in ("epoch_history.jsonl", "fixed_loss_samples.jsonl"):
            (self.run / name).unlink(missing_ok=True)
        recovered = retention.read_receipt(self.run, self.manifest, self.checkpoint)
        self.assertEqual(recovered, row)
        retention.rebuild_aggregates(self.run, [recovered])
        self.assertEqual(len((self.run / "fixed_loss_samples.jsonl").read_text().splitlines()), 5)
        retention.prune_intermediate(self.run, self.manifest, self.checkpoint)

    def test_partial_cleanup_and_symlink_guards(self):
        self.commit()
        folder = self.run / "checkpoint-6"
        (folder / "model-00001-of-00002.safetensors").unlink()
        retention.prune_intermediate(self.run, self.manifest, self.checkpoint)
        self.assertFalse(list(folder.glob("*.safetensors")))
        outside = self.root / "external-model.safetensors"
        outside.write_bytes(b"protected")
        (folder / "model-00001-of-00002.safetensors").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "non-regular"):
            retention.prune_intermediate(self.run, self.manifest, self.checkpoint)
        self.assertEqual(outside.read_bytes(), b"protected")

    def test_online_latest_snapshot_is_deleted_before_final_exists(self):
        self.manifest["final_model_step"] = 60
        self.manifest["checkpoints"] = [self.checkpoint]
        self.commit()
        retention.prune_intermediate(self.run, self.manifest, self.checkpoint)
        self.assertFalse(list((self.run / "checkpoint-6").glob("*.safetensors")))

    def test_storage_budget_includes_one_active_snapshot_and_previous_finals(self):
        budget = storage_budget(1 << 30)
        self.assertEqual(budget["peak_weight_bytes"], 3 << 30)
        self.assertEqual(budget["final_weight_bytes"], 3 << 30)
        self.assertEqual(budget["required_free_bytes"], int(8.15 * (1 << 30)))


if __name__ == "__main__":
    unittest.main()
