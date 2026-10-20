"""Frozen per-sample experiment, legacy compatibility, and launch boundaries."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/palu"))
import forget_until_threshold as experiment


def write_plan(study, plan):
    (study / "frozen").mkdir(exist_ok=True)
    experiment.write_json(study / "frozen/config.json", {"forget_threshold_plan": plan})
    experiment.write_json(study / "frozen/manifest.json", {"frozen_file_sha256": {
        "config.json": experiment.sha(study / "frozen/config.json")}})
    experiment.write_json(study / "experiment.json", dict(plan, status="prepared"))


class SamplePlanTests(unittest.TestCase):
    def test_old_defaults_and_missing_scope_keep_global_semantics(self):
        plan = experiment.protocol()
        self.assertEqual(plan["stopping_scope"], "global")
        self.assertEqual(plan["learning_rate"], 1e-5)
        self.assertNotIn("sample_stop_policy", plan)
        legacy = experiment.protocol(probe_interval="update")
        legacy.pop("stopping_scope")
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder)
            write_plan(study, legacy)
            self.assertEqual(experiment.frozen_plan(study), legacy)
            self.assertEqual(experiment.validate_stopping_policy(legacy), "global")

    def test_sample_plan_preserves_batch_pairing_and_evaluation_budget(self):
        plan = experiment.protocol(stopping_scope="sample_permanent", learning_rate=5e-6,
                                   zero_weight_retain_no_grad=True)
        self.assertEqual(plan["phase_weights"], {"forget": [1., 0., 2.]})
        self.assertEqual(plan["retain_updates"], 0)
        self.assertEqual((plan["world_size"], plan["microbatch"], plan["accumulation"]), (4, 1, 8))
        self.assertEqual(plan["probe_steps"], [7, 13, 19, 25, 32, 38, 44, 50, 57, 63])
        self.assertEqual([row["step"] for row in experiment.evaluation_events(plan)], [0, 13, 25, 38, 50, 63])
        policy = plan["sample_stop_policy"]
        self.assertFalse(policy["retain_reference_feedback"])
        self.assertEqual(policy["common_kl"], "all_samples_unchanged")
        self.assertEqual(policy["denominator"], "original_whole_update_sensitive_tokens_including_inactive")
        self.assertIn("excluded", policy["zero_sensitive_token_samples"])
        self.assertEqual(policy["terminal"], "all_eligible_deactivated_or_budget")

    def test_incompatible_sample_policies_are_rejected(self):
        for changes in (dict(stopping_scope="anything"),
                        dict(stopping_scope="sample_permanent", probe_interval="update"),
                        dict(stopping_scope="sample_permanent", threshold_comparison="le")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                experiment.protocol(**changes)

    def test_even_rehashed_frozen_sample_policy_cannot_change_semantics(self):
        original = experiment.protocol(stopping_scope="sample_permanent")
        changes = [dict(evaluation_policy="legacy"), dict(probe_interval="update"),
                   dict(threshold_comparison="le"), dict(sample_stop_policy={}),
                   dict(stopping_scope="unknown")]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                plan = copy.deepcopy(original)
                plan.update(change)
                # Recompute steps as an attacker/user might when rewriting all visible files.
                if change.get("probe_interval") == "update":
                    plan["probe_steps"] = list(range(1, 64))
                study = Path(folder)
                write_plan(study, plan)
                with self.assertRaises(ValueError):
                    experiment.frozen_plan(study)

    def test_train_dispatches_sample_controller_without_loading_gpu(self):
        plan = experiment.protocol(stopping_scope="sample_permanent")
        sample_control = object()
        seen = []
        fake_modules = {
            "torch": SimpleNamespace(distributed=SimpleNamespace(is_initialized=lambda: False)),
            "layout_token_training": SimpleNamespace(train=lambda args, torch, control: seen.append(control)),
            "forget_threshold_control": SimpleNamespace(ForgetThresholdControl=lambda p: self.fail("global controller selected")),
            "sample_forget_control": SimpleNamespace(SampleForgetControl=lambda p: sample_control)}
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder)
            (study / "frozen").mkdir()
            experiment.write_json(study / "frozen/manifest.json", {"frozen_file_sha256": {}})
            experiment.write_json(study / "code_sha256.json", {})
            with patch.dict(sys.modules, fake_modules), patch.object(experiment, "frozen_plan", return_value=plan):
                experiment.train(study)
        self.assertEqual(seen, [sample_control])

    def test_stopping_scope_cannot_override_prepared_modes(self):
        for mode in ("train", "analyze", "export"):
            result = subprocess.run([sys.executable, "-S", str(Path(experiment.__file__)),
                "--mode", mode, "--study", "/unused", "--stopping-scope", "sample_permanent"],
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("use the frozen plan", result.stderr)


class SampleLauncherTests(unittest.TestCase):
    def launch(self, name="run_sample_forget_threshold.sh", **overrides):
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder) / "never-created"
            env = dict(os.environ, MODE="dry-run", PYTHON_BIN=sys.executable,
                       STUDY_DIR=str(study), SOURCE_STUDY=str(Path(folder) / "unused"))
            for key in ("SEEDS", "LEARNING_RATE", "MAX_UPDATES", "STOPPING_SCOPE", "PROBE_INTERVAL",
                        "THRESHOLD_COMPARISON", "ZERO_WEIGHT_RETAIN_NO_GRAD"):
                env.pop(key, None)
            env.update(overrides)
            result = subprocess.run(["bash", str(ROOT / "scripts/palu" / name)], env=env,
                                    text=True, capture_output=True)
            self.assertFalse(study.exists())
        plans, rest = [], result.stdout
        while "{" in rest:
            start = rest.index("{")
            plan, consumed = json.JSONDecoder().raw_decode(rest[start:])
            plans.append(plan)
            rest = rest[start + consumed:]
        return result, plans

    def test_new_launcher_freezes_three_requested_plans_without_cuda_or_files(self):
        result, plans = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([plan["seed"] for plan in plans], [103, 104, 105])
        for plan in plans:
            self.assertEqual(plan["stopping_scope"], "sample_permanent")
            self.assertEqual(plan["learning_rate"], 5e-6)
            self.assertEqual(plan["max_updates"], 63)
            self.assertTrue(plan["zero_weight_retain_no_grad"])
            self.assertEqual(plan["threshold_comparison"], "lt")

    def test_old_launcher_defaults_are_unchanged(self):
        result, plans = self.launch("run_forget_until_threshold.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([plan["seed"] for plan in plans], [103, 104, 105, 106])
        for plan in plans:
            self.assertEqual(plan["stopping_scope"], "global")
            self.assertEqual(plan["learning_rate"], 1e-5)
            self.assertFalse(plan["zero_weight_retain_no_grad"])

    def test_invalid_scope_or_interval_fails_before_prepare(self):
        for changes in (dict(STOPPING_SCOPE="unknown"), dict(PROBE_INTERVAL="update"),
                        dict(THRESHOLD_COMPARISON="le")):
            with self.subTest(changes=changes):
                result, plans = self.launch(**changes)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(plans, [])


if __name__ == "__main__":
    unittest.main()
