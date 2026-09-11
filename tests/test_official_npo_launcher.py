import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/unlearn/tofu/run_official_npo.py"
SPEC = importlib.util.spec_from_file_location("official_npo_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class OfficialNPOLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.retain = self.root / "TOFU_EVAL.json"
        self.retain.write_text(json.dumps({"forget_truth_ratio": {"value_by_index": {}}}))
        self.args = launcher.parse_args([
            "--upstream-dir", str(self.root / "official"),
            "--output-dir", str(self.root / "result with spaces"),
            "--retain-logs", str(self.retain),
        ])

    def fake_command(self, command, args, logfile, evaluation=False):
        if evaluation:
            target = args.output_dir / "evals"
            target.mkdir()
            scores = {"forget_quality": 0.2, "model_utility": 0.5}
            launcher.write_json(target / "TOFU_SUMMARY.json", scores)
            launcher.write_json(target / "TOFU_EVAL.json", {
                key: {"agg_value": value} for key, value in scores.items()
            })
        elif "accelerate.commands.launch" in command:
            launcher.write_json(args.output_dir / "trainer_state.json", {
                "global_step": 60, "max_steps": 60, "epoch": 10,
            })
            (args.output_dir / "model-00001-of-00001.safetensors").write_bytes(b"test weights")
            (args.output_dir / "config.json").write_text("{}")

    def execute_mocked(self, run=None):
        with patch.object(launcher, "check_upstream"), \
             patch.object(launcher, "check_environment", return_value={}), \
             patch.object(launcher, "run_command", side_effect=run or self.fake_command), \
             contextlib.redirect_stdout(io.StringIO()):
            launcher.execute(self.args)

    def test_dry_run_has_no_side_effects_or_subprocesses(self):
        self.args.dry_run = True
        with patch.object(launcher.subprocess, "check_output", side_effect=AssertionError), \
             patch.object(launcher.subprocess, "run", side_effect=AssertionError), \
             contextlib.redirect_stdout(io.StringIO()):
            launcher.execute(self.args)
        self.assertFalse(self.args.output_dir.exists())

    def test_existing_directory_is_never_overwritten(self):
        self.args.output_dir.mkdir()
        sentinel = self.args.output_dir / "original.txt"
        sentinel.write_text("original")
        with self.assertRaisesRegex(ValueError, "must be NEW"):
            self.execute_mocked()
        self.assertEqual(sentinel.read_text(), "original")

    def test_valid_evaluation_cleans_only_weight_files(self):
        self.execute_mocked()
        self.assertFalse((self.args.output_dir / "model-00001-of-00001.safetensors").exists())
        self.assertTrue((self.args.output_dir / "config.json").exists())
        status = json.loads((self.args.output_dir / "baseline_audit/status.json").read_text())
        self.assertEqual(status["status"], "completed")

    def test_missing_evaluation_preserves_weights_and_records_failure(self):
        def incomplete(command, args, logfile, evaluation=False):
            if not evaluation:
                self.fake_command(command, args, logfile, evaluation)
        with self.assertRaises(FileNotFoundError):
            self.execute_mocked(incomplete)
        self.assertTrue((self.args.output_dir / "model-00001-of-00001.safetensors").exists())
        status = json.loads((self.args.output_dir / "baseline_audit/status.json").read_text())
        self.assertEqual(status["status"], "failed")

    def test_keep_model_preserves_weights(self):
        self.args.keep_model = True
        self.execute_mocked()
        self.assertTrue((self.args.output_dir / "model-00001-of-00001.safetensors").exists())

    def test_incomplete_training_never_runs_evaluation(self):
        evaluated = []
        def partial(command, args, logfile, evaluation=False):
            evaluated.append(evaluation)
            self.fake_command(command, args, logfile, evaluation)
            if "accelerate.commands.launch" in command:
                launcher.write_json(args.output_dir / "trainer_state.json", {
                    "global_step": 1, "max_steps": 60,
                })
        with self.assertRaisesRegex(ValueError, "planned optimizer-step count"):
            self.execute_mocked(partial)
        self.assertNotIn(True, evaluated)
        self.assertTrue((self.args.output_dir / "model-00001-of-00001.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
