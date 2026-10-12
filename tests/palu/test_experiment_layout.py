import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/palu"))
from experiment_layout import experiment_name, protocol_id
from run_experiment import read_json, register_launch, update_manifest
from test_experiment_protocol import settings


def options(root, seed):
    args = settings()
    args.output, args.output_root, args.experiment = None, root / "palu", "batch8_first_sensitive"
    args.seeds, args.gpu = [seed], "2"
    args.new_group, args.dry_run, args.skip_eval = False, False, True
    args.summarize, args.retain_model_path = None, None
    return args


class ExperimentLayoutTests(unittest.TestCase):
    def snapshot(self):
        return {str(ROOT / "src/trainer/unlearn/palu.py"): "algorithm",
                str(ROOT / "scripts/palu/run.sh"): "launcher"}

    def test_separate_launches_join_matching_timestamp_and_keep_seed_provenance(self):
        with tempfile.TemporaryDirectory() as temporary, patch("run_experiment.input_snapshot", return_value=self.snapshot()):
            root = Path(temporary)
            first = options(root, 0)
            register_launch(first)
            second = options(root, 1)
            second.gpu = "7"  # Device ordinal is provenance, not a different batch protocol.
            manifest, planned, _ = register_launch(second)
            self.assertEqual(first.output, second.output)
            self.assertEqual(first.output.parent, root / "palu/batch8_first_sensitive")
            self.assertEqual(manifest["seeds"], [0, 1])
            self.assertEqual(manifest["settings"]["seeds"], [0, 1])
            self.assertEqual([row["seed"] for row in planned], [1])
            self.assertEqual([row["train_cuda_visible_devices"] for row in manifest["runs"]], ["2", "7"])
            self.assertTrue(all(Path(row["output_dir"]).parent == first.output for row in manifest["runs"]))
            self.assertEqual(protocol_id(manifest), manifest["layout"]["config_id"])

    def test_repeated_seed_changed_config_and_new_group_never_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary, patch("run_experiment.input_snapshot", return_value=self.snapshot()):
            root = Path(temporary)
            initial = options(root, 0)
            register_launch(initial)
            original = (initial.output / "manifest.json").read_bytes()
            repeated = options(root, 0)
            register_launch(repeated)
            changed = options(root, 1)
            changed.lr *= 2
            register_launch(changed)
            forced = options(root, 2)
            forced.new_group = True
            register_launch(forced)
            self.assertEqual(len({args.output for args in (initial, repeated, changed, forced)}), 4)
            repeated_newest = options(root, 2)
            register_launch(repeated_newest)
            self.assertNotIn(repeated_newest.output, {args.output for args in (initial, repeated, changed, forced)})
            self.assertEqual((initial.output / "manifest.json").read_bytes(), original)

    def test_algorithm_change_splits_groups_but_launcher_seed_edit_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = options(root, 0)
            with patch("run_experiment.input_snapshot", return_value=self.snapshot()):
                register_launch(first)
            edited_launcher = self.snapshot()
            edited_launcher[str(ROOT / "scripts/palu/run.sh")] = "different seed default"
            second = options(root, 1)
            with patch("run_experiment.input_snapshot", return_value=edited_launcher):
                register_launch(second)
            self.assertEqual(first.output, second.output)
            edited_algorithm = dict(edited_launcher)
            edited_algorithm[str(ROOT / "src/trainer/unlearn/palu.py")] = "different algorithm"
            third = options(root, 2)
            with patch("run_experiment.input_snapshot", return_value=edited_algorithm):
                register_launch(third)
            self.assertNotEqual(first.output, third.output)

    def test_manifest_updates_preserve_other_registered_seed_commands(self):
        with tempfile.TemporaryDirectory() as temporary, patch("run_experiment.input_snapshot", return_value=self.snapshot()):
            root = Path(temporary)
            first, second = options(root, 0), options(root, 1)
            _, first_plan, _ = register_launch(first)
            _, second_plan, _ = register_launch(second)
            first_plan[0]["train_argv"].append("seed0_update")
            second_plan[0]["train_argv"].append("seed1_update")
            update_manifest(first.output, first_plan)
            manifest = update_manifest(second.output, second_plan)
            self.assertEqual([row["train_argv"][-1] for row in manifest["runs"]], ["seed0_update", "seed1_update"])

    def test_concurrent_seed_registration_uses_one_timestamp_without_lost_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            code = ("import json, sys; sys.path.insert(0, sys.argv.pop(1)); "
                    "from run_experiment import parse_args, register_launch; "
                    "args=parse_args(); register_launch(args); print(json.dumps(str(args.output)))")
            processes = [subprocess.Popen([sys.executable, "-c", code, str(ROOT / "scripts/palu"),
                "--output-root", temporary, "--experiment", "parallel", "--seeds", str(seed),
                "--gpu", gpu, "--skip-eval"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                for seed, gpu in ((0, "2"), (1, "7"))]
            outputs = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, stderr)
                outputs.append(Path(json.loads(stdout)))
            self.assertEqual(outputs[0], outputs[1])
            manifest = read_json(outputs[0] / "manifest.json")
            self.assertEqual(manifest["seeds"], [0, 1])
            self.assertEqual(len(manifest["runs"]), 2)
            self.assertEqual(len(manifest["launches"]), 2)

    def test_experiment_names_cannot_escape_the_baseline_directory(self):
        self.assertEqual(experiment_name("batch8_sensitive_probe"), "batch8_sensitive_probe")
        for value in ("../other", "a/b", "..", "", "a\\b", "a\nb"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                experiment_name(value)


if __name__ == "__main__":
    unittest.main()
