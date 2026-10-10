import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/palu_mean_ablation"))
from run_comparison import (build_commands, collect_metrics, hash_file, ks_statistic,
                            summarize, validate_overrides, validate_pair, write_json)


def settings():
    return SimpleNamespace(python=sys.executable, model="Llama-3.1-8B-Instruct", model_path="/tmp/model with spaces",
        model_revision=None, forget_split="forget05", holdout_split="holdout05", retain_split="retain95",
        lr=2e-5, batch_size=8, grad_accum=4, epochs=10., max_steps=-1, max_grad_norm=1.,
        alpha=.2, top_k=5000, first_n=3, override=[], retain_logs=Path("/tmp/reference logs.json"),
        eval_batch_size=8, eval_dataset=None, classifier_model=None, skip_eval=False,
        num_processes=1, main_process_port=29513)


def compose_training(args, run, variant):
    command, _ = build_commands(args, run, 42, variant)
    overrides = command[command.index("--config-name=unlearn.yaml") + 1:]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name="unlearn", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(config)
        return OmegaConf.to_container(OmegaConf.masked_copy(config, [key for key in config if key != "hydra"]), resolve=True)


class ComparisonProtocolTests(unittest.TestCase):
    def test_real_hydra_configs_differ_only_in_flag_and_output_metadata(self):
        args = settings()
        full = compose_training(args, Path("/tmp/full with spaces"), "full_mean")
        stop = compose_training(args, Path("/tmp/stop with spaces"), "stop_mean")
        validate_pair(full, stop)
        self.assertEqual(full["model"]["model_args"]["pretrained_model_name_or_path"], args.model_path)
        self.assertFalse(full["trainer"]["args"]["do_eval"])
        self.assertFalse(full["trainer"]["method_args"]["detach_mean_target"])
        self.assertTrue(stop["trainer"]["method_args"]["detach_mean_target"])
        stop["trainer"]["args"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "configs differ"):
            validate_pair(full, stop)

    def test_evaluation_loads_saved_arm_and_matched_reference(self):
        args, run = settings(), Path("/tmp/run with spaces")
        _, command = build_commands(args, run, 42, "stop_mean")
        overrides = command[2:]
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
            cfg = compose(config_name="eval", overrides=overrides, return_hydra_config=True)
            HydraConfig.instance().set_config(cfg)
            self.assertEqual(cfg.model.model_args.pretrained_model_name_or_path, str(run))
            self.assertEqual(cfg.eval.tofu.retain_logs_path, str(args.retain_logs))
            self.assertEqual(cfg.eval.tofu.output_dir, str(run / "evals"))
            self.assertEqual(cfg.eval.tofu.forget_split, "forget05")
            self.assertTrue(cfg.eval.tofu.overwrite)

    def test_override_cannot_change_arm_seed_or_enable_in_training_evaluation(self):
        for override in ("++trainer.method_args.detach_mean_target=false", "trainer.args.seed=1", "trainer.args.do_eval=true",
                         "~eval", "trainer.args={seed:1}", "forget_split=forget10"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_overrides([override])
        validate_overrides(["trainer.args.optim=adamw_torch", "+trainer.args.use_cpu=true"])

    def test_dry_run_is_model_free_and_alternates_paired_order(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "plan with spaces"
            execution = subprocess.run([sys.executable, str(ROOT / "scripts/palu_mean_ablation/run_comparison.py"),
                "--dry-run", "--output", str(output), "--seeds", "42", "43", "--python", "/no/such/python"],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(execution.returncode, 0, execution.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual([(run["seed"], run["variant"]) for run in manifest["runs"]],
                             [(42, "full_mean"), (42, "stop_mean"), (43, "stop_mean"), (43, "full_mean")])
            self.assertFalse((output / "gradient_probe.json").exists())

    def test_ks_statistic_matches_scipy_including_ties(self):
        from scipy.stats import ks_2samp
        left, right = [.1, .1, .4, .7], [.1, .3, .5, .5, .9]
        self.assertAlmostEqual(ks_statistic(left, right), ks_2samp(left, right).statistic)

    def fixture(self, root, variant, runtime, fq):
        run = root / "seed_42" / variant
        write_json(run / "resolved_config.json", compose_training(settings(), run, variant))
        write_json(run / "trainer_state.json", {"global_step": 10, "epoch": 1., "log_history": [
            {"train_runtime": runtime, "train_steps_per_second": 10 / runtime}]})
        write_json(run / "telemetry.json", {"entrypoint_wall_seconds": runtime + 2., "cuda_peak_allocated_bytes": 100,
                                           "cuda_peak_reserved_bytes": 200})
        reference = root / "reference.json"
        write_json(reference, {"forget_truth_ratio": {"value_by_index": {"0": {"score": .5}, "1": {"score": 1.}}}})
        write_json(run / "evals/TOFU_SUMMARY.json", {"forget_quality": fq, "model_utility": .6,
            "exact_memorization": .5, "forget_Q_A_gibberish": .8})
        write_json(run / "evals/TOFU_EVAL.json", {"forget_truth_ratio": {"value_by_index": {"0": {"score": .2}, "1": {"score": .5}}}})
        write_json(run / "evals/evaluation_provenance.json", {"status": "completed", "reference_sha256": hash_file(reference),
            "config": {"model": {"model_args": {"pretrained_model_name_or_path": str(run)}}}})
        metrics = collect_metrics(run, reference, True)
        write_json(run / "result.json", {"status": "completed", "metrics": metrics})
        return run, reference

    def test_reporting_pairs_by_seed_and_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            full, _ = self.fixture(root, "full_mean", 20., .4)
            stop, _ = self.fixture(root, "stop_mean", 15., .6)
            write_json(root / "manifest.json", {"seeds": [42], "evaluation_enabled": True, "runs": [
                {"seed": 42, "variant": variant, "output_dir": str(run)}
                for variant, run in (("stop_mean", stop), ("full_mean", full))]})
            report = summarize(root)
            self.assertEqual(report["status"], "completed")
            delta = report["paired_stop_minus_full"][0]
            self.assertAlmostEqual(delta["forget_quality"], .2)
            self.assertEqual(delta["train_runtime"], -5.)
            self.assertIsNone(report["aggregate"]["train_runtime"]["stddev_paired_difference"])
            self.assertTrue((root / "comparison.csv").exists())
            cfg = json.loads((stop / "resolved_config.json").read_text())
            cfg["trainer"]["args"]["seed"] = 43
            write_json(stop / "resolved_config.json", cfg)
            with self.assertRaises(ValueError):
                summarize(root)

    def test_invalid_or_incomplete_evaluation_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            run, reference = self.fixture(Path(temp), "full_mean", 20., .4)
            write_json(run / "evals/evaluation_provenance.json", {"status": "completed_unverified_cache"})
            with self.assertRaisesRegex(ValueError, "provenance"):
                collect_metrics(run, reference, True)
            write_json(run / "evals/evaluation_provenance.json", {"status": "completed", "reference_sha256": hash_file(reference),
                "config": {"model": {"model_args": {"pretrained_model_name_or_path": str(run)}}}})
            write_json(run / "evals/TOFU_SUMMARY.json", {"forget_quality": .4})
            with self.assertRaisesRegex(ValueError, "Missing/non-finite"):
                collect_metrics(run, reference, True)

    def test_reporting_rejects_different_step_counts_and_runtime_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            full, _ = self.fixture(root, "full_mean", 20., .4)
            stop, _ = self.fixture(root, "stop_mean", 15., .6)
            write_json(root / "manifest.json", {"seeds": [42], "evaluation_enabled": True, "runs": [
                {"seed": 42, "variant": name, "output_dir": str(run)}
                for name, run in (("full_mean", full), ("stop_mean", stop))]})
            result = json.loads((stop / "result.json").read_text())
            result["metrics"]["optimizer_steps"] = 9
            write_json(stop / "result.json", result)
            with self.assertRaisesRegex(ValueError, "step counts"):
                summarize(root)
            result["metrics"]["optimizer_steps"] = 10
            write_json(stop / "result.json", result)
            telemetry = json.loads((stop / "telemetry.json").read_text())
            telemetry["packages"] = {"torch": "changed"}
            write_json(stop / "telemetry.json", telemetry)
            with self.assertRaisesRegex(ValueError, "environment differs"):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
