import contextlib
import importlib.util
import io
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/unlearn/tofu/run_official_npo.py"
UPSTREAM_TEST_ROOT = Path(os.environ.get("NPO_UPSTREAM_TEST_DIR", str(SCRIPT.parents[4] / "open-unlearning-npo")))
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

    def test_four_gpu_sdpa_commands_and_two_gpu_reference_batch(self):
        self.args.seed = 7
        plan = launcher.commands(self.args)
        train = plan["train"]
        self.assertEqual(train[train.index("--num_processes") + 1], "4")
        self.assertIn("trainer.args.gradient_accumulation_steps=2", train)
        self.assertIn("model.model_args.attn_implementation=sdpa", train)
        self.assertIn("model.model_args.attn_implementation=sdpa", plan["evaluate"])
        self.assertIn("seed=7", plan["evaluate"])
        self.args.gpu = "0,1"
        self.assertIn("trainer.args.gradient_accumulation_steps=4", launcher.commands(self.args)["train"])

    @unittest.skipUnless(importlib.util.find_spec("hydra") and (UPSTREAM_TEST_ROOT / "configs").is_dir(),
                         "Requires Hydra and a pinned checkout (NPO_UPSTREAM_TEST_DIR)")
    def test_real_upstream_hydra_configs_match_launch_protocol(self):
        from hydra import compose, initialize_config_dir
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import OmegaConf

        launcher.check_upstream(UPSTREAM_TEST_ROOT)
        self.args.upstream_dir = UPSTREAM_TEST_ROOT
        self.args.seed = 7
        plan = launcher.commands(self.args)
        train = plan["train"]
        configs = {}
        for name, config_name, overrides in [
            ("train", "unlearn", train[train.index("src/train.py") + 2:]),
            ("evaluate", "eval", plan["evaluate"][2:]),
        ]:
            with initialize_config_dir(config_dir=str(UPSTREAM_TEST_ROOT / "configs"), version_base=None):
                cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
                HydraConfig.instance().set_config(cfg)
                configs[name] = OmegaConf.to_container(
                    OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"]), resolve=True)
        train, evaluate = configs["train"], configs["evaluate"]
        self.assertEqual(train["trainer"]["handler"], "NPO")
        expected = {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 2,
                    "num_train_epochs": 10, "learning_rate": 1e-5, "optim": "paged_adamw_32bit", "seed": 7}
        for key, value in expected.items():
            self.assertEqual(train["trainer"]["args"][key], value, key)
        self.assertEqual(train["model"]["model_args"]["pretrained_model_name_or_path"], launcher.MODEL_ID)
        self.assertEqual(evaluate["model"]["model_args"]["pretrained_model_name_or_path"], str(self.args.output_dir))
        for cfg in (train, evaluate):
            self.assertEqual(cfg["model"]["model_args"]["attn_implementation"], "sdpa")
        for subset, split in (("forget", "forget05"), ("retain", "retain95")):
            hf_args = train["data"][subset][f"TOFU_QA_{subset}"]["args"]["hf_args"]
            self.assertEqual(hf_args["name"], split)
            self.assertEqual(hf_args["path"], "locuslab/TOFU")
        self.assertEqual(evaluate["seed"], 7)
        self.assertTrue({"forget_quality", "model_utility"} <= set(evaluate["eval"]["tofu"]["metrics"]))
        self.assertFalse(self.args.output_dir.exists())

    def test_check_only_does_not_train_or_create_results(self):
        self.args.check_only = True
        with patch.object(launcher, "check_upstream"), \
             patch.object(launcher, "check_environment", return_value={}), \
             patch.object(launcher.subprocess, "run") as run, \
             contextlib.redirect_stdout(io.StringIO()):
            launcher.execute(self.args)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertIn("--cfg", call.args[0])
            self.assertNotIn("accelerate.commands.launch", call.args[0])
            self.assertEqual(call.kwargs["cwd"], self.args.upstream_dir)
        self.assertFalse(self.args.output_dir.exists())

    def test_upstream_process_does_not_inherit_local_module_paths(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/local/PALU_demo/src", "PYTHONHOME": "/old/env"}), \
             patch.object(launcher.subprocess, "run") as run, \
             contextlib.redirect_stdout(io.StringIO()):
            launcher.run_command([self.args.python, "src/eval.py"], self.args, self.root / "eval.log", evaluation=True)
        env = run.call_args.kwargs["env"]
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(run.call_args.kwargs["cwd"], self.args.upstream_dir)

    def test_bash_entry_works_outside_palu_directory(self):
        result = subprocess.run([
            "bash", str(SCRIPT.with_name("train_tofu_npo_official.sh")),
            "--upstream-dir", str(self.args.upstream_dir),
            "--output-dir", str(self.args.output_dir), "--retain-logs", str(self.retain), "--dry-run",
        ], cwd=self.root, text=True, capture_output=True, check=True,
            env={**os.environ, "NPO_LAUNCHER_PYTHON": sys.executable})
        self.assertIn("--num_processes 4", result.stdout)
        self.assertIn("attn_implementation=sdpa", result.stdout)
        self.assertFalse(self.args.output_dir.exists())

    def test_python_path_keeps_venv_symlink(self):
        venv_python = self.root / "new env/bin/python"
        venv_python.parent.mkdir(parents=True)
        venv_python.symlink_to(sys.executable)
        args = launcher.parse_args([
            "--upstream-dir", str(self.args.upstream_dir), "--output-dir", str(self.args.output_dir),
            "--retain-logs", str(self.retain), "--python", str(venv_python),
        ])
        self.assertEqual(args.python, str(venv_python))

    def run_environment_probe(self, versions=None, cuda="12.8", python_version=(3, 11, 9)):
        # Exercise the actual child-interpreter program with CPU stubs. This
        # tests policy enforcement, not CUDA or DeepSpeed runtime compatibility.
        import packaging.specifiers  # Import before mocking the child's Python version.
        installed = {"torch": "2.7.1+cu128", "bitsandbytes": "0.46.1", "deepspeed": "0.15.4",
                     "transformers": "4.51.3", "lm-eval": "0.4.11"}
        installed.update(versions or {})
        requirements = "torch==2.4.1\nbitsandbytes==0.44.1\ndeepspeed==0.15.4\ntransformers==4.51.3\n"
        fake_torch = SimpleNamespace(version=SimpleNamespace(cuda=cuda), cuda=SimpleNamespace(
            device_count=lambda: 4, get_device_name=lambda i: "test Blackwell",
            get_device_capability=lambda i: (12, 0), get_arch_list=lambda: ["sm_120"],
        ))
        def probe(command, **kwargs):
            stdout = io.StringIO()
            with patch.object(importlib.metadata, "version", side_effect=lambda name: installed[name]), \
                 patch.object(importlib.metadata, "distributions", return_value=[]), \
                 patch.object(Path, "read_text", return_value=requirements), \
                 patch.object(sys, "argv", command[2:]), \
                 patch.object(sys, "version_info", python_version), \
                 patch.dict(sys.modules, {"torch": fake_torch, "flash_attn": None}), \
                 contextlib.redirect_stdout(stdout):
                exec(command[2], {})
            return stdout.getvalue()
        with patch.object(launcher.subprocess, "check_output", side_effect=probe):
            return launcher.check_environment(self.args)

    def test_hardware_policy_records_deviations_without_requiring_flash_attention(self):
        result = self.run_environment_probe()
        self.assertEqual(set(result["differences_from_upstream"]), {"torch", "bitsandbytes"})
        self.assertEqual(result["torch_cuda"], "12.8")

    def test_hardware_policy_still_rejects_old_transformers(self):
        with self.assertRaisesRegex(SystemExit, "transformers: installed=4.45.1"):
            self.run_environment_probe({"transformers": "4.45.1"})

    def test_official_environment_policy_rejects_hardware_overrides(self):
        self.args.environment = "official"
        with self.assertRaisesRegex(SystemExit, "torch: installed=2.7.1"):
            self.run_environment_probe()

    def test_blackwell_rejects_old_cuda_build_before_training(self):
        with self.assertRaisesRegex(SystemExit, "Blackwell requires"):
            self.run_environment_probe(cuda="12.4")

    def test_upstream_python_requirement_is_checked_before_training(self):
        with self.assertRaisesRegex(SystemExit, "Python >=3.11"):
            self.run_environment_probe(python_version=(3, 10, 14))

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
