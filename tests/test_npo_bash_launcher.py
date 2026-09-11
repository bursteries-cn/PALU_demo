"""CPU-only launch/cleanup integration checks; these do not train a model."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/unlearn/tofu/train_tofu_npo_llama3.sh"


class NPOBashLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "results with spaces"
        self.retain = self.root / "TOFU_EVAL.json"
        self.retain.write_text('{"forget_truth_ratio": {"value_by_index": {}}}')

    def launch(self, *extra, env=None):
        return subprocess.run([
            "bash", str(SCRIPT), "--output-dir", str(self.output),
            "--retain-logs", str(self.retain), *extra,
        ], cwd=self.root, env={**os.environ, **(env or {})}, text=True, capture_output=True)

    def test_dry_run_selects_four_rank_npo_and_same_evaluation_dataset(self):
        result = self.launch("--dry-run", "--gpu=4,5,6,7", "--seed=3", "--epochs", "12", "--dataset", "/data/TOFU copy")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("env ")]
        self.assertEqual(len(lines), 2)
        train, evaluate = lines
        self.assertEqual(train[train.index("--num_processes") + 1], "4")
        for value in ["CUDA_VISIBLE_DEVICES=4,5,6,7", "trainer=NPO", "trainer.args.per_device_train_batch_size=4", "trainer.args.gradient_accumulation_steps=2", "trainer.args.seed=3", "trainer.args.num_train_epochs=12", "trainer.args.learning_rate=1e-5", "++trainer.args.max_grad_norm=0", "++trainer.args.lr_scheduler_type=linear", "~eval.tofu"]:
            self.assertIn(value, train)
        self.assertIn("CUDA_VISIBLE_DEVICES=4", evaluate)
        self.assertIn("+tofu_dataset_path=/data/TOFU copy", evaluate)
        self.assertIn("data.forget.TOFU_QA_forget.args.hf_args.path=/data/TOFU copy", train)
        self.assertNotIn("flash_attention_2", result.stdout)
        self.assertFalse(self.output.exists())

    def test_rejects_bad_gpu_and_missing_or_empty_values(self):
        for flags in [("--gpu", "0,1"), ("--gpu", "0,1,1,3"), ("--epochs", "0"), ("--lr", "nan"), ("--lr", " "), ("--seed",)]:
            with self.subTest(flags=flags):
                result = self.launch("--dry-run", *flags)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())

    def test_sweep_cannot_reuse_exact_output_directory(self):
        result = self.launch("--dry-run", "--lr", "1e-5 2e-5")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--output-root", result.stderr)

    def test_hydra_composition_disables_only_the_training_evaluator(self):
        try:
            from hydra import compose, initialize_config_dir
        except ImportError:
            self.skipTest("Hydra is not installed in this CPU test environment")
        result = self.launch("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        for line in result.stdout.splitlines():
            if not line.startswith("env "):
                continue
            command = shlex.split(line)
            training = "src/train.py" in command
            start = command.index("src/train.py" if training else "src/eval.py") + 1
            overrides = [a for a in command[start:] if not a.startswith("--")]
            overrides += ["hydra/job_logging=default", "hydra/hydra_logging=default"]
            with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
                cfg = compose(config_name="unlearn" if training else "eval", overrides=overrides)
                if training:
                    self.assertFalse(cfg.get("eval"))
                    self.assertEqual(cfg.trainer.handler, "NPO")
                    self.assertEqual(cfg.trainer.args.max_grad_norm, 0)
                    self.assertEqual(cfg.model.model_args.attn_implementation, "sdpa")
                else:
                    self.assertEqual(len(cfg.eval.tofu.metrics), 4)
                    self.assertEqual(cfg.eval.tofu.retain_logs_path, str(self.retain))

    def fake_python(self):
        path = self.root / "fake-python"
        path.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''
            import json, os, pathlib, subprocess, sys
            args = sys.argv[1:]
            if args[:1] == ['-c'] and 'torch.cuda.device_count()' in args[1]:
                raise SystemExit(0)
            training = args[:2] == ['-m', 'accelerate.commands.launch']
            evaluating = args[:1] == ['src/eval.py']
            if not training and not evaluating:
                raise SystemExit(subprocess.call([sys.executable, *args]))
            with open(os.environ['NPO_TEST_CALLS'], 'a') as f:
                f.write(json.dumps({'args': args, 'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'cwd': os.getcwd()}) + '\\n')
            output = pathlib.Path(next(a.split('=', 1)[1] for a in args if a.startswith('paths.output_dir=')))
            output.mkdir(parents=True, exist_ok=True)
            if training:
                (output / 'trainer_state.json').write_text(json.dumps({'global_step': 60, 'max_steps': 60}))
                (output / 'model.safetensors').write_bytes(b'mock weights')
                (output / 'config.json').write_text('{}')
            elif os.environ.get('NPO_TEST_FAIL_EVAL'):
                raise SystemExit(42)
            else:
                scores = {k: .5 for k in ['forget_quality', 'model_utility', 'exact_memorization', 'forget_Q_A_gibberish']}
                (output / 'TOFU_SUMMARY.json').write_text(json.dumps(scores))
                (output / 'TOFU_EVAL.json').write_text(json.dumps({k: {'agg_value': v} for k, v in scores.items()}))
                (output / 'evaluation_provenance.json').write_text(json.dumps({'status': 'completed', 'config': {'model': {'model_args': {'pretrained_model_name_or_path': str(output.parent)}}}}))
        '''))
        path.chmod(0o755)
        return path

    def test_full_shell_pipeline_evaluates_on_one_gpu_then_cleans_weights(self):
        calls = self.root / "calls.jsonl"
        result = self.launch("--python", str(self.fake_python()), env={"NPO_TEST_CALLS": str(calls)})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertEqual([r['gpu'] for r in records], ['0,1,2,3', '0'])
        self.assertTrue(all(Path(r['cwd']) == ROOT for r in records))
        self.assertFalse((self.output / "model.safetensors").exists())
        self.assertTrue((self.output / "config.json").exists())
        self.assertTrue((self.output / "npo_run_complete.json").exists())

    def test_evaluation_failure_through_tee_preserves_weights(self):
        result = self.launch("--python", str(self.fake_python()), env={
            "NPO_TEST_CALLS": str(self.root / "calls.jsonl"), "NPO_TEST_FAIL_EVAL": "1",
        })
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertTrue((self.output / "model.safetensors").exists())
        self.assertFalse((self.output / "npo_run_complete.json").exists())

    def test_keep_model(self):
        result = self.launch("--python", str(self.fake_python()), "--keep-model", env={"NPO_TEST_CALLS": str(self.root / "calls.jsonl")})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.output / "model.safetensors").exists())


if __name__ == '__main__':
    unittest.main()
