import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/palu"))
sys.path.insert(0, str(ROOT / "src"))
from run_experiment import cleanup_completed_weights, merge_telemetry, read_json, write_json
import test_experiment_protocol as protocol


class RuntimeAndCleanupTests(unittest.TestCase):
    def test_multi_gpu_plan_preserves_batch_and_uses_single_gpu_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "plan"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/palu/run_experiment.py"),
                "--dry-run", "--output", str(output), "--gpu", "1,3,4,5", "--seeds", "42"],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = read_json(output / "manifest.json")
            self.assertFalse(manifest["settings"]["keep_model"])
            self.assertEqual(manifest["settings"]["effective_batch_size"], 32)
            self.assertEqual(manifest["settings"]["grad_accum"], 8)
            self.assertEqual(manifest["eval_cuda_visible_devices"], "1")
            train, evaluate = manifest["runs"][0]["train_argv"], manifest["runs"][0]["eval_argv"]
            self.assertEqual(train[1:3], ["-m", "accelerate.commands.launch"])
            self.assertEqual(train[train.index("--num_processes") + 1], "4")
            self.assertNotIn("accelerate.commands.launch", evaluate)
            self.assertIn("CUDA_VISIBLE_DEVICES=1 ", result.stdout)
            args = protocol.settings()
            args.num_processes, args.batch_size, args.grad_accum = 4, 1, 8
            config = protocol.compose_training(args, Path(temp) / "seed_42")
            from run_experiment import validate_config
            validate_config(config, 42)
            self.assertEqual(config["trainer"]["args"]["gradient_accumulation_steps"], 8)
            self.assertEqual(len(manifest["runs"]), 1)

    def test_invalid_gpu_topology_and_ambiguous_default_batch_are_rejected(self):
        for flags in (["--gpu", "0,0"], ["--gpu", "0,"], ["--gpu", "0,1,2"],
                      ["--gpu", "0", "--main-process-port", "0"]):
            with self.subTest(flags=flags):
                result = subprocess.run([sys.executable, str(ROOT / "scripts/palu/run_experiment.py"),
                    "--dry-run", *flags], cwd=ROOT, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)

    def test_rank_aggregation_uses_maximum_and_rejects_missing_failed_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            for rank, seconds, allocated in ((0, 12., 100), (1, 11., 200)):
                write_json(run / f"telemetry.rank{rank}.json", {"status": "completed", "rank": rank,
                    "world_size": 2, "packages": {"torch": "2.4.1"}, "cuda_available": True,
                    "cuda_version": "12.1", "cuda_visible_devices": "1,3", "gpu_name": "GPU",
                    "entrypoint_wall_seconds": seconds, "cuda_peak_allocated_bytes": allocated,
                    "cuda_peak_reserved_bytes": allocated + 10})
            telemetry = merge_telemetry(run, 2)
            self.assertEqual(telemetry["entrypoint_wall_seconds"], 12.)
            self.assertEqual(telemetry["cuda_peak_allocated_bytes"], 200)
            self.assertEqual(len(telemetry["ranks"]), 2)
            failed = read_json(run / "telemetry.rank1.json")
            failed["status"] = "failed"
            write_json(run / "telemetry.rank1.json", failed)
            with self.assertRaisesRegex(ValueError, "rank 1"):
                merge_telemetry(run, 2)
            (run / "telemetry.rank1.json").unlink()
            with self.assertRaises(FileNotFoundError):
                merge_telemetry(run, 2)

    def test_cleanup_removes_shards_but_preserves_reports_and_failed_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                     "model.safetensors.index.json", "pytorch_model.bin"]
            for name in [*files, "config.json", "tokenizer.json", "trainer_state.json", "train.log"]:
                (run / name).write_text("test")
            write_json(run / "result.json", {"status": "eval_failed"})
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                cleanup_completed_weights(run, False)
            self.assertTrue(all((run / name).exists() for name in files))
            write_json(run / "result.json", {"status": "completed", "metrics": {"optimizer_steps": 2}})
            cleanup_completed_weights(run, True)
            self.assertTrue(all((run / name).exists() for name in files))
            result = cleanup_completed_weights(run, False)
            self.assertCountEqual(result["removed_weight_files"], files)
            self.assertTrue((run / "trainer_state.json").exists())
            self.assertTrue((run / "tokenizer.json").exists())
            self.assertFalse(any((run / name).exists() for name in files))

    def test_cleanup_refuses_symlinks_before_removing_any_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            run.mkdir()
            original = Path(temp) / "initial_model"
            original.write_text("original")
            (run / "model.safetensors").write_text("owned")
            (run / "pytorch_model.bin").symlink_to(original)
            write_json(run / "result.json", {"status": "completed", "metrics": {"optimizer_steps": 2}})
            with self.assertRaisesRegex(ValueError, "non-regular"):
                cleanup_completed_weights(run, False)
            self.assertEqual(original.read_text(), "original")
            self.assertTrue((run / "model.safetensors").exists())

    def test_evaluation_of_another_checkpoint_is_rejected_before_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            run, reference = protocol.ExperimentProtocolTests().fixture(Path(temp), 42, 20., .4)
            provenance = read_json(run / "evals/evaluation_provenance.json")
            provenance["config"]["model"]["model_args"]["pretrained_model_name_or_path"] = "/other/checkpoint"
            write_json(run / "evals/evaluation_provenance.json", provenance)
            from run_experiment import collect_metrics
            with self.assertRaisesRegex(ValueError, "this run's checkpoint"):
                collect_metrics(run, reference, True)

    def test_worker_selects_local_device_and_writes_separate_rank_telemetry(self):
        import train_with_telemetry as wrapper
        with tempfile.TemporaryDirectory() as temp:
            telemetry = Path(temp) / "telemetry.json"
            with patch.dict(wrapper.os.environ, {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"}), \
                 patch.object(wrapper.sys, "argv", ["wrapper", "--telemetry", str(telemetry)]), \
                 patch.object(torch.cuda, "is_available", return_value=True), \
                 patch.object(torch.cuda, "set_device") as select, \
                 patch.object(torch.cuda, "synchronize"), \
                 patch.object(torch.cuda, "reset_peak_memory_stats"), \
                 patch.object(torch.cuda, "get_device_name", return_value="GPU1"), \
                 patch.object(torch.cuda, "max_memory_allocated", return_value=100), \
                 patch.object(torch.cuda, "max_memory_reserved", return_value=200), \
                 patch.object(wrapper.runpy, "run_path", side_effect=lambda *a, **kw: select.assert_called_once_with(1)):
                wrapper.main()
            self.assertFalse(telemetry.exists())
            record = read_json(Path(temp) / "telemetry.rank1.json")
            self.assertEqual((record["rank"], record["local_rank"], record["world_size"]), (1, 1, 2))
            self.assertEqual(record["gpu_name"], "GPU1")

    def test_reference_zero3_config_resolves_nested_auto_buckets(self):
        import trainer.unlearn.base as base
        config = {"zero_optimization": {"stage": 3, "reduce_bucket_size": "auto",
                  "stage3_prefetch_bucket_size": "auto", "stage3_param_persistence_threshold": "auto"}}
        original = copy.deepcopy(config)
        trainer = object.__new__(base.UnlearnTrainer)
        trainer.accelerator = SimpleNamespace(state=SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(deepspeed_config=config)))
        model = torch.nn.Linear(2, 2)
        model.config = SimpleNamespace(hidden_size=10)
        def initialize(model, config):
            zero = config["zero_optimization"]
            self.assertEqual(zero["reduce_bucket_size"], 100)
            self.assertEqual(zero["stage3_prefetch_bucket_size"], 90)
            self.assertEqual(zero["stage3_param_persistence_threshold"], 100)
            self.assertEqual(set(config), {"zero_optimization"})
            return model, None, None, None
        with patch.object(base, "deepspeed", SimpleNamespace(initialize=initialize), create=True):
            reference = trainer._prepare_deepspeed(model)
        self.assertFalse(reference.training)
        self.assertEqual(config, original)


if __name__ == "__main__":
    unittest.main()
