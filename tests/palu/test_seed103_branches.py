"""Branch timing, shared seed103 controls and state inheritance safeguards.

The CPU tests do not establish CUDA/ZeRO-3 numerical equivalence. A real run
also verifies per-rank checkpoint contents and the restored live runtime.
"""
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/palu"))
sys.path.insert(0, str(ROOT / "src"))

import layout_loss_curve_experiment as replay
from online_epoch_evaluation import epoch_events
import seed103_branch_training as branches
import seed103_branch_experiment as experiment


class ProtocolTests(unittest.TestCase):
    def test_only_forget_weight_changes_after_six_complete_updates(self):
        self.assertEqual(branches.FORK_STEP, 6)
        self.assertEqual(branches.BRANCH_GAMMAS, {"continue": 1.0, "repair": 0.0, "weak": 0.1})
        for arm, coefficient in branches.BRANCH_GAMMAS.items():
            for step in range(1, 7):
                self.assertEqual(branches.branch_gamma(step, arm), 1.0)
            for step in (7, 8, 25, 60):
                self.assertEqual(branches.branch_gamma(step, arm), coefficient)

    def test_complete_fork_precedes_first_actual_epoch_and_pending_gradients(self):
        first = epoch_events()[1]
        fork_pairs = branches.FORK_STEP * 32
        self.assertEqual(fork_pairs, 192)
        self.assertEqual(first["consumed_pairs"], 200)
        self.assertEqual(first["step"], 6)
        self.assertEqual(first["pending_pairs"], 8)
        # These are two four-rank microbatches of update7; no update may be
        # flushed or pair repeated merely to place the split at an epoch edge.
        self.assertEqual((first["consumed_pairs"] - fork_pairs) // 4, 2)

    def test_branch_continues_same_immutable_pair_stream_and_seed103_method(self):
        config, pairs, _, _ = replay.validate_assets()
        records = replay.replay_records(pairs)
        fork = branches.FORK_STEP * 32
        self.assertEqual(len(records), 1920)
        self.assertEqual(len(records[fork:]), 54 * 32)
        self.assertEqual(records[fork]["draw_index"], 193)
        self.assertEqual(records[-1]["draw_index"], 1920)
        method = config["trainer"]["method_args"]
        self.assertEqual((method["alpha"], method["gamma"], method["top_k"], method["first_n"]), (2, 1, 5000, 3))
        self.assertEqual([event["step"] for event in epoch_events()],
                         [0, 6, 12, 18, 25, 31, 37, 43, 50, 56, 60])

    def test_dry_run_exposes_global_tokens_and_original_four_gpu_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = experiment.plan(SimpleNamespace(assets=experiment.ASSETS,
                study=Path(folder), model_path=None, retain_logs=None))
        self.assertEqual((plan["world_size"], plan["microbatch"], plan["accumulation"]), (4, 1, 8))
        self.assertEqual((plan["normalization"], plan["padding_scope"]), ("global", "microbatch"))
        self.assertEqual((plan["fork_after_updates"], plan["fork_consumed_pairs"]), (6, 192))
        self.assertEqual(plan["optimizer_updates_per_path"], 60)
        self.assertEqual(plan["actual_training_updates_executed"], 6 + 3 * 54)
        self.assertEqual({row["name"]: row["gamma"] for row in plan["arms"]}, branches.BRANCH_GAMMAS)

    def test_storage_accounts_for_full_adam_restart_and_three_final_models(self):
        model_bytes = 10_000_000
        prefix = experiment.storage_budget(model_bytes)
        self.assertEqual(prefix["estimated_fork_state_bytes"], 7 * model_bytes)
        self.assertEqual(prefix["final_weight_bytes"], 3 * model_bytes)
        self.assertEqual(prefix["remaining_weight_and_restart_bytes"], 10 * model_bytes)
        for index, arm in enumerate(("continue", "repair", "weak")):
            budget = experiment.storage_budget(model_bytes, arm)
            self.assertEqual(budget["remaining_weight_and_restart_bytes"], (3 - index) * model_bytes)
        with self.assertRaises(ValueError):
            experiment.storage_budget(model_bytes, "unknown")

    def test_missing_optimizer_rank_and_modified_sidecar_stop_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder)
            tensors = study / "fork_weights"
            tensors.mkdir()
            receipt = {"status": "ready", "fork_step": 6, "consumed_pairs": 192,
                       "world_size": 4, "rank_fingerprints": [{} for _ in range(4)], "files": {}}
            for rank in range(4):
                for name in (f"step6/zero_pp_rank_{rank}_mp_rank_00_optim_states.pt", f"rank-{rank}.pt"):
                    path = tensors / name
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(b"synthetic inventory fixture, not a model")
                    receipt["files"][name] = {"bytes": path.stat().st_size, "sha256": replay.sha(path)}
            self.assertEqual(branches.validate_fork_files(study, receipt), tensors.resolve())
            incomplete = copy.deepcopy(receipt)
            del incomplete["files"]["step6/zero_pp_rank_3_mp_rank_00_optim_states.pt"]
            with self.assertRaisesRegex(ValueError, "all four ZeRO optimizer shards"):
                branches.validate_fork_files(study, incomplete)
            (tensors / "rank-2.pt").write_bytes(b"changed RNG state")
            with self.assertRaisesRegex(ValueError, "changed or is missing"):
                branches.validate_fork_files(study, receipt)


class ResultAuditTests(unittest.TestCase):
    """Synthetic aggregate records; native per-sample validation is tested elsewhere."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.study = Path(temporary.name)
        frozen = self.study / "frozen"
        frozen.mkdir()
        (frozen / "schedule.pt").write_bytes(b"synthetic CPU fixture, not a Torch schedule")
        (frozen / "config.json").write_text("{}")
        config, pairs, _, provenance = replay.validate_assets()
        self.pairs = [{"step": index // 32 + 1, **record} for index, record in
                      enumerate(replay.replay_records(pairs))]
        self.write_rows(frozen / "pair_schedule.jsonl", self.pairs)
        events = epoch_events()
        experiment.write_json(self.study / "experiment.json", {
            "experiment": "seed103_four_gpu_global_token_step6_branches",
            "evaluation_steps": [event["step"] for event in events],
            "method": config["trainer"]["method_args"], "source_provenance": provenance})
        weights = self.study / "fork_weights"
        weights.mkdir()
        inventory = {}
        for rank in range(4):
            for name in (f"step6/zero_pp_rank_{rank}_mp_rank_00_optim_states.pt", f"rank-{rank}.pt"):
                path = weights / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"synthetic restart fixture")
                inventory[name] = {"bytes": path.stat().st_size, "sha256": replay.sha(path)}
        (self.study / "fork").mkdir()
        experiment.write_json(self.study / "fork/receipt.json", {
            "schema": 1, "status": "ready", "fork_step": 6, "consumed_pairs": 192, "world_size": 4,
            "schedule_sha256": replay.sha(frozen / "schedule.pt"),
            "config_sha256": replay.sha(frozen / "config.json"),
            "rank_fingerprints": [{} for _ in range(4)], "files": inventory})
        self.rows = {}
        for arm in experiment.ARMS:
            name = arm["name"]
            folder = self.study / name
            folder.mkdir()
            rows = [{**event, "raw_sensitive_loss": 1.0, "raw_retain_nll": .2,
                "raw_common_kl": .1, "loss_total": 1.6, "forget_quality": .5,
                "model_utility": .7, "ks_d": .1, "training_state_unchanged_all_ranks": True,
                "fixed_dataset_sha256": provenance["fixed_dataset_sha256"],
                "reference_sha256": provenance["reference_sha256"]} for event in events]
            self.rows[name] = rows
            self.write_rows(folder / "epoch_history.jsonl", rows)
            history = [{"step": step, "gamma_used": 1.0 if step <= 6 else arm["gamma"],
                        "alpha_used": 2.0, "raw_sensitive_loss": 1.0, "raw_retain_nll": .2,
                        "raw_common_kl": .1, "loss_total": 1.6} for step in range(1, 61)]
            self.write_rows(folder / "loss_history.jsonl", history)
            self.write_rows(folder / "training_pairs.jsonl", self.pairs)
            final = folder / "checkpoint-60"
            final.mkdir()
            (final / "model.safetensors").write_bytes(b"synthetic final fixture, not a research model")
            weight = {"model.safetensors": {"bytes": (final / "model.safetensors").stat().st_size,
                                          "sha256": replay.sha(final / "model.safetensors")}}
            checkpoints = [{"step": event["step"], "checkpoint": str(folder / f"checkpoint-{event['step']}"),
                            "files": copy.deepcopy(weight) if event["step"] else {},
                            "weights_deleted_after_verified_evaluation": event["step"] not in (0, 60)}
                           for event in events]
            experiment.write_json(folder / "manifest.json", {
                "status": "completed", "completed_updates": 60, "seed": 103,
                "schedule_sha256": replay.sha(frozen / "schedule.pt"),
                "loss_mode": "global", "padding_scope": "microbatch", "branch_gamma": arm["gamma"],
                "method": config["trainer"]["method_args"], "initial_parameter_sha256": "fixture-start",
                "teacher_parameter_sha256": "fixture-start", "checkpoints": checkpoints,
                "checkpoint_retention": "final_only_after_verified_epoch_evaluation",
                "schedule_metadata": {"pair_stream_sha256": "fixture-pairs"}})
            experiment.write_json(folder / "fork_restore_receipt.json", {
                "status": "verified", "fork_receipt_sha256": replay.sha(self.study / "fork/receipt.json"),
                "fork_step": 6, "completed_updates": 6, "consumed_pairs": 192,
                "arm": name, "branch_gamma": arm["gamma"], "rank_guards": [
                    {"rank": rank, "unchanged": True, "before_sha256": "same", "after_sha256": "same"}
                    for rank in range(4)]})

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def collect(self):
        def committed(folder, _manifest, checkpoint):
            return next(row for row in self.rows[Path(folder).name] if row["step"] == checkpoint["step"])
        with patch("checkpoint_retention.read_receipt", side_effect=committed):
            return experiment.collect_results(self.study)

    def test_complete_synthetic_records_are_collectable(self):
        epochs, online, manifests = self.collect()
        self.assertEqual(set(epochs), set(branches.BRANCH_GAMMAS))
        self.assertTrue(all(len(rows) == 11 for rows in epochs.values()))
        self.assertTrue(all(len(rows) == 60 for rows in online.values()))
        self.assertTrue(all(row["completed_updates"] == 60 for row in manifests.values()))

    def test_changed_suffix_gamma_is_rejected(self):
        path = self.study / "repair/loss_history.jsonl"
        rows = replay.read_rows(path)
        rows[6]["gamma_used"] = 1.0
        self.write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "gamma"):
            self.collect()

    def test_changed_pair_is_rejected(self):
        path = self.study / "weak/training_pairs.jsonl"
        rows = replay.read_rows(path)
        rows[192]["retain_index"] += 1
        self.write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "pair|Pair|F/R"):
            self.collect()

    def test_missing_intermediate_receipt_inventory_is_rejected(self):
        path = self.study / "continue/manifest.json"
        manifest = replay.read_json(path)
        manifest["checkpoints"] = [row for row in manifest["checkpoints"] if row["step"] != 12]
        experiment.write_json(path, manifest)
        with self.assertRaises(ValueError):
            self.collect()

    def test_empty_final_weight_inventory_is_rejected(self):
        path = self.study / "continue/manifest.json"
        manifest = replay.read_json(path)
        manifest["checkpoints"][-1]["files"] = {}
        experiment.write_json(path, manifest)
        with self.assertRaises(ValueError):
            self.collect()


class ResumeFromForkTests(unittest.TestCase):
    """Exercise recovery metadata and filesystem mutations with tiny fixtures."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name).resolve()
        self.study, self.code_root = base / "study", base / "current_code"
        self.study.mkdir()
        self.provenance = {"fixture": "same-seed103-source"}
        self.write_json("experiment.json", {
            "experiment": "seed103_four_gpu_global_token_step6_branches",
            "source_provenance": self.provenance})
        frozen = self.study / "frozen"
        frozen.mkdir()
        names = ("config.json", "schedule.pt", "retain_reference.json", "epoch_schedule.json", "pair_schedule.jsonl")
        for name in names:
            (frozen / name).write_bytes(("unchanged fixture " + name).encode())
        self.frozen_hashes = {name: replay.sha(frozen / name) for name in names}
        self.write_json("frozen/manifest.json", {
            "seed103_source_provenance": self.provenance, "frozen_file_sha256": self.frozen_hashes})
        self.fork = {"status": "ready", "fork_step": 6, "consumed_pairs": 192, "world_size": 4,
            "config_sha256": self.frozen_hashes["config.json"],
            "schedule_sha256": self.frozen_hashes["schedule.pt"],
            "initial_parameter_sha256": "same-initial", "teacher_parameter_sha256": "same-initial"}
        self.write_json("fork/receipt.json", self.fork)
        self.fork_sha = replay.sha(self.study / "fork/receipt.json")
        self.prefix = {**self.fork, "status": "prefix_completed", "completed_updates": 6,
                       "pending_pairs": 0, "fork_receipt_sha256": self.fork_sha}
        self.write_json("prefix/manifest.json", self.prefix)
        self.prefix_sha = replay.sha(self.study / "prefix/manifest.json")
        self.code_names = sorted(experiment.COMPATIBILITY_PATCH_FILES) + ["src/trainer/unlearn/palu.py"]
        self.old_bytes = {name: ("old fixture " + name).encode() for name in self.code_names}
        old_hashes = {}
        for name in self.code_names:
            current = self.code_root / name
            snapshot = self.study / "code_snapshot" / name
            current.parent.mkdir(parents=True, exist_ok=True)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(self.old_bytes[name])
            old_hashes[name] = replay.sha(snapshot)
            current.write_bytes(("compatibility patch " + name).encode()
                                if name in experiment.COMPATIBILITY_PATCH_FILES else self.old_bytes[name])
        self.old_hashes = old_hashes
        self.write_json("code_sha256.json", old_hashes)
        self.failed = {**self.fork, "status": "training", "completed_updates": 0}
        self.write_json("continue/manifest.json", self.failed)
        self.write_json("continue/resolved_config.json", {"fixture": "failed initialization config"})
        self.failed_manifest_bytes = (self.study / "continue/manifest.json").read_bytes()
        self.failed_config_bytes = (self.study / "continue/resolved_config.json").read_bytes()
        (self.study / "logs").mkdir()
        self.failed_logs = {}
        for suffix in ("train", "disk_before", "gpu_before"):
            name = f"continue_{suffix}.log"
            content = ("original failed-load evidence " + suffix).encode()
            (self.study / "logs" / name).write_bytes(content)
            self.failed_logs[name] = content

    def write_json(self, relative, value):
        path = self.study / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        experiment.write_json(path, value)

    def resume(self, fork_error=None):
        with patch.object(experiment, "ROOT", self.code_root), \
                patch.object(experiment, "code_paths", return_value=[self.code_root / name for name in self.code_names]), \
                patch.object(experiment.source, "validate_assets", return_value=(None, None, None, self.provenance)), \
                patch.object(experiment, "validate_fork_receipt", return_value=self.fork, side_effect=fork_error) as validator, \
                patch("builtins.print"):
            result = experiment.resume_from_fork(SimpleNamespace(study=self.study, assets=Path("fixture-assets")))
            validator.assert_called_once_with(self.study, require_tensors=True)
            return result

    def assert_not_mutated(self):
        self.assertTrue((self.study / "continue/manifest.json").is_file())
        self.assertFalse((self.study / "compatibility_audit").exists())
        self.assertFalse((self.study / "failed_load_attempts").exists())
        self.assertEqual(replay.read_json(self.study / "code_sha256.json"), self.old_hashes)

    def test_only_compatibility_sources_refresh_and_failed_attempt_is_preserved(self):
        receipt = self.resume()
        self.assertEqual(receipt["status"], "ready_to_resume_from_fork")
        self.assertEqual(set(receipt["changed_files"]), experiment.COMPATIBILITY_PATCH_FILES)
        self.assertEqual(receipt["branch_states_before"], {
            "continue": "failed_initialization_only", "repair": "new", "weak": "new"})
        self.assertFalse((self.study / "continue").exists())
        archived = self.study / receipt["quarantined_attempts"][0]
        self.assertEqual((archived / "manifest.json").read_bytes(), self.failed_manifest_bytes)
        self.assertEqual((archived / "resolved_config.json").read_bytes(), self.failed_config_bytes)
        for name, content in self.failed_logs.items():
            self.assertEqual((archived / "logs" / name).read_bytes(), content)
            self.assertEqual((self.study / "logs" / name).read_bytes(), content)
        audit, = (self.study / "compatibility_audit").iterdir()
        self.assertEqual(replay.read_json(audit / "previous_code_sha256.json"), self.old_hashes)
        for name in self.code_names:
            self.assertEqual((audit / "previous_code_snapshot" / name).read_bytes(), self.old_bytes[name])
            self.assertEqual((self.study / "code_snapshot" / name).read_bytes(), (self.code_root / name).read_bytes())
        self.assertEqual(replay.sha(self.study / "fork/receipt.json"), self.fork_sha)
        self.assertEqual(replay.sha(self.study / "prefix/manifest.json"), self.prefix_sha)
        self.assertEqual({name: replay.sha(self.study / "frozen" / name) for name in self.frozen_hashes}, self.frozen_hashes)

    def test_modified_frozen_input_is_rejected_before_any_mutation(self):
        (self.study / "frozen/schedule.pt").write_bytes(b"changed sample schedule")
        with self.assertRaisesRegex(ValueError, "Frozen scientific input"):
            self.resume()
        self.assert_not_mutated()

    def test_scientific_source_change_is_rejected_before_any_mutation(self):
        (self.code_root / "src/trainer/unlearn/palu.py").write_bytes(b"changed scientific objective")
        with self.assertRaisesRegex(ValueError, "outside the compatibility patch"):
            self.resume()
        self.assert_not_mutated()

    def test_modified_old_snapshot_is_rejected_before_any_mutation(self):
        (self.study / "code_snapshot/scripts/palu/seed103_branch_training.py").write_bytes(b"altered evidence")
        with self.assertRaisesRegex(ValueError, "Original code snapshot changed"):
            self.resume()
        self.assert_not_mutated()

    def test_any_restored_or_trained_suffix_is_preserved_and_rejected(self):
        for filename in ("fork_restore_receipt.json", "loss_history.jsonl"):
            with self.subTest(file=filename):
                path = self.study / "continue" / filename
                path.write_text("{}\n")
                with self.assertRaisesRegex(ValueError, "restored/partial training results"):
                    self.resume()
                self.assert_not_mutated()
                path.unlink()
        self.write_json("continue/manifest.json", {**self.failed, "completed_updates": 1})
        with self.assertRaisesRegex(ValueError, "restored/partial training results"):
            self.resume()
        self.assert_not_mutated()

    def test_only_complete_six_update_prefix_and_full_restart_can_resume(self):
        for key, invalid in (("completed_updates", 5), ("consumed_pairs", 200), ("pending_pairs", 8)):
            with self.subTest(field=key):
                self.write_json("prefix/manifest.json", {**self.prefix, key: invalid})
                with self.assertRaisesRegex(ValueError, "six-update, 192-pair prefix"):
                    self.resume()
                self.assert_not_mutated()
        self.write_json("prefix/manifest.json", self.prefix)
        with self.assertRaisesRegex(ValueError, "missing fork tensor"):
            self.resume(fork_error=ValueError("missing fork tensor"))
        self.assert_not_mutated()


def tiny_runtime(torch):
    """Ordinary CPU Adam, deliberately not a stand-in for native ZeRO restore."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)).double()
    model.config = SimpleNamespace(use_cache=False)
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    optimizer.micro_step_id = 0
    optimizer.is_gradient_accumulation_boundary = True
    optimizer.overflow = False
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: step / 6 if step < 6 else max(0.0, (60 - step) / 54))
    model.global_steps = model.global_samples = model.micro_steps = model.skipped_steps = 0
    model._is_gradient_accumulation_boundary = False
    return {"engine": model, "model": model, "reference": reference, "optimizer": optimizer,
            "scheduler": scheduler, "device": torch.device("cpu"), "rank": 0, "world": 1,
            "distributed": False, "warmup_steps": 6, "scheduler_total_updates": 60}


def advance_tiny(runtime, gamma, torch):
    import numpy as np
    # Consume all three RNG streams so a re-seed, setup-time draw or missing
    # restore changes the next actual parameter update, not only an audit hash.
    x = torch.randn(5, 3, dtype=torch.float64) + 0.01 * random.random() + 0.01 * np.random.random()
    model, optimizer = runtime["model"], runtime["optimizer"]
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    with torch.no_grad():
        reference = runtime["reference"](x)
    f = prediction[:, 0].square().mean()
    r = (prediction[:, 1] - 0.2).square().mean()
    common = (prediction - reference).square().mean()
    loss = gamma * f + 2 * r + 2 * common
    loss.backward()
    optimizer.step()
    runtime["scheduler"].step()
    model.global_steps += 1
    model.global_samples += 32
    model.micro_steps += 8
    return float(loss.detach())


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is installed in the server PALU environment")
class TinyStateTests(unittest.TestCase):
    def test_six_update_adam_scheduler_rng_and_next_update_survive_restore(self):
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(103)
        random.seed(103)
        np.random.seed(103)
        continuous = tiny_runtime(torch)
        for _ in range(6):
            advance_tiny(continuous, 1.0, torch)
        self.assertEqual(continuous["engine"].global_steps, 6)
        self.assertEqual(continuous["engine"].micro_steps, 48)
        self.assertEqual(continuous["engine"].global_samples, 192)
        self.assertEqual(continuous["scheduler"].last_epoch, 6)
        self.assertEqual(continuous["optimizer"].param_groups[0]["lr"], 2e-5)
        self.assertTrue(all(float(state["step"]) == 6 for state in continuous["optimizer"].state.values()))
        before = branches.training_fingerprint(continuous, torch)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "ordinary_cpu_adam.pt"
            torch.save({"model": continuous["model"].state_dict(),
                        "reference": continuous["reference"].state_dict(),
                        "optimizer": continuous["optimizer"].state_dict(),
                        "scheduler": continuous["scheduler"].state_dict(),
                        "sidecar": branches.rank_sidecar(continuous, torch)}, checkpoint)
            continued_loss = advance_tiny(continuous, 1.0, torch)
            expected_after = branches.training_fingerprint(continuous, torch)
            endpoints = {}
            for arm, gamma in branches.BRANCH_GAMMAS.items():
                resumed = tiny_runtime(torch)  # Fresh setup consumes Torch RNG.
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                resumed["model"].load_state_dict(state["model"])
                resumed["reference"].load_state_dict(state["reference"])
                resumed["optimizer"].load_state_dict(state["optimizer"])
                resumed["scheduler"].load_state_dict(state["scheduler"])
                # Emulate the native loader's omission of micro-step counters;
                # the actual experiment sidecar must restore those as well.
                self.assertEqual(resumed["engine"].micro_steps, 0)
                branches.restore_sidecar(state["sidecar"], resumed, torch)
                self.assertEqual(branches.training_fingerprint(resumed, torch), before)
                actual_loss = advance_tiny(resumed, gamma, torch)
                endpoints[arm] = torch.cat([p.detach().flatten() for p in resumed["model"].parameters()])
                if arm == "continue":
                    self.assertEqual(actual_loss, continued_loss)
                    self.assertEqual(branches.training_fingerprint(resumed, torch), expected_after)
                self.assertEqual(resumed["scheduler"].last_epoch, 7)
                self.assertEqual(resumed["engine"].global_samples, 224)
                self.assertTrue(all(float(item["step"]) == 7 for item in resumed["optimizer"].state.values()))
            self.assertFalse(torch.equal(endpoints["continue"], endpoints["repair"]))
            self.assertFalse(torch.equal(endpoints["repair"], endpoints["weak"]))
            self.assertFalse(torch.equal(endpoints["continue"], endpoints["weak"]))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("deepspeed"),
                     "Native TorchCheckpointEngine is installed in the server PALU environment")
class NativeCheckpointLoaderTests(unittest.TestCase):
    """Use the installed DS loader and enum; no ZeRO engine or GPU is created."""

    def setUp(self):
        import torch
        from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
        from deepspeed.runtime.zero.config import ZeroStageEnum
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.study = Path(temporary.name).resolve()
        self.folder = self.study / "fork_weights"
        (self.folder / "step6").mkdir(parents=True)
        self.payload = {"zero_stage": ZeroStageEnum.weights, "moment": torch.tensor([1.0, 2.0])}
        self.receipt = {"status": "ready", "tag": "step6", "fork_step": 6, "consumed_pairs": 192,
                        "world_size": 4, "rank_fingerprints": [{} for _ in range(4)], "files": {}}
        for rank in range(4):
            for name in (f"step6/zero_pp_rank_{rank}_mp_rank_00_optim_states.pt", f"rank-{rank}.pt"):
                target = self.folder / name
                torch.save(self.payload, target)
                self.receipt["files"][name] = {"bytes": target.stat().st_size, "sha256": replay.sha(target)}
        self.path = self.folder / "step6/zero_pp_rank_0_mp_rank_00_optim_states.pt"
        self.loader = TorchCheckpointEngine()
        self.runtime = {"engine": SimpleNamespace(checkpoint_engine=self.loader)}
        self.assertEqual(branches.validate_fork_files(self.study, self.receipt), self.folder.resolve())

    def test_real_native_enum_fails_by_default_and_only_owned_loader_is_relaxed(self):
        import torch
        from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
        from deepspeed.runtime.zero.config import ZeroStageEnum
        original = self.loader.load
        class_method = TorchCheckpointEngine.load
        torch_loader = torch.load
        with self.assertRaisesRegex(pickle.UnpicklingError, "ZeroStageEnum"):
            self.loader.load(str(self.path), map_location="cpu")
        with branches.trusted_native_checkpoint_loading(self.runtime, self.receipt, self.folder, torch):
            restored = self.loader.load(str(self.path), map_location="cpu")
            self.assertEqual(restored["zero_stage"], ZeroStageEnum.weights)
            torch.testing.assert_close(restored["moment"], self.payload["moment"])
            self.assertIs(TorchCheckpointEngine.load, class_method)
            self.assertIs(torch.load, torch_loader)
            with self.assertRaisesRegex(pickle.UnpicklingError, "ZeroStageEnum"):
                TorchCheckpointEngine().load(str(self.path), map_location="cpu")
            with self.assertRaisesRegex(pickle.UnpicklingError, "ZeroStageEnum"):
                torch.load(self.path, map_location="cpu")
        self.assertEqual(self.loader.load, original)
        with self.assertRaisesRegex(pickle.UnpicklingError, "ZeroStageEnum"):
            self.loader.load(str(self.path), map_location="cpu")

    def test_exception_exit_restores_the_original_loader(self):
        import torch
        original = self.loader.load
        with self.assertRaisesRegex(RuntimeError, "synthetic restore interruption"):
            with branches.trusted_native_checkpoint_loading(self.runtime, self.receipt, self.folder, torch):
                self.loader.load(str(self.path), map_location="cpu")
                raise RuntimeError("synthetic restore interruption")
        self.assertEqual(self.loader.load, original)
        with self.assertRaisesRegex(pickle.UnpicklingError, "ZeroStageEnum"):
            self.loader.load(str(self.path), map_location="cpu")

    def test_unlisted_path_and_changed_file_are_rejected_before_unpickling(self):
        import torch
        unlisted = self.folder / "step6/unlisted_model_states.pt"
        torch.save(self.payload, unlisted)
        outside = self.study / "outside_optim_states.pt"
        torch.save(self.payload, outside)
        for target in (unlisted, outside):
            with self.subTest(target=target.name), self.assertRaises((ValueError, RuntimeError, OSError)):
                with branches.trusted_native_checkpoint_loading(self.runtime, self.receipt, self.folder, torch):
                    self.loader.load(str(target), map_location="cpu")
        # A valid native payload with new tensor contents must fail its receipt
        # hash rather than being accepted because the filename is registered.
        torch.save({**self.payload, "moment": torch.tensor([3.0, 4.0])}, self.path)
        with self.assertRaises((ValueError, RuntimeError, OSError)):
            with branches.trusted_native_checkpoint_loading(self.runtime, self.receipt, self.folder, torch):
                self.loader.load(str(self.path), map_location="cpu")

    def test_direct_symlink_and_symlink_parent_are_rejected(self):
        import torch
        direct = self.folder / "step6/symlink_optim_states.pt"
        direct.symlink_to(self.path)
        linked_directory = self.folder / "linked_step6"
        linked_directory.symlink_to(self.folder / "step6", target_is_directory=True)
        self.receipt["files"][str(direct.relative_to(self.folder))] = copy.deepcopy(
            self.receipt["files"][str(self.path.relative_to(self.folder))])
        nested = linked_directory / self.path.name
        self.receipt["files"][str(nested.relative_to(self.folder))] = copy.deepcopy(
            self.receipt["files"][str(self.path.relative_to(self.folder))])
        for target in (direct, nested):
            with self.subTest(target=str(target)), self.assertRaises((ValueError, RuntimeError, OSError)):
                with branches.trusted_native_checkpoint_loading(self.runtime, self.receipt, self.folder, torch):
                    self.loader.load(str(target), map_location="cpu")


if __name__ == "__main__":
    unittest.main()
