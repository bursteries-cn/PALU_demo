"""CPU arithmetic/state-contract tests; these do not certify CUDA/ZeRO restore."""
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import random
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/palu"))
import forget_checkpoint_state as checkpoint


class InventoryTests(unittest.TestCase):
    def fixture(self, folder):
        files = {}
        for name in ["step3/mp_rank_00_model_states.pt",
                     *[f"step3/zero_pp_rank_{rank}_mp_rank_00_optim_states.pt" for rank in range(4)],
                     *[f"rank-{rank}.pt" for rank in range(4)]]:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture, not a real model")
            files[name] = {"bytes": path.stat().st_size, "sha256": checkpoint.gradient.file_sha(path)}
        return dict(status="ready", tag="step3", step=3, consumed_pairs=96, world_size=4,
                    rank_fingerprints=[{}] * 4, files=files)

    def test_missing_optimizer_rank_and_changed_sidecar_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            receipt = self.fixture(folder)
            self.assertEqual(checkpoint.validate_resume_files(folder, receipt), folder.resolve())
            incomplete = copy.deepcopy(receipt)
            del incomplete["files"]["step3/zero_pp_rank_3_mp_rank_00_optim_states.pt"]
            with self.assertRaisesRegex(ValueError, "four optimizer"):
                checkpoint.validate_resume_files(folder, incomplete)
            (folder / "rank-2.pt").write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "changed or is missing"):
                checkpoint.validate_resume_files(folder, receipt)

    def test_partial_update_and_wrong_world_are_rejected(self):
        runtime = dict(distributed=True, world=4, accumulation=8,
                       engine=SimpleNamespace(global_steps=3, micro_steps=25))
        with self.assertRaisesRegex(ValueError, "complete optimizer"):
            checkpoint._complete_boundary(runtime, 3)
        runtime["engine"].micro_steps = 24
        checkpoint._complete_boundary(runtime, 3)
        runtime["world"] = 1
        with self.assertRaisesRegex(ValueError, "four-rank"):
            checkpoint._complete_boundary(runtime, 3)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch required")
class ProbeTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        self.torch, self.np = torch, np
        torch.set_num_threads(1)
        class Tiny(torch.nn.Module):
            def __init__(self, offset=0):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(35, dtype=torch.float64).reshape(7, 5).square() / 11 + offset)
                self.dropout = torch.nn.Dropout(.7)
                self.calls = []
            def forward(self, input_ids, attention_mask, use_cache=False, labels=None):
                self.calls.append(tuple(input_ids.shape))
                # Consume all RNG streams to verify diagnostics preserve them.
                random.random()
                np.random.random()
                torch.rand(())
                logits = self.dropout(self.weight[input_ids])
                loss = None if labels is None else torch.nn.functional.cross_entropy(
                    logits[..., :-1, :].reshape(-1, 5), labels[..., 1:].reshape(-1))
                return SimpleNamespace(logits=logits, loss=loss)
        self.model, self.teacher = Tiny(), Tiny(.1).requires_grad_(False).eval()
        self.model.train()
        self.model.dropout.eval()  # Preserve mixed module modes, not only root.training.
        self.runtime = dict(engine=self.model, model=self.model, reference=self.teacher,
            rank=0, world=1, distributed=False, device=torch.device("cpu"),
            tokenizer=SimpleNamespace(eos_token_id=4, pad_token_id=0))
        samples = [
            dict(input_ids=torch.tensor([1, 2, 3]), labels=torch.tensor([-100, -200, 1])),
            dict(input_ids=torch.tensor([2, 3, 4, 5, 6]), labels=torch.tensor([-100, 1, 2, 3, 4])),
        ]
        self.schedule = dict(samples=dict(forget=samples), metadata=dict(
            fixed_dataset_sha256="fixed", fixed_target_counts=dict(sensitive=4)))
        self.method = dict(top_k=3, first_n=3, target_mode="mean")

    def test_token_weighting_matches_native_loss_and_preserves_rng_modes_weights(self):
        torch = self.torch
        rng = checkpoint.state.capture_rng(self.runtime, torch)
        modes = [(module, module.training) for model in (self.model, self.teacher) for module in model.modules()]
        before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        row = checkpoint.fixed_forget_probe(self.runtime, self.schedule, self.method, torch)
        self.assert_rng_equal(checkpoint.state.capture_rng(self.runtime, torch), rng)
        self.assertTrue(all(module.training == flag for module, flag in modes))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertTrue(all(torch.equal(parameter, before[name]) for name, parameter in self.model.named_parameters()))
        expected = []
        with checkpoint._evaluation_state(self.runtime, torch):
            for sample in self.schedule["samples"]["forget"]:
                inputs = {key: value.unsqueeze(0) for key, value in sample.items()}
                inputs["attention_mask"] = inputs["input_ids"].ne(0)
                paired = dict(forget=inputs, retain={**inputs, "labels": inputs["input_ids"].clamp(max=4)})
                native = checkpoint.gradient.raw_components(self.model, self.teacher, paired,
                    top_k=3, first_n=3, eos=4, torch=torch)["sensitive"]
                # Independent analytic PALU expression, including EOS exclusion.
                flat = self.model.weight[inputs["input_ids"]][:, :-1].reshape(-1, 5)
                top = self.teacher.weight[inputs["input_ids"]][:, :-1].reshape(-1, 5).topk(3, dim=-1).indices
                mse = (flat.gather(1, top) - flat.mean(-1, keepdim=True)).square().mean(-1)
                mask = checkpoint.gradient.sensitive_mask(inputs["labels"], 4, 3, torch).flatten()
                expected.append((float(mse[mask].mean()), int(mask.sum())))
                self.assertAlmostEqual(expected[-1][0], float(native))
        self.assertEqual(row["sensitive_token_count"], 4)
        self.assertEqual(row["forget_count"], 2)
        self.assertAlmostEqual(row["raw_sensitive_loss"], sum(loss * count for loss, count in expected) / 4)
        self.assertNotAlmostEqual(row["raw_sensitive_loss"], row["sample_mean_raw_sensitive_loss"])
        self.assertEqual(len(self.model.calls), 6)  # Probe 2; independent native comparison F/R 4.
        self.assertEqual(len(self.teacher.calls), 4)

    def assert_rng_equal(self, actual, expected):
        self.assertEqual(actual["python"], expected["python"])
        self.assertEqual(pickle.dumps(actual["numpy"]), pickle.dumps(expected["numpy"]))
        self.assertTrue(self.torch.equal(actual["torch_cpu"], expected["torch_cpu"]))

    def test_common_probe_reuses_forwards_and_matches_native_token_weighting(self):
        torch = self.torch
        # Include unequal common counts, zero-common and zero-sensitive examples.
        self.schedule["samples"]["forget"].append(dict(
            input_ids=torch.tensor([1, 2, 3, 4]),
            labels=torch.tensor([-100, -200, -200, 4])))
        self.schedule["metadata"]["fixed_target_counts"]["common"] = 3
        with torch.no_grad():
            self.teacher.weight.mul_(torch.tensor([.95, 1.05, .99, 1.01, 1.02]))
        for dtype in (torch.float64, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.model.to(dtype)
                self.teacher.to(dtype)
                self.model.calls.clear()
                self.teacher.calls.clear()
                rng = checkpoint.state.capture_rng(self.runtime, torch)
                row = checkpoint.fixed_forget_probe(self.runtime, self.schedule,
                    self.method, torch, include_common_kl=True)
                self.assert_rng_equal(checkpoint.state.capture_rng(self.runtime, torch), rng)
                self.assertTrue(self.model.training)
                self.assertFalse(self.model.dropout.training)
                self.assertFalse(self.teacher.training)
                self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
                self.assertEqual(len(self.model.calls), 3)
                self.assertEqual(len(self.teacher.calls), 3)
                self.assertEqual(row["sensitive_token_count"], 4)
                self.assertEqual(row["common_token_count"], 3)
                self.assertEqual(row["forget_zero_common_examples"], 1)
                self.assertEqual(row["forget_zero_sensitive_examples"], 1)
                expected_common = expected_sensitive = 0.0
                expected_common_macro = 0.0
                with checkpoint._evaluation_state(self.runtime, torch):
                    for index, sample in enumerate(self.schedule["samples"]["forget"]):
                        inputs = {key: value.unsqueeze(0) for key, value in sample.items()}
                        inputs["attention_mask"] = inputs["input_ids"].ne(0)
                        paired = dict(forget=inputs, retain={**inputs,
                            "labels": inputs["input_ids"].clamp(max=4)})
                        native = checkpoint.gradient.raw_components(self.model, self.teacher,
                            paired, top_k=3, first_n=3, eos=4, torch=torch)
                        recorded = row["fixed_loss_samples"][index]
                        self.assertEqual(recorded["dataset_index"], index)
                        self.assertEqual(recorded["split"], "forget")
                        self.assertIsNone(recorded["raw_retain_nll"])
                        self.assertIsNone(recorded["retain_token_count"])
                        self.assertEqual(recorded["raw_sensitive_loss"], float(native["sensitive"].double()))
                        self.assertEqual(recorded["raw_common_kl"], float(native["common"].double()))
                        expected_sensitive += recorded["raw_sensitive_loss"] * recorded["sensitive_token_count"]
                        expected_common += recorded["raw_common_kl"] * recorded["common_token_count"]
                        expected_common_macro += recorded["raw_common_kl"]
                self.assertEqual(row["raw_sensitive_loss"], expected_sensitive / 4)
                self.assertEqual(row["raw_common_kl"], expected_common / 3)
                self.assertEqual(row["sample_mean_raw_common_kl"], expected_common_macro / 3)

    def test_common_probe_handles_zero_targets_and_checks_frozen_count(self):
        torch = self.torch
        self.schedule["metadata"]["fixed_target_counts"]["common"] = 2
        with self.assertRaisesRegex(ValueError, "common target count"):
            checkpoint.fixed_forget_probe(self.runtime, self.schedule,
                self.method, torch, include_common_kl=True)
        for sample in self.schedule["samples"]["forget"]:
            sample["labels"][sample["labels"] == -200] = -100
        self.schedule["metadata"]["fixed_target_counts"]["common"] = 0
        row = checkpoint.fixed_forget_probe(self.runtime, self.schedule,
            self.method, torch, include_common_kl=True)
        self.assertEqual(row["raw_common_kl"], 0)
        self.assertEqual(row["common_token_count"], 0)
        self.assertEqual(row["forget_zero_common_examples"], 2)
        self.assertTrue(all(sample["raw_common_kl"] == 0 for sample in row["fixed_loss_samples"]))
        legacy = checkpoint.fixed_forget_probe(self.runtime, self.schedule, self.method, torch)
        self.assertNotIn("raw_common_kl", legacy)
        self.assertNotIn("fixed_loss_samples", legacy)

    def test_error_restores_rng_and_modes(self):
        torch = self.torch
        before = checkpoint.state.capture_rng(self.runtime, torch)
        with patch.object(checkpoint, "raw_forget_loss", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                checkpoint.fixed_forget_probe(self.runtime, self.schedule, self.method, torch)
        self.assert_rng_equal(checkpoint.state.capture_rng(self.runtime, torch), before)
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.dropout.training)
        self.assertFalse(self.teacher.training)

    def test_frozen_teacher_and_exact_pool_required(self):
        self.teacher.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "teacher must be frozen"):
            checkpoint.fixed_forget_probe(self.runtime, self.schedule, self.method, self.torch)
        self.teacher.requires_grad_(False)
        self.runtime.update(world=4, distributed=True)
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            checkpoint.fixed_forget_probe(self.runtime, self.schedule, self.method, self.torch)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch required")
class NativeContractMockTests(unittest.TestCase):
    """Fake native I/O with real CPU Adam/RNG; never a distributed GPU test."""

    def runtime(self):
        import torch
        model = torch.nn.Linear(2, 1).double()
        with torch.no_grad():
            model.weight.fill_(.25)
            model.bias.fill_(.1)
        model.config = SimpleNamespace(use_cache=False)
        teacher = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        optimizer.micro_step_id = 0
        optimizer.is_gradient_accumulation_boundary = True
        optimizer.overflow = False
        model.global_steps = model.global_samples = model.micro_steps = model.skipped_steps = 0
        model._is_gradient_accumulation_boundary = False
        model.checkpoint_engine = SimpleNamespace(
            load=lambda path, map_location=None: torch.load(path, map_location=map_location, weights_only=False))
        runtime = dict(engine=model, model=model, reference=teacher, optimizer=optimizer,
            scheduler=scheduler, rank=0, world=4, distributed=True, accumulation=8,
            device=torch.device("cpu"))

        def save(instance, folder, *, tag, client_state, save_latest):
            self.assertFalse(save_latest)
            folder = Path(folder)
            (folder / tag).mkdir()
            torch.save(dict(model=instance.state_dict(), scheduler=scheduler.state_dict(), client=client_state),
                       folder / tag / "mp_rank_00_model_states.pt")
            for rank in range(4):
                torch.save(optimizer.state_dict(), folder / tag / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")
                if rank:
                    torch.save(checkpoint.state.rank_sidecar(runtime, torch), folder / f"rank-{rank}.pt")
            random.random()
            torch.rand(())  # The helper must restore RNG consumed by save.
            return True

        def load(instance, folder, *, tag, load_module_strict, load_optimizer_states,
                 load_lr_scheduler_states, load_module_only):
            self.assertTrue(load_module_strict and load_optimizer_states and load_lr_scheduler_states)
            self.assertFalse(load_module_only)
            path = Path(folder) / tag
            saved = instance.checkpoint_engine.load(path / "mp_rank_00_model_states.pt", map_location="cpu")
            instance.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(instance.checkpoint_engine.load(
                path / "zero_pp_rank_0_mp_rank_00_optim_states.pt", map_location="cpu"))
            scheduler.load_state_dict(saved["scheduler"])
            return str(path), saved["client"]

        model.save_checkpoint = MethodType(save, model)
        model.load_checkpoint = MethodType(load, model)
        return runtime

    def advance(self, runtime):
        import numpy as np
        import torch
        x = torch.randn(3, 2, dtype=torch.float64) + np.random.random() + random.random()
        runtime["optimizer"].zero_grad(set_to_none=True)
        loss = runtime["model"](x).square().mean()
        loss.backward()
        runtime["optimizer"].step()
        runtime["scheduler"].step()
        runtime["engine"].global_steps += 1
        runtime["engine"].global_samples += 32
        runtime["engine"].micro_steps += 8
        return float(loss.detach())

    def test_save_restore_preserves_adam_scheduler_rng_and_next_update(self):
        import torch
        live = self.runtime()
        for _ in range(3):
            self.advance(live)
        manifest = dict.fromkeys(checkpoint.IDENTITY_KEYS, "synthetic-consistent-fingerprint")
        fingerprint = checkpoint.state.training_fingerprint(live, torch)
        # Emulate collective plumbing only; all four mock ranks share CPU state.
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(torch.distributed, "broadcast_object_list", side_effect=lambda *a, **k: None), \
                patch.object(checkpoint.state, "all_ranks", side_effect=lambda value, *a: [copy.deepcopy(value) for _ in range(4)]):
            folder = Path(temporary) / "native"
            receipt = checkpoint.save_resume_checkpoint(folder, live, manifest, 3, torch,
                probe={"raw_sensitive_loss": .48}, stop_reason="threshold")
            self.assertEqual(receipt["completed_updates"], 3)
            self.assertEqual(receipt["next_update"], 4)
            self.assertTrue((folder / "receipt.json").is_file())
            self.assertEqual(checkpoint.state.training_fingerprint(live, torch), fingerprint)
            expected_loss = self.advance(live)
            expected = checkpoint.state.training_fingerprint(live, torch)
            restored = self.runtime()
            checkpoint.load_resume_checkpoint(folder, restored, manifest, torch)
            self.assertEqual(checkpoint.state.training_fingerprint(restored, torch), fingerprint)
            self.assertEqual(self.advance(restored), expected_loss)
            self.assertEqual(checkpoint.state.training_fingerprint(restored, torch), expected)
            self.assertEqual(restored["engine"].global_steps, 4)
            self.assertEqual(restored["scheduler"].last_epoch, 4)


if __name__ == "__main__":
    unittest.main()
