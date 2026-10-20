"""CPU correctness/contracts only; no CUDA/ZeRO equivalence claim."""
import copy
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/palu"))
import zero_retain_backward_check as check
import gradient_equivalence_step as gradient
import layout_token_training as training


class InstrumentationTests(unittest.TestCase):
    def test_training_has_exactly_one_pre_optimizer_hook(self):
        _, source = check.instrument_training(training.train, object())
        self.assertEqual(source.count("check_recorder.before_optimizer(runtime, step)"), 1)
        self.assertEqual(source.count("engine.step()"), 2)
        self.assertIn("with check_recorder.update(runtime, int(window['step'])):", source)

    def test_components_add_only_one_no_grad_scope(self):
        _, baseline = check.instrument_components(gradient.raw_components, "baseline")
        _, modified = check.instrument_components(gradient.raw_components, "no_grad")
        self.assertEqual(modified.count("with torch.no_grad():"), baseline.count("with torch.no_grad():") + 1)
        self.assertIn("retain_loss = model(**retain).loss", modified)
        with self.assertRaises(ValueError):
            check.instrument_components(gradient.raw_components, "detach")

    def test_differences_are_not_silently_toleranced(self):
        self.assertEqual(check.differing_paths({"x": [1.0]}, {"x": [1.0]}), [])
        self.assertEqual(check.differing_paths({"x": [1.0]}, {"x": [1.0 + 1e-14]}), ["/x/0"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch required")
class TensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(1)

    def test_native_losses_gradients_adam_and_rng_match(self):
        torch = self.torch
        torch.manual_seed(103)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(9, 6, dtype=torch.float64)
                self.projection = torch.nn.Linear(6, 9, dtype=torch.float64)
                self.grad_enabled_calls = []
            def forward(self, input_ids, attention_mask, use_cache=False, labels=None):
                self.grad_enabled_calls.append((labels is not None, torch.is_grad_enabled(), self.training))
                logits = self.projection(self.embedding(input_ids))
                loss = None if labels is None else torch.nn.functional.cross_entropy(
                    logits[..., :-1, :].reshape(-1, 9), labels[..., 1:].reshape(-1))
                return SimpleNamespace(logits=logits, loss=loss)
        model, teacher = Tiny(), Tiny().requires_grad_(False).eval()
        initial = copy.deepcopy(model.state_dict())
        samples = dict(forget=dict(input_ids=torch.tensor([[1, 2, 3, 4]]),
                                  attention_mask=torch.ones(1, 4), labels=torch.tensor([[-100, -200, 2, 3]])),
                       retain=dict(input_ids=torch.tensor([[2, 4, 5, 7]]),
                                  attention_mask=torch.ones(1, 4), labels=torch.tensor([[-100, 4, 5, 7]])))
        rng = torch.random.get_rng_state().clone()
        arms = []
        for arm in ("baseline", "no_grad"):
            model.load_state_dict(initial)
            model.grad_enabled_calls.clear()
            torch.random.set_rng_state(rng)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            function, _ = check.instrument_components(gradient.raw_components, arm)
            records = []
            for step in range(3):
                optimizer.zero_grad()
                components = function(model, teacher, samples, top_k=5, first_n=3, eos=8, torch=torch)
                counts = gradient.target_counts(samples, 8, 3, torch)
                total = gradient.weighted_loss(components, counts, counts, gamma=1, alpha=0, common=2,
                                               normalization="global", accumulation=8, world=4)
                (total / 32).backward()
                records.append(dict(losses={key: float(value.detach()) for key, value in components.items()},
                                    gradient=[parameter.grad.clone() for parameter in model.parameters()]))
                self.assertEqual(components["retain"].requires_grad, arm == "baseline")
                optimizer.step()
            arms.append(dict(records=records, weights=copy.deepcopy(model.state_dict()),
                             optimizer=copy.deepcopy(optimizer.state_dict()), rng=torch.random.get_rng_state(),
                             calls=copy.deepcopy(model.grad_enabled_calls)))
        stream = check.TensorStream(torch)
        for key in ("records", "weights", "optimizer", "rng"):
            self.assertEqual(stream.state(key, arms[0][key]), stream.state(key, arms[1][key]))
        self.assertTrue(all(grad and mode for retain, grad, mode in arms[0]["calls"] if retain))
        self.assertTrue(all(not grad and mode for retain, grad, mode in arms[1]["calls"] if retain))
        self.assertTrue(all(grad and mode for retain, grad, mode in arms[1]["calls"] if not retain))

    def test_binary_snapshot_reports_full_numeric_difference(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as temp:
            prefix = Path(temp) / "snapshot"
            baseline = check.TensorStream(torch, prefix=prefix)
            original = baseline.tensor("x", torch.tensor([1., 2., 3.], dtype=torch.bfloat16))
            baseline.close()
            candidate = check.TensorStream(torch, reference=prefix)
            changed = candidate.tensor("x", torch.tensor([1., 2.5, 3.], dtype=torch.bfloat16))
            candidate.close()
            self.assertNotEqual(original["sha256"], changed["sha256"])
            self.assertEqual(candidate.comparisons["x"]["max_abs_diff"], .5)
            self.assertEqual(candidate.comparisons["x"]["l2_diff"], .5)
            self.assertEqual(candidate.comparisons["x"]["unequal_elements"], 1)

    def test_empty_state_tensor_is_supported_and_nonfinite_rejected(self):
        stream = check.TensorStream(self.torch)
        self.assertEqual(stream.tensor("empty", self.torch.tensor([]))["numel"], 0)
        with self.assertRaisesRegex(AssertionError, "Nonfinite"):
            stream.tensor("nan", self.torch.tensor([float("nan")]))

    def test_adam_moments_cannot_pass_as_empty_after_update(self):
        torch = self.torch
        model = torch.nn.Linear(2, 2)
        model.config = SimpleNamespace(use_cache=False)
        master = torch.zeros(6)
        underlying = SimpleNamespace(state={}, param_groups=[dict(params=[], lr=1e-5)])
        optimizer = SimpleNamespace(optimizer=underlying, fp32_partitioned_groups_flat=[master])
        runtime = dict(engine=model, model=model, reference=model, optimizer=optimizer,
                       scheduler=SimpleNamespace(state_dict=lambda: dict(last_epoch=1)),
                       device=torch.device("cpu"))
        with patch.object(torch.cuda, "get_rng_state", return_value=torch.zeros(4, dtype=torch.uint8)):
            check.state_fingerprint(runtime, check.TensorStream(torch), completed_updates=0)
            with self.assertRaisesRegex(AssertionError, "Missing persistent Adam"):
                check.state_fingerprint(runtime, check.TensorStream(torch), completed_updates=1)
            underlying.state[master] = dict(step=1, state1=torch.zeros(6), state2=torch.zeros(6))
            result = check.state_fingerprint(runtime, check.TensorStream(torch), completed_updates=1)
            self.assertEqual(result["optimizer"]["moment_coverage"][0]["moment_numel"], {"state1": 6, "state2": 6})
            underlying.state[master]["state2"] = torch.zeros(3)
            with self.assertRaisesRegex(AssertionError, "Incomplete persistent Adam"):
                check.state_fingerprint(runtime, check.TensorStream(torch), completed_updates=1)
            underlying.state[master] = dict(step=2, exp_avg=torch.zeros(6), exp_avg_sq=torch.zeros(6))
            with self.assertRaisesRegex(AssertionError, "Adam step differs"):
                check.state_fingerprint(runtime, check.TensorStream(torch), completed_updates=1)

    def test_gradient_uses_real_local_accumulator_and_rejects_missing(self):
        torch = self.torch
        parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
        parameter.ds_tensor = torch.tensor([1., 2.])
        parameter.ds_shape, parameter.ds_numel = torch.Size([8]), 8
        model = SimpleNamespace(named_parameters=lambda: [("w", parameter)])
        runtime = dict(distributed=True, world=4, model=model,
                       optimizer=SimpleNamespace(fp32_partitioned_groups_flat=[torch.zeros(2)]))
        self.assertIsNone(parameter.grad)
        result = check.gradient_fingerprint(runtime, check.TensorStream(torch), lambda p: torch.tensor([3., 4.]))
        self.assertEqual(result["w"]["partition"]["l2_norm"], 5.0)
        with self.assertRaisesRegex(AssertionError, "Missing real"):
            check.gradient_fingerprint(runtime, check.TensorStream(torch), lambda p: None)
        with self.assertRaisesRegex(AssertionError, "Incomplete"):
            check.gradient_fingerprint(runtime, check.TensorStream(torch), lambda p: torch.ones(1))


if __name__ == "__main__":
    unittest.main()
