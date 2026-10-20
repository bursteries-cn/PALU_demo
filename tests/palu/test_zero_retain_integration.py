"""Default/positive-R compatibility for the opt-in formal integration."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/palu"))
import gradient_equivalence_step as gradient
import layout_token_training as training
import forget_until_threshold as experiment


class PolicyTests(unittest.TestCase):
    def test_default_and_old_plan_preserve_backward(self):
        self.assertFalse(experiment.protocol()["zero_weight_retain_no_grad"])
        for config in ({}, {"forget_threshold_plan": {}}):
            for alpha in (0.0, 2.0):
                self.assertTrue(training.retain_backward_policy(config, None, alpha)["retain_requires_grad"])

    def test_only_explicit_frozen_flag_and_zero_coefficient_disable(self):
        plan = experiment.protocol(zero_weight_retain_no_grad=True)
        control = SimpleNamespace(plan=plan)
        config = {"forget_threshold_plan": plan}
        self.assertFalse(training.retain_backward_policy(config, control, 0.0)["retain_requires_grad"])
        self.assertTrue(training.retain_backward_policy(config, control, 1e-12)["retain_requires_grad"])
        self.assertTrue(training.retain_backward_policy(config, control, 2.0)["retain_requires_grad"])
        with self.assertRaisesRegex(ValueError, "exact frozen"):
            training.retain_backward_policy(config, SimpleNamespace(plan={}), 0.0)
        for value in (1, "true", None):
            with self.assertRaises(ValueError):
                experiment.protocol(zero_weight_retain_no_grad=value)
            with self.assertRaises(ValueError):
                training.retain_backward_policy({"forget_threshold_plan": {"zero_weight_retain_no_grad": value}}, control, 0.0)

    def test_cli_only_opt_in_changes_preparation(self):
        parser=experiment.parser()
        self.assertIsNone(parser.parse_args(["--mode", "prepare", "--study", "/tmp/new"]).zero_weight_retain_no_grad)
        self.assertTrue(parser.parse_args(["--mode", "prepare", "--study", "/tmp/new", "--zero-weight-retain-no-grad"]).zero_weight_retain_no_grad)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch required")
class ArithmeticTests(unittest.TestCase):
    def test_native_default_positive_retain_and_zero_retain_gradients(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(103)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(9, 9, dtype=torch.float64))
                self.calls = []
            def forward(self, input_ids, attention_mask, labels=None, use_cache=False):
                self.calls.append((labels is not None, torch.is_grad_enabled(), self.training))
                logits = self.weight[input_ids]
                loss = None if labels is None else torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1,9), labels[:,1:].flatten())
                return SimpleNamespace(logits=logits,loss=loss)
        model = Tiny()
        teacher = copy.deepcopy(model).requires_grad_(False).eval()
        inputs = {"forget": dict(input_ids=torch.tensor([[1,2,3,4]]), attention_mask=torch.ones(1,4), labels=torch.tensor([[-100,-200,2,3]])),
                  "retain": dict(input_ids=torch.tensor([[2,3,5,7]]), attention_mask=torch.ones(1,4), labels=torch.tensor([[-100,3,5,7]]))}
        counts = gradient.target_counts(inputs,8,3,torch)
        plan = experiment.protocol(zero_weight_retain_no_grad=True)
        config, control = {"forget_threshold_plan": plan}, SimpleNamespace(plan=plan)
        for alpha in (0.0, 2.0):
            collected=[]
            for option in ({}, {"retain_requires_grad":training.retain_backward_policy(config,control,alpha)["retain_requires_grad"]}):
                model.zero_grad(set_to_none=True)
                raw = gradient.raw_components(model,teacher,inputs,top_k=5,first_n=3,eos=8,torch=torch,**option)
                loss = gradient.weighted_loss(raw,counts,counts,gamma=1.,alpha=alpha,common=2.,normalization="global",accumulation=8,world=4)
                loss.backward()
                collected.append(({k:float(v.detach()) for k,v in raw.items()},model.weight.grad.clone(),raw["retain"].requires_grad))
            self.assertEqual(collected[0][0],collected[1][0])
            self.assertTrue(torch.equal(collected[0][1],collected[1][1]))
            self.assertEqual(collected[1][2],alpha != 0.)
        self.assertTrue(all(mode for _,_,mode in model.calls))

if __name__ == "__main__":
    unittest.main()
