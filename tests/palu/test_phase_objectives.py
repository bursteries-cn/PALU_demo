"""Phase coefficients and LR guards; Torch checks do not certify GPU arithmetic."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/palu"))
import gradient_equivalence_step as gradient
import layout_token_training as training


class ObjectiveTests(unittest.TestCase):
    def test_native_default_retains_historical_objective(self):
        raw = {"sensitive": 3.0, "retain": 5.0, "common": 7.0}
        counts = dict.fromkeys(raw, 1)
        value = gradient.weighted_loss(raw, counts, counts, gamma=1, alpha=2,
                                      normalization="native", accumulation=4, world=1)
        self.assertEqual(value, 27)
        self.assertEqual(training.loss_record(raw, 1, 2)["loss_total"], value)

    def test_inactive_components_are_still_recorded(self):
        raw = {"sensitive": 3.0, "retain": 5.0, "common": 7.0}
        forget = training.loss_record(raw, 1, 0, 0)
        repair = training.loss_record(raw, 0, 2, 0)
        self.assertEqual(forget["loss_total"], 3)
        self.assertEqual(repair["loss_total"], 10)
        self.assertEqual(repair["loss_forget"], 0)
        self.assertEqual(repair["loss_common_kl"], 0)
        for row in (forget, repair):
            self.assertEqual((row["raw_sensitive_loss"], row["raw_retain_nll"], row["raw_common_kl"]), (3, 5, 7))

    def test_independent_coefficients_reconstruct_global_token_means(self):
        parts = [({"sensitive": 3., "retain": 4., "common": 8.},
                  {"sensitive": 2, "retain": 3, "common": 1}),
                 ({"sensitive": 5., "retain": 2., "common": 6.},
                  {"sensitive": 1, "retain": 1, "common": 3})]
        totals = {key: sum(counts[key] for _, counts in parts) for key in parts[0][0]}
        for gamma, alpha, common in ((1, 0, 0), (0, 2, 2), (0, 2, 0), (1, 2, 2)):
            observed = sum(gradient.weighted_loss(raw, counts, totals,
                gamma=gamma, alpha=alpha, common=common, normalization="global", accumulation=2, world=1) / 2
                for raw, counts in parts)
            means = {key: sum(raw[key] * counts[key] for raw, counts in parts) / totals[key] for key in totals}
            self.assertAlmostEqual(observed, gamma * means["sensitive"] + alpha * means["retain"] + common * means["common"])

    def test_legacy_and_phase_hooks_are_distinct(self):
        method = {"gamma": 1, "alpha": 2}
        legacy = SimpleNamespace(gamma_for_step=lambda step: 0 if step > 6 else 1)
        phases = SimpleNamespace(coefficients_for_step=lambda step: (1, 0, 0) if step <= 63 else (0, 2, 2))
        self.assertEqual(training.objective_coefficients(None, 1, method), (1, 2, 2))
        self.assertEqual(training.objective_coefficients(legacy, 7, method), (0, 2, 2))
        self.assertEqual(training.objective_coefficients(phases, 63, method), (1, 0, 0))
        self.assertEqual(training.objective_coefficients(phases, 64, method), (0, 2, 2))
        for bad in ((-1, 0, 0), (0, float("nan"), 1), (1, 0, float("inf"))):
            with self.assertRaises(ValueError):
                training.objective_coefficients(SimpleNamespace(coefficients_for_step=lambda step: bad), 1, method)

    def test_historical_budget_is_default_and_new_budget_is_explicit(self):
        self.assertEqual(training.planned_update_budget(None), 60)
        self.assertEqual(training.planned_update_budget(SimpleNamespace()), 60)
        self.assertEqual(training.planned_update_budget(SimpleNamespace(planned_updates=126)), 126)
        for bad in (0, -1, 62.5, True, "126"):
            with self.assertRaises(ValueError):
                training.planned_update_budget(SimpleNamespace(planned_updates=bad))


class LearningRateTests(unittest.TestCase):
    def test_constant_rate_stays_constant_across_switch_and_endpoint(self):
        runtime = {"warmup_steps": 0, "scheduler_total_updates": 126}
        args = {"learning_rate": 1e-5, "lr_scheduler_type": "constant"}
        for step in (0, 1, 63, 64, 126):
            training.assert_learning_rate([1e-5, 1e-5], step, runtime, args)
        with self.assertRaises(AssertionError):
            training.assert_learning_rate([0], 126, runtime, args)
        with self.assertRaises(ValueError):
            training.assert_learning_rate([1e-5], 0, dict(runtime, warmup_steps=6), args)

    def test_linear_schedule_still_matches_historical_warmup_and_decay(self):
        runtime = {"warmup_steps": 6, "scheduler_total_updates": 60}
        args = {"learning_rate": 2e-5, "lr_scheduler_type": "linear"}
        for step, value in ((0, 0), (3, 1e-5), (6, 2e-5), (33, 1e-5), (60, 0)):
            training.assert_learning_rate([value], step, runtime, args)
        with self.assertRaises(AssertionError):
            training.assert_learning_rate([2e-5], 60, runtime, args)
        with self.assertRaises(ValueError):
            training.assert_learning_rate([2e-5], 60, runtime, dict(args, lr_scheduler_type="cosine"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable; GPU equivalence is not established")
class GradientTests(unittest.TestCase):
    def test_unequal_microbatch_tokens_match_full_objective_in_both_phases(self):
        import torch
        counts = [dict(sensitive=1 + i % 3, retain=2 + i % 2, common=1 + i % 4) for i in range(8)]
        targets = [{key: torch.arange(count, dtype=torch.float64) + i / 10
                    for key, count in row.items()} for i, row in enumerate(counts)]
        totals = {key: sum(row[key] for row in counts) for key in counts[0]}
        for coefficients in ((1, 0, 0), (0, 2, 2), (0, 2, 0)):
            x = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
            for values, local in zip(targets, counts):
                raw = {key: ((x - tokens) ** 2).mean() for key, tokens in values.items()}
                loss = gradient.weighted_loss(raw, local, totals, gamma=coefficients[0], alpha=coefficients[1],
                    common=coefficients[2], normalization="global", accumulation=2, world=4)
                (loss / 8).backward()  # accumulation2 and four-rank gradient mean
            expected_x = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
            expected = sum(coefficient * ((expected_x - torch.cat([row[key] for row in targets])) ** 2).mean()
                for key, coefficient in zip(("sensitive", "retain", "common"), coefficients))
            expected.backward()
            torch.testing.assert_close(x.grad, expected_x.grad, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
