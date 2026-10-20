"""CPU evidence for permanent sample masks; not a CUDA/ZeRO numerical benchmark."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/palu"))
import forget_until_threshold as experiment
import gradient_equivalence_step as gradient
import layout_token_training as training
import sample_forget_control as sample_control
from prepare_layout_schedule import canonical_sha


def plan(max_updates=13):
    return experiment.protocol(seed=103, max_updates=max_updates, learning_rate=5e-6,
        stopping_scope="sample_permanent", zero_weight_retain_no_grad=True)


def samples(values, counts):
    return [dict(dataset_index=i, split="forget", raw_sensitive_loss=value,
                 sensitive_token_count=count, raw_common_kl=.2, common_token_count=2)
            for i, (value, count) in enumerate(zip(values, counts))]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


class StateTests(unittest.TestCase):
    def test_upper_strict_lower_inclusive_and_zero_target_excluded(self):
        state = sample_control.SampleStopState([1, 2, 3, 0], plan())
        changes, summary = state.observe(7, samples([.5, .4, .399999, 0], [1, 2, 3, 0]))
        self.assertEqual([row["active"] for row in state.rows], [True, False, False, False])
        self.assertEqual([row["current_region"] for row in changes],
            ["above_or_equal_high", "in_band", "below_low", "not_applicable"])
        self.assertEqual(summary["eligible_samples"], 3)
        self.assertEqual(summary["ineligible_zero_sensitive_samples"], 1)
        self.assertEqual(summary["stopped_in_band"], 1)
        self.assertEqual(summary["stopped_below_low"], 1)
        self.assertIsNone(state.rows[3]["first_stop_step"])
        self.assertIsNone(state.decision(7, .45))  # Global mean is not the stopping rule.

    def test_rebound_does_not_reactivate_or_change_first_crossing(self):
        state = sample_control.SampleStopState([2, 1, 0], plan())
        state.observe(7, samples([.45, .8, 0], [2, 1, 0]))
        first = copy.deepcopy(state.rows[0])
        changes, summary = state.observe(13, samples([.9, .49, 0], [2, 1, 0]))
        self.assertFalse(state.rows[0]["active"])
        for key in ("first_stop_step", "first_stop_loss", "first_stop_below_low"):
            self.assertEqual(state.rows[0][key], first[key])
        self.assertTrue(changes[0]["stopped_sample_rebounded"])
        self.assertAlmostEqual(changes[0]["drift_from_stop"], .45)
        self.assertEqual(summary["inactive_above_or_equal_high"], 1)
        decision = state.decision(13, .76)
        self.assertEqual(decision["stop_reason"], "all_eligible_samples_deactivated")
        self.assertTrue(decision["band_hit"])
        self.assertGreater(decision["raw_sensitive_loss"], .5)

    def test_late_below_low_is_drift_not_a_rewritten_first_crossing(self):
        state = sample_control.SampleStopState([1, 1], plan())
        state.observe(7, samples([.45, .8], [1, 1]))
        _, summary = state.observe(13, samples([.1, .49], [1, 1]))
        self.assertEqual(summary["stopped_below_low"], 0)
        self.assertEqual(summary["current_below_low"], 1)
        self.assertEqual(summary["ever_below_low"], 1)
        self.assertTrue(state.decision(13, .295)["band_hit"])

    def test_budget_exhaustion_preserves_remaining_active_samples(self):
        state = sample_control.SampleStopState([1, 2], plan(max_updates=3))
        state.observe(3, samples([.3, .5], [1, 2]))
        decision = state.decision(3, .433333)
        self.assertEqual(decision["stop_reason"], "budget_exhausted")
        self.assertEqual(decision["active_samples"], 1)
        self.assertFalse(decision["threshold_reached"])
        self.assertFalse(decision["band_hit"])

    def test_all_overshot_still_stops_but_reports_missed_band(self):
        state = sample_control.SampleStopState([1, 2], plan())
        state.observe(7, samples([.39, .1], [1, 2]))
        decision = state.decision(7, .2)
        self.assertTrue(decision["threshold_reached"])
        self.assertFalse(decision["band_hit"])
        self.assertEqual(decision["stopped_below_low"], 2)
        self.assertEqual(decision["sample_stop_state_sha256"], canonical_sha(state.snapshot()))

    def test_snapshot_is_detached_from_mutable_controller_state(self):
        state = sample_control.SampleStopState([1, 0], plan())
        snapshot = state.snapshot()
        snapshot["rows"][0]["active"] = False
        self.assertTrue(state.rows[0]["active"])

    def test_invalid_probes_are_rejected_atomically(self):
        good = samples([.45, .7, 0], [1, 2, 0])
        cases = [good[:-1], good + [good[0]], [good[0], good[0], good[2]]]
        for key, value in (("raw_sensitive_loss", float("nan")),
                           ("raw_sensitive_loss", float("inf")),
                           ("raw_sensitive_loss", -.1), ("raw_sensitive_loss", True),
                           ("sensitive_token_count", 99), ("sensitive_token_count", True),
                           ("split", "retain"), ("dataset_index", True)):
            bad = copy.deepcopy(good)
            bad[1][key] = value
            cases.append(bad)
        for bad in cases:
            with self.subTest(probe=bad):
                state = sample_control.SampleStopState([1, 2, 0], plan())
                before = state.snapshot()
                with self.assertRaises((ValueError, TypeError)):
                    state.observe(7, bad)
                self.assertEqual(state.snapshot(), before)

    def test_probe_order_and_decision_must_match_observed_state(self):
        state = sample_control.SampleStopState([1], plan())
        for step in (0, 1, 13, 14):
            with self.subTest(step=step), self.assertRaises(ValueError):
                state.observe(step, samples([.7], [1]))
        with self.assertRaises(ValueError):
            state.decision(13, .7)
        state.observe(7, samples([.7], [1]))
        for step in (7, 8):
            with self.subTest(step=step), self.assertRaises(ValueError):
                state.observe(step, samples([.7], [1]))
        for step, loss in ((13, .7), (7, float("nan")), (7, -1)):
            with self.subTest(step=step, loss=loss), self.assertRaises(ValueError):
                state.decision(step, loss)

    def test_invalid_target_counts_cannot_create_a_vacuous_success(self):
        for counts in ([], [0, 0], [1, -1], [True, 1], [1.0, 2]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                sample_control.SampleStopState(counts, plan())


class ControllerTests(unittest.TestCase):
    def test_initialization_binds_eligibility_to_frozen_200_samples(self):
        controller = sample_control.SampleForgetControl(plan())
        counts = [3] * 196 + [0] * 4
        schedule = dict(samples={"forget": [dict(labels=[n]) for n in counts]},
            metadata=dict(fixed_target_counts={"sensitive": sum(counts)}))
        runtime = dict(rank=0, world=4, distributed=False, tokenizer=SimpleNamespace(eos_token_id=99))
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), layout="four", padding_scope="microbatch",
                online_evaluation=True, normalization="global")
            manifest = {}
            with patch.object(sample_control, "selected_sensitive_count", side_effect=lambda labels, eos, first_n: labels[0]):
                controller.initialize(runtime, manifest, {}, schedule, args, None)
            self.assertEqual(controller.state.summary()["eligible_samples"], 196)
            self.assertEqual(controller.state.summary()["ineligible_zero_sensitive_samples"], 4)
            self.assertEqual(manifest["threshold_plan"], controller.plan)
            self.assertEqual(manifest["sample_stop_state_sha256"], canonical_sha(controller.state.snapshot()))
            self.assertEqual(controller.forget_weight(195), 1.)
            self.assertEqual(controller.forget_weight(196), 0.)

    def test_initialize_rejects_unsupported_layout_or_changed_target_counts(self):
        runtime = dict(rank=0, world=4, distributed=False, tokenizer=SimpleNamespace(eos_token_id=99))
        schedule = dict(samples={"forget": [dict(labels=[1])] * 200},
            metadata=dict(fixed_target_counts={"sensitive": 200}))
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), layout="four", padding_scope="microbatch",
                online_evaluation=True, normalization="global")
            for failure in ("world", "layout", "padding", "count"):
                rt, sc, ar = copy.deepcopy(runtime), copy.deepcopy(schedule), copy.deepcopy(args)
                if failure == "world": rt["world"] = 1
                elif failure == "layout": ar.layout = "single"
                elif failure == "padding": ar.padding_scope = "update"
                else: sc["metadata"]["fixed_target_counts"]["sensitive"] = 201
                controller = sample_control.SampleForgetControl(plan())
                with self.subTest(failure=failure), patch.object(sample_control, "selected_sensitive_count", return_value=1):
                    with self.assertRaises(ValueError):
                        controller.initialize(rt, {}, {}, sc, ar, None)
            self.assertFalse((Path(tmp) / "sample_stop_state.json").exists())

    def test_resume_identity_must_preserve_sample_mask_and_its_hash(self):
        from forget_checkpoint_state import IDENTITY_KEYS
        from forget_threshold_control import validate_resume_identity
        state = sample_control.SampleStopState([1, 2], plan())
        state.observe(7, samples([.45, .49], [1, 2]))
        manifest = {key: f"original-{key}" for key in IDENTITY_KEYS}
        manifest.update(threshold_plan=plan(), sample_stop_state=state.snapshot(),
            sample_stop_state_sha256=canonical_sha(state.snapshot()))
        receipt = {key: manifest[key] for key in IDENTITY_KEYS}
        receipt.update(step=7, completed_updates=7, consumed_pairs=224, next_update=8,
            sample_stop_state=state.snapshot(), sample_stop_state_sha256=canonical_sha(state.snapshot()))
        validate_resume_identity(receipt, manifest, 7)
        for field in ("sample_stop_state", "sample_stop_state_sha256"):
            for missing in (False, True):
                bad = copy.deepcopy(receipt)
                if missing: del bad[field]
                else: bad[field] = "different"
                with self.subTest(field=field, missing=missing), self.assertRaisesRegex(ValueError, "sample stopping state"):
                    validate_resume_identity(bad, manifest, 7)

    def test_epoch_hooks_activate_mask_only_for_following_update(self):
        controller = sample_control.SampleForgetControl(plan())
        controller.state = sample_control.SampleStopState([1, 2, 0], controller.plan)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp))
            manifest = dict(evaluation_steps=[0, 13], final_model_step=13)
            online = SimpleNamespace(events={0: dict(step=0), 416: dict(step=13)})
            checked = []
            for step in range(1, 14):
                before = controller.training_row_metadata(step)
                self.assertEqual(before["sample_mask_observed_after_step"], 0 if step <= 7 else 7)
                self.assertEqual(controller.forget_weight(0), 1.0 if step <= 7 else 0.0)
                values = [.45, .8, 0] if step == 7 else [.9, .39, 0]
                fixed = dict(raw_sensitive_loss=(values[0]+2*values[1])/3,
                    fixed_loss_samples=samples(values, [1, 2, 0]), raw_common_kl=.2)
                with patch("forget_checkpoint_state.fixed_forget_probe", return_value=fixed) as probe:
                    stopped = controller.after_update(dict(rank=0, distributed=False), manifest,
                        {"trainer": {"method_args": {}}}, {}, args, step, online, None)
                if probe.called:
                    checked.append(step)
                self.assertEqual(stopped, step == 13)
            self.assertEqual(checked, [7, 13])
            self.assertEqual(controller.forget_weight(0), 0.)
            self.assertEqual(controller.forget_weight(1), 0.)
            self.assertEqual(manifest["final_model_step"], 13)
            self.assertEqual(manifest["evaluation_steps"], [0, 13])
            self.assertEqual(online.events[416]["evaluation_kind"], "forget_stop")
            self.assertEqual(online.fixed_forget_probe_result[0], 13)
            self.assertEqual(len(json.loads((args.output / "sample_stop_state.json").read_text())["rows"]), 3)
            self.assertEqual(len((args.output / "sample_stop_history.jsonl").read_text().splitlines()), 6)
            self.assertEqual(controller.last_probe["sample_masked_sensitive_loss_next_update"], 0.)

    def test_rank_mask_disagreement_prevents_persisting_inconsistent_state(self):
        controller = sample_control.SampleForgetControl(plan())
        controller.state = sample_control.SampleStopState([1], controller.plan)
        def gather(result, value):
            result[:] = [value, value, "different", value]
        torch = SimpleNamespace(distributed=SimpleNamespace(all_gather_object=gather))
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {}
            with self.assertRaisesRegex(AssertionError, "Ranks disagree"):
                controller.persist_state(dict(rank=0, distributed=True, world=4), manifest,
                    SimpleNamespace(output=Path(tmp)), torch)
            self.assertEqual(manifest, {})
            self.assertFalse((Path(tmp) / "sample_stop_state.json").exists())

    def test_pair_index_mapping_and_binary_mask_guard(self):
        window = dict(pair_records=[dict(forget_index=19), dict(forget_index=3)])
        controller = SimpleNamespace(forget_weight=lambda index: float(index == 3))
        self.assertEqual(training.microbatch_forget_weight(controller, window, [0]), 0.)
        self.assertEqual(training.microbatch_forget_weight(controller, window, [1]), 1.)
        self.assertEqual(training.microbatch_forget_weight(None, window, [0, 1]), 1.)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            training.microbatch_forget_weight(controller, window, [0, 1])
        for weight in (-1, .5, float("nan"), True):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                training.microbatch_forget_weight(SimpleNamespace(forget_weight=lambda _: weight), window, [0])


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch required")
class ArithmeticTests(unittest.TestCase):
    def test_32_unequal_samples_match_direct_token_objective_and_gradient(self):
        import torch
        torch.set_num_threads(1)
        counts = [dict(sensitive=0 if i == 7 else 1+i%7, retain=2+i%5, common=1+i%9) for i in range(32)]
        totals = {key: sum(row[key] for row in counts) for key in counts[0]}
        for mode in ("all_active", "selected", "all_inactive"):
            with self.subTest(mode=mode):
                x = torch.tensor([.3, -.7, .2], dtype=torch.float64, requires_grad=True)
                masks = [float(mode == "all_active" or (mode == "selected" and i%3 == 0)) for i in range(32)]
                raw, micros, distributed_parts, direct_parts = [], [], [], []
                for microstep in range(8):
                    for rank in range(4):
                        i = microstep*4+rank
                        # Each component has different per-token derivatives and token counts.
                        sums = {key: sum(((x[j] + .03*(i+k+1))**2 for k in range(counts[i][key])),
                                         x[j]*0.) for j, key in enumerate(totals)}
                        components = {key: sums[key]/max(counts[i][key], 1) for key in totals}
                        distributed_parts.append(gradient.weighted_loss(components, counts[i], totals,
                            gamma=masks[i], alpha=0., common=2., normalization="global", accumulation=8, world=4)/32)
                        direct_parts.append(masks[i]*sums["sensitive"]/totals["sensitive"] + 2*sums["common"]/totals["common"])
                        raw.append(components)
                        micros.append(dict(rank=rank, microstep=microstep, pair_indices=[i], counts=counts[i],
                            sample_forget_weight=masks[i], raw={key: float(value.detach()) for key, value in components.items()}))
                observed, expected = sum(distributed_parts), sum(direct_parts)
                self.assertTrue(torch.allclose(observed, expected, rtol=1e-14, atol=1e-14))
                observed_grad = torch.autograd.grad(observed, x, retain_graph=True)[0]
                expected_grad = torch.autograd.grad(expected, x)[0]
                self.assertTrue(torch.allclose(observed_grad, expected_grad, rtol=1e-14, atol=1e-14))
                self.assertEqual(float(observed_grad[1]), 0.)  # R is logging only.
                self.assertGreater(float(observed_grad[2]), 0.)  # Common KL survives even all F inactive.
                if mode == "all_inactive":
                    self.assertEqual(float(observed_grad[0]), 0.)
                token_raw = {key: sum(float(raw[i][key].detach())*counts[i][key] for i in range(32))/totals[key] for key in totals}
                diagnostic = training.loss_record(token_raw, 1., 0., 2.)
                row = dict(diagnostic, global_target_counts=totals, window_token_mean=copy.deepcopy(diagnostic))
                training.masked_objective_record(row, micros, 1., 0., 2.)
                self.assertAlmostEqual(row["loss_total"], float(expected.detach()), places=13)
                self.assertEqual(row["raw_sensitive_loss"], diagnostic["raw_sensitive_loss"])
                self.assertEqual(row["unmasked_objective_loss"], diagnostic["loss_total"])
                active = sum(counts[i]["sensitive"]*masks[i] for i in range(32))
                self.assertEqual(row["active_sensitive_tokens_in_update"], int(active))
                self.assertAlmostEqual(row["active_sensitive_token_fraction"], active/totals["sensitive"])
                self.assertEqual(row["window_token_mean"]["loss_total"], row["loss_total"])
                if mode == "all_active":
                    self.assertAlmostEqual(row["loss_total"], diagnostic["loss_total"], places=14)
                elif mode == "selected":
                    renormalized = row["masked_sensitive_loss"]*totals["sensitive"]/active
                    self.assertGreater(renormalized, row["masked_sensitive_loss"])


class ReplayValidationTests(unittest.TestCase):
    def fixture(self, root):
        frozen = plan()
        counts = [1, 2, 3, 0]
        state = sample_control.SampleStopState(counts, frozen)
        losses, pairs, probes, fixed_samples, transitions = [], [], [], [], []
        for step in range(1, 14):
            micros = [dict(rank=i%4, microstep=i//4, pair_indices=[i], forget_dataset_index=i%4,
                sample_forget_weight=float(state.rows[i%4]["active"]), counts=dict(sensitive=counts[i%4], retain=2, common=3),
                raw=dict(sensitive=.6+.02*(i%4), retain=1.2, common=.2)) for i in range(32)]
            totals = {key: sum(m["counts"][key] for m in micros) for key in micros[0]["counts"]}
            token_raw = {key: sum(m["raw"][key]*m["counts"][key] for m in micros)/totals[key] for key in totals}
            diagnostic = training.loss_record(token_raw, 1., 0., 2.)
            row = dict(diagnostic, step=step, global_target_counts=totals, window_token_mean=copy.deepcopy(diagnostic),
                sample_stop_state_sha256=canonical_sha(state.snapshot()), sample_mask_observed_after_step=state.step,
                microsteps=micros, **state.summary())
            training.masked_objective_record(row, micros, 1., 0., 2.)
            losses.append(row)
            pairs.extend(dict(step=step, forget_index=i%4) for i in range(32))
            if step in (7, 13):
                current = samples([.45, .7, .3, 0] if step == 7 else [.9, .4, .2, 0], counts)
                changes, summary = state.observe(step, current)
                transitions.extend(changes)
                fixed_samples.extend(dict(r, step=step) for r in current)
                aggregate = sum(r["raw_sensitive_loss"]*r["sensitive_token_count"] for r in current)/sum(counts)
                probes.append(dict(step=step, raw_sensitive_loss=aggregate,
                    sample_masked_sensitive_loss_next_update=sum(r["last_loss"]*r["sensitive_token_count"]
                        for r in state.rows if r["active"])/sum(counts),
                    sample_stop_state=state.snapshot(), sample_stop_summary=summary))
        decision = state.decision(13, probes[-1]["raw_sensitive_loss"])
        manifest = dict(sample_stop_state=state.snapshot(), sample_stop_state_sha256=canonical_sha(state.snapshot()))
        write_rows(root / "training_pairs.jsonl", pairs)
        write_rows(root / "forget_probe_samples.jsonl", fixed_samples)
        write_rows(root / "sample_stop_history.jsonl", transitions)
        (root / "sample_stop_state.json").write_text(json.dumps(state.snapshot()))
        return manifest, frozen, losses, probes, decision

    def test_replay_accepts_next_update_mask_and_drift_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_control.validate_sample_run(root, *self.fixture(root))

    def test_tampered_mask_objective_state_or_termination_is_rejected(self):
        for kind in ("premature_mask", "wrong_sample", "weight", "objective", "missing_pair", "duplicate_micro", "token_count", "probe_state", "decision", "history", "final_hash",
                     "raw_forget", "raw_retain", "raw_common", "unmasked_objective", "next_masked_probe"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest, frozen, losses, probes, decision = self.fixture(root)
                if kind == "premature_mask": losses[6]["sample_mask_observed_after_step"] = 7
                elif kind == "wrong_sample": losses[7]["microsteps"][0]["forget_dataset_index"] = 1
                elif kind == "weight": losses[7]["microsteps"][0]["sample_forget_weight"] = 1.
                elif kind == "objective": losses[7]["loss_total"] += .01
                elif kind == "missing_pair":
                    rows = [json.loads(line) for line in (root / "training_pairs.jsonl").read_text().splitlines()]
                    write_rows(root / "training_pairs.jsonl", rows[1:])
                elif kind == "duplicate_micro": losses[7]["microsteps"][0]["pair_indices"] = [1]
                elif kind == "token_count": losses[7]["global_target_counts"]["sensitive"] += 1
                elif kind == "probe_state": probes[0]["sample_stop_state"]["rows"][0]["active"] = True
                elif kind == "decision": decision["band_hit"] = not decision["band_hit"]
                elif kind == "history": (root / "sample_stop_history.jsonl").write_text("")
                elif kind == "raw_forget": losses[7]["raw_sensitive_loss"] += .01
                elif kind == "raw_retain": losses[7]["raw_retain_nll"] += .01
                elif kind == "raw_common": losses[7]["raw_common_kl"] += .01
                elif kind == "unmasked_objective": losses[7]["unmasked_objective_loss"] += .01
                elif kind == "next_masked_probe": probes[0]["sample_masked_sensitive_loss_next_update"] += .01
                else: manifest["sample_stop_state_sha256"] = "wrong"
                with self.assertRaises(ValueError):
                    sample_control.validate_sample_run(root, manifest, frozen, losses, probes, decision)


if __name__ == "__main__":
    unittest.main()
