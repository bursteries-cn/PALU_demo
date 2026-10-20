"""Early stopping must preserve the final state and not evaluate it twice."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/palu"))
import forget_threshold_control as control
import layout_token_training as training


PLAN = dict(experiment="forget_threshold", seed=104, total_updates=63,
            threshold_low=.4, threshold_high=.5, learning_rate=1e-5)


class StoppingTests(unittest.TestCase):
    def test_crossing_and_exhaustion_are_separate_outcomes(self):
        self.assertIsNone(control.stop_decision(25, .54, PLAN))
        for value in (.4, .45, .5):
            row = control.stop_decision(26, value, PLAN)
            self.assertEqual(row["stop_reason"], "threshold_reached")
            self.assertTrue(row["band_hit"])
        row = control.stop_decision(26, .39, PLAN)
        self.assertEqual(row["stop_reason"], "threshold_overshot")
        self.assertTrue(row["threshold_reached"])
        self.assertFalse(row["band_hit"])
        row = control.stop_decision(63, .6, PLAN)
        self.assertEqual(row["stop_reason"], "budget_exhausted")
        self.assertFalse(row["threshold_reached"])
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                control.stop_decision(1, value, PLAN)

    def exercise_stop(self, stop_step, loss):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            runtime = dict(rank=0, distributed=False)
            manifest = dict(evaluation_steps=[0, 13, 25, 38, 50, 63], final_model_step=63)
            online = SimpleNamespace(events={n * 32: dict(step=n, pending_pairs=0)
                                             for n in manifest["evaluation_steps"]})
            obj = control.ForgetThresholdControl(PLAN)
            # Advance the controller to a chosen stopping update without GPU work.
            for step in range(1, stop_step + 1):
                measured = loss if step == stop_step else .7
                with patch("forget_checkpoint_state.fixed_forget_probe",
                           return_value=dict(raw_sensitive_loss=measured, elapsed_seconds=.1)):
                    stopped = obj.after_update(runtime, manifest, {"trainer": {"method_args": {}}},
                        {}, SimpleNamespace(output=output), step, online, None)
                self.assertEqual(stopped, step == stop_step)
            # Pruning will see the actual final step BEFORE saving this event.
            self.assertEqual(manifest["final_model_step"], stop_step)
            self.assertEqual(manifest["evaluation_steps"][-1], stop_step)
            self.assertEqual(sum(e["step"] == stop_step for e in online.events.values()), 1)
            self.assertTrue(all(n <= stop_step * 32 for n in online.events))
            self.assertEqual(online.events[stop_step * 32]["pending_pairs"], 0)
            self.assertEqual(online.events[stop_step * 32]["evaluation_kind"], "forget_stop")
            rows = [json.loads(x) for x in (output / "forget_probe_history.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in rows], list(range(1, stop_step + 1)))
            return manifest

    def test_stop_at_existing_event_replaces_instead_of_duplicating_it(self):
        manifest = self.exercise_stop(25, .49)
        self.assertEqual(manifest["evaluation_steps"], [0, 13, 25])

    def test_stop_between_events_adds_exact_final_evaluation(self):
        manifest = self.exercise_stop(27, .47)
        self.assertEqual(manifest["evaluation_steps"], [0, 13, 25, 27])

    def test_epoch_rule_skips_intermediate_probes_and_uses_strict_upper_boundary(self):
        from forget_until_threshold import protocol
        plan = protocol(seed=104, probe_interval="epoch")
        obj = control.ForgetThresholdControl(plan)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            manifest = dict(evaluation_steps=[0, 13, 25, 38, 50, 63], final_model_step=63)
            online = SimpleNamespace(events={s * 32: dict(step=s) for s in manifest["evaluation_steps"]})
            checked = []
            for step in range(1, 33):
                # Exactly 0.5 does not trigger the new strict rule at step25.
                loss = .39 if step == 32 else .5 if step == 25 else .7
                with patch("forget_checkpoint_state.fixed_forget_probe",
                           return_value=dict(raw_sensitive_loss=loss, elapsed_seconds=.1)) as probe:
                    stopped = obj.after_update(dict(rank=0, distributed=False), manifest,
                        {"trainer": {"method_args": {}}}, {},
                        SimpleNamespace(output=output), step, online, None)
                    if probe.called:
                        checked.append(step)
                self.assertEqual(stopped, step == 32)
            self.assertEqual(checked, [7, 13, 19, 25, 32])
            rows = [json.loads(x) for x in (output / "forget_probe_history.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in rows], checked)
            self.assertEqual(manifest["evaluation_steps"], [0, 13, 25, 32])
            self.assertEqual(obj.decision["stop_reason"], "threshold_overshot")
            self.assertFalse(obj.decision["band_hit"])
            self.assertEqual(rows[-1]["actual_data_epochs"], 5.12)
            with self.assertRaisesRegex(ValueError, "probe schedule"):
                control.stop_decision(28, .49, plan)

    def test_epoch_budget_end_is_checked_even_before_first_epoch(self):
        from forget_until_threshold import protocol
        plan = protocol(max_updates=3, probe_interval="epoch")
        self.assertEqual(control.stop_decision(3, .5, plan)["stop_reason"], "budget_exhausted")
        self.assertEqual(control.stop_decision(3, .4, plan)["stop_reason"], "threshold_reached")
        self.assertEqual(control.stop_decision(3, .399, plan)["stop_reason"], "threshold_overshot")

    def test_budget_exhaustion_still_marks_and_retains_final_model(self):
        manifest = self.exercise_stop(63, .6)
        self.assertEqual(manifest["stop_decision"]["stop_reason"], "budget_exhausted")
        self.assertFalse(manifest["early_stopped"])

    def test_old_controllers_get_no_new_probe_or_collective(self):
        for old in (None, SimpleNamespace()):
            self.assertFalse(training.after_complete_update(old, {}, {}, {}, {}, None, 1, None, None))

    def test_disagreement_across_ranks_is_rejected(self):
        def gather(results, value):
            results[:] = [value, not value, value, value]
        torch = SimpleNamespace(distributed=SimpleNamespace(all_gather_object=gather))
        obj = SimpleNamespace(after_update=lambda *args: True)
        with self.assertRaisesRegex(AssertionError, "Ranks disagree"):
            training.after_complete_update(obj, dict(distributed=True, world=4),
                {}, {}, {}, None, 1, None, torch)

    def test_native_archive_must_belong_to_this_stop_and_configuration(self):
        from forget_checkpoint_state import IDENTITY_KEYS
        manifest = {key: f"original-{key}" for key in IDENTITY_KEYS}
        receipt = dict(manifest, step=27, completed_updates=27,
                       consumed_pairs=864, next_update=28)
        control.validate_resume_identity(receipt, manifest, 27)
        for key in (*IDENTITY_KEYS, "step", "completed_updates", "consumed_pairs", "next_update"):
            wrong = dict(receipt)
            wrong[key] = "different" if key in IDENTITY_KEYS else receipt[key] + 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "stopping state"):
                control.validate_resume_identity(wrong, manifest, 27)

    def test_final_evaluation_read_error_is_broadcast_before_checkpoint_collectives(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = control.ForgetThresholdControl(PLAN)
            obj.decision = control.stop_decision(27, .47, PLAN)
            manifest = dict(completed_updates=27, evaluated_steps=[0, 13, 25, 27],
                            evaluation_steps=[0, 13, 25, 27])
            broadcasts = []
            torch = SimpleNamespace(distributed=SimpleNamespace(
                broadcast_object_list=lambda payload, src: broadcasts.append(payload[0])))
            with patch("forget_checkpoint_state.save_resume_checkpoint") as save:
                with self.assertRaisesRegex(AssertionError, "Final evaluation validation failed"):
                    obj.finish(dict(rank=0, distributed=True), manifest,
                               SimpleNamespace(output=Path(tmp)), torch)
                save.assert_not_called()
            self.assertEqual(len(broadcasts), 1)
            self.assertIn("Final evaluation validation failed", broadcasts[0])


if __name__ == "__main__":
    unittest.main()
