"""Phase/sampling invariants without model weights, CUDA or optional packages."""
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/palu'))
import layout_loss_curve_experiment as source
import sequential_forget_retain as experiment


class SequentialTests(unittest.TestCase):
    def test_switch_is_after_complete_update_and_never_blends_objectives(self):
        plan = experiment.protocol()
        control = experiment.PhaseControl(plan)
        self.assertEqual((plan['forget_updates'], plan['retain_updates']), (63, 63))
        self.assertEqual(plan['actual_forget_phase_epochs'], 10.08)
        self.assertEqual(control.coefficients_for_step(63), (1., 0., 0.))
        self.assertEqual(control.coefficients_for_step(64), (0., 2., 2.))
        self.assertEqual(control.training_row_metadata(64)['phase_epoch'], .16)
        pure = experiment.PhaseControl(experiment.protocol(common_kl=False))
        self.assertEqual(pure.coefficients_for_step(64), (0., 2., 0.))

    def test_evaluations_have_no_path_collisions_and_keep_pending_gradients(self):
        plan = experiment.protocol()
        events = experiment.evaluation_events(plan)
        self.assertEqual([e['step'] for e in events], [0, 12, 25, 37, 50, 62, 63, 75, 88, 100, 113, 125, 126])
        self.assertEqual(len(events), len({e['step'] for e in events}))
        self.assertEqual(events[1]['pending_pairs'], 16)
        boundary = next(e for e in events if e['step'] == 63)
        self.assertEqual((boundary['pending_pairs'], boundary['training_phase']), (0, 'forget'))
        self.assertEqual(events[-1]['step'], plan['total_updates'])
        self.assertEqual(events[-1]['pending_pairs'], 0)
        for e in events:
            self.assertEqual(e['updated_pairs'] + e['pending_pairs'], e['consumed_pairs'])
            self.assertEqual(e['consumed_pairs'] % 4, 0)  # reachable four-rank microstep

    def test_extension_preserves_historical_pairs_and_finishes_partial_epoch(self):
        _, pairs, _, _ = source.validate_assets()
        prefix = source.replay_records(pairs)
        state = random.getstate()
        first = experiment.extend_records(prefix, 4032)
        self.assertEqual(state, random.getstate())
        self.assertEqual(first, experiment.extend_records(prefix, 4032))
        self.assertEqual(first[:1920], prefix)
        for start in range(0, 4000, 200):
            self.assertEqual(sorted(p['forget_index'] for p in first[start:start+200]), list(range(200)))
        self.assertEqual([p['draw_index'] for p in first], list(range(1, 4033)))
        self.assertTrue(all(0 <= p['retain_index'] < 3800 for p in first))

    def test_retain_clock_and_short_phases_are_explicit(self):
        plan = experiment.protocol(retain_epoch_size=3800)
        self.assertEqual(plan['retain_updates'], 1188)
        for plan in (plan, experiment.protocol(2, 2), experiment.protocol(.1, .1), experiment.protocol(20, 10)):
            events = experiment.evaluation_events(plan)
            self.assertEqual(len(events), len({e['step'] for e in events}))
            self.assertEqual(events[-1]['step'], plan['total_updates'])
            self.assertTrue(any(e['step'] == plan['forget_updates'] for e in events))
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                experiment.protocol(learning_rate=value)

    def test_mutating_visible_phase_plan_cannot_change_training(self):
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder)
            (study/'frozen').mkdir()
            plan = experiment.protocol()
            source.write_json(study/'frozen/config.json', {'sequential_phase_plan': plan})
            source.write_json(study/'frozen/manifest.json', {'frozen_file_sha256': {
                'config.json': source.sha(study/'frozen/config.json')}})
            source.write_json(study/'experiment.json', dict(plan, status='prepared'))
            self.assertEqual(experiment.frozen_plan(study), plan)
            altered = dict(plan, forget_updates=62)
            source.write_json(study/'experiment.json', altered)
            with self.assertRaisesRegex(ValueError, 'differs from frozen'):
                experiment.frozen_plan(study)


if __name__ == '__main__':
    unittest.main()
