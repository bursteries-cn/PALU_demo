"""Phase/sampling invariants without model weights, CUDA or optional packages."""
import copy
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
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

    def test_common_kl_can_remain_active_across_the_complete_update_boundary(self):
        plan = experiment.protocol(forget_common_kl_weight=2)
        control = experiment.PhaseControl(plan)
        self.assertEqual(control.coefficients_for_step(1), (1., 0., 2.))
        self.assertEqual(control.coefficients_for_step(63), (1., 0., 2.))
        self.assertEqual(control.coefficients_for_step(64), (0., 2., 2.))
        self.assertEqual(control.coefficients_for_step(126), (0., 2., 2.))
        self.assertEqual(plan['forget_common_kl_weight'], 2)
        self.assertEqual(experiment.objective_label(plan), 'F + 2common_KL -> 2R + 2common_KL')
        self.assertEqual(experiment.objective_label(experiment.protocol()), 'F -> 2R + 2common_KL')
        for value in (-1, float('nan'), float('inf'), -float('inf')):
            with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                experiment.protocol(forget_common_kl_weight=value)

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


class PairedStudyTests(unittest.TestCase):
    def fixture(self, folder):
        baseline, study = Path(folder)/'baseline', Path(folder)/'new'
        frozen = baseline/'frozen'
        frozen.mkdir(parents=True)
        plan = experiment.protocol(forget_common_kl_weight=2)
        baseline_plan = experiment.protocol()
        baseline_plan.pop('forget_common_kl_weight')  # the historical run predates this field
        config = {'sequential_phase_plan': plan, 'model': {'path': '/original/full/model'},
                  'trainer': {'method_args': {'gamma': 1, 'alpha': 2},
                              'args': {'output_dir': str(study/'run'), 'seed': 103}}}
        baseline_config = copy.deepcopy(config)
        baseline_config['sequential_phase_plan'] = baseline_plan
        baseline_config['trainer']['args']['output_dir'] = str(baseline/'run')
        keys = ('seed', 'updates', 'effective_batch', 'total_training_pairs', 'data_sizes',
                'eos_token_id', 'first_n', 'pair_stream_sha256', 'tensor_schedule_sha256',
                'fixed_dataset_sha256', 'fixed_target_counts', 'collator', 'normalization')
        payload = {'metadata': dict.fromkeys(keys, 'unchanged'), 'windows': []}
        for step in range(1, 127):
            pairs = [{'draw_index': (step-1)*32+i+1, 'forget_index': i, 'retain_index': step+i}
                     for i in range(32)]
            payload['windows'].append(dict(step=step, window_sha256=f'window-{step}',
                pair_sha256=f'pair-{step}', pair_records=pairs, tensor_hashes={'tensor': f'tensor-{step}'},
                counts={'sensitive': 32, 'retain': 64, 'common': 96}, sample_lengths=[3]*32,
                global_padding_shapes={'forget': [32, 3], 'retain': [32, 3]}))
        payload['metadata']['epoch_evaluation_schedule'] = experiment.evaluation_events(plan)
        reference = Path(folder)/'retain_reference.json'
        source.write_json(reference, {'scores': [1., 2.]})
        source.write_json(frozen/'retain_reference.json', {'scores': [1., 2.]})
        source.write_json(frozen/'config.json', baseline_config)
        source.write_json(frozen/'schedule.pt', payload)  # fake CPU loader keeps guard tests dependency-free
        source.write_json(frozen/'epoch_schedule.json', experiment.evaluation_events(plan))
        rows = [{'step': w['step'], **p} for w in payload['windows'] for p in w['pair_records']]
        (frozen/'pair_schedule.jsonl').write_text(''.join(experiment.json.dumps(p)+'\n' for p in rows))
        source.write_json(baseline/'experiment.json', dict(baseline_plan, status='prepared'))
        # A stopped run is still a valid paired source: all future inputs were frozen beforehand.
        (baseline/'run').mkdir()
        source.write_json(baseline/'run/manifest.json', {'status': 'stopped', 'completed_updates': 108})
        self.rehash(frozen)
        loader = SimpleNamespace(load=lambda path, **kwargs: source.read_json(path))
        return baseline, study, plan, config, payload, reference, loader

    def rehash(self, frozen):
        names = ('config.json', 'schedule.pt', 'retain_reference.json', 'epoch_schedule.json', 'pair_schedule.jsonl')
        source.write_json(frozen/'manifest.json', {'frozen_file_sha256': {n: source.sha(frozen/n) for n in names}})

    def test_stopped_baseline_matches_all_126_updates_without_modification(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(folder)
            before = source.read_json(args[0]/'frozen/manifest.json')
            receipt = experiment.verify_paired_study(*args)
            self.assertEqual(receipt['matched_updates'], 126)
            self.assertEqual(receipt['matched_pairs'], 4032)
            self.assertEqual(receipt['baseline_forget_common_kl_weight'], 0)
            self.assertEqual(receipt['forget_common_kl_weight'], 2)
            self.assertEqual(before, source.read_json(args[0]/'frozen/manifest.json'))
            self.assertTrue(all(source.sha(args[0]/'frozen'/n) == v
                                for n, v in before['frozen_file_sha256'].items()))
            self.assertFalse(args[1].exists())

    def test_incompatible_baselines_fail_before_new_study_is_written(self):
        for mismatch in ('learning_rate', 'method', 'fixed_data', 'reference', 'epoch',
                         'last_window', 'last_pair', 'incomplete_schedule', 'changed_file'):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as folder:
                args = self.fixture(folder)
                baseline, study, plan, config, payload, reference, loader = args
                frozen = baseline/'frozen'
                if mismatch == 'learning_rate':
                    plan['learning_rate'] = 2e-5
                elif mismatch == 'method':
                    config['trainer']['method_args']['alpha'] = 3
                elif mismatch == 'fixed_data':
                    payload['metadata']['fixed_dataset_sha256'] = 'other-data'
                elif mismatch == 'reference':
                    source.write_json(reference, {'scores': [3., 4.]})
                elif mismatch == 'epoch':
                    events = source.read_json(frozen/'epoch_schedule.json')
                    events[-1]['step'] -= 1
                    source.write_json(frozen/'epoch_schedule.json', events)
                    self.rehash(frozen)
                elif mismatch == 'last_window':
                    payload['windows'][-1]['tensor_hashes']['tensor'] = 'changed'
                elif mismatch == 'last_pair':
                    payload['windows'][-1]['pair_records'].reverse()
                elif mismatch == 'incomplete_schedule':
                    payload['windows'].pop()
                elif mismatch == 'changed_file':
                    (frozen/'pair_schedule.jsonl').write_text('{}\n')
                with self.assertRaisesRegex(ValueError, 'Paired baseline'):
                    experiment.verify_paired_study(*args)
                self.assertFalse(study.exists())


if __name__ == '__main__':
    unittest.main()
