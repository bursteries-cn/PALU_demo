import ast
import inspect
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/palu'))
import layout_token_training as training
import gradient_equivalence_step as gradient
from profile_seed103_training import instrument_training, instrument_components
from analyze_training_profile import duration, intersection_duration, trace_summary


class RemoveProfilingScopes(ast.NodeTransformer):
    def visit_With(self, node):
        node = self.generic_visit(node)
        if any(isinstance(item.context_expr, ast.Call)
               and ast.unparse(item.context_expr.func).startswith('profile_recorder.') for item in node.items):
            return node.body
        return node


class InstrumentationTests(unittest.TestCase):
    def assert_only_contexts_added(self, function, source):
        original = ast.parse(inspect.getsource(function))
        stripped = RemoveProfilingScopes().visit(ast.parse(source))
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(stripped, include_attributes=False))

    def test_training_arithmetic_control_flow_and_logging_unchanged(self):
        _, source, scopes = instrument_training(training.train, None)
        self.assert_only_contexts_added(training.train, source)
        for name in ('inputs_and_target_counts', 'loss_construction', 'backward_including_recompute',
                     'optimizer_step', 'logging_and_scalar_sync'):
            self.assertGreater(scopes[name], 0)

    def test_three_model_calls_and_loss_arithmetic_unchanged(self):
        _, source = instrument_components(gradient.raw_components, None)
        self.assert_only_contexts_added(gradient.raw_components, source)
        for name in ('student_forget_forward', 'teacher_forward', 'student_retain_forward'):
            self.assertEqual(source.count("profile_recorder.phase('" + name + "')"), 1)

    def test_changed_training_structure_is_rejected(self):
        def empty_function():
            return None
        with self.assertRaises((AssertionError, IndentationError)):
            instrument_training(empty_function, None)


class TimelineTests(unittest.TestCase):
    def test_overlapping_stream_intervals_are_counted_once(self):
        self.assertEqual(duration([(0, 10), (5, 15), (30, 40), (4, 8)]), 25)
        self.assertEqual(duration([]), 0)
        self.assertEqual(intersection_duration([(0,10),(5,15)],[(8,12),(20,30)]),4)

    def test_communication_overlap_and_launch_phase_are_separate(self):
        def event(name, category, start, length, tid=1, **args):
            return dict(ph='X', name=name, cat=category, ts=start, dur=length,
                        tid=tid, pid=1, args=args)
        events = [event('PALU/update', 'user_annotation', 0, 100),
                  event('PALU/student_forget_forward', 'user_annotation', 0, 50),
                  event('aten::fake_collective', 'cpu_op', 5, 1, **{'External id': 1}),
                  event('aten::fake_matmul', 'cpu_op', 15, 1, **{'External id': 2}),
                  event('nccl_AllGather', 'kernel', 10, 20, tid=7, **{'External id': 1}),
                  event('gemm', 'kernel', 20, 20, tid=8, **{'External id': 2})]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trace.json.gz'
            with gzip.open(path, 'wt') as handle:
                json.dump({'traceEvents': events}, handle)
            result = trace_summary(path)
        self.assertAlmostEqual(result['gpu_activity_union_seconds'], 30e-6)
        self.assertAlmostEqual(result['nccl_other_overlap_seconds'], 10e-6)
        self.assertAlmostEqual(result['nccl_only_seconds'], 10e-6)
        self.assertAlmostEqual(result['no_process_gpu_activity_seconds'], 70e-6)
        self.assertEqual(result['gpu_phase_event_counts'], {'student_forget_forward': 2})
        self.assertAlmostEqual(result['gpu_activity_during_host_phase']['student_forget_forward']['nccl_only_seconds'], 10e-6)


if __name__ == '__main__':
    unittest.main()
