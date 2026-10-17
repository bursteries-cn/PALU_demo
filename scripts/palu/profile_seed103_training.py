#!/usr/bin/env python3
"""Six-update, weight-free profiling replay of the frozen seed103 prefix.

The installed training and loss functions are instrumented in memory with AST
context managers. Their arithmetic, sample order, optimizer and existing scalar
reads are unchanged. No evaluation, checkpoint save or source-study edit occurs.
Wall phases are exclusive CPU execution/wait intervals, not pure GPU compute.
CUDA/NCCL work is measured separately from one complete update's Kineto trace.
"""
import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def call_name(node):
    if isinstance(node, ast.Call):
        return ast.unparse(node.func)
    return ''


def wrap(node, phase):
    item = ast.withitem(context_expr=ast.Call(
        func=ast.Attribute(value=ast.Name(id='profile_recorder', ctx=ast.Load()),
                           attr='phase', ctx=ast.Load()),
        args=[ast.Constant(phase)], keywords=[]), optional_vars=None)
    return ast.copy_location(ast.With(items=[item], body=[node], type_comment=None), node)


class TrainingScopes(ast.NodeTransformer):
    """Only wrap statements; never replace an arithmetic expression."""
    def __init__(self):
        self.counts = defaultdict(int)

    def tag(self, node, phase):
        self.counts[phase] += 1
        return wrap(node, phase)

    def visit_Assign(self, node):
        name = call_name(node.value)
        if name in ('microbatch_inputs', 'gradient.target_counts'):
            return self.tag(node, 'inputs_and_target_counts')
        if any(isinstance(n, ast.Call) and call_name(n) == 'float' for n in ast.walk(node.value)):
            return self.tag(node, 'logging_and_scalar_sync')
        return node

    def visit_Expr(self, node):
        name = call_name(node.value)
        if name.endswith('.backward'):
            return self.tag(node, 'backward_including_recompute')
        if name in ('engine.step', 'optimizer.step', 'scheduler.step'):
            return self.tag(node, 'optimizer_step')
        if name in ('append_json', 'atomic_json', 'print', 'micro_records.append',
                    'torch.distributed.all_gather_object', 'torch.cuda.synchronize'):
            return self.tag(node, 'logging_and_scalar_sync')
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if any(isinstance(n, ast.Call) and call_name(n) == 'torch.isfinite' for n in ast.walk(node.test)):
            return self.tag(node, 'logging_and_scalar_sync')
        return node

    def visit_With(self, node):
        if any(call_name(item.context_expr) == 'torch.autocast' for item in node.items):
            return self.tag(node, 'loss_construction')
        return self.generic_visit(node)


def instrument_training(function, recorder):
    tree = ast.parse(inspect.getsource(function))
    loops = [node for node in tree.body[0].body if isinstance(node, ast.For)
             and any(isinstance(inner, ast.For) and isinstance(inner.target, ast.Name)
                     and inner.target.id == 'microstep' for inner in ast.walk(node))]
    if len(loops) != 1:
        raise AssertionError('Training source changed: expected one optimizer-window loop')
    loop = loops[0]
    scopes = TrainingScopes()
    loop.body = [scopes.visit(node) for node in loop.body]
    context = ast.parse('with profile_recorder.update(window["step"]):\n    pass').body[0]
    context.body = loop.body
    loop.body = [context]
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__, profile_recorder=recorder)
    source = ast.unparse(tree)
    exec(compile(tree, function.__code__.co_filename, 'exec'), namespace)
    return namespace[function.__name__], source, dict(scopes.counts)


def instrument_components(function, recorder):
    tree = ast.parse(inspect.getsource(function))
    targets = {'output': 'student_forget_forward', 'reference_output': 'teacher_forward',
               'retain_loss': 'student_retain_forward'}
    observed = []
    class Visitor(ast.NodeTransformer):
        def visit_Assign(self, node):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in targets:
                observed.append(target.id)
                return wrap(node, targets[target.id])
            return node
    tree = Visitor().visit(tree)
    if sorted(observed) != sorted(targets):
        raise AssertionError('PALU components changed: expected exactly three model calls')
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__, profile_recorder=recorder)
    exec(compile(tree, function.__code__.co_filename, 'exec'), namespace)
    return namespace[function.__name__], ast.unparse(tree)


class Recorder:
    def __init__(self, torch, output, backend, trace_step=6, warmup=3):
        self.torch, self.output, self.backend = torch, output, backend
        self.trace_step, self.warmup = trace_step, warmup
        self.rank = int(os.environ['RANK'])
        self.active = self.tracing = False
        self.stack, self.rows = [], []

    @contextmanager
    def phase(self, name):
        if not self.active:
            yield
            return
        marker = self.torch.profiler.record_function('PALU/' + name) if self.tracing else nullcontext()
        with marker:
            frame = [name, time.perf_counter(), 0.0]
            self.stack.append(frame)
            try:
                yield
            finally:
                elapsed = time.perf_counter() - frame[1]
                self.stack.pop()
                self.times[name] += elapsed - frame[2]
                self.calls[name] += 1
                if self.stack:
                    self.stack[-1][2] += elapsed

    @contextmanager
    def update(self, step):
        prof = None
        self.tracing = self.backend == 'trace' and step == self.trace_step
        if self.tracing:
            prof = self.torch.profiler.profile(
                activities=[self.torch.profiler.ProfilerActivity.CPU,
                            self.torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False)
            prof.__enter__()
        # Same alignment barrier in both runs, outside measured update6.
        # No CUDA synchronize/barrier is added at microbatch/phase boundaries.
        if step == self.trace_step:
            self.torch.distributed.barrier()
        self.times, self.calls = defaultdict(float), defaultdict(int)
        self.active = True
        start = time.perf_counter()
        marker = self.torch.profiler.record_function('PALU/update') if self.tracing else nullcontext()
        try:
            with marker:
                yield
        finally:
            elapsed = time.perf_counter() - start
            self.active = False
            row = {'rank': self.rank, 'step': step, 'warmup': step <= self.warmup,
                   'detailed_trace': self.tracing, 'wall_seconds': elapsed,
                   'exclusive_host_seconds': dict(self.times), 'phase_calls': dict(self.calls),
                   'unclassified_seconds': elapsed - sum(self.times.values())}
            self.rows.append(row)
            with (self.output / f'timings-rank-{self.rank}.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            print(json.dumps({'event': 'profile_update_complete', **row}), flush=True)
            if prof:
                prof.__exit__(None, None, None)
                path = self.output / f'trace-rank-{self.rank}.json'
                prof.export_chrome_trace(str(path))
                with path.open('rb') as src, gzip.open(str(path) + '.gz', 'wb', compresslevel=3) as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
                path.unlink()  # Only the owned raw trace; gzip retains all events.
            self.tracing = False


class NoEvaluation:
    def __init__(self, *args):
        self.events = {}
    def at(self, *args):
        return None


class Control:
    run_initial_evaluation = False
    def __init__(self, recorder, study):
        self.recorder, self.study = recorder, study
    def select_windows(self, windows):
        return windows
    def gamma_for_step(self, step):
        return 1.0
    def configure_online(self, online):
        pass
    def initialize(self, runtime, manifest, config, schedule, args, torch):
        self.recorder.output.mkdir(parents=True, exist_ok=True)
        # Distinguish this bounded training-only replay from a formal study.
        manifest.update(profiling_only=True, evaluation_protocol='disabled for training profiling',
                        save_weights=False, measured_steps=[4, 5, 6], trace_step=6)
    def finish(self, runtime, manifest, args, torch):
        from seed103_branch_training import training_fingerprint, fingerprint_sha
        torch.distributed.barrier()
        fingerprint = training_fingerprint(runtime, torch)
        historical = read(self.study / 'fork/receipt.json')['rank_fingerprints'][runtime['rank']]
        differences = [key for key in fingerprint if fingerprint[key] != historical[key]]
        write(args.output / f'final-state-rank-{runtime["rank"]}.json',
              {'fingerprint': fingerprint, 'sha256': fingerprint_sha(fingerprint),
               'historical_step6_exact': not differences, 'historical_different_sections': differences})
        if runtime['rank'] == 0:
            manifest.update(status='profiling_completed', completed_updates=6,
                            profiling_backend=self.recorder.backend, completed_unix=time.time())
            write(args.output / 'manifest.json', manifest)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('baseline', 'trace'), required=True)
    args = parser.parse_args()
    import torch
    import layout_token_training as base
    import gradient_equivalence_step as gradient
    import online_epoch_evaluation as online
    study = args.study.resolve()
    frozen = read(study / 'frozen/manifest.json')
    config, schedule = (study / 'frozen' / name for name in ('config.json', 'schedule.pt'))
    for path in (config, schedule):
        if gradient.file_sha(path) != frozen['frozen_file_sha256'][path.name]:
            raise ValueError('Frozen input changed: ' + path.name)
    recorder = Recorder(torch, args.output, args.backend)
    train, train_source, counts = instrument_training(base.train, recorder)
    components, component_source = instrument_components(gradient.raw_components, recorder)
    options = SimpleNamespace(config=config, schedule=schedule, output=args.output,
        max_updates=6, layout='four', seed=103, normalization='global',
        padding_scope='microbatch', online_evaluation=True, delete_evaluated_intermediate=False)
    original_components, original_online = gradient.raw_components, online.OnlineEpochEvaluation
    gradient.raw_components, online.OnlineEpochEvaluation = components, NoEvaluation
    try:
        train(options, torch, Control(recorder, study))
        if recorder.rank == 0:
            (args.output / 'instrumented_train.py').write_text(train_source + '\n')
            (args.output / 'instrumented_components.py').write_text(component_source + '\n')
            source_names = ('profile_seed103_training.py', 'layout_token_training.py',
                            'gradient_equivalence_step.py', 'seed103_branch_training.py',
                            'online_epoch_evaluation.py')
            write(args.output / 'profiling_protocol.json', {
                'backend': args.backend, 'seed': 103, 'warmup_updates': [1, 2, 3],
                'measured_updates': [4, 5, 6], 'detailed_trace_update': 6,
                'source_study': str(study), 'config_sha256': gradient.file_sha(config),
                'schedule_sha256': gradient.file_sha(schedule), 'scope_counts': counts,
                'source_sha256': {name: gradient.file_sha(ROOT / 'scripts/palu' / name) for name in source_names},
                'limits': ['Exclusive host phases include CUDA waits, not pure compute.',
                           'CUDA communication is overlapping work, never an extra additive phase.',
                           'One complete trace update; only three measured updates per run.',
                           'Evaluation/checkpointing excluded; full scheduler still 60 updates/warmup6.',
                           'Only one extra alignment barrier before update6, outside its timer.']})
    finally:
        gradient.raw_components, online.OnlineEpochEvaluation = original_components, original_online
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
