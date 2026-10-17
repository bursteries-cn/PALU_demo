#!/usr/bin/env python3
"""Read-only analysis of seed103 profiling: exclusive host phases and GPU unions."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def merged(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def duration(intervals):
    return sum(b - a for a, b in merged(intervals))


def intersection_duration(left, right):
    left, right = merged(left), merged(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def trace_summary(path):
    with gzip.open(path, 'rt') as handle:
        events = json.load(handle)['traceEvents']
    spans = [e for e in events if e.get('ph') == 'X']
    updates = [e for e in spans if e.get('name') == 'PALU/update']
    if len(updates) != 1:
        raise AssertionError('Expected exactly one complete annotated update')
    start, end = updates[0]['ts'], updates[0]['ts'] + updates[0]['dur']
    annotations = defaultdict(list)
    for event in spans:
        if event.get('name', '').startswith('PALU/') and event['name'] != 'PALU/update':
            annotations[event['tid']].append(event)
    for values in annotations.values():
        values.sort(key=lambda e: e['ts'])
    starts = {tid: [e['ts'] for e in values] for tid, values in annotations.items()}

    def phase(event):
        candidates = annotations.get(event['tid'], [])
        i = bisect_right(starts.get(event['tid'], []), event['ts']) - 1
        # Nesting is shallow; enclosing loss scope may precede finished children.
        while i >= 0:
            item = candidates[i]
            if item['ts'] <= event['ts'] < item['ts'] + item['dur']:
                return item['name'].removeprefix('PALU/')
            i -= 1
        return 'unattributed'

    external, correlation = {}, {}
    for event in spans:
        if event.get('cat') in ('cpu_op', 'user_annotation'):
            key = event.get('args', {}).get('External id')
            if key is not None:
                external[key] = phase(event)
        if event.get('cat') == 'cuda_runtime':
            key = event.get('args', {}).get('correlation')
            if key is not None:
                correlation[key] = event
    comm, other, by_kind, by_phase = [], [], defaultdict(list), defaultdict(list)
    kernels, gpu_categories, phase_counts = Counter(), Counter(), Counter()
    gpu_count = 0
    for event in spans:
        cat = event.get('cat')
        if cat not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        a, b = max(start, event['ts']), min(end, event['ts'] + event['dur'])
        if b <= a:
            continue
        gpu_count += 1
        gpu_categories[cat] += 1
        is_comm = 'nccl' in event.get('name', '').lower()
        (comm if is_comm else other).append((a, b))
        kind = 'other_gpu'
        if is_comm:
            lower = event['name'].lower().replace('_', '')
            kind = next((key for key in ('allgather', 'reducescatter', 'allreduce', 'broadcast') if key in lower), 'other_nccl')
            kernels[event['name']] += 1
        by_kind[kind].append((a, b))
        args = event.get('args', {})
        attribution = external.get(args.get('External id'), 'unattributed')
        if attribution == 'unattributed':
            launch = correlation.get(args.get('correlation'))
            if launch:
                attribution = external.get(launch.get('args', {}).get('External id'), phase(launch))
        phase_counts[attribution] += 1
        by_phase[attribution + ('/nccl' if is_comm else '/other_gpu')].append((a, b))
    if not gpu_count or not comm:
        raise AssertionError('Trace does not contain CUDA kernels and NCCL work')
    c, o, union = duration(comm), duration(other), duration(comm + other)
    phase_windows = defaultdict(list)
    for values in annotations.values():
        for event in values:
            phase_windows[event['name'].removeprefix('PALU/')].append(
                (max(start, event['ts']), min(end, event['ts'] + event['dur'])))
    temporal = {}
    for key, windows in phase_windows.items():
        nccl = intersection_duration(comm, windows)
        non_nccl = intersection_duration(other, windows)
        active = intersection_duration(comm + other, windows)
        temporal[key] = {'inclusive_host_span_seconds': duration(windows) / 1e6,
                         'nccl_active_seconds': nccl / 1e6,
                         'other_gpu_active_seconds': non_nccl / 1e6,
                         'nccl_only_seconds': (active - non_nccl) / 1e6,
                         'nccl_other_overlap_seconds': (nccl + non_nccl - active) / 1e6}
    sync_calls, scalar_calls, collective_calls = defaultdict(list), defaultdict(list), defaultdict(list)
    for event in spans:
        if not start <= event['ts'] < end:
            continue
        if event.get('cat') == 'cuda_runtime' and 'Synchronize' in event.get('name', ''):
            sync_calls[event['name'] + '/' + phase(event)].append(event['dur'])
        if event.get('name') == 'aten::_local_scalar_dense':
            scalar_calls[phase(event)].append(event['dur'])
        if event.get('cat') in ('cpu_op', 'user_annotation') and any(
                text in event.get('name', '').lower() for text in ('c10d::', 'nccl:', 'processgroupnccl')):
            collective_calls[event['name']].append(event['dur'])
    summarize_calls = lambda values: {key: {'calls': len(v), 'summed_host_seconds': sum(v) / 1e6}
                                     for key, v in values.items()}
    return {'trace': str(path), 'update_seconds': (end - start) / 1e6,
            'gpu_event_count': gpu_count, 'gpu_categories': dict(gpu_categories),
            'gpu_activity_union_seconds': union / 1e6, 'nccl_union_seconds': c / 1e6,
            'other_gpu_union_seconds': o / 1e6, 'nccl_other_overlap_seconds': (c + o - union) / 1e6,
            'nccl_only_seconds': (union - o) / 1e6, 'other_gpu_only_seconds': (union - c) / 1e6,
            'no_process_gpu_activity_seconds': ((end - start) - union) / 1e6,
            'nccl_kernel_calls': sum(kernels.values()), 'nccl_kernel_counts': dict(kernels),
            'kind_union_seconds': {k: duration(v) / 1e6 for k, v in by_kind.items()},
            'phase_gpu_union_seconds': {k: duration(v) / 1e6 for k, v in by_phase.items()},
            'gpu_phase_event_counts': dict(phase_counts), 'host_cuda_sync_calls': summarize_calls(sync_calls),
            'host_scalar_reads': summarize_calls(scalar_calls),
            'host_collective_api_calls': summarize_calls(collective_calls),
            'gpu_activity_during_host_phase': temporal,
            'limits': ['NCCL kernel duration includes waiting and is not pure network transfer.',
                       'Communication overlaps GPU work and is already inside host-phase wall time.',
                       'No-process-GPU-activity is not necessarily whole-device idle.',
                       'Per-kind/per-phase union durations can overlap; do not add them.',
                       'Autograd worker-thread launches can lack a same-thread PALU scope and remain unattributed.',
                       'GPU activity during host phase is a timestamp intersection, not launch-origin attribution.',
                       'Loss-construction host windows include nested forward scopes; do not sum phase windows.']}


def verify(root, study):
    from benchmark_sharded_evaluation import compare_values
    a, b = (lines(root / backend / 'loss_history.jsonl') for backend in ('baseline', 'trace'))
    historical = lines(study / 'prefix/loss_history.jsonl')
    drop = lambda rows: [{k: v for k, v in row.items() if k != 'seconds_since_start'} for row in rows]
    for rows in (a, b, historical):
        if [row['step'] for row in rows] != list(range(1, 7)):
            raise AssertionError('Require six complete updates')
        for row in rows:
            micros = row['microsteps']
            assert sorted(i for m in micros for i in m['pair_indices']) == list(range(32))
            assert all(m['pair_indices'] == [m['microstep'] * 4 + m['rank']] for m in micros)
    matched = compare_values(drop(a), drop(b), atol=0, rtol=0)
    # Historical prefix has the same scientific keys; ignore optional metadata.
    common = sorted(set(a[0]) & set(historical[0]) - {'seconds_since_start'})
    history = compare_values([{k: r[k] for k in common} for r in a],
                             [{k: r[k] for k in common} for r in historical], atol=0, rtol=0)
    states = []
    for rank in range(4):
        left, right = (read(root / backend / f'final-state-rank-{rank}.json') for backend in ('baseline', 'trace'))
        states.append({'rank': rank, 'exact_equal': left['fingerprint'] == right['fingerprint'],
                       'baseline_matches_historical': left['historical_step6_exact'],
                       'trace_matches_historical': right['historical_step6_exact'],
                       'historical_different_sections': right['historical_different_sections']})
    protocols = [read(root / backend / 'profiling_protocol.json') for backend in ('baseline', 'trace')]
    for key in ('source_sha256', 'config_sha256', 'schedule_sha256'):
        assert protocols[0][key] == protocols[1][key]
    return {'passed': matched['passed'] and history['passed'] and all(
        v['exact_equal'] and v['baseline_matches_historical'] and v['trace_matches_historical'] for v in states),
        'loss_comparison': matched, 'historical_loss_comparison': history, 'rank_states': states,
        'historical_loss_fields': common, 'pair_positions_per_update': 32, 'updates': 6}


def summarize_host(root):
    result = {}
    for backend in ('baseline', 'trace'):
        rows = [r for rank in range(4) for r in lines(root / backend / f'timings-rank-{rank}.jsonl')]
        critical = [max((r for r in rows if r['step'] == step), key=lambda r: r['wall_seconds']) for step in (4, 5, 6)]
        names = set().union(*(r['exclusive_host_seconds'] for r in critical))
        result[backend] = {'critical_rank_by_update': [r['rank'] for r in critical],
            'update_seconds': [r['wall_seconds'] for r in critical],
            'mean_update_seconds': statistics.mean(r['wall_seconds'] for r in critical),
            'mean_exclusive_host_seconds': {name: statistics.mean(r['exclusive_host_seconds'].get(name, 0) for r in critical) for name in sorted(names)},
            'mean_unclassified_seconds': statistics.mean(r['unclassified_seconds'] for r in critical),
            'all_rank_timings': rows}
    result['trace_update6_wall_ratio'] = result['trace']['update_seconds'][-1] / result['baseline']['update_seconds'][-1]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1, choices=(1, 2, 4),
                        help='CPU trace parsers; larger values require RAM for multiple expanded JSON traces')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    verification, host = verify(args.runs, args.study), summarize_host(args.runs)
    paths = [args.runs / 'trace' / f'trace-rank-{rank}.json.gz' for rank in range(4)]
    if args.workers == 1:
        traces = [trace_summary(path) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            traces = list(pool.map(trace_summary, paths))
    result = {'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'verification': verification, 'host': host, 'traces': traces}
    write(args.output / 'profile_summary.json', result)
    print(json.dumps({'verified': result['verification']['passed'],
                     'baseline_seconds': result['host']['baseline']['mean_update_seconds'],
                     'trace_update6_wall_ratio': result['host']['trace_update6_wall_ratio'],
                     'gpu': [{k: row[k] for k in ('update_seconds', 'nccl_union_seconds', 'nccl_only_seconds',
                                                   'other_gpu_union_seconds', 'nccl_other_overlap_seconds',
                                                   'no_process_gpu_activity_seconds')} for row in result['traces']]}, indent=2))
    if not result['verification']['passed']:
        raise SystemExit('Numerical/state equivalence failed; inspect before interpreting timings')


if __name__ == '__main__':
    main()
