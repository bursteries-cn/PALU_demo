#!/usr/bin/env python3
"""One complete seed103 token-equal trajectory, using existing training/evaluation."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from layout_loss_curve_experiment import read_json, read_rows, write_json, sha


def relocate(value, old, new):
    if isinstance(value, dict):
        return {k: relocate(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, old, new) for v in value]
    if isinstance(value, str) and (value == old or value.startswith(old + '/')):
        return new + value[len(old):]
    return value


def prepare(study, source):
    import torch
    from online_epoch_evaluation import epoch_events
    from seed103_branch_experiment import code_paths
    if study.exists():
        raise ValueError('Choose a new study directory; existing results are never replaced')
    old = source / 'frozen'
    original = read_json(old / 'manifest.json')
    for name, digest in original['frozen_file_sha256'].items():
        if sha(old / name) != digest:
            raise ValueError('Source frozen input changed: ' + name)
    config = read_json(old / 'config.json')
    required = {'gamma': 1.0, 'alpha': 2, 'top_k': 5000, 'first_n': 3,
                'target_mode': 'mean', 'retain_loss_type': 'NLL', 'non_sensitive_constraint_type': 'kl'}
    if any(config['trainer']['method_args'].get(k) != v for k, v in required.items()):
        raise ValueError('Not the validated seed103 PALU method')
    train = config['trainer']['args']
    required_train = {'seed': 103, 'data_seed': 103, 'learning_rate': 2e-5,
                      'warmup_steps': 6, 'max_steps': 60, 'lr_scheduler_type': 'linear'}
    if any(train.get(k) != v for k, v in required_train.items()):
        raise ValueError('Not the validated seed103 training configuration')
    model = Path(config['model']['model_args']['pretrained_model_name_or_path'])
    model_bytes = sum(p.stat().st_size for p in model.glob('*.safetensors'))
    if model_bytes == 0:
        raise ValueError('Full model weights not found')
    parent = study.parent
    parent.mkdir(parents=True, exist_ok=True)
    budget = int(model_bytes * 2.2 + 8 * (1 << 30))
    if shutil.disk_usage(parent).free < budget:
        raise ValueError('Insufficient disk for temporary/final weights and results')
    frozen = study / 'frozen'
    frozen.mkdir(parents=True)
    moved = relocate(config, str(source), str(study))
    assert relocate(moved, str(study), str(source)) == config
    write_json(frozen / 'config.json', moved)
    payload = torch.load(old / 'schedule.pt', map_location='cpu', weights_only=True)
    before_windows = [w['window_sha256'] for w in payload['windows']]
    assert len(before_windows) == 60 and payload['metadata']['effective_batch'] == 32
    events = epoch_events(every_epochs=2)
    payload['metadata'].update(source_config_sha256=sha(frozen / 'config.json'),
        evaluation_policy='initial, every two actual data epochs, and final tail; in-memory sharded evaluation')
    torch.save(payload, frozen / 'schedule.pt')
    check = torch.load(frozen / 'schedule.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(w[s][k], v[s][k]) for w, v in zip(payload['windows'], check['windows'])
               for s in ('forget', 'retain') for k in ('input_ids', 'attention_mask', 'labels'))
    assert before_windows == [w['window_sha256'] for w in check['windows']]
    for name in ('retain_reference.json', 'pair_schedule.jsonl'):
        shutil.copyfile(old / name, frozen / name)
    write_json(frozen / 'epoch_schedule.json', events)
    metadata = copy.deepcopy(original)
    for key in list(metadata):
        if key.startswith('fork_') or key == 'snapshot_storage_budget':
            metadata.pop(key)
    metadata.update(source_study=str(source), source_config_sha256=sha(frozen / 'config.json'),
        eval_every_epochs=2, evaluation_backend='sharded', actual_epoch_events=events,
        evaluation_steps=[e['step'] for e in events],
        full_update_denominators='Separate sensitive, retain and common token totals over each 32-pair update')
    metadata['frozen_file_sha256'] = {name: sha(frozen / name) for name in
        ('config.json', 'schedule.pt', 'retain_reference.json', 'pair_schedule.jsonl', 'epoch_schedule.json')}
    write_json(frozen / 'manifest.json', metadata)
    paths = code_paths() + [Path(__file__), ROOT / 'scripts/palu/run_seed103_token_equal.sh']
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        target = study / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = sha(path)
    write_json(study / 'code_sha256.json', hashes)
    receipt = dict(experiment='seed103_four_gpu_token_equal_single', status='prepared',
        source_study=str(source), source_frozen_sha256=original['frozen_file_sha256'],
        seed=103, layout='four', microbatch=1, accumulation=8, effective_batch=32,
        normalization='global', padding_scope='microbatch', updates=60, pairs=1920,
        gamma=1, alpha=2, branches=0, start='original Full model; fresh original optimizer',
        evaluation_steps=[e['step'] for e in events], eval_every_epochs=2, evaluation_backend='sharded',
        checkpoint_retention='final only after verified evaluation', final_models=1,
        required_free_bytes=budget, all_60_window_hashes_unchanged=True,
        source_pair_stream_sha256=sha(old / 'pair_schedule.jsonl'),
        new_pair_stream_sha256=sha(frozen / 'pair_schedule.jsonl'),
        cuda_initialized=torch.cuda.is_initialized())
    write_json(study / 'experiment.json', receipt)
    print(json.dumps(receipt), flush=True)


class SingleControl:
    run_initial_evaluation = True

    def select_windows(self, windows):
        return windows

    def gamma_for_step(self, step):
        return 1.0

    def initialize(self, runtime, manifest, config, schedule, args, torch):
        import layout_token_training as base
        manifest.update(experiment='seed103_four_gpu_token_equal_single',
            model=config['model']['handler'], eval_every_epochs=2, evaluation_backend='sharded',
            branches=0, training_objective='F + 2R + 2 common KL; gamma=1 throughout')
        if runtime['rank'] == 0:
            base.atomic_json(args.output / 'manifest.json', manifest)

    def configure_online(self, online):
        pass

    def finish(self, runtime, manifest, args, torch):
        return False


def train(study):
    import torch
    import layout_token_training as base
    metadata = read_json(study / 'frozen/manifest.json')
    for name, digest in metadata['frozen_file_sha256'].items():
        if sha(study / 'frozen' / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    for name, digest in read_json(study / 'code_sha256.json').items():
        if sha(ROOT / name) != digest:
            raise ValueError('Prepared source changed: ' + name)
    args = argparse.Namespace(config=study / 'frozen/config.json', schedule=study / 'frozen/schedule.pt',
        output=study / 'run', max_updates=None, layout='four', seed=103, normalization='global',
        padding_scope='microbatch', online_evaluation=True, delete_evaluated_intermediate=True,
        eval_every_epochs=2, eval_backend='sharded')
    try:
        base.train(args, torch, SingleControl())
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def analyze(study):
    import checkpoint_retention as retention
    run = study / 'run'
    manifest = read_json(run / 'manifest.json')
    losses = read_rows(run / 'loss_history.jsonl')
    if manifest['status'] != 'completed' or manifest['completed_updates'] != 60:
        raise ValueError('Formal training is incomplete')
    if [r['step'] for r in losses] != list(range(1, 61)):
        raise ValueError('Expected all 60 update records')
    for row in losses:
        if row['gamma_used'] != 1 or row['alpha_used'] != 2 or row['loss_mode'] != 'global':
            raise ValueError('Training objective changed')
    expected = [0, 12, 25, 37, 50, 60]
    if manifest['evaluated_steps'] != expected:
        raise ValueError('Missing two-epoch evaluations')
    for checkpoint in manifest['checkpoints']:
        row = retention.read_receipt(run, manifest, checkpoint)
        if not row['training_state_unchanged_all_ranks'] or row['evaluation_backend'] != 'sharded':
            raise ValueError('Evaluation state/backend guard failed')
    if manifest['deleted_intermediate_steps'] != [12, 25, 37, 50]:
        raise ValueError('Intermediate checkpoint retention mismatch')
    if not list((run / 'checkpoint-60').glob('*.safetensors')):
        raise ValueError('Final weights missing')
    write_json(study / 'completion_validation.json', {'passed': True, 'updates': 60,
        'evaluated_steps': expected, 'gamma': 1, 'final_models': 1})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare', 'train', 'analyze'), required=True)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--source-study', type=Path)
    args = p.parse_args()
    if args.mode == 'prepare':
        if args.source_study is None:
            p.error('prepare requires --source-study')
        prepare(args.study.resolve(), args.source_study.resolve())
    elif args.mode == 'train':
        train(args.study.resolve())
    else:
        analyze(args.study.resolve())


if __name__ == '__main__':
    main()
