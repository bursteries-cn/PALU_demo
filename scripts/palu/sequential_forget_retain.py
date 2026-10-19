#!/usr/bin/env python3
"""One continuous PALU F -> R(+common-KL) experiment with frozen inputs."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from layout_loss_curve_experiment import read_json, read_rows, write_json, sha


def protocol(forget_epochs=10, retain_epochs=10, retain_epoch_size=200,
             learning_rate=1e-5, common_kl=True):
    if any(not math.isfinite(v) or v <= 0 for v in (forget_epochs, retain_epochs, learning_rate)):
        raise ValueError('Epoch budgets and learning rate must be finite and positive')
    if retain_epoch_size not in (200, 3800):
        raise ValueError('Retain epoch size must explicitly be 200 or 3800 draws')
    f, r = math.ceil(forget_epochs * 200 / 32), math.ceil(retain_epochs * retain_epoch_size / 32)
    return dict(experiment='seed103_sequential_forget_retain', seed=103,
        learning_rate=learning_rate, scheduler='constant', warmup_steps=0,
        forget_epochs_requested=forget_epochs, retain_epochs_requested=retain_epochs,
        phase_epoch_size=200, retain_epoch_size=retain_epoch_size,
        forget_updates=f, retain_updates=r, total_updates=f+r,
        forget_draws=f*32, retain_draws=r*32,
        actual_forget_phase_epochs=f*32/200,
        actual_retain_phase_epochs=r*32/retain_epoch_size,
        phase_weights={'forget': [1., 0., 0.], 'retain': [0., 2., 2. if common_kl else 0.]},
        common_kl_in_retain=common_kl, effective_batch=32, layout='four',
        microbatch=1, accumulation=8, normalization='global',
        eval_every_epochs=2, evaluation_backend='sharded',
        optimizer_transition='carry all Adam states; no reset, no restart',
        epoch_semantics='Phase-local sample exposure clock; retain sampling is with replacement, not full coverage',
        switch_policy='Round each phase UP to a complete 32-pair update; never mix phase objectives in one update',
        checkpoint_retention='final only after verified evaluation')


def evaluation_events(plan):
    """Phase-local two-epoch cadence plus exact complete-update phase endpoints."""
    from online_epoch_evaluation import epoch_events
    result = []
    for phase, offset, count, size in (
        ('forget', 0, plan['forget_updates']*32, 200),
        ('retain', plan['forget_updates']*32, plan['retain_updates']*32, plan['retain_epoch_size'])):
        for local in epoch_events(count, size, 32, 2):
            if phase == 'retain' and local['consumed_pairs'] == 0:
                continue  # the shared boundary has already been evaluated
            consumed = offset + local['consumed_pairs']
            event = dict(local, step=consumed//32, consumed_pairs=consumed,
                updated_pairs=consumed//32*32, actual_data_epochs=consumed/200,
                weight_update_epochs=consumed//32*32/200,
                completed_data_epochs=consumed//200,
                source_epoch_index=math.ceil(consumed/200),
                training_phase=phase, phase_consumed_pairs=local['consumed_pairs'],
                phase_epoch=local['actual_data_epochs'], phase_epoch_size=size,
                phase_boundary=local['consumed_pairs'] == count)
            # A very short final tail can have the same weight state as the
            # previous nominal epoch observation. Keep its endpoint metadata.
            if result and result[-1]['step'] == event['step']:
                result[-1] = event
            else:
                result.append(event)
    return result


def extend_records(prefix, total, seed=103, forget_size=200, retain_size=3800):
    """Preserve every historical draw; finish its partial epoch, then extend."""
    records = copy.deepcopy(prefix[:total])
    if len(records) == total:
        return records
    frng, rrng = random.Random(seed), random.Random(seed)
    while len(records) < total:
        position = len(records) % forget_size
        seen = {p['forget_index'] for p in records[-position:]} if position else set()
        if len(seen) != position:
            raise ValueError('Historical partial Forget epoch has repeated samples')
        remaining = [i for i in range(forget_size) if i not in seen]
        frng.shuffle(remaining)
        for index in remaining:
            n = len(records)
            records.append(dict(draw_index=n+1, epoch_index=n//forget_size+1,
                index_in_epoch=n%forget_size, forget_index=index,
                retain_index=rrng.randrange(retain_size),
                pair_origin='deterministic_extension_python_random_seed103'))
            if len(records) == total:
                break
    return records


class PhaseControl:
    run_initial_evaluation = True

    def __init__(self, plan):
        self.plan = plan
        self.planned_updates = plan['total_updates']

    def select_windows(self, windows):
        return windows

    def phase(self, step):
        if not 1 <= step <= self.planned_updates:
            raise ValueError('Update outside the frozen phase plan')
        return 'forget' if step <= self.plan['forget_updates'] else 'retain'

    def coefficients_for_step(self, step):
        return tuple(self.plan['phase_weights'][self.phase(step)])

    def training_row_metadata(self, step):
        phase = self.phase(step)
        offset = 0 if phase == 'forget' else self.plan['forget_updates']
        size = 200 if phase == 'forget' else self.plan['retain_epoch_size']
        return dict(training_phase=phase, phase_update=step-offset,
                    phase_epoch=(step-offset)*32/size, phase_epoch_size=size)

    def initialize(self, runtime, manifest, config, schedule, args, torch):
        import layout_token_training as base
        manifest.update(experiment=self.plan['experiment'], model=config['model']['handler'],
            phase_plan=self.plan, branches=0, eval_every_epochs=2,
            evaluation_backend='sharded', training_objective='F -> 2R + optional 2common_KL')
        if runtime['rank'] == 0:
            base.atomic_json(args.output / 'manifest.json', manifest)

    def configure_online(self, online):
        online.events = {e['consumed_pairs']: e for e in evaluation_events(self.plan)}

    def finish(self, runtime, manifest, args, torch):
        return False


def code_paths():
    from seed103_branch_experiment import code_paths as existing
    return sorted(set(existing() + [Path(__file__),
        ROOT / 'scripts/palu/seed103_token_equal.py',
        ROOT / 'scripts/palu/run_sequential_forget_retain.sh',
        ROOT / 'scripts/palu/plot_sequential_phases.py',
        ROOT / 'scripts/palu/README_sequential_forget_retain.md']))


def prepare(study, source, plan):
    import torch
    from omegaconf import OmegaConf
    from model import get_tokenizer
    from data import get_collators
    from prepare_layout_schedule import build_schedule
    from seed103_token_equal import relocate
    if study.exists():
        raise ValueError('Choose a fresh STUDY_DIR; existing results are never overwritten')
    old = source / 'frozen'
    provenance = read_json(old / 'manifest.json')
    for name, expected in provenance['frozen_file_sha256'].items():
        if sha(old / name) != expected:
            raise ValueError('Changed frozen source: ' + name)
    config = read_json(old / 'config.json')
    method = config['trainer']['method_args']
    expected = dict(alpha=2, gamma=1., top_k=5000, first_n=3, target_mode='mean',
                    retain_loss_type='NLL', non_sensitive_constraint_type='kl')
    if any(method.get(k) != v for k, v in expected.items()):
        raise ValueError('Expected the validated seed103 PALU method')
    if config['model']['handler'] != 'Llama-3.1-8B-Instruct' or config['trainer']['args']['seed'] != 103:
        raise ValueError('Expected Llama-3.1-8B seed103')
    original = torch.load(old / 'schedule.pt', map_location='cpu', weights_only=True)
    if len(original['windows']) != 60 or original['metadata']['data_sizes'] != {'forget': 200, 'retain': 3800}:
        raise ValueError('Source must contain the original 60 complete seed103 updates')
    config = relocate(config, str(source), str(study))
    config['sequential_phase_plan'] = copy.deepcopy(plan)
    config['trainer']['args'].update(learning_rate=plan['learning_rate'],
        lr_scheduler_type='constant', warmup_steps=0, warmup_epochs=0, warmup_ratio=0,
        max_steps=plan['total_updates'], output_dir=str(study / 'run'),
        logging_dir=str(study / 'logs'), per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        deepspeed=str(ROOT / 'configs/accelerate/zero_stage3_offload_config.json'))
    config['eval']['tofu']['retain_logs_path'] = str(study / 'frozen/retain_reference.json')
    model = Path(config['model']['model_args']['pretrained_model_name_or_path'])
    index = model / 'model.safetensors.index.json'
    weights = [model / n for n in set(read_json(index)['weight_map'].values())] if index.exists() else [model / 'model.safetensors']
    if not weights or any(not p.is_file() for p in weights):
        raise ValueError('Full model weights missing: ' + str(model))
    budget = int(sum(p.stat().st_size for p in weights) * 2.2 + 8*(1<<30))
    study.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(study.parent).free < budget:
        raise ValueError('Insufficient disk for temporary/final weights and full observations')
    tokenizer = get_tokenizer(OmegaConf.create(config['model']['tokenizer_args']))
    collator = get_collators(OmegaConf.create(config['collator']), tokenizer=tokenizer)
    prefix = [p for w in original['windows'] for p in w['pair_records']]
    records = extend_records(prefix, plan['total_updates']*32)
    payload = build_schedule(original['samples'], collator, eos=tokenizer.eos_token_id,
        first_n=3, updates=plan['total_updates'], effective_batch=32, seed=103,
        torch=torch, records=records)
    for new, old_window in zip(payload['windows'], original['windows']):
        if new['window_sha256'] != old_window['window_sha256']:
            raise ValueError('Historical prefix changed during schedule extension')
    if payload['metadata']['fixed_dataset_sha256'] != original['metadata']['fixed_dataset_sha256']:
        raise ValueError('Frozen sample pool changed')
    frozen = study / 'frozen'
    frozen.mkdir(parents=True)
    write_json(frozen / 'config.json', config)
    shutil.copyfile(old / 'retain_reference.json', frozen / 'retain_reference.json')
    events = evaluation_events(plan)
    payload['metadata'].update(model_path=str(model), source_config_sha256=sha(frozen / 'config.json'),
        seed103_source_provenance=original['metadata']['seed103_source_provenance'],
        evaluation_policy='Phase-local every two exposure epochs, phase boundary, initial and final; preserve pending gradients',
        actual_epoch_events=events, evaluation_steps=[e['step'] for e in events],
        epoch_evaluation_schedule=events, phase_plan=plan,
        prefix_preserved_updates=min(60, plan['total_updates']),
        extension_policy='Finish historical partial epoch without repeats; Python Random(103) independent Forget/Retain generators; Retain replacement')
    torch.save(payload, frozen / 'schedule.pt')
    write_json(frozen / 'epoch_schedule.json', events)
    (frozen / 'pair_schedule.jsonl').write_text(''.join(json.dumps({'step': w['step'], **p})+'\n'
        for w in payload['windows'] for p in w['pair_records']))
    metadata = copy.deepcopy(payload['metadata'])
    metadata['frozen_file_sha256'] = {n: sha(frozen/n) for n in
        ('config.json', 'schedule.pt', 'retain_reference.json', 'epoch_schedule.json', 'pair_schedule.jsonl')}
    write_json(frozen / 'manifest.json', metadata)
    for path in code_paths():
        dest = study / 'code_snapshot' / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    write_json(study / 'code_sha256.json', {str(p.relative_to(ROOT)): sha(p) for p in code_paths()})
    r_pairs = records[plan['forget_updates']*32:]
    plan.update(status='prepared', source_study=str(source),
        source_frozen_sha256=provenance['frozen_file_sha256'], required_free_bytes=budget,
        retain_phase_unique_retain=len({p['retain_index'] for p in r_pairs}),
        evaluation_steps=[e['step'] for e in events], cuda_initialized=torch.cuda.is_initialized())
    write_json(study / 'experiment.json', plan)
    print(json.dumps(plan), flush=True)


def train(study):
    import torch
    import layout_token_training as base
    for name, digest in read_json(study / 'frozen/manifest.json')['frozen_file_sha256'].items():
        if sha(study / 'frozen' / name) != digest:
            raise ValueError('Changed frozen input: ' + name)
    for name, digest in read_json(study / 'code_sha256.json').items():
        if sha(ROOT / name) != digest:
            raise ValueError('Prepared code changed: ' + name)
    plan = frozen_plan(study)
    args = argparse.Namespace(config=study/'frozen/config.json', schedule=study/'frozen/schedule.pt',
        output=study/'run', max_updates=None, layout='four', seed=103, normalization='global',
        padding_scope='microbatch', online_evaluation=True, delete_evaluated_intermediate=True,
        eval_every_epochs=2, eval_backend='sharded')
    try:
        base.train(args, torch, PhaseControl(plan))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def frozen_plan(study):
    config = study/'frozen/config.json'
    expected = read_json(study/'frozen/manifest.json')['frozen_file_sha256']['config.json']
    if sha(config) != expected:
        raise ValueError('Frozen configuration changed')
    plan = read_json(config)['sequential_phase_plan']
    visible = read_json(study/'experiment.json')
    if any(visible.get(key) != value for key, value in plan.items()):
        raise ValueError('Experiment description differs from frozen phase plan')
    return plan


def validate(study):
    import checkpoint_retention as retention
    plan = frozen_plan(study)
    control = PhaseControl(plan)
    run = study/'run'
    manifest = read_json(run/'manifest.json')
    if manifest['status'] != 'completed' or manifest['completed_updates'] != plan['total_updates']:
        raise ValueError('Full two-phase training is incomplete; use plot mode to inspect partial results')
    losses = read_rows(run/'loss_history.jsonl')
    if [r['step'] for r in losses] != list(range(1, plan['total_updates']+1)):
        raise ValueError('Missing update losses')
    for row in losses:
        if (row['gamma_used'], row['alpha_used'], row['common_used']) != control.coefficients_for_step(row['step']):
            raise ValueError('Objective did not follow phase plan')
        if not all(math.isclose(lr, plan['learning_rate'], rel_tol=1e-10) for lr in row['lr_before']):
            raise ValueError('Learning rate changed')
    events = evaluation_events(plan)
    if manifest['evaluated_steps'] != [e['step'] for e in events]:
        raise ValueError('Missing phase-local evaluations')
    if [c['step'] for c in manifest['checkpoints']] != [e['step'] for e in events]:
        raise ValueError('Checkpoint observation inventory is incomplete')
    for checkpoint, event in zip(manifest['checkpoints'], events):
        receipt = retention.read_receipt(run, manifest, checkpoint)
        if not receipt['training_state_unchanged_all_ranks']:
            raise ValueError('Evaluation modified training state')
        if any(receipt.get(k) != event[k] for k in
               ('consumed_pairs', 'updated_pairs', 'pending_pairs', 'training_phase', 'phase_epoch')):
            raise ValueError('Evaluation timing or phase attribution differs from plan')
    pairs = read_rows(study/'frozen/pair_schedule.jsonl')
    consumed = read_rows(run/'training_pairs.jsonl')
    if len(pairs) != len(consumed) or any(any(b.get(k) != v for k, v in a.items())
                                        for a, b in zip(pairs, consumed)):
        raise ValueError('Actual F/R order or pairing differs from the frozen schedule')
    if manifest.get('deleted_intermediate_steps') != [e['step'] for e in events[1:-1]]:
        raise ValueError('Intermediate checkpoint retention mismatch')
    if not list((run/f"checkpoint-{plan['total_updates']}").glob('*.safetensors')):
        raise ValueError('Final model missing')
    write_json(study/'completion_validation.json', dict(passed=True,
        completed_updates=plan['total_updates'], evaluation_steps=manifest['evaluated_steps']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('dry-run', 'prepare', 'train', 'analyze', 'plot'), required=True)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--source-study', type=Path)
    p.add_argument('--forget-epochs', type=float, default=10)
    p.add_argument('--retain-epochs', type=float, default=10)
    p.add_argument('--retain-epoch-size', type=int, choices=(200, 3800), default=200)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--no-common-kl', action='store_true')
    args = p.parse_args()
    plan = protocol(args.forget_epochs, args.retain_epochs, args.retain_epoch_size,
                    args.learning_rate, not args.no_common_kl)
    study = args.study.resolve()
    if args.mode == 'dry-run':
        print(json.dumps({**plan, 'evaluation_events': evaluation_events(plan)}, indent=2))
    elif args.mode == 'prepare':
        if args.source_study is None:
            p.error('prepare requires --source-study')
        prepare(study, args.source_study.resolve(), plan)
    elif args.mode == 'train':
        train(study)
    else:
        frozen_plan(study)
        if args.mode == 'analyze':
            validate(study)
        from plot_sequential_phases import analyze
        analyze(study)


if __name__ == '__main__':
    main()
