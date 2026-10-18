"""Prepare, launch and analyze nine single-GPU PALU overshoot replications."""
import argparse
import concurrent.futures
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'configs/analysis/overshoot_seed_replicates'
IDS = ('P336', 'P421', 'P386')
EXPECTED = {'P336': (5e-5, 5, 3, 5000), 'P421': (4e-5, 5, 5, 1000), 'P386': (5e-5, 5, 4, 10000)}
EVAL_STEPS = [0, 12, 25, 37, 50, 60]
SCRIPTS = ['overshoot_seed_replicates.py', 'run_overshoot_seed_replicates.sh', 'seed_epoch_experiment.py', 'plot_seed_epoch.py']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_csv(path, data):
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        w.writeheader(); w.writerows(data)


def sources():
    provenance = read(ASSETS / 'provenance.json')
    configs = {}
    for cid in IDS:
        path = ASSETS / f'{cid}.json'
        if sha(path) != provenance['configs'][cid]['asset_sha256']:
            raise ValueError(f'Changed historical config asset: {cid}')
        c = read(path); t = c['trainer']; m = t['method_args']; a = t['args']
        if (a['learning_rate'], m['alpha'], m['first_n'], m['top_k']) != EXPECTED[cid]:
            raise ValueError(f'Unexpected historical hyperparameters: {cid}')
        assert c['model']['handler'] == 'Llama-3.1-8B-Instruct' and t['handler'] == 'PALU'
        assert a['per_device_train_batch_size'] == 8 and a['gradient_accumulation_steps'] == 4
        assert a['seed'] == 0 and a['num_train_epochs'] == 10 and a['warmup_epochs'] == 1
        assert m['target_mode'] == 'mean' and m['gamma'] == 1
        configs[cid] = c
    return configs, provenance


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['dry-run', 'prepare', 'run', 'analyze'], required=True)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--gpus', default='0')
    p.add_argument('--seeds', default='201,202,203')
    p.add_argument('--min-free-gib', type=float, default=78)
    p.add_argument('--model-path', type=Path)
    p.add_argument('--retain-logs', type=Path)
    p.add_argument('--tofu-data', type=Path)
    args = p.parse_args(); args.study = args.study.resolve()
    args.seeds = [int(s) for s in args.seeds.split(',')]
    args.gpus = [s.strip() for s in args.gpus.split(',')]
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3 or min(args.seeds) < 1:
        p.error('Use exactly three distinct positive seeds; historical seed 0 is excluded')
    if not all(args.gpus) or len(set(args.gpus)) != len(args.gpus):
        p.error('GPUS must contain distinct nonempty GPU identifiers')
    if args.min_free_gib <= 0:
        p.error('min-free-gib must be positive')
    return args


def code_files():
    return sorted(list((ROOT / 'src').rglob('*.py')) + [ROOT / 'scripts/palu' / n for n in SCRIPTS])


def prepare(args):
    configs, provenance = sources(); study = args.study
    if (study / 'manifest.json').is_file():
        old = read(study / 'manifest.json')
        if old['seeds'] != args.seeds or old['source_provenance'] != provenance:
            raise ValueError('Existing study has different seeds or source configs')
        for name, value in old['code_sha256'].items():
            if sha(ROOT / name) != value: raise ValueError(f'Code changed since preparation: {name}')
        for name, value in old['frozen_sha256'].items():
            if sha(study / name) != value: raise ValueError(f'Frozen study asset changed: {name}')
        if args.model_path and str(args.model_path.resolve()) != old['model_path']:
            raise ValueError('Existing study has a different Full model')
        if args.tofu_data and str(args.tofu_data.resolve()) != old['tofu_data_path']:
            raise ValueError('Existing study has a different TOFU data path')
        return old
    if study.exists() and any(study.iterdir()):
        raise ValueError('Use a new STUDY_DIR; incomplete preparation is preserved')
    packages = {k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'accelerate', 'datasets', 'bitsandbytes', 'numpy', 'scipy', 'matplotlib']}
    if packages['transformers'] != '4.45.1' or packages['accelerate'] != '0.34.2':
        raise ValueError('Historical Trainer budget/sampling requires transformers 4.45.1 and accelerate 0.34.2')
    if args.model_path:
        model = args.model_path.resolve()
    else:
        from huggingface_hub import snapshot_download
        model = Path(snapshot_download('open-unlearning/tofu_Llama-3.1-8B-Instruct_full',
            revision='1a5c5b1a557f8c99bdadecd5168ebd03f640b00e', local_files_only=True)).resolve()
    reference = (args.retain_logs or ROOT / 'saves/eval/tofu_Llama-3.1-8B-Instruct_retain95/TOFU_EVAL.json').resolve()
    tofu_data = (args.tofu_data or ROOT / 'datasets/TOFU').resolve()
    if not tofu_data.is_dir(): raise ValueError('TOFU dataset directory missing; set TOFU_DATA_PATH')
    if not (model / 'config.json').is_file(): raise ValueError(f'Full snapshot missing: {model}; set MODEL_PATH to the TOFU Full model')
    if not reference.is_file() or sha(reference) != provenance['reference_sha256']:
        raise ValueError('The frozen Retain95 evaluation reference is missing or changed')
    index = model / 'model.safetensors.index.json'
    weights = [model / n for n in sorted(set(read(index)['weight_map'].values()))] if index.is_file() else [model / 'model.safetensors']
    if not all(p.is_file() for p in weights): raise ValueError('Full model weight shards are incomplete')
    model_bytes = sum(p.stat().st_size for p in weights)
    parent = study
    while not parent.exists(): parent = parent.parent
    required = int(9 * model_bytes * 1.05 + 5 * 2**30)
    if shutil.disk_usage(parent).free < required:
        raise ValueError(f'Nine final models require about {required / 2**30:.1f} GiB free disk; intermediate weights are not saved')
    from omegaconf import OmegaConf
    sys.path.insert(0, str(ROOT / 'src'))
    from model import get_tokenizer
    from data import get_data
    frozen = study / 'frozen'; frozen.mkdir(parents=True)
    shutil.copyfile(reference, frozen / 'retain_reference.json')
    resolved = {}
    for cid, original in configs.items():
        c = copy.deepcopy(original)
        c['paths'].update(root_dir=str(ROOT), work_dir=str(ROOT), output_dir=str(study / cid), tofu_data=str(tofu_data))
        for key in ['model_args', 'tokenizer_args']: c['model'][key]['pretrained_model_name_or_path'] = str(model)
        c['eval']['tofu']['retain_logs_path'] = str(frozen / 'retain_reference.json')
        # Explicitly freeze the historical 60-update budget and single-GPU warmup.
        c['trainer']['args'].update(max_steps=60, dataloader_num_workers=0)
        c = OmegaConf.to_container(OmegaConf.create(c), resolve=True)
        if '${' in json.dumps(c): raise ValueError('Unresolved source config interpolation')
        resolved[cid] = c; write(frozen / f'{cid}.json', c)
    # CPU-only tokenization preflight. Full fixed-split losses are measured during training.
    first = OmegaConf.create(resolved[IDS[0]])
    tokenizer = get_tokenizer(first.model.tokenizer_args)
    data = get_data(first.data, mode='unlearn', tokenizer=tokenizer, template_args=first.model.template_args,
                    model_name=first.model.handler)['train']
    if data.anchor != 'forget' or len(data.forget) != 200 or len(data.retain) != 3800:
        raise ValueError('Expected the unchanged Forget05=200 / Retain95=3800 training splits')
    digest = hashlib.sha256()
    for split in ['forget', 'retain']:
        for i in range(len(getattr(data, split))):
            sample = getattr(data, split)[i]
            digest.update(json.dumps([split, i], separators=(',', ':')).encode())
            for key in ['input_ids', 'labels']:
                digest.update(key.encode()); digest.update(sample[key].long().numpy().tobytes())
    planned = [dict(model=cid, architecture='Llama-3.1-8B-Instruct', seed=s,
                    planned_seeds=args.seeds, output_dir=str(study / cid / f'seed_{s}'),
                    config=str(frozen / f'{cid}.json')) for cid in IDS for s in args.seeds]
    manifest = dict(schema=1, study_type='historical_overshoot_seed_replication', seeds=args.seeds,
                    runs=planned, source_provenance=provenance, model_path=str(model), tofu_data_path=str(tofu_data),
                    model_weight_bytes=model_bytes, planned_final_weight_bytes=model_bytes * 9,
                    packages=packages, training_world_size=1, microbatch=8, accumulation=4,
                    loss_normalization='original microbatch target-token means; equal microbatch accumulation',
                    evaluation_steps=EVAL_STEPS, eval_every_actual_epochs=2, fixed_loss_data='all F200/R3800, batch1, token-weighted and sample means',
                    primary_intervals={cid: [provenance['configs'][cid]['primary_anchor_step'], 60] for cid in IDS},
                    checkpoint_policy='no intermediate weights written; current in-memory evaluation; final weights only',
                    tokenized_data_sha256=digest.hexdigest(), created_unix=time.time(),
                    code_sha256={str(p.relative_to(ROOT)): sha(p) for p in code_files()},
                    frozen_sha256={str(p.relative_to(study)): sha(p) for p in frozen.iterdir()})
    write(study / 'manifest.json', manifest)
    print(f'Prepared nine runs; final weights total about {model_bytes * 9 / 2**30:.1f} GiB', flush=True)
    return manifest


def verify_run(entry):
    output = Path(entry['output_dir']); manifest = read(output / 'manifest.json')
    if manifest.get('status') != 'completed' or manifest['global_step'] != 60:
        raise ValueError(f'Unfinished run is preserved: {output}; use a new STUDY_DIR')
    epochs = rows(output / 'epoch_history.jsonl')
    if [r['step'] for r in epochs] != EVAL_STEPS: raise ValueError('Unexpected evaluation cadence')
    losses = rows(output / 'loss_history.jsonl')
    if [r['step'] for r in losses] != list(range(1,61)): raise ValueError('Incomplete optimizer-step loss history')
    for epoch in epochs:
        folder = output / f"checkpoint-{epoch['step']}" / 'evals'
        receipt = read(folder / 'verification.json')
        if receipt['status'] != 'verified': raise ValueError('Unverified epoch evaluation')
        for name, expected in receipt['files'].items():
            if sha(folder / name) != expected: raise ValueError(f'Evaluation artifact changed: {folder / name}')
    if not (output / 'model.safetensors.index.json').is_file() and not (output / 'model.safetensors').is_file():
        raise ValueError('Final model is missing')


def train(args, manifest):
    logdir = args.study / 'logs'; logdir.mkdir(exist_ok=True)
    queues = [[] for _ in args.gpus]
    for i, entry in enumerate(manifest['runs']): queues[i % len(queues)].append(entry)
    def worker(gpu, entries):
        for entry in entries:
            output = Path(entry['output_dir']); cid, seed = entry['model'], entry['seed']
            if (output / 'manifest.json').exists():
                verify_run(entry); print(f'Skip verified completed {cid}/seed_{seed}', flush=True); continue
            if output.exists() and any(output.iterdir()): raise ValueError(f'Nonempty unfinished output: {output}')
            query = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid,name,memory.free,memory.total', '--format=csv,noheader,nounits'], text=True).strip()
            fields = query.split(',')
            if len(fields) != 4 or float(fields[-2]) / 1024 < args.min_free_gib:
                raise ValueError(f'GPU {gpu} needs {args.min_free_gib} GiB free: {query}')
            write(logdir / f'{cid}_seed_{seed}_gpu.json', {'gpu': gpu, 'query': query, 'time': time.time()})
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            command = [sys.executable, str(ROOT / 'scripts/palu/seed_epoch_experiment.py'), '--config', entry['config'],
                       '--output', str(output), '--seed', str(seed), '--eval-every-epochs', '2',
                       '--expected-updates', '60', '--verify-observations', '--forget-samples', 'all', '--retain-samples', 'all',
                       '--expected-fixed-data-sha256', manifest['tokenized_data_sha256']]
            log = logdir / f'{cid}_seed_{seed}.log'
            print(f'Running {cid}/seed_{seed} on GPU {gpu}; log: {log}', flush=True)
            with log.open('w') as stream:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            verify_run(entry); print(f'Completed {cid}/seed_{seed}', flush=True)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, gpu, queue) for gpu, queue in zip(args.gpus, queues)]
        for future in concurrent.futures.as_completed(futures):
            try: future.result()
            except Exception as e: errors.append(str(e)); print(str(e), file=sys.stderr, flush=True)
    analyze(args.study)
    if errors: raise RuntimeError('; '.join(errors))


def analyze(study):
    import numpy as np
    manifest = read(study / 'manifest.json'); comparisons = []; all_epochs = []
    for entry in manifest['runs']:
        output = Path(entry['output_dir']); cid = entry['model']; seed = entry['seed']
        record = dict(config=cid, seed=seed, status='missing', primary_confirmed=False)
        if not (output / 'epoch_history.jsonl').is_file(): comparisons.append(record); continue
        points = rows(output / 'epoch_history.jsonl'); all_epochs.extend(dict(p, config=cid) for p in points)
        bystep = {p['step']: p for p in points}; anchor_step = manifest['primary_intervals'][cid][0]
        if anchor_step not in bystep or 60 not in bystep: record['status']='incomplete'; comparisons.append(record); continue
        anchor, end = bystep[anchor_step], bystep[60]
        samples = lambda step: list(csv.DictReader((output / f'checkpoint-{step}/evals/truth_ratios.csv').open()))
        ar, er = samples(anchor_step), samples(60)
        assert [r['index'] for r in ar] == [r['index'] for r in er]
        at, et, ref = [np.array([float(r[key]) for r in data]) for data, key in [(ar,'TR'), (er,'TR'), (er,'reference_TR')]]
        grid = np.unique(np.r_[et,ref]); gap = np.searchsorted(np.sort(et),grid,side='right')/len(et)-np.searchsorted(np.sort(ref),grid,side='right')/len(ref)
        flags = dict(anchor_close=anchor['ks_d']<=.15+1e-12, D_increased=end['ks_d']-anchor['ks_d']>=.05-1e-12,
                     FQ_dropped_100x=end['forget_quality']<=anchor['forget_quality']/100,
                     MU_recovered=end['model_utility']-anchor['model_utility']>=.01-1e-12,
                     F_decreased=end['raw_sensitive_loss']<anchor['raw_sensitive_loss'],
                     R_decreased=end['raw_retain_nll']<anchor['raw_retain_nll'],
                     TR_high_side=-float(gap.min())>float(gap.max()) and np.median(et)>np.median(ref),
                     majority_TR_increased=bool((et>at).mean()>.5 and np.log(et).mean()>np.log(at).mean()))
        record.update(status='complete_observations', anchor_step=anchor_step, end_step=60,
                      **{k: bool(v) for k,v in flags.items()}, primary_confirmed=all(flags.values()),
                      fraction_TR_increased=float((et>at).mean()), end_D_high=-float(gap.min()), end_D_low=float(gap.max()))
        for name,p in [('anchor',anchor),('end',end)]:
            for key in ['forget_quality','model_utility','ks_d','raw_sensitive_loss','raw_retain_nll','raw_common_kl']:
                record[f'{name}_{key}']=p[key]
        comparisons.append(record)
    analysis = study / 'analysis'; analysis.mkdir(exist_ok=True)
    write_csv(analysis / 'primary_seed_comparisons.csv', comparisons)
    write_csv(analysis / 'all_epoch_points.csv', all_epochs)
    summary = {cid: {'planned_seeds':manifest['seeds'], 'completed_intervals':sum(r['status']=='complete_observations' for r in comparisons if r['config']==cid),
                     'primary_confirmed':sum(r['primary_confirmed'] for r in comparisons if r['config']==cid)} for cid in IDS}
    write(analysis / 'summary.json', summary)
    pairing = []
    for seed in manifest['seeds']:
        hashes = {entry['model']: sha(Path(entry['output_dir']) / 'training_pairs.jsonl')
                  for entry in manifest['runs'] if entry['seed'] == seed and (Path(entry['output_dir']) / 'training_pairs.jsonl').is_file()}
        pairing.append(dict(seed=seed, complete_configs=len(hashes), pair_file_sha256=hashes,
                            same_pairs_across_all_configs=len(hashes)==3 and len(set(hashes.values()))==1))
    write(analysis / 'pairing_checks.json', pairing)
    if all_epochs:
        subprocess.run([sys.executable, str(ROOT / 'scripts/palu/plot_seed_epoch.py'), str(study), '--output', str(study / 'plots')], cwd=ROOT, check=True)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    args = parse(); configs, provenance = sources()
    if args.mode == 'dry-run':
        print(json.dumps(dict(configs={k: EXPECTED[k] for k in IDS}, tuple_order=['lr','alpha','first_n','top_k'],
                              seeds=args.seeds, gpus=args.gpus, runs=9, world_size_per_run=1, microbatch=8, accumulation=4,
                              evaluation_steps=EVAL_STEPS, primary_intervals={k:[v['primary_anchor_step'],60] for k,v in provenance['configs'].items()},
                              study=str(args.study), training_started=False), indent=2)); return
    if args.mode == 'analyze': analyze(args.study); return
    manifest = prepare(args)
    if args.mode == 'run': train(args, manifest)


if __name__ == '__main__':
    main()
