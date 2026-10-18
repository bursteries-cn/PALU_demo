"""Validate saved epoch measurements before cleaning owned intermediate weights."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

WEIGHT = re.compile(r'(?:model(?:-\d{5}-of-\d{5})?\.safetensors|model\.safetensors\.index\.json|pytorch_model(?:-\d{5}-of-\d{5})?\.bin|pytorch_model\.bin\.index\.json)$')


def write_json(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def prune_weights(run, step, final_step):
    run = Path(run).resolve()
    folder = run / f'checkpoint-{step}'
    if folder.is_symlink() or folder.resolve() != folder:
        raise ValueError('Refusing a symlinked or unowned checkpoint')
    if step <= 0 or step >= final_step:
        return []
    receipt = folder / 'evals/verification.json'
    if not receipt.is_file() or json.loads(receipt.read_text()).get('status') != 'verified':
        raise ValueError('Intermediate cleanup requires verified evaluation')
    paths = [p for p in folder.iterdir() if WEIGHT.fullmatch(p.name)]
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError('Refusing unexpected checkpoint weight entries')
    for p in paths:
        p.unlink()
    return [p.name for p in paths]


def finalize_observation(run, row, reference_path, final_step):
    import numpy as np
    from scipy.stats import ks_2samp
    run = Path(run).resolve(); step = row['step']
    folder = run / f'checkpoint-{step}' / 'evals'
    native = json.loads((folder / 'TOFU_EVAL.json').read_text())
    ref = json.loads(Path(reference_path).read_text())['forget_truth_ratio']['value_by_index']
    values = native['forget_truth_ratio']['value_by_index']
    ids = [str(i) for i in range(row['forget_count'])]
    if set(values) != set(ids) or set(ref) != set(ids):
        raise ValueError('Incomplete or mismatched per-question TR indices')
    records = []
    for i in ids:
        lc = native['forget_Q_A_PARA_Prob']['value_by_index'][i]['avg_loss']
        lw = native['forget_Q_A_PERT_Prob']['value_by_index'][i]['avg_loss']
        pc, pw = float(np.exp(-lc)), float(np.exp(-np.mean(lw)))
        tr, reference_tr = values[i]['score'], ref[i]['score']
        if len(lw) != 5 or not math.isfinite(tr) or tr <= 0 or not math.isclose(tr, pw / (pc + 1e-10), rel_tol=1e-11):
            raise ValueError('Invalid geometric-mean TR reconstruction')
        records.append(dict(index=int(i), TR=tr, reference_TR=reference_tr,
                            correct_score=pc, wrong_geometric_score=pw,
                            correct_eval_NLL=lc, mean_wrong_eval_NLL=float(np.mean(lw))))
    result = ks_2samp([r['TR'] for r in records], [r['reference_TR'] for r in records], method='exact')
    if not math.isclose(result.pvalue, row['forget_quality'], rel_tol=1e-10, abs_tol=0):
        raise ValueError('Saved FQ does not match the same-reference exact KS')
    for key in ['forget_quality', 'model_utility']:
        if not math.isclose(native[key]['agg_value'], row[key], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError(f'Inconsistent {key}')
    expected = {'retain': (400, ['Q_A_Prob', 'Q_A_PARA_Prob', 'Q_A_PERT_Prob', 'Q_A_ROUGE', 'Truth_Ratio']),
                'ra': (100, ['Q_A_Prob', 'Q_A_PERT_Prob', 'Q_A_Prob_normalised', 'Q_A_ROUGE', 'Truth_Ratio']),
                'wf': (117, ['Q_A_Prob', 'Q_A_PERT_Prob', 'Q_A_Prob_normalised', 'Q_A_ROUGE', 'Truth_Ratio'])}
    for prefix, (size, suffixes) in expected.items():
        for suffix in suffixes:
            if len(native[f'{prefix}_{suffix}']['value_by_index']) != size:
                raise ValueError('Incomplete MU per-sample metrics')
    fixed = [json.loads(line) for line in (run / 'fixed_loss_samples.jsonl').read_text().splitlines()
             if line.strip() and json.loads(line)['step'] == step]
    for split, size in [('forget', row['forget_count']), ('retain', row['retain_count'])]:
        samples = [s for s in fixed if s['split'] == split]
        if len(samples) != size or len({s['dataset_index'] for s in samples}) != size:
            raise ValueError('Incomplete fixed-dataset component losses')
    for value, count in [('raw_sensitive_loss', 'sensitive_token_count'), ('raw_common_kl', 'common_token_count'), ('raw_retain_nll', 'retain_token_count')]:
        items = [s for s in fixed if s[value] is not None]
        reconstructed = sum(s[value] * s[count] for s in items) / max(sum(s[count] for s in items), 1)
        if not math.isfinite(reconstructed) or not math.isclose(reconstructed, row[value], rel_tol=1e-8, abs_tol=1e-9):
            raise ValueError('Fixed per-sample losses do not reproduce the epoch record')
    with (folder / 'truth_ratios.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (folder / 'fixed_loss_samples.jsonl').write_text(''.join(json.dumps(s, allow_nan=False) + '\n' for s in fixed))
    inventory = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()
                 if p.is_file() and p.name != 'verification.json'}
    receipt = dict(status='verified', step=step, FQ=float(result.pvalue), KS_D=float(result.statistic),
                   TR_count=len(records), fixed_samples=len(fixed), reference_sha256=row['reference_sha256'],
                   files=inventory, intermediate_weights_written_by_trainer=False)
    write_json(folder / 'verification.json', receipt)
    removed = prune_weights(run, step, final_step)
    write_json(folder / 'verification.json', dict(receipt, removed_intermediate_weight_files=removed))
