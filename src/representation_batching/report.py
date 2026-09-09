"""Offline experiment ledger. Standard library only; optional PyYAML for legacy runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("forget_quality", "model_utility", "forget_Q_A_gibberish", "exact_memorization")
DIAGNOSTICS = ("forget_loss", "retain_loss", "npo_weight_mean", "npo_weight_near_zero_fraction")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def load_lines(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def config_for(run):
    for name in ("resolved_config.json", "resolved_config.yaml", ".hydra/config.yaml"):
        path = run / name
        if path.exists():
            try:
                return read_json(path)
            except ValueError:
                try:
                    import yaml
                except ImportError as exc:
                    raise ValueError("旧版 YAML 配置需要 PyYAML；新运行会同时保存 JSON") from exc
                return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError("缺少 resolved_config.json/yaml")


def training_protocol(config, meta, run):
    # Exclude only identity/logging fields; preserve every optimization/data setting.
    config = json.loads(json.dumps(config))
    trainer = config.get("trainer", {})
    for key in ("output_dir", "logging_dir", "run_name", "seed", "report_to"):
        trainer.get("args", {}).pop(key, None)
    trainer.get("method_args", {}).pop("batch_manifest_path", None)
    runtime_path = run / "batch_audit/runtime_contract.json"
    runtime = read_json(runtime_path) if runtime_path.exists() else {}
    snapshot = run / "batch_audit/code_snapshot"
    code = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(snapshot.rglob("*")) if p.is_file()}
    return digest({
        "model": config.get("model"), "data": config.get("data"),
        "collator": config.get("collator"), "trainer": trainer,
        "feature": {k: meta.get(k) for k in (
            "feature_sha256", "feature_key", "source_qa_content_sha256",
            "effective_batch_size", "retain_size", "num_epochs", "world_size",
            "per_device_batch_size", "gradient_accumulation_steps")},
        "packages": runtime.get("environment", {}).get("packages"), "code": code,
    })


def audit_schedule(run, meta, steps):
    """Compare every rank with the archived plan, including epoch/step/microstep."""
    observed_counts, expected_counts = [], []
    for rank in range(int(meta["world_size"])):
        base = run / "batch_audit"
        planned = base / f"rank-{rank}-planned-microbatches.jsonl"
        observed = base / f"rank-{rank}-observed-microbatches.jsonl"
        if not planned.exists() or not observed.exists():
            return "missing", None, None
        expected, actual = load_lines(planned), load_lines(observed)
        derived = []
        size = int(meta["per_device_batch_size"])
        stride = int(meta["world_size"]) * size
        for step in steps:
            for microstep, start in enumerate(range(0, len(step["forget_indices"]), stride)):
                offset = start + rank * size
                derived.append({"epoch": step["epoch"], "optimizer_step": step["optimizer_step"],
                    "microstep": microstep, "pairs": [
                        {"forget_index": f, "retain_index": r} for f, r in zip(
                            step["forget_indices"][offset:offset+size],
                            step["retain_indices"][offset:offset+size])]})
        if expected != derived:
            return "mismatch", len(actual), len(derived)
        expected_counts.append(len(expected))
        observed_counts.append(len(actual))
        if len(actual) > len(expected) or actual != expected[:len(actual)]:
            return "mismatch", min(observed_counts), max(expected_counts)
    if len(set(expected_counts)) != 1:
        return "mismatch", min(observed_counts), max(expected_counts)
    # Logs alone do not prove the final optimizer step returned successfully.
    status = "complete" if observed_counts == expected_counts else "prefix"
    return status, min(observed_counts), max(expected_counts)


def evaluation_protocol(provenance, run, summary):
    if provenance.get("status") != "completed":
        return None
    cfg = json.loads(json.dumps(provenance.get("config", {})))
    model = cfg.get("model", {})
    target = model.get("model_args", {}).get("pretrained_model_name_or_path")
    training = config_for(run)
    recorded_paths = [str(run), training.get("paths", {}).get("output_dir"),
                      training.get("trainer", {}).get("args", {}).get("output_dir")]
    allowed = {Path(p).expanduser().resolve() for p in recorded_paths if p}
    if not target or Path(target).expanduser().resolve() not in allowed:
        return None  # Never silently attach checkpoint/baseline metrics to the final model.
    if not provenance.get("code_sha256"):
        return None
    for key in ("model_args", "tokenizer_args"):
        model.get(key, {}).pop("pretrained_model_name_or_path", None)
    evaluation = cfg.get("eval", {})
    for entry in evaluation.values():
        if isinstance(entry, dict):
            entry.pop("output_dir", None)
            entry.pop("overwrite", None)
    return digest({"model": model, "eval": evaluation, "seed": cfg.get("seed"),
                   "code": provenance["code_sha256"], "references": provenance.get("reference_sha256")})


def collect_run(run, links, warnings, include_evaluation=True):
    run = run.expanduser().resolve()
    candidates = []
    for path in sorted((run / "batch_audit").glob("*.jsonl")):
        if "microbatches" in path.name:
            continue
        records = load_lines(path)
        if records and records[0].get("type") == "metadata":
            candidates.append((path, records))
    if len(candidates) != 1:
        raise ValueError(f"需要恰好一份归档 manifest，发现 {len(candidates)} 份")
    manifest_path, records = candidates[0]
    meta, steps = records[0], records[1:]
    config = config_for(run)
    args = config.get("trainer", {}).get("args", {})
    status_path = run / "training_status.json"
    status = read_json(status_path) if status_path.exists() else {}
    audit, consumed, expected = audit_schedule(run, meta, steps)
    limited = int(args.get("max_steps", -1)) > 0
    state_path = run / "trainer_state.json"
    state = read_json(state_path) if state_path.exists() else {}
    expected_steps = len(steps)
    full_complete = (status.get("status") == "completed" or
                     (not status and state.get("global_step") == expected_steps))
    if limited:
        training_state = "limited"
    elif status.get("status") == "failed":
        training_state = "failed"
    elif audit == "mismatch":
        training_state = "audit_mismatch"
    elif full_complete and audit == "complete":
        training_state = "completed"
    else:
        training_state = "incomplete_or_unknown"
    arm = str(meta["method"]) + "/" + str(meta.get("batch_order", "random"))
    row = {
        "run": run.name, "run_dir": str(run), "arm": arm, "seed": meta["seed"],
        "training_status": training_state, "audit": audit,
        "observed_microbatches": consumed, "expected_microbatches": expected,
        "feature_key": meta.get("feature_key"), "learning_rate": args.get("learning_rate"),
        "global_step": status.get("global_step", state.get("global_step")),
        "protocol": training_protocol(config, meta, run),
        "manifest": str(manifest_path), "evaluation_status": "missing",
        "summary_path": None, "evaluation_protocol": None,
        "error": status.get("error"),
    }
    for name in ("within_batch_cosine", "previous_batch_cosine", "question_tokens_mean", "answer_tokens_mean", "initial_answer_nll_mean"):
        values = [s[name] for s in steps if finite(s.get(name))]
        row[name] = statistics.mean(values) if values else None
    diagnostics_path = run / "training_diagnostics.jsonl"
    if diagnostics_path.exists():
        curves = load_lines(diagnostics_path)
    else:
        curves = [x for x in state.get("log_history", []) if "npo_weight_mean" in x]
    row["diagnostic_points"] = len(curves)
    for name in DIAGNOSTICS:
        row["last_" + name] = next((x[name] for x in reversed(curves) if finite(x.get(name))), None)
    if not include_evaluation:
        row["eligible"] = False
        return row, curves, steps
    candidates = set()
    conventional = run / "evals/TOFU_SUMMARY.json"
    if conventional.exists():
        candidates.add(conventional.resolve())
    for link in links:
        if Path(link["run_dir"]).resolve() == run:
            candidates.add(Path(link["summary"]).resolve())
    if len(candidates) > 1:
        row["evaluation_status"] = "ambiguous"
        warnings.append(f"{run}: 多份最终评估，未自动择优或取最新")
    elif candidates:
        path = candidates.pop()
        metrics = read_json(path)
        row["summary_path"] = str(path)
        row.update({key: value for key, value in metrics.items() if finite(value) and key not in row})
        provenance_path = path.parent / "evaluation_provenance.json"
        provenance = read_json(provenance_path) if provenance_path.exists() else {}
        protocol = evaluation_protocol(provenance, run, path)
        row["evaluation_protocol"] = protocol
        required = set(provenance.get("config", {}).get("eval", {}).get("tofu", {}).get("metrics", {}))
        available = {key for key, value in metrics.items() if finite(value)}
        row["evaluation_status"] = "verified" if protocol and required and required <= available else "unverified_or_partial"
    row["eligible"] = row["training_status"] == "completed" and row["evaluation_status"] == "verified"
    return row, curves, steps


def summarize(rows, reference, warnings):
    # Duplicate seed/arm runs are explicit ambiguities, never extra independent seeds.
    groups = defaultdict(list)
    for row in rows:
        if row["eligible"]:
            groups[(row["protocol"], row["evaluation_protocol"], row["arm"], row["seed"])].append(row)
    unique = []
    for key, matches in groups.items():
        if len(matches) != 1:
            warnings.append(f"重复 seed/arm（{key}），{len(matches)} 次运行均不自动进入均值或配对比较")
        else:
            unique.append(matches[0])
    by_arm = defaultdict(list)
    for row in unique:
        by_arm[(row["protocol"], row["evaluation_protocol"], row["arm"])].append(row)
    aggregates = []
    pairs = []
    lookup = {(r["protocol"], r["evaluation_protocol"], r["arm"], r["seed"]): r for r in unique}
    for (protocol, evaluation, arm), members in sorted(by_arm.items()):
        for metric in METRICS:
            values = [r[metric] for r in members if finite(r.get(metric))]
            if values:
                aggregates.append(dict(protocol=protocol, evaluation_protocol=evaluation, arm=arm,
                    metric=metric, n_seeds=len(values), mean=statistics.mean(values),
                    std=statistics.stdev(values) if len(values) > 1 else None))
        if arm == reference:
            continue
        for row in members:
            base = lookup.get((protocol, evaluation, reference, row["seed"]))
            if base:
                for metric in METRICS:
                    if finite(row.get(metric)) and finite(base.get(metric)):
                        pairs.append(dict(protocol=protocol, evaluation_protocol=evaluation, arm=arm,
                            reference=reference, seed=row["seed"], metric=metric,
                            delta=row[metric] - base[metric], run=row["run"], reference_run=base["run"]))
    return aggregates, pairs


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value is None:
        return "—"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def table(rows, columns):
    if not rows:
        return '<p class="empty">暂无符合条件的数据。</p>'
    def cell(row, key):
        value = html.escape(fmt(row.get(key)))
        if key in ("run_dir", "summary_path", "manifest") and row.get(key):
            return f'<a href="{html.escape(Path(row[key]).as_uri(), quote=True)}">{value}</a>'
        return value
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in columns) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{cell(r,c)}</td>' for c in columns) + '</tr>' for r in rows) + '</tbody></table></div>'


def chart(records, key, x_key="optimizer_step"):
    points = [(r.get(x_key, r.get("step", i+1)), r[key]) for i,r in enumerate(records) if finite(r.get(key))]
    points = [(x,y) for x,y in points if finite(x)]
    if not points:
        return '<span class="empty">无曲线数据</span>'
    xmin, xmax = min(x for x,y in points), max(x for x,y in points)
    ymin, ymax = min(y for x,y in points), max(y for x,y in points)
    coords = ' '.join(f'{45+(x-xmin)/max(xmax-xmin,1)*460:.2f},{135-(y-ymin)/(ymax-ymin or 1)*105:.2f}' for x,y in points)
    return (f'<svg viewBox="0 0 550 175" role="img" aria-label="{html.escape(key)}">'
            f'<text x="8" y="15">{html.escape(key)} · {len(points)} points</text>'
            f'<path d="M45 25 V135 H510" fill="none" stroke="#aaa"/>'
            f'<polyline points="{coords}" fill="none" stroke="#176a89" stroke-width="2"/>'
            f'<text x="2" y="40">{ymax:.3g}</text><text x="2" y="135">{ymin:.3g}</text>'
            f'<text x="45" y="158">{xmin:g}</text><text x="390" y="158">step {xmax:g}</text></svg>')


def build_report(config, out):
    out.mkdir(parents=True, exist_ok=True)
    warnings, rows, details = [], [], []
    roots = [Path(p).expanduser().resolve() for p in config.get("roots", [])]
    runs = set()
    for root in roots:
        if not root.exists():
            warnings.append(f"扫描目录不存在：{root}")
            continue
        if (root / "batch_audit").is_dir():
            runs.add(root)
        runs.update(p.parent for p in root.rglob("batch_audit") if p.is_dir())
    for run in sorted(runs):
        try:
            row, curves, steps = collect_run(run, config.get("evaluations", []), warnings)
            rows.append(row)
            details.append((row, curves, steps))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"{run}: {exc}")
    aggregates, pairs = summarize(rows, config.get("reference_arm", "R/random"), warnings)
    coverage = []
    for arm in config.get("expected_arms", ["R/random", "S/random", "D/random", "P/random"]):
        for seed in config.get("expected_seeds", [0, 1, 2]):
            matches = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
            coverage.append(dict(arm=arm, seed=seed, runs=len(matches),
                eligible=sum(r["eligible"] for r in matches),
                status="missing" if not matches else "; ".join(sorted({r["training_status"]+" / "+r["evaluation_status"] for r in matches}))))
    baselines = []
    for base in config.get("baselines", []):
        try:
            baselines.append({"label": base["label"], "summary_path": base["summary"],
                "note": "参考值；未自动认定与本次评估协议一致",
                **{k:v for k,v in read_json(base["summary"]).items() if finite(v)}})
        except (OSError, ValueError) as exc:
            warnings.append(f"基线 {base}: {exc}")
    for name, data in (("runs", rows), ("coverage", coverage), ("aggregates", aggregates), ("paired_deltas", pairs)):
        write_csv(out / f"{name}.csv", data)
    batch_rows = [{"run": row["run"], "run_dir": row["run_dir"], "arm": row["arm"], "seed": row["seed"],
                   **{key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value)
                      for key, value in step.items()}} for row, _, steps in details for step in steps]
    write_csv(out / "batch_details.csv", batch_rows)
    payload = dict(roots=[str(p) for p in roots], runs=rows, coverage=coverage,
                   aggregates=aggregates, paired_deltas=pairs, baselines=baselines, warnings=warnings)
    (out / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    sections = [f'<h1>NPO 分组与顺序实验</h1><p>{len(rows)} 次运行 · {sum(r["eligible"] for r in rows)} 次可用于正式比较</p>',
        '<p>先看完成状态与审计，再看分组是否拉开差异，最后比较同协议、同 seed 的评估结果。缺失值显示为 —，不填零、不自动选最佳重复运行。</p>',
        '<nav><a href="runs.csv">实验总表 CSV</a> · <a href="batch_details.csv">逐批次统计 CSV</a> · <a href="paired_deltas.csv">配对差值 CSV</a> · <a href="results.json">汇总数据 JSON</a></nav>',
        '<h2>扫描范围</h2><p>'+html.escape(' · '.join(str(p) for p in roots))+'</p>',
        '<h2>计划覆盖</h2><p>跨配置的总体覆盖；正式汇总仍按 protocol 分开。limited 为限步/冒烟运行。</p>'+table(coverage,["arm","seed","runs","eligible","status"]),
        '<h2>实验总表</h2><input id="filter" placeholder="筛选：seed、S/random、completed、目录等"><div id="ledger">'+table(rows,["run","arm","seed","training_status","audit","evaluation_status","protocol","feature_key","learning_rate","within_batch_cosine",*METRICS,"run_dir","summary_path"])+"</div>",
        '<h2>同协议跨 seed 汇总</h2><p>只有完整训练、所有 rank 审计通过且评估来源已验证的唯一 seed/arm 进入汇总。std 是样本标准差；只有一个 seed 时不报告 std。</p>'+table(aggregates,["protocol","evaluation_protocol","arm","metric","n_seeds","mean","std"]),
        '<h2>同 seed 配对差值</h2><p>delta = 当前组 − 参考组；参考组为 '+html.escape(config.get("reference_arm","R/random"))+'. 差值正负不自动代表优劣。</p>'+table(pairs,["protocol","arm","reference","seed","metric","delta"]),
        '<h2>Full / Retain 参考</h2>'+table(baselines,["label",*METRICS,"note","summary_path"]),
        '<h2>指标解释</h2><p>forget_quality 是 KS 检验 p 值，不是遗忘百分比；较大 p 值不能证明分布相同或知识删除。model_utility 衡量保留能力；forget_Q_A_gibberish 是 clean 类概率。exact_memorization 应结合匹配的 Retain 参考解释，不能只追求越低越好。训练损失或表征相似度不能替代遗忘评估。</p>',
        '<h2>逐次运行：分组检查与训练轨迹</h2>']
    for row, curves, steps in details:
        sections.append('<details><summary>'+html.escape(row["run"]+' · '+row["arm"]+' · seed '+str(row["seed"]))+'</summary>'+table([row],["question_tokens_mean","answer_tokens_mean","initial_answer_nll_mean","observed_microbatches","expected_microbatches","error"]))
        sections.append('<div class="charts">'+''.join(chart(curves,k) for k in DIAGNOSTICS)+chart(steps,"within_batch_cosine","report_step")+chart(steps,"previous_batch_cosine","report_step")+'</div>'+table(steps,["epoch","optimizer_step","within_batch_cosine","previous_batch_cosine","question_tokens_mean","answer_tokens_mean","initial_answer_nll_mean","author_counts"])+'</details>')
    sections.append('<h2>待处理项 / 读取问题</h2><ul>'+''.join('<li>'+html.escape(w)+'</li>' for w in warnings)+'</ul>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NPO 实验总览</title><style>
body{font:15px/1.6 system-ui,sans-serif;color:#20313e;background:#f5f7fa;margin:0;padding:32px}h1{margin-top:0}h2{margin-top:32px}p{max-width:1100px}a{color:#126a8a}.scroll{overflow:auto;background:white;border:1px solid #dce2e8;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px 12px;text-align:left;border-bottom:1px solid #eee;white-space:nowrap}th{background:#e7eef4;position:sticky;top:0}input{padding:10px;width:360px;max-width:90%;margin-bottom:12px}.empty{color:#7b8793}details{background:white;padding:15px;margin:12px 0;border-radius:8px}summary{cursor:pointer;font-weight:600}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:15px}svg{width:100%;font-size:11px}nav{margin:20px 0}</style><body>'''+''.join(sections)+'''<script>document.getElementById('filter').addEventListener('input',e=>{let q=e.target.value.toLowerCase();document.querySelectorAll('#ledger tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></body></html>'''
    (out / "index.html").write_text(document, encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", action="append", type=Path, help="Repeat to scan multiple training roots")
    parser.add_argument("--out", type=Path, default=Path("reports/representation_batching"))
    args = parser.parse_args()
    config = read_json(args.config) if args.config else {}
    # Config paths resolve relative to the config file; CLI paths relative to cwd.
    base = args.config.resolve().parent if args.config else Path.cwd()
    def resolve(p):
        path = Path(p).expanduser()
        return str((base / path).resolve() if not path.is_absolute() else path.resolve())
    config["roots"] = [resolve(p) for p in config.get("roots", [])]
    if args.root:
        config["roots"] = [str(p.expanduser().resolve()) for p in args.root]
    if not config["roots"]:
        config["roots"] = [str(Path("saves/unlearn").resolve())]
    for link in config.get("evaluations", []):
        for key in ("run_dir", "summary"):
            link[key] = resolve(link[key])
    for baseline in config.get("baselines", []):
        baseline["summary"] = resolve(baseline["summary"])
    payload = build_report(config, args.out.resolve())
    print(f"Report: {args.out.resolve() / 'index.html'}")
    print(f"Runs: {len(payload['runs'])}; warnings: {len(payload['warnings'])}")


if __name__ == "__main__":
    main()
