"""临时脚本：汇总 saves/unlearn 下的 TOFU 评估结果，按各指标分别排名。

用法示例：
    python analyze_results.py                      # 各指标 Top-10
    python analyze_results.py --topk 20            # 各指标 Top-20
    python analyze_results.py --last-only          # 只看每个 run 的最后一个 checkpoint
    python analyze_results.py --min-utility 0.6    # 先过滤掉 model_utility 太低的
    python analyze_results.py --model llama-3.1    # 只看某个模型，避免不同模型混排
    python analyze_results.py --csv results.csv    # 导出全部记录
"""

import argparse
import csv
import json
import re
from pathlib import Path

# 指标方向：True 表示越大越好
METRIC_DIRECTION = {
    "forget_quality": True,  # KS 检验 p 值，越大越接近 retain 模型
    "model_utility": True,  # 保留能力的调和平均
    "forget_Q_A_gibberish": True,  # class_id=0 即 "clean"，越大说明输出没退化成乱码
}

DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SUFFIX_RE = re.compile(
    r"target_mode(?P<target_mode>[a-zA-Z]+)"
    r"_first_n(?P<first_n>-?\d+)"
    r"_lr(?P<lr>[0-9.eE+-]+?)"
    r"_b(?P<bsz>\d+)"
    r"_ga(?P<ga>\d+)"
    r"_a(?P<alpha>[0-9.]+)"
    r"_topk(?P<topk>\d+)"
    r"_e(?P<epochs>\d+)"
    r"_day(?P<day>\d+)"
    r"_time(?P<time>\d+)$"
)


def parse_run(summary_path, root):
    """从 .../<split>/<model>/<date>/<trainer>/<suffix>/checkpoint-N/evals/TOFU_SUMMARY.json 解析元信息。

    日期层是后来加的，没有这一层的旧结构也能解析。
    """
    rel = summary_path.relative_to(root).parts
    # rel = (tofu, <split>, <model>, [<date>], <trainer>, <suffix>, checkpoint-N, evals, 文件名)
    if len(rel) < 8 or not rel[-3].startswith("checkpoint-"):
        return None

    i = 3
    date = ""
    if DATE_DIR_RE.match(rel[i]):
        date, i = rel[i], i + 1

    info = {
        "split": rel[1],
        "model": rel[2],
        "date": date,
        "trainer": rel[i],
        "run": rel[i + 1],
        "step": int(rel[-3].split("-")[1]),
        "run_dir": summary_path.parents[2],
    }
    m = SUFFIX_RE.match(info["run"])
    if m:
        info.update(m.groupdict())
    return info


def collect(root):
    records = []
    for path in sorted(root.rglob("checkpoint-*/evals/TOFU_SUMMARY.json")):
        info = parse_run(path, root)
        if info is None:
            continue
        try:
            metrics = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        info["metrics"] = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        records.append(info)

    # 同一个 run 内按 step 排序，checkpoint 的先后顺序即 epoch 序号（step 0 是训练前）
    by_run = {}
    for r in records:
        by_run.setdefault(r["run_dir"], []).append(r)
    for run_records in by_run.values():
        run_records.sort(key=lambda r: r["step"])
        last_step = run_records[-1]["step"]
        for epoch, r in enumerate(run_records):
            r["epoch"] = epoch
            r["is_last"] = r["step"] == last_step
    return records


def fmt_params(r):
    prefix = f"[{r['date']}] " if r.get("date") else ""
    if "lr" not in r:
        return prefix + r["run"]
    return prefix + (
        f"lr={r['lr']} a={r['alpha']} topk={r['topk']} "
        f"first_n={r['first_n']} b={r['bsz']}x{r['ga']} mode={r['target_mode']}"
    )


def match_model(r, pattern):
    """pattern 为 None 时不过滤，否则按大小写不敏感的子串匹配。"""
    return pattern is None or pattern.lower() in r["model"].lower()


def print_table(title, rows, metric, extra_metrics):
    print(f"\n{title}")
    print("-" * 118)
    header = f"{'#':>3}  {metric:>12}  " + "  ".join(f"{m:>12}" for m in extra_metrics)
    print(f"{header}  {'ep':>3}  {'step':>4}  参数")
    print("-" * 118)
    for i, r in enumerate(rows, 1):
        main = r["metrics"].get(metric)
        others = "  ".join(
            f"{r['metrics'].get(m, float('nan')):>12.4f}" for m in extra_metrics
        )
        print(
            f"{i:>3}  {main:>12.4f}  {others}  "
            f"{r['epoch']:>3}  {r['step']:>4}  {fmt_params(r)}"
        )


def pareto_front(records, metrics):
    """返回在给定指标上非被支配的记录（都按越大越好处理）。"""
    front = []
    for a in records:
        va = [a["metrics"].get(m) for m in metrics]
        if any(v is None for v in va):
            continue
        dominated = False
        for b in records:
            if b is a:
                continue
            vb = [b["metrics"].get(m) for m in metrics]
            if any(v is None for v in vb):
                continue
            if all(y >= x for x, y in zip(va, vb)) and any(y > x for x, y in zip(va, vb)):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="saves/unlearn", help="评估结果根目录")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument(
        "--include-start",
        action="store_true",
        help="把 checkpoint-0（训练前的原始模型）也纳入排名",
    )
    ap.add_argument("--last-only", action="store_true", help="只看每个 run 的最后一个 checkpoint")
    ap.add_argument("--min-utility", type=float, default=None)
    ap.add_argument("--min-forget-quality", type=float, default=None)
    ap.add_argument("--split", default=None, help="只看某个 forget split，如 forget05")
    ap.add_argument(
        "--model",
        default=None,
        help="只看某个模型，子串匹配且大小写不敏感，如 llama-2 / llama-3.1",
    )
    ap.add_argument("--date", default=None, help="只看某一天的实验，如 2026-08-30")
    ap.add_argument("--csv", default=None, help="把全部记录导出到 csv")
    args = ap.parse_args()

    root = Path(args.root)
    records = collect(root)
    if not records:
        print(f"在 {root} 下没找到 TOFU_SUMMARY.json")
        return

    all_metrics = sorted({m for r in records for m in r["metrics"]})
    print(f"共扫描到 {len(records)} 条评估记录，来自 {len({r['run_dir'] for r in records})} 个 run")
    print(f"指标：{', '.join(all_metrics)}")
    print(f"模型：{', '.join(sorted({r['model'] for r in records}))}")
    dates = sorted({r["date"] for r in records if r["date"]})
    if dates:
        print(f"日期：{', '.join(dates)}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            meta_cols = [
                "split", "model", "date", "trainer", "run", "epoch", "step", "is_last",
                "target_mode", "first_n", "lr", "bsz", "ga", "alpha", "topk", "epochs",
            ]
            writer.writerow(meta_cols + all_metrics)
            for r in records:
                writer.writerow(
                    [r.get(c, "") for c in meta_cols]
                    + [r["metrics"].get(m, "") for m in all_metrics]
                )
        print(f"已导出：{args.csv}")

    # checkpoint-0 是训练前的原始模型，作为参考基线单独打印
    baselines = [r for r in records if r["step"] == 0 and match_model(r, args.model)]
    if baselines:
        print("\n训练前基线（checkpoint-0，各 run 应当一致）")
        print("-" * 118)
        for m in all_metrics:
            vals = [r["metrics"][m] for r in baselines if m in r["metrics"]]
            if vals:
                print(f"  {m:>22}: min={min(vals):.4f}  max={max(vals):.4f}  n={len(vals)}")

    pool = records
    if args.split:
        pool = [r for r in pool if r["split"] == args.split]
    if args.model:
        pool = [r for r in pool if match_model(r, args.model)]
    if args.date:
        pool = [r for r in pool if r["date"] == args.date]
    if not args.include_start:
        pool = [r for r in pool if r["step"] != 0]
    if args.last_only:
        pool = [r for r in pool if r.get("is_last")]
    if args.min_utility is not None:
        pool = [r for r in pool if r["metrics"].get("model_utility", -1) >= args.min_utility]
    if args.min_forget_quality is not None:
        pool = [
            r for r in pool if r["metrics"].get("forget_quality", -1) >= args.min_forget_quality
        ]

    print(f"\n参与排名的记录数：{len(pool)}")
    if not pool:
        return

    for metric in all_metrics:
        rows = [r for r in pool if metric in r["metrics"]]
        reverse = METRIC_DIRECTION.get(metric, True)
        rows.sort(key=lambda r: r["metrics"][metric], reverse=reverse)
        extra = [m for m in all_metrics if m != metric]
        arrow = "越大越好" if reverse else "越小越好"
        print_table(f"按 {metric} 排名（{arrow}）Top-{args.topk}", rows[: args.topk], metric, extra)

    trade_off = [m for m in ("forget_quality", "model_utility") if m in all_metrics]
    if len(trade_off) == 2:
        front = pareto_front(pool, trade_off)
        front.sort(key=lambda r: r["metrics"]["forget_quality"], reverse=True)
        print_table(
            f"forget_quality / model_utility 帕累托前沿（共 {len(front)} 个）",
            front,
            "forget_quality",
            [m for m in all_metrics if m != "forget_quality"],
        )


if __name__ == "__main__":
    main()
