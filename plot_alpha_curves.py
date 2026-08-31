"""把不同 alpha 的 TOFU 评估结果画成折线图。

--root 可以指向任意一层（模型层、日期层、trainer 层都行），脚本会递归找 run 目录。

用法示例：
    # 默认：取每个 run 的最后一个 checkpoint，画 3 个指标随 alpha 的变化
    python plot_alpha_curves.py

    # 指定结果目录 + 输出路径
    python plot_alpha_curves.py \
        --root saves/unlearn/tofu/forget05/Llama-2-7b-chat-hf \
        --out figures/alpha_curves.png

    # 只看某一天的实验
    python plot_alpha_curves.py --date 2026-08-30

    # 同时对比多个 epoch（0 是训练前的原始模型）
    python plot_alpha_curves.py --epochs 3,5,10

    # 三个指标画在同一张图里（各自独立 y 轴）
    python plot_alpha_curves.py --layout single
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["forget_quality", "model_utility", "forget_Q_A_gibberish"]

METRIC_LABEL = {
    "forget_quality": "Forget Quality (KS p-value)",
    "model_utility": "Model Utility",
    "forget_Q_A_gibberish": "Fluency (1 - gibberish)",
}

# forget_quality 是 p 值，跨了十几个数量级，线性轴看不出区别
LOG_SCALE_METRICS = {"forget_quality"}

ALPHA_RE = re.compile(r"_a(?P<alpha>[0-9.]+)_topk")
TIME_RE = re.compile(r"_day(?P<day>\d+)_time(?P<time>\d+)$")
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def run_date(run_dir):
    """run 目录上面的日期层，没有这一层（旧结构）时返回空串。"""
    for part in reversed(run_dir.parts):
        if DATE_DIR_RE.match(part):
            return part
    return ""


def collect(root, date=None):
    """递归扫描 root 下的 checkpoint-N/evals/TOFU_SUMMARY.json，按 alpha 聚合。

    返回 {alpha: {"run": 相对 root 的路径, "points": [(epoch, step, metrics), ...]}}。
    同一个 alpha 出现多个 run 时，保留 checkpoint 数最多、时间戳最新的那个。
    """
    by_run = {}
    for summary in root.rglob("checkpoint-*/evals/TOFU_SUMMARY.json"):
        run_dir = summary.parents[2]
        if date is not None and run_date(run_dir) != date:
            continue
        try:
            metrics = json.loads(summary.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        step = int(summary.parents[1].name.split("-")[1])
        by_run.setdefault(run_dir, []).append((step, metrics))

    candidates = {}
    for run_dir, points in by_run.items():
        m = ALPHA_RE.search(run_dir.name)
        if m is None:
            continue
        alpha = float(m.group("alpha"))
        points.sort(key=lambda x: x[0])

        stamp = TIME_RE.search(run_dir.name)
        rank = (len(points), stamp.group("day", "time") if stamp else ("", ""))
        prev = candidates.get(alpha)
        if prev is None or rank > prev[0]:
            # checkpoint 的先后顺序即 epoch 序号，step 0 是训练前的原始模型
            series = [(epoch, step, metrics) for epoch, (step, metrics) in enumerate(points)]
            label = run_dir.relative_to(root).as_posix()
            candidates[alpha] = (rank, {"run": label, "date": run_date(run_dir), "points": series})

    return {alpha: data for alpha, (_, data) in sorted(candidates.items())}


def pick(points, epoch):
    """epoch 为 None 表示取最后一个 checkpoint。"""
    if epoch is None:
        return points[-1]
    for p in points:
        if p[0] == epoch:
            return p
    return None


def print_table(data, epoch_specs):
    for epoch in epoch_specs:
        tag = "last" if epoch is None else f"epoch={epoch}"
        print(f"\n[{tag}]")
        print(f"{'alpha':>8}  {'step':>5}  " + "  ".join(f"{m:>22}" for m in METRICS))
        for alpha, entry in data.items():
            got = pick(entry["points"], epoch)
            if got is None:
                continue
            _, step, metrics = got
            cells = "  ".join(f"{metrics.get(m, float('nan')):>22.6g}" for m in METRICS)
            print(f"{alpha:>8g}  {step:>5}  {cells}")


def series_for(data, metric, epoch):
    xs, ys = [], []
    for alpha, entry in data.items():
        got = pick(entry["points"], epoch)
        if got is None or metric not in got[2]:
            continue
        xs.append(alpha)
        ys.append(got[2][metric])
    return xs, ys


def style_axis(ax, metric, alphas):
    ax.set_xscale("log")
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.minorticks_off()
    ax.set_xlabel(r"$\alpha$")
    ax.grid(True, alpha=0.3, linestyle="--")
    if metric in LOG_SCALE_METRICS:
        ax.set_yscale("log")


def plot_grid(data, epoch_specs, out, title):
    alphas = sorted(data)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.2 * len(METRICS), 4.2))
    for ax, metric in zip(axes, METRICS):
        for epoch in epoch_specs:
            xs, ys = series_for(data, metric, epoch)
            if not xs:
                continue
            label = "last ckpt" if epoch is None else f"epoch {epoch}"
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.8, label=label)
        style_axis(ax, metric, alphas)
        ax.set_ylabel(METRIC_LABEL[metric])
        ax.set_title(METRIC_LABEL[metric])
        if len(epoch_specs) > 1:
            ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save(fig, out)


def plot_single(data, epoch_specs, out, title):
    """三个指标叠在一张图上，每个指标一个 y 轴。"""
    if len(epoch_specs) > 1:
        raise SystemExit("--layout single 只支持一个 epoch，请去掉多余的 --epochs 取值")
    epoch = epoch_specs[0]
    alphas = sorted(data)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    axes = [ax, ax.twinx(), ax.twinx()]
    axes[2].spines["right"].set_position(("axes", 1.16))
    colors = ["tab:red", "tab:blue", "tab:green"]

    handles = []
    for cur_ax, metric, color in zip(axes, METRICS, colors):
        xs, ys = series_for(data, metric, epoch)
        (line,) = cur_ax.plot(
            xs, ys, marker="o", markersize=5, linewidth=1.8, color=color,
            label=METRIC_LABEL[metric],
        )
        handles.append(line)
        cur_ax.set_ylabel(METRIC_LABEL[metric], color=color)
        cur_ax.tick_params(axis="y", colors=color)
        if metric in LOG_SCALE_METRICS:
            cur_ax.set_yscale("log")

    style_axis(ax, "none", alphas)
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    save(fig, out)


def save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"\n已保存：{out}\n         {pdf}")


def parse_epochs(spec):
    if spec is None or spec.strip().lower() in {"", "last"}:
        return [None]
    epochs = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        epochs.append(None if tok.lower() == "last" else int(tok))
    return epochs or [None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="saves/unlearn/tofu/forget05/Llama-2-7b-chat-hf",
        help="结果目录，递归查找其下的 run；可以是模型层、日期层或 trainer 层",
    )
    ap.add_argument("--date", default=None, help="只用某一天的实验，如 2026-08-30")
    ap.add_argument("--out", default="figures/alpha_curves.png")
    ap.add_argument(
        "--epochs",
        default="last",
        help="要画的 epoch，逗号分隔；last 表示最后一个 checkpoint，0 是训练前的原始模型",
    )
    ap.add_argument("--layout", choices=["grid", "single"], default="grid")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"目录不存在：{root}")

    data = collect(root, args.date)
    if not data:
        raise SystemExit(f"在 {root} 下没找到可用的 TOFU_SUMMARY.json")

    print(f"共 {len(data)} 个 alpha：{', '.join(f'{a:g}' for a in data)}")
    for alpha, entry in data.items():
        print(f"  alpha={alpha:<5g} {len(entry['points'])} 个 checkpoint  <- {entry['run']}")

    dates = {entry["date"] for entry in data.values() if entry["date"]}
    if len(dates) > 1:
        print(f"\n注意：这些曲线混了多天的实验（{', '.join(sorted(dates))}），")
        print("      如需只看一天，加 --date YYYY-MM-DD。")

    epoch_specs = parse_epochs(args.epochs)
    print_table(data, epoch_specs)

    title = args.title or root.as_posix()
    if args.date:
        title = f"{title} @ {args.date}"
    out = Path(args.out)
    if args.layout == "grid":
        plot_grid(data, epoch_specs, out, title)
    else:
        plot_single(data, epoch_specs, out, title)


if __name__ == "__main__":
    main()
