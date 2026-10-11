#!/usr/bin/env python3
"""Four PALU loss curves and Retain-reference KL, with pointwise seed-level t CIs."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

# Use a writable cache and a non-interactive backend on servers.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "palu-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t

VARIANTS = ("full_mean", "stop_mean")
LOSSES = (("loss_total", "Total", "#222222"), ("loss_forget", "Forget (weighted)", "#D55E00"),
          ("loss_retain_nll", "Retain NLL (weighted)", "#0072B2"),
          ("loss_common_kl", "Common-token KL (weighted)", "#009E73"))
KLS = (("forget_repr_kl", "Forget: KL(Retain || current)", "#D55E00"),
       ("retain_repr_kl", "Retain: KL(Retain || current)", "#0072B2"))


def mean_interval(values):
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Confidence interval requires finite observations")
    n, mean = len(values), statistics.mean(values)
    std = statistics.stdev(values) if n >= 2 else None
    half = float(t.ppf(.975, n - 1)) * std / math.sqrt(n) if std is not None else None
    return {"n": n, "mean": mean, "sample_stddev": std, "ci95_half_width": half,
            "ci95_low": mean - half if half is not None else None,
            "ci95_high": mean + half if half is not None else None}


def read_history(path):
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    rows = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not text.endswith("\n"):
                break  # A concurrently running worker may be halfway through its final line.
            raise
        step = int(row["step"])
        if step in rows:
            raise ValueError(f"Duplicate step {step}: {path}")
        if "loss_total" in row:
            components = sum(row[key] for key, _, _ in LOSSES[1:])
            if not math.isclose(row["loss_total"], components, rel_tol=2e-3, abs_tol=1e-6):
                raise ValueError(f"Loss components do not add to total: {path}, step={step}")
        rows[step] = row
    return rows


def aggregate(histories, seeds, source, metrics):
    """Use the same available paired seeds in both panels at each step; no interpolation."""
    steps = sorted({step for arm in histories.values() for rows in arm.values() for step in rows})
    output = []
    for step in steps:
        paired = [seed for seed in seeds if all(step in histories[variant].get(seed, {}) for variant in VARIANTS)]
        if not paired:
            continue
        for variant in VARIANTS:
            for key, _, _ in metrics:
                values = [float(histories[variant][seed][step][key]) for seed in paired]
                output.append({"source": source, "variant": variant, "step": step, "metric": key,
                    **mean_interval(values), "seeds": " ".join(map(str, paired)),
                    "expected_n": len(seeds), "all_seeds_present": len(paired) == len(seeds)})
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(ax, rows, variant, metrics, ylabel):
    for key, label, color in metrics:
        points = sorted((row for row in rows if row["variant"] == variant and row["metric"] == key), key=lambda row: row["step"])
        if not points:
            continue
        ax.plot([row["step"] for row in points], [row["mean"] for row in points],
                "o-", color=color, label=label, markersize=3, linewidth=1.4)
        bounded = [row for row in points if row["ci95_half_width"] is not None]
        if bounded:
            ax.errorbar([row["step"] for row in bounded], [row["mean"] for row in bounded],
                yerr=[row["ci95_half_width"] for row in bounded], fmt="none", color=color,
                capsize=3, capthick=.9, elinewidth=.8, errorevery=1)
    ns = sorted({row["n"] for row in rows if row["variant"] == variant})
    ax.set_title(f"{variant} | paired seeds n={','.join(map(str, ns))}")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)


def trend_intervals(histories, seeds):
    rows = []
    for variant in VARIANTS:
        for seed in seeds:
            points = sorted(histories[variant].get(seed, {}).values(), key=lambda row: row["step"])
            for before, after in zip(points, points[1:]):
                dl = after["loss_total"] - before["loss_total"]
                df = after["loss_forget"] - before["loss_forget"]
                dk = after["forget_repr_kl"] - before["forget_repr_kl"]
                rows.append({"seed": seed, "variant": variant, "step_from": before["step"], "step_to": after["step"],
                    "delta_total_loss": dl, "delta_forget_loss": df, "delta_forget_repr_kl": dk,
                    "delta_retain_repr_kl": after["retain_repr_kl"] - before["retain_repr_kl"],
                    "total_down_forget_kl_up": dl < 0 and dk > 0, "forget_down_forget_kl_up": df < 0 and dk > 0})
    return rows


def render(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    seeds = manifest["seeds"]
    training, probes = {name: {} for name in VARIANTS}, {name: {} for name in VARIANTS}
    for planned in manifest["runs"]:
        run, seed, variant = Path(planned["output_dir"]), planned["seed"], planned["variant"]
        training[variant][seed] = read_history(run / "loss_history.jsonl")
        probes[variant][seed] = read_history(run / "probe_history.jsonl")
    train_rows = aggregate(training, seeds, "training", LOSSES)
    probe_rows = aggregate(probes, seeds, "fixed_probe", LOSSES + KLS)
    if not train_rows and not probe_rows:
        return None  # Historical experiments did not record components.
    output = root / "plots"
    output.mkdir(exist_ok=True)
    write_csv(output / "curve_statistics.csv", train_rows + probe_rows)
    write_csv(output / "trend_intervals.csv", trend_intervals(probes, seeds))
    footnote = ("Points: mean across paired training seeds. Caps: pointwise 95% Student-t CI; no CI for n=1.\n"
                "Raw steps, no smoothing/interpolation. CIs describe seed variability, not simultaneous curve coverage.")
    if train_rows:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
        for ax, variant in zip(axes, VARIANTS):
            draw(ax, train_rows, variant, LOSSES, "Training loss (weighted components)")
        fig.suptitle("PALU training losses: total = forget + retain NLL + common-token KL", fontsize=12)
        fig.text(.5, .015, footnote, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .10, 1, .94))
        for extension in ("png", "pdf"):
            fig.savefig(output / f"loss_curves.{extension}", dpi=300)
        plt.close(fig)
    if probe_rows:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey="row")
        for column, variant in enumerate(VARIANTS):
            draw(axes[0, column], probe_rows, variant, LOSSES, "Fixed-probe loss (per-example mean)")
            draw(axes[1, column], probe_rows, variant, KLS, "Pre-head representation KL (nats)")
        fig.suptitle("Same fixed training examples and post-update model state | h[first answer - 1]", fontsize=12)
        fig.text(.5, .015, footnote, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .07, 1, .96))
        for extension in ("png", "pdf"):
            fig.savefig(output / f"loss_vs_retain_kl.{extension}", dpi=300)
        plt.close(fig)
    (output / "README.txt").write_text(footnote + "\nWeighted NLL is measured on Retain training examples.\n"
        "Weighted common-token KL is on Forget examples against the initial Full reference.\n"
        "Representation KL compares softmax(hidden/T) against a separate Retain-only reference.\n"
        "Fixed-probe losses are per-example means; training losses average actual microbatch objectives.\n"
        "trend_intervals.csv marks exploratory opposite-direction intervals, not significant change points.\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(f"Plots: {render(args.root) or 'no paired component histories yet'}")


if __name__ == "__main__":
    main()
