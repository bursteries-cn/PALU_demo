#!/usr/bin/env python3
"""Plot PALU seed experiments using complete, unsmoothed epoch and step histories.

Usage: python scripts/palu/plot_seed_epoch.py EXPERIMENT_ROOT
The group manifest contains runs with model, seed and output_dir. A single-run
manifest or a directory of single-run manifests is also accepted. Absolute paths
in a downloaded manifest can be remapped with --path-map REMOTE_ROOT=LOCAL_ROOT.
Only fixed-data, post-update epoch losses are paired with FQ/MU in scatter plots.
Online losses are pre-update microbatch averages and have separate figures.
"""
import argparse
import csv
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "palu-seed-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogFormatterMathtext,
                               LogLocator, MaxNLocator, NullFormatter, NullLocator)
import numpy as np

LOSS_METRICS = (
    ("raw_sensitive_loss", "Raw sensitive-token loss"),
    ("raw_retain_nll", "Raw retain NLL"),
    ("raw_common_kl", "Raw common-token KL"),
    ("loss_total", "Weighted total loss"),
)
EVAL_METRICS = (
    ("forget_quality", "FQ: KS p-value"),
    ("model_utility", "MU"),
    ("ks_d", "Truth-ratio KS D"),
)
ALIASES = {"forget_quality": ("forget_quality", "FQ", "fq"),
           "model_utility": ("model_utility", "MU", "mu"),
           "ks_d": ("ks_d", "forget_ks_statistic", "ks_statistic", "D")}
# Okabe-Ito colors; seed identity remains stable across all figures for a model.
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p")
FOOTNOTE = ("Lines: individual training seeds and their arithmetic mean; shading: sample SD across seeds.\n"
            "All logged points; no smoothing or interpolation. Seed SD is not a confidence interval.")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    rows = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(text.splitlines()) - 1 and not text.endswith("\n"):
                break  # A running worker may be appending its final record.
            raise ValueError(f"Invalid complete JSONL line {i + 1}: {path}") from None
    return rows


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def portable_hash(value):
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value


def remap_path(value, base, mappings):
    path = Path(value)
    for before, after in mappings:
        try:
            path = after / path.relative_to(before)
            break
        except ValueError:
            continue
    return path if path.is_absolute() else base / path


def load_runs(root, mappings):
    """Preserve missing runs and their planned seeds instead of dropping coverage."""
    top = root / "manifest.json"
    documents = [(top, read_json(top))] if top.is_file() else [
        (path, read_json(path)) for path in sorted(root.rglob("manifest.json"))
        if "plots" not in path.relative_to(root).parts]
    runs, planned = [], {}
    for manifest_path, manifest in documents:
        settings = manifest.get("settings", {})
        entries = manifest.get("runs")
        if not entries:
            if "model" in manifest and "seed" in manifest:
                entries = [dict(manifest, output_dir=str(manifest_path.parent))]
            else:
                continue
        for entry in entries:
            model = entry.get("model", manifest.get("model", settings.get("model")))
            seed = entry.get("seed")
            if model is None or seed is None or "output_dir" not in entry:
                raise ValueError(f"Each run needs model, seed and output_dir: {manifest_path}")
            model, seed = str(model), int(seed)
            path = remap_path(entry["output_dir"], manifest_path.parent, mappings)
            detail_path = path / "manifest.json"
            detail = read_json(detail_path) if detail_path.is_file() else {}
            run = {"model": model, "seed": seed, "path": path, "entry": entry,
                   "manifest": detail, "group_manifest": manifest}
            runs.append(run)
            expected = entry.get("planned_seeds", manifest.get("planned_seeds", manifest.get("seeds", [seed])))
            # Optional per-model planning avoids mixing seed lists for different models.
            groups = manifest.get("models", {})
            if isinstance(groups, dict) and isinstance(groups.get(model), dict):
                expected = groups[model].get("planned_seeds", groups[model].get("seeds", expected))
            planned.setdefault(model, set()).update(int(s) for s in expected)
            planned[model].add(seed)
    unique = {}
    for run in runs:
        key = run["model"], run["seed"]
        if key in unique and unique[key]["path"].resolve() != run["path"].resolve():
            raise ValueError(f"Duplicate model/seed with different outputs: {key}")
        unique[key] = run
    if not unique:
        raise ValueError(f"No run manifests found under {root}")
    return list(unique.values()), planned


def normalize_rows(run, filename, source):
    rows, seen = [], {}
    for original in read_jsonl(run["path"] / filename):
        if not finite(original.get("epoch")):
            raise ValueError(f"Missing real epoch in {run['path'] / filename}")
        row = dict(original)
        if "seed" in row and int(row["seed"]) != run["seed"]:
            raise ValueError(f"Record seed differs from manifest: {run['path'] / filename}")
        row.update(model=run["model"], seed=run["seed"], source=source,
                   output_dir=str(run["path"]), epoch=float(row["epoch"]))
        row["run_status"] = run["manifest"].get("status", run["entry"].get("status", "unknown"))
        row["step"] = int(row.get("step", row.get("global_step", -1)))
        if row["step"] < 0:
            raise ValueError(f"Missing optimizer step: {run['path'] / filename}")
        for key, names in ALIASES.items():
            for name in names:
                value = original.get(name, original.get("metrics", {}).get(name))
                if isinstance(value, dict):
                    value = value.get("agg_value", value.get("value"))
                if value is not None:
                    row[key] = value
                    break
        for key, _ in LOSS_METRICS + EVAL_METRICS:
            if row.get(key) is not None and not finite(row[key]):
                raise ValueError(f"Nonfinite {key}, seed {run['seed']}, step {row['step']}")
        for key in ("forget_quality", "model_utility", "ks_d"):
            if finite(row.get(key)) and not 0 <= row[key] <= 1:
                raise ValueError(f"{key} outside [0, 1] in {run['path'] / filename}")
        row["fixed_dataset_sha256"] = portable_hash(row.get("fixed_dataset_sha256",
            run["manifest"].get("fixed_dataset_sha256", run["manifest"].get("fixed_dataset_sha"))))
        row["config_sha256"] = run["manifest"].get("config_sha256", run["manifest"].get("config_sha", run["entry"].get("config_sha")))
        row["protocol_sha256"] = run["manifest"].get("protocol_sha256", run["entry"].get("protocol_sha256"))
        # Re-evaluation of the same state must agree; retain only one plotted point.
        identity = row["step"], round(row["epoch"], 10)
        if identity in seen:
            before = seen[identity]
            if any(before.get(key) != row.get(key) for key, _ in LOSS_METRICS + EVAL_METRICS):
                raise ValueError(f"Conflicting repeated state: {run['path'] / filename}, {identity}")
            if row.get("evaluation_kind") == "final_tail":
                before["evaluation_kind"] = "final_tail"
            continue
        seen[identity] = row
        rows.append(row)
    return sorted(rows, key=lambda row: (row["step"], row["epoch"]))


def write_csv(path, rows, fields=None):
    fields = fields or sorted({key for row in rows for key in row if key != "metrics"})
    # Nested diagnostic summaries stay in source JSONL; scalar CSV remains inspectable.
    fields = [key for key in fields if not any(isinstance(row.get(key), (dict, list)) for row in rows)]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def curve_statistics(rows, seeds, x_key, metrics):
    output = []
    for x in sorted({round(row[x_key], 10) for row in rows}):
        point = [row for row in rows if round(row[x_key], 10) == x]
        for key, _ in metrics:
            present = [row for row in point if finite(row.get(key))]
            values = [row[key] for row in present]
            if not values:
                continue
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else None
            output.append({"model": point[0]["model"], "source": point[0]["source"],
                x_key: x, "metric": key, "n_seeds": len(values), "planned_n_seeds": len(seeds),
                "seeds": " ".join(str(row["seed"]) for row in present),
                "all_planned_seeds_present": {row["seed"] for row in present} == set(seeds),
                "mean": mean, "sample_sd": sd, "mean_minus_sd": mean - sd if sd is not None else None,
                "mean_plus_sd": mean + sd if sd is not None else None,
                "zero_p_values": sum(value == 0 for value in values) if key == "forget_quality" else None})
    return output


def style_axis(ax, label, x_label, log=False):
    ax.set_xlabel(x_label)
    ax.set_ylabel(label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.18, linewidth=.5)
    if log:
        ax.set_yscale("log")


def format_log_ticks(ax, coordinate, endpoint=False):
    """Sparse labels on log axes, with compact values for a narrow endpoint span."""
    axis = ax.xaxis if coordinate == "x" else ax.yaxis
    low, high = ax.get_xlim() if coordinate == "x" else ax.get_ylim()
    axis.set_minor_formatter(NullFormatter())
    if endpoint and low > 0 and high / low < 10:
        ticks = MaxNLocator(nbins=4, min_n_ticks=3).tick_values(low, high)
        ticks = [float(value) for value in ticks if low <= value <= high and value > 0]
        if len(ticks) < 2:
            ticks = np.geomspace(low, high, 4).tolist()
        if len(ticks) > 5:
            ticks = [ticks[index] for index in np.linspace(0, len(ticks) - 1, 5, dtype=int)]
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4g}"))
        axis.set_minor_locator(NullLocator())
    else:
        axis.set_major_locator(LogLocator(base=10, subs=(1,), numticks=5))
        axis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
        axis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=12))


def annotate_endpoint_points(fig, axes_points):
    """Place seed labels after layout; offsets/leader lines leave data unchanged."""
    fig.canvas.draw()
    pixels_per_point = fig.dpi / 72
    candidates = ((6, 9), (6, -10), (-6, 9), (-6, -10), (6, 18), (-6, 18),
                  (6, -18), (-6, -18), (0, 25), (0, -25))
    for ax, points in axes_points:
        occupied = []
        point_positions = [ax.transData.transform((x, y)) for _, x, y in points]
        bounds = ax.get_window_extent()
        for seed, x, y in sorted(points, key=lambda point: (point[1], point[2], point[0])):
            px, py = ax.transData.transform((x, y))
            width = len(str(seed)) * 7 * .62 * pixels_per_point
            height = 7 * 1.2 * pixels_per_point
            chosen = None
            for dx, dy in candidates:
                tx, ty = px + dx * pixels_per_point, py + dy * pixels_per_point
                left = tx if dx > 0 else tx - width if dx < 0 else tx - width / 2
                box = (left - 2 * pixels_per_point, ty - height / 2 - pixels_per_point,
                       left + width + 2 * pixels_per_point, ty + height / 2 + pixels_per_point)
                inside = box[0] >= bounds.x0 and box[2] <= bounds.x1 and box[1] >= bounds.y0 and box[3] <= bounds.y1
                overlaps = any(box[0] < old[2] and box[2] > old[0] and box[1] < old[3] and box[3] > old[1]
                               for old in occupied)
                covers_point = any(box[0] < ox + 3 * pixels_per_point and box[2] > ox - 3 * pixels_per_point
                                   and box[1] < oy + 3 * pixels_per_point and box[3] > oy - 3 * pixels_per_point
                                   for ox, oy in point_positions)
                if inside and not overlaps and not covers_point:
                    chosen = dx, dy, box
                    break
            if chosen is None:
                dx, dy = candidates[len(occupied) % len(candidates)]
            else:
                dx, dy, box = chosen
                occupied.append(box)
            ax.annotate(str(seed), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                        ha="left" if dx > 0 else "right" if dx < 0 else "center", va="center",
                        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": .45}, zorder=6)


def observed_coverage(rows, seeds, x_key=None):
    """Count distinct training seeds, including measurements whose p-value is zero."""
    if x_key is None:
        counts = [len({row["seed"] for row in rows})]
        label = "Observed endpoint seeds"
    else:
        points = sorted({round(row[x_key], 10) for row in rows})
        counts = [len({row["seed"] for row in rows if round(row[x_key], 10) == point}) for point in points]
        unit = "logged epoch" if x_key == "epoch" else "optimizer step"
        label = f"Observed seeds per {unit}"
    low, high = (min(counts), max(counts)) if counts else (0, 0)
    count_text = str(low) if low == high else f"{low}-{high}"
    return f"{label}: n={count_text} / {len(seeds)} planned", low, high


def coverage_footer(rows, seeds, x_key=None):
    text, low, high = observed_coverage(rows, seeds, x_key)
    if high < 2:
        text += "; seed SD unavailable (n < 2)."
    elif low < 2:
        text += "; seed SD unavailable at points with n < 2."
    if x_key is None:
        observed = {row["seed"] for row in rows}
        incomplete = sum(row.get("run_status") != "completed" for row in rows)
        pending = len(set(seeds) - observed)
        if incomplete or pending:
            text += f"\nInterim last observed states: {incomplete} observed run(s) incomplete/unknown; {pending} planned seed(s) without data."
        else:
            text += "\nAll observed runs are completed; points remain their last observed states."
    return text


def draw_curve(ax, rows, stats, key, label, seeds, x_key, single=False):
    log = key == "forget_quality"
    zeros = 0
    for seed in seeds:
        values = [row for row in rows if row["seed"] == seed]
        xs = [row[x_key] for row in values]
        ys = [row.get(key, math.nan) for row in values]
        zeros += sum(value == 0 for value in ys) if log else 0
        # NaN preserves visible gaps for unrepresentable zero p-values.
        ys = [value if finite(value) and (value > 0 or not log) else math.nan for value in ys]
        if not any(math.isfinite(value) for value in ys):
            continue
        ax.plot(xs, ys, marker="o" if x_key == "epoch" else None, markersize=2.8,
                color=COLORS[seeds.index(seed) % len(COLORS)], linewidth=1.1,
                alpha=1 if single else .75, label=f"Seed {seed}")
    if not single:
        selected = [row for row in stats if row["metric"] == key and x_key in row]
        xs = np.array([row[x_key] for row in selected], dtype=float)
        means = np.array([row["mean"] for row in selected], dtype=float)
        plotted = np.where(means > 0, means, np.nan) if log else means
        if np.any(np.isfinite(plotted)):
            ax.plot(xs, plotted, color="#222222", linewidth=1.8, label="Seed mean", zorder=5)
        lows = np.array([row["mean_minus_sd"] if row["sample_sd"] is not None else math.nan for row in selected])
        highs = np.array([row["mean_plus_sd"] if row["sample_sd"] is not None else math.nan for row in selected])
        where = np.isfinite(lows) & np.isfinite(highs)
        if log:
            where &= lows > 0  # Do not create an arbitrary lower plotting epsilon.
        if np.any(where):
            ax.fill_between(xs, lows, highs, where=where, color="#777777", alpha=.20, label="Seed SD")
    style_axis(ax, label, "Logged training epoch" if x_key == "epoch" else "Optimizer step", log)
    if log and zeros:
        ax.text(.02, .02, f"{zeros} zero p-value(s) retained in CSV; absent on log axis",
                transform=ax.transAxes, fontsize=7, va="bottom")


def save_figure(fig, output, name, footnote, endpoint_points=None):
    fig.text(.5, .012, footnote, ha="center", va="bottom", fontsize=7)
    footer_height = max(.075, (8.8 * len(footnote.splitlines()) + 10) / (72 * fig.get_figheight()))
    fig.tight_layout(rect=(0, footer_height, 1, .945))
    if endpoint_points:
        annotate_endpoint_points(fig, endpoint_points)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def render_curves(output, model, epoch, online, seeds, stats):
    common_note = FOOTNOTE
    partial = [row for row in epoch if row.get("partial_epoch")]
    if partial:
        positions = sorted({row["epoch"] for row in partial})
        common_note += "\nRecorded partial epoch(s) at HF state: " + ", ".join(f"{value:g}" for value in positions)
    if epoch:
        epoch_coverage = observed_coverage(epoch, seeds, "epoch")[0]
        common_note += "\n" + coverage_footer(epoch, seeds, "epoch")
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        for ax, (key, label) in zip(axes.flat, LOSS_METRICS):
            draw_curve(ax, epoch, stats, key, label, seeds, "epoch")
        axes.flat[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{model}: fixed-data post-update losses\n{epoch_coverage}", fontsize=12)
        save_figure(fig, output, "fixed_loss_curves", common_note)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
        for ax, (key, label) in zip(axes, EVAL_METRICS):
            draw_curve(ax, epoch, stats, key, label, seeds, "epoch")
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{model}: epoch evaluations\n{epoch_coverage}", fontsize=12)
        save_figure(fig, output, "metric_curves", common_note +
                    "\nFQ uses a logarithmic axis; SD bands crossing zero are omitted on that axis.")
        for seed in seeds:
            selected = [row for row in epoch if row["seed"] == seed]
            if not selected:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
            for ax, (key, label) in zip(axes.flat, LOSS_METRICS[:3] + EVAL_METRICS):
                draw_curve(ax, selected, [], key, label, seeds, "epoch", single=True)
            tail = " (recorded partial final epoch)" if selected[-1].get("partial_epoch") else ""
            fig.suptitle(f"{model}: seed {seed}, fixed losses and evaluation{tail}", fontsize=12)
            save_figure(fig, output, f"seed_{seed}_epoch_curves", "Same fixed data at each observed epoch. All points; no smoothing.")
    if online:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        for ax, (key, label) in zip(axes.flat, LOSS_METRICS):
            draw_curve(ax, online, stats, key, label, seeds, "step")
        axes.flat[0].legend(fontsize=7, ncol=2)
        online_coverage = observed_coverage(online, seeds, "step")[0]
        fig.suptitle(f"{model}: online training losses\n{online_coverage}", fontsize=12)
        save_figure(fig, output, "online_loss_curves", FOOTNOTE +
                    "\nPre-update train-mode microbatch means; these are a different loss measurement from fixed-data epoch losses.\n" +
                    coverage_footer(online, seeds, "step"))


def scatter_figure(output, model, rows, seeds, endpoints=False, log_loss=False):
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    epochs = sorted({round(row["epoch"], 10) for row in rows})
    pairs = (("raw_sensitive_loss", "forget_quality", "Raw sensitive-token loss", "FQ: KS p-value"),
             ("raw_retain_nll", "model_utility", "Raw retain NLL", "MU"))
    zero_count = 0
    endpoint_points = []
    omitted_loss_counts = {key: {"zero": 0, "negative": 0} for key, _, _, _ in pairs}
    for ax, (xkey, ykey, xlabel, ylabel) in zip(axes, pairs):
        plotted_count = 0
        labels = []
        for seed in seeds:
            selected = [row for row in rows if row["seed"] == seed]
            color = COLORS[seeds.index(seed) % len(COLORS)]
            final_step = max((row["step"] for row in selected), default=-1)
            for row in selected:
                x, y = row.get(xkey), row.get(ykey)
                if not finite(x) or not finite(y):
                    continue
                omitted_fq = ykey == "forget_quality" and y == 0
                omitted_loss = log_loss and x <= 0
                if omitted_fq:
                    zero_count += 1
                if omitted_loss:
                    omitted_loss_counts[xkey]["zero" if x == 0 else "negative"] += 1
                if omitted_fq or omitted_loss:
                    continue
                epoch_index = epochs.index(round(row["epoch"], 10))
                alpha = .35 + .65 * epoch_index / max(1, len(epochs) - 1)
                final = row["step"] == final_step
                ax.scatter([x], [y], c=[to_rgba(color, alpha)], s=52 if final else 28,
                    marker="*" if final else MARKERS[epoch_index % len(MARKERS)],
                    linewidths=.6, edgecolors="#222222" if final else "none", zorder=4 if final else 2)
                plotted_count += 1
                if endpoints:
                    labels.append((seed, x, y))
        style_axis(ax, ylabel, xlabel, ykey == "forget_quality")
        if log_loss:
            ax.set_xscale("log")
            format_log_ticks(ax, "x", endpoint=endpoints)
        if ykey == "forget_quality":
            format_log_ticks(ax, "y", endpoint=endpoints)
        if endpoints:
            endpoint_points.append((ax, labels))
        if not plotted_count:
            ax.text(.5, .5, "No representable pairs on these axes", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
    observed_seeds = sorted({row["seed"] for row in rows})
    legend = [Line2D([], [], color=COLORS[seeds.index(seed) % len(COLORS)], marker="o", linestyle="none", label=f"Seed {seed}")
              for seed in observed_seeds]
    legend.append(Line2D([], [], color="#777777", marker="*", linestyle="none", markersize=9, label="Last observed point"))
    axes[1].legend(handles=legend, fontsize=7, loc="best")
    if not endpoints:
        epoch_legend = [Line2D([], [], color="#777777", marker=MARKERS[i % len(MARKERS)],
            linestyle="none", label=f"Epoch {epoch:g}") for i, epoch in enumerate(epochs[:-1])]
        if epoch_legend:
            axes[0].legend(handles=epoch_legend, fontsize=6.5, ncol=2, loc="best", title="Earlier epochs")
    suffix = "endpoints" if endpoints else "all_epochs"
    scale_note = "; logarithmic loss axes" if log_loss else ""
    coverage = observed_coverage(rows, seeds, None if endpoints else "epoch")[0]
    fig.suptitle(f"{model}: loss versus metric ({'last observed state per seed' if endpoints else 'all epochs'}{scale_note})\n{coverage}", fontsize=12)
    note = ("Fixed-data post-update losses only. Color identifies seed; markers/opacity identify epoch; star marks last observed point.\n"
            "Epochs from one seed are repeated measurements; pooled trends cannot establish a seed effect.")
    if zero_count:
        note += f"\n{zero_count} zero FQ p-value(s) retained in CSV and omitted from the logarithmic panel."
    if log_loss:
        omitted = "; ".join(f"{key}: {counts['zero']} zero, {counts['negative']} negative"
                            for key, counts in omitted_loss_counts.items())
        note += f"\nNonpositive original losses retained in CSV and omitted from log-loss axes ({omitted})."
    note += "\n" + coverage_footer(rows, seeds, None if endpoints else "epoch")
    prefix = "loss_metric_scatter_log_loss" if log_loss else "loss_metric_scatter"
    save_figure(fig, output, f"{prefix}_{suffix}", note, endpoint_points=endpoint_points)


def relative_difference(a, b):
    denominator = (abs(a) + abs(b)) / 2
    return abs(a - b) / denominator if denominator else 0.0


def match_pairs(rows, absolute_tolerance, endpoint=False):
    """Descriptive within-model matches; no hypothesis tests on dependent epochs."""
    output = []
    for before, after in combinations(rows, 2):
        if before["seed"] == after["seed"]:
            continue
        keys = ("raw_sensitive_loss", "raw_retain_nll")
        if not all(finite(row.get(key)) for row in (before, after) for key in keys):
            continue
        # A loss match is meaningful only with a verified shared fixed dataset.
        hash_before, hash_after = before.get("fixed_dataset_sha256"), after.get("fixed_dataset_sha256")
        if not hash_before or hash_before != hash_after:
            continue
        protocol_before, protocol_after = before.get("protocol_sha256"), after.get("protocol_sha256")
        if protocol_before and protocol_after and protocol_before != protocol_after:
            continue
        relative = {key: relative_difference(before[key], after[key]) for key in keys}
        qualified = [tol for tol in (.01, .05, .10) if all(
            abs(before[key] - after[key]) <= absolute_tolerance + tol * (abs(before[key]) + abs(after[key])) / 2
            for key in keys)]
        if not qualified:
            continue
        fq_a, fq_b = before.get("forget_quality"), after.get("forget_quality")
        gap = abs(math.log10(fq_a) - math.log10(fq_b)) if finite(fq_a) and finite(fq_b) and min(fq_a, fq_b) > 0 else None
        item = {"model": before["model"], "comparison": "endpoints" if endpoint else "epoch_states",
            "seed_a": before["seed"], "seed_b": after["seed"], "epoch_a": before["epoch"], "epoch_b": after["epoch"],
            "step_a": before["step"], "step_b": after["step"],
            "same_logged_epoch": math.isclose(before["epoch"], after["epoch"], abs_tol=1e-9),
            "same_update_budget": before["step"] == after["step"],
            "minimum_relative_tolerance": min(qualified), "absolute_tolerance": absolute_tolerance,
            "forget_relative_difference": relative["raw_sensitive_loss"],
            "retain_relative_difference": relative["raw_retain_nll"], "FQ_log10_gap": gap,
            "FQ_gap_at_least_2_orders": gap is not None and gap >= 2,
            "FQ_zero_in_pair": fq_a == 0 or fq_b == 0,
            "KS_D_absolute_gap": abs(before["ks_d"] - after["ks_d"]) if all(finite(row.get("ks_d")) for row in (before, after)) else None,
            "fixed_dataset_sha256": hash_before}
        if all(finite(row.get("raw_common_kl")) for row in (before, after)):
            item["common_kl_relative_difference"] = relative_difference(before["raw_common_kl"], after["raw_common_kl"])
            item["all_three_raw_components_match"] = abs(before["raw_common_kl"] - after["raw_common_kl"]) <= (
                absolute_tolerance + min(qualified) * (abs(before["raw_common_kl"]) + abs(after["raw_common_kl"])) / 2)
        item["protocol_hash_verified"] = bool(protocol_before and protocol_before == protocol_after)
        for suffix, row in (("a", before), ("b", after)):
            for key, _ in LOSS_METRICS + EVAL_METRICS:
                item[f"{key}_{suffix}"] = row.get(key)
        output.append(item)
    return sorted(output, key=lambda row: (row["minimum_relative_tolerance"], -(row["FQ_log10_gap"] or 0)))


def render(root, output, mappings, absolute_tolerance):
    runs, planned = load_runs(root, mappings)
    summary = {"models": {}, "root": str(root), "output": str(output), "statistical_unit": "training seed"}
    output.mkdir(parents=True, exist_ok=True)
    for model in sorted(planned):
        directory = output / re.sub(r"[^A-Za-z0-9._-]+", "_", model)
        directory.mkdir(exist_ok=True)
        seeds = sorted(planned[model])
        epoch, online, endpoints = [], [], []
        for run in (run for run in runs if run["model"] == model):
            records = normalize_rows(run, "epoch_history.jsonl", "fixed_epoch")
            epoch.extend(records)
            online.extend(normalize_rows(run, "loss_history.jsonl", "online_training"))
            if records:
                endpoint = dict(records[-1])
                endpoint["run_status"] = run["manifest"].get("status", run["entry"].get("status", "unknown"))
                endpoint["endpoint_kind"] = endpoint.get("evaluation_kind", "last_observed")
                endpoints.append(endpoint)
        # Prevent accidentally mixing different fixed-data measurements across seeds.
        hashes = {row.get("fixed_dataset_sha256") for row in epoch if row.get("fixed_dataset_sha256")}
        if len(hashes) > 1:
            raise ValueError(f"Different fixed datasets within model {model}; create separate groups")
        protocols = {row.get("protocol_sha256") for row in epoch if row.get("protocol_sha256")}
        if len(protocols) > 1:
            raise ValueError(f"Different training protocols within model {model}; create separate groups")
        fixed_counts = {key: sorted({row[key] for row in epoch if row.get(key) is not None})
                        for key in ("forget_count", "retain_count")}
        if any(len(values) > 1 for values in fixed_counts.values()):
            raise ValueError(f"Fixed-data example counts change within model {model}")
        stats = curve_statistics(epoch, seeds, "epoch", LOSS_METRICS + EVAL_METRICS) + curve_statistics(
            online, seeds, "step", LOSS_METRICS)
        write_csv(directory / "epoch_points.csv", epoch)
        write_csv(directory / "online_step_points.csv", online)
        write_csv(directory / "curve_statistics.csv", stats)
        write_csv(directory / "endpoints.csv", endpoints)
        endpoint_stats = []
        for key, _ in LOSS_METRICS + EVAL_METRICS:
            values = [row[key] for row in endpoints if finite(row.get(key))]
            endpoint_stats.append({"model": model, "metric": key, "n_seeds": len(values), "planned_n_seeds": len(seeds),
                "mean": statistics.mean(values) if values else None,
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                "min": min(values) if values else None, "max": max(values) if values else None,
                "same_update_budget": len(endpoints) > 1 and len({row["step"] for row in endpoints}) == 1,
                "same_logged_epoch": len(endpoints) > 1 and len({round(row["epoch"], 10) for row in endpoints}) == 1})
        write_csv(directory / "endpoint_statistics.csv", endpoint_stats)
        pairs = match_pairs(epoch, absolute_tolerance)
        final_pairs = match_pairs(endpoints, absolute_tolerance, endpoint=True)
        write_csv(directory / "matched_epoch_pairs.csv", pairs)
        write_csv(directory / "matched_endpoint_pairs.csv", final_pairs)
        render_curves(directory, model, epoch, online, seeds, stats)
        scatter_figure(directory, model, epoch, seeds)
        scatter_figure(directory, model, endpoints, seeds, endpoints=True)
        scatter_figure(directory, model, epoch, seeds, log_loss=True)
        scatter_figure(directory, model, endpoints, seeds, endpoints=True, log_loss=True)
        summary["models"][model] = {"planned_seeds": seeds, "observed_epoch_seeds": sorted({r["seed"] for r in epoch}),
            "epoch_points": len(epoch), "online_points": len(online), "endpoints": len(endpoints),
            "zero_FQ_points": sum(row.get("forget_quality") == 0 for row in epoch),
            "fixed_dataset_sha256": next(iter(hashes), None), "matched_epoch_pairs": len(pairs),
            "protocol_sha256": next(iter(protocols), None),
            "fixed_sample_counts": fixed_counts,
            "fixed_hash_verified_all_points": bool(epoch) and all(row.get("fixed_dataset_sha256") for row in epoch),
            "protocol_hash_verified_all_points": bool(epoch) and all(row.get("protocol_sha256") for row in epoch),
            "matched_endpoint_pairs": len(final_pairs), "output_dir": str(directory)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# PALU seed and epoch plots\n\n"
        "Each model is analyzed separately. `epoch_points.csv` preserves every observed fixed-data evaluation; "
        "`online_step_points.csv` preserves the unsmoothed pre-update training losses. Plots use real logged epochs, "
        "including any recorded partial final epoch. Partial flags come from actual batch coverage, not from rounding "
        "HF state. The last available state is labeled as an observed endpoint and can be "
        "incomplete while training runs. `endpoints.csv` records its evaluation kind and manifest status.\n\n"
        "`curve_statistics.csv` reports arithmetic means and sample SD across the available training seeds at each "
        "epoch or step, with planned and observed seed coverage. Epoch observations are not independent training "
        "replicates. FQ is a KS p-value and uses a logarithmic axis. Zero p-values remain zero in CSV; they are omitted "
        "from log axes without substituting an epsilon. SD bands that cross zero are omitted from the FQ log panel.\n\n"
        "`loss_metric_scatter_all_epochs` shows the whole trajectory; `loss_metric_scatter_endpoints` compares last "
        "observed states only. A pooled time trend does not identify a seed effect. FQ scatter uses raw sensitive-token "
        "loss; MU scatter uses raw retain NLL. These fixed-data losses use the runner's token-weighted reduction; "
        "the sample means are also retained in CSV. Common-token KL remains separately recorded. The additional "
        "`loss_metric_scatter_log_loss_all_epochs` and `loss_metric_scatter_log_loss_endpoints` figures use logarithmic "
        "loss axes to show small terminal losses alongside the initial state. Nonpositive losses retain their original "
        "values in CSV and are omitted from these log-loss figures; no epsilon is substituted.\n\n"
        "`matched_epoch_pairs.csv` and `matched_endpoint_pairs.csv` are exploratory comparisons of different seeds "
        "with verified matching fixed-dataset hashes. Both raw sensitive loss and retain NLL must satisfy absolute "
        f"difference <= {absolute_tolerance:g} + tolerance * (|a| + |b|)/2, at 1%, 5%, or 10%. They also report common "
        "KL, real epoch, update count, KS D and the FQ log gap. Matching loss after training is a conditional descriptive "
        "analysis, not a causal effect of seed or proof of equal functions. Full-history matches and endpoint matches "
        "are separate to make temporal confounding visible. No checkpoints are selected by best FQ.\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, help="Default: ROOT/seed_epoch_plots")
    parser.add_argument("--path-map", action="append", default=[], metavar="REMOTE_ROOT=LOCAL_ROOT")
    parser.add_argument("--absolute-loss-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    if args.absolute_loss_tolerance < 0:
        parser.error("absolute loss tolerance must be nonnegative")
    mappings = []
    for mapping in args.path_map:
        before, separator, after = mapping.partition("=")
        if not separator or not before or not after:
            parser.error("path mapping must be REMOTE_ROOT=LOCAL_ROOT")
        mappings.append((Path(before), Path(after)))
    result = render(args.root, args.output or args.root / "seed_epoch_plots", mappings, args.absolute_loss_tolerance)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
