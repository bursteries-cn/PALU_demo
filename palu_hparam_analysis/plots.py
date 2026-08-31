from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

_mpl_cache = Path(tempfile.gettempdir()) / "palu-hparam-matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .schema import FACTORS, FACTOR_LABELS, METRICS, METRIC_LABELS, display_factor_value

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000"]


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
        }
    )


def save_figure(
    fig: plt.Figure,
    output_stem: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        extension = extension.lower().lstrip(".")
        path = output_stem.with_suffix(f".{extension}")
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_coverage(
    audit: dict[str, Any], output_stem: Path, config: dict[str, Any]
) -> list[str]:
    apply_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), constrained_layout=True)
    for panel, (ax, table) in enumerate(zip(axes.flat, audit["pair_tables"])):
        left, right = table["left"], table["right"]
        left_levels = table["left_levels"]
        right_levels = table["right_levels"]
        matrix = np.array(
            [
                [table["counts"].get(f"{left_value}|{right_value}", 0) for right_value in right_levels]
                for left_value in left_levels
            ],
            dtype=float,
        )
        image = ax.imshow(matrix, cmap="cividis", vmin=0, vmax=max(1.0, matrix.max()), aspect="auto")
        ax.set_xticks(range(len(right_levels)))
        ax.set_xticklabels([display_factor_value(right, value) for value in right_levels], rotation=35, ha="right")
        ax.set_yticks(range(len(left_levels)))
        ax.set_yticklabels([display_factor_value(left, value) for value in left_levels])
        ax.set_xlabel(FACTOR_LABELS[right])
        ax.set_ylabel(FACTOR_LABELS[left])
        ax.set_title(f"{chr(65 + panel)}  {FACTOR_LABELS[left]} × {FACTOR_LABELS[right]}", loc="left", fontweight="bold")
        threshold = max(matrix.max() * 0.55, 1)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = int(matrix[row, column])
                ax.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value >= threshold else "black",
                    fontsize=7,
                )
                if value == 0:
                    ax.add_patch(
                        Rectangle(
                            (column - 0.5, row - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="////",
                            edgecolor="#666666",
                            linewidth=0,
                        )
                    )
        fig.colorbar(image, ax=ax, shrink=0.65, label="Observed rows")
    fig.suptitle(
        f"Sparse-design coverage: {audit['observed_configurations']}/{audit['planned_combinations']} configurations "
        f"({audit['coverage_fraction']:.1%})",
        fontsize=11,
    )
    return save_figure(
        fig,
        output_stem,
        config["output"]["static_formats"],
        int(config["output"]["dpi"]),
    )


def plot_parameter_guidance(
    summary: list[dict[str, Any]], output_stem: Path, config: dict[str, Any]
) -> list[str]:
    """Compact evidence plot for sparse data; untested levels stay visibly untested."""
    if not any(entry.get("score_median") is not None for entry in summary):
        return []
    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)
    for panel, (ax, factor) in enumerate(zip(axes.flat, FACTORS)):
        entries = [entry for entry in summary if entry["factor"] == factor]
        x = np.arange(len(entries), dtype=float)
        tested_x = []
        medians = []
        lower = []
        upper = []
        for index, entry in enumerate(entries):
            score = entry.get("score_median")
            if score is None:
                ax.scatter(index, 0.035, marker="x", s=22, color="#8a98a6", linewidths=1.0)
                ax.text(index, 0.085, "not run", ha="center", va="bottom", fontsize=6, color="#6f7c88")
                continue
            tested_x.append(index)
            medians.append(float(score))
            lower.append(float(entry.get("score_q25", score)))
            upper.append(float(entry.get("score_q75", score)))
            if entry.get("promising"):
                ax.axvspan(index - 0.38, index + 0.38, color=OKABE_ITO[1], alpha=0.16, zorder=0)
            ax.text(
                index,
                min(0.98, float(entry.get("score_q75", score)) + 0.055),
                f"n={entry['n_configurations']}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
        if tested_x:
            means = np.array(medians, dtype=float)
            low = np.array(lower, dtype=float)
            high = np.array(upper, dtype=float)
            ax.errorbar(
                tested_x,
                means,
                yerr=np.vstack([means - low, high - means]),
                color=OKABE_ITO[0],
                marker="o",
                markersize=4.5,
                linewidth=1.2,
                capsize=3,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [display_factor_value(factor, entry["value"]) for entry in entries],
            rotation=28,
            ha="right",
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Observed desirability percentile")
        ax.set_xlabel(FACTOR_LABELS[factor])
        ax.set_title(
            f"{chr(65 + panel)}  {FACTOR_LABELS[factor]}",
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="y", color="#e4e8ec", linewidth=0.6)
    fig.suptitle(
        "Observed parameter guidance (median and IQR; orange bands mark current promising levels)\n"
        "Descriptive under sparse, confounded coverage — not a causal effect estimate",
        fontsize=10,
    )
    return save_figure(
        fig,
        output_stem,
        config["output"]["static_formats"],
        int(config["output"]["dpi"]),
    )


def plot_main_effects(
    result: dict[str, Any], output_stem: Path, config: dict[str, Any]
) -> list[str]:
    apply_publication_style()
    metric = result["metric"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    rng = np.random.default_rng(int(config["modeling"]["random_seed"]))
    for panel, (ax, factor) in enumerate(zip(axes.flat, FACTORS)):
        effects = result["main_effects"][factor]
        x = np.arange(len(effects), dtype=float)
        means = np.array([entry["adjusted_mean"] for entry in effects], dtype=float)
        lower = np.array(
            [entry["lower"] if entry["lower"] is not None else entry["adjusted_mean"] for entry in effects]
        )
        upper = np.array(
            [entry["upper"] if entry["upper"] is not None else entry["adjusted_mean"] for entry in effects]
        )
        ax.errorbar(
            x,
            means,
            yerr=np.vstack(
                [np.maximum(means - lower, 0.0), np.maximum(upper - means, 0.0)]
            ),
            color=OKABE_ITO[0],
            marker="o",
            linewidth=1.5,
            capsize=3,
            label="Adjusted mean (95% bootstrap interval)",
            zorder=3,
        )
        for index, entry in enumerate(effects):
            observed = np.array(entry["observed_values"], dtype=float)
            if len(observed):
                jitter = rng.uniform(-0.08, 0.08, size=len(observed))
                ax.scatter(
                    np.full(len(observed), index) + jitter,
                    observed,
                    s=13,
                    facecolors="none",
                    edgecolors=OKABE_ITO[1],
                    linewidths=0.8,
                    alpha=0.75,
                    label="Observed" if index == 0 else None,
                    zorder=2,
                )
            ax.text(index, upper[index], f"n={entry['n_observed']}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([display_factor_value(factor, entry["level"]) for entry in effects], rotation=30, ha="right")
        if factor in {"lr", "alpha"}:
            ax.set_xlabel(f"{FACTOR_LABELS[factor]} (configured levels; log-spaced display order)")
        else:
            ax.set_xlabel(FACTOR_LABELS[factor])
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_title(f"{chr(65 + panel)}  {FACTOR_LABELS[factor]}", loc="left", fontweight="bold")
        if metric == "forget_quality" and np.all(means > 0):
            ax.set_yscale("log")
        if panel == 0:
            ax.legend(frameon=False, loc="best")
    status = result["diagnostics"].get("status", "unknown")
    fig.suptitle(f"Adjusted main effects — {METRIC_LABELS[metric]} [{status}]", fontsize=11)
    return save_figure(
        fig,
        output_stem,
        config["output"]["static_formats"],
        int(config["output"]["dpi"]),
    )


def plot_interactions(
    result: dict[str, Any], output_stem: Path, config: dict[str, Any]
) -> list[str]:
    apply_publication_style()
    metric = result["metric"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), constrained_layout=True)
    all_values = [cell["adjusted_mean"] for pair in result["pair_effects"] for cell in pair["cells"]]
    vmin, vmax = min(all_values), max(all_values)
    for panel, (ax, pair) in enumerate(zip(axes.flat, result["pair_effects"])):
        left, right = pair["left"], pair["right"]
        left_levels = config["search_space"][left]
        right_levels = config["search_space"][right]
        lookup = {
            (str(cell[left]), str(cell[right])): cell for cell in pair["cells"]
        }
        matrix = np.zeros((len(left_levels), len(right_levels)), dtype=float)
        for row_index, left_level in enumerate(left_levels):
            for column_index, right_level in enumerate(right_levels):
                matrix[row_index, column_index] = lookup[(str(left_level), str(right_level))]["adjusted_mean"]
        image = ax.imshow(matrix, cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(right_levels)))
        ax.set_xticklabels([display_factor_value(right, value) for value in right_levels], rotation=35, ha="right")
        ax.set_yticks(range(len(left_levels)))
        ax.set_yticklabels([display_factor_value(left, value) for value in left_levels])
        ax.set_xlabel(FACTOR_LABELS[right])
        ax.set_ylabel(FACTOR_LABELS[left])
        ax.set_title(
            f"{chr(65 + panel)}  {FACTOR_LABELS[left]} × {FACTOR_LABELS[right]}\n"
            f"interaction RMS={pair['interaction_rms']:.3g}",
            loc="left",
            fontweight="bold",
        )
        for row_index, left_level in enumerate(left_levels):
            for column_index, right_level in enumerate(right_levels):
                cell = lookup[(str(left_level), str(right_level))]
                ax.text(
                    column_index,
                    row_index,
                    f"{cell['adjusted_mean']:.3g}\nn={cell['n_observed']}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if cell["adjusted_mean"] > (vmin + vmax) / 2 else "black",
                )
                if cell["n_observed"] == 0:
                    ax.add_patch(
                        Rectangle(
                            (column_index - 0.5, row_index - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="////",
                            edgecolor="#777777",
                            linewidth=0,
                        )
                    )
        fig.colorbar(image, ax=ax, shrink=0.65, label="Adjusted prediction")
    fig.suptitle(
        f"Pairwise adjusted surfaces — {METRIC_LABELS[metric]}\n"
        "Hatched cells have no direct observation and are model estimates",
        fontsize=11,
    )
    return save_figure(
        fig,
        output_stem,
        config["output"]["static_formats"],
        int(config["output"]["dpi"]),
    )


def plot_tradeoffs(
    rows: list[dict[str, Any]], em_target: Optional[float], output_stem: Path, config: dict[str, Any]
) -> list[str]:
    apply_publication_style()
    complete = [row for row in rows if all(row.get(metric) is not None for metric in METRICS)]
    fig, axes = plt.subplots(4, 4, figsize=(9, 8.8), constrained_layout=True)
    for row_index, y_metric in enumerate(METRICS):
        for column_index, x_metric in enumerate(METRICS):
            ax = axes[row_index, column_index]
            if not complete:
                ax.axis("off")
                continue
            x = np.array([float(row[x_metric]) for row in complete])
            y = np.array([float(row[y_metric]) for row in complete])
            pareto = np.array([bool(row.get("pareto_observed")) for row in complete])
            if row_index == column_index:
                ax.hist(x, bins=min(10, max(3, int(math.sqrt(len(x))))), color=OKABE_ITO[0], alpha=0.8)
            elif row_index > column_index:
                ax.scatter(x[~pareto], y[~pareto], s=16, color=OKABE_ITO[0], alpha=0.55, label="Observed")
                ax.scatter(
                    x[pareto],
                    y[pareto],
                    s=38,
                    facecolors="none",
                    edgecolors=OKABE_ITO[1],
                    linewidths=1.4,
                    label="Pareto",
                )
                if y_metric == "exact_memorization" and em_target is not None:
                    ax.axhline(em_target, color="#555555", linestyle="--", linewidth=0.8)
                if x_metric == "exact_memorization" and em_target is not None:
                    ax.axvline(em_target, color="#555555", linestyle="--", linewidth=0.8)
            else:
                ax.axis("off")
            if row_index == 3:
                ax.set_xlabel(METRIC_LABELS[x_metric], rotation=15)
            if column_index == 0:
                ax.set_ylabel(METRIC_LABELS[y_metric])
            if x_metric == "forget_quality" and row_index >= column_index and np.all(x > 0):
                ax.set_xscale("log")
            if y_metric == "forget_quality" and row_index > column_index and np.all(y > 0):
                ax.set_yscale("log")
    fig.suptitle("Observed four-metric trade-offs and Pareto configurations", fontsize=11)
    return save_figure(
        fig,
        output_stem,
        config["output"]["static_formats"],
        int(config["output"]["dpi"]),
    )


def plot_parallel_coordinates(
    rows: list[dict[str, Any]], em_target: Optional[float], output_stem: Path, config: dict[str, Any]
) -> list[str]:
    """Scale-free direction-aligned view; raw values remain in tables and trade-off plots."""
    apply_publication_style()
    complete = [row for row in rows if all(row.get(metric) is not None for metric in METRICS)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    if not complete:
        ax.text(0.5, 0.5, "No complete four-metric configurations", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        objective_columns = []
        for metric in METRICS:
            if metric == "exact_memorization":
                target = em_target if em_target is not None else 0.0
                values = np.array([-abs(float(row[metric]) - target) for row in complete])
            else:
                values = np.array([float(row[metric]) for row in complete])
            lo, hi = float(values.min()), float(values.max())
            objective_columns.append(
                np.ones(len(values)) if math.isclose(lo, hi) else (values - lo) / (hi - lo)
            )
        matrix = np.vstack(objective_columns).T
        x = np.arange(len(METRICS))
        for row, values in zip(complete, matrix):
            is_pareto = bool(row.get("pareto_observed"))
            ax.plot(
                x,
                values,
                color=OKABE_ITO[1] if is_pareto else OKABE_ITO[0],
                linewidth=1.8 if is_pareto else 0.7,
                alpha=0.9 if is_pareto else 0.18,
                marker="o" if is_pareto else None,
                markersize=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS[metric] for metric in METRICS], rotation=18, ha="right")
        ax.set_ylabel("Within-observed desirability (0–1; larger is better)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_title(
            "Observed parallel coordinates (orange = Pareto; scale-free display only)",
            loc="left",
            fontweight="bold",
        )
    return save_figure(
        fig,
        output_stem,
        config["output"]["static_formats"],
        int(config["output"]["dpi"]),
    )


def plot_model_diagnostics(
    models: dict[str, dict[str, Any]], output_stem: Path, config: dict[str, Any]
) -> list[str]:
    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    for panel, (ax, metric) in enumerate(zip(axes.flat, METRICS)):
        residuals = models.get(metric, {}).get("residuals", [])
        observed = np.array([entry["observed"] for entry in residuals if entry.get("cv_prediction") is not None])
        predicted = np.array([entry["cv_prediction"] for entry in residuals if entry.get("cv_prediction") is not None])
        if len(observed):
            ax.scatter(observed, predicted, s=18, color=OKABE_ITO[0], alpha=0.65)
            limits = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
            ax.plot(limits, limits, color="#555555", linestyle="--", linewidth=0.8)
        else:
            ax.text(0.5, 0.5, "Cross-validation unavailable", ha="center", va="center", transform=ax.transAxes)
        diagnostics = models.get(metric, {}).get("diagnostics", {})
        selected = diagnostics.get("cv", {}).get("selected", {})
        label = f"status={diagnostics.get('status', 'not fitted')}"
        if selected:
            label += f"\nCV RMSE={float(selected.get('rmse', float('nan'))):.3g}"
            if selected.get("r2") is not None:
                label += f", R²={float(selected['r2']):.3g}"
        ax.text(0.03, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontsize=7)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Grouped-CV prediction")
        ax.set_title(f"{chr(65 + panel)}  {METRIC_LABELS[metric]}", loc="left", fontweight="bold")
        ax.grid(color="#e5e5e5", linewidth=0.5)
    fig.suptitle("Surrogate-model diagnostics", fontsize=11)
    return save_figure(
        fig,
        output_stem,
        config["output"]["static_formats"],
        int(config["output"]["dpi"]),
    )


def plot_epoch_trajectories(
    rows: list[dict[str, Any]], output_stem: Path, config: dict[str, Any]
) -> list[str]:
    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    runs = sorted({str(row.get("run_id")) for row in rows})
    for panel, (ax, metric) in enumerate(zip(axes.flat, METRICS)):
        for run_index, run_id in enumerate(runs):
            run_rows = sorted(
                [row for row in rows if str(row.get("run_id")) == run_id and row.get(metric) is not None],
                key=lambda row: float(row.get("epoch", 0)),
            )
            if not run_rows:
                continue
            ax.plot(
                [float(row["epoch"]) for row in run_rows],
                [float(row[metric]) for row in run_rows],
                color=OKABE_ITO[run_index % len(OKABE_ITO)],
                alpha=0.25 if len(runs) > 12 else 0.55,
                linewidth=0.8,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{chr(65 + panel)}  {METRIC_LABELS[metric]}", loc="left", fontweight="bold")
        if metric == "forget_quality":
            positive = [float(row[metric]) for row in rows if row.get(metric) is not None and float(row[metric]) > 0]
            if positive:
                ax.set_yscale("log")
        ax.grid(color="#e5e5e5", linewidth=0.5)
    fig.suptitle(f"Checkpoint trajectories ({len(runs)} runs; descriptive only)", fontsize=11)
    return save_figure(
        fig,
        output_stem,
        config["output"]["static_formats"],
        int(config["output"]["dpi"]),
    )
