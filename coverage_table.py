"""把 PALU 超参实验的分布密度画成表格式热图，用于一眼看出哪些参数区间和组合做得稀疏。

只做覆盖度可视化，不做推荐、排名或 HTML 报告。数据管线直接复用
palu_hparam_analysis 的 ingest/schema/coverage，本脚本只负责聚合与绘图。

用法示例：
    python3 coverage_table.py
    python3 coverage_table.py --model Llama-3.1-8B-Instruct
    python3 coverage_table.py --rows lr,top_k --cols alpha,first_n
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Optional, Sequence

from palu_hparam_analysis.coverage import coverage_audit, normalized_search_space
from palu_hparam_analysis.ingest import (
    collect_records,
    filter_records,
    parse_run_name,
    select_checkpoints,
    select_protocol_cohort,
)
from palu_hparam_analysis.schema import (
    FACTOR_LABELS,
    FACTORS,
    METRIC_LABELS,
    canonical_factor_value,
    display_factor_value,
    load_analysis_config,
)

# plots 在导入时设置 Agg 后端与 MPLCONFIGDIR，必须先于 pyplot 导入。
from palu_hparam_analysis.plots import apply_publication_style, save_figure

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.patches import Polygon, Rectangle

# 单个组合通常只跑 1-2 次，成对投影后可以到几十次，两者需要不同粗细的分箱。
COUNT_BOUNDS_FINE = [1, 2, 3, 4, 5]
COUNT_BOUNDS_COARSE = [1, 2, 3, 5, 10, 20]
EMPTY_COLOR = "#f2f2f2"
NO_METRIC_COLOR = "#e2e2e2"
GRID_COLOR = "#ffffff"
MAJOR_LINE_COLOR = "#333333"

METRIC_MODES = ("forget_quality", "model_utility")


# --------------------------------------------------------------------------
# 数据层
# --------------------------------------------------------------------------


def model_slug(name: str) -> str:
    lowered = name.lower()
    if "llama-3.1" in lowered:
        return "llama31"
    if "llama-2" in lowered:
        return "llama2"
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")


def _sort_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def factor_tuple(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(canonical_factor_value(factor, row.get(factor)) for factor in FACTORS)


def scan_run_dirs(root: Path) -> list[dict[str, Any]]:
    """列出所有 run 目录（含没有任何评估结果的），仅靠 .hydra 与目录名识别。

    collect_records 从 TOFU_SUMMARY.json 出发，跑了但没评估的 run 不会出现在其中，
    这里补上它们以便在图上标记为"启动过但无结果"。
    """
    runs: list[dict[str, Any]] = []
    if not root.is_dir():
        return runs
    for hydra_dir in sorted(root.rglob(".hydra")):
        run_dir = hydra_dir.parent
        parsed = parse_run_name(run_dir.name)
        if not parsed:
            continue
        try:
            rel = run_dir.relative_to(root).parts
        except ValueError:
            continue
        if len(rel) < 5:
            continue
        row: dict[str, Any] = dict(parsed)
        row["split"] = rel[1]
        row["model"] = rel[2]
        row["trainer"] = rel[-2]
        row["run"] = run_dir.name
        row["run_id"] = str(run_dir.resolve())
        row["has_eval"] = any(run_dir.glob("checkpoint-*/evals/TOFU_SUMMARY.json"))
        runs.append(row)
    return runs


def levels_for(factor: str, config: dict[str, Any], observed: Sequence[Any]) -> list[Any]:
    """轴上的档位 = search_space 声明值 ∪ 磁盘观测值，按数值排序。

    计划内但一次没跑的档位因此会作为整行/整列空白显现出来。
    """
    values = list(normalized_search_space(config)[factor])
    for value in observed:
        if value is not None and value not in values:
            values.append(value)
    return sorted(values, key=_sort_key)


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    """按四因子组合聚合成 run 数与指标中位数。"""
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[factor_tuple(row)].append(row)

    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, group in buckets.items():
        entry: dict[str, Any] = {"n_runs": len({row["run_id"] for row in group})}
        for metric in METRIC_MODES:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            entry[metric] = median(values) if values else None
        out[key] = entry
    return out


def prepare_model_data(
    records: Sequence[dict[str, Any]],
    all_runs: Sequence[dict[str, Any]],
    config: dict[str, Any],
    include_short_runs: bool,
) -> dict[str, Any]:
    """把某个模型的原始 checkpoint 行整理成绘图所需的一切。"""
    full_length = int(config["checkpoint"].get("epoch", 10))
    rows = filter_records(records, config)
    short_run_ids = set()
    if not include_short_runs:
        kept = []
        for row in rows:
            if int(row.get("num_train_epochs", 0)) < full_length:
                short_run_ids.add(row["run_id"])
                continue
            kept.append(row)
        rows = kept

    selected = select_checkpoints(rows, config)

    # 台账要反映"到底跑过什么"，因此不做 protocol cohort 过滤，只取回 cohort 表用作脚注。
    mixed_config = {**config, "protocol": {"allow_mixed": True}}
    selected, _, cohort_table = select_protocol_cohort(selected, mixed_config)

    run_pool = filter_records(all_runs, config)
    if not include_short_runs:
        run_pool = [
            row for row in run_pool if int(row.get("num_train_epochs", 0)) >= full_length
        ]
    started_only = {
        factor_tuple(row) for row in run_pool if not row.get("has_eval")
    }

    agg = aggregate(selected)
    levels = {
        factor: levels_for(factor, config, [row.get(factor) for row in selected] + [row.get(factor) for row in run_pool])
        for factor in FACTORS
    }
    return {
        "selected": selected,
        "agg": agg,
        "levels": levels,
        "started_only": started_only,
        "cohort_table": cohort_table,
        "n_run_dirs": len(run_pool),
        "n_short_runs": len(short_run_ids),
        "audit": coverage_audit(selected, config),
    }


def build_axis_keys(factors: Sequence[str], levels: dict[str, list[Any]]) -> list[tuple[Any, ...]]:
    return list(itertools.product(*(levels[factor] for factor in factors)))


def full_key(
    row_factors: Sequence[str],
    row_key: Sequence[Any],
    col_factors: Sequence[str],
    col_key: Sequence[Any],
) -> tuple[Any, ...]:
    mapping = dict(zip(row_factors, row_key))
    mapping.update(zip(col_factors, col_key))
    return tuple(mapping[factor] for factor in FACTORS)


# --------------------------------------------------------------------------
# 绘图辅助
# --------------------------------------------------------------------------


def _fmt_metric(metric: str, value: float) -> str:
    if metric == "forget_quality":
        return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")
    return f"{value:.2f}"


def _count_scale(
    max_count: int, bounds: Optional[Sequence[int]] = None
) -> tuple[ListedColormap, BoundaryNorm, list[float], list[str]]:
    """离散计数配色。分箱按数据上限裁剪，避免大片格子挤在同一个最深色里。

    上界必须有限，否则 colorbar 无法设置轴范围。
    """
    max_count = max(int(max_count), 1)
    candidates = list(bounds) if bounds else COUNT_BOUNDS_FINE
    edges = [value for value in candidates if value <= max_count] or [1]
    edges = edges + [max(max_count + 1, edges[-1] + 1)]
    n_bins = len(edges) - 1

    cmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.28, 0.96, n_bins)))
    cmap.set_bad(EMPTY_COLOR)
    norm = BoundaryNorm(edges, cmap.N)
    ticks = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
    labels = []
    for index in range(n_bins):
        low, high = int(edges[index]), int(edges[index + 1]) - 1
        labels.append(str(low) if low == high else f"{low}-{high}")
    return cmap, norm, ticks, labels


def _metric_norm(metric: str, values: Sequence[float]) -> Optional[Any]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None
    if metric == "forget_quality":
        positive = [v for v in finite if v > 0]
        if not positive:
            return None
        low, high = min(positive), max(positive)
        if math.isclose(low, high):
            high = low * 10
        return LogNorm(vmin=low, vmax=high)
    low, high = min(finite), max(finite)
    if math.isclose(low, high):
        high = low + 1e-9
    return Normalize(vmin=low, vmax=high)


def _group_boundaries(outer: int, inner: int) -> list[float]:
    return [index * inner - 0.5 for index in range(1, outer)]


def _draw_matrix_axes(
    ax,
    n_rows: int,
    n_cols: int,
    row_factors: Sequence[str],
    col_factors: Sequence[str],
    levels: dict[str, list[Any]],
) -> None:
    """次分组取值作为刻度标签，主分组名称与取值标在坐标轴外侧。"""
    row_inner = len(levels[row_factors[1]])
    col_inner = len(levels[col_factors[1]])

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [display_factor_value(col_factors[1], value) for value in levels[col_factors[1]]]
        * len(levels[col_factors[0]]),
        fontsize=6,
        rotation=90,
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [display_factor_value(row_factors[1], value) for value in levels[row_factors[1]]]
        * len(levels[row_factors[0]]),
        fontsize=6,
    )
    ax.tick_params(length=2, pad=1.5)

    for boundary in _group_boundaries(len(levels[row_factors[0]]), row_inner):
        ax.axhline(boundary, color=MAJOR_LINE_COLOR, linewidth=1.1)
    for boundary in _group_boundaries(len(levels[col_factors[0]]), col_inner):
        ax.axvline(boundary, color=MAJOR_LINE_COLOR, linewidth=1.1)

    # 主分组只标取值，因子名各自出现一次，放在最外层。
    for index, value in enumerate(levels[row_factors[0]]):
        center = index * row_inner + (row_inner - 1) / 2
        ax.text(
            -1.7,
            center,
            display_factor_value(row_factors[0], value),
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            rotation=90,
            clip_on=False,
        )
    ax.text(
        -2.6,
        (n_rows - 1) / 2,
        f"{FACTOR_LABELS[row_factors[0]]}  (outer)   /   {FACTOR_LABELS[row_factors[1]]}  (inner)",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        rotation=90,
        clip_on=False,
    )
    for index, value in enumerate(levels[col_factors[0]]):
        center = index * col_inner + (col_inner - 1) / 2
        ax.text(
            center,
            n_rows + 0.4,
            display_factor_value(col_factors[0], value),
            ha="center",
            va="top",
            fontsize=7.5,
            fontweight="bold",
            clip_on=False,
        )
    ax.text(
        (n_cols - 1) / 2,
        n_rows + 1.5,
        f"{FACTOR_LABELS[col_factors[0]]}  (outer)   /   {FACTOR_LABELS[col_factors[1]]}  (inner)",
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        clip_on=False,
    )

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)


def _cell_grid_lines(ax, n_rows: int, n_cols: int) -> None:
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=0.5)
    ax.tick_params(which="minor", length=0)


def _marginal_panel(ax, levels: dict[str, list[Any]], marginals: dict[str, dict[Any, int]]) -> None:
    """左上角的单因子边际计数速查表。"""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n_factors = len(FACTORS)
    max_rows = max(len(levels[factor]) for factor in FACTORS)
    for column, factor in enumerate(FACTORS):
        # 收窄到 0.84 宽，给右邻的顶部条形图 y 轴刻度让位。
        x = 0.02 + (column + 0.5) / n_factors * 0.84
        ax.text(
            x,
            0.97,
            FACTOR_LABELS[factor],
            ha="center",
            va="top",
            fontsize=6.5,
            fontweight="bold",
        )
        for index, value in enumerate(levels[factor]):
            count = marginals[factor].get(value, 0)
            ax.text(
                x,
                0.97 - (index + 1) * (0.82 / (max_rows + 0.6)) - 0.06,
                f"{display_factor_value(factor, value)}   {count}",
                ha="center",
                va="top",
                fontsize=6,
                color="#222222" if count else "#a0a0a0",
            )
    ax.text(
        0.5,
        0.015,
        "single-factor marginal run counts",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#666666",
        style="italic",
    )


def _started_only_marker(ax, column: int, row: int) -> None:
    """格子右上角的小三角：该组合启动过但没有任何评估结果。"""
    ax.add_patch(
        Polygon(
            [
                (column + 0.16, row - 0.5),
                (column + 0.5, row - 0.5),
                (column + 0.5, row - 0.16),
            ],
            closed=True,
            facecolor="#d55e00",
            edgecolor="none",
            zorder=4,
        )
    )


# --------------------------------------------------------------------------
# 图 1 / 图 3 / 图 4：四维展平大矩阵
# --------------------------------------------------------------------------


def render_flat(
    data: dict[str, Any],
    *,
    mode: str,
    row_factors: Sequence[str],
    col_factors: Sequence[str],
    model: str,
    config: dict[str, Any],
    output_stem: Path,
    formats: Sequence[str],
) -> list[str]:
    levels = data["levels"]
    agg = data["agg"]
    row_keys = build_axis_keys(row_factors, levels)
    col_keys = build_axis_keys(col_factors, levels)
    n_rows, n_cols = len(row_keys), len(col_keys)

    counts = np.zeros((n_rows, n_cols), dtype=float)
    values = np.full((n_rows, n_cols), np.nan, dtype=float)
    started = np.zeros((n_rows, n_cols), dtype=bool)
    for r, row_key in enumerate(row_keys):
        for c, col_key in enumerate(col_keys):
            key = full_key(row_factors, row_key, col_factors, col_key)
            entry = agg.get(key)
            if entry is not None:
                counts[r, c] = entry["n_runs"]
                if mode != "count" and entry.get(mode) is not None:
                    values[r, c] = entry[mode]
            started[r, c] = key in data["started_only"]

    apply_publication_style()
    cell = 0.30
    corner_w, right_w, top_h = 2.9, 1.5, 1.7
    matrix_w, matrix_h = n_cols * cell, n_rows * cell
    fig_w = corner_w + matrix_w + right_w + 1.1
    fig_h = top_h + matrix_h + 1.85
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[corner_w, matrix_w, right_w],
        height_ratios=[top_h, matrix_h],
        left=0.02,
        right=0.985,
        top=1 - 0.55 / fig_h,
        bottom=1.15 / fig_h,
        wspace=0.04,
        hspace=0.04,
    )
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_cbar_host = fig.add_subplot(gs[0, 2])
    ax = fig.add_subplot(gs[1, 1])
    ax_right = fig.add_subplot(gs[1, 2])

    # aspect="auto" 让矩阵填满 gridspec 分配的框，边际条才能逐行逐列对齐；
    # 格子的近似方形由 width_ratios / height_ratios 保证。
    count_ticks: list[float] = []
    count_labels: list[str] = []
    if mode == "count":
        cmap, norm, count_ticks, count_labels = _count_scale(int(counts.max()))
        painted = np.where(counts > 0, counts, np.nan)
        image = ax.imshow(painted, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    else:
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(EMPTY_COLOR)
        norm = _metric_norm(mode, values[np.isfinite(values)].tolist())
        image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    for r in range(n_rows):
        for c in range(n_cols):
            n_runs = int(counts[r, c])
            if n_runs == 0:
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="#cccccc",
                        linewidth=0,
                        zorder=2,
                    )
                )
            elif mode != "count" and not math.isfinite(values[r, c]):
                # 有 run 但该指标缺失，与"完全没跑"用不同底纹区分。
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        facecolor=NO_METRIC_COLOR,
                        hatch="...",
                        edgecolor="#b0b0b0",
                        linewidth=0,
                        zorder=2,
                    )
                )
            if started[r, c]:
                _started_only_marker(ax, c, r)

            if mode == "count":
                if n_runs:
                    ax.text(
                        c,
                        r,
                        str(n_runs),
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="#ffffff" if norm(n_runs) >= cmap.N / 2 else "#10334d",
                        zorder=3,
                    )
            elif math.isfinite(values[r, c]) and norm is not None:
                shade = norm(values[r, c])
                ax.text(
                    c,
                    r,
                    _fmt_metric(mode, values[r, c]),
                    ha="center",
                    va="center",
                    fontsize=4.4,
                    color="#ffffff" if shade < 0.55 else "#20140a",
                    zorder=3,
                )

    _cell_grid_lines(ax, n_rows, n_cols)
    _draw_matrix_axes(ax, n_rows, n_cols, row_factors, col_factors, levels)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color(MAJOR_LINE_COLOR)

    col_totals = counts.sum(axis=0)
    row_totals = counts.sum(axis=1)
    ax_top.bar(range(n_cols), col_totals, color="#20639b", width=0.82)
    ax_top.set_xlim(-0.5, n_cols - 0.5)
    ax_top.set_xticks([])
    ax_top.set_ylabel("runs", fontsize=6.5)
    ax_top.tick_params(labelsize=6, length=2)
    ax_top.margins(y=0.18)
    for index, total in enumerate(col_totals):
        if total:
            ax_top.text(index, total, str(int(total)), ha="center", va="bottom", fontsize=5)

    ax_right.barh(range(n_rows), row_totals, color="#20639b", height=0.82)
    ax_right.set_ylim(n_rows - 0.5, -0.5)
    ax_right.set_yticks([])
    ax_right.set_xlabel("runs", fontsize=6.5)
    ax_right.tick_params(labelsize=6, length=2)
    ax_right.margins(x=0.22)
    for index, total in enumerate(row_totals):
        if total:
            ax_right.text(total, index, f" {int(total)}", ha="left", va="center", fontsize=5)

    marginals = {
        factor: defaultdict(int) for factor in FACTORS
    }
    for key, entry in agg.items():
        for factor, value in zip(FACTORS, key):
            marginals[factor][value] += entry["n_runs"]
    _marginal_panel(ax_corner, levels, marginals)

    ax_cbar_host.axis("off")
    if mode == "count":
        colorbar = fig.colorbar(
            image,
            ax=ax_cbar_host,
            fraction=0.34,
            aspect=9,
            spacing="uniform",
            ticks=count_ticks,
        )
        colorbar.ax.set_yticklabels(count_labels, fontsize=6)
        colorbar.set_label("runs per combination", fontsize=6.5)
    elif norm is not None:
        colorbar = fig.colorbar(image, ax=ax_cbar_host, fraction=0.34, aspect=9)
        colorbar.ax.tick_params(labelsize=6)
        colorbar.set_label(METRIC_LABELS[mode], fontsize=6.5)

    audit = data["audit"]
    if mode == "count":
        headline = (
            f"Hyperparameter coverage: {audit['observed_configurations']}/{audit['planned_combinations']} "
            f"combinations ({audit['coverage_fraction']:.1%}), {len(data['selected'])} runs"
        )
    else:
        headline = (
            f"{METRIC_LABELS[mode]} across the {audit['planned_combinations']}-cell grid "
            f"({audit['observed_configurations']} combinations observed)"
        )
    fig.suptitle(f"{model}  —  {headline}", fontsize=10, y=1 - 0.12 / fig_h)

    fig.text(
        0.5,
        1 - 0.36 / fig_h,
        f"rows: {FACTOR_LABELS[row_factors[0]]} x {FACTOR_LABELS[row_factors[1]]}   |   "
        f"columns: {FACTOR_LABELS[col_factors[0]]} x {FACTOR_LABELS[col_factors[1]]}   |   "
        f"hatched = never run, dotted = run without this metric, orange corner = started without evaluation",
        ha="center",
        va="top",
        fontsize=6.5,
        color="#555555",
    )
    fig.text(0.02, 0.012, _footnote(data, config), ha="left", va="bottom", fontsize=5.8, color="#666666")

    return save_figure(fig, output_stem, formats, int(config["output"]["dpi"]))


def _footnote(data: dict[str, Any], config: dict[str, Any]) -> str:
    filters = config.get("filters", {})
    checkpoint = config.get("checkpoint", {})
    policy = checkpoint.get("policy")
    policy_text = (
        "last evaluated checkpoint of each run"
        if policy == "last"
        else f"checkpoint at epoch {checkpoint.get('epoch')}"
    )
    parts = [
        f"split={filters.get('split')}, trainer contains '{filters.get('trainer_contains')}', "
        f"counting the {policy_text}",
        f"{data['n_run_dirs']} run directories in scope; {len(data['selected'])} have a usable result; "
        f"{len(data['started_only'])} combinations started without any evaluation"
        + (f"; {data['n_short_runs']} short runs excluded" if data["n_short_runs"] else ""),
    ]
    cohorts = data["cohort_table"]
    if len(cohorts) > 1:
        sizes = ", ".join(str(entry["n_rows"]) for entry in cohorts[:6])
        parts.append(
            f"protocol cohorts not merged away: {len(cohorts)} distinct evaluation protocols with sizes {sizes}"
        )
    return "\n".join(parts)


# --------------------------------------------------------------------------
# 图 2：成对覆盖下三角
# --------------------------------------------------------------------------


def render_pairwise(
    data: dict[str, Any],
    *,
    model: str,
    config: dict[str, Any],
    output_stem: Path,
    formats: Sequence[str],
) -> list[str]:
    audit = data["audit"]
    levels = data["levels"]
    pair_lookup = {(table["left"], table["right"]): table for table in audit["pair_tables"]}

    marginals: dict[str, dict[Any, int]] = {factor: defaultdict(int) for factor in FACTORS}
    for key, entry in data["agg"].items():
        for factor, value in zip(FACTORS, key):
            marginals[factor][value] += entry["n_runs"]

    apply_publication_style()
    n = len(FACTORS)
    fig, axes = plt.subplots(n, n, figsize=(9.4, 9.3), constrained_layout=True)
    # 给底部脚注留出位置，否则会压住最下一行的轴标签。
    fig.get_layout_engine().set(rect=(0.0, 0.045, 1.0, 0.955))

    vmax = max(
        (
            max(table["counts"].values(), default=0)
            for table in audit["pair_tables"]
        ),
        default=0,
    )
    cmap, norm, count_ticks, count_labels = _count_scale(int(vmax), COUNT_BOUNDS_COARSE)
    image = None

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                factor = FACTORS[i]
                factor_levels = levels[factor]
                heights = [marginals[factor].get(value, 0) for value in factor_levels]
                ax.bar(range(len(factor_levels)), heights, color="#20639b", width=0.75)
                ax.set_xticks(range(len(factor_levels)))
                ax.set_xticklabels(
                    [display_factor_value(factor, value) for value in factor_levels],
                    rotation=40,
                    ha="right",
                    fontsize=6,
                )
                ax.tick_params(labelsize=6, length=2)
                ax.set_title(FACTOR_LABELS[factor], fontsize=8, fontweight="bold")
                ax.margins(y=0.2)
                for index, height in enumerate(heights):
                    ax.text(
                        index,
                        height,
                        str(int(height)),
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                    )
                continue

            x_factor, y_factor = FACTORS[j], FACTORS[i]
            table = pair_lookup[(x_factor, y_factor)]
            x_levels = levels[x_factor]
            y_levels = levels[y_factor]
            matrix = np.array(
                [
                    [table["counts"].get(f"{x_value}|{y_value}", 0) for x_value in x_levels]
                    for y_value in y_levels
                ],
                dtype=float,
            )
            painted = np.where(matrix > 0, matrix, np.nan)
            image = ax.imshow(painted, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            ax.set_xticks(range(len(x_levels)))
            ax.set_xticklabels(
                [display_factor_value(x_factor, value) for value in x_levels],
                rotation=40,
                ha="right",
                fontsize=6,
            )
            ax.set_yticks(range(len(y_levels)))
            ax.set_yticklabels(
                [display_factor_value(y_factor, value) for value in y_levels], fontsize=6
            )
            ax.tick_params(length=2)
            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    count = int(matrix[r, c])
                    if count == 0:
                        ax.add_patch(
                            Rectangle(
                                (c - 0.5, r - 0.5),
                                1,
                                1,
                                fill=False,
                                hatch="///",
                                edgecolor="#cccccc",
                                linewidth=0,
                            )
                        )
                        continue
                    ax.text(
                        c,
                        r,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="#ffffff" if norm(count) >= cmap.N / 2 else "#10334d",
                    )
            if j == 0:
                ax.set_ylabel(FACTOR_LABELS[y_factor], fontsize=7.5, fontweight="bold")
            if i == n - 1:
                ax.set_xlabel(FACTOR_LABELS[x_factor], fontsize=7.5, fontweight="bold")

    if image is not None:
        colorbar = fig.colorbar(
            image,
            ax=axes[0, 1:].tolist(),
            fraction=0.16,
            aspect=14,
            spacing="uniform",
            ticks=count_ticks,
        )
        colorbar.ax.set_yticklabels(count_labels, fontsize=6.5)
        colorbar.set_label("runs per pair of levels", fontsize=7)

    fig.suptitle(
        f"{model}  —  pairwise coverage (max cell = {int(vmax)} runs), "
        f"{audit['observed_configurations']}/{audit['planned_combinations']} full combinations",
        fontsize=10,
    )
    fig.text(0.01, 0.005, _footnote(data, config), ha="left", va="bottom", fontsize=5.8, color="#666666")
    return save_figure(fig, output_stem, formats, int(config["output"]["dpi"]))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_factor_pair(text: str, name: str) -> tuple[str, str]:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--{name} needs exactly two factors, got {text!r}")
    for factor in parts:
        if factor not in FACTORS:
            raise argparse.ArgumentTypeError(f"unknown factor {factor!r}; choose from {list(FACTORS)}")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="saves/unlearn", type=Path)
    parser.add_argument("--config", default="configs/analysis/palu_hparams.yaml", type=Path)
    parser.add_argument("--out", default="figures/coverage", type=Path)
    parser.add_argument("--model", action="append", default=None, help="可重复；默认画磁盘上出现的所有模型")
    parser.add_argument("--rows", default="lr,alpha")
    parser.add_argument("--cols", default="top_k,first_n")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument(
        "--checkpoint-policy",
        default="last",
        choices=("last", "epoch"),
        help=(
            "台账默认用每个 run 的最后一个 checkpoint。"
            "注意 trainer_state.json 记录的末尾 epoch 是 9.6 而非 10，"
            "用 'epoch' 精确匹配 config 里的 10 会漏掉几乎所有 run。"
        ),
    )
    parser.add_argument("--include-short-runs", action="store_true", help="不排除 epoch 数少于目标 checkpoint 的 run")
    args = parser.parse_args()

    row_factors = parse_factor_pair(args.rows, "rows")
    col_factors = parse_factor_pair(args.cols, "cols")
    if set(row_factors) | set(col_factors) != set(FACTORS):
        parser.error(f"--rows and --cols together must cover all four factors {list(FACTORS)}")
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]

    base_config = load_analysis_config(args.config if args.config.is_file() else None)
    records, warnings = collect_records(args.root)
    if not records:
        parser.error(f"no TOFU_SUMMARY.json found under {args.root}")
    for warning in warnings[:5]:
        print(f"[warn] {warning}")
    if len(warnings) > 5:
        print(f"[warn] ... and {len(warnings) - 5} more warnings")

    all_runs = scan_run_dirs(args.root)
    models = args.model or sorted({row["model"] for row in records if row.get("model")})
    print(f"扫描到 {len(records)} 条评估行、{len(all_runs)} 个 run 目录；模型：{', '.join(models)}")

    for model in models:
        config = {
            **base_config,
            "filters": {**base_config.get("filters", {}), "model": model},
            "checkpoint": {**base_config.get("checkpoint", {}), "policy": args.checkpoint_policy},
        }
        data = prepare_model_data(records, all_runs, config, args.include_short_runs)
        if not data["selected"]:
            print(f"[skip] {model}: 没有匹配所选 checkpoint 口径的结果")
            continue
        slug = model_slug(model)
        audit = data["audit"]
        print(
            f"\n{model} ({slug})："
            f"{audit['observed_configurations']}/{audit['planned_combinations']} 组合 "
            f"({audit['coverage_fraction']:.1%})，{len(data['selected'])} 个 run 进入统计"
        )

        written = render_flat(
            data,
            mode="count",
            row_factors=row_factors,
            col_factors=col_factors,
            model=model,
            config=config,
            output_stem=args.out / f"coverage_flat_{slug}",
            formats=formats,
        )
        written += render_pairwise(
            data,
            model=model,
            config=config,
            output_stem=args.out / f"coverage_pairwise_{slug}",
            formats=formats,
        )
        for metric in METRIC_MODES:
            written += render_flat(
                data,
                mode=metric,
                row_factors=row_factors,
                col_factors=col_factors,
                model=model,
                config=config,
                output_stem=args.out / f"metric_flat_{metric}_{slug}",
                formats=formats,
            )
        for path in written:
            print(f"  写出 {path}")


if __name__ == "__main__":
    main()
