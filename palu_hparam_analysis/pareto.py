from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Iterable, Optional

from .schema import FACTORS, METRICS, METRIC_OBJECTIVES, factor_key


def load_baselines(repo_root: str, config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    import json
    from pathlib import Path

    root = Path(repo_root)
    baseline_cfg = config.get("baselines", {})
    values: dict[str, Any] = {"full": {}, "retain": {}}
    warnings: list[str] = []
    for name, key in (("full", "full_summary"), ("retain", "retain_summary")):
        configured = baseline_cfg.get(key)
        if not configured:
            continue
        path = Path(configured)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            warnings.append(f"{name} baseline summary not found: {path}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"cannot read {name} baseline {path}: {exc}")
            continue
        values[name] = {
            metric: float(data[metric])
            for metric in METRICS
            if isinstance(data.get(metric), (int, float))
        }
        values[name]["summary_path"] = str(path.resolve())
    for dotted, value in baseline_cfg.get("overrides", {}).items():
        try:
            name, metric = dotted.split(".", 1)
        except ValueError:
            warnings.append(f"invalid baseline override key: {dotted}")
            continue
        if name in values and metric in METRICS and value is not None:
            values[name][metric] = float(value)
    return values, warnings


def resolve_em_target(config: dict[str, Any], baselines: dict[str, Any]) -> Optional[float]:
    configured = config.get("ranking", {}).get("exact_memorization_target")
    if configured is not None:
        return float(configured)
    retain = baselines.get("retain", {}).get("exact_memorization")
    return float(retain) if retain is not None else None


def aggregate_configurations(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[factor_key(row)].append(row)
    aggregated = []
    for key, group in sorted(grouped.items()):
        out = {factor: group[0].get(factor) for factor in FACTORS}
        out["config_id"] = "|".join(f"{factor}={out[factor]}" for factor in FACTORS)
        out["n_runs"] = len({row.get("run_id") for row in group})
        out["run_ids"] = sorted({str(row.get("run_id")) for row in group})
        out["seeds"] = sorted({str(row.get("seed")) for row in group})
        out["epoch"] = group[0].get("epoch")
        for metric in METRICS:
            vals = [float(row[metric]) for row in group if row.get(metric) is not None]
            out[metric] = mean(vals) if vals else None
            out[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else None
            out[f"{metric}_min"] = min(vals) if vals else None
            out[f"{metric}_max"] = max(vals) if vals else None
            out[f"{metric}_n"] = len(vals)
        aggregated.append(out)
    return aggregated


def objective_value(metric: str, value: float, em_target: Optional[float]) -> float:
    objective = METRIC_OBJECTIVES[metric]
    if objective == "max":
        return value
    if objective == "min":
        return -value
    if em_target is None:
        return -value
    return -abs(value - em_target)


def passes_constraints(
    row: dict[str, Any], config: dict[str, Any], em_target: Optional[float]
) -> bool:
    constraints = config.get("ranking", {}).get("constraints", {})
    for metric in ("model_utility", "forget_quality", "forget_Q_A_gibberish"):
        threshold = constraints.get(f"{metric}_min")
        if threshold is not None and (
            row.get(metric) is None or float(row[metric]) < float(threshold)
        ):
            return False
    tolerance = constraints.get("exact_memorization_tolerance")
    if tolerance is not None and em_target is not None:
        if row.get("exact_memorization") is None:
            return False
        if abs(float(row["exact_memorization"]) - em_target) > float(tolerance):
            return False
    return True


def pareto_front(
    rows: Iterable[dict[str, Any]], em_target: Optional[float], metric_suffix: str = ""
) -> list[dict[str, Any]]:
    complete = []
    for row in rows:
        values = [row.get(f"{metric}{metric_suffix}") for metric in METRICS]
        if all(value is not None and math.isfinite(float(value)) for value in values):
            complete.append(row)
    front = []
    for candidate in complete:
        candidate_values = [
            objective_value(metric, float(candidate[f"{metric}{metric_suffix}"]), em_target)
            for metric in METRICS
        ]
        dominated = False
        for other in complete:
            if other is candidate:
                continue
            other_values = [
                objective_value(metric, float(other[f"{metric}{metric_suffix}"]), em_target)
                for metric in METRICS
            ]
            if all(right >= left for left, right in zip(candidate_values, other_values)) and any(
                right > left for left, right in zip(candidate_values, other_values)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def _percentile_ranks(values: list[float]) -> list[float]:
    if len(values) == 1:
        return [1.0]
    result = []
    for value in values:
        below = sum(other < value for other in values)
        equal = sum(other == value for other in values)
        result.append((below + 0.5 * (equal - 1)) / (len(values) - 1))
    return result


def add_balanced_ranks(rows: list[dict[str, Any]], em_target: Optional[float]) -> None:
    complete_indices = [
        index for index, row in enumerate(rows) if all(row.get(metric) is not None for metric in METRICS)
    ]
    if not complete_indices:
        return
    metric_ranks: dict[str, list[float]] = {}
    for metric in METRICS:
        values = [
            objective_value(metric, float(rows[index][metric]), em_target)
            for index in complete_indices
        ]
        metric_ranks[metric] = _percentile_ranks(values)
    for local_index, row_index in enumerate(complete_indices):
        ranks = [metric_ranks[metric][local_index] for metric in METRICS]
        rows[row_index]["balanced_min_percentile"] = min(ranks)
        rows[row_index]["balanced_mean_percentile"] = mean(ranks)


def rank_observed(
    rows: list[dict[str, Any]], config: dict[str, Any], em_target: Optional[float]
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if passes_constraints(row, config, em_target)]
    front_ids = {id(row) for row in pareto_front(eligible, em_target)}
    for row in rows:
        row["passes_constraints"] = row in eligible
        row["pareto_observed"] = id(row) in front_ids
        ranges = [
            float(row[f"{metric}_max"]) - float(row[f"{metric}_min"])
            for metric in METRICS
            if row.get(f"{metric}_n", 0) > 1
            and row.get(f"{metric}_max") is not None
            and row.get(f"{metric}_min") is not None
        ]
        row["replicate_supported"] = len(ranges) == len(METRICS)
        row["mean_metric_replicate_range"] = mean(ranges) if ranges else None
    add_balanced_ranks(rows, em_target)
    return sorted(
        rows,
        key=lambda row: (
            not row.get("passes_constraints", False),
            not row.get("pareto_observed", False),
            not row.get("replicate_supported", False),
            float(row["mean_metric_replicate_range"])
            if row.get("mean_metric_replicate_range") is not None
            else math.inf,
            -float(row.get("balanced_min_percentile", -1)),
            -float(row.get("balanced_mean_percentile", -1)),
        ),
    )
