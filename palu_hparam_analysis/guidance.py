from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Any, Iterable, Optional

from .coverage import normalized_search_space, planned_grid
from .schema import FACTORS, METRICS, canonical_factor_value, factor_key


def _finite(values: Iterable[Any]) -> list[float]:
    out = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _quantile(values: list[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_factor_value_summary(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Describe tested factor levels without imputing unobserved grid cells."""
    levels = normalized_search_space(config)
    summary: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    recommendation_cfg = config.get("recommendation", {})
    max_levels = max(1, int(recommendation_cfg.get("max_promising_levels", 2)))
    tolerance = float(recommendation_cfg.get("promising_score_tolerance", 0.10))

    for factor in FACTORS:
        configured = list(levels[factor])
        extras = []
        for row in rows:
            value = canonical_factor_value(factor, row.get(factor))
            if value not in configured and value not in extras:
                extras.append(value)
        factor_entries = []
        for value in [*configured, *extras]:
            group = [
                row
                for row in rows
                if canonical_factor_value(factor, row.get(factor)) == value
            ]
            scores = _finite(row.get("observed_score") for row in group)
            entry: dict[str, Any] = {
                "factor": factor,
                "value": value,
                "configured": value in configured,
                "status": "tested" if group else "missing",
                "n_configurations": len(group),
                "n_runs": sum(int(row.get("n_runs", 0) or 0) for row in group),
                "score_median": median(scores) if scores else None,
                "score_q25": _quantile(scores, 0.25),
                "score_q75": _quantile(scores, 0.75),
                "score_best": max(scores) if scores else None,
                "pareto_configurations": sum(bool(row.get("pareto_observed")) for row in group),
                "promising": False,
            }
            for metric in METRICS:
                metric_values = _finite(row.get(metric) for row in group)
                entry[f"{metric}_median"] = median(metric_values) if metric_values else None
                entry[f"{metric}_n"] = len(metric_values)
            if not group:
                entry["evidence"] = "not tested"
            elif len(group) == 1:
                entry["evidence"] = "single configuration"
            elif len(group) <= 3:
                entry["evidence"] = "limited"
            else:
                entry["evidence"] = "moderate"
            factor_entries.append(entry)

        ranked = sorted(
            [entry for entry in factor_entries if entry["score_median"] is not None],
            key=lambda entry: (
                -float(entry["score_median"]),
                -int(entry["n_configurations"]),
                configured.index(entry["value"]) if entry["value"] in configured else math.inf,
            ),
        )
        chosen: list[dict[str, Any]] = []
        if ranked:
            best_score = float(ranked[0]["score_median"])
            chosen = [
                entry
                for entry in ranked
                if best_score - float(entry["score_median"]) <= tolerance
            ][:max_levels]
            if not chosen:
                chosen = ranked[:1]
            for entry in chosen:
                entry["promising"] = True

        chosen_values = [entry["value"] for entry in chosen]
        recommendations.append(
            {
                "factor": factor,
                "promising_values": chosen_values,
                "best_value": chosen_values[0] if chosen_values else None,
                "best_score_median": chosen[0]["score_median"] if chosen else None,
                "support_configurations": sum(
                    int(entry["n_configurations"]) for entry in chosen
                ),
                "evidence": (
                    "no scored observations"
                    if not chosen
                    else "very sparse"
                    if sum(int(entry["n_configurations"]) for entry in chosen) <= 1
                    else "limited"
                    if sum(int(entry["n_configurations"]) for entry in chosen) <= 4
                    else "moderate"
                ),
            }
        )
        summary.extend(factor_entries)
    return summary, recommendations


def build_experiment_ledger(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    all_checkpoint_rows: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Return one row per planned configuration plus observed out-of-space rows."""
    observed = {factor_key(row): row for row in rows}
    checkpoint_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in all_checkpoint_rows or []:
        checkpoint_groups[factor_key(row)].append(row)
    ledger: list[dict[str, Any]] = []
    planned_keys = set()
    for combination in planned_grid(config):
        key = factor_key(combination)
        planned_keys.add(key)
        source = observed.get(key)
        other_checkpoints = checkpoint_groups.get(key, [])
        entry = {
            "status": (
                "tested"
                if source
                else "other_checkpoint"
                if other_checkpoints
                else "missing"
            ),
            **combination,
        }
        if source:
            for field in (
                "config_id",
                "n_runs",
                "run_ids",
                "epoch",
                *METRICS,
                "observed_score",
                "score_metrics_used",
                "score_metrics_expected",
                "pareto_observed",
            ):
                entry[field] = source.get(field)
        elif other_checkpoints:
            entry["n_runs"] = len(
                {str(row.get("run_id")) for row in other_checkpoints}
            )
            entry["available_epochs"] = sorted(
                {
                    float(row["epoch"])
                    for row in other_checkpoints
                    if row.get("epoch") is not None
                }
            )
        ledger.append(entry)
    for row in rows:
        if factor_key(row) in planned_keys:
            continue
        ledger.append({"status": "tested_outside_space", **row})
    return ledger


def recommend_neighbor_experiments(
    rows: list[dict[str, Any]], ledger: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Suggest a few untested one-step neighbors of the best observed rows.

    These are local controlled probes, not surrogate predictions.
    """
    recommendation_cfg = config.get("recommendation", {})
    source_limit = max(1, int(recommendation_cfg.get("top_observed_sources", 3)))
    output_limit = max(1, int(recommendation_cfg.get("max_next_experiments", 8)))
    levels = normalized_search_space(config)
    missing = {
        factor_key(row): row for row in ledger if row.get("status") == "missing"
    }
    sources = sorted(
        [
            row
            for row in rows
            if row.get("observed_score") is not None
            and row.get("passes_constraints", True)
        ],
        key=lambda row: (
            -float(row.get("observed_score", -1.0)),
            -float(row.get("balanced_min_percentile", -1.0)),
            str(row.get("config_id", "")),
        ),
    )[:source_limit]
    candidates: dict[tuple[str, ...], dict[str, Any]] = {}
    for source_rank, source in enumerate(sources, start=1):
        for factor in FACTORS:
            factor_levels = levels[factor]
            current = canonical_factor_value(factor, source.get(factor))
            if current not in factor_levels:
                continue
            index = factor_levels.index(current)
            for neighbor_index in (index - 1, index + 1):
                if neighbor_index < 0 or neighbor_index >= len(factor_levels):
                    continue
                candidate = {name: source.get(name) for name in FACTORS}
                candidate[factor] = factor_levels[neighbor_index]
                key = factor_key(candidate)
                if key not in missing:
                    continue
                entry = candidates.setdefault(
                    key,
                    {
                        **candidate,
                        "source_ranks": [],
                        "source_config_ids": [],
                        "changes": [],
                        "source_evidence": [],
                        "best_source_score": float(source["observed_score"]),
                    },
                )
                entry["source_ranks"].append(source_rank)
                entry["source_config_ids"].append(source.get("config_id"))
                entry["changes"].append(
                    f"{factor}: {source.get(factor)} -> {factor_levels[neighbor_index]}"
                )
                entry["source_evidence"].append(
                    {
                        "source_rank": source_rank,
                        "source_config_id": source.get("config_id"),
                        "change": f"{factor}: {source.get(factor)} -> {factor_levels[neighbor_index]}",
                    }
                )
                entry["best_source_score"] = max(
                    float(entry["best_source_score"]), float(source["observed_score"])
                )

    ranked = sorted(
        candidates.values(),
        key=lambda row: (
            min(row["source_ranks"]),
            -len(set(row["source_config_ids"])),
            -float(row["best_source_score"]),
            factor_key(row),
        ),
    )[:output_limit]
    for priority, row in enumerate(ranked, start=1):
        row["priority"] = priority
        row["source_ranks"] = sorted(set(row["source_ranks"]))
        row["source_config_ids"] = sorted(set(str(value) for value in row["source_config_ids"]))
        row["changes"] = sorted(set(row["changes"]))
        row["source_evidence"] = sorted(
            row["source_evidence"],
            key=lambda item: (int(item["source_rank"]), str(item["change"])),
        )
        row["primary_change"] = row["source_evidence"][0]["change"]
        row["neighbor_support"] = len(row["source_config_ids"])
        row["reason"] = (
            "one-step neighbor of observed rank(s) "
            + ", ".join(str(value) for value in row["source_ranks"])
        )
    return ranked
