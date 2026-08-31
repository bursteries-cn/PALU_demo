from __future__ import annotations

import itertools
from collections import Counter
from typing import Any, Iterable

from .schema import FACTORS, METRICS, canonical_factor_value, factor_key


def normalized_search_space(config: dict[str, Any]) -> dict[str, list[Any]]:
    search_space = config.get("search_space", {})
    normalized: dict[str, list[Any]] = {}
    for factor in FACTORS:
        values = search_space.get(factor, [])
        normalized[factor] = [canonical_factor_value(factor, value) for value in values]
        if not normalized[factor]:
            raise ValueError(f"search_space.{factor} must contain at least one level")
    return normalized


def planned_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    levels = normalized_search_space(config)
    return [
        dict(zip(FACTORS, values))
        for values in itertools.product(*(levels[factor] for factor in FACTORS))
    ]


def coverage_audit(
    rows: Iterable[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    rows = list(rows)
    grid = planned_grid(config)
    observed_keys = Counter(factor_key(row) for row in rows)
    planned_keys = {factor_key(row) for row in grid}
    outside = [row for row in rows if factor_key(row) not in planned_keys]
    missing = [row for row in grid if factor_key(row) not in observed_keys]
    metric_complete = {
        metric: sum(row.get(metric) is not None for row in rows) for metric in METRICS
    }
    all_complete = sum(all(row.get(metric) is not None for metric in METRICS) for row in rows)

    pair_tables = []
    for left_index, left in enumerate(FACTORS):
        for right in FACTORS[left_index + 1 :]:
            counts = Counter(
                (str(canonical_factor_value(left, row.get(left))), str(canonical_factor_value(right, row.get(right))))
                for row in rows
            )
            pair_tables.append(
                {
                    "left": left,
                    "right": right,
                    "left_levels": config["search_space"][left],
                    "right_levels": config["search_space"][right],
                    "counts": {
                        f"{left_value}|{right_value}": counts[(str(left_value), str(right_value))]
                        for left_value in config["search_space"][left]
                        for right_value in config["search_space"][right]
                    },
                }
            )

    return {
        "planned_combinations": len(grid),
        "observed_configurations": len(observed_keys),
        "observed_rows": len(rows),
        "coverage_fraction": len(observed_keys) / len(grid) if grid else 0.0,
        "complete_four_metric_rows": all_complete,
        "metric_complete": metric_complete,
        "missing_combinations": missing,
        "outside_search_space": outside,
        "pair_tables": pair_tables,
    }


def pair_support(
    rows: Iterable[dict[str, Any]], candidate: dict[str, Any]
) -> tuple[int, dict[str, int]]:
    rows = list(rows)
    supports: dict[str, int] = {}
    for left_index, left in enumerate(FACTORS):
        for right in FACTORS[left_index + 1 :]:
            count = sum(
                canonical_factor_value(left, row.get(left))
                == canonical_factor_value(left, candidate.get(left))
                and canonical_factor_value(right, row.get(right))
                == canonical_factor_value(right, candidate.get(right))
                for row in rows
            )
            supports[f"{left}:{right}"] = count
    return min(supports.values(), default=0), supports
