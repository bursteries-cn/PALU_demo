from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

__version__ = "0.1.0"

FACTORS = ("lr", "alpha", "top_k", "first_n")
METRICS = (
    "model_utility",
    "forget_quality",
    "forget_Q_A_gibberish",
    "exact_memorization",
)

FACTOR_LABELS = {
    "lr": "Learning rate",
    "alpha": "Alpha",
    "top_k": "Top-K",
    "first_n": "Initial-N",
}

METRIC_LABELS = {
    "model_utility": "Model utility",
    "forget_quality": "Forget quality (KS p-value)",
    "forget_Q_A_gibberish": "Fluency (clean probability)",
    "exact_memorization": "Exact memorization",
}

# Objective values are converted to "larger is better" before Pareto ranking.
METRIC_OBJECTIVES = {
    "model_utility": "max",
    "forget_quality": "max",
    "forget_Q_A_gibberish": "max",
    "exact_memorization": "target",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "search_space": {
        "lr": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
        "alpha": [0.2, 0.5, 1.0, 2.0, 5.0],
        "top_k": [1, 1000, 5000, 10000],
        "first_n": [1, 2, 3, 4, 5, 10],
    },
    "filters": {
        "model": "Llama-3.1-8B-Instruct",
        "split": "forget05",
        "trainer_contains": "PALU",
        "date": None,
    },
    "checkpoint": {"policy": "epoch", "epoch": 10},
    "protocol": {"allow_mixed": False},
    "baselines": {
        "full_summary": "saves/eval/tofu_Llama-3.1-8B-Instruct_full/evals_forget05/TOFU_SUMMARY.json",
        "retain_summary": "saves/eval/tofu_Llama-3.1-8B-Instruct_retain95/TOFU_SUMMARY.json",
        "overrides": {},
    },
    "ranking": {
        "exact_memorization_target": None,
        "constraints": {
            "model_utility_min": None,
            "forget_quality_min": None,
            "forget_Q_A_gibberish_min": None,
            "exact_memorization_tolerance": None,
        },
    },
    "modeling": {
        "min_rows": 16,
        "cv_folds": 5,
        "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
        "bootstrap_samples": 100,
        "random_seed": 20260831,
        "min_cv_r2_for_candidates": 0.0,
        "max_predicted_candidates": 30,
    },
    "output": {"static_formats": ["png", "pdf", "svg"], "dpi": 300},
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_mapping(path: Path) -> dict[str, Any]:
    """Load JSON or YAML, with JSON kept as a dependency-free YAML subset."""
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is not JSON-compatible YAML and PyYAML is unavailable"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def load_analysis_config(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    return _merge(DEFAULT_CONFIG, load_mapping(path))


def canonical_factor_value(
    factor: str, value: Any
) -> Optional[Union[int, float, str]]:
    if value is None or value == "":
        return None
    if isinstance(value, str) and value.strip().lower() == "all":
        return "all"
    if factor in {"top_k", "first_n"}:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return str(value)
    if factor in {"lr", "alpha"}:
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)
    return value


def factor_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(canonical_factor_value(f, row.get(f))) for f in FACTORS)


def display_factor_value(factor: str, value: Any) -> str:
    value = canonical_factor_value(factor, value)
    if value == "all" or value == -1:
        return "All"
    if factor == "lr" and isinstance(value, (float, int)):
        return f"{float(value):g}"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
