from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from .coverage import pair_support, planned_grid
from .pareto import add_balanced_ranks, pareto_front, passes_constraints
from .schema import FACTORS, METRICS, canonical_factor_value, factor_key


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(float), 1e-9, 1 - 1e-9)
    return np.log(clipped / (1 - clipped))


def _inverse_logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


class DesignEncoder:
    """Categorical main effects plus all six pairwise interactions."""

    def __init__(self, search_space: dict[str, list[Any]]):
        self.levels = {
            factor: [canonical_factor_value(factor, value) for value in search_space[factor]]
            for factor in FACTORS
        }
        self.feature_names = ["intercept"]
        self.main_columns: dict[str, list[int]] = {}
        for factor in FACTORS:
            columns = []
            for level in self.levels[factor][1:]:
                columns.append(len(self.feature_names))
                self.feature_names.append(f"{factor}={level}")
            self.main_columns[factor] = columns
        self.interaction_columns: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
        for left_index, left in enumerate(FACTORS):
            for right in FACTORS[left_index + 1 :]:
                columns = []
                for left_column, right_column in itertools.product(
                    self.main_columns[left], self.main_columns[right]
                ):
                    output_column = len(self.feature_names)
                    columns.append((output_column, left_column, right_column))
                    self.feature_names.append(
                        f"{self.feature_names[left_column]}:{self.feature_names[right_column]}"
                    )
                self.interaction_columns[(left, right)] = columns

    def encode(self, rows: Iterable[dict[str, Any]]) -> np.ndarray:
        rows = list(rows)
        output = np.zeros((len(rows), len(self.feature_names)), dtype=float)
        output[:, 0] = 1.0
        for row_index, row in enumerate(rows):
            for factor in FACTORS:
                value = canonical_factor_value(factor, row.get(factor))
                try:
                    level_index = self.levels[factor].index(value)
                except ValueError as exc:
                    raise ValueError(
                        f"value {value!r} for {factor} is outside configured search space"
                    ) from exc
                if level_index > 0:
                    output[row_index, self.main_columns[factor][level_index - 1]] = 1.0
            for columns in self.interaction_columns.values():
                for output_column, left_column, right_column in columns:
                    output[row_index, output_column] = (
                        output[row_index, left_column] * output[row_index, right_column]
                    )
        return output


@dataclass
class RidgeFit:
    beta: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def predict_transformed(self, x: np.ndarray) -> np.ndarray:
        scaled = x.copy().astype(float)
        if scaled.shape[1] > 1:
            scaled[:, 1:] = (scaled[:, 1:] - self.means) / self.scales
        return scaled @ self.beta


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeFit:
    scaled = x.copy().astype(float)
    means = scaled[:, 1:].mean(axis=0) if scaled.shape[1] > 1 else np.array([])
    scales = scaled[:, 1:].std(axis=0) if scaled.shape[1] > 1 else np.array([])
    # Bootstrap samples can make a rare dummy column almost constant. Treat it as
    # constant rather than amplifying floating-point noise during standardization.
    scales = np.where(scales > 1e-6, scales, 1.0)
    if scaled.shape[1] > 1:
        scaled[:, 1:] = (scaled[:, 1:] - means) / scales
    penalty = np.eye(scaled.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    # Some macOS Accelerate/NumPy combinations leave benign floating-point flags
    # after finite BLAS matmuls. Suppress the flag, then enforce finiteness explicitly.
    with np.errstate(all="ignore"):
        matrix = scaled.T @ scaled + penalty
        vector = scaled.T @ y
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
        raise np.linalg.LinAlgError("non-finite Ridge normal equations")
    try:
        beta = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(matrix, vector, rcond=None)[0]
    return RidgeFit(beta=beta, means=means, scales=scales)


def _group_folds(groups: list[str], requested: int, seed: int) -> list[np.ndarray]:
    unique = sorted(set(groups))
    if len(unique) < 3:
        return []
    folds = min(requested, len(unique))
    rng = np.random.default_rng(seed)
    shuffled = np.array(unique, dtype=object)
    rng.shuffle(shuffled)
    group_chunks = np.array_split(shuffled, folds)
    return [
        np.array([index for index, group in enumerate(groups) if group in set(chunk)], dtype=int)
        for chunk in group_chunks
        if len(chunk)
    ]


def _scores(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, Optional[float]]:
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    if denominator <= 1e-15:
        return rmse, None
    r2 = 1.0 - float(np.sum((actual - predicted) ** 2)) / denominator
    return rmse, r2


def _cross_validate(
    x: np.ndarray,
    y_raw: np.ndarray,
    groups: list[str],
    alphas: list[float],
    folds: int,
    seed: int,
) -> tuple[float, dict[str, Any], Optional[np.ndarray]]:
    split_indices = _group_folds(groups, folds, seed)
    if not split_indices:
        return float(alphas[len(alphas) // 2]), {"status": "too_few_groups"}, None
    y = _logit(y_raw)
    all_indices = np.arange(len(y_raw))
    results = []
    predictions_by_alpha = []
    for alpha in alphas:
        predicted = np.full(len(y_raw), np.nan, dtype=float)
        for test_indices in split_indices:
            train_mask = np.ones(len(y_raw), dtype=bool)
            train_mask[test_indices] = False
            if not train_mask.any():
                continue
            fit = _fit_ridge(x[train_mask], y[train_mask], alpha)
            predicted[test_indices] = _inverse_logit(fit.predict_transformed(x[test_indices]))
        valid = np.isfinite(predicted)
        rmse, r2 = _scores(y_raw[valid], predicted[valid])
        results.append({"alpha": float(alpha), "rmse": rmse, "r2": r2})
        predictions_by_alpha.append(predicted)
    best_index = min(range(len(results)), key=lambda index: results[index]["rmse"])
    best = results[best_index]
    return (
        float(best["alpha"]),
        {"status": "ok", "selected": best, "all_alphas": results},
        predictions_by_alpha[best_index],
    )


def _tree_sensitivity(
    x: np.ndarray,
    y: np.ndarray,
    grid_x: np.ndarray,
    ridge_grid: np.ndarray,
    groups: list[str],
    folds: int,
    seed: int,
) -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesRegressor  # type: ignore
    except ImportError:
        return {"status": "dependency_unavailable"}
    split_indices = _group_folds(groups, folds, seed)
    if not split_indices:
        return {"status": "too_few_groups"}
    predicted = np.full(len(y), np.nan, dtype=float)
    all_indices = np.arange(len(y))
    for test_indices in split_indices:
        train_indices = np.setdiff1d(all_indices, test_indices)
        model = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            max_features=0.8,
        )
        model.fit(x[train_indices, 1:], y[train_indices])
        predicted[test_indices] = model.predict(x[test_indices, 1:])
    rmse, r2 = _scores(y, np.clip(predicted, 0, 1))
    final_model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=seed,
        max_features=0.8,
    )
    final_model.fit(x[:, 1:], y)
    tree_grid = np.clip(final_model.predict(grid_x[:, 1:]), 0, 1)
    if np.std(ridge_grid) <= 1e-12 or np.std(tree_grid) <= 1e-12:
        correlation = None
    else:
        correlation = float(np.corrcoef(ridge_grid, tree_grid)[0, 1])
    return {
        "status": "ok",
        "cv_rmse": rmse,
        "cv_r2": r2,
        "ridge_grid_correlation": correlation,
    }


def fit_metric_model(
    rows: list[dict[str, Any]], metric: str, config: dict[str, Any]
) -> dict[str, Any]:
    model_cfg = config.get("modeling", {})
    metric_rows = [row for row in rows if row.get(metric) is not None]
    diagnostics: dict[str, Any] = {
        "metric": metric,
        "n_rows": len(metric_rows),
        "n_total_rows": len(rows),
        "missing_rows": len(rows) - len(metric_rows),
    }
    if len(metric_rows) < int(model_cfg.get("min_rows", 16)):
        diagnostics["status"] = "insufficient_rows"
        return {"metric": metric, "diagnostics": diagnostics}

    seen_levels = {
        factor: sorted(
            {canonical_factor_value(factor, row.get(factor)) for row in metric_rows},
            key=str,
        )
        for factor in FACTORS
    }
    diagnostics["seen_levels"] = {factor: [str(value) for value in values] for factor, values in seen_levels.items()}
    if any(len(values) < 2 for values in seen_levels.values()):
        diagnostics["status"] = "insufficient_factor_levels"
        return {"metric": metric, "diagnostics": diagnostics}

    encoder = DesignEncoder(config["search_space"])
    grid = planned_grid(config)
    x = encoder.encode(metric_rows)
    grid_x = encoder.encode(grid)
    y_raw = np.array([float(row[metric]) for row in metric_rows], dtype=float)
    groups = [str(row.get("config_id", row.get("run_id", index))) for index, row in enumerate(metric_rows)]
    alphas = [float(value) for value in model_cfg.get("ridge_alphas", [0.1, 1, 10])]
    seed = int(model_cfg.get("random_seed", 0))
    selected_alpha, cv, cv_predictions = _cross_validate(
        x,
        y_raw,
        groups,
        alphas,
        int(model_cfg.get("cv_folds", 5)),
        seed,
    )
    fit = _fit_ridge(x, _logit(y_raw), selected_alpha)
    fitted = _inverse_logit(fit.predict_transformed(x))
    grid_predictions = _inverse_logit(fit.predict_transformed(grid_x))
    train_rmse, train_r2 = _scores(y_raw, fitted)
    diagnostics.update(
        {
            "status": "exploratory",
            "n_features": len(encoder.feature_names),
            "selected_alpha": selected_alpha,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "cv": cv,
        }
    )
    cv_r2 = cv.get("selected", {}).get("r2")
    if cv.get("status") == "ok" and cv_r2 is not None and cv_r2 >= float(
        model_cfg.get("min_cv_r2_for_candidates", 0.0)
    ):
        diagnostics["status"] = "validated"

    bootstrap_samples = int(model_cfg.get("bootstrap_samples", 100))
    rng = np.random.default_rng(seed)
    bootstrap_grid = []
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(metric_rows), size=len(metric_rows))
        try:
            bootstrap_fit = _fit_ridge(x[indices], _logit(y_raw[indices]), selected_alpha)
            prediction = _inverse_logit(bootstrap_fit.predict_transformed(grid_x))
            if np.all(np.isfinite(prediction)):
                bootstrap_grid.append(prediction)
        except np.linalg.LinAlgError:
            continue
    if bootstrap_grid:
        bootstrap_array = np.vstack(bootstrap_grid)
        grid_lower = np.quantile(bootstrap_array, 0.025, axis=0)
        grid_upper = np.quantile(bootstrap_array, 0.975, axis=0)
    else:
        bootstrap_array = np.empty((0, len(grid)))
        grid_lower = np.full(len(grid), np.nan)
        grid_upper = np.full(len(grid), np.nan)
    diagnostics["bootstrap_completed"] = int(len(bootstrap_array))
    tree_sensitivity = _tree_sensitivity(
        x,
        y_raw,
        grid_x,
        grid_predictions,
        groups,
        int(model_cfg.get("cv_folds", 5)),
        seed,
    )
    diagnostics["tree_sensitivity"] = tree_sensitivity
    if (
        diagnostics["status"] == "validated"
        and tree_sensitivity.get("status") == "ok"
        and (
            tree_sensitivity.get("ridge_grid_correlation") is None
            or float(tree_sensitivity["ridge_grid_correlation"]) < 0.7
        )
    ):
        diagnostics["status"] = "unstable_surrogate_disagreement"

    grid_rows = []
    for index, candidate in enumerate(grid):
        supported = all(
            canonical_factor_value(factor, candidate[factor]) in seen_levels[factor]
            for factor in FACTORS
        )
        grid_rows.append(
            {
                **candidate,
                "prediction": float(grid_predictions[index]),
                "lower": float(grid_lower[index]),
                "upper": float(grid_upper[index]),
                "level_supported": supported,
            }
        )

    main_effects: dict[str, list[dict[str, Any]]] = {}
    for factor in FACTORS:
        effects = []
        for level in config["search_space"][factor]:
            indices = [
                index
                for index, candidate in enumerate(grid)
                if canonical_factor_value(factor, candidate[factor])
                == canonical_factor_value(factor, level)
            ]
            boot_means = (
                bootstrap_array[:, indices].mean(axis=1) if len(bootstrap_array) else np.array([])
            )
            observed_values = [
                float(row[metric])
                for row in metric_rows
                if canonical_factor_value(factor, row[factor])
                == canonical_factor_value(factor, level)
            ]
            effects.append(
                {
                    "level": level,
                    "adjusted_mean": float(grid_predictions[indices].mean()),
                    "lower": float(np.quantile(boot_means, 0.025)) if len(boot_means) else None,
                    "upper": float(np.quantile(boot_means, 0.975)) if len(boot_means) else None,
                    "n_observed": len(observed_values),
                    "observed_values": observed_values,
                }
            )
        main_effects[factor] = effects

    pair_effects = []
    grand_mean = float(grid_predictions.mean())
    for left_index, left in enumerate(FACTORS):
        left_means = {str(item["level"]): item["adjusted_mean"] for item in main_effects[left]}
        for right in FACTORS[left_index + 1 :]:
            right_means = {str(item["level"]): item["adjusted_mean"] for item in main_effects[right]}
            cells = []
            interaction_values = []
            for left_level in config["search_space"][left]:
                for right_level in config["search_space"][right]:
                    indices = [
                        index
                        for index, candidate in enumerate(grid)
                        if canonical_factor_value(left, candidate[left])
                        == canonical_factor_value(left, left_level)
                        and canonical_factor_value(right, candidate[right])
                        == canonical_factor_value(right, right_level)
                    ]
                    mean_prediction = float(grid_predictions[indices].mean())
                    boot_means = (
                        bootstrap_array[:, indices].mean(axis=1)
                        if len(bootstrap_array)
                        else np.array([])
                    )
                    support = sum(
                        canonical_factor_value(left, row[left])
                        == canonical_factor_value(left, left_level)
                        and canonical_factor_value(right, row[right])
                        == canonical_factor_value(right, right_level)
                        for row in metric_rows
                    )
                    interaction = (
                        mean_prediction
                        - left_means[str(left_level)]
                        - right_means[str(right_level)]
                        + grand_mean
                    )
                    interaction_values.append(interaction)
                    cells.append(
                        {
                            left: left_level,
                            right: right_level,
                            "adjusted_mean": mean_prediction,
                            "lower": float(np.quantile(boot_means, 0.025))
                            if len(boot_means)
                            else None,
                            "upper": float(np.quantile(boot_means, 0.975))
                            if len(boot_means)
                            else None,
                            "n_observed": support,
                            "interaction_residual": interaction,
                        }
                    )
            pair_effects.append(
                {
                    "left": left,
                    "right": right,
                    "interaction_rms": float(np.sqrt(np.mean(np.square(interaction_values)))),
                    "cells": cells,
                }
            )

    residual_rows = []
    for index, row in enumerate(metric_rows):
        residual_rows.append(
            {
                "config_id": row.get("config_id", row.get("run_id")),
                "observed": float(y_raw[index]),
                "fitted": float(fitted[index]),
                "residual": float(y_raw[index] - fitted[index]),
                "cv_prediction": float(cv_predictions[index])
                if cv_predictions is not None and np.isfinite(cv_predictions[index])
                else None,
            }
        )
    return {
        "metric": metric,
        "diagnostics": diagnostics,
        "grid_predictions": grid_rows,
        "main_effects": main_effects,
        "pair_effects": pair_effects,
        "residuals": residual_rows,
    }


def fit_all_metrics(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {metric: fit_metric_model(rows, metric, config) for metric in METRICS}


def predicted_candidates(
    observed: list[dict[str, Any]],
    models: dict[str, dict[str, Any]],
    config: dict[str, Any],
    em_target: Optional[float],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings = []
    invalid = [
        metric
        for metric, result in models.items()
        if result.get("diagnostics", {}).get("status") != "validated"
    ]
    if invalid:
        return [], [
            "predicted candidates disabled because these metric models are not cross-validated: "
            + ", ".join(invalid)
        ]
    observed_keys = {factor_key(row) for row in observed}
    prediction_maps = {
        metric: {factor_key(row): row for row in result["grid_predictions"]}
        for metric, result in models.items()
    }
    candidates = []
    for candidate in planned_grid(config):
        key = factor_key(candidate)
        if key in observed_keys:
            continue
        metric_predictions = {metric: prediction_maps[metric][key] for metric in METRICS}
        if not all(item.get("level_supported") for item in metric_predictions.values()):
            continue
        minimum_support, supports = pair_support(observed, candidate)
        if minimum_support < 1:
            continue
        row = {**candidate, "config_id": "|".join(f"{f}={candidate[f]}" for f in FACTORS)}
        row["minimum_pair_support"] = minimum_support
        row["pair_support"] = supports
        interval_widths = []
        for metric, item in metric_predictions.items():
            row[metric] = item["prediction"]
            row[f"{metric}_pred"] = item["prediction"]
            row[f"{metric}_lower"] = item["lower"]
            row[f"{metric}_upper"] = item["upper"]
            interval_widths.append(item["upper"] - item["lower"])
        row["mean_prediction_interval_width"] = float(np.mean(interval_widths))
        row["passes_constraints"] = passes_constraints(row, config, em_target)
        candidates.append(row)
    eligible_candidates = [row for row in candidates if row["passes_constraints"]]
    eligible_observed = [
        row
        for row in observed
        if passes_constraints(row, config, em_target)
        and all(row.get(metric) is not None for metric in METRICS)
    ]
    # A proposed run is marked Pareto only when its prediction is not dominated
    # by another proposal or by an already observed configuration.
    front = {
        id(row) for row in pareto_front([*eligible_observed, *eligible_candidates], em_target)
    }
    for row in candidates:
        row["pareto_predicted"] = id(row) in front
    add_balanced_ranks(candidates, em_target)
    candidates.sort(
        key=lambda row: (
            not row["passes_constraints"],
            not row["pareto_predicted"],
            -float(row.get("balanced_min_percentile", -1)),
            -float(row.get("balanced_mean_percentile", -1)),
            -int(row["minimum_pair_support"]),
            float(row["mean_prediction_interval_width"]),
        )
    )
    limit = int(config.get("modeling", {}).get("max_predicted_candidates", 30))
    return candidates[:limit], warnings
