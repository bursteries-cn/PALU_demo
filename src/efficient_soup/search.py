"""Deterministic two-edge, greedy midpoint search in the model simplex.

The paper does not specify all bracket/tie rules. This implementation replaces
the worse endpoint by the midpoint (ties replace the right endpoint), while
retaining the best evaluated point. It is a heuristic, not global optimization.
"""
import math


def coefficients(values):
    values = tuple(float(v) for v in values)
    if len(values) != 3 or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Expected three finite, nonnegative coefficients")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-10):
        raise ValueError("Coefficients must sum to one")
    return values


def deviation(es_forget, es_retain):
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (es_forget, es_retain)):
        raise ValueError("ES must be finite and within [0, 1]")
    return math.hypot(es_forget, 1 - es_retain)


def efficient_search(evaluate, depth=3):
    """Return endpoint, stage-one, final records; lower DS is better.

    evaluate(weights, stage) returns a mapping with `ds_es`. Cache is scoped to
    this invocation; no results are reused across models/datasets or runs.
    At most 3 + 2 * depth evaluations. Equal scores keep the incumbent.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    cache = {}
    history = []

    def score(weights, stage):
        weights = coefficients(weights)
        if weights not in cache:
            result = dict(evaluate(weights, stage))
            if not math.isfinite(result["ds_es"]):
                raise ValueError("Non-finite search score")
            result.update(weights=list(weights), stage=stage, evaluation=len(history))
            cache[weights] = result
            history.append(result)
        return cache[weights]

    original = score((1, 0, 0), "original")
    candidates = [score((0, 1, 0), "candidate_1"), score((0, 0, 1), "candidate_2")]
    better, other = sorted(candidates, key=lambda item: item["ds_es"])

    def edge(left, right, stage):
        best = min((left, right), key=lambda item: item["ds_es"])
        for _ in range(depth):
            midpoint = score(tuple((a + b) / 2 for a, b in zip(left["weights"], right["weights"])), stage)
            if midpoint["ds_es"] < best["ds_es"]:
                best = midpoint
            if left["ds_es"] <= right["ds_es"]:
                right = midpoint
            else:
                left = midpoint
        return best

    stage_one = edge(original, better, "anchor_edge")
    final = edge(stage_one, other, "second_edge")
    return {"original": original, "candidates": candidates, "best_single": better,
            "stage_one": stage_one, "final": final, "history": history}
