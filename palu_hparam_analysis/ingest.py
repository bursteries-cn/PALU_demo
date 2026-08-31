from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .schema import FACTORS, METRICS, canonical_factor_value

DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NUMBER = r"[0-9.eE+-]+"
RUN_RE = re.compile(
    rf"target_mode(?P<target_mode>[a-zA-Z]+)"
    rf"_first_n(?P<first_n>-?\d+|all)"
    rf"_lr(?P<lr>{NUMBER})"
    rf"_b(?P<batch_size>\d+)"
    rf"_ga(?P<gradient_accumulation>\d+)"
    rf"_a(?P<alpha>{NUMBER})"
    rf"_topk(?P<top_k>-?\d+|all)"
    rf"_e(?P<num_train_epochs>\d+)"
    rf"_day(?P<day>\d+)_time(?P<time>\d+)$",
    re.IGNORECASE,
)

CONFIG_FIELDS = {
    "trainer.args.learning_rate": "lr",
    "trainer.method_args.alpha": "alpha",
    "trainer.method_args.top_k": "top_k",
    "trainer.method_args.first_n": "first_n",
    "trainer.args.per_device_train_batch_size": "batch_size",
    "trainer.args.gradient_accumulation_steps": "gradient_accumulation",
    "trainer.args.num_train_epochs": "num_train_epochs",
    "trainer.args.seed": "seed",
    "trainer.method_args.gamma": "gamma",
    "trainer.method_args.target_mode": "target_mode",
    "trainer.method_args.retain_loss_type": "retain_loss_type",
    "model.model_args.attn_implementation": "attention",
    "model.model_args.torch_dtype": "torch_dtype",
    "model.handler": "model",
    "forget_split": "split",
    "eval.tofu.batch_size": "eval_batch_size",
    "eval.tofu.retain_logs_path": "retain_logs_path",
    "eval.tofu.metrics.forget_Q_A_gibberish.classifier_model_args.pretrained_model_name_or_path": "gibberish_classifier",
    "eval.tofu.metrics.forget_Q_A_gibberish.class_id": "gibberish_class_id",
    "eval.tofu.metrics.forget_Q_A_gibberish.max_length": "gibberish_max_length",
    "eval.tofu.metrics.forget_Q_A_ROUGE.generation_args.do_sample": "generation_do_sample",
    "eval.tofu.metrics.forget_Q_A_ROUGE.generation_args.max_new_tokens": "generation_max_new_tokens",
    "eval.tofu.metrics.forget_Q_A_ROUGE.generation_args.temperature": "generation_temperature",
    "eval.tofu.metrics.forget_Q_A_ROUGE.generation_args.top_p": "generation_top_p",
}


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _minimal_yaml_dotted(text: str, wanted: set[str]) -> dict[str, Any]:
    """Read scalar dotted paths from Hydra YAML when PyYAML is unavailable.

    It intentionally ignores lists and complex YAML constructs. All fields used by
    this module are nested scalar mappings in Hydra's resolved config.
    """
    found: dict[str, Any] = {}
    stack: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith(("#", "- ")):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().strip("'\"")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([entry[1] for entry in stack] + [key])
        if value.strip():
            if path in wanted:
                found[path] = _scalar(value.split(" #", 1)[0])
        else:
            stack.append((indent, key))
    return found


def _nested_get(value: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def read_hydra_fields(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    wanted = set(CONFIG_FIELDS)
    resolved: dict[str, Any]
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        resolved = {key: _nested_get(data, key) for key in wanted}
        resolved = {key: value for key, value in resolved.items() if value is not None}
    except (ImportError, ValueError, TypeError):
        resolved = _minimal_yaml_dotted(text, wanted)
    return {CONFIG_FIELDS[key]: value for key, value in resolved.items()}


def parse_run_name(name: str) -> dict[str, Any]:
    match = RUN_RE.fullmatch(name)
    if match is None:
        return {}
    values: dict[str, Any] = match.groupdict()
    for factor in FACTORS:
        values[factor] = canonical_factor_value(factor, values[factor])
    for key in ("batch_size", "gradient_accumulation", "num_train_epochs"):
        values[key] = int(values[key])
    return values


def _path_metadata(summary: Path, root: Path, run_dir: Path) -> dict[str, Any]:
    try:
        rel = run_dir.relative_to(root).parts
    except ValueError:
        rel = run_dir.parts
    out: dict[str, Any] = {"run": run_dir.name, "run_dir": str(run_dir.resolve())}
    # Expected root-relative layout: tofu/split/model/[date]/trainer/run.
    if len(rel) >= 5:
        out["benchmark"] = rel[0]
        out["split"] = rel[1]
        out["model"] = rel[2]
        index = 3
        if index < len(rel) and DATE_DIR_RE.fullmatch(rel[index]):
            out["date"] = rel[index]
            index += 1
        else:
            out["date"] = ""
        if index < len(rel) - 1:
            out["trainer"] = rel[index]
    checkpoint = summary.parents[1].name
    out["step"] = int(checkpoint.split("-", 1)[1])
    return out


def _equivalent(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return True
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(left).lower() == str(right).lower()


def _trainer_epoch_map(run_dir: Path) -> dict[int, float]:
    state_path = run_dir / "trainer_state.json"
    if not state_path.is_file():
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    mapping: dict[int, float] = {0: 0.0}
    for item in state.get("log_history", []):
        if isinstance(item, dict) and "step" in item and "epoch" in item:
            try:
                mapping[int(item["step"])] = float(item["epoch"])
            except (TypeError, ValueError):
                continue
    return mapping


def collect_records(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not root.is_dir():
        return [], [f"result root does not exist: {root}"]

    for summary in sorted(root.rglob("checkpoint-*/evals/TOFU_SUMMARY.json")):
        run_dir = summary.parents[2]
        path_values = {**_path_metadata(summary, root, run_dir), **parse_run_name(run_dir.name)}
        config_values = read_hydra_fields(run_dir / ".hydra" / "config.yaml")
        mismatch = []
        for key, config_value in config_values.items():
            if key in path_values and not _equivalent(path_values[key], config_value):
                mismatch.append(f"{key}: path={path_values[key]!r}, config={config_value!r}")
        if mismatch:
            warnings.append(f"metadata mismatch in {run_dir}: " + "; ".join(mismatch))
        row = {**path_values, **config_values}
        row["metadata_source"] = "hydra+path" if config_values else "path"
        row["summary_path"] = str(summary.resolve())
        try:
            metrics = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"cannot read {summary}: {exc}")
            continue
        for metric in METRICS:
            value = metrics.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                row[metric] = float(value)
            else:
                row[metric] = None
        for factor in FACTORS:
            row[factor] = canonical_factor_value(factor, row.get(factor))
        try:
            row["effective_batch"] = int(row["batch_size"]) * int(
                row["gradient_accumulation"]
            )
        except (KeyError, TypeError, ValueError):
            row["effective_batch"] = None
        row["seed"] = row.get("seed", 0)
        row["run_id"] = str(run_dir.resolve())
        records.append(row)

    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_run[row["run_id"]].append(row)
    for run_id, run_rows in by_run.items():
        run_rows.sort(key=lambda item: item["step"])
        epoch_map = _trainer_epoch_map(Path(run_id))
        for fallback_epoch, row in enumerate(run_rows):
            if row["step"] in epoch_map:
                row["epoch"] = epoch_map[row["step"]]
                row["epoch_source"] = "trainer_state"
            else:
                row["epoch"] = float(fallback_epoch)
                row["epoch_source"] = "checkpoint_order"
        last_step = max(item["step"] for item in run_rows)
        for row in run_rows:
            row["is_last"] = row["step"] == last_step
    return records, warnings


def filter_records(records: Iterable[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    filters = config.get("filters", {})
    out = []
    for row in records:
        if filters.get("model") and row.get("model") != filters["model"]:
            continue
        if filters.get("split") and row.get("split") != filters["split"]:
            continue
        trainer_contains = filters.get("trainer_contains")
        if trainer_contains and trainer_contains.lower() not in str(row.get("trainer", "")).lower():
            continue
        if filters.get("date") and row.get("date") != filters["date"]:
            continue
        if any(row.get(factor) is None for factor in FACTORS):
            continue
        out.append(row)
    return out


def select_checkpoints(records: Iterable[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoint = config.get("checkpoint", {})
    policy = checkpoint.get("policy", "epoch")
    if policy == "last":
        return [row for row in records if row.get("is_last")]
    if policy != "epoch":
        raise ValueError(f"unsupported checkpoint policy: {policy}")
    wanted = float(checkpoint.get("epoch", 10))
    return [row for row in records if math.isclose(float(row.get("epoch", -1)), wanted)]


PROTOCOL_FIELDS = (
    "model",
    "split",
    "trainer",
    "attention",
    "torch_dtype",
    "effective_batch",
    "target_mode",
    "gamma",
    "retain_loss_type",
    "eval_batch_size",
    "retain_logs_path",
    "gibberish_classifier",
    "gibberish_class_id",
    "gibberish_max_length",
    "generation_do_sample",
    "generation_max_new_tokens",
    "generation_temperature",
    "generation_top_p",
)


def protocol_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "unknown")) for field in PROTOCOL_FIELDS)


def select_protocol_cohort(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    counts = Counter(protocol_key(row) for row in records)
    cohort_table = [
        {**dict(zip(PROTOCOL_FIELDS, key)), "n_rows": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    if not records or config.get("protocol", {}).get("allow_mixed", False):
        return records, [], cohort_table
    largest = counts.most_common(1)[0][0]
    included = [row for row in records if protocol_key(row) == largest]
    excluded = [row for row in records if protocol_key(row) != largest]
    return included, excluded, cohort_table
