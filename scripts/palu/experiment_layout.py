"""Shared timestamp layout and seed-level summary statistics for PALU experiments."""
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
LAYOUT_SCHEMA = 1


def experiment_name(value):
    if not value.strip() or value in (".", "..") or any(char in value for char in "/\\\r\n"):
        raise ValueError("Experiment must be one nonempty directory name")
    return value


def timestamp_directory(experiment):
    # Microseconds distinguish repeated experiment groups, including dry runs.
    return Path(experiment) / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")


def protocol_id(manifest):
    settings = dict(manifest["settings"])
    for key in ("summarize", "output", "output_root", "experiment", "experiment_dir", "planned_seeds",
                "seeds", "gpu", "main_process_port", "keep_model", "dry_run", "python",
                "probe_cache_path", "probe_cache_sha256", "new_group"):
        settings.pop(key, None)
    # Changing the bash seed list or report layout must not create a new scientific protocol.
    orchestration = {"run.sh", "run_experiment.py", "plot_history.py", "experiment_layout.py"}
    inputs = {}
    for name, digest in manifest["code_and_data_sha256"].items():
        path = Path(name)
        if path.parent == ROOT / "scripts/palu" and path.name in orchestration:
            continue
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        inputs[key] = digest
    payload = json.dumps({"settings": settings, "inputs": inputs}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def mean_interval(values):
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Confidence interval requires finite observations")
    n, mean = len(values), statistics.mean(values)
    std = statistics.stdev(values) if n >= 2 else None
    if std is None:
        half = None
    else:
        from scipy.stats import t
        half = float(t.ppf(.975, n - 1)) * std / math.sqrt(n)
    return {"n": n, "mean": mean, "sample_stddev": std, "ci95_half_width": half,
            "ci95_low": mean - half if half is not None else None,
            "ci95_high": mean + half if half is not None else None}


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
