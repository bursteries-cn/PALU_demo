from __future__ import annotations

from pathlib import Path


_LOCAL_DATASET_SUFFIXES = (
    ("json", ".json"),
    ("json", ".jsonl"),
    ("parquet", ".parquet"),
)


def resolve_local_dataset_config(path: str, name: str | None):
    """Resolve a Hub-style config name to one file in a local directory.

    TOFU stores each configuration as ``<config>.json``. A directory copied
    from the Hub can lose its Dataset Card config metadata, in which case
    ``datasets.load_dataset(directory, config)`` exposes only ``default``.
    This resolver lets callers select the data file explicitly instead.
    """

    root = Path(path).expanduser()
    if not root.is_dir() or not name:
        return None

    matches = [
        (builder, candidate.resolve())
        for builder, suffix in _LOCAL_DATASET_SUFFIXES
        if (candidate := root / f"{name}{suffix}").is_file()
    ]
    if not matches:
        return None
    if len(matches) > 1:
        files = ", ".join(str(candidate) for _, candidate in matches)
        raise ValueError(
            f"Local dataset config {name!r} is ambiguous; multiple files exist: {files}"
        )
    return matches[0]
