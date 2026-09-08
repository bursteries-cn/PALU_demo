from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = 1
SUPPORTED_METHODS = ("R", "S", "D", "P")
SUPPORTED_BATCH_ORDERS = ("random", "similar", "diverse")


@dataclass(frozen=True)
class BatchManifest:
    metadata: Mapping[str, object]
    epochs: Mapping[int, Tuple[Mapping[str, object], ...]]

    def steps_for_epoch(self, epoch: int) -> Tuple[Mapping[str, object], ...]:
        if epoch not in self.epochs:
            raise KeyError(
                f"Manifest has epochs {sorted(self.epochs)}, requested epoch {epoch}."
            )
        return self.epochs[epoch]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qa_content_sha256(rows, question_key: str, answer_key: str) -> str:
    """Hash ordered QA content while ignoring framework-specific dataset fingerprints."""
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        payload = [index, row[question_key], row[answer_key]]
        digest.update(
            (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def l2_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be rank 2, got shape {features.shape}")
    if not np.isfinite(features).all():
        bad = np.argwhere(~np.isfinite(features))
        raise ValueError(f"non-finite feature values at {bad[:10].tolist()}")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 0):
        bad = np.flatnonzero(norms.reshape(-1) <= 0).tolist()
        raise ValueError(f"zero-norm feature rows: {bad[:10]}")
    return features / norms


def mean_pairwise_cosine(batch_ids: Sequence[int], similarity: np.ndarray, row: Mapping[int, int]) -> float:
    if len(batch_ids) < 2:
        return 0.0
    positions = [row[int(sample_id)] for sample_id in batch_ids]
    block = similarity[np.ix_(positions, positions)]
    upper = block[np.triu_indices(len(positions), k=1)]
    return float(upper.mean())


def _seed(base_seed: int, epoch: int, stream: int) -> int:
    return int(base_seed + 100_003 * epoch + 10_007 * stream)


def _partition_random(sample_ids: Sequence[int], batch_size: int) -> List[List[int]]:
    return [
        [int(x) for x in sample_ids[start : start + batch_size]]
        for start in range(0, len(sample_ids), batch_size)
    ]


def _partition_greedy(
    sample_ids: Sequence[int],
    batch_size: int,
    similarity: np.ndarray,
    row: Mapping[int, int],
    priority: Mapping[int, int],
    mode: str,
) -> List[List[int]]:
    if mode not in ("similar", "diverse"):
        raise ValueError(f"Unknown greedy grouping mode: {mode}")
    remaining = {int(x) for x in sample_ids}
    batches: List[List[int]] = []
    while remaining:
        anchor = min(remaining, key=lambda sample_id: priority[sample_id])
        batch = [anchor]
        remaining.remove(anchor)
        while remaining and len(batch) < batch_size:
            batch_rows = [row[x] for x in batch]
            scored = []
            for candidate in remaining:
                score = float(similarity[row[candidate], batch_rows].mean())
                scored.append((candidate, score))
            if mode == "similar":
                chosen = min(scored, key=lambda item: (-item[1], priority[item[0]]))[0]
            else:
                chosen = min(scored, key=lambda item: (item[1], priority[item[0]]))[0]
            batch.append(chosen)
            remaining.remove(chosen)
        batches.append(batch)
    return batches


def _cross_batch_cosine(
    left: Sequence[int],
    right: Sequence[int],
    similarity: np.ndarray,
    row: Mapping[int, int],
) -> float:
    left_rows = [row[int(sample_id)] for sample_id in left]
    right_rows = [row[int(sample_id)] for sample_id in right]
    return float(similarity[np.ix_(left_rows, right_rows)].mean())


def _order_batches(
    batches: Sequence[Sequence[int]],
    *,
    mode: str,
    seed_order: Sequence[int],
    similarity: np.ndarray,
    row: Mapping[int, int],
) -> List[List[int]]:
    if mode not in SUPPORTED_BATCH_ORDERS:
        raise ValueError(f"batch order must be one of {SUPPORTED_BATCH_ORDERS}, got {mode}")
    if not batches:
        return []
    if mode == "random":
        return [[int(x) for x in batches[index]] for index in seed_order]

    priority = {batch_index: rank for rank, batch_index in enumerate(seed_order)}
    current = seed_order[0]
    remaining = set(range(len(batches)))
    remaining.remove(current)
    ordered = [current]
    while remaining:
        scored = [
            (
                candidate,
                _cross_batch_cosine(
                    batches[current], batches[candidate], similarity, row
                ),
            )
            for candidate in remaining
        ]
        if mode == "similar":
            current = min(scored, key=lambda item: (-item[1], priority[item[0]]))[0]
        else:
            current = min(scored, key=lambda item: (item[1], priority[item[0]]))[0]
        ordered.append(current)
        remaining.remove(current)
    return [[int(x) for x in batches[index]] for index in ordered]


def build_arm_steps(
    sample_ids: Sequence[int],
    features: np.ndarray,
    *,
    method: str,
    effective_batch_size: int,
    retain_size: int,
    num_epochs: int,
    seed: int,
    batch_order: str = "random",
) -> List[dict]:
    """Create deterministic optimizer-step manifests for one grouping arm.

    R randomly partitions samples, S greedily groups similar samples, D greedily
    groups dissimilar samples, and P runs the S algorithm after a fixed feature
    permutation. Full-batch order and retain ids are shared across methods when
    all other arguments are identical.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}, got {method}")
    if batch_order not in SUPPORTED_BATCH_ORDERS:
        raise ValueError(
            f"batch_order must be one of {SUPPORTED_BATCH_ORDERS}, got {batch_order}"
        )
    if effective_batch_size <= 0 or retain_size <= 0 or num_epochs <= 0:
        raise ValueError("batch size, retain size, and num_epochs must be positive")

    ids = [int(x) for x in sample_ids]
    if not ids:
        raise ValueError("sample_ids must not be empty")
    if len(ids) != len(set(ids)):
        raise ValueError("sample_ids must be unique")
    normalized = l2_normalize(features)
    if normalized.shape[0] != len(ids):
        raise ValueError("sample_ids and features must have the same number of rows")
    if len(ids) % effective_batch_size:
        raise ValueError(
            f"forget set size {len(ids)} is not divisible by effective batch size "
            f"{effective_batch_size}. Transformers 4.45.1 does not commit a normal "
            "partial gradient-accumulation window at an epoch boundary; choose a "
            "batch size that preserves every manifest step exactly."
        )
    row = {sample_id: index for index, sample_id in enumerate(ids)}

    grouping_features = normalized
    if method == "P":
        permutation = np.random.default_rng(_seed(seed, 0, 91)).permutation(len(ids))
        grouping_features = normalized[permutation]
    grouping_similarity = np.einsum(
        "id,jd->ij", grouping_features, grouping_features, optimize=True
    )
    true_similarity = np.einsum(
        "id,jd->ij", normalized, normalized, optimize=True
    )

    steps: List[dict] = []
    for epoch in range(num_epochs):
        priority_order = np.random.default_rng(_seed(seed, epoch, 1)).permutation(ids).tolist()
        priority = {sample_id: index for index, sample_id in enumerate(priority_order)}

        if method == "R":
            full_batches = _partition_random(priority_order, effective_batch_size)
        else:
            grouping_mode = "diverse" if method == "D" else "similar"
            full_batches = _partition_greedy(
                priority_order,
                effective_batch_size,
                grouping_similarity,
                row,
                priority,
                grouping_mode,
            )

        order_rng = np.random.default_rng(_seed(seed, epoch, 2))
        seed_order = order_rng.permutation(len(full_batches)).tolist()
        batches = _order_batches(
            full_batches,
            mode=batch_order,
            seed_order=seed_order,
            similarity=true_similarity,
            row=row,
        )
        retain_rng = np.random.default_rng(_seed(seed, epoch, 3))
        for optimizer_step, forget_ids in enumerate(batches):
            retain_ids = retain_rng.integers(0, retain_size, size=len(forget_ids)).tolist()
            steps.append(
                {
                    "type": "step",
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "method": method,
                    "batch_order": batch_order,
                    "forget_indices": [int(x) for x in forget_ids],
                    "retain_indices": [int(x) for x in retain_ids],
                    "within_batch_cosine": mean_pairwise_cosine(
                        forget_ids, true_similarity, row
                    ),
                }
            )
    return steps


def make_metadata(
    *,
    method: str,
    sample_ids: Sequence[int],
    feature_file: str | Path,
    feature_key: str,
    effective_batch_size: int,
    retain_size: int,
    num_epochs: int,
    seed: int,
    world_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    batch_order: str = "random",
) -> dict:
    return {
        "type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "batch_order": batch_order,
        "sample_ids": [int(x) for x in sample_ids],
        "feature_file": str(Path(feature_file).resolve()),
        "feature_sha256": sha256_file(feature_file),
        "feature_key": feature_key,
        "effective_batch_size": int(effective_batch_size),
        "retain_size": int(retain_size),
        "num_epochs": int(num_epochs),
        "seed": int(seed),
        "world_size": int(world_size),
        "per_device_batch_size": int(per_device_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
    }


def write_batch_manifest(path: str | Path, metadata: Mapping[str, object], steps: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(metadata), ensure_ascii=False) + "\n")
        for step in steps:
            handle.write(json.dumps(dict(step), ensure_ascii=False) + "\n")


def write_batch_stats(path: str | Path, manifests: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "batch_order",
                "epoch",
                "optimizer_step",
                "batch_size",
                "within_batch_cosine",
                "question_tokens_mean",
                "answer_tokens_mean",
                "initial_answer_nll_mean",
                "num_authors",
                "author_counts_json",
            ),
        )
        writer.writeheader()
        for method, steps in manifests.items():
            for step in steps:
                writer.writerow(
                    {
                        "method": method,
                        "batch_order": step.get("batch_order", "random"),
                        "epoch": step["epoch"],
                        "optimizer_step": step["optimizer_step"],
                        "batch_size": len(step["forget_indices"]),
                        "within_batch_cosine": step["within_batch_cosine"],
                        "question_tokens_mean": step.get("question_tokens_mean"),
                        "answer_tokens_mean": step.get("answer_tokens_mean"),
                        "initial_answer_nll_mean": step.get(
                            "initial_answer_nll_mean"
                        ),
                        "num_authors": len(step.get("author_counts", {})),
                        "author_counts_json": json.dumps(
                            step.get("author_counts", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )


def load_batch_manifest(path: str | Path, *, validate: bool = True) -> BatchManifest:
    path = Path(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or records[0].get("type") != "metadata":
        raise ValueError(f"{path} must start with a metadata record")
    metadata = records[0]
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema {metadata.get('schema_version')}; expected {SCHEMA_VERSION}"
        )
    epochs: Dict[int, List[Mapping[str, object]]] = {}
    for record in records[1:]:
        if record.get("type") != "step":
            raise ValueError(f"Unexpected manifest record type: {record.get('type')}")
        epochs.setdefault(int(record["epoch"]), []).append(record)
    frozen_epochs = {epoch: tuple(steps) for epoch, steps in epochs.items()}
    manifest = BatchManifest(metadata=metadata, epochs=frozen_epochs)
    if validate:
        validate_batch_manifest(manifest)
    return manifest


def validate_batch_manifest(manifest: BatchManifest) -> None:
    meta = manifest.metadata
    required = {
        "method",
        "batch_order",
        "sample_ids",
        "effective_batch_size",
        "retain_size",
        "num_epochs",
        "seed",
        "world_size",
        "per_device_batch_size",
        "gradient_accumulation_steps",
    }
    missing = required.difference(meta)
    if missing:
        raise ValueError(f"Manifest metadata missing fields: {sorted(missing)}")
    if meta["method"] not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported manifest method: {meta['method']}")
    if meta["batch_order"] not in SUPPORTED_BATCH_ORDERS:
        raise ValueError(f"Unsupported manifest batch order: {meta['batch_order']}")
    world_size = int(meta["world_size"])
    per_device = int(meta["per_device_batch_size"])
    grad_accum = int(meta["gradient_accumulation_steps"])
    effective = int(meta["effective_batch_size"])
    retain_size = int(meta["retain_size"])
    num_epochs = int(meta["num_epochs"])
    if min(world_size, per_device, grad_accum, effective, retain_size, num_epochs) <= 0:
        raise ValueError("Manifest batch dimensions must be positive")
    expected_effective = world_size * per_device * grad_accum
    if effective != expected_effective:
        raise ValueError(
            f"effective_batch_size={effective}, but world_size * per_device_batch_size * "
            f"gradient_accumulation_steps={expected_effective}"
        )

    expected_ids = [int(x) for x in meta["sample_ids"]]
    if not expected_ids:
        raise ValueError("metadata sample_ids must not be empty")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("metadata sample_ids must be unique")
    expected_epochs = set(range(num_epochs))
    if set(manifest.epochs) != expected_epochs:
        raise ValueError(
            f"Manifest epochs are {sorted(manifest.epochs)}, expected {sorted(expected_epochs)}"
        )

    for epoch, steps in manifest.epochs.items():
        seen: List[int] = []
        for expected_step, step in enumerate(steps):
            if int(step["optimizer_step"]) != expected_step:
                raise ValueError(f"Epoch {epoch} optimizer steps are not contiguous")
            if step.get("method") != meta["method"]:
                raise ValueError(f"Epoch {epoch} step {expected_step} method mismatch")
            if step.get("batch_order") != meta["batch_order"]:
                raise ValueError(f"Epoch {epoch} step {expected_step} batch-order mismatch")
            forget_ids = [int(x) for x in step["forget_indices"]]
            retain_ids = [int(x) for x in step["retain_indices"]]
            if len(forget_ids) != effective:
                raise ValueError(
                    f"Epoch {epoch} step {expected_step} has {len(forget_ids)} samples; "
                    f"every optimizer step must contain exactly {effective}"
                )
            if len(forget_ids) != len(retain_ids):
                raise ValueError(f"Forget/retain size mismatch in epoch {epoch} step {expected_step}")
            if len(forget_ids) % world_size:
                raise ValueError(
                    f"Epoch {epoch} step {expected_step} size {len(forget_ids)} is not divisible "
                    f"by world_size {world_size}; no sample duplication is allowed"
                )
            local_microsteps = len(forget_ids) // (world_size * per_device)
            if local_microsteps <= 0 or local_microsteps > grad_accum:
                raise ValueError(
                    f"Epoch {epoch} step {expected_step} cannot be split into the configured microbatches"
                )
            if any(x < 0 or x >= retain_size for x in retain_ids):
                raise ValueError(f"Retain index out of range in epoch {epoch} step {expected_step}")
            seen.extend(forget_ids)
        if sorted(seen) != sorted(expected_ids):
            raise ValueError(f"Epoch {epoch} does not cover each forget sample exactly once")


def split_step_for_rank(
    step: Mapping[str, object],
    *,
    rank: int,
    world_size: int,
    per_device_batch_size: int,
) -> List[List[Tuple[int, int]]]:
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} outside [0, {world_size})")
    forget_ids = [int(x) for x in step["forget_indices"]]
    retain_ids = [int(x) for x in step["retain_indices"]]
    stride = world_size * per_device_batch_size
    if len(forget_ids) % stride:
        raise ValueError(
            f"Batch of {len(forget_ids)} samples cannot be split across {world_size} ranks "
            f"with per-device batch {per_device_batch_size}"
        )
    microbatches: List[List[Tuple[int, int]]] = []
    for start in range(0, len(forget_ids), stride):
        rank_start = start + rank * per_device_batch_size
        rank_stop = rank_start + per_device_batch_size
        microbatches.append(
            list(zip(forget_ids[rank_start:rank_stop], retain_ids[rank_start:rank_stop]))
        )
    return microbatches
