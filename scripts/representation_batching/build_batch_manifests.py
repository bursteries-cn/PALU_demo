#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from representation_batching.manifest import (  # noqa: E402
    SUPPORTED_BATCH_ORDERS,
    SUPPORTED_METHODS,
    build_arm_steps,
    load_batch_manifest,
    make_metadata,
    write_batch_manifest,
    write_batch_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic R/S/D/P NPO effective-batch manifests."
    )
    parser.add_argument("--features", type=Path, required=True, help="NPZ produced by extract_tofu_features.py")
    parser.add_argument("--feature-key", required=True, help="Feature array key inside the NPZ file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retain-size", type=int, required=True)
    parser.add_argument("--effective-batch-size", type=int, default=20)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=10)
    parser.add_argument(
        "--batch-order",
        choices=SUPPORTED_BATCH_ORDERS,
        default="random",
        help="Order fixed effective batches randomly or by adjacent batch similarity.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(SUPPORTED_METHODS),
        choices=SUPPORTED_METHODS,
    )
    return parser.parse_args()


def enrich_steps(steps, records_by_id):
    if not records_by_id:
        return
    for step in steps:
        records = [records_by_id[int(sample_id)] for sample_id in step["forget_indices"]]
        authors = [
            str(record.get("source_metadata", {}).get("author"))
            for record in records
            if record.get("source_metadata", {}).get("author") is not None
        ]
        step.update(
            {
                "question_tokens_mean": float(
                    np.mean([record["question_length"] for record in records])
                ),
                "answer_tokens_mean": float(
                    np.mean([record["answer_length"] for record in records])
                ),
                "initial_answer_nll_mean": float(
                    np.mean([record["initial_answer_nll"] for record in records])
                ),
                "author_counts": dict(sorted(Counter(authors).items())),
            }
        )


def main() -> None:
    args = parse_args()
    expected = args.world_size * args.per_device_batch_size * args.gradient_accumulation_steps
    if args.effective_batch_size != expected:
        raise ValueError(
            f"effective batch {args.effective_batch_size} != world_size * per-device batch * "
            f"gradient accumulation ({expected})"
        )

    archive = np.load(args.features, allow_pickle=False)
    if "sample_ids" not in archive:
        raise KeyError(f"{args.features} has no sample_ids array")
    if args.feature_key not in archive:
        available = sorted(archive.files)
        raise KeyError(f"Feature key {args.feature_key!r} not found. Available: {available}")
    sample_ids = archive["sample_ids"].astype(np.int64).tolist()
    features = archive[args.feature_key].astype(np.float32)
    feature_manifest_path = args.features.parent / "feature_manifest.json"
    feature_manifest = None
    records_by_id = None
    if feature_manifest_path.is_file():
        feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        records = feature_manifest.get("records", [])
        records_by_id = {int(record["sample_id"]): record for record in records}
        if set(records_by_id) != set(sample_ids):
            raise ValueError(
                "feature_manifest.json records do not exactly match NPZ sample_ids"
            )

    if len(sample_ids) % args.world_size:
        raise ValueError(
            f"forget set size {len(sample_ids)} is not divisible by world_size {args.world_size}; "
            "the implementation refuses to duplicate or drop samples"
        )
    if len(sample_ids) % args.effective_batch_size:
        raise ValueError(
            f"forget set size {len(sample_ids)} is not divisible by effective batch size "
            f"{args.effective_batch_size}; choose a full accumulation window so epoch "
            "boundaries cannot mix manifest steps"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_steps = {}
    for method in args.methods:
        steps = build_arm_steps(
            sample_ids,
            features,
            method=method,
            effective_batch_size=args.effective_batch_size,
            retain_size=args.retain_size,
            num_epochs=args.num_epochs,
            seed=args.seed,
            batch_order=args.batch_order,
        )
        enrich_steps(steps, records_by_id)
        metadata = make_metadata(
            method=method,
            sample_ids=sample_ids,
            feature_file=args.features,
            feature_key=args.feature_key,
            effective_batch_size=args.effective_batch_size,
            retain_size=args.retain_size,
            num_epochs=args.num_epochs,
            seed=args.seed,
            world_size=args.world_size,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            batch_order=args.batch_order,
        )
        if feature_manifest is not None:
            declared_digest = feature_manifest.get("feature_sha256")
            if declared_digest and declared_digest != metadata["feature_sha256"]:
                raise ValueError(
                    "feature_manifest.json does not match the supplied NPZ feature file"
                )
            metadata.update(
                {
                    "source_feature_manifest": str(feature_manifest_path.resolve()),
                    "source_model": feature_manifest.get("model"),
                    "source_model_revision": feature_manifest.get("model_revision"),
                    "source_dataset": feature_manifest.get("dataset"),
                    "source_dataset_config": feature_manifest.get("dataset_config"),
                    "source_dataset_split": feature_manifest.get("dataset_split"),
                    "source_dataset_revision": feature_manifest.get("dataset_revision"),
                    "source_dataset_fingerprint": feature_manifest.get(
                        "dataset_fingerprint"
                    ),
                    "source_qa_content_sha256": feature_manifest.get(
                        "qa_content_sha256"
                    ),
                    "source_question_key": feature_manifest.get("question_key"),
                    "source_answer_key": feature_manifest.get("answer_key"),
                }
            )
        suffix = "" if args.batch_order == "random" else f"_order-{args.batch_order}"
        output = args.output_dir / f"{method}{suffix}.jsonl"
        write_batch_manifest(output, metadata, steps)
        load_batch_manifest(output, validate=True)
        all_steps[method] = steps
        print(f"wrote {output}")

    write_batch_stats(args.output_dir / "batch_stats.csv", all_steps)
    summary = {
        "features": str(args.features.resolve()),
        "feature_key": args.feature_key,
        "methods": args.methods,
        "seed": args.seed,
        "forget_size": len(sample_ids),
        "retain_size": args.retain_size,
        "effective_batch_size": args.effective_batch_size,
        "world_size": args.world_size,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
        "batch_order": args.batch_order,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
