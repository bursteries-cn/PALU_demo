#!/usr/bin/env python3
"""Compare original and sample-parallel evaluation of one immutable checkpoint.

Run each backend in a separate four-rank torchrun, then --backend compare. No
training update, resume, checkpoint deletion or modification of source artifacts.
--smoke uses F7/R11 and the first 65 examples of each native dataset; it is not a
full-dataset scientific validation. Tolerances are fixed before the measurements.
"""
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def compare_values(left, right, atol=1e-6, rtol=1e-6):
    stats = {"numeric_count": 0, "numeric_unequal": 0, "max_absolute_error": 0.0,
             "max_relative_error": 0.0, "outside_tolerance": 0, "structure_or_text_errors": 0,
             "examples": [], "atol": atol, "rtol": rtol}

    def visit(a, b, path):
        bad = False
        if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
            for key in a:
                visit(a[key], b[key], path + "/" + str(key))
            return
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + "/" + str(i))
            return
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            stats["numeric_count"] += 1
            error = abs(a - b)
            stats["numeric_unequal"] += a != b
            stats["max_absolute_error"] = max(stats["max_absolute_error"], error)
            stats["max_relative_error"] = max(stats["max_relative_error"], error / max(abs(a), abs(b), 1e-300))
            bad = not math.isfinite(a) or not math.isfinite(b) or error > atol + rtol * abs(a)
            stats["outside_tolerance"] += bad
        elif a != b:
            stats["structure_or_text_errors"] += 1
            bad = True
        if bad and len(stats["examples"]) < 20:
            stats["examples"].append({"path": path, "original": str(a)[:300], "sharded": str(b)[:300]})
    visit(left, right, "")
    stats["passed"] = stats["outside_tolerance"] == stats["structure_or_text_errors"] == 0
    return stats


def compare(output):
    receipts = {name: read(output / name / "benchmark.json") for name in ("legacy", "sharded")}
    a, b = receipts.values()
    for key in ("checkpoint_parameter_sha256", "teacher_parameter_sha256", "smoke",
                "schedule_sha256", "config_sha256", "world_size", "cuda_visible_devices",
                "source_sha256", "software", "gpu_inventory"):
        if a[key] != b[key]:
            raise ValueError("Benchmark conditions differ: " + key)
    def artifacts(name):
        workspace = output / name / "online_evaluation_workspace"
        fixed = [json.loads(line) for line in (workspace / "fixed_loss_samples.jsonl").read_text().splitlines()]
        native = read(Path(receipts[name]["row"]["evaluation_dir"]) / "TOFU_EVAL.json")
        return fixed, native
    fixed_a, native_a = artifacts("legacy")
    fixed_b, native_b = artifacts("sharded")
    if not a["smoke"] and (len(fixed_a) != 4000 or len(fixed_b) != 4000):
        raise AssertionError("Full validation requires every fixed F200/R3800 sample")
    fixed, native = compare_values(fixed_a, fixed_b), compare_values(native_a, native_b)
    metrics = {key: {"legacy": a["row"][key], "sharded": b["row"][key],
                     "absolute_error": abs(a["row"][key] - b["row"][key])}
               for key in ("forget_quality", "model_utility", "ks_d")}
    timing = {}
    for name, receipt in receipts.items():
        row = receipt["row"]
        timing[name] = {"fixed_seconds": row["fixed_loss_wall_seconds"],
                        "tofu_seconds": row["evaluation_wall_seconds"] - row["fixed_loss_wall_seconds"],
                        "native_total_seconds": row["evaluation_wall_seconds"],
                        "barrier_total_seconds": receipt["barrier_total_seconds"]}
    result = {"passed": fixed["passed"] and native["passed"] and all(v["absolute_error"] <= 1e-12 for v in metrics.values())
                         and all(r["state_unchanged_all_ranks"] for r in receipts.values()),
              "smoke": a["smoke"], "checkpoint_parameter_sha256": a["checkpoint_parameter_sha256"],
              "fixed_samples": len(fixed_a), "fixed_comparison": fixed, "native_comparison": native,
              "metrics": metrics, "metric_absolute_tolerance": 1e-12, "timing": timing,
              "native_sample_counts": {key: len(value["value_by_index"]) for key, value in native_a.items()
                                       if "value_by_index" in value},
              "speedup": {key: timing["legacy"][key] / timing["sharded"][key] for key in timing["legacy"]},
              "limits": "One checkpoint and one timing per backend; standalone same ZeRO setup, optimizer not resumed; full training throughput is not measured."}
    write(output / "comparison.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


@contextmanager
def smoke_datasets(enabled):
    if not enabled:
        yield
        return
    from evals.metrics.base import UnlearningMetric
    from torch.utils.data import Subset
    original = UnlearningMetric.get_datasets
    def limited(instance, *args, **kwargs):
        data = original(instance, *args, **kwargs)
        limit = lambda value: Subset(value, range(min(65, len(value))))
        return {key: limit(value) for key, value in data.items()} if isinstance(data, dict) else limit(data)
    UnlearningMetric.get_datasets = limited
    try:
        yield
    finally:
        UnlearningMetric.get_datasets = original


def run(args):
    import torch
    import layout_token_training as training
    import gradient_equivalence_step as gradient
    from online_epoch_evaluation import evaluate_probe, runtime_fingerprint
    from sharded_epoch_evaluation import gather
    rank = int(os.environ.get("RANK", "0"))
    destination = args.output / args.backend
    if rank == 0:
        destination.mkdir(parents=True, exist_ok=False)
        provenance = read(args.checkpoint / "checkpoint_provenance.json")
        for name, spec in provenance["files"].items():
            path = args.checkpoint / name
            if path.stat().st_size != spec["bytes"] or gradient.file_sha(path) != spec["sha256"]:
                raise ValueError("Checkpoint inventory differs: " + name)
    config_path, schedule_path = (args.study / "frozen" / name for name in ("config.json", "schedule.pt"))
    config = read(config_path)
    frozen = read(args.study / "frozen/manifest.json")
    for path in (config_path, schedule_path):
        if gradient.file_sha(path) != frozen["frozen_file_sha256"][path.name]:
            raise ValueError("Frozen input changed: " + path.name)
    schedule = torch.load(schedule_path, map_location="cpu", weights_only=True)
    if args.smoke:
        schedule["samples"] = {split: samples[:7 if split == "forget" else 11]
                               for split, samples in schedule["samples"].items()}
    runtime = training.setup(config, "four", 103, torch, student_checkpoint=str(args.checkpoint))
    provenance = read(args.checkpoint / "checkpoint_provenance.json")
    student_hash = gradient.parameter_hash(runtime["model"], torch, rank, True)
    teacher_hash = gradient.parameter_hash(runtime["reference"].module, torch, rank, True)
    if student_hash != provenance["parameter_sha256"]:
        raise AssertionError("Loaded student does not match saved checkpoint")
    if teacher_hash != provenance["initial_parameter_sha256"]:
        raise AssertionError("Teacher is not the original frozen Full model")
    before = runtime_fingerprint(runtime, torch)
    torch.distributed.barrier()
    started = time.perf_counter()
    with smoke_datasets(args.smoke):
        probe, row, stats = evaluate_probe(config, schedule, runtime, destination, torch,
            backend=args.backend, step=provenance["step"], epoch=provenance["weight_update_epochs"],
            epoch_index=provenance["source_epoch_index"],
            expected_fixed_hash=None if args.smoke else schedule["metadata"]["fixed_dataset_sha256"])
    torch.cuda.synchronize()
    torch.distributed.barrier()
    elapsed = time.perf_counter() - started
    after = runtime_fingerprint(runtime, torch)
    stats["changed_state_sections"] = [key for key in before if before[key] != after[key]]
    all_stats = gather(stats, torch)
    if rank == 0:
        paths = ["scripts/palu/" + name + ".py" for name in ("benchmark_sharded_evaluation",
            "sharded_epoch_evaluation", "online_epoch_evaluation", "layout_token_training", "gradient_equivalence_step")]
        paths += [str(path.relative_to(ROOT)) for path in sorted((ROOT / "src/evals").rglob("*.py"))]
        paths += ["src/trainer/" + name + ".py" for name in ("palu_epoch_eval", "palu_diagnostics", "unlearn/palu", "unlearn/grad_diff")]
        receipt = {"backend": args.backend, "smoke": args.smoke, "checkpoint": str(args.checkpoint),
            "checkpoint_parameter_sha256": student_hash, "teacher_parameter_sha256": teacher_hash,
            "config_sha256": gradient.file_sha(config_path), "schedule_sha256": gradient.file_sha(schedule_path),
            "world_size": runtime["world"], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "barrier_total_seconds": elapsed, "row": row, "ranks": all_stats,
            "state_unchanged_all_ranks": all(not item["changed_state_sections"] for item in all_stats),
            "completed_unix": time.time(), "loss_atol": 1e-6, "loss_rtol": 1e-6, "metric_atol": 1e-12}
        receipt.update(source_sha256={name: gradient.file_sha(ROOT / name) for name in paths},
            software={name: importlib.metadata.version(name) for name in ("torch", "transformers", "deepspeed", "accelerate", "numpy", "scipy")},
            gpu_inventory=subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], text=True).splitlines())
        write(destination / "benchmark.json", receipt)
        print(json.dumps({"event": "benchmark_complete", **receipt}), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("legacy", "sharded", "compare"), required=True)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.backend == "compare":
        compare(args.output)
    else:
        if not args.study or not args.checkpoint:
            parser.error("Evaluation requires --study and --checkpoint")
        run(args)
