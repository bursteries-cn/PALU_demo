#!/usr/bin/env python3
"""Run the unchanged training entry point and record process timing/GPU peaks."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    sys.argv = [str(ROOT / "src/train.py"), *overrides]
    cuda = torch.cuda.is_available()
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    # Select the rank's device before measuring memory or loading either model.
    if cuda:
        torch.cuda.set_device(local_rank)
    telemetry = {
        "status": "started",
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "accelerate", "datasets")},
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "timing_scope": "training entry point including setup and final save; evaluation disabled",
    }
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        telemetry["gpu_name"] = torch.cuda.get_device_name()
    started = time.perf_counter()
    try:
        runpy.run_path(str(ROOT / "src/train.py"), run_name="__main__")
        telemetry["status"] = "completed"
    except BaseException as exc:
        telemetry["status"] = "failed"
        telemetry["error_type"] = type(exc).__name__
        raise
    finally:
        if cuda:
            torch.cuda.synchronize()
            telemetry["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            telemetry["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        telemetry["entrypoint_wall_seconds"] = time.perf_counter() - started
        output = args.telemetry
        if world_size > 1:
            output = output.with_name(f"{output.stem}.rank{rank}{output.suffix}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(telemetry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
