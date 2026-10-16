#!/usr/bin/env python3
"""Prepare and compare three PALU loss reductions using the actual seed103 pairs.

Dry-run, trace validation and CSV analysis use only the Python standard library.
Preparation tokenizes on CPU; each epoch is evaluated before training continues.
"""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "configs/analysis/layout_loss_seed103"
sys.path.insert(0, str(Path(__file__).resolve().parent))

ARMS = (
    {"name": "old_single", "label": "Old single: b8 x GA4, microbatch means", "layout": "single",
     "normalization": "native", "padding_scope": "microbatch", "world_size": 1, "microbatch": 8, "accumulation": 4},
    {"name": "new_single", "label": "New single: b8 x GA4, update token mean", "layout": "single",
     "normalization": "global", "padding_scope": "update", "world_size": 1, "microbatch": 8, "accumulation": 4},
    {"name": "four_native", "label": "Four GPUs: b1 x GA8, microbatch means", "layout": "four",
     "normalization": "native", "padding_scope": "microbatch", "world_size": 4, "microbatch": 1, "accumulation": 8},
)
MEASURES = ("raw_sensitive_loss", "raw_retain_nll", "raw_common_kl", "loss_total",
            "forget_quality", "model_utility", "ks_d")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_assets(assets=ASSETS):
    provenance = read_json(assets / "provenance.json")
    for name, expected in provenance["source_files"].items():
        if sha(assets / name) != expected:
            raise ValueError(f"Frozen seed103 source changed: {name}")
    config = read_json(assets / "resolved_config.json")
    if config["model"]["handler"] != "Llama-3.1-8B-Instruct":
        raise ValueError("This launcher targets the Llama3.1 seed103 experiment")
    required = {"alpha": 2, "gamma": 1.0, "top_k": 5000, "first_n": 3, "target_mode": "mean",
                "retain_loss_type": "NLL", "non_sensitive_constraint_type": "kl"}
    for key, expected in required.items():
        if config["trainer"]["method_args"].get(key) != expected:
            raise ValueError(f"Seed103 method mismatch: {key}")
    required = {"seed": 103, "data_seed": 103, "learning_rate": 2e-5, "warmup_steps": 6,
                "lr_scheduler_type": "linear", "optim": "paged_adamw_32bit", "weight_decay": 0.01,
                "max_grad_norm": 1.0, "bf16": True, "gradient_checkpointing": True,
                "per_device_train_batch_size": 8, "gradient_accumulation_steps": 4,
                "num_train_epochs": 10, "dataloader_num_workers": 0, "dataloader_drop_last": False}
    for key, expected in required.items():
        if config["trainer"]["args"].get(key) != expected:
            raise ValueError(f"Seed103 TrainingArguments mismatch: {key}")
    pairs = read_rows(assets / "training_pairs.jsonl")
    history = read_rows(assets / "epoch_history.jsonl")
    if len(pairs) != 1920:
        raise ValueError("Expected 1920 consumed pairs, not prefetched pairs")
    for index, pair in enumerate(pairs):
        if pair["optimizer_step_before_update"] != index // 32 or pair["example_in_microbatch"] != index % 8:
            raise ValueError("Recorded pair order no longer matches 60 x (8 x 4)")
        if pair["epoch_index"] != index // 200 + 1 or pair["microbatch_in_epoch"] != index % 200 // 8 + 1:
            raise ValueError("Recorded source epoch/microbatch identity is inconsistent")
        if not 0 <= pair["forget_index"] < 200 or not 0 <= pair["retain_index"] < 3800:
            raise ValueError("Recorded pair has an invalid source index")
    for start in range(0, 1800, 200):
        if sorted(row["forget_index"] for row in pairs[start:start + 200]) != list(range(200)):
            raise ValueError("Source full Forget traversal is not a permutation")
    if [row["step"] for row in history] != [0, 6, 12, 18, 25, 31, 37, 43, 50, 56, 60]:
        raise ValueError("Unexpected source epoch checkpoint steps")
    if any(row["fixed_dataset_sha256"] != provenance["fixed_dataset_sha256"] or
           row["reference_sha256"] != provenance["reference_sha256"] for row in history):
        raise ValueError("Historical measurement/reference changed within the source run")
    return config, pairs, history, provenance


def replay_records(pairs):
    return [{"draw_index": index + 1, "epoch_index": pair["epoch_index"],
             "index_in_epoch": index % 200, "forget_index": pair["forget_index"],
             "retain_index": pair["retain_index"], "source_microbatch_in_epoch": pair["microbatch_in_epoch"]}
            for index, pair in enumerate(pairs)]


def replay_events(history):
    # HF epoch evaluation can occur with pending accumulated gradients. Its
    # model weights are the last completed update, so compare exactly that
    # update in every arm. Pending gradients are not checkpoint loss inputs.
    return [{"step": row["step"], "completed_epoch": row["step"] * 32 // 200,
             "source_epoch_index": row["epoch_index"], "sample_epoch": row["step"] * 32 / 200,
             "boundary_exposure": row["epoch_index"] * 200 if not row["partial_epoch"] else 1920,
             "updated_exposure": row["step"] * 32,
             "overshoot_examples": row["step"] * 32 - row["epoch_index"] * 200 if not row["partial_epoch"] else 0,
             "final_partial_epoch": row["partial_epoch"]} for row in history if row["step"]]


def plan(args):
    config, pairs, history, provenance = validate_assets(args.assets)
    return {"model": config["model"]["handler"], "seed": 103,
            "method": config["trainer"]["method_args"],
            "learning_rate": 2e-5, "optimizer": "paged_adamw_32bit", "weight_decay": 0.01,
            "max_grad_norm": 1.0, "scheduler": "linear", "warmup_steps": 6,
            "optimizer_updates": 60, "effective_batch": 32, "training_pairs": len(pairs),
            "nominal_num_train_epochs": 10, "actual_updated_data_epochs": 9.6,
            "unique_retain_examples": len({row["retain_index"] for row in pairs}),
            "evaluation_steps": [row["step"] for row in history],
            "measurement": "same fixed F200/R3800; batch1 token-weighted losses; native TOFU FQ, KS D, MU",
            "evaluation_timing": "pause at actual 200-pair epoch edges and final1920; evaluate current in-memory weights, validate observations and delete intermediate weights before next training microbatch; preserve pending gradients and optimizer state",
            "input_replay": "the actual 1920 consumed seed103 F/R pairs, in order; no fresh sampling",
            "checkpoint_retention": "delete each intermediate weight snapshot after verified epoch evaluation; retain all final models and observations",
            "peak_model_snapshots": 3,
            "final_model_snapshots": 3,
            "model_path": str(args.model_path or config["model"]["model_args"]["pretrained_model_name_or_path"]),
            "retain_logs": str(args.retain_logs or config["eval"]["tofu"]["retain_logs_path"]),
            "study": str(args.study.resolve()), "arms": list(ARMS),
            "source_provenance": provenance,
            "limits": ["One selected seed per arm; curve comparison does not estimate seed variance.",
                       "Four-GPU ZeRO3 uses FP32 master parameters; single GPU updates BF16 parameters directly.",
                       "Native padding follows each microbatch; new_single keeps full-update padding.",
                       "Historical seed103 is an additional reference, not a guaranteed bit-identical replay."]}


def replace_strings(value, before, after):
    if isinstance(value, dict):
        return {key: replace_strings(item, before, after) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_strings(item, before, after) for item in value]
    return after if isinstance(value, str) and value == before else value


def existing_parent(path):
    while not path.exists():
        path = path.parent
    return path


def storage_budget(model_bytes):
    # Sequential arms: two previous finals plus the one currently being evaluated.
    return {"peak_weight_bytes": model_bytes * 3, "final_weight_bytes": model_bytes * 3,
            "required_free_bytes": int(model_bytes * 3 * 1.05 + 5 * (1 << 30))}


def prepare(args):
    description = plan(args)
    study = args.study.resolve()
    frozen = study / "frozen"
    if (frozen / "manifest.json").exists():
        metadata = read_json(frozen / "manifest.json")
        if metadata["seed103_source_provenance"] != description["source_provenance"]:
            raise ValueError("Existing study uses different source assets")
        for name, expected in metadata["frozen_file_sha256"].items():
            if sha(frozen / name) != expected:
                raise ValueError(f"Existing frozen study file changed: {name}")
        for name, expected in read_json(study / "code_sha256.json").items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"Code changed since this study was prepared: {name}")
        if Path(metadata["model_path"]).resolve() != Path(description["model_path"]).resolve():
            raise ValueError("Existing frozen study uses a different Full model path")
        print(json.dumps({"status": "reuse_verified_frozen_schedule", "study": str(study)}))
        return
    if study.exists() and any(study.iterdir()):
        raise ValueError("Use a fresh STUDY_DIR; unfinished preparation is preserved")
    model_path = Path(description["model_path"]).resolve()
    reference = Path(description["retain_logs"]).resolve()
    if not (model_path / "config.json").is_file():
        raise ValueError(f"Full model snapshot is missing: {model_path}")
    index_path = model_path / "model.safetensors.index.json"
    weight_paths = ([model_path / name for name in sorted(set(read_json(index_path)["weight_map"].values()))]
                    if index_path.is_file() else [model_path / "model.safetensors"])
    if any(not path.is_file() for path in weight_paths):
        raise ValueError("Full model has missing safetensors shards")
    model_bytes = sum(path.stat().st_size for path in weight_paths)
    budget = storage_budget(model_bytes)
    needed = budget["required_free_bytes"]
    available = shutil.disk_usage(existing_parent(study)).free
    if available < needed:
        raise ValueError(f"Immediate epoch evaluation with intermediate cleanup needs about {needed / (1 << 30):.1f} GiB peak free space; available {available / (1 << 30):.1f} GiB")
    if not reference.is_file() or sha(reference) != description["source_provenance"]["reference_sha256"]:
        raise ValueError("Retain95 FQ reference is missing or differs from seed103")
    for name, expected in description["source_provenance"].get("historical_core_code_sha256", {}).items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Scientific source differs from the seed103 source: {name}")
    required_packages = ("torch", "transformers", "datasets", "accelerate", "deepspeed",
                         "bitsandbytes", "omegaconf", "safetensors", "scipy", "matplotlib")
    packages = {name: importlib.metadata.version(name) for name in required_packages}
    # No model weights or CUDA contexts are loaded during preparation.
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(ROOT / "src"))
    from data import get_collators, get_data
    from model import get_tokenizer
    from prepare_layout_schedule import build_schedule
    from online_epoch_evaluation import epoch_events
    source, pairs, history, _ = validate_assets(args.assets)
    config = copy.deepcopy(source)
    for key in ("model_args", "tokenizer_args"):
        config["model"][key]["pretrained_model_name_or_path"] = str(model_path)
    old_reference = config["eval"]["tofu"]["retain_logs_path"]
    config = replace_strings(config, old_reference, str(frozen / "retain_reference.json"))
    config["trainer"]["args"].update(max_steps=60, adam_beta1=0.9, adam_beta2=0.999,
        adam_epsilon=1e-8, tf32=False, output_dir=str(study), logging_dir=str(study / "logs"))
    config["trainer"]["args"]["deepspeed"] = str(ROOT / "configs/accelerate/zero_stage3_offload_config.json")
    config["eval"]["tofu"]["output_dir"] = str(study / "evaluation_unused")
    config["paths"]["output_dir"] = str(study)
    for key in ("diagnostics", "detach_mean_target"):
        config["trainer"]["method_args"].pop(key, None)
    cfg = OmegaConf.create(config)
    tokenizer = get_tokenizer(cfg.model.tokenizer_args)
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer,
                    template_args=cfg.model.template_args, model_name=cfg.model.handler)["train"]
    if data.anchor != "forget" or len(data.forget) != 200 or len(data.retain) != 3800:
        raise ValueError("Expected Forget05=200 and Retain95=3800, anchor=forget")
    samples = {}
    for split in ("forget", "retain"):
        samples[split] = []
        for index in range(len(getattr(data, split))):
            sample = getattr(data, split)[index]
            if int(sample["index"]) != index:
                raise ValueError("Source dataset indices changed")
            samples[split].append({key: sample[key].detach().cpu().clone() for key in ("input_ids", "labels")})
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    if collator.padding_side != "right":
        raise ValueError("Original native collator uses right padding")
    payload = build_schedule(samples, collator, eos=tokenizer.eos_token_id, first_n=3,
                             updates=60, effective_batch=32, seed=103, torch=torch,
                             records=replay_records(pairs), evaluations=replay_events(history))
    metadata = payload["metadata"]
    if metadata["fixed_dataset_sha256"] != description["source_provenance"]["fixed_dataset_sha256"]:
        raise ValueError("Tokenized fixed F200/R3800 datasets differ from seed103")
    frozen.mkdir(parents=True)
    shutil.copyfile(reference, frozen / "retain_reference.json")
    write_json(frozen / "config.json", config)
    metadata.update(model_path=str(model_path), source_config_sha256=sha(frozen / "config.json"),
        seed103_source_provenance=description["source_provenance"], packages=packages,
        model_weight_files={path.name: path.stat().st_size for path in weight_paths},
        evaluation_policy=description["evaluation_timing"],
        evaluation_steps=description["evaluation_steps"],
        actual_epoch_events=epoch_events(),
        historical_evaluation_difference="Actual consumed-pair epoch edges and pending gradients match source HF timing; all arms evaluate current model without optimizer restart.",
        normalization="Chosen per arm; fixed measurement reduction is shared across all arms.",
        estimated_snapshot_disk_bytes=needed, snapshot_storage_budget=budget)
    torch.save(payload, frozen / "schedule.pt")
    write_json(frozen / "epoch_schedule.json", metadata["epoch_evaluation_schedule"])
    (frozen / "pair_schedule.jsonl").write_text("".join(json.dumps({"step": window["step"], **pair}) + "\n"
        for window in payload["windows"] for pair in window["pair_records"]))
    metadata["frozen_file_sha256"] = {name: sha(frozen / name) for name in
        ("config.json", "schedule.pt", "retain_reference.json", "epoch_schedule.json", "pair_schedule.jsonl")}
    write_json(frozen / "manifest.json", metadata)
    (study / "historical_seed103").mkdir()
    for name in ("epoch_history.jsonl", "loss_history.jsonl", "provenance.json"):
        shutil.copyfile(args.assets / name, study / "historical_seed103" / name)
    description.update(status="prepared", packages=packages, required_snapshot_disk_gib=needed / (1 << 30),
                       peak_weight_gib=budget["peak_weight_bytes"] / (1 << 30),
                       final_weight_gib=budget["final_weight_bytes"] / (1 << 30))
    write_json(study / "experiment.json", description)
    code_paths = [Path(__file__), ROOT / "scripts/palu/run_layout_loss_curves_seed103.sh",
        ROOT / "scripts/palu/layout_token_training.py", ROOT / "scripts/palu/gradient_equivalence_step.py",
        ROOT / "scripts/palu/prepare_layout_schedule.py", ROOT / "scripts/palu/checkpoint_retention.py",
        ROOT / "scripts/palu/online_epoch_evaluation.py",
        ROOT / "src/trainer/palu_epoch_eval.py",
        ROOT / "src/trainer/unlearn/palu.py", ROOT / "src/trainer/unlearn/grad_diff.py",
        ROOT / "src/trainer/palu_diagnostics.py", ROOT / "configs/accelerate/zero_stage3_offload_config.json"]
    for path in code_paths:
        destination = study / "code_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    write_json(study / "code_sha256.json", {str(path.relative_to(ROOT)): sha(path) for path in code_paths})
    print(json.dumps({"status": "prepared", "study": str(study), "pairs": 1920,
        "pair_stream_sha256": metadata["pair_stream_sha256"], "evaluation_steps": metadata["evaluation_steps"],
        "required_snapshot_disk_gib": needed / (1 << 30),
        "peak_weight_gib": budget["peak_weight_bytes"] / (1 << 30),
        "final_weight_gib": budget["final_weight_bytes"] / (1 << 30),
        "cuda_initialized": torch.cuda.is_initialized()}))


def check_gpus(args):
    devices = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(devices) not in (1, 4) or len(set(devices)) != len(devices):
        raise ValueError("Choose one GPU or four distinct GPUs")
    result = subprocess.run(["nvidia-smi", "--id=" + ",".join(devices),
        "--query-gpu=index,uuid,name,memory.free,memory.total", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != len(devices):
        raise ValueError("nvidia-smi did not return every selected GPU")
    for line in lines:
        fields = [item.strip() for item in line.split(",")]
        if float(fields[-2]) < args.min_free_gib * 1024:
            raise ValueError(f"Insufficient GPU memory (need {args.min_free_gib} GiB free): {line}")
    print(result.stdout.strip())


def relative_difference(left, right):
    return 2 * abs(left - right) / max(abs(left) + abs(right), 1e-300)


def collect_results(study, allow_partial=False):
    description = read_json(study / "experiment.json")
    epochs, online, manifests = {}, {}, {}
    frozen_sha = sha(study / "frozen/schedule.pt")
    for arm in ARMS:
        folder = study / arm["name"]
        manifest = read_json(folder / "manifest.json")
        if manifest["status"] != "completed" and not allow_partial:
            raise ValueError(f"Arm not fully trained and evaluated: {arm['name']}")
        if manifest["seed"] != 103 or manifest["schedule_sha256"] != frozen_sha:
            raise ValueError("Arms did not replay the same seed103 schedule")
        if manifest.get("loss_mode") != arm["normalization"] or manifest.get("padding_scope") != arm["padding_scope"]:
            raise ValueError("Arm loss/padding treatment is mislabeled")
        if any(manifest["method"].get(key) != description["method"].get(key) for key in
               ("alpha", "gamma", "top_k", "first_n", "target_mode", "retain_loss_type", "non_sensitive_constraint_type")):
            raise ValueError("Arm changed the seed103 objective settings")
        rows = read_rows(folder / "epoch_history.jsonl")
        if not allow_partial and [row["step"] for row in rows] != description["evaluation_steps"]:
            raise ValueError("Missing an initial/epoch/final state")
        for row in rows:
            if row["fixed_dataset_sha256"] != description["source_provenance"]["fixed_dataset_sha256"] or row["reference_sha256"] != description["source_provenance"]["reference_sha256"]:
                raise ValueError("Fixed loss data or FQ reference differs between arms")
            if any(not math.isfinite(row[key]) for key in MEASURES):
                raise ValueError("Nonfinite fixed loss or metric")
        if manifest.get("online_epoch_evaluation"):
            from online_epoch_evaluation import epoch_events
            expected = epoch_events()
            if len(rows) != len(expected) or any(any(row.get(key) != event[key] for key in
                    ("step", "consumed_pairs", "updated_pairs", "pending_pairs")) or
                    not row.get("training_state_unchanged_all_ranks") for row, event in zip(rows, expected)):
                raise ValueError("Immediate evaluation did not occur at actual epoch edges with preserved training state")
        if manifest.get("checkpoint_retention") == "final_only_after_verified_epoch_evaluation":
            import checkpoint_retention as retention
            by_step = {row["step"]: row for row in rows}
            for checkpoint in manifest["checkpoints"]:
                committed = retention.read_receipt(folder, manifest, checkpoint)
                if committed is None or any(committed[key] != by_step[checkpoint["step"]][key] for key in MEASURES):
                    raise ValueError("Missing or inconsistent committed observations after weight cleanup")
            final = max(manifest["checkpoints"], key=lambda item: item["step"])
            final_path = retention.owned_checkpoint(folder, final["step"])
            for name, inventory in final["files"].items():
                if retention.WEIGHT_FILE.fullmatch(name) and (not (final_path / name).is_file() or (final_path / name).stat().st_size != inventory["bytes"]):
                    raise ValueError("Final model weights are incomplete")
        epochs[arm["name"]] = rows
        online[arm["name"]] = read_rows(folder / "loss_history.jsonl")
        if len(online[arm["name"]]) != 60 or manifest["completed_updates"] != 60:
            raise ValueError("Training did not complete exactly 60 updates")
        manifests[arm["name"]] = manifest
    if len({manifest["initial_parameter_sha256"] for manifest in manifests.values()}) != 1:
        raise ValueError("Arms did not start with identical parameters")
    if len({manifest["teacher_parameter_sha256"] for manifest in manifests.values()}) != 1:
        raise ValueError("Arms did not use identical Full teacher parameters")
    if len({manifest["schedule_metadata"]["pair_stream_sha256"] for manifest in manifests.values()}) != 1:
        raise ValueError("Pair identities/order differ")
    epochs["historical_seed103"] = read_rows(study / "historical_seed103/epoch_history.jsonl")
    return epochs, online, manifests


def analyze(args):
    study = args.study.resolve()
    epochs, online, manifests = collect_results(study)
    output = study / "analysis"
    output.mkdir(exist_ok=True)
    write_csv(output / "fixed_states.csv", [{"arm": name, "step": row["step"], "epoch": row["epoch"],
        **{key: row[key] for key in MEASURES}} for name, rows in epochs.items() for row in rows])
    write_csv(output / "online_states.csv", [{"arm": name, "step": row["step"], "epoch": row["epoch"],
        **{key: row[key] for key in MEASURES[:4]}} for name, rows in online.items() for row in rows])
    names = [arm["name"] for arm in ARMS]
    matching = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for a in epochs[left]:
                for b in epochs[right]:
                    if not a["step"] or not b["step"]:
                        continue
                    f_diff = relative_difference(a["raw_sensitive_loss"], b["raw_sensitive_loss"])
                    r_diff = relative_difference(a["raw_retain_nll"], b["raw_retain_nll"])
                    if f_diff <= 0.1:
                        matching.append({"left": left, "right": right, "left_step": a["step"], "right_step": b["step"],
                            "same_step": a["step"] == b["step"], "f_relative_difference": f_diff, "r_relative_difference": r_diff,
                            "joint_FR_within_10pct": r_diff <= 0.1, "joint_FR_within_5pct": max(f_diff, r_diff) <= 0.05,
                            "joint_FR_within_1pct": max(f_diff, r_diff) <= 0.01,
                            "left_F": a["raw_sensitive_loss"], "right_F": b["raw_sensitive_loss"],
                            "left_R": a["raw_retain_nll"], "right_R": b["raw_retain_nll"],
                            "left_FQ": a["forget_quality"], "right_FQ": b["forget_quality"],
                            "absolute_log10_FQ_gap": abs(math.log10(max(a["forget_quality"], 1e-300)) - math.log10(max(b["forget_quality"], 1e-300))),
                            "left_D": a["ks_d"], "right_D": b["ks_d"]})
    write_csv(output / "near_F_loss_pairs.csv", matching)
    historical = {row["step"]: row for row in epochs["historical_seed103"]}
    bridge = [{"step": row["step"], **{key + "_replay": row[key] for key in MEASURES},
        **{key + "_historical": historical[row["step"]][key] for key in MEASURES},
        **{key + "_relative_difference": relative_difference(row[key], historical[row["step"]][key]) for key in MEASURES[:3]}}
        for row in epochs["old_single"]]
    write_csv(output / "historical_seed103_bridge.csv", bridge)
    summary = {"status": "validated", "fresh_arms": names, "states_per_arm": {name: len(rows) for name, rows in epochs.items()},
        "same_inputs_initial_model_teacher_measurement_reference": True,
        "single_seed_only": True, "near_F_pairs_10pct": len(matching),
        "same_step_joint_FR_pairs_10pct": sum(row["same_step"] and row["joint_FR_within_10pct"] for row in matching),
        "endpoints": {name: {key: rows[-1][key] for key in MEASURES} for name, rows in epochs.items()},
        "interpretation": "Compare fixed-state FQ/F and MU/R. Cross-step matches have different update budgets. No seed-variance or curve-invariance conclusion is automatic.",
        "historical_bridge": "Old-single replay is shown against original seed103 at the same weight-update steps; inspect this bridge before pooling curves."}
    write_json(output / "summary.json", summary)
    if not args.no_plots:
        plot(output, epochs, online)
    print(json.dumps(summary, ensure_ascii=False))


def plot(output, epochs, online):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {arm["name"]: arm["label"] for arm in ARMS}
    labels["historical_seed103"] = "Historical seed103 reference"
    colors = {"old_single": "#0072B2", "new_single": "#D55E00", "four_native": "#009E73", "historical_seed103": "#777777"}
    marker = {"old_single": "o", "new_single": "s", "four_native": "^", "historical_seed103": "x"}
    footnote = "Fixed losses: batch1, token-weighted F200/R3800. One selected seed per arm; no uncertainty estimate.\nLines connect observed states in training order; no smoothing. Four-GPU changes include the ZeRO3 optimizer backend."

    def save(fig, name):
        fig.text(0.02, 0.015, footnote, fontsize=8, va="bottom")
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def line(ax, name, rows, x, y):
        ax.plot([row[x] for row in rows], [max(row[y], 1e-300) if y == "forget_quality" else row[y] for row in rows],
                color=colors[name], marker=marker[name], ms=4, linewidth=1.2,
                linestyle="--" if name == "historical_seed103" else "-", alpha=0.8, label=labels[name])

    for keyset, name, histories in (
        (("raw_sensitive_loss", "raw_retain_nll", "raw_common_kl", "loss_total"), "fixed_loss_curves", epochs),
        (("raw_sensitive_loss", "raw_retain_nll", "raw_common_kl", "loss_total"), "online_loss_curves", online),
        (("forget_quality", "model_utility", "ks_d"), "metric_curves", epochs)):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, key in zip(axes.flat, keyset):
            for arm, rows in histories.items():
                line(ax, arm, rows, "step", key)
            ax.set(xlabel="Completed optimizer update", ylabel=key)
            if key == "forget_quality":
                ax.set_yscale("log")
            ax.grid(alpha=0.2)
        if len(keyset) == 3:
            axes.flat[3].axis("off")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle("Actual pre-update training reductions (different definitions by arm)" if name.startswith("online") else "Common fixed-state measurement")
        save(fig, name)
    for log_x in (False, True):
        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        for ax, (x, y) in zip(axes, (("raw_sensitive_loss", "forget_quality"),
                                    ("raw_retain_nll", "model_utility"), ("raw_sensitive_loss", "ks_d"))):
            for arm, rows in epochs.items():
                line(ax, arm, rows, x, y)
                for row in rows:
                    if row["step"] in (0, 60):
                        ax.annotate(str(row["step"]), (row[x], max(row[y], 1e-300)), fontsize=7)
            if log_x:
                ax.set_xscale("log")
            if y == "forget_quality":
                ax.set_yscale("log")
            ax.set(xlabel=x + " (common fixed measurement)", ylabel=y)
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=7)
        save(fig, "loss_metric_scatter_log_loss" if log_x else "loss_metric_scatter")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "prepare", "check-gpus", "analyze"), required=True)
    parser.add_argument("--study", type=Path, default=ROOT / "artifacts/layout_loss_curves_seed103/preview")
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--retain-logs", type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--min-free-gib", type=float, default=78)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.mode == "dry-run":
        print(json.dumps(plan(args), ensure_ascii=False, indent=2))
    elif args.mode == "prepare":
        prepare(args)
    elif args.mode == "check-gpus":
        check_gpus(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
