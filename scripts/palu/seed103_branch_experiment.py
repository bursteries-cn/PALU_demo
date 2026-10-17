#!/usr/bin/env python3
"""Prepare, validate and plot the four-GPU seed103 step-six branch experiment.

Dry-run and manifest validation need only the Python standard library. Preparation
uses CPU tokenization; this controller never launches a training process itself.
"""
import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout_loss_curve_experiment as source

ROOT, ASSETS = source.ROOT, source.ASSETS
ARMS = (
    {"name": "continue", "gamma": 1.0, "label": "Continue: gamma = 1"},
    {"name": "repair", "gamma": 0.0, "label": "Repair: gamma = 0"},
    {"name": "weak", "gamma": 0.1, "label": "Weak forgetting: gamma = 0.1"},
)
MEASURES = source.MEASURES
read_json, read_rows = source.read_json, source.read_rows
write_json, write_csv, sha = source.write_json, source.write_csv, source.sha


def storage_budget(model_bytes, stage="prefix"):
    """Added free space, excluding the pre-existing Full model and completed arms.

ZeRO restart: BF16 model + FP32 master + two FP32 moments = 14 bytes per
parameter, approximately seven times the BF16 safetensors files. This is an
estimate; stage checks use current free space and completed final inventories.
"""
    if stage not in ("prefix", *(arm["name"] for arm in ARMS)):
        raise ValueError("Unknown storage stage")
    remaining = 3 if stage == "prefix" else 3 - [arm["name"] for arm in ARMS].index(stage)
    fork_bytes = 7 * model_bytes if stage == "prefix" else 0
    added = fork_bytes + remaining * model_bytes
    return {"bf16_model_bytes": model_bytes, "estimated_fork_state_bytes": 7 * model_bytes,
            "remaining_final_models": remaining, "remaining_weight_and_restart_bytes": added,
            "final_weight_bytes": 3 * model_bytes, "estimated_peak_weight_and_restart_bytes": 10 * model_bytes,
            "required_free_bytes": math.ceil(added * 1.10 + 8 * (1 << 30)),
            "estimate": "Restart uses 14 bytes/parameter; allow 10 percent plus 8 GiB for observations and serialization overhead."}


def plan(args):
    from online_epoch_evaluation import epoch_events
    every = getattr(args, "eval_every_epochs", 2)
    backend = getattr(args, "eval_backend", "sharded")
    events = epoch_events(every_epochs=every)
    config, pairs, history, provenance = source.validate_assets(args.assets)
    result = source.plan(args)
    result.update(experiment="seed103_four_gpu_global_token_step6_branches", arms=list(ARMS),
        layout="four", world_size=4, microbatch=1, accumulation=8,
        normalization="global", padding_scope="microbatch", fork_after_updates=6,
        fork_consumed_pairs=192, branch_first_update=7, optimizer_updates_per_path=60,
        prefix_updates=6, suffix_updates_per_arm=54, actual_training_updates_executed=168,
        method_prefix=config["trainer"]["method_args"],
        branch_objective="gamma*F + 2*R + 2*common_KL; gamma=1,0,0.1 after complete update 6",
        fork_policy="Save native DeepSpeed model, Adam moments, FP32 master, scheduler, update counters and per-rank RNG at 192 consumed pairs; each arm restores and validates the same restart state.",
        epoch_one_policy="Fork after update 6 at 192 pairs. Evaluation cadence is independent of this fork; pending gradients remain intact at selected epoch edges.",
        eval_every_epochs=every, evaluation_backend=backend,
        evaluation_steps=[event["step"] for event in events],
        evaluation_timing=f"Initial state plus every {every} actual 200-pair epochs and final1920; evaluate immediately before the next training microbatch, preserve live state, validate all observations, then delete intermediate model weights.",
        checkpoint_retention="Keep three final models and all observations; delete intermediate weights after verified evaluation, and shared restart tensors only after all arms pass result validation.",
        temporary_restart_snapshots=1, final_model_snapshots=3,
        peak_model_snapshots=3,
        storage_estimate="Shared ZeRO restart approximately seven BF16 models plus three final models; dynamically checked from actual Full shard sizes and currently free space.",
        limits=["One selected seed; branches are paired interventions, not independent seed replicates.",
                "All paths use four-GPU ZeRO3 with FP32 master, full-update token normalization and original per-sample padding.",
                "Gamma=0 retains common KL on Forget non-sensitive positions; it is not pure retain-only training.",
                "The selected step-six fork tests one predeclared switch time; it does not estimate timing sensitivity.",
                "Weight cleanup is immediate after verified epoch evaluation; failed or uncommitted artifacts are preserved."])
    result.pop("optimizer_updates", None)
    return result


def code_paths():
    return [ROOT / name for name in (
        "scripts/palu/seed103_branch_experiment.py", "scripts/palu/run_seed103_branches.sh",
        "scripts/palu/seed103_branch_training.py", "scripts/palu/README_seed103_branches.md",
        "scripts/palu/layout_loss_curve_experiment.py", "scripts/palu/layout_token_training.py",
        "scripts/palu/gradient_equivalence_step.py", "scripts/palu/prepare_layout_schedule.py",
        "scripts/palu/checkpoint_retention.py", "scripts/palu/online_epoch_evaluation.py",
        "scripts/palu/sharded_epoch_evaluation.py", "scripts/palu/benchmark_sharded_evaluation.py",
        "scripts/palu/run_evaluation_benchmark.sh",
        "scripts/palu/plot_seed_epoch.py", "src/trainer/palu_epoch_eval.py",
        "src/trainer/unlearn/palu.py", "src/trainer/unlearn/grad_diff.py",
        "src/trainer/palu_diagnostics.py", "src/data/qa.py", "src/data/unlearn.py",
        "src/data/collators.py", "src/trainer/utils.py",
        "src/evals/base.py", "src/evals/tofu.py", "src/evals/metrics/base.py",
        "src/evals/metrics/utils.py", "src/evals/metrics/memorization.py",
        "src/evals/metrics/privacy.py", "src/evals/metrics/utility.py",
        "configs/accelerate/zero_stage3_offload_config.json")]


def check_storage(args):
    study = args.study.resolve()
    metadata = read_json(study / "frozen/manifest.json")
    model_bytes = sum(metadata["model_weight_files"].values())
    budget = storage_budget(model_bytes, args.stage)
    available = shutil.disk_usage(source.existing_parent(study)).free
    status = {"stage": args.stage, "available_free_gib": available / (1 << 30),
              "required_free_gib": budget["required_free_bytes"] / (1 << 30), **budget}
    if available < budget["required_free_bytes"]:
        raise ValueError(f"Stage {args.stage} needs approximately {status['required_free_gib']:.1f} GiB additional free space; available {status['available_free_gib']:.1f} GiB. Existing observations/restart files are preserved.")
    print(json.dumps(status, ensure_ascii=False))
    return status


def prepare(args):
    description = plan(args)
    study, frozen = args.study.resolve(), args.study.resolve() / "frozen"
    if (frozen / "manifest.json").exists():
        metadata = read_json(frozen / "manifest.json")
        if metadata["seed103_source_provenance"] != description["source_provenance"]:
            raise ValueError("Existing study uses different source assets")
        previous = read_json(study / "experiment.json")
        if previous.get("experiment") != description["experiment"]:
            raise ValueError("Existing folder is not this branch experiment; choose a new STUDY_DIR")
        if (previous.get("eval_every_epochs", 1) != description["eval_every_epochs"] or
                previous.get("evaluation_backend", "legacy") != description["evaluation_backend"]):
            raise ValueError("Existing study has a different frozen evaluation protocol; use a new STUDY_DIR")
        for name, expected in metadata["frozen_file_sha256"].items():
            if sha(frozen / name) != expected:
                raise ValueError(f"Existing frozen study file changed: {name}")
        for name, expected in read_json(study / "code_sha256.json").items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"Code changed since preparation: {name}; use a new STUDY_DIR")
        if Path(metadata["model_path"]).resolve() != Path(description["model_path"]).resolve():
            raise ValueError("Existing frozen study uses a different Full model path")
        print(json.dumps({"status": "reuse_verified_frozen_schedule", "study": str(study)}))
        return
    if study.exists() and any(study.iterdir()):
        raise ValueError("Use a fresh STUDY_DIR; incomplete preparation is preserved")
    model_path, reference = Path(description["model_path"]).resolve(), Path(description["retain_logs"]).resolve()
    if not (model_path / "config.json").is_file():
        raise ValueError(f"Full model snapshot missing: {model_path}")
    index_path = model_path / "model.safetensors.index.json"
    weight_paths = ([model_path / name for name in sorted(set(read_json(index_path)["weight_map"].values()))]
                    if index_path.is_file() else [model_path / "model.safetensors"])
    if any(not path.is_file() for path in weight_paths):
        raise ValueError("Full model has missing safetensors shards")
    model_bytes = sum(path.stat().st_size for path in weight_paths)
    budget = storage_budget(model_bytes)
    available = shutil.disk_usage(source.existing_parent(study)).free
    if available < budget["required_free_bytes"]:
        raise ValueError(f"Shared ZeRO fork and three finals need about {budget['required_free_bytes'] / (1 << 30):.1f} GiB free space; available {available / (1 << 30):.1f} GiB")
    if not reference.is_file() or sha(reference) != description["source_provenance"]["reference_sha256"]:
        raise ValueError("Retain95 reference is missing or differs from seed103")
    for name, expected in description["source_provenance"].get("historical_core_code_sha256", {}).items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Scientific source differs from seed103: {name}")
    paths = code_paths()
    if any(not path.is_file() for path in paths):
        raise ValueError("Experiment source bundle incomplete: " + ", ".join(str(p) for p in paths if not p.is_file()))
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "datasets", "accelerate", "deepspeed", "bitsandbytes",
                 "omegaconf", "safetensors", "scipy", "matplotlib")}
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(ROOT / "src"))
    from data import get_collators, get_data
    from model import get_tokenizer
    from prepare_layout_schedule import build_schedule
    from online_epoch_evaluation import epoch_events
    config, pairs, history, _ = source.validate_assets(args.assets)
    config = copy.deepcopy(config)
    for key in ("model_args", "tokenizer_args"):
        config["model"][key]["pretrained_model_name_or_path"] = str(model_path)
    config = source.replace_strings(config, config["eval"]["tofu"]["retain_logs_path"], str(frozen / "retain_reference.json"))
    config["trainer"]["args"].update(max_steps=60, adam_beta1=.9, adam_beta2=.999,
        adam_epsilon=1e-8, tf32=False, output_dir=str(study), logging_dir=str(study / "logs"),
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        deepspeed=str(ROOT / "configs/accelerate/zero_stage3_offload_config.json"))
    config["eval"]["tofu"]["output_dir"] = str(study / "evaluation_unused")
    config["paths"]["output_dir"] = str(study)
    for key in ("diagnostics", "detach_mean_target"):
        config["trainer"]["method_args"].pop(key, None)
    cfg = OmegaConf.create(config)
    tokenizer = get_tokenizer(cfg.model.tokenizer_args)
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer,
                    template_args=cfg.model.template_args, model_name=cfg.model.handler)["train"]
    if data.anchor != "forget" or len(data.forget) != 200 or len(data.retain) != 3800:
        raise ValueError("Expected Forget05=200, Retain95=3800, anchor=forget")
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
        raise ValueError("Original native collator requires right padding")
    payload = build_schedule(samples, collator, eos=tokenizer.eos_token_id, first_n=3,
        updates=60, effective_batch=32, seed=103, torch=torch,
        records=source.replay_records(pairs), evaluations=source.replay_events(history))
    metadata = payload["metadata"]
    if metadata["fixed_dataset_sha256"] != description["source_provenance"]["fixed_dataset_sha256"]:
        raise ValueError("Tokenized fixed F200/R3800 datasets differ from seed103")
    frozen.mkdir(parents=True)
    shutil.copyfile(reference, frozen / "retain_reference.json")
    write_json(frozen / "config.json", config)
    metadata.update(model_path=str(model_path), source_config_sha256=sha(frozen / "config.json"),
        seed103_source_provenance=description["source_provenance"], packages=packages,
        model_weight_files={path.name: path.stat().st_size for path in weight_paths},
        evaluation_policy=description["evaluation_timing"], evaluation_steps=description["evaluation_steps"],
        actual_epoch_events=epoch_events(every_epochs=description["eval_every_epochs"]),
        eval_every_epochs=description["eval_every_epochs"], evaluation_backend=description["evaluation_backend"],
        normalization="global", padding_scope="microbatch",
        fork_after_updates=6, fork_consumed_pairs=192, snapshot_storage_budget=budget,
        full_update_denominators="Separate sensitive, retain and common token totals over each 32-pair update")
    torch.save(payload, frozen / "schedule.pt")
    # Windows retain the source replay metadata; this file describes the
    # actual selected pause points of this newly frozen online protocol.
    write_json(frozen / "epoch_schedule.json", metadata["actual_epoch_events"])
    (frozen / "pair_schedule.jsonl").write_text("".join(json.dumps({"step": window["step"], **pair}) + "\n"
        for window in payload["windows"] for pair in window["pair_records"]), encoding="utf-8")
    metadata["frozen_file_sha256"] = {name: sha(frozen / name) for name in
        ("config.json", "schedule.pt", "retain_reference.json", "epoch_schedule.json", "pair_schedule.jsonl")}
    write_json(frozen / "manifest.json", metadata)
    (study / "historical_seed103").mkdir()
    for name in ("epoch_history.jsonl", "loss_history.jsonl", "provenance.json"):
        shutil.copyfile(args.assets / name, study / "historical_seed103" / name)
    for path in paths:
        destination = study / "code_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    write_json(study / "code_sha256.json", {str(path.relative_to(ROOT)): sha(path) for path in paths})
    description.update(status="prepared", packages=packages,
        required_initial_free_gib=budget["required_free_bytes"] / (1 << 30), storage_budget=budget)
    write_json(study / "experiment.json", description)
    print(json.dumps({"status": "prepared", "study": str(study), "pairs": 1920,
        "pair_stream_sha256": metadata["pair_stream_sha256"],
        "required_initial_free_gib": description["required_initial_free_gib"],
        "cuda_initialized": torch.cuda.is_initialized()}, ensure_ascii=False))


def validate_fork_receipt(study, require_tensors=False):
    receipt_path = study / "fork/receipt.json"
    receipt = read_json(receipt_path)
    if (receipt.get("status") != "ready" or receipt.get("fork_step") != 6 or
            receipt.get("consumed_pairs") != 192 or receipt.get("world_size") != 4):
        raise ValueError("Shared fork must be a ready four-rank state after exactly six complete updates and 192 pairs")
    if receipt.get("schedule_sha256") != sha(study / "frozen/schedule.pt") or receipt.get("config_sha256") != sha(study / "frozen/config.json"):
        raise ValueError("Fork uses a different frozen schedule/config")
    if len(receipt.get("rank_fingerprints", [])) != 4 or not receipt.get("files"):
        raise ValueError("Fork is missing complete rank fingerprints or restart inventory")
    weights = study / "fork_weights"
    if weights.exists():
        for name, expected in receipt["files"].items():
            target = weights / name
            if target.is_symlink() or not target.resolve().is_relative_to(weights.resolve()):
                raise ValueError("Restart inventory points outside the owned fork folder")
            if not target.is_file() or target.stat().st_size != expected["bytes"] or sha(target) != expected["sha256"]:
                raise ValueError(f"Shared fork tensor inventory mismatch: {name}")
    elif require_tensors:
        raise ValueError("Shared fork tensors are missing; cannot restore a new branch")
    elif not (study / "fork/cleanup_receipt.json").is_file():
        raise ValueError("Shared fork tensors are missing without a validated cleanup receipt")
    else:
        cleanup = read_json(study / "fork/cleanup_receipt.json")
        if cleanup.get("fork_receipt_sha256") != sha(receipt_path) or cleanup.get("status") != "deleted_after_validated_branches":
            raise ValueError("Shared fork cleanup receipt differs from the original fork")
    return receipt


COMPATIBILITY_PATCH_FILES = frozenset((
    "scripts/palu/seed103_branch_training.py",
    "scripts/palu/seed103_branch_experiment.py",
    "scripts/palu/run_seed103_branches.sh",
    "scripts/palu/README_seed103_branches.md",
))


def resume_from_fork(args):
    """CPU-only recovery from a loader failure before any suffix was restored.

    Frozen scientific inputs and the complete six-update native fork must match.
    Failed initialization folders are moved intact; trained partial runs cannot
    be reset. Only the four files belonging to the reviewed compatibility patch
    may differ from their preparation snapshots, which are first archived.
    """
    import checkpoint_retention as retention
    study = args.study.resolve()
    description = read_json(study / "experiment.json")
    if description.get("experiment") != "seed103_four_gpu_global_token_step6_branches":
        raise ValueError("Resume needs an existing seed103 branch experiment")
    _, _, _, provenance = source.validate_assets(args.assets)
    metadata = read_json(study / "frozen/manifest.json")
    if getattr(args, "model_path", None) is not None and args.model_path.resolve() != Path(metadata["model_path"]).resolve():
        raise ValueError("Resume cannot change the original Full model path")
    if metadata.get("seed103_source_provenance") != provenance or description.get("source_provenance") != provenance:
        raise ValueError("Resume source provenance differs from the original seed103 assets")
    expected_frozen = {"config.json", "schedule.pt", "retain_reference.json", "epoch_schedule.json", "pair_schedule.jsonl"}
    if set(metadata.get("frozen_file_sha256", {})) != expected_frozen:
        raise ValueError("Resume requires the complete immutable five-file frozen input inventory")
    for name, expected in metadata["frozen_file_sha256"].items():
        if sha(study / "frozen" / name) != expected:
            raise ValueError(f"Frozen scientific input changed: {name}")
    prefix = read_json(study / "prefix/manifest.json")
    if (prefix.get("status") != "prefix_completed" or prefix.get("completed_updates") != 6 or
            prefix.get("consumed_pairs") != 192 or prefix.get("pending_pairs") != 0):
        raise ValueError("Resume requires the committed six-update, 192-pair prefix with no pending gradients")
    fork = validate_fork_receipt(study, require_tensors=True)
    fork_sha = sha(study / "fork/receipt.json")
    if prefix.get("fork_receipt_sha256") != fork_sha:
        raise ValueError("Prefix points to a different native fork receipt")
    for key in ("config_sha256", "schedule_sha256", "initial_parameter_sha256", "teacher_parameter_sha256"):
        if prefix.get(key) != fork.get(key):
            raise ValueError(f"Prefix and native fork disagree: {key}")
    old_codes = read_json(study / "code_sha256.json")
    expected_codes = {str(path.relative_to(ROOT)) for path in code_paths()}
    if set(old_codes) != expected_codes:
        raise ValueError("Frozen code inventory differs from this reviewed experiment bundle")
    changed = []
    for name, expected in old_codes.items():
        snapshot = study / "code_snapshot" / name
        if snapshot.is_symlink() or not snapshot.is_file() or sha(snapshot) != expected:
            raise ValueError(f"Original code snapshot changed: {name}")
        if sha(ROOT / name) != expected:
            if name not in COMPATIBILITY_PATCH_FILES:
                raise ValueError(f"Scientific code changed outside the compatibility patch: {name}")
            changed.append(name)
    candidates = []
    candidate_logs = {}
    statuses = {}
    for arm in ARMS:
        folder = study / arm["name"]
        if not folder.exists():
            statuses[arm["name"]] = "new"
            continue
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Branch output must be a real owned directory")
        manifest = read_json(folder / "manifest.json")
        if manifest.get("status") == "completed" and manifest.get("completed_updates") == 60:
            restore = read_json(folder / "fork_restore_receipt.json")
            if restore.get("status") != "verified" or restore.get("fork_receipt_sha256") != fork_sha:
                raise ValueError("Completed branch has no verified matching fork restore")
            statuses[arm["name"]] = "completed"
            continue
        files = {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()}
        if (any(path.is_symlink() for path in folder.rglob("*")) or
                not files <= {"manifest.json", "resolved_config.json"} or
                manifest.get("status") not in ("training", "failed") or manifest.get("completed_updates") != 0):
            raise ValueError(f"Branch {arm['name']} contains restored/partial training results; preserve it. This mode only resets a loader failure before any suffix restore or training.")
        if any(manifest.get(key) != fork[key] for key in
               ("config_sha256", "schedule_sha256", "initial_parameter_sha256", "teacher_parameter_sha256")):
            raise ValueError("Failed initialization belongs to another model, teacher or frozen schedule")
        logs = [study / "logs" / f"{arm['name']}_{suffix}.log"
                for suffix in ("train", "disk_before", "gpu_before")]
        if any(path.is_symlink() or (path.exists() and not path.is_file()) for path in logs):
            raise ValueError("Failed-load logs must be ordinary files")
        candidate_logs[arm["name"]] = [path for path in logs if path.is_file()]
        candidates.append(folder)
        statuses[arm["name"]] = "failed_initialization_only"
    # Every input, snapshot, restart shard and branch has passed read-only
    # verification before the first mutation. No native state is deserialized.
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + f"_{time.time_ns() % 1000000000:09d}"
    audit = study / "compatibility_audit" / stamp
    audit.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(study / "code_sha256.json", audit / "previous_code_sha256.json")
    shutil.copytree(study / "code_snapshot", audit / "previous_code_snapshot")
    receipt = {"schema": 1, "status": "verified_pending_refresh", "operation": "resume_from_complete_step6_fork",
        "created_unix": time.time(), "fork_receipt_sha256": fork_sha,
        "prefix_manifest_sha256": sha(study / "prefix/manifest.json"),
        "frozen_file_sha256": metadata["frozen_file_sha256"], "old_code_sha256": old_codes,
        "allowed_compatibility_files": sorted(COMPATIBILITY_PATCH_FILES), "changed_files": sorted(changed),
        "branch_states_before": statuses, "quarantined_attempts": [],
        "scope": "CPU metadata/restart-inventory verification and failed-load quarantine only; no model load, GPU initialization or optimizer update"}
    retention.atomic_json(audit / "receipt.json", receipt)
    for folder in candidates:
        destination = study / "failed_load_attempts" / f"{folder.name}_{stamp}"
        destination.parent.mkdir(exist_ok=True)
        folder.rename(destination)
        if candidate_logs[folder.name]:
            (destination / "logs").mkdir()
            for log in candidate_logs[folder.name]:
                shutil.copyfile(log, destination / "logs" / log.name)
        receipt["quarantined_attempts"].append(str(destination.relative_to(study)))
        retention.atomic_json(audit / "receipt.json", receipt)
    current_codes = dict(old_codes)
    for name in sorted(COMPATIBILITY_PATCH_FILES):
        target = study / "code_snapshot" / name
        temporary = target.with_name(target.name + ".compat.tmp")
        shutil.copyfile(ROOT / name, temporary)
        temporary.replace(target)
        current_codes[name] = sha(ROOT / name)
    retention.atomic_json(study / "code_sha256.json", current_codes)
    receipt.update(status="ready_to_resume_from_fork", new_code_sha256=current_codes,
        prefix_and_fork_unmodified=True)
    retention.atomic_json(audit / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "study": str(study), "audit": str(audit),
        "fork_step": 6, "fork_consumed_pairs": 192, "changed_files": receipt["changed_files"],
        "quarantined_attempts": receipt["quarantined_attempts"], "gpu_training_started": False}, ensure_ascii=False))
    return receipt


def collect_results(study):
    study = Path(study).resolve()
    description = read_json(study / "experiment.json")
    if description.get("experiment") != "seed103_four_gpu_global_token_step6_branches":
        raise ValueError("Not a seed103 four-GPU branch study")
    validate_fork_receipt(study)
    fork_sha = sha(study / "fork/receipt.json")
    frozen_sha = sha(study / "frozen/schedule.pt")
    expected_pairs = read_rows(study / "frozen/pair_schedule.jsonl")
    if len(expected_pairs) != 1920:
        raise ValueError("Frozen schedule must contain the actual 1920 consumed F/R pairs")
    epochs, online, manifests = {}, {}, {}
    from online_epoch_evaluation import epoch_events
    expected_events = epoch_events(every_epochs=description.get("eval_every_epochs", 1))
    import checkpoint_retention as retention
    for arm in ARMS:
        folder = study / arm["name"]
        manifest = read_json(folder / "manifest.json")
        if manifest.get("status") != "completed" or manifest.get("completed_updates") != 60:
            raise ValueError(f"Arm not completed at update 60: {arm['name']}")
        if manifest.get("seed") != 103 or manifest.get("schedule_sha256") != frozen_sha:
            raise ValueError("Branches did not replay the same seed103 schedule")
        if manifest.get("loss_mode") != "global" or manifest.get("padding_scope") != "microbatch":
            raise ValueError("Branch loss/padding treatment differs from the preregistered protocol")
        if manifest.get("branch_gamma") != arm["gamma"]:
            raise ValueError("Branch gamma is mislabeled")
        if any(manifest["method"].get(key) != description["method"].get(key) for key in
               ("alpha", "gamma", "top_k", "first_n", "target_mode", "retain_loss_type", "non_sensitive_constraint_type")):
            raise ValueError("Canonical evaluation or other seed103 method settings changed")
        restore = read_json(folder / "fork_restore_receipt.json")
        if (restore.get("status") != "verified" or restore.get("fork_receipt_sha256") != fork_sha or
                restore.get("fork_step") != 6 or restore.get("completed_updates") != 6 or
                restore.get("consumed_pairs") != 192 or restore.get("arm") != arm["name"] or
                restore.get("branch_gamma") != arm["gamma"]):
            raise ValueError("Branch did not restore the exact shared step-six fork")
        guards = restore.get("rank_guards", [])
        if len(guards) != 4 or any(not row.get("unchanged") or row.get("before_sha256") != row.get("after_sha256") for row in guards):
            raise ValueError("Branch restart state did not pass all four rank equivalence guards")
        rows = read_rows(folder / "epoch_history.jsonl")
        if len(rows) != len(expected_events) or [row["step"] for row in rows] != description["evaluation_steps"]:
            raise ValueError("Branch is missing initial, epoch or final observations")
        for row, event in zip(rows, expected_events):
            if (row.get("evaluation_backend", "legacy") != description.get("evaluation_backend", "legacy") or
                    row.get("eval_every_epochs", 1) != description.get("eval_every_epochs", 1)):
                raise ValueError("Evaluation backend or cadence differs from the frozen protocol")
            if any(row.get(key) != event[key] for key in ("step", "consumed_pairs", "updated_pairs", "pending_pairs")):
                raise ValueError("Evaluation did not occur at an actual 200-pair epoch boundary")
            if not row.get("training_state_unchanged_all_ranks"):
                raise ValueError("Evaluation changed live training state")
            if (row["fixed_dataset_sha256"] != description["source_provenance"]["fixed_dataset_sha256"] or
                    row["reference_sha256"] != description["source_provenance"]["reference_sha256"]):
                raise ValueError("Fixed losses or FQ reference changed across branches")
            if any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) for key in MEASURES):
                raise ValueError("Nonfinite or missing fixed losses/metrics")
        if manifest.get("checkpoint_retention") != "final_only_after_verified_epoch_evaluation":
            raise ValueError("Intermediate checkpoint cleanup policy is missing")
        by_step = {row["step"]: row for row in rows}
        if [checkpoint["step"] for checkpoint in manifest.get("checkpoints", [])] != description["evaluation_steps"]:
            raise ValueError("Checkpoint receipt inventory must cover every initial, epoch and final state exactly once")
        for checkpoint in manifest["checkpoints"]:
            committed = retention.read_receipt(folder, manifest, checkpoint)
            if committed is None or any(committed[key] != by_step[checkpoint["step"]][key] for key in MEASURES):
                raise ValueError("Committed per-sample results or checkpoint cleanup receipts are missing")
        final = max(manifest["checkpoints"], key=lambda item: item["step"])
        if final["step"] != 60:
            raise ValueError("Final retained checkpoint is not update 60")
        final_path = retention.owned_checkpoint(folder, 60)
        if not any(retention.WEIGHT_FILE.fullmatch(name) and name.endswith(".safetensors") for name in final["files"]):
            raise ValueError("Final model has no safetensors weights")
        for name, inventory in final["files"].items():
            target = final_path / name
            if retention.WEIGHT_FILE.fullmatch(name) and (not target.is_file() or target.stat().st_size != inventory["bytes"]):
                raise ValueError("Final model weight inventory is incomplete")
        pairs = read_rows(folder / "training_pairs.jsonl")
        if len(pairs) != 1920 or any(any(actual.get(key) != value for key, value in expected.items())
                for actual, expected in zip(pairs, expected_pairs)):
            raise ValueError("Actual consumed F/R sample identities, order or pairing differ from the frozen seed103 schedule")
        history = read_rows(folder / "loss_history.jsonl")
        if len(history) != 60 or [row["step"] for row in history] != list(range(1, 61)):
            raise ValueError("Expected exactly 60 complete update losses per branch")
        for row in history:
            expected_gamma = 1.0 if row["step"] <= 6 else arm["gamma"]
            if row.get("gamma_used") != expected_gamma or row.get("alpha_used") != 2:
                raise ValueError("Online objective gamma did not switch immediately after update six")
            if any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) for key in MEASURES[:4]):
                raise ValueError("Nonfinite online loss")
        epochs[arm["name"]], online[arm["name"]], manifests[arm["name"]] = rows, history, manifest
    for key in ("initial_parameter_sha256", "teacher_parameter_sha256"):
        if len({manifest.get(key) for manifest in manifests.values()}) != 1 or not next(iter(manifests.values())).get(key):
            raise ValueError("Branches differ in their initial/teacher parameters")
    if len({manifest["schedule_metadata"]["pair_stream_sha256"] for manifest in manifests.values()}) != 1:
        raise ValueError("Branch F/R pairs or their order changed")
    for arm in ARMS[1:]:
        if online[arm["name"]][:6] != online["continue"][:6]:
            raise ValueError("Shared prefix update records differ across branches")
        shared = [row for row in epochs["continue"] if row["step"] <= 6]
        other_shared = [row for row in epochs[arm["name"]] if row["step"] <= 6]
        for left, right in zip(shared, other_shared):
            if any(not math.isclose(left[key], right[key], rel_tol=1e-6, abs_tol=1e-12) for key in MEASURES):
                raise ValueError("Identical initial/step-six weights produced inconsistent branch evaluations")
    return epochs, online, manifests


def analyze(args):
    study = args.study.resolve()
    epochs, online, manifests = collect_results(study)
    output = study / "analysis"
    output.mkdir(exist_ok=True)
    fixed_rows, online_rows = [], []
    for arm in ARMS:
        for history, target in ((epochs[arm["name"]], fixed_rows), (online[arm["name"]], online_rows)):
            for row in history:
                target.append({"arm": arm["name"], "branch_gamma": arm["gamma"], **row,
                    "canonical_loss_total": row["raw_sensitive_loss"] + 2 * row["raw_retain_nll"] + 2 * row["raw_common_kl"]})
    write_csv(output / "fixed_states.csv", fixed_rows)
    write_csv(output / "online_states.csv", online_rows)
    comparisons = []
    for i, left in enumerate(ARMS):
        for right in ARMS[i + 1:]:
            for a in epochs[left["name"]]:
                for b in epochs[right["name"]]:
                    if a["step"] <= 6 or b["step"] <= 6:
                        continue
                    f_diff = source.relative_difference(a["raw_sensitive_loss"], b["raw_sensitive_loss"])
                    r_diff = source.relative_difference(a["raw_retain_nll"], b["raw_retain_nll"])
                    if max(f_diff, r_diff) <= .1:
                        comparisons.append({"left": left["name"], "right": right["name"],
                            "left_step": a["step"], "right_step": b["step"], "same_update_budget": a["step"] == b["step"],
                            "F_relative_difference": f_diff, "R_relative_difference": r_diff,
                            "KL_relative_difference": source.relative_difference(a["raw_common_kl"], b["raw_common_kl"]),
                            "joint_FR_within_1pct": max(f_diff, r_diff) <= .01,
                            "joint_FR_within_5pct": max(f_diff, r_diff) <= .05,
                            "left_FQ": a["forget_quality"], "right_FQ": b["forget_quality"],
                            "FQ_log10_gap": abs(math.log10(a["forget_quality"]) - math.log10(b["forget_quality"])) if min(a["forget_quality"], b["forget_quality"]) > 0 else None,
                            "left_MU": a["model_utility"], "right_MU": b["model_utility"],
                            "KS_D_absolute_gap": abs(a["ks_d"] - b["ks_d"])})
    write_csv(output / "matched_FR_pairs.csv", comparisons)
    summary = {"status": "validated", "fork_step": 6, "fork_consumed_pairs": 192,
        "fork_restore_verified_all_arms": True,
        "shared_prefix_observation_steps": [row["step"] for row in epochs["continue"] if row["step"] <= 6],
        "initial_step6_observations_consistent": True if any(row["step"] == 6 for row in epochs["continue"]) else None,
        "same_inputs_initial_model_teacher_measurement_reference": True, "single_seed_only": True,
        "states_per_arm": {name: len(rows) for name, rows in epochs.items()},
        "endpoints": {name: {key: rows[-1][key] for key in MEASURES} for name, rows in epochs.items()},
        "matched_FR_pairs_10pct": len(comparisons),
        "interpretation": "Three paired strategies, one seed and a fixed switch time; no seed variance estimate. FQ is a KS p-value; inspect KS D, TR, per-sample answers and MU jointly."}
    write_json(output / "summary.json", summary)
    if not args.no_plots:
        plot(output, epochs, online)
    if args.cleanup_fork:
        cleanup_fork(study)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def cleanup_fork(study):
    # Only the owned restart tensors are deleted, after collect_results verified
    # every final model, complete epoch observation and restore receipt.
    validate_fork_receipt(study)
    folder = study / "fork_weights"
    if not folder.exists():
        return
    if folder.is_symlink() or folder.resolve().parent != study.resolve():
        raise ValueError("Refusing to remove a fork folder outside this study")
    inventory = read_json(study / "fork/receipt.json")["files"]
    actual = {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()}
    if actual != set(inventory):
        raise ValueError("Unexpected files in restart folder; preserve it for inspection")
    cleanup = {"status": "validated_pending_deletion", "fork_receipt_sha256": sha(study / "fork/receipt.json"),
        "removed_bytes": sum(item["bytes"] for item in inventory.values()), "verified_arms": [arm["name"] for arm in ARMS]}
    # Write a pending receipt first, so an interrupted deletion is inspectable.
    write_json(study / "fork/cleanup_receipt.json", cleanup)
    shutil.rmtree(folder)
    cleanup["status"] = "deleted_after_validated_branches"
    write_json(study / "fork/cleanup_receipt.json", cleanup)


def plot(output, epochs, online):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import plot_seed_epoch as plotting
    plt = plotting.plt
    labels = {arm["name"]: arm["label"] for arm in ARMS}
    colors = {arm["name"]: plotting.COLORS[index] for index, arm in enumerate(ARMS)}
    markers = {arm["name"]: plotting.MARKERS[index] for index, arm in enumerate(ARMS)}
    observed_steps = [row["step"] for row in epochs["continue"]]
    note = ("One shared seed103 prefix; three paired interventions, not independent training seeds. No smoothing or seed SD.\n"
            f"Fork after complete update 6 (192 pairs); measured updates: {', '.join(map(str, observed_steps))}.\n"
            "Fixed losses: token-weighted F200/R3800. FQ is a KS p-value. Zero values stay in CSV and are omitted on log axes.")
    def line(ax, name, rows, xkey, ykey):
        ax.plot([row[xkey] for row in rows],
                [float("nan") if ykey == "forget_quality" and row[ykey] == 0 else row[ykey] for row in rows],
                color=colors[name], marker=markers[name], ms=4, linewidth=1.2, label=labels[name])
    for metrics, filename, histories in (
        (plotting.LOSS_METRICS, "fixed_loss_curves", epochs),
        (plotting.LOSS_METRICS, "online_loss_curves", online),
        (plotting.EVAL_METRICS, "metric_curves", epochs)):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (key, label) in zip(axes.flat, metrics):
            for name, rows in histories.items():
                line(ax, name, rows, "step", key)
            plotting.style_axis(ax, label, "Completed optimizer update", key == "forget_quality")
            ax.axvline(6, color="#777777", linestyle=":", linewidth=.8)
            if key == "forget_quality":
                plotting.format_log_ticks(ax, "y")
        if len(metrics) == 3:
            axes.flat[3].axis("off")
        axes.flat[0].legend(fontsize=8)
        title = "Online training objective (gamma changes after update 6)" if filename.startswith("online") else "Common fixed-state measurement (canonical gamma = 1)"
        fig.suptitle(title)
        plotting.save_figure(fig, output, filename, note)
    for log_loss in (False, True):
        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        for ax, (xkey, ykey, xlabel, ylabel) in zip(axes, (
            ("raw_sensitive_loss", "forget_quality", "Fixed sensitive-token loss", "FQ: KS p-value"),
            ("raw_retain_nll", "model_utility", "Fixed retain NLL", "MU"),
            ("raw_sensitive_loss", "ks_d", "Fixed sensitive-token loss", "Truth-ratio KS D"))):
            for name, history in epochs.items():
                rows = [row for row in history if (not log_loss or row[xkey] > 0)]
                line(ax, name, rows, xkey, ykey)
                for row in rows:
                    if row["step"] in (0, 6 if 6 in observed_steps else observed_steps[1], 60) and (ykey != "forget_quality" or row[ykey] > 0):
                        ax.annotate(str(row["step"]), (row[xkey], row[ykey]), fontsize=6)
            plotting.style_axis(ax, ylabel, xlabel, ykey == "forget_quality")
            if log_loss:
                ax.set_xscale("log")
                plotting.format_log_ticks(ax, "x")
            if ykey == "forget_quality":
                plotting.format_log_ticks(ax, "y")
        axes[0].legend(fontsize=7)
        fig.suptitle("Fixed-data loss versus metric; lines connect observed states in training order")
        plotting.save_figure(fig, output, "loss_metric_scatter_log_loss" if log_loss else "loss_metric_scatter", note)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "prepare", "check-gpus", "check-storage", "resume-from-fork", "analyze"), required=True)
    parser.add_argument("--study", type=Path, default=ROOT / "artifacts/seed103_branches/preview")
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--retain-logs", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--min-free-gib", type=float, default=40)
    parser.add_argument("--stage", choices=("prefix", "continue", "repair", "weak"), default="prefix")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--cleanup-fork", action="store_true")
    parser.add_argument("--eval-every-epochs", type=int, default=2)
    parser.add_argument("--eval-backend", choices=("legacy", "sharded"), default="sharded")
    args = parser.parse_args()
    if args.mode == "dry-run":
        print(json.dumps(plan(args), ensure_ascii=False, indent=2))
    elif args.mode == "prepare":
        prepare(args)
    elif args.mode == "check-gpus":
        if len(args.gpus.split(",")) != 4:
            parser.error("This experiment needs four GPUs")
        if args.min_free_gib <= 0 or not math.isfinite(args.min_free_gib):
            parser.error("min-free-gib must be positive and finite")
        source.check_gpus(args)
    elif args.mode == "check-storage":
        check_storage(args)
    elif args.mode == "resume-from-fork":
        resume_from_fork(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
