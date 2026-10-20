#!/usr/bin/env python3
"""Prepare only F + 2 common-KL, stopping at a fixed F200 loss threshold.

Dry-run needs only the standard library. Training leaves a verified model and
full optimizer/RNG restart state; no retain-stage objective is selected here.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from layout_loss_curve_experiment import read_json, read_rows, write_json, sha


def probe_steps(plan):
    """Observe epoch boundaries only after the next completed optimizer update.

    Plans saved before probe_interval was introduced retain their original
    every-update schedule. Always inspect the budget endpoint.
    """
    total = plan["total_updates"]
    interval = plan.get("probe_interval", "update")
    if interval == "update":
        result = list(range(1, total + 1))
    elif interval == "epoch":
        batch, size = plan["effective_batch"], plan["phase_epoch_size"]
        result = sorted({math.ceil(exposure/batch)
                         for exposure in range(size, total*batch + 1, size)}
                        | {total})
    else:
        raise ValueError("probe_interval must be epoch or update")
    if "probe_steps" in plan and plan["probe_steps"] != result:
        raise ValueError("Frozen probe_steps differ from the declared interval and budget")
    return result


def protocol(seed=103, threshold_low=.4, threshold_high=.5, max_updates=63,
             learning_rate=1e-5, probe_interval="epoch", threshold_comparison=None,
             zero_weight_retain_no_grad=False):
    if type(zero_weight_retain_no_grad) is not bool:
        raise ValueError("zero_weight_retain_no_grad must be boolean")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates <= 0:
        raise ValueError("Maximum updates must be a positive integer")
    if (not all(math.isfinite(v) for v in (threshold_low, threshold_high, learning_rate))
            or not 0 <= threshold_low < threshold_high or learning_rate <= 0):
        raise ValueError("Require finite 0 <= lower < upper threshold and positive learning rate")
    if probe_interval not in ("epoch", "update"):
        raise ValueError("probe_interval must be epoch or update")
    comparison = (("lt" if probe_interval == "epoch" else "le")
                  if threshold_comparison is None else threshold_comparison)
    if comparison not in ("lt", "le"):
        raise ValueError("threshold_comparison must be lt or le")
    sign = "<" if comparison == "lt" else "<="
    plan = dict(experiment="palu_forget_until_threshold", seed=seed,
        threshold_low=float(threshold_low), threshold_high=float(threshold_high),
        learning_rate=learning_rate, scheduler="constant", warmup_steps=0,
        max_updates=max_updates, total_updates=max_updates, forget_updates=max_updates,
        retain_updates=0, phase_epoch_size=200, retain_epoch_size=200,
        phase_weights={"forget": [1., 0., 2.]}, forget_common_kl_weight=2.,
        zero_weight_retain_no_grad=zero_weight_retain_no_grad,
        effective_batch=32, world_size=4, layout="four", microbatch=1,
        accumulation=8, normalization="global", padding_scope="microbatch",
        max_forget_draws=max_updates*32, max_forget_exposure_epochs=max_updates*32/200,
        probe_interval=probe_interval, threshold_comparison=comparison,
        probe_every_updates=1 if probe_interval == "update" else None, probe_samples=200,
        probe_loss="post-update raw PALU F on fixed F200; full-dataset sensitive-token mean",
        probe_policy=("First complete update at/after each F200 exposure epoch, plus budget endpoint"
                      if probe_interval == "epoch" else "Every complete update"),
        stop_policy=f"First scheduled probe with fixed F {sign} upper threshold; lower threshold classifies overshoot only",
        budget_exhaustion_policy="Save and evaluate the last complete update; mark threshold not reached",
        evaluation_backend="sharded", eval_every_epochs=2,
        evaluation_policy="forget_light_v1",
        evaluation_policy_description="Initial/2-epoch FQ and TR; MU once at Forget stop; fixed F200 F+KL only, reuse same-step probes",
        checkpoint_retention="Delete verified intermediate evaluation weights; preserve stop model and native optimizer/RNG restart",
        next_stage="Undecided: do not execute A, B or C",
        schedule_origin=("historical_seed103_first_60_updates_then_recorded_extension"
                         if seed == 103 else "fresh_independent_torch_generators_for_F_permutation_and_R_replacement"),
        seed_comparison_note="103 is the historical-pair anchor; 104+ use the recorded fresh sampler, not a replay of seed103")
    plan["probe_steps"] = probe_steps(plan)
    return plan


def evaluation_events(plan):
    """Two exposure epochs, rounded UP so observations have no pending gradients."""
    total, batch, size = plan["total_updates"], plan["effective_batch"], plan["phase_epoch_size"]
    targets = {0: 0}
    for exposure in range(2*size, total*batch + 1, 2*size):
        targets[math.ceil(exposure/batch)] = exposure
    targets.setdefault(total, total*batch)
    result = []
    for step, target in sorted(targets.items()):
        consumed = step*batch
        result.append(dict(step=step, consumed_pairs=consumed, updated_pairs=consumed,
            pending_pairs=0, actual_data_epochs=consumed/size,
            weight_update_epochs=consumed/size, completed_data_epochs=consumed//size,
            source_epoch_index=math.ceil(consumed/size),
            evaluation_kind="initial" if step == 0 else "budget_end" if step == total else "epoch_end",
            training_phase="forget", phase_consumed_pairs=consumed,
            phase_epoch=consumed/size, phase_epoch_size=size,
            phase_boundary=step == total, nominal_boundary_pairs=target,
            rounded_up_pairs=consumed-target))
    return result


def prepare_config(original, source, study, plan):
    from seed103_token_equal import relocate
    config = copy.deepcopy(original)
    method = config["trainer"]["method_args"]
    expected = dict(alpha=2, gamma=1., top_k=5000, first_n=3, target_mode="mean",
                    retain_loss_type="NLL", non_sensitive_constraint_type="kl")
    if any(method.get(k) != value for k, value in expected.items()):
        raise ValueError("Expected the validated seed103 PALU method")
    if config["model"]["handler"] != "Llama-3.1-8B-Instruct" or config["trainer"]["args"]["seed"] != 103:
        raise ValueError("Source must be the original Llama-3.1-8B seed103 study")
    config = relocate(config, str(source), str(study))
    # This explicit controller plan is the only source of the active [1,0,2] weights.
    config["forget_threshold_plan"] = copy.deepcopy(plan)
    config["trainer"]["args"].update(learning_rate=plan["learning_rate"],
        lr_scheduler_type="constant", warmup_steps=0, warmup_epochs=0, warmup_ratio=0,
        max_steps=plan["total_updates"], seed=plan["seed"], data_seed=plan["seed"],
        output_dir=str(study / "run"), logging_dir=str(study / "logs"),
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        deepspeed=str(ROOT / "configs/accelerate/zero_stage3_offload_config.json"))
    config["eval"]["tofu"]["retain_logs_path"] = str(study / "frozen/retain_reference.json")
    return config


def pair_records(original, plan, torch):
    """Historical anchor for 103; fresh, independently seeded streams for others."""
    if plan["seed"] == 103:
        from sequential_forget_retain import extend_records
        prefix = [p for w in original["windows"] for p in w["pair_records"]]
        return extend_records(prefix, plan["total_updates"]*32)
    from prepare_layout_schedule import pair_stream
    records = pair_stream(200, 3800, plan["total_updates"], 32, plan["seed"], torch)
    for record in records:
        record["pair_origin"] = f"fresh_torch_generators_seed{plan['seed']}"
    return records


def storage_budget(model_bytes):
    # Native ZeRO restart ~= BF16 weights + FP32 master + two FP32 moments,
    # plus one separately readable BF16 stop model and one temporary eval model.
    return max(160*(1 << 30), math.ceil(9*model_bytes*1.10 + 8*(1 << 30)))


def code_paths():
    from sequential_forget_retain import code_paths as existing
    return sorted(set(existing() + [ROOT / "scripts/palu" / name for name in
        ("forget_until_threshold.py", "forget_threshold_control.py", "forget_checkpoint_state.py",
         "forget_light_evaluation.py", "run_forget_until_threshold.sh", "README_forget_until_threshold.md")]))


def prepare(study, source, plan):
    if study.exists():
        raise ValueError("Choose a fresh study directory; existing results are never overwritten")
    import torch
    from omegaconf import OmegaConf
    from model import get_tokenizer
    from data import get_collators
    from prepare_layout_schedule import build_schedule

    old = source / "frozen"
    provenance = read_json(old / "manifest.json")
    if not {"config.json", "schedule.pt", "retain_reference.json"}.issubset(provenance["frozen_file_sha256"]):
        raise ValueError("Source lacks required frozen input hashes")
    for name, digest in provenance["frozen_file_sha256"].items():
        if sha(old / name) != digest:
            raise ValueError("Changed frozen source: " + name)
    config = prepare_config(read_json(old / "config.json"), source, study, plan)
    original = torch.load(old / "schedule.pt", map_location="cpu", weights_only=True)
    if len(original["windows"]) != 60 or original["metadata"]["data_sizes"] != {"forget": 200, "retain": 3800}:
        raise ValueError("Source must contain the original 60 complete seed103 updates and F200/R3800")
    model = Path(config["model"]["model_args"]["pretrained_model_name_or_path"])
    index = model / "model.safetensors.index.json"
    weights = [model / n for n in set(read_json(index)["weight_map"].values())] if index.exists() else [model / "model.safetensors"]
    if not weights or any(not p.is_file() for p in weights):
        raise ValueError("Original Full model weights missing: " + str(model))
    budget = storage_budget(sum(path.stat().st_size for path in weights))
    study.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(study.parent).free < budget:
        raise ValueError(f"Need approximately {budget/(1<<30):.1f} GiB free for model and native Adam restart")
    tokenizer = get_tokenizer(OmegaConf.create(config["model"]["tokenizer_args"]))
    collator = get_collators(OmegaConf.create(config["collator"]), tokenizer=tokenizer)
    records = pair_records(original, plan, torch)
    payload = build_schedule(original["samples"], collator, eos=tokenizer.eos_token_id,
        first_n=3, updates=plan["total_updates"], effective_batch=32,
        seed=plan["seed"], torch=torch, records=records)
    if plan["seed"] == 103:
        for current, old_window in zip(payload["windows"], original["windows"]):
            if current["window_sha256"] != old_window["window_sha256"]:
                raise ValueError("Historical seed103 prefix changed during extension")
    if payload["metadata"]["fixed_dataset_sha256"] != original["metadata"]["fixed_dataset_sha256"]:
        raise ValueError("Fixed F/R sample pool changed")
    frozen = study / "frozen"
    frozen.mkdir(parents=True)
    write_json(frozen / "config.json", config)
    shutil.copyfile(old / "retain_reference.json", frozen / "retain_reference.json")
    events = evaluation_events(plan)
    payload["metadata"].update(model_path=str(model), source_config_sha256=sha(frozen / "config.json"),
        source_study=str(source), source_seed=103, schedule_origin=plan["schedule_origin"],
        selection=("Recorded seed103 pair prefix, followed by its deterministic extension"
                   if plan["seed"] == 103 else "Independent Torch CPU generators seeded by this run; F randperm epochs, R randint replacement"),
        not_a_reconstruction_of_historical_unrecorded_pairs=plan["seed"] != 103,
        historical_prefix_preserved_updates=min(60, plan["total_updates"]) if plan["seed"] == 103 else 0,
        seed103_source_provenance=original["metadata"]["seed103_source_provenance"],
        evaluation_policy=plan["evaluation_policy"], actual_epoch_events=events,
        evaluation_steps=[e["step"] for e in events], epoch_evaluation_schedule=events,
        forget_threshold_plan=copy.deepcopy(plan), possible_early_stop=True)
    torch.save(payload, frozen / "schedule.pt")
    write_json(frozen / "epoch_schedule.json", events)
    (frozen / "pair_schedule.jsonl").write_text("".join(json.dumps({"step": w["step"], **p})+"\n"
        for w in payload["windows"] for p in w["pair_records"]))
    metadata = copy.deepcopy(payload["metadata"])
    metadata["frozen_file_sha256"] = {name: sha(frozen/name) for name in
        ("config.json", "schedule.pt", "retain_reference.json", "epoch_schedule.json", "pair_schedule.jsonl")}
    metadata["model_weight_files"] = {str(path): path.stat().st_size for path in weights}
    write_json(frozen / "manifest.json", metadata)
    paths = code_paths()
    for path in paths:
        destination = study / "code_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    write_json(study / "code_sha256.json", {str(path.relative_to(ROOT)): sha(path) for path in paths})
    model_bytes = sum(path.stat().st_size for path in weights)
    result = dict(plan, status="prepared", source_study=str(source),
        source_frozen_sha256=provenance["frozen_file_sha256"], required_free_bytes=budget,
        estimated_persistent_checkpoint_bytes=8*model_bytes,
        planned_unique_retain_examples=len({p["retain_index"] for p in records}),
        evaluation_steps=[e["step"] for e in events], cuda_initialized=torch.cuda.is_initialized())
    write_json(study / "experiment.json", result)
    print(json.dumps(result), flush=True)


def frozen_plan(study):
    path = study / "frozen/config.json"
    if sha(path) != read_json(study / "frozen/manifest.json")["frozen_file_sha256"]["config.json"]:
        raise ValueError("Frozen configuration changed")
    plan = read_json(path)["forget_threshold_plan"]
    visible = read_json(study / "experiment.json")
    if any(visible.get(key) != value for key, value in plan.items()):
        raise ValueError("Experiment description differs from frozen threshold plan")
    probe_steps(plan)
    if type(plan.get("zero_weight_retain_no_grad", False)) is not bool:
        raise ValueError("Frozen zero_weight_retain_no_grad must be boolean")
    if plan.get("threshold_comparison", "le") not in ("lt", "le"):
        raise ValueError("Frozen threshold_comparison must be lt or le")
    return plan


def train(study):
    import torch
    import layout_token_training as base
    from forget_threshold_control import ForgetThresholdControl
    for name, digest in read_json(study / "frozen/manifest.json")["frozen_file_sha256"].items():
        if sha(study / "frozen" / name) != digest:
            raise ValueError("Changed frozen input: " + name)
    for name, digest in read_json(study / "code_sha256.json").items():
        if sha(ROOT / name) != digest:
            raise ValueError("Prepared code changed: " + name)
    plan = frozen_plan(study)
    args = argparse.Namespace(config=study/"frozen/config.json", schedule=study/"frozen/schedule.pt",
        output=study/"run", max_updates=None, layout="four", seed=plan["seed"],
        normalization="global", padding_scope="microbatch", online_evaluation=True,
        delete_evaluated_intermediate=True, eval_every_epochs=2, eval_backend="sharded")
    try:
        base.train(args, torch, ForgetThresholdControl(plan))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def analyze(study, validate_completed=True):
    frozen_plan(study)
    if validate_completed:
        from forget_threshold_control import validate
        validate(study)
    from plot_sequential_phases import analyze as export
    report = export(study, plots=False)
    decision = study / "run/stop_decision.json"
    if decision.exists():
        stop = read_json(decision)
        report.update(status="forget_stage_complete_awaiting_retention_choice" if validate_completed else "stop_observation_available",
                      stop_decision=stop, planned_retain_updates=0)
    report["forget_to_retain_switch_step"] = None
    write_json(study / "analysis/analysis_manifest.json", report)
    (study / "analysis/README.md").write_text(
        "# Forget-only threshold experiment\n\n"
        "This run does not enter any retain-training stage. `stop_decision.json` records "
        "threshold hit, overshoot or exhausted update budget; `completion_validation.json` "
        "checks the saved model and native state inventory, not a GPU save/restore round trip. "
        "A budget stop is not a threshold hit.\n\n"
        "`evaluation_summary.csv` contains native FQ/MU and measured fixed losses. "
        "For forget_light_v1, fixed R is unmeasured and MU is measured only at the Forget endpoint; "
        "missing measurements remain null/blank, never zero. "
        "`truth_ratio_samples.csv` and `ecdf_points.csv` preserve all evaluated sample IDs. "
        "TR uses geometric wrong-answer probability divided by correct-answer probability "
        "plus 1e-10. `D_low`/`D_high` describe empirical CDF differences, not proof of deletion.\n\n"
        "The scheduled fixed F200 threshold probes are separate from online minibatch losses. "
        "Light evaluation reuses same-step F/KL probes and preserves their per-example records. "
        "The frozen plan records probe_interval, probe_steps and the threshold comparison; "
        "old plans without these fields retain their every-update, <= behavior. "
        "The lower boundary is only an overshoot diagnostic, not an additional stopping condition.\n")
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", required=True, choices=("dry-run", "prepare", "train", "analyze", "export"))
    result.add_argument("--study", required=True, type=Path)
    result.add_argument("--source-study", type=Path)
    # Defaults apply only to prepare/dry-run. Supplying controls to train/analyze
    # fails explicitly rather than silently overriding or ignoring a frozen plan.
    result.add_argument("--seed", type=int)
    result.add_argument("--threshold-low", type=float)
    result.add_argument("--threshold-high", type=float)
    result.add_argument("--max-updates", type=int)
    result.add_argument("--learning-rate", type=float)
    result.add_argument("--probe-interval", choices=("epoch", "update"))
    result.add_argument("--threshold-comparison", choices=("lt", "le"))
    result.add_argument("--zero-weight-retain-no-grad", action="store_true", default=None,
                        help="Explicit future-plan opt-in after GPU equivalence validation; never changes an existing plan")
    return result


def main():
    p = parser()
    args = p.parse_args()
    controls = {name: getattr(args, name) for name in
                ("seed", "threshold_low", "threshold_high", "max_updates", "learning_rate",
                 "probe_interval", "threshold_comparison", "zero_weight_retain_no_grad")
                if getattr(args, name) is not None}
    if args.mode not in ("prepare", "dry-run") and (controls or args.source_study is not None):
        p.error("train/analyze/export use the frozen plan; do not pass preparation settings")
    study = args.study.resolve()
    if args.mode == "dry-run":
        plan = protocol(**controls)
        estimate = dict(assumed_bf16_model_bytes=16_000_000_000,
            per_seed_persistent_checkpoint_bytes=128_000_000_000,
            per_seed_required_free_bytes=storage_budget(16_000_000_000),
            note="Approximate 8B model estimate; prepare checks actual Full weight bytes. Native Adam restart dominates storage.")
        print(json.dumps(dict(plan, storage_estimate=estimate,
                              evaluation_events=evaluation_events(plan)), indent=2))
    elif args.mode == "prepare":
        if args.source_study is None:
            p.error("prepare requires --source-study")
        prepare(study, args.source_study.resolve(), protocol(**controls))
    elif args.mode == "train":
        train(study)
    else:
        print(json.dumps(analyze(study, validate_completed=args.mode == "analyze"), indent=2))


if __name__ == "__main__":
    main()
