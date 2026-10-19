#!/usr/bin/env python3
"""Controlled PALU layout training with optional immediate epoch evaluation.

Train consumes an immutable schedule of complete 32-pair windows. Each F/R/KL
term is normalized by that whole window's target count. Single b8 x GA4 and
ZeRO-3 b1 x GA8 x 4 ranks retain their actual historical optimizer arithmetic.
--online-evaluation pauses at actual 200-pair epoch edges, including inside
gradient accumulation, commits evaluation and prunes weights before continuing.
Without that flag, the earlier separate checkpoint-evaluation mode is available.
--normalization native preserves the original mean of microbatch means instead.
--padding-scope microbatch restores each actual microbatch's original padding.
Core PALU code is not modified. Defaults preserve the whole-update control.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gradient_equivalence_step as gradient


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate", "self-test"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", type=Path, help="Completed training output to evaluate")
    parser.add_argument("--layout", choices=("single", "four"), default="single")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalization", choices=("native", "global"), default="global")
    parser.add_argument("--padding-scope", choices=("microbatch", "update"), default="update")
    parser.add_argument("--online-evaluation", action="store_true",
                        help="Evaluate the current model at actual epoch ends, preserving pending gradients and optimizer state")
    parser.add_argument("--delete-evaluated-intermediate", action="store_true",
                        help="Delete only intermediate weight files after durable, validated per-example evaluation")
    parser.add_argument("--resume-evaluation", action="store_true",
                        help="Reuse verified epoch receipts; reevaluate only unfinished states")
    parser.add_argument("--max-updates", type=int, default=None, help="Smoke-only bounded schedule prefix, recorded explicitly")
    return parser.parse_args()


def append_json(path, payload):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    gradient.write_json(temporary, payload)
    temporary.replace(path)


def loss_record(raw, gamma, alpha, common=None):
    common = alpha if common is None else common
    weighted = [gamma * raw["sensitive"], alpha * raw["retain"], common * raw["common"]]
    return {"loss": sum(weighted), "loss_total": sum(weighted), "loss_forget": weighted[0],
            "loss_retain_nll": weighted[1], "loss_common_kl": weighted[2],
            "raw_sensitive_loss": raw["sensitive"], "raw_retain_nll": raw["retain"],
            "raw_common_kl": raw["common"]}


def microbatch_inputs(window, indices, device, padding_scope):
    """Slice the same frozen examples; trim only documented right padding."""
    result = {}
    for split in ("forget", "retain"):
        width = window[split]["input_ids"].shape[1]
        if padding_scope == "microbatch":
            if "sample_lengths" not in window:
                raise ValueError("Microbatch padding requires a schedule with original sample lengths")
            width = max(window["sample_lengths"][split][index] for index in indices)
        result[split] = {key: window[split][key][indices, :width].to(device)
                         for key in ("input_ids", "attention_mask", "labels")}
    return result


def assert_learning_rate(values, completed_updates, runtime, train_args):
    warmup, total = runtime["warmup_steps"], runtime["scheduler_total_updates"]
    kind = train_args.get("lr_scheduler_type", "linear")
    if kind == "constant":
        if warmup != 0:
            raise ValueError("Constant learning rate requires explicit zero warmup")
        factor = 1.0
    elif kind == "linear":
        factor = completed_updates / max(1, warmup) if completed_updates < warmup else max(0.0, (total - completed_updates) / max(1, total - warmup))
    else:
        raise ValueError(f"Unsupported controlled learning-rate schedule: {kind}")
    expected = float(train_args["learning_rate"]) * factor
    if any(not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-14) for value in values):
        raise AssertionError(f"Learning-rate schedule mismatch after {completed_updates} updates: {values} vs {expected}")


def planned_update_budget(control):
    """Only an explicit experiment controller may replace the historical budget."""
    planned = getattr(control, "planned_updates", 60)
    if isinstance(planned, bool) or not isinstance(planned, int) or planned <= 0:
        raise ValueError("Controller planned_updates must be a positive integer")
    return planned


def objective_coefficients(control, step, method):
    """Set phase weights once per complete update, preserving old gamma hooks."""
    gamma, alpha = float(method["gamma"]), float(method["alpha"])
    if control is not None and hasattr(control, "coefficients_for_step"):
        gamma, alpha, common = map(float, control.coefficients_for_step(step))
    else:
        if control is not None:
            gamma = float(control.gamma_for_step(step))
        common = alpha
    if any(not math.isfinite(value) or value < 0 for value in (gamma, alpha, common)):
        raise ValueError("PALU objective coefficients must be finite and nonnegative")
    return gamma, alpha, common


def optimizer_groups(model, train_args):
    from transformers.trainer_pt_utils import get_parameter_names
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    decay = {name for name in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if "bias" not in name}
    return [{"params": [p for n, p in model.named_parameters() if p.requires_grad and n in decay],
             "weight_decay": float(train_args.get("weight_decay", 0.01))},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and n not in decay],
             "weight_decay": 0.0}]


def make_optimizer(model, train_args):
    import bitsandbytes as bnb
    if train_args.get("optim", "paged_adamw_32bit") != "paged_adamw_32bit":
        raise ValueError("This historical backend comparison requires paged_adamw_32bit")
    return bnb.optim.PagedAdamW32bit(optimizer_groups(model, train_args),
            lr=float(train_args.get("learning_rate", 2e-5)),
            betas=(float(train_args.get("adam_beta1", 0.9)), float(train_args.get("adam_beta2", 0.999))),
            eps=float(train_args.get("adam_epsilon", 1e-8)))


def optimizer_precision(optimizer, torch):
    underlying = getattr(optimizer, "optimizer", optimizer)
    result = {"wrapper_class": type(optimizer).__name__, "underlying_class": type(underlying).__name__,
              "optimizer_parameter_dtypes": sorted({str(p.dtype) for group in underlying.param_groups for p in group["params"]}),
              "state_tensor_dtypes": sorted({str(value.dtype) for state in underlying.state.values()
                                             for value in state.values() if torch.is_tensor(value)})}
    for key in ("fp32_partitioned_groups_flat", "fp16_partitioned_groups_flat", "fp16_groups"):
        values = getattr(optimizer, key, None)
        if values:
            flattened = [item for group in values for item in (group if isinstance(group, list) else [group])]
            result[key + "_dtypes"] = sorted({str(value.dtype) for value in flattened if torch.is_tensor(value)})
    for key in ("gradient_accumulation_dtype", "grad_accum_dtype"):
        result[key] = str(getattr(optimizer, key, None))
    return result


def checkpoint_state(model, torch, rank, distributed):
    """Export read-only canonical BF16 weights from shards; do not invoke hooks."""
    state = {}
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    for name, parameter in model.named_parameters():
        if distributed:
            shard = parameter.ds_tensor.detach().contiguous()
            group = getattr(parameter, "ds_process_group", None)
            full = torch.empty(shard.numel() * torch.distributed.get_world_size(group),
                               dtype=shard.dtype, device=shard.device)
            torch.distributed.all_gather_into_tensor(full, shard, group=group)
            tensor = full[:int(parameter.ds_numel)].view(parameter.ds_shape)
        else:
            tensor = parameter.detach()
        if rank == 0:
            state[name] = tensor.contiguous().cpu().clone()
        if distributed:
            del full, tensor
    # Include persistent state buffers, with the same keys as ordinary HF saves.
    parameter_names = {name for name, _ in model.named_parameters()}
    if rank == 0:
        for name, tensor in model.state_dict().items():
            if name not in parameter_names:
                state[name] = tensor.detach().cpu().clone()
    return state


def save_checkpoint(model, tokenizer, output, step, initial_hash, torch, rank, distributed, window):
    path = output / f"checkpoint-{step}"
    digest = gradient.parameter_hash(model, torch, rank, distributed)
    state = checkpoint_state(model, torch, rank, distributed)
    if rank == 0:
        path.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(path, state_dict=state, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(path)
        inventory = {item.name: {"bytes": item.stat().st_size, "sha256": gradient.file_sha(item)}
                     for item in sorted(path.iterdir()) if item.is_file()}
        row = {"step": step, "checkpoint": str(path), "parameter_sha256": digest,
               "initial_parameter_sha256": initial_hash,
               "consumed_pairs": step * 32, "actual_data_epochs": step * 32 / 200,
               "completed_data_epochs": int(window.get("epoch_completed", step * 32 // 200)),
               "evaluation_kind": window.get("snapshot_kind", "final_tail" if step * 32 % 200 and window.get("final_window") else "update_after_epoch_crossing"),
               "source_epoch_index": window.get("source_epoch_index", step * 32 // 200),
               "files": inventory}
        for key in ("consumed_pairs", "updated_pairs", "pending_pairs", "actual_data_epochs",
                    "weight_update_epochs", "completed_data_epochs", "source_epoch_index", "evaluation_kind"):
            if key in window:
                row[key] = window[key]
        atomic_json(path / "checkpoint_provenance.json", row)
        del state
    else:
        row = None
    if distributed:
        torch.distributed.barrier()
        values = [row]
        torch.distributed.broadcast_object_list(values, src=0)
        row = values[0]
    return row


def setup(config, layout, seed, torch, *, student_checkpoint=None):
    if student_checkpoint is not None and layout != "four":
        raise ValueError("Checkpoint evaluation setup requires four ranks to preserve the original Full teacher")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler
    from trainer.utils import seed_everything
    distributed = layout == "four"
    rank, world, local_rank = (int(os.getenv(key, default)) for key, default in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    if world != (4 if distributed else 1) or not torch.cuda.is_available():
        raise ValueError("Launch single with one CUDA process, four with exactly four CUDA ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        import deepspeed
        deepspeed.init_distributed(dist_backend="nccl")
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_args = copy.deepcopy(config["model"]["model_args"])
    model_path = model_args.pop("pretrained_model_name_or_path")
    model_args.pop("torch_dtype", None)
    model_args.pop("device_map", None)
    model_args["attn_implementation"] = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(**config["model"]["tokenizer_args"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if float(getattr(model_config, "attention_dropout", 0.0)) != 0.0:
        raise ValueError("Matched-pair comparison requires dropout zero")
    train_args = config["trainer"]["args"]
    accumulation = 8 if distributed else 4
    holder, ds_config = None, None
    if distributed:
        from transformers.integrations import HfDeepSpeedConfig
        import deepspeed
        ds_path = train_args.get("deepspeed") or ROOT / "configs/accelerate/zero_stage3_offload_config.json"
        ds_config = gradient.resolved_deepspeed(json.loads(Path(ds_path).read_text()), model_config.hidden_size, accumulation, world)
        ds_config["gradient_clipping"] = float(train_args.get("max_grad_norm", 1.0))
        holder = HfDeepSpeedConfig(ds_config)
        model = AutoModelForCausalLM.from_pretrained(student_checkpoint or model_path, torch_dtype=torch.bfloat16, **model_args)
        reference = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, **model_args)
        reference.requires_grad_(False)
        reference.config.use_cache = False
        reference.eval()
        reference, *_ = deepspeed.initialize(model=reference, config=ds_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, **model_args).to(device)
        reference = copy.deepcopy(model).to(device)
        reference.requires_grad_(False)
        reference.eval()
        reference.config.use_cache = False
    model.config.use_cache = False
    model.train()
    if train_args.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=train_args.get("gradient_checkpointing_kwargs"))
    if any(float(module.p) for module in model.modules() if isinstance(module, torch.nn.Dropout)):
        raise ValueError("Nonzero dropout is excluded")
    optimizer = make_optimizer(model, train_args)
    total_updates = int(train_args.get("max_steps", 60))
    if total_updates <= 0:
        total_updates = 60
    warmup = int(train_args.get("warmup_steps", 0))
    if train_args.get("warmup_epochs"):
        warmup = int(float(train_args["warmup_epochs"]) * 200 / 32)
    scheduler = get_scheduler(train_args.get("lr_scheduler_type", "linear"), optimizer=optimizer,
                              num_warmup_steps=warmup, num_training_steps=total_updates)
    if distributed:
        engine, optimizer, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer,
                         model_parameters=[p for p in model.parameters() if p.requires_grad],
                         lr_scheduler=scheduler, config=ds_config)
        engine.train()
        engine.zero_grad()
    else:
        engine = model
        model.zero_grad(set_to_none=True)
    return {"model": model, "reference": reference, "tokenizer": tokenizer,
            "optimizer": optimizer, "scheduler": scheduler, "engine": engine,
            "holder": holder, "ds_config": ds_config, "rank": rank, "world": world,
            "device": device, "distributed": distributed, "accumulation": accumulation,
            "warmup_steps": warmup, "scheduler_total_updates": total_updates,
            "model_path": str(model_path)}


def train(args, torch, control=None):
    if not args.config or not args.schedule or not args.output:
        raise ValueError("train requires --config --schedule --output")
    config = json.loads(args.config.read_text())
    method, train_args = config["trainer"]["method_args"], config["trainer"]["args"]
    planned_updates = planned_update_budget(control)
    explicit_budget = control is not None and hasattr(control, "planned_updates")
    if explicit_budget and train_args.get("max_steps") != planned_updates:
        raise AssertionError("Controller budget differs from the explicit configuration max_steps")
    for key, expected in (("target_mode", "mean"), ("retain_loss_type", "NLL"), ("non_sensitive_constraint_type", "kl")):
        if method.get(key, expected) != expected:
            raise ValueError(f"Unsupported PALU {key}")
    schedule = torch.load(args.schedule, map_location="cpu", weights_only=True)
    windows = schedule["windows"]
    if args.max_updates is None:
        if len(windows) != planned_updates or schedule["metadata"]["effective_batch"] != 32:
            raise AssertionError(f"Formal study requires exactly {planned_updates} complete windows of 32 pairs")
        if Path(config["model"]["model_args"]["pretrained_model_name_or_path"]).resolve() != Path(schedule["metadata"]["model_path"]).resolve():
            raise AssertionError("Formal initial model path differs from frozen schedule provenance")
        if schedule["metadata"].get("source_config_sha256") != gradient.file_sha(args.config):
            raise AssertionError("Formal source configuration differs from frozen schedule provenance")
    if args.max_updates is not None:
        windows = windows[:args.max_updates]
    if not windows or len(windows) > planned_updates or any(int(window["step"]) != i + 1 for i, window in enumerate(windows)):
        raise ValueError(f"Expected a contiguous at-most-{planned_updates} update schedule")
    from prepare_layout_schedule import tensor_hashes, canonical_sha
    for window in windows:
        for split in ("forget", "retain"):
            shapes = [window[split][key].shape for key in ("input_ids", "attention_mask", "labels")]
            if any(len(shape) != 2 or shape[0] != 32 for shape in shapes) or len(set(shapes)) != 1:
                raise AssertionError(f"Invalid frozen32 tensors at update {window['step']}")
        observed = tensor_hashes(window)
        if window.get("tensor_hashes") != observed or window.get("pair_sha256") != canonical_sha(window["pair_records"]):
            raise AssertionError(f"Frozen schedule tensor/pair integrity failed at step {window['step']}")
        expected = canonical_sha({"pairs": window["pair_records"], "counts": window["counts"], "tensors": observed})
        if expected != window["window_sha256"]:
            raise AssertionError("Frozen update identity disagrees with its contents")
    # Optional experiment controller selects a prefix or continuation only after
    # the complete immutable input schedule has passed the ordinary checks.
    if control is not None:
        windows = control.select_windows(windows)
    runtime = setup(config, args.layout, args.seed, torch)
    scheduler_kind = train_args.get("lr_scheduler_type", "linear")
    supported_schedulers = ("linear", "constant") if explicit_budget else ("linear",)
    if runtime["scheduler_total_updates"] != planned_updates or scheduler_kind not in supported_schedulers:
        raise AssertionError(f"Controlled study requires {planned_updates} updates and one of {supported_schedulers}")
    if scheduler_kind == "constant" and runtime["warmup_steps"] != 0:
        raise AssertionError("Constant learning rate requires explicit zero warmup")
    model, reference, engine = (runtime[key] for key in ("model", "reference", "engine"))
    optimizer, scheduler = runtime["optimizer"], runtime["scheduler"]
    rank, world, distributed = runtime["rank"], runtime["world"], runtime["distributed"]
    accumulation, device = runtime["accumulation"], runtime["device"]
    train_args.update(per_device_train_batch_size=1 if distributed else 8,
                      gradient_accumulation_steps=accumulation, seed=args.seed, data_seed=args.seed)
    tokenizer = runtime["tokenizer"]
    output = args.output.resolve()
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Refusing to replace existing training measurements")
        output.mkdir(parents=True, exist_ok=True)
        gradient.write_json(output / "resolved_config.json", config)
    if distributed:
        torch.distributed.barrier()
    initial_hash = gradient.parameter_hash(model, torch, rank, distributed)
    reference_base = reference.module if distributed else reference
    reference_hash = gradient.parameter_hash(reference_base, torch, rank, distributed)
    if initial_hash != reference_hash:
        raise AssertionError("Teacher and student do not have identical initial weights")
    if schedule["metadata"]["eos_token_id"] != tokenizer.eos_token_id or schedule["metadata"]["first_n"] != int(method["first_n"]):
        raise AssertionError("Schedule target-mask settings differ from the training model")
    initial = {"step": 0, "checkpoint": runtime["model_path"], "parameter_sha256": initial_hash,
               "consumed_pairs": 0, "actual_data_epochs": 0.0,
               "completed_data_epochs": 0, "evaluation_kind": "initial"}
    started = time.time()
    manifest = {"schema": 1, "status": "training", "layout": args.layout,
                "normalization": "whole_update_token_denominators" if args.normalization == "global" else "native_microbatch_means",
                "loss_mode": args.normalization, "padding_scope": args.padding_scope, "seed": args.seed,
                "world_size": world, "batch_size": 1 if distributed else 8,
                "gradient_accumulation_steps": accumulation, "effective_batch": 32,
                "planned_updates": planned_updates, "requested_updates": len(windows),
                "final_model_step": len(windows),
                "smoke_prefix": args.max_updates is not None,
                "schedule_path": str(args.schedule.resolve()), "schedule_sha256": gradient.file_sha(args.schedule),
                "schedule_metadata": schedule.get("metadata"), "config_sha256": gradient.file_sha(args.config),
                "initial_parameter_sha256": initial_hash, "teacher_parameter_sha256": reference_hash,
                "checkpoints": [initial], "completed_updates": 0,
                "model_path": runtime["model_path"], "method": method,
                "deepspeed_config": runtime["ds_config"],
                "warmup_steps": runtime["warmup_steps"], "scheduler_total_updates": runtime["scheduler_total_updates"],
                "optimizer_precision_initial": optimizer_precision(optimizer, torch),
                "max_grad_norm": float(train_args.get("max_grad_norm", 1.0)),
                "code_sha256": {"layout_token_training.py": gradient.file_sha(__file__),
                                  "gradient_equivalence_step.py": gradient.file_sha(Path(gradient.__file__)),
                                  "src/trainer/unlearn/palu.py": gradient.file_sha(ROOT / "src/trainer/unlearn/palu.py")},
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name_rank0": torch.cuda.get_device_name(device),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "packages": {key: importlib.metadata.version(key) for key in ("torch", "transformers", "accelerate", "deepspeed", "bitsandbytes")},
                "evaluation_protocol": ("immediate actual-epoch evaluation of current in-memory model; all ranks preserve pending gradients, optimizer, scheduler and RNG; delete verified intermediate weights before next training microbatch"
                    if args.online_evaluation else "separate single-GPU process reloads each saved student snapshot, keeps initial Full teacher, fixed F200/R3800 losses and native TOFU"),
                "online_epoch_evaluation": args.online_evaluation,
                "evaluation_steps": [0] + [int(window["step"]) for window in windows if window.get("evaluation_boundary") or int(window["step"]) == len(windows)],
                "limitations": ["Layouts retain native BF16/communication/optimizer master arithmetic; remaining difference is layout plus backend.",
                                 schedule["metadata"]["evaluation_policy"]]}
    if rank == 0:
        atomic_json(output / "manifest.json", manifest)
    if control is not None:
        control.initialize(runtime, manifest, config, schedule, args, torch)
    online = None
    if args.online_evaluation:
        from online_epoch_evaluation import OnlineEpochEvaluation
        online = OnlineEpochEvaluation(args, config, schedule, runtime, manifest, torch)
        if control is not None:
            control.configure_online(online)
        manifest["evaluation_steps"] = [event["step"] for event in online.events.values()]

        def save_event(event):
            return save_checkpoint(model, tokenizer, output, event["step"], initial_hash,
                                   torch, rank, distributed, event)

        if control is None or control.run_initial_evaluation:
            online.at(0, 0, save_event)
    for window in windows:
        step = int(window["step"])
        gamma, alpha, common = objective_coefficients(control, step, method)
        counts = gradient.target_counts(window, tokenizer.eos_token_id, int(method["first_n"]), torch)
        if counts != window["counts"] or any(counts[key] <= 0 for key in counts):
            raise AssertionError(f"Invalid whole-window denominators at {step}")
        if rank == 0 and not online:
            for pair in window.get("pair_records", []):
                append_json(output / "training_pairs.jsonl", {"step": step, "optimizer_step_before_update": step - 1, **pair})
        if not distributed:
            model.zero_grad(set_to_none=True)
        lr_before = [float(group["lr"]) for group in optimizer.param_groups]
        assert_learning_rate(lr_before, step - 1, runtime, train_args)
        contributions, token_contributions, micro_records = ({key: 0.0 for key in counts},
                                                           {key: 0.0 for key in counts}, [])
        before = int(engine.global_steps) if distributed else step - 1
        skipped_before = int(getattr(engine, "skipped_steps", 0)) if distributed else 0
        for microstep in range(accumulation):
            indices = [microstep * world + rank] if distributed else list(range(microstep * 8, (microstep + 1) * 8))
            inputs = microbatch_inputs(window, indices, device, args.padding_scope)
            local = gradient.target_counts(inputs, tokenizer.eos_token_id, int(method["first_n"]), torch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                components = gradient.raw_components(engine, reference, inputs, top_k=int(method["top_k"]),
                           first_n=int(method["first_n"]), eos=tokenizer.eos_token_id, torch=torch)
                loss = gradient.weighted_loss(components, local, counts, gamma=gamma, alpha=alpha,
                               common=common, normalization=args.normalization, accumulation=accumulation, world=world)
            if not bool(torch.isfinite(loss.detach())):
                raise AssertionError(f"Nonfinite loss at update {step}")
            for key in counts:
                value = float(components[key].detach())
                token_contributions[key] += value * local[key] / counts[key]
                contributions[key] += value * (local[key] / counts[key] if args.normalization == "global" else 1 / (accumulation * world))
            micro_records.append({"rank": rank, "microstep": microstep, "pair_indices": indices,
                                  "counts": local, "component_dtypes": {key: str(value.dtype) for key, value in components.items()},
                                  "raw": {key: float(value.detach()) for key, value in components.items()},
                                  "input_shapes": {split: list(inputs[split]["input_ids"].shape) for split in inputs},
                                  "scaled_forward_loss": float(loss.detach())})
            if distributed:
                engine.backward(loss)
                if microstep < accumulation - 1:
                    if engine.is_gradient_accumulation_boundary():
                        raise AssertionError("Unexpected early optimizer boundary")
                    engine.step()
                    if int(engine.global_steps) != before:
                        raise AssertionError("Optimizer updated inside frozen32 window")
            else:
                (loss / accumulation).backward()
            del loss, components, inputs
            if online:
                global_microbatch = world if distributed else 8
                consumed = (step - 1) * 32 + (microstep + 1) * global_microbatch
                if rank == 0:
                    start = microstep * global_microbatch
                    for pair in window["pair_records"][start:start + global_microbatch]:
                        append_json(output / "training_pairs.jsonl", {"step": step,
                            "optimizer_step_before_update": step - 1, **pair})
                # At a non-boundary microstep, weights are still the previous
                # update. Do not zero, flush or otherwise complete this gradient.
                if microstep < accumulation - 1:
                    online.at(consumed, step - 1, save_event)
        if distributed:
            if not engine.is_gradient_accumulation_boundary():
                raise AssertionError("Frozen32 window is not an optimizer boundary")
            engine.step()
            if int(engine.global_steps) != before + 1:
                raise AssertionError("Expected exactly one optimizer update")
            if int(getattr(engine, "skipped_steps", 0)) != skipped_before or bool(getattr(optimizer, "overflow", False)):
                raise AssertionError("DeepSpeed skipped an update due to overflow")
            norm = engine.get_global_grad_norm()
            norm_value = float(norm) if norm is not None else None
            norm_dtype = str(getattr(norm, "dtype", None))
            if norm_value is None or not math.isfinite(norm_value):
                raise AssertionError("Missing or nonfinite DeepSpeed gradient norm")
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_args.get("max_grad_norm", 1.0)))
            norm_value, norm_dtype = float(norm), str(norm.dtype)
            if not math.isfinite(norm_value):
                raise AssertionError("Nonfinite single-GPU gradient norm")
            optimizer.step()
            scheduler.step()
        lr_after = [float(group["lr"]) for group in optimizer.param_groups]
        assert_learning_rate(lr_after, step, runtime, train_args)
        if distributed:
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, {"raw": contributions, "token_raw": token_contributions, "microsteps": micro_records})
            raw = {key: sum(part["raw"][key] for part in gathered) for key in counts}
            token_raw = {key: sum(part["token_raw"][key] for part in gathered) for key in counts}
            micros = sorted([row for part in gathered for row in part["microsteps"]], key=lambda row: (row["microstep"], row["rank"]))
        else:
            raw, token_raw, micros = contributions, token_contributions, micro_records
        if sorted(index for row in micros for index in row["pair_indices"]) != list(range(32)):
            raise AssertionError("Microbatches do not consume frozen32 exactly once")
        if {key: sum(row["counts"][key] for row in micros) for key in counts} != counts:
            raise AssertionError("Microbatch counts do not reconstruct global denominators")
        torch.cuda.synchronize(device)
        manifest["completed_updates"] = step
        if rank == 0:
            row = {"schema": 1, "step": step, "global_step": step, "epoch": step * 32 / 200,
                   "completed_data_epochs": step * 32 // 200, "consumed_pairs": step * 32,
                   "global_target_counts": counts, "window_sha256": window.get("window_sha256"),
                   "pair_sha256": window.get("pair_sha256"), "microbatches_all_ranks": accumulation * world,
                   "loss_reduction": "pre-update whole-window target-token mean" if args.normalization == "global" else "pre-update equal mean over all native rank/microbatch means",
                   "loss_mode": args.normalization, "padding_scope": args.padding_scope,
                   "window_token_mean": loss_record(token_raw, gamma, alpha, common),
                   **loss_record(raw, gamma, alpha, common), "grad_norm": norm_value, "grad_norm_dtype": norm_dtype,
                   "grad_norm_semantics": "before clipping; native backend norm computation",
                   "clip_threshold": float(train_args.get("max_grad_norm", 1.0)),
                   "clip_factor_nominal": min(1.0, float(train_args.get("max_grad_norm", 1.0)) / (norm_value + 1e-6)) if norm_value else 1.0,
                   "lr_before": lr_before, "lr_after": lr_after, "microsteps": micros,
                   "optimizer_precision": optimizer_precision(optimizer, torch),
                   "seconds_since_start": time.time() - started}
            if control is not None:
                row.update(gamma_used=gamma, alpha_used=alpha, common_used=common,
                           canonical_total=float(method["gamma"]) * raw["sensitive"] + float(method["alpha"]) * (raw["retain"] + raw["common"]))
                if hasattr(control, "training_row_metadata"):
                    row.update(control.training_row_metadata(step))
            append_json(output / "loss_history.jsonl", row)
            manifest["completed_updates"] = step
            atomic_json(output / "manifest.json", manifest)
            print(json.dumps({"step": step, "loss": row["loss"], "grad_norm": norm_value, "lr_used": lr_before[0]}), flush=True)
        if online:
            # An epoch edge aligned with an optimizer boundary is evaluated
            # after the update and scheduler step, as in the source HF run.
            online.at(step * 32, step, save_event)
        elif window.get("evaluation_boundary") or step == len(windows):
            checkpoint_window = {**window, "final_window": step == len(windows)}
            checkpoint = save_checkpoint(model, tokenizer, output, step, initial_hash, torch, rank, distributed, checkpoint_window)
            manifest["checkpoints"].append(checkpoint)
            if rank == 0:
                atomic_json(output / "manifest.json", manifest)
    if control is not None and control.finish(runtime, manifest, args, torch):
        return
    manifest.update(status="completed" if online else "training_completed", completed_unix=time.time(),
                    training_wall_seconds=time.time() - started,
                    optimizer_precision_final=optimizer_precision(optimizer, torch),
                    peak_allocated_bytes_rank0=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes_rank0=torch.cuda.max_memory_reserved(device))
    if online:
        if manifest.get("evaluated_steps") != manifest["evaluation_steps"]:
            raise AssertionError("Training completed without all immediate epoch observations")
        manifest.update(evaluation_config_sha256=gradient.file_sha(args.config),
                        evaluation_completed_unix=time.time(),
                        checkpoint_retention="final_only_after_verified_epoch_evaluation" if args.delete_evaluated_intermediate else "all",
                        deleted_intermediate_steps=[item["step"] for item in manifest["checkpoints"]
                            if item.get("weights_deleted_after_verified_evaluation")])
    if rank == 0:
        atomic_json(output / "manifest.json", manifest)
    if distributed:
        torch.distributed.barrier()


def evaluate(args, torch):
    if not args.run or not args.config:
        raise ValueError("evaluate requires --run --config")
    if int(os.getenv("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise ValueError("Evaluation must be a new process with exactly one visible GPU")
    from omegaconf import OmegaConf
    from data import get_collators, get_data
    from model import get_model
    from trainer import load_trainer_args
    from trainer.palu_epoch_eval import PALUEpochTrainer
    from trainer.utils import seed_everything
    from evals import get_evaluators
    from transformers.modeling_utils import load_sharded_checkpoint
    from safetensors.torch import load_file
    import checkpoint_retention as retention
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "training_completed":
        raise ValueError("Only a completed uninterrupted training run can be evaluated")
    workspace = run / "evaluation_workspace"
    if workspace.exists() and any(workspace.iterdir()) and not args.resume_evaluation:
        raise ValueError("Refusing to replace previous evaluation; preserve it before retrying")
    if args.resume_evaluation:
        workspace.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while (workspace / f"attempt-{attempt:04d}").exists():
            attempt += 1
        evaluation = workspace / f"attempt-{attempt:04d}"
    else:
        evaluation = workspace
    evaluation.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    for key in ("alpha", "gamma", "top_k", "first_n", "target_mode", "retain_loss_type", "non_sensitive_constraint_type"):
        if config["trainer"]["method_args"].get(key) != manifest["method"].get(key):
            raise AssertionError(f"Evaluation PALU objective differs from training: {key}")
    train_model_path = manifest["model_path"]
    config["model"]["model_args"]["pretrained_model_name_or_path"] = train_model_path
    config["model"]["model_args"]["attn_implementation"] = "sdpa"
    config["model"]["model_args"]["torch_dtype"] = "bfloat16"
    training_args = config["trainer"]["args"]
    training_args.pop("deepspeed", None)
    training_args.pop("evaluation_strategy", None)
    training_args.update(output_dir=str(evaluation), logging_dir=str(evaluation / "logs"),
                seed=manifest["seed"], data_seed=manifest["seed"], report_to="none", eval_strategy="no",
                dataloader_num_workers=0, per_device_train_batch_size=8, gradient_accumulation_steps=4)
    config["eval"]["tofu"]["metrics"] = {key: config["eval"]["tofu"]["metrics"][key] for key in ("forget_quality", "model_utility")}
    cfg = OmegaConf.create(config)
    seed_everything(manifest["seed"])
    model, tokenizer = get_model(cfg.model)
    data = get_data(cfg.data, mode=cfg.get("mode", "unlearn"), tokenizer=tokenizer,
                    template_args=cfg.model.template_args, model_name=cfg.model.handler)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    evaluators = get_evaluators(cfg.eval, template_args=cfg.model.template_args, model=model, tokenizer=tokenizer)
    trainer = PALUEpochTrainer(model=model, tokenizer=tokenizer,
              args=load_trainer_args(cfg.trainer.args, data["train"]), train_dataset=data["train"],
              eval_dataset=data.get("eval"), data_collator=collator, evaluators=evaluators,
              template_args=cfg.model.template_args, **dict(cfg.trainer.method_args),
              epoch_options={"forget_samples": "all", "retain_samples": "all", "model_name": config["model"]["handler"]})
    model = trainer.model
    initial_hash = gradient.parameter_hash(model, torch, 0, False)
    teacher_hash = gradient.parameter_hash(trainer.ref_model, torch, 0, False)
    if initial_hash != manifest["initial_parameter_sha256"] or teacher_hash != initial_hash:
        raise AssertionError("Independent evaluator initial model/teacher hash disagrees with training")
    trainer._prepare_fixed_data()
    if trainer._epoch_fixed_manifest["fixed_dataset_sha256"] != manifest["schedule_metadata"]["fixed_dataset_sha256"]:
        raise AssertionError("Evaluator fixed full datasets differ from the immutable training schedule pools")
    atomic_json(evaluation / "resolved_config.json", config)
    completed_rows = []
    for checkpoint in manifest["checkpoints"]:
        step = int(checkpoint["step"])
        cached = retention.read_receipt(run, manifest, checkpoint) if args.resume_evaluation else None
        if cached is not None:
            completed_rows.append(cached)
            retention.rebuild_aggregates(run, completed_rows)
            if args.delete_evaluated_intermediate:
                retention.prune_intermediate(run, manifest, checkpoint)
            manifest["evaluated_steps"] = [row["step"] for row in completed_rows]
            atomic_json(run / "manifest.json", manifest)
            print(json.dumps({"step": step, "evaluation_reused": True}), flush=True)
            continue
        path = Path(checkpoint["checkpoint"])
        if step:
            if (path / "model.safetensors.index.json").exists():
                load_sharded_checkpoint(model, path, strict=True, prefer_safe=True)
            else:
                state = load_file(str(path / "model.safetensors"), device="cpu")
                model.load_state_dict(state, strict=True)
                del state
        current_hash = gradient.parameter_hash(model, torch, 0, False)
        if current_hash != checkpoint["parameter_sha256"]:
            raise AssertionError(f"Reloaded checkpoint hash disagrees at step {step}")
        trainer.state.global_step = step
        trainer.state.epoch = checkpoint["actual_data_epochs"]
        trainer._epoch_context.update(epoch_index=checkpoint["completed_data_epochs"],
                         microbatches=0, expected_microbatches=0, epoch_end_seen=True)
        trainer.evaluate()
        rows = [json.loads(line) for line in (evaluation / "epoch_history.jsonl").read_text().splitlines()]
        row = rows[-1]
        row.update(completed_data_epochs=checkpoint["completed_data_epochs"],
                   source_epoch_index=checkpoint.get("source_epoch_index", checkpoint["completed_data_epochs"]),
                   consumed_pairs=checkpoint["consumed_pairs"], actual_data_epochs=checkpoint["actual_data_epochs"],
                   evaluation_kind=checkpoint["evaluation_kind"],
                   partial_epoch=checkpoint["evaluation_kind"] == "final_tail",
                   checkpoint_parameter_sha256=current_hash,
                   initial_parameter_sha256=initial_hash, teacher_parameter_sha256=teacher_hash,
                   checkpoint_path=str(path), training_layout=manifest["layout"],
                   training_normalization=manifest["normalization"],
                   training_padding_scope=manifest.get("padding_scope", "update"),
                   training_schedule_sha256=manifest["schedule_sha256"],
                   evaluation_model="separate process reloads saved BF16 student; teacher remains initial Full model")
        # Replace training-specific base evaluator bookkeeping by schedule facts.
        for key in ("actual_epoch_microbatches", "expected_epoch_microbatches", "training_examples_this_epoch",
                    "unique_forget_examples_this_epoch", "unique_retain_examples_this_epoch", "unique_retain_examples_cumulative"):
            row.pop(key, None)
        if gradient.parameter_hash(trainer.ref_model, torch, 0, False) != teacher_hash:
            raise AssertionError("Teacher changed during evaluation; checkpoint weights are retained")
        row = retention.commit_observation(run, manifest, checkpoint, row,
                Path(row["evaluation_dir"]), evaluation / "fixed_loss_samples.jsonl")
        completed_rows.append(row)
        retention.rebuild_aggregates(run, completed_rows)
        if args.delete_evaluated_intermediate:
            retention.prune_intermediate(run, manifest, checkpoint)
        manifest["evaluated_steps"] = [item["step"] for item in completed_rows]
        atomic_json(run / "manifest.json", manifest)
        print(json.dumps({"step": step, "fq": row["forget_quality"], "mu": row["model_utility"], "ks_d": row["ks_d"]}), flush=True)
    final_teacher_hash = gradient.parameter_hash(trainer.ref_model, torch, 0, False)
    if teacher_hash != final_teacher_hash:
        raise AssertionError("Reference teacher changed during evaluation")
    manifest.update(status="completed", evaluation_config_sha256=gradient.file_sha(args.config),
                    evaluated_states=len(completed_rows), evaluation_completed_unix=time.time(),
                    fixed_dataset_sha256=completed_rows[0]["fixed_dataset_sha256"],
                    reference_sha256=completed_rows[0]["reference_sha256"],
                    checkpoint_retention="final_only_after_verified_epoch_evaluation" if args.delete_evaluated_intermediate else "all",
                    deleted_intermediate_steps=[item["step"] for item in manifest["checkpoints"] if item.get("weights_deleted_after_verified_evaluation")],
                    evaluation_teacher_unchanged=True)
    atomic_json(run / "manifest.json", manifest)


def main():
    args = arguments()
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    if args.mode == "train":
        train(args, torch)
    elif args.mode == "evaluate":
        evaluate(args, torch)
    else:
        gradient.self_test(torch)
        # Scheduler and canonical labels can be tested without GPUs/8B weights.
        from transformers import get_scheduler
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=2e-5)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=6, num_training_steps=60)
        used = []
        for _ in range(60):
            used.append(float(optimizer.param_groups[0]["lr"]))
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
        assert used[0] == 0.0 and math.isclose(used[6], 2e-5) and optimizer.param_groups[0]["lr"] == 0.0
        boundaries = sorted(set(math.ceil(epoch * 200 / 32) for epoch in range(1, 10))) + [60]
        assert boundaries == [7, 13, 19, 25, 32, 38, 44, 50, 57, 60]
        print(json.dumps({"status": "training_self_test_passed", "epoch_crossing_update_steps": boundaries,
                          "lr_used_first": used[0], "lr_used_at_step7": used[6], "lr_after_step60": 0.0}), flush=True)


if __name__ == "__main__":
    main()
