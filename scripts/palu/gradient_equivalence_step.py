#!/usr/bin/env python3
"""Capture PALU's *unclipped* gradient for one immutable 32-pair update window.

No optimizer update is executed.  Single: b8 x accumulation4; distributed:
ZeRO-3 b1 x accumulation8 x 4 ranks.  Every input is sliced from the same
globally padded CPU tensor bundle.  `global` changes only objective weights;
it does not silently promote the model's native objective arithmetic.

python scripts/palu/gradient_equivalence_step.py --config config.json
  --pairs pairs.pt --output native_single --layout single --normalization native
torchrun --nproc_per_node=4 scripts/palu/gradient_equivalence_step.py
  --config config.json --pairs pairs.pt --output global_four
  --layout four --normalization global

Snapshot format: gradients.f32.bin, native-endian float32 contiguous values in
model.named_parameters() order; gradient_layout.json describes float offsets.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("single", "four"), required=True)
    parser.add_argument("--normalization", choices=("native", "global"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true", help="CPU tiny-model objective and gradient tests; config/pairs are not read")
    return parser.parse_args()


def sensitive_mask(labels, eos, first_n, torch):
    """Exactly PALU's shifted-label selection, independent of microbatch shape."""
    shifted = labels[..., 1:]
    mask = shifted >= 0
    if eos is not None:
        mask = mask & (shifted != eos)
    if first_n > 0:
        refined = torch.zeros_like(mask)
        for row in range(mask.shape[0]):
            positions = torch.nonzero(mask[row], as_tuple=False).flatten().tolist()
            last, in_interval = None, 0
            for position in positions:
                in_interval = in_interval + 1 if last is not None and position == last + 1 else 1
                if in_interval <= first_n:
                    refined[row, position] = True
                last = position
        mask = refined
    return mask


def target_counts(inputs, eos, first_n, torch):
    return {
        "sensitive": int(sensitive_mask(inputs["forget"]["labels"], eos, first_n, torch).sum()),
        "common": int((inputs["forget"]["labels"][..., 1:] == -200).sum()),
        "retain": int((inputs["retain"]["labels"][..., 1:] != -100).sum()),
    }


def raw_components(model, reference, inputs, *, top_k, first_n, eos, torch):
    """PALU native arithmetic; retain CE is obtained from the native model loss.

    This isolated measurement copy is checked against repository PALU.compute_loss
    on the first actual microbatch and by --self-test; core training is untouched.
    Sensitive/common numerators retain their original dtype and reduction order.
    """
    F = torch.nn.functional
    forget = {key: inputs["forget"][key] for key in ("input_ids", "attention_mask")}
    forget["use_cache"] = False
    output = model(**forget)
    shifted_logits = output.logits[..., :-1, :].contiguous()
    flat = shifted_logits.view(-1, shifted_logits.shape[-1])
    vocabulary = flat.shape[-1]
    with torch.no_grad():
        reference_output = reference(**forget)
    reference_logits = reference_output.logits[..., :-1, :].contiguous()
    reference_flat = reference_logits.view(-1, vocabulary)
    _, indices = torch.topk(reference_flat, min(top_k, vocabulary), dim=-1)
    batch_indices = torch.arange(flat.shape[0], device=flat.device).unsqueeze(1).expand_as(indices)
    selected = flat[batch_indices, indices]
    target = flat.mean(dim=-1, keepdim=True).expand_as(selected)
    per_token = ((selected - target) ** 2).mean(dim=-1)
    sensitive = sensitive_mask(inputs["forget"]["labels"], eos, first_n, torch).reshape(-1)
    forget_loss = (per_token * sensitive).sum() / sensitive.sum().clamp(min=1)
    common = (inputs["forget"]["labels"][..., 1:] == -200).reshape(-1)
    if common.sum() > 0:
        ref_log_prob = F.log_softmax(reference_logits, dim=-1).view(-1, vocabulary)
        log_prob = F.log_softmax(shifted_logits, dim=-1).view(-1, vocabulary)
        per_token_kl = torch.sum(torch.exp(log_prob) * (log_prob - ref_log_prob), dim=-1)
        common_loss = (per_token_kl * common).sum() / common.sum().clamp(min=1)
    else:
        common_loss = torch.tensor(0.0, device=flat.device)
    retain = {key: inputs["retain"][key] for key in ("input_ids", "attention_mask", "labels")}
    retain["use_cache"] = False
    retain_loss = model(**retain).loss
    return {"sensitive": forget_loss, "retain": retain_loss, "common": common_loss}


def weighted_loss(components, local_counts, global_counts, *, gamma, alpha,
                  normalization, accumulation, world):
    coefficient = {"sensitive": gamma, "retain": alpha, "common": alpha}
    if normalization == "native":
        # Preserve the actual PALU operand/order: gamma*F + alpha*R + alpha*KL.
        return gamma * components["sensitive"] + alpha * components["retain"] + alpha * components["common"]
    return sum(coefficient[key] * components[key]
               * (accumulation * world * local_counts[key] / max(global_counts[key], 1))
               for key in ("sensitive", "retain", "common"))


def slice_inputs(bundle, indices, device):
    return {split: {key: bundle[split][key][indices].to(device)
                    for key in ("input_ids", "attention_mask", "labels")}
            for split in ("forget", "retain")}


def check_original_loss(model, reference, inputs, components, tokenizer, method, torch):
    from trainer.unlearn.palu import PALU
    original = object.__new__(PALU)
    for key, value in {"top_k": int(method["top_k"]), "num_sensitive_tokens": int(method["first_n"]),
                       "target_mode": "mean", "non_sensitive_constraint_type": "kl",
                       "gamma": float(method["gamma"]), "alpha": float(method["alpha"]),
                       "retain_loss_type": "NLL", "ref_model": reference,
                       "tokenizer": tokenizer, "_palu_monitor": None,
                       "_palu_diagnostic_mode": True}.items():
        setattr(original, key, value)
    with torch.no_grad():
        expected = original.compute_loss(model, inputs)
    measured = method["gamma"] * components["sensitive"] + method["alpha"] * components["retain"] + method["alpha"] * components["common"]
    # SDPA on identical inputs is deterministic on this invocation. A tiny
    # fp32 model may differ only in its last arithmetic bit.
    equal = bool(torch.allclose(expected, measured.detach(), rtol=1e-6, atol=1e-7))
    if not equal:
        raise AssertionError(f"Measurement copy differs from native PALU: {expected.item()} vs {measured.item()}")
    return {"original": float(expected), "measurement_copy": float(measured.detach()), "passed": equal}


def parameter_hash(model, torch, rank, distributed):
    """Hash full BF16 weights without changing ZeRO parameter/hook state.

    A live final-backward graph can retain active ZeRO submodules, so entering
    GatheredParameters and then repartitioning is unsafe at this measurement
    boundary.  Instead assemble each immutable parameter's existing contiguous
    rank shards in fresh storage, and discard ZeRO's end padding.  This yields
    the same canonical bytes/header as ordinary unsharded named parameters.
    """
    digest = hashlib.sha256()
    if distributed and torch.cuda.is_available():
        torch.cuda.synchronize()
    for name, parameter in model.named_parameters():
        if distributed:
            shard = parameter.ds_tensor.detach().contiguous()
            group = getattr(parameter, "ds_process_group", None)
            group_size = torch.distributed.get_world_size(group)
            full = torch.empty(shard.numel() * group_size, device=shard.device, dtype=shard.dtype)
            torch.distributed.all_gather_into_tensor(full, shard, group=group)
            value = full[:int(parameter.ds_numel)]
            shape = list(parameter.ds_shape)
            if value.dtype != parameter.dtype:
                raise AssertionError(f"Parameter/shard dtype differs: {name}")
            # A subset of persistent or graph-active parameters may still have
            # a full resident view. Check those too without freeing that view.
            if parameter.numel() == int(parameter.ds_numel) and not torch.equal(parameter.detach().reshape(-1), value):
                raise AssertionError(f"Resident full weight differs from canonical shards: {name}")
        else:
            value, shape = parameter.detach(), list(parameter.shape)
        if rank == 0:
            header = json.dumps([name, shape, str(parameter.dtype)], separators=(",", ":")).encode()
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            tensor = value.contiguous().cpu()
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        if distributed:
            del full, value
    result = digest.hexdigest() if rank == 0 else None
    if distributed:
        values = [result]
        torch.distributed.broadcast_object_list(values, src=0)
        result = values[0]
    return result


def write_snapshot(model, output, initial_hash, torch, rank, distributed):
    if distributed:
        from deepspeed.utils import safe_get_full_grad
    parameters, offset, absent, nonfinite = [], 0, [], []
    handle = (output / "gradients.f32.bin").open("wb") if rank == 0 else None
    digest = hashlib.sha256()
    try:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            # ALL ranks call safe_get_full_grad in identical order.
            gradient = safe_get_full_grad(parameter) if distributed else parameter.grad
            shape = list(getattr(parameter, "ds_shape", parameter.shape))
            numel = int(getattr(parameter, "ds_numel", parameter.numel()))
            is_absent = gradient is None
            if is_absent:
                absent.append(name)
            if rank == 0:
                value = torch.zeros(numel, dtype=torch.float32) if is_absent else gradient.detach().reshape(-1).float().cpu()
                if value.numel() != numel:
                    raise AssertionError(f"Incomplete gradient {name}: {value.numel()} vs {numel}")
                if not bool(torch.isfinite(value).all()):
                    nonfinite.append(name)
                encoded = value.numpy().tobytes()
                handle.write(encoded)
                digest.update(encoded)
                parameters.append({"name": name, "shape": shape, "numel": numel,
                                   "offset": offset, "dtype": "float32", "requires_grad": True,
                                   "gradient_none": is_absent})
            offset += numel
            del gradient
    finally:
        if handle is not None:
            handle.close()
    if rank == 0:
        payload = {"schema": 1, "dtype": "float32", "byte_order": sys.byteorder,
                   "offset_unit": "float32_elements", "parameters": parameters,
                   "total_numel": offset, "file_bytes": offset * 4,
                   "gradient_file": "gradients.f32.bin", "gradient_sha256": digest.hexdigest(),
                   "initial_parameter_sha256": initial_hash,
                   "gradient_none_parameters": absent, "nonfinite_parameters": nonfinite,
                   "gradient_stage": "after complete accumulation backward, before clipping or optimizer update"}
        write_json(output / "gradient_layout.json", payload)
        if nonfinite:
            raise AssertionError(f"Nonfinite gradients: {nonfinite}")
        if absent:
            raise AssertionError(f"Missing trainable gradients: {absent}")
        return payload
    return None


def resolved_deepspeed(source, hidden, accumulation, world):
    stage = copy.deepcopy(source)
    stage.update(train_batch_size=32, train_micro_batch_size_per_gpu=1,
                 gradient_accumulation_steps=accumulation, gradient_clipping=1.0,
                 bf16={"enabled": True}, zero_allow_untested_optimizer=True)
    zero = stage.setdefault("zero_optimization", {})
    zero.update(stage=3, reduce_bucket_size=hidden * hidden,
                stage3_param_persistence_threshold=10 * hidden,
                stage3_prefetch_bucket_size=int(0.9 * hidden * hidden))
    for key in ("optimizer", "scheduler"):
        stage.pop(key, None)
    return stage


def json_dtype(value):
    return None if value is None else str(value)


def main(args):
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from trainer.utils import seed_everything

    if args.self_test:
        self_test(torch)
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    method = config["trainer"]["method_args"]
    for key, expected in (("target_mode", "mean"), ("non_sensitive_constraint_type", "kl"), ("retain_loss_type", "NLL")):
        if method.get(key, expected) != expected:
            raise ValueError(f"Only native PALU {key}={expected} is supported")
    rank = int(os.getenv("RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    distributed = args.layout == "four"
    if world != (4 if distributed else 1):
        raise ValueError(f"Expected {4 if distributed else 1} ranks, received {world}")
    if not torch.cuda.is_available():
        raise ValueError("Actual model gradient capture requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    accumulation = 8 if distributed else 4
    batch_size = 1 if distributed else 8
    if distributed:
        import deepspeed
        deepspeed.init_distributed(dist_backend="nccl")
    output = args.output.resolve()
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Refusing to replace existing measurement output")
        output.mkdir(parents=True, exist_ok=True)
    if distributed:
        torch.distributed.barrier()
    seed_everything(args.seed)
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bundle = torch.load(args.pairs, map_location="cpu", weights_only=False)
    for split in ("forget", "retain"):
        if any(bundle[split][key].shape[0] != 32 for key in ("input_ids", "attention_mask", "labels")):
            raise ValueError("Expected exactly 32 globally padded pairs")
        if bundle[split]["input_ids"].shape != bundle[split]["labels"].shape:
            raise ValueError("Mismatched global input/label shape")
    model_args = copy.deepcopy(config["model"]["model_args"])
    model_path = model_args.pop("pretrained_model_name_or_path")
    model_args.pop("torch_dtype", None)
    model_args.pop("device_map", None)
    model_args["attn_implementation"] = "sdpa"
    tokenizer_args = copy.deepcopy(config["model"]["tokenizer_args"])
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)
    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if float(getattr(model_config, "attention_dropout", 0.0)) != 0.0:
        raise ValueError("This matched-input test requires zero model attention dropout")
    counts = target_counts(bundle, tokenizer.eos_token_id, int(method["first_n"]), torch)
    supplied = bundle.get("counts")
    if supplied and supplied != counts:
        raise ValueError(f"Frozen bundle counts disagree: {supplied} vs {counts}")
    if any(counts[key] == 0 for key in counts):
        raise ValueError(f"All three global target pools must be nonempty: {counts}")
    ds_holder, ds_config = None, None
    if distributed:
        from transformers.integrations import HfDeepSpeedConfig
        ds_path = config["trainer"]["args"].get("deepspeed")
        if not ds_path:
            ds_path = ROOT / "configs/accelerate/zero_stage3_offload_config.json"
        ds_config = resolved_deepspeed(json.loads(Path(ds_path).read_text()),
                                       model_config.hidden_size, accumulation, world)
        ds_holder = HfDeepSpeedConfig(ds_config)  # Keep alive throughout load/capture.
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                    **model_args)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                    **model_args).to(device)
    model.config.use_cache = False
    model.train()
    checkpoint_kwargs = config["trainer"]["args"].get("gradient_checkpointing_kwargs")
    if config["trainer"]["args"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
    if any(float(module.p) for module in model.modules() if isinstance(module, torch.nn.Dropout)):
        raise ValueError("Nonzero dropout confounds this deterministic partition comparison")
    # Reference model has exactly the same starting weights, but no checkpointing
    # and no dropout.  ZeRO-3 partitions it without optimizer state.
    if distributed:
        reference = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                        **model_args)
        reference.requires_grad_(False)
        reference.config.use_cache = False
        reference.eval()
        reference, *_ = deepspeed.initialize(model=reference, config=ds_config)
        import bitsandbytes as bnb
        from transformers.trainer_pt_utils import get_parameter_names
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        decay_names = set(get_parameter_names(model, ALL_LAYERNORM_LAYERS))
        decay_names = {name for name in decay_names if "bias" not in name}
        train_args = config["trainer"]["args"]
        optimizer = bnb.optim.PagedAdamW32bit([
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and n in decay_names],
             "weight_decay": float(train_args.get("weight_decay", 0.01))},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and n not in decay_names],
             "weight_decay": 0.0}], lr=float(train_args.get("learning_rate", 2e-5)),
            betas=(float(train_args.get("adam_beta1", 0.9)), float(train_args.get("adam_beta2", 0.999))),
            eps=float(train_args.get("adam_epsilon", 1e-8)))
        engine, *_ = deepspeed.initialize(model=model, optimizer=optimizer,
                                         model_parameters=[p for p in model.parameters() if p.requires_grad],
                                         config=ds_config)
        engine.train()
        engine.zero_grad()
        initial_global_steps = int(engine.global_steps)
    else:
        reference = copy.deepcopy(model).to(device)
        reference.requires_grad_(False)
        reference.gradient_checkpointing_disable()
        reference.eval()
        model.zero_grad(set_to_none=True)
        engine, initial_global_steps = model, 0
    initial_hash = parameter_hash(model, torch, rank, distributed)
    if rank == 0:
        write_json(output / "started.json", {"model_path": str(model_path), "layout": args.layout,
                   "normalization": args.normalization, "initial_parameter_sha256": initial_hash,
                   "pairs_sha256": file_sha(args.pairs), "global_counts": counts,
                   "torch": torch.__version__, "started_epoch": started})
    records, all_indices = [], []
    native_check = None
    for microstep in range(accumulation):
        indices = [microstep * world + rank] if distributed else list(range(microstep * 8, (microstep + 1) * 8))
        all_indices.extend(indices)
        inputs = slice_inputs(bundle, indices, device)
        local = target_counts(inputs, tokenizer.eos_token_id, int(method["first_n"]), torch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            components = raw_components(engine, reference, inputs, top_k=int(method["top_k"]),
                                        first_n=int(method["first_n"]), eos=tokenizer.eos_token_id, torch=torch)
            if microstep == 0:
                native_check = check_original_loss(engine, reference, inputs, components,
                                                   tokenizer, method, torch)
            loss = weighted_loss(components, local, counts, gamma=float(method["gamma"]),
                                 alpha=float(method["alpha"]), normalization=args.normalization,
                                 accumulation=accumulation, world=world)
        records.append({"rank": rank, "microstep": microstep, "pair_indices": indices,
                        "counts": local, "raw": {key: float(value.detach()) for key, value in components.items()},
                        "raw_dtypes": {key: str(value.dtype) for key, value in components.items()},
                        "forward_loss": float(loss.detach()),
                        "backward_divisor": accumulation,
                        "dp_average_divisor": world,
                        "effective_contribution": float(loss.detach()) / accumulation / world})
        if distributed:
            engine.backward(loss)
            if microstep < accumulation - 1:
                # DeepSpeed increments its microstep even on a non-boundary step;
                # the zero-update assertions prove that no optimizer action occurred.
                if engine.is_gradient_accumulation_boundary():
                    raise AssertionError("Unexpected optimizer boundary before full 32-pair window")
                engine.step()
                if int(engine.global_steps) != initial_global_steps:
                    raise AssertionError("Optimizer updated before gradient capture")
            elif not engine.is_gradient_accumulation_boundary():
                raise AssertionError("Final backward is not a complete accumulation boundary")
        else:
            (loss / accumulation).backward()
        del loss, components, inputs
        torch.cuda.synchronize(device)
        print(json.dumps({"rank": rank, "microstep_completed": microstep + 1,
                          "microsteps": accumulation, "optimizer_updates": 0}), flush=True)
    gathered = [None] * world if distributed else [records]
    if distributed:
        torch.distributed.all_gather_object(gathered, records)
    capture = write_snapshot(model, output, initial_hash, torch, rank, distributed)
    # Hash after capture rather than trusting that `.step` was a non-boundary.
    final_hash = parameter_hash(model, torch, rank, distributed)
    if final_hash != initial_hash:
        raise AssertionError("Model weights changed during a no-update gradient measurement")
    if rank == 0:
        flattened = sorted((row for group in gathered for row in group), key=lambda row: (row["microstep"], row["rank"]))
        actual_indices = [index for row in flattened for index in row["pair_indices"]]
        if sorted(actual_indices) != list(range(32)):
            raise AssertionError(f"Actual update input is not a permutation of frozen32: {actual_indices}")
        summed_counts = {key: sum(row["counts"][key] for row in flattened) for key in counts}
        if summed_counts != counts:
            raise AssertionError("Microbatch target counts do not reconstruct frozen global denominators")
        versions = {key: importlib.metadata.version(key) for key in ("torch", "transformers", "accelerate", "deepspeed", "bitsandbytes")}
        summary = {"schema": 1, "status": "completed", "layout": args.layout,
                   "normalization": args.normalization, "world_size": world,
                   "batch_size_per_rank": batch_size, "gradient_accumulation_steps": accumulation,
                   "global_pair_count": 32, "pairs_sha256": file_sha(args.pairs),
                   "global_target_counts": counts, "microstep_target_counts_sum": summed_counts,
                   "pair_records": bundle.get("pair_records"), "pair_metadata": bundle.get("metadata"),
                   "initial_parameter_sha256": initial_hash, "final_parameter_sha256": final_hash,
                   "parameter_hash_method": "named BF16 header+bytes; ZeRO shards freshly all-gathered and unpadded, without touching hook state",
                   "weights_unchanged": final_hash == initial_hash, "optimizer_updates": 0,
                   "gradient_stage": capture["gradient_stage"], "gradient_file": capture["gradient_file"],
                   "gradient_sha256": capture["gradient_sha256"], "gradient_total_numel": capture["total_numel"],
                   "global_steps_before": initial_global_steps,
                   "global_steps_after": int(engine.global_steps) if distributed else 0,
                   "effective_scalar_objective": sum(row["effective_contribution"] for row in flattened),
                   "microsteps": flattened, "native_original_loss_check_rank0_first_microstep": native_check,
                   "method": method, "model_path": str(model_path),
                   "model_precision": "BF16", "loss_reduction_precision": "native; actual per-component dtypes recorded for every microstep",
                   "attention": "sdpa", "dropout": 0.0, "versions": versions,
                   "runner_sha256": file_sha(__file__),
                   "palu_source_sha256": file_sha(ROOT / "src/trainer/unlearn/palu.py"),
                   "config_sha256": file_sha(args.config), "deepspeed_config": ds_config,
                   "global_loss_scaling": "Each raw local mean times local_count/global_count * GA * world; backward /GA and DP /world",
                   "limitations": ["Model BF16 arithmetic and gradient accumulation remain layout dependent; native loss reduction dtypes are measured, not assumed.",
                                    "No clipping, optimizer update, FQ evaluation or training trajectory is measured."],
                   "duration_seconds": time.time() - started,
                   "peak_allocated_bytes_rank0": torch.cuda.max_memory_allocated(device),
                   "peak_reserved_bytes_rank0": torch.cuda.max_memory_reserved(device)}
        if distributed:
            summary["deepspeed_gradient_dtypes"] = {
                "communication_data_type": json_dtype(getattr(engine, "communication_data_type", None)),
                "gradient_accumulation_dtype": json_dtype(getattr(engine.optimizer, "gradient_accumulation_dtype", None)),
                "grad_accum_dtype": json_dtype(getattr(engine.optimizer, "grad_accum_dtype", None)),
                "optimizer_micro_step_id": getattr(engine.optimizer, "micro_step_id", None),
                "engine_micro_steps": engine.micro_steps,
                "engine_get_data_types": [str(value) for value in engine.get_data_types()]}
        write_json(output / "summary.json", summary)
        print(json.dumps({"status": "completed", "output": str(output), "optimizer_updates": 0,
                          "weights_unchanged": True, "effective_scalar_objective": summary["effective_scalar_objective"]}), flush=True)
    if distributed:
        torch.distributed.barrier()


def self_test(torch):
    """Unequal-token test: FP64 model with model-native FP32 output/CE reductions."""
    from types import SimpleNamespace
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(31)
    config = LlamaConfig(vocab_size=29, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, attention_dropout=0.0,
                         pad_token_id=0, eos_token_id=2, use_cache=False)
    config._attn_implementation = "eager"
    base = LlamaForCausalLM(config).double()
    reference = copy.deepcopy(base).eval()
    reference.requires_grad_(False)
    ids = torch.randint(3, 29, (32, 9))
    attention = torch.ones_like(ids)
    f_labels = torch.full_like(ids, -100)
    r_labels = torch.full_like(ids, -100)
    for i in range(32):
        count = i % 5 + 1
        f_labels[i, 2:2+count] = ids[i, 2:2+count]
        f_labels[i, 2+count:] = -200
        r_labels[i, 1:2 + i % 7] = ids[i, 1:2 + i % 7]
    bundle = {"forget": {"input_ids": ids, "attention_mask": attention, "labels": f_labels},
              "retain": {"input_ids": ids.flip(0), "attention_mask": attention, "labels": r_labels}}
    method = {"top_k": 7, "first_n": 3, "alpha": 0.2, "gamma": 1.0}
    counts = target_counts(bundle, 2, 3, torch)
    outputs = {}
    for normalization in ("native", "global"):
        for layout in ("single", "four"):
            model = copy.deepcopy(base).train()
            accumulation, world = (4, 1) if layout == "single" else (8, 4)
            total_loss = 0.0
            for microstep in range(accumulation):
                for rank in range(world):
                    indices = list(range(microstep * 8, (microstep + 1) * 8)) if layout == "single" else [microstep * 4 + rank]
                    inputs = slice_inputs(bundle, indices, "cpu")
                    components = raw_components(model, reference, inputs, top_k=7, first_n=3, eos=2, torch=torch)
                    if microstep == 0 and rank == 0:
                        check_original_loss(model, reference, inputs, components,
                                            SimpleNamespace(eos_token_id=2), method, torch)
                    local = target_counts(inputs, 2, 3, torch)
                    loss = weighted_loss(components, local, counts, gamma=1.0, alpha=0.2,
                                         normalization=normalization, accumulation=accumulation, world=world)
                    (loss / accumulation / world).backward()
                    total_loss += float(loss.detach()) / accumulation / world
            flat = torch.cat([parameter.grad.reshape(-1) for parameter in model.parameters()])
            outputs[f"{normalization}_{layout}"] = {"loss": total_loss, "gradient": flat}
    native_difference = torch.linalg.vector_norm(outputs["native_single"]["gradient"] - outputs["native_four"]["gradient"]).item()
    global_difference = torch.linalg.vector_norm(outputs["global_single"]["gradient"] - outputs["global_four"]["gradient"]).item()
    if native_difference < 1e-6 or global_difference > 1e-7:
        raise AssertionError(f"Unexpected FP64 gradient checks: native={native_difference}, global={global_difference}")
    print(json.dumps({"status": "self_test_passed", "global_counts": counts,
                      "native_gradient_l2_difference": native_difference,
                      "global_gradient_l2_difference": global_difference,
                      "losses": {key: value["loss"] for key, value in outputs.items()},
                      "native_copy_checked_against_original": True}), flush=True)


if __name__ == "__main__":
    main(arguments())
