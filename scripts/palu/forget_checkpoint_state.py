"""Small fixed-F probe and an owned, restartable ZeRO-3 stopping checkpoint.

All ranks call these functions together. CPU tests exercise the arithmetic and
state contract, not native CUDA/DeepSpeed save/restore equivalence.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import gradient_equivalence_step as gradient
import seed103_branch_training as state
from checkpoint_retention import atomic_json


IDENTITY_KEYS = ("config_sha256", "schedule_sha256",
                 "initial_parameter_sha256", "teacher_parameter_sha256")


def sample_stop_state_fields(container):
    """Copy and authenticate optional controller state without changing old runs."""
    keys = ("sample_stop_state", "sample_stop_state_sha256")
    if not any(key in container for key in keys):
        return {}
    from prepare_layout_schedule import canonical_sha
    if (not all(key in container for key in keys)
            or not isinstance(container["sample_stop_state"], dict)
            or canonical_sha(container["sample_stop_state"]) != container["sample_stop_state_sha256"]):
        raise ValueError("Sample stopping state is missing or differs from its canonical SHA256")
    # Do not let subsequent in-memory controller updates mutate saved client state.
    return json.loads(json.dumps({key: container[key] for key in keys}, allow_nan=False))


@contextmanager
def _evaluation_state(runtime, torch):
    rng = state.capture_rng(runtime, torch)
    models = (runtime["engine"], runtime["reference"])
    modes = [(module, module.training) for model in models for module in model.modules()]
    try:
        for model in models:
            model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        state.restore_rng(rng, runtime, torch)


def raw_forget_loss(model, reference, inputs, *, top_k, first_n, eos, torch,
                    include_common_kl=False):
    """Native PALU F, optionally common KL from the same two forwards.

    The legacy result is ``(F, sensitive_count)``. With common KL enabled it is
    ``(F, sensitive_count, KL, common_count)``. KL is student-to-Full-teacher,
    retaining native dtype and reduction order before converting the scalar.
    """
    forward = {key: inputs[key] for key in ("input_ids", "attention_mask")}
    forward["use_cache"] = False
    logits = model(**forward).logits[..., :-1, :].contiguous()
    teacher = reference(**forward).logits[..., :-1, :].contiguous()
    flat = logits.view(-1, logits.shape[-1])
    indices = teacher.view_as(flat).topk(min(top_k, flat.shape[-1]), dim=-1).indices
    selected = flat.gather(1, indices)
    target = flat.mean(dim=-1, keepdim=True).expand_as(selected)
    per_token = ((selected - target) ** 2).mean(dim=-1)
    mask = gradient.sensitive_mask(inputs["labels"], eos, first_n, torch).reshape(-1)
    count = int(mask.sum())
    loss = (per_token * mask).sum() / mask.sum().clamp(min=1)
    result = (float(loss.double()), count)
    if not include_common_kl:
        return result
    common = (inputs["labels"][..., 1:] == -200).reshape(-1)
    common_count = int(common.sum())
    if common_count:
        reference_log_prob = torch.nn.functional.log_softmax(teacher, dim=-1).view_as(flat)
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1).view_as(flat)
        per_token_kl = torch.sum(torch.exp(log_prob) * (log_prob - reference_log_prob), dim=-1)
        common_loss = (per_token_kl * common).sum() / common.sum().clamp(min=1)
    else:
        common_loss = torch.tensor(0.0, device=flat.device)
    return (*result, float(common_loss.double()), common_count)


def fixed_forget_probe(runtime, schedule, method, torch, *, include_common_kl=False):
    """Probe all fixed Forget samples using equal target-token weights.

    Rank r handles r, r+world, ... . The fixed pool must divide evenly across
    ranks, ensuring every rank executes the same number of student/teacher
    ZeRO forwards. The production pool is 200, world=4 (50 pairs of forwards).
    No generation, Retain samples or optimizer calls are performed. Common KL
    is optional for compatibility with frozen old plans; it reuses the F
    student/teacher logits and never requires another model forward.
    """
    samples = schedule["samples"]["forget"]
    rank, world = runtime["rank"], runtime["world"]
    if not samples or len(samples) % world:
        raise ValueError("Fixed Forget pool must divide evenly across ranks")
    if method.get("target_mode", "mean") != "mean" or int(method["top_k"]) <= 0:
        raise ValueError("Probe supports PALU mean-target with positive top_k")
    if any(parameter.requires_grad for parameter in runtime["reference"].parameters()):
        raise ValueError("The Full teacher must be frozen before probing")
    started = time.perf_counter()
    rows = []
    with _evaluation_state(runtime, torch):
        for index in range(rank, len(samples), world):
            sample = samples[index]
            ids = sample["input_ids"].unsqueeze(0).to(runtime["device"])
            inputs = {"input_ids": ids,
                      "attention_mask": ids.ne(runtime["tokenizer"].pad_token_id),
                      "labels": sample["labels"].unsqueeze(0).to(runtime["device"])}
            values = raw_forget_loss(runtime["model"], runtime["reference"], inputs,
                top_k=int(method["top_k"]), first_n=int(method["first_n"]),
                eos=runtime["tokenizer"].eos_token_id, torch=torch,
                include_common_kl=include_common_kl)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Fixed Forget probe produced a nonfinite loss")
            rows.append((index, *values))
        if runtime["distributed"]:
            rows = [row for shard in state.all_ranks(rows, runtime, torch) for row in shard]
    rows.sort()
    if [row[0] for row in rows] != list(range(len(samples))):
        raise ValueError("Fixed Forget probe omitted or duplicated sample indices")
    tokens = sum(row[2] for row in rows)
    expected = schedule["metadata"]["fixed_target_counts"]["sensitive"]
    if not tokens or tokens != expected:
        raise ValueError("Fixed Forget probe target count differs from frozen schedule")
    result = {"raw_sensitive_loss": sum(row[1] * row[2] for row in rows) / tokens,
            "sample_mean_raw_sensitive_loss": sum(row[1] for row in rows) / len(rows),
            "sensitive_token_count": tokens, "forget_count": len(rows),
            "forget_zero_sensitive_examples": sum(row[2] == 0 for row in rows),
            "fixed_dataset_sha256": schedule["metadata"]["fixed_dataset_sha256"],
            "elapsed_seconds": time.perf_counter() - started,
            "reduction": "fixed Forget pool; native per-sample F times shifted sensitive token count",
            "rng_and_module_modes_restored": True}
    if include_common_kl:
        common_tokens = sum(row[4] for row in rows)
        if common_tokens != schedule["metadata"]["fixed_target_counts"]["common"]:
            raise ValueError("Fixed Forget probe common target count differs from frozen schedule")
        result.update(
            raw_common_kl=sum(row[3] * row[4] for row in rows) / max(common_tokens, 1),
            sample_mean_raw_common_kl=sum(row[3] for row in rows) / len(rows),
            common_token_count=common_tokens,
            forget_zero_common_examples=sum(row[4] == 0 for row in rows),
            reduction="fixed Forget pool; native per-sample F/KL times respective shifted target token counts",
            fixed_loss_samples=[{
                "split": "forget", "dataset_index": index,
                "raw_sensitive_loss": loss, "raw_common_kl": common_loss,
                "raw_retain_nll": None, "sensitive_token_count": count,
                "common_token_count": common_count, "retain_token_count": None,
            } for index, loss, count, common_loss, common_count in rows])
    return result


def _broadcast_error(error, runtime, torch):
    errors = [error]
    torch.distributed.broadcast_object_list(errors, src=0)
    if errors[0]:
        raise ValueError(errors[0])


def validate_resume_files(folder, receipt):
    """Validate every owned restart file before full native deserialization."""
    sample_stop_state_fields(receipt)
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Missing owned restart checkpoint directory")
    folder = folder.resolve()
    step, world = receipt.get("step"), receipt.get("world_size")
    if (receipt.get("status") != "ready" or world != 4 or not isinstance(step, int)
            or step <= 0 or receipt.get("consumed_pairs") != step * 32
            or receipt.get("tag") != f"step{step}"
            or len(receipt.get("rank_fingerprints", [])) != world):
        raise ValueError("Restart receipt has an invalid complete-update boundary or world size")
    files = receipt.get("files", {})
    optimizers = [name for name in files if name.endswith("_optim_states.pt")]
    models = [name for name in files if name.endswith("_model_states.pt")]
    if len(optimizers) != world or not models:
        raise ValueError("Restart requires all four optimizer shards and native model state")
    tag = Path(receipt["tag"])
    if any(Path(name).parent != tag for name in optimizers + models):
        raise ValueError("Native model and optimizer files must use the receipt tag")
    for rank in range(world):
        if f"rank-{rank}.pt" not in files:
            raise ValueError("Restart requires per-rank RNG and engine counters")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid restart inventory path")
        path = folder / relative
        if (path.is_symlink() or any(parent.is_symlink() for parent in path.parents
                                    if parent.is_relative_to(folder))
                or not path.is_file() or path.stat().st_size != expected["bytes"]
                or gradient.file_sha(path) != expected["sha256"]):
            raise ValueError(f"Restart file changed or is missing: {name}")
    return folder


def _complete_boundary(runtime, step):
    if not runtime["distributed"] or runtime["world"] != 4 or runtime["accumulation"] != 8:
        raise ValueError("Native restart requires the four-rank, accumulation-eight layout")
    if (step <= 0 or int(runtime["engine"].global_steps) != step
            or int(runtime["engine"].micro_steps) != step * 8):
        raise ValueError("Restart must be saved after a complete optimizer update")


def save_resume_checkpoint(folder, runtime, manifest, step, torch, *,
                           probe=None, stop_reason="forget_stage_complete"):
    """Save native BF16/FP32/Adam/scheduler plus rank RNG; never prune it."""
    _complete_boundary(runtime, step)
    folder, rank = Path(folder), runtime["rank"]
    client = {key: manifest[key] for key in IDENTITY_KEYS}
    client.update(step=step, completed_updates=step, consumed_pairs=step * 32,
                  next_update=step + 1)
    client.update(sample_stop_state_fields(manifest))
    error = None
    if rank == 0:
        try:
            folder.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            error = str(exc)
    _broadcast_error(error, runtime, torch)
    before = state.training_fingerprint(runtime, torch)
    extra = state.rank_sidecar(runtime, torch)
    try:
        saved = runtime["engine"].save_checkpoint(str(folder), tag=f"step{step}",
            client_state=client, save_latest=False)
        if saved is False:
            raise ValueError("DeepSpeed reported checkpoint save failure")
        torch.save(extra, folder / f"rank-{rank}.pt")
    finally:
        state.restore_rng(extra["rng"], runtime, torch)
    after = state.training_fingerprint(runtime, torch)
    guards = state.all_ranks({"rank": rank, "unchanged": before == after,
        "before_sha256": state.fingerprint_sha(before),
        "after_sha256": state.fingerprint_sha(after)}, runtime, torch)
    fingerprints = state.all_ranks(before, runtime, torch)
    if not all(guard["unchanged"] for guard in guards):
        raise AssertionError("Saving restart changed persistent training state")
    receipt = None
    if rank == 0:
        try:
            files = {str(path.relative_to(folder)): {
                "bytes": path.stat().st_size, "sha256": gradient.file_sha(path)}
                for path in sorted(folder.rglob("*")) if path.is_file()}
            receipt = {"schema": 1, "status": "ready", "tag": f"step{step}",
                "world_size": 4, **client, "stop_reason": stop_reason, "probe": probe,
                "files": files, "rank_fingerprints": fingerprints, "save_guards": guards,
                "created_unix": time.time(),
                "restore_contract": "fresh engines; same world size, software, config and teacher; "
                                    "restore native student/master/Adam/scheduler plus rank RNG/counters",
                "excluded_ephemeral_state": "spent gradient caches; no pending sample gradients",
                "limitations": "Native GPU restore equivalence is checked only when actually loaded; "
                               "CPU tests and file validation are not a CUDA resume test."}
            validate_resume_files(folder, receipt)
            atomic_json(folder / "receipt.json", receipt)
        except Exception as exc:
            error = str(exc)
    _broadcast_error(error, runtime, torch)
    payload = [receipt]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def load_resume_checkpoint(folder, runtime, manifest, torch):
    """Strict future continuation helper, without selecting a retention policy."""
    folder = Path(folder)
    if folder.is_symlink():
        raise ValueError("Restart folder cannot be a symlink")
    folder = folder.resolve()
    receipt = json.loads((folder / "receipt.json").read_text())
    controller_state = sample_stop_state_fields(receipt)
    expected_controller_state = sample_stop_state_fields(manifest)
    if expected_controller_state and expected_controller_state != controller_state:
        raise ValueError("Restart and current sample stopping state differ")
    if (runtime["world"] != 4 or not runtime["distributed"]
            or runtime["accumulation"] != 8 or int(runtime["engine"].global_steps) != 0):
        raise ValueError("Restore requires a fresh four-rank accumulation-eight engine")
    if any(receipt[key] != manifest[key] for key in IDENTITY_KEYS):
        raise ValueError("Restart and current configuration/schedule/initial teacher differ")
    error = None
    if runtime["rank"] == 0:
        try:
            validate_resume_files(folder, receipt)
        except Exception as exc:
            error = str(exc)
    _broadcast_error(error, runtime, torch)
    with state.trusted_native_checkpoint_loading(runtime, receipt, folder, torch):
        path, client = runtime["engine"].load_checkpoint(str(folder), tag=receipt["tag"],
            load_module_strict=True, load_optimizer_states=True,
            load_lr_scheduler_states=True, load_module_only=False)
    if (not path or sample_stop_state_fields(client) != controller_state
            or any(client.get(key) != receipt[key]
                       for key in (*IDENTITY_KEYS, "step", "completed_updates",
                                   "consumed_pairs", "next_update", *controller_state))):
        raise ValueError("DeepSpeed did not restore the complete expected restart state")
    sidecar = folder / f"rank-{runtime['rank']}.pt"
    # As in the existing fork loader, only an owned, hash-validated sidecar is read.
    extra = torch.load(sidecar, map_location="cpu", weights_only=False)
    state.restore_sidecar(extra, runtime, torch)
    actual = state.training_fingerprint(runtime, torch)
    expected = receipt["rank_fingerprints"][runtime["rank"]]
    guards = state.all_ranks({"rank": runtime["rank"], "unchanged": actual == expected,
        "changed_sections": [key for key in expected if expected[key] != actual.get(key)]},
        runtime, torch)
    if not all(guard["unchanged"] for guard in guards):
        raise AssertionError("Native restart failed persistent training-state equality: " + str(guards))
    return receipt
