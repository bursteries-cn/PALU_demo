#!/usr/bin/env python3
"""Seed103 four-GPU token-normalized prefix and exact-state continuations.

The parent loop remains shared with the existing online-epoch experiment.
A native ZeRO checkpoint is temporary; each branch loads it in a fresh process.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout_token_training as base
import gradient_equivalence_step as gradient
import checkpoint_retention as retention

FORK_STEP = 6
BRANCH_GAMMAS = {"continue": 1.0, "repair": 0.0, "weak": 0.1}
TAG = "step6"


def branch_gamma(step, arm):
    if arm not in BRANCH_GAMMAS or not 1 <= step <= 60:
        raise ValueError("Expected one of the three registered branches and update 1..60")
    return 1.0 if step <= FORK_STEP else BRANCH_GAMMAS[arm]


def capture_rng(runtime, torch):
    import numpy as np
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch_cpu": torch.random.get_rng_state()}
    if runtime["device"].type == "cuda":
        result["torch_cuda"] = torch.cuda.get_rng_state(runtime["device"])
    return result


def restore_rng(state, runtime, torch):
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if runtime["device"].type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], runtime["device"])


def training_fingerprint(runtime, torch):
    """Persistent causal state at a complete update; exclude spent grad caches.

    ZeRO param_groups can be temporarily empty. Read moments by their persistent
    FP32 master keys instead, so a silently missing Adam shard cannot pass.
    """
    from online_epoch_evaluation import runtime_fingerprint, _state_value
    result = runtime_fingerprint(runtime, torch)
    for name in ("student", "teacher"):
        result[name].pop("gradients", None)
    state = result["optimizer"]
    for key in ("grad_partitions_flat_buffer", "averaged_gradients", "fp32_partitioned_groups_flat_grad"):
        state.pop(key, None)
    optimizer = runtime["optimizer"]
    inner = getattr(optimizer, "optimizer", optimizer)
    masters = getattr(optimizer, "fp32_partitioned_groups_flat", None)
    if masters is None:
        masters = [p for group in inner.param_groups for p in group["params"]]
    state["persistent_parameter_states"] = [_state_value(inner.state.get(p, {}), torch) for p in masters]
    return result


def fingerprint_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rank_sidecar(runtime, torch):
    engine, optimizer = runtime["engine"], runtime["optimizer"]
    return {"rng": capture_rng(runtime, torch),
            "engine_counters": {key: getattr(engine, key) for key in
                ("global_steps", "global_samples", "micro_steps", "skipped_steps", "_is_gradient_accumulation_boundary") if hasattr(engine, key)},
            "optimizer_counters": {key: getattr(optimizer, key) for key in
                ("micro_step_id", "is_gradient_accumulation_boundary", "overflow") if hasattr(optimizer, key)}}


def restore_sidecar(state, runtime, torch):
    for key, value in state["engine_counters"].items():
        setattr(runtime["engine"], key, value)
    for key, value in state["optimizer_counters"].items():
        setattr(runtime["optimizer"], key, value)
    restore_rng(state["rng"], runtime, torch)


def all_ranks(value, runtime, torch):
    values = [None] * runtime["world"]
    torch.distributed.all_gather_object(values, value)
    return values


def validate_fork_files(study, receipt):
    folder = Path(study).resolve() / "fork_weights"
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Missing owned native fork checkpoint")
    if receipt.get("status") != "ready" or receipt.get("fork_step") != FORK_STEP or receipt.get("consumed_pairs") != 192 or receipt.get("world_size") != 4:
        raise ValueError("Fork receipt has an invalid boundary or world size")
    if len(receipt.get("rank_fingerprints", [])) != 4:
        raise ValueError("Missing rank fingerprints")
    names = receipt.get("files", {})
    optimizer_files = [name for name in names if name.endswith("optim_states.pt")]
    if len(optimizer_files) != 4:
        raise ValueError("Native fork must include all four ZeRO optimizer shards")
    for name, inventory in names.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid fork inventory path")
        path = folder / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != inventory["bytes"] or gradient.file_sha(path) != inventory["sha256"]:
            raise ValueError(f"Fork file changed or is missing: {name}")
    for rank in range(4):
        if f"rank-{rank}.pt" not in names:
            raise ValueError("Missing per-rank RNG/counter sidecar")
    return folder


@contextmanager
def trusted_native_checkpoint_loading(runtime, receipt, folder, torch):
    """Read only this run's verified native DS files with full deserialization.

    DS 0.15 does not specify torch.load(weights_only=...), whereas PyTorch 2.6+
    defaults to weights_only=True. Native optimizer state includes DS enums and
    cannot use that restricted tensor-only reader. Override just this student's
    checkpoint reader for the duration of its native restore, never torch.load.
    """
    folder = Path(folder)
    if folder.is_symlink():
        raise ValueError("Native fork folder cannot be a symlink")
    root = folder.resolve()
    allowed = {}
    for name, item in receipt["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid native fork inventory path")
        if (relative.parent == Path(receipt["tag"]) and
                relative.name.endswith(("_model_states.pt", "_optim_states.pt"))):
            allowed[root / relative] = item
    if not allowed:
        raise ValueError("No inventoried native DeepSpeed checkpoint files")
    checkpoint_engine = runtime["engine"].checkpoint_engine
    own_attribute = "load" in vars(checkpoint_engine)
    original = checkpoint_engine.load

    def load_owned_native(path, map_location=None):
        target = Path(path).absolute()
        if (target not in allowed or target.is_symlink() or
                not target.resolve().is_relative_to(root) or
                any(parent.is_symlink() for parent in target.parents
                    if parent.is_relative_to(root))):
            raise ValueError("Native checkpoint load is outside the verified fork inventory")
        expected = allowed[target]
        # Hash and deserialize the same open file; changing or substituting a
        # native checkpoint after the earlier global inventory check must fail.
        with target.open("rb") as handle:
            if os.fstat(handle.fileno()).st_size != expected["bytes"]:
                raise ValueError("Native checkpoint changed since fork validation")
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError("Native checkpoint changed since fork validation")
            handle.seek(0)
            return torch.load(handle, map_location=map_location, weights_only=False)

    checkpoint_engine.load = load_owned_native
    try:
        yield
    finally:
        if own_attribute:
            checkpoint_engine.load = original
        else:
            del checkpoint_engine.load


def save_fork(study, runtime, manifest, torch):
    rank, engine = runtime["rank"], runtime["engine"]
    if int(engine.global_steps) != FORK_STEP or int(engine.micro_steps) != FORK_STEP * 8:
        raise AssertionError("Fork must be a complete sixth update with 192 consumed pairs")
    folder = Path(study) / "fork_weights"
    audit = Path(study) / "fork"
    if rank == 0:
        folder.mkdir(exist_ok=False)
        audit.mkdir(exist_ok=False)
    torch.distributed.barrier()
    before = training_fingerprint(runtime, torch)
    extra = rank_sidecar(runtime, torch)
    client = {"fork_step": FORK_STEP, "consumed_pairs": 192, "schedule_sha256": manifest["schedule_sha256"],
              "config_sha256": manifest["config_sha256"], "initial_parameter_sha256": manifest["initial_parameter_sha256"],
              "teacher_parameter_sha256": manifest["teacher_parameter_sha256"]}
    # All ranks participate. Native DS carries BF16 weights, FP32 masters,
    # Adam moments/steps, scheduler and its saved engine counters.
    engine.save_checkpoint(str(folder), tag=TAG, client_state=client, save_latest=False)
    torch.save(extra, folder / f"rank-{rank}.pt")
    restore_rng(extra["rng"], runtime, torch)
    after = training_fingerprint(runtime, torch)
    guard = {"rank": rank, "unchanged": before == after,
             "before_sha256": fingerprint_sha(before), "after_sha256": fingerprint_sha(after)}
    guards = all_ranks(guard, runtime, torch)
    fingerprints = all_ranks(before, runtime, torch)
    if not all(item["unchanged"] for item in guards):
        raise AssertionError("Saving native fork modified persistent training state")
    if rank == 0:
        files = {str(path.relative_to(folder)): {"bytes": path.stat().st_size, "sha256": gradient.file_sha(path)}
                 for path in sorted(folder.rglob("*")) if path.is_file()}
        receipt = {"schema": 1, "status": "ready", "tag": TAG, "world_size": 4, **client,
                   "rank_fingerprints": fingerprints, "save_guards": guards, "files": files,
                   "created_unix": time.time(),
                   "excluded_ephemeral_state": "spent gradient caches at a complete update; no pending sample gradients",
                   "restore_contract": "fresh ZeRO engines, native model/master/Adam/scheduler restore plus rank RNG and omitted counters"}
        retention.atomic_json(audit / "receipt.json", receipt)
        validate_fork_files(study, receipt)
    torch.distributed.barrier()


def load_fork(study, runtime, manifest, arm, torch):
    rank = runtime["rank"]
    receipt_path = Path(study) / "fork/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if any(receipt[key] != manifest[key] for key in ("config_sha256", "schedule_sha256", "initial_parameter_sha256", "teacher_parameter_sha256")):
        raise ValueError("Native fork and current seed103 inputs/configuration differ")
    # Read the large inventory once, then broadcast the outcome; never silently
    # accept DS's fallback from missing optimizer shards to BF16 initialization.
    error = None
    if rank == 0:
        try:
            validate_fork_files(study, receipt)
        except Exception as exc:
            error = str(exc)
    errors = [error]
    torch.distributed.broadcast_object_list(errors, src=0)
    if errors[0]:
        raise ValueError(errors[0])
    with trusted_native_checkpoint_loading(runtime, receipt, Path(study) / "fork_weights", torch):
        path, client = runtime["engine"].load_checkpoint(str(Path(study) / "fork_weights"), tag=receipt["tag"],
            load_module_strict=True, load_optimizer_states=True, load_lr_scheduler_states=True, load_module_only=False)
    if not path or any(client.get(key) != receipt[key] for key in
            ("fork_step", "consumed_pairs", "schedule_sha256", "config_sha256")):
        raise ValueError("DeepSpeed did not restore the expected complete native checkpoint")
    # Only this experiment's own hash-validated native sidecar is deserialized.
    extra = torch.load(Path(study) / f"fork_weights/rank-{rank}.pt", map_location="cpu", weights_only=False)
    restore_sidecar(extra, runtime, torch)
    base.assert_learning_rate([float(group["lr"]) for group in runtime["optimizer"].param_groups], FORK_STEP,
                              runtime, json.loads((Path(study) / "frozen/config.json").read_text())["trainer"]["args"])
    restored = training_fingerprint(runtime, torch)
    expected = receipt["rank_fingerprints"][rank]
    guard = {"rank": rank, "unchanged": restored == expected,
             "changed_sections": [key for key in expected if expected[key] != restored.get(key)],
             "before_sha256": fingerprint_sha(expected), "after_sha256": fingerprint_sha(restored)}
    guards = all_ranks(guard, runtime, torch)
    if not all(item["unchanged"] for item in guards):
        raise AssertionError("Fork restore failed persistent model/master/Adam/scheduler/RNG equivalence: " + str(guards))
    if rank == 0:
        retention.atomic_json(Path(study) / arm / "fork_restore_receipt.json", {
            "schema": 1, "status": "verified", "arm": arm, "branch_gamma": BRANCH_GAMMAS[arm],
            "fork_step": FORK_STEP, "completed_updates": FORK_STEP, "consumed_pairs": 192,
            "fork_receipt_sha256": gradient.file_sha(receipt_path), "rank_guards": guards,
            "native_load_policy": "weights_only=False scoped to hash-verified owned native files; global torch.load unchanged"})
    return receipt


class BranchControl:
    def __init__(self, study, phase, arm=None):
        self.study, self.phase, self.arm = Path(study).resolve(), phase, arm
        self.run_initial_evaluation = phase == "prefix"

    def select_windows(self, windows):
        return windows[:FORK_STEP] if self.phase == "prefix" else windows[FORK_STEP:]

    def gamma_for_step(self, step):
        return 1.0 if self.phase == "prefix" else branch_gamma(step, self.arm)

    def initialize(self, runtime, manifest, config, schedule, args, torch):
        self.runtime, self.manifest, self.torch = runtime, manifest, torch
        manifest.update(requested_updates=60, final_model_step=60, planned_updates=60, smoke_prefix=False,
            experiment="seed103_token_normalized_three_branches", fork_step=FORK_STEP, fork_consumed_pairs=192,
            eval_every_epochs=getattr(args, "eval_every_epochs", 1),
            evaluation_backend=getattr(args, "eval_backend", "legacy"),
            phase=self.phase, branch_gamma=None if self.phase == "prefix" else BRANCH_GAMMAS[self.arm],
            fixed_loss_total_definition="canonical F+2R+2KL; gamma=1 for every fixed diagnostic",
            training_gamma_definition="gamma=1 through update6; branch gamma from update7",
            limitations=["One selected seed; no seed variance estimate.",
                "Common KL is retained in every branch; gamma0 is not retain-only repair.",
                "Native four-GPU BF16/FP32 master, communication and clipping are shared."])
        if self.phase == "branch":
            load_fork(self.study, runtime, manifest, self.arm, torch)
            prefix = self.study / "prefix"
            source = json.loads((prefix / "manifest.json").read_text())
            if source["status"] != "prefix_completed" or source["completed_updates"] != FORK_STEP:
                raise ValueError("The common prefix was not committed at update6")
            manifest["checkpoints"] = [copy.deepcopy(source["checkpoints"][0])]
            manifest["completed_updates"] = FORK_STEP
            if runtime["rank"] == 0:
                for name in ("loss_history.jsonl", "training_pairs.jsonl"):
                    shutil.copyfile(prefix / name, args.output / name)
                source_row = retention.read_receipt(prefix, source, source["checkpoints"][0])
                row = retention.commit_observation(args.output, manifest, manifest["checkpoints"][0], source_row,
                    source_row["evaluation_dir"], prefix / "fixed_loss_samples.jsonl")
                retention.rebuild_aggregates(args.output, [row])
            torch.distributed.barrier()
        if runtime["rank"] == 0:
            base.atomic_json(args.output / "manifest.json", manifest)

    def configure_online(self, online):
        if self.phase == "prefix":
            online.events = {0: online.events[0]}
        else:
            if self.runtime["rank"] == 0:
                online.rows = [retention.read_receipt(online.output, self.manifest, self.manifest["checkpoints"][0])]
            self.manifest.update(evaluated_steps=[0], evaluated_states=1)

    def finish(self, runtime, manifest, args, torch):
        if self.phase != "prefix":
            return False
        save_fork(self.study, runtime, manifest, torch)
        manifest.update(status="prefix_completed", completed_updates=FORK_STEP, consumed_pairs=192,
            pending_pairs=0, fork_receipt_sha256=gradient.file_sha(self.study / "fork/receipt.json"),
            completed_unix=time.time(), checkpoint_retention="no_prefix_model_copy; native_fork_is_temporary")
        if runtime["rank"] == 0:
            base.atomic_json(args.output / "manifest.json", manifest)
        torch.distributed.barrier()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prefix", "branch"), required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(BRANCH_GAMMAS))
    args = parser.parse_args()
    if (args.phase == "branch") != (args.arm is not None):
        parser.error("--arm is required only for branch phase")
    import torch
    study = args.study.resolve()
    description = json.loads((study / "frozen/manifest.json").read_text())
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise ValueError("Launch with torchrun --nproc_per_node=4")
    options = argparse.Namespace(config=study / "frozen/config.json", schedule=study / "frozen/schedule.pt",
        output=study / ("prefix" if args.phase == "prefix" else args.arm), layout="four", seed=103,
        normalization="global", padding_scope="microbatch", online_evaluation=True,
        eval_every_epochs=description.get("eval_every_epochs", 1),
        eval_backend=description.get("evaluation_backend", "legacy"),
        delete_evaluated_intermediate=True, max_updates=None)
    base.train(options, torch, control=BranchControl(study, args.phase, args.arm))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
