"""Pause PALU at actual epoch edges, evaluate, commit, prune, then resume.

No optimizer is rebuilt and no pending gradient is flushed. The legacy backend
mirrors rank zero's forwards. The sharded backend distributes distinct fixed
examples and original TOFU batches, then merges per-example observations before
native metric aggregation. Existing studies retain their frozen backend/cadence.
"""
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
from types import SimpleNamespace


def epoch_events(total_pairs=1920, forget_size=200, effective_batch=32, every_epochs=1):
    if min(total_pairs, forget_size, effective_batch, every_epochs) <= 0 or int(every_epochs) != every_epochs:
        raise ValueError("Training sizes must be positive")
    interval = forget_size * int(every_epochs)
    exposures = [0, *range(interval, total_pairs + 1, interval)]
    if exposures[-1] != total_pairs:
        exposures.append(total_pairs)
    return [{"step": consumed // effective_batch,
             "consumed_pairs": consumed,
             "updated_pairs": consumed // effective_batch * effective_batch,
             "pending_pairs": consumed % effective_batch,
             "actual_data_epochs": consumed / forget_size,
             "weight_update_epochs": consumed // effective_batch * effective_batch / forget_size,
             "completed_data_epochs": consumed // forget_size,
             "source_epoch_index": (consumed + forget_size - 1) // forget_size,
             "evaluation_kind": "initial" if not consumed else "final_tail" if consumed % forget_size else "epoch_end"}
            for consumed in exposures]


def map_tensors(value, function, torch):
    if torch.is_tensor(value):
        return function(value)
    if isinstance(value, dict):
        return {key: map_tensors(item, function, torch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(map_tensors(item, function, torch) for item in value)
    if isinstance(value, list):
        return [map_tensors(item, function, torch) for item in value]
    return value


def _tensor_sha(value, torch):
    digest = hashlib.sha256()
    flat = value.detach().reshape(-1)
    # Bound host memory even for a multi-GiB embedding or Adam state.
    chunk = max(1, (8 << 20) // max(1, value.element_size()))
    for start in range(0, flat.numel(), chunk):
        digest.update(flat[start:start + chunk].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": digest.hexdigest()}


def _state_value(value, torch):
    if torch.is_tensor(value):
        return _tensor_sha(value, torch)
    if isinstance(value, dict):
        return {str(key): _state_value(item, torch) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_state_value(item, torch) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def runtime_fingerprint(runtime, torch):
    """Read exact training tensor contents, counters, RNG and module modes.

    This deliberately reads live optimizer buffers, not optimizer.state_dict(),
    which can consolidate or otherwise alter ZeRO state. Inference-only gather
    caches are excluded; trainable shards and accumulated gradients are included.
    """
    import numpy as np
    device = runtime["device"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {}
    for tag, wrapper in (("student", runtime["engine"]), ("teacher", runtime["reference"])):
        model = getattr(wrapper, "module", wrapper)
        result[tag] = {
            "parameters": {name: _tensor_sha(getattr(parameter, "ds_tensor", parameter), torch)
                           for name, parameter in model.named_parameters()},
            "gradients": {name: _state_value(parameter.grad, torch) for name, parameter in model.named_parameters()},
            "buffers": {name: _tensor_sha(buffer, torch) for name, buffer in model.named_buffers()},
            "module_modes": {name: module.training for name, module in wrapper.named_modules()},
            "use_cache": getattr(model.config, "use_cache", None),
        }
    optimizer = runtime["optimizer"]
    underlying = getattr(optimizer, "optimizer", optimizer)
    result["optimizer"] = {
        "groups": [{key: _state_value(value, torch) for key, value in group.items() if key != "params"}
                   for group in underlying.param_groups],
        "state": [[_state_value(underlying.state.get(parameter, {}), torch) for parameter in group["params"]]
                  for group in underlying.param_groups],
    }
    for key in ("fp32_partitioned_groups_flat", "grad_partitions_flat_buffer", "averaged_gradients"):
        if hasattr(optimizer, key):
            value = getattr(optimizer, key)
            result["optimizer"][key] = _state_value(value, torch)
            if key == "fp32_partitioned_groups_flat":
                result["optimizer"][key + "_grad"] = [_state_value(item.grad, torch) for item in value]
    for key in ("micro_step_id", "is_gradient_accumulation_boundary", "overflow"):
        if hasattr(optimizer, key):
            result["optimizer"][key] = _state_value(getattr(optimizer, key), torch)
    result["scheduler"] = _state_value(runtime["scheduler"].state_dict(), torch)
    result["engine_counters"] = {key: getattr(runtime["engine"], key) for key in
        ("global_steps", "global_samples", "micro_steps", "skipped_steps") if hasattr(runtime["engine"], key)}
    result["rng"] = {"python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                     "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                     "torch_cpu": _tensor_sha(torch.random.get_rng_state(), torch)}
    if device.type == "cuda":
        result["rng"]["torch_cuda"] = _tensor_sha(torch.cuda.get_rng_state(device), torch)
    return result


def _cache_length(cache):
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(cache[0][0].shape[-2]) if len(cache) else 0


class CoordinatedForward:
    """Coordinate the existing ZeRO shards without a fifth GPU or extra model.

    The top module's __call__ is wrapped rather than forward: ZeRO pre-hooks
    gather parameters before forward, so workers must be notified before them.
    KV tensors are never sent through the object broadcast.
    """

    def __init__(self, models, device, torch):
        self.models, self.device, self.torch = models, device, torch
        self.calls = 0
        self.generated_tokens = 0

    def _broadcast(self, command):
        payload = [command]
        self.torch.distributed.broadcast_object_list(payload, src=0)
        return payload[0]

    def send_forward(self, tag, args, kwargs):
        torch = self.torch
        values = dict(kwargs)
        present = "past_key_values" in values
        cache = values.pop("past_key_values", None)
        kind = "none" if cache is None else "dynamic" if hasattr(cache, "get_seq_length") else "legacy"
        if kind == "dynamic" and type(cache).__name__ != "DynamicCache":
            raise ValueError("Online ZeRO evaluation supports the source DynamicCache/legacy cache only")
        length = _cache_length(cache)
        cpu = lambda tensor: tensor.detach().cpu()
        self._broadcast({"op": "forward", "model": tag,
            "args": map_tensors(args, cpu, torch), "kwargs": map_tensors(values, cpu, torch),
            "cache": {"present": present, "kind": kind, "length": length},
            "autocast": torch.is_autocast_enabled(self.device.type),
            "autocast_dtype": torch.get_autocast_dtype(self.device.type)})
        self.calls += 1
        self.generated_tokens += int(length > 0)

    @contextmanager
    def root_models(self):
        originals = {}
        try:
            for tag, model in self.models.items():
                original = type(model)
                originals[tag] = original
                rpc = self

                def make_class(base, model_tag):
                    class OnlineForward(base):
                        def __call__(instance, *args, **kwargs):
                            rpc.send_forward(model_tag, args, kwargs)
                            return base.__call__(instance, *args, **kwargs)

                        def generate(instance, *args, **kwargs):
                            generation = kwargs.get("generation_config") or instance.generation_config
                            if kwargs.get("num_beams", generation.num_beams) != 1:
                                raise ValueError("Source TOFU generation uses one beam; beam cache reordering is unsupported")
                            if kwargs.get("do_sample", generation.do_sample):
                                raise ValueError("Source TOFU generation must explicitly set do_sample=False")
                            # Only root executes the generation loop; workers
                            # mirror its forwards, including locally held caches.
                            kwargs["synced_gpus"] = False
                            return base.generate(instance, *args, **kwargs)
                    return OnlineForward

                model.__class__ = make_class(original, tag)
            yield
        finally:
            for tag, original in originals.items():
                self.models[tag].__class__ = original

    def workers(self):
        from transformers.cache_utils import DynamicCache
        torch, caches = self.torch, {}
        while True:
            command = self._broadcast(None)
            if command["op"] == "finish":
                caches.clear()
                return command.get("error")
            if command["op"] != "forward":
                raise ValueError("Unexpected online forward command")
            tag, spec = command["model"], command["cache"]
            move = lambda tensor: tensor.to(self.device)
            args = map_tensors(command["args"], move, torch)
            kwargs = map_tensors(command["kwargs"], move, torch)
            cache = None
            if spec["kind"] != "none":
                if spec["length"] == 0:
                    cache = DynamicCache() if spec["kind"] == "dynamic" else ()
                else:
                    cache = caches.get(tag)
                    if cache is None or _cache_length(cache) != spec["length"]:
                        raise AssertionError("Worker KV cache differs from root sequence length")
                    if spec["kind"] == "dynamic" and not isinstance(cache, DynamicCache):
                        cache = DynamicCache.from_legacy_cache(cache)
                    if spec["kind"] == "legacy" and isinstance(cache, DynamicCache):
                        cache = cache.to_legacy_cache()
            if spec["present"]:
                kwargs["past_key_values"] = cache
            with torch.autocast(device_type=self.device.type, dtype=command["autocast_dtype"], enabled=command["autocast"]):
                outputs = self.models[tag](*args, **kwargs)
            caches[tag] = getattr(outputs, "past_key_values", None)
            self.calls += 1
            self.generated_tokens += int(spec["length"] > 0)
            del outputs, args, kwargs, cache

    def finish(self, error=None):
        self._broadcast({"op": "finish", "error": error})


class FrozenSplit:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return dict(self.samples[index], index=index)


def build_probe(config, schedule, runtime, output, torch):
    """Reuse native diagnostic methods, without invoking a Trainer constructor.

    The constructor would build another Full teacher/optimizer and reject ZeRO.
    This object exposes evaluation only and owns no additional model weights.
    """
    from omegaconf import OmegaConf
    from data import get_collators
    from evals import get_evaluators
    from trainer.palu_epoch_eval import PALUEpochTrainer
    model = runtime["model"]
    reference = getattr(runtime["reference"], "module", runtime["reference"])
    cfg = OmegaConf.create(copy.deepcopy(config))
    cfg.eval.tofu.metrics = {key: cfg.eval.tofu.metrics[key] for key in ("forget_quality", "model_utility")}
    probe = object.__new__(PALUEpochTrainer)
    probe.model = probe.model_wrapped = model
    probe.ref_model = reference
    probe.tokenizer = runtime["tokenizer"]
    probe.template_args = cfg.model.template_args
    probe.args = SimpleNamespace(seed=int(config["trainer"]["args"].get("seed", 103)),
                                 device=runtime["device"])
    probe.accelerator = SimpleNamespace(device=runtime["device"], num_processes=1)
    probe.state = SimpleNamespace(global_step=0, epoch=0.0)
    probe.train_dataset = SimpleNamespace(**{split: FrozenSplit(schedule["samples"][split]) for split in ("forget", "retain")})
    probe.data_collator = get_collators(cfg.collator, tokenizer=probe.tokenizer)
    probe.evaluators = get_evaluators(cfg.eval, template_args=probe.template_args, model=model, tokenizer=probe.tokenizer)
    method = config["trainer"]["method_args"]
    for name, source in (("gamma", "gamma"), ("alpha", "alpha"), ("top_k", "top_k"),
                         ("num_sensitive_tokens", "first_n"), ("target_mode", "target_mode"),
                         ("retain_loss_type", "retain_loss_type"), ("non_sensitive_constraint_type", "non_sensitive_constraint_type")):
        setattr(probe, name, method[source])
    probe.epoch_options = {"forget_samples": "all", "retain_samples": "all", "model_name": config["model"]["handler"]}
    probe.epoch_output = Path(output) / "online_evaluation_workspace"
    probe.epoch_output.mkdir(parents=True, exist_ok=True)
    probe._epoch_loss_forget_only = False
    probe._last_epoch_evaluated_step = None
    probe._epoch_fixed = probe._epoch_fixed_manifest = probe._epoch_reference_sha256 = None
    probe._epoch_context = {"epoch_index": 0, "microbatches": 0, "expected_microbatches": 0, "epoch_end_seen": False}
    probe._palu_diagnostic_mode = True
    # PALU records its seven component tensors only with a non-None monitor.
    # Diagnostic mode suppresses training callbacks and pairing writes.
    probe._palu_monitor = SimpleNamespace(record=lambda *_: None)
    probe.compute_loss_context_manager = nullcontext
    probe._prepare_inputs = lambda value: map_tensors(value, lambda tensor: tensor.to(runtime["device"]), torch)
    probe.log = lambda *_args, **_kwargs: None
    return probe


def evaluate_probe(config, schedule, runtime, output, torch, *, backend="legacy", step=0,
                   epoch=0.0, epoch_index=0, expected_fixed_hash=None, probe=None):
    """Shared online/standalone entry; all ranks participate and restore RNG/modes."""
    from trainer.palu_diagnostics import preserve_rng_and_modes
    rank, distributed = runtime["rank"], runtime["distributed"]
    if backend not in ("legacy", "sharded") or backend == "sharded" and not distributed:
        raise ValueError("Unknown backend or sample-parallel evaluation without distributed ranks")
    models = {"student": runtime["model"], "reference": getattr(runtime["reference"], "module", runtime["reference"])}
    rpc = CoordinatedForward(models, runtime["device"], torch) if distributed and backend == "legacy" else None
    parallel = None
    if backend == "sharded":
        from sharded_epoch_evaluation import ShardedEvaluation
        parallel = ShardedEvaluation(torch)
    row, error = None, None
    with preserve_rng_and_modes([runtime["engine"], runtime["reference"]], runtime["device"]), torch.no_grad():
        if rank == 0 or parallel:
            try:
                if probe is None:
                    destination = Path(output) if rank == 0 else Path(output) / f"rank-{rank}"
                    probe = build_probe(config, schedule, runtime, destination, torch)
                with parallel.activate(probe, models) if parallel else rpc.root_models() if rpc else nullcontext():
                    probe._prepare_fixed_data()
                    if expected_fixed_hash and probe._epoch_fixed_manifest["fixed_dataset_sha256"] != expected_fixed_hash:
                        raise AssertionError("Online fixed datasets differ from the frozen schedule")
                    probe.state.global_step, probe.state.epoch = step, epoch
                    probe._epoch_context.update(epoch_index=epoch_index, epoch_end_seen=True)
                    probe.evaluate()
                if rank == 0:
                    row = json.loads((probe.epoch_output / "epoch_history.jsonl").read_text().splitlines()[-1])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                # A local error inside sharded model work may leave peers in a
                # collective. Let torchrun abort the process group immediately.
                if parallel:
                    raise
            finally:
                if rpc:
                    rpc.finish(error)
        else:
            error = rpc.workers()
    if error:
        raise RuntimeError(error)
    stats = parallel.stats if parallel else {"rank": rank, "forward_calls": rpc.calls if rpc else None,
                                            "cached_decode_forwards": rpc.generated_tokens if rpc else None}
    return probe, row, stats


class OnlineEpochEvaluation:
    def __init__(self, args, config, schedule, runtime, manifest, torch):
        self.args, self.config, self.schedule = args, config, schedule
        self.runtime, self.manifest, self.torch = runtime, manifest, torch
        self.output = args.output.resolve()
        self.events = {event["consumed_pairs"]: event for event in epoch_events(
            total_pairs=32 * manifest["requested_updates"], forget_size=schedule["metadata"]["data_sizes"]["forget"],
            every_epochs=getattr(args, "eval_every_epochs", 1))}
        self.backend = getattr(args, "eval_backend", "legacy")
        self.probe, self.rows = None, []

    def at(self, consumed_pairs, completed_updates, save_checkpoint):
        event = self.events.get(consumed_pairs)
        if event is None:
            return
        if event["step"] != completed_updates:
            raise AssertionError("Epoch evaluation is on the wrong side of an optimizer update")
        import checkpoint_retention as retention
        runtime, manifest, torch = self.runtime, self.manifest, self.torch
        rank, distributed = runtime["rank"], runtime["distributed"]
        before = runtime_fingerprint(runtime, torch)
        if consumed_pairs:
            checkpoint = save_checkpoint(event)
            manifest["checkpoints"].append(checkpoint)
        else:
            checkpoint = manifest["checkpoints"][0]
            checkpoint.update(event)
        manifest.update(status="evaluating_epoch", completed_updates=completed_updates,
                        consumed_pairs=consumed_pairs, pending_pairs=event["pending_pairs"])
        if rank == 0:
            retention.atomic_json(self.output / "manifest.json", manifest)
            print(json.dumps({"event": "epoch_evaluation_begin", **event}), flush=True)
        error, row, stats = None, None, {}
        try:
            self.probe, row, stats = evaluate_probe(self.config, self.schedule, runtime, self.output, torch,
                backend=self.backend, step=completed_updates, epoch=event["weight_update_epochs"],
                epoch_index=event["source_epoch_index"],
                expected_fixed_hash=manifest["schedule_metadata"]["fixed_dataset_sha256"], probe=self.probe)
        except Exception as exc:
            if self.backend == "sharded":
                raise
            error = f"{type(exc).__name__}: {exc}"
        after = runtime_fingerprint(runtime, torch)
        differences = [key for key in before if before[key] != after[key]]
        guard = {"rank": rank, "step": completed_updates, "consumed_pairs": consumed_pairs,
                 "pending_pairs": event["pending_pairs"], "unchanged": not differences,
                 "changed_sections": differences,
                 "before_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
                 "after_sha256": hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
                 **stats, "evaluation_backend": self.backend}
        guards = [None] * runtime["world"] if distributed else [guard]
        if distributed:
            torch.distributed.all_gather_object(guards, guard)
        if rank == 0:
            try:
                (self.output / f"checkpoint-{completed_updates}").mkdir(exist_ok=True)
                retention.atomic_json(self.output / f"checkpoint-{completed_updates}" / "training_state_guard.json", {"ranks": guards})
                if error:
                    raise RuntimeError(error)
                if not all(item["unchanged"] for item in guards):
                    raise AssertionError("Evaluation changed training state; checkpoint retained: " + str(guards))
                row.update(event, partial_epoch=event["evaluation_kind"] == "final_tail",
                    checkpoint_parameter_sha256=checkpoint["parameter_sha256"],
                    initial_parameter_sha256=manifest["initial_parameter_sha256"],
                    teacher_parameter_sha256=manifest["teacher_parameter_sha256"],
                    checkpoint_path=checkpoint["checkpoint"], training_layout=manifest["layout"],
                    training_normalization=manifest["normalization"], training_padding_scope=manifest["padding_scope"],
                    training_schedule_sha256=manifest["schedule_sha256"],
                    evaluation_model="current in-memory training model; " + self.backend + " native evaluation; no optimizer restart",
                    evaluation_backend=self.backend, eval_every_epochs=getattr(self.args, "eval_every_epochs", 1),
                    training_state_unchanged_all_ranks=True)
                for key in ("actual_epoch_microbatches", "expected_epoch_microbatches", "training_examples_this_epoch",
                            "unique_forget_examples_this_epoch", "unique_retain_examples_this_epoch", "unique_retain_examples_cumulative"):
                    row.pop(key, None)
                retention.atomic_json(Path(row["evaluation_dir"]) / "training_state_guard.json", {"ranks": guards})
                row = retention.commit_observation(self.output, manifest, checkpoint, row,
                    Path(row["evaluation_dir"]), self.probe.epoch_output / "fixed_loss_samples.jsonl")
                self.rows.append(row)
                retention.rebuild_aggregates(self.output, self.rows)
                if self.args.delete_evaluated_intermediate:
                    retention.prune_intermediate(self.output, manifest, checkpoint)
                manifest.update(status="training", evaluated_steps=[item["step"] for item in self.rows],
                                evaluated_states=len(self.rows), fixed_dataset_sha256=row["fixed_dataset_sha256"],
                                reference_sha256=row["reference_sha256"], evaluation_teacher_unchanged=True)
                retention.atomic_json(self.output / "manifest.json", manifest)
                print(json.dumps({"event": "epoch_evaluation_committed_training_resumes", **event,
                    "fq": row["forget_quality"], "mu": row["model_utility"], "ks_d": row["ks_d"],
                    "intermediate_weights_deleted": checkpoint.get("weights_deleted_after_verified_evaluation", False)}), flush=True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                manifest.update(status="online_evaluation_failed", error=error)
                retention.atomic_json(self.output / "manifest.json", manifest)
        if distributed:
            payload = [error, manifest if rank == 0 else None]
            torch.distributed.broadcast_object_list(payload, src=0)
            error = payload[0]
            # In-place update keeps the training loop's manifest reference valid.
            if rank != 0:
                manifest.clear()
                manifest.update(payload[1])
        if error:
            raise RuntimeError("Online evaluation failed; no further training is permitted: " + error)
