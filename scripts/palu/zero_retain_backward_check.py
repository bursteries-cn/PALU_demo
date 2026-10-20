#!/usr/bin/env python3
"""Isolated seed103 zero-weight Retain backward equivalence and timing replay.

No production training source is edited. The no_grad arm wraps exactly the
existing Retain forward; its numerical loss and the 0*R term are preserved.
Four fresh CUDA processes replay the same immutable 32-pair update windows.
Correctness fingerprints include real ZeRO-3 local gradients before clipping,
all student/teacher shards, FP32 masters, Adam state, scheduler and RNG.
Use separate timing runs to keep hashing/cache effects outside speed claims.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def instrument_components(function, arm):
    if arm not in ("baseline", "no_grad"):
        raise ValueError("Unknown arm")
    tree = ast.parse(inspect.getsource(function))
    matches = []
    class Visitor(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(target, ast.Name) and target.id == "retain_loss" for target in node.targets):
                if ast.unparse(node.value) != "model(**retain).loss":
                    raise AssertionError("Retain expression changed; refuse automatic instrumentation")
                matches.append(node)
                if arm == "no_grad":
                    context = ast.parse("with torch.no_grad():\n    pass").body[0]
                    context.body = [node]
                    return ast.copy_location(context, node)
            return node
    tree = Visitor().visit(tree)
    if len(matches) != 1:
        raise AssertionError("Expected exactly one native Retain forward")
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, function.__code__.co_filename, "exec"), namespace)
    return namespace[function.__name__], ast.unparse(tree)


def instrument_training(function, recorder):
    tree = ast.parse(inspect.getsource(function))
    loops = [node for node in tree.body[0].body if isinstance(node, ast.For)
             and any(isinstance(inner, ast.For) and isinstance(inner.target, ast.Name)
                     and inner.target.id == "microstep" for inner in ast.walk(node))]
    if len(loops) != 1:
        raise AssertionError("Expected one complete optimizer-window loop")
    loop = loops[0]
    # Insert only before the final optimizer step, never the seven accumulation
    # bookkeeping engine.step() calls inside the microstep loop.
    matches = 0
    for statement in loop.body:
        if not isinstance(statement, ast.If) or ast.unparse(statement.test) != "distributed":
            continue
        for index, child in enumerate(statement.body):
            if isinstance(child, ast.Expr) and ast.unparse(child.value) == "engine.step()":
                statement.body.insert(index, ast.parse("check_recorder.before_optimizer(runtime, step)").body[0])
                matches += 1
                break
    if matches != 1:
        raise AssertionError("Expected exactly one complete-update optimizer step")
    context = ast.parse('with check_recorder.update(runtime, int(window["step"])):\n    pass').body[0]
    context.body = loop.body
    loop.body = [context]
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__, check_recorder=recorder)
    exec(compile(tree, function.__code__.co_filename, "exec"), namespace)
    return namespace[function.__name__], ast.unparse(tree)


class TensorStream:
    """Bounded-memory digest, optionally write or compare one step's raw shards.

    Optional numerical fallback writes only an explicitly chosen step. A failed
    hash-only run is not declared equivalent; rerun that step with snapshots.
    """
    def __init__(self, torch, prefix=None, reference=None):
        if prefix is not None and reference is not None:
            raise ValueError("Choose snapshot write or compare, not both")
        self.torch, self.prefix, self.reference = torch, prefix, reference
        self.index, self.offset, self.comparisons = {}, 0, {}
        self.handle = None
        if prefix is not None:
            prefix.parent.mkdir(parents=True, exist_ok=True)
            self.handle = Path(str(prefix) + ".bin").open("xb")
        if reference is not None:
            self.index_ref = read(str(reference) + ".json")
            self.handle = Path(str(reference) + ".bin").open("rb")

    def tensor(self, name, value):
        torch = self.torch
        if value is None or not torch.is_tensor(value):
            raise TypeError("Missing tensor: " + name)
        flat = value.detach().reshape(-1)
        dtype, shape = str(value.dtype), list(value.shape)
        metadata = dict(dtype=dtype, shape=shape, numel=value.numel(), offset=self.offset,
                        bytes=value.numel() * value.element_size())
        ref = None
        if self.reference is not None:
            ref = self.index_ref.get(name)
            if ref is None or any(ref[key] != metadata[key] for key in ("dtype", "shape", "numel", "bytes")):
                raise AssertionError("Snapshot tensor layout differs: " + name)
            self.handle.seek(ref["offset"])
        sha, square_sum, max_abs = hashlib.sha256(), 0.0, 0.0
        diff_square, ref_square, diff_max, unequal = 0.0, 0.0, 0.0, 0
        chunk = max(1, (8 << 20) // value.element_size())
        for start in range(0, flat.numel(), chunk):
            cpu = flat[start:start + chunk].contiguous().cpu()
            encoded = cpu.view(torch.uint8).numpy().tobytes()
            sha.update(encoded)
            numeric = cpu.double()
            if not bool(torch.isfinite(numeric).all()):
                raise AssertionError("Nonfinite tensor: " + name)
            square_sum += float(numeric.square().sum())
            max_abs = max(max_abs, float(numeric.abs().max()))
            if self.prefix is not None:
                self.handle.write(encoded)
            if ref is not None:
                encoded_ref = self.handle.read(len(encoded))
                if len(encoded_ref) != len(encoded):
                    raise AssertionError("Truncated reference snapshot")
                expected = torch.frombuffer(bytearray(encoded_ref), dtype=cpu.dtype).double()
                difference = numeric - expected
                diff_square += float(difference.square().sum())
                ref_square += float(expected.square().sum())
                diff_max = max(diff_max, float(difference.abs().max()))
                unequal += int(numeric.ne(expected).sum())
        metadata["sha256"] = sha.hexdigest()
        self.index[name] = metadata
        self.offset += metadata["bytes"]
        if ref is not None:
            self.comparisons[name] = dict(bitwise_equal=metadata["sha256"] == ref["sha256"],
                unequal_elements=unequal, unequal_fraction=unequal / max(1, value.numel()),
                rms_diff=math.sqrt(diff_square / max(1, value.numel())), max_abs_diff=diff_max, l2_diff=math.sqrt(diff_square),
                relative_l2_diff=math.sqrt(diff_square / ref_square) if ref_square else (0.0 if not diff_square else None))
        return dict(dtype=dtype, shape=shape, numel=value.numel(), sha256=sha.hexdigest(),
                    l2_norm=math.sqrt(square_sum), max_abs=max_abs)

    def state(self, name, value):
        if self.torch.is_tensor(value):
            return self.tensor(name, value)
        if isinstance(value, dict):
            return {str(key): self.state(name + "/" + str(key), item)
                    for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (tuple, list)):
            return [self.state(name + "/" + str(index), item) for index, item in enumerate(value)]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise TypeError("Unsupported state value: " + name + " " + type(value).__name__)

    def close(self):
        if self.handle:
            self.handle.close()
        if self.prefix is not None:
            write(str(self.prefix) + ".json", self.index)
        if self.reference is not None and set(self.index) != set(self.index_ref):
            raise AssertionError("Reference snapshot contains missing/extra tensors")


def gradient_fingerprint(runtime, stream, get_local_grad=None):
    if not runtime.get("distributed") or runtime.get("world") != 4:
        raise ValueError("Real gradient capture requires four-rank ZeRO-3")
    if get_local_grad is None:
        from deepspeed.utils import safe_get_local_grad
        get_local_grad = safe_get_local_grad
    optimizer = runtime["optimizer"]
    if not hasattr(optimizer, "fp32_partitioned_groups_flat"):
        raise ValueError("Expected ZeRO-3 FP32 partitioned optimizer")
    result = {}
    for name, parameter in runtime["model"].named_parameters():
        if not parameter.requires_grad:
            continue
        # safe_get_local_grad reads the actual ZeRO3 accumulator, including CPU
        # optimizer offload; parameter.grad is generally None and is not used.
        gradient = get_local_grad(parameter)
        if gradient is None:
            raise AssertionError("Missing real ZeRO gradient: " + name)
        if not hasattr(parameter, "ds_tensor") or gradient.numel() != parameter.ds_tensor.numel():
            raise AssertionError("Incomplete local gradient partition: " + name)
        result[name] = dict(ds_numel=int(parameter.ds_numel), ds_shape=list(parameter.ds_shape),
                            partition=stream.tensor("gradient/" + name, gradient))
    if not result:
        raise AssertionError("No trainable gradients captured")
    return result


def state_fingerprint(runtime, stream, completed_updates=0):
    import numpy as np
    torch, result = stream.torch, {}
    for tag, wrapper in (("student", runtime["engine"]), ("teacher", runtime["reference"])):
        model = getattr(wrapper, "module", wrapper)
        result[tag] = dict(
            parameters={name: stream.tensor(tag + "/parameter/" + name, getattr(parameter, "ds_tensor", parameter))
                        for name, parameter in model.named_parameters()},
            buffers={name: stream.tensor(tag + "/buffer/" + name, value) for name, value in model.named_buffers()},
            module_modes={name: module.training for name, module in wrapper.named_modules()},
            use_cache=getattr(model.config, "use_cache", None))
    optimizer = runtime["optimizer"]
    underlying = getattr(optimizer, "optimizer", optimizer)
    masters = getattr(optimizer, "fp32_partitioned_groups_flat", None)
    if masters is None or not masters:
        raise ValueError("Missing persistent ZeRO3 FP32 masters")
    moment_states = [underlying.state.get(parameter, {}) for parameter in masters]
    moment_coverage = []
    for index, (parameter, values) in enumerate(zip(masters, moment_states)):
        keys = ("exp_avg", "exp_avg_sq") if "exp_avg" in values else ("state1", "state2")
        if completed_updates > 0 and parameter.numel():
            if "step" not in values or any(key not in values for key in keys):
                raise AssertionError(f"Missing persistent Adam state for master subgroup {index}")
            if any(not torch.is_tensor(values[key]) or values[key].numel() != parameter.numel() for key in keys):
                raise AssertionError(f"Incomplete persistent Adam moments for master subgroup {index}")
            observed_step = values["step"].item() if torch.is_tensor(values["step"]) else values["step"]
            if int(observed_step) != completed_updates:
                raise AssertionError(f"Adam step differs from completed updates for subgroup {index}")
        moment_coverage.append(dict(subgroup=index, master_numel=parameter.numel(),
            keys=sorted(values), moment_numel={key: values[key].numel() for key in keys if key in values and torch.is_tensor(values[key])}))
    result["optimizer"] = dict(
        moment_coverage=moment_coverage,
        masters=[stream.tensor("optimizer/master/" + str(i), parameter) for i, parameter in enumerate(masters)],
        moments=[stream.state("optimizer/moments/" + str(i), moment_states[i])
                 for i, parameter in enumerate(masters)],
        groups=[stream.state("optimizer/group/" + str(i), {key: value for key, value in group.items() if key != "params"})
                for i, group in enumerate(underlying.param_groups)],
        counters={key: stream.state("optimizer/counter/" + key, getattr(optimizer, key)) for key in
                  ("micro_step_id", "is_gradient_accumulation_boundary", "overflow") if hasattr(optimizer, key)})
    result["scheduler"] = stream.state("scheduler", runtime["scheduler"].state_dict())
    result["engine_counters"] = {key: getattr(runtime["engine"], key) for key in
        ("global_steps", "global_samples", "micro_steps", "skipped_steps") if hasattr(runtime["engine"], key)}
    result["rng"] = dict(python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        torch_cpu=stream.tensor("rng/cpu", torch.random.get_rng_state()),
        torch_cuda=stream.tensor("rng/cuda", torch.cuda.get_rng_state(runtime["device"])))
    return result


class Recorder:
    def __init__(self, args, torch):
        self.args, self.torch, self.rank = args, torch, int(os.environ.get("RANK", "0"))
        self.timings = []
        self.before_steps = []

    def capture(self, runtime, step, stage):
        selected = self.args.snapshot_step == step
        suffix = f"step-{step}-{stage}-rank-{self.rank}"
        prefix = self.args.snapshot_write / suffix if selected and self.args.snapshot_write else None
        reference = self.args.snapshot_compare / suffix if selected and self.args.snapshot_compare else None
        stream = TensorStream(self.torch, prefix=prefix, reference=reference)
        try:
            value = gradient_fingerprint(runtime, stream) if stage == "gradient" else state_fingerprint(runtime, stream, completed_updates=step)
        finally:
            stream.close()
        write(self.args.output / f"{suffix}.json", dict(step=step, stage=stage, rank=self.rank,
              fingerprint=value, numerical_comparison=stream.comparisons))

    def before_optimizer(self, runtime, step):
        self.before_steps.append(step)
        if self.args.mode == "check":
            self.capture(runtime, step, "gradient")

    @contextmanager
    def update(self, runtime, step):
        torch = self.torch
        torch.cuda.synchronize(runtime["device"])
        torch.distributed.barrier()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize(runtime["device"])
        elapsed = time.perf_counter() - start
        # Capture happens after this wall-clock interval. Correctness intervals
        # still include the pre-step gradient read and are NOT timing evidence.
        torch.distributed.barrier()
        timing = dict(rank=self.rank, step=step, warmup=step <= self.args.warmup,
                      wall_seconds=elapsed, valid_for_timing=self.args.mode == "time")
        self.timings.append(timing)
        append(self.args.output / f"timings-rank-{self.rank}.jsonl", timing)
        if self.args.mode == "check":
            self.capture(runtime, step, "state")
        if self.rank == 0:
            print(json.dumps(dict(event="zero_retain_update_complete", arm=self.args.arm, **timing)), flush=True)


class NoEvaluation:
    def __init__(self, *args):
        self.events = {}
    def at(self, *args):
        return None


class Control:
    run_initial_evaluation = False
    def __init__(self, recorder, planned_updates):
        self.recorder, self.planned_updates = recorder, planned_updates
    def select_windows(self, windows):
        return windows
    def coefficients_for_step(self, step):
        return 1.0, 0.0, 2.0
    def configure_online(self, online):
        pass
    def initialize(self, runtime, manifest, config, schedule, args, torch):
        metadata = schedule["metadata"]
        if metadata.get("effective_batch") != 32 or metadata.get("source_config_sha256") != file_sha(args.config):
            raise AssertionError("Frozen schedule is not bound to this exact configuration/effective batch")
        if Path(metadata["model_path"]).resolve() != Path(config["model"]["model_args"]["pretrained_model_name_or_path"]).resolve():
            raise AssertionError("Frozen schedule initial model path mismatch")
        source_manifest_path = self.recorder.args.study / "run" / "manifest.json"
        if not source_manifest_path.is_file():
            raise ValueError("Missing historical run manifest for seed103 weight identity check")
        source_manifest = read(source_manifest_path)
        for key in ("initial_parameter_sha256", "teacher_parameter_sha256", "deepspeed_config", "method", "packages"):
            if manifest[key] != source_manifest[key]:
                raise AssertionError("Replay differs from historical seed103 runtime: " + key)
        if source_manifest.get("seed") != 103:
            raise AssertionError("Historical run is not seed103")
        self.recorder.capture(runtime, 0, "state")
        manifest.update(isolated_zero_retain_check=True, save_weights=False,
                        evaluation_protocol="disabled for isolated training replay")
    def finish(self, runtime, manifest, args, torch):
        expected = list(range(1, self.recorder.args.updates + 1))
        if self.recorder.before_steps != expected:
            raise AssertionError("Gradient hook failed to visit every complete update exactly once")
        if self.recorder.args.mode == "time":
            self.recorder.capture(runtime, len(expected), "state")
        if runtime["rank"] == 0:
            manifest.update(status="isolated_check_completed", completed_updates=len(expected),
                            completed_unix=time.time(), arm=self.recorder.args.arm,
                            diagnostic_mode=self.recorder.args.mode)
            write(args.output / "manifest.json", manifest)
        return True


def differing_paths(left, right, prefix=""):
    if type(left) != type(right):
        return [prefix]
    if isinstance(left, dict):
        return [path for key in sorted(set(left) | set(right)) for path in
                ([prefix + "/" + key] if key not in left or key not in right else
                 differing_paths(left[key], right[key], prefix + "/" + key))]
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix + "/length"]
        return [path for index, (a, b) in enumerate(zip(left, right))
                for path in differing_paths(a, b, prefix + "/" + str(index))]
    return [] if left == right else [prefix]


def compare(left, right):
    left, right = Path(left), Path(right)
    protocols = [read(folder / "check_protocol.json") for folder in (left, right)]
    for key in ("config_sha256", "schedule_sha256", "updates", "warmup", "seed", "mode", "code_sha256"):
        if protocols[0][key] != protocols[1][key]:
            raise AssertionError("Arm protocols differ: " + key)
    comparison_kind = "same_arm_repeat" if protocols[0]["arm"] == protocols[1]["arm"] else "baseline_vs_no_grad"
    if comparison_kind != "same_arm_repeat" and (protocols[0]["arm"], protocols[1]["arm"]) != ("baseline", "no_grad"):
        raise ValueError("Different-arm comparison expects baseline then no_grad")
    count, mode = protocols[0]["updates"], protocols[0]["mode"]
    manifests = [read(folder / "manifest.json") for folder in (left, right)]
    bound_fields = ("deepspeed_config", "packages", "tf32_matmul", "tf32_cudnn", "method", "max_grad_norm",
                    "initial_parameter_sha256", "teacher_parameter_sha256", "layout", "effective_batch",
                    "normalization", "padding_scope", "world_size", "batch_size", "gradient_accumulation_steps")
    for key in bound_fields:
        if manifests[0].get(key) != manifests[1].get(key):
            raise AssertionError("Runtime configurations differ: " + key)
    for folder, manifest in zip((left, right), manifests):
        if manifest.get("status") != "isolated_check_completed" or manifest.get("completed_updates") != count:
            raise AssertionError("Incomplete arm: " + str(folder))
    differences, check_rows = [], []
    if mode in ("check", "time"):
        for step in (range(count + 1) if mode == "check" else (0, count)):
            for stage in (("state",) if step == 0 or mode == "time" else ("gradient", "state")):
                for rank in range(4):
                    suffix = f"step-{step}-{stage}-rank-{rank}.json"
                    a, b = (read(folder / suffix) for folder in (left, right))
                    paths = differing_paths(a["fingerprint"], b["fingerprint"])
                    differences += [{"step": step, "stage": stage, "rank": rank, "path": path} for path in paths]
                    check_rows.append(dict(step=step, stage=stage, rank=rank,
                                           bitwise_equal=not paths, numerical_comparison=b.get("numerical_comparison", {})))
    losses = [rows(folder / "loss_history.jsonl") for folder in (left, right)]
    if any([row["step"] for row in arm] != list(range(1, count + 1)) for arm in losses):
        raise AssertionError("Incomplete loss records")
    loss_comparisons = []
    for a, b in zip(*losses):
        stripped = [{key: value for key, value in row.items() if key != "seconds_since_start"} for row in (a, b)]
        paths = differing_paths(*stripped)
        differences += [dict(step=a["step"], stage="loss", rank="all", path=path) for path in paths]
        loss_comparisons.append(dict(step=a["step"], bitwise_equal=not paths,
            scalar_differences={key: b[key] - a[key] for key in ("raw_sensitive_loss", "raw_retain_nll", "raw_common_kl", "loss", "grad_norm")}))
    timing = []
    for folder in (left, right):
        rank_rows = [rows(folder / f"timings-rank-{rank}.jsonl") for rank in range(4)]
        if any([row["step"] for row in values] != list(range(1, count + 1)) for values in rank_rows):
            raise AssertionError("Incomplete timing ranks")
        measured = [max(rank_rows[rank][step - 1]["wall_seconds"] for rank in range(4))
                    for step in range(protocols[0]["warmup"] + 1, count + 1)]
        timing.append(dict(arm=read(folder / "check_protocol.json")["arm"],
                           measured_max_rank_seconds=measured, mean_seconds=sum(measured) / len(measured)))
    return dict(schema=1, mode=mode, comparison_kind=comparison_kind, baseline=str(left), no_grad=str(right),
        all_bitwise_equal=not differences,
        update_equivalence_basis=(
            "identical initial and every post-step BF16 shards and FP32 masters implies identical parameter updates"
            if mode == "check" else
            "only initial and final BF16 shards, FP32 masters and optimizer states are compared; intermediate updates and gradients are not captured"),
        differences=differences, state_checks=check_rows, loss_checks=loss_comparisons,
        timing_valid=mode == "time", timing=timing,
        speedup=timing[0]["mean_seconds"] / timing[1]["mean_seconds"] if mode == "time" else None,
        gradient_capture_available=mode == "check",
        limits=[f"Only this {count}-update prefix and zero Retain coefficient are tested.",
                "Complete local shards cover the whole model collectively across all four ranks.",
                "Hash mismatch without selected-step snapshots is not accepted as numerical equivalence.",
                "Correctness-run durations include gradient inspection and are not speed measurements."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "time", "compare"), required=True)
    parser.add_argument("--arm", choices=("baseline", "no_grad"))
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--snapshot-step", type=int)
    parser.add_argument("--snapshot-write", type=Path)
    parser.add_argument("--snapshot-compare", type=Path)
    args = parser.parse_args()
    if args.mode == "compare":
        if not args.baseline or not args.candidate:
            parser.error("compare requires --baseline and --candidate")
        result = compare(args.baseline, args.candidate)
        write(args.output, result)
        print(json.dumps({key: value for key, value in result.items() if key not in ("state_checks", "loss_checks", "differences")}, indent=2))
        if result["all_bitwise_equal"] is False:
            raise SystemExit(2)
        return
    if not args.arm or not args.study or not 0 <= args.warmup < args.updates <= 63:
        parser.error("replay requires arm/study and 0 <= warmup < updates <= 63")
    if args.snapshot_write and args.snapshot_compare:
        parser.error("snapshot write and compare are mutually exclusive")
    if (args.snapshot_write or args.snapshot_compare) and (args.mode != "check" or args.snapshot_step is None):
        parser.error("snapshots require check mode and an explicit snapshot step")
    if args.snapshot_step is not None and not 1 <= args.snapshot_step <= args.updates:
        parser.error("snapshot step must fall within replay")
    import torch
    import layout_token_training as base
    import gradient_equivalence_step as gradient
    import online_epoch_evaluation as online
    config_path, schedule_path = (args.study / "frozen" / name for name in ("config.json", "schedule.pt"))
    config = read(config_path)
    train_args = config["trainer"]["args"]
    if float(train_args["learning_rate"]) != 1e-5 or train_args.get("lr_scheduler_type") != "constant" or int(train_args.get("warmup_steps", 0)) != 0:
        raise ValueError("This diagnostic requires the current seed103 constant-1e-5 protocol")
    if int(train_args.get("max_steps", 0)) != 63:
        raise ValueError("Expected the frozen 63-update threshold protocol")
    frozen = read(args.study / "frozen" / "manifest.json")
    for path in (config_path, schedule_path):
        if file_sha(path) != frozen["frozen_file_sha256"][path.name]:
            raise ValueError("Frozen input changed: " + path.name)
    if int(config["trainer"]["args"].get("seed", 103)) != 103:
        raise ValueError("Expected seed103 frozen configuration")
    recorder = Recorder(args, torch)
    train, train_source = instrument_training(base.train, recorder)
    components, component_source = instrument_components(gradient.raw_components, args.arm)
    options = SimpleNamespace(config=config_path, schedule=schedule_path, output=args.output,
        max_updates=args.updates, layout="four", seed=103, normalization="global",
        padding_scope="microbatch", online_evaluation=True, delete_evaluated_intermediate=False)
    original_components, original_online = gradient.raw_components, online.OnlineEpochEvaluation
    gradient.raw_components, online.OnlineEpochEvaluation = components, NoEvaluation
    try:
        train(options, torch, Control(recorder, planned_updates=63))
        if recorder.rank == 0:
            (args.output / "instrumented_train.py").write_text(train_source + "\n")
            (args.output / "instrumented_components.py").write_text(component_source + "\n")
            historical = args.study / "run" / "loss_history.jsonl"
            if args.arm == "baseline" and historical.is_file():
                old_rows = {row["step"]: row for row in rows(historical)}
                history_check = []
                for current in rows(args.output / "loss_history.jsonl"):
                    old = old_rows[current["step"]]
                    scalar_keys = ("raw_sensitive_loss", "raw_retain_nll", "raw_common_kl", "loss", "grad_norm")
                    history_check.append(dict(step=current["step"],
                        scalar_difference={key: current[key] - old[key] for key in scalar_keys},
                        scalar_exact=all(current[key] == old[key] for key in scalar_keys),
                        pair_sha256_equal=current["pair_sha256"] == old["pair_sha256"],
                        window_sha256_equal=current["window_sha256"] == old["window_sha256"],
                        learning_rate_equal=current["lr_before"] == old["lr_before"]))
                write(args.output / "historical_prefix_comparison.json", dict(
                    source=str(historical), source_sha256=file_sha(historical), steps=history_check,
                    limitation="Historical training also ran fixed probes and evaluation; replay disables these."))
            sources = [Path(__file__), Path(base.__file__), Path(gradient.__file__), Path(online.__file__)]
            write(args.output / "check_protocol.json", dict(schema=1, mode=args.mode, arm=args.arm,
                seed=103, updates=args.updates, warmup=args.warmup,
                source_study=str(args.study.resolve()), config_sha256=file_sha(config_path),
                schedule_sha256=file_sha(schedule_path), coefficients=[1.0, 0.0, 2.0],
                effective_batch=32, layout="4 x batch1 x accumulation8", normalization="global",
                code_sha256={path.name: file_sha(path) for path in sources},
                gradient_capture="safe_get_local_grad after 8th backward, before clipping/optimizer step",
                snapshot_step=args.snapshot_step, snapshot_write=str(args.snapshot_write) if args.snapshot_write else None,
                snapshot_compare=str(args.snapshot_compare) if args.snapshot_compare else None,
                timing="separate time-mode replay; CUDA synchronization and barrier at identical complete-step boundaries"))
    finally:
        gradient.raw_components, online.OnlineEpochEvaluation = original_components, original_online
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
