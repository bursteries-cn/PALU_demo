#!/usr/bin/env python3
"""Run paired full-mean/stop-mean PALU training, separate TOFU evaluation, and reporting."""
import argparse
from bisect import bisect_right
import copy
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {"full_mean": False, "stop_mean": True}
WEIGHT_FILE = re.compile(r"(?:model(?:-\d+-of-\d+)?\.safetensors(?:\.index\.json)?|"
                         r"pytorch_model(?:-\d+-of-\d+)?\.bin(?:\.index\.json)?|"
                         r"adapter_model\.(?:bin|safetensors))")
QUALITY_METRICS = ("forget_quality", "forget_ks_statistic", "model_utility",
                   "exact_memorization", "forget_Q_A_gibberish")
PERFORMANCE_METRICS = ("train_runtime", "train_steps_per_second", "seconds_per_optimizer_step",
                       "entrypoint_wall_seconds", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
                       "diagnostic_wall_seconds")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def assignment(key, value):
    return f"{key}={json.dumps(value, ensure_ascii=False)}"


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_snapshot(args):
    files = []
    for folder, pattern in (("src", "*.py"), ("configs", "*.yaml"), ("configs", "*.json"),
                            ("scripts/palu_mean_ablation", "*.py"), ("scripts/palu_mean_ablation", "*.sh")):
        files.extend((ROOT / folder).rglob(pattern))
    files.extend((ROOT / "data/palu/TOFU" / f"{args.forget_split}_with_common_words_gpt").rglob("*"))
    if args.retain_logs.is_file():
        files.append(args.retain_logs)
    # Include local data supplied through Hydra overrides and optional eval data.
    data_paths = [Path(args.eval_dataset)] if args.eval_dataset else []
    for override in args.override:
        key, _, value = override.partition("=")
        if key.endswith("hf_args.path"):
            data_paths.append(Path(value.strip('\"\'')))
    for path in data_paths:
        if path.is_dir():
            files.extend(path.rglob("*"))
        elif path.is_file():
            files.append(path)
    return {str(path.resolve()): hash_file(path) for path in sorted(set(files)) if path.is_file()}


def validate_overrides(overrides):
    protected = ("trainer", "trainer.handler", "trainer.method_args.detach_mean_target",
                 "trainer.method_args.target_mode", "paths.output_dir", "trainer.args.output_dir",
                 "trainer.args.logging_dir", "trainer.args.run_name", "task_name", "trainer.args.seed",
                 "trainer.args.data_seed", "trainer.args.eval_strategy", "trainer.args.eval_on_start",
                 "trainer.args.do_eval", "trainer.args.do_train", "trainer.args.report_to",
                 "trainer.args.save_strategy", "do_save", "eval", "model",
                 "forget_split", "holdout_split", "retain_split", "trainer.args.deepspeed",
                 "trainer.method_args.diagnostics", "trainer.method_args.track_loss_components",
                 "model.model_args.device_map", "trainer.args.per_device_train_batch_size",
                 "trainer.args.gradient_accumulation_steps")
    for override in overrides:
        key = override.partition("=")[0].lstrip("+~")
        if (key in protected or any(item.startswith(key + ".") for item in protected)
                or key.startswith("eval.") or key.startswith("trainer.method_args.diagnostics.")):
            raise ValueError(f"Override {key} would invalidate the paired protocol")


def build_commands(args, run, seed, variant):
    train = [args.python, str(ROOT / "scripts/palu_mean_ablation/train_with_telemetry.py"),
             "--telemetry", str(run / "telemetry.json"), "--config-name=unlearn.yaml",
             "experiment=unlearn/tofu/tpo", "trainer=PALU", f"model={args.model}",
             assignment("model.model_args.pretrained_model_name_or_path", args.model_path),
             assignment("model.tokenizer_args.pretrained_model_name_or_path", args.model_path),
             f"forget_split={args.forget_split}", f"holdout_split={args.holdout_split}",
             f"retain_split={args.retain_split}", assignment("task_name", f"palu_mean_seed{seed}_{variant}"),
             assignment("paths.output_dir", str(run)), "do_save=true", "~eval.tofu",
             "trainer.args.do_train=true", "trainer.args.do_eval=false", "trainer.args.eval_strategy=no",
             "trainer.args.eval_on_start=false", "trainer.args.save_strategy=no", "trainer.args.report_to=none",
             "trainer.args.logging_steps=1", f"trainer.args.seed={seed}", f"+trainer.args.data_seed={seed}",
             f"trainer.args.learning_rate={args.lr}", f"trainer.args.per_device_train_batch_size={args.batch_size}",
             f"trainer.args.gradient_accumulation_steps={args.grad_accum}", f"trainer.args.num_train_epochs={args.epochs}",
             f"+trainer.args.max_steps={args.max_steps}", f"+trainer.args.max_grad_norm={args.max_grad_norm}",
             "trainer.args.gradient_checkpointing=true", f"trainer.method_args.gamma={getattr(args, 'gamma', 1.0)}",
             f"trainer.method_args.alpha={args.alpha}", "trainer.method_args.retain_loss_type=NLL",
             f"trainer.method_args.top_k={args.top_k}", f"trainer.method_args.first_n={args.first_n}",
             "trainer.method_args.target_mode=mean",
             assignment("trainer.method_args.detach_mean_target", VARIANTS[variant]), *args.override]
    if getattr(args, "track_loss_components", True):
        train.append("+trainer.method_args.track_loss_components=true")
    if getattr(args, "retain_model_path", None):
        options = {"cache_path": str(args.output / "reference_cache.pt"),
                   "cache_sha256": getattr(args, "probe_cache_sha256", None),
                   "every": args.probe_every, "temperature": args.probe_temperature}
        train.append("+trainer.method_args.diagnostics=" +
                     "{" + ",".join(f"{key}:{json.dumps(value, ensure_ascii=False)}" for key, value in options.items()) + "}")
    if args.model_revision:
        train.extend([assignment("+model.model_args.revision", args.model_revision),
                      assignment("+model.tokenizer_args.revision", args.model_revision)])
    if args.num_processes > 1:
        train = [args.python, "-m", "accelerate.commands.launch", "--config_file",
                 str(ROOT / "configs/accelerate/default_config.yaml"),
                 "--num_processes", str(args.num_processes),
                 "--main_process_port", str(args.main_process_port), "--mixed_precision", "bf16", *train[1:]]
    evaluate = [args.python, str(ROOT / "src/eval.py"), "experiment=eval/tofu/default", f"model={args.model}",
                assignment("model.model_args.pretrained_model_name_or_path", str(run)),
                assignment("model.tokenizer_args.pretrained_model_name_or_path", str(run)),
                f"forget_split={args.forget_split}", f"holdout_split={args.holdout_split}",
                assignment("retain_logs_path", str(args.retain_logs)), assignment("paths.output_dir", str(run / "evals")),
                assignment("task_name", f"palu_mean_seed{seed}_{variant}_eval"), f"seed={seed}",
                "eval.tofu.overwrite=true", f"eval.tofu.batch_size={args.eval_batch_size}"]
    if args.eval_dataset:
        evaluate.append(assignment("+tofu_dataset_path", args.eval_dataset))
    if args.classifier_model:
        for key in ("classifier_model_args", "classifier_tokenization_args"):
            evaluate.append(assignment(f"eval.tofu.metrics.forget_Q_A_gibberish.{key}.pretrained_model_name_or_path",
                                       args.classifier_model))
    return train, None if args.skip_eval else evaluate


def merge_telemetry(run, num_processes):
    """Aggregate after all workers exit; memory is the maximum per-rank peak."""
    if num_processes == 1:
        records = [read_json(run / "telemetry.json")]
    else:
        records = [read_json(run / f"telemetry.rank{rank}.json") for rank in range(num_processes)]
    for rank, record in enumerate(records):
        if record.get("status") != "completed" or record.get("rank") != rank or record.get("world_size") != num_processes:
            raise ValueError(f"Missing/failed/misconfigured telemetry for rank {rank}")
        for key in ("packages", "cuda_available", "cuda_version", "cuda_visible_devices"):
            if record.get(key) != records[0].get(key):
                raise ValueError(f"Distributed runtime differs on rank {rank}: {key}")
    merged = dict(records[0])
    for key in ("entrypoint_wall_seconds", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        values = [record.get(key) for record in records]
        merged[key] = max(values) if all(value is not None for value in values) else None
    merged["gpu_names_by_rank"] = [record.get("gpu_name") for record in records]
    merged["memory_scope"] = "maximum per-rank PyTorch allocator peak; not summed GPU memory"
    merged["ranks"] = records
    write_json(run / "telemetry.json", merged)
    return merged


def cleanup_completed_weights(run, keep_model):
    """Only clean this run's regular weight files after its metrics are verified."""
    result = read_json(run / "result.json")
    if result.get("status") != "completed" or not result.get("metrics"):
        raise ValueError("Incomplete experiment; weights preserved")
    files = [] if keep_model else sorted(path for path in run.iterdir() if WEIGHT_FILE.fullmatch(path.name))
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("Refusing to remove non-regular model artifacts")
    for path in files:
        path.unlink()
    result.update(keep_model_weights=keep_model, removed_weight_files=[path.name for path in files])
    write_json(run / "result.json", result)
    return result


def normalized_config(config):
    config = copy.deepcopy(config)
    config.pop("task_name", None)
    config["paths"].pop("output_dir", None)
    for key in ("output_dir", "logging_dir", "run_name"):
        config["trainer"]["args"].pop(key, None)
    config["trainer"]["method_args"].pop("detach_mean_target", None)
    return config


def validate_pair(full, stop):
    for name, config in (("full_mean", full), ("stop_mean", stop)):
        if config["trainer"]["handler"] != "PALU":
            raise ValueError("Resolved trainer is not PALU")
        methods = config["trainer"]["method_args"]
        if methods["target_mode"] != "mean" or methods["detach_mean_target"] is not VARIANTS[name]:
            raise ValueError(f"Resolved mean gradient setting is wrong for {name}")
    if normalized_config(full) != normalized_config(stop):
        raise ValueError("Resolved paired configs differ beyond gradient flag and output paths")


def ks_statistic(left, right):
    left, right = sorted(left), sorted(right)
    if not left or not right:
        raise ValueError("Empty truth-ratio samples")
    return max(abs(bisect_right(left, point) / len(left) - bisect_right(right, point) / len(right))
               for point in set(left + right))


def collect_metrics(run, retain_logs, evaluate):
    state = read_json(run / "trainer_state.json")
    last = next((item for item in reversed(state["log_history"]) if "train_runtime" in item), {})
    if not last or state["global_step"] <= 0:
        raise ValueError("Training did not record optimizer steps and runtime")
    if state.get("max_steps", state["global_step"]) != state["global_step"]:
        raise ValueError("Training did not reach its planned optimizer-step count; weights preserved")
    telemetry = read_json(run / "telemetry.json")
    metrics = {key: last.get(key) for key in ("train_runtime", "train_steps_per_second")}
    metrics.update({key: telemetry.get(key) for key in PERFORMANCE_METRICS if key not in metrics})
    metrics["seconds_per_optimizer_step"] = last["train_runtime"] / state["global_step"]
    metrics["optimizer_steps"] = state["global_step"]
    metrics["epoch"] = state["epoch"]
    timing = run / "diagnostic_timing.json"
    metrics["diagnostic_wall_seconds"] = read_json(timing)["diagnostic_wall_seconds"] if timing.is_file() else 0.0
    if evaluate:
        provenance = read_json(run / "evals/evaluation_provenance.json")
        if provenance["status"] != "completed" or provenance.get("reference_sha256") != hash_file(retain_logs):
            raise ValueError("Evaluation provenance or Retain reference is invalid")
        evaluated_model = provenance.get("config", {}).get("model", {}).get("model_args", {}).get("pretrained_model_name_or_path")
        if not evaluated_model or Path(evaluated_model).resolve() != run.resolve():
            raise ValueError("Evaluation did not use this run's checkpoint; weights preserved")
        summary = read_json(run / "evals/TOFU_SUMMARY.json")
        for key in QUALITY_METRICS:
            if key != "forget_ks_statistic":
                value = summary.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Missing/non-finite evaluation metric: {key}")
                metrics[key] = value
        logs = read_json(run / "evals/TOFU_EVAL.json")
        reference = read_json(retain_logs)
        scores = lambda payload: [entry["score"] for entry in payload["forget_truth_ratio"]["value_by_index"].values()]
        metrics["forget_ks_statistic"] = ks_statistic(scores(logs), scores(reference))
    return metrics


def summarize(output):
    manifest = read_json(output / "manifest.json")
    rows, missing = [], []
    for planned in manifest["runs"]:
        run = Path(planned["output_dir"])
        result_file = run / "result.json"
        if not result_file.is_file() or read_json(result_file).get("status") != "completed":
            missing.append({"seed": planned["seed"], "variant": planned["variant"]})
            continue
        result = read_json(result_file)
        rows.append({"seed": planned["seed"], "variant": planned["variant"], **result["metrics"]})
    paired, configs = [], {}
    for seed in manifest["seeds"]:
        pair = {row["variant"]: row for row in rows if row["seed"] == seed}
        if set(pair) != set(VARIANTS):
            continue
        for variant in VARIANTS:
            configs[variant] = read_json(output / f"seed_{seed}" / variant / "resolved_config.json")
            recorded_args = configs[variant]["trainer"]["args"]
            if recorded_args["seed"] != seed or recorded_args["data_seed"] != seed:
                raise ValueError(f"Resolved seed does not match run metadata for seed {seed}")
        validate_pair(configs["full_mean"], configs["stop_mean"])
        if pair["full_mean"]["optimizer_steps"] != pair["stop_mean"]["optimizer_steps"]:
            raise ValueError(f"Optimizer step counts differ for seed {seed}")
        telemetry = {name: read_json(output / f"seed_{seed}" / name / "telemetry.json") for name in VARIANTS}
        for key in ("packages", "cuda_available", "cuda_version", "gpu_name", "cuda_visible_devices",
                    "world_size", "gpu_names_by_rank"):
            if telemetry["full_mean"].get(key) != telemetry["stop_mean"].get(key):
                raise ValueError(f"Runtime environment differs for seed {seed}: {key}")
        delta = {"seed": seed}
        for key in (*QUALITY_METRICS, *PERFORMANCE_METRICS):
            full, stop = pair["full_mean"].get(key), pair["stop_mean"].get(key)
            delta[key] = stop - full if full is not None and stop is not None else None
        paired.append(delta)
    aggregate = {}
    for key in (*QUALITY_METRICS, *PERFORMANCE_METRICS):
        values = [item[key] for item in paired if item[key] is not None]
        if values:
            aggregate[key] = {"n_pairs": len(values), "mean_stop_minus_full": statistics.mean(values),
                              "stddev_paired_difference": statistics.stdev(values) if len(values) > 1 else None}
    result = {"status": "completed" if not missing else "incomplete", "rows": rows,
              "paired_stop_minus_full": paired, "aggregate": aggregate, "missing_runs": missing,
              "scope": "TOFU training comparison" if manifest["evaluation_enabled"] else "training smoke/performance only; no forgetting evaluation"}
    write_json(output / "comparison.json", result)
    fields = ["seed", "variant", "optimizer_steps", "epoch", *QUALITY_METRICS, *PERFORMANCE_METRICS]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# PALU mean-gradient comparison", "", result["scope"], "",
             f"Completed pairs: {len(paired)} / {len(manifest['seeds'])}", "",
             "Differences below are stop_mean minus full_mean. FQ is a KS p-value; it is not an erasure percentage.",
             "Training runtime excludes evaluation; entrypoint wall time also includes setup and final save.", "",
             "GPU memory metrics are the maximum per-rank allocator peak; evaluation uses one GPU.", "",
             "| Metric | Mean paired difference | Paired sample stddev | n |", "|---|---:|---:|---:|"]
    for key, values in aggregate.items():
        std = values["stddev_paired_difference"]
        lines.append(f"| {key} | {values['mean_stop_minus_full']:.6g} | {std:.6g} | {values['n_pairs']} |"
                     if std is not None else f"| {key} | {values['mean_stop_minus_full']:.6g} | — | {values['n_pairs']} |")
    if missing:
        lines += ["", f"Missing/incomplete runs: {json.dumps(missing)}"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def execute(command, log, env):
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env={**env, "PYTHONUNBUFFERED": "1"},
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
    return returncode, time.perf_counter() - started


def sample_count(value):
    count = -1 if value == "all" else int(value)
    if count == 0 or count < -1:
        raise argparse.ArgumentTypeError("Probe sample count must be positive or all")
    return count


def render_histories(output):
    # Keep the old model-free driver usable without plotting packages for old results/dry runs.
    if any(output.glob("seed_*/*/loss_history.jsonl")):
        from plot_history import render
        render(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", type=Path, help="Rebuild reports without running any training")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/palu_mean_ablation" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="0,1,2,3", help="Visible GPU IDs; multi-GPU training uses ZeRO-3, evaluation uses the first GPU")
    parser.add_argument("--main-process-port", type=int, default=29513)
    parser.add_argument("--keep-model", action="store_true", help="Retain final weights; default deletes them after successful evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--model", choices=["Llama-3.1-8B-Instruct", "Llama-2-7b-chat-hf", "Llama-2-7b-hf"], default="Llama-3.1-8B-Instruct")
    parser.add_argument("--model-path", help="Same initial checkpoint for every arm")
    parser.add_argument("--model-revision", help="Optional immutable Hub checkpoint revision")
    parser.add_argument("--retain-logs", type=Path, help="Matched Retain model TOFU_EVAL.json")
    parser.add_argument("--forget-split", default="forget05")
    parser.add_argument("--holdout-split", default="holdout05")
    parser.add_argument("--retain-split", default="retain95")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--first-n", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device training batch size")
    parser.add_argument("--grad-accum", type=int, help="Default chooses global effective batch 32")
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-grad-norm", type=float, default=1.)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-dataset", help="Optional local TOFU evaluation dataset")
    parser.add_argument("--classifier-model", help="Optional local fluency classifier")
    parser.add_argument("--override", action="append", default=[], help="Shared Hydra override for both arms")
    parser.add_argument("--skip-eval", action="store_true", help="Smoke/performance only; does not measure forgetting")
    parser.add_argument("--dry-run", action="store_true", help="Write the exact plan without loading models")
    parser.add_argument("--retain-model-path", help="Enable diagnostic KL against this separate Retain-only checkpoint")
    parser.add_argument("--retain-model-revision", help="Optional immutable Retain-only Hub revision")
    parser.add_argument("--probe-every", type=int, default=5)
    parser.add_argument("--probe-forget-samples", type=sample_count, default=32)
    parser.add_argument("--probe-retain-samples", type=sample_count, default=32)
    parser.add_argument("--probe-seed", type=int, default=20261011)
    parser.add_argument("--probe-temperature", type=float, default=1.0)
    parser.add_argument("--no-loss-history", action="store_false", dest="track_loss_components", default=True,
                        help="Disable loss history for performance-only measurements")
    args = parser.parse_args()
    if args.summarize:
        return args
    gpu_ids = [value.strip() for value in args.gpu.split(",")]
    if any(not value.isdecimal() for value in gpu_ids) or len(gpu_ids) != len(set(gpu_ids)):
        parser.error("--gpu must contain unique non-negative GPU IDs, e.g. 0,1,2,3")
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        parser.error("Launch this driver once with python; it launches distributed workers itself")
    args.gpu = ",".join(gpu_ids)
    args.num_processes = len(gpu_ids)
    if not 1 <= args.main_process_port <= 65535:
        parser.error("--main-process-port must be between 1 and 65535")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.grad_accum is None:
        micro_batch = args.batch_size * args.num_processes
        if 32 % micro_batch:
            parser.error("GPU count x batch size must divide 32, or set --grad-accum explicitly")
        args.grad_accum = 32 // micro_batch
    args.effective_batch_size = args.num_processes * args.batch_size * args.grad_accum
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be unique")
    if min(args.batch_size, args.grad_accum, args.top_k, args.eval_batch_size, args.epochs, args.lr) <= 0:
        parser.error("Batch sizes, top-k, epochs and learning rate must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be positive or -1")
    if (args.probe_every <= 0 or not math.isfinite(args.probe_temperature) or args.probe_temperature <= 0
            or not all(math.isfinite(value) and value >= 0 for value in (args.alpha, args.gamma))):
        parser.error("Probe interval/temperature must be positive; alpha/gamma must be finite and nonnegative")
    if args.retain_model_path and not args.track_loss_components:
        parser.error("Retain-reference diagnostics require loss history")
    try:
        validate_overrides(args.override)
    except ValueError as exc:
        parser.error(str(exc))
    args.model_path = args.model_path or f"open-unlearning/tofu_{args.model}_full"
    args.output = args.output.expanduser().resolve()
    args.retain_logs = (args.retain_logs or ROOT / f"saves/eval/tofu_{args.model}_{args.retain_split}/TOFU_EVAL.json").expanduser().resolve()
    if not args.dry_run and not args.skip_eval:
        if not args.retain_logs.is_file():
            parser.error("--retain-logs must point to the matched Retain model's TOFU_EVAL.json")
        try:
            reference = read_json(args.retain_logs)["forget_truth_ratio"]["value_by_index"]
            if not reference:
                raise ValueError("Empty reference scores")
        except (KeyError, ValueError) as exc:
            parser.error(f"Invalid Retain reference: {exc}")
    return args


def main():
    args = parse_args()
    if args.summarize:
        report = summarize(args.summarize.expanduser().resolve())
        render_histories(args.summarize.expanduser().resolve())
        print(f"Report: {args.summarize / 'comparison.md'} ({report['status']})")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"seeds": args.seeds, "evaluation_enabled": not args.skip_eval,
                "train_cuda_visible_devices": args.gpu, "eval_cuda_visible_devices": args.gpu.split(",")[0],
                "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "code_and_data_sha256": input_snapshot(args), "runs": []}
    for index, seed in enumerate(args.seeds):
        order = list(VARIANTS) if index % 2 == 0 else list(reversed(VARIANTS))
        for variant in order:
            run = args.output / f"seed_{seed}" / variant
            train, evaluate = build_commands(args, run, seed, variant)
            manifest["runs"].append({"seed": seed, "variant": variant, "output_dir": str(run),
                                     "train_argv": train, "eval_argv": evaluate})
    if args.retain_model_path:
        train = manifest["runs"][0]["train_argv"]
        request = {"overrides": train[train.index("--config-name=unlearn.yaml") + 1:],
                   "retain_model_path": args.retain_model_path, "retain_model_revision": args.retain_model_revision,
                   "probe_seed": args.probe_seed, "forget_samples": args.probe_forget_samples,
                   "retain_samples": args.probe_retain_samples, "cache_path": str(args.output / "reference_cache.pt")}
        write_json(args.output / "diagnostic_request.json", request)
        manifest["reference_precompute_argv"] = [args.python, str(ROOT / "scripts/palu_mean_ablation/prepare_diagnostics.py"),
                                               "--request", str(args.output / "diagnostic_request.json")]
        manifest["timing_scope"] = "training runtime includes periodic diagnostics; reference precompute is shared and separate"
    write_json(args.output / "manifest.json", manifest)
    if args.dry_run:
        print(f"Plan: {args.output / 'manifest.json'}")
        print(f"Training GPUs: {args.gpu}; evaluation GPU: {args.gpu.split(',')[0]}; "
              f"effective batch: {args.num_processes} x {args.batch_size} x {args.grad_accum} = {args.effective_batch_size}; "
              f"keep weights: {args.keep_model}")
        if args.retain_model_path:
            print("Reference precompute: " + shlex.join(manifest["reference_precompute_argv"]))
        for run in manifest["runs"]:
            print(shlex.join(["env", f"CUDA_VISIBLE_DEVICES={args.gpu}", *run["train_argv"]]))
            if run["eval_argv"]:
                print(shlex.join(["env", f"CUDA_VISIBLE_DEVICES={args.gpu.split(',')[0]}", *run["eval_argv"]]))
        return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false"}
    eval_env = {**env, "CUDA_VISIBLE_DEVICES": args.gpu.split(",")[0]}
    if args.num_processes > 1:
        preflight = [args.python, "-c", "import importlib.metadata, torch; "
                     f"assert torch.cuda.is_available() and torch.cuda.device_count() == {args.num_processes}, "
                     "'Requested GPUs are not all available'; print('DeepSpeed:', importlib.metadata.version('deepspeed'))"]
        code, _ = execute(preflight, args.output / "environment_check.log", env)
        if code:
            raise RuntimeError(f"Multi-GPU environment check failed; see {args.output / 'environment_check.log'}")
    probe = [args.python, str(ROOT / "scripts/palu_mean_ablation/probe_gradients.py"),
             "--output", str(args.output / "gradient_probe.json")]
    code, _ = execute(probe, args.output / "gradient_probe.log", env)
    if code:
        raise RuntimeError(f"Gradient probe failed; see {args.output / 'gradient_probe.log'}")
    if args.retain_model_path:
        code, elapsed = execute(manifest["reference_precompute_argv"], args.output / "reference_precompute.log", env)
        if code:
            raise RuntimeError(f"Reference precompute failed; see {args.output / 'reference_precompute.log'}")
        args.probe_cache_sha256 = hash_file(args.output / "reference_cache.pt")
        manifest["reference_cache"] = read_json(args.output / "reference_cache.json")
        manifest["reference_precompute_wall_seconds"] = elapsed
        for planned in manifest["runs"]:
            planned["train_argv"], planned["eval_argv"] = build_commands(args, Path(planned["output_dir"]),
                                                                        planned["seed"], planned["variant"])
        write_json(args.output / "manifest.json", manifest)
    try:
        for planned in manifest["runs"]:
            if input_snapshot(args) != manifest["code_and_data_sha256"]:
                raise RuntimeError("Experiment source or input data changed during the comparison")
            if args.retain_model_path and hash_file(args.output / "reference_cache.pt") != args.probe_cache_sha256:
                raise RuntimeError("Retain reference cache changed during the comparison")
            run = Path(planned["output_dir"])
            run.mkdir(parents=True)
            result = {"seed": planned["seed"], "variant": planned["variant"], "status": "started"}
            write_json(run / "result.json", result)
            print(f"Training seed={planned['seed']} variant={planned['variant']}; log={run / 'train.log'}", flush=True)
            code, _ = execute(planned["train_argv"], run / "train.log", env)
            result.update(status="train_failed" if code else "trained", train_exit_code=code)
            write_json(run / "result.json", result)
            if code:
                raise RuntimeError(f"Training failed; see {run / 'train.log'}")
            merge_telemetry(run, args.num_processes)
            if planned["eval_argv"]:
                print(f"Evaluating seed={planned['seed']} variant={planned['variant']}", flush=True)
                code, elapsed = execute(planned["eval_argv"], run / "eval.log", eval_env)
                result.update(eval_exit_code=code, eval_wall_seconds=elapsed)
                if code:
                    result["status"] = "eval_failed"
                    write_json(run / "result.json", result)
                    raise RuntimeError(f"Evaluation failed; see {run / 'eval.log'}")
            if input_snapshot(args) != manifest["code_and_data_sha256"]:
                raise RuntimeError("Experiment source or input data changed during the comparison")
            result["metrics"] = collect_metrics(run, args.retain_logs, planned["eval_argv"] is not None)
            result["status"] = "completed"
            write_json(run / "result.json", result)
            # Verify any completed pair before discarding either arm's weights.
            pair_runs = [args.output / f"seed_{planned['seed']}" / name for name in VARIANTS]
            if all((path / "result.json").is_file() and read_json(path / "result.json").get("status") == "completed"
                   for path in pair_runs):
                summarize(args.output)
                render_histories(args.output)
                for path in pair_runs:
                    cleanup_completed_weights(path, args.keep_model)
    finally:
        summarize(args.output)
        render_histories(args.output)
    print(f"Comparison: {args.output / 'comparison.md'}")


if __name__ == "__main__":
    main()
