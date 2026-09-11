#!/usr/bin/env python3
"""Run one pinned upstream NPO baseline without importing PALU's trainer/configs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


UPSTREAM_COMMIT = "4ad738aaf60f6a4385f6e2506d01da99e76c31f3"
MODEL = "Llama-3.1-8B-Instruct"
MODEL_ID = f"open-unlearning/tofu_{MODEL}_full"
WEIGHT_FILE = re.compile(r"(?:model(?:-\d+-of-\d+)?\.safetensors|pytorch_model(?:-\d+-of-\d+)?\.bin|model\.safetensors\.index\.json|pytorch_model\.bin\.index\.json)")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def commands(args):
    """Hydra overrides follow upstream scripts/tofu_unlearn.sh, selecting one run."""
    task = f"official_npo_{MODEL}_forget05_seed{args.seed}"
    common = [
        f"model={MODEL}", f"task_name={task}", "forget_split=forget05",
        "holdout_split=holdout05", f"retain_logs_path={args.retain_logs}",
    ]
    train_args = [
        "--config-name=unlearn.yaml", "experiment=unlearn/tofu/default.yaml",
        "trainer=NPO", *common, "retain_split=retain95",
        f"model.model_args.pretrained_model_name_or_path={MODEL_ID}",
        "trainer.args.per_device_train_batch_size=4",
        "trainer.args.gradient_accumulation_steps=4",
        "trainer.args.ddp_find_unused_parameters=true",
        "trainer.args.gradient_checkpointing=true", f"trainer.args.seed={args.seed}",
        f"paths.output_dir={args.output_dir}",
    ]
    train = [
        args.python, "-m", "accelerate.commands.launch",
        "--config_file", "configs/accelerate/default_config.yaml",
        "--num_processes", "2", "--main_process_port", str(args.port),
        "src/train.py", *train_args,
    ]
    evaluate = [
        args.python, "src/eval.py", "experiment=eval/tofu/default.yaml", *common,
        f"model.model_args.pretrained_model_name_or_path={args.output_dir}",
        f"paths.output_dir={args.output_dir / 'evals'}", "eval.tofu.overwrite=true",
    ]
    resolve = [args.python, "src/train.py", *train_args, "--cfg", "job", "--resolve"]
    return {"resolve_config": resolve, "train": train, "evaluate": evaluate}


def check_upstream(root):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Upstream HEAD must be {UPSTREAM_COMMIT}; got {commit}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--",
         "src", "configs", "scripts", "requirements.txt"], cwd=root, text=True,
    ).strip()
    if dirty:
        raise ValueError("Upstream training/evaluation sources must be clean:\n" + dirty)
    for relative in ["src/train.py", "src/eval.py", "configs/accelerate/default_config.yaml"]:
        if not (root / relative).is_file():
            raise ValueError(f"Upstream checkout is incomplete: {relative}")


def check_environment(args):
    # Run in the selected training interpreter; the launcher itself needs only stdlib.
    code = """
import importlib.metadata as m, json, pathlib, sys
from packaging.specifiers import SpecifierSet
versions, errors = {}, []
for line in pathlib.Path('requirements.txt').read_text().splitlines():
    line = line.split('#', 1)[0].strip()
    if '==' not in line:
        continue
    name, expected = line.split('==', 1)
    try:
        actual = m.version(name)
    except m.PackageNotFoundError:
        actual = None
    versions[name] = actual
    if actual is None or not SpecifierSet('==' + expected).contains(actual, prereleases=True):
        errors.append(f'{name}: installed={actual}, required={expected}')
if errors:
    raise SystemExit('Use a separate upstream environment:\\n' + '\\n'.join(errors))
import torch, flash_attn
versions['flash-attn'] = m.version('flash-attn')
if torch.cuda.device_count() != 2:
    raise SystemExit('Exactly two visible CUDA GPUs are required for this baseline.')
print(json.dumps({'python': sys.version, 'packages': versions,
                  'gpus': [torch.cuda.get_device_name(i) for i in range(2)]}))
"""
    return json.loads(subprocess.check_output(
        [args.python, "-c", code], cwd=args.upstream_dir, env=train_env(args), text=True,
    ))


def train_env(args):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Prevent a caller's local PALU PYTHONPATH from shadowing the official modules.
    env.pop("PYTHONPATH", None)
    return env


def run_command(command, args, logfile, evaluation=False):
    env = train_env(args)
    if evaluation:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu.split(",")[0]
    print(f"Running; log: {logfile}", flush=True)
    with logfile.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=args.upstream_dir, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def check_evaluation(output):
    raw = json.loads((output / "evals/TOFU_EVAL.json").read_text())
    summary = json.loads((output / "evals/TOFU_SUMMARY.json").read_text())
    for name in ("forget_quality", "model_utility"):
        value = summary.get(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Missing/non-finite evaluation metric: {name}")
        if not isinstance(raw.get(name), dict) or "agg_value" not in raw[name]:
            raise ValueError(f"Missing raw evaluation record: {name}")
    return summary


def remove_weights(output):
    # Only weights owned by this new run, after successful evaluation. Keep logs/configs.
    files = sorted(p for p in output.iterdir() if WEIGHT_FILE.fullmatch(p.name))
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError("Refusing to remove non-regular model artifacts")
    for path in files:
        path.unlink()
    return [p.name for p in files]


def execute(args):
    plan = commands(args)
    for phase, command in plan.items():
        print(f"[{phase}] cwd={args.upstream_dir}\n{shlex.join(command)}")
    if args.dry_run:
        print("DRY RUN: no training, downloads, environment changes, or output writes.")
        return
    check_upstream(args.upstream_dir)
    if args.output_dir == args.upstream_dir or args.upstream_dir in args.output_dir.parents:
        raise ValueError("Use an output directory outside the official source checkout")
    if args.output_dir.exists():
        raise ValueError("--output-dir must be NEW; existing results will not be overwritten")
    if not args.retain_logs.is_file():
        raise ValueError("--retain-logs must point to the matching Retain95 TOFU_EVAL.json")
    retain = json.loads(args.retain_logs.read_text())
    if not isinstance(retain.get("forget_truth_ratio"), dict):
        raise ValueError("Retain log lacks forget_truth_ratio; do not pass TOFU_SUMMARY.json")
    environment = check_environment(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit_dir = args.output_dir / "baseline_audit"
    audit_dir.mkdir()
    contract = {
        "upstream_commit": UPSTREAM_COMMIT, "upstream_dir": str(args.upstream_dir),
        "model_id": MODEL_ID, "seed": args.seed, "commands": plan,
        "gpu_ids": args.gpu, "environment": environment,
        "retain_logs_path": str(args.retain_logs),
        "retain_logs_sha256": hashlib.sha256(args.retain_logs.read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "keep_model_weights": args.keep_model,
        "scope": "Unmodified upstream full run. Per-component first-step audit is a separate follow-up.",
    }
    write_json(audit_dir / "run_contract.json", contract)
    state = {"status": "resolving_config", "upstream_commit": UPSTREAM_COMMIT}
    state_path = audit_dir / "status.json"
    try:
        write_json(state_path, state)
        run_command(plan["resolve_config"], args, audit_dir / "resolved_config.yaml")
        state["status"] = "training"
        write_json(state_path, state)
        run_command(plan["train"], args, audit_dir / "train.log")
        trainer_state = json.loads((args.output_dir / "trainer_state.json").read_text())
        planned_steps = trainer_state.get("max_steps", 0)
        if planned_steps <= 0 or trainer_state.get("global_step") != planned_steps:
            raise ValueError("Trainer did not reach its planned optimizer-step count")
        state.update(status="evaluating", global_step=trainer_state["global_step"],
                     reported_epoch=trainer_state.get("epoch"))
        write_json(state_path, state)
        run_command(plan["evaluate"], args, audit_dir / "eval.log", evaluation=True)
        state.update(status="evaluation_complete", metrics=check_evaluation(args.output_dir))
        write_json(state_path, state)
        if not args.keep_model:
            state["removed_weight_files"] = remove_weights(args.output_dir)
        state["status"] = "completed"
        write_json(state_path, state)
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        write_json(state_path, state)
        raise
    print(f"Completed: {args.output_dir / 'evals/TOFU_SUMMARY.json'}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True,
                        help=f"Separate clean OpenUnlearning checkout at {UPSTREAM_COMMIT}")
    parser.add_argument("--output-dir", type=Path, required=True, help="Exact new result directory")
    parser.add_argument("--retain-logs", type=Path, required=True, help="Matching Retain95 TOFU_EVAL.json")
    parser.add_argument("--python", default=sys.executable, help="Python from a separate upstream environment")
    parser.add_argument("--gpu", default="0,1", help="Two distinct GPU IDs; first is used for evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=29501)
    parser.add_argument("--keep-model", action="store_true", help="Keep final weights after successful evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing anything")
    args = parser.parse_args(argv)
    gpu_ids = args.gpu.split(",")
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(s.isdigit() for s in gpu_ids):
        parser.error("--gpu requires two distinct numeric IDs, e.g. 0,1")
    if args.seed < 0 or not 1 <= args.port <= 65535:
        parser.error("--seed must be nonnegative; --port must be in 1..65535")
    for name in ("upstream_dir", "output_dir", "retain_logs"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    try:
        execute(parse_args())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
