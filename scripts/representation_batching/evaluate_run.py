#!/usr/bin/env python3
"""Evaluate one completed RepresentationNPO final model into <run>/evals."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from representation_batching.report import config_for


def build_command(run, retain_logs, dataset=None, classifier_model=None, batch_size=None):
    config = config_for(run)
    if not (run / "config.json").exists():
        raise ValueError("No saved final model config.json in run directory")
    model = config["model"]["handler"]
    command = [sys.executable, "src/eval.py", "experiment=eval/tofu/default",
        f"model={model}", f"model.model_args.pretrained_model_name_or_path={run}",
        f"model.tokenizer_args.pretrained_model_name_or_path={run}",
        f"forget_split={config['forget_split']}", f"holdout_split={config['holdout_split']}",
        f"retain_logs_path={retain_logs}", f"paths.output_dir={run / 'evals'}",
        f"task_name={run.name}_final_eval", "eval.tofu.overwrite=true"]
    if dataset:
        command.append(f"+tofu_dataset_path={dataset}")
    if classifier_model:
        prefix = "eval.tofu.metrics.forget_Q_A_gibberish"
        command += [f"{prefix}.classifier_model_args.pretrained_model_name_or_path={classifier_model}",
                    f"{prefix}.classifier_tokenization_args.pretrained_model_name_or_path={classifier_model}"]
    if batch_size is not None:
        command.append(f"eval.tofu.batch_size={batch_size}")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--retain-logs", type=Path, required=True, help="Matched Retain95 TOFU_EVAL.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", help="Override all TOFU evaluation dataset paths")
    parser.add_argument("--classifier-model", help="Gibberish detector directory or Hub id")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    run, retain = args.run.expanduser().resolve(), args.retain_logs.expanduser().resolve()
    if not retain.is_file():
        parser.error("--retain-logs must exist")
    if "," in args.gpu:
        parser.error("Use one GPU for the repository TOFU evaluator")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    command = build_command(run, retain, args.dataset, args.classifier_model, args.batch_size)
    if args.dry_run:
        import shlex
        print(shlex.join(command))
    else:
        subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu}, check=True)


if __name__ == "__main__":
    main()
