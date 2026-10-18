#!/usr/bin/env python3
"""Run one seed from a resolved PALU config with fixed-loss and TOFU epoch eval.

Example: CUDA_VISIBLE_DEVICES=2 python scripts/palu/seed_epoch_experiment.py
  --config experiment.json --output artifacts/seed_100 --seed 100
Dry runs load no model. --check-data additionally tokenizes both datasets on CPU.
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


def read_config(path):
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:
            raise ValueError("YAML requires PyYAML; a resolved JSON config also works") from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a complete, resolved config mapping")
    if "${" in json.dumps(value):
        raise ValueError("Config contains unresolved interpolations; provide resolved_config.json/yaml")
    return value


def configured_run(source, output, seed, overrides=()):
    config = copy.deepcopy(source)
    for assignment in overrides:
        key, separator, encoded = assignment.partition("=")
        if not separator or not key:
            raise ValueError("--set must have dotted.path=JSON syntax")
        value, cursor = json.loads(encoded), config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise ValueError(f"Unknown config mapping in --set: {key}")
            cursor = cursor[part]
        cursor[parts[-1]] = value
    if config.get("trainer", {}).get("handler") != "PALU":
        raise ValueError("This entrypoint only accepts PALU configs")
    method = config["trainer"].setdefault("method_args", {})
    if method.get("target_mode", "mean") != "mean" or method.get("retain_loss_type", "NLL") != "NLL" or method.get("non_sensitive_constraint_type", "kl") != "kl":
        raise ValueError("Required PALU objective: vocabulary-mean sensitive MSE, Retain NLL, common KL")
    config.setdefault("paths", {})["output_dir"] = str(output)
    config["task_name"] = f"palu_epoch_seed_{seed}"
    args = config["trainer"]["args"]
    args.pop("evaluation_strategy", None)
    args.update(output_dir=str(output), logging_dir=str(output / "logs"), run_name=config["task_name"],
                seed=seed, data_seed=seed, eval_strategy="epoch", eval_on_start=True,
                do_train=True, do_eval=True, save_strategy="no", logging_steps=1, report_to="none")
    if args.get("deepspeed"):
        raise ValueError("This first seed study uses one GPU; remove the distributed DeepSpeed config explicitly")
    config["do_save"] = False
    method.update(track_loss_components=True)
    method.pop("diagnostics", None)
    suites = config.get("eval", {})
    if len(suites) != 1 or "tofu" not in suites:
        raise ValueError("Config must contain exactly eval.tofu")
    tofu = suites["tofu"]
    tofu.update(output_dir=str(output), overwrite=True)
    if not {"forget_quality", "model_utility"}.issubset(tofu.get("metrics", {})):
        raise ValueError("TOFU config requires forget_quality and model_utility metrics")
    # Preserve these metrics' recursive precompute configuration in full. The
    # unrelated top-level classifier/exact-memorization metrics need no forwards.
    tofu["metrics"] = {key: tofu["metrics"][key] for key in ("forget_quality", "model_utility")}
    return config


def protocol_config(config):
    value = copy.deepcopy(config)
    value.pop("task_name", None)
    value["paths"].pop("output_dir", None)
    for key in ("output_dir", "logging_dir", "run_name", "seed", "data_seed"):
        value["trainer"]["args"].pop(key, None)
    value["eval"]["tofu"].pop("output_dir", None)
    return value


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--set", action="append", default=[], help="Explicit override as dotted.path=JSON")
    parser.add_argument("--forget-samples", default="all")
    parser.add_argument("--retain-samples", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-data", action="store_true", help="CPU-only dataset/tokenizer preflight")
    parser.add_argument("--no-save-final-model", action="store_true")
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--verify-observations", action="store_true")
    parser.add_argument("--expected-updates", type=int)
    parser.add_argument("--expected-fixed-data-sha256")
    return parser.parse_args()


def main():
    args = cli()
    output = args.output.resolve()
    source = read_config(args.config)
    if args.eval_every_epochs < 1:
        raise ValueError("--eval-every-epochs must be positive")
    config = configured_run(source, output, args.seed, args.set)
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if visible and len([device for device in visible.split(",") if device.strip()]) != 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one GPU")
    payload = {"schema": 1, "seed": args.seed, "model": config["model"]["handler"],
        "source_config_path": str(args.config.resolve()),
        "source_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "protocol_sha256": hashlib.sha256(json.dumps(protocol_config(config), sort_keys=True).encode()).hexdigest(),
        "output_dir": str(output), "cuda_visible_devices": visible,
        "forget_samples": args.forget_samples, "retain_samples": args.retain_samples,
        "save_final_model": not args.no_save_final_model,
        "metric_scope": ["forget_quality", "model_utility"],
        "num_train_epochs": config["trainer"]["args"].get("num_train_epochs"),
        "max_steps": config["trainer"]["args"].get("max_steps", -1),
        "eval_every_epochs": args.eval_every_epochs,
        "checkpoint_policy": "in-memory evaluation; no intermediate weights; final model retained",
        "batch_size": config["trainer"]["args"].get("per_device_train_batch_size"),
        "gradient_accumulation_steps": config["trainer"]["args"].get("gradient_accumulation_steps"),
        "protocol": "one uninterrupted Trainer run; epoch-end TOFU and fixed full-split loss; preserve RNG and module modes"}
    if args.dry_run and not args.check_data:
        print(json.dumps({**payload, "resolved_config": config}, ensure_ascii=False, indent=2))
        return
    # Import the training stack only after a model-free config dry run succeeds.
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from omegaconf import OmegaConf
    from data import get_collators, get_data
    from model import get_model, get_tokenizer
    from trainer.utils import seed_everything
    from trainer.palu_epoch_eval import PALUEpochTrainer, fixed_indices
    from trainer import load_trainer_args
    from evals import get_evaluators

    cfg = OmegaConf.create(config)
    if args.check_data:
        tokenizer = get_tokenizer(cfg.model.tokenizer_args)
        data = get_data(cfg.data, mode=cfg.get("mode", "unlearn"), tokenizer=tokenizer,
                        template_args=cfg.model.template_args, model_name=cfg.model.handler)
        dataset = data["train"]
        payload["dataset_sizes"] = {split: len(getattr(dataset, split)) for split in ("forget", "retain")}
        payload["fixed_counts"] = {split: len(fixed_indices(payload["dataset_sizes"][split],
            getattr(args, split + "_samples"))) for split in ("forget", "retain")}
        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
    if int(os.getenv("WORLD_SIZE", "1")) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Launch this seed experiment directly with exactly one visible CUDA GPU")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Run directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", config)
    payload.update(status="initializing", started_unix=time.time(),
                   packages={package: importlib.metadata.version(package)
                             for package in ("torch", "transformers", "accelerate", "datasets")},
                   code_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted((ROOT / "src").rglob("*.py"))})
    payload["code_sha256"]["scripts/palu/seed_epoch_experiment.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(output / "manifest.json", payload)
    started = time.perf_counter()
    try:
        seed_everything(args.seed)
        model, tokenizer = get_model(cfg.model)
        data = get_data(cfg.data, mode=cfg.get("mode", "unlearn"), tokenizer=tokenizer,
                        template_args=cfg.model.template_args, model_name=cfg.model.handler)
        collator = get_collators(cfg.collator, tokenizer=tokenizer)
        evaluators = get_evaluators(cfg.eval, template_args=cfg.model.template_args,
                                    model=model, tokenizer=tokenizer)
        trainer_args = load_trainer_args(cfg.trainer.args, data["train"])
        method_args = dict(cfg.trainer.method_args)
        trainer = PALUEpochTrainer(model=model, tokenizer=tokenizer, args=trainer_args,
            train_dataset=data["train"], eval_dataset=data.get("eval"), data_collator=collator,
            evaluators=evaluators, template_args=cfg.model.template_args, **method_args,
            epoch_options={"forget_samples": args.forget_samples, "retain_samples": args.retain_samples,
                           "model_name": config["model"]["handler"],
                           "eval_every_epochs": args.eval_every_epochs,
                           "expected_fixed_data_sha256": args.expected_fixed_data_sha256,
                           "verify_observations": args.verify_observations})
        trainer.train()
        if args.expected_updates is not None and trainer.state.global_step != args.expected_updates:
            raise ValueError(f"Expected {args.expected_updates} updates, got {trainer.state.global_step}")
        # Covers unusual early-stop paths without repeating a completed endpoint.
        trainer.evaluate()
        trainer.save_state()
        if not args.no_save_final_model:
            trainer.save_model(str(output))
        final_manifest = json.loads((output / "manifest.json").read_text())
        final_manifest.update(status="completed", completed_unix=time.time(),
            entrypoint_wall_seconds=time.perf_counter() - started, global_step=trainer.state.global_step,
            epoch=trainer.state.epoch, planned_max_steps=trainer.state.max_steps,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(output / "manifest.json", final_manifest)
    except BaseException as error:
        final_manifest = json.loads((output / "manifest.json").read_text())
        final_manifest.update(status="failed", failed_unix=time.time(), error_type=type(error).__name__,
                              error=str(error), entrypoint_wall_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", final_manifest)
        raise


if __name__ == "__main__":
    main()
