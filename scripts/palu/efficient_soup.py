#!/usr/bin/env python3
"""PALU + EfficientSoup: immutable inputs, fixed proxy, one live model."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="Initial Full checkpoint: local directory or Hub ID")
    parser.add_argument("--revision", help="Revision of the original Hub model")
    parser.add_argument("--candidates", nargs=2, type=Path, required=True, help="Two final PALU safetensors directories")
    parser.add_argument("--output", type=Path, required=True, help="New directory, never overwritten")
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct", help="Repository model/template config name")
    parser.add_argument("--dataset", required=True, help="TOFU local directory or Hub dataset ID")
    parser.add_argument("--forget-config", default="forget05_perturbed")
    parser.add_argument("--retain-config", default="retain_perturbed")
    parser.add_argument("--proxy-questions", type=int, default=64, help="Unique questions per split; remainder held out")
    parser.add_argument("--split-seed", type=int, default=20261013)
    parser.add_argument("--depth", type=int, default=3, help="Midpoints per edge (3 + 2*depth proxy evaluations maximum)")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32", "float16"], default="bfloat16")
    parser.add_argument("--dry-run", action="store_true", help="Write plan only; no model/data loading")
    parser.add_argument("--full-eval", action="store_true", help="Also run standard TOFU on Full, both seeds, edge, soup")
    parser.add_argument("--forget-split", default="forget05")
    parser.add_argument("--holdout-split", default="holdout05")
    parser.add_argument("--retain-logs", type=Path)
    parser.add_argument("--classifier-model")
    args = parser.parse_args(argv)
    if min(args.proxy_questions, args.depth, args.max_length, args.batch_size) < 1:
        parser.error("Counts, depth, max-length and batch-size must be positive")
    if Path(args.model).name != args.model or not (ROOT / "configs/model" / f"{args.model}.yaml").is_file():
        parser.error("--model must name an existing repository model config")
    if args.full_eval and (args.retain_logs is None or (not args.dry_run and not args.retain_logs.is_file())):
        parser.error("--full-eval requires existing --retain-logs")
    if args.full_eval and args.forget_config != args.forget_split + "_perturbed":
        parser.error("Full evaluation and proxy Forget splits must match")
    args.output = args.output.expanduser().resolve()
    args.candidates = [path.expanduser().resolve() for path in args.candidates]
    if len(set(args.candidates)) != 2:
        parser.error("Candidates must be distinct checkpoint directories")
    return args


def resolve_original(original, revision):
    if Path(original).expanduser().is_dir():
        return Path(original).expanduser().resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(original, revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]))


def validate_compatibility(paths):
    from transformers import AutoConfig, AutoTokenizer
    configs, tokenizers = [], []
    ignore = {"_name_or_path", "transformers_version", "torch_dtype", "architectures", "use_cache",
              "_attn_implementation_autoset"}
    for path in paths:
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        if getattr(config, "quantization_config", None):
            raise ValueError("Quantized checkpoints are unsupported; use full floating weights")
        configs.append({k: v for k, v in config.to_dict().items() if k not in ignore})
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an existing EOS token")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizers.append(tokenizer)
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("Model configs differ; all ingredients must share architecture and semantics")
    def signature(tokenizer):
        backend = json.loads(tokenizer.backend_tokenizer.to_str()) if tokenizer.is_fast else None
        if backend:
            backend.pop("padding", None)
            backend.pop("truncation", None)
        payload = {"vocab": tokenizer.get_vocab(), "special": tokenizer.special_tokens_map,
                   "backend": backend, "chat_template": tokenizer.chat_template,
                   "class": type(tokenizer).__name__}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    signatures = [signature(t) for t in tokenizers]
    if len(set(signatures)) != 1:
        raise ValueError("Ingredient tokenizers differ")
    return tokenizers[0], signatures[0]


def candidate_provenance(paths, original=None, revision=None, forget_split=None, template=None):
    records = []
    normalized = []
    for path in paths:
        config_path = path / "resolved_config.json"
        if not config_path.is_file():
            records.append({"path": str(path), "provenance": "unverified: no resolved_config.json"})
            continue
        config = json.loads(config_path.read_text())
        if config["trainer"]["handler"] != "PALU":
            raise ValueError(f"Candidate is not a PALU run: {path}")
        initial = config["model"]["model_args"].get("pretrained_model_name_or_path")
        if original is not None and initial is not None:
            same = str(initial) == str(original)
            if Path(str(initial)).is_dir() and Path(str(original)).is_dir():
                same = Path(str(initial)).resolve() == Path(str(original)).resolve()
            if not same:
                raise ValueError("--original differs from the PALU initialization recorded in resolved_config.json")
        recorded_revision = config["model"]["model_args"].get("revision")
        if recorded_revision != revision:
            raise ValueError("Original revision differs from the PALU candidate revision")
        if forget_split and config.get("forget_split", forget_split) != forget_split:
            raise ValueError("Candidate and evaluation Forget splits differ")
        if template is not None and config["model"]["template_args"] != template:
            raise ValueError("Proxy template differs from PALU training template")
        seed = config["trainer"]["args"]["seed"]
        records.append({"path": str(path), "seed": seed, "config": config,
                        "provenance": "PALU config recorded; common initialization must match --original"})
        # Compare scientific settings while allowing run-specific diagnostics/output.
        clone = json.loads(json.dumps(config))
        for key in ("task_name", "seed", "paths", "hydra"):
            clone.pop(key, None)
        for key in ("output_dir", "logging_dir", "run_name", "seed", "data_seed"):
            clone["trainer"]["args"].pop(key, None)
        clone["trainer"]["method_args"].pop("diagnostics", None)
        normalized.append(clone)
    if len(normalized) == 2:
        if records[0]["seed"] == records[1]["seed"]:
            raise ValueError("Use two different seeds for the first EfficientSoup experiment")
        if normalized[0] != normalized[1]:
            raise ValueError("PALU candidate settings differ beyond random seed and output/diagnostics")
    return records


def full_eval_commands(args, paths):
    quote = lambda key, value: f"{key}={json.dumps(str(value), ensure_ascii=False)}"
    commands = {}
    for name, path in paths.items():
        command = [sys.executable, str(ROOT / "src/eval.py"), "experiment=eval/tofu/default",
            f"model={args.model}", quote("model.model_args.pretrained_model_name_or_path", path),
            quote("model.tokenizer_args.pretrained_model_name_or_path", path),
            "model.model_args.attn_implementation=sdpa", f"model.model_args.torch_dtype={args.dtype}",
            quote("model.model_args.device_map", args.device),
            f"forget_split={args.forget_split}", f"holdout_split={args.holdout_split}",
            quote("retain_logs_path", args.retain_logs), quote("paths.output_dir", args.output / "full_eval" / name),
            quote("task_name", f"efficient_soup_{name}"), "eval.tofu.overwrite=true",
            f"eval.tofu.batch_size={args.batch_size}", quote("+tofu_dataset_path", args.dataset)]
        command.append(quote("eval.tofu.metrics.forget_Q_A_gibberish.device", args.device))
        if args.classifier_model:
            for key in ("classifier_model_args", "classifier_tokenization_args"):
                command.append(quote(f"eval.tofu.metrics.forget_Q_A_gibberish.{key}.pretrained_model_name_or_path",
                                     args.classifier_model))
        commands[name] = command
    return commands


def run(args, manifest):
    import torch
    import transformers
    from omegaconf import OmegaConf
    from efficient_soup.search import efficient_search
    from efficient_soup.weights import WeightMixer, sha256
    from efficient_soup.proxy import prepare_split, evaluate_pair
    started = time.perf_counter()
    torch.manual_seed(args.split_seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu --dtype float32 for tiny smoke tests")
    original = resolve_original(args.original, args.revision)
    paths = [original, *args.candidates]
    tokenizer, fingerprint = validate_compatibility(paths)
    template = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/model" / f"{args.model}.yaml"))["template_args"]
    manifest.update(candidate_provenance=candidate_provenance(args.candidates, args.original, args.revision,
                                                              args.forget_config.removesuffix("_perturbed"), template), tokenizer_sha256=fingerprint,
                    runtime={"torch": torch.__version__, "transformers": transformers.__version__,
                             "cuda": torch.version.cuda, "device": args.device,
                             "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None})
    manifest["template"] = template
    splits = {key: prepare_split(args.dataset, name, tokenizer, template, args.max_length,
                                 args.proxy_questions, args.split_seed)
              for key, name in (("forget", args.forget_config), ("retain", args.retain_config))}
    overlap = set(splits["forget"]["metadata"]["question_sha256"]) & set(splits["retain"]["metadata"]["question_sha256"])
    if overlap:
        raise ValueError("Forget and Retain proxy pools contain overlapping questions")
    write_json(args.output / "samples.json", {k: v["metadata"] for k, v in splits.items()})
    model = transformers.AutoModelForCausalLM.from_pretrained(original, local_files_only=True,
        torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa", low_cpu_mem_usage=True).to(args.device).eval()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    mixer = WeightMixer(model, paths)
    try:
        manifest["ingredients"] = [source.fingerprint() for source in mixer.sources]
        manifest["setup_wall_seconds"] = time.perf_counter() - started
        write_json(args.output / "manifest.json", manifest)
        def timed_score(weights, stage, partition="proxy"):
            begin = time.perf_counter()
            mixer.apply(weights)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            merged = time.perf_counter()
            scores = evaluate_pair(model, splits, partition, args.batch_size, tokenizer.pad_token_id)
            scores.update(merge_wall_seconds=merged - begin, eval_wall_seconds=time.perf_counter() - merged)
            print(f"{stage}: weights={weights}, DS_ES={scores['ds_es']:.6f}, "
                  f"ES_F={scores['es_forget']:.6f}, ES_R={scores['es_retain']:.6f}", flush=True)
            if partition == "proxy":
                with (args.output / "search.jsonl").open("a") as stream:
                    stream.write(json.dumps({"weights": weights, "stage": stage, **scores}, allow_nan=False) + "\n")
            return scores
        result = efficient_search(timed_score, args.depth)
        # Freeze all choices before observing held-out scores.
        write_json(args.output / "selection.json", result)
        comparisons = {"original": result["original"], "candidate_1": result["candidates"][0],
            "candidate_2": result["candidates"][1], "anchor_edge": result["stage_one"], "soup": result["final"]}
        heldout, cache = {}, {}
        for name, record in comparisons.items():
            weights = tuple(record["weights"])
            if weights not in cache:
                cache[weights] = timed_score(weights, name + "_heldout", "heldout")
            heldout[name] = {"weights": weights, **cache[weights]}
        write_json(args.output / "heldout.json", heldout)
        save_begin = time.perf_counter()
        for name in ("anchor_edge", "soup"):
            mixer.apply(comparisons[name]["weights"])
            model.save_pretrained(args.output / name, safe_serialization=True, max_shard_size="4GB")
            tokenizer.save_pretrained(args.output / name)
        manifest["save_wall_seconds"] = time.perf_counter() - save_begin
        manifest["proxy_evaluations"] = len(result["history"])
        manifest["heldout_evaluations"] = len(cache)
        manifest["search_merge_wall_seconds"] = sum(x["merge_wall_seconds"] for x in result["history"])
        manifest["search_eval_wall_seconds"] = sum(x["eval_wall_seconds"] for x in result["history"])
        manifest["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None
        for source in mixer.sources:
            source.assert_unchanged()
    finally:
        mixer.close()
    del mixer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.full_eval:
        commands = full_eval_commands(args, {"original": original, "candidate_1": paths[1], "candidate_2": paths[2],
                                            "anchor_edge": args.output / "anchor_edge", "soup": args.output / "soup"})
        write_json(args.output / "full_eval_commands.json", commands)
        summary = {}
        from scipy.stats import ks_2samp
        reference = json.loads(args.retain_logs.read_text())
        scores = lambda log: [x["score"] for x in log["forget_truth_ratio"]["value_by_index"].values()]
        manifest["retain_logs_sha256"] = sha256(args.retain_logs)
        for name, command in commands.items():
            begin = time.perf_counter()
            with (args.output / f"full_eval_{name}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            directory = args.output / "full_eval" / name
            summary[name] = json.loads((directory / "TOFU_SUMMARY.json").read_text())
            raw = json.loads((directory / "TOFU_EVAL.json").read_text())
            summary[name]["forget_ks_statistic"] = float(ks_2samp(scores(raw), scores(reference)).statistic)
            summary[name]["eval_wall_seconds"] = time.perf_counter() - begin
            write_json(args.output / "full_eval_summary.json", summary)
    manifest.update(status="completed", soup_total_wall_seconds=time.perf_counter() - started)
    write_json(args.output / "manifest.json", manifest)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = sorted((ROOT / "src/efficient_soup").glob("*.py")) + [Path(__file__), ROOT / "configs/model" / f"{args.model}.yaml"]
    files += sorted((ROOT / "src/data").glob("*.py"))
    manifest = {"schema": 1, "status": "planned" if args.dry_run else "started",
        "settings": {k: [str(x) for x in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v
                     for k, v in vars(args).items()},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "protocol": {"search": "greedy_midpoint_replace_worse_endpoint_v1", "depth": args.depth,
            "maximum_proxy_evaluations": 3 + 2 * args.depth, "ds_scale": "raw and 100x, both recorded",
            "es": "whole answer incl. EOS/template terminators; longest fully correct suffix; empty invalid",
            "heldout": "disjoint questions, not necessarily disjoint authors; never used to select weights",
            "full_eval": "standard TOFU overlaps proxy pool; descriptive benchmark, not independent held-out validation",
            "cost": "soup wall times exclude candidate training; no automatic speedup claim",
            "reproduction": "paper-inspired implementation; bracket/tie/budget rules are explicit engineering choices"}}
    write_json(args.output / "manifest.json", manifest)
    if args.dry_run:
        print(f"Plan: {args.output / 'manifest.json'}")
        return
    try:
        run(args, manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.output / "manifest.json", manifest)
        raise
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
