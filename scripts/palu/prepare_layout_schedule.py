#!/usr/bin/env python3
"""Freeze a full PALU update schedule on CPU, without loading model weights.

Each update always consumes 32 fixed Forget/Retain pairs. Forget permutations
form a continuous stream across dataset traversals; Retain is independently
sampled with replacement. Both layouts slice the same globally padded tensors.
This is a new controlled schedule, not a reconstruction of historical draws.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def selected_sensitive_count(labels, eos, first_n):
    """Count actual shifted PALU targets, including interval-wise first_n."""
    values = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    positions = [i for i, value in enumerate(values[1:])
                 if value >= 0 and (eos is None or value != eos)]
    selected, previous, interval_size = 0, None, 0
    for position in positions:
        if previous is None or position != previous + 1:
            interval_size = 0
        if first_n <= 0 or interval_size < first_n:
            selected += 1
        interval_size += 1
        previous = position
    return selected


def sample_counts(forget_labels, retain_labels, eos, first_n):
    f = forget_labels.tolist() if hasattr(forget_labels, "tolist") else list(forget_labels)
    r = retain_labels.tolist() if hasattr(retain_labels, "tolist") else list(retain_labels)
    return {"sensitive": selected_sensitive_count(f, eos, first_n),
            "common": sum(value == -200 for value in f[1:]),
            "retain": sum(value != -100 for value in r[1:])}


def epoch_evaluation_schedule(forget_size, updates, effective_batch):
    """Evaluate only after the complete update crossing each traversal edge."""
    if min(forget_size, updates, effective_batch) <= 0:
        raise ValueError("Sizes and training budget must be positive")
    exposure = updates * effective_batch
    events = []
    for epoch in range(1, exposure // forget_size + 1):
        step = math.ceil(epoch * forget_size / effective_batch)
        events.append({"step": step, "completed_epoch": epoch,
                       "boundary_exposure": epoch * forget_size,
                       "updated_exposure": step * effective_batch,
                       "overshoot_examples": step * effective_batch - epoch * forget_size,
                       "sample_epoch": step * effective_batch / forget_size})
    if not events or events[-1]["step"] != updates:
        events.append({"step": updates, "completed_epoch": exposure // forget_size,
                       "boundary_exposure": exposure,
                       "updated_exposure": exposure, "overshoot_examples": 0,
                       "sample_epoch": exposure / forget_size, "final_partial_epoch": True})
    return events


def pair_stream(forget_size, retain_size, updates, effective_batch, seed, torch):
    """Dedicated generators leave process-global CPU/CUDA RNG untouched."""
    if min(forget_size, retain_size, updates, effective_batch) <= 0:
        raise ValueError("Sizes and training budget must be positive")
    total = updates * effective_batch
    forget_rng = torch.Generator(device="cpu").manual_seed(seed)
    retain_rng = torch.Generator(device="cpu").manual_seed(seed)
    forget = []
    while len(forget) < total:
        forget.extend(torch.randperm(forget_size, generator=forget_rng).tolist())
    forget = forget[:total]
    retain = torch.randint(retain_size, (total,), generator=retain_rng).tolist()
    return [{"draw_index": i + 1, "epoch_index": i // forget_size + 1,
             "index_in_epoch": i % forget_size, "forget_index": f,
             "retain_index": r} for i, (f, r) in enumerate(zip(forget, retain))]


def tensor_hashes(batch):
    hashes = {}
    for split in ("forget", "retain"):
        for key in ("input_ids", "attention_mask", "labels"):
            tensor = batch[split][key].detach().cpu().contiguous()
            hashes[split + "." + key] = {
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}
    return hashes


def fixed_data_hash(samples, torch):
    """Same canonical unpadded full-split hash as PALUEpochTrainer."""
    digest = hashlib.sha256()
    for split in ("forget", "retain"):
        for index, sample in enumerate(samples[split]):
            digest.update(json.dumps([split, index], separators=(",", ":")).encode())
            for key in ("input_ids", "labels"):
                digest.update(key.encode())
                digest.update(sample[key].to(torch.int64).contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_schedule(samples, collator, *, eos, first_n, updates, effective_batch, seed, torch,
                   records=None, evaluations=None):
    """Return immutable CPU inputs plus independently countable provenance."""
    sizes = {split: len(samples[split]) for split in ("forget", "retain")}
    replay = records is not None
    records = copy.deepcopy(records) if replay else pair_stream(
        sizes["forget"], sizes["retain"], updates, effective_batch, seed, torch)
    if len(records) != updates * effective_batch:
        raise ValueError("Pair replay must contain exactly the requested update budget")
    for pair in records:
        for split in ("forget", "retain"):
            if not 0 <= pair[split + "_index"] < sizes[split]:
                raise ValueError("Pair replay contains an out-of-range dataset index")
    evaluations = copy.deepcopy(evaluations) if evaluations is not None else epoch_evaluation_schedule(
        sizes["forget"], updates, effective_batch)
    if any(not 1 <= event["step"] <= updates for event in evaluations):
        raise ValueError("Evaluation event outside the training budget")
    events = {}
    for event in evaluations:
        events.setdefault(event["step"], []).append(event)
    windows = []
    for index in range(updates):
        pairs = records[index * effective_batch:(index + 1) * effective_batch]
        instances = [{split: {key: samples[split][pair[split + "_index"]][key]
                              for key in ("input_ids", "labels")}
                      for split in ("forget", "retain")} for pair in pairs]
        inputs = collator(instances)
        counts = {name: 0 for name in ("sensitive", "common", "retain")}
        for row, pair in enumerate(pairs):
            observed = sample_counts(inputs["forget"]["labels"][row],
                                     inputs["retain"]["labels"][row], eos, first_n)
            pair.update({name + "_tokens": value for name, value in observed.items()})
            for name, value in observed.items():
                counts[name] += value
        if any(value <= 0 for value in counts.values()):
            raise ValueError(f"Update {index + 1} has an empty global target set: {counts}")
        hashes = tensor_hashes(inputs)
        step = index + 1
        crossed = events.get(step, [])
        window = {"step": step, **inputs, "inputs": inputs,
                  "sample_lengths": {split: [len(samples[split][pair[split + "_index"]]["input_ids"])
                                               for pair in pairs] for split in ("forget", "retain")},
                  "counts": counts, "pair_records": pairs,
                  "pair_sha256": canonical_sha(pairs), "tensor_hashes": hashes,
                  "window_sha256": canonical_sha({"pairs": pairs, "counts": counts, "tensors": hashes}),
                  "global_padding_shapes": {split: list(inputs[split]["input_ids"].shape)
                                             for split in ("forget", "retain")},
                  "epoch_completed": step * effective_batch // sizes["forget"],
                  "epoch_fraction": step * effective_batch / sizes["forget"],
                  "evaluation_boundary": bool(crossed),
                  "epoch_label": crossed[-1]["sample_epoch"] if crossed else None,
                  "completed_epoch_boundaries": [event["completed_epoch"] for event in crossed
                                                 if not event.get("final_partial_epoch")],
                  "evaluation_events": crossed}
        if crossed and "source_epoch_index" in crossed[-1]:
            window["source_epoch_index"] = crossed[-1]["source_epoch_index"]
            window["snapshot_kind"] = "final_tail" if crossed[-1].get("final_partial_epoch") else "seed103_epoch_weight_state"
        windows.append(window)
    fixed_counts = {"sensitive": 0, "common": 0, "retain": 0}
    zero_sensitive = 0
    for sample in samples["forget"]:
        observed = sample_counts(sample["labels"], [], eos, first_n)
        zero_sensitive += int(observed["sensitive"] == 0)
        for name in ("sensitive", "common"):
            fixed_counts[name] += observed[name]
    for sample in samples["retain"]:
        fixed_counts["retain"] += sample_counts([], sample["labels"], eos, first_n)["retain"]
    metadata = {"schema": 1, "seed": seed, "updates": updates,
                "effective_batch": effective_batch, "total_training_pairs": len(records),
                "data_sizes": sizes, "eos_token_id": eos, "first_n": first_n,
                "selection": "replay supplied, recorded F/R pairs" if replay else "continuous Forget randperm epochs; independent Retain randint replacement; separate CPU generators seeded identically",
                "pair_stream_sha256": canonical_sha(records),
                "tensor_schedule_sha256": canonical_sha([window["window_sha256"] for window in windows]),
                "fixed_dataset_sha256": fixed_data_hash(samples, torch),
                "fixed_target_counts": fixed_counts, "forget_zero_sensitive_examples": zero_sensitive,
                "evaluation_steps": sorted({0, *events}), "epoch_evaluation_schedule": evaluations,
                "evaluation_policy": "initial state; first complete optimizer update at/after each 200-example Forget traversal boundary; final update",
                "completed_forget_epochs": len(records) // sizes["forget"],
                "tail_forget_examples": len(records) % sizes["forget"],
                "unique_retain_examples": len({pair["retain_index"] for pair in records}),
                "not_a_reconstruction_of_historical_unrecorded_pairs": not replay,
                "historical_evaluation_difference": "old HF epoch-end evaluation could run with a partially accumulated update; this control evaluates only completed 32-pair updates",
                "collator": {"handler": collator.__class__.__name__,
                             "padding_side": collator.padding_side,
                             "pad_token_id": collator.tokenizer.pad_token_id,
                             "attention_mask": "native input_ids.ne(pad_token_id)",
                             "label_padding": -100,
                             "padding_scope": "each full update window, separately for Forget and Retain; never re-pad microbatches"},
                "normalization": "independent global sensitive/common/retain shifted target-token averages"}
    return {"schema": 1, "windows": windows, "samples": samples, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--effective-batch", type=int, default=32)
    args = parser.parse_args()
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(ROOT / "src"))
    from data import get_collators, get_data
    from model import get_tokenizer
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to replace an existing frozen schedule: {output}")
    source = json.loads(args.source_config.read_text())
    config = copy.deepcopy(source)
    model_path = args.model_path.resolve()
    for section in ("model_args", "tokenizer_args"):
        config["model"][section]["pretrained_model_name_or_path"] = str(model_path)
    method = config["trainer"]["method_args"]
    for key, required in {"gamma": 1.0, "alpha": 0.2, "top_k": 5000, "first_n": 3,
                          "target_mode": "mean", "retain_loss_type": "NLL",
                          "non_sensitive_constraint_type": "kl"}.items():
        if method.get(key) != required:
            raise ValueError(f"Source config must preserve historical {key}={required}")
    if config["trainer"]["args"]["learning_rate"] != 2e-5:
        raise ValueError("Source config must preserve historical learning_rate=2e-5")
    for key in ("diagnostics", "detach_mean_target", "track_loss_components"):
        method.pop(key, None)
    cfg = OmegaConf.create(config)
    tokenizer = get_tokenizer(cfg.model.tokenizer_args)
    dataset = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer,
                       template_args=cfg.model.template_args, model_name=cfg.model.handler)["train"]
    if dataset.anchor != "forget":
        raise ValueError("This controlled schedule requires anchor=forget")
    samples, source_rows = {}, {}
    for split in ("forget", "retain"):
        actual = getattr(dataset, split)
        samples[split] = []
        for index in range(len(actual)):
            sample = actual[index]  # Direct split access: tokenize each source item once.
            if int(sample["index"]) != index:
                raise ValueError("Dataset index is not its frozen source-pool index")
            samples[split].append({key: sample[key].detach().cpu().clone()
                                   for key in ("input_ids", "labels")})
        source_rows[split] = {"fingerprint": getattr(actual.data, "_fingerprint", None),
                              "cache_files": [{"filename": row["filename"],
                                               "sha256": sha256_file(row["filename"])}
                                              for row in getattr(actual.data, "cache_files", [])]}
    if {split: len(samples[split]) for split in samples} != {"forget": 200, "retain": 3800}:
        raise ValueError("Historical Forget05/Retain95 control requires 200/3800 items")
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    if collator.padding_side != "right":
        raise ValueError("Historical control requires right padding")
    payload = build_schedule(samples, collator, eos=tokenizer.eos_token_id,
                             first_n=method["first_n"], updates=args.updates,
                             effective_batch=args.effective_batch, seed=args.seed, torch=torch)
    metadata = payload["metadata"]
    model_cfg = json.loads((model_path / "config.json").read_text())
    if float(model_cfg.get("attention_dropout", 0.0)) != 0.0:
        raise ValueError("Layout control requires zero attention dropout")
    code = ["scripts/palu/prepare_layout_schedule.py", "src/data/qa.py", "src/data/utils.py",
            "src/data/unlearn.py", "src/data/collators.py", "src/trainer/unlearn/palu.py",
            "src/trainer/unlearn/grad_diff.py", "src/trainer/utils.py",
            "configs/accelerate/default_config.yaml", "configs/accelerate/zero_stage3_offload_config.json"]
    metadata.update(model_path=str(model_path), attention_dropout=model_cfg.get("attention_dropout", 0.0),
                    model_file_sha256={str(p.relative_to(model_path)): sha256_file(p)
                                       for p in sorted(model_path.iterdir())
                                       if p.is_file() and (p.suffix == ".safetensors" or p.name in
                                           ("config.json", "model.safetensors.index.json"))},
                    tokenizer_file_sha256={str(p.relative_to(model_path)): sha256_file(p)
                                            for p in sorted(model_path.iterdir())
                                            if p.is_file() and p.suffix in (".json", ".model", ".tiktoken")
                                            and p.name != "model.safetensors.index.json"},
                    source_dataset=source_rows,
                    source_config_path=str(args.source_config.resolve()),
                    source_config_sha256=sha256_file(args.source_config),
                    training_code_sha256={name: sha256_file(ROOT / name) for name in code},
                    packages={name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "datasets", "accelerate")},
                    training_budget={"optimizer_updates": args.updates,
                                     "effective_batch_pairs": args.effective_batch,
                                     "total_pairs": args.updates * args.effective_batch,
                                     "warmup_steps": 6, "scheduler": "linear", "learning_rate": 2e-5})
    output.mkdir(parents=True)
    torch.save(payload, output / "schedule.pt")
    metadata["schedule_pt_sha256"] = sha256_file(output / "schedule.pt")
    for name, value in (("manifest.json", metadata), ("config.json", config),
                        ("epoch_schedule.json", metadata["epoch_evaluation_schedule"])):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (output / "pair_schedule.jsonl").open("w") as handle:
        for window in payload["windows"]:
            for pair in window["pair_records"]:
                handle.write(json.dumps({"step": window["step"], **pair}, separators=(",", ":")) + "\n")
    with (output / "window_manifest.jsonl").open("w") as handle:
        for window in payload["windows"]:
            summary = {key: value for key, value in window.items() if key not in ("forget", "retain", "inputs", "pair_records")}
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "schedule_pt_sha256": metadata["schedule_pt_sha256"],
                      "fixed_dataset_sha256": metadata["fixed_dataset_sha256"],
                      "pair_stream_sha256": metadata["pair_stream_sha256"],
                      "evaluation_steps": metadata["evaluation_steps"],
                      "unique_retain_examples": metadata["unique_retain_examples"]}))


if __name__ == "__main__":
    main()
