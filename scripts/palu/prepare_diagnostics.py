#!/usr/bin/env python3
"""Cache Retain-only pre-head representations once before launching training seeds."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data import get_collators, get_data
from model import get_tokenizer
from trainer.palu_diagnostics import (PROBE_CACHE_SCHEMA, PROBE_POSITIONS, probe_prediction_positions,
                                      sha256_file, tokenizer_fingerprint)


def fixed_samples(dataset, collator, count, seed, eos, *, position):
    """Use independent RNG, and exactly the masks used by the training collator."""
    order = random.Random(seed).sample(range(len(dataset)), len(dataset))
    wanted = len(dataset) if count == -1 else min(count, len(dataset))
    selected, skipped = [], []
    for index in order:
        actual = dataset[index]
        batch = collator([actual])
        sample = {key: batch[key][0].cpu() for key in ("input_ids", "labels", "attention_mask")}
        try:
            prediction = probe_prediction_positions(sample["labels"][None], sample["attention_mask"][None],
                sample["input_ids"][None], eos, position=position)
        except ValueError as error:
            skipped.append({"dataset_index": index, "reason": str(error)})
            continue
        selected.append({**sample, "dataset_index": index, "prediction_index": int(prediction[0])})
        if len(selected) == wanted:
            break
    if not selected:
        raise ValueError("No valid training examples for diagnostics")
    if count != -1 and len(selected) < wanted:
        raise ValueError(f"Requested {wanted} valid diagnostic examples, found {len(selected)}")
    return selected, skipped


def prepare(request):
    started = time.perf_counter()
    print(f"Resolving diagnostic data and Retain reference: {request['retain_model_path']}", flush=True)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name="unlearn", overrides=request["overrides"], return_hydra_config=True)
        HydraConfig.instance().set_config(config)
        config = OmegaConf.create(OmegaConf.to_container(
            OmegaConf.masked_copy(config, [key for key in config if key != "hydra"]), resolve=True))
    tokenizer = get_tokenizer(config.model.tokenizer_args)
    full_path = str(config.model.model_args.pretrained_model_name_or_path)
    full_config = AutoConfig.from_pretrained(full_path, revision=config.model.model_args.get("revision"))
    teacher_path, requested_revision = request["retain_model_path"], request.get("retain_model_revision")
    teacher_config = AutoConfig.from_pretrained(teacher_path, revision=requested_revision)
    keys = ("model_type", "hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads")
    for key in keys:
        if getattr(full_config, key) != getattr(teacher_config, key):
            raise ValueError(f"Full/Retain-only model architecture differs: {key}")
    if teacher_config.model_type != "llama":
        raise ValueError("This final-RMSNorm/pre-head diagnostic currently supports Llama models")
    resolved_revision = getattr(teacher_config, "_commit_hash", None) or requested_revision
    teacher_tokenizer_args = OmegaConf.to_container(config.model.tokenizer_args, resolve=True)
    teacher_tokenizer_args.update(pretrained_model_name_or_path=teacher_path)
    teacher_tokenizer_args.pop("revision", None)
    if resolved_revision:
        teacher_tokenizer_args["revision"] = resolved_revision
    teacher_tokenizer = get_tokenizer(OmegaConf.create(teacher_tokenizer_args))
    if tokenizer_fingerprint(tokenizer) != tokenizer_fingerprint(teacher_tokenizer):
        raise ValueError("Full/Retain-only tokenizer vocabularies or special tokens differ")
    datasets = get_data(config.data, mode=config.mode, tokenizer=tokenizer,
                        template_args=config.model.template_args, model_name=config.model.handler)["train"]
    collator = get_collators(config.collator, tokenizer=tokenizer)
    samples, skipped = {}, {}
    for split, offset in (("forget", 0), ("retain", 1)):
        samples[split], skipped[split] = fixed_samples(getattr(datasets, split), collator,
            request[f"{split}_samples"], request["probe_seed"] + offset, tokenizer.eos_token_id,
            position=PROBE_POSITIONS[split])

    local_hashes = {}
    if Path(teacher_path).is_dir():
        for path in sorted(Path(teacher_path).iterdir()):
            if path.is_file() and (path.suffix in (".safetensors", ".bin") or path.name == "config.json"):
                local_hashes[path.name] = sha256_file(path)
    dtype_name = config.model.model_args.get("torch_dtype", "float32")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name, torch.float32)
    print(f"Preparing Retain reference {teacher_path}; "
          f"Forget={len(samples['forget'])}, Retain={len(samples['retain'])}", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_path, revision=resolved_revision,
        torch_dtype=dtype, attn_implementation=config.model.model_args.get("attn_implementation", "sdpa"),
        device_map="auto" if torch.cuda.is_available() else None)
    teacher.eval().requires_grad_(False)
    # Calling the decoder here avoids materializing full-vocabulary logits. The training
    # diagnostic still uses the wrapped LM forward so ZeRO-3 collectives remain intact.
    input_device = teacher.get_input_embeddings().weight.device
    if input_device.type == "meta":
        raise ValueError("Reference precompute requires a materialized input embedding device")
    with torch.no_grad():
        for split in ("forget", "retain"):
            for index, sample in enumerate(samples[split]):
                outputs = teacher.model(input_ids=sample["input_ids"][None].to(input_device),
                    attention_mask=sample["attention_mask"][None].to(input_device), use_cache=False, return_dict=True)
                sample["reference_hidden"] = outputs.last_hidden_state[0, sample["prediction_index"]].float().cpu().clone()
                if (index + 1) % 8 == 0 or index + 1 == len(samples[split]):
                    print(f"Reference {split}: {index + 1}/{len(samples[split])}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    metadata = {"schema": PROBE_CACHE_SCHEMA, "positions": PROBE_POSITIONS,
        "layer": "final RMSNorm output before lm_head",
        "input_scope": "same teacher-forced training sequence; causal h[t-1] sees only tokens before the target, "
                       "including the ordinary answer prefix before the first sensitive token on Forget",
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "architecture": {key: getattr(teacher.config, key) for key in keys},
        "retain_model_path": teacher_path, "requested_revision": requested_revision,
        "resolved_revision": resolved_revision, "local_checkpoint_sha256": local_hashes,
        "probe_seed": request["probe_seed"], "data_config": OmegaConf.to_container(config.data, resolve=True),
        "template_args": OmegaConf.to_container(config.model.template_args, resolve=True),
        "forward_dtype": str(dtype), "skipped": skipped,
        "sample_indices": {split: [sample["dataset_index"] for sample in items] for split, items in samples.items()},
        "reference_precompute_wall_seconds": time.perf_counter() - started}
    cache_path = Path(request["cache_path"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    torch.save({"metadata": metadata, "samples": samples}, temporary)
    temporary.replace(cache_path)
    metadata["cache_sha256"] = sha256_file(cache_path)
    cache_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Reference cache: {cache_path}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    prepare(json.loads(args.request.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
