#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from representation_batching.manifest import qa_content_sha256  # noqa: E402
from data.utils import load_hf_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cached LLaMA question representations and initial answer NLL for TOFU."
    )
    parser.add_argument(
        "--model",
        default="open-unlearning/tofu_Llama-3.1-8B-Instruct_full",
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset", default="locuslab/TOFU")
    parser.add_argument("--dataset-config", default="forget05")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--date-string", default="10 Apr 2025")
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Fail above this length; do not truncate away training tokens.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--block",
        default="two_thirds",
        help="One-indexed decoder block or 'two_thirds'.",
    )
    parser.add_argument("--torch-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map; use 'none' to call model.to(device).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _template_kwargs(date_string: str | None) -> dict:
    return {"date_string": date_string} if date_string else {}


def _render_and_tokenize(tokenizer, question: str, answer: str, args: argparse.Namespace) -> dict:
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": question})
    prompt_messages = list(messages)
    messages.append({"role": "assistant", "content": answer})
    template_kwargs = _template_kwargs(args.date_string)

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        **template_kwargs,
    )
    chat_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        **template_kwargs,
    )
    rendered_prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    offset_encoding = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offset_ids = list(offset_encoding["input_ids"])
    offsets = list(offset_encoding["offset_mapping"])
    if offset_ids != list(chat_ids):
        raise ValueError(
            "Rendered-template tokenization does not match apply_chat_template; "
            "cannot identify a trustworthy question-token span."
        )
    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids = list(chat_ids) + [tokenizer.eos_token_id]
        offsets.append((0, 0))
    else:
        chat_ids = list(chat_ids)
    if args.max_length > 0 and len(chat_ids) > args.max_length:
        raise ValueError(
            f"Full QA sequence has {len(chat_ids)} tokens, above --max-length "
            f"{args.max_length}. Increase the guard or shorten the input; truncation would "
            "disagree with the repository's chat-template training path."
        )

    if not rendered.startswith(rendered_prompt):
        raise ValueError("Rendered generation prompt is not a character prefix of full QA")
    question_start = rendered_prompt.rfind(question)
    if question_start < 0:
        raise ValueError("Question text was not found in the rendered chat template")
    question_end = question_start + len(question)
    question_positions = [
        index
        for index, (start, stop) in enumerate(offsets)
        if start < question_end and stop > question_start
    ]
    if not question_positions:
        raise ValueError("Question has no tokens in the rendered training sequence")
    prompt_length = len(prompt_ids)
    if prompt_length >= len(chat_ids):
        raise ValueError("No answer token remains after tokenization")
    if list(prompt_ids) != chat_ids[:prompt_length]:
        raise ValueError(
            "The generation prompt is not an exact prefix of the full QA tokenization; "
            "this would disagree with the repository's training mask."
        )
    return {
        "input_ids": chat_ids,
        "attention_mask": [1] * len(chat_ids),
        "question_positions": question_positions,
        "assistant_prefix_last": prompt_length - 1,
        "answer_start": prompt_length,
    }


def _decoder_and_layers(model):
    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if decoder is None or layers is None:
        raise TypeError("This extractor currently supports LLaMA-style model.model.layers only")
    return decoder, layers


def _block_number(value: str, num_layers: int) -> int:
    if value == "two_thirds":
        return int(math.ceil(2 * num_layers / 3))
    block = int(value)
    if block < 1 or block > num_layers:
        raise ValueError(f"block must be in [1, {num_layers}], got {block}")
    return block


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def _source_metadata(row: dict) -> dict:
    keys = ("author", "name", "fact_id", "id")
    return {key: row[key] for key in keys if key in row}


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise TypeError("A fast tokenizer is required for offset mappings")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "revision": args.model_revision,
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.device_map == "none":
        model.to(args.device)
    model.eval()

    dataset = load_hf_dataset(
        path=args.dataset,
        name=args.dataset_config,
        split=args.dataset_split,
        revision=args.dataset_revision,
    )
    decoder, layers = _decoder_and_layers(model)
    block_number = _block_number(args.block, len(layers))
    captured: Dict[str, torch.Tensor] = {}

    def capture_block(_module, _inputs, output):
        captured["block"] = output[0] if isinstance(output, tuple) else output

    hook = layers[block_number - 1].register_forward_hook(capture_block)
    arrays: Dict[str, List[np.ndarray]] = {
        f"block_{block_number}_question_last": [],
        f"block_{block_number}_assistant_prefix_last": [],
        f"block_{block_number}_question_mean": [],
        "final_question_last": [],
        "final_assistant_prefix_last": [],
        "final_question_mean": [],
    }
    sample_ids: List[int] = []
    initial_nll: List[float] = []
    records: List[dict] = []

    try:
        for start in range(0, len(dataset), args.batch_size):
            rows = [dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))]
            prepared = []
            for local_offset, row in enumerate(rows):
                answer = row[args.answer_key]
                if not isinstance(answer, str):
                    raise TypeError("The extractor expects one string answer per TOFU row")
                prepared.append(
                    _render_and_tokenize(
                        tokenizer,
                        str(row[args.question_key]),
                        answer,
                        args,
                    )
                )
                sample_id = start + local_offset
                sample_ids.append(sample_id)
                records.append(
                    {
                        "sample_id": sample_id,
                        "question": str(row[args.question_key]),
                        "question_token_positions": prepared[-1]["question_positions"],
                        "assistant_prefix_last": prepared[-1]["assistant_prefix_last"],
                        "question_length": len(prepared[-1]["question_positions"]),
                        "answer_length": len(prepared[-1]["input_ids"]) - prepared[-1]["answer_start"],
                        "source_metadata": _source_metadata(dict(row)),
                    }
                )

            padded = tokenizer.pad(
                {
                    "input_ids": [item["input_ids"] for item in prepared],
                    "attention_mask": [item["attention_mask"] for item in prepared],
                },
                padding=True,
                return_tensors="pt",
            )
            device = _input_device(model)
            inputs = {key: value.to(device) for key, value in padded.items()}
            captured.clear()
            with torch.inference_mode():
                decoder_output = decoder(**inputs, use_cache=False, return_dict=True)
            if "block" not in captured:
                raise RuntimeError("Decoder block hook did not run")
            block_hidden = captured["block"]
            final_hidden = decoder_output.last_hidden_state

            for batch_index, item in enumerate(prepared):
                question_positions = torch.tensor(item["question_positions"], device=final_hidden.device)
                question_last = int(item["question_positions"][-1])
                prefix_last = int(item["assistant_prefix_last"])
                block_sample = block_hidden[batch_index]
                final_sample = final_hidden[batch_index]
                arrays[f"block_{block_number}_question_last"].append(
                    block_sample[question_last].float().cpu().numpy()
                )
                arrays[f"block_{block_number}_assistant_prefix_last"].append(
                    block_sample[prefix_last].float().cpu().numpy()
                )
                arrays[f"block_{block_number}_question_mean"].append(
                    block_sample.index_select(0, question_positions.to(block_sample.device))
                    .mean(dim=0)
                    .float()
                    .cpu()
                    .numpy()
                )
                arrays["final_question_last"].append(
                    final_sample[question_last].float().cpu().numpy()
                )
                arrays["final_assistant_prefix_last"].append(
                    final_sample[prefix_last].float().cpu().numpy()
                )
                arrays["final_question_mean"].append(
                    final_sample.index_select(0, question_positions)
                    .mean(dim=0)
                    .float()
                    .cpu()
                    .numpy()
                )

                answer_start = int(item["answer_start"])
                sequence_length = len(item["input_ids"])
                predictor_hidden = final_sample[answer_start - 1 : sequence_length - 1]
                output_embeddings = model.get_output_embeddings()
                output_device = next(output_embeddings.parameters()).device
                predictor_hidden = predictor_hidden.to(output_device)
                targets = torch.tensor(
                    item["input_ids"][answer_start:sequence_length],
                    device=output_device,
                    dtype=torch.long,
                )
                with torch.inference_mode():
                    logits = output_embeddings(predictor_hidden)
                    nll = F.cross_entropy(logits.float(), targets, reduction="sum")
                initial_nll.append(float(nll.cpu()))
                records[len(records) - len(prepared) + batch_index][
                    "initial_answer_nll"
                ] = float(nll.cpu())
    finally:
        hook.remove()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "features.npz"
    np.savez_compressed(
        feature_path,
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        initial_answer_nll=np.asarray(initial_nll, dtype=np.float32),
        **{key: np.stack(value).astype(np.float32) for key, value in arrays.items()},
    )
    feature_sha256 = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "model": args.model,
        "model_revision": args.model_revision,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "resolved_tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "dataset_revision": args.dataset_revision,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "qa_content_sha256": qa_content_sha256(
            dataset, args.question_key, args.answer_key
        ),
        "question_key": args.question_key,
        "answer_key": args.answer_key,
        "tokenizer_class": tokenizer.__class__.__name__,
        "num_decoder_layers": len(layers),
        "selected_block_one_indexed": block_number,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "system_prompt": args.system_prompt,
        "date_string": args.date_string,
        "max_length": args.max_length,
        "feature_file": str(feature_path.resolve()),
        "feature_sha256": feature_sha256,
        "feature_keys": sorted(arrays),
        "records": records,
    }
    (args.output_dir / "feature_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {feature_path}")
    print(f"wrote {args.output_dir / 'feature_manifest.json'}")


if __name__ == "__main__":
    main()
