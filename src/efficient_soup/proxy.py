"""Fixed, whole-answer ES protocol with disjoint search/held-out questions."""
import hashlib
import json
import random

import torch

from data.utils import load_hf_dataset, preprocess_chat_instance
from .search import deviation


def extraction_strength(predictions, labels):
    """Fraction after the last wrong token; empty targets are invalid."""
    if predictions.ndim != 1 or labels.ndim != 1 or predictions.shape != labels.shape or not len(labels):
        raise ValueError("ES requires equally sized nonempty 1-D answer sequences")
    wrong = torch.nonzero(predictions != labels, as_tuple=False).flatten()
    prefix = int(wrong[-1]) + 1 if len(wrong) else 0
    return 1 - prefix / len(labels)


def prepare_split(path, name, tokenizer, template, max_length, count, seed):
    raw = load_hf_dataset(path, name=name, split="train")
    groups, encoded = {}, []
    fingerprints = []
    for index, row in enumerate(raw):
        question, answer = row["question"], row["answer"]
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError("Soup proxy requires scalar question/answer strings")
        # Leave enough tokenization room to detect overlength in non-chat
        # templates too, where the repository helper otherwise truncates.
        tokenization_limit = max_length if template["apply_chat_template"] else max_length + 1
        item = preprocess_chat_instance(tokenizer, template, [question], [answer], tokenization_limit)
        # The repository's chat-template path does not truncate. Reject rather
        # than silently changing which answers count or allocating huge logits.
        if len(item["input_ids"]) > max_length:
            raise ValueError(f"{name}[{index}] exceeds max_length; increase it explicitly")
        if (item["labels"] == -200).any() or not (item["labels"][1:] >= 0).any():
            raise ValueError(f"{name}[{index}] is not a valid whole-answer ES example")
        key = hashlib.sha256(question.strip().encode()).hexdigest()
        groups.setdefault(key, []).append(index)
        fingerprint = hashlib.sha256(json.dumps({"question": question, "answer": answer,
            "tokens": {k: v.tolist() for k, v in item.items()}}, sort_keys=True).encode()).hexdigest()
        fingerprints.append(fingerprint)
        encoded.append(item)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 2 or count < 1 or count >= len(keys):
        raise ValueError(f"{name}: proxy count must leave at least one held-out question ({len(keys)} available)")
    proxy_ids = sorted(i for key in keys[:count] for i in groups[key])
    heldout_ids = sorted(i for key in keys[count:] for i in groups[key])
    metadata = {"dataset": path, "config": name, "proxy_indices": proxy_ids,
                "heldout_indices": heldout_ids, "example_sha256": fingerprints,
                "question_sha256": sorted(groups), "grouping": "identical stripped question"}
    return {"proxy": [encoded[i] for i in proxy_ids], "heldout": [encoded[i] for i in heldout_ids],
            "metadata": metadata}


@torch.inference_mode()
def evaluate_es(model, items, batch_size, pad_token_id):
    model.eval()
    device = next(model.parameters()).device
    scores = []
    for offset in range(0, len(items), batch_size):
        batch = items[offset:offset + batch_size]
        ids = torch.nn.utils.rnn.pad_sequence([x["input_ids"] for x in batch], batch_first=True,
                                             padding_value=pad_token_id).to(device)
        labels = torch.nn.utils.rnn.pad_sequence([x["labels"] for x in batch], batch_first=True,
                                                padding_value=-100).to(device)
        # Do not infer mask from token value: PAD may equal EOS.
        mask = torch.nn.utils.rnn.pad_sequence([x["attention_mask"] for x in batch], batch_first=True,
                                              padding_value=0).to(device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        predictions = logits[:, :-1].argmax(-1)
        for prediction, label, attention in zip(predictions, labels[:, 1:], mask[:, 1:]):
            valid = (label >= 0) & attention.bool()
            scores.append(extraction_strength(prediction[valid], label[valid]))
        del logits
    if not scores:
        raise ValueError("Empty ES split")
    return {"mean": sum(scores) / len(scores), "scores": scores, "n": len(scores)}


def evaluate_pair(model, splits, partition, batch_size, pad_token_id):
    results = {key: evaluate_es(model, split[partition], batch_size, pad_token_id)
               for key, split in splits.items()}
    ds = deviation(results["forget"]["mean"], results["retain"]["mean"])
    return {"es_forget": results["forget"]["mean"], "es_retain": results["retain"]["mean"],
            "ds_es": ds, "ds_es_percent": 100 * ds, "per_example": results}
