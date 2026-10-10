#!/usr/bin/env python3
"""Verify both backward paths through the actual PALU.compute_loss."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from trainer.unlearn.palu import PALU


class LogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.detach().clone())

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(
            logits=self.logits.view(1, 1, -1).expand(1, input_ids.shape[1], -1) + 0
        )


def run_probe(logits=None, top_k=2):
    if logits is None:
        logits = torch.tensor([5., 3., 0., -2., -1.], dtype=torch.float64)
    if not 0 < top_k <= logits.numel():
        raise ValueError("top_k must be between 1 and vocabulary size")
    selected = torch.topk(logits, top_k).indices
    outside = torch.ones_like(logits, dtype=torch.bool)
    outside[selected] = False
    batch = {
        "input_ids": torch.tensor([[0, 1]], device=logits.device),
        "attention_mask": torch.ones(1, 2, dtype=torch.long, device=logits.device),
        "labels": torch.tensor([[-100, 1]], device=logits.device),
    }
    result = {"vocab_size": logits.numel(), "top_k": top_k, "variants": {}}
    if logits.numel() <= 32:
        result["logits"] = logits.tolist()
        result["selected_indices"] = selected.tolist()
    for name, detach in (("full_mean", False), ("stop_mean", True)):
        model = LogitModel(logits)
        # Isolate the real loss without constructing a Trainer or a dataset.
        trainer = object.__new__(PALU)
        trainer.ref_model = copy.deepcopy(model)
        trainer.tokenizer = SimpleNamespace(eos_token_id=None)
        trainer.top_k = top_k
        trainer.num_sensitive_tokens = 1
        trainer.target_mode = "mean"
        trainer.detach_mean_target = detach
        trainer.non_sensitive_constraint_type = "kl"
        trainer.gamma, trainer.alpha = 1., 0.
        trainer.compute_retain_loss = lambda **kwargs: logits.new_zeros(())
        loss = trainer.compute_loss(model, {"forget": batch, "retain": batch})
        loss.backward()
        gradient = model.logits.grad
        expected = torch.zeros_like(logits)
        expected[selected] = 2 * (logits[selected] - logits.mean()) / top_k
        if not detach:
            expected -= 2 * (logits[selected].mean() - logits.mean()) / logits.numel()
        torch.testing.assert_close(gradient, expected)
        if any(p.grad is not None for p in trainer.ref_model.parameters()):
            raise RuntimeError("Reference model received a gradient")
        result["variants"][name] = {
            "loss": loss.item(),
            "outside_nonzero_count": torch.count_nonzero(gradient[outside]).item(),
            "outside_gradient_l1": gradient[outside].abs().sum().item(),
            "outside_gradient_l2": gradient[outside].norm().item(),
            "gradient_sum": gradient.sum().item(),
            "formula_max_error": (gradient - expected).abs().max().item(),
        }
        if logits.numel() <= 32:
            result["variants"][name]["gradient"] = gradient.tolist()
    if result["variants"]["full_mean"]["loss"] != result["variants"]["stop_mean"]["loss"]:
        raise RuntimeError("Forward losses differ on identical logits")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocab-size", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    if args.vocab_size == 5:
        logits = torch.tensor([5., 3., 0., -2., -1.], dtype=torch.float64)
    elif args.vocab_size > 1:
        generator = torch.Generator().manual_seed(0)
        logits = torch.randn(args.vocab_size, generator=generator, dtype=torch.float64)
    else:
        parser.error("--vocab-size must exceed 1")
    result = run_probe(logits.to(args.device), args.top_k)
    result["device"] = str(logits.to(args.device).device)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
