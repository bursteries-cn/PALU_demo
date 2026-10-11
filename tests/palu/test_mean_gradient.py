import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, TrainingArguments

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/palu")]
from trainer.unlearn.palu import PALU


class MeanGradientTests(unittest.TestCase):
    def test_official_mean_gradient_matches_formula_and_sums_to_zero(self):
        model, trainer, batch = self.position_fixture()
        loss = trainer.compute_loss(model, batch)
        loss.backward()
        gradient = model.logits.grad.sum(dim=(0, 1))
        self.assertEqual(loss.item(), 10.)
        torch.testing.assert_close(gradient, torch.tensor([2.8, .8, -1.2, -1.2, -1.2]))
        self.assertAlmostEqual(gradient.sum().item(), 0., places=6)

    def test_formula_on_llama_sized_vocabulary(self):
        model, trainer, batch = self.position_fixture()
        values = torch.randn(128256, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        model.logits = torch.nn.Parameter(values.repeat(1, 7, 1))
        trainer.ref_model = copy.deepcopy(model)
        trainer.top_k = 5000
        trainer.compute_loss(model, batch).backward()
        selected = values.topk(trainer.top_k).indices
        residual = values[selected] - values.mean()
        expected = torch.full_like(values, -2 * residual.mean() / values.numel())
        expected[selected] += 2 * residual / trainer.top_k
        torch.testing.assert_close(model.logits.grad.sum(dim=(0, 1)), expected, atol=1e-12, rtol=1e-12)

    def test_zero_tail_gradient_when_selected_mean_equals_global_mean(self):
        # The reference still selects dimensions 0/1, but the current selected
        # mean equals the global mean. This checks the qualification "usually".
        model, trainer, batch = self.position_fixture()
        with torch.no_grad():
            model.logits[:] = torch.tensor([1., -1., 2., -2., 0.])
        trainer.compute_loss(model, batch).backward()
        self.assertTrue(torch.equal(model.logits.grad[..., 2:], torch.zeros_like(model.logits.grad[..., 2:])))
        self.assertGreater(model.logits.grad[..., :2].abs().sum().item(), 0)

    @staticmethod
    def position_fixture():
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([5., 3., 0., -2., -1.]).repeat(1, 7, 1))

            def forward(self, **kwargs):
                # A real LM computes outputs inside no_grad; returning a bare
                # Parameter would bypass that context in the reference fixture.
                return SimpleNamespace(logits=self.logits + 0)
        model = Model()
        trainer = object.__new__(PALU)
        trainer.ref_model = copy.deepcopy(model)
        trainer.tokenizer = SimpleNamespace(eos_token_id=2)
        trainer.top_k, trainer.num_sensitive_tokens = 2, 1
        trainer.target_mode = "mean"
        trainer.non_sensitive_constraint_type = "kl"
        trainer.gamma, trainer.alpha = 1., 0.
        trainer.compute_retain_loss = lambda **kwargs: model.logits.new_zeros(())
        sample = {"input_ids": torch.zeros(1, 7, dtype=torch.long), "attention_mask": torch.ones(1, 7, dtype=torch.long),
                  "labels": torch.tensor([[-100, 1, 4, -200, 3, 2, -100]])}
        return model, trainer, {"forget": sample, "retain": sample}

    def test_prefix_mask_eos_and_reference_selection_are_preserved(self):
        model, trainer, batch = self.position_fixture()
        # Current top-K changes; selected dimensions must still be 0/1.
        with torch.no_grad():
            model.logits[..., 4] = 10.
        trainer.compute_loss(model, batch).backward()
        for position in (1, 2, 4, 5, 6):
            self.assertEqual(torch.count_nonzero(model.logits.grad[0, position]).item(), 0)
        for position in (0, 3):
            expected = torch.tensor([5., 3., 0., -2., 10.])
            gradient = torch.zeros(5)
            gradient[:2] = (expected[:2] - expected.mean()) / 2
            gradient -= (expected[:2].mean() - expected.mean()) / 5
            torch.testing.assert_close(model.logits.grad[0, position], gradient)
        self.assertTrue(all(parameter.grad is None for parameter in trainer.ref_model.parameters()))

    def test_no_sensitive_positions_have_zero_forget_gradient(self):
        model, trainer, batch = self.position_fixture()
        batch["forget"]["labels"].fill_(-100)
        loss = trainer.compute_loss(model, batch)
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(torch.count_nonzero(model.logits.grad).item(), 0)


class TinyTrainingTests(unittest.TestCase):
    def test_training_updates_model_and_keeps_reference_unchanged(self):
        torch.manual_seed(24)
        initial = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32, pad_token_id=0, eos_token_id=2))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}, unk_token="[UNK]")),
            pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
        forget = {"input_ids": torch.tensor([[1, 3, 4, 5, 2]]), "attention_mask": torch.ones(1, 5, dtype=torch.long),
                  "labels": torch.tensor([[-100, 3, 4, -200, 2]])}
        retain = {**forget, "labels": torch.tensor([[-100, 3, 4, 5, 2]])}
        batch = {"forget": forget, "retain": retain}
        with tempfile.TemporaryDirectory() as temp:
            model = copy.deepcopy(initial)
            args = TrainingArguments(output_dir=temp, use_cpu=True,
                per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=3,
                learning_rate=1e-3, optim="adamw_torch", report_to=[], save_strategy="no",
                eval_strategy="no", logging_strategy="no", remove_unused_columns=False,
                disable_tqdm=True, seed=7)
            trainer = PALU(model=model, args=args, train_dataset=[batch] * 3,
                data_collator=lambda samples: samples[0], tokenizer=tokenizer,
                top_k=3, first_n=1, alpha=.2)
            reference = {key: value.detach().clone() for key, value in trainer.ref_model.state_dict().items()}
            trainer.train()
            self.assertEqual(trainer.state.global_step, 3)
            self.assertTrue(all(torch.equal(value, trainer.ref_model.state_dict()[key]) for key, value in reference.items()))
            self.assertTrue(all(parameter.grad is None for parameter in trainer.ref_model.parameters()))
            self.assertTrue(any(not torch.equal(value, model.state_dict()[key]) for key, value in initial.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
