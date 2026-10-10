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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/palu_mean_ablation")]
from trainer.unlearn.palu import PALU
from probe_gradients import run_probe


class MeanGradientTests(unittest.TestCase):
    def test_actual_loss_has_equal_values_but_different_tail_gradients(self):
        result = run_probe()
        full, stop = (result["variants"][name] for name in ("full_mean", "stop_mean"))
        self.assertEqual(full["loss"], stop["loss"])
        torch.testing.assert_close(torch.tensor(full["gradient"]), torch.tensor([2.8, .8, -1.2, -1.2, -1.2]))
        torch.testing.assert_close(torch.tensor(stop["gradient"]), torch.tensor([4., 2., 0., 0., 0.]))
        self.assertEqual(full["outside_nonzero_count"], 3)
        self.assertEqual(stop["outside_nonzero_count"], 0)

    def test_formula_on_llama_sized_vocabulary(self):
        logits = torch.randn(128256, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        result = run_probe(logits, 5000)
        self.assertEqual(result["variants"]["full_mean"]["outside_nonzero_count"], 123256)
        self.assertEqual(result["variants"]["stop_mean"]["outside_nonzero_count"], 0)
        self.assertLess(result["variants"]["full_mean"]["formula_max_error"], 1e-12)

    def test_all_vocabulary_selection_has_identical_gradients(self):
        result = run_probe(top_k=5)
        self.assertEqual(result["variants"]["full_mean"]["gradient"], result["variants"]["stop_mean"]["gradient"])

    def test_zero_tail_gradient_is_possible_without_detach(self):
        # The reference still selects dimensions 0/1, but the current selected
        # mean equals the global mean. This checks the qualification "usually".
        model, trainer, batch = self.position_fixture(False)
        with torch.no_grad():
            model.logits[:] = torch.tensor([1., -1., 2., -2., 0.])
        trainer.compute_loss(model, batch).backward()
        self.assertTrue(torch.equal(model.logits.grad[..., 2:], torch.zeros_like(model.logits.grad[..., 2:])))
        self.assertGreater(model.logits.grad[..., :2].abs().sum().item(), 0)

    @staticmethod
    def position_fixture(detach):
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
        trainer.target_mode, trainer.detach_mean_target = "mean", detach
        trainer.non_sensitive_constraint_type = "kl"
        trainer.gamma, trainer.alpha = 1., 0.
        trainer.compute_retain_loss = lambda **kwargs: model.logits.new_zeros(())
        sample = {"input_ids": torch.zeros(1, 7, dtype=torch.long), "attention_mask": torch.ones(1, 7, dtype=torch.long),
                  "labels": torch.tensor([[-100, 1, 4, -200, 3, 2, -100]])}
        return model, trainer, {"forget": sample, "retain": sample}

    def test_prefix_mask_eos_and_reference_selection_are_preserved(self):
        for detach in (False, True):
            with self.subTest(detach=detach):
                model, trainer, batch = self.position_fixture(detach)
                # Current top-K changes; selected dimensions must still be 0/1.
                with torch.no_grad():
                    model.logits[..., 4] = 10.
                trainer.compute_loss(model, batch).backward()
                for position in (1, 2, 4, 5, 6):
                    self.assertEqual(torch.count_nonzero(model.logits.grad[0, position]).item(), 0)
                for position in (0, 3):
                    expected = torch.tensor([5., 3., 0., -2., 10.])
                    gradient = torch.zeros(5)
                    gradient[:2] = (expected[:2] - expected.mean()) / 2  # K=2, two sensitive positions
                    if not detach:
                        gradient -= (expected[:2].mean() - expected.mean()) / 5
                    torch.testing.assert_close(model.logits.grad[0, position], gradient)
                self.assertTrue(all(parameter.grad is None for parameter in trainer.ref_model.parameters()))

    def test_no_sensitive_positions_have_zero_forget_gradient(self):
        for detach in (False, True):
            model, trainer, batch = self.position_fixture(detach)
            batch["forget"]["labels"].fill_(-100)
            loss = trainer.compute_loss(model, batch)
            loss.backward()
            self.assertEqual(loss.item(), 0.)
            self.assertEqual(torch.count_nonzero(model.logits.grad).item(), 0)


class TinyTrainingTests(unittest.TestCase):
    def test_both_modes_train_from_same_weights_and_keep_reference_unchanged(self):
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
        initial_losses, final_states = [], []
        with tempfile.TemporaryDirectory() as temp:
            for detach in (False, True):
                model = copy.deepcopy(initial)
                args = TrainingArguments(output_dir=str(Path(temp) / str(detach)), use_cpu=True,
                    per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=3,
                    learning_rate=1e-3, optim="adamw_torch", report_to=[], save_strategy="no",
                    eval_strategy="no", logging_strategy="no", remove_unused_columns=False,
                    disable_tqdm=True, seed=7)
                trainer = PALU(model=model, args=args, train_dataset=[batch] * 3,
                    data_collator=lambda samples: samples[0], tokenizer=tokenizer,
                    top_k=3, first_n=1, alpha=.2, detach_mean_target=detach)
                initial_losses.append(trainer.compute_loss(model, batch).item())
                reference = {key: value.detach().clone() for key, value in trainer.ref_model.state_dict().items()}
                trainer.train()
                self.assertEqual(trainer.state.global_step, 3)
                self.assertTrue(all(torch.equal(value, trainer.ref_model.state_dict()[key]) for key, value in reference.items()))
                self.assertTrue(all(parameter.grad is None for parameter in trainer.ref_model.parameters()))
                self.assertTrue(any(not torch.equal(value, model.state_dict()[key]) for key, value in initial.state_dict().items()))
                final_states.append(copy.deepcopy(model.state_dict()))
        self.assertEqual(initial_losses[0], initial_losses[1])
        self.assertTrue(any(not torch.equal(value, final_states[1][key]) for key, value in final_states[0].items()))


if __name__ == "__main__":
    unittest.main()
