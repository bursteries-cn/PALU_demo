"""CPU-only scientific-input, checkpoint-alignment and comparison safeguards."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/palu"))
sys.path.insert(0, str(ROOT / "src"))
import layout_loss_curve_experiment as experiment


class ReplayTests(unittest.TestCase):
    def test_actual_consumed_pairs_and_budget(self):
        config, pairs, history, provenance = experiment.validate_assets()
        self.assertEqual(config["trainer"]["method_args"]["alpha"], 2)
        self.assertEqual(len(pairs), 1920)
        self.assertEqual(len({row["retain_index"] for row in pairs}), 1502)
        records = experiment.replay_records(pairs)
        self.assertEqual(records[0]["forget_index"], 95)
        self.assertEqual(records[0]["retain_index"], 2251)
        self.assertEqual(records[-1]["forget_index"], 82)
        self.assertEqual(records[-1]["retain_index"], 341)
        self.assertEqual(records[-1]["draw_index"], 1920)

    def test_epoch_boundary_model_states_match_original_hf(self):
        _, _, history, _ = experiment.validate_assets()
        events = experiment.replay_events(history)
        self.assertEqual([row["step"] for row in events], [6, 12, 18, 25, 31, 37, 43, 50, 56, 60])
        self.assertEqual(events[0]["source_epoch_index"], 1)
        self.assertEqual(events[0]["updated_exposure"], 192)
        self.assertEqual(events[0]["overshoot_examples"], -8)
        self.assertEqual(events[-1]["sample_epoch"], 9.6)
        self.assertTrue(events[-1]["final_partial_epoch"])

    def test_three_distinct_treatments_and_equal_effective_batch(self):
        self.assertEqual([row["normalization"] for row in experiment.ARMS], ["native", "global", "native"])
        self.assertEqual([row["padding_scope"] for row in experiment.ARMS], ["microbatch", "update", "microbatch"])
        self.assertTrue(all(row["microbatch"] * row["accumulation"] * row["world_size"] == 32 for row in experiment.ARMS))


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.study = Path(self.temporary.name)
        (self.study / "frozen").mkdir()
        (self.study / "frozen/schedule.pt").write_text("CPU test fixture, not a Torch schedule")
        config, _, history, provenance = experiment.validate_assets()
        description = {"method": config["trainer"]["method_args"], "evaluation_steps": [row["step"] for row in history],
                       "source_provenance": provenance}
        experiment.write_json(self.study / "experiment.json", description)
        (self.study / "historical_seed103").mkdir()
        self.write_rows(self.study / "historical_seed103/epoch_history.jsonl", history)
        for arm in experiment.ARMS:
            folder = self.study / arm["name"]
            folder.mkdir()
            manifest = {"status": "completed", "seed": 103,
                        "schedule_sha256": experiment.sha(self.study / "frozen/schedule.pt"),
                        "loss_mode": arm["normalization"], "padding_scope": arm["padding_scope"],
                        "method": description["method"], "completed_updates": 60,
                        "initial_parameter_sha256": "test-initial", "teacher_parameter_sha256": "test-initial",
                        "schedule_metadata": {"pair_stream_sha256": "test-pairs"}}
            experiment.write_json(folder / "manifest.json", manifest)
            self.write_rows(folder / "epoch_history.jsonl", history)
            self.write_rows(folder / "loss_history.jsonl", [{"step": i, "epoch": i * 32 / 200,
                **{key: history[-1][key] for key in experiment.MEASURES[:4]}} for i in range(1, 61)])

    def write_rows(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_complete_comparison_and_historical_bridge(self):
        experiment.analyze(SimpleNamespace(study=self.study, no_plots=True))
        summary = experiment.read_json(self.study / "analysis/summary.json")
        self.assertEqual(summary["states_per_arm"]["old_single"], 11)
        self.assertTrue(summary["single_seed_only"])
        self.assertTrue((self.study / "analysis/historical_seed103_bridge.csv").is_file())

    def test_reference_change_rejected(self):
        path = self.study / "four_native/epoch_history.jsonl"
        rows = experiment.read_rows(path)
        rows[-1]["reference_sha256"] = "different-reference"
        self.write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "FQ reference"):
            experiment.collect_results(self.study)

    def test_initial_model_change_rejected(self):
        path = self.study / "new_single/manifest.json"
        value = experiment.read_json(path)
        value["initial_parameter_sha256"] = "different-start"
        experiment.write_json(path, value)
        with self.assertRaisesRegex(ValueError, "identical parameters"):
            experiment.collect_results(self.study)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is installed only in the server PALU environment")
class TensorReplayTests(unittest.TestCase):
    def test_three_objectives_gradients_and_nonzero_common_kl(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from gradient_equivalence_step import (check_original_loss, raw_components, slice_inputs,
                                               target_counts, weighted_loss)
        torch.set_num_threads(1)
        torch.manual_seed(53)
        config = LlamaConfig(vocab_size=29, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            attention_dropout=0.0, pad_token_id=0, eos_token_id=2, use_cache=False)
        config._attn_implementation = "eager"
        reference = LlamaForCausalLM(config).double().eval()
        reference.requires_grad_(False)
        student = copy.deepcopy(reference).train().requires_grad_(True)
        with torch.no_grad():
            student.lm_head.weight.add_(torch.randn_like(student.lm_head.weight) * 0.02)
        ids = torch.randint(3, 29, (32, 9))
        f_labels = torch.full_like(ids, -100)
        r_labels = torch.full_like(ids, -100)
        for i in range(32):
            count = i % 4
            f_labels[i, 2:2 + count] = ids[i, 2:2 + count]
            f_labels[i, 2 + count:] = -200
            r_labels[i, 1:2 + i % 7] = ids[i, 1:2 + i % 7]
        inputs = {"forget": {"input_ids": ids, "labels": f_labels, "attention_mask": torch.ones_like(ids)},
                  "retain": {"input_ids": ids.flip(0), "labels": r_labels, "attention_mask": torch.ones_like(ids)}}
        counts = target_counts(inputs, 2, 3, torch)
        method = {"top_k": 7, "first_n": 3, "alpha": 2.0, "gamma": 1.0}
        outputs = {}
        for name, world, accumulation, normalization in (("full", 1, 1, "global"),
                ("old_single", 1, 4, "native"), ("new_single", 1, 4, "global"),
                ("four_native", 4, 8, "native"), ("four_global", 4, 8, "global")):
            model = copy.deepcopy(student)
            value, expected = 0.0, 0.0
            common_seen = False
            for micro in range(accumulation):
                for rank in range(world):
                    indices = list(range(32)) if name == "full" else [4 * micro + rank] if world == 4 else list(range(8 * micro, 8 * micro + 8))
                    batch = slice_inputs(inputs, indices, "cpu")
                    terms = raw_components(model, reference, batch, top_k=7, first_n=3, eos=2, torch=torch)
                    common_seen |= float(terms["common"].detach()) > 0
                    if micro == rank == 0:
                        check_original_loss(model, reference, batch, terms, SimpleNamespace(eos_token_id=2), method, torch)
                    local = target_counts(batch, 2, 3, torch)
                    loss = weighted_loss(terms, local, counts, gamma=1.0, alpha=2.0,
                                         normalization=normalization, accumulation=accumulation, world=world)
                    (loss / accumulation / world).backward()
                    value += float(loss.detach()) / accumulation / world
                    if normalization == "native":
                        expected += sum(weight * float(terms[key].detach()) for key, weight in
                            (("sensitive", 1), ("retain", 2), ("common", 2))) / accumulation / world
            self.assertTrue(common_seen)
            if normalization == "native":
                self.assertAlmostEqual(value, expected, places=6)
            outputs[name] = (value, torch.cat([p.grad.reshape(-1) for p in model.parameters()]))
        self.assertAlmostEqual(outputs["full"][0], outputs["new_single"][0], places=5)
        torch.testing.assert_close(outputs["full"][1], outputs["new_single"][1], rtol=1e-5, atol=1e-7)
        self.assertAlmostEqual(outputs["full"][0], outputs["four_global"][0], places=5)
        torch.testing.assert_close(outputs["full"][1], outputs["four_global"][1], rtol=1e-5, atol=1e-7)
        self.assertGreater(torch.linalg.vector_norm(outputs["old_single"][1] - outputs["four_native"][1]).item(), 1e-6)

    def test_original_microbatch_padding_and_global_schedule(self):
        import torch
        sys.path.insert(0, str(ROOT / "src"))
        from data.collators import DataCollatorForSupervisedDataset
        from prepare_layout_schedule import build_schedule
        from layout_token_training import microbatch_inputs
        from gradient_equivalence_step import target_counts
        collator = DataCollatorForSupervisedDataset(SimpleNamespace(pad_token_id=99), padding_side="right")
        samples = {
            "forget": [{"input_ids": torch.tensor([10, 11, 99]), "labels": torch.tensor([-100, 11, 99])},
                       {"input_ids": torch.tensor([10, 12, 13, 14, 99]), "labels": torch.tensor([-100, 12, -200, 14, 99])}],
            "retain": [{"input_ids": torch.tensor([10, 21, 99]), "labels": torch.tensor([-100, 21, 99])},
                       {"input_ids": torch.tensor([10, 22, 23, 99]), "labels": torch.tensor([-100, 22, 23, 99])}]}
        records = [{"draw_index": i + 1, "forget_index": (i // 8) % 2, "retain_index": i % 2} for i in range(32)]
        rng_before = torch.random.get_rng_state().clone()
        schedule = build_schedule(samples, collator, eos=99, first_n=3, updates=1, effective_batch=32,
                                  seed=103, torch=torch, records=records)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        window = schedule["windows"][0]
        self.assertEqual([r["forget_index"] for r in window["pair_records"]], [r["forget_index"] for r in records])
        for size in (1, 8):
            total = {key: 0 for key in window["counts"]}
            for start in range(0, 32, size):
                indices = list(range(start, start + size))
                actual = microbatch_inputs(window, indices, "cpu", "microbatch")
                expected = collator([{split: samples[split][records[i][split + "_index"]] for split in samples} for i in indices])
                for split in samples:
                    for key in ("input_ids", "labels", "attention_mask"):
                        self.assertTrue(torch.equal(actual[split][key], expected[split][key]))
                for key, value in target_counts(actual, 99, 3, torch).items():
                    total[key] += value
            self.assertEqual(total, window["counts"])
        old = microbatch_inputs(window, list(range(8)), "cpu", "microbatch")
        new = microbatch_inputs(window, list(range(8)), "cpu", "update")
        self.assertEqual(old["forget"]["input_ids"].shape[1], 3)
        self.assertEqual(new["forget"]["input_ids"].shape[1], 5)


if __name__ == "__main__":
    unittest.main()
