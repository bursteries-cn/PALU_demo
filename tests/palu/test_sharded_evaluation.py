import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/palu"), str(ROOT / "src"), str(Path(__file__).parent)]
from online_epoch_evaluation import epoch_events
from sharded_epoch_evaluation import shard_slots
from benchmark_sharded_evaluation import compare_values


class PartitionTests(unittest.TestCase):
    def test_checkpoint_setup_rejects_single_rank_teacher_copy(self):
        from layout_token_training import setup
        with self.assertRaisesRegex(ValueError, "original Full teacher"):
            setup({}, "single", 103, None, student_checkpoint="saved-student")

    def test_uneven_and_small_shards_cover_once_with_equal_work(self):
        for size in (1, 2, 3, 4, 5, 7, 11, 200, 3800):
            assignments = [shard_slots(size, rank, 4) for rank in range(4)]
            self.assertEqual(len({len(items) for items in assignments}), 1)
            self.assertEqual(sorted(i for items in assignments for i, valid in items if valid), list(range(size)))
            self.assertTrue(all(0 <= i < size for items in assignments for i, _ in items))

    def test_every_two_epochs_preserves_initial_final_and_pending_gradients(self):
        events = epoch_events(every_epochs=2)
        self.assertEqual([v["step"] for v in events], [0, 12, 25, 37, 50, 60])
        self.assertEqual([v["consumed_pairs"] for v in events], [0, 400, 800, 1200, 1600, 1920])
        self.assertEqual([v["pending_pairs"] for v in events], [0, 16, 0, 16, 0, 0])
        self.assertEqual(epoch_events(total_pairs=800, every_epochs=2)[-1]["step"], 25)
        self.assertEqual(len(epoch_events(every_epochs=20)), 2)
        for invalid in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                epoch_events(every_epochs=invalid)

    def test_comparison_rejects_missing_samples_different_text_and_nonfinite(self):
        self.assertFalse(compare_values({"a": 1}, {"b": 1})["passed"])
        self.assertFalse(compare_values([1, 2], [1])["passed"])
        self.assertFalse(compare_values({"generation": "a"}, {"generation": "b"})["passed"])
        self.assertFalse(compare_values(float("nan"), float("nan"))["passed"])
        self.assertTrue(compare_values([1.0], [1.0])["passed"])


def distributed_worker(rank, init_file, output):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from types import SimpleNamespace
    from test_online_epoch_evaluation import tiny_model, tiny_runtime
    from online_epoch_evaluation import runtime_fingerprint
    from sharded_epoch_evaluation import ShardedEvaluation
    from evals.metrics.utils import run_batchwise_evals
    from trainer.palu_diagnostics import preserve_rng_and_modes
    from transformers.cache_utils import DynamicCache
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=4)
    torch.manual_seed(103)
    model = tiny_model(torch)
    runtime = tiny_runtime(torch, model)
    ids = torch.tensor([[3, 4, 5]])
    model(input_ids=ids, labels=ids).loss.backward()
    runtime["optimizer"].step()
    runtime["optimizer"].zero_grad(set_to_none=True)
    model(input_ids=ids, labels=ids).loss.backward()  # pending accumulation
    data = [{"index": i, "ids": [3 + i] * (i % 4 + 1)} for i in range(13)]

    def collate(items):
        width = max(len(item["ids"]) for item in items)
        ids = torch.tensor([[0] * (width - len(item["ids"])) + item["ids"] for item in items])
        return {"a": {"index": torch.tensor([item["index"] for item in items]),
                      "input_ids": ids, "attention_mask": (ids != 0).long()},
                "b": {"index": torch.tensor([item["index"] for item in items]),
                      "input_ids": ids + 1, "attention_mask": (ids != 0).long()}}

    loader = DataLoader(data, batch_size=3, collate_fn=collate)
    def evaluate(model, batch):
        # Different whole batches terminate after different numbers of forwards.
        length = 1 + int(batch["input_ids"][0, -1]) % 3
        generated = model.generate(**batch, max_new_tokens=length, do_sample=False,
                                   eos_token_id=None, use_cache=True, past_key_values=DynamicCache())
        with torch.no_grad():
            logits = model(**batch, use_cache=False).logits
        return [{"score": float(logits[i, -1].sum()), "generation": generated[i].tolist(),
                 "batch_width": batch["input_ids"].shape[1], "batch_size": len(generated)}
                for i in range(len(generated))]

    with preserve_rng_and_modes([model], torch.device("cpu")):
        expected = run_batchwise_evals(model, loader, evaluate, {}, "baseline") if rank == 0 else None
    dist.barrier()
    probe = SimpleNamespace(_fixed_losses=lambda *_: None)
    before = runtime_fingerprint(runtime, torch)
    driver = ShardedEvaluation(torch)
    with preserve_rng_and_modes([model], torch.device("cpu")), driver.activate(probe, {"student": model}):
        actual = driver.run_batches(model, loader, evaluate, {}, "sample-parallel")
    assert runtime_fingerprint(runtime, torch) == before
    assert "_fixed_losses" in probe.__dict__  # restore instance override
    # Exercise the unchanged PALU formula, ordered aggregation, zero targets,
    # and padding when fewer fixed examples than ranks are present.
    from online_epoch_evaluation import build_probe
    f = [{"input_ids": torch.tensor([3, 4, 5, 6]), "labels": torch.tensor([-100, 4, -200, 6])},
         {"input_ids": torch.tensor([3, 7, 8]), "labels": torch.tensor([-100, -200, -200])}]
    r = [{"input_ids": torch.tensor([3, 9, 10]), "labels": torch.tensor([-100, 9, 10])},
         {"input_ids": torch.tensor([3, 11, 12, 13]), "labels": torch.tensor([-100, 11, 12, 13])},
         {"input_ids": torch.tensor([3, 14]), "labels": torch.tensor([-100, 14])}]
    runtime["tokenizer"] = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    config = json.loads((ROOT / "configs/analysis/layout_loss_seed103/resolved_config.json").read_text())
    config["trainer"]["method_args"].update(top_k=7, first_n=3)
    before_fixed = runtime_fingerprint(runtime, torch)
    with preserve_rng_and_modes([model, runtime["reference"]], torch.device("cpu")), torch.no_grad():
        base_probe = build_probe(config, {"samples": {"forget": f, "retain": r}}, runtime,
                                Path(output).parent / f"baseline-{rank}", torch)
        baseline_fixed = base_probe._fixed_losses(6, 0.96)
        parallel_probe = build_probe(config, {"samples": {"forget": f, "retain": r}}, runtime,
                                    Path(output).parent / f"parallel-{rank}", torch)
        fixed_driver = ShardedEvaluation(torch)
        with fixed_driver.activate(parallel_probe, {"student": model, "reference": runtime["reference"]}):
            parallel_fixed = parallel_probe._fixed_losses(6, 0.96)
        assert parallel_fixed == baseline_fixed
        if rank == 0:
            assert (base_probe.epoch_output / "fixed_loss_samples.jsonl").read_text() == (parallel_probe.epoch_output / "fixed_loss_samples.jsonl").read_text()
        assert "_fixed_losses" not in parallel_probe.__dict__  # restore class method
    assert runtime_fingerprint(runtime, torch) == before_fixed
    gathered = [None] * 4
    dist.all_gather_object(gathered, driver.stats)
    if rank == 0:
        assert actual == expected, compare_values(expected, actual)
        assert sum(v["native_real_batches"] for v in gathered) == 5
        assert sum(v["native_dummy_batches"] for v in gathered) == 3
        assert sum(v["native_real_answer_samples"] for v in gathered) == 26
        Path(output).write_text(json.dumps(gathered))
    dist.destroy_process_group()


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch tests run in PALU server environment")
class DistributedTests(unittest.TestCase):
    def test_four_rank_batch_shapes_generation_tails_and_live_training_state(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(distributed_worker, args=(str(Path(folder) / "init"), str(Path(folder) / "result")),
                     nprocs=4, join=True)
            self.assertTrue((Path(folder) / "result").exists())


if __name__ == "__main__":
    unittest.main()
