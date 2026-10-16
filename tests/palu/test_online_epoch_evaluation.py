"""Actual epoch timing and CPU checks for pause/resume and coordinated inference.

The Gloo test exercises four processes and HF generation/cache behavior, not
CUDA/ZeRO-3 numerical equivalence. Full GPU runs remain user-started experiments.
"""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/palu"))
sys.path.insert(0, str(ROOT / "src"))
from online_epoch_evaluation import epoch_events


class TimingTests(unittest.TestCase):
    def test_actual_edges_and_pending_gradients_match_seed103(self):
        events = epoch_events()
        self.assertEqual([row["step"] for row in events], [0, 6, 12, 18, 25, 31, 37, 43, 50, 56, 60])
        self.assertEqual([row["consumed_pairs"] for row in events], [0, *range(200, 1801, 200), 1920])
        self.assertEqual([row["pending_pairs"] for row in events], [0, 8, 16, 24, 0, 8, 16, 24, 0, 8, 0])
        self.assertEqual(events[1]["actual_data_epochs"], 1.0)
        self.assertEqual(events[1]["weight_update_epochs"], 0.96)
        self.assertEqual(events[-1]["source_epoch_index"], 10)

    def test_both_microbatch_layouts_reach_edges_without_flushing(self):
        for global_microbatch in (8, 4):
            events = {row["consumed_pairs"]: row for row in epoch_events()}
            observed, completed = [], 0
            for consumed in range(global_microbatch, 1921, global_microbatch):
                # Training updates first only at the last microbatch of a window.
                if consumed % 32 == 0:
                    completed += 1
                if consumed in events:
                    observed.append((consumed, completed))
            self.assertEqual(observed, [(row["consumed_pairs"], row["step"]) for row in events.values() if row["consumed_pairs"]])


def tiny_model(torch):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        attention_dropout=0.0, pad_token_id=0, eos_token_id=2, use_cache=False)
    cfg._attn_implementation = "eager"
    return LlamaForCausalLM(cfg).double()


def tiny_runtime(torch, model):
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    return {"engine": model, "model": model, "reference": reference, "optimizer": optimizer,
            "scheduler": scheduler, "device": torch.device("cpu"), "rank": 0, "world": 1, "distributed": False}


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch available in server PALU environment")
class StateTests(unittest.TestCase):
    def lifecycle(self, mutate_at=None):
        """Use synthetic metric files to test the real commit/delete/resume path."""
        import torch
        import checkpoint_retention as retention
        from online_epoch_evaluation import OnlineEpochEvaluation
        torch.manual_seed(103)
        runtime = tiny_runtime(torch, tiny_model(torch))
        ids = torch.tensor([[3, 4, 5]])
        runtime["model"](input_ids=ids, labels=ids).loss.backward()
        runtime["optimizer"].step()
        runtime["optimizer"].zero_grad(set_to_none=True)
        (runtime["model"](input_ids=ids, labels=ids).loss / 4).backward()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        run = Path(temporary.name)
        args = SimpleNamespace(output=run, delete_evaluated_intermediate=True)
        manifest = {"requested_updates": 7, "final_model_step": 7, "layout": "single",
            "normalization": "native_microbatch_means", "padding_scope": "microbatch",
            "schedule_sha256": "test-schedule", "initial_parameter_sha256": "test-initial",
            "teacher_parameter_sha256": "test-initial", "schedule_metadata": {
                "data_sizes": {"forget": 200, "retain": 3}, "fixed_dataset_sha256": "test-fixed",
                "fixed_target_counts": {"sensitive": 200, "common": 200, "retain": 6},
                "seed103_source_provenance": {"reference_sha256": "test-ref"}},
            "checkpoints": [{"step": 0, "checkpoint": "external-Full", "parameter_sha256": "test-initial"}]}
        trace = []
        workspace = run / "online_evaluation_workspace"
        workspace.mkdir()

        class SyntheticProbe:
            epoch_output = workspace
            state = SimpleNamespace(global_step=0, epoch=0.0)
            _epoch_context = {}
            _epoch_fixed_manifest = {"fixed_dataset_sha256": "test-fixed"}

            def _prepare_fixed_data(self):
                pass

            def evaluate(self):
                step = self.state.global_step
                trace.append(("evaluate", step))
                output = workspace / f"checkpoint-{step}" / "evals"
                output.mkdir(parents=True)
                row = {"step": step, "evaluation_dir": str(output), "fixed_dataset_sha256": "test-fixed",
                    "reference_sha256": "test-ref", "raw_sensitive_loss": 1.0, "raw_common_kl": 0.3,
                    "raw_retain_nll": 0.2, "forget_quality": 0.5, "model_utility": 0.7, "ks_d": 0.1}
                with (workspace / "epoch_history.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                samples = [{"step": step, "split": "forget", "dataset_index": i,
                    "raw_sensitive_loss": 1.0, "sensitive_token_count": 1,
                    "raw_common_kl": 0.3, "common_token_count": 1} for i in range(200)]
                samples += [{"step": step, "split": "retain", "dataset_index": i,
                    "raw_retain_nll": 0.2, "retain_token_count": 2} for i in range(3)]
                with (workspace / "fixed_loss_samples.jsonl").open("a") as handle:
                    handle.writelines(json.dumps(sample) + "\n" for sample in samples)
                sizes = {"forget_truth_ratio": 200, "forget_Q_A_PARA_Prob": 200, "forget_Q_A_PERT_Prob": 200}
                for prefix, size, suffixes in (
                    ("retain", 400, ("Q_A_Prob", "Q_A_PARA_Prob", "Q_A_PERT_Prob", "Q_A_ROUGE", "Truth_Ratio")),
                    ("ra", 100, ("Q_A_Prob", "Q_A_PERT_Prob", "Q_A_Prob_normalised", "Q_A_ROUGE", "Truth_Ratio")),
                    ("wf", 117, ("Q_A_Prob", "Q_A_PERT_Prob", "Q_A_Prob_normalised", "Q_A_ROUGE", "Truth_Ratio"))):
                    sizes.update({prefix + "_" + suffix: size for suffix in suffixes})
                native = {key: {"value_by_index": {str(i): {"score": 0.1} for i in range(size)}} for key, size in sizes.items()}
                native.update({key: {"agg_value": row[key]} for key in ("forget_quality", "model_utility")})
                retention.atomic_json(output / "TOFU_EVAL.json", native)
                if step == mutate_at:
                    next(runtime["model"].parameters()).grad.add_(0.01)

        def save(event):
            trace.append(("save", event["step"]))
            folder = run / f"checkpoint-{event['step']}"
            folder.mkdir()
            weight = folder / "model.safetensors"
            weight.write_bytes(b"Synthetic test weight, not a research model")
            return {**event, "checkpoint": str(folder), "parameter_sha256": "test-weight-" + str(event["step"]),
                    "files": {weight.name: {"bytes": weight.stat().st_size, "sha256": retention.sha(weight)}}}

        driver = OnlineEpochEvaluation(args, {}, {"metadata": manifest["schedule_metadata"]}, runtime, manifest, torch)
        with patch("online_epoch_evaluation.build_probe", return_value=SyntheticProbe()):
            driver.at(0, 0, save)
            if mutate_at:
                with self.assertRaisesRegex(RuntimeError, "Evaluation changed training state"):
                    driver.at(200, 6, save)
                self.assertTrue((run / "checkpoint-6/model.safetensors").exists())
                self.assertEqual(manifest["status"], "online_evaluation_failed")
                self.assertFalse((run / "checkpoint-6/evaluation_receipt.json").exists())
                return
            driver.at(200, 6, save)
            # Training may resume only once intermediate weights are gone and
            # both native and fixed per-example observations are committed.
            self.assertFalse((run / "checkpoint-6/model.safetensors").exists())
            self.assertTrue((run / "checkpoint-6/evaluation_receipt.json").is_file())
            trace.append(("resume", 6))
            driver.at(224, 7, save)
        self.assertTrue((run / "checkpoint-7/model.safetensors").is_file())
        self.assertEqual(trace, [("evaluate", 0), ("save", 6), ("evaluate", 6), ("resume", 6), ("save", 7), ("evaluate", 7)])
        self.assertEqual(manifest["evaluated_steps"], [0, 6, 7])

    def test_actual_driver_commits_deletes_then_resumes_and_retains_final(self):
        self.lifecycle()

    def test_actual_driver_stops_and_retains_weight_if_state_changes(self):
        self.lifecycle(mutate_at=6)

    def test_pending_gradients_adam_lr_rng_and_next_update_unchanged(self):
        import numpy as np
        import torch
        from online_epoch_evaluation import runtime_fingerprint
        from trainer.palu_diagnostics import preserve_rng_and_modes
        torch.set_num_threads(1)
        torch.manual_seed(103)
        random.seed(103)
        np.random.seed(103)
        runtime = tiny_runtime(torch, tiny_model(torch))
        model, optimizer, scheduler = (runtime[key] for key in ("model", "optimizer", "scheduler"))
        ids = torch.tensor([[3, 4, 5, 6]])
        model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        # A partial update is already accumulated at evaluation time.
        (model(input_ids=ids, labels=ids).loss / 4).backward()
        control = copy.deepcopy(model)
        for original, cloned in zip(model.parameters(), control.parameters()):
            cloned.grad = original.grad.clone()
        control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.01)
        control_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        before = runtime_fingerprint(runtime, torch)
        with preserve_rng_and_modes([model, runtime["reference"]], runtime["device"]), torch.no_grad():
            random.random()
            np.random.random()
            torch.rand(7)
            model.generate(ids, do_sample=False, max_new_tokens=4, use_cache=True, synced_gpus=False)
            runtime["reference"](input_ids=ids)
        self.assertEqual(before, runtime_fingerprint(runtime, torch))
        (model(input_ids=ids, labels=ids).loss * 3 / 4).backward()
        (control(input_ids=ids, labels=ids).loss * 3 / 4).backward()
        optimizer.step()
        control_optimizer.step()
        for actual, expected in zip(model.parameters(), control.parameters()):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(before["scheduler"], runtime_fingerprint(runtime, torch)["scheduler"])

    def test_guard_detects_changed_moment_or_pending_gradient(self):
        import torch
        from online_epoch_evaluation import runtime_fingerprint
        torch.manual_seed(103)
        runtime = tiny_runtime(torch, tiny_model(torch))
        ids = torch.tensor([[3, 4, 5]])
        runtime["model"](input_ids=ids, labels=ids).loss.backward()
        runtime["optimizer"].step()
        before = runtime_fingerprint(runtime, torch)
        next(runtime["model"].parameters()).grad.add_(0.01)
        self.assertNotEqual(before["student"]["gradients"], runtime_fingerprint(runtime, torch)["student"]["gradients"])
        next(iter(runtime["optimizer"].state.values()))["exp_avg"].add_(0.01)
        self.assertNotEqual(before["optimizer"], runtime_fingerprint(runtime, torch)["optimizer"])

    def test_native_fixed_loss_probe_reuses_weights_and_matches_palu(self):
        import torch
        from online_epoch_evaluation import build_probe, runtime_fingerprint
        from gradient_equivalence_step import raw_components, target_counts
        from trainer.palu_diagnostics import preserve_rng_and_modes
        torch.manual_seed(103)
        runtime = tiny_runtime(torch, tiny_model(torch))
        runtime["tokenizer"] = SimpleNamespace(eos_token_id=2, pad_token_id=0)
        with torch.no_grad():
            runtime["model"].lm_head.weight.add_(0.01)
        f = [{"input_ids": torch.tensor([3, 4, 5, 6]), "labels": torch.tensor([-100, 4, -200, 6])},
             {"input_ids": torch.tensor([3, 7, 8]), "labels": torch.tensor([-100, -200, 8])}]
        r = [{"input_ids": torch.tensor([3, 9, 10]), "labels": torch.tensor([-100, 9, 10])}]
        config = json.loads((ROOT / "configs/analysis/layout_loss_seed103/resolved_config.json").read_text())
        config["trainer"]["method_args"].update(top_k=7, first_n=3)
        with tempfile.TemporaryDirectory() as temporary:
            before = runtime_fingerprint(runtime, torch)
            with preserve_rng_and_modes([runtime["model"], runtime["reference"]], runtime["device"]), torch.no_grad():
                probe = build_probe(config, {"samples": {"forget": f, "retain": r}}, runtime, temporary, torch)
                measured = probe._fixed_losses(6, 0.96)
                weighted, counts = {key: 0.0 for key in ("sensitive", "common", "retain")}, {key: 0 for key in ("sensitive", "common", "retain")}
                for sample in f:
                    batch = probe.data_collator([{"forget": sample, "retain": r[0]}])
                    raw = raw_components(runtime["model"], runtime["reference"], batch, top_k=7, first_n=3, eos=2, torch=torch)
                    local = target_counts(batch, 2, 3, torch)
                    for key in ("sensitive", "common"):
                        weighted[key] += float(raw[key]) * local[key]
                        counts[key] += local[key]
                retain = float(runtime["model"](input_ids=r[0]["input_ids"][None], labels=r[0]["labels"][None]).loss)
            self.assertIs(probe.model, runtime["model"])
            self.assertIs(probe.ref_model, runtime["reference"])
            self.assertAlmostEqual(measured["raw_sensitive_loss"], weighted["sensitive"] / counts["sensitive"], places=10)
            self.assertAlmostEqual(measured["raw_common_kl"], weighted["common"] / counts["common"], places=10)
            self.assertAlmostEqual(measured["raw_retain_nll"], retain, places=10)
            self.assertEqual(before, runtime_fingerprint(runtime, torch))


def gloo_worker(rank, world, init_path, output):
    import numpy as np
    import torch
    from online_epoch_evaluation import CoordinatedForward, runtime_fingerprint
    from trainer.palu_diagnostics import preserve_rng_and_modes
    from transformers.cache_utils import DynamicCache
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method="file://" + init_path, rank=rank, world_size=world)
    torch.manual_seed(103)
    random.seed(103)
    np.random.seed(103)
    runtime = tiny_runtime(torch, tiny_model(torch))
    model, ref = runtime["model"], runtime["reference"]
    ids = torch.tensor([[3, 4, 5], [6, 7, 8]])
    model(input_ids=ids, labels=ids).loss.backward()
    runtime["optimizer"].step()
    runtime["optimizer"].zero_grad(set_to_none=True)
    (model(input_ids=ids, labels=ids).loss / 4).backward()
    baseline = copy.deepcopy(model).eval()
    before = runtime_fingerprint(runtime, torch)
    rpc = CoordinatedForward({"student": model, "reference": ref}, runtime["device"], torch)
    error = None
    with preserve_rng_and_modes([model, ref], runtime["device"]), torch.no_grad():
        if rank == 0:
            try:
                with rpc.root_models():
                    torch.testing.assert_close(model(input_ids=ids, use_cache=False).logits,
                                               baseline(input_ids=ids, use_cache=False).logits, rtol=0, atol=0)
                    ref(input_ids=ids[:1], use_cache=False)
                    for batch in (ids, ids[:1]):
                        actual = model.generate(batch, do_sample=False, max_new_tokens=5, eos_token_id=None,
                                                pad_token_id=0, use_cache=True)
                        expected = baseline.generate(batch, do_sample=False, max_new_tokens=5, eos_token_id=None,
                                                    pad_token_id=0, use_cache=True, synced_gpus=False)
                        if not torch.equal(actual, expected):
                            raise AssertionError("Coordinated greedy generation differs from local model")
                    # Exercise explicit dynamic-cache input as well as HF's
                    # installed generation cache representation.
                    cache = DynamicCache()
                    result = model(input_ids=ids, past_key_values=cache, use_cache=True)
                    model(input_ids=ids[:, -1:], past_key_values=result.past_key_values, use_cache=True)
                random.random()
                torch.rand(4)
                np.random.random()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                rpc.finish(error)
        else:
            error = rpc.workers()
    unchanged = before == runtime_fingerprint(runtime, torch)
    result = {"rank": rank, "unchanged": unchanged, "calls": rpc.calls,
              "cached_decode_forwards": rpc.generated_tokens, "error": error,
              "cuda_initialized": torch.cuda.is_initialized()}
    Path(output, f"rank-{rank}.json").write_text(json.dumps(result))
    torch.distributed.destroy_process_group()
    if error or not unchanged:
        raise AssertionError(result)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch available in server PALU environment")
class DistributedTests(unittest.TestCase):
    def test_four_gloo_ranks_generation_cache_and_pending_state(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as temporary:
            mp.spawn(gloo_worker, args=(4, str(Path(temporary, "init")), temporary), nprocs=4, join=True)
            rows = [json.loads(Path(temporary, f"rank-{rank}.json").read_text()) for rank in range(4)]
            self.assertTrue(all(row["unchanged"] and row["error"] is None and not row["cuda_initialized"] for row in rows))
            self.assertEqual(len({row["calls"] for row in rows}), 1)
            self.assertGreater(rows[0]["cached_decode_forwards"], 0)


if __name__ == "__main__":
    unittest.main()
