"""Freeze a real multi-seed experiment without silently changing its plan."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/palu"))
import forget_until_threshold as experiment


class PrepareTests(unittest.TestCase):
    def test_budget_and_complete_update_evaluations(self):
        plan = experiment.protocol()
        self.assertEqual(plan["phase_weights"], {"forget": [1., 0., 2.]})
        self.assertEqual(plan["retain_updates"], 0)
        self.assertEqual(plan["probe_every_updates"], 1)
        self.assertEqual(plan["max_forget_exposure_epochs"], 10.08)
        events = experiment.evaluation_events(plan)
        self.assertEqual([row["step"] for row in events], [0, 13, 25, 38, 50, 63])
        self.assertTrue(all(row["pending_pairs"] == 0 for row in events))
        self.assertEqual(events[1]["rounded_up_pairs"], 16)
        self.assertEqual(events[2]["rounded_up_pairs"], 0)
        for count in (1, 12, 13, 25, 62, 63):
            events = experiment.evaluation_events(experiment.protocol(max_updates=count))
            self.assertEqual(events[-1]["step"], count)
            self.assertEqual(len(events), len({row["step"] for row in events}))
            self.assertTrue(all(row["step"] <= count for row in events))

    def test_invalid_plan_fails(self):
        for changes in (dict(seed=-1), dict(seed=2**32), dict(seed=True),
                        dict(max_updates=0), dict(max_updates=1.5),
                        dict(threshold_low=.5), dict(threshold_high=float("nan")),
                        dict(learning_rate=0), dict(learning_rate=float("inf"))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                experiment.protocol(**changes)
        self.assertGreaterEqual(experiment.storage_budget(16_000_000_000), 160*(1 << 30))

    def test_source_method_preserved_and_all_seed_controls_replaced(self):
        source = {"model": {"handler": "Llama-3.1-8B-Instruct"},
            "trainer": {"args": {"seed": 103, "data_seed": 103, "warmup_ratio": .1},
                        "method_args": dict(alpha=2, gamma=1., top_k=5000, first_n=3,
                            target_mode="mean", retain_loss_type="NLL",
                            non_sensitive_constraint_type="kl")},
            "eval": {"tofu": {"retain_logs_path": "/old/frozen/retain_reference.json"}}}
        before = copy.deepcopy(source)
        plan = experiment.protocol(seed=105)
        config = experiment.prepare_config(source, Path("/old"), Path("/new"), plan)
        args = config["trainer"]["args"]
        self.assertEqual((args["seed"], args["data_seed"]), (105, 105))
        self.assertEqual((args["warmup_steps"], args["warmup_epochs"], args["warmup_ratio"]), (0, 0, 0))
        self.assertEqual(config["trainer"]["method_args"], source["trainer"]["method_args"])
        self.assertEqual(config["forget_threshold_plan"]["phase_weights"]["forget"], [1., 0., 2.])
        self.assertEqual(source, before)
        source["trainer"]["method_args"]["first_n"] = 4
        with self.assertRaisesRegex(ValueError, "validated seed103"):
            experiment.prepare_config(source, Path("/old"), Path("/new"), plan)

    def test_frozen_plan_rejects_editing_visible_description_or_config(self):
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder)
            (study/"frozen").mkdir()
            plan = experiment.protocol(seed=104)
            experiment.write_json(study/"frozen/config.json", {"forget_threshold_plan": plan})
            experiment.write_json(study/"frozen/manifest.json", {"frozen_file_sha256": {
                "config.json": experiment.sha(study/"frozen/config.json")}})
            experiment.write_json(study/"experiment.json", dict(plan, status="prepared"))
            self.assertEqual(experiment.frozen_plan(study), plan)
            experiment.write_json(study/"experiment.json", dict(plan, threshold_high=.6))
            with self.assertRaisesRegex(ValueError, "differs from frozen"):
                experiment.frozen_plan(study)
            experiment.write_json(study/"frozen/config.json", {"forget_threshold_plan": dict(plan, seed=105)})
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                experiment.frozen_plan(study)

    def test_dry_run_works_without_site_packages_and_train_rejects_overrides(self):
        script = ROOT/"scripts/palu/forget_until_threshold.py"
        completed = subprocess.run([sys.executable, "-S", str(script), "--mode", "dry-run",
            "--study", "/unused", "--seed", "105"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["seed"], 105)
        rejected = subprocess.run([sys.executable, "-S", str(script), "--mode", "train",
            "--study", "/unused", "--seed", "105"], capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("frozen plan", rejected.stderr)

    def test_seed103_reuses_recorded_pairs_without_torch(self):
        original = {"windows": [{"pair_records": [
            {"draw_index": i+1, "forget_index": i, "retain_index": i+10}
            for i in range(32)]}]}
        result = experiment.pair_records(original, experiment.protocol(max_updates=1), None)
        self.assertEqual(result, original["windows"][0]["pair_records"])
        result[0]["retain_index"] = 999
        self.assertNotEqual(result, original["windows"][0]["pair_records"])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU torch needed for true fresh-generator sampling test")
class FreshScheduleTests(unittest.TestCase):
    def test_new_seeds_change_draws_preserve_epoch_coverage_and_global_rng(self):
        state = torch.random.get_rng_state().clone()
        first = experiment.pair_records({}, experiment.protocol(seed=104), torch)
        repeated = experiment.pair_records({}, experiment.protocol(seed=104), torch)
        other = experiment.pair_records({}, experiment.protocol(seed=105), torch)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(len(first), 2016)
        for start in range(0, 2000, 200):
            self.assertEqual(sorted(row["forget_index"] for row in first[start:start+200]), list(range(200)))
        self.assertNotEqual([row["forget_index"] for row in first], [row["forget_index"] for row in other])
        self.assertNotEqual([row["retain_index"] for row in first], [row["retain_index"] for row in other])
        self.assertTrue(all(0 <= row["retain_index"] < 3800 for row in first))
        self.assertTrue(all(row["pair_origin"] == "fresh_torch_generators_seed104" for row in first))

    def test_prepare_freezes_real_tensors_and_preserves_historical_prefix(self):
        from prepare_layout_schedule import build_schedule
        tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=0)

        class Collator:
            padding_side = "right"

            def __init__(self):
                self.tokenizer = tokenizer

            def __call__(self, rows):
                return {split: {**{key: torch.stack([row[split][key] for row in rows])
                                   for key in ("input_ids", "labels")},
                                "attention_mask": torch.stack([row[split]["input_ids"].ne(0) for row in rows])}
                        for split in ("forget", "retain")}

        samples = {split: [{"input_ids": torch.tensor([1, 2, 3, 4]),
                             "labels": torch.tensor([-100, -200, 3, 4] if split == "forget"
                                                    else [-100, 2, 3, 4])}
                            for _ in range(size)] for split, size in (("forget", 200), ("retain", 3800))}
        original = build_schedule(samples, Collator(), eos=0, first_n=3, updates=60,
                                  effective_batch=32, seed=103, torch=torch)
        original["metadata"]["seed103_source_provenance"] = {"test": "recorded"}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/"source"
            (source/"frozen").mkdir(parents=True)
            model = root/"model"
            model.mkdir()
            (model/"model.safetensors").write_bytes(b"test_weight_inventory_only")
            config = {"model": {"handler": "Llama-3.1-8B-Instruct", "tokenizer_args": {},
                                "model_args": {"pretrained_model_name_or_path": str(model)}},
                      "trainer": {"args": {"seed": 103},
                                  "method_args": dict(alpha=2, gamma=1., top_k=5000, first_n=3,
                                      target_mode="mean", retain_loss_type="NLL",
                                      non_sensitive_constraint_type="kl")},
                      "collator": {}, "eval": {"tofu": {}}}
            experiment.write_json(source/"frozen/config.json", config)
            experiment.write_json(source/"frozen/retain_reference.json", {"test_reference": True})
            torch.save(original, source/"frozen/schedule.pt")
            names = ("config.json", "schedule.pt", "retain_reference.json")
            experiment.write_json(source/"frozen/manifest.json", {"frozen_file_sha256": {
                name: experiment.sha(source/"frozen"/name) for name in names}})
            fake_imports = {
                "model": SimpleNamespace(get_tokenizer=lambda _: tokenizer),
                "data": SimpleNamespace(get_collators=lambda _, **kwargs: Collator()),
                "omegaconf": SimpleNamespace(OmegaConf=SimpleNamespace(create=lambda value: value))}
            for seed in (103, 104):
                study = root/f"seed_{seed}"
                with patch.dict(sys.modules, fake_imports), \
                     patch.object(experiment.shutil, "disk_usage", return_value=SimpleNamespace(free=1 << 50)), \
                     patch.object(experiment, "code_paths", return_value=[Path(experiment.__file__)]), \
                     patch("builtins.print"):
                    experiment.prepare(study, source, experiment.protocol(seed=seed))
                plan = experiment.frozen_plan(study)
                frozen = torch.load(study/"frozen/schedule.pt", weights_only=True)
                self.assertEqual(plan["seed"], seed)
                self.assertEqual(len(frozen["windows"]), 63)
                self.assertEqual(frozen["metadata"]["fixed_dataset_sha256"],
                                 original["metadata"]["fixed_dataset_sha256"])
                hashes = experiment.read_json(study/"frozen/manifest.json")["frozen_file_sha256"]
                self.assertTrue(all(experiment.sha(study/"frozen"/name) == digest
                                    for name, digest in hashes.items()))
                if seed == 103:
                    self.assertEqual([w["window_sha256"] for w in frozen["windows"][:60]],
                                     [w["window_sha256"] for w in original["windows"]])
                else:
                    self.assertNotEqual(frozen["windows"][0]["pair_sha256"],
                                        original["windows"][0]["pair_sha256"])


if __name__ == "__main__":
    unittest.main()
