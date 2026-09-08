from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from representation_batching.manifest import (  # noqa: E402
    build_arm_steps,
    load_batch_manifest,
    make_metadata,
    split_step_for_rank,
    write_batch_manifest,
)
from representation_batching.local_dataset import resolve_local_dataset_config  # noqa: E402


def clustered_features() -> np.ndarray:
    return np.asarray(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.98, -0.02],
            [0.97, 0.03],
            [-1.00, 0.00],
            [-0.99, -0.01],
            [-0.98, 0.02],
            [-0.97, -0.03],
        ],
        dtype=np.float32,
    )


class LocalDatasetResolutionTests(unittest.TestCase):
    def test_resolves_hub_style_config_to_local_json(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_file = root / "forget05.json"
            data_file.write_text('{"question":"q","answer":"a"}\n', encoding="utf-8")
            self.assertEqual(
                resolve_local_dataset_config(str(root), "forget05"),
                ("json", data_file.resolve()),
            )

    def test_missing_local_config_falls_back_to_normal_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(
                resolve_local_dataset_config(temp, "forget05")
            )

    def test_ambiguous_local_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "forget05.json").touch()
            (root / "forget05.parquet").touch()
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                resolve_local_dataset_config(str(root), "forget05")


class GroupingTests(unittest.TestCase):
    def setUp(self):
        self.ids = list(range(8))
        self.features = clustered_features()
        self.common = {
            "effective_batch_size": 4,
            "retain_size": 20,
            "num_epochs": 2,
            "seed": 7,
        }

    def build(self, method: str, batch_order: str = "random"):
        return build_arm_steps(
            self.ids,
            self.features,
            method=method,
            batch_order=batch_order,
            **self.common,
        )

    def test_each_arm_has_identical_coverage_and_retain_schedule(self):
        arms = {method: self.build(method) for method in ("R", "S", "D", "P")}
        for epoch in range(2):
            epoch_arms = {
                method: [step for step in steps if step["epoch"] == epoch]
                for method, steps in arms.items()
            }
            for steps in epoch_arms.values():
                seen = [sample for step in steps for sample in step["forget_indices"]]
                self.assertEqual(sorted(seen), self.ids)
            retain_sequences = [
                [step["retain_indices"] for step in steps]
                for steps in epoch_arms.values()
            ]
            self.assertTrue(all(sequence == retain_sequences[0] for sequence in retain_sequences))

    def test_similar_and_diverse_grouping_change_within_batch_cosine(self):
        similar = np.mean([step["within_batch_cosine"] for step in self.build("S")])
        diverse = np.mean([step["within_batch_cosine"] for step in self.build("D")])
        self.assertGreater(similar, diverse)

    def test_pseudo_feature_control_is_deterministic(self):
        first = self.build("P")
        second = self.build("P")
        self.assertEqual(
            [step["forget_indices"] for step in first],
            [step["forget_indices"] for step in second],
        )

    def test_non_finite_features_are_rejected(self):
        features = self.features.copy()
        features[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite feature"):
            build_arm_steps(
                self.ids,
                features,
                method="S",
                **self.common,
            )

    def test_batch_order_changes_order_without_changing_partition(self):
        random_steps = self.build("S", batch_order="random")
        similar_steps = self.build("S", batch_order="similar")
        for epoch in range(2):
            random_batches = [
                tuple(sorted(step["forget_indices"]))
                for step in random_steps
                if step["epoch"] == epoch
            ]
            similar_batches = [
                tuple(sorted(step["forget_indices"]))
                for step in similar_steps
                if step["epoch"] == epoch
            ]
            self.assertEqual(sorted(random_batches), sorted(similar_batches))


class ManifestRuntimeContractTests(unittest.TestCase):
    def test_round_trip_validation_and_rank_split(self):
        ids = list(range(8))
        features = clustered_features()
        steps = build_arm_steps(
            ids,
            features,
            method="S",
            effective_batch_size=4,
            retain_size=20,
            num_epochs=1,
            seed=3,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            feature_file = root / "features.npz"
            np.savez(feature_file, sample_ids=np.asarray(ids), feature=features)
            metadata = make_metadata(
                method="S",
                sample_ids=ids,
                feature_file=feature_file,
                feature_key="feature",
                effective_batch_size=4,
                retain_size=20,
                num_epochs=1,
                seed=3,
                world_size=2,
                per_device_batch_size=1,
                gradient_accumulation_steps=2,
            )
            manifest_path = root / "S.jsonl"
            write_batch_manifest(manifest_path, metadata, steps)
            loaded = load_batch_manifest(manifest_path)
            step = loaded.steps_for_epoch(0)[0]
            rank_zero = split_step_for_rank(
                step, rank=0, world_size=2, per_device_batch_size=1
            )
            rank_one = split_step_for_rank(
                step, rank=1, world_size=2, per_device_batch_size=1
            )
            self.assertEqual(len(rank_zero), 2)
            self.assertEqual(len(rank_one), 2)
            reconstructed = []
            for microstep in range(2):
                reconstructed.extend(pair[0] for pair in rank_zero[microstep])
                reconstructed.extend(pair[0] for pair in rank_one[microstep])
            self.assertEqual(reconstructed, step["forget_indices"])

    def test_manifest_rejects_a_partial_optimizer_step(self):
        ids = list(range(8))
        features = clustered_features()
        steps = build_arm_steps(
            ids,
            features,
            method="R",
            effective_batch_size=4,
            retain_size=20,
            num_epochs=1,
            seed=3,
        )
        steps[-1] = {
            **steps[-1],
            "forget_indices": steps[-1]["forget_indices"][:2],
            "retain_indices": steps[-1]["retain_indices"][:2],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            feature_file = root / "features.npz"
            np.savez(feature_file, sample_ids=np.asarray(ids), feature=features)
            metadata = make_metadata(
                method="R",
                sample_ids=ids,
                feature_file=feature_file,
                feature_key="feature",
                effective_batch_size=4,
                retain_size=20,
                num_epochs=1,
                seed=3,
                world_size=2,
                per_device_batch_size=1,
                gradient_accumulation_steps=2,
            )
            manifest_path = root / "R.jsonl"
            write_batch_manifest(manifest_path, metadata, steps)
            with self.assertRaisesRegex(ValueError, "every optimizer step"):
                load_batch_manifest(manifest_path)

    def test_grouping_rejects_partial_accumulation_window(self):
        ids = list(range(9))
        features = np.vstack([clustered_features(), np.asarray([[0.0, 1.0]])])
        with self.assertRaisesRegex(ValueError, "not divisible by effective batch size"):
            build_arm_steps(
                ids,
                features,
                method="R",
                effective_batch_size=4,
                retain_size=20,
                num_epochs=1,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
