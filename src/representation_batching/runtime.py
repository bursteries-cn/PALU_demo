from __future__ import annotations

from typing import Iterator, List, Tuple

from torch.utils.data import DataLoader, Dataset, Sampler

from representation_batching.manifest import BatchManifest, split_step_for_rank


class ScheduledForgetRetainDataset(Dataset):
    """Read deterministic forget/retain pairs without mutating the base dataset."""

    def __init__(self, base_dataset: Dataset):
        if not hasattr(base_dataset, "forget") or not hasattr(base_dataset, "retain"):
            raise TypeError("Scheduled dataset requires a ForgetRetainDataset-like object")
        if base_dataset.forget is None or base_dataset.retain is None:
            raise ValueError("Both forget and retain datasets are required")
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset.forget)

    def __getitem__(self, pair: Tuple[int, int]):
        forget_index, retain_index = pair
        return {
            "forget": self.base_dataset.forget[int(forget_index)],
            "retain": self.base_dataset.retain[int(retain_index)],
        }


class ManifestBatchSampler(Sampler[List[Tuple[int, int]]]):
    """Yield the exact per-rank microbatches encoded by a global manifest."""

    def __init__(
        self,
        manifest: BatchManifest,
        *,
        rank: int,
        world_size: int,
        per_device_batch_size: int,
    ):
        self.manifest = manifest
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.per_device_batch_size = int(per_device_batch_size)
        self.epoch = 0
        meta = manifest.metadata
        expected = (
            int(meta["world_size"]),
            int(meta["per_device_batch_size"]),
        )
        actual = (self.world_size, self.per_device_batch_size)
        if actual != expected:
            raise ValueError(
                f"Runtime (world_size, per_device_batch_size)={actual} does not match manifest {expected}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        for step in self.manifest.steps_for_epoch(self.epoch):
            yield from split_step_for_rank(
                step,
                rank=self.rank,
                world_size=self.world_size,
                per_device_batch_size=self.per_device_batch_size,
            )

    def __len__(self) -> int:
        return sum(
            len(step["forget_indices"])
            // (self.world_size * self.per_device_batch_size)
            for step in self.manifest.steps_for_epoch(self.epoch)
        )


class ManifestDataLoader(DataLoader):
    """Expose set_epoch because Trainer constructs the dataloader only once."""

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)
