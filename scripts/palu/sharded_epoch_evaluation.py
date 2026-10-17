"""Sample-parallel evaluation on existing ZeRO-3 shards.

Fixed losses retain batch size one. TOFU distributes *whole original batches*,
preserving their members, padding, precision and native aggregation formulas.
Dummy work aligns ZeRO collectives at uneven tails and is never scored. All
ranks reconstruct globally ordered per-example results before FQ/MU aggregation.
"""
from collections import defaultdict
from contextlib import contextmanager
import json
from types import MethodType


def shard_slots(size, rank, world):
    if size <= 0 or not 0 <= rank < world:
        raise ValueError("Nonempty data and a valid rank are required")
    return [(ordinal if ordinal < size else 0, ordinal < size)
            for ordinal in range(rank, ((size + world - 1) // world) * world, world)]


def gather(value, torch):
    values = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(values, value)
    return values


def summarize_fixed(rows, manifest, gamma, alpha):
    from trainer.palu_diagnostics import LOSS_KEYS
    sums = dict.fromkeys(("sensitive", "common", "retain"), 0.0)
    macro, counts = dict(sums), dict.fromkeys(sums, 0)
    sizes, zeros = {"forget": 0, "retain": 0}, {"sensitive": 0, "common": 0}
    for row in rows:
        split = row["split"]
        sizes[split] += 1
        for key, field in (("sensitive", "raw_sensitive_loss"), ("common", "raw_common_kl"),
                           ("retain", "raw_retain_nll")):
            if row.get(field) is None:
                continue
            value, count = row[field], row[key + "_token_count"]
            sums[key] += value * count
            macro[key] += value
            counts[key] += count
            if key in zeros:
                zeros[key] += count == 0

    def seven(values):
        f, r, c = (values[key] for key in ("sensitive", "retain", "common"))
        weighted = [gamma * f, alpha * r, alpha * c]
        return dict(zip(LOSS_KEYS, [sum(weighted), *weighted, f, r, c]))

    token = seven({key: sums[key] / max(counts[key], 1) for key in sums})
    mean = seven({key: macro[key] / sizes["retain" if key == "retain" else "forget"] for key in sums})
    return {**token, **{"sample_mean_" + key: value for key, value in mean.items()},
            "forget_count": sizes["forget"], "retain_count": sizes["retain"],
            **{key + "_token_count": value for key, value in counts.items()},
            "forget_zero_sensitive_examples": zeros["sensitive"],
            "forget_zero_common_examples": zeros["common"],
            "fixed_dataset_sha256": manifest["fixed_dataset_sha256"]}


class ShardedEvaluation:
    def __init__(self, torch):
        self.torch = torch
        self.rank, self.world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        self.stats = {"rank": self.rank, "fixed_real_samples": 0, "fixed_dummy_samples": 0,
                      "native_real_batches": 0, "native_dummy_batches": 0,
                      "native_real_answer_samples": 0, "forward_calls": 0,
                      "cached_decode_forwards": 0}

    def fixed_losses(self, probe, original, step, epoch):
        from trainer.palu_diagnostics import append_history
        probe._prepare_fixed_data()
        full, output = probe._epoch_fixed, probe.epoch_output
        slots = {split: shard_slots(len(full[split]), self.rank, self.world) for split in full}
        local_output = output / "fixed_shards" / f"step-{step}-rank-{self.rank}"
        local_output.mkdir(parents=True, exist_ok=False)
        try:
            probe.epoch_output = local_output
            probe._epoch_fixed = {split: [full[split][i] for i, _ in values] for split, values in slots.items()}
            original(step, epoch)
        finally:
            probe.epoch_output, probe._epoch_fixed = output, full
        local_rows = [json.loads(line) for line in (local_output / "fixed_loss_samples.jsonl").read_text().splitlines()]
        flags = [valid for split in ("forget", "retain") for _, valid in slots[split]]
        if len(local_rows) != len(flags):
            raise AssertionError("Fixed-loss shard row count differs from assignment")
        self.stats["fixed_real_samples"] += sum(flags)
        self.stats["fixed_dummy_samples"] += len(flags) - sum(flags)
        rows = [row for shard in gather([row for row, valid in zip(local_rows, flags) if valid], self.torch)
                for row in shard]
        by_key = {(row["split"], row["dataset_index"]): row for row in rows}
        expected = [(split, index) for split in ("forget", "retain") for index, _ in full[split]]
        if len(by_key) != len(rows) or set(by_key) != set(expected):
            raise AssertionError("Fixed-loss shards duplicate or omit samples")
        ordered = [by_key[key] for key in expected]
        if self.rank == 0:
            for row in ordered:
                append_history(output / "fixed_loss_samples.jsonl", row)
        return summarize_fixed(ordered, probe._epoch_fixed_manifest, probe.gamma, probe.alpha)

    def run_batches(self, model, dataloader, batch_eval_fn, batch_eval_fn_args, eval_msg):
        from torch.utils.data import SequentialSampler
        from evals.metrics.utils import dict_transpose
        if not isinstance(dataloader.sampler, SequentialSampler) or dataloader.drop_last:
            raise ValueError("Sample-parallel TOFU requires the original sequential, non-dropping loader")
        batches = list(dataloader.batch_sampler)
        records = []
        for ordinal, valid in shard_slots(len(batches), self.rank, self.world):
            batch = dataloader.collate_fn([dataloader.dataset[i] for i in batches[ordinal]])
            if "input_ids" in batch:
                batch = {"0": batch}
            keys = list(batch)
            if any(other != keys for other in gather(keys, self.torch)):
                raise ValueError("Different answer-option counts would desynchronize ZeRO forwards")
            self.stats["native_real_batches" if valid else "native_dummy_batches"] += 1
            answers = []
            for intra, mini in batch.items():
                indices = mini.pop("index").cpu().numpy().tolist()
                values = batch_eval_fn(model=model, batch=mini, **batch_eval_fn_args)
                if len(indices) != len(values):
                    raise AssertionError("Per-example result count differs from batch")
                answers.append((intra, indices, values))
                if valid:
                    self.stats["native_real_answer_samples"] += len(indices)
            if valid:
                records.append((ordinal, answers))
        records = sorted((record for shard in gather(records, self.torch) for record in shard), key=lambda x: x[0])
        if [ordinal for ordinal, _ in records] != list(range(len(batches))):
            raise AssertionError("Native evaluation shards duplicate or omit batches")
        evals = defaultdict(dict)
        for _, answers in records:
            for intra, indices, values in answers:
                if len(set(indices)) != len(indices) or evals[intra].keys() & set(indices):
                    raise AssertionError("Native evaluation repeated sample indices")
                evals[intra].update(zip(indices, values))
        if any(len(values) != len(dataloader.dataset) for values in evals.values()):
            raise AssertionError("Native evaluation sample coverage incomplete")
        if self.rank == 0:
            print(json.dumps({"event": "sharded_metric_complete", "metric": eval_msg,
                              "samples": len(dataloader.dataset), "original_batches": len(batches)}), flush=True)
        return next(iter(evals.values())) if len(evals) == 1 else dict_transpose(evals)

    @contextmanager
    def activate(self, probe, models):
        from evals.metrics import memorization
        from online_epoch_evaluation import _cache_length
        old_runner, old_fixed = memorization.run_batchwise_evals, probe._fixed_losses
        had_fixed = "_fixed_losses" in probe.__dict__
        originals = {tag: type(model) for tag, model in models.items()}
        driver = self
        try:
            memorization.run_batchwise_evals = self.run_batches
            probe._fixed_losses = MethodType(lambda instance, step, epoch: driver.fixed_losses(instance, old_fixed, step, epoch), probe)
            for tag, model in models.items():
                def wrapped(base):
                    peer_finished = False
                    class SampleParallelModel(base):
                        def _has_unfinished_sequences(instance, this_peer_finished, synced_gpus, *args, **kwargs):
                            nonlocal peer_finished
                            peer_finished = this_peer_finished
                            return base._has_unfinished_sequences(instance, this_peer_finished, synced_gpus, *args, **kwargs)

                        def __call__(instance, *args, **kwargs):
                            driver.stats["forward_calls"] += 1
                            cache = kwargs.get("past_key_values")
                            length = _cache_length(cache)
                            driver.stats["cached_decode_forwards"] += int(length > 0)
                            # Transformers 4.45 skips kwargs updates on finished
                            # peers, but DynamicCache mutates in-place during their
                            # required ZeRO dummy forwards. Undo ONLY that unused
                            # append, keeping mask/cache aligned on the next dummy.
                            rollback = peer_finished and cache is not None and hasattr(cache, "get_seq_length")
                            if rollback and type(cache).__name__ != "DynamicCache":
                                raise ValueError("Finished-peer cache rollback requires DynamicCache")
                            try:
                                return base.__call__(instance, *args, **kwargs)
                            finally:
                                if rollback:
                                    cache.crop(length)

                        def generate(instance, *args, **kwargs):
                            nonlocal peer_finished
                            generation = kwargs.get("generation_config") or instance.generation_config
                            if kwargs.get("num_beams", generation.num_beams) != 1 or kwargs.get("do_sample", generation.do_sample):
                                raise ValueError("This protocol preserves native greedy single-beam generation")
                            # Finished ranks must continue parameter-gather forwards
                            # until every rank finishes its distinct original batch.
                            kwargs["synced_gpus"] = True
                            peer_finished = False
                            try:
                                return base.generate(instance, *args, **kwargs)
                            finally:
                                peer_finished = False
                    return SampleParallelModel
                model.__class__ = wrapped(originals[tag])
            yield self
        finally:
            memorization.run_batchwise_evals = old_runner
            if had_fixed:
                probe._fixed_losses = old_fixed
            else:
                del probe._fixed_losses
            for tag, model in models.items():
                model.__class__ = originals[tag]
