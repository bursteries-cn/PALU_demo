"""PALU epoch diagnostics on fixed examples and the existing TOFU evaluator.

Training remains one uninterrupted HF Trainer run. Evaluation uses the current
model, restores RNG/module modes, and never starts a second model on the GPU.
The fixed loss pass uses batch size one, so both macro means and token weighted
means have an explicit, stable definition independent of the training batch.
"""
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from transformers import TrainerCallback

from data.unlearn import ForgetRetainDataset
from trainer.base import add_neg_log_pvalue_metrics
from trainer.palu_diagnostics import (LOSS_KEYS, append_history,
                                      preserve_rng_and_modes, sha256_file)
from trainer.unlearn.palu import PALU


def fixed_indices(length, requested="all"):
    """All examples, or evenly spaced deterministic indices, with no RNG calls."""
    if length <= 0:
        raise ValueError("Fixed diagnostic dataset is empty")
    count = length if str(requested) == "all" else int(requested)
    if count <= 0 or count > length:
        raise ValueError(f"Requested {count} examples from a dataset of {length}")
    return [index * length // count for index in range(count)]


def target_counts(forget_labels, retain_labels, first_n, eos_token_id):
    """Match the shifted PALU sensitive/common masks and causal-LM NLL targets."""
    labels = list(forget_labels)[1:]
    sensitive = [index for index, label in enumerate(labels)
                 if label >= 0 and label != eos_token_id]
    if first_n > 0:
        chosen, interval_count, previous = [], 0, None
        for index in sensitive:
            if previous is None or index > previous + 1:
                interval_count = 0
            if interval_count < first_n:
                chosen.append(index)
            interval_count += 1
            previous = index
        sensitive = chosen
    return len(sensitive), sum(label == -200 for label in labels), sum(
        label != -100 for label in list(retain_labels)[1:])


def ks_statistic(left, right):
    left, right = sorted(left), sorted(right)
    if not left or not right or not all(math.isfinite(value) for value in left + right):
        raise ValueError("Truth-ratio samples must be nonempty and finite")
    return max(abs(bisect_right(left, point) / len(left)
                   - bisect_right(right, point) / len(right))
               for point in set(left + right))


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class AuditedForgetRetainDataset(ForgetRetainDataset):
    """Observe one original paired-data draw without drawing any extra RNG."""

    def __init__(self, original, output, epoch_context):
        super().__init__(forget=original.forget, retain=original.retain, anchor=original.anchor)
        self.output, self.epoch_context = Path(output), epoch_context
        self.draw_index = 0

    def __getitem__(self, index):
        item = super().__getitem__(index)
        def source_index(split):
            value = item[split].get("index")
            if value is None:
                raise ValueError(f"Training-pair audit requires the {split} dataset index")
            return int(value)
        self.draw_index += 1
        append_history(self.output / "prepared_pairs.jsonl", {
            "draw_index": self.draw_index, "epoch_index": self.epoch_context.get("epoch_index", 0),
            "anchor_index": int(index), "anchor": self.anchor,
            "forget_index": source_index("forget"), "retain_index": source_index("retain")})
        return item


class PairAuditCollator:
    """Carry indices outside tensor inputs until a training forward consumes them."""

    def __init__(self, original):
        self.original = original

    def __call__(self, instances):
        batch = self.original(instances)
        if all("forget" in item and "retain" in item and "index" in item["forget"]
               and "index" in item["retain"] for item in instances):
            batch["_palu_pair_indices"] = [{"forget_index": int(item["forget"]["index"]),
                                             "retain_index": int(item["retain"]["index"])}
                                            for item in instances]
        return batch


class EpochBoundaryCallback(TrainerCallback):
    """Count actual dataloader traversals instead of rounding HF state.epoch."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        context = self.trainer._epoch_context
        context["epoch_index"] += 1
        context["microbatches"] = 0
        context["expected_microbatches"] = len(kwargs["train_dataloader"])
        context["epoch_end_seen"] = False

    def on_epoch_end(self, args, state, control, **kwargs):
        self.trainer._epoch_context["epoch_end_seen"] = True


class PALUEpochTrainer(PALU):
    """Single GPU PALU with complete epoch metrics and canonical fixed losses."""

    def __init__(self, epoch_options=None, *args, **kwargs):
        kwargs["track_loss_components"] = True
        # Epoch fixed-loss observations supersede the old representation cache.
        kwargs.pop("diagnostics", None)
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError("Epoch TOFU evaluation requires exactly one process/GPU")
        if self.retain_loss_type != "NLL" or self.non_sensitive_constraint_type != "kl":
            raise ValueError("This protocol requires Retain NLL and common-token KL")
        self.epoch_options = dict(epoch_options or {})
        self.epoch_output = Path(self.args.output_dir)
        self.epoch_output.mkdir(parents=True, exist_ok=True)
        for name in ("epoch_history.jsonl", "fixed_loss_samples.jsonl"):
            if (self.epoch_output / name).exists():
                raise ValueError(f"Existing {name}; use a fresh output directory")
        if not self.evaluators or len(self.evaluators) != 1:
            raise ValueError("Configure exactly one TOFU evaluator")
        self._epoch_loss_forget_only = False
        self._last_epoch_evaluated_step = None
        self._epoch_fixed = None
        self._epoch_fixed_manifest = None
        self._epoch_reference_sha256 = None
        self._epoch_context = {"epoch_index": 0, "microbatches": 0,
                               "expected_microbatches": 0, "epoch_end_seen": False}
        if self.args.dataloader_num_workers != 0:
            raise ValueError("Training-pair audit requires dataloader_num_workers=0")
        if not isinstance(self.train_dataset, ForgetRetainDataset):
            raise ValueError("Expected ForgetRetainDataset for the pair audit")
        for name in ("training_pairs.jsonl", "prepared_pairs.jsonl"):
            if (self.epoch_output / name).exists():
                raise ValueError(f"Existing {name}; use a fresh output directory")
        self.train_dataset = AuditedForgetRetainDataset(self.train_dataset, self.epoch_output,
                                                       self._epoch_context)
        self.data_collator = PairAuditCollator(self.data_collator)
        self.add_callback(EpochBoundaryCallback(self))

    def compute_loss(self, model, inputs, return_outputs=False):
        pairs = inputs.pop("_palu_pair_indices", None)
        result = super().compute_loss(model, inputs, return_outputs=return_outputs)
        if model.training and not getattr(self, "_palu_diagnostic_mode", False):
            if pairs is None:
                raise RuntimeError("A training forward is missing its pairing audit metadata")
            self._epoch_context["microbatches"] += 1
            for batch_index, pair in enumerate(pairs):
                append_history(self.epoch_output / "training_pairs.jsonl", {
                    "epoch_index": self._epoch_context["epoch_index"],
                    "microbatch_in_epoch": self._epoch_context["microbatches"],
                    "optimizer_step_before_update": int(self.state.global_step),
                    "example_in_microbatch": batch_index, **pair})
        return result

    def compute_retain_loss(self, model, retain_inputs):
        # Reuse the actual PALU sensitive/common objective without evaluating a
        # duplicate Retain example in each fixed Forget pass. This flag is only
        # set inside evaluate(), under no_grad, and restored in a finally block.
        if self._epoch_loss_forget_only:
            return torch.zeros((), device=retain_inputs["input_ids"].device)
        return super().compute_retain_loss(model, retain_inputs)

    def _prepare_fixed_data(self):
        if self._epoch_fixed is not None:
            return
        if not hasattr(self.train_dataset, "forget") or not hasattr(self.train_dataset, "retain"):
            raise ValueError("Fixed diagnostics require ForgetRetainDataset")
        fixed, metadata, digest = {}, {}, hashlib.sha256()
        for split in ("forget", "retain"):
            dataset = getattr(self.train_dataset, split)
            indices = fixed_indices(len(dataset), self.epoch_options.get(split + "_samples", "all"))
            fixed[split] = []
            for index in indices:
                # Access each actual split directly: never trigger random paired
                # sampling through ForgetRetainDataset.__getitem__.
                source = dataset[index]
                sample = {key: source[key].detach().cpu().clone()
                          for key in ("input_ids", "labels")}
                digest.update(json.dumps([split, index], separators=(",", ":")).encode())
                for key in ("input_ids", "labels"):
                    digest.update(key.encode())
                    digest.update(sample[key].to(torch.int64).numpy().tobytes())
                fixed[split].append((index, sample))
            metadata[split] = {"dataset_size": len(dataset), "count": len(indices), "indices": indices}
        self._epoch_fixed = fixed
        self._epoch_fixed_manifest = {
            "fixed_dataset_sha256": digest.hexdigest(), "fixed_indices": metadata,
            "fixed_loss_batch_size": 1,
            "fixed_loss_reduction": "raw_* = target-token-weighted mean over fixed split; sample_mean_raw_* = arithmetic mean of per-example PALU losses",
            "online_loss_reduction": "train-mode pre-update mean of microbatch losses across gradient accumulation; see loss_history.jsonl",
        }
        manifest_path = self.epoch_output / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest.update(self._epoch_fixed_manifest)
        manifest.update({"schema": 1, "seed": self.args.seed, "world_size": 1,
                         "model": self.epoch_options.get("model_name", self.model.config.model_type)})
        _atomic_json(manifest_path, manifest)

    def _fixed_losses(self, step, epoch):
        self._prepare_fixed_data()
        model = self.model_wrapped
        eos, first_n = self.tokenizer.eos_token_id, self.num_sensitive_tokens
        sums = {key: 0.0 for key in ("sensitive", "common", "retain")}
        macro = dict(sums)
        counts = {key: 0 for key in sums}
        zero_sensitive = zero_common = 0
        dummy_retain = self._epoch_fixed["retain"][0][1]
        self._epoch_loss_forget_only = True
        try:
            for index, sample in self._epoch_fixed["forget"]:
                batch = self._prepare_inputs(self.data_collator([{"forget": sample, "retain": dummy_retain}]))
                with self.compute_loss_context_manager():
                    self.compute_loss(model, batch)
                values = self._last_palu_terms.double().cpu().tolist()
                sensitive_count, common_count, _ = target_counts(
                    sample["labels"].tolist(), dummy_retain["labels"].tolist(), first_n, eos)
                zero_sensitive += sensitive_count == 0
                zero_common += common_count == 0
                for key, value, count in (("sensitive", values[4], sensitive_count),
                                          ("common", values[6], common_count)):
                    sums[key] += value * count
                    macro[key] += value
                    counts[key] += count
                append_history(self.epoch_output / "fixed_loss_samples.jsonl", {
                    "step": step, "epoch": epoch, "split": "forget", "dataset_index": index,
                    "raw_sensitive_loss": values[4], "raw_common_kl": values[6],
                    "raw_retain_nll": None, "sensitive_token_count": sensitive_count,
                    "common_token_count": common_count, "retain_token_count": None})
        finally:
            self._epoch_loss_forget_only = False
        for index, sample in self._epoch_fixed["retain"]:
            batch = self._prepare_inputs(self.data_collator([sample]))
            batch["use_cache"] = False
            with self.compute_loss_context_manager():
                loss = super().compute_retain_loss(model, batch)
            value = float(loss.double())
            count = sum(label != -100 for label in sample["labels"].tolist()[1:])
            if count == 0 or not math.isfinite(value):
                raise ValueError(f"Retain diagnostic example {index} has no targets or a nonfinite loss")
            sums["retain"] += value * count
            macro["retain"] += value
            counts["retain"] += count
            append_history(self.epoch_output / "fixed_loss_samples.jsonl", {
                "step": step, "epoch": epoch, "split": "retain", "dataset_index": index,
                "raw_sensitive_loss": None, "raw_common_kl": None, "raw_retain_nll": value,
                "sensitive_token_count": None, "common_token_count": None, "retain_token_count": count})
        f_count, r_count = (len(self._epoch_fixed[split]) for split in ("forget", "retain"))
        token_values = {key: sums[key] / max(counts[key], 1) for key in sums}
        macro_values = {key: macro[key] / (r_count if key == "retain" else f_count) for key in sums}

        def seven(values):
            sensitive, retain, common = values["sensitive"], values["retain"], values["common"]
            weighted = [self.gamma * sensitive, self.alpha * retain, self.alpha * common]
            return dict(zip(LOSS_KEYS, [sum(weighted), *weighted, sensitive, retain, common]))

        return {**seven(token_values), **{"sample_mean_" + key: value for key, value in seven(macro_values).items()},
                "forget_count": f_count, "retain_count": r_count,
                "sensitive_token_count": counts["sensitive"], "common_token_count": counts["common"],
                "retain_token_count": counts["retain"], "forget_zero_sensitive_examples": zero_sensitive,
                "forget_zero_common_examples": zero_common,
                "fixed_dataset_sha256": self._epoch_fixed_manifest["fixed_dataset_sha256"]}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", trial=None):
        # HF calls this at eval_on_start and after each on_epoch_end, including
        # its final partial epoch when the historical 60-update budget ends at 9.6.
        step, epoch = int(self.state.global_step), float(self.state.epoch or 0.0)
        if step == self._last_epoch_evaluated_step:
            return {}
        output = self.epoch_output / f"checkpoint-{step}" / "evals"
        output.mkdir(parents=True, exist_ok=True)
        device = self.accelerator.device
        model = self.model_wrapped
        started = time.perf_counter()
        previous_diagnostic = getattr(self, "_palu_diagnostic_mode", False)
        try:
            self._palu_diagnostic_mode = True
            with preserve_rng_and_modes([model, self.ref_model], device), torch.no_grad():
                self._prepare_fixed_data()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                fixed_started = time.perf_counter()
                losses = self._fixed_losses(step, epoch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                fixed_seconds = time.perf_counter() - fixed_started
                metrics = {}
                for evaluator in self.evaluators.values():
                    metrics.update(evaluator.evaluate(model=model, output_dir=str(output),
                        overwrite=True, template_args=self.template_args, tokenizer=self.tokenizer))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
        finally:
            self._palu_diagnostic_mode = previous_diagnostic
            self._epoch_loss_forget_only = False
        for key in ("forget_quality", "model_utility"):
            if not isinstance(metrics.get(key), (int, float)) or not math.isfinite(metrics[key]):
                raise ValueError(f"Epoch evaluation is missing a finite {key}")
        evaluator = next(iter(self.evaluators.values()))
        fq_cfg = evaluator.eval_cfg.metrics["forget_quality"]
        reference_path = Path(str(fq_cfg.reference_logs.retain_model_logs.path))
        reference_sha256 = sha256_file(reference_path)
        if self._epoch_reference_sha256 is not None and reference_sha256 != self._epoch_reference_sha256:
            raise ValueError("Retain reference logs changed between epoch evaluations")
        self._epoch_reference_sha256 = reference_sha256
        current = json.loads((output / "TOFU_EVAL.json").read_text())
        reference = json.loads(reference_path.read_text())
        scores = lambda payload: [float(value["score"]) for value in payload["forget_truth_ratio"]["value_by_index"].values()]
        current_scores, reference_scores = scores(current), scores(reference)
        d = ks_statistic(current_scores, reference_scores)
        context = self._epoch_context
        partial = step > 0 and context["microbatches"] < context["expected_microbatches"]
        pair_path = self.epoch_output / "training_pairs.jsonl"
        pairs = [json.loads(line) for line in pair_path.read_text().splitlines()] if pair_path.exists() else []
        current_pairs = [pair for pair in pairs if pair["epoch_index"] == context["epoch_index"]]
        row = {"schema": 1, "seed": self.args.seed,
               "model": self.epoch_options.get("model_name", self.model.config.model_type),
               "step": step, "global_step": step, "epoch": epoch,
               "state_epoch": epoch, "epoch_index": context["epoch_index"],
               "actual_epoch_microbatches": context["microbatches"],
               "expected_epoch_microbatches": context["expected_microbatches"],
               "training_examples_this_epoch": len(current_pairs),
               "unique_forget_examples_this_epoch": len({pair["forget_index"] for pair in current_pairs}),
               "unique_retain_examples_this_epoch": len({pair["retain_index"] for pair in current_pairs}),
               "unique_retain_examples_cumulative": len({pair["retain_index"] for pair in pairs}),
               "evaluation_kind": "initial" if step == 0 else "final_tail" if partial else "epoch_end",
               "partial_epoch": partial, **losses,
               "forget_quality": metrics["forget_quality"], "model_utility": metrics["model_utility"],
               "ks_d": d, "forget_ks_statistic": d,
               "forget_truth_ratio_count": len(current_scores),
               "reference_truth_ratio_count": len(reference_scores),
               "fixed_loss_wall_seconds": fixed_seconds,
               "evaluation_wall_seconds": time.perf_counter() - started,
               "evaluation_dir": str(output), "reference_sha256": reference_sha256}
        append_history(self.epoch_output / "epoch_history.jsonl", row)
        _atomic_json(output / "epoch_provenance.json", {**row, "metrics": metrics,
                     "evaluation_model": "current in-memory training model; no reload",
                     "rng_and_module_modes_restored": True})
        self._last_epoch_evaluated_step = step
        self.log(add_neg_log_pvalue_metrics({**metrics, "forget_ks_statistic": d}))
        return metrics
