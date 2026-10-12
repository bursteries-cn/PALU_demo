"""PALU scalar history and fixed-example diagnostics; never part of the objective."""
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import TrainerCallback


LOSS_KEYS = ("loss_total", "loss_forget", "loss_retain_nll", "loss_common_kl",
             "raw_sensitive_loss", "raw_retain_nll", "raw_common_kl")
PROBE_CACHE_SCHEMA = 2
PROBE_POSITIONS = {"forget": "first_sensitive", "retain": "first_answer"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer):
    payload = {"vocab": tokenizer.get_vocab(), "special_tokens": tokenizer.special_tokens_map,
               "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def probe_prediction_positions(labels, attention_mask, input_ids, eos_token_id=None, *, position):
    """Select h[t-1] for the first sensitive or answer token in each sequence."""
    if position not in ("first_sensitive", "first_answer"):
        raise ValueError(f"Unknown diagnostic token position: {position}")
    sensitive = position == "first_sensitive"
    positions = []
    for label, mask, ids in zip(labels, attention_mask, input_ids):
        valid = (label >= 0 if sensitive else label != -100) & mask.bool()
        if sensitive and eos_token_id is not None:
            valid &= label != eos_token_id  # Match PALU's sensitive mask; EOS is not a target.
        targets = torch.nonzero(valid, as_tuple=False).flatten()
        target_name = "sensitive" if sensitive else "answer"
        if not len(targets):
            raise ValueError(f"No {target_name} token remains after masking/truncation")
        start = int(targets[0])
        if start == 0 or not bool(mask[start - 1]):
            raise ValueError(f"First {target_name} token has no valid preceding context token")
        if eos_token_id is not None and int(ids[start]) == eos_token_id:
            raise ValueError(f"First {target_name} token is EOS")
        positions.append(start - 1)
    return torch.tensor(positions, device=labels.device, dtype=torch.long)


def representation_distances(reference, current, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Representation KL temperature must be finite and positive")
    if reference.shape != current.shape:
        raise ValueError("Reference/current hidden dimensions differ")
    ref_log = F.log_softmax(reference.float() / temperature, dim=-1)
    cur_log = F.log_softmax(current.float() / temperature, dim=-1)
    # Sum feature dimensions, then let the caller average independent examples.
    kl = (ref_log.exp() * (ref_log - cur_log)).sum(dim=-1)
    cosine = 1 - F.cosine_similarity(reference.float(), current.float(), dim=-1)
    return kl, cosine


@contextmanager
def preserve_rng_and_modes(models, device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    flags = [(module, module.training) for model in models for module in model.modules()]
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            for model in models:
                model.eval()
            yield
    finally:
        for module, training in flags:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def capture_pre_head(model, positions):
    """Capture only selected final-normalized hidden vectors via the actual LM head."""
    captured = []

    def hook(module, inputs):
        hidden = inputs[0]
        selected = positions[len(captured)].to(hidden.device)
        captured.append(hidden[torch.arange(hidden.shape[0], device=hidden.device), selected].detach().float())

    handle = model.get_output_embeddings().register_forward_pre_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def append_history(path, row):
    """Flush JSONL and CSV at each optimizer step/checkpoint, rank zero only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    csv_path = path.with_suffix(".csv")
    fresh = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if fresh:
            writer.writeheader()
        writer.writerow(row)


class PaluMonitor(TrainerCallback):
    def __init__(self, trainer, diagnostics=None):
        self.trainer = trainer
        self.options = dict(diagnostics or {})
        self.accumulated = None
        self.microbatches = 0
        self.last_probe_step = None
        self.probe_seconds = 0.0
        self.probe_seconds_after_runtime = 0.0
        self.cache = None
        self.output = Path(trainer.args.output_dir)
        self.every = int(self.options.get("every", 5))
        self.temperature = float(self.options.get("temperature", 1.0))
        if self.every <= 0 or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Diagnostic interval and temperature must be positive")
        if trainer.retain_loss_type != "NLL" or trainer.non_sensitive_constraint_type != "kl":
            raise ValueError("Four-curve history requires Retain NLL and common-token KL")

    def record(self, terms):
        terms = terms.detach().double()
        self.accumulated = terms.clone() if self.accumulated is None else self.accumulated + terms
        self.microbatches += 1

    def _reduce(self, values):
        return self.trainer.accelerator.reduce(values, reduction="sum")

    def _write(self, name, row):
        if self.trainer.accelerator.is_main_process:
            append_history(self.output / name, row)

    def on_train_begin(self, args, state, control, **kwargs):
        # Fresh run directories prevent histories from silently mixing executions.
        for name in ("loss_history.jsonl", "probe_history.jsonl", "probe_samples.jsonl"):
            if (self.output / name).exists():
                raise ValueError(f"Existing diagnostic history: {self.output / name}; use a new output directory")
        if self.options:
            path = Path(self.options["cache_path"])
            expected_hash = self.options.get("cache_sha256")
            if not expected_hash or sha256_file(path) != expected_hash:
                raise ValueError("Diagnostic reference cache hash is absent or has changed")
            self.cache = torch.load(path, map_location="cpu", weights_only=True)
            self._validate_cache()
            self._probe(state.global_step)

    def _validate_cache(self):
        metadata = self.cache["metadata"]
        model = self.trainer.accelerator.unwrap_model(self.trainer.model_wrapped)
        if metadata.get("schema") != PROBE_CACHE_SCHEMA or metadata.get("positions") != PROBE_POSITIONS:
            raise ValueError("Reference cache uses an outdated schema or token position; regenerate it in a new run")
        if metadata["tokenizer_sha256"] != tokenizer_fingerprint(self.trainer.tokenizer):
            raise ValueError("Diagnostic cache tokenizer differs from training")
        for key in ("model_type", "hidden_size", "vocab_size"):
            if metadata["architecture"][key] != getattr(model.config, key):
                raise ValueError(f"Diagnostic cache architecture differs: {key}")
        if getattr(model.config, "pretraining_tp", 1) != 1:
            raise ValueError("Pre-head capture requires pretraining_tp=1")
        for split in ("forget", "retain"):
            samples = self.cache["samples"][split]
            if not samples:
                raise ValueError(f"Empty diagnostic split: {split}")
            dataset = getattr(self.trainer.train_dataset, split)
            for cached in samples:
                actual = dataset[cached["dataset_index"]]
                actual_batch = self.trainer.data_collator([actual])
                for key in ("input_ids", "labels", "attention_mask"):
                    if not torch.equal(actual_batch[key][0].cpu(), cached[key]):
                        raise ValueError(f"Diagnostic cache input differs: {split}/{cached['dataset_index']}/{key}")
                positions = probe_prediction_positions(cached["labels"][None], cached["attention_mask"][None],
                    cached["input_ids"][None], self.trainer.tokenizer.eos_token_id, position=PROBE_POSITIONS[split])
                if int(positions[0]) != cached["prediction_index"]:
                    raise ValueError(f"Diagnostic cache {split} prediction position differs")
                if cached["reference_hidden"].shape != (model.config.hidden_size,):
                    raise ValueError("Invalid cached hidden representation")

    def on_step_end(self, args, state, control, **kwargs):
        if self.accumulated is None or not self.microbatches:
            raise RuntimeError("Optimizer step has no PALU component observations")
        values = self._reduce(torch.cat((self.accumulated, self.accumulated.new_tensor([self.microbatches]))))
        row = {"step": state.global_step, "epoch": float(state.epoch or 0),
               "microbatches_all_ranks": int(values[-1]),
               **dict(zip(LOSS_KEYS, (values[:-1] / values[-1]).cpu().tolist()))}
        self._write("loss_history.jsonl", row)
        self.trainer.log({key: row[key] for key in LOSS_KEYS[:4]})
        self.accumulated, self.microbatches = None, 0
        if self.cache is not None and (state.global_step % self.every == 0
                or state.global_step >= state.max_steps or control.should_training_stop):
            self._probe(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.cache is not None and self.last_probe_step != state.global_step:
            # HF computes train_runtime before on_train_end. Normally the final
            # step is handled above; label any unusual early-stop fallback explicitly.
            self._probe(state.global_step, included_in_runtime=False)

    def _probe(self, step, included_in_runtime=True):
        trainer, accelerator = self.trainer, self.trainer.accelerator
        device, world, rank = accelerator.device, accelerator.num_processes, accelerator.process_index
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model = trainer.model_wrapped  # Keep DeepSpeed/DDP collectives on the real forward path.
        unwrapped = accelerator.unwrap_model(model)
        samples = self.cache["samples"]
        counts = {split: len(samples[split]) for split in samples}
        rounds = math.ceil(max(counts.values()) / world)
        # F sums: sensitive, common KL, repr KL, cosine, count; R: NLL, repr KL, cosine, count.
        totals = torch.zeros(9, dtype=torch.float64, device=device)
        observations = []
        try:
            trainer._palu_diagnostic_mode = True
            with preserve_rng_and_modes([model, trainer.ref_model], device), torch.no_grad():
                for iteration in range(rounds):
                    index = iteration * world + rank
                    chosen = {split: samples[split][index % counts[split]] for split in samples}
                    # Every rank makes identical numbers/types of forwards. Repeated padding gets zero weight.
                    batch = trainer.data_collator([{split: {key: value[key] for key in ("input_ids", "labels")}
                                                    for split, value in chosen.items()}])
                    batch = trainer._prepare_inputs(batch)
                    positions = [probe_prediction_positions(batch[split]["labels"], batch[split]["attention_mask"],
                        batch[split]["input_ids"], trainer.tokenizer.eos_token_id, position=PROBE_POSITIONS[split])
                        for split in ("forget", "retain")]
                    with capture_pre_head(unwrapped, positions) as hidden, trainer.compute_loss_context_manager():
                        trainer.compute_loss(model, batch)
                    if len(hidden) != 2:
                        raise RuntimeError("Expected one current-model LM-head call per Forget/Retain forward")
                    terms = trainer._last_palu_terms.double()
                    for offset, split in enumerate(("forget", "retain")):
                        if index >= counts[split]:
                            continue
                        reference = chosen[split]["reference_hidden"][None].to(hidden[offset].device)
                        kl, cosine = representation_distances(reference, hidden[offset], self.temperature)
                        if split == "forget":
                            totals[:5] += torch.stack((terms[4], terms[6], kl[0].double(), cosine[0].double(), terms.new_ones(()))).to(device)
                        else:
                            totals[5:] += torch.stack((terms[5], kl[0].double(), cosine[0].double(), terms.new_ones(()))).to(device)
                        observations.append({"step": step, "split": split, "dataset_index": chosen[split]["dataset_index"],
                                             "position": PROBE_POSITIONS[split], "prediction_index": int(positions[offset][0]),
                                             "repr_kl": float(kl[0]), "cosine_distance": float(cosine[0])})
        finally:
            trainer._palu_diagnostic_mode = False
        totals = self._reduce(totals).cpu().tolist()
        f_count, r_count = int(totals[4]), int(totals[8])
        sensitive, common, retain = totals[0] / f_count, totals[1] / f_count, totals[5] / r_count
        weighted = [trainer.gamma * sensitive, trainer.alpha * retain, trainer.alpha * common]
        row = {"step": step, "forget_count": f_count, "retain_count": r_count,
               "forget_position": PROBE_POSITIONS["forget"], "retain_position": PROBE_POSITIONS["retain"],
               **dict(zip(LOSS_KEYS, [sum(weighted), *weighted, sensitive, retain, common])),
               "forget_repr_kl": totals[2] / f_count, "retain_repr_kl": totals[6] / r_count,
               "forget_cosine_distance": totals[3] / f_count, "retain_cosine_distance": totals[7] / r_count,
               "temperature": self.temperature, "included_in_trainer_runtime": included_in_runtime}
        # Gather variable-length records after all forward collectives have completed.
        if world > 1:
            import torch.distributed as distributed
            gathered = [None] * world
            distributed.all_gather_object(gathered, observations)
            observations = [entry for part in gathered for entry in part]
        for observation in observations:
            self._write("probe_samples.jsonl", observation)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = torch.tensor([time.perf_counter() - started], dtype=torch.float64, device=device)
        if world > 1:
            distributed.all_reduce(elapsed, op=distributed.ReduceOp.MAX)
        elapsed = float(elapsed)
        self.probe_seconds += elapsed
        if not included_in_runtime:
            self.probe_seconds_after_runtime += elapsed
        row["diagnostic_wall_seconds"] = elapsed
        self._write("probe_history.jsonl", row)
        if accelerator.is_main_process:
            timing = {"diagnostic_wall_seconds": self.probe_seconds,
                      "includes_training_runtime": self.probe_seconds_after_runtime == 0,
                      "diagnostic_seconds_after_trainer_runtime": self.probe_seconds_after_runtime,
                      "reference_precompute_included": False,
                      "measurement": "maximum per-rank wall time including sample gathering/writes; excludes final summary serialization"}
            (self.output / "diagnostic_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
            print(f"[diagnostic step={step}] Forget KL={row['forget_repr_kl']:.6g}; "
                  f"Retain KL={row['retain_repr_kl']:.6g}; seconds={elapsed:.2f}", flush=True)
        self.last_probe_step = step
