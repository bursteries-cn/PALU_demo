from __future__ import annotations

import importlib.metadata
import json
import logging
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from representation_batching.manifest import (
    load_batch_manifest,
    qa_content_sha256,
    sha256_file,
    split_step_for_rank,
)
from representation_batching.runtime import (
    ManifestBatchSampler,
    ManifestDataLoader,
    ScheduledForgetRetainDataset,
)
from trainer.unlearn.npo import NPO
from trainer.utils import compute_dpo_loss


logger = logging.getLogger(__name__)


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_output(project_root, *args):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


class RepresentationNPO(NPO):
    """NPO whose effective forget batches come from an offline manifest.

    The dataloader is intentionally not passed to ``accelerator.prepare``: the
    manifest has already been split by global rank, and a second Accelerate
    sharding pass would change its optimizer-step composition. Trainer moves
    every input tensor to the process device in ``_prepare_inputs``.
    """

    def __init__(
        self,
        batch_manifest_path,
        npo_near_zero_threshold=0.1,
        verify_feature_hash=True,
        *args,
        **kwargs,
    ):
        self.batch_manifest_path = str(Path(batch_manifest_path).expanduser().resolve())
        self.batch_manifest = load_batch_manifest(self.batch_manifest_path)
        self.npo_near_zero_threshold = float(npo_near_zero_threshold)
        self.verify_feature_hash = bool(verify_feature_hash)
        super().__init__(*args, **kwargs)
        self._validate_deepspeed_gradient_clipping()
        self._validate_manifest_against_runtime()
        self._archive_manifest()
        self._expected_rank_microbatches = self._build_expected_rank_microbatches()
        self._observed_microbatch_count = 0
        self._diagnostic_buffer = {}

    def _deepspeed_gradient_clipping(self):
        if not self.is_deepspeed_enabled:
            return None
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        config = getattr(plugin, "deepspeed_config", None)
        if not isinstance(config, dict):
            return None
        return config.get("gradient_clipping")

    def _validate_deepspeed_gradient_clipping(self):
        """Ensure the Trainer clipping contract is active under DeepSpeed.

        Accelerate delegates clipping to DeepSpeed and does not call
        ``torch.nn.utils.clip_grad_norm_`` in this mode. DeepSpeed defaults a
        missing ``gradient_clipping`` entry to 0, so omitting the key silently
        disables ``TrainingArguments.max_grad_norm``.
        """
        if not self.is_deepspeed_enabled or float(self.args.max_grad_norm) <= 0:
            return
        expected = float(self.args.max_grad_norm)
        configured = self._deepspeed_gradient_clipping()
        try:
            configured = float(configured)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "DeepSpeed gradient_clipping must resolve to "
                f"TrainingArguments.max_grad_norm={expected}; got {configured!r}. "
                "Set gradient_clipping to 'auto' in the DeepSpeed config."
            ) from exc
        if not math.isclose(configured, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"DeepSpeed gradient_clipping={configured} does not match "
                f"TrainingArguments.max_grad_norm={expected}. Set the DeepSpeed "
                "value to 'auto' so Transformers resolves one clipping threshold."
            )

    def _validate_manifest_against_runtime(self):
        metadata = self.batch_manifest.metadata
        expected = {
            "world_size": int(self.args.world_size),
            "per_device_batch_size": int(self.args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(self.args.gradient_accumulation_steps),
        }
        for key, actual in expected.items():
            configured = int(metadata[key])
            if actual != configured:
                raise ValueError(
                    f"Runtime {key}={actual} does not match manifest {configured}. "
                    "Regenerate the manifest instead of allowing Trainer to reshape batches."
                )

        epochs = float(self.args.num_train_epochs)
        if not epochs.is_integer() or int(epochs) != int(metadata["num_epochs"]):
            raise ValueError(
                f"Runtime num_train_epochs={epochs} does not match manifest "
                f"{metadata['num_epochs']}"
            )
        if int(self.args.seed) != int(metadata["seed"]):
            raise ValueError(
                f"Runtime seed={self.args.seed} does not match manifest seed={metadata['seed']}"
            )
        train_dataset = self.train_dataset
        if not hasattr(train_dataset, "forget") or not hasattr(train_dataset, "retain"):
            raise TypeError("RepresentationNPO requires the standard ForgetRetainDataset")
        expected_forget = list(range(len(train_dataset.forget)))
        manifest_forget = sorted(int(x) for x in metadata["sample_ids"])
        if manifest_forget != expected_forget:
            raise ValueError(
                "Manifest sample_ids must exactly match the current forget dataset row indices"
            )
        if int(metadata["retain_size"]) != len(train_dataset.retain):
            raise ValueError(
                f"Manifest retain_size={metadata['retain_size']} but dataset has "
                f"{len(train_dataset.retain)} rows"
            )
        expected_content_digest = metadata.get("source_qa_content_sha256")
        if expected_content_digest:
            actual_content_digest = qa_content_sha256(
                train_dataset.forget.data,
                train_dataset.forget.question_key,
                train_dataset.forget.answer_key,
            )
        else:
            actual_content_digest = None
        if expected_content_digest and actual_content_digest != expected_content_digest:
            raise ValueError(
                "Ordered forget QA content differs from the dataset used to extract features: "
                f"{actual_content_digest} != {expected_content_digest}"
            )

        feature_path = Path(str(metadata.get("feature_file", "")))
        feature_digest = metadata.get("feature_sha256")
        if self.verify_feature_hash and feature_path.is_file() and feature_digest:
            actual_digest = sha256_file(feature_path)
            if actual_digest != feature_digest:
                raise ValueError(
                    f"Feature file hash mismatch for {feature_path}: {actual_digest} != {feature_digest}"
                )
        elif self.verify_feature_hash and feature_digest:
            logger.warning(
                "Feature file %s is not present on this host; the manifest remains usable, "
                "but its source feature hash could not be rechecked.",
                feature_path,
            )

    def _archive_manifest(self):
        if int(self.args.process_index) != 0:
            return
        audit_dir = Path(self.args.output_dir) / "batch_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            self.batch_manifest_path,
            audit_dir / Path(self.batch_manifest_path).name,
        )
        source_feature_manifest = Path(
            str(self.batch_manifest.metadata.get("source_feature_manifest", ""))
        )
        if source_feature_manifest.is_file():
            shutil.copy2(source_feature_manifest, audit_dir / "feature_manifest.json")
        runtime = {
            "batch_manifest_path": self.batch_manifest_path,
            "batch_manifest_sha256": sha256_file(self.batch_manifest_path),
            "world_size": int(self.args.world_size),
            "per_device_batch_size": int(self.args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(self.args.gradient_accumulation_steps),
            "num_train_epochs": float(self.args.num_train_epochs),
            "max_grad_norm": float(self.args.max_grad_norm),
            "deepspeed_gradient_clipping": self._deepspeed_gradient_clipping(),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch_cuda": torch.version.cuda,
                "cuda_device_count": torch.cuda.device_count(),
                "packages": {
                    name: _package_version(name)
                    for name in ("torch", "transformers", "accelerate", "deepspeed")
                },
            },
        }
        project_root = Path(__file__).resolve().parents[3]
        runtime["git"] = {
            "head": (_git_output(project_root, "rev-parse", "HEAD") or "").strip()
            or None,
            "status": (_git_output(project_root, "status", "--short") or "").splitlines(),
        }
        (audit_dir / "runtime_contract.json").write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        code_diff = _git_output(project_root, "diff", "--binary", "HEAD")
        if code_diff is not None:
            (audit_dir / "tracked_code_diff.patch").write_text(
                code_diff,
                encoding="utf-8",
            )
        snapshot_files = (
            ".gitignore",
            "requirements.txt",
            "src/train.py",
            "src/trainer/__init__.py",
            "src/trainer/utils.py",
            "src/trainer/unlearn/representation_npo.py",
            "src/representation_batching/__init__.py",
            "src/representation_batching/manifest.py",
            "src/representation_batching/runtime.py",
            "configs/trainer/RepresentationNPO.yaml",
            "configs/experiment/unlearn/tofu/representation_npo.yaml",
            "configs/model/Llama-3.1-8B-Instruct.yaml",
            "configs/accelerate/default_config.yaml",
            "configs/accelerate/zero_stage3_offload_config.json",
            "scripts/representation_batching/extract_tofu_features.py",
            "scripts/representation_batching/build_batch_manifests.py",
            "scripts/representation_batching/run_npo_representation.sh",
        )
        snapshot_root = audit_dir / "code_snapshot"
        for relative in snapshot_files:
            source = project_root / relative
            if source.is_file():
                destination = snapshot_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

    def _build_expected_rank_microbatches(self):
        expected = []
        for epoch, steps in sorted(self.batch_manifest.epochs.items()):
            for step in steps:
                microbatches = split_step_for_rank(
                    step,
                    rank=int(self.args.process_index),
                    world_size=int(self.args.world_size),
                    per_device_batch_size=int(self.args.per_device_train_batch_size),
                )
                for microstep, pairs in enumerate(microbatches):
                    expected.append(
                        {
                            "epoch": epoch,
                            "optimizer_step": int(step["optimizer_step"]),
                            "microstep": microstep,
                            "pairs": [(int(forget), int(retain)) for forget, retain in pairs],
                        }
                    )
        return tuple(expected)

    def _write_rank_schedule(self):
        audit_dir = Path(self.args.output_dir) / "batch_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        output = audit_dir / f"rank-{int(self.args.process_index)}-planned-microbatches.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for record in self._expected_rank_microbatches:
                serializable = dict(record)
                serializable["pairs"] = [
                    {"forget_index": forget, "retain_index": retain}
                    for forget, retain in record["pairs"]
                ]
                handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")
        observed = audit_dir / f"rank-{int(self.args.process_index)}-observed-microbatches.jsonl"
        observed.write_text("", encoding="utf-8")

    def _audit_observed_batch(self, forget_batch, retain_batch):
        if "index" not in forget_batch or "index" not in retain_batch:
            raise KeyError(
                "RepresentationNPO requires a collator that preserves the dataset index"
            )
        position = self._observed_microbatch_count
        if position >= len(self._expected_rank_microbatches):
            raise RuntimeError("Trainer consumed more microbatches than the manifest defines")
        expected = self._expected_rank_microbatches[position]
        forget_ids = [int(x) for x in forget_batch["index"].detach().cpu().tolist()]
        retain_ids = [int(x) for x in retain_batch["index"].detach().cpu().tolist()]
        observed_pairs = list(zip(forget_ids, retain_ids))
        if observed_pairs != expected["pairs"]:
            raise RuntimeError(
                "Observed forget/retain rows do not match the manifest at "
                f"epoch={expected['epoch']} optimizer_step={expected['optimizer_step']} "
                f"microstep={expected['microstep']}: {observed_pairs} != {expected['pairs']}"
            )
        audit_path = (
            Path(self.args.output_dir)
            / "batch_audit"
            / f"rank-{int(self.args.process_index)}-observed-microbatches.jsonl"
        )
        record = {
            **{key: expected[key] for key in ("epoch", "optimizer_step", "microstep")},
            "pairs": [
                {"forget_index": forget, "retain_index": retain}
                for forget, retain in observed_pairs
            ],
        }
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._observed_microbatch_count += 1
        return expected

    def _buffer_effective_batch_diagnostics(self, expected, values):
        for key, value in values.items():
            self._diagnostic_buffer.setdefault(key, []).append(
                value.detach().float().reshape(-1)
            )
        if expected["microstep"] + 1 != int(self.args.gradient_accumulation_steps):
            return

        gathered = {}
        for key, chunks in self._diagnostic_buffer.items():
            local_values = torch.cat(chunks)
            gathered[key] = self.accelerator.gather(local_values)
        if self.is_world_process_zero():
            weights = gathered["npo_weight"]
            diagnostics = {
                    "manifest_epoch": expected["epoch"],
                    "manifest_optimizer_step": expected["optimizer_step"],
                    "forget_loss": gathered["forget_loss"].mean().item(),
                    "retain_loss": gathered["retain_loss"].mean().item(),
                    "npo_weight_mean": weights.mean().item(),
                    "npo_weight_min": weights.min().item(),
                    "npo_weight_max": weights.max().item(),
                    "npo_weight_near_zero_fraction": (
                        weights < self.npo_near_zero_threshold
                    ).float().mean().item(),
                    "npo_log_ratio_mean": gathered["npo_log_ratio"].mean().item(),
                    "forget_sequence_nll_mean": gathered[
                        "forget_sequence_nll"
                    ].mean().item(),
                    "forget_answer_tokens_mean": gathered[
                        "forget_answer_tokens"
                    ].mean().item(),
                    "forget_padding_tokens_mean": gathered[
                        "forget_padding_tokens"
                    ].mean().item(),
                }
            # Persist independently of W&B and save_model, including no-save smoke runs.
            # Losses describe the incoming update; this is not a completion marker.
            diagnostics["optimizer_step"] = int(self.state.global_step) + 1
            with (Path(self.args.output_dir) / "training_diagnostics.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(diagnostics, ensure_ascii=False) + "\n")
            self.log(diagnostics)
        self._diagnostic_buffer = {}

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset")
        dataset = ScheduledForgetRetainDataset(self.train_dataset)
        sampler = ManifestBatchSampler(
            self.batch_manifest,
            rank=int(self.args.process_index),
            world_size=int(self.args.world_size),
            per_device_batch_size=int(self.args.per_device_train_batch_size),
        )
        self._write_rank_schedule()
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_sampler": sampler,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if self.args.dataloader_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.args.dataloader_persistent_workers
            if self.args.dataloader_prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return ManifestDataLoader(**dataloader_kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        forget_batch = inputs["forget"]
        retain_batch = inputs["retain"]
        expected = self._audit_observed_batch(forget_batch, retain_batch)
        forget_inputs = {
            "input_ids": forget_batch["input_ids"],
            "attention_mask": forget_batch["attention_mask"],
            "labels": forget_batch["labels"],
        }
        forget_loss, forget_outputs, details = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
            return_details=True,
        )

        retain_inputs = {
            "input_ids": retain_batch["input_ids"],
            "attention_mask": retain_batch["attention_mask"],
            "labels": retain_batch["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        self._buffer_effective_batch_diagnostics(
            expected,
            {
                "forget_loss": forget_loss,
                "retain_loss": retain_loss,
                "npo_weight": details["npo_weight"],
                "npo_log_ratio": details["lose_log_ratio"],
                "forget_sequence_nll": details["lose_nll"],
                "forget_answer_tokens": (
                    forget_batch["labels"][..., 1:] != -100
                ).sum(dim=-1),
                "forget_padding_tokens": (
                    forget_batch["attention_mask"] == 0
                ).sum(dim=-1),
            },
        )
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss

    def train(self, *args, **kwargs):
        def write_status(status, error=None):
            if self.is_world_process_zero():
                payload = {
                    "status": status,
                    "global_step": int(self.state.global_step),
                    "max_steps": int(self.args.max_steps),
                    "observed_microbatches_per_rank": self._observed_microbatch_count,
                    "expected_microbatches_per_rank": len(self._expected_rank_microbatches),
                    "error": error,
                }
                path = Path(self.args.output_dir) / "training_status.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                temporary.replace(path)

        if self.is_world_process_zero():
            (Path(self.args.output_dir) / "training_diagnostics.jsonl").write_text("", encoding="utf-8")
        write_status("started")
        try:
            output = super().train(*args, **kwargs)
        except BaseException as exc:
            write_status("failed", f"{type(exc).__name__}: {exc}")
            raise
        if self.args.max_steps <= 0 and (
            self._observed_microbatch_count != len(self._expected_rank_microbatches)
        ):
            write_status("failed", "Incomplete manifest consumption")
            raise RuntimeError(
                f"Trainer consumed {self._observed_microbatch_count} microbatches, but the "
                f"manifest defines {len(self._expected_rank_microbatches)} for this rank"
            )
        write_status("limited" if self.args.max_steps > 0 else "completed")
        return output
