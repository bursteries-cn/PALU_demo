"""Commit complete epoch observations before removing owned intermediate weights.

This module uses only the standard library. Receipts allow interrupted evaluation
to resume from committed observations even after their model weights are gone.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import time

WEIGHT_FILE = re.compile(r"(?:model(?:-\d{5}-of-\d{5})?\.safetensors|model\.safetensors\.index\.json)\Z")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_directory(path):
    import os
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_text(path, text):
    import os
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def owned_checkpoint(run, step):
    run = Path(run).resolve()
    checkpoint = run / f"checkpoint-{step}"
    if checkpoint.is_symlink() or checkpoint.resolve() != checkpoint:
        raise ValueError("Checkpoint must be a real directory directly under this run")
    return checkpoint


def observation_scope(epoch, manifest):
    """Resolve omissions from the run's frozen policy, never from row claims alone."""
    policy = manifest.get("threshold_plan", {}).get("evaluation_policy")
    full_metrics = ["forget_quality", "model_utility"]
    if policy != "forget_light_v1":
        if epoch.get("fixed_loss_scope", "full") != "full" or epoch.get("evaluation_metrics", full_metrics) != full_metrics:
            raise ValueError("Reduced observation scope is not authorized by the run plan")
        return full_metrics, False
    final = manifest.get("final_model_step")
    step = epoch["step"]
    if type(final) is not int or final <= 0 or type(step) is not int or not 0 <= step <= final:
        raise ValueError("Invalid terminal step for the scoped observation")
    metrics = full_metrics if step == final else ["forget_quality"]
    if epoch.get("evaluation_metrics") != metrics:
        raise ValueError("Observation metrics do not match the planned terminal/intermediate scope")
    if epoch.get("fixed_loss_scope") != "forget_only":
        raise ValueError("Lightweight Forget observations require fixed_loss_scope=forget_only")
    if epoch.get("raw_retain_nll", "missing") is not None or any(
            type(epoch.get(key)) is not int or epoch[key] != 0 for key in ("retain_count", "retain_token_count")):
        raise ValueError("Unmeasured Retain loss must be null with zero sample and token counts")
    if epoch.get("forget_count") != manifest["schedule_metadata"]["data_sizes"]["forget"]:
        raise ValueError("Lightweight observations must cover the complete fixed Forget set")
    if "model_utility" not in metrics and epoch.get("model_utility", "missing") is not None:
        raise ValueError("Unmeasured MU must be explicitly null")
    return metrics, True


def validate_fixed_samples(rows, epoch, manifest):
    _, forget_only = observation_scope(epoch, manifest)
    expected_sizes = dict(manifest["schedule_metadata"]["data_sizes"])
    expected_counts = dict(manifest["schedule_metadata"]["fixed_target_counts"])
    if forget_only:
        expected_sizes["retain"] = expected_counts["retain"] = 0
    observed = {split: set() for split in ("forget", "retain")}
    sums = {key: 0.0 for key in ("sensitive", "common", "retain")}
    counts = {key: 0 for key in sums}
    for sample in rows:
        split, index = sample["split"], sample["dataset_index"]
        if split not in observed or index in observed[split] or sample["step"] != epoch["step"]:
            raise ValueError("Fixed per-example records have a wrong step, split, or duplicate index")
        if not 0 <= index < expected_sizes[split]:
            raise ValueError("Fixed per-example dataset index is out of range")
        observed[split].add(index)
        components = (("retain", "raw_retain_nll", "retain_token_count"),) if split == "retain" else (
            ("sensitive", "raw_sensitive_loss", "sensitive_token_count"),
            ("common", "raw_common_kl", "common_token_count"))
        for key, value_key, count_key in components:
            value, count = sample[value_key], sample[count_key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or type(count) is not int or count < 0:
                raise ValueError("Nonfinite per-example loss or invalid target count")
            sums[key] += value * count
            counts[key] += count
    if any(len(observed[split]) != expected_sizes[split] for split in observed):
        raise ValueError("Incomplete fixed per-example evaluation; weights must be retained")
    if counts != expected_counts:
        raise ValueError("Per-example target counts differ from the frozen dataset")
    for key, field in (("sensitive", "raw_sensitive_loss"), ("common", "raw_common_kl"), ("retain", "raw_retain_nll")):
        if forget_only and key == "retain":
            continue
        if forget_only and epoch.get(key + "_token_count") != counts[key]:
            raise ValueError("Reported target counts differ from the fixed per-example records")
        if not math.isclose(sums[key] / max(counts[key], 1), epoch[field], rel_tol=1e-8, abs_tol=1e-9):
            raise ValueError("Per-example losses do not reconstruct the reported epoch loss")


def validate_observation(evals, epoch, manifest):
    evals = Path(evals)
    if evals.is_symlink():
        raise ValueError("Evaluation directory cannot be a symlink")
    if epoch["fixed_dataset_sha256"] != manifest["schedule_metadata"]["fixed_dataset_sha256"]:
        raise ValueError("Epoch used a different fixed dataset")
    expected_reference = manifest["schedule_metadata"].get("seed103_source_provenance", {}).get("reference_sha256")
    if expected_reference and epoch["reference_sha256"] != expected_reference:
        raise ValueError("Epoch used a different FQ reference")
    metrics, _ = observation_scope(epoch, manifest)
    native = json.loads((evals / "TOFU_EVAL.json").read_text())
    for key in metrics + ["ks_d"]:
        if isinstance(epoch.get(key), bool) or not isinstance(epoch.get(key), (int, float)) or not math.isfinite(epoch[key]):
            raise ValueError("Missing or nonfinite FQ/MU/KS D")
    for key in metrics:
        if key not in native:
            raise ValueError(f"Missing required native metric: {key}")
        value = native[key]
        value = value["agg_value"] if isinstance(value, dict) else value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing or nonfinite native metric: {key}")
        if not math.isclose(value, epoch[key], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("Native TOFU metrics differ from the epoch record")
    if len(native["forget_truth_ratio"]["value_by_index"]) != manifest["schedule_metadata"]["data_sizes"]["forget"]:
        raise ValueError("Missing per-example Forget truth ratios")
    per_item = {"forget_Q_A_PARA_Prob": manifest["schedule_metadata"]["data_sizes"]["forget"],
                "forget_Q_A_PERT_Prob": manifest["schedule_metadata"]["data_sizes"]["forget"]}
    if "model_utility" in metrics:
        per_item.update({key: size for prefix, size, keys in (
            ("retain", 400, ("Q_A_Prob", "Q_A_PARA_Prob", "Q_A_PERT_Prob", "Q_A_ROUGE", "Truth_Ratio")),
            ("ra", 100, ("Q_A_Prob", "Q_A_PERT_Prob", "Q_A_Prob_normalised", "Q_A_ROUGE", "Truth_Ratio")),
            ("wf", 117, ("Q_A_Prob", "Q_A_PERT_Prob", "Q_A_Prob_normalised", "Q_A_ROUGE", "Truth_Ratio")))
            for key in (prefix + "_" + suffix for suffix in keys)})
    for key, size in per_item.items():
        if key not in native or len(native[key].get("value_by_index") or {}) != size:
            raise ValueError(f"Incomplete native per-example evaluation: {key}")
    rows = [json.loads(line) for line in (evals / "fixed_loss_samples.jsonl").read_text().splitlines() if line.strip()]
    validate_fixed_samples(rows, epoch, manifest)


def read_receipt(run, manifest, checkpoint):
    folder = owned_checkpoint(run, int(checkpoint["step"]))
    receipt_path = folder / "evaluation_receipt.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt["step"] != checkpoint["step"] or receipt["parameter_sha256"] != checkpoint["parameter_sha256"] or receipt["schedule_sha256"] != manifest["schedule_sha256"]:
        raise ValueError("Evaluation receipt belongs to different weights or input schedule")
    for name, inventory in receipt["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid artifact path in evaluation receipt")
        path = folder / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != inventory["bytes"] or sha(path) != inventory["sha256"]:
            raise ValueError(f"Committed evaluation artifact changed or is missing: {path}")
    epoch = receipt["epoch_record"]
    if epoch["step"] != checkpoint["step"] or epoch["checkpoint_parameter_sha256"] != checkpoint["parameter_sha256"] or epoch["training_schedule_sha256"] != manifest["schedule_sha256"]:
        raise ValueError("Committed epoch record belongs to a different state")
    validate_observation(folder / "evals", epoch, manifest)
    return epoch


def commit_observation(run, manifest, checkpoint, epoch, source_evals, fixed_source):
    import os
    run = Path(run).resolve()
    folder = owned_checkpoint(run, int(checkpoint["step"]))
    folder.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=".evals-", dir=folder))
    target = folder / "evals"
    shutil.copytree(source_evals, pending, dirs_exist_ok=True)
    lines = [line for line in Path(fixed_source).read_text().splitlines()
             if line.strip() and json.loads(line)["step"] == checkpoint["step"]]
    atomic_text(pending / "fixed_loss_samples.jsonl", "\n".join(lines) + "\n")
    epoch = dict(epoch, evaluation_dir=str(target))
    atomic_json(pending / "epoch_provenance.json", epoch)
    validate_observation(pending, epoch, manifest)
    if target.exists():
        # A prior interrupted attempt without a receipt is preserved for audit.
        preserved = Path(tempfile.mkdtemp(prefix=f"incomplete-checkpoint-{checkpoint['step']}-", dir=run / "evaluation_workspace"))
        target.replace(preserved / "evals")
    pending.replace(target)
    inventory = {}
    for path in sorted(target.rglob("*")):
        if path.is_file():
            if path.is_symlink():
                raise ValueError("Evaluation artifact cannot be a symlink")
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            inventory[str(path.relative_to(folder))] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    sync_directory(target)
    sync_directory(folder)
    atomic_json(folder / "evaluation_receipt.json", {"schema": 1, "status": "committed",
        "step": checkpoint["step"], "parameter_sha256": checkpoint["parameter_sha256"],
        "schedule_sha256": manifest["schedule_sha256"], "epoch_record": epoch,
        "files": inventory, "committed_unix": time.time()})
    return read_receipt(run, manifest, checkpoint)


def rebuild_aggregates(run, rows):
    run = Path(run)
    atomic_text(run / "epoch_history.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    # These source files are immutable, individually validated receipt artifacts.
    atomic_text(run / "fixed_loss_samples.jsonl", "".join(
        (Path(row["evaluation_dir"]) / "fixed_loss_samples.jsonl").read_text() for row in rows))


def prune_intermediate(run, manifest, checkpoint):
    step = int(checkpoint["step"])
    # Online training has not created the final checkpoint yet. Never mistake
    # its newest intermediate state for the final model.
    final = int(manifest.get("final_model_step", max(int(item["step"]) for item in manifest["checkpoints"])))
    if step == 0 or step == final:
        return {"step": step, "weights_retained": True, "reason": "initial Full or final model"}
    if not 0 < step < final:
        raise ValueError("Only an intermediate update can be pruned")
    if read_receipt(run, manifest, checkpoint) is None:
        raise ValueError("No validated evaluation receipt; weights must be retained")
    folder = owned_checkpoint(run, step)
    if Path(checkpoint["checkpoint"]).resolve() != folder:
        raise ValueError("Refusing to remove weights outside this run")
    names = [name for name in checkpoint["files"] if WEIGHT_FILE.fullmatch(name)]
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("No owned safetensors inventory; weights must be retained")
    for name in names:
        path = folder / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("Refusing a non-regular weight file")
    journal = folder / "weight_cleanup.json"
    record = {"schema": 1, "step": step, "status": "deleting", "files": names,
              "evaluation_receipt_sha256": sha(folder / "evaluation_receipt.json"),
              "planned_bytes": sum(checkpoint["files"][name]["bytes"] for name in names)}
    atomic_json(journal, record)
    for name in names:
        (folder / name).unlink(missing_ok=True)
    sync_directory(folder)
    record.update(status="deleted", weights_retained=False, completed_unix=time.time())
    atomic_json(journal, record)
    checkpoint.update(weights_retained=False, weights_deleted_after_verified_evaluation=True,
                      weight_cleanup_path=str(journal), deleted_weight_bytes=record["planned_bytes"])
    provenance = folder / "checkpoint_provenance.json"
    if provenance.is_file():
        value = json.loads(provenance.read_text())
        value.update(weights_retained=False, weight_cleanup_path=str(journal))
        atomic_json(provenance, value)
    return record
