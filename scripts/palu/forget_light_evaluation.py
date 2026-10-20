"""Evaluation-only policy for Forget threshold runs; no optimizer changes."""
import json
import math


def sample_masked_diagnostic(result, manifest):
    """Unmasked fixed-pool losses plus the F mask effective after this probe.

    The denominator remains every sensitive token in F200. This diagnostic is
    not a forecast of a future minibatch loss or a claim that inactive samples
    will remain inside the stopping band.
    """
    from forget_checkpoint_state import sample_stop_state_fields
    fields = sample_stop_state_fields(manifest)
    if not fields:
        raise ValueError("Sample stopping evaluation requires its authenticated mask")
    state = fields["sample_stop_state"]
    rows = state.get("rows", [])
    samples = result["fixed_loss_samples"]
    indices = [row.get("dataset_index") for row in rows]
    expected = [row["dataset_index"] for row in samples]
    if len(indices) != len(set(indices)) or set(indices) != set(expected):
        raise ValueError("Sample stopping mask differs from the fixed Forget pool")
    by_index = {row["dataset_index"]: row for row in rows}
    numerator = 0.0
    for sample in samples:
        row = by_index[sample["dataset_index"]]
        count = sample["sensitive_token_count"]
        if (type(row.get("active")) is not bool
                or row.get("sensitive_token_count") != count
                or (not count and row["active"])):
            raise ValueError("Invalid sample stopping mask or sensitive target count")
        numerator += sample["raw_sensitive_loss"] * count * row["active"]
    denominator = result["sensitive_token_count"]
    if denominator <= 0 or sum(row["sensitive_token_count"] for row in samples) != denominator:
        raise ValueError("Sample stopping diagnostic requires the original F200 token denominator")
    masked = numerator / denominator
    recorded = result.get("sample_masked_sensitive_loss_next_update")
    if recorded is not None and not math.isclose(recorded, masked, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Recorded sample-masked loss does not reconstruct from its mask")
    return dict(
        stopping_scope="sample_permanent",
        sample_stop_state_sha256=fields["sample_stop_state_sha256"],
        loss_total_semantics="unmasked fixed F200 diagnostic F + 2common_KL; not the masked training objective",
        sample_masked_sensitive_loss_next_update=masked,
        sample_masked_loss_total_next_update=masked + 2 * result["raw_common_kl"],
        sample_masked_loss_semantics="fixed current-model F200 diagnostic using the mask effective next update and original full-pool token denominators")


def prepare_observation(online, step):
    """All ranks reuse only the probe from this exact completed update."""
    if online.manifest["threshold_plan"]["evaluation_policy"] != "forget_light_v1":
        raise ValueError("Light evaluation requires a frozen opt-in plan")
    cached = getattr(online, "fixed_forget_probe_result", None)
    reused = cached is not None and cached[0] == step
    if reused:
        result = cached[1]
    else:
        from forget_checkpoint_state import fixed_forget_probe
        result = fixed_forget_probe(online.runtime, online.schedule,
            online.config["trainer"]["method_args"], online.torch, include_common_kl=True)
        online.fixed_forget_probe_result = (step, result)
    expected_hash = online.manifest["schedule_metadata"]["fixed_dataset_sha256"]
    if result["fixed_dataset_sha256"] != expected_hash:
        raise ValueError("Cached Forget probe belongs to a different fixed dataset")
    metrics = ["forget_quality"]
    if step == online.manifest["final_model_step"]:
        metrics.append("model_utility")
    payload = dict(result, fixed_loss_reused=reused)
    if online.manifest["threshold_plan"].get("stopping_scope") == "sample_permanent":
        payload.update(sample_masked_diagnostic(result, online.manifest))
    return dict(metric_names=metrics, fixed_forget_result=payload), (0.0 if reused else result["elapsed_seconds"])


def fixed_loss_provider(probe, result):
    """Record the F200 probe without invoking the full F200/R3800 evaluator."""
    def provide(step, epoch):
        rows = result["fixed_loss_samples"]
        if len(rows) != result["forget_count"] or any(r["split"] != "forget" for r in rows):
            raise ValueError("Light fixed losses require complete Forget-only samples")
        with (probe.epoch_output / "fixed_loss_samples.jsonl").open("a") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row, step=step, epoch=epoch), allow_nan=False) + "\n")
        values = {key: result[key] for key in (
            "raw_sensitive_loss", "raw_common_kl", "sample_mean_raw_sensitive_loss",
            "sample_mean_raw_common_kl", "forget_count", "sensitive_token_count",
            "common_token_count", "forget_zero_sensitive_examples", "forget_zero_common_examples",
            "fixed_dataset_sha256")}
        if any(not math.isfinite(values[k]) for k in ("raw_sensitive_loss", "raw_common_kl")):
            raise ValueError("Nonfinite Forget fixed loss")
        # Preserve the unmasked fixed F + 2 KL diagnostic. R is unmeasured.
        for prefix in ("", "sample_mean_"):
            f, kl = values[prefix + "raw_sensitive_loss"], values[prefix + "raw_common_kl"]
            values.update({prefix + "raw_retain_nll": None,
                prefix + "loss_total": f + 2 * kl, prefix + "loss_forget": f,
                prefix + "loss_retain_nll": None, prefix + "loss_common_kl": 2 * kl})
        values.update(retain_count=0, retain_token_count=0, fixed_loss_scope="forget_only",
            fixed_loss_reused=result["fixed_loss_reused"],
            fixed_probe_wall_seconds=result["elapsed_seconds"])
        if result.get("stopping_scope") == "sample_permanent":
            values.update({key: result[key] for key in (
                "stopping_scope", "sample_stop_state_sha256", "loss_total_semantics",
                "sample_masked_sensitive_loss_next_update", "sample_masked_loss_total_next_update",
                "sample_masked_loss_semantics")})
        return values
    return provide
