"""Evaluation-only policy for Forget threshold runs; no optimizer changes."""
import json
import math


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
        # The active Forget objective is F + 2 KL. R is unmeasured, never zero.
        for prefix in ("", "sample_mean_"):
            f, kl = values[prefix + "raw_sensitive_loss"], values[prefix + "raw_common_kl"]
            values.update({prefix + "raw_retain_nll": None,
                prefix + "loss_total": f + 2 * kl, prefix + "loss_forget": f,
                prefix + "loss_retain_nll": None, prefix + "loss_common_kl": 2 * kl})
        values.update(retain_count=0, retain_token_count=0, fixed_loss_scope="forget_only",
            fixed_loss_reused=result["fixed_loss_reused"],
            fixed_probe_wall_seconds=result["elapsed_seconds"])
        return values
    return provide
