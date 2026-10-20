#!/usr/bin/env python3
"""Complete-update stopping for the Forget-only experiment; no retention policy."""
import math
from pathlib import Path

import layout_token_training as base
from layout_loss_curve_experiment import read_json, read_rows, write_json


def threshold_reached(loss, plan):
    comparison = plan.get("threshold_comparison", "le")
    if comparison not in ("lt", "le"):
        raise ValueError("Threshold comparison must be lt or le")
    return loss < plan["threshold_high"] if comparison == "lt" else loss <= plan["threshold_high"]


def stop_decision(step, loss, plan):
    from forget_until_threshold import probe_steps
    if not math.isfinite(loss) or loss < 0:
        raise ValueError("Fixed Forget loss must be finite and nonnegative")
    if not 1 <= step <= plan["total_updates"]:
        raise ValueError("Stopping probe is outside the update budget")
    if step not in probe_steps(plan):
        raise ValueError("Stopping decision is outside the frozen probe schedule")
    reached = threshold_reached(loss, plan)
    if not reached and step < plan["total_updates"]:
        return None
    hit = plan["threshold_low"] <= loss and reached
    reason = ("threshold_reached" if hit else "threshold_overshot") if reached else "budget_exhausted"
    return dict(step=step, raw_sensitive_loss=loss, stop_reason=reason,
                threshold_reached=reached, band_hit=hit,
                threshold_low=plan["threshold_low"], threshold_high=plan["threshold_high"],
                consumed_pairs=step * 32, pending_pairs=0)


def terminal_event(step, reason):
    return dict(step=step, consumed_pairs=step * 32, updated_pairs=step * 32,
                pending_pairs=0, actual_data_epochs=step * 32 / 200,
                weight_update_epochs=step * 32 / 200, completed_data_epochs=step * 32 // 200,
                source_epoch_index=math.ceil(step * 32 / 200), evaluation_kind="forget_stop",
                training_phase="forget", phase_consumed_pairs=step * 32,
                phase_epoch=step * 32 / 200, phase_epoch_size=200,
                phase_boundary=True, stop_reason=reason)


def validate_resume_identity(receipt, manifest, step):
    """Bind a self-consistent native archive to this experiment's stopping state."""
    from forget_checkpoint_state import IDENTITY_KEYS
    expected = dict(step=step, completed_updates=step,
                    consumed_pairs=step * 32, next_update=step + 1)
    expected.update({key: manifest[key] for key in IDENTITY_KEYS})
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Native checkpoint does not match this experiment's stopping state")
    if manifest.get("threshold_plan", {}).get("stopping_scope") == "sample_permanent":
        for key in ("sample_stop_state", "sample_stop_state_sha256"):
            if key not in manifest or receipt.get(key) != manifest[key]:
                raise ValueError("Native checkpoint lost the sample stopping state")


class ForgetThresholdControl:
    run_initial_evaluation = True

    def __init__(self, plan):
        from forget_until_threshold import probe_steps
        self.plan = plan
        self.planned_updates = plan["total_updates"]
        self.decision = None
        self.last_probe = None
        self._next_update_step = 1
        self.probe_steps = set(probe_steps(plan))

    def select_windows(self, windows):
        return windows

    def coefficients_for_step(self, step):
        if not 1 <= step <= self.planned_updates:
            raise ValueError("Update outside Forget budget")
        return 1., 0., 2.

    def training_row_metadata(self, step):
        return dict(training_phase="forget", phase_update=step,
                    phase_epoch=step * 32 / 200, phase_epoch_size=200)

    def initialize(self, runtime, manifest, config, schedule, args, torch):
        if not args.online_evaluation or args.normalization != "global":
            raise ValueError("Threshold stopping requires online evaluation and global token means")
        manifest.update(experiment=self.plan["experiment"], phase_plan=self.plan,
                        training_objective="F + 2common_KL; stop before choosing retention policy",
                        threshold_plan=self.plan, branches=0, retain_updates=0,
                        continuation_policy="not chosen", resume_restore_validated=False)
        if runtime["rank"] == 0:
            base.atomic_json(args.output / "manifest.json", manifest)

    def configure_online(self, online):
        from forget_until_threshold import evaluation_events
        online.events = {r["consumed_pairs"]: r for r in evaluation_events(self.plan)}

    def process_probe(self, step, runtime, manifest, args, torch):
        """Global stopping remains the default; sample control overrides only this decision."""
        return stop_decision(step, float(self.last_probe["raw_sensitive_loss"]), self.plan)

    def after_update(self, runtime, manifest, config, schedule, args, step, online, torch):
        from forget_checkpoint_state import fixed_forget_probe
        if online is None or step != self._next_update_step or self.decision is not None:
            raise ValueError("Expected one control hook after each complete update, before its evaluation")
        self._next_update_step += 1
        if step not in self.probe_steps:
            return False
        light = self.plan.get("evaluation_policy") == "forget_light_v1"
        options = {"include_common_kl": True} if light else {}
        self.last_probe = fixed_forget_probe(runtime, schedule, config["trainer"]["method_args"], torch, **options)
        if light:
            online.fixed_forget_probe_result = (step, self.last_probe)
        loss = float(self.last_probe["raw_sensitive_loss"])
        self.decision = self.process_probe(step, runtime, manifest, args, torch)
        row = dict({k: v for k, v in self.last_probe.items() if k != "fixed_loss_samples"},
                   step=step, measurement="post-update fixed Forget token mean",
                   seed=self.plan["seed"], consumed_pairs=step * 32, pending_pairs=0,
                   probe_interval=self.plan.get("probe_interval", "update"),
                   actual_data_epochs=step * 32 / 200,
                   threshold_low=self.plan["threshold_low"], threshold_high=self.plan["threshold_high"],
                   band_hit=self.plan["threshold_low"] <= loss and threshold_reached(loss, self.plan),
                   stop_reason=self.decision["stop_reason"] if self.decision else None)
        if self.plan.get("stopping_scope") == "sample_permanent":
            row["aggregate_in_band_diagnostic"] = row.pop("band_hit")
        if runtime["rank"] == 0:
            for sample in self.last_probe.get("fixed_loss_samples", []):
                base.append_json(args.output / "forget_probe_samples.jsonl",
                                 dict(sample, step=step, epoch=step * 32 / 200))
            base.append_json(args.output / "forget_probe_history.jsonl", row)
            print(__import__("json").dumps({"event": "fixed_forget_probe",
                **{k: v for k, v in row.items() if k != "sample_stop_state"}}), flush=True)
        if self.decision is None:
            return False
        # This happens before OnlineEpochEvaluation saves/prunes this state.
        # Replace (rather than duplicate) any regular evaluation at this step.
        event = terminal_event(step, self.decision["stop_reason"])
        online.events = {n: r for n, r in online.events.items() if n < step * 32}
        online.events[step * 32] = event
        manifest["planned_evaluation_steps"] = list(manifest["evaluation_steps"])
        manifest.update(final_model_step=step, actual_total_updates=step,
                        early_stopped=step < self.planned_updates, pending_pairs=0,
                        stop_decision=self.decision,
                        evaluation_steps=[r["step"] for _, r in sorted(online.events.items())])
        if runtime["rank"] == 0:
            base.atomic_json(args.output / "stop_decision.json", self.decision)
            base.atomic_json(args.output / "manifest.json", manifest)
        return True

    def finish(self, runtime, manifest, args, torch):
        from forget_checkpoint_state import save_resume_checkpoint
        if not self.decision or manifest["completed_updates"] != self.decision["step"]:
            raise AssertionError("Forget run ended without a valid complete-update stopping decision")
        if manifest.get("evaluated_steps") != manifest["evaluation_steps"]:
            raise AssertionError("Stopped state has not been fully evaluated")
        # Online rows exist only on rank 0. Broadcast its validation error before
        # entering the native checkpoint collectives so peers cannot hang.
        error = None
        if runtime["rank"] == 0:
            try:
                row = read_rows(args.output / "epoch_history.jsonl")[-1]
                if (row["step"] != self.decision["step"] or
                        not math.isclose(row["raw_sensitive_loss"], self.decision["raw_sensitive_loss"],
                                         rel_tol=1e-4, abs_tol=1e-5)):
                    raise ValueError("Stopping probe and final fixed-data evaluation disagree")
            except Exception as exc:
                error = f"Final evaluation validation failed: {type(exc).__name__}: {exc}"
        if runtime["distributed"]:
            payload = [error]
            torch.distributed.broadcast_object_list(payload, src=0)
            error = payload[0]
        if error:
            raise AssertionError(error)
        receipt = save_resume_checkpoint(args.output / "resume_checkpoint", runtime, manifest,
                                         self.decision["step"], torch, probe=self.last_probe,
                                         stop_reason=self.decision["stop_reason"])
        manifest.update(resume_checkpoint=str(args.output / "resume_checkpoint"),
                        resume_checkpoint_status=receipt["status"],
                        continuation_policy="not chosen", resume_restore_validated=False)
        return False  # shared loop writes completion timestamps and final checks


def validate_light_probes(probes, samples, metadata):
    """Check every epoch probe, including epochs with no TOFU evaluation."""
    grouped = {row["step"]: [] for row in probes}
    for sample in samples:
        if sample["step"] not in grouped:
            raise ValueError("Fixed probe samples contain an unrecorded step")
        grouped[sample["step"]].append(sample)
    size = metadata["data_sizes"]["forget"]
    for row in probes:
        rows = grouped[row["step"]]
        if (row["fixed_dataset_sha256"] != metadata["fixed_dataset_sha256"]
                or len(rows) != size or row["forget_count"] != size
                or {r["dataset_index"] for r in rows} != set(range(size))
                or any(r["split"] != "forget" for r in rows)):
            raise ValueError("Fixed probe samples differ from the frozen F200 set")
        for component, field in (("sensitive", "raw_sensitive_loss"), ("common", "raw_common_kl")):
            count_key = component + "_token_count"
            if any(not isinstance(r[field], (int, float)) or not math.isfinite(r[field])
                   or type(r[count_key]) is not int or r[count_key] < 0 for r in rows):
                raise ValueError("Invalid fixed probe per-example loss or count")
            count = sum(r[count_key] for r in rows)
            value = sum(r[field] * r[count_key] for r in rows) / max(count, 1)
            if (count != metadata["fixed_target_counts"][component] or row[count_key] != count
                    or not math.isclose(value, row[field], rel_tol=1e-8, abs_tol=1e-9)):
                raise ValueError("Fixed probe token counts or aggregate loss do not reconstruct")


def validate(study):
    """Audit actual early-stop results, never require all maximum-budget updates."""
    import checkpoint_retention as retention
    from forget_checkpoint_state import validate_resume_files
    from forget_until_threshold import frozen_plan, probe_steps
    study, run = Path(study), Path(study) / "run"
    manifest = read_json(run / "manifest.json")
    plan = manifest["threshold_plan"]
    if plan != frozen_plan(study):
        raise ValueError("Manifest threshold plan differs from the frozen experiment")
    decision = read_json(run / "stop_decision.json")
    step = decision["step"]
    if manifest["status"] != "completed" or manifest["completed_updates"] != step:
        raise ValueError("Forget experiment has not completed")
    sample_mode = plan.get("stopping_scope", "global") == "sample_permanent"
    if not sample_mode and decision != stop_decision(step, decision["raw_sensitive_loss"], plan):
        raise ValueError("Recorded stopping reason does not match the measured threshold")
    losses = read_rows(run / "loss_history.jsonl")
    probes = read_rows(run / "forget_probe_history.jsonl")
    expected_probes = [s for s in probe_steps(plan) if s <= step]
    if [r["step"] for r in losses] != list(range(1, step + 1)) or [r["step"] for r in probes] != expected_probes:
        raise ValueError("Incomplete per-update losses or scheduled fixed Forget probes")
    if any(not math.isfinite(r["raw_sensitive_loss"]) or r["raw_sensitive_loss"] < 0
           for r in probes):
        raise ValueError("Fixed Forget probes contain an invalid loss")
    if plan.get("evaluation_policy") == "forget_light_v1":
        validate_light_probes(probes, read_rows(run / "forget_probe_samples.jsonl"),
                              manifest["schedule_metadata"])
    expected_retain = base.retain_backward_policy(
        {"forget_threshold_plan": plan}, ForgetThresholdControl(plan), 0.0)
    if ("retain_backward_policy" in manifest or expected_retain["zero_weight_retain_no_grad"]):
        if manifest.get("retain_backward_policy") != expected_retain:
            raise ValueError("Manifest Retain backward mode differs from the frozen plan")
    for row in losses:
        if expected_retain["zero_weight_retain_no_grad"] or any(key in row for key in expected_retain):
            if any(row.get(key) != value for key, value in expected_retain.items()):
                raise ValueError("Recorded Retain backward mode differs from the frozen plan")
        if [row[k] for k in ("gamma_used", "alpha_used", "common_used")] != [1, 0, 2]:
            raise ValueError("Forget objective changed")
        if any(not math.isclose(lr, plan["learning_rate"], rel_tol=1e-10) for lr in row["lr_before"] + row["lr_after"]):
            raise ValueError("Learning rate changed")
    if sample_mode:
        from sample_forget_control import validate_sample_run
        validate_sample_run(run, manifest, plan, losses, probes, decision)
    elif any(threshold_reached(r["raw_sensitive_loss"], plan) for r in probes[:-1]):
        raise ValueError("Training continued after an earlier threshold crossing")
    if probes[-1]["raw_sensitive_loss"] != decision["raw_sensitive_loss"]:
        raise ValueError("Last fixed probe differs from stopping decision")
    from forget_until_threshold import evaluation_events
    expected_steps = [e["step"] for e in evaluation_events(plan) if e["step"] < step] + [step]
    if manifest["evaluated_steps"] != expected_steps:
        raise ValueError("Missing initial, periodic or final evaluation")
    if [c["step"] for c in manifest["checkpoints"]] != expected_steps:
        raise ValueError("Checkpoint inventory differs from completed evaluation events")
    for checkpoint in manifest["checkpoints"]:
        receipt = retention.read_receipt(run, manifest, checkpoint)
        if not receipt["training_state_unchanged_all_ranks"] or receipt["pending_pairs"]:
            raise ValueError("Evaluation changed training state or has pending gradients")
    scheduled = read_rows(study / "frozen/pair_schedule.jsonl")[:step * 32]
    consumed = read_rows(run / "training_pairs.jsonl")
    if len(consumed) != step * 32 or any(any(b.get(k) != v for k, v in a.items())
                                        for a, b in zip(scheduled, consumed)):
        raise ValueError("Consumed F/R pair stream differs from the frozen prefix")
    if manifest["deleted_intermediate_steps"] != expected_steps[1:-1]:
        raise ValueError("Intermediate checkpoint cleanup differs from completed events")
    if not list((run / f"checkpoint-{step}").glob("*.safetensors")):
        raise ValueError("Stopped model weights are missing")
    native = read_json(run / "resume_checkpoint/receipt.json")
    validate_resume_identity(native, manifest, step)
    validate_resume_files(run / "resume_checkpoint", native)
    result = dict(passed=True, stop_step=step, completed_updates=step,
                  stop_reason=decision["stop_reason"], threshold_reached=decision["threshold_reached"],
                  band_hit=decision["band_hit"], actual_final_loss=decision["raw_sensitive_loss"],
                  evaluation_steps=expected_steps, native_checkpoint_saved=True,
                  resume_restore_validated=False, continuation_policy="not chosen")
    if sample_mode:
        result.update(stopping_scope="sample_permanent", sample_stop_state_sha256=manifest["sample_stop_state_sha256"],
                      sample_stop_summary=probes[-1]["sample_stop_summary"],
                      threshold_reached_semantics=decision["threshold_reached_semantics"],
                      band_hit_semantics=decision["band_hit_semantics"])
    write_json(study / "completion_validation.json", result)
    return result
