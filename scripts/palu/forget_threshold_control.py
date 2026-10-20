#!/usr/bin/env python3
"""Complete-update stopping for the Forget-only experiment; no retention policy."""
import math
from pathlib import Path

import layout_token_training as base
from layout_loss_curve_experiment import read_json, read_rows, write_json


def stop_decision(step, loss, plan):
    if not math.isfinite(loss) or loss < 0:
        raise ValueError("Fixed Forget loss must be finite and nonnegative")
    if not 1 <= step <= plan["total_updates"]:
        raise ValueError("Stopping probe is outside the update budget")
    reached = loss <= plan["threshold_high"]
    if not reached and step < plan["total_updates"]:
        return None
    hit = plan["threshold_low"] <= loss <= plan["threshold_high"]
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


class ForgetThresholdControl:
    run_initial_evaluation = True

    def __init__(self, plan):
        self.plan = plan
        self.planned_updates = plan["total_updates"]
        self.decision = None
        self.last_probe = None
        self._next_probe_step = 1

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

    def after_update(self, runtime, manifest, config, schedule, args, step, online, torch):
        from forget_checkpoint_state import fixed_forget_probe
        if online is None or step != self._next_probe_step or self.decision is not None:
            raise ValueError("Expected one probe after each complete update, before its evaluation")
        self._next_probe_step += 1
        self.last_probe = fixed_forget_probe(runtime, schedule, config["trainer"]["method_args"], torch)
        loss = float(self.last_probe["raw_sensitive_loss"])
        self.decision = stop_decision(step, loss, self.plan)
        row = dict(self.last_probe, step=step, measurement="post-update fixed Forget token mean",
                   seed=self.plan["seed"], consumed_pairs=step * 32, pending_pairs=0,
                   threshold_low=self.plan["threshold_low"], threshold_high=self.plan["threshold_high"],
                   band_hit=self.plan["threshold_low"] <= loss <= self.plan["threshold_high"],
                   stop_reason=self.decision["stop_reason"] if self.decision else None)
        if runtime["rank"] == 0:
            base.append_json(args.output / "forget_probe_history.jsonl", row)
            print(__import__("json").dumps({"event": "fixed_forget_probe", **row}), flush=True)
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


def validate(study):
    """Audit actual early-stop results, never require all maximum-budget updates."""
    import checkpoint_retention as retention
    from forget_checkpoint_state import validate_resume_files
    from forget_until_threshold import frozen_plan
    study, run = Path(study), Path(study) / "run"
    manifest = read_json(run / "manifest.json")
    plan = manifest["threshold_plan"]
    if plan != frozen_plan(study):
        raise ValueError("Manifest threshold plan differs from the frozen experiment")
    decision = read_json(run / "stop_decision.json")
    step = decision["step"]
    if manifest["status"] != "completed" or manifest["completed_updates"] != step:
        raise ValueError("Forget experiment has not completed")
    if decision != stop_decision(step, decision["raw_sensitive_loss"], plan):
        raise ValueError("Recorded stopping reason does not match the measured threshold")
    losses = read_rows(run / "loss_history.jsonl")
    probes = read_rows(run / "forget_probe_history.jsonl")
    if [r["step"] for r in losses] != list(range(1, step + 1)) or [r["step"] for r in probes] != list(range(1, step + 1)):
        raise ValueError("Incomplete per-update losses or fixed Forget probes")
    if any(not math.isfinite(r["raw_sensitive_loss"]) or r["raw_sensitive_loss"] < 0
           for r in probes):
        raise ValueError("Fixed Forget probes contain an invalid loss")
    for row in losses:
        if [row[k] for k in ("gamma_used", "alpha_used", "common_used")] != [1, 0, 2]:
            raise ValueError("Forget objective changed")
        if any(not math.isclose(lr, plan["learning_rate"], rel_tol=1e-10) for lr in row["lr_before"] + row["lr_after"]):
            raise ValueError("Learning rate changed")
    if any(r["raw_sensitive_loss"] <= plan["threshold_high"] for r in probes[:-1]):
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
    write_json(study / "completion_validation.json", result)
    return result
