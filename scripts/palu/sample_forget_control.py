"""Permanent per-example PALU-F stopping; shared parameters may still drift."""
import copy
import math

from forget_threshold_control import ForgetThresholdControl
from layout_loss_curve_experiment import read_json, read_rows
from prepare_layout_schedule import canonical_sha, selected_sensitive_count
import layout_token_training as base


class SampleStopState:
    def __init__(self, counts, plan):
        if not counts or any(type(n) is not int or n < 0 for n in counts) or not sum(counts):
            raise ValueError("Sample stopping requires valid sensitive token counts")
        self.plan = plan
        self.step = 0
        self.rows = [dict(dataset_index=i, sensitive_token_count=n, eligible=n > 0,
                          active=n > 0, first_stop_step=None, first_stop_loss=None,
                          first_stop_below_low=None, last_loss=None, ever_below_low=False)
                     for i, n in enumerate(counts)]

    def snapshot(self):
        return dict(schema=1, policy="sample_permanent", step=self.step,
                    rows=copy.deepcopy(self.rows))

    def observe(self, step, samples):
        from forget_until_threshold import probe_steps
        expected = [s for s in probe_steps(self.plan) if s > self.step]
        if not expected or step != expected[0]:
            raise ValueError("Sample stop probe skipped or repeated a scheduled epoch")
        if (len(samples) != len(self.rows) or
                any(type(s.get("dataset_index")) is not int for s in samples) or
                sorted(s["dataset_index"] for s in samples) != list(range(len(self.rows)))):
            raise ValueError("Sample stop probe omitted or duplicated fixed Forget IDs")
        ordered = sorted(samples, key=lambda s: s["dataset_index"])
        # Validate the complete probe before mutating any state.
        for old, sample in zip(self.rows, ordered):
            value = sample["raw_sensitive_loss"]
            if (sample.get("split") != "forget" or
                    type(sample["sensitive_token_count"]) is not int or
                    sample["sensitive_token_count"] != old["sensitive_token_count"] or
                    type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError("Invalid fixed per-sample Forget loss or target count")
        changes = []
        low, high = self.plan["threshold_low"], self.plan["threshold_high"]
        for row, sample in zip(self.rows, ordered):
            value, before = float(sample["raw_sensitive_loss"]), row["active"]
            previous_stop = row["first_stop_step"]
            if row["eligible"]:
                row["ever_below_low"] |= value < low
                if before and value < high:
                    row.update(active=False, first_stop_step=step, first_stop_loss=value,
                               first_stop_below_low=value < low)
            row["last_loss"] = value
            changes.append(dict(row, step=step, active_before=before,
                newly_stopped=before and not row["active"],
                current_region=("not_applicable" if not row["eligible"] else
                                "below_low" if value < low else "in_band" if value < high else "above_or_equal_high"),
                drift_from_stop=(value-row["first_stop_loss"] if previous_stop is not None else None),
                stopped_sample_rebounded=(previous_stop is not None and value >= high)))
        self.step = step
        summary = self.summary()
        return changes, summary

    def summary(self):
        valid = [r for r in self.rows if r["eligible"]]
        stopped = [r for r in valid if not r["active"]]
        measured = [r for r in valid if r["last_loss"] is not None]
        low, high = self.plan["threshold_low"], self.plan["threshold_high"]
        return dict(eligible_samples=len(valid), ineligible_zero_sensitive_samples=len(self.rows)-len(valid),
            active_samples=sum(r["active"] for r in valid), stopped_samples=len(stopped),
            stopped_in_band=sum(not r["first_stop_below_low"] for r in stopped),
            stopped_below_low=sum(r["first_stop_below_low"] for r in stopped),
            current_below_low=sum(r["last_loss"] < low for r in measured),
            current_in_band=sum(low <= r["last_loss"] < high for r in measured),
            current_above_or_equal_high=sum(r["last_loss"] >= high for r in measured),
            inactive_above_or_equal_high=sum(r["last_loss"] is not None and r["last_loss"] >= high for r in stopped),
            ever_below_low=sum(r["ever_below_low"] for r in valid))

    def decision(self, step, aggregate_loss):
        if (type(step) is not int or step != self.step or step == 0 or
                type(aggregate_loss) not in (int, float) or not math.isfinite(aggregate_loss) or aggregate_loss < 0):
            raise ValueError("Sample stopping decision requires the most recent finite fixed probe")
        summary = self.summary()
        reached = summary["active_samples"] == 0
        if not reached and step < self.plan["total_updates"]:
            return None
        return dict(step=step, raw_sensitive_loss=aggregate_loss,
            stop_reason="all_eligible_samples_deactivated" if reached else "budget_exhausted",
            threshold_reached=reached,
            band_hit=reached and summary["stopped_below_low"] == 0,
            threshold_low=self.plan["threshold_low"], threshold_high=self.plan["threshold_high"],
            consumed_pairs=step*32, pending_pairs=0, stopping_scope="sample_permanent",
            threshold_reached_semantics="all eligible samples have crossed at some observed epoch; not all current losses are below threshold",
            band_hit_semantics="all first crossings were in band; does not constrain final drift",
            sample_stop_state_sha256=canonical_sha(self.snapshot()), **summary)


class SampleForgetControl(ForgetThresholdControl):
    def initialize(self, runtime, manifest, config, schedule, args, torch):
        from forget_until_threshold import validate_stopping_policy
        validate_stopping_policy(self.plan)
        if (self.plan.get("stopping_scope") != "sample_permanent" or args.layout != "four"
                or runtime["world"] != 4 or args.padding_scope != "microbatch"):
            raise ValueError("Sample stopping requires the frozen four-card b1 global-token layout")
        counts = [selected_sensitive_count(s["labels"], runtime["tokenizer"].eos_token_id, 3)
                  for s in schedule["samples"]["forget"]]
        if len(counts) != 200 or sum(counts) != schedule["metadata"]["fixed_target_counts"]["sensitive"]:
            raise ValueError("Sample eligibility differs from frozen F200 counts")
        self.state = SampleStopState(counts, self.plan)
        self.persist_state(runtime, manifest, args, torch)
        super().initialize(runtime, manifest, config, schedule, args, torch)
        manifest["training_objective"] = "masked F + 2common_KL; original whole-window token denominators"
        if runtime["rank"] == 0:
            base.atomic_json(args.output / "manifest.json", manifest)

    def persist_state(self, runtime, manifest, args, torch):
        snapshot = self.state.snapshot()
        digest = canonical_sha(snapshot)
        if runtime["distributed"]:
            hashes = [None] * runtime["world"]
            torch.distributed.all_gather_object(hashes, digest)
            if any(h != digest for h in hashes):
                raise AssertionError("Ranks disagree on the per-sample Forget mask")
        manifest.update(sample_stop_state=snapshot, sample_stop_state_sha256=digest)
        if runtime["rank"] == 0:
            base.atomic_json(args.output / "sample_stop_state.json", snapshot)
            base.atomic_json(args.output / "manifest.json", manifest)

    def forget_weight(self, index):
        if type(index) is not int or not 0 <= index < len(self.state.rows):
            raise ValueError("Forget ID outside the fixed pool")
        return float(self.state.rows[index]["active"])

    def training_row_metadata(self, step):
        return dict(super().training_row_metadata(step), stopping_scope="sample_permanent",
                    sample_stop_state_sha256=canonical_sha(self.state.snapshot()),
                    sample_mask_observed_after_step=self.state.step, **self.state.summary())

    def process_probe(self, step, runtime, manifest, args, torch):
        changes, summary = self.state.observe(step, self.last_probe["fixed_loss_samples"])
        self.persist_state(runtime, manifest, args, torch)
        self.last_probe.update(sample_stop_state=self.state.snapshot(), sample_stop_summary=summary,
            sample_masked_sensitive_loss_next_update=sum(r["last_loss"] * r["sensitive_token_count"]
                for r in self.state.rows if r["active"]) / sum(r["sensitive_token_count"] for r in self.state.rows))
        if runtime["rank"] == 0:
            for row in changes:
                base.append_json(args.output / "sample_stop_history.jsonl", row)
        return self.state.decision(step, float(self.last_probe["raw_sensitive_loss"]))


def validate_sample_run(run, manifest, plan, losses, probes, decision):
    """Replay every mask from raw probes; verify the mask used by all 32 pairs."""
    samples = read_rows(run / "forget_probe_samples.jsonl")
    first = sorted([r for r in samples if r["step"] == probes[0]["step"]], key=lambda r: r["dataset_index"])
    state = SampleStopState([r["sensitive_token_count"] for r in first], plan)
    probe_by_step = {r["step"]: r for r in probes}
    transitions = []
    all_pairs = read_rows(run / "training_pairs.jsonl")
    for loss in losses:
        step = loss["step"]
        if (loss.get("sample_stop_state_sha256") != canonical_sha(state.snapshot()) or
                loss.get("sample_mask_observed_after_step") != state.step):
            raise ValueError("Training used a different or prematurely updated sample mask")
        if any(loss.get(k) != v for k, v in state.summary().items()):
            raise ValueError("Training sample status counts differ from replay")
        pairs = [r for r in all_pairs if r["step"] == step]
        micros = loss["microsteps"]
        if (len(pairs) != 32 or len(micros) != 32 or
                sorted(i for m in micros for i in m["pair_indices"]) != list(range(32)) or
                any(sum(m["counts"][k] for m in micros) != n for k, n in loss["global_target_counts"].items())):
            raise ValueError("Sample mask log does not cover the complete 32-pair update")
        for micro in micros:
            indices = micro["pair_indices"]
            if len(indices) != 1:
                raise ValueError("Sample masking requires b1 microbatches")
            index = pairs[indices[0]]["forget_index"]
            if (micro["forget_dataset_index"] != index or
                    micro["sample_forget_weight"] != float(state.rows[index]["active"]) or
                    micro["counts"]["sensitive"] != state.rows[index]["sensitive_token_count"]):
                raise ValueError("Recorded sample mask differs from frozen pair or previous probe")
        reconstructed = copy.deepcopy(loss)
        for component, field in (("sensitive", "raw_sensitive_loss"), ("retain", "raw_retain_nll"), ("common", "raw_common_kl")):
            value = sum(m["raw"][component] * m["counts"][component] for m in micros) / loss["global_target_counts"][component]
            if not math.isclose(value, loss[field], rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError("Unmasked training diagnostic does not reconstruct")
        base.masked_objective_record(reconstructed, micros, 1., 0., 2.)
        for field in ("loss", "loss_total", "loss_forget", "masked_sensitive_loss",
                      "active_sensitive_tokens_in_update", "active_sensitive_token_fraction", "unmasked_objective_loss"):
            if not math.isclose(loss[field], reconstructed[field], rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError("Masked training objective does not reconstruct")
        if step in probe_by_step:
            probe = probe_by_step[step]
            changes, summary = state.observe(step, [r for r in samples if r["step"] == step])
            transitions.extend(changes)
            if probe.get("sample_stop_state") != state.snapshot() or probe.get("sample_stop_summary") != summary:
                raise ValueError("Saved sample stop state differs from fixed-probe replay")
            masked = sum(r["last_loss"] * r["sensitive_token_count"] for r in state.rows if r["active"]) / sum(r["sensitive_token_count"] for r in state.rows)
            if not math.isclose(probe["sample_masked_sensitive_loss_next_update"], masked, rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError("Masked fixed probe diagnostic does not reconstruct")
            expected = state.decision(step, probe["raw_sensitive_loss"])
            if (step == decision["step"] and expected != decision) or (step < decision["step"] and expected is not None):
                raise ValueError("Sample termination decision differs from replay")
    if transitions != read_rows(run / "sample_stop_history.jsonl"):
        raise ValueError("Per-sample stop/drift history differs from probes")
    snapshot = state.snapshot()
    if (snapshot != read_json(run / "sample_stop_state.json") or
            snapshot != manifest.get("sample_stop_state") or
            canonical_sha(snapshot) != manifest.get("sample_stop_state_sha256")):
        raise ValueError("Final sample stopping state/hash mismatch")
