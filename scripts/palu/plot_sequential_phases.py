#!/usr/bin/env python3
"""Export and plot committed observations from a sequential PALU trajectory.

Analysis uses the native TOFU wrong/correct geometric Truth Ratio, including its
1e-10 denominator stabilizer. It never applies the utility metric's 1-TR clip.
FQ and MU are checked against the native evaluation cache, not re-estimated.
Use --no-plots for a standard-library-only export during an unfinished run.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
import hashlib
import json
import math
from pathlib import Path

EPSILON = 1e-10
LOSS_FIELDS = ("raw_sensitive_loss", "raw_retain_nll", "raw_common_kl")
QUANTILES = (("q10", .1), ("q25", .25), ("q50", .5), ("q75", .75), ("q90", .9))


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def read_rows(path):
    """Permit only a missing file or an unfinished last JSONL line in a live run."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines, result = text.splitlines(), []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as error:
            if number == len(lines) and not text.endswith("\n"):
                break
            raise ValueError(f"Invalid complete JSONL row: {path}:{number}") from error
    return result


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Nonfinite or nonnumeric {name}: {value!r}")
    return float(value)


def mean_loss(value, name):
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError(f"Empty {name}")
    return math.fsum(finite_number(item, name) for item in values) / len(values)


def close(observed, expected, name, *, rtol=1e-10, atol=1e-14):
    if not math.isclose(observed, expected, rel_tol=rtol, abs_tol=atol):
        raise ValueError(f"{name} differs: {observed!r} vs {expected!r}")


def logaddexp(a, b):
    high, low = max(a, b), min(a, b)
    return high + math.log1p(math.exp(low - high))


def numeric_id_order(value):
    return (0, int(value), value) if value.isdecimal() else (1, value, value)


def load_truth_ratios(payload):
    """Match IDs and verify every native TR against its primitive answer losses.

    A finite zero cached TR is retained. It must agree with underflow in the
    evaluator formula; the loss-derived log(TR) remains available and is flagged.
    Invalid/nonfinite inputs fail loudly rather than reducing the sample count.
    """
    correct = payload["forget_Q_A_PARA_Prob"]["value_by_index"]
    wrong = payload["forget_Q_A_PERT_Prob"]["value_by_index"]
    cached = payload["forget_truth_ratio"]["value_by_index"]
    if not correct or set(correct) != set(wrong) or set(correct) != set(cached):
        raise ValueError("Correct/wrong/native TR sample IDs differ or are empty")
    result = {}
    for index in sorted(correct, key=numeric_id_order):
        lc = mean_loss(correct[index]["avg_loss"], f"correct loss [{index}]")
        lw = mean_loss(wrong[index]["avg_loss"], f"wrong-candidate losses [{index}]")
        try:
            pc, pw = math.exp(-lc), math.exp(-lw)
        except OverflowError as error:
            raise ValueError(f"Invalid answer loss exponent at sample {index}") from error
        score = finite_number(cached[index]["score"], f"native TR [{index}]")
        expected = pw / (pc + EPSILON)
        if score < 0 or (score == 0) != (expected == 0):
            raise ValueError(f"Invalid or unexplained zero TR at sample {index}")
        close(score, expected, f"native TR reconstruction [{index}]", atol=0)
        log_tr = -lw - logaddexp(-lc, math.log(EPSILON))
        if not math.isfinite(log_tr):
            raise ValueError(f"Nonfinite reconstructed log(TR) at sample {index}")
        result[index] = dict(sample_id=index, correct_nll=lc, wrong_mean_nll=lw,
            correct_geometric_probability=pc, wrong_geometric_probability=pw,
            truth_ratio=score, log_truth_ratio=log_tr, unregularized_log_ratio=lc-lw,
            native_tr_underflow=score == 0)
    return result


def signed_ks(current, reference):
    """D_high means the current distribution lies toward *larger* raw TR."""
    current, reference = sorted(current), sorted(reference)
    if not current or not reference:
        raise ValueError("KS requires two nonempty samples")
    n, m = len(current), len(reference)
    differences = [bisect_right(current, point) / n - bisect_right(reference, point) / m
                   for point in sorted(set(current + reference))]
    low, high = max(0., max(differences)), max(0., -min(differences))
    return dict(ks_d_reconstructed=max(low, high), D_low=low, D_high=high,
                cdf_has_both_signs=low > 0 and high > 0,
                dominant_side="high_TR" if high > low + 1e-12 else
                              "low_TR" if low > high + 1e-12 else "tied")


def quantile(values, probability):
    values = sorted(values)
    if not values or not 0 <= probability <= 1:
        raise ValueError("Quantile requires nonempty values and a probability")
    index = (len(values) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    fraction = index - low
    return (1 - fraction) * values[low] + fraction * values[high]


def distribution_summary(samples):
    rows = list(samples.values())
    result = {"sample_count": len(rows),
              "native_tr_underflow_count": sum(row["native_tr_underflow"] for row in rows)}
    for field, prefix in (("truth_ratio", "TR"), ("log_truth_ratio", "log_TR")):
        values = [row[field] for row in rows]
        result.update({prefix + "_" + name: quantile(values, probability) for name, probability in QUANTILES})
        result[prefix + "_mean"] = math.fsum(values) / len(values)
    return result


def phase_at(step, experiment):
    return "initial" if step == 0 else "forget" if step <= experiment["forget_updates"] else "retain"


def phase_epoch_at(step, experiment):
    pairs = experiment.get("effective_batch", 32)
    forget = experiment["forget_updates"]
    return (step * pairs / experiment["phase_epoch_size"] if step <= forget else
            (step-forget) * pairs / experiment.get("retain_epoch_size", experiment["phase_epoch_size"]))


def native_metric(payload, key):
    value = payload[key]
    return finite_number(value["agg_value"] if isinstance(value, dict) else value, "native " + key)


def collect(study):
    study = Path(study)
    experiment = read_json(study / "experiment.json")
    reference_path = study / "frozen/retain_reference.json"
    reference = load_truth_ratios(read_json(reference_path))
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    summaries, all_samples, observations = [], [], []
    seen = set()
    for row in sorted(read_rows(study / "run/epoch_history.jsonl"), key=lambda item: item["step"]):
        step = row["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0 or step in seen:
            raise ValueError(f"Invalid or duplicate evaluation step: {step!r}")
        if step > experiment["forget_updates"] + experiment["retain_updates"]:
            raise ValueError(f"Evaluation extends past planned final step: {step}")
        seen.add(step)
        if row.get("reference_sha256") and row["reference_sha256"] != reference_sha:
            raise ValueError(f"Retain reference differs from the recorded evaluation at step {step}")
        evaluation = study / f"run/checkpoint-{step}/evals"
        if not (evaluation / "TOFU_EVAL.json").is_file() and row.get("evaluation_dir"):
            evaluation = Path(row["evaluation_dir"])
        payload = read_json(evaluation / "TOFU_EVAL.json")
        current = load_truth_ratios(payload)
        if set(current) != set(reference):
            raise ValueError(f"Run/Retain sample IDs differ at step {step}; no inner-join dropping is allowed")
        summary = dict(step=step, phase=phase_at(step, experiment),
            phase_update_epochs=phase_epoch_at(step, experiment),
            phase_epoch_recorded=row.get("phase_epoch"),
            actual_data_epochs=row.get("actual_data_epochs", row.get("epoch")),
            training_phase_recorded=row.get("training_phase", ""))
        for key in ("forget_quality", "model_utility", "ks_d") + LOSS_FIELDS:
            summary[key] = finite_number(row[key], f"step {step}: {key}")
        for key in ("forget_quality", "model_utility", "ks_d"):
            if not 0 <= summary[key] <= 1:
                raise ValueError(f"Out-of-range {key} at step {step}")
        for key in ("forget_quality", "model_utility"):
            close(summary[key], native_metric(payload, key), f"step {step}: cached {key}", atol=0)
        diagnostic = signed_ks([item["truth_ratio"] for item in current.values()],
                               [item["truth_ratio"] for item in reference.values()])
        close(summary["ks_d"], diagnostic["ks_d_reconstructed"], f"step {step}: native KS D")
        summary.update(diagnostic, **distribution_summary(current))
        summary["fraction_above_paired_reference"] = sum(
            item["truth_ratio"] > reference[index]["truth_ratio"] for index, item in current.items()) / len(current)
        summary["mean_log_TR_minus_reference"] = math.fsum(
            item["log_truth_ratio"] - reference[index]["log_truth_ratio"] for index, item in current.items()) / len(current)
        for index, item in current.items():
            paired = reference[index]
            all_samples.append(dict(step=step, phase=summary["phase"], **item,
                reference_truth_ratio=paired["truth_ratio"], reference_log_truth_ratio=paired["log_truth_ratio"],
                reference_correct_nll=paired["correct_nll"], reference_wrong_mean_nll=paired["wrong_mean_nll"],
                delta_log_truth_ratio=item["log_truth_ratio"] - paired["log_truth_ratio"]))
        summaries.append(summary)
        observations.append((summary, current))
    return experiment, reference, summaries, all_samples, observations


def plot(study, output, experiment, reference, summaries, observations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 160, "pdf.fonttype": 42})
    boundary = experiment["forget_updates"]
    x = [row["step"] for row in summaries]
    colors = {"initial": "#555555", "forget": "#D55E00", "retain": "#0072B2"}
    point_colors = [colors[row["phase"]] for row in summaries]

    def save(figure, name):
        figure.tight_layout()
        figure.savefig(output / (name + ".png"))
        figure.savefig(output / (name + ".pdf"))
        plt.close(figure)

    def mark(axis):
        axis.axvline(boundary, color="#666666", linestyle=":", label="F to R switch")
        axis.set_xlabel("Complete optimizer update")
        axis.grid(alpha=.2)

    def series(axis, key, label):
        axis.plot(x, [row[key] for row in summaries], color="#555555", linewidth=1, zorder=1)
        axis.scatter(x, [row[key] for row in summaries], c=point_colors, s=28, zorder=2)
        axis.set_ylabel(label)
        mark(axis)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    series(axes[0, 0], "forget_quality", "FQ (native KS p-value)")
    series(axes[0, 1], "forget_quality", "FQ (symlog; zero retained)")
    axes[0, 1].set_yscale("symlog", linthresh=1e-12)
    series(axes[1, 0], "model_utility", "Model utility")
    for key, label, color in (("D_low", "Low-TR side", "#0072B2"), ("D_high", "High-TR side", "#D55E00")):
        axes[1, 1].plot(x, [row[key] for row in summaries], "-o", label=label, color=color)
    axes[1, 1].set_ylabel("Signed KS gaps")
    mark(axes[1, 1]); axes[1, 1].legend(fontsize=8)
    figure.suptitle("Fixed-dataset observations: orange = forgetting, blue = retention", fontsize=11)
    save(figure, "metrics_and_ks_direction")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, key, label in zip(axes, LOSS_FIELDS, ("Forget sensitive-token loss", "Retain NLL", "Common-token KL")):
        series(axis, key, label)
    figure.suptitle("All three raw losses measured on the same fixed probe sets (including inactive objectives)", fontsize=11)
    save(figure, "fixed_loss_curves")

    online = read_rows(Path(study) / "run/loss_history.jsonl")
    if online:
        figure, axes = plt.subplots(1, 3, figsize=(14, 4))
        for axis, key, label in zip(axes, LOSS_FIELDS, ("Forget sensitive-token loss", "Retain NLL", "Common-token KL")):
            valid = [(row.get("step", row.get("global_step")), row[key]) for row in online
                     if key in row and row[key] is not None]
            for step, value in valid:
                finite_number(value, f"online {key} at step {step}")
            if valid:
                axis.plot([v[0] for v in valid], [v[1] for v in valid], linewidth=1)
            axis.set_ylabel(label); mark(axis)
        figure.suptitle("Online pre-update training-batch losses; these are not the fixed-probe losses", fontsize=11)
        save(figure, "online_loss_curves")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, loss, metric, label in ((axes[0], "raw_sensitive_loss", "forget_quality", "Forget sensitive-token loss"),
                                     (axes[1], "raw_retain_nll", "model_utility", "Retain NLL")):
        for phase in ("initial", "forget", "retain"):
            selected = [row for row in summaries if row["phase"] == phase]
            axis.scatter([row[loss] for row in selected], [row[metric] for row in selected], color=colors[phase], label=phase)
        axis.plot([row[loss] for row in summaries], [row[metric] for row in summaries], color="#777777", alpha=.5, zorder=0)
        for row in summaries:
            axis.annotate(str(row["step"]), (row[loss], row[metric]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        axis.set(xlabel=label, ylabel="FQ" if metric == "forget_quality" else "MU")
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    figure.suptitle("Fixed-probe loss versus metric; labels are optimizer updates", fontsize=11)
    save(figure, "loss_metric_scatter")

    reference_summary = distribution_summary(reference)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, prefix, label in ((axes[0], "TR", "Raw Truth Ratio"), (axes[1], "log_TR", "log(Truth Ratio)")):
        for suffix, color in (("q10", "#0072B2"), ("q50", "#222222"), ("q90", "#D55E00")):
            key = prefix + "_" + suffix
            axis.plot(x, [row[key] for row in summaries], "-o", color=color, label=suffix)
            axis.axhline(reference_summary[key], color=color, linestyle="--", linewidth=1)
        axis.fill_between(x, [row[prefix + "_q25"] for row in summaries],
                          [row[prefix + "_q75"] for row in summaries], color="#555555", alpha=.1)
        axis.set_ylabel(label); mark(axis); axis.legend(fontsize=8)
    figure.suptitle("TR quantile trajectories; dashed = Retain quantiles; shading = current IQR", fontsize=11)
    save(figure, "tr_quantile_trajectory")

    def ecdf(axis, samples, field, **kwargs):
        values = sorted(item[field] for item in samples.values())
        axis.step([values[0]] + values, [0.] + [(i+1)/len(values) for i in range(len(values))], where="post", **kwargs)
        axis.set_ylabel("Empirical cumulative probability"); axis.set_ylim(-.02, 1.02); axis.grid(alpha=.2)

    ecdf_folder = output / "ecdf_by_step"
    ecdf_folder.mkdir(exist_ok=True)
    for summary, samples in observations:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for axis, field, label in ((axes[0], "truth_ratio", "Raw TR (wrong / correct)"),
                                    (axes[1], "log_truth_ratio", "log(TR), reconstructed stably from answer losses")):
            ecdf(axis, reference, field, color="black", linestyle="--", linewidth=1.6, label="Retain reference")
            ecdf(axis, samples, field, color=colors[summary["phase"]], label=f"Step {summary['step']} ({summary['phase']})")
            axis.set_xlabel(label); axis.legend(fontsize=8)
        figure.suptitle(f"Step {summary['step']}: FQ={summary['forget_quality']:.5g}, MU={summary['model_utility']:.5g}, "
                       f"D_low={summary['D_low']:.3f}, D_high={summary['D_high']:.3f}", fontsize=11)
        save(figure, f"ecdf_by_step/step_{summary['step']:06d}")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.get_cmap("viridis")
    for number, (summary, samples) in enumerate(observations):
        for axis, field in zip(axes, ("truth_ratio", "log_truth_ratio")):
            ecdf(axis, samples, field, color=cmap(number / max(len(observations)-1, 1)), alpha=.8,
                 label=f"{summary['step']} {summary['phase']}")
    for axis, field, label in zip(axes, ("truth_ratio", "log_truth_ratio"), ("Raw TR", "log(TR)")):
        ecdf(axis, reference, field, color="black", linestyle="--", linewidth=2, label="Retain")
        axis.set_xlabel(label)
    axes[1].legend(fontsize=7, ncol=2)
    figure.suptitle("Every committed evaluation against the same Retain reference", fontsize=11)
    save(figure, "tr_ecdf_all_steps")


def analyze(study, output=None, plots=True):
    study = Path(study).resolve()
    output = Path(output).resolve() if output else study / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    experiment, reference, summaries, samples, observations = collect(study)
    write_csv(output / "reference_truth_ratios.csv", list(reference.values()))
    write_csv(output / "truth_ratio_samples.csv", samples, list(samples[0]) if samples else ["step", "sample_id"])
    write_csv(output / "evaluation_summary.csv", summaries, list(summaries[0]) if summaries else ["step", "phase"])
    ecdfs = []
    for step, current in [("Retain", reference)] + [(summary["step"], current) for summary, current in observations]:
        for field in ("truth_ratio", "log_truth_ratio"):
            for rank, value in enumerate(sorted(item[field] for item in current.values()), 1):
                ecdfs.append(dict(step=step, measurement=field, value=value, ecdf=rank/len(current)))
    write_csv(output / "ecdf_points.csv", ecdfs)
    expected_final = experiment["forget_updates"] + experiment["retain_updates"]
    report = dict(status="final_observation_available" if summaries and summaries[-1]["step"] == expected_final else "partial",
        completed_evaluation_steps=[row["step"] for row in summaries], planned_final_step=expected_final,
        forget_to_retain_switch_step=experiment["forget_updates"], sample_count=len(reference),
        reference_sha256=hashlib.sha256((study / "frozen/retain_reference.json").read_bytes()).hexdigest(),
        reference_summary=distribution_summary(reference),
        native_tr_underflow_count=sum(row["native_tr_underflow"] for row in samples),
        native_FQ_MU_match_history=bool(summaries), KS_D_reconstructed_from_native_raw_TR=bool(summaries),
        FQ_pvalue_recomputed=False, plotted=bool(plots and summaries))
    if plots and summaries:
        plot(study, output, experiment, reference, summaries, observations)
    atomic_text(output / "README.md", "# Sequential PALU trajectory\n\n"
        "Plots and CSVs include every committed evaluation currently available. Re-running analysis updates them as training progresses. "
        "The final observation alone does not certify that the training process has exited successfully.\n\n"
        "`truth_ratio_samples.csv` joins the run and Retain reference by exact evaluation sample ID; missing or duplicate IDs are errors. "
        "TR is the un-clipped native TOFU score: exp(-mean(wrong-answer NLL)) / (exp(-correct-answer NLL) + 1e-10). "
        "Wrong-candidate averaging is geometric at the probability level. Log(TR) is reconstructed with a stable log denominator. "
        "Any finite zero native TR caused by numerical underflow is kept and flagged; no sample is silently removed. "
        "`unregularized_log_ratio` additionally exposes correct NLL minus mean wrong NLL, without the native epsilon.\n\n"
        "`D_low=max(CDF_run-CDF_Retain)` and `D_high=max(CDF_Retain-CDF_run)`; KS D is their maximum. "
        "High-TR dominance indicates the direction of the largest CDF discrepancy, not that every sample has crossed the reference. "
        "FQ is the saved native KS p-value; this analysis checks cache/history agreement and independently reconstructs D, without recomputing p-values. "
        "MU is checked against its native cached value; its nine components are not independently re-evaluated here.\n\n"
        "Metric/loss scatter plots use post-update fixed-probe losses. Online batch losses are plotted separately and are not interchangeable. "
        "All figures use complete optimizer updates on the horizontal axis; the dotted line marks the objective switch. "
        "Quantiles and ECDFs describe across-question distributions, not uncertainty over seeds or proof of knowledge deletion. "
        "The two-epoch evaluation interval can miss a shorter-lived minimum of distribution distance.\n")
    atomic_text(output / "analysis_manifest.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(analyze(args.study, args.output, plots=not args.no_plots), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
