from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from palu_hparam_analysis.coverage import coverage_audit
from palu_hparam_analysis.effects import fit_metric_model
from palu_hparam_analysis.ingest import (
    collect_records,
    select_checkpoints,
    select_protocol_cohort,
)
from palu_hparam_analysis.pareto import pareto_front
from palu_hparam_analysis.report import run_analysis
from palu_hparam_analysis.schema import DEFAULT_CONFIG, FACTORS


def synthetic_config() -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    config["search_space"] = {
        "lr": [1e-5, 2e-5],
        "alpha": [0.5, 1.0],
        "top_k": [1, 1000],
        "first_n": [1, 3],
    }
    config["filters"] = {
        "model": "Llama-3.1-8B-Instruct",
        "split": "forget05",
        "trainer_contains": "PALU",
        "date": None,
    }
    config["checkpoint"] = {"policy": "epoch", "epoch": 10}
    config["baselines"] = {
        "full_summary": None,
        "retain_summary": None,
        "overrides": {},
    }
    config["ranking"]["exact_memorization_target"] = 0.6
    config["modeling"].update(
        {
            "min_rows": 8,
            "cv_folds": 4,
            "bootstrap_samples": 5,
            "ridge_alphas": [0.01, 0.1, 1.0],
            "min_cv_r2_for_candidates": -1.0,
            "random_seed": 7,
        }
    )
    config["output"] = {"static_formats": ["png"], "dpi": 90}
    return config


def synthetic_metric_values(lr: float, alpha: float, top_k: int, first_n: int) -> dict:
    x_lr = int(lr > 1e-5)
    x_alpha = int(alpha > 0.5)
    x_top = int(top_k > 1)
    x_first = int(first_n > 1)
    interaction = x_lr * x_alpha - x_top * x_first
    return {
        "model_utility": 0.53 + 0.06 * x_alpha - 0.025 * x_lr + 0.01 * x_top,
        "forget_quality": 0.22 + 0.18 * x_lr + 0.08 * x_top + 0.025 * interaction,
        "forget_Q_A_gibberish": 0.58 + 0.07 * x_alpha - 0.04 * x_first + 0.015 * x_top,
        "exact_memorization": 0.79 - 0.10 * x_lr - 0.045 * x_first + 0.015 * x_alpha,
    }


def make_run(
    result_root: Path,
    lr: float,
    alpha: float,
    top_k: int,
    first_n: int,
    metrics: dict,
    hydra_alpha: float = None,
) -> Path:
    suffix = (
        f"target_modemean_first_n{first_n}_lr{lr:g}_b8_ga4_a{alpha:g}_topk{top_k}"
        "_e10_day0831_time120000"
    )
    run = (
        result_root
        / "tofu"
        / "forget05"
        / "Llama-3.1-8B-Instruct"
        / "2026-08-31"
        / "PALU_acl"
        / suffix
    )
    evals = run / "checkpoint-100" / "evals"
    evals.mkdir(parents=True)
    (evals / "TOFU_SUMMARY.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"step": 100, "epoch": 10.0}]}), encoding="utf-8"
    )
    if hydra_alpha is not None:
        hydra = run / ".hydra"
        hydra.mkdir()
        (hydra / "config.yaml").write_text(
            f"""trainer:
  args:
    learning_rate: {lr}
    per_device_train_batch_size: 8
    gradient_accumulation_steps: 4
    num_train_epochs: 10
    seed: 0
  method_args:
    alpha: {hydra_alpha}
    top_k: {top_k}
    first_n: {first_n}
    gamma: 1.0
    target_mode: mean
    retain_loss_type: NLL
model:
  handler: Llama-3.1-8B-Instruct
  model_args:
    attn_implementation: sdpa
forget_split: forget05
""",
            encoding="utf-8",
        )
    return run


class IngestionTests(unittest.TestCase):
    def test_hydra_metadata_wins_and_epoch_comes_from_trainer_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "saves" / "unlearn"
            make_run(
                root,
                2e-5,
                0.5,
                1000,
                3,
                synthetic_metric_values(2e-5, 0.5, 1000, 3),
                hydra_alpha=1.0,
            )
            rows, warnings = collect_records(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["alpha"], 1.0)
            self.assertEqual(rows[0]["epoch"], 10.0)
            self.assertEqual(rows[0]["epoch_source"], "trainer_state")
            self.assertEqual(rows[0]["effective_batch"], 32)
            self.assertTrue(any("metadata mismatch" in warning for warning in warnings))
            selected = select_checkpoints(rows, synthetic_config())
            self.assertEqual(len(selected), 1)

    def test_mixed_protocols_are_separated_by_default(self):
        base = {
            "model": "Llama-3.1-8B-Instruct",
            "split": "forget05",
            "trainer": "PALU_acl",
            "effective_batch": 32,
            "target_mode": "mean",
            "gamma": 1.0,
            "retain_loss_type": "NLL",
        }
        records = [{**base, "attention": "sdpa"}, {**base, "attention": "flash_attention_2"}]
        included, excluded, cohorts = select_protocol_cohort(records, synthetic_config())
        self.assertEqual(len(included), 1)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(len(cohorts), 2)


class AnalysisTests(unittest.TestCase):
    def _rows(self, missing_last: bool = True):
        rows = []
        combinations = list(
            itertools.product([1e-5, 2e-5], [0.5, 1.0], [1, 1000], [1, 3])
        )
        if missing_last:
            combinations = combinations[:-1]
        for index, (lr, alpha, top_k, first_n) in enumerate(combinations):
            rows.append(
                {
                    "lr": lr,
                    "alpha": alpha,
                    "top_k": top_k,
                    "first_n": first_n,
                    "config_id": f"config-{index}",
                    **synthetic_metric_values(lr, alpha, top_k, first_n),
                }
            )
        return rows

    def test_coverage_reports_missing_grid_cell(self):
        audit = coverage_audit(self._rows(), synthetic_config())
        self.assertEqual(audit["planned_combinations"], 16)
        self.assertEqual(audit["observed_configurations"], 15)
        self.assertEqual(len(audit["missing_combinations"]), 1)

    def test_em_target_is_used_in_pareto_dominance(self):
        rows = [
            {
                "name": "a",
                "model_utility": 0.7,
                "forget_quality": 0.8,
                "forget_Q_A_gibberish": 0.85,
                "exact_memorization": 0.6,
            },
            {
                "name": "b",
                "model_utility": 0.6,
                "forget_quality": 0.7,
                "forget_Q_A_gibberish": 0.8,
                "exact_memorization": 0.72,
            },
        ]
        front = pareto_front(rows, em_target=0.6)
        self.assertEqual([row["name"] for row in front], ["a"])

    def test_sparse_pairwise_ridge_fits_and_predicts_full_grid(self):
        result = fit_metric_model(self._rows(), "model_utility", synthetic_config())
        self.assertIn(result["diagnostics"]["status"], {"validated", "exploratory"})
        self.assertEqual(len(result["grid_predictions"]), 16)
        self.assertEqual(set(result["main_effects"]), set(FACTORS))
        self.assertEqual(len(result["pair_effects"]), 6)


class EndToEndTests(unittest.TestCase):
    def test_report_generation_with_incomplete_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            root = repo / "saves" / "unlearn"
            combinations = list(
                itertools.product([1e-5, 2e-5], [0.5, 1.0], [1, 1000], [1, 3])
            )[:-1]
            for lr, alpha, top_k, first_n in combinations:
                make_run(
                    root,
                    lr,
                    alpha,
                    top_k,
                    first_n,
                    synthetic_metric_values(lr, alpha, top_k, first_n),
                )
            output = repo / "report"
            manifest = run_analysis(root, synthetic_config(), output, repo)
            self.assertTrue((output / "report.html").is_file())
            self.assertTrue((output / "figures" / "coverage.png").is_file())
            self.assertTrue((output / "tables" / "observed_configurations.csv").is_file())
            self.assertEqual(manifest["counts"]["observed_configurations"], 15)
            report = (output / "report.html").read_text(encoding="utf-8")
            self.assertIn("PALU sparse four-factor analysis", report)
            self.assertIn("Predicted candidates are proposals", report)
            self.assertNotIn("https://", report)
            self.assertNotIn("http://", report)


if __name__ == "__main__":
    unittest.main()
