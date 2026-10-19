"""CPU-only checks of TR direction, sample matching, and partial-run export."""
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/palu"))
import plot_sequential_phases as analysis


def payload(losses, fq=.5, mu=.6):
    correct, wrong, scores = {}, {}, {}
    for index, (lc, lw) in enumerate(losses):
        key = str(index)
        correct[key] = {"avg_loss": lc}
        wrong[key] = {"avg_loss": [lw-0.1, lw+0.1]}
        scores[key] = {"score": math.exp(-lw) / (math.exp(-lc) + 1e-10)}
    return {"forget_Q_A_PARA_Prob": {"value_by_index": correct},
            "forget_Q_A_PERT_Prob": {"value_by_index": wrong},
            "forget_truth_ratio": {"value_by_index": scores},
            "forget_quality": {"agg_value": fq}, "model_utility": {"agg_value": mu}}


class SequentialPlots(unittest.TestCase):
    def test_signed_ks_direction_and_ties(self):
        high = analysis.signed_ks([3., 4.], [1., 2.])
        self.assertEqual((high["D_low"], high["D_high"], high["dominant_side"]), (0., 1., "high_TR"))
        low = analysis.signed_ks([1., 2.], [3., 4.])
        self.assertEqual((low["D_low"], low["D_high"], low["dominant_side"]), (1., 0., "low_TR"))
        crossing = analysis.signed_ks([1., 4.], [2., 3.])
        self.assertEqual(crossing["ks_d_reconstructed"], .5)
        self.assertTrue(crossing["cdf_has_both_signs"])
        self.assertEqual(analysis.signed_ks([1., 1.], [1., 1.])["ks_d_reconstructed"], 0)

    def test_geometric_wrong_candidates_and_no_clipping(self):
        native = payload([(3., 2.), (2., 4.)])
        native["forget_Q_A_PERT_Prob"]["value_by_index"] = dict(reversed(list(
            native["forget_Q_A_PERT_Prob"]["value_by_index"].items())))
        result = analysis.load_truth_ratios(native)
        self.assertGreater(result["0"]["truth_ratio"], 1)
        self.assertAlmostEqual(result["0"]["wrong_geometric_probability"], math.exp(-2.))
        self.assertAlmostEqual(result["0"]["log_truth_ratio"], math.log(result["0"]["truth_ratio"]))
        native["forget_truth_ratio"]["value_by_index"]["0"]["score"] = 0.9
        with self.assertRaisesRegex(ValueError, "TR reconstruction"):
            analysis.load_truth_ratios(native)

    def test_zero_is_retained_with_stable_log_and_nonfinite_is_rejected(self):
        native = payload([(2., 1000.), (3., 4.)])
        result = analysis.load_truth_ratios(native)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["0"]["truth_ratio"], 0)
        self.assertTrue(result["0"]["native_tr_underflow"])
        self.assertTrue(math.isfinite(result["0"]["log_truth_ratio"]))
        native["forget_truth_ratio"]["value_by_index"]["0"]["score"] = float("inf")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            analysis.load_truth_ratios(native)

    def test_sample_mismatch_rejected(self):
        native = payload([(2., 3.), (3., 2.)])
        del native["forget_Q_A_PERT_Prob"]["value_by_index"]["1"]
        with self.assertRaisesRegex(ValueError, "sample IDs"):
            analysis.load_truth_ratios(native)

    def test_partial_export_matches_ids_cache_and_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study / "frozen").mkdir(); (study / "run/checkpoint-13/evals").mkdir(parents=True)
            plan = dict(forget_updates=63, retain_updates=63, phase_epoch_size=200,
                        retain_epoch_size=200, learning_rate=1e-5)
            (study / "experiment.json").write_text(json.dumps(plan))
            native = payload([(3., 2.), (2., 4.)])
            (study / "frozen/retain_reference.json").write_text(json.dumps(native))
            (study / "run/checkpoint-13/evals/TOFU_EVAL.json").write_text(json.dumps(native))
            row = dict(step=13, forget_quality=.5, model_utility=.6, ks_d=0.,
                       raw_sensitive_loss=.3, raw_retain_nll=.4, raw_common_kl=.05,
                       evaluation_dir="/old/server/path/checkpoint-13/evals")
            history = study / "run/epoch_history.jsonl"
            history.write_text(json.dumps(row) + "\n" + '{"step":')
            report = analysis.analyze(study, plots=False)
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["completed_evaluation_steps"], [13])
            with (study / "analysis/truth_ratio_samples.csv").open() as stream:
                samples = list(csv.DictReader(stream))
            self.assertEqual([sample["sample_id"] for sample in samples], ["0", "1"])
            self.assertEqual({float(sample["delta_log_truth_ratio"]) for sample in samples}, {0.})
            row["forget_quality"] = .8
            history.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "cached forget_quality"):
                analysis.analyze(study, plots=False)

    def test_empty_live_study_and_malformed_complete_line(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory); (study / "frozen").mkdir()
            (study / "experiment.json").write_text(json.dumps(dict(
                forget_updates=63, retain_updates=63, phase_epoch_size=200, retain_epoch_size=3800)))
            (study / "frozen/retain_reference.json").write_text(json.dumps(payload([(2., 3.)])))
            report = analysis.analyze(study, plots=False)
            self.assertEqual(report["completed_evaluation_steps"], [])
            self.assertAlmostEqual(analysis.phase_epoch_at(64, analysis.read_json(study / "experiment.json")), 32/3800)
            history = study / "run/epoch_history.jsonl"; history.parent.mkdir()
            history.write_text('{"step":\n')
            with self.assertRaisesRegex(ValueError, "complete JSONL"):
                analysis.read_rows(history)


if __name__ == "__main__":
    unittest.main()
