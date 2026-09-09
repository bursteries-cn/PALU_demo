from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from representation_batching.report import build_report, collect_run, summarize


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def lines(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in records), encoding="utf-8")


def fixture(root, arm="R", seed=0, max_steps=-1, name=None):
    run = root / (name or f"{arm}-{seed}")
    meta = dict(type="metadata", method=arm, batch_order="random", seed=seed,
        world_size=2, per_device_batch_size=1, gradient_accumulation_steps=1,
        effective_batch_size=2, num_epochs=1, retain_size=4, sample_ids=[0,1],
        feature_key="block_22_question_last", feature_sha256="features")
    step = dict(type="step", epoch=0, optimizer_step=0,
        forget_indices=[0,1], retain_indices=[2,3], within_batch_cosine=.8)
    lines(run / "batch_audit" / f"{arm}.jsonl", [meta, step])
    for rank in range(2):
        records = [dict(epoch=0, optimizer_step=0, microstep=0,
            pairs=[dict(forget_index=rank, retain_index=rank+2)])]
        for kind in ("planned", "observed"):
            lines(run / "batch_audit" / f"rank-{rank}-{kind}-microbatches.jsonl", records)
    cfg = dict(model={"handler":"Llama", "model_args":{"pretrained_model_name_or_path":"Full"}},
        data={"forget":"forget05"}, trainer={"args":{"max_steps":max_steps,"seed":seed,"learning_rate":2e-5},
        "method_args":{"batch_manifest_path":"different-"+arm}})
    write(run / "resolved_config.json", cfg)
    write(run / "training_status.json", {"status":"completed", "global_step":1})
    lines(run / "training_diagnostics.jsonl", [{"optimizer_step":1,"npo_weight_mean":1.}])
    write(run / "evals/TOFU_SUMMARY.json", {"forget_quality":.2 if arm=="R" else .4,"model_utility":.6})
    provenance = {"status":"completed", "code_sha256":{"eval.py":"same"}, "config":{
        "seed":0, "model":{"model_args":{"pretrained_model_name_or_path":str(run)},
        "tokenizer_args":{"pretrained_model_name_or_path":str(run)}},
        "eval":{"tofu":{"metrics":{"forget_quality":{},"model_utility":{}},"output_dir":str(run / "evals")}}}}
    write(run / "evals/evaluation_provenance.json", provenance)
    return run


class ReportTests(unittest.TestCase):
    def test_pairing_and_html_csv_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, "R"); fixture(root,"S")
            result = build_report({"roots":[str(root)],"expected_seeds":[0,1]}, root / "out")
            self.assertEqual(len(result["runs"]),2)
            self.assertEqual(len(result["paired_deltas"]),2)
            self.assertAlmostEqual(result["paired_deltas"][0]["delta"],.2)
            self.assertTrue(any(r["status"]=="missing" for r in result["coverage"]))
            self.assertIn("NPO", (root / "out/index.html").read_text())
            self.assertTrue((root / "out/runs.csv").exists())

    def test_limited_run_not_eligible_even_with_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            run = fixture(Path(temp), max_steps=2)
            row, _, _ = collect_run(run,[],[])
            self.assertEqual(row["training_status"],"limited")
            self.assertFalse(row["eligible"])

    def test_missing_rank_and_mismatch_not_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            run = fixture(Path(temp))
            path = run / "batch_audit/rank-1-observed-microbatches.jsonl"
            path.unlink()
            self.assertFalse(collect_run(run,[],[])[0]["eligible"])
            lines(path,[{"wrong":"row"}])
            self.assertEqual(collect_run(run,[],[])[0]["audit"],"mismatch")

    def test_edited_planned_and_observed_logs_detected_against_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            run=fixture(Path(temp))
            for kind in ("planned","observed"):
                lines(run / f"batch_audit/rank-0-{kind}-microbatches.jsonl",[])
            self.assertEqual(collect_run(run,[],[])[0]["audit"],"mismatch")

    def test_duplicate_seed_excluded_from_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            paths=[fixture(root), fixture(root,name="retry"), fixture(root,"S")]
            rows=[collect_run(p,[],[])[0] for p in paths]
            warnings=[]
            aggregates,pairs=summarize(rows,"R/random",warnings)
            self.assertTrue(warnings)
            self.assertEqual(pairs,[])
            self.assertTrue(all(r["arm"]=="S/random" for r in aggregates))

    def test_partial_eval_and_wrong_checkpoint_not_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            run=fixture(Path(temp))
            path=run / "evals/evaluation_provenance.json"
            provenance=json.loads(path.read_text())
            provenance["status"]="started"
            write(path,provenance)
            self.assertFalse(collect_run(run,[],[])[0]["eligible"])
            provenance["status"]="completed"
            provenance["config"]["model"]["model_args"]["pretrained_model_name_or_path"]=str(run / "checkpoint-1")
            write(path,provenance)
            self.assertFalse(collect_run(run,[],[])[0]["eligible"])

    def test_protocol_changes_do_not_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            r=fixture(root); s=fixture(root,"S")
            path=s / "resolved_config.json"
            cfg=json.loads(path.read_text()); cfg["trainer"]["args"]["learning_rate"]=9e-5
            write(path,cfg)
            rows=[collect_run(p,[],[])[0] for p in (r,s)]
            self.assertEqual(summarize(rows,"R/random",[])[1],[])

    def test_multiple_external_results_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); run=fixture(root)
            extra=root / "external/TOFU_SUMMARY.json"; write(extra,{"forget_quality":.99})
            row,_,_=collect_run(run,[{"run_dir":str(run),"summary":str(extra)}],[])
            self.assertEqual(row["evaluation_status"],"ambiguous")
            self.assertFalse(row["eligible"])

    def test_corrupt_one_run_preserves_other_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture(root); bad=fixture(root,"S")
            (bad / "training_status.json").write_text("{")
            result=build_report({"roots":[str(root)]},root / "out")
            self.assertEqual(len(result["runs"]),1)
            self.assertEqual(len(result["warnings"]),1)

    def test_unproven_completion_and_absent_eval_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            run=fixture(Path(temp))
            (run / "training_status.json").unlink()
            row,_,_=collect_run(run,[],[])
            self.assertEqual(row["training_status"],"incomplete_or_unknown")
            write(run / "trainer_state.json",{"global_step":1})
            (run / "evals/evaluation_provenance.json").unlink()
            row,_,_=collect_run(run,[],[])
            self.assertEqual(row["training_status"],"completed")
            self.assertEqual(row["evaluation_status"],"unverified_or_partial")
            self.assertEqual(row["forget_quality"],.2)
            self.assertFalse(row["eligible"])

    def test_copied_server_run_matches_archived_output_path(self):
        with tempfile.TemporaryDirectory() as temp:
            run=fixture(Path(temp))
            config=json.loads((run / "resolved_config.json").read_text())
            config["paths"]={"output_dir":"/server/results/run-1"}
            write(run / "resolved_config.json",config)
            provenance=json.loads((run / "evals/evaluation_provenance.json").read_text())
            provenance["config"]["model"]["model_args"]["pretrained_model_name_or_path"]="/server/results/run-1"
            write(run / "evals/evaluation_provenance.json",provenance)
            self.assertTrue(collect_run(run,[],[])[0]["eligible"])

if __name__ == "__main__":
    unittest.main()
