from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from representation_batching.report import build_report, collect_run, summarize, LEGACY_MODEL_LOADER_SHA256, report_lock
from representation_batching.seed_comparison import select_cells


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
    def test_timestamp_and_nested_seed_layouts_are_scanned_once_by_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            old=fixture(root,'R',0,name='representation_npo_R_seed0_20260909-120000')
            new=fixture(root/'seed_runs/seed-7/S','S',7,name='attempt-0001')
            report=build_report({'roots':[str(root),str(root/'seed_runs')],'expected_seeds':[0,1,2]},root/'report')
            self.assertEqual(len(report['runs']),2)
            rows={r['run_dir']:r for r in report['runs']}
            self.assertEqual((rows[str(old)]['seed'],rows[str(old)]['arm']),(0,'R/random'))
            self.assertEqual((rows[str(new)]['seed'],rows[str(new)]['arm']),(7,'S/random'))
            self.assertEqual(rows[str(old)]['protocol'],rows[str(new)]['protocol'])
            wide,_,warnings=select_cells(report['runs'])
            self.assertFalse(warnings)
            self.assertEqual(next(r for r in wide if r['seed']==0 and r['metric']=='model_utility')['R'],.6)
            self.assertEqual(next(r for r in wide if r['seed']==7 and r['metric']=='model_utility')['S'],.6)
            self.assertTrue(any(r['seed']==7 for r in report['coverage']))

    def test_report_lock_excludes_other_processes_and_releases_after_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            code='''import fcntl,sys
with open(sys.argv[1], 'a') as handle:
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('blocked')
    else:
        print('acquired')
'''
            def attempt():
                return subprocess.check_output([sys.executable,'-c',code,str(out/'.report.lock')],text=True).strip()
            with self.assertRaisesRegex(ValueError,'test failure'):
                with report_lock(out):
                    self.assertEqual(attempt(),'blocked')
                    raise ValueError('test failure')
            self.assertEqual(attempt(),'acquired')

    def legacy_provenance(self, run):
        path=run/'evals/evaluation_provenance.json'
        provenance=json.loads(path.read_text())
        provenance['config']['model']['model_args']['torch_dtype']='bfloat16'
        original=json.loads(json.dumps(provenance))
        write(run/'evals/.hydra/config.yaml',provenance['config'])
        provenance['code_sha256']['model/__init__.py']=LEGACY_MODEL_LOADER_SHA256
        for key in ('pretrained_model_name_or_path','torch_dtype'):
            provenance['config']['model']['model_args'].pop(key)
        write(path,provenance)
        return original,provenance

    def test_known_loader_mutation_is_recovered_without_rewriting_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            run=fixture(Path(temp))
            original,legacy=self.legacy_provenance(run)
            before=(run/'evals/evaluation_provenance.json').read_bytes()
            row,_,_=collect_run(run,[],[])
            self.assertTrue(row['eligible'])
            self.assertIn('归档配置恢复',row['evaluation_note'])
            self.assertEqual(before,(run/'evals/evaluation_provenance.json').read_bytes())
            original['code_sha256']=legacy['code_sha256']
            write(run/'evals/evaluation_provenance.json',original)
            self.assertEqual(row['evaluation_protocol'],collect_run(run,[],[])[0]['evaluation_protocol'])

    def test_legacy_recovery_rejects_missing_conflicting_and_newer_archives(self):
        with tempfile.TemporaryDirectory() as temp:
            run=fixture(Path(temp)); self.legacy_provenance(run)
            archive=run/'evals/.hydra/config.yaml'
            provenance_path=run/'evals/evaluation_provenance.json'
            original=archive.read_text()
            archive.unlink()
            self.assertFalse(collect_run(run,[],[])[0]['eligible'])
            archive.write_text(original)
            os.utime(archive,ns=(1,1))
            changed=json.loads(original)
            changed['model']['model_args']['pretrained_model_name_or_path']=str(run/'checkpoint-1')
            write(archive,changed); os.utime(archive,ns=(1,1))
            self.assertFalse(collect_run(run,[],[])[0]['eligible'])
            archive.write_text(original)
            later=provenance_path.stat().st_mtime_ns+1_000_000_000
            os.utime(archive,ns=(later,later))
            self.assertFalse(collect_run(run,[],[])[0]['eligible'])

    def test_legacy_recovery_rejects_unknown_loader_and_changed_model_args(self):
        with tempfile.TemporaryDirectory() as temp:
            run=fixture(Path(temp)); _,legacy=self.legacy_provenance(run)
            legacy['code_sha256']['model/__init__.py']='unknown'
            write(run/'evals/evaluation_provenance.json',legacy)
            self.assertFalse(collect_run(run,[],[])[0]['eligible'])
            self.legacy_provenance(fixture(Path(temp)))
            archive=run/'evals/.hydra/config.yaml'
            changed=json.loads(archive.read_text())
            changed['model']['model_args']['attn_implementation']='different'
            write(archive,changed); os.utime(archive,ns=(1,1))
            self.assertFalse(collect_run(run,[],[])[0]['eligible'])

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
