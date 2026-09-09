from __future__ import annotations
import importlib.util
import json
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from representation_batching.pipeline import run_seed, commands, hash_file, load_settings, saved_model, run_command
from representation_batching.seed_comparison import select_cells, export_comparison, manual_rows
from representation_batching.evaluation_config import set_tofu_dataset_paths, validate_local_tofu_files


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")


def settings(root):
    return {"features":str(root/"features.npz"),"feature_key":"block_22_question_last","retain_size":3800,
        "training_gpus":"0,1","evaluation_gpu":"0","evaluation_batch_size":32,
        "model":"/models/Full","dataset":"/data/TOFU","learning_rate":2e-5,
        "retain_logs":"/eval/retain/TOFU_EVAL.json","classifier_model":"/models/detector",
        "manifest_root":str(root/"manifests"),"output_root":str(root/"runs"),"report_dir":str(root/"report")}


def item(seed,arm,value,protocol="a",status="completed",eligible=True):
    return {"seed":seed,"arm":arm+"/random","training_status":status,
        "evaluation_status":"verified" if eligible else "unverified_or_partial", "protocol":protocol,
        "evaluation_protocol":"eval1","eligible":eligible,"summary_path":"/example/TOFU_SUMMARY.json",
        "model_utility":value}


class ComparisonTests(unittest.TestCase):
    def test_table_missing_zero_and_duplicate_cells(self):
        rows=[item(1,"R",0),item(1,"S",.5),item(1,"S",.9),item(2,"D",.6),item(0,"P",.3,status="limited")]
        wide,selected,warnings=select_cells(rows,[0,1,2])
        r=next(r for r in wide if r['seed']==1 and r['metric']=='model_utility')
        self.assertEqual(r['R'],0)
        self.assertIsNone(r['S']); self.assertIsNone(r['P'])
        self.assertEqual(len(warnings),1)
        self.assertNotIn((0,'P'),selected)

    def test_unverified_is_visible_and_labelled(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            wide,warnings=export_comparison([item(0,'R',.6,eligible=False)],out,plots=False)
            self.assertIn('unverified_or_partial',(out/'seed_comparison.html').read_text())
            self.assertEqual(next(r for r in wide if r['metric']=='model_utility')['R'],.6)

    def test_manual_results_are_never_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'reported.csv'
            path.write_text('seed,arm,model_utility\n0,R,0.56\n')
            rows=manual_rows(path)
            self.assertFalse(rows[0]['eligible'])
            self.assertEqual(rows[0]['training_status'],'user_reported')


class PipelineTests(unittest.TestCase):
    def test_child_exit_code_and_log_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'stage.log'
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                run_command([sys.executable,'-c','print("test-stage"); raise SystemExit(7)'],path,Path(temp))
            self.assertEqual(caught.exception.returncode,7)
            self.assertIn('test-stage',path.read_text())

    def test_incomplete_model_shards_are_not_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            write(run/'config.json',{})
            write(run/'model.safetensors.index.json',{'weight_map':{'a':'part1.safetensors','b':'part2.safetensors'}})
            (run/'part1.safetensors').write_bytes(b'weights')
            self.assertFalse(saved_model(run))
            (run/'part2.safetensors').write_bytes(b'weights')
            self.assertTrue(saved_model(run))

    def fake_executor(self,config,seed,fail_evaluation_once=False):
        calls=[]; completed={}; failed=[False]
        def runner(command,log):
            calls.append(command)
            if 'build_batch_manifests.py' in command[1]:
                directory=Path(command[command.index('--output-dir')+1]); directory.mkdir(parents=True,exist_ok=True)
                for arm in ('R','S','D','P'): (directory/f'{arm}.jsonl').write_text('manifest')
            elif command[0]=='bash':
                path=Path(command[command.index('--output-dir')+1]); path.mkdir(parents=True)
                completed[str(path)]=(True,False)
            else:
                path=command[command.index('--run')+1]
                if fail_evaluation_once and not failed[0]:
                    failed[0]=True; raise ValueError('simulated evaluation failure')
                completed[path]=(True,True)
        return calls,runner,lambda path:completed.get(str(path),(False,False))

    def test_one_seed_then_restart_skips_all_completed_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=settings(Path(temp)); calls,runner,inspector=self.fake_executor(cfg,2)
            state=run_seed(cfg,2,'sig',runner=runner,inspector=inspector,refresher=lambda *_:None)
            self.assertEqual(len(calls),9)
            self.assertEqual(state['status'],'completed')
            self.assertEqual(sum(c[0]=='bash' for c in calls),4)
            self.assertTrue(all(c[c.index('--seed')+1]=='2' for c in calls if '--seed' in c))
            run_seed(cfg,2,'sig',runner=runner,inspector=inspector,refresher=lambda *_:None)
            self.assertEqual(len(calls),9)

    def test_failed_evaluation_resumes_without_retraining(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=settings(Path(temp)); calls,runner,inspector=self.fake_executor(cfg,1,True)
            with self.assertRaisesRegex(ValueError,'simulated'):
                run_seed(cfg,1,'sig',runner=runner,inspector=inspector,refresher=lambda *_:None)
            run_seed(cfg,1,'sig',runner=runner,inspector=inspector,refresher=lambda *_:None)
            self.assertEqual(sum(c[0]=='bash' for c in calls),4)
            self.assertEqual(sum('--run' in c for c in calls),5)

    def test_changed_protocol_and_foreign_manifests_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=settings(Path(temp)); calls,runner,inspector=self.fake_executor(cfg,1)
            run_seed(cfg,1,'sig',runner=runner,inspector=inspector,refresher=lambda *_:None)
            with self.assertRaisesRegex(ValueError,'配置/特征/代码'):
                run_seed(cfg,1,'different',runner=runner,inspector=inspector,refresher=lambda *_:None)
            folder=Path(cfg['manifest_root'])/'seed-2'; folder.mkdir()
            (folder/'R.jsonl').write_text('foreign')
            with self.assertRaisesRegex(ValueError,'其他来源'):
                run_seed(cfg,2,'sig',runner=runner,inspector=inspector,refresher=lambda *_:None)

    def test_nonzero_process_exit_prevents_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=settings(Path(temp)); calls=[]
            def runner(command,log):
                calls.append(command)
                if command[0]=='bash': raise ValueError('training failed')
            with self.assertRaisesRegex(ValueError,'training failed'):
                run_seed(cfg,3,'sig',runner=runner,inspector=lambda _: (False,False),refresher=lambda *_:None)
            self.assertFalse(any('--run' in c for c in calls))

    def test_settings_infer_sources_and_reject_bad_feature_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            feature=root/'artifacts/representation_batching/features-forget05/features.npz'
            feature.parent.mkdir(parents=True); feature.write_bytes(b'fake cache for metadata test')
            meta={'model':'org/Full','dataset':'org/TOFU','dataset_config':'forget05',
                'feature_keys':['block_22_question_last'],'feature_sha256':hash_file(feature)}
            write(feature.parent/'feature_manifest.json',meta)
            write(root/'detector/config.json',{})
            write(root/'retain.json',{'forget_Truth_Ratio':{'agg_value':1}})
            cfg=root/'pipeline.json'; write(cfg,{'classifier_model':'detector','retain_logs':'retain.json'})
            loaded=load_settings(cfg,root)
            self.assertEqual(loaded['model'],'org/Full')
            self.assertEqual(loaded['dataset'],'org/TOFU')
            feature.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'哈希'):
                load_settings(cfg,root)


class EvaluationConfigTests(unittest.TestCase):
    def test_override_nested_tofu_only_and_check_files(self):
        cfg={'pre_compute':{'x':{'datasets':{'TOFU_QA_forget':{'args':{'hf_args':{'path':'old','name':'forget05'}}},
            'OTHER':{'args':{'hf_args':{'path':'keep','name':'other'}}}}}}}
        with tempfile.TemporaryDirectory() as temp:
            set_tofu_dataset_paths(cfg,temp)
            entries=cfg['pre_compute']['x']['datasets']
            self.assertEqual(entries['TOFU_QA_forget']['args']['hf_args']['path'],temp)
            self.assertEqual(entries['OTHER']['args']['hf_args']['path'],'keep')
            with self.assertRaisesRegex(ValueError,'forget05'):
                validate_local_tofu_files(cfg,temp)
            (Path(temp)/'forget05.json').write_text('{}')
            validate_local_tofu_files(cfg,temp)

    def test_evaluation_command_forwards_path_and_device_independent_flags(self):
        spec=importlib.util.spec_from_file_location('evaluate_run',ROOT/'scripts/representation_batching/evaluate_run.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            write(run/'config.json',{})
            write(run/'resolved_config.json',{'model':{'handler':'Llama-3.1-8B-Instruct'},'forget_split':'forget05','holdout_split':'holdout05'})
            command=module.build_command(run,run/'retain.json','/local/TOFU','/local/detector',8)
            self.assertIn('+tofu_dataset_path=/local/TOFU',command)
            self.assertIn('eval.tofu.batch_size=8',command)
            self.assertIn('eval.tofu.overwrite=true',command)

if __name__=='__main__':
    unittest.main()
