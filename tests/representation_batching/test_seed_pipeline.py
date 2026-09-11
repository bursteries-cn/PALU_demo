from __future__ import annotations
import importlib.util
import json
import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from representation_batching.pipeline import (run_seed, commands, hash_file, load_settings,
    saved_model, discarded_model, discard_model_weights, run_command, inspect_run, save_seed_results, fingerprint,
    fingerprint_payload, payload_digest, compatible_legacy_signature, runtime_settings, build_parser)
from representation_batching.seed_comparison import select_cells, export_comparison, manual_rows
from representation_batching.evaluation_config import set_tofu_dataset_paths, validate_local_tofu_files
from representation_batching.report import (LEGACY_MODEL_LOADER_SHA256, METRICS,
    PORT_LAUNCHER_SHA256, LEGACY_LAUNCHER_SHA256, normalize_launch_hash)
from test_report import fixture


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
    def test_gpu_cli_overrides_and_automatic_evaluation_gpu_and_port(self):
        args=build_parser().parse_args(['2','--gpu','2,3'])
        raw={'training_gpus':'0,1','evaluation_gpu':'0'}
        values=runtime_settings(raw,{k:getattr(args,k) for k in ('training_gpus','evaluation_gpu','main_process_port')})
        self.assertEqual(values,{'training_gpus':'2,3','evaluation_gpu':'2','main_process_port':29502})
        args=build_parser().parse_args(['2','--gpus','4,5','--eval-gpu','5','--port','29604'])
        values=runtime_settings(raw,{k:getattr(args,k) for k in ('training_gpus','evaluation_gpu','main_process_port')})
        self.assertEqual(values,{'training_gpus':'4,5','evaluation_gpu':'5','main_process_port':29604})
        self.assertEqual(raw,{'training_gpus':'0,1','evaluation_gpu':'0'})

    def test_bash_dry_run_routes_overrides_without_writing_config_or_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); feature=root/'features.npz'; feature.write_bytes(b'feature metadata fixture')
            write(root/'feature_manifest.json',{'model':'org/Full','dataset':'org/TOFU',
                'dataset_config':'forget05','feature_keys':['block_22_question_last'],'feature_sha256':hash_file(feature)})
            write(root/'classifier/config.json',{})
            write(root/'retain.json',{'forget_truth_ratio':{'agg_value':1}})
            cfg=root/'pipeline.json'
            write(cfg,{'features':str(feature),'classifier_model':str(root/'classifier'),
                'retain_logs':str(root/'retain.json'),'output_root':str(root/'runs'),
                'manifest_root':str(root/'manifests'),'training_gpus':'0,1','evaluation_gpu':'0'})
            before=cfg.read_bytes()
            completed=subprocess.run(['bash',str(ROOT/'scripts/representation_batching/run_seed.sh'),
                '7','--config',str(cfg),'--gpu','2,3','--eval-gpu','3','--port','29602','--dry-run'],
                cwd=root,env={**os.environ,'PYTHON_BIN':sys.executable},capture_output=True,text=True,check=True)
            self.assertIn('训练 GPU=2,3; 评估 GPU=3; port=29602',completed.stdout)
            self.assertEqual(completed.stdout.count('--main-process-port 29602'),4)
            self.assertEqual(completed.stdout.count('--gpu 2,3'),4)
            self.assertEqual(completed.stdout.count('--gpu 3'),4)
            self.assertEqual(cfg.read_bytes(),before)
            self.assertFalse((root/'runs').exists())
            self.assertFalse((root/'manifests').exists())

    def test_invalid_gpu_layouts_and_ports_are_rejected(self):
        for value in ('0','0,1,2,3','1,1','01,1','-1,2','x,1','0,'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                runtime_settings({'training_gpus':value})
        for overrides in ({'evaluation_gpu':'2,3'},{'main_process_port':0},{'main_process_port':65536}):
            with self.subTest(overrides=overrides),self.assertRaises(ValueError):
                runtime_settings({},overrides)

    def test_gpu_and_port_changes_do_not_change_training_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); cfg=settings(root)
            before=fingerprint(cfg,root)
            cfg.update(training_gpus='2,3',evaluation_gpu='3',main_process_port=29502)
            self.assertEqual(before,fingerprint(cfg,root))
            _,train,evaluate=commands(cfg,2,'R',root/'run')
            self.assertEqual(train[train.index('--gpu')+1],'2,3')
            self.assertEqual(train[train.index('--main-process-port')+1],'29502')
            self.assertEqual(evaluate[evaluate.index('--gpu')+1],'3')
            cfg['learning_rate']=9e-5
            self.assertNotEqual(before,fingerprint(cfg,root))

    def test_shell_passes_gpu_and_port_to_accelerate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); manifest=root/'R.jsonl'; manifest.write_text('manifest')
            fake=root/'accelerate'
            fake.write_text('#!'+sys.executable+'\nimport json,os,sys\nprint(json.dumps({"args":sys.argv[1:],"gpus":os.environ.get("CUDA_VISIBLE_DEVICES")}))\n')
            fake.chmod(0o755)
            result=subprocess.run(['bash',str(ROOT/'scripts/representation_batching/run_npo_representation.sh'),
                '--manifest',str(manifest),'--gpu','2,3','--main-process-port','29502','--epochs','10','--no-save'],
                cwd=ROOT,env={**os.environ,'PATH':str(root)+os.pathsep+os.environ['PATH']},
                check=True,capture_output=True,text=True)
            payload=json.loads(result.stdout)
            self.assertEqual(payload['gpus'],'2,3')
            args=payload['args']; index=args.index('--main_process_port')
            self.assertEqual(args[index+1],'29502')
            self.assertLess(index,args.index('src/train.py'))
            self.assertIn('do_save=false',args)
            self.assertIn('trainer.args.num_train_epochs=10',args)

    def test_only_known_port_launcher_change_has_the_same_protocol_hash(self):
        name='scripts/representation_batching/run_npo_representation.sh'
        self.assertEqual(normalize_launch_hash(name,PORT_LAUNCHER_SHA256),LEGACY_LAUNCHER_SHA256)
        self.assertEqual(normalize_launch_hash(name,'changed optimizer flags'),'changed optimizer flags')
        # The epoch-capable launcher is deliberately not covered by the old
        # port-only exemption. Numerical cohort validation stays conservative.
        self.assertNotEqual(hash_file(ROOT/name),PORT_LAUNCHER_SHA256)
        self.assertEqual(normalize_launch_hash(name,hash_file(ROOT/name)),hash_file(ROOT/name))

    def test_epochs_cli_routes_to_both_stages_and_isolates_existing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); feature=root/'features.npz'; feature.write_bytes(b'feature fixture')
            write(root/'feature_manifest.json',{'model':'org/Full','dataset':'org/TOFU',
                'dataset_config':'forget05','feature_keys':['block_22_question_last'],'feature_sha256':hash_file(feature)})
            write(root/'classifier/config.json',{})
            write(root/'retain.json',{'forget_truth_ratio':{'agg_value':1}})
            cfg=root/'pipeline.json'
            write(cfg,{'features':str(feature),'classifier_model':str(root/'classifier'),
                'retain_logs':str(root/'retain.json'),'output_root':str(root/'runs'),
                'manifest_root':str(root/'manifests'),'report_dir':str(root/'report'),'num_epochs':5})
            before=cfg.read_bytes()
            default=load_settings(cfg,root,overrides={'num_epochs':3})
            ten=load_settings(cfg,root,overrides={'num_epochs':10})
            self.assertEqual(default['output_root'],str((root/'runs').resolve()))
            self.assertEqual(ten['output_root'],str((root/'runs/epochs-10').resolve()))
            self.assertEqual(ten['manifest_root'],str((root/'manifests/epochs-10').resolve()))
            self.assertEqual(ten['report_dir'],str((root/'report/epochs-10').resolve()))
            self.assertEqual(ten['scan_roots'],[ten['output_root']])
            self.assertNotEqual(fingerprint(default,root),fingerprint(ten,root))
            self.assertEqual(load_settings(cfg,root)['num_epochs'],5)
            for arm in ('R','S','D','P'):
                build,train,_=commands(ten,2,arm,root/'new-run',root)
                self.assertEqual(build[build.index('--num-epochs')+1],'10')
                self.assertEqual(train[train.index('--epochs')+1],'10')
                self.assertEqual(train[train.index('--model')+1],'org/Full')
                self.assertIn('epochs-10',build[build.index('--output-dir')+1])
            result=subprocess.run(['bash',str(ROOT/'scripts/representation_batching/run_seed.sh'),
                '2','--config',str(cfg),'--epochs','10','--dry-run'],
                env={**os.environ,'PYTHON_BIN':sys.executable},capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout.count('--num-epochs 10'),1)
            self.assertEqual(result.stdout.count('--epochs 10'),4)
            self.assertEqual(cfg.read_bytes(),before)
            self.assertFalse((root/'runs').exists())
            for value in (0,-1,1.5,True,'invalid'):
                with self.subTest(value=value),self.assertRaisesRegex(ValueError,'正整数'):
                    load_settings(cfg,root,overrides={'num_epochs':value})

    def test_default_commands_keep_three_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            build,train,_=commands(settings(root),0,'R',root/'run')
            self.assertEqual(build[build.index('--num-epochs')+1],'3')
            self.assertEqual(train[train.index('--epochs')+1],'3')

    def test_model_retention_cli_aliases(self):
        parser=build_parser()
        self.assertIsNone(parser.parse_args(['0']).keep_model_weights)
        self.assertTrue(parser.parse_args(['0','--keep-model']).keep_model_weights)
        for flag in ('--discard-model-after-eval','--no-keep-model','--no-save'):
            self.assertFalse(parser.parse_args(['0',flag]).keep_model_weights)

    def test_single_run_rejects_invalid_epochs_before_launch(self):
        launcher=str(ROOT/'scripts/representation_batching/run_npo_representation.sh')
        for value in ('0','-1','1.5','abc'):
            with self.subTest(value=value):
                result=subprocess.run(['bash',launcher,'--epochs',value],capture_output=True,text=True)
                self.assertEqual(result.returncode,2)
                self.assertIn('positive integer',result.stderr)

    def test_existing_r_with_legacy_provenance_is_reused_before_s_d_p(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); cfg=settings(root); seed_root=root/'runs/seed-2'
            run=fixture(seed_root/'R',arm='R',seed=2,name='attempt-0001')
            write(run/'config.json',{})
            write(run/'model_save_complete.json',{'status':'completed'})
            (run/'model.safetensors').write_bytes(b'weights')
            write(run/'evals/TOFU_SUMMARY.json',dict.fromkeys(METRICS,.5))
            path=run/'evals/evaluation_provenance.json'
            provenance=json.loads(path.read_text())
            provenance['config']['model']['model_args']['torch_dtype']='bfloat16'
            write(run/'evals/.hydra/config.yaml',provenance['config'])
            provenance['code_sha256']['model/__init__.py']=LEGACY_MODEL_LOADER_SHA256
            for key in ('pretrained_model_name_or_path','torch_dtype'):
                provenance['config']['model']['model_args'].pop(key)
            write(path,provenance)
            write(seed_root/'pipeline_state.json',{'seed':2,'signature':'sig','settings':cfg,
                'arms':{'R':{'run_dir':str(run),'attempts':[str(run)],'status':'evaluating'}}})
            manifests=root/'manifests/seed-2'; manifests.mkdir(parents=True)
            write(manifests/'pipeline_owner.json',{'signature':'sig'})
            for arm in ('R','S','D','P'): (manifests/f'{arm}.jsonl').write_text('manifest')
            calls=[]
            def runner(command,log):
                calls.append(command)
                if command[0]=='bash':
                    path=Path(command[command.index('--output-dir')+1])
                    fixture(path.parent,arm=path.parent.name,seed=2,name=path.name)
                    write(path/'config.json',{})
                    write(path/'model_save_complete.json',{'status':'completed'})
                    (path/'model.safetensors').write_bytes(b'weights')
                else:
                    path=Path(command[command.index('--run')+1])
                    write(path/'evals/TOFU_SUMMARY.json',dict.fromkeys(METRICS,.5))
            state=run_seed(cfg,2,'sig',runner=runner,refresher=lambda *_:None)
            self.assertEqual(state['status'],'completed')
            self.assertEqual(sum(c[0]=='bash' for c in calls),3)
            self.assertEqual(sum('--run' in c for c in calls),3)
            self.assertFalse(any(str(run) in c for c in calls))
            rows=json.loads((seed_root/'seed_results.json').read_text())['runs']
            self.assertTrue(all(r['model_utility']==.5 for r in rows))
            self.assertIn('归档配置恢复',rows[0]['evaluation_note'])

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

    def test_discard_weights_keeps_verified_evaluation_and_resume_state(self):
        with tempfile.TemporaryDirectory() as temp:
            run=fixture(Path(temp))
            write(run/'config.json',{})
            write(run/'model_save_complete.json',{'status':'completed'})
            (run/'model-00001-of-00002.safetensors').write_bytes(b'a'*11)
            (run/'model-00002-of-00002.safetensors').write_bytes(b'b'*13)
            write(run/'model.safetensors.index.json',{'weight_map':{
                'a':'model-00001-of-00002.safetensors','b':'model-00002-of-00002.safetensors'}})
            expected_removed=24+(run/'model.safetensors.index.json').stat().st_size
            summary=(run/'evals/TOFU_SUMMARY.json').read_bytes()
            row={'training_status':'completed','eligible':True,**dict.fromkeys(METRICS,.5)}
            with patch('representation_batching.pipeline.collect_run',return_value=(row,[],[])):
                self.assertEqual(inspect_run(run),(True,True))
                payload=discard_model_weights(run)
                self.assertEqual(payload['removed_bytes'],expected_removed)
                self.assertFalse(saved_model(run))
                self.assertTrue(discarded_model(run))
                self.assertEqual((run/'evals/TOFU_SUMMARY.json').read_bytes(),summary)
                self.assertTrue((run/'config.json').exists())
                self.assertEqual(inspect_run(run),(True,True))
                self.assertEqual(discard_model_weights(run)['removed_bytes'],expected_removed)

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

    def test_no_keep_policy_discards_each_arm_only_after_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=settings(Path(temp)); cfg['keep_model_weights']=False
            calls,runner,inspector=self.fake_executor(cfg,2)
            with patch('representation_batching.pipeline.discard_model_weights',
                       return_value={'removed_bytes':1024}) as discard:
                state=run_seed(cfg,2,'sig',runner=runner,inspector=inspector,refresher=lambda *_:None)
            self.assertEqual(discard.call_count,4)
            self.assertTrue(all(entry['keep_model_weights'] is False for entry in state['arms'].values()))
            self.assertTrue(all(entry['weight_cleanup']['removed_bytes']==1024 for entry in state['arms'].values()))

    def test_failed_evaluation_resumes_without_retraining(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=settings(Path(temp)); calls,runner,inspector=self.fake_executor(cfg,1,True)
            with self.assertRaisesRegex(ValueError,'simulated'):
                run_seed(cfg,1,'sig',runner=runner,inspector=inspector,refresher=lambda *_:None)
            self.assertEqual(sum(c[0]=='bash' for c in calls),4)
            state=json.loads((Path(cfg['output_root'])/'seed-1/pipeline_state.json').read_text())
            self.assertEqual(state['status'],'partial_failed')
            self.assertEqual(state['arms']['R']['status'],'evaluation_failed')
            self.assertTrue(all(state['arms'][a]['status']=='completed' for a in ('S','D','P')))
            run_seed(cfg,1,'sig',runner=runner,inspector=inspector,refresher=lambda *_:None)
            self.assertEqual(sum(c[0]=='bash' for c in calls),4)
            self.assertEqual(sum('--run' in c for c in calls),5)

    def test_changed_protocol_and_foreign_manifests_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=settings(Path(temp)); calls,runner,inspector=self.fake_executor(cfg,1)
            run_seed(cfg,1,'sig',runner=runner,inspector=inspector,refresher=lambda *_:None)
            with self.assertRaisesRegex(ValueError,'配置/特征/计算代码'):
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
            self.assertEqual(sum(c[0]=='bash' for c in calls),4)

    def test_corrupt_evaluation_does_not_repeat_successful_training(self):
        with tempfile.TemporaryDirectory() as temp:
            run=fixture(Path(temp))
            write(run/'config.json',{})
            write(run/'model_save_complete.json',{'status':'completed'})
            (run/'model.safetensors').write_bytes(b'weights')
            (run/'evals/TOFU_SUMMARY.json').write_text('{')
            self.assertEqual(inspect_run(run),(True,False))
            (run/'evals/TOFU_SUMMARY.json').unlink()
            self.assertEqual(inspect_run(run),(True,False))

    def test_seed_results_include_four_arms_metrics_paths_and_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); run=fixture(root)
            write(run/'evals/TOFU_SUMMARY.json',dict.fromkeys(
                ('forget_quality','model_utility','exact_memorization','forget_Q_A_gibberish'),0))
            write(run/'evals/TOFU_EVAL.json',{'examples':[]})
            state={'seed':0,'status':'partial_failed','arms':{
                'R':{'status':'completed','run_dir':str(run)},
                'S':{'status':'evaluation_failed','run_dir':str(run),'error':'failed eval'}}}
            save_seed_results(root,state)
            rows=json.loads((root/'seed_results.json').read_text())['runs']
            self.assertEqual([r['arm'] for r in rows],['R','S','D','P'])
            self.assertEqual(rows[0]['model_utility'],0)
            self.assertEqual(rows[0]['evaluation_path'],str(run/'evals/TOFU_EVAL.json'))
            self.assertIsNone(rows[1]['model_utility'])
            self.assertEqual(rows[1]['error'],'failed eval')
            self.assertEqual(rows[2]['status'],'pending')
            self.assertTrue((root/'seed_results.csv').exists())

    def test_real_inspection_continues_after_r_failure_and_resumes_only_r(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); cfg=settings(root); calls=[]; fail=[True]
            def runner(command,log):
                calls.append(command)
                if 'build_batch_manifests.py' in command[1]:
                    path=Path(command[command.index('--output-dir')+1]); path.mkdir(exist_ok=True)
                    for arm in ('R','S','D','P'): (path/f'{arm}.jsonl').write_text('manifest')
                elif command[0]=='bash':
                    path=Path(command[command.index('--output-dir')+1])
                    fixture(path.parent,arm=path.parent.name,seed=1,name=path.name)
                    write(path/'config.json',{})
                    write(path/'model_save_complete.json',{'status':'completed'})
                    (path/'model.safetensors').write_bytes(b'weights')
                    (path/'evals/TOFU_SUMMARY.json').unlink()
                else:
                    path=Path(command[command.index('--run')+1])
                    if path.parent.name=='R' and fail[0]:
                        fail[0]=False
                        (path/'evals/TOFU_SUMMARY.json').write_text('{')
                        raise subprocess.CalledProcessError(9,command)
                    write(path/'evals/TOFU_SUMMARY.json',dict.fromkeys(
                        ('forget_quality','model_utility','exact_memorization','forget_Q_A_gibberish'),.5))
            with self.assertRaisesRegex(ValueError,'四组已全部尝试'):
                run_seed(cfg,1,'sig',runner=runner,refresher=lambda *_:None)
            self.assertEqual(sum(c[0]=='bash' for c in calls),4)
            rows=json.loads((root/'runs/seed-1/seed_results.json').read_text())['runs']
            self.assertIsNone(rows[0]['model_utility'])
            self.assertTrue(all(r['model_utility']==.5 for r in rows[1:]))
            state=run_seed(cfg,1,'sig',runner=runner,refresher=lambda *_:None)
            self.assertEqual(state['status'],'completed')
            self.assertEqual(sum(c[0]=='bash' for c in calls),4)
            self.assertEqual(sum('--run' in c for c in calls),5)

    def test_interrupt_does_not_launch_remaining_arms(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=settings(Path(temp)); calls=[]
            def runner(command,log):
                calls.append(command)
                if command[0]=='bash': raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                run_seed(cfg,0,'sig',runner=runner,refresher=lambda *_:None)
            self.assertEqual(sum(c[0]=='bash' for c in calls),1)
            state=json.loads((Path(cfg['output_root'])/'seed-0/pipeline_state.json').read_text())
            self.assertEqual(state['arms']['R']['status'],'interrupted')
            self.assertEqual(state['arms']['S']['status'],'pending')

    def test_protocol_hash_ignores_reporting_but_detects_training_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); cfg=settings(root)
            path=root/'src/representation_batching/pipeline.py'
            path.parent.mkdir(parents=True); path.write_text('old controls')
            trainer=root/'src/trainer/unlearn.py'
            trainer.parent.mkdir(); trainer.write_text('training')
            before=fingerprint(cfg,root)
            path.write_text('new controls')
            self.assertEqual(before,fingerprint(cfg,root))
            cfg['keep_model_weights']=False
            self.assertEqual(before,fingerprint(cfg,root))
            trainer.write_text('different training')
            self.assertNotEqual(before,fingerprint(cfg,root))

    def test_legacy_signature_requires_matching_settings_and_numerical_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); cfg=settings(root)
            legacy=fingerprint_payload(cfg,root)
            legacy['code']['src/representation_batching/pipeline.py']='old hash'
            old_signature=payload_digest(legacy,False)
            original=fingerprint_payload
            def payload(settings,root,revision=None):
                return legacy if revision else original(settings,root)
            with patch('representation_batching.pipeline.fingerprint_payload',side_effect=payload):
                self.assertTrue(compatible_legacy_signature(old_signature,cfg,root))
                changed={**cfg,'learning_rate':9e-5}
                self.assertFalse(compatible_legacy_signature(old_signature,changed,root))

    def test_legacy_state_migration_keeps_models_and_manifest_ownership(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); cfg=settings(root)
            current_signature=fingerprint(cfg,root)
            legacy=fingerprint_payload(cfg,root)
            legacy['code']['src/representation_batching/pipeline.py']='old hash'
            old_signature=payload_digest(legacy,False)
            calls,runner,inspector=self.fake_executor(cfg,1)
            run_seed(cfg,1,old_signature,root,runner,inspector,lambda *_:None)
            original=fingerprint_payload
            def payload(settings,root,revision=None):
                return legacy if revision else original(settings,root)
            with patch('representation_batching.pipeline.fingerprint_payload',side_effect=payload):
                state=run_seed(cfg,1,current_signature,root,runner,inspector,lambda *_:None)
            self.assertEqual(len(calls),9)
            self.assertEqual(state['signature'],current_signature)
            owner=json.loads((root/'manifests/seed-1/pipeline_owner.json').read_text())
            self.assertEqual(owner['signature'],current_signature)

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
            self.assertFalse(loaded['keep_model_weights'])
            self.assertTrue(load_settings(cfg,root,{'keep_model_weights':True})['keep_model_weights'])
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
