"""Exercise the evaluator entry point with the actual config-consuming loader.

Only model construction and metric computation are stubbed; no GPU is needed.
"""
import ast
from contextlib import nullcontext
import json
import logging
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from representation_batching.pipeline import (
    EVAL_PROVENANCE_FIX_SHA256, LEGACY_EVAL_SHA256, hash_file, payload_digest)
from representation_batching.report import collect_run, METRICS
from test_report import fixture, write


class Config(dict):
    def __getattr__(self, key):
        return self[key]


def config(value):
    if isinstance(value,dict): return Config({k:config(v) for k,v in value.items()})
    if isinstance(value,list): return [config(v) for v in value]
    return value


def module(name, **values):
    result=ModuleType(name)
    result.__dict__.update(values)
    return result


class EvaluationProvenanceTests(unittest.TestCase):
    def test_entrypoint_retains_path_and_dtype_consumed_by_real_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            run=fixture(Path(temp))
            cfg=config({
                'seed':0,
                'model':{'handler':'Llama',
                    'model_args':{'pretrained_model_name_or_path':str(run),'torch_dtype':'bfloat16'},
                    'tokenizer_args':{'pretrained_model_name_or_path':str(run)},
                    'template_args':{}},
                'eval':{'tofu':{'output_dir':str(run/'evals'),'overwrite':True,
                    'metrics':dict.fromkeys(METRICS,{})}}})
            loaded=[]
            def construct(**kwargs):
                loaded.append(kwargs)
                return 'fake model'
            # Execute the real get_model/get_dtype bodies against fake model IO.
            tree=ast.parse((ROOT/'src/model/__init__.py').read_text())
            tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef)
                       and node.name in ('get_model','get_dtype')]
            env={'DictConfig':Config,'open_dict':lambda _:nullcontext(),
                 'torch':SimpleNamespace(float16='float16',bfloat16='bfloat16',float32='float32'),
                 'MODEL_REGISTRY':{'AutoModelForCausalLM':SimpleNamespace(from_pretrained=construct)},
                 'get_tokenizer':lambda _: 'fake tokenizer','hf_home':None,
                 'logger':logging.getLogger(__name__)}
            exec(compile(tree,'model-loader-under-test','exec'),env)
            def evaluate(**kwargs):
                self.assertEqual(kwargs['model'],'fake model')
                write(run/'evals/TOFU_SUMMARY.json',dict.fromkeys(METRICS,.5))
            evaluator=SimpleNamespace(evaluate=evaluate)
            omega=SimpleNamespace(to_container=lambda obj,resolve=True:json.loads(json.dumps(obj)))
            modules={
                'hydra':module('hydra',main=lambda **kwargs:lambda func:func),
                'omegaconf':module('omegaconf',DictConfig=Config,OmegaConf=omega),
                'model':module('model',get_model=env['get_model']),
                'evals':module('evals',get_evaluators=lambda _: {'tofu':evaluator}),
                'trainer.utils':module('trainer.utils',seed_everything=lambda _:None)}
            with patch.dict(sys.modules,modules):
                entrypoint=runpy.run_path(str(ROOT/'src/eval.py'))['main']
                entrypoint(cfg)
            self.assertNotIn('pretrained_model_name_or_path',cfg.model.model_args)
            self.assertNotIn('torch_dtype',cfg.model.model_args)
            self.assertEqual(loaded[0]['pretrained_model_name_or_path'],str(run))
            provenance=json.loads((run/'evals/evaluation_provenance.json').read_text())
            recorded=provenance['config']['model']['model_args']
            self.assertEqual(recorded['pretrained_model_name_or_path'],str(run))
            self.assertEqual(recorded['torch_dtype'],'bfloat16')
            self.assertEqual(provenance['status'],'completed')
            self.assertTrue(collect_run(run,[],[])[0]['eligible'])

    def test_only_exact_metadata_fix_is_compatible_with_previous_protocol(self):
        self.assertEqual(hash_file(ROOT/'src/eval.py'),EVAL_PROVENANCE_FIX_SHA256)
        old=payload_digest({'code':{'src/eval.py':LEGACY_EVAL_SHA256}})
        fixed=payload_digest({'code':{'src/eval.py':EVAL_PROVENANCE_FIX_SHA256}})
        changed=payload_digest({'code':{'src/eval.py':'different evaluation code'}})
        self.assertEqual(old,fixed)
        self.assertNotEqual(old,changed)


if __name__=='__main__':
    unittest.main()
