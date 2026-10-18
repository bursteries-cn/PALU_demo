"""CPU checks for source identity, predeclared intervals and safe output cleanup."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/palu'))
import overshoot_seed_replicates as control
spec = importlib.util.spec_from_file_location('epoch_artifacts', ROOT / 'src/trainer/palu_epoch_artifacts.py')
artifacts = importlib.util.module_from_spec(spec); spec.loader.exec_module(artifacts)


class Protocol(unittest.TestCase):
    def test_original_configs_and_primary_intervals(self):
        configs, provenance = control.sources()
        self.assertEqual(set(configs), {'P336', 'P421', 'P386'})
        self.assertEqual({k: v['primary_anchor_step'] for k,v in provenance['configs'].items()},
                         {'P336':25, 'P421':12, 'P386':12})
        self.assertTrue(all(v['trainer']['args']['seed'] == 0 for v in configs.values()))

    def test_cleanup_requires_receipt_and_never_deletes_final_or_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder = root / 'checkpoint-12'; folder.mkdir()
            weight = folder / 'model.safetensors'; weight.write_text('keep')
            with self.assertRaises(ValueError): artifacts.prune_weights(root,12,60)
            self.assertTrue(weight.exists())
            (folder / 'evals').mkdir(); artifacts.write_json(folder / 'evals/verification.json', {'status':'verified'})
            other = root / 'external.safetensors'; other.write_text('outside')
            weight.unlink(); weight.symlink_to(other)
            with self.assertRaises(ValueError): artifacts.prune_weights(root,12,60)
            self.assertTrue(other.exists())
            self.assertEqual(artifacts.prune_weights(root,12,12), [])

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('scipy'), 'numpy/scipy unavailable')
    def test_complete_observation_is_validated_before_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder = root / 'checkpoint-2/evals'; folder.mkdir(parents=True)
            reference = root / 'reference.json'
            native = {'forget_truth_ratio':{'value_by_index':{str(i):{'score':1/(1+1e-10)} for i in range(2)}},
                      'forget_Q_A_PARA_Prob':{'value_by_index':{str(i):{'avg_loss':0.} for i in range(2)}},
                      'forget_Q_A_PERT_Prob':{'value_by_index':{str(i):{'avg_loss':[0.]*5} for i in range(2)}},
                      'forget_quality':{'agg_value':1.}, 'model_utility':{'agg_value':.5}}
            for prefix,n in [('retain',400),('ra',100),('wf',117)]:
                for suffix in ['Q_A_Prob','Q_A_PARA_Prob','Q_A_PERT_Prob','Q_A_Prob_normalised','Q_A_ROUGE','Truth_Ratio']:
                    native[f'{prefix}_{suffix}']={'value_by_index':{str(i):{'score':.5} for i in range(n)}}
            artifacts.write_json(reference, native); artifacts.write_json(folder / 'TOFU_EVAL.json',native)
            fixed=[]
            for split,n in [('forget',2),('retain',3)]:
                for i in range(n):
                    fixed.append(dict(step=2,split=split,dataset_index=i,
                        raw_sensitive_loss=1. if split=='forget' else None, sensitive_token_count=1 if split=='forget' else None,
                        raw_common_kl=2. if split=='forget' else None, common_token_count=1 if split=='forget' else None,
                        raw_retain_nll=3. if split=='retain' else None, retain_token_count=1 if split=='retain' else None))
            history=root / 'fixed_loss_samples.jsonl'
            history.write_text(''.join(json.dumps(r)+'\n' for r in fixed[:-1]))
            weight=folder.parent / 'model.safetensors'; weight.write_text('temporary')
            row=dict(step=2,forget_count=2,retain_count=3,forget_quality=1.,model_utility=.5,
                     raw_sensitive_loss=1.,raw_common_kl=2.,raw_retain_nll=3.,reference_sha256='test')
            with self.assertRaises(ValueError): artifacts.finalize_observation(root,row,reference,4)
            self.assertTrue(weight.exists())
            history.write_text(''.join(json.dumps(r)+'\n' for r in fixed))
            artifacts.finalize_observation(root,row,reference,4)
            self.assertFalse(weight.exists()); self.assertTrue((folder/'truth_ratios.csv').is_file())
            self.assertEqual(json.loads((folder/'verification.json').read_text())['status'],'verified')


if __name__ == '__main__': unittest.main()
