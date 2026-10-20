"""Launcher dry-run must freeze an explicit opt-in, with backward-compatible defaults."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class LauncherTests(unittest.TestCase):
    def launch(self, value=None):
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)/"future-study"
            env=dict(os.environ, MODE="dry-run", PYTHON_BIN=sys.executable, SEEDS="103",
                     STUDY_DIR=str(destination), SOURCE_STUDY=str(Path(folder)/"unused-source"))
            env.pop("ZERO_WEIGHT_RETAIN_NO_GRAD",None)
            if value is not None: env["ZERO_WEIGHT_RETAIN_NO_GRAD"]=value
            result=subprocess.run(["bash",str(ROOT/"scripts/palu/run_forget_until_threshold.sh")],
                                  env=env,text=True,capture_output=True)
            self.assertFalse(destination.exists())
            return result

    def test_default_and_explicit_zero_keep_native_backward(self):
        for value in (None,"0"):
            result=self.launch(value)
            self.assertEqual(result.returncode,0,result.stderr)
            payload,_=json.JSONDecoder().raw_decode(result.stdout[result.stdout.index("{"):])
            self.assertIs(payload["zero_weight_retain_no_grad"],False)

    def test_explicit_one_freezes_opt_in(self):
        result=self.launch("1")
        self.assertEqual(result.returncode,0,result.stderr)
        payload,_=json.JSONDecoder().raw_decode(result.stdout[result.stdout.index("{"):])
        plan=payload
        self.assertIs(plan["zero_weight_retain_no_grad"],True)
        self.assertEqual(plan["phase_weights"],{"forget":[1.,0.,2.]})
        self.assertEqual(plan["effective_batch"],32)
        self.assertEqual(plan["seed"],103)

    def test_invalid_values_fail_before_preparation(self):
        for value in ("true","false","2","0.0","-1"):
            result=self.launch(value)
            self.assertEqual(result.returncode,2)
            self.assertIn("must be 0 or 1",result.stderr)
            self.assertNotIn('"plan"',result.stdout)

    def test_train_cannot_override_an_existing_frozen_plan(self):
        result=subprocess.run([sys.executable,str(ROOT/"scripts/palu/forget_until_threshold.py"),
            "--mode","train","--study","/unused-frozen-study","--zero-weight-retain-no-grad"],
            text=True,capture_output=True)
        self.assertEqual(result.returncode,2)
        self.assertIn("use the frozen plan",result.stderr)

if __name__=="__main__": unittest.main()
