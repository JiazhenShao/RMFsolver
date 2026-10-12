import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]/'new_paper_calculations/26-10-01/isothermal_numeric_logT'
sys.path.insert(0,str(WORK))

class RunnerTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('run_isothermal_numerical')

    def test_cli_has_lightweight_modes(self):
        args = self.module().parser().parse_args(['--dry-run','--workers','2'])
        self.assertTrue(args.dry_run)
        self.assertEqual(args.workers,2)
        self.assertFalse(args.smoke)

    def test_boundary_checkpoint_precedes_progress_and_is_not_repeated(self):
        from _logT_engine import ContourRun
        from _logT_common import load
        import numpy as np
        inputs = dict(points=[],n0=1e6,fingerprint='test',warm={
            'inner_boundary':np.array([[3,.01],[2.9,20]]),
            'outer_boundary':np.array([[5.3,.01],[5.2,20]])})
        with tempfile.TemporaryDirectory() as d:
            run = ContourRun(d,inputs,{'test':1})
            observed=[]
            run.boundary_saved=lambda row: observed.append(load(run.output)['boundaries'][float(.01).hex()]['status'])
            with patch('_logT_trial.bounded_call',return_value=dict(success=True,status='success',lower=3.,upper=5.3)) as call:
                self.module().prepare_boundaries(run,[.01],1)
                self.module().prepare_boundaries(run,[.01],1)
            self.assertEqual(call.call_count,1)
            self.assertEqual(observed,['success'])

    def test_smoke_checkpoint_is_separate(self):
        args = self.module().parser().parse_args(['--smoke'])
        self.assertEqual(self.module().output_directory(args),WORK/'smoke')

    def test_interrupted_boundary_gets_only_one_additional_attempt(self):
        m=self.module()
        self.assertTrue(m.boundary_pending({'status':'interrupted','attempts':1}))
        self.assertFalse(m.boundary_pending({'status':'interrupted','attempts':2}))
        self.assertFalse(m.boundary_pending({'status':'trial_timeout','attempts':1}))

if __name__ == '__main__':
    unittest.main()
