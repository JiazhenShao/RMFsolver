import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'new_paper_calculations/26-10-01/isothermal_numeric_logT'))

class TrialTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('_logT_trial')

    def test_density_root_uses_local_brackets_and_memoization(self):
        seen = []
        def f(d):
            seen.append(d)
            return d-3.00001
        root = self.module().density_root(f, 3.01, (1., 5.5))
        self.assertAlmostEqual(root, 3.00001, places=8)
        self.assertLess(len(seen), 15)
        self.assertTrue(all(1 <= x <= 5.5 for x in seen))

    def test_invalid_probe_does_not_abort_valid_bracket(self):
        def f(d):
            if d > 4:
                raise ValueError('EOS invalid')
            return d-3
        self.assertAlmostEqual(self.module().density_root(f, 4.1, (1.,5.5)), 3., places=8)

    def test_no_root_has_explicit_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'bracket'):
            self.module().density_root(lambda d: 1., 3., (1.,5.5))

    def test_validator_rejects_unconverged_rms(self):
        m = self.module()
        result = dict(success=True, bvp_status=0, a_0plus_source='maximum',
            a_0plus_max_status='interior', a_0plus=.2, a_0plus_derived=.2,
            rate_model='exact_nonleptonic', diffusion_model='local_muB_fixed_T',
            composition_definition='nK_over_local_nB', current_definition='u_nK_minus_D_K_dnK_dx',
            jB=1., u_0minus=1e-6, boundary_residuals=np.zeros(3), compact_scale=1.,
            tail_residual_norm=0., momentum_flux_ratio=1., bvp_max_rms_residual=1e-5)
        self.assertTrue(m.valid_result(result, {'tol_bvp':1e-4}))
        result['bvp_max_rms_residual'] = .1
        self.assertFalse(m.valid_result(result, {'tol_bvp':1e-4}))

    def test_boundary_calls_tuple_interface_and_fixed_density_upstream(self):
        seen=[]
        def upstream(density,T,**kwargs):
            seen.append(density)
            return dict(P_0minus=density,muB_0minus=1000.)
        fake = SimpleNamespace(_validated_pnm_state_from_nB=upstream,
            _solve_muB_inf_at_muK0_for_given_Pi=lambda pressure,*a,**kw:1003.-pressure,
            _solve_interface_0plus_from_local_a_and_Pi=lambda a,pressure,*args,**kw:(1005.-pressure,0.))
        rmf = SimpleNamespace(phase_velocity=fake,RMFparameter=SimpleNamespace(paraQMCRMF3=None))
        with patch.dict(sys.modules,{'RMFsolver':rmf}):
            answer=self.module().boundary(.01,dict(lower=3.01,upper=4.99),1.)
        self.assertAlmostEqual(answer['lower'],3.,places=8)
        self.assertAlmostEqual(answer['upper'],5.,places=8)
        self.assertEqual(len(set(seen)),len(seen))
        self.assertLess(answer['upper_composition'],1.)

if __name__ == '__main__':
    unittest.main()
