"""No BVP calls or live EOS work in these tests."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'new_paper_calculations/26-10-14'))


class CoreMfpTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'new_paper_calculations/26-10-14/isothermal_core_mfp.py').exists())
        return importlib.import_module('isothermal_core_mfp')

    def test_core_formula_and_cold_trend(self):
        m=self.module()
        self.assertAlmostEqual(m.core_ratio(.5, 2., 3.), 1.5*np.sqrt(6.))
        self.assertGreater(m.core_ratio(.5, 100., 3.),m.core_ratio(.5, 2., 3.))
        for a in (0.,1.,-1.):
            with self.assertRaises(ValueError): m.core_ratio(a,2.,3.)

    def test_temperature_root_and_failures(self):
        m=self.module()
        def state(T):
            return dict(status='interior',ratio=1.e-4*(T/.003)**(-5/6),T_0minus=T)
        result=m.temperature_root(state,1.e-4, temperature_limits=(1.e-8,1.), reference_temperature=.01)
        self.assertEqual(result['status'],'success')
        self.assertLess(abs(result['root']['T_0minus']/.003-1),1.e-6)
        self.assertLessEqual(result['relative_ratio_residual'],1.e-6)
        self.assertGreater(len(result['evaluations']),1)
        failed=m.temperature_root(lambda T:dict(status='stable'),1.e-4)
        self.assertEqual(failed['status'],'outside_domain')
        broken=m.temperature_root(lambda T:(_ for _ in ()).throw(RuntimeError('EOS failed')),1.e-4)
        self.assertEqual(broken['status'],'failed')

    def test_budget_window_and_gap(self):
        m=self.module()
        outside=m.temperature_root(lambda T:dict(status='interior',ratio=2.e-4,T_0minus=T),1.e-4,
                                   temperature_limits=(.001,.1),max_evaluations=20)
        self.assertNotEqual(outside['status'],'success')
        rows=[dict(status='success',nB_0minus_over_n0=d,root=dict(T_0minus=t)) for d,t in ((3.1,.01),(3.2,.02),(3.3,.03))]
        rows[1]['status']='failed'
        self.assertEqual([len(s) for s in m.segments(rows)],[1,1])

    def test_save_before_progress_and_resume(self):
        m=self.module(); calls=[]; physics=dict(n0=1.,B_one_forth=189.1565957288247,ms=0.,NM_type='PNM',upB=5000)
        def solve(density):
            calls.append(density)
            return dict(status='success',nB_0minus_over_n0=density,root=dict(T_0minus=.01),evaluations=[])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'core.npy'
            def progress(row):
                saved=np.load(path,allow_pickle=True).item()
                self.assertEqual(saved['rows'][float(row['nB_0minus_over_n0']).hex()]['status'],'success')
            for _ in range(2):
                result=m.calculate((3.1,3.2),physics,path,solver=solve,progress=progress,show_progress=False)
            self.assertEqual(len(calls),2)
            self.assertEqual(len(result['rows']),2)
            with self.assertRaises(ValueError):
                m.calculate((3.1,3.2),physics,path,fraction=.1,solver=solve,show_progress=False)


if __name__=='__main__': unittest.main()
