"""Lightweight checks for the core figure's static allowed-region shading."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'new_paper_calculations/26-10-14'))


class CoreRegionsTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'new_paper_calculations/26-10-14/isothermal_core_regions.py').exists())
        return importlib.import_module('isothermal_core_regions')

    def test_reuse_resume_and_save_before_progress(self):
        module=self.module()
        calls=[]
        saved=dict(status='success',T_0minus=.01,lower=3.,upper=5.3)
        inputs=dict(physics={},source_hashes={},boundary_cache={float(.01).hex():saved})
        def solve(T,seeds):
            calls.append(T)
            self.assertEqual(seeds['lower'],3.)
            return dict(status='success',T_0minus=T,lower=3.,upper=5.3)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bounds.npy'
            def progress(row):
                result=np.load(path,allow_pickle=True).item()
                self.assertEqual(result['rows'][float(row['T_0minus']).hex()]['status'],'success')
            for _ in range(2):
                result=module.calculate(inputs,(1.e-4,.001,.01),path,solver=solve,progress=progress,show_progress=False)
            self.assertEqual(calls,[1.e-4,.001])
            self.assertEqual(result['rows'][float(.01).hex()],saved)
            with self.assertRaises(ValueError):
                module.calculate(inputs,(1.e-4,.01),path,solver=solve,show_progress=False)

    def test_geometry_breaks_at_failures_and_covers_requested_window(self):
        module=self.module()
        rows=[dict(status='success',T_0minus=T,lower=3.,upper=5.3) for T in (1.e-4,.001,.01,1.)]
        parts=module.segments(rows)
        np.testing.assert_allclose(parts[0][:,0],[1.e-4,.001,.01,1.])
        rows[1]['status']='failed'
        self.assertEqual([len(p) for p in module.segments(rows)],[1,2])

    def test_notebook_draws_both_regions_without_changing_core_line(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import nbformat
        import isothermal_core_mfp
        book=nbformat.read(ROOT/'26-10-14.ipynb',as_version=4)
        source=book.cells[8].source
        self.assertTrue(source.startswith('# Tunable plotting parameters'))
        self.assertNotIn('.calculate(',source)
        roots=[dict(status='success',nB_0minus_over_n0=d,root=dict(T_0minus=T))
               for d,T in ((3.1,.01),(4.,.1),(5.2,.15))]
        regions=[dict(status='success',T_0minus=T,lower=3.,upper=5.3) for T in (1.e-4,.01,1.)]
        with tempfile.TemporaryDirectory() as temp:
            namespace=dict(isothermal_core_mfp=isothermal_core_mfp,isothermal_core_regions=self.module(),
                core_result=dict(identity=dict(fraction=1.e-4),rows=dict(enumerate(roots))),
                core_regions=dict(rows=dict(enumerate(regions))),
                CORE_OUTPUT_DIRECTORY=Path(temp),CORE_ROOT=Path(temp))
            exec(compile(source,'core-plot','exec'),namespace)
            ax=namespace['core_axis']
            self.assertEqual(len(ax.collections),2)
            for collection,limits in zip(ax.collections,((1.,3.),(5.3,5.5))):
                vertices=collection.get_paths()[0].vertices
                np.testing.assert_allclose([min(vertices[:,0]),max(vertices[:,0])],limits)
                np.testing.assert_allclose([min(vertices[:,1]),max(vertices[:,1])],[1.e-4,1.])
            np.testing.assert_allclose(ax.lines[-1].get_xydata(),[[3.1,.01],[4.,.1],[5.2,.15]])
            plt.close(namespace['core_figure'])


if __name__=='__main__': unittest.main()
