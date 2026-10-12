"""Bounded new numerical dots; no live BVP in the unit tests."""
from pathlib import Path
import importlib
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'new_paper_calculations/26-10-02'


class ColdPointsTests(unittest.TestCase):
    def module(self):
        self.assertTrue((HERE/'notebook_isothermal_cold.py').exists(),'Missing cold numerical helper')
        sys.path.insert(0,str(HERE))
        return importlib.import_module('notebook_isothermal_cold')

    def inputs(self):
        return dict(n0=1.,physics=dict(B_one_forth=189.1565957288247,ms=0.,NM_type='PNM',upB=5000),
            fingerprint='fixture',source_hashes={},profiles=[],
            boundaries={.01:dict(lower=3.,upper=5.4)},
            seeds={(500.,.01):3.006})

    def result(self,p):
        v=(p['nB_0minus_over_n0']-3.)*1.e5
        return dict(success=True,validated=True,velocity_m_s=v,jB=p['nB_0minus']*v/299792458.,
            bvp_status=0,bvp_max_rms_residual=1.e-6,boundary_residuals=np.zeros(3),
            endpoint_flux_residual_max=1.e-10,profile_eos_residual_max=1.e-10,
            profile_physical=True,a_0plus=.2,a_0plus_derived=.2,a_0plus_source='maximum',
            a_0plus_max_status='interior',rate_model='exact_nonleptonic',
            diffusion_model='local_muB_fixed_T',composition_definition='nK_over_local_nB',
            current_definition='u_nK_minus_D_K_dnK_dx',u_0minus=v/299792458.,
            momentum_flux_ratio=1.e-7,tail_residual_norm=1.e-10)

    def test_numerical_roots_resume_and_save_precedes_progress(self):
        m=self.module();calls=[];saved=[]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'cold.npy'
            def evaluate(p,**kwargs):
                calls.append(p.copy());return self.result(p)
            def notified(event):
                data=np.load(path,allow_pickle=True).item()
                self.assertEqual(data['evaluations'][-1]['status'],event['status'])
                saved.append(event)
            payload=m.calculate(self.inputs(),path,temperatures=(.01,),levels=(500.,),workers=1,
                evaluator=evaluate,evaluation_saved=notified,show_progress=False)
            self.assertEqual(len(saved),len(calls))
            root=next(iter(payload['targets'].values()))['root']
            self.assertLessEqual(abs(root['velocity_m_s']/500.-1),.005)
            self.assertNotEqual(root['nB_0minus_over_n0'],3.006)
            n=len(calls)
            m.calculate(self.inputs(),path,temperatures=(.01,),levels=(500.,),workers=1,
                evaluator=evaluate,show_progress=False)
            self.assertEqual(len(calls),n)
            with self.assertRaises(ValueError):
                m.calculate(self.inputs(),path,temperatures=(.01,),levels=(500.,),workers=1,
                    max_evaluations=6,evaluator=evaluate,show_progress=False)

    def test_failed_and_interrupted_work_is_bounded_and_not_plotted(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'cold.npy'
            def interrupt(p,**kwargs): raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                m.calculate(self.inputs(),path,temperatures=(.01,),levels=(500.,),workers=1,
                    evaluator=interrupt,show_progress=False)
            result=m.calculate(self.inputs(),path,temperatures=(.01,),levels=(500.,),workers=1,
                evaluator=lambda p,**kw:dict(success=False,status='timeout'),show_progress=False)
            self.assertEqual(result['evaluations'][0]['status'],'interrupted')
            self.assertLessEqual(len(result['evaluations']),4)
            self.assertEqual(len(m.new_points(result,500.)),0)

    def test_distant_continuation_never_overrides_layer_initializer(self):
        m=self.module();p=dict(T_0minus=.01,nB_0minus_over_n0=3.01,nB_0minus=3.01)
        near=dict(T_0minus=.01,nB_0minus=3.011,nB_0minus_over_n0=3.011,lower=3.,jB=1.)
        far=dict(near,T_0minus=1.)
        self.assertIsNone(m.nearby_profile([far],p,3.))
        self.assertEqual(m.nearby_profile([far,near],p,3.),near)

    def test_nearly_on_target_density_uses_small_numerical_correction(self):
        m=self.module()
        self.assertTrue(hasattr(m,'correct_density'),'Missing local cold density correction')
        samples=[dict(nB_0minus_over_n0=3.0147,velocity_m_s=489.26)]
        next_density=m.correct_density(samples,500.,(3.,5.3))
        self.assertTrue(3.0147<next_density<3.016)
        bracket=samples+[dict(nB_0minus_over_n0=3.016,velocity_m_s=540.)]
        self.assertTrue(3.0147<m.correct_density(bracket,500.,(3.,5.3))<3.016)

    def test_validation_rejects_unconverged_or_bad_flux_even_when_success_true(self):
        m=self.module();r=self.result(dict(nB_0minus_over_n0=3.005,nB_0minus=3.005))
        self.assertTrue(m.valid_result(r,1.e-4))
        for bad in (dict(bvp_status=1),dict(endpoint_flux_residual_max=1.e-2),
                    dict(profile_physical=False),dict(bvp_max_rms_residual=np.nan)):
            self.assertFalse(m.valid_result(dict(r,**bad),1.e-4))

    def test_new_cell_precedes_log_plot_and_plot_never_calls_solver(self):
        import nbformat
        nb=nbformat.read(ROOT/'26-10-02.ipynb',as_version=4)
        index=next(i for i,c in enumerate(nb.cells) if "EXPORT_STEM = 'isothermal-analytic-numerical-logT'" in c.source)
        self.assertIn('cold_numerical_extra',nb.cells[index-1].source)
        self.assertIn('notebook_isothermal_cold.calculate(',nb.cells[index-1].source)
        self.assertIn('HIGH_COLD_LEVELS = (10000., 50000., 500000.)',nb.cells[index-1].source)
        self.assertIn("'isothermal-cold-numerical-high-speed.npy'",nb.cells[index-1].source)
        self.assertIn('HIGH_COLD_FALLBACK_TEMPERATURES = (0.09779327685429287,)',nb.cells[index-1].source)
        self.assertNotIn('.calculate(',nb.cells[index].source)
        self.assertNotIn('solve_front_isothermal(',nb.cells[index].source)

    def test_input_bundle_uses_matching_boundaries_and_profiles_only_as_seeds(self):
        m=self.module();a=m.analytic.load_reference(ROOT)
        inputs=m.load_inputs(a)
        self.assertEqual(len(inputs['seeds']),6)
        self.assertEqual(len(inputs['profiles']),3)
        self.assertNotIn('xi',inputs['physics'])
        self.assertNotIn('targets',inputs)
        self.assertEqual(len(inputs['boundaries']),3)

    def test_plot_adds_only_new_accepted_dots_and_preserves_old_numerical_data(self):
        import contextlib,io,nbformat
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        m=self.module();nb=nbformat.read(ROOT/'26-10-02.ipynb',as_version=4)
        p=dict(T_0minus=.01,nB_0minus_over_n0=3.005,nB_0minus=3.005)
        root=dict(p,velocity_m_s=500.,result=self.result(p))
        extra=dict(targets={'good':dict(level=500.,status='success',root=root),
                            'bad':dict(level=2000.,status='unresolved',root=None)})
        high=dict(targets={str(level):dict(level=level,status='success',
            root=dict(p,nB_0minus_over_n0=density,velocity_m_s=level,
                      result=dict(self.result(p),velocity_m_s=level)))
            for level,density in ((10000.,3.1),(50000.,3.4),(500000.,4.5))})
        space={'__name__':'__main__','cold_numerical_extra':extra,
               'cold_numerical_high':high,'notebook_isothermal_cold':m,
               'cold_numerical_high_fallback':dict(targets={'another':dict(level=50000.,status='success',
                   root=dict(p,T_0minus=.09779327685429287,nB_0minus_over_n0=4.3,
                             velocity_m_s=50000.,result=dict(self.result(p),velocity_m_s=50000.)))})}
        with tempfile.TemporaryDirectory(dir=ROOT/'local_analysis_results') as directory, contextlib.redirect_stdout(io.StringIO()):
            exec(compile(nb.cells[4].source.replace('CALCULATE_ANALYTICAL = True','CALCULATE_ANALYTICAL = False'),
                         'load-isothermal','exec'),space)
            space['OUTPUT_DIRECTORY']=Path(directory)
            exec(compile(nb.cells[-1].source,'log-figure','exec'),space)
            dots=[a for a in space['axis'].collections if str(a.get_gid()).startswith('numerical-cold:')]
            self.assertEqual(sum(len(a.get_offsets()) for a in dots),63)
            for level in (10000.,50000.,500000.):
                artist=next(a for a in dots if a.get_gid()==f'numerical-cold:{level:g}')
                self.assertEqual(len(artist.get_offsets()),2 if level==50000. else 1)
            np.testing.assert_array_equal(space['numerical_contours'].levels,
                                          [50.,100.,200.,500.,1000.,2000.,5000.])
            original=m.analytic.load_numerical(ROOT)
            for key in original[0]:np.testing.assert_array_equal(original[0][key],space['warm_data'][key])
        plt.close('all')


if __name__=='__main__': unittest.main()
