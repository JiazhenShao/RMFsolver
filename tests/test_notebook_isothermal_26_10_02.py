"""Fresh notebook analytical calculation without expensive EOS tests."""
from pathlib import Path
import importlib
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'new_paper_calculations/26-10-02'


class NotebookCalculationTests(unittest.TestCase):
    def module(self):
        self.assertTrue((HERE / 'notebook_isothermal.py').exists(), 'Notebook calculation helper is missing')
        sys.path.insert(0, str(HERE))
        return importlib.import_module('notebook_isothermal')

    def inputs(self):
        return dict(n0=1., fingerprint='fixture', source_hashes={}, physics=dict(
            B_one_forth=189.1565957288247, ms=0., NM_type='PNM', upB=5000, xi=-.85),
            boundary_cache={float(t).hex():dict(T_0minus=t,status='success',lower=1.,upper=5.,
                lower_clipped=(t==20.),upper_clipped=False) for t in (.01,20.)})

    def evaluator(self, calls):
        def evaluate(point):
            calls.append(point.copy())
            speed=1000.*(point['nB_0minus_over_n0']-1.) + (80. if point['T_0minus']==20. else 0.)
            return dict(success=True,validated=True,status='moving_front',velocity_m_s=speed)
        return evaluate

    def test_axes_and_boundaries_only_no_old_analytical_points(self):
        module=self.module()
        self.assertEqual(len(module.TEMPERATURES),46)
        self.assertEqual(len(module.LEVELS),14)
        inputs=module.load_boundary_inputs(ROOT)
        self.assertNotIn('points',inputs)
        self.assertNotIn('targets',inputs)
        self.assertTrue(all(float(t).hex() in inputs['boundary_cache'] for t in module.TEMPERATURES))

    def test_fresh_calls_solver_and_resume_does_not_repeat_successes(self):
        module=self.module();calls=[]
        with tempfile.TemporaryDirectory() as directory:
            result=module.calculate(self.inputs(),directory,temperatures=(.01,20.),levels=(50.,1000.),
                workers=1,evaluator=self.evaluator(calls),show_progress=False)
            self.assertGreater(len(calls),0)
            self.assertEqual(sum(r['status']=='success' for r in result['targets'].values()),3)
            self.assertEqual(sum(r['status']=='outside_window' for r in result['targets'].values()),1)
            count=len(calls)
            resumed=module.calculate(self.inputs(),directory,temperatures=(.01,20.),levels=(50.,1000.),
                workers=1,evaluator=self.evaluator(calls),show_progress=False)
            self.assertEqual(len(calls),count)
            self.assertTrue(resumed['contours_complete'])
            with self.assertRaises(ValueError):
                module.calculate(self.inputs(),directory,temperatures=(.01,20.),levels=(50.,500.),
                    workers=1,evaluator=self.evaluator(calls),show_progress=False)

    def test_save_precedes_evaluation_notification_and_invalid_result_is_not_root(self):
        module=self.module();notified=[]
        with tempfile.TemporaryDirectory() as directory:
            def saved(event):
                payload=np.load(Path(directory)/'isothermal-analytic-logT.npy',allow_pickle=True).item()
                self.assertEqual(payload['evaluations'][-1]['status'],event['status'])
                self.assertIn('result',payload['evaluations'][-1])
                notified.append(event)
            result=module.calculate(self.inputs(),directory,temperatures=(.01,),levels=(50.,),workers=1,
                evaluator=lambda point:dict(success=False,validated=False,status='failed'),
                evaluation_saved=saved,show_progress=False)
            self.assertTrue(notified)
            self.assertEqual(next(iter(result['targets'].values()))['status'],'unresolved')

    def test_exact_upper_composition_boundary_is_never_proposed(self):
        module=self.module()
        rows=[dict(nB_0minus_over_n0=4.9,velocity_m_s=100.)]
        self.assertTrue(4.9 < module.propose_density(rows,1.e6,(1.,5.)) < 5.)
        self.assertEqual(module.propose_density([dict(nB_0minus_over_n0=1.,velocity_m_s=80.)],1000.,(1.,5.)),3.)

    def test_interrupted_reserved_evaluation_resumes_at_a_new_density(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as directory:
            def interrupt(point):
                raise KeyboardInterrupt('simulated interruption during a trial')
            with self.assertRaises(KeyboardInterrupt):
                module.calculate(self.inputs(),directory,temperatures=(.01,),levels=(50.,),workers=1,
                    evaluator=interrupt,show_progress=False)
            calls=[]
            result=module.calculate(self.inputs(),directory,temperatures=(.01,),levels=(50.,),workers=1,
                evaluator=self.evaluator(calls),show_progress=False)
            self.assertTrue(result['contours_complete'])
            self.assertEqual(result['evaluations'][0]['status'],'interrupted')
            self.assertEqual(result['evaluations'][0]['runtime_s'],90.)
            self.assertNotEqual(calls[0]['id'],result['evaluations'][0]['point']['id'])

    def test_calculation_cell_is_standalone_and_fresh_enabled_by_default(self):
        import nbformat
        nb=nbformat.read(ROOT/'26-10-02.ipynb',as_version=4)
        self.assertIn('CALCULATE_ANALYTICAL = True',nb.cells[4].source)
        self.assertIn('notebook_isothermal.calculate(',nb.cells[4].source)
        self.assertIn('from pathlib import Path',nb.cells[4].source)
        self.assertIn('WORKER_IMPORT_PATHS = None',nb.cells[4].source)
        self.assertIn('importlib.reload(notebook_isothermal)',nb.cells[4].source)
        for index in (5,len(nb.cells)-1):
            self.assertNotIn('notebook_isothermal.calculate(',nb.cells[index].source)

    def test_runtime_selection_prefers_persistent_matching_sidecar_and_allows_override(self):
        module=self.module()
        self.assertTrue(hasattr(module,'worker_import_paths'), 'Missing persistent runtime selection')
        with tempfile.TemporaryDirectory() as directory:
            runtime=module.bundled_runtime_path(directory)
            self.assertTrue(runtime.is_relative_to(Path(directory).resolve()/'local_analysis_results'))
            self.assertIn(sys.implementation.cache_tag,runtime.name)
            self.assertEqual(module.worker_import_paths(None,root=directory),())
            (runtime/'scipy').mkdir(parents=True)
            (runtime/'scipy/__init__.py').touch()
            self.assertEqual(module.worker_import_paths(None,root=directory),(str(runtime),))
            self.assertEqual(module.worker_import_paths((),root=directory),())
            self.assertEqual(module.worker_import_paths((directory,),root=directory),(str(Path(directory).resolve()),))

    def test_progress_counts_every_saved_evaluation_separately_from_targets(self):
        module=self.module();bars=[];calls=[]
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'isothermal-analytic-logT.npy'
            class Bar:
                def __init__(self,**kwargs):
                    self.options=kwargs;self.n=0;bars.append(self)
                def __enter__(self): return self
                def __exit__(self,*args): pass
                def set_postfix(self,*args,**kwargs): pass
                def update(self,n):
                    self.n+=n
                    if 'evaluation' in self.options['desc'].lower() and n:
                        data=np.load(output,allow_pickle=True).item()
                        completed=[e for e in data['evaluations'] if e['status'] not in ('running','interrupted')]
                        assert self.n==len(completed), 'Evaluation progress advanced before atomic save'
            with patch('tqdm.auto.tqdm',Bar):
                result=module.calculate(self.inputs(),directory,temperatures=(.01,),levels=(50.,1000.),
                    workers=1,evaluator=self.evaluator(calls))
                self.assertEqual(len(bars),2,'Need separate target and evaluation displays')
                self.assertEqual(bars[0].options['total'],2)
                self.assertEqual(bars[0].n,2)
                self.assertEqual(bars[1].n,len(result['evaluations']))
                self.assertEqual(bars[1].options['mininterval'],0.)
                self.assertEqual(bars[1].options['miniters'],1)
                before=len(calls)
                module.calculate(self.inputs(),directory,temperatures=(.01,),levels=(50.,1000.),
                    workers=1,evaluator=self.evaluator(calls))
                self.assertEqual(len(calls),before)
                self.assertEqual(bars[-1].n,bars[1].n)

    def test_resolved_runtime_paths_reach_preflight_and_each_worker(self):
        module=self.module();calls=[];numeric_calls=[]
        expected=('/runtime-fixture',)
        def bounded(module_name,function,*,args,kwargs=None,**controls):
            calls.append((args,kwargs))
            if args[0]=='preflight':
                self.assertEqual(kwargs['import_paths'],expected)
                return dict(scipy='fixture',scipy_path='/runtime-fixture/scipy/__init__.py')
            self.assertEqual(args[3],expected)
            return self.evaluator(numeric_calls)(args[1])
        self.assertTrue(hasattr(module,'worker_import_paths'), 'Missing persistent runtime selection')
        with tempfile.TemporaryDirectory() as directory, patch.object(module,'worker_import_paths',return_value=expected), \
                patch.object(module,'run_module_call_with_hard_timeout',side_effect=bounded):
            result=module.calculate(self.inputs(),directory,temperatures=(.01,),levels=(50.,),workers=1,show_progress=False)
            self.assertTrue(result['contours_complete'])
            self.assertGreater(len(calls),1)

    def test_failed_preflight_explains_setup_without_reserving_solver_work(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module,'run_module_call_with_hard_timeout',side_effect=RuntimeError('binary loader failure')):
            with self.assertRaisesRegex(RuntimeError,'pip install --no-deps --target'):
                module.calculate(self.inputs(),directory,temperatures=(.01,),levels=(50.,),
                    workers=1,show_progress=False)
            self.assertFalse((Path(directory)/'isothermal-analytic-logT.npy').exists())


if __name__=='__main__':
    unittest.main()
