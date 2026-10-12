"""Lightweight workflow contracts; no real EOS or velocity solves."""
import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

RUN = Path(__file__).resolve().parents[1] / 'new_paper_calculations/26-10-02/isothermal_analytic_logT'


class AnalyticLogTTests(unittest.TestCase):
    def setUp(self):
        self.names = ('_logT_common', '_logT_trial', '_logT_inputs', '_logT_engine', 'run_isothermal_analytic')
        self.previous = {name: sys.modules.pop(name, None) for name in self.names}
        sys.path.insert(0, str(RUN))

    def tearDown(self):
        sys.path.remove(str(RUN))
        for name in self.names:
            sys.modules.pop(name, None)
            if self.previous[name] is not None:
                sys.modules[name] = self.previous[name]

    def module(self, name):
        return importlib.import_module(name)

    def test_analytical_physics_and_full_temperature_axis(self):
        c = self.module('_logT_common')
        self.assertEqual(c.PHYSICS['xi'], -.85)
        self.assertEqual(len(c.TEMPERATURES), 46)
        np.testing.assert_array_equal(c.TEMPERATURES[:31], np.geomspace(.01, 20., 31))
        self.assertEqual(c.TEMPERATURES[-1], 120.)
        self.assertEqual(sum(c.TEMPERATURES == 20.), 1)
        self.assertTrue(np.all(np.diff(c.TEMPERATURES) > 0))
        self.assertEqual(len(c.base_targets()), 322)
        self.assertNotIn('tol_bvp', c.CONTROLS)

    @staticmethod
    def answer(speed=100.):
        u = speed / 299792458.
        return dict(success=True, status='moving_front', front_exists=True,
            a_0plus_source='maximum', a_0plus_max_status='interior', a_0plus=.2,
            u_0minus=u, jB=u, u_0minus_formula_squared=u*u,
            analytic_velocity_residual=0., composition_residual=0.,
            momentum_flux_inf_residual=0., momentum_flux_0plus_residual=0., Pi=100.,
            momentum_flux_ratio=1.e-10, slow_front_consistent=True)

    def test_worker_calls_only_public_analytical_solver_with_xi(self):
        trial = self.module('_logT_trial')
        self.assertTrue(hasattr(trial, 'analytical'), 'Replace numerical worker with analytical')
        calls = []
        def solver(*args, **kwargs):
            calls.append((args, kwargs))
            return self.answer()
        package = SimpleNamespace(phase_velocity=SimpleNamespace(analytic_velocity_isothermal=solver))
        with patch.dict(sys.modules, {'RMFsolver': package}):
            result = trial.analytical(dict(T_0minus=.01, nB_0minus=3., nB_0minus_over_n0=3.))
        self.assertTrue(result['validated'])
        self.assertAlmostEqual(result['velocity_m_s'], 100., places=7)
        self.assertEqual(calls[0][1]['xi'], -.85)
        self.assertTrue(np.isnan(calls[0][1]['a_0plus']))

    def test_validation_checks_analytical_not_bvp_residuals(self):
        trial = self.module('_logT_trial')
        self.assertTrue(hasattr(trial, 'valid_analytical_result'))
        good = self.answer()
        self.assertTrue(trial.valid_analytical_result(good))
        for key in ('composition_residual','analytic_velocity_residual','momentum_flux_inf_residual'):
            bad = dict(good, **{key:1.})
            self.assertFalse(trial.valid_analytical_result(bad))
        self.assertFalse(trial.valid_analytical_result(dict(good, success=False)))

    def test_input_bundle_has_only_static_reuse_not_velocity_results(self):
        m = self.module('_logT_inputs')
        inputs = m.load_inputs(RUN)
        self.assertNotIn('points', list(inputs))
        self.assertIn('boundary_cache', inputs)
        self.assertIn('boundary_geometry', inputs)
        self.assertGreaterEqual(len(inputs['boundary_cache']), 31)
        self.assertEqual(inputs['manifest']['use'], 'static_boundaries_only')
        c=self.module('_logT_common')
        self.assertTrue(all(float(T).hex() in inputs['boundary_cache'] for T in c.TEMPERATURES[:31]),
                        'Reuse cold snapshots despite platform-level last-bit geomspace rounding')

    def test_no_repeated_trial_after_two_failed_guesses(self):
        c = self.module('_logT_common')
        self.assertTrue(hasattr(c, 'next_density'))
        failures = []
        for _ in range(4):
            guess = c.next_density([], failures, 3.01, 500., (3., 5.3))
            self.assertIsNotNone(guess)
            self.assertNotIn(guess, failures)
            failures.append(guess)
        self.assertTrue(all(3 < d < 5.3 for d in failures))

    def test_engine_uses_analytical_samples_and_saves_before_progress(self):
        engine = self.module('_logT_engine')
        self.assertTrue(hasattr(engine, 'AnalyticRun'))
        c = self.module('_logT_common')
        inputs = dict(n0=1., fingerprint='domain', boundary_cache={.01.hex():
            dict(status='success', lower=3., upper=5., T_0minus=.01, lower_clipped=False)},
            boundary_geometry={})
        calls, seen = [], []
        def evaluator(point):
            calls.append(point['id'])
            speed = (point['nB_0minus_over_n0']-3.)*10000.
            return dict(self.answer(speed), velocity_m_s=speed, validated=True)
        with tempfile.TemporaryDirectory() as temporary:
            def saved(event):
                checkpoint = c.load(Path(temporary)/'isothermal-analytic-logT.npy')
                self.assertEqual(checkpoint['evaluations'][-1]['status'], event['status'])
                self.assertNotEqual(event['status'], 'running')
                seen.append(event['id'])
            run = engine.AnalyticRun(temporary, inputs, {'xi':-.85},
                                     evaluator=evaluator, evaluation_saved=saved)
            run.solve_temperature(.01, [50., 100., 500.])
            self.assertEqual(sum(t['status']=='success' for t in run.payload['targets'].values()), 3)
            self.assertEqual(len(calls), len(set(calls)))
            self.assertEqual(len(seen), len(calls))
            resumed = engine.AnalyticRun(temporary, inputs, {'xi':-.85}, evaluator=evaluator)
            resumed.solve_temperature(.01, [50., 100., 500.])
            self.assertEqual(len(calls), len(seen))
            with self.assertRaisesRegex(ValueError, 'identity|compatible'):
                engine.AnalyticRun(temporary, inputs, {'xi':-.5}, evaluator=evaluator)

    def test_outside_window_does_not_launch_analytical_solve(self):
        engine = self.module('_logT_engine')
        self.assertTrue(hasattr(engine, 'AnalyticRun'))
        with tempfile.TemporaryDirectory() as temporary:
            inputs = dict(n0=1., fingerprint='domain', boundary_cache={120.0.hex():
                dict(status='outside_window', T_0minus=120., reason='composition')}, boundary_geometry={})
            run = engine.AnalyticRun(temporary, inputs, {}, evaluator=lambda **kw:self.fail('No physical window'))
            run.solve_temperature(120., [50., 5000.])
            self.assertTrue(all(t['status']=='outside_window' for t in run.payload['targets'].values()))
            self.assertEqual(run.payload['evaluations'], [])

    def test_new_entrypoint_and_no_bvp_machinery(self):
        self.assertTrue((RUN/'run_isothermal_analytic.py').is_file())
        for name in ('_logT_trial.py','_logT_engine.py','_logT_common.py'):
            source=(RUN/name).read_text()
            self.assertNotIn('solve_front_isothermal',source)
            self.assertNotIn('continuation_guess',source)
        self.assertFalse((RUN/'run_isothermal_numerical.py').exists())

    def test_clipped_upper_edge_classifies_unreachable_speed_without_search(self):
        engine=self.module('_logT_engine')
        inputs=dict(n0=1.,fingerprint='domain',boundary_geometry={},boundary_cache={80.0.hex():
            dict(status='success',T_0minus=80.,lower=3.,upper=5.5,upper_clipped=True)})
        calls=[]
        def evaluator(p):
            calls.append(p)
            speed=(p['nB_0minus_over_n0']-3.)*100.
            return dict(self.answer(speed),velocity_m_s=speed,validated=True)
        with tempfile.TemporaryDirectory() as temporary:
            run=engine.AnalyticRun(temporary,inputs,{},evaluator=evaluator)
            run.solve_temperature(80.,[5000.])
            self.assertEqual(next(iter(run.payload['targets'].values()))['status'],'outside_window')
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0]['nB_0minus_over_n0'],5.5)

    def test_interrupted_reservation_remains_spent_on_resume(self):
        engine=self.module('_logT_engine');c=self.module('_logT_common')
        inputs=dict(n0=1.,fingerprint='domain',boundary_geometry={},boundary_cache={})
        with tempfile.TemporaryDirectory() as temporary:
            run=engine.AnalyticRun(temporary,inputs,{})
            task=c.target(100.,.01);task['status']='running'
            run.payload['targets'][task['id']]=task
            run.payload['evaluations'].append(dict(id=0,target_id=task['id'],status='running',
                reserved_seconds=180.,point=c.point(.01,3.01,1.)))
            run.save()
            resumed=engine.AnalyticRun(temporary,inputs,{})
            self.assertEqual(resumed.payload['evaluations'][0]['status'],'interrupted')
            self.assertEqual(resumed.spent(task),180.)
            self.assertEqual(resumed.payload['targets'][task['id']]['status'],'pending')

    def test_boundary_can_classify_composition_excluded_warm_window(self):
        trial=self.module('_logT_trial')
        fake=SimpleNamespace(_validated_pnm_state_from_nB=lambda d,T:dict(P_0minus=d,muB_0minus=1000.),
            _solve_interface_0plus_from_local_a_and_Pi=lambda a,P,*args,**kw:(1000.+.8-P,0.))
        with patch.dict(sys.modules,{'RMFsolver':SimpleNamespace(phase_velocity=fake)}):
            row=trial.boundary(120.,dict(lower=1.,upper=1.),1.)
        self.assertEqual(row['status'],'outside_window')

    def test_mocked_master_run_and_resume_do_not_call_numerical_solver(self):
        runner=self.module('run_isothermal_analytic');c=self.module('_logT_common')
        engine=self.module('_logT_engine')
        tasks=c.base_targets(smoke=True)
        inputs=dict(n0=1.,fingerprint='domain',manifest={},boundary_geometry={},boundary_cache={
            float(t['T_0minus']).hex():dict(status='success',T_0minus=t['T_0minus'],lower=3.,upper=5.)
            for t in tasks})
        calls=[]
        locks=[]
        original_save=engine.AnalyticRun.save
        def save(run):
            locks.append(run.lock)
            return original_save(run)
        def refresh(*args):
            self.assertFalse(locks[-1]._is_owned(), 'Rendering must not hold the checkpoint lock')
        def evaluate(kind,timeout,point):
            self.assertEqual(kind,'analytical')
            calls.append(point)
            speed=(point['nB_0minus_over_n0']-3.)*10000.
            return dict(self.answer(speed),validated=True,velocity_m_s=speed)
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner,'load_inputs',return_value=inputs), \
                patch.object(runner,'identity',return_value={}),patch.object(runner,'refresh',side_effect=refresh), \
                patch.object(engine.AnalyticRun,'save',new=save), \
                patch.object(self.module('_logT_trial'),'bounded_call',side_effect=evaluate):
            args=['--smoke','--workers','2','--output-dir',temporary]
            self.assertEqual(runner.main(args),0)
            count=len(calls)
            self.assertEqual(runner.main(args),0)
            self.assertEqual(len(calls),count)
            saved=c.load(Path(temporary)/'isothermal-analytic-logT.npy')
            self.assertEqual(len(saved['targets']),4)
            self.assertTrue(saved['contours_complete'])


if __name__ == '__main__':
    unittest.main()
