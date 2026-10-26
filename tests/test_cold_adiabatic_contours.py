"""Cold contour orchestration contracts, without importing the live EOS solver."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'new_paper_calculations/26-10-26'
sys.path.insert(0, str(RUN / 'shared'))
import bounded_call
import cold_physics
import cold_scan


def identity():
    return dict(format='cold_adiabatic_contours_v1', physics=dict(
        T_0plus_MeV=0., xi=-.85, ms_MeV=0., NM_type='PNM',
        B14_MeV=189., n0_MeV3=1., path_rtol=1e-6),
        code_sha256={'solver.py': 'cold-source'}, input_sha256='cold-seeds',
        method_timeout=120., angular_rounds=5, smoke=False)


def valid_result():
    return dict(success=True, message='accepted', velocity_m_s=500., jB=1., I2=2.,
        T_0plus=0., D_K_0plus=np.inf, invD_K_0plus=0., a_0plus=.5,
        conservation_errors={'pressure': 1e-12},
        hydrodynamic_model='nonrelativistic_isobaric', closure='adiabatic',
        alpha_s=.4, rate_approximation='madsen_muB_over_3')


def row(method, theta, rho, status='accepted', speed=500., exclusion=None):
    return dict(method=method, point=cold_scan.point_on_ray(theta, rho),
        status=status, speed=speed, exclusion=exclusion,
        record_key=cold_scan.coordinate_key(method, theta, rho))


class BoundedCallTests(unittest.TestCase):
    def test_hard_timeout_terminates_child_and_later_calls_work(self):
        children_before = {p.pid for p in mp.active_children()}
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            bounded_call.run_module_call_with_hard_timeout(
                'time', 'sleep', args=(30,), timeout_s=.15)
        self.assertLess(time.monotonic() - started, 5.)
        self.assertEqual({p.pid for p in mp.active_children()}, children_before)
        self.assertEqual(bounded_call.run_module_call_with_hard_timeout(
            'builtins', 'sum', args=([2, 3],), timeout_s=5.), 5)


class RecordTests(unittest.TestCase):
    def test_cancelled_scan_starts_no_new_call_or_ray(self):
        with tempfile.TemporaryDirectory() as directory:
            scan = cold_scan.Scan(Path(directory) / 'scan.npy', identity(), [], 120.)
            scan.cancel.set()
            with patch.object(cold_scan, 'run_module_call_with_hard_timeout') as call:
                with self.assertRaises(InterruptedError):
                    scan.evaluate('analytic', 0., .4)
                with self.assertRaises(InterruptedError):
                    scan.trace_ray('semi_analytic', 0.)
                call.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_resume_rejects_warm_source_and_settings_changes(self):
        base = identity()
        mutations = [('physics', 'T_0plus_MeV', .1),
                     ('code_sha256', 'solver.py', 'changed'),
                     ('physics', 'path_rtol', 1e-5)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.npy'
            scan = cold_scan.Scan(path, base, [], 120.)
            with patch.object(cold_scan, 'run_module_call_with_hard_timeout',
                              return_value=row('analytic', 0., .4)):
                scan.evaluate('analytic', 0., .4)
            for group, key, value in mutations:
                changed = deepcopy(base)
                changed[group][key] = value
                with self.subTest(group=group, key=key):
                    with patch.object(cold_scan, 'run_module_call_with_hard_timeout') as call:
                        with self.assertRaisesRegex(ValueError, 'differ'):
                            cold_scan.Scan(path, changed, [], 120.).evaluate('analytic', 0., .4)
                        call.assert_not_called()

    def test_methods_have_independent_records_and_success_reuses_failure_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.npy'
            def calculate(module, function, *, args, **kwargs):
                method, point, config = args
                if method == 'analytic':
                    raise TimeoutError('full method timed out')
                return row(method, point['theta_rad'], point['rho'])
            scan = cold_scan.Scan(path, identity(), [], 120.)
            with patch.object(cold_scan, 'run_module_call_with_hard_timeout', side_effect=calculate) as call:
                full = scan.evaluate('analytic', 0., .4)
                semi = scan.evaluate('semi_analytic', 0., .4)
                self.assertEqual(call.call_count, 2)
                self.assertTrue(all(c.kwargs['timeout_s'] == 120. for c in call.call_args_list))
            self.assertEqual(full['status'], 'failed')
            self.assertEqual(semi['status'], 'accepted')
            self.assertNotEqual(full['record_key'], semi['record_key'])
            self.assertEqual(len(list(scan.record_dir.glob('*.npy'))), 2)
            resumed = cold_scan.Scan(path, identity(), [], 120.)
            with patch.object(cold_scan, 'run_module_call_with_hard_timeout',
                              return_value=row('analytic', 0., .4)) as call:
                retried = resumed.evaluate('analytic', 0., .4)
                reused = resumed.evaluate('semi_analytic', 0., .4)
                self.assertEqual(call.call_count, 1)
            self.assertEqual(retried['status'], 'accepted')
            self.assertEqual(len(retried['previous_failures']), 1)
            self.assertEqual(reused['record_key'], semi['record_key'])

    def test_geometry_cache_cannot_replace_a_speed_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            scan = cold_scan.Scan(Path(directory) / 'scan.npy', identity(), [], 120.)
            def calculate(module, function, *, args, **kwargs):
                method, point, config = args
                return row(method, point['theta_rad'], point['rho'],
                           status='interior' if function == 'classify' else 'accepted')
            with patch.object(cold_scan, 'run_module_call_with_hard_timeout', side_effect=calculate) as call:
                geometry = scan.geometry('semi_analytic', 0., .4)
                speed = scan.evaluate('semi_analytic', 0., .4)
                self.assertEqual(call.call_count, 2)
                self.assertEqual([c.args[1] for c in call.call_args_list], ['classify', 'evaluate'])
            self.assertEqual(geometry['status'], 'interior')
            self.assertEqual(speed['status'], 'accepted')
            self.assertNotEqual(geometry['record_key'], speed['record_key'])
            self.assertEqual(len(list(scan.record_dir.glob('*.npy'))), 2)

    def test_atomic_save_keeps_previous_record_if_serialization_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.npy'
            cold_scan.atomic_save(path, {'status': 'accepted'})
            with patch.object(cold_scan.np, 'save', side_effect=RuntimeError('interrupted write')):
                with self.assertRaises(RuntimeError):
                    cold_scan.atomic_save(path, {'status': 'failed'})
            self.assertEqual(np.load(path, allow_pickle=True).item(), {'status': 'accepted'})


class ColdPhysicsTests(unittest.TestCase):
    def test_exact_cold_infinite_diffusion_is_valid_and_floor_is_rejected(self):
        cold_physics.validate_result(valid_result())
        for change in (dict(T_0plus=1e-12), dict(D_K_0plus=1e100),
                       dict(D_K_0plus=-np.inf), dict(invD_K_0plus=1e-100),
                       dict(velocity_m_s=np.nan), dict(I2=0.)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                cold_physics.validate_result(dict(valid_result(), **change))

    def evaluate_with_result(self, result):
        calls = []
        def solve(*args, **kwargs):
            calls.append((args, kwargs))
            return result.copy()
        pv = SimpleNamespace(_PNM_BRANCH_SEEDS=(),
            analytic_velocity_adiabatic_slow=solve,
            semi_analytic_velocity_adiabatic_slow=solve)
        package = ModuleType('RMFsolver')
        package.phase_velocity = pv
        with patch.dict(sys.modules, {'RMFsolver': package}):
            answer = cold_physics.evaluate('analytic', cold_scan.point_on_ray(0., .5), identity()['physics'])
        return answer, calls

    def test_physical_exclusions_are_distinct_from_numerical_failure(self):
        for message, expected in cold_physics.EXCLUSIONS.items():
            with self.subTest(message=message):
                answer, _ = self.evaluate_with_result(dict(success=False, message=message))
                self.assertEqual((answer['status'], answer['exclusion']), ('excluded', expected))
        answer, _ = self.evaluate_with_result(dict(success=False, message='EOS solve did not converge'))
        self.assertEqual(answer['status'], 'failed')
        self.assertIsNone(answer['exclusion'])

    def test_calls_use_exact_cold_massless_pnm_inputs(self):
        answer, calls = self.evaluate_with_result(valid_result())
        self.assertEqual(answer['status'], 'accepted')
        self.assertEqual(calls[0][1], dict(T_0plus=0., xi=-.85, ms=0., NM_type='PNM'))


@unittest.skipUnless(os.environ.get('COLD_CONTOUR_LIVE') == '1',
    'Set COLD_CONTOUR_LIVE=1 with the isolated SciPy runtime for live classifier checks')
class LiveClassifierTests(unittest.TestCase):
    def test_classifier_matches_each_methods_direct_thermodynamic_domain(self):
        from RMFsolver import phase_velocity as pv
        seed = json.loads((RUN / 'inputs/cold-scan-seeds.json').read_text())
        config = dict(identity()['physics'], **seed['calibration'])
        for temperature, density in ((0., 2.99), (0., 3.00080902615833),
                                     (0., 3.5), (0., 5.4), (70., 2.5)):
            point = dict(T_0minus_MeV=temperature, nB_0minus_over_n0=density)
            for method in cold_physics.METHODS:
                with self.subTest(method=method, temperature=temperature, density=density):
                    classification = cold_physics.classify(method, point, config)
                    if method == 'analytic':
                        direct = cold_physics.evaluate(method, point, config)
                        expected = ('interior' if direct['status'] == 'accepted' else direct['status'],
                                    direct['exclusion'])
                    else:
                        nB = density * config['n0_MeV3']
                        upstream = pv._isothermal_upstream_nuclear_state(
                            temperature, nB, pv.para.paraQMCRMF3, 'PNM')
                        try:
                            # Validate with the actual EOS path endpoints but do not build
                            # the unrelated diffusion table responsible for the timeout.
                            with patch.object(pv._SlowFrontPath, '_tabulate', return_value=None):
                                path = pv._SlowFrontPath('adiabatic', upstream['P_0minus'],
                                    upstream['h_0minus'] / nB, temperature, config['B14_MeV'],
                                    0., None, 0., config['path_rtol'])
                            self.assertEqual(path.interface['T'], 0.)
                            self.assertGreater(path.interface['a'], 0.)
                            self.assertLess(path.interface['a'], 1.)
                            expected = ('interior', None)
                        except RuntimeError as exc:
                            self.assertIn(str(exc), cold_physics.EXCLUSIONS)
                            expected = ('excluded', cold_physics.EXCLUSIONS[str(exc)])
                    self.assertEqual((classification['status'], classification['exclusion']), expected)
                    if temperature == 0. and density == 5.4:
                        # The consistently second-order ceiling now excludes
                        # this state in both methods; the old leading-density
                        # analytical ceiling incorrectly admitted it.
                        self.assertEqual(classification['status'], 'excluded')
                        self.assertEqual(classification['exclusion'], 'composition_ceiling')


class ContourTests(unittest.TestCase):
    def trace(self, evaluate, method='analytic'):
        scan = cold_scan.Scan('/unused/scan.npy', identity(), [], 120.)
        def geometry(method, theta, rho):
            result = evaluate(method, theta, rho).copy()
            if result['status'] == 'accepted':
                result['status'] = 'interior'
            return result
        with patch.object(scan, 'evaluate', side_effect=evaluate), \
                patch.object(scan, 'geometry', side_effect=geometry), \
                patch.object(cold_scan, 'LEVELS', (500.,)):
            return scan.trace_ray(method, 0.)

    def test_boundary_refinement_uses_geometry_without_near_zero_speed_integrals(self):
        scan = cold_scan.Scan('/unused/scan.npy', identity(), [], 120.)
        speed_calls, geometry_calls = [], []
        def geometry(method, theta, rho):
            geometry_calls.append(rho)
            return row(method, theta, rho, status='excluded' if rho < .3 else 'interior',
                       exclusion='stable_matter' if rho < .3 else None)
        def evaluate(method, theta, rho):
            speed_calls.append(rho)
            self.assertGreaterEqual(abs(rho - .3), .005,
                'Boundary bisection must not invoke a nearly-zero-speed integral')
            return row(method, theta, rho, status='excluded' if rho < .3 else 'accepted',
                       speed=100 + 1000*(rho - .3),
                       exclusion='stable_matter' if rho < .3 else None)
        with patch.object(scan, 'evaluate', side_effect=evaluate), \
                patch.object(scan, 'geometry', side_effect=geometry), \
                patch.object(cold_scan, 'LEVELS', (500.,)):
            answer = scan.trace_ray('semi_analytic', 0.)
        self.assertTrue(answer['boundaries']['stable_matter']['resolved'])
        self.assertTrue(any(abs(rho - .3) < 1e-4 for rho in geometry_calls))
        self.assertTrue(speed_calls)
        self.assertEqual(answer['contours'][500.]['status'], 'resolved')

    def test_geometry_failure_is_recorded_for_resume_retry(self):
        scan = cold_scan.Scan('/unused/scan.npy', identity(), [], 120.)
        def evaluate(method, theta, rho):
            return row(method, theta, rho, status='excluded' if rho < .3 else 'accepted',
                       speed=100 + 1000*rho, exclusion='stable_matter' if rho < .3 else None)
        failed = row('semi_analytic', 0., .28125, status='failed')
        failed['record_key'] = 'geometry-failed-fixture'
        with patch.object(scan, 'evaluate', side_effect=evaluate), \
                patch.object(scan, 'geometry', return_value=failed), \
                patch.object(cold_scan, 'LEVELS', (500.,)):
            answer = scan.trace_ray('semi_analytic', 0.)
        self.assertFalse(answer['boundaries']['stable_matter']['resolved'])
        self.assertIn(failed['record_key'], answer['failures'])

    def test_failed_sample_never_bridges_a_root(self):
        def evaluate(method, theta, rho):
            return row(method, theta, rho, status='failed' if rho == .5 else 'accepted', speed=10 + 980*rho)
        answer = self.trace(evaluate)
        self.assertEqual(answer['contours'][500.]['status'], 'unresolved')
        self.assertEqual(len(answer['failures']), 1)

    def test_failed_root_refinement_is_unresolved(self):
        initial = set(np.linspace(0., 1., 17))
        def evaluate(method, theta, rho):
            return row(method, theta, rho, status='accepted' if rho in initial else 'failed', speed=11 + 980*rho)
        answer = self.trace(evaluate)
        self.assertEqual(answer['contours'][500.]['status'], 'unresolved')
        self.assertTrue(answer['failures'])

    def test_failed_interior_does_not_establish_outside_window(self):
        def evaluate(method, theta, rho):
            return row(method, theta, rho, status='failed' if rho == .5 else 'accepted', speed=100.)
        self.assertEqual(self.trace(evaluate)['contours'][500.]['status'], 'unresolved')

    def test_exact_sampled_root_is_one_resolved_crossing(self):
        answer = self.trace(lambda m, t, r: row(m, t, r, speed=10 + 980*r))
        root = answer['contours'][500.]
        self.assertEqual(root['status'], 'resolved')
        self.assertLess(root['relative_speed_error'], 1e-3)

    def test_exact_sampled_root_does_not_hide_another_crossing(self):
        answer = self.trace(lambda m, t, r: row(m, t, r, speed=500 + 1000*(r - .25)*(r - .71)))
        self.assertEqual(answer['contours'][500.]['status'], 'unresolved')

    def test_decreasing_crossing_is_resolved(self):
        answer = self.trace(lambda m, t, r: row(m, t, r, speed=997 - 980*r))
        self.assertEqual(answer['contours'][500.]['status'], 'resolved')

    def test_methods_keep_their_own_physical_domain(self):
        def evaluate(method, theta, rho):
            threshold = .3 if method == 'analytic' else .6
            return row(method, theta, rho, status='excluded' if rho < threshold else 'accepted',
                       speed=1000*rho, exclusion='stable_matter' if rho < threshold else None)
        full, semi = (self.trace(evaluate, method) for method in cold_physics.METHODS)
        self.assertAlmostEqual(full['boundaries']['stable_matter']['point']['rho'], .3, delta=1e-4)
        self.assertAlmostEqual(semi['boundaries']['stable_matter']['point']['rho'], .6, delta=1e-4)

    def test_angular_refinement_detects_boundary_disappearance_and_unresolved_boundary(self):
        contours = {level: dict(status='outside_window') for level in cold_scan.LEVELS}
        def ray(theta, boundary=True, resolved=True):
            return dict(contours=contours, boundaries={'stable_matter': dict(
                point=cold_scan.point_on_ray(theta, .4), resolved=resolved)} if boundary else {})
        for middle in (ray(.1, boundary=False), ray(.1, resolved=False)):
            with self.subTest(middle=middle):
                error, transition = cold_scan.interpolation_error(ray(0.), middle, ray(.2))
                self.assertTrue(transition or error > .0025)


class EntrypointTests(unittest.TestCase):
    def test_default_angular_budget_can_certify_seed_interval_transitions(self):
        args = cold_scan.make_parser(1, Path('/unused/scan.npy')).parse_args([])
        self.assertGreaterEqual(args.angular_rounds, 6)

    def test_six_angular_rounds_certify_a_status_transition(self):
        seed = json.loads((RUN / 'inputs/cold-scan-seeds.json').read_text())
        lower = float(np.arctan2(115., 120.))
        spacing = float(np.pi / 58.)
        upper = lower + spacing
        seed['coordinates'] = [cold_scan.point_on_ray(theta, .5) for theta in (lower, upper)]
        transition = lower + spacing / 3
        def trace(scan, method, theta):
            contour = (dict(status='resolved', point=cold_scan.point_on_ray(0., .5),
                            speed=500., relative_speed_error=0.)
                       if theta < transition else dict(status='outside_window'))
            return dict(method=method, theta_rad=theta, contours={500.: contour},
                        boundaries={}, records=[], failures=[])
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory) / 'seeds.json'
            inputs.write_text(json.dumps(seed))
            for rounds in (5, 6):
                with self.subTest(rounds=rounds):
                    output = Path(directory) / f'rounds-{rounds}.npy'
                    argv = ['scan', '--workers', '1', '--angular-rounds', str(rounds),
                            '--method', 'analytic', '--input', str(inputs), '--output', str(output)]
                    with patch.object(sys, 'argv', argv), \
                            patch.object(cold_scan.Scan, 'trace_ray', trace), \
                            patch.object(cold_scan, 'LEVELS', (500.,)), redirect_stdout(io.StringIO()):
                        result = cold_scan.main(default_output=output)
                    payload = np.load(output, allow_pickle=True).item()
                    self.assertEqual(payload['complete'], rounds == 6)
                    self.assertEqual(result, 0 if rounds == 6 else 2)
                    self.assertEqual(bool(payload['unresolved_angular_intervals']), rounds == 5)
                    self.assertEqual(payload['coverage']['analytic']['unresolved'], 0)

    def test_interruption_cancels_workers_before_executor_waits(self):
        cancellation_at_shutdown = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.npy'
            scan = cold_scan.Scan(path, identity(), [], 120.)
            class Future:
                def cancel(self):
                    return True

            class Pool:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    cancellation_at_shutdown.append(scan.cancel.is_set())

                def submit(self, *args):
                    return Future()

            def interrupt(_futures):
                signal.raise_signal(signal.SIGINT)

            previous_sigint = signal.getsignal(signal.SIGINT)
            with patch.object(cold_scan, 'Scan', return_value=scan), \
                    patch.object(cold_scan, 'ThreadPoolExecutor', return_value=Pool()), \
                    patch.object(cold_scan, 'as_completed', side_effect=interrupt), \
                    patch.object(sys, 'argv', ['scan', '--method', 'analytic', '--output', str(path)]):
                with self.assertRaises(KeyboardInterrupt):
                    cold_scan.main(default_workers=1, default_output=path)
            self.assertEqual(cancellation_at_shutdown, [True])
            self.assertEqual(signal.getsignal(signal.SIGINT), previous_sigint)

    def test_local_cluster_dry_run_match_physics_and_make_no_files(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for variant in ('local', 'cluster'):
                completed = subprocess.run([sys.executable, '-B', str(RUN / variant / 'run_adiabatic_contours.py'),
                    '--dry-run', '--output', str(Path(directory) / variant / 'scan.npy')],
                    capture_output=True, text=True, check=True, timeout=15)
                config = json.loads(completed.stdout)
                config.pop('workers')
                config.pop('output')
                outputs.append(config)
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[0]['methods'], ['analytic', 'semi_analytic'])
            self.assertEqual(outputs[0]['identity']['method_timeout'], 120.)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_ray_window_has_exact_cold_edge_and_common_bounds(self):
        for theta in np.linspace(0., np.pi / 2, 41):
            point = cold_scan.point_on_ray(theta, cold_scan.ray_limit(theta))
            self.assertGreaterEqual(point['nB_0minus_over_n0'], 1.)
            self.assertLessEqual(point['nB_0minus_over_n0'], 5.5 + 1e-12)
            self.assertGreaterEqual(point['T_0minus_MeV'], 0.)
            self.assertLessEqual(point['T_0minus_MeV'], 115. + 1e-12)
        self.assertEqual(cold_scan.point_on_ray(0., .5)['T_0minus_MeV'], 0.)


if __name__ == '__main__':
    unittest.main()
