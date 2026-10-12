"""No live EOS/BVP calls: test the dedicated xi=-0.85 cluster orchestration."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'new_paper_calculations/26-10-02/adiabatic_xi_minus0p85'


def load_scan():
    sys.path.insert(0, str(WORK))
    try:
        spec = importlib.util.spec_from_file_location('adiabatic_xi_scan', WORK / 'run_T0plus0p1_analytical.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(WORK))


def result(scan, point, speed=4000., success=True):
    method = dict(success=success, u_0minus=speed / scan.SPEED_OF_LIGHT,
                  jB=1., message='' if success else 'test failure')
    return dict(point=point, analytic=dict(method), semi_analytic=dict(method))


class ClusterTests(unittest.TestCase):
    def scan(self):
        try:
            return load_scan()
        except ImportError as exc:
            self.fail(f'Runner must import without eager EOS/SciPy integration imports: {exc}')

    def test_dry_run_from_another_directory_has_new_xi_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'untouched.npy'
            process = subprocess.run([sys.executable, '-B', str(WORK / 'run_T0plus0p1_analytical.py'),
                                      '--dry-run', '--workers', '2', '--output', str(output)],
                                     cwd=tmp, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn('xi=-0.85', process.stdout)
            self.assertIn('660', process.stdout)
            self.assertFalse(output.exists())

    def test_source_coordinates_preserved_and_inner_fraction_added(self):
        scan = self.scan()
        domain, numerical = scan.load_inputs()
        points, matched, original = scan.build_points(domain, numerical)
        self.assertEqual(domain['xi'], -.5)  # Immutable geometry-source provenance.
        self.assertEqual(scan.RUN_XI, -.85)
        self.assertEqual(len(points), 660)
        self.assertEqual(matched.shape, (30, 18))
        self.assertEqual(original.shape, (30, 20))
        for ij in np.ndindex(matched.shape):
            self.assertEqual(points[matched[ij]]['T_0minus_MeV'], numerical['plot_T_0minus_MeV_grid'][ij])
            self.assertEqual(points[matched[ij]]['nB_0minus_over_n0'], numerical['plot_nB_0minus_over_n0_grid'][ij])
        for theta in range(30):
            key = scan.point_key(scan.point_on_ray(domain, theta, .005))
            self.assertIn(key, points)
        self.assertTrue(all(p['T_0minus_MeV'] == 0 for p in points.values() if p['theta_index'] == 0))

    def test_workers_use_allocation_without_cap_and_affinity_fallback(self):
        scan = self.scan()
        with patch.dict(scan.engine.os.environ, {'SLURM_CPUS_PER_TASK': '128'}, clear=True):
            self.assertEqual(scan.engine.default_workers(), 128)
            self.assertEqual(self.scan().parser().parse_args([]).workers, 128)
        with patch.dict(scan.engine.os.environ, {'SLURM_CPUS_PER_TASK': '0', 'PBS_NP': '96'}, clear=True):
            self.assertEqual(scan.engine.default_workers(), 96)
        with patch.dict(scan.engine.os.environ, {'NSLOTS': '72'}, clear=True):
            self.assertEqual(scan.engine.default_workers(), 72)
        with patch.dict(scan.engine.os.environ, {}, clear=True), \
                patch.object(scan.engine.os, 'sched_getaffinity', return_value={1, 3, 5}, create=True), \
                patch.object(scan.engine.os, 'cpu_count', return_value=64):
            self.assertEqual(scan.engine.default_workers(), 3)

    def test_per_method_timeout_retains_completed_other_method(self):
        scan = self.scan()
        domain, _ = scan.load_inputs()
        domain = dict(domain, xi=scan.RUN_XI)
        point = scan.point_on_ray(domain, 0, .1)
        full = result(scan, point)['analytic']
        with patch.object(scan, 'run_module_call_with_hard_timeout',
                          side_effect=[full, TimeoutError('test timeout')]) as bounded:
            row = scan.evaluate_point(point, domain, 120.)
        self.assertEqual(row['analytic'], full)
        self.assertFalse(row['semi_analytic']['success'])
        self.assertEqual(bounded.call_count, 2)
        self.assertTrue(all(c.kwargs['timeout_s'] == 120. for c in bounded.call_args_list))

    def test_hard_timeout_stops_real_nonphysics_child(self):
        scan = self.scan()
        started = time.monotonic()
        # CLI execution retains the script directory; spawned children inherit it.
        with patch.object(sys, 'path', [str(WORK), *sys.path]), self.assertRaises(TimeoutError):
            scan.run_module_call_with_hard_timeout('time', 'sleep', args=(20,), timeout_s=.3)
        self.assertLess(time.monotonic() - started, 6.)

    def test_direct_density_backend_passes_requested_xi_to_both_existing_methods(self):
        scan = self.scan()
        domain, _ = scan.load_inputs()
        domain = dict(domain, xi=scan.RUN_XI)
        point = scan.point_on_ray(domain, 0, .1)
        calls = []
        def direct(density, temperature, **kwargs):
            calls.append(('density', density, temperature))
            return dict(muB_0minus=1100., upstream_density_relative_residual=1e-10)
        def bound(*args, **kwargs):
            calls.append(('method', args, kwargs))
            return dict(u_0minus=1e-6, jB=1.)
        phase = types.SimpleNamespace(_validated_pnm_state_from_nB=direct,
                                      analytic_velocity_bound=bound, semi_analytic_velocity_bound=bound)
        package = types.ModuleType('RMFsolver')
        package.phase_velocity = phase
        package.RMFparameter = types.SimpleNamespace(paraQMCRMF3=object())
        with patch.dict(sys.modules, {'RMFsolver': package}):
            for method in scan.METHODS:
                row = scan.engine.evaluate_method(point, domain, method)
                self.assertTrue(scan.good(row))
        self.assertEqual([c[0] for c in calls], ['density', 'method', 'density', 'method'])
        for call in calls[1::2]:
            self.assertEqual(call[2]['xi'], -.85)
            self.assertEqual(call[2]['T_0plus'], .1)

    def test_backend_rejects_explicit_solver_failure_and_bad_density_residual(self):
        scan = self.scan()
        domain, _ = scan.load_inputs()
        domain = dict(domain, xi=scan.RUN_XI)
        point = scan.point_on_ray(domain, 0, .1)
        package = types.ModuleType('RMFsolver')
        package.RMFparameter = types.SimpleNamespace(paraQMCRMF3=object())
        from unittest.mock import Mock
        bound = Mock(return_value=dict(success=False, u_0minus=1e-6, jB=1., message='Rejected physical branch'))
        direct = Mock(return_value=dict(muB_0minus=1100., upstream_density_relative_residual=0.))
        package.phase_velocity = types.SimpleNamespace(_validated_pnm_state_from_nB=direct,
            analytic_velocity_bound=bound, semi_analytic_velocity_bound=bound)
        with patch.dict(sys.modules, {'RMFsolver': package}):
            for method in scan.METHODS:
                with self.assertRaisesRegex(RuntimeError, 'Rejected physical branch'):
                    scan.engine.evaluate_method(point, domain, method)
            bound.reset_mock()
            for residual in (np.nan, np.inf, 1.):
                direct.return_value = dict(muB_0minus=1100., upstream_density_relative_residual=residual)
                with self.assertRaisesRegex(RuntimeError, 'density residual'):
                    scan.engine.evaluate_method(point, domain, 'analytic')
            bound.assert_not_called()

    def test_failed_sample_does_not_limit_outward_search_or_close_bracket(self):
        scan = self.scan()
        domain, _ = scan.load_inputs()
        records = {}
        for fraction, speed, success in ((.85, 3000, True), (.90, 0, False), (.99, 10000, True)):
            point = scan.point_on_ray(domain, 1, fraction)
            records[scan.point_key(point)] = result(scan, point, speed, success)
        self.assertIsNone(scan.ray_bracket(records, 1, 5000, .1)['keys'])
        self.assertGreater(scan.next_probe(records, 1, 5000, .01, .999), .90)

    def test_failed_midpoint_is_not_scheduled_repeatedly(self):
        scan = self.scan()
        domain, _ = scan.load_inputs()
        records = {}
        for fraction, speed, success in ((.95, 4000, True), (.97, 0, False), (.99, 9000, True)):
            point = scan.point_on_ray(domain, 0, fraction)
            records[scan.point_key(point)] = result(scan, point, speed, success)
        probe = scan.next_probe(records, 0, 5000, .01, .999)
        self.assertNotIn(scan.point_key(scan.point_on_ray(domain, 0, probe)), records)
        self.assertLess(probe, .99)
        self.assertGreater(probe, .95)

    def test_checkpoint_precedes_progress_and_retries_keep_previous_success(self):
        scan = self.scan()
        point = dict(theta_index=0, radial_fraction=.1)
        first = result(scan, point)
        first['semi_analytic']['success'] = False
        rows, events = {}, []
        save = lambda: events.append(('save', rows[scan.point_key(point)]['semi_analytic']['success']))
        count = lambda: events.append(('progress',))
        scan.accept_result(rows, first, save, count)
        successful = rows[scan.point_key(point)]['analytic'].copy()
        scan.accept_result(rows, result(scan, point, 9000), save, count)
        self.assertEqual(rows[scan.point_key(point)]['analytic'], successful)
        self.assertTrue(rows[scan.point_key(point)]['semi_analytic']['success'])
        self.assertEqual([e[0] for e in events], ['save', 'progress', 'save', 'progress'])
        scan.accept_result(rows, result(scan, point, 12000), save, count)
        self.assertEqual(len(events), 4)

    def test_mocked_full_run_resume_and_changed_contract(self):
        scan = self.scan()
        original_batch, original_bar = scan.run_batch, scan.tqdm
        calls = []
        def evaluator(point, *args, **kwargs):
            calls.append(point)
            return result(scan, point, 5000 * point['radial_fraction'] / .965)
        def batch(*args, **kwargs):
            return original_batch(*args, **kwargs, evaluator=evaluator)
        with tempfile.TemporaryDirectory() as tmp, patch.object(scan, 'run_batch', batch), \
                patch.object(scan, 'tqdm', lambda *a, **k: original_bar(*a, **k, disable=True)):
            output = Path(tmp) / 'comparison.npy'
            command = ['--workers', '4', '--output', str(output)]
            self.assertEqual(scan.main(command), 0)
            payload = np.load(output, allow_pickle=True).item()
            self.assertEqual(payload['xi'], -.85)
            self.assertEqual(payload['geometry_source_xi'], -.5)
            self.assertEqual(payload['numerical_reference_grid']['xi'], -.5)
            self.assertTrue(payload['complete_5000_contour'])
            self.assertEqual(payload['plot_grid']['point_keys'].shape, (30, 21))
            self.assertTrue(any('RMFsolver/phase_velocity.py' in key
                                for key in payload['contract']['source_sha256']))
            call_count = len(calls)
            self.assertEqual(scan.main(command), 0)
            self.assertEqual(len(calls), call_count)
            with self.assertRaisesRegex(ValueError, 'contract'):
                scan.main(command + ['--method-timeout', '30'])
            self.assertFalse(list(Path(tmp).glob('tmp*.npy')))

    def test_resume_retries_only_failed_method_and_keeps_success_exactly(self):
        scan = self.scan()
        original_batch, original_bar = scan.run_batch, scan.tqdm
        attempts = []
        def evaluator(point, *args, methods, **kwargs):
            attempts.append((scan.point_key(point), methods))
            row = result(scan, point)
            if len(attempts) == 1:
                row['semi_analytic']['success'] = False
            return dict(point=point, **{method: row[method] for method in methods})
        def batch(*args, **kwargs):
            return original_batch(*args, **kwargs, evaluator=evaluator)
        with tempfile.TemporaryDirectory() as tmp, patch.object(scan, 'run_batch', batch), \
                patch.object(scan, 'tqdm', lambda *a, **k: original_bar(*a, **k, disable=True)):
            output = Path(tmp) / 'trial.npy'
            actual_output = Path(tmp) / 'trial-smoke.npy'
            command = ['--smoke', '--workers', '1', '--output', str(output)]
            self.assertEqual(scan.main(command), 2)
            first = np.load(actual_output, allow_pickle=True).item()
            first_key = attempts[0][0]
            original_method = first['records'][first_key]['analytic'].copy()
            self.assertEqual(scan.main(command), 0)
            self.assertEqual(len(attempts), 4)
            self.assertEqual(attempts[-1], (first_key, ('semi_analytic',)))
            second = np.load(actual_output, allow_pickle=True).item()
            self.assertEqual(second['records'][first_key]['analytic'], original_method)
            self.assertEqual(len(second['records'][first_key]['retry_history']), 1)
            self.assertFalse(output.exists())

    def test_legacy_checkpoint_is_rejected_before_any_work(self):
        scan = self.scan()
        with tempfile.TemporaryDirectory() as tmp, patch.object(scan, 'run_batch') as batch:
            output = Path(tmp) / 'legacy.npy'
            np.save(output, dict(xi=-.5, records={}), allow_pickle=True)
            original = output.read_bytes()
            with self.assertRaisesRegex(ValueError, 'contract'):
                scan.main(['--output', str(output)])
            batch.assert_not_called()
            self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
