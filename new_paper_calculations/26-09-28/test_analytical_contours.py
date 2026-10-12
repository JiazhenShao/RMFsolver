import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_T0plus0p1_analytical as scan


def result(point, velocity=4000.0, success=True):
    method = dict(success=success, u_0minus=velocity / scan.SPEED_OF_LIGHT,
                  jB=1.0, message="" if success else "mock failure")
    return dict(point=point, analytic=dict(method), semi_analytic=dict(method))


class GridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain, cls.numerical = scan.load_inputs()
        cls.points, cls.matched_keys, cls.domain_keys = scan.build_points(
            cls.domain, cls.numerical)

    def test_union_preserves_exact_numerical_coordinates_and_endpoints(self):
        self.assertEqual(len(self.points), 630)
        self.assertEqual(self.matched_keys.shape, (30, 18))
        self.assertEqual(self.domain_keys.shape, (30, 20))
        for ij in np.ndindex(self.matched_keys.shape):
            p = self.points[self.matched_keys[ij]]
            self.assertEqual(p['T_0minus_MeV'], self.numerical['plot_T_0minus_MeV_grid'][ij])
            self.assertEqual(p['nB_0minus_over_n0'], self.numerical['plot_nB_0minus_over_n0_grid'][ij])
        self.assertEqual(self.points[self.matched_keys[0, 0]]['theta_rad'], 0)
        self.assertEqual(self.points[self.matched_keys[-1, 0]]['theta_rad'], np.pi / 2)
        self.assertTrue(all(self.points[k]['T_0minus_MeV'] == 0 for k in self.matched_keys[0]))

    def test_rejects_mismatched_calibration(self):
        bad = dict(self.numerical, calibration=dict(self.numerical['calibration'], B14_MeV=180.0))
        with self.assertRaisesRegex(ValueError, 'calibration'):
            scan.build_points(self.domain, bad)

    def test_new_point_uses_saved_ray_transform_and_stays_inside(self):
        p = scan.point_on_ray(self.domain, 0, 0.975)
        self.assertEqual(p['T_0minus_MeV'], 0)
        self.assertGreater(p['nB_0minus_over_n0'], 3)
        with self.assertRaises(ValueError):
            scan.point_on_ray(self.domain, 0, 1.0)


class BracketTests(unittest.TestCase):
    def setUp(self):
        self.domain, _ = scan.load_inputs()

    def records(self, samples):
        rows = {}
        for rho, velocity, good in samples:
            p = scan.point_on_ray(self.domain, 0, rho)
            rows[scan.point_key(p)] = result(p, velocity, good)
        return rows

    def test_refines_bracket_then_accepts_only_narrow_converged_pair(self):
        rows = self.records([(0.95, 4000, True), (0.99, 9000, True)])
        report = scan.ray_bracket(rows, 0, 5000, 0.01)
        self.assertFalse(report['complete'])
        rho = scan.next_probe(rows, 0, 5000, 0.01, 0.999)
        self.assertAlmostEqual(rho, 0.97)
        rows.update(self.records([(0.96, 4800, True), (0.965, 5400, True)]))
        self.assertTrue(scan.ray_bracket(rows, 0, 5000, 0.01)['complete'])

    def test_never_bridges_failed_cell_or_fabricates_solver_values(self):
        rows = self.records([(0.95, 4000, True), (0.97, 0, False), (0.99, 9000, True)])
        self.assertIsNone(scan.ray_bracket(rows, 0, 5000, 0.1)['keys'])
        self.assertAlmostEqual(scan.next_probe(rows, 0, 5000, 0.01, 0.999), 0.96)

    def test_extends_outward_but_never_to_boundary(self):
        rows = self.records([(0.95, 4000, True), (0.99, 4500, True)])
        probe = scan.next_probe(rows, 0, 5000, 0.01, 0.999)
        self.assertGreater(probe, 0.99)
        self.assertLess(probe, 1)

    def test_every_ray_required_for_complete_curve(self):
        rows = {}
        for theta in range(30):
            for rho, v in ((0.97, 4900), (0.975, 5200)):
                p = scan.point_on_ray(self.domain, theta, rho)
                rows[scan.point_key(p)] = result(p, v)
        self.assertTrue(scan.coverage(rows, 30, 5000, 0.01)['complete'])
        rows.pop(scan.point_key(scan.point_on_ray(self.domain, 29, 0.975)))
        self.assertFalse(scan.coverage(rows, 30, 5000, 0.01)['complete'])


class CheckpointTests(unittest.TestCase):
    def test_each_terminal_point_is_saved_and_counted_once(self):
        domain, numerical = scan.load_inputs()
        points, matched, original = scan.build_points(domain, numerical)
        subset = list(points.values())[:3]
        rows, saves, updates = {}, [], []
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'checkpoint.npy'
            def save():
                scan.atomic_save(output, dict(records=rows))
                saves.append(len(np.load(output, allow_pickle=True).item()['records']))
            for p in subset:
                scan.accept_result(rows, result(p), save, lambda: updates.append(1))
            scan.accept_result(rows, result(subset[0]), save, lambda: updates.append(1))
        self.assertEqual(saves, [1, 2, 3])
        self.assertEqual(len(updates), 3)

    def test_invalid_success_is_not_accepted(self):
        p = dict(theta_index=0, radial_fraction=0.1)
        for value in (np.nan, np.inf, -1.0, 0.0):
            self.assertFalse(scan.good(result(p, value)['semi_analytic']))

    def test_resume_rejects_changed_contract(self):
        with self.assertRaisesRegex(ValueError, 'contract'):
            scan.validate_resume(dict(contract={'target': 5000}), {'target': 6000})

    def test_parallel_mocked_points_checkpoint_individually(self):
        domain, numerical = scan.load_inputs()
        points, _, _ = scan.build_points(domain, numerical)
        rows, counts = {}, []
        class Progress:
            n = 0
            def update(self, n): self.n += n
            def set_postfix(self, **kwargs): pass
        bar = Progress()
        scan.run_batch(list(points.values())[:5], rows, domain,
                       Namespace(workers=2, point_timeout=1),
                       lambda: counts.append(len(rows)), bar,
                       evaluator=lambda point, *_: result(point))
        self.assertEqual(counts, [1, 2, 3, 4, 5])
        self.assertEqual(bar.n, 5)

    def test_hard_timeout_really_stops_child(self):
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            scan.run_module_call_with_hard_timeout('time', 'sleep', args=(30,), timeout_s=0.3)
        self.assertLess(time.monotonic() - started, 6)

    def test_mocked_cli_smoke_saves_readable_output_and_resumes(self):
        original_batch = scan.run_batch
        def batch(*args):
            return original_batch(*args, evaluator=lambda point, *_: result(point))
        with tempfile.TemporaryDirectory() as tmp, patch.object(scan, 'run_batch', batch):
            output = Path(tmp) / 'trial.npy'
            command = ['--smoke', '--workers', '2', '--output', str(output)]
            self.assertEqual(scan.main(command), 0)
            saved = np.load(Path(tmp) / 'trial-smoke.npy', allow_pickle=True).item()
            self.assertEqual(saved['completed_point_count'], 3)
            self.assertFalse(saved['complete_5000_contour'])
            self.assertTrue(saved['run_complete'])
            with patch.object(scan, 'evaluate_point', side_effect=AssertionError('must resume')):
                self.assertEqual(scan.main(command), 0)

    def test_target_below_all_successes_does_not_repeat_failed_innermost(self):
        domain, _ = scan.load_inputs()
        outer = scan.point_on_ray(domain, 0, 0.02)
        inner = scan.point_on_ray(domain, 0, 0.01)
        rows = {scan.point_key(outer): result(outer, 6000),
                scan.point_key(inner): result(inner, 0, False)}
        probe = scan.next_probe(rows, 0, 5000, 0.01, 0.999)
        self.assertIsNotNone(probe)
        self.assertNotIn(scan.point_key(scan.point_on_ray(domain, 0, probe)), rows)

    def test_mocked_full_grid_refines_all_rays_to_complete_target(self):
        original_batch, original_bar = scan.run_batch, scan.tqdm
        def batch(*args):
            return original_batch(*args, evaluator=lambda point, *_:
                                  result(point, 5000 * point['radial_fraction'] / 0.965))
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(scan, 'run_batch', batch), \
                patch.object(scan, 'tqdm', lambda *a, **k: original_bar(*a, **k, disable=True)):
            output = Path(tmp) / 'comparison.npy'
            self.assertEqual(scan.main(['--workers', '4', '--output', str(output)]), 0)
            saved = np.load(output, allow_pickle=True).item()
            self.assertTrue(saved['complete_5000_contour'])
            self.assertGreater(saved['completed_point_count'], 630)
            grid = saved['semi_analytic_bracket_grid']
            self.assertTrue(np.all(grid['semi_analytic_success']))
            self.assertTrue(np.all(grid['semi_analytic_u_0minus_m_s'][:, 0] <= 5000))
            self.assertTrue(np.all(grid['semi_analytic_u_0minus_m_s'][:, 1] >= 5000))
            self.assertTrue(np.all(np.diff(grid['radial_fraction'], axis=1) <= 0.01 + 1e-12))


if __name__ == '__main__':
    unittest.main()
