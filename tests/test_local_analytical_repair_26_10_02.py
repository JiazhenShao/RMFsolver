import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'new_paper_calculations/26-10-02/repair_analytical_local.py'


class LocalRepairTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('local_analytic_repair_test', SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_secant_reuses_saved_bracket(self):
        rows = [dict(nB_0minus_over_n0=2., velocity_m_s=100.),
                dict(nB_0minus_over_n0=3., velocity_m_s=300.)]
        self.assertEqual(self.module.propose_density(rows, 200., (1., 5.)), 2.5)

    def test_clipped_edge_does_not_stagnate(self):
        rows = [dict(nB_0minus_over_n0=1.000001, velocity_m_s=2300.)]
        density = self.module.propose_density(rows, 5000., (1., 1.016))
        self.assertGreater(density, 1.008)
        self.assertLess(density, 1.016)

    def test_overlay_preserves_original_success_and_updates_only_missing(self):
        baseline = dict(records={'x': {'analytic': {'success': True, 'u_0minus': 1.}}},
                        plot_grid={}, numerical_reference_grid={'u_0minus_m_s': np.array([5.])})
        overlay = dict(adiabatic={'x': {'analytic': {'success': True, 'u_0minus': 2.},
                                       'semi_analytic': {'success': True, 'u_0minus': 3.}}})
        merged = self.module.merge_adiabatic(baseline, overlay)
        self.assertEqual(merged['records']['x']['analytic']['u_0minus'], 1.)
        self.assertEqual(merged['records']['x']['semi_analytic']['u_0minus'], 3.)
        self.assertNotIn('semi_analytic', baseline['records']['x'])
        np.testing.assert_equal(merged['numerical_reference_grid']['u_0minus_m_s'], [5.])

    def test_overlay_isothermal_does_not_overwrite_success(self):
        baseline = dict(targets={'x': {'status': 'success', 'root': {'id': 'old'}},
                                 'y': {'status': 'unresolved'}})
        overlay = dict(isothermal={'x': {'status': 'success', 'root': {'id': 'new'}},
                                    'y': {'status': 'success', 'root': {'id': 'recovered'}}})
        merged = self.module.merge_isothermal(baseline, overlay)
        self.assertEqual(merged['targets']['x']['root']['id'], 'old')
        self.assertEqual(merged['targets']['y']['root']['id'], 'recovered')
        self.assertEqual(baseline['targets']['y']['status'], 'unresolved')

    def test_missing_entire_ray_is_selected_from_neighboring_rays(self):
        velocity = np.array([[20., 200., 3000.], [np.nan]*3, [20., 200., 3000.]])
        valid = np.isfinite(velocity)
        keys = np.array([[f'{i}:{j/3:.12f}' for j in range(3)] for i in range(3)])
        grid = dict(point_keys=keys, radial_fraction=np.tile([.1, .5, .9], (3, 1)))
        for method in ('analytic', 'semi_analytic'):
            grid[method+'_success'] = valid.copy()
            grid[method+'_u_0minus_m_s'] = velocity.copy()
        records = {key: {'analytic': {'success': bool(valid[ij])}} for ij,key in np.ndenumerate(keys)}
        tasks = self.module.adiabatic_tasks(dict(plot_grid=grid, records=records))
        self.assertTrue(all((key, 'semi_analytic') in tasks for key in keys[1]))


if __name__ == '__main__':
    unittest.main()
