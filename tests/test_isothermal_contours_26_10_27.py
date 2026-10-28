"""Isothermal contour workflow of 2026-10-27 (high-T polar mesh, shared engine)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "new_paper_calculations/26-10-27/isothermal"
sys.path[:0] = [str(WORKFLOW), str(WORKFLOW / "highT")]

from RMFsolver import phase_velocity as pv  # noqa: E402

SAVED_DOMAIN = ROOT / "new_paper_calculations/26-09-16/isothermal_numeric_repair/isothermal-domain.npy"


class BoundaryTests(unittest.TestCase):
    def test_polar_round_trip(self):
        import _iso_common as common
        rho = np.array([0.1, 0.5, 0.93])
        theta = np.array([0.05, 0.8, np.pi / 2])
        x, T = common.to_physical(rho, theta)
        back_rho, back_theta = common.to_polar(x, T)
        np.testing.assert_allclose(back_rho, rho, rtol=1e-12)
        np.testing.assert_allclose(back_theta, theta, rtol=1e-12)
        self.assertEqual(float(common.to_physical(0.7, np.pi / 2)[0]), 1.0)

    def test_boundary_status_on_each_side(self):
        import _iso_common as common
        import _iso_boundaries as boundaries
        expected = (("stable_neutron_matter", "moving_front"),
                    ("moving_front", "composition_ceiling_saturated"))
        for degrees in (10.0, 45.0, 80.0):
            ray = boundaries.ray_boundaries(np.radians(degrees))
            for rho, (inside, outside) in zip((ray["rho_inner"], ray["rho_outer"]), expected):
                for factor, status in ((1 - 1e-4, inside), (1 + 1e-4, outside)):
                    x, T = common.to_physical(rho * factor, ray["theta"])
                    result = pv.analytic_velocity_isothermal_slow(
                        float(T), float(x) * common.N0_MEV3, common.PHYSICS["B_one_forth"], xi=common.XI)
                    with self.subTest(degrees=degrees, factor=factor):
                        self.assertEqual(result["status"], status)

    def test_boundaries_match_probe(self):
        import _iso_boundaries as boundaries
        window = boundaries.fixed_T_window(10.0)
        self.assertEqual(window["status"], "open")
        self.assertTrue(2.75 < window["x_inner"] < 3.0, window)
        self.assertTrue(5.25 < window["x_outer"] < 5.5, window)
        self.assertEqual(boundaries.fixed_T_window(97.0)["status"], "closed")

    def test_boundaries_cross_check_saved_domain(self):
        import _iso_common as common
        import _iso_boundaries as boundaries
        saved = np.load(SAVED_DOMAIN, allow_pickle=True).item()
        for kind, prefix in (("rho_inner", "lower_phase"), ("rho_outer", "upper_phase")):
            x = np.asarray(saved[prefix + "_nB_0minus"], dtype=float)
            x = x / common.N0_MEV3 if np.max(x) > 100 else x
            saved_rho, saved_theta = common.to_polar(x, np.asarray(saved[prefix + "_T_0minus_MeV"], dtype=float))
            for index in (3, 8, 12, 16, 20, 23):
                ray = boundaries.ray_boundaries(float(saved_theta[index]))
                with self.subTest(kind=kind, index=index):
                    self.assertLessEqual(abs(ray[kind] - saved_rho[index]), 1e-3)

    def test_boundary_curves_cover_both_figures(self):
        import _iso_boundaries as boundaries
        curves = boundaries.boundary_curves(n_rows=6, n_rays=8, workers=4)
        for name, top in (("inner", 60.0), ("outer", 94.0)):
            x, T = curves[name]
            self.assertLessEqual(T.min(), 0.0100001)
            self.assertGreaterEqual(T.max(), top)
            self.assertTrue(np.all(np.diff(T) > 0))
            self.assertTrue(np.all(np.diff(x) < 0), name)


if __name__ == "__main__":
    unittest.main()
