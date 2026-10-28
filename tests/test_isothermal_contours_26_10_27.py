"""Isothermal contour workflow of 2026-10-27 (high-T polar mesh, shared engine)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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


class PolarMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import polar_mesh
        cls.mesh = polar_mesh.build_mesh(workers=4)

    def test_mesh_is_strictly_inside_ring(self):
        mesh = self.mesh
        inner = np.array([b["rho_inner"] for b in mesh["boundaries"]])[:, None]
        outer = np.array([b["rho_outer"] for b in mesh["boundaries"]])[:, None]
        pending = mesh["point_status"] == "pending"
        self.assertTrue(np.all((mesh["rho"] > inner)[pending]))
        self.assertTrue(np.all((mesh["rho"] < outer)[pending]))

    def test_mesh_panel_mask(self):
        mesh = self.mesh
        pending = mesh["point_status"] == "pending"
        self.assertTrue(np.all((mesh["T"][pending] >= 8.0) & (mesh["T"][pending] <= 100.0)))
        outside = mesh["point_status"] == "outside_panel"
        self.assertTrue(np.all((mesh["T"][outside] < 8.0) | (mesh["T"][outside] > 100.0)))

    def test_mesh_counts(self):
        self.assertEqual(self.mesh["rho"].shape, (41, 29))
        self.assertGreaterEqual(np.sum(self.mesh["point_status"] == "pending"), 900)

    def test_ninety_degree_ray_is_n0_edge(self):
        self.assertEqual(self.mesh["theta"][-1], np.pi / 2)
        np.testing.assert_array_equal(self.mesh["x"][-1], 1.0)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        import _iso_evaluate as evaluation
        self.evaluation = evaluation

    def numerical_once(self):
        if not hasattr(EvaluationTests, "_numerical"):
            import _iso_common as common
            EvaluationTests._numerical = pv.solve_front_isothermal_slow(
                10.0, 3.5 * common.N0_MEV3, common.PHYSICS["B_one_forth"], return_profile=True,
                **common.NUMERICAL)
        return dict(EvaluationTests._numerical)

    def test_reference_points(self):
        for method, speed in (("analytic", 221.7), ("numerical", 154.5)):
            record, profile = self.evaluation.evaluate(method, 3.5, 10.0)
            with self.subTest(method=method):
                self.assertEqual(record["status"], "accepted", record["message"])
                self.assertAlmostEqual(record["speed_m_s"], speed, delta=0.1)
                self.assertEqual(profile is None, method == "analytic")

    def test_analytic_intermediates_reproduce_speed(self):
        import _iso_common as common
        record, _ = self.evaluation.evaluate("analytic", 4.0, 30.0)
        a = record["a_0plus"]
        speed = common.C_M_S * np.sqrt(2 * record["I2"] / (record["lambda_n"]**2 * (1 - a)
                                                           * (1 + record["xi_used"] * a)))
        self.assertLess(abs(speed / record["speed_m_s"] - 1), 1e-12)

    def test_numerical_profile_integrals(self):
        numerical, profile = self.evaluation.evaluate("numerical", 3.5, 10.0)
        analytic, _ = self.evaluation.evaluate("analytic", 3.5, 10.0)
        self.assertLess(abs(numerical["I2_num"] / analytic["I2"] - 1), 0.3)
        self.assertTrue(-1 < numerical["xi_profile"] < 1, numerical["xi_profile"])
        self.assertEqual(len(profile["x"]), 256)

    def test_roundoff_inside_mesh_is_masked_not_failed(self):
        with patch.object(pv, "analytic_velocity_isothermal_slow",
                          return_value=dict(success=True, front_exists=False, message="",
                                            status="composition_ceiling_saturated")):
            record, _ = self.evaluation.evaluate("analytic", 5.2, 10.0)
        self.assertEqual(record["status"], "boundary_roundoff")
        with patch.object(pv, "solve_front_isothermal_slow",
                          return_value=dict(success=False, status="thermodynamic_path_failed",
                                            message="Automatic interface ceiling is not interior")):
            record, _ = self.evaluation.evaluate("numerical", 5.2, 10.0)
        self.assertEqual(record["status"], "boundary_roundoff")
        self.assertEqual(record["attempts"], 1)

    def test_nan_rms_collocation_retries_then_accepts(self):
        good = self.numerical_once()
        calls = []
        def fake(*args, **kwargs):
            calls.append(kwargs)
            return dict(good, bvp_max_rms_residual=np.nan) if len(calls) == 1 else good
        with patch.object(pv, "solve_front_isothermal_slow", side_effect=fake):
            record, _ = self.evaluation.evaluate("numerical", 3.5, 10.0)
        self.assertEqual(record["status"], "accepted")
        self.assertEqual(record["attempts"], 2)
        self.assertEqual(calls[1]["n_mesh"], 600)

    def test_shooting_branch_requires_flux_change_and_mismatch(self):
        good = self.numerical_once()
        for change, expected in ((2e-6, "unresolved"), (1e-9, "accepted")):
            shot = dict(good, solution_method="shooting", bvp_max_rms_residual=np.nan,
                        shooting_flux_change=change, shooting_mismatch=1e-10)
            with patch.object(pv, "solve_front_isothermal_slow", return_value=shot):
                record, _ = self.evaluation.evaluate("numerical", 3.5, 10.0)
            with self.subTest(change=change):
                self.assertEqual(record["status"], expected)

    def test_two_attempt_cap(self):
        calls = []
        def failing(*args, **kwargs):
            calls.append(kwargs)
            return dict(success=False, status="bvp_failed", message="singular Jacobian")
        with patch.object(pv, "solve_front_isothermal_slow", side_effect=failing):
            record, _ = self.evaluation.evaluate("numerical", 3.5, 10.0)
        self.assertEqual(record["status"], "unresolved")
        self.assertEqual(len(calls), 2)
        self.assertEqual(record["message"], "singular Jacobian")


if __name__ == "__main__":
    unittest.main()
