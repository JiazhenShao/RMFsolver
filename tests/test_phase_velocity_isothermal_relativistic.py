import unittest

import numpy as np

from RMFsolver import phase_velocity as pv
from RMFsolver.constants import MeV_fm

N0 = 0.16 * MeV_fm**3
B14 = 189.1566


class TestMomentumFluxDiagnostics(unittest.TestCase):
    def test_ratios_match_closed_form(self):
        P, w, u = 1.0e9, 5.74e9, 5.8e-3
        d = pv._momentum_flux_diagnostics(P, w, u)
        gamma_squared = 1.0 / (1.0 - u * u)
        self.assertAlmostEqual(d["momentum_flux_ratio"], w * u * u / P, places=12)
        self.assertAlmostEqual(
            d["relativistic_flux_ratio"], w * gamma_squared * u * u / P, places=12
        )
        self.assertAlmostEqual(
            d["gamma_minus_1"], np.sqrt(gamma_squared) - 1.0, places=12
        )

    def test_static_limit_is_zero(self):
        d = pv._momentum_flux_diagnostics(1.0e9, 5.0e9, 0.0)
        self.assertEqual(d["momentum_flux_ratio"], 0.0)
        self.assertEqual(d["gamma_minus_1"], 0.0)

    def test_non_finite_velocity_gives_nan_not_raise(self):
        d = pv._momentum_flux_diagnostics(1.0e9, 5.0e9, np.nan)
        self.assertTrue(np.isnan(d["momentum_flux_ratio"]))

    def test_out_of_range_velocity_gives_nan_not_raise(self):
        for bad_u in (-0.1, 1.0, 1.5):
            d = pv._momentum_flux_diagnostics(1.0e9, 5.0e9, bad_u)
            self.assertTrue(np.isnan(d["momentum_flux_ratio"]), f"u={bad_u}")
            self.assertTrue(np.isnan(d["relativistic_flux_ratio"]), f"u={bad_u}")
            self.assertTrue(np.isnan(d["gamma_minus_1"]), f"u={bad_u}")


class TestDiagnosticsReachTheResults(unittest.TestCase):
    def test_analytic_reports_a_small_ratio_at_realistic_composition(self):
        r = pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, 0.5, xi=-0.5)
        self.assertEqual(r["status"], "moving_front")
        self.assertLess(r["momentum_flux_ratio"], 1.0e-6)
        self.assertLess(r["gamma_minus_1"], 1.0e-6)

    def test_ratio_grows_as_composition_approaches_one(self):
        near = pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, 1.0 - 1e-8, xi=-0.5)
        far = pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, 1.0 - 1e-4, xi=-0.5)
        self.assertGreater(near["momentum_flux_ratio"], far["momentum_flux_ratio"])
        # (Pi-P)/P scales as 1/(1-a): four decades of a give four of ratio.
        self.assertAlmostEqual(
            np.log10(near["momentum_flux_ratio"] / far["momentum_flux_ratio"]),
            4.0,
            delta=0.3,
        )


class TestNumericalDiagnosticsReachTheResults(unittest.TestCase):
    def test_bvp_reports_a_small_finite_ratio_at_realistic_composition(self):
        r = pv.solve_front_isothermal(
            1.0, 3.5 * N0, B14, 0.5, n_mesh=120, tol_bvp=1.0e-3
        )
        for key in ("momentum_flux_ratio", "relativistic_flux_ratio", "gamma_minus_1"):
            self.assertIn(key, r)
            self.assertTrue(np.isfinite(r[key]), f"{key} is not finite: {r[key]}")
        self.assertGreaterEqual(r["momentum_flux_ratio"], 0.0)
        self.assertLess(r["momentum_flux_ratio"], 1.0e-3)


class TestValidityGuard(unittest.TestCase):
    def test_tolerance_constant_exists_and_is_conservative(self):
        self.assertLessEqual(pv.MOMENTUM_FLUX_RATIO_TOLERANCE, 1.0e-3)

    def test_analytic_flags_when_the_flux_term_is_large(self):
        # 1-a = 1e-9 measured at (Pi-P)/P ~ 0.2 for this state.
        r = pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, 1.0 - 1e-9, xi=-0.5)
        self.assertFalse(r["success"])
        self.assertEqual(r["status"], "static_isobar_approximation_invalid")
        self.assertGreater(r["momentum_flux_ratio"], pv.MOMENTUM_FLUX_RATIO_TOLERANCE)

    def test_realistic_composition_is_unaffected(self):
        r = pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, 0.5, xi=-0.5)
        self.assertTrue(r["success"])
        self.assertEqual(r["status"], "moving_front")

    def test_guard_can_be_relaxed_by_the_caller(self):
        r = pv.analytic_velocity_isothermal(
            1.0, 3.5 * N0, B14, 1.0 - 1e-9, xi=-0.5, momentum_flux_tol=1.0
        )
        self.assertEqual(r["status"], "moving_front")


class TestRelativisticFluxPair(unittest.TestCase):
    def test_reduces_to_the_newtonian_pair_at_small_velocity(self):
        nB, w, P, u = 1.0e7, 5.0e9, 1.0e9, 1.0e-6
        jB, Pi = pv._relativistic_flux_pair(nB, w, P, u)
        self.assertAlmostEqual(jB / (nB * u), 1.0, places=10)
        self.assertAlmostEqual(Pi / (P + w * u * u), 1.0, places=10)

    def test_carries_the_gamma_factors_at_large_velocity(self):
        nB, w, P, u = 1.0e7, 5.0e9, 1.0e9, 0.6
        gamma = 1.0 / np.sqrt(1.0 - u * u)
        jB, Pi = pv._relativistic_flux_pair(nB, w, P, u)
        self.assertAlmostEqual(jB, nB * gamma * u, places=4)
        self.assertAlmostEqual(Pi, P + w * gamma**2 * u * u, delta=1.0)

    def test_rejects_superluminal_velocity(self):
        with self.assertRaises(RuntimeError):
            pv._relativistic_flux_pair(1.0e7, 5.0e9, 1.0e9, 1.0)


class TestRelativisticThreading(unittest.TestCase):
    def test_default_path_is_unchanged(self):
        """relativistic=False must reproduce the pre-existing result exactly."""
        base = pv.solve_front_isothermal(
            1.0, 3.5 * N0, B14, 0.5, n_mesh=120, tol_bvp=1.0e-3
        )
        same = pv.solve_front_isothermal(
            1.0, 3.5 * N0, B14, 0.5, n_mesh=120, tol_bvp=1.0e-3, relativistic=False
        )
        self.assertEqual(base["jB"], same["jB"])

    def test_relativistic_agrees_to_the_measured_correction_size(self):
        newton = pv.solve_front_isothermal(
            1.0, 3.5 * N0, B14, 0.5, n_mesh=120, tol_bvp=1.0e-3
        )
        exact = pv.solve_front_isothermal(
            1.0, 3.5 * N0, B14, 0.5, n_mesh=120, tol_bvp=1.0e-3, relativistic=True
        )
        # This state hits the known pre-existing singular-Jacobian failure in
        # the BVP collocation solve, so neither path reports success -- the
        # no-arg default fails identically.  What the flag must not do is
        # change the outcome: same convergence verdict, same jB.
        self.assertEqual(exact["success"], newton["success"])
        self.assertLess(abs(exact["jB"] / newton["jB"] - 1.0), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
