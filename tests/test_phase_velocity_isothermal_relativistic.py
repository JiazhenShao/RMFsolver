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
        # u is the proper velocity gamma*v, so gamma = sqrt(1 + u**2).
        gamma_squared = 1.0 + u * u
        self.assertAlmostEqual(d["momentum_flux_ratio"], w * u * u / P, places=12)
        self.assertAlmostEqual(
            d["gamma_minus_1"], np.sqrt(gamma_squared) - 1.0, places=12
        )

    def test_gamma_uses_the_proper_velocity_convention(self):
        # Same convention as _relativistic_gamma_from_u, which the
        # energy-conserving solver has used all along.
        for u in (0.0, 5.8e-3, 0.5, 1.0, 7.0):
            d = pv._momentum_flux_diagnostics(1.0e9, 5.0e9, u)
            self.assertAlmostEqual(
                d["gamma_minus_1"] + 1.0, pv._relativistic_gamma_from_u(u), places=12
            )

    def test_no_separate_relativistic_ratio_is_reported(self):
        # w*gamma**2*v**2 == w*u**2 identically, so a second key would just
        # duplicate momentum_flux_ratio -- or, as before, double-count gamma**2.
        d = pv._momentum_flux_diagnostics(1.0e9, 5.0e9, 5.8e-3)
        self.assertNotIn("relativistic_flux_ratio", d)

    def test_static_limit_is_zero(self):
        d = pv._momentum_flux_diagnostics(1.0e9, 5.0e9, 0.0)
        self.assertEqual(d["momentum_flux_ratio"], 0.0)
        self.assertEqual(d["gamma_minus_1"], 0.0)

    def test_non_finite_velocity_gives_nan_not_raise(self):
        d = pv._momentum_flux_diagnostics(1.0e9, 5.0e9, np.nan)
        self.assertTrue(np.isnan(d["momentum_flux_ratio"]))

    def test_negative_velocity_gives_nan_not_raise(self):
        d = pv._momentum_flux_diagnostics(1.0e9, 5.0e9, -0.1)
        self.assertTrue(np.isnan(d["momentum_flux_ratio"]))
        self.assertTrue(np.isnan(d["gamma_minus_1"]))

    def test_proper_velocity_above_one_is_physical_not_nan(self):
        # u = 1 is v = 1/sqrt(2); the old guard NaN-ed the whole diagnostic.
        for u in (1.0, 1.5, 12.0):
            d = pv._momentum_flux_diagnostics(1.0e9, 5.0e9, u)
            self.assertTrue(np.isfinite(d["momentum_flux_ratio"]), f"u={u}")
            self.assertTrue(np.isfinite(d["gamma_minus_1"]), f"u={u}")


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
        for key in ("momentum_flux_ratio", "gamma_minus_1"):
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


class TestClosureIsAlreadyRelativistic(unittest.TestCase):
    def test_quark_momentum_flux_equals_the_exact_relativistic_pair(self):
        """Pi - P must equal w*(jB/nB)**2 exactly, at every flux.

        ``u = jB/nB`` is the proper velocity gamma*v, so w*u**2 IS
        w*gamma**2*v**2: inverting jB = nB*gamma*v gives v = x/sqrt(1 + x**2)
        with x = jB/nB, hence gamma**2 = 1 + x**2 and gamma**2*v**2 = x**2
        identically.  The closure needs no relativistic correction, and this
        test is the guard against anyone "fixing" it by adding one.
        """
        muB, muK, T = 1100.0, 20.0, 1.0
        nB = float(pv.nB_QM(muB, muK, B14, T, upB=5000))
        P = float(pv.PQM(muB, muK, B14, T, upB=5000))
        w = P + float(pv.edensQM(muB, muK, B14, T, include_em=False, upB=5000))
        self.assertGreater(nB, 0.0)
        for jB in (1.0e5, 1.0e6, 5.0e6, 2.0e7):
            with self.subTest(jB=jB):
                x = jB / nB
                Pi = pv._Pi_QM_state(muB, muK, B14, T, jB)
                self.assertAlmostEqual(Pi / (P + w * x * x), 1.0, delta=1.0e-13)
                # ... and spelled out through the 3-velocity, the long way.
                v = x / np.sqrt(1.0 + x * x)
                gamma_squared = 1.0 / (1.0 - v * v)
                self.assertAlmostEqual(
                    Pi / (P + w * gamma_squared * v * v), 1.0, delta=1.0e-13
                )


if __name__ == "__main__":
    unittest.main()
