"""Tests for the a_0plus=NaN thermodynamic-maximum branch.

Covers the shared ceiling solve and both isothermal entry points.  The
numerical BVP is exercised in a single case only; it costs minutes per solve.
"""

import unittest

import numpy as np

from RMFsolver import RMFparameter as para
from RMFsolver import phase_velocity as pv
from RMFsolver.constants import MeV_fm


N0 = 0.16 * MeV_fm**3
B14 = 189.1566


def _upstream(nB_over_n0, T):
    """Return (muB_0minus, P_0minus) on the branch-validated PNM branch."""
    muB = float(
        pv.muB_from_nB_physical(
            nB_over_n0 * N0, T, param=para.paraQMCRMF3,
            NM_type="PNM", auto_expand=True,
        )
    )
    state = pv._analytic_nuclear_state(
        muB, T, param=para.paraQMCRMF3, NM_type="PNM"
    )
    return muB, float(state["P_0minus"])


class TestCeilingSolve(unittest.TestCase):
    """The scalar ceiling solve itself."""

    def test_interior_root_satisfies_the_balance(self):
        muB, P = _upstream(3.5, 1.0)
        result = pv._solve_a_0plus_max(muB, P, 1.0, B14)
        self.assertEqual(result["status"], "interior")
        self.assertTrue(0.0 < result["a_0plus_max"] < 1.0)
        # muB(0-) = muB(0+) + a*muK(0+) must close at the root.
        self.assertAlmostEqual(
            result["muB_0plus"]
            + result["a_0plus_max"] * result["muK_0plus"]
            - muB,
            0.0,
            places=6,
        )

    def test_ceiling_vanishes_on_the_coexistence_boundary(self):
        """delta_muB = 0 at 3 n0 for this calibration, so a_0plus_max -> 0."""
        muB, P = _upstream(3.0, 0.01)
        result = pv._solve_a_0plus_max(muB, P, 0.01, B14)
        self.assertLess(result["a_0plus_max"], 0.05)

    def test_ceiling_saturates_beyond_the_strangeness_free_boundary(self):
        """Past nB ~ 5.3 n0 cold, even a_0plus = 1 stays favorable."""
        muB, P = _upstream(5.45, 0.01)
        result = pv._solve_a_0plus_max(muB, P, 0.01, B14)
        self.assertEqual(result["status"], "saturated")
        self.assertEqual(result["a_0plus_max"], 1.0)

    def test_residual_at_zero_composition_reduces_to_delta_muB(self):
        """g(a->0) must equal delta_muB, the a_0plus_max = 0 level set."""
        T, nB_over_n0 = 20.0, 3.2
        muB, P = _upstream(nB_over_n0, T)
        muB_equilibrated = float(
            pv._solve_muB_inf_at_muK0_for_given_Pi(P, 0.0, B14, T, ms=0.0)
        )
        delta_muB = muB_equilibrated - muB
        interface = pv._solve_interface_0plus_from_local_a_and_Pi(
            1.0e-8, P, 0.0, B14, T, ms=0.0
        )
        residual = interface[0] + 1.0e-8 * interface[1] - muB
        self.assertAlmostEqual(residual / delta_muB, 1.0, places=3)


class TestAnalyticBranch(unittest.TestCase):
    """analytic_velocity_isothermal with a_0plus omitted."""

    def test_auto_matches_explicitly_passing_the_ceiling(self):
        auto = pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, xi=-0.5)
        self.assertEqual(auto["status"], "moving_front")
        self.assertEqual(auto["a_0plus_source"], "maximum")
        explicit = pv.analytic_velocity_isothermal(
            1.0, 3.5 * N0, B14, auto["a_0plus_max"], xi=-0.5
        )
        self.assertEqual(explicit["a_0plus_source"], "input")
        self.assertAlmostEqual(
            auto["u_0minus"] / explicit["u_0minus"], 1.0, places=6
        )

    def test_resolved_composition_is_the_reported_ceiling(self):
        result = pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, xi=-0.5)
        self.assertAlmostEqual(
            result["a_0plus"], result["a_0plus_max"], places=12
        )
        self.assertTrue(0.0 < result["a_0plus"] < 1.0)

    def test_explicit_input_leaves_the_ceiling_unevaluated(self):
        result = pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, 0.3, xi=-0.5)
        self.assertEqual(result["a_0plus_source"], "input")
        self.assertEqual(result["a_0plus"], 0.3)
        self.assertTrue(np.isnan(result["a_0plus_max"]))

    def test_saturated_ceiling_reports_instead_of_returning_a_speed(self):
        result = pv.analytic_velocity_isothermal(0.01, 5.45 * N0, B14, xi=-0.5)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "composition_ceiling_saturated")
        self.assertEqual(result["a_0plus_max"], 1.0)

    def test_stable_neutron_matter_still_gates_before_the_ceiling(self):
        result = pv.analytic_velocity_isothermal(1.0, 2.0 * N0, B14, xi=-0.5)
        self.assertFalse(result["front_exists"])
        self.assertGreater(result["delta_muB"], 0.0)

    def test_out_of_range_composition_still_raises(self):
        with self.assertRaises(RuntimeError):
            pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, 1.5)
        with self.assertRaises(RuntimeError):
            pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, -0.1)


class TestNumericalBranch(unittest.TestCase):
    """solve_front_isothermal with a_0plus omitted."""

    def test_auto_resolves_to_the_same_ceiling_as_the_helper(self):
        T, nB_over_n0 = 1.0, 3.5
        muB, P = _upstream(nB_over_n0, T)
        expected = pv._solve_a_0plus_max(muB, P, T, B14)["a_0plus_max"]
        result = pv.solve_front_isothermal(
            T, nB_over_n0 * N0, B14, n_mesh=120, tol_bvp=1.0e-3
        )
        self.assertEqual(result["a_0plus_source"], "maximum")
        self.assertAlmostEqual(result["a_0plus"], expected, places=8)

    def test_out_of_range_composition_still_raises(self):
        with self.assertRaises(RuntimeError):
            pv.solve_front_isothermal(1.0, 3.5 * N0, B14, 0.0)
        with self.assertRaises(RuntimeError):
            pv.solve_front_isothermal(1.0, 3.5 * N0, B14, 1.0)


if __name__ == "__main__":
    unittest.main()
