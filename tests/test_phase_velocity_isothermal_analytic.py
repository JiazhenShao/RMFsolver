import inspect
import unittest
from unittest.mock import patch

import numpy as np

from RMFsolver import RMFparameter as para
from RMFsolver import constants as const
from RMFsolver import phase_velocity as pv


N0 = 0.16 * const.MeV_fm**3


class AnalyticVelocityIsothermalApiTests(unittest.TestCase):
    def test_adaptive_log_root_is_not_clipped_at_one_or_one_trillionth(self):
        for target in (2.0, 1.0e-15):
            evaluations = []

            def evaluate_log_u(theta):
                u_0minus = float(np.exp(theta))
                evaluations.append(u_0minus)
                return target**2 - u_0minus**2, {"u_0minus": u_0minus}

            result = pv._solve_analytic_isothermal_log_root(evaluate_log_u)

            self.assertAlmostEqual(
                result["u_0minus"] / target,
                1.0,
                places=10,
            )
            self.assertLess(len(evaluations), 40)

    def test_adaptive_log_root_reuses_bracket_endpoint_evaluations(self):
        target = 2.0
        call_counts = {}

        def branch_sensitive_evaluate(theta):
            theta = float(theta)
            call_counts[theta] = call_counts.get(theta, 0) + 1
            u_0minus = float(np.exp(theta))
            residual = target**2 - u_0minus**2
            if call_counts[theta] > 1:
                residual = -residual
            return residual, {"u_0minus": u_0minus}

        result = pv._solve_analytic_isothermal_log_root(branch_sensitive_evaluate)

        self.assertAlmostEqual(result["u_0minus"] / target, 1.0, places=10)
        self.assertEqual(max(call_counts.values()), 1)

    def test_adaptive_log_root_reports_slow_front_limit_without_branch_jump(self):
        for target in (1.0, 2.0):
            def evaluate_log_u(theta):
                u_0minus = float(np.exp(theta))
                return target**2 - u_0minus**2, {"u_0minus": u_0minus}

            with self.assertRaises(pv._AnalyticIsothermalSlowFrontInvalid) as raised:
                pv._solve_analytic_isothermal_log_root(
                    evaluate_log_u,
                    max_u_0minus=1.0,
                )

            self.assertEqual(raised.exception.limit_data["u_0minus"], 1.0)

    def test_public_api_has_the_requested_signature(self):
        self.assertIn("analytic_velocity_isothermal", pv.__all__)
        function = getattr(pv, "analytic_velocity_isothermal")
        signature = inspect.signature(function)

        self.assertEqual(
            list(signature.parameters),
            [
                "T_0minus",
                "nB_0minus",
                "B_one_forth",
                "a_0plus",
                "xi",
                "ms",
                "param",
                "NM_type",
                "upB",
                "delta_muB_tol",
            ],
        )
        for name in (
            "xi",
            "ms",
            "param",
            "NM_type",
            "upB",
            "delta_muB_tol",
        ):
            self.assertEqual(
                signature.parameters[name].kind,
                inspect.Parameter.KEYWORD_ONLY,
            )
        self.assertEqual(signature.parameters["xi"].default, 0.0)
        self.assertEqual(signature.parameters["ms"].default, 0.0)
        self.assertIs(signature.parameters["param"].default, para.paraQMCRMF3)
        self.assertEqual(signature.parameters["NM_type"].default, "PNM")
        self.assertEqual(signature.parameters["upB"].default, 5000)
        self.assertEqual(signature.parameters["delta_muB_tol"].default, 1.0e-6)

    @staticmethod
    def _nuclear_state(nB_0minus):
        return {
            "P_0minus": 20.0,
            "e_0minus": 80.0,
            "h_0minus": 100.0,
            "nB_0minus": float(nB_0minus),
            "h_over_nB_0minus": 100.0 / float(nB_0minus),
        }

    def _call_with_static_candidate(self, muB_qm_candidate, *, temperature=10.0, a_0plus=0.4):
        nB_0minus = 2.0
        with (
            patch.object(pv, "muB_from_nB_physical", return_value=1000.0),
            patch.object(
                pv,
                "_analytic_nuclear_state",
                return_value=self._nuclear_state(nB_0minus),
            ),
            patch.object(
                pv,
                "_solve_muB_inf_at_muK0_for_given_Pi",
                return_value=float(muB_qm_candidate),
            ),
            patch.object(pv, "nB_QM", return_value=3.0),
            patch.object(pv, "nK_QM", return_value=0.0),
        ):
            return pv.analytic_velocity_isothermal(
                temperature,
                nB_0minus,
                180.0,
                a_0plus,
            )

    def test_positive_delta_muB_returns_stable_neutron_matter_zero(self):
        result = self._call_with_static_candidate(1010.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "stable_neutron_matter")
        self.assertEqual(result["phase_region"], "stable_neutron_matter")
        self.assertFalse(result["front_exists"])
        self.assertEqual(result["u_0minus"], 0.0)
        self.assertEqual(result["u_0minus_squared"], 0.0)
        self.assertEqual(result["jB"], 0.0)
        self.assertEqual(result["delta_muB"], 10.0)
        self.assertEqual(result["muB_qm_candidate"], 1010.0)
        self.assertEqual(result["a_0minus"], 1.0)
        self.assertTrue(np.isnan(result["D_K"]))
        self.assertTrue(np.isnan(result["gamma_K"]))
        for key in (
            "lambda_n_squared",
            "mu_q",
            "qD",
            "alpha_s",
            "analytic_denominator",
            "composition_residual",
            "momentum_flux_inf_residual",
            "momentum_flux_0plus_residual",
            "slow_front_consistent",
        ):
            self.assertIn(key, result)
        self.assertEqual(
            result["composition_definition"],
            "a_0plus_equals_nK_0plus_over_nB_0plus",
        )

    def test_delta_muB_within_tolerance_returns_coexistence_zero(self):
        result = self._call_with_static_candidate(1000.0 + 0.5e-6)

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "isothermal_coexistence")
        self.assertEqual(result["phase_region"], "isothermal_coexistence")
        self.assertFalse(result["front_exists"])
        self.assertEqual(result["u_0minus"], 0.0)
        self.assertAlmostEqual(result["muK_qm_candidate"], 0.0)

    def test_zero_temperature_quark_favored_state_is_classified_but_not_solved(self):
        result = self._call_with_static_candidate(
            990.0,
            temperature=0.0,
            a_0plus=0.4,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "zero_temperature_transport_invalid")
        self.assertEqual(result["phase_region"], "quark_matter_favored")
        self.assertFalse(result["front_exists"])
        self.assertTrue(np.isnan(result["u_0minus"]))
        self.assertEqual(result["delta_muB"], -10.0)

    def test_zero_interface_composition_returns_zero_without_transport(self):
        result = self._call_with_static_candidate(
            990.0,
            temperature=0.0,
            a_0plus=0.0,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "zero_interface_composition")
        self.assertEqual(result["phase_region"], "quark_matter_favored")
        self.assertFalse(result["front_exists"])
        self.assertEqual(result["u_0minus"], 0.0)
        self.assertEqual(result["u_0minus_formula_squared"], 0.0)

    def test_public_api_returns_structured_slow_front_invalid_result(self):
        nB_0minus = 2.0
        limit_data = {
            "u_0minus": 1.0,
            "u_0minus_formula_squared": 4.0,
        }
        with (
            patch.object(pv, "muB_from_nB_physical", return_value=1000.0),
            patch.object(
                pv,
                "_analytic_nuclear_state",
                return_value=self._nuclear_state(nB_0minus),
            ),
            patch.object(
                pv,
                "_solve_muB_inf_at_muK0_for_given_Pi",
                return_value=990.0,
            ),
            patch.object(pv, "nB_QM", return_value=3.0),
            patch.object(pv, "nK_QM", return_value=0.0),
            patch.object(
                pv,
                "_solve_analytic_isothermal_log_root",
                side_effect=pv._AnalyticIsothermalSlowFrontInvalid(limit_data),
            ),
        ):
            result = pv.analytic_velocity_isothermal(
                10.0,
                nB_0minus,
                180.0,
                0.4,
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "slow_front_approximation_invalid")
        self.assertEqual(result["phase_region"], "quark_matter_favored")
        self.assertFalse(result["front_exists"])
        self.assertTrue(np.isnan(result["u_0minus"]))
        self.assertEqual(result["u_0minus_trial_limit"], 1.0)
        self.assertEqual(result["u_0minus_formula_squared_at_limit"], 4.0)

    def test_moving_branch_solves_finite_flux_formula_with_local_interface_fraction(self):
        nB_0minus = 2.0
        a_0plus = 0.4
        D_K = 0.2
        gamma_K = 0.03
        eta = 0.5
        xi = 0.25
        lambda_n = nB_0minus / 3.0
        expected_I2 = (
            D_K
            * gamma_K
            * a_0plus**2
            * (a_0plus**2 + 2.0 * eta)
            / 4.0
        )
        expected_u_squared = (
            2.0
            * expected_I2
            / (
                lambda_n**2
                * (1.0 - a_0plus)
                * (1.0 + xi * a_0plus)
            )
        )

        def fake_nB_QM(_muB, muK, *_args, **_kwargs):
            return 4.0 if float(muK) > 0.0 else 3.0

        def fake_nK_QM(_muB, muK, *_args, **_kwargs):
            return a_0plus * 4.0 if float(muK) > 0.0 else 0.0

        def fake_Pi_QM(_muB, _muK, _bag, _temperature, jB, **_kwargs):
            u_0minus = float(jB) / nB_0minus
            return 20.0 + 100.0 * u_0minus**2

        with (
            patch.object(pv, "muB_from_nB_physical", return_value=1000.0),
            patch.object(
                pv,
                "_analytic_nuclear_state",
                return_value=self._nuclear_state(nB_0minus),
            ),
            patch.object(
                pv,
                "_solve_muB_inf_at_muK0_for_given_Pi",
                return_value=990.0,
            ),
            patch.object(
                pv,
                "_solve_muB_inf_at_muK0_for_given_Pi_ms",
                return_value=1100.0,
            ),
            patch.object(
                pv,
                "_solve_interface_0plus_from_local_a_and_Pi",
                return_value=(1200.0, 50.0),
                create=True,
            ) as interface_solve,
            patch.object(pv, "nB_QM", side_effect=fake_nB_QM),
            patch.object(pv, "nK_QM", side_effect=fake_nK_QM),
            patch.object(pv, "_Pi_QM_state", side_effect=fake_Pi_QM),
            patch.object(
                pv,
                "_microphysics_from_quark_state_isothermal_baseline",
                return_value={
                    "alpha_s": 0.3,
                    "muQ": 400.0,
                    "qD": 1.0,
                    "D": D_K,
                    "eta": eta,
                    "gamma": gamma_K,
                    "tau": 7.0,
                },
            ),
        ):
            result = pv.analytic_velocity_isothermal(
                10.0,
                nB_0minus,
                180.0,
                a_0plus,
                xi=xi,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "moving_front")
        self.assertEqual(result["phase_region"], "quark_matter_favored")
        self.assertTrue(result["front_exists"])
        self.assertAlmostEqual(result["I2"], expected_I2, places=14)
        self.assertAlmostEqual(
            result["u_0minus_squared"],
            expected_u_squared,
            places=12,
        )
        self.assertAlmostEqual(
            result["u_0minus_formula_squared"],
            expected_u_squared,
            places=12,
        )
        self.assertLess(abs(result["analytic_velocity_residual"]), 1.0e-12)
        self.assertLess(abs(result["momentum_flux_inf_residual"]), 1.0e-12)
        self.assertLess(abs(result["momentum_flux_0plus_residual"]), 1.0e-12)
        self.assertLess(abs(result["composition_residual"]), 1.0e-12)
        self.assertAlmostEqual(result["nK_0plus"] / result["nB_0plus"], a_0plus)
        self.assertAlmostEqual(result["lambda_n"], lambda_n)
        self.assertEqual(result["D_K"], D_K)
        self.assertEqual(result["gamma_K"], gamma_K)
        self.assertGreater(interface_solve.call_count, 0)
        for call in interface_solve.call_args_list:
            self.assertEqual(call.args[0], a_0plus)

    def test_live_eos_classifies_both_sides_of_isothermal_coexistence(self):
        stable = pv.analytic_velocity_isothermal(
            20.0,
            N0,
            180.0,
            0.3,
        )
        moving = pv.analytic_velocity_isothermal(
            20.0,
            3.0 * N0,
            180.0,
            0.3,
        )

        self.assertEqual(stable["status"], "stable_neutron_matter")
        self.assertGreater(stable["delta_muB"], 0.0)
        self.assertEqual(stable["u_0minus"], 0.0)
        self.assertFalse(stable["front_exists"])

        self.assertEqual(moving["status"], "moving_front")
        self.assertLess(moving["delta_muB"], 0.0)
        self.assertGreater(moving["u_0minus"], 0.0)
        self.assertTrue(moving["front_exists"])
        self.assertLess(abs(moving["analytic_velocity_residual"]), 1.0e-18)
        self.assertLess(abs(moving["composition_residual"]), 1.0e-8)

    def test_live_low_temperature_case_reports_slow_front_breakdown(self):
        result = pv.analytic_velocity_isothermal(
            1.0e-8,
            3.0 * N0,
            180.0,
            0.3,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "slow_front_approximation_invalid")
        self.assertEqual(result["phase_region"], "quark_matter_favored")
        self.assertTrue(np.isnan(result["u_0minus"]))
        self.assertEqual(result["u_0minus_trial_limit"], 1.0)
        self.assertGreater(result["u_0minus_formula_squared_at_limit"], 1.0)

    def test_input_domain_is_validated_before_eos_evaluation(self):
        invalid_calls = (
            ((-1.0, 2.0, 180.0, 0.4), {}, "T_0minus"),
            ((10.0, 0.0, 180.0, 0.4), {}, "nB_0minus"),
            ((10.0, 2.0, 0.0, 0.4), {}, "B_one_forth"),
            ((10.0, 2.0, 180.0, -0.1), {}, "a_0plus"),
            ((10.0, 2.0, 180.0, 1.0), {}, "a_0plus"),
            ((10.0, 2.0, 180.0, 0.4), {"xi": 1.0}, "xi"),
            ((10.0, 2.0, 180.0, 0.4), {"ms": 1.0}, "ms=0"),
            (
                (10.0, 2.0, 180.0, 0.4),
                {"delta_muB_tol": -1.0},
                "delta_muB_tol",
            ),
            ((10.0, 2.0, 180.0, 0.4), {"upB": np.inf}, "upB"),
            ((10.0, 2.0, 180.0, 0.4), {"upB": 1.9}, "upB"),
        )
        for args, kwargs, message in invalid_calls:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaisesRegex(RuntimeError, message):
                    pv.analytic_velocity_isothermal(*args, **kwargs)

    def test_current_analytic_model_rejects_non_pure_neutron_matter(self):
        with patch.object(
            pv,
            "muB_from_nB_physical",
            side_effect=AssertionError("EOS evaluation must not be reached"),
        ):
            with self.assertRaisesRegex(RuntimeError, "NM_type='PNM'"):
                pv.analytic_velocity_isothermal(
                    10.0,
                    2.0,
                    180.0,
                    0.4,
                    NM_type="SYM",
                )


if __name__ == "__main__":
    unittest.main()
