import inspect
import unittest

import numpy as np

from RMFsolver import phase_velocity as pv


class FixedTemperatureAnalyticApiTests(unittest.TestCase):
    def test_public_analytic_methods_have_distinct_fixed_temperature_apis(self):
        full_signature = inspect.signature(pv.analytic_velocity_bound)
        semi_function = getattr(pv, "semi_analytic_velocity_bound", None)
        lte_function = getattr(pv, "analytic_velocity_bound_lte", None)

        self.assertIsNotNone(semi_function)
        self.assertIsNotNone(lte_function)
        semi_signature = inspect.signature(semi_function)
        lte_signature = inspect.signature(lte_function)

        for signature in (full_signature, semi_signature):
            self.assertIn("T_0plus", signature.parameters)
            self.assertEqual(
                signature.parameters["T_0plus"].kind,
                inspect.Parameter.KEYWORD_ONLY,
            )
            self.assertEqual(signature.parameters["xi"].default, 0.0)
            self.assertNotIn("velocity_closure", signature.parameters)
            self.assertNotIn("interface_fraction_mode", signature.parameters)
            self.assertNotIn("a_0plus_max", signature.parameters)

        self.assertEqual(lte_signature.parameters["xi"].default, 0.0)
        self.assertNotIn("T_0plus", lte_signature.parameters)

    def test_fixed_muB_interface_state_uses_exact_eos_at_requested_temperature(self):
        muB_inf = 1100.0
        T_inf = 20.0
        T_0plus = 5.0
        bag = 180.0
        P_inf = float(pv.PQM(muB_inf, 0.0, bag, T_inf))

        state = pv._analytic_fixed_muB_interface_state(
            muB_inf,
            P_inf,
            T_0plus,
            bag,
        )

        self.assertAlmostEqual(state["T_0plus"], T_0plus)
        self.assertAlmostEqual(state["muB_0plus"], muB_inf)
        pressure = float(
            pv.PQM(
                state["muB_0plus"],
                state["muK_0plus"],
                bag,
                state["T_0plus"],
            )
        )
        self.assertLess(abs(pressure - P_inf), 1.0e-8 * max(abs(P_inf), 1.0))
        expected_a = float(
            pv.nK_QM(
                state["muB_0plus"],
                state["muK_0plus"],
                bag,
                state["T_0plus"],
            )
            / pv.nB_QM(
                state["muB_0plus"],
                state["muK_0plus"],
                bag,
                state["T_0plus"],
            )
        )
        self.assertTrue(np.isfinite(state["a_0plus"]))
        self.assertAlmostEqual(state["a_0plus"], expected_a)
        self.assertGreater(state["a_0plus"], 0.0)
        self.assertLess(state["a_0plus"], state["A_boundary"])


if __name__ == "__main__":
    unittest.main()
