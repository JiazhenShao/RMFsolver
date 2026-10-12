import inspect
import unittest
from pathlib import Path

from RMFsolver import phase_velocity as pv


class EndpointNotationContractTests(unittest.TestCase):
    def assert_parameter_names(self, function, expected_prefix, forbidden):
        names = tuple(inspect.signature(function).parameters)
        self.assertEqual(names[: len(expected_prefix)], expected_prefix)
        for old_name in forbidden:
            self.assertNotIn(old_name, names)

    def test_public_solver_parameters_use_explicit_endpoint_labels(self):
        self.assert_parameter_names(
            pv.analytic_velocity_bound,
            ("muB_0minus", "T_0minus", "B_one_forth"),
            ("muB_N", "T_N"),
        )
        self.assert_parameter_names(
            pv.solve_front_energy_conserving_nK,
            ("T_0minus", "nB_0minus", "B_one_forth", "a_0plus"),
            ("T", "nB_N", "aQstar"),
        )
        self.assert_parameter_names(
            pv.solve_front_energy_conserving_uNmax,
            ("T_0minus", "nB_0minus", "B_one_forth", "T_0plus"),
            ("T", "nB_N", "TQstar", "TQ_guess"),
        )
        self.assertIn(
            "T_inf_guess",
            inspect.signature(pv.solve_front_energy_conserving_uNmax).parameters,
        )
        self.assert_parameter_names(
            pv.solve_front_thermal_conducting,
            ("T_0minus", "nB_0minus", "B_one_forth", "T_0plus"),
            ("T", "nB_N", "T_interface"),
        )

    def test_interface_helpers_use_0plus_not_qstar(self):
        self.assertTrue(hasattr(pv, "_solve_interface_0plus_from_a_0plus_and_Pi"))
        self.assertTrue(hasattr(pv, "_solve_interface_0plus_from_T_0plus_E_and_Pi"))
        self.assertTrue(hasattr(pv, "_fixed_T_0plus_E_residual"))
        self.assertFalse(hasattr(pv, "_solve_interface_Qstar_from_aQstar_and_Pi"))
        self.assertFalse(hasattr(pv, "_solve_interface_Qstar_from_TQstar_E_and_Pi"))
        self.assertFalse(hasattr(pv, "_fixed_TQstar_E_residual"))

    def test_phase_velocity_does_not_emit_legacy_endpoint_keys(self):
        source = Path(pv.__file__).read_text()
        for legacy_token in (
            "Qstar",
            "TQstar",
            "aQstar",
            '"a_N"',
            '"muB_N"',
            '"nB_N"',
            '"T_N"',
            '"muB_Q"',
            '"nB_Q"',
            '"T_Q"',
        ):
            self.assertNotIn(legacy_token, source)


if __name__ == "__main__":
    unittest.main()
