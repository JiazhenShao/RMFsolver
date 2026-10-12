import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("run_hidden_rho_pilot.py")
SPEC = importlib.util.spec_from_file_location("hidden_rho_pilot", MODULE_PATH)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class HiddenRhoPilotTests(unittest.TestCase):
    def test_full_step_matches_observed_growth(self):
        predicted = pilot.predict_jB(
            100.0,
            previous_rho=0.84,
            target_rho=0.90,
            exponent=1.26,
        )
        self.assertAlmostEqual(predicted / 100.0, 1.8077, places=3)

    def test_interpolation_stays_on_fixed_theta_ray(self):
        left = {
            "theta_index": 3,
            "radial_fraction_used": 0.84,
            "T_0minus_MeV": 20.0,
            "nB_0minus_over_n0": 4.0,
            "nB_0minus_MeV3": 40.0,
            "rho": 1.0,
        }
        right = {
            **left,
            "radial_fraction_used": 0.90,
            "T_0minus_MeV": 26.0,
            "nB_0minus_over_n0": 4.6,
            "nB_0minus_MeV3": 46.0,
            "rho": 1.6,
        }
        point = pilot.interpolate_hidden_point(
            left,
            right,
            target_rho=0.87,
            synthetic_point_index=123,
            synthetic_radial_index=101,
        )
        self.assertEqual(point["theta_index"], 3)
        self.assertAlmostEqual(point["T_0minus_MeV"], 23.0)
        self.assertAlmostEqual(point["nB_0minus_over_n0"], 4.3)

    def test_candidates_center_on_prediction(self):
        candidates = pilot.jB_candidates(200.0, (1.0, 0.95, 1.05))
        np.testing.assert_allclose(
            [candidate["jB_guess"] for candidate in candidates],
            [200.0, 190.0, 210.0],
        )
        self.assertTrue(all(candidate["use_profile"] for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
