from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "run_rho5k_adaptive_bracket.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rho5k_adaptive_bracket", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AdaptiveBracketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_module()

    def test_target_rho_matches_growth_model_and_cap(self):
        target = self.runner.predict_target_rho(
            4075.87479167,
            base_rho=0.86,
            target_velocity_m_s=5200.0,
            exponent=1.26,
            minimum_increment=0.0025,
            maximum_rho=0.885,
        )
        self.assertAlmostEqual(target, 0.884608, places=5)
        capped = self.runner.predict_target_rho(
            100.0,
            base_rho=0.86,
            target_velocity_m_s=5200.0,
            exponent=1.26,
            minimum_increment=0.0025,
            maximum_rho=0.885,
        )
        self.assertEqual(capped, 0.885)

    def test_hidden_ladder_uses_small_steps_and_exact_target(self):
        ladder = self.runner.build_hidden_ladder(
            base_rho=0.86,
            target_rho=0.8846,
            maximum_step=0.005,
        )
        self.assertAlmostEqual(ladder[-1], 0.8846)
        previous = 0.86
        for rho in ladder:
            self.assertLessEqual(rho - previous, 0.005 + 1.0e-12)
            previous = rho

    def test_adjacent_angular_candidates_follow_radial_candidates(self):
        candidates = self.runner.build_jB_candidates(
            predicted_radial_jB=100.0,
            perturbations=(1.0, 0.95, 1.05),
            theta_index=8,
            target_rho=0.875,
            base_rho=0.86,
            exponent=1.26,
            base_jB_by_theta={7: 90.0, 8: 92.0, 9: 96.0},
        )
        sources = [candidate["source"] for candidate in candidates]
        self.assertEqual(sources[:3], [
            "adaptive_radial_prediction",
            "adaptive_radial_prediction",
            "adaptive_radial_prediction",
        ])
        self.assertIn("adjacent_theta_two_sided", sources)
        angular = next(
            candidate for candidate in candidates
            if candidate["source"] == "adjacent_theta_two_sided"
        )
        self.assertEqual(angular["angular_seed_theta_indices"], (7, 9))

    def test_angular_seed_never_jumps_over_missing_neighbor(self):
        candidates = self.runner.build_jB_candidates(
            predicted_radial_jB=100.0,
            perturbations=(1.0,),
            theta_index=8,
            target_rho=0.875,
            base_rho=0.86,
            exponent=1.26,
            base_jB_by_theta={6: 88.0, 8: 92.0, 9: 96.0},
        )
        angular = [
            candidate for candidate in candidates
            if candidate["source"].startswith("adjacent_theta")
        ]
        self.assertEqual(len(angular), 1)
        self.assertEqual(angular[0]["angular_seed_theta_indices"], (9,))
        self.assertEqual(angular[0]["source"], "adjacent_theta_one_sided")

    def test_extension_is_small_and_capped(self):
        self.assertEqual(
            self.runner.next_extension_rho(0.881, increment=0.0025, maximum_rho=0.885),
            0.8835,
        )
        self.assertEqual(
            self.runner.next_extension_rho(0.884, increment=0.0025, maximum_rho=0.885),
            0.885,
        )
        self.assertIsNone(
            self.runner.next_extension_rho(0.885, increment=0.0025, maximum_rho=0.885)
        )


if __name__ == "__main__":
    unittest.main()
