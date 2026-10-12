import importlib.util
import sys
import unittest
from pathlib import Path


RUNNER_DIR = (
    Path(__file__).resolve().parents[1]
    / "new_paper_calculations"
    / "26-08-18"
)


def _load_radial_module():
    sys.path.insert(0, str(RUNNER_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "t0plus0p1_radial_shell_continuation",
            RUNNER_DIR / "_radial_shell_continuation.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class T0plus0p1RadialSeedingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.radial = _load_radial_module()

    def test_first_shell_starts_inside_observed_numerical_branch(self):
        candidates = self.radial.first_shell_candidates()

        self.assertEqual(candidates[:3], (0.4, 0.3, 0.5))
        self.assertLess(max(candidates[:3]), 0.6)

    def test_first_shell_keeps_larger_guesses_as_late_fallbacks(self):
        candidates = self.radial.first_shell_candidates()

        self.assertEqual(candidates[3:], (0.6, 0.8))

    def test_matched_one_mev_seed_precedes_fixed_fallbacks(self):
        self.assertTrue(hasattr(self.radial, "matched_reference_candidates"))

        candidates = self.radial.matched_reference_candidates(
            0.25,
            scale=1.18,
        )

        self.assertAlmostEqual(candidates[0]["jB_guess"], 0.295)
        self.assertEqual(
            candidates[0]["source"],
            "matched_T0plus1_numerical_jB_scaled",
        )
        self.assertAlmostEqual(candidates[1]["jB_guess"], 0.25)
        self.assertEqual(
            candidates[1]["source"],
            "matched_T0plus1_numerical_jB",
        )
        self.assertEqual(
            tuple(candidate["jB_guess"] for candidate in candidates[2:]),
            self.radial.first_shell_candidates(),
        )

    def test_missing_reference_uses_fixed_first_shell_ladder(self):
        self.assertTrue(hasattr(self.radial, "matched_reference_candidates"))

        candidates = self.radial.matched_reference_candidates(
            float("nan"),
            scale=1.18,
        )

        self.assertEqual(
            tuple(candidate["jB_guess"] for candidate in candidates),
            self.radial.first_shell_candidates(),
        )


if __name__ == "__main__":
    unittest.main()
