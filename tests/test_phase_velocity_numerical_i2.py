import unittest

from RMFsolver import phase_velocity as pv


class NumericalI2ClosureTests(unittest.TestCase):
    def test_dense_state_keeps_positive_muK_continuation_branch(self):
        result = pv.semi_analytic_velocity_bound(
            1372.5773250328914,
            20.0,
            189.156595729,
            T_0plus=0.5,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["velocity_closure"], "numerical_I2")
        self.assertEqual(result["velocity_method"], "semi_analytic_numerical_I2")
        self.assertAlmostEqual(result["T_0plus"], 0.5)
        self.assertAlmostEqual(
            result["a_0plus"],
            result["nK_0plus"] / result["nB_0plus"],
        )
        self.assertGreater(result["I2_numerical"], 0.0)
        self.assertGreaterEqual(result["layer_trajectory_points"], 2)


if __name__ == "__main__":
    unittest.main()
