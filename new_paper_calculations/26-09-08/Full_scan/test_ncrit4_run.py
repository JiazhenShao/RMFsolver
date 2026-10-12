"""Fast checks using saved roots, without launching BVP workers."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import _T0plus0p1_runner as runner
import prepare_T0plus0p1_domain as prepare
import run_T0plus0p1_numerical as numerical


class FourN0RunTests(unittest.TestCase):
    def setUp(self):
        self.args = runner._prepare_parser().parse_args(prepare.default_arguments())
        self.transform = runner.benchmark.ray_grid.RayTransform(1.0, 6.0, 170.0)
        self.saved = np.load(runner._input_path(self.args.boundary_file), allow_pickle=True).item()
        self.calibration = dict(ncrit_over_n0=4.0, B14_MeV=self.saved["B14_MeV"],
                                n0_MeV3=0.16 * runner.const.MeV_fm**3)

    def test_four_n0_defaults_and_unchanged_numerical_caps(self):
        self.assertEqual(self.args.ncrit_over_n0, 4.0)
        self.assertEqual(runner.benchmark.DEFAULT_NCRIT_OVER_N0, 4.0)
        self.assertEqual(prepare.DEFAULT_OUTPUT, numerical.DEFAULT_DOMAIN_FILE)
        self.assertIn("ncrit4", numerical.DEFAULT_OUTPUT)
        self.assertEqual(numerical.DEFAULT_TRIAL_TIMEOUT_S, 300)
        self.assertEqual(numerical.DEFAULT_CELL_TIMEOUT_S, 1200)
        self.assertEqual(numerical.DEFAULT_SAVE_EVERY, 1)

    def test_cached_prepare_never_launches_boundary_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "domain.npy"
            with mock.patch.object(runner, "_calibration", return_value=self.calibration), \
                 mock.patch.object(runner, "ProcessPoolExecutor", side_effect=AssertionError("no boundary pool")):
                runner.prepare_domain_main([*prepare.default_arguments(), "--output", str(output)])
            payload = np.load(output, allow_pickle=True).item()
        boundary = payload["boundary_diagnostics"]["semi_analytic"]
        self.assertTrue(boundary["reused_saved_roots"])
        self.assertEqual(boundary["saved_boundary_theta_count"], 18)
        self.assertEqual(len(payload["points"]), 600)
        theta = payload["boundary_diagnostics"]["selected_geometry"]["theta_axis_rad"]
        self.assertEqual(len(theta), 30)
        np.testing.assert_allclose(theta[[0, -1]], [0.0, np.pi / 2])
        np.testing.assert_array_equal(boundary["rho_outer"], [r["rho"] for r in self.saved["A_records"]])
        self.assertEqual(payload["calibration"]["ncrit_over_n0"], 4.0)

    def test_wrong_calibration_or_window_is_rejected(self):
        for key, value in (("ncrit_over_n0", 2.0), ("B14_MeV", 176.0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.load_saved_boundaries(self.args, {**self.calibration, key: value}, self.transform)
        with self.assertRaisesRegex(ValueError, "geometry"):
            runner.load_saved_boundaries(self.args, self.calibration,
                                        runner.benchmark.ray_grid.RayTransform(1.0, 4.5, 120.0))

    def test_wrong_temperature_or_bad_residual_is_rejected(self):
        self.args.T0plus_values = (1.0,)
        with self.assertRaises(ValueError):
            runner.load_saved_boundaries(self.args, self.calibration, self.transform)
        self.args.T0plus_values = (0.1,)
        self.saved["A_records"][0]["residual_norm"] = float("nan")
        with mock.patch.object(runner.np, "load") as load:
            load.return_value.item.return_value = self.saved
            with self.assertRaisesRegex(ValueError, "residual"):
                runner.load_saved_boundaries(self.args, self.calibration, self.transform)

    def test_cold_solver_uses_density_constraint_and_not_fixed_density_inversion(self):
        task = dict(condition="ns", theta_rad=0.0, transform=self.transform,
                    density_max_over_n0=7.0, temperature_max_MeV=170.0,
                    n0=self.calibration["n0_MeV3"], B14=self.calibration["B14_MeV"],
                    ms=0.0, upB=5000, nm_type="PNM", T_0plus_MeV=0.001)
        saved = self.saved["A_records"][0]

        def accept_saved_root(residual, seed, **kwargs):
            self.assertEqual(len(seed), 4)
            vector = np.asarray(saved["solver_vector"])
            self.assertLess(np.max(np.abs(residual(vector))), runner.DIRECT_BOUNDARY_ROOT_TOLERANCE)
            return type("Solution", (), dict(x=vector, nfev=1))()

        with mock.patch.object(runner, "least_squares", side_effect=accept_saved_root), \
             mock.patch.object(runner, "_direct_upstream_state", side_effect=AssertionError("wrong RMF branch")):
            result = runner._solve_direct_interface_root(task)
        self.assertAlmostEqual(result["A_boundary"], 1.0, places=8)
        self.assertAlmostEqual(result["n_s_over_nB"], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
