from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "new_paper_calculations"
    / "26-07-29"
    / "run_uNmax_ray_grid.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_uNmax_ray_grid_26_07_29",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NumericalSafetyRayGridRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    @staticmethod
    def _analytic_payload():
        return {
            "theta_axis_rad": np.array([0.1, 0.5]),
            "rho_grid": np.array(
                [
                    [0.50, 0.60, 0.70, 0.80],
                    [0.55, 0.65, 0.75, 0.85],
                ]
            ),
            "velocity_grid_m_s": np.array(
                [
                    [100.0, 500.0, 1700.0, 3000.0],
                    [100.0, 500.0, 1700.0, 3000.0],
                ]
            ),
            "jB_grid": np.array(
                [
                    [1.0, 5.0, 17.0, 30.0],
                    [2.0, 6.0, 18.0, 31.0],
                ]
            ),
        }

    def test_defaults_use_1700_safety_cap_and_zero_TQstar(self):
        args = self.runner.build_parser().parse_args([])
        self.assertEqual(args.seed_velocity, 500.0)
        self.assertEqual(args.safety_velocity, 1700.0)
        self.assertEqual(args.TQstar, 0.0)
        self.assertEqual(args.output, "uNmax-elliptical-ray-grid.npy")

    def test_velocity_contour_interpolates_crossing_across_theta(self):
        contour = self.runner.extract_analytic_velocity_contour(
            self._analytic_payload(),
            theta_axis_rad=np.array([0.1, 0.3, 0.5]),
            target_velocity_m_s=1700.0,
        )
        np.testing.assert_allclose(
            contour["rho_by_theta"],
            np.array([0.70, 0.725, 0.75]),
        )
        np.testing.assert_allclose(
            contour["jB_by_theta"],
            np.array([17.0, 17.5, 18.0]),
        )

    def test_safety_boundary_stays_between_gap_and_A_boundaries(self):
        theta = np.array([0.1, 0.3, 0.5])
        transform = self.runner.RayTransform(1.0, 4.0, 100.0)
        safety = self.runner.build_analytic_safety_boundary(
            self._analytic_payload(),
            transform=transform,
            theta_axis_rad=theta,
            rho_inner=np.array([0.45, 0.475, 0.50]),
            rho_A_boundary=np.array([0.85, 0.875, 0.90]),
            target_velocity_m_s=1700.0,
        )
        np.testing.assert_allclose(
            safety["rho_by_theta"],
            np.array([0.70, 0.725, 0.75]),
        )
        self.assertTrue(
            np.all(safety["clearance_to_A_rho_by_theta"] > 0.0)
        )
        mesh = self.runner.build_safe_ray_mesh(
            transform=transform,
            theta_axis_rad=theta,
            radial_fraction_axis=self.runner.chebyshev_open_axis(
                5,
                1.0e-3,
            ),
            rho_inner=np.array([0.45, 0.475, 0.50]),
            rho_outer=safety["rho_by_theta"],
        )
        self.assertTrue(
            np.all(
                mesh["rho_grid"]
                < np.asarray(safety["rho_by_theta"])[:, None]
            )
        )

    def test_safety_boundary_rejects_contour_beyond_A_boundary(self):
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            self.runner.build_analytic_safety_boundary(
                self._analytic_payload(),
                transform=self.runner.RayTransform(1.0, 4.0, 100.0),
                theta_axis_rad=np.array([0.1, 0.5]),
                rho_inner=np.array([0.45, 0.50]),
                rho_A_boundary=np.array([0.65, 0.90]),
                target_velocity_m_s=1700.0,
            )

    def test_safety_velocity_must_exceed_seed_velocity(self):
        args = self.runner.build_parser().parse_args(
            ["--safety-velocity", "400"]
        )
        with self.assertRaisesRegex(ValueError, "greater than seed"):
            self.runner._validate_args(args)

    def test_main_uses_safety_contour_but_preserves_A_boundary(self):
        theta = np.array([0.1, 0.3, 0.5])
        payload = self._analytic_payload()
        payload.update(
            {
                "arguments": {
                    "density_min_over_n0": 1.0,
                    "density_max_over_n0": 5.0,
                    "t_min": 1.0e-2,
                    "t_max": 100.0,
                },
                "calibration": {"B14_calibrated_MeV": 189.0},
                "interface_fraction_mode": "extreme_endothermic",
            }
        )
        boundary_payload = {
            "theta_axis_rad": theta,
            "rho_inner": np.array([0.45, 0.475, 0.50]),
            "rho_outer": np.array([0.85, 0.875, 0.90]),
            "gap_records": [],
            "A_records": [],
            "gap_curve_records": [],
            "A_curve_records": [],
            "endpoint_records": {},
        }

        class FakeProgress:
            def __init__(self, *, total, **_kwargs):
                self.total = int(total)

            def update(self, _count=1):
                return None

            def set_postfix_str(self, _text):
                return None

            def refresh(self):
                return None

            def close(self):
                return None

        def fake_calculate(task):
            return {
                **vars(task),
                "success": True,
                "message": "ok",
                "u_N": 1.0e-6,
                "u_N_m_s": 300.0,
                "jB": float(task.jB_guess),
                "lambda_n": 0.8,
                "A_boundary": 0.5,
                "a_local_Qstar": 0.3,
                "entropy_jump_per_baryon": 0.2,
                "point_elapsed_s": 0.01,
                "point_timed_out": False,
            }

        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            seed_path = temporary_path / "analytic.npy"
            output_path = temporary_path / "numerical.npy"
            np.save(seed_path, payload, allow_pickle=True)
            with (
                mock.patch.object(
                    self.runner,
                    "resolve_calibration",
                    return_value={
                        "B14_calibrated_MeV": 189.0,
                        "muB_crit0_MeV": 1100.0,
                        "P_crit0_MeV4": 1.0,
                        "n0_MeV3": 1.0,
                        "nB_crit0_MeV3": 3.0,
                        "nB_crit0_over_n0": 3.0,
                    },
                ),
                mock.patch.object(
                    self.runner,
                    "solve_boundary_curves",
                    return_value=boundary_payload,
                ),
                mock.patch.object(
                    self.runner,
                    "calculate_numerical_point",
                    side_effect=fake_calculate,
                ),
                mock.patch.object(self.runner, "tqdm", FakeProgress),
            ):
                status = self.runner.main(
                    [
                        "--output",
                        str(output_path),
                        "--analytic-seed-payload",
                        str(seed_path),
                        "--theta-count",
                        "3",
                        "--radial-count",
                        "2",
                        "--workers",
                        "1",
                        "--no-neighbor-retry",
                    ]
                )
            saved = np.load(output_path, allow_pickle=True).item()

        self.assertEqual(status, 0)
        np.testing.assert_allclose(
            saved["rho_outer_boundary"],
            np.array([0.70, 0.725, 0.75]),
        )
        np.testing.assert_allclose(
            saved["A_mesh_boundary_rho_by_theta"],
            np.array([0.85, 0.875, 0.90]),
        )
        self.assertTrue(
            np.all(
                saved["rho_grid"]
                < saved["rho_outer_boundary"][:, None]
            )
        )
        self.assertEqual(saved["safety_velocity_m_s"], 1700.0)


if __name__ == "__main__":
    unittest.main()
