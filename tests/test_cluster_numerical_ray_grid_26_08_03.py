from __future__ import annotations

import importlib.util
import signal
import sys
import time
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "new_paper_calculations"
    / "26-08-03"
    / "run_uNmax_ray_grid.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_uNmax_ray_grid_26_08_03",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NumericalI2SeededUNmaxRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    @staticmethod
    def _analytic_payload():
        theta = np.array([0.1, 0.5])
        rho = np.array(
            [
                [0.40, 0.50, 0.60, 0.70],
                [0.45, 0.55, 0.65, 0.75],
            ]
        )
        theta_grid = np.repeat(theta[:, None], rho.shape[1], axis=1)
        return {
            "theta_axis_rad": theta,
            "rho_grid": rho,
            "nB_grid_over_n0": 1.0 + 9.0 * rho * np.cos(theta_grid),
            "temperature_grid_MeV": 200.0 * rho * np.sin(theta_grid),
            "velocity_grid_m_s": np.array(
                [
                    [10.0, 20.0, 500.0, 1700.0],
                    [10.0, 20.0, 500.0, 1700.0],
                ]
            ),
            "jB_grid": np.array(
                [
                    [0.1, 0.2, 5.0, 17.0],
                    [0.2, 0.3, 6.0, 18.0],
                ]
            ),
            "rho_inner_boundary": np.array([0.35, 0.40]),
            "rho_outer_boundary": np.array([0.80, 0.85]),
            "gap_boundary_records": [],
            "A_boundary_records": [],
            "gap_mesh_boundary_records": [],
            "A_mesh_boundary_records": [],
            "endpoint_records": {},
            "velocity_closure": "numerical_I2",
            "interface_fraction_mode": "extreme_endothermic",
            "transform": {
                "density_origin_over_n0": 1.0,
                "density_scale_over_n0": 9.0,
                "temperature_scale_MeV": 200.0,
            },
            "arguments": {
                "density_min_over_n0": 1.0,
                "density_max_over_n0": 10.0,
                "t_min": 1.0e-2,
                "t_max": 200.0,
                "xi": 0.0,
                "ms": 0.0,
                "nm_type": "PNM",
                "ncrit_over_n0": 3.0,
                "upB": 5000,
            },
            "calibration": {"B14_calibrated_MeV": 189.0},
        }

    def test_defaults_use_requested_window_and_20_m_s_seed(self):
        args = self.runner.build_parser().parse_args([])
        self.assertEqual(args.density_min_over_n0, 1.0)
        self.assertEqual(args.density_max_over_n0, 5.5)
        self.assertEqual(args.t_min, 1.0e-2)
        self.assertEqual(args.t_max, 120.0)
        self.assertEqual(args.theta_count, 30)
        self.assertEqual(args.radial_count, 20)
        self.assertEqual(args.seed_velocity, 20.0)
        self.assertEqual(args.safety_velocity, 1700.0)
        self.assertEqual(args.T_0plus, 0.0)
        self.assertEqual(
            args.output,
            "uNmax-T-nB-ray-grid-numerical-I2-seeded.npy",
        )

    def test_source_geometry_reuses_numerical_I2_transform_and_boundaries(self):
        payload = self._analytic_payload()
        geometry = self.runner.source_geometry_from_payload(
            payload,
            theta_count=3,
        )
        transform = geometry["transform"]
        self.assertEqual(transform.density_origin_over_n0, 1.0)
        self.assertEqual(transform.density_scale_over_n0, 9.0)
        self.assertEqual(transform.temperature_scale_MeV, 200.0)
        np.testing.assert_allclose(
            geometry["theta_axis_rad"],
            np.array([0.1, 0.3, 0.5]),
        )
        np.testing.assert_allclose(
            geometry["rho_inner"],
            np.array([0.35, 0.375, 0.40]),
        )
        np.testing.assert_allclose(
            geometry["rho_A_boundary"],
            np.array([0.80, 0.825, 0.85]),
        )

    def test_validation_requires_numerical_I2_and_covering_source_window(self):
        args = self.runner.build_parser().parse_args([])
        calibration = {"B14_calibrated_MeV": 189.0}
        payload = self._analytic_payload()
        self.runner.validate_analytic_seed_payload(
            payload,
            args=args,
            calibration=calibration,
        )

        payload["velocity_closure"] = "closed_form"
        with self.assertRaisesRegex(ValueError, "numerical_I2"):
            self.runner.validate_analytic_seed_payload(
                payload,
                args=args,
                calibration=calibration,
            )

    def test_successor_uses_analytic_primary_then_propagated_fallback(self):
        args = self.runner.build_parser().parse_args([])
        arrays = {
            "theta_grid_rad": np.full((1, 5), 0.5),
            "rho_grid": np.array([[0.50, 0.60, 0.70, 0.80, 0.90]]),
            "radial_fraction_grid": np.array(
                [[0.1, 0.3, 0.5, 0.7, 0.9]]
            ),
            "nB_grid_over_n0": np.array(
                [[2.0, 2.2, 2.4, 2.6, 2.8]]
            ),
            "temperature_grid_MeV": np.array(
                [[10.0, 12.0, 14.0, 16.0, 18.0]]
            ),
            "analytic_jB_guess_grid": np.array(
                [[1.0, 2.0, 3.0, 4.0, 5.0]]
            ),
        }
        seed = self.runner._grid_point_task(
            args=args,
            arrays=arrays,
            row_index=0,
            col_index=2,
            direction="seed",
            n0=1.0,
            B14=189.0,
            jB_guess=3.0,
            jB_guess_source="analytic_numerical_I2_seed_20m_s",
        )
        solved_seed = {**vars(seed), "success": True, "jB": 8.0}
        successors = self.runner._continuation_successor_tasks(
            completed_task=seed,
            completed_row=solved_seed,
            args=args,
            arrays=arrays,
            n0=1.0,
            B14=189.0,
        )

        self.assertEqual(
            [(task.col_index, task.continuation_direction) for task in successors],
            [(3, "rho_out"), (1, "rho_in")],
        )
        self.assertEqual([task.jB_guess for task in successors], [4.0, 2.0])
        self.assertTrue(
            all(
                task.jB_guess_source == "analytic_2d_numerical_I2_primary"
                for task in successors
            )
        )
        self.assertTrue(
            all(task.propagated_jB_guess == 8.0 for task in successors)
        )

        failed_primary = {**vars(successors[0]), "success": False}
        fallback = self.runner._propagated_fallback_task(
            successors[0],
            failed_primary,
        )
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback.jB_guess, 8.0)
        self.assertEqual(
            fallback.jB_guess_source,
            "immediate_same_ray_propagated_fallback",
        )
        self.assertEqual(fallback.retry_attempt, 1)

    def test_two_dimensional_interpolation_evaluates_actual_target_points(self):
        payload = {
            "nB_grid_over_n0": np.array([[1.0, 2.0], [1.0, 2.0]]),
            "temperature_grid_MeV": np.array([[0.0, 0.0], [1.0, 1.0]]),
            "theta_axis_rad": np.array([0.0, 1.0]),
            "rho_grid": np.array([[0.1, 0.2], [0.1, 0.2]]),
            "jB_grid": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "velocity_grid_m_s": np.array(
                [[10.0, 20.0], [13.0, 23.0]]
            ),
        }
        interpolated = self.runner.interpolate_analytic_fields_2d(
            payload,
            nB_grid_over_n0=np.array([[1.5]]),
            temperature_grid_MeV=np.array([[0.25]]),
            theta_grid_rad=np.array([[0.25]]),
            rho_grid=np.array([[0.15]]),
        )
        np.testing.assert_allclose(interpolated["jB_grid"], [[2.0]])
        np.testing.assert_allclose(
            interpolated["velocity_grid_m_s"],
            [[15.75]],
        )
        self.assertFalse(interpolated["fallback_mask"][0, 0])

    def test_seed_selection_uses_jB_at_selected_numerical_cell(self):
        selected = self.runner.select_seed_cells_from_interpolated_grid(
            rho_grid=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            velocity_grid_m_s=np.array(
                [[5.0, 18.0, 35.0], [8.0, 21.0, 40.0]]
            ),
            jB_grid=np.array(
                [[0.05, 0.18, 0.35], [0.08, 0.21, 0.40]]
            ),
            target_velocity_m_s=20.0,
        )
        np.testing.assert_array_equal(
            selected["seed_radial_index_by_theta"],
            np.array([1, 1]),
        )
        np.testing.assert_allclose(
            selected["seed_analytic_jB_by_theta"],
            np.array([0.18, 0.21]),
        )
        np.testing.assert_allclose(
            selected["seed_actual_velocity_m_s_by_theta"],
            np.array([18.0, 21.0]),
        )

    def test_successful_primary_does_not_create_propagated_fallback(self):
        args = self.runner.build_parser().parse_args([])
        arrays = {
            "theta_grid_rad": np.array([[0.5]]),
            "rho_grid": np.array([[0.7]]),
            "radial_fraction_grid": np.array([[0.5]]),
            "nB_grid_over_n0": np.array([[2.5]]),
            "temperature_grid_MeV": np.array([[10.0]]),
        }
        task = self.runner._grid_point_task(
            args=args,
            arrays=arrays,
            row_index=0,
            col_index=0,
            direction="rho_out",
            n0=1.0,
            B14=189.0,
            jB_guess=0.2,
            jB_guess_source="analytic_numerical_I2_primary",
            propagated_jB_guess=0.3,
        )
        self.assertIsNone(
            self.runner._propagated_fallback_task(
                task,
                {**vars(task), "success": True, "jB": 0.25},
            )
        )

    @unittest.skipUnless(
        hasattr(signal, "setitimer"),
        "requires POSIX interval timers",
    )
    def test_point_timeout_is_not_swallowed_by_solver_exception_handlers(self):
        with self.assertRaises(self.runner.PointTimeoutError):
            with self.runner.point_timeout(0.01):
                deadline = time.monotonic() + 0.1
                while time.monotonic() < deadline:
                    try:
                        time.sleep(0.02)
                    except Exception:
                        pass


if __name__ == "__main__":
    unittest.main()
