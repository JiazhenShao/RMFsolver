from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "new_paper_calculations"
    / "26-07-25"
    / "run_uNmax_ray_grid.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_uNmax_ray_grid_26_07_25",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NumericalRayGridRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_defaults_preserve_ray_grid_and_use_numerical_uNmax(self):
        args = self.runner.build_parser().parse_args([])
        self.assertEqual(args.theta_count, 30)
        self.assertEqual(args.radial_count, 20)
        self.assertEqual(args.seed_velocity, 500.0)
        self.assertEqual(args.TQstar, 0.5)
        self.assertEqual(args.output, "uNmax-elliptical-ray-grid.npy")
        self.assertEqual(
            self.runner.NUMERICAL_ROOT_METHOD,
            "solve_bvp_uNmax_nK_parameter_1d",
        )

    def test_extract_seed_spine_interpolates_500_crossing_per_theta(self):
        source_theta = np.radians(np.array([10.0, 40.0]))
        source_rho = np.array(
            [
                [0.50, 0.60, 0.70, 0.80],
                [0.55, 0.65, 0.75, 0.85],
            ]
        )
        source_velocity = np.array(
            [
                [100.0, 300.0, 700.0, 900.0],
                [100.0, 300.0, 700.0, 900.0],
            ]
        )
        source_jB = np.array(
            [
                [1.0, 3.0, 7.0, 9.0],
                [2.0, 4.0, 8.0, 10.0],
            ]
        )
        payload = {
            "theta_axis_rad": source_theta,
            "rho_grid": source_rho,
            "velocity_grid_m_s": source_velocity,
            "jB_grid": source_jB,
        }
        target_theta = np.radians(np.array([10.0, 25.0, 40.0]))
        target_rho = np.array(
            [
                [0.55, 0.65, 0.75],
                [0.575, 0.675, 0.775],
                [0.60, 0.70, 0.80],
            ]
        )

        spine = self.runner.extract_seed_spine(
            payload,
            theta_axis_rad=target_theta,
            rho_grid=target_rho,
            target_velocity_m_s=500.0,
        )

        np.testing.assert_allclose(
            spine["seed_rho_by_theta"],
            np.array([0.65, 0.675, 0.70]),
        )
        np.testing.assert_array_equal(
            spine["seed_radial_index_by_theta"],
            np.array([1, 1, 1]),
        )
        np.testing.assert_allclose(
            spine["seed_analytic_jB_by_theta"],
            np.array([5.0, 5.5, 6.0]),
        )

    def test_extract_seed_spine_rejects_missing_velocity_crossing(self):
        payload = {
            "theta_axis_rad": np.array([0.5]),
            "rho_grid": np.array([[0.5, 0.6, 0.7]]),
            "velocity_grid_m_s": np.array([[10.0, 20.0, 30.0]]),
            "jB_grid": np.array([[1.0, 2.0, 3.0]]),
        }
        with self.assertRaisesRegex(ValueError, "500"):
            self.runner.extract_seed_spine(
                payload,
                theta_axis_rad=np.array([0.5]),
                rho_grid=np.array([[0.5, 0.6, 0.7]]),
                target_velocity_m_s=500.0,
            )

    def test_successor_uses_immediate_numerical_jB_then_analytic_fallback(self):
        args = self.runner.build_parser().parse_args([])
        arrays = {
            "theta_grid_rad": np.full((1, 5), 0.5),
            "rho_grid": np.array([[0.50, 0.60, 0.70, 0.80, 0.90]]),
            "radial_fraction_grid": np.array([[0.1, 0.3, 0.5, 0.7, 0.9]]),
            "nB_grid_over_n0": np.array([[2.0, 2.2, 2.4, 2.6, 2.8]]),
            "temperature_grid_MeV": np.array([[10.0, 12.0, 14.0, 16.0, 18.0]]),
            "analytic_jB_guess_grid": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        }
        seed_task = self.runner._grid_point_task(
            args=args,
            arrays=arrays,
            row_index=0,
            col_index=2,
            direction="seed",
            n0=1.0,
            B14=189.0,
            jB_guess=3.0,
            jB_guess_source="analytic_500_seed",
        )
        seed_result = {**vars(seed_task), "success": True, "jB": 8.0}

        successors = self.runner._continuation_successor_tasks(
            completed_task=vars(seed_task),
            completed_row=seed_result,
            args=args,
            arrays=arrays,
            n0=1.0,
            B14=189.0,
        )
        self.assertEqual(
            [(task.col_index, task.continuation_direction) for task in successors],
            [(3, "rho_out"), (1, "rho_in")],
        )
        self.assertTrue(all(task.jB_guess == 8.0 for task in successors))
        self.assertTrue(
            all(task.jB_guess_source == "immediate_same_ray_neighbor"
                for task in successors)
        )

        failed_inward = {
            **vars(successors[1]),
            "success": False,
            "jB": np.nan,
        }
        fallback = self.runner._continuation_successor_tasks(
            completed_task=vars(successors[1]),
            completed_row=failed_inward,
            args=args,
            arrays=arrays,
            n0=1.0,
            B14=189.0,
        )
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0].col_index, 0)
        self.assertEqual(fallback[0].jB_guess, 1.0)
        self.assertEqual(
            fallback[0].jB_guess_source,
            "analytic_fallback_after_failed_immediate_neighbor",
        )

    def test_numerical_adapter_calls_fixed_TQstar_nK_solver(self):
        task = self.runner.NumericalRayPointTask(
            row_index=2,
            col_index=3,
            theta_rad=0.5,
            rho=0.7,
            radial_fraction=0.5,
            nB_over_n0=2.5,
            T_N=10.0,
            n0=2.0,
            B14=189.0,
            ms=0.0,
            nm_type="PNM",
            upB=5000,
            TQstar=0.5,
            tail_eps=1.0e-8,
            n_mesh=40,
            tol_bvp=1.0e-3,
            max_nodes=1000,
            jB_guess=2.5,
            TQ_guess=None,
            jB_bounds=None,
            auto_jB_bounds=True,
            shooting_u_max=1.0e-4,
            shooting_jB_lower=1.0e-12,
            point_timeout=1.0,
            jB_guess_source="analytic_500_seed",
            continuation_direction="seed",
        )
        solved = {
            "success": True,
            "message": "ok",
            "u_N": 1.0e-6,
            "u_N_max_candidate": 1.0e-6,
            "jB": 1.0,
            "muB_N": 1000.0,
            "muB_Q": 1050.0,
            "T_Q": 20.0,
            "nB_Q": 6.0,
            "h_N": 5050.0,
            "h_Q": 6360.0,
            "lambda_n": 5.0 / 6.0,
            "aQstar": 0.5,
            "aQstar_derived": 0.5,
            "a_end": 0.0,
            "bvp_parameters": np.array([1.0]),
            "_root_method": "solve_bvp_uNmax_nK_parameter_1d",
            "solver_variant": "uNmax_direct_TQstar_nK",
        }
        with mock.patch.object(
            self.runner.phase_velocity,
            "solve_front_energy_conserving_uNmax",
            return_value=solved,
        ) as calculate:
            row = self.runner.calculate_numerical_point(task)

        self.assertTrue(row["success"])
        self.assertEqual(row["nB_N_MeV3"], 5.0)
        self.assertEqual(row["root_method"], "solve_bvp_uNmax_nK_parameter_1d")
        self.assertEqual(row["bvp_parameter_count"], 1)
        self.assertEqual(calculate.call_args.kwargs["TQstar"], 0.5)
        self.assertEqual(calculate.call_args.kwargs["jB_guess"], 2.5)
        self.assertEqual(
            calculate.call_args.kwargs["jB_bounds"],
            (1.0e-12, 5.0e-4),
        )

    def test_payload_records_seed_spine_and_numerical_contract(self):
        args = self.runner.build_parser().parse_args([])
        arrays = {
            "theta_axis_rad": np.array([0.5]),
            "nB_grid_over_n0": np.array([[2.0]]),
            "temperature_grid_MeV": np.array([[10.0]]),
            "seed_rho_by_theta": np.array([0.7]),
            "seed_radial_index_by_theta": np.array([0]),
            "seed_analytic_jB_by_theta": np.array([3.0]),
        }
        payload = self.runner.build_numerical_payload(
            args=args,
            transform=self.runner.RayTransform(1.0, 4.0, 100.0),
            arrays=arrays,
            records=[],
            boundary_payload={"gap_records": [], "A_records": []},
            calibration={
                "B14_calibrated_MeV": 189.0,
                "n0_MeV3": 1.0,
                "nB_crit0_MeV3": 3.0,
            },
            analytic_seed_path=Path("/tmp/analytic.npy"),
        )
        self.assertEqual(
            payload["solver_variant"],
            "uNmax_direct_TQstar_nK",
        )
        self.assertEqual(
            payload["root_method"],
            "solve_bvp_uNmax_nK_parameter_1d",
        )
        self.assertEqual(payload["seed_velocity_m_s"], 500.0)
        self.assertEqual(payload["TQstar_target_MeV"], 0.5)
        self.assertEqual(
            payload["mesh_parameterization"],
            "dimensionless_elliptical_theta_safe_radial_fraction",
        )
        self.assertIn("seed_rho_by_theta", payload)

    def test_mocked_main_scans_every_cell_and_saves_cartesian_payload(self):
        theta_axis = np.array([0.1, 0.8, 1.5])
        source_rho = np.repeat(
            np.linspace(0.51, 0.89, 5)[None, :],
            theta_axis.size,
            axis=0,
        )
        source_velocity = np.repeat(
            np.array([[100.0, 300.0, 500.0, 700.0, 900.0]]),
            theta_axis.size,
            axis=0,
        )
        source_jB = np.repeat(
            np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
            theta_axis.size,
            axis=0,
        )
        analytic_payload = {
            "theta_axis_rad": theta_axis,
            "rho_grid": source_rho,
            "velocity_grid_m_s": source_velocity,
            "jB_grid": source_jB,
            "arguments": {
                "density_min_over_n0": 1.0,
                "density_max_over_n0": 5.0,
                "t_min": 1.0e-2,
                "t_max": 100.0,
            },
            "calibration": {"B14_calibrated_MeV": 189.0},
        }
        boundary_payload = {
            "theta_axis_rad": theta_axis,
            "rho_inner": np.full(theta_axis.size, 0.50),
            "rho_outer": np.full(theta_axis.size, 0.90),
            "gap_records": [],
            "A_records": [],
            "gap_curve_records": [],
            "A_curve_records": [],
            "endpoint_records": {},
        }
        progress_instances = []

        class FakeProgress:
            def __init__(self, *, total, **_kwargs):
                self.total = int(total)
                self.completed = 0
                progress_instances.append(self)

            def update(self, count=1):
                self.completed += int(count)

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
            np.save(seed_path, analytic_payload, allow_pickle=True)
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
                        "5",
                        "--workers",
                        "1",
                        "--no-neighbor-retry",
                    ]
                )
            saved = np.load(output_path, allow_pickle=True).item()

        self.assertEqual(status, 0)
        self.assertEqual(progress_instances[-1].total, 15)
        self.assertEqual(progress_instances[-1].completed, 15)
        self.assertEqual(len(saved["records"]), 15)
        self.assertEqual(saved["velocity_grid_m_s"].shape, (3, 5))
        self.assertTrue(
            np.all(saved["status_grid"] == self.runner.STATUS_FINITE)
        )
        self.assertEqual(
            {record["continuation_direction"] for record in saved["records"]},
            {"seed", "rho_in", "rho_out"},
        )


if __name__ == "__main__":
    unittest.main()
