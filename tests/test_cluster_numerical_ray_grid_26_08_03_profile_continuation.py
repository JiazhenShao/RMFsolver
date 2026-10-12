from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "new_paper_calculations"
    / "26-08-03"
    / "run_uNmax_profile_continuation_ray_grid.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_uNmax_profile_continuation_ray_grid_26_08_03",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ProfileContinuationUNmaxRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_new_runner_has_distinct_output_and_profile_first_metadata(self):
        args = self.runner.build_parser().parse_args([])
        self.assertEqual(
            args.output,
            "uNmax-T-nB-ray-grid-full-profile-continuation.npy",
        )
        self.assertEqual(
            self.runner.DEFAULT_RUN_TAG,
            "numerical_uNmax_T_nB_full_profile_continuation_20260803_v2",
        )
        self.assertEqual(args.max_nodes, 10000)
        self.assertEqual(args.point_timeout, 1800.0)

    def test_effective_xi_recovers_linear_flux_profile(self):
        result = {
            "success": True,
            "a": np.array([0.5, 0.25, 0.0]),
            "jK": np.array([0.2, 0.1, 0.0]),
            "nB": np.ones(3),
            "u": np.zeros(3),
            "lambda_n": 1.0,
            "u_0minus": 0.4,
        }
        diagnostics = self.runner.effective_xi_from_profile(result)
        self.assertAlmostEqual(diagnostics["xi_effective"], 0.0)
        self.assertAlmostEqual(diagnostics["I1_profile"], 0.05)
        self.assertAlmostEqual(diagnostics["interface_diffusive_flux"], 0.2)

    def test_continuation_bundle_is_compact_and_contains_solver_profile(self):
        result = {
            "success": True,
            "jB": 3.0,
            "T_inf": 10.0,
            "muB_inf": 1000.0,
            "muB_0plus": 1020.0,
            "muK_0plus": 30.0,
            "T_0plus": 0.0,
            "nK_inf": 0.0,
            "s_coord": np.linspace(0.0, 0.99, 11),
            "nK": np.linspace(5.0, 0.0, 11),
            "jK": np.linspace(3.0, 0.0, 11),
        }
        bundle = self.runner.extract_continuation_bundle(result, max_points=5)
        self.assertEqual(bundle["jB"], 3.0)
        self.assertEqual(bundle["s_coord"].size, 5)
        np.testing.assert_allclose(bundle["s_coord"][[0, -1]], [0.0, 0.99])
        np.testing.assert_allclose(bundle["nK"][[0, -1]], [5.0, 0.0])
        np.testing.assert_allclose(bundle["jK"][[0, -1]], [3.0, 0.0])

    def test_point_solver_requests_profile_and_passes_continuation(self):
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
            jB_guess_source="test",
            analytic_jB_guess=0.2,
        )
        continuation = {
            "jB": 0.3,
            "s_coord": np.array([0.0, 0.99]),
            "nK": np.array([1.0, 0.0]),
            "jK": np.array([0.3, 0.0]),
            "nK_inf": 0.0,
        }
        with patch.object(
            self.runner.phase_velocity,
            "solve_front_energy_conserving_uNmax",
            return_value={"success": False, "message": "mock failure"},
        ) as solve:
            row, bundle = self.runner.calculate_numerical_point(
                task,
                continuation_guess=continuation,
                return_continuation=True,
            )
        self.assertFalse(row["success"])
        self.assertIsNone(bundle)
        self.assertTrue(solve.call_args.kwargs["return_profile"])
        self.assertIs(solve.call_args.kwargs["continuation_guess"], continuation)

    def test_direction_tries_profile_first_then_analytic_fallback(self):
        job = self.runner.RayDirectionJob(
            row_index=0,
            direction="rho_out",
            tasks=(
                SimpleNamespace(col_index=1, analytic_jB_guess=11.0),
                SimpleNamespace(col_index=2, analytic_jB_guess=12.0),
            ),
            initial_continuation={"jB": 9.0},
        )
        calls = []

        def fake_solve(task, *, continuation_guess, attempt_kind, guess_source):
            calls.append(
                (
                    task.col_index,
                    None if continuation_guess is None else continuation_guess["jB"],
                    attempt_kind,
                    guess_source,
                )
            )
            if task.col_index == 1 and continuation_guess is not None:
                return {"success": False, "col_index": 1}, None
            result = {"success": True, "col_index": task.col_index, "jB": task.analytic_jB_guess}
            return result, {"jB": task.analytic_jB_guess}

        outcomes = self.runner.solve_ray_direction(
            job,
            solve_point=fake_solve,
        )
        self.assertEqual(
            calls,
            [
                (1, 9.0, "profile_continuation_primary", "immediate_same_ray_full_profile"),
                (1, None, "analytic_fallback", "analytic_2d_numerical_I2_fallback"),
                (2, 11.0, "profile_continuation_primary", "immediate_same_ray_full_profile"),
            ],
        )
        self.assertEqual(len(outcomes), 3)

    def test_failed_immediate_neighbor_prevents_distant_profile_reuse(self):
        job = self.runner.RayDirectionJob(
            row_index=0,
            direction="rho_out",
            tasks=(
                SimpleNamespace(col_index=1, analytic_jB_guess=11.0),
                SimpleNamespace(col_index=2, analytic_jB_guess=12.0),
            ),
            initial_continuation={"jB": 9.0},
        )
        calls = []

        def fake_solve(task, *, continuation_guess, attempt_kind, guess_source):
            calls.append((task.col_index, continuation_guess, attempt_kind))
            return {"success": False, "col_index": task.col_index}, None

        self.runner.solve_ray_direction(job, solve_point=fake_solve)
        self.assertEqual(calls[0][1], {"jB": 9.0})
        self.assertIsNone(calls[1][1])
        self.assertIsNone(calls[2][1])
        self.assertEqual(calls[2][2], "analytic_primary_no_neighbor")

    def test_direction_jobs_walk_adjacent_cells_away_from_seed(self):
        args = self.runner.build_parser().parse_args([])
        shape = (1, 5)
        arrays = {
            "theta_grid_rad": np.full(shape, 0.5),
            "rho_grid": np.array([[0.5, 0.6, 0.7, 0.8, 0.9]]),
            "radial_fraction_grid": np.array([[0.1, 0.3, 0.5, 0.7, 0.9]]),
            "nB_grid_over_n0": np.array([[2.0, 2.2, 2.4, 2.6, 2.8]]),
            "temperature_grid_MeV": np.array([[10.0, 12.0, 14.0, 16.0, 18.0]]),
            "analytic_jB_guess_grid": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
            "seed_radial_index_by_theta": np.array([2]),
        }
        jobs = self.runner.build_ray_direction_jobs(
            args=args,
            arrays=arrays,
            n0=1.0,
            B14=189.0,
            seed_continuations={0: {"jB": 8.0}},
        )
        self.assertEqual([job.direction for job in jobs], ["rho_out", "rho_in"])
        self.assertEqual(
            [[task.col_index for task in job.tasks] for job in jobs],
            [[3, 4], [1, 0]],
        )
        self.assertTrue(all(job.initial_continuation["jB"] == 8.0 for job in jobs))

    def test_xi_map_interpolates_converged_profile_measurements(self):
        arrays = {
            "nB_grid_over_n0": np.array([[1.0, 2.0], [1.0, 2.0]]),
            "temperature_grid_MeV": np.array([[0.0, 0.0], [1.0, 1.0]]),
            "xi_effective_grid": np.array([[-0.2, -0.4], [-0.6, -0.8]]),
        }
        self.runner.update_effective_xi_map(arrays)
        np.testing.assert_allclose(
            arrays["xi_effective_interpolated_grid"],
            arrays["xi_effective_grid"],
        )
        self.assertFalse(np.any(arrays["xi_interpolation_fallback_mask"]))


if __name__ == "__main__":
    unittest.main()
