from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner(relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(
        f"cluster_runner_{path.parent.name}",
        path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import runner at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClusterUNmaxRunnerTests(unittest.TestCase):
    def _load_23jun_runner(self):
        return _load_runner("new_paper_calculations/26-06-23/run_hydro_numerical_uNmax_contour.py")

    def _load_26_07_17_runner(self):
        return _load_runner("new_paper_calculations/26-07-17/run_hydro_numerical_uNmax_contour.py")

    def _load_26_07_17_tqstar_0p5_runner(self):
        return _load_runner("new_paper_calculations/26-07-17/run_hydro_numerical_uNmax_contour_TQstar_0p5.py")

    def _load_26_07_17_linear_finite_T_runner(self):
        return _load_runner(
            "new_paper_calculations/26-07-17/"
            "run_hydro_numerical_uNmax_contour_linearT_finiteT_ncrit.py"
        )

    def test_26_07_17_runners_use_v2_run_tags(self):
        runner = self._load_26_07_17_runner()
        tqstar_runner = self._load_26_07_17_tqstar_0p5_runner()

        self.assertEqual(runner.DEFAULT_RUN_TAG, "numerical_uNmax_nK_logT_v2")
        self.assertEqual(
            tqstar_runner.DEFAULT_RUN_TAG,
            "numerical_uNmax_nK_logT_TQstar0p5_v2",
        )

    def test_26_07_17_tqstar_0p5_runner_has_distinct_default_output(self):
        runner = self._load_26_07_17_tqstar_0p5_runner()

        self.assertEqual(runner.DEFAULT_TQSTAR_MEV, 0.5)
        self.assertEqual(runner.DEFAULT_RUN_TAG, "numerical_uNmax_nK_logT_TQstar0p5_v2")
        self.assertEqual(runner.DEFAULT_TEMPERATURE_MIN_MEV, 1.0e-2)
        self.assertEqual(runner.DEFAULT_TEMPERATURE_MAX_MEV, 100.0)
        self.assertEqual(runner.TEMPERATURE_MESH_LABEL, "adaptive_log")

    def test_26_07_17_runners_define_metastability_from_finite_T_critical_density(self):
        for load_runner in (
            self._load_26_07_17_runner,
            self._load_26_07_17_tqstar_0p5_runner,
            self._load_26_07_17_linear_finite_T_runner,
        ):
            runner = load_runner()
            runner.critical_state_for_temperature.cache_clear()
            with self.subTest(path=runner.__file__):
                with (
                    mock.patch.object(runner, "_coexistence_root", return_value=1100.0) as root,
                    mock.patch.object(runner.phase_velocity, "PQM", return_value=42.0) as pqm,
                    mock.patch.object(runner.phase_velocity, "nB_NM", return_value=2.75) as nB_nm,
                ):
                    nB_N, nB_crit_T, muB_crit_T, P_crit_T = (
                        runner.upstream_density_from_metastability(
                            T=25.0,
                            metastability=0.4,
                            n0=1.0,
                            B14=189.0,
                            ms=0.0,
                            nm_type="PNM",
                            upB=5000,
                        )
                    )

                self.assertEqual(nB_crit_T, 2.75)
                self.assertEqual(nB_N, 3.15)
                self.assertEqual(muB_crit_T, 1100.0)
                self.assertEqual(P_crit_T, 42.0)
                root.assert_called_once()
                pqm.assert_called_with(1100.0, 0.0, 189.0, 25.0, ms=0.0, upB=5000)
                nB_nm.assert_called_with(
                    1100.0,
                    25.0,
                    param=runner.para.paraQMCRMF3,
                    NM_type="PNM",
                )

                with (
                    mock.patch.object(
                        runner,
                        "upstream_density_from_metastability",
                        return_value=(3.15, 2.75, 1100.0, 42.0),
                    ),
                    mock.patch.object(runner, "muB_from_nB", return_value=1125.0),
                    mock.patch.object(
                        runner.phase_velocity,
                        "solve_front_energy_conserving_uNmax",
                        return_value={"success": False, "message": "synthetic solver result"},
                    ) as solver,
                ):
                    row = runner.calculate_point(
                        (
                            25.0,
                            0.4,
                            1.0,
                            3.0,
                            189.0,
                            0.0,
                            0.0,
                            "PNM",
                            5000,
                            runner.DEFAULT_TQSTAR_MEV,
                            1.0e-8,
                            20,
                            1.0e-4,
                            100,
                            None,
                            None,
                            None,
                        )
                    )

                self.assertEqual(solver.call_args.args[1], 3.15)
                self.assertEqual(row["nB_crit_T_MeV3"], 2.75)
                self.assertEqual(row["nB_N_MeV3"], 3.15)
                self.assertEqual(row["metastability_definition"], "(nB_N - nB_crit(T))/n0")

    def test_26_07_17_linear_finite_T_runner_has_flat_distinct_output(self):
        runner = self._load_26_07_17_linear_finite_T_runner()

        self.assertEqual(runner.DEFAULT_TQSTAR_MEV, 0.5)
        self.assertEqual(runner.DEFAULT_TEMPERATURE_MIN_MEV, 0.1)
        self.assertEqual(runner.TEMPERATURE_MESH_LABEL, "adaptive_linear")
        self.assertEqual(
            runner.DEFAULT_RUN_TAG,
            "numerical_uNmax_nK_linearT_finiteT_ncrit_v1",
        )
        self.assertEqual(Path(runner.DEFAULT_OUTPUT_TEMPLATE).parent, Path("."))
        self.assertEqual(
            runner.DEFAULT_OUTPUT_TEMPLATE,
            "hydro-numerical-uNmax-adaptive-contour-finiteT-ncrit__{run_tag}.npy",
        )

    def test_26_07_17_runner_builds_adaptive_log_temperature_columns(self):
        runner = self._load_26_07_17_runner()

        self.assertEqual(runner.DEFAULT_TEMPERATURE_MIN_MEV, 1.0e-2)
        self.assertEqual(runner.DEFAULT_TEMPERATURE_MAX_MEV, 100.0)
        self.assertEqual(runner.DEFAULT_SEED_TEMPERATURE_MEV, 10.0)
        self.assertEqual(runner.TEMPERATURE_MESH_LABEL, "adaptive_log")
        self.assertEqual(runner.TEMPERATURE_AXIS_SCALE, "log")

        temperatures = runner.build_adaptive_temperature_column(
            temperature_min=1.0e-2,
            temperature_max=100.0,
            temperature_cut=80.0,
            boundary_eps=0.005,
            temperature_count=30,
        )
        self.assertEqual(temperatures.size, 30)
        self.assertAlmostEqual(temperatures[0], 1.0e-2)
        self.assertAlmostEqual(temperatures[-1], 79.995)
        np.testing.assert_allclose(np.diff(np.log(temperatures)), np.diff(np.log(temperatures))[0])
        self.assertEqual(
            runner.choose_seed_row_index(temperatures, runner.DEFAULT_SEED_TEMPERATURE_MEV),
            int(np.argmin(np.abs(temperatures - 10.0))),
        )

        capped_temperatures = runner.build_adaptive_temperature_column(
            temperature_min=1.0e-2,
            temperature_max=100.0,
            temperature_cut=120.0,
            boundary_eps=0.005,
            temperature_count=30,
        )
        self.assertAlmostEqual(capped_temperatures[-1], 100.0)

        with self.assertRaisesRegex(ValueError, "temperature_min must be positive"):
            runner.build_adaptive_temperature_column(
                temperature_min=0.0,
                temperature_max=100.0,
                temperature_cut=80.0,
                boundary_eps=0.005,
                temperature_count=30,
            )

    def test_26_07_17_runner_propagates_same_column_jB_from_10_mev_seed(self):
        runner = self._load_26_07_17_runner()
        temperatures = runner.build_adaptive_temperature_column(
            temperature_min=1.0e-2,
            temperature_max=100.0,
            temperature_cut=80.0,
            boundary_eps=0.005,
            temperature_count=30,
        )
        seed_row = runner.choose_seed_row_index(temperatures, 10.0)
        order = runner._column_solve_order(temperatures.size, seed_row)

        self.assertEqual(order[0], (seed_row, "seed"))
        self.assertEqual(order[1], (seed_row + 1, "up"))
        self.assertEqual(order[temperatures.size - seed_row], (seed_row - 1, "down"))

        column_task = (
            0,
            0.2,
            temperatures,
            1.0,
            3.0,
            180.0,
            0.0,
            0.0,
            "PNM",
            5000,
            0.0,
            1.0e-8,
            20,
            1.0e-4,
            100,
            None,
            None,
            None,
            80.0,
            0.005,
            10.0,
            True,
            1.0e-4,
            1.0e-12,
        )
        solved_seed = {seed_row: {"row": {"success": True, "jB": 42.0}}}
        upward_task = runner._make_column_point_task(column_task, seed_row + 1, "up", solved_seed)
        downward_task = runner._make_column_point_task(column_task, seed_row - 1, "down", solved_seed)

        self.assertEqual(upward_task[16], 42.0)
        self.assertEqual(upward_task[21], "same_column_lower_neighbor")
        self.assertEqual(downward_task[16], 42.0)
        self.assertEqual(downward_task[21], "same_column_upper_neighbor")

    def test_23jun_runner_defaults_are_tunable_constants(self):
        runner = self._load_23jun_runner()

        self.assertEqual(runner.DEFAULT_RUN_TAG, "numerical_uNmax_nK_v2")
        self.assertEqual(runner.DEFAULT_TEMPERATURE_COUNT, 30)
        self.assertEqual(runner.DEFAULT_METASTABILITY_COUNT, 20)
        self.assertEqual(runner.DEFAULT_BOUNDARY_PROBE_COUNT, 8)
        self.assertEqual(runner.DEFAULT_BOUNDARY_BISECT_STEPS, 10)
        self.assertEqual(runner.DEFAULT_SEED_TEMPERATURE_MEV, 10.0)
        self.assertTrue(runner.DEFAULT_NEIGHBOR_RETRY)
        self.assertEqual(runner.DEFAULT_SHOOTING_U_MAX, 1.0e-4)
        self.assertEqual(runner.DEFAULT_SHOOTING_JB_LOWER, 1.0e-12)

        with mock.patch.object(sys, "argv", ["runner"]):
            with mock.patch.dict(os.environ, {}, clear=False):
                args = runner.parse_args()
        self.assertEqual(args.run_tag, "numerical_uNmax_nK_v2")
        self.assertEqual(args.temperature_count, 30)
        self.assertEqual(args.metastability_count, 20)
        self.assertEqual(args.boundary_probe_count, 8)
        self.assertEqual(args.boundary_bisect_steps, 10)
        self.assertEqual(args.seed_temperature, 10.0)
        self.assertTrue(args.neighbor_retry)
        self.assertTrue(args.auto_jB_bounds)
        self.assertEqual(args.shooting_u_max, 1.0e-4)
        self.assertEqual(args.shooting_jB_lower, 1.0e-12)

        with mock.patch.object(sys, "argv", ["runner"]):
            with mock.patch.dict(os.environ, {"RUN_TAG": "env_tag"}, clear=False):
                args = runner.parse_args()
        self.assertEqual(args.run_tag, "env_tag")

        with mock.patch.object(sys, "argv", ["runner", "--seed-temperature", "15", "--no-neighbor-retry"]):
            args = runner.parse_args()
        self.assertEqual(args.seed_temperature, 15.0)
        self.assertFalse(args.neighbor_retry)

        with mock.patch.object(sys, "argv", ["runner", "--shooting-u-max", "2e-4", "--shooting-jB-lower", "1e-9", "--no-auto-jB-bounds"]):
            args = runner.parse_args()
        self.assertEqual(args.shooting_u_max, 2.0e-4)
        self.assertEqual(args.shooting_jB_lower, 1.0e-9)
        self.assertFalse(args.auto_jB_bounds)

    def test_23jun_runner_failure_rows_use_direct_nK_schema(self):
        runner = self._load_23jun_runner()

        self.assertEqual(runner.NUMERICAL_CLOSURE_LABEL, "uNmax_direct_TQstar_nK")
        failure = runner.failure_values("synthetic failure")
        self.assertEqual(failure["solver_variant"], "uNmax_direct_TQstar_nK")
        self.assertEqual(failure["rate_model"], "exact_nonleptonic")
        self.assertEqual(failure["composition_definition"], "nK_over_local_nB")
        self.assertEqual(failure["density_ratio_definition"], "lambda_n_equals_nB_N_over_nB_Q")
        self.assertEqual(failure["current_definition"], "u_nK_minus_D_dnK_dx")
        self.assertEqual(failure["root_method"], "solve_bvp_uNmax_nK_parameter_1d")
        self.assertEqual(failure["bvp_parameter_count"], -1)
        self.assertIn("lambda_n", failure)
        self.assertIn("a_local_Qstar", failure)
        self.assertIn("a_local_end", failure)
        self.assertIn("aQstar", failure)
        self.assertIn("aQstar_derived", failure)
        self.assertIn("aQstar_max", failure)
        self.assertIn("downstream_root_calls", failure)
        self.assertIn("local_root_calls", failure)
        self.assertIn("qstar_root_calls", failure)
        self.assertIn("nK_Qstar_MeV3", failure)
        self.assertIn("nK_Qstar", failure)
        self.assertIn("nK_end", failure)
        self.assertIn("jK_end", failure)
        self.assertIn("jK_left_residual_norm", failure)
        self.assertNotIn("a_N", failure)
        self.assertNotIn("q_end", failure)
        self.assertNotIn("q_left_residual_norm", failure)

    def test_23jun_runner_payload_saves_local_a_grid_and_legacy_alias(self):
        runner = self._load_23jun_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.npy"
            grid = np.array([[0.125]], dtype=float)
            runner.save_payload(
                path,
                temperature_values=np.array([0.1], dtype=float),
                boundary_temperature_values=np.array([1.0], dtype=float),
                boundary_metastability_values=np.array([0.01], dtype=float),
                temperature_grid=np.array([[0.1]], dtype=float),
                metastability_grid=np.array([[0.01]], dtype=float),
                velocity_grid_m_s=np.array([[2.0]], dtype=float),
                u_N_grid=np.array([[3.0]], dtype=float),
                a_local_grid=grid,
                records=[],
                boundary_records=[],
                B14=180.0,
                n0=1.0,
                nB_crit=3.0,
                xi=0.0,
                ms=0.0,
                nm_type="PNM",
                run_tag="unit",
                boundary_eps=0.005,
                TQstar=0.5,
            )
            payload = np.load(path, allow_pickle=True).item()

        self.assertIn("a_local_grid", payload)
        self.assertIn("aQstar_grid", payload)
        np.testing.assert_allclose(payload["a_local_grid"], grid)
        np.testing.assert_allclose(payload["aQstar_grid"], grid)
        self.assertEqual(payload["root_method"], "solve_bvp_uNmax_nK_parameter_1d")
        self.assertEqual(payload["density_ratio_definition"], "lambda_n_equals_nB_N_over_nB_Q")

    def test_23jun_seed_row_and_column_continuation_order(self):
        runner = self._load_23jun_runner()
        temperatures = np.array([0.1, 5.0, 10.0, 15.0, 20.0], dtype=float)
        self.assertEqual(runner.choose_seed_row_index(temperatures, 10.0), 2)

        calls = []

        def fake_calculate_velocity_point(task):
            calls.append(task)
            row_index = int(task[0])
            col_index = int(task[1])
            jB_guess = task[16]
            source = task[21]
            direction = task[22]
            success = row_index != 3
            return {
                "row_index": row_index,
                "col_index": col_index,
                "T_MeV": float(task[2]),
                "metastability": float(task[3]),
                "row": {
                    "success": success,
                    "jB": float(row_index + 100) if success else np.nan,
                    "jB_guess_used": np.nan if jB_guess is None else float(jB_guess),
                    "jB_guess_source": source,
                    "continuation_direction": direction,
                },
            }

        column_task = (
            0,
            0.2,
            temperatures,
            1.0,
            3.0,
            180.0,
            0.0,
            0.0,
            "PNM",
            5000,
            0.5,
            1.0e-8,
            20,
            1.0e-4,
            100,
            None,
            None,
            None,
            20.0,
            0.005,
            10.0,
            True,
            1.0e-4,
            1.0e-12,
        )
        with mock.patch.object(runner, "calculate_velocity_point", side_effect=fake_calculate_velocity_point):
            result = runner.solve_velocity_column(column_task)

        self.assertEqual([item["row_index"] for item in result["results"]], [2, 3, 4, 1, 0])
        self.assertEqual([task[22] for task in calls], ["seed", "up", "up", "down", "down"])
        self.assertEqual(calls[0][21], "default")
        self.assertEqual(calls[1][16], 102.0)
        self.assertEqual(calls[1][21], "same_column_lower_neighbor")
        self.assertIsNone(calls[2][16])
        self.assertEqual(calls[2][21], "fallback_after_failed_neighbor")
        self.assertEqual(calls[3][16], 102.0)
        self.assertEqual(calls[3][21], "same_column_upper_neighbor")

    def test_23jun_retry_uses_only_immediate_successful_neighbors(self):
        runner = self._load_23jun_runner()

        successful_cells = {
            (1, 0): {"row": {"success": True, "jB": 11.0}},
            (0, 1): {"row": {"success": True, "jB": 21.0}},
            (3, 0): {"row": {"success": True, "jB": 31.0}},
        }
        seed, source = runner._immediate_retry_seed(2, 0, successful_cells)
        self.assertEqual(seed, 11.0)
        self.assertEqual(source, "retry_same_column_lower_neighbor")

        successful_cells = {
            (3, 0): {"row": {"success": True, "jB": 31.0}},
            (0, 1): {"row": {"success": True, "jB": 21.0}},
        }
        seed, source = runner._immediate_retry_seed(2, 0, successful_cells)
        self.assertEqual(seed, 31.0)
        self.assertEqual(source, "retry_same_column_upper_neighbor")

        successful_cells = {
            (0, 0): {"row": {"success": True, "jB": 1.0}},
            (4, 0): {"row": {"success": True, "jB": 4.0}},
        }
        seed, source = runner._immediate_retry_seed(2, 0, successful_cells)
        self.assertIsNone(seed)
        self.assertEqual(source, "")

    def test_23jun_auto_jB_bounds_are_velocity_based_and_overridable(self):
        runner = self._load_23jun_runner()

        bounds, diagnostics = runner.resolve_jB_bounds_for_point(
            nB_N=4.0e6,
            explicit_jB_bounds=None,
            auto_jB_bounds=True,
            shooting_u_max=1.0e-4,
            shooting_jB_lower=1.0e-12,
        )
        self.assertEqual(bounds, (1.0e-12, 400.0))
        self.assertTrue(diagnostics["jB_bounds_used"])
        self.assertTrue(diagnostics["auto_jB_bounds"])
        self.assertEqual(diagnostics["jB_upper_bound"], 400.0)

        bounds, diagnostics = runner.resolve_jB_bounds_for_point(
            nB_N=4.0e6,
            explicit_jB_bounds=(2.0, 3.0),
            auto_jB_bounds=True,
            shooting_u_max=1.0e-4,
            shooting_jB_lower=1.0e-12,
        )
        self.assertEqual(bounds, (2.0, 3.0))
        self.assertTrue(diagnostics["jB_bounds_used"])
        self.assertFalse(diagnostics["auto_jB_bounds"])
        self.assertEqual(diagnostics["jB_lower_bound"], 2.0)
        self.assertEqual(diagnostics["jB_upper_bound"], 3.0)

        bounds, diagnostics = runner.resolve_jB_bounds_for_point(
            nB_N=4.0e6,
            explicit_jB_bounds=None,
            auto_jB_bounds=False,
            shooting_u_max=1.0e-4,
            shooting_jB_lower=1.0e-12,
        )
        self.assertIsNone(bounds)
        self.assertFalse(diagnostics["jB_bounds_used"])

    def test_23jun_velocity_point_passes_bounds_and_records_diagnostics(self):
        runner = self._load_23jun_runner()
        captured = {}

        def fake_calculate_point(task):
            captured["task"] = task
            return {
                "success": True,
                "jB": 2.5,
                "u_N": 1.0e-6,
                "u_N_max_candidate": 1.0e-6,
            }

        task = runner._make_velocity_task(
            row_index=0,
            col_index=0,
            T_N=10.0,
            metastability=0.25,
            n0=10.0,
            nB_crit=30.0,
            B14=180.0,
            xi=0.0,
            ms=0.0,
            nm_type="PNM",
            upB=5000,
            TQstar=0.5,
            tail_eps=1.0e-8,
            n_mesh=20,
            tol_bvp=1.0e-4,
            max_nodes=100,
            jB_guess=None,
            TQ_guess=None,
            jB_bounds=None,
            temperature_cut=20.0,
            boundary_eps=0.005,
            jB_guess_source="default",
            continuation_direction="seed",
            continuation_seed_temperature=10.0,
            auto_jB_bounds=True,
            shooting_u_max=1.0e-4,
            shooting_jB_lower=1.0e-12,
        )
        with mock.patch.object(runner, "calculate_point", side_effect=fake_calculate_point):
            point = runner.calculate_velocity_point(task)

        passed_bounds = captured["task"][16]
        self.assertEqual(passed_bounds[0], 1.0e-12)
        self.assertAlmostEqual(passed_bounds[1], 0.00325)
        row = point["row"]
        self.assertTrue(row["jB_bounds_used"])
        self.assertTrue(row["auto_jB_bounds"])
        self.assertEqual(row["jB_lower_bound"], 1.0e-12)
        self.assertAlmostEqual(row["jB_upper_bound"], 0.00325)
        self.assertEqual(row["shooting_u_max"], 1.0e-4)


if __name__ == "__main__":
    unittest.main()
