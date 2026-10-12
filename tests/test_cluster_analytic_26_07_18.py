from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = REPO_ROOT / "new_paper_calculations" / "26-07-18"


def _load_module(name: str, filename: str):
    path = RUNNER_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AnalyticCluster26July18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = _load_module(
            "hydro_analytic_contour_common_26_07_18",
            "_hydro_analytic_contour_common.py",
        )
        cls.log_runner = _load_module(
            "hydro_analytic_log_runner_26_07_18",
            "run_hydro_analytic_velocity_contour_logT.py",
        )
        cls.linear_runner = _load_module(
            "hydro_analytic_linear_runner_26_07_18",
            "run_hydro_analytic_velocity_contour_linearT.py",
        )
        cls.safe_runner = _load_module(
            "hydro_analytic_safe_runner_26_07_18",
            "run_uN_discontinuity_contour.py",
        )

    def test_runner_defaults_differ_only_in_mesh_and_output_identity(self):
        log_config = self.log_runner.RUN_CONFIG
        linear_config = self.linear_runner.RUN_CONFIG

        self.assertEqual(log_config.temperature_mesh, "adaptive_log")
        self.assertEqual(log_config.temperature_axis_scale, "log")
        self.assertEqual(linear_config.temperature_mesh, "adaptive_linear")
        self.assertEqual(linear_config.temperature_axis_scale, "linear")
        self.assertNotEqual(log_config.run_tag, linear_config.run_tag)
        self.assertNotEqual(log_config.output_template, linear_config.output_template)

        ignored = {
            "temperature_mesh",
            "temperature_axis_scale",
            "run_tag",
            "output_template",
        }
        log_values = vars(log_config).copy()
        linear_values = vars(linear_config).copy()
        for key in ignored:
            log_values.pop(key)
            linear_values.pop(key)
        self.assertEqual(log_values, linear_values)
        self.assertEqual(log_config.temperature_min, 1.0e-2)
        self.assertEqual(log_config.temperature_max, 100.0)
        self.assertEqual(log_config.temperature_count, 30)
        self.assertEqual(log_config.metastability_count, 20)
        self.assertEqual(log_config.seed_temperature, 10.0)
        self.assertEqual(log_config.interface_fraction_mode, "LTE")

    def test_adaptive_log_and_linear_columns_have_same_endpoints(self):
        common = self.common
        kwargs = dict(
            temperature_min=1.0e-2,
            temperature_max=100.0,
            temperature_cut=80.0,
            boundary_eps=0.005,
            temperature_count=30,
        )
        log_values = common.build_adaptive_temperature_column(
            temperature_mesh="adaptive_log",
            **kwargs,
        )
        linear_values = common.build_adaptive_temperature_column(
            temperature_mesh="adaptive_linear",
            **kwargs,
        )

        self.assertAlmostEqual(log_values[0], 1.0e-2)
        self.assertAlmostEqual(linear_values[0], 1.0e-2)
        self.assertAlmostEqual(log_values[-1], 79.995)
        self.assertAlmostEqual(linear_values[-1], 79.995)
        np.testing.assert_allclose(
            np.diff(np.log(log_values)),
            np.diff(np.log(log_values))[0],
        )
        np.testing.assert_allclose(
            np.diff(linear_values),
            np.diff(linear_values)[0],
        )

    def test_metastability_uses_finite_temperature_critical_density(self):
        common = self.common
        common.critical_state_for_temperature.cache_clear()
        with (
            mock.patch.object(common, "_coexistence_root", return_value=1100.0),
            mock.patch.object(common.phase_velocity, "PQM", return_value=42.0) as pqm,
            mock.patch.object(common.phase_velocity, "nB_NM", return_value=2.75) as nB_nm,
        ):
            result = common.upstream_density_from_metastability(
                T=25.0,
                metastability=0.4,
                n0=1.0,
                B14=189.0,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
            )

        nB_N, nB_crit_T, muB_crit_T, P_crit_T = result
        self.assertEqual(nB_N, 3.15)
        self.assertEqual(nB_crit_T, 2.75)
        self.assertEqual(muB_crit_T, 1100.0)
        self.assertEqual(P_crit_T, 42.0)
        pqm.assert_called_with(1100.0, 0.0, 189.0, 25.0, ms=0.0, upB=5000)
        nB_nm.assert_called_with(
            1100.0,
            25.0,
            param=common.para.paraQMCRMF3,
            NM_type="PNM",
        )

    def test_analytic_point_uses_lte_and_neighbor_endpoint_seed(self):
        common = self.common
        analytic_result = {
            "success": True,
            "message": "ok",
            "u_N_max": 2.0e-6,
            "u_N_squared": 4.0e-12,
            "u_N_formula_squared": 4.0e-12,
            "analytic_velocity_residual": 0.0,
            "jB": 6.0,
            "u_Q": 1.0e-6,
            "gamma_N": 1.0,
            "gamma_Q": 1.0,
            "muB_Q": 1050.0,
            "T_Q": 20.0,
            "P_N": 1.0,
            "P_Q": 1.0,
            "pressure_jump": 0.0,
            "pressure_jump_balance": 0.0,
            "pressure_jump_residual": 0.0,
            "energy_flux_residual": 0.0,
            "momentum_flux_residual": 0.0,
            "endpoint_scaled_residual": 0.0,
            "mu_q": 350.0,
            "nB_N": 3.15,
            "nB_Q": 4.0,
            "h_N": 10.0,
            "h_Q": 12.0,
            "h_over_nB_N": 3.0,
            "h_over_nB_Q": 3.0,
            "lambda_n": 0.7875,
            "A_boundary": 0.8,
            "a_interface": 0.79,
            "a_interface_LTE": 0.79,
            "interface_fraction_mode": "LTE",
            "beta_LTE": 1.0e-5,
            "lte_correction": 0.2,
            "muKstar_max": 5.0,
            "gamma": 1.0,
            "tau": 2.0,
            "tau_seconds": 3.0,
            "slow_front_consistent": True,
            "composition_definition": "a_local_equals_nK_over_nB",
            "density_ratio_definition": "lambda_n_equals_nB_N_over_nB_Q",
            "analytic_formula_variant": "piecewise_constant_lambda_n",
        }
        with (
            mock.patch.object(
                common,
                "upstream_density_from_metastability",
                return_value=(3.15, 2.75, 1100.0, 42.0),
            ),
            mock.patch.object(common, "muB_from_nB", return_value=1125.0),
            mock.patch.object(
                common.phase_velocity,
                "analytic_velocity_bound",
                return_value=analytic_result,
            ) as solver,
        ):
            row = common.calculate_point(
                common.AnalyticPointTask(
                    T_N=25.0,
                    metastability=0.4,
                    n0=1.0,
                    nB_crit0=3.0,
                    B14=189.0,
                    xi=0.0,
                    ms=0.0,
                    nm_type="PNM",
                    upB=5000,
                    interface_fraction_mode="LTE",
                    endpoint_initial_guess=(1040.0, 19.0),
                )
            )

        self.assertTrue(row["success"])
        self.assertEqual(row["nB_N_MeV3"], 3.15)
        self.assertEqual(row["nB_crit_T_MeV3"], 2.75)
        self.assertEqual(row["metastability_definition"], "(nB_N - nB_crit(T))/n0")
        self.assertEqual(row["endpoint_guess_used"], (1040.0, 19.0))
        self.assertEqual(row["endpoint_solution"], (1050.0, 20.0))
        self.assertNotIn("a_N", row)
        self.assertEqual(solver.call_args.kwargs["interface_fraction_mode"], "LTE")
        self.assertEqual(solver.call_args.kwargs["initial_guess"], (1040.0, 19.0))

    def test_column_continuation_uses_only_immediate_successful_endpoint(self):
        common = self.common
        temperatures = np.array([1.0, 5.0, 10.0, 20.0, 40.0])
        seed_row = common.choose_seed_row_index(temperatures, 10.0)
        self.assertEqual(
            common.column_solve_order(temperatures.size, seed_row),
            [(2, "seed"), (3, "up"), (4, "up"), (1, "down"), (0, "down")],
        )

        solved = {2: {"success": True, "endpoint_solution": (1050.0, 20.0)}}
        guess, source = common.neighbor_endpoint_guess(3, "up", solved)
        self.assertEqual(guess, (1050.0, 20.0))
        self.assertEqual(source, "same_column_lower_neighbor")

        solved[3] = {"success": False, "endpoint_solution": None}
        guess, source = common.neighbor_endpoint_guess(4, "up", solved)
        self.assertIsNone(guess)
        self.assertEqual(source, "fallback_after_failed_neighbor")

    def test_velocity_work_records_configured_seed_and_retry_diagnostics(self):
        common = self.common
        task = common.AnalyticPointTask(
            T_N=12.0,
            metastability=0.3,
            n0=1.0,
            nB_crit0=3.0,
            B14=189.0,
            xi=0.0,
            ms=0.0,
            nm_type="PNM",
            upB=5000,
            interface_fraction_mode="LTE",
            endpoint_initial_guess=(1050.0, 20.0),
        )
        with mock.patch.object(
            common,
            "calculate_point",
            return_value={"success": False, "endpoint_solution": None},
        ):
            result = common.calculate_velocity_point(
                (0, 1, task, "retry", "retry_same_row_left_neighbor", 15.0, 1, (0, 1))
            )

        row = result["row"]
        self.assertEqual(row["continuation_seed_temperature_MeV"], 15.0)
        self.assertEqual(row["retry_attempt"], 1)
        self.assertEqual(row["retry_of"], (0, 1))

    def test_A_and_lte_boundary_hit_one_together(self):
        common = self.common
        for beta in (0.0, 1.0e-5, 1.0e-3, 0.1, 0.9, 1.1):
            lambda_n = 0.75
            u_N = beta / (5.0 * lambda_n)
            result = common.phase_velocity._analytic_a_0plus_lte(
                1.0,
                u_N,
                lambda_n,
            )
            self.assertEqual(result["a_0plus_LTE"], 1.0)

    def test_nonendpoint_probe_failure_is_not_a_boundary(self):
        common = self.common
        task = common.BoundaryTask(
            metastability=0.2,
            n0=1.0,
            nB_crit0=3.0,
            B14=189.0,
            xi=0.0,
            ms=0.0,
            nm_type="PNM",
            upB=5000,
            interface_fraction_mode="LTE",
            temperature_min=0.01,
            temperature_max=100.0,
            probe_count=3,
            bisect_steps=2,
        )
        rows = iter(
            [
                {"success": True, "endpoint_domain_failure": False, "A_boundary": 0.2, "a_interface_LTE": 0.2},
                {"success": False, "endpoint_domain_failure": False, "message": "RMF root failed"},
                {"success": True, "endpoint_domain_failure": False, "A_boundary": 0.8, "a_interface_LTE": 0.8},
            ]
        )
        with mock.patch.object(common, "_calculate_boundary_point", side_effect=lambda *_: next(rows)):
            result = common.solve_boundary_for_metastability(task)

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "nonendpoint_probe_failure")
        self.assertTrue(np.isnan(result["temperature_cut_MeV"]))

    def test_payload_records_mesh_identity_and_modern_schema(self):
        common = self.common
        config = self.log_runner.RUN_CONFIG
        args = common.parse_args(config, argv=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.npy"
            common.save_payload(
                path=path,
                config=config,
                args=args,
                B14=189.0,
                muB_crit0=1190.0,
                P_crit0=42.0,
                n0=1.0,
                nB_crit0=3.0,
                temperature_grid=np.array([[1.0]]),
                metastability_grid=np.array([[0.2]]),
                velocity_grid_m_s=np.array([[100.0]]),
                u_N_grid=np.array([[1.0e-6]]),
                lambda_n_grid=np.array([[0.75]]),
                A_boundary_grid=np.array([[0.8]]),
                a_interface_grid=np.array([[0.7]]),
                a_interface_LTE_grid=np.array([[0.7]]),
                nB_crit_T_grid=np.array([[2.8]]),
                muB_crit_T_grid=np.array([[1100.0]]),
                boundary_temperature_values=np.array([80.0]),
                boundary_metastability_values=np.array([0.2]),
                records=[],
                boundary_records=[],
            )
            payload = np.load(path, allow_pickle=True).item()

        self.assertEqual(payload["temperature_mesh"], "adaptive_log")
        self.assertEqual(payload["temperature_axis_scale"], "log")
        self.assertEqual(payload["metastability_definition"], "(nB_N - nB_crit(T))/n0")
        self.assertEqual(
            payload["density_ratio_definition"],
            "lambda_n_equals_nB_N_over_nB_Q",
        )
        self.assertNotIn("a_N_grid", payload)

    def test_safe_runner_defaults_to_two_block_extreme_endothermic_scan(self):
        runner = self.safe_runner
        parser = runner.build_parser()
        args = parser.parse_args([])

        self.assertEqual(runner.DEFAULT_OUTPUT, "uN-two-block-adaptive-contour.npy")
        self.assertEqual(args.interface_fraction_mode, "extreme_endothermic")
        self.assertEqual(args.temperature_mesh, "adaptive_log")
        self.assertEqual(args.seed_temperature, 10.0)
        self.assertEqual(args.split_temperature, 80.0)
        self.assertEqual(args.m_min, 0.0)
        self.assertGreater(args.metastability_boundary_eps, 0.0)
        self.assertEqual(args.save_every, 10)
        self.assertTrue(args.neighbor_retry)

    def test_gap_boundary_search_includes_metastability_axis(self):
        runner = self.safe_runner
        task = runner.GapBoundaryTask(
            T_N=30.0,
            B14=189.0,
            n0=1.0,
            ms=0.0,
            nm_type="PNM",
            upB=5000,
            m_lo=0.0,
            m_hi=0.9,
        )
        with (
            mock.patch.object(
                runner,
                "critical_state_for_temperature",
                return_value=(1100.0, 42.0, 3.0),
            ),
            mock.patch.object(
                runner,
                "gap_at_metastability",
                side_effect=lambda m, *_: 0.2 - float(m),
            ) as gap,
        ):
            result = runner.solve_gap_boundary_at_temperature(task)

        self.assertEqual(result["status"], runner.GAP_BOUNDARY_ROOT)
        self.assertAlmostEqual(result["metastability_boundary"], 0.2)
        self.assertEqual(gap.call_args_list[0].args[0], 0.0)
        self.assertTrue(result["allowed_above_boundary"])

    def test_low_block_row_starts_right_of_gap_boundary(self):
        runner = self.safe_runner
        values = runner.build_low_metastability_row(
            metastability_boundary=0.2,
            boundary_active=True,
            metastability_max=0.9,
            boundary_eps=1.0e-3,
            metastability_count=5,
        )
        self.assertEqual(values.size, 5)
        self.assertAlmostEqual(values[0], 0.201)
        self.assertAlmostEqual(values[-1], 0.9)
        self.assertTrue(np.all(values > 0.2))

        all_allowed = runner.build_low_metastability_row(
            metastability_boundary=0.0,
            boundary_active=False,
            metastability_max=0.9,
            boundary_eps=1.0e-3,
            metastability_count=5,
        )
        self.assertEqual(all_allowed[0], 0.0)

    def test_two_blocks_share_80_MeV_seam_and_stay_below_boundaries(self):
        runner = self.safe_runner
        low_temperatures = runner.build_low_temperature_axis(
            temperature_mesh="adaptive_log",
            temperature_min=0.01,
            split_temperature=80.0,
            temperature_count=30,
        )
        high_temperatures = runner.build_high_temperature_column(
            temperature_mesh="adaptive_linear",
            split_temperature=80.0,
            A_temperature_boundary=95.0,
            boundary_eps=0.005,
            temperature_count=8,
        )

        self.assertAlmostEqual(low_temperatures[0], 0.01)
        self.assertAlmostEqual(low_temperatures[-1], 80.0)
        self.assertAlmostEqual(high_temperatures[0], 80.0)
        self.assertAlmostEqual(high_temperatures[-1], 94.995)
        self.assertTrue(np.all(high_temperatures < 95.0))
        np.testing.assert_allclose(
            np.diff(np.log(low_temperatures)),
            np.diff(np.log(low_temperatures))[0],
        )

    def test_low_block_continuation_walks_from_high_m_and_rows_from_10_MeV(self):
        runner = self.safe_runner
        temperatures = np.array([1.0, 5.0, 10.0, 20.0, 40.0, 80.0])
        self.assertEqual(
            runner.low_temperature_row_order(temperatures, 10.0),
            [2, 3, 4, 5, 1, 0],
        )
        self.assertEqual(runner.low_metastability_solve_order(5), [4, 3, 2, 1, 0])

        solved = {4: {"success": True, "endpoint_solution": (1050.0, 20.0)}}
        guess, source = runner.low_row_neighbor_endpoint_guess(3, solved)
        self.assertEqual(guess, (1050.0, 20.0))
        self.assertEqual(source, "same_row_higher_metastability_neighbor")
        solved[3] = {"success": False, "endpoint_solution": None}
        guess, source = runner.low_row_neighbor_endpoint_guess(2, solved)
        self.assertIsNone(guess)
        self.assertEqual(source, "fallback_after_failed_neighbor")

    def test_two_block_payload_keeps_meshes_separate(self):
        runner = self.safe_runner
        args = runner.build_parser().parse_args([])
        arrays = {
            "low_temperature_grid": np.array([[10.0, 10.0], [80.0, 80.0]]),
            "low_metastability_grid": np.array([[0.2, 0.9], [0.0, 0.9]]),
            "high_temperature_grid": np.array([[80.0, 80.0], [95.0, 94.0]]),
            "high_metastability_grid": np.array([[0.0, 0.9], [0.0, 0.9]]),
        }
        payload = runner._payload(
            args=args,
            calibration=(189.0, 1190.0, 42.0, 1.0, 3.0),
            arrays=arrays,
            records=[],
            A_boundary_records=[],
            gap_boundary_records=[],
        )

        self.assertEqual(payload["mesh_layout"], "two_block_orthogonal_adaptive")
        self.assertEqual(payload["split_temperature_MeV"], 80.0)
        self.assertIn("low_temperature_grid", payload)
        self.assertIn("high_temperature_grid", payload)
        self.assertNotIn("safe_temperature_cut_values", payload)

    def test_safe_runner_passes_explicit_extreme_endothermic_mode(self):
        runner = self.safe_runner
        point_result = {
            "success": True,
            "message": "ok",
            "u_N_max": 1.0e-6,
            "u_N": 1.0e-6,
            "muB_Q": 1100.0,
            "T_Q": 12.0,
            "A_boundary": 0.7,
            "lambda_n": 0.8,
            "a_interface": 0.7,
            "a_interface_LTE": 0.69,
        }
        with mock.patch.object(
            runner.phase_velocity,
            "analytic_velocity_bound",
            return_value=point_result,
        ) as solver:
            result = runner.solve_analytic_velocity(
                muB_N=1200.0,
                T_N=20.0,
                B14=189.0,
                xi=0.0,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
                endpoint_initial_guess=(1090.0, 11.0),
                interface_fraction_mode="extreme_endothermic",
            )

        self.assertTrue(result["success"])
        self.assertEqual(
            solver.call_args.kwargs["interface_fraction_mode"],
            "extreme_endothermic",
        )
        self.assertEqual(solver.call_args.kwargs["initial_guess"], (1090.0, 11.0))

    def test_A_boundary_preserves_existing_bracket_after_numerical_limit(self):
        runner = self.safe_runner
        task = runner.ABoundaryTask(
            metastability=0.0,
            B14=189.0,
            n0=1.0,
            nB_crit0=3.0,
            xi=0.0,
            ms=0.0,
            nm_type="PNM",
            upB=5000,
            temperature_min=80.0,
            temperature_max=100.0,
            probe_count=2,
            bisect_steps=3,
            interface_fraction_mode="extreme_endothermic",
        )
        evaluations = iter(
            [
                {"classification": "allowed", "endpoint_solution": (1050.0, 30.0)},
                {"classification": "A_forbidden", "endpoint_solution": None},
                {"classification": "allowed", "endpoint_solution": (1060.0, 35.0)},
                {"classification": "A_forbidden", "endpoint_solution": None},
                {"classification": "numerical_failure", "endpoint_solution": None},
            ]
        )
        with mock.patch.object(
            runner,
            "_evaluate_A_boundary_point",
            side_effect=lambda *_: next(evaluations).copy(),
        ):
            result = runner.solve_A_boundary_for_metastability(task)

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], runner.A_BOUNDARY_SOLVED)
        self.assertAlmostEqual(result["temperature_lower_bracket_MeV"], 90.0)
        self.assertAlmostEqual(result["temperature_upper_bracket_MeV"], 95.0)
        self.assertAlmostEqual(result["temperature_cut_MeV"], 92.5)
        self.assertTrue(result["boundary_numerical_limit"])


if __name__ == "__main__":
    unittest.main()
