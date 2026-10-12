from __future__ import annotations

import importlib.util
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "new_paper_calculations"
    / "26-07-20"
    / "run_uNmax_logTm_entropy_safe.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_uNmax_logTm_entropy_safe", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NumericalEntropySafeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_defaults_define_requested_log_log_domain(self):
        args = self.runner.build_parser().parse_args([])
        self.assertEqual(args.t_min, 1.0e-2)
        self.assertEqual(args.t_max, 50.0)
        self.assertEqual(args.m_min, 1.0e-5)
        self.assertEqual(args.m_max, 1.0)
        self.assertEqual(args.t_count, 30)
        self.assertEqual(args.m_count, 20)
        self.assertEqual(args.seed_temperature, 10.0)
        self.assertEqual(args.TQstar, 0.5)
        self.assertGreater(args.point_timeout, 0.0)
        self.assertEqual(args.output, "uNmax-logTm-entropy-safe.npy")

    def test_bag_calibration_receives_physical_critical_density(self):
        args = self.runner.build_parser().parse_args([])
        expected_n0 = 0.16 * self.runner.const.MeV_fm**3
        with mock.patch.object(
            self.runner,
            "calibrate_b14",
            return_value=(189.0, 1100.0, 2.0),
        ) as calibrate:
            calibration = self.runner.resolve_calibration(args)
        calibrate.assert_called_once_with(
            args.ncrit_over_n0 * expected_n0, args.ms, args.nm_type
        )
        self.assertEqual(calibration[0], 189.0)
        self.assertAlmostEqual(calibration[3], expected_n0)
        self.assertAlmostEqual(calibration[4], args.ncrit_over_n0 * expected_n0)

    def test_temperature_axis_is_logarithmic(self):
        values = self.runner.build_temperature_axis(1.0e-2, 50.0, 30)
        self.assertEqual(values[0], 1.0e-2)
        self.assertEqual(values[-1], 50.0)
        np.testing.assert_allclose(
            np.diff(np.log(values)), np.diff(np.log(values))[0]
        )

    def test_temperature_schedule_restarts_downward_sweep_at_seed(self):
        schedule = self.runner.temperature_row_schedule(
            np.array([0.1, 1.0, 10.0, 20.0, 40.0]), seed_temperature=10.0
        )
        self.assertEqual(
            schedule,
            [
                (2, None, "seed"),
                (3, 2, "up"),
                (4, 3, "up"),
                (1, 2, "down"),
                (0, 1, "down"),
            ],
        )

    def test_entropy_root_sets_stricter_log_metastability_lower_bound(self):
        values = self.runner.build_entropy_safe_metastability_row(
            gap_boundary=1.0e-2,
            gap_status=self.runner.GAP_BOUNDARY_ROOT,
            gap_allowed_above=True,
            entropy_boundary=2.0e-2,
            entropy_status=self.runner.ENTROPY_BOUNDARY_ROOT,
            entropy_allowed_above=True,
            m_min=1.0e-5,
            m_max=1.0,
            count=20,
            gap_buffer_fraction=1.0e-3,
            entropy_buffer_fraction=2.0e-3,
        )
        self.assertAlmostEqual(values[0], 2.0e-2 * 1.002)
        self.assertEqual(values[-1], 1.0)
        np.testing.assert_allclose(
            np.diff(np.log(values)), np.diff(np.log(values))[0]
        )

    def test_entropy_forbidden_or_ambiguous_row_is_not_scanned(self):
        common = dict(
            gap_boundary=np.nan,
            gap_status=self.runner.GAP_BOUNDARY_ALL_ALLOWED,
            gap_allowed_above=True,
            entropy_boundary=np.nan,
            entropy_allowed_above=True,
            m_min=1.0e-5,
            m_max=1.0,
            count=20,
            gap_buffer_fraction=1.0e-3,
            entropy_buffer_fraction=1.0e-3,
        )
        self.assertIsNone(
            self.runner.build_entropy_safe_metastability_row(
                entropy_status=self.runner.ENTROPY_BOUNDARY_ALL_FORBIDDEN,
                **common,
            )
        )
        self.assertIsNone(
            self.runner.build_entropy_safe_metastability_row(
                entropy_status=self.runner.ENTROPY_BOUNDARY_MULTIPLE_ROOTS,
                **common,
            )
        )

    def test_entropy_zero_crossing_uses_log_metastability(self):
        crossing = self.runner.entropy_zero_crossing(
            metastability=np.array([1.0e-3, 1.0e-2]),
            entropy_jump=np.array([-1.0, 1.0]),
        )
        self.assertAlmostEqual(crossing, np.sqrt(1.0e-3 * 1.0e-2))

    def test_adjacent_temperature_uses_nearest_log_m_jB(self):
        adjacent = {
            0: {"success": True, "metastability": 1.0e-3, "jB": 2.0},
            1: {"success": True, "metastability": 1.0e-2, "jB": 3.0},
        }
        guess, source = self.runner.adjacent_row_jB_guess(
            metastability=8.0e-3,
            adjacent_row=adjacent,
            default=1.0,
        )
        self.assertEqual(guess, 3.0)
        self.assertEqual(source, "adjacent_temperature_nearest_log_m")

    def test_failed_nearest_jB_does_not_use_farther_success(self):
        adjacent = {
            0: {"success": True, "metastability": 1.0e-3, "jB": 2.0},
            1: {"success": False, "metastability": 1.0e-2, "jB": np.nan},
        }
        guess, source = self.runner.adjacent_row_jB_guess(
            metastability=8.0e-3,
            adjacent_row=adjacent,
            default=1.0,
        )
        self.assertEqual(guess, 1.0)
        self.assertEqual(source, "fallback_after_failed_nearest_log_m")

    def test_timeout_interrupts_a_stalled_point(self):
        with self.assertRaises(self.runner.PointTimeoutError):
            with self.runner.point_timeout(0.01):
                time.sleep(0.1)

    def test_numerical_adapter_uses_uNmax_nK_and_fixed_TQstar(self):
        task = self.runner.NumericalPointTask(
            row_index=2,
            col_index=3,
            T_N=10.0,
            metastability=0.2,
            n0=1.0,
            nB_crit0=3.0,
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
            jB_guess_source="test",
            continuation_direction="seed",
            continuation_seed_temperature=10.0,
        )
        critical = (900.0, 1.0, 3.0)
        solved = {
            "success": True,
            "message": "ok",
            "u_N": 1.0e-6,
            "u_N_max_candidate": 1.0e-6,
            "jB": 1.0,
            "muB_N": 1000.0,
            "muB_Q": 1050.0,
            "T_Q": 20.0,
            "nB_Q": 4.0,
            "h_N": 4040.0,
            "h_Q": 4240.0,
            "lambda_n": 1.0,
            "aQstar": 0.5,
            "aQstar_derived": 0.5,
            "a_end": 0.0,
            "bvp_parameters": np.array([1.0]),
            "_root_method": "solve_bvp_uNmax_nK_parameter_1d",
            "solver_variant": "uNmax_direct_TQstar_nK",
        }
        with (
            mock.patch.object(
                self.runner, "critical_state_for_temperature", return_value=critical
            ),
            mock.patch.object(
                self.runner.phase_velocity,
                "solve_front_energy_conserving_uNmax",
                return_value=solved,
            ) as calculate,
        ):
            row = self.runner.calculate_numerical_point(task)
        self.assertTrue(row["success"])
        self.assertEqual(row["root_method"], "solve_bvp_uNmax_nK_parameter_1d")
        self.assertEqual(row["bvp_parameter_count"], 1)
        self.assertEqual(calculate.call_args.kwargs["TQstar"], 0.5)
        self.assertEqual(calculate.call_args.kwargs["jB_guess"], 2.5)
        self.assertEqual(row["jB_guess_source"], "test")
        self.assertFalse(row["point_timed_out"])

    def test_payload_records_entropy_safe_numerical_contract(self):
        args = self.runner.build_parser().parse_args([])
        arrays = {
            "temperature_values": np.array([1.0]),
            "temperature_grid": np.array([[1.0]]),
            "metastability_grid": np.array([[0.1]]),
        }
        payload = self.runner.build_payload(
            args=args,
            calibration=(189.0, 1000.0, 1.0, 2.0, 6.0),
            arrays=arrays,
            records=[],
            gap_boundary_records=[],
            entropy_boundary_records=[],
        )
        self.assertEqual(payload["temperature_axis_scale"], "log")
        self.assertEqual(payload["metastability_axis_scale"], "log")
        self.assertTrue(payload["entropy_affects_mesh"])
        self.assertEqual(
            payload["safe_region_definition"],
            "Delta_h <= 0 and Delta_s_per_baryon >= 0",
        )
        self.assertEqual(
            payload["root_method"], "solve_bvp_uNmax_nK_parameter_1d"
        )
        self.assertEqual(payload["TQstar_target_MeV"], 0.5)


if __name__ == "__main__":
    unittest.main()
