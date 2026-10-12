from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "new_paper_calculations"
    / "26-07-19"
    / "scan_analytic_logTm.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("scan_analytic_logTm", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnalyticLogTLogMRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_defaults_define_requested_domain(self):
        args = self.runner.build_parser().parse_args([])
        self.assertEqual(args.t_min, 1.0e-2)
        self.assertEqual(args.t_max, 50.0)
        self.assertEqual(args.m_min, 1.0e-5)
        self.assertEqual(args.m_max, 1.0)
        self.assertEqual(args.t_count, 30)
        self.assertEqual(args.m_count, 20)
        self.assertEqual(args.seed_temperature, 10.0)
        self.assertEqual(args.interface_fraction_mode, "extreme_endothermic")
        self.assertEqual(args.output, "analytic-uN-logTm.npy")

    def test_runner_contains_no_numerical_or_A_boundary_contract(self):
        self.assertFalse(hasattr(self.runner, "NumericalPointTask"))
        self.assertFalse(hasattr(self.runner, "resolve_jB_bounds"))
        self.assertFalse(hasattr(self.runner, "ABoundaryTask"))
        self.assertFalse(hasattr(self.runner, "solve_A_boundary_for_metastability"))

    def test_temperature_axis_is_logarithmic(self):
        values = self.runner.build_temperature_axis(1.0e-2, 50.0, 30)
        self.assertEqual(values[0], 1.0e-2)
        self.assertEqual(values[-1], 50.0)
        np.testing.assert_allclose(np.diff(np.log(values)), np.diff(np.log(values))[0])

    def test_gap_root_sets_log_metastability_lower_bound(self):
        values = self.runner.build_metastability_row(
            boundary=2.0e-2,
            boundary_status=self.runner.GAP_BOUNDARY_ROOT,
            allowed_above_boundary=True,
            m_min=1.0e-5,
            m_max=1.0,
            count=20,
            gap_buffer_fraction=1.0e-3,
        )
        self.assertAlmostEqual(values[0], 2.0e-2 * 1.001)
        self.assertEqual(values[-1], 1.0)
        np.testing.assert_allclose(np.diff(np.log(values)), np.diff(np.log(values))[0])

    def test_all_allowed_and_forbidden_rows(self):
        allowed = self.runner.build_metastability_row(
            boundary=np.nan,
            boundary_status=self.runner.GAP_BOUNDARY_ALL_ALLOWED,
            allowed_above_boundary=True,
            m_min=1.0e-5,
            m_max=1.0,
            count=20,
            gap_buffer_fraction=1.0e-3,
        )
        self.assertEqual(allowed[0], 1.0e-5)
        forbidden = self.runner.build_metastability_row(
            boundary=np.nan,
            boundary_status=self.runner.GAP_BOUNDARY_ALL_FORBIDDEN,
            allowed_above_boundary=True,
            m_min=1.0e-5,
            m_max=1.0,
            count=20,
            gap_buffer_fraction=1.0e-3,
        )
        self.assertIsNone(forbidden)

    def test_adjacent_temperature_uses_nearest_log_m_endpoint(self):
        adjacent = {
            0: {
                "success": True,
                "metastability": 1.0e-3,
                "endpoint_solution": (1000.0, 20.0),
            },
            1: {
                "success": True,
                "metastability": 1.0e-2,
                "endpoint_solution": (1050.0, 25.0),
            },
        }
        guess, source = self.runner.adjacent_row_endpoint_guess(
            metastability=8.0e-3,
            adjacent_row=adjacent,
            default=None,
        )
        self.assertEqual(guess, (1050.0, 25.0))
        self.assertEqual(source, "adjacent_temperature_nearest_log_m")

    def test_failed_nearest_endpoint_does_not_use_farther_success(self):
        adjacent = {
            0: {
                "success": True,
                "metastability": 1.0e-3,
                "endpoint_solution": (1000.0, 20.0),
            },
            1: {
                "success": False,
                "metastability": 1.0e-2,
                "endpoint_solution": None,
            },
        }
        guess, source = self.runner.adjacent_row_endpoint_guess(
            metastability=8.0e-3,
            adjacent_row=adjacent,
            default=(1100.0, 30.0),
        )
        self.assertEqual(guess, (1100.0, 30.0))
        self.assertEqual(source, "fallback_after_failed_nearest_log_m")

    def test_point_adapter_uses_extreme_endothermic_analytic_task(self):
        task = self.runner.PointWork(
            row_index=2,
            col_index=3,
            T_N=10.0,
            metastability=0.2,
            n0=1.0,
            nB_crit0=3.0,
            B14=189.0,
            xi=0.0,
            ms=0.0,
            nm_type="PNM",
            upB=5000,
            endpoint_guess=(1000.0, 20.0),
            endpoint_guess_source="test",
            continuation_direction="up",
            continuation_seed_temperature=10.0,
        )
        solved = {
            "success": True,
            "T_MeV": 10.0,
            "metastability": 0.2,
            "u_N_m_s": 100.0,
            "endpoint_solution": (1010.0, 21.0),
        }
        with mock.patch.object(
            self.runner,
            "calculate_analytic_point",
            return_value=solved,
        ) as calculate:
            result = self.runner.calculate_velocity_point(task)
        self.assertTrue(result["row"]["success"])
        analytic_task = calculate.call_args.args[0]
        self.assertEqual(analytic_task.interface_fraction_mode, "extreme_endothermic")
        self.assertEqual(analytic_task.endpoint_initial_guess, (1000.0, 20.0))

    def test_entropy_jump_is_downstream_minus_upstream(self):
        row = {
            "T_MeV": 10.0,
            "T_Q_MeV": 20.0,
            "h_over_nB_N_MeV": 1100.0,
            "muB_N_MeV": 1000.0,
            "h_over_nB_Q_MeV": 1050.0,
            "muB_Q_MeV": 1010.0,
        }
        self.assertEqual(self.runner.entropy_jump_per_baryon(row), 2.0 - 10.0)

    def test_entropy_zero_uses_log_m_interpolation(self):
        crossing = self.runner.entropy_zero_crossing(
            metastability=np.array([1.0e-3, 1.0e-2]),
            entropy_jump=np.array([-1.0, 1.0]),
        )
        self.assertAlmostEqual(crossing, np.sqrt(1.0e-3 * 1.0e-2))

    def test_payload_metadata_has_no_A_boundary_scan(self):
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
        )
        self.assertEqual(payload["temperature_axis_scale"], "log")
        self.assertEqual(payload["metastability_axis_scale"], "log")
        self.assertEqual(payload["safe_region_definition"], "Delta_h <= 0")
        self.assertEqual(payload["A_boundary_handling"], "not_scanned_T_max_50_MeV")
        self.assertFalse(payload["entropy_affects_mesh"])
        self.assertNotIn("A_boundary_records", payload)
        self.assertNotIn("root_method", payload)


if __name__ == "__main__":
    unittest.main()
