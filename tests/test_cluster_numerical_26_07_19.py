from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = REPO_ROOT / "new_paper_calculations" / "26-07-19"


def _load_runner():
    path = RUNNER_DIR / "run_uN_discontinuity_contour.py"
    spec = importlib.util.spec_from_file_location("numerical_uN_runner_26_07_19", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NumericalCluster26July19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _load_runner()

    def test_defaults_select_linear_numerical_uNmax(self):
        runner = self.runner
        parser = runner.build_parser()
        args = parser.parse_args([])

        self.assertEqual(runner.DEFAULT_TEMPERATURE_MESH, "adaptive_linear")
        self.assertEqual(args.temperature_mesh, "adaptive_linear")
        self.assertEqual(args.TQstar, 0.5)
        self.assertEqual(args.t_count, 30)
        self.assertEqual(args.m_count, 20)
        self.assertEqual(args.seed_temperature, 10.0)
        self.assertEqual(runner.NUMERICAL_CLOSURE_LABEL, "uNmax_direct_TQstar_nK")

    def test_numerical_point_calls_one_parameter_uNmax(self):
        runner = self.runner
        task = runner.NumericalPointTask(
            T_N=25.0,
            metastability=0.4,
            n0=1.0,
            nB_crit0=3.0,
            B14=189.0,
            ms=0.0,
            nm_type="PNM",
            upB=5000,
            TQstar=0.5,
            tail_eps=1.0e-8,
            n_mesh=200,
            tol_bvp=1.0e-4,
            max_nodes=10000,
            jB_guess=6.0,
            TQ_guess=None,
            jB_bounds=(1.0e-12, 300.0),
        )
        solver_result = {
            "success": True,
            "message": "ok",
            "u_N_max_candidate": 2.0e-6,
            "jB": 6.1,
            "u_Q": 1.0e-6,
            "lambda_n": 0.8,
            "aQstar": 0.2,
            "aQstar_derived": 0.2,
            "a_end": 1.0e-6,
            "muB_Q": 1050.0,
            "T_Q": 20.0,
            "T_Qstar": 0.5,
            "_root_method": "solve_bvp_uNmax_nK_parameter_1d",
            "solver_variant": "uNmax_direct_TQstar_nK",
            "composition_definition": "nK_over_local_nB",
            "density_ratio_definition": "lambda_n_equals_nB_N_over_nB_Q",
            "current_definition": "u_nK_minus_D_dnK_dx",
            "rate_model": "exact_nonleptonic",
            "bvp_parameters": np.array([1.0]),
        }

        with (
            mock.patch.object(
                runner,
                "critical_state_for_temperature",
                return_value=(1100.0, 42.0, 2.75),
            ),
            mock.patch.object(runner, "muB_from_nB", return_value=1125.0),
            mock.patch.object(
                runner.phase_velocity,
                "solve_front_energy_conserving_uNmax",
                return_value=solver_result,
            ) as solver,
        ):
            row = runner.calculate_numerical_point(task)

        self.assertTrue(row["success"])
        self.assertEqual(row["jB"], 6.1)
        self.assertEqual(row["root_method"], "solve_bvp_uNmax_nK_parameter_1d")
        self.assertEqual(row["bvp_parameter_count"], 1)
        self.assertEqual(row["a_local_Qstar"], 0.2)
        self.assertNotIn("analytic_velocity_residual", row)
        solver.assert_called_once()
        kwargs = solver.call_args.kwargs
        self.assertEqual(kwargs["TQstar"], 0.5)
        self.assertEqual(kwargs["jB_guess"], 6.0)
        self.assertEqual(kwargs["jB_bounds"], (1.0e-12, 300.0))
        self.assertFalse(kwargs["return_profile"])

    def test_neighbor_seed_uses_only_immediate_success(self):
        runner = self.runner
        solved = {
            3: {"success": True, "jB": 8.0},
            2: {"success": False, "jB": np.nan},
        }

        guess, source = runner.low_row_neighbor_jB_guess(1, solved, default=4.0)

        self.assertEqual(guess, 4.0)
        self.assertEqual(source, "fallback_after_failed_neighbor")

        solved[2] = {"success": True, "jB": 7.0}
        guess, source = runner.low_row_neighbor_jB_guess(1, solved, default=4.0)
        self.assertEqual(guess, 7.0)
        self.assertEqual(source, "same_row_higher_metastability_neighbor")

    def test_linear_low_temperature_axis(self):
        values = self.runner.build_low_temperature_axis(
            temperature_mesh=self.runner.DEFAULT_TEMPERATURE_MESH,
            temperature_min=1.0e-2,
            split_temperature=80.0,
            temperature_count=30,
        )
        np.testing.assert_allclose(np.diff(values), np.diff(values)[0])


if __name__ == "__main__":
    unittest.main()
