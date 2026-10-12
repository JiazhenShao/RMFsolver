from __future__ import annotations

from contextlib import redirect_stderr
import importlib.util
import inspect
import io
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = REPO_ROOT / "new_paper_calculations" / "26-07-23"


def _load_module(name: str, filename: str):
    path = RUNNER_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LTEInterfaceTemperatureJump26July23Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _load_module(
            "lte_temperature_jump_runner_26_07_23",
            "run_lte_temperature_jump_contour.py",
        )

    @staticmethod
    def _successful_analytic_row():
        return {
            "success": True,
            "u_N": 0.02,
            "jB": 3.0,
            "P_N_MeV4": 40.0,
            "h_N_MeV4": 1000.0,
            "muB_Q_MeV": 1100.0,
            "T_Q_MeV": 20.0,
            "P_Q_MeV4": 75.0,
            "A_boundary": 0.8,
            "a_interface_LTE": 0.4,
            "lambda_n": 0.75,
            "beta_LTE": 0.075,
        }

    def _patch_exact_isobar_for_current_fraction(self, row):
        a_target = float(row["a_interface_LTE"])
        beta = 5.0 * float(row["lambda_n"]) * float(row["u_N"])
        A_exact = float(
            np.sqrt(a_target * a_target + beta * a_target * (1.0 - a_target))
        )
        return patch.object(
            self.runner,
            "_exact_local_A_on_zero_temperature_isobar",
            return_value={
                "success": True,
                "status": 0,
                "message": "exact local isobar converged",
                "muK_A_MeV": 85.0,
                "A_boundary_exact_local": A_exact,
                "pressure_residual_norm": 0.0,
            },
            create=True,
        )

    def test_exact_zero_temperature_isobar_returns_local_eos_fraction(self):
        self.assertTrue(
            hasattr(
                self.runner,
                "_exact_local_A_on_zero_temperature_isobar",
            ),
            "runner must provide a private exact local-isobar helper",
        )

        def fake_pressure(muB, muK, B14, T, *, ms, upB):
            self.assertEqual(muB, 1100.0)
            self.assertEqual(B14, 180.0)
            self.assertEqual(T, 0.0)
            self.assertEqual(ms, 0.0)
            self.assertEqual(upB, 5000)
            return 50.0 + muK

        with (
            patch.object(self.runner, "PQM", side_effect=fake_pressure),
            patch.object(self.runner, "nB_QM", return_value=8.0),
            patch.object(self.runner, "nK_QM", return_value=2.0),
        ):
            result = self.runner._exact_local_A_on_zero_temperature_isobar(
                muB_Q=1100.0,
                P_Q=56.0,
                B14=180.0,
                ms=0.0,
                upB=5000,
                muK_probe=1.0,
            )

        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["muK_A_MeV"], 6.0)
        self.assertAlmostEqual(result["A_boundary_exact_local"], 0.25)
        self.assertAlmostEqual(result["pressure_residual_norm"], 0.0)

    def test_lte_fraction_is_recomputed_from_exact_local_isobar_ceiling(self):
        row = self._successful_analytic_row()
        exact_A = 0.3
        expected_lte = self.runner.phase_velocity._analytic_a_0plus_lte(
            exact_A,
            row["u_N"],
            row["lambda_n"],
        )
        captured = {}

        def fake_jump(a_target, E, Pi, jB, B14, **kwargs):
            captured["a_target"] = float(a_target)
            return {
                "muB": 1120.0,
                "muK": 70.0,
                "T": 8.0,
                "nB": 10.0,
                "nK": 10.0 * float(a_target),
                "u": 0.3,
                "E": E,
                "Pi": Pi,
            }

        with (
            patch.object(
                self.runner,
                "_exact_local_A_on_zero_temperature_isobar",
                return_value={
                    "success": True,
                    "status": 0,
                    "message": "exact local isobar converged",
                    "muK_A_MeV": 72.0,
                    "A_boundary_exact_local": exact_A,
                    "pressure_residual_norm": 0.0,
                },
                create=True,
            ),
            patch.object(
                self.runner.phase_velocity,
                "_solve_interface_state_from_local_a_E_and_Pi",
                side_effect=fake_jump,
            ),
        ):
            self.runner.add_lte_temperature_fields(
                row,
                T_N_MeV=25.0,
                B14=180.0,
                ms=0.0,
                upB=5000,
            )

        self.assertAlmostEqual(row["A_boundary_analytic"], 0.8)
        self.assertAlmostEqual(row["A_boundary_exact_local"], exact_A)
        self.assertAlmostEqual(
            row["a_interface_LTE"],
            expected_lte["a_0plus_LTE"],
        )
        self.assertAlmostEqual(captured["a_target"], row["a_interface_LTE"])
        self.assertNotAlmostEqual(captured["a_target"], 0.4)

    def test_exact_isobar_failure_marks_temperature_nan_without_jump_attempt(self):
        row = self._successful_analytic_row()
        row["beta_LTE"] = 999.0
        with (
            patch.object(
                self.runner,
                "_exact_local_A_on_zero_temperature_isobar",
                return_value={
                    "success": False,
                    "status": 2,
                    "message": "exact T=0 isobar did not bracket",
                    "muK_A_MeV": np.nan,
                    "A_boundary_exact_local": np.nan,
                    "pressure_residual_norm": np.nan,
                },
                create=True,
            ),
            patch.object(
                self.runner.phase_velocity,
                "_solve_interface_state_from_local_a_E_and_Pi",
            ) as jump_helper,
        ):
            self.runner.add_lte_temperature_fields(
                row,
                T_N_MeV=25.0,
                B14=180.0,
                ms=0.0,
                upB=5000,
            )

        jump_helper.assert_not_called()
        self.assertTrue(np.isnan(row["T_Qstar_LTE_MeV"]))
        self.assertTrue(np.isnan(row["delta_T_interface_MeV"]))
        self.assertAlmostEqual(
            row["beta_LTE"],
            5.0 * row["lambda_n"] * row["u_N"],
        )
        self.assertEqual(
            row["lte_isobar_closure_status"],
            self.runner.LTE_ISOBAR_ROOT_FAILURE,
        )
        self.assertEqual(
            row["lte_jump_closure_status"],
            self.runner.LTE_JUMP_NOT_ATTEMPTED,
        )
        self.assertIn("did not bracket", row["lte_isobar_closure_message"])

    def test_exact_lte_interface_temperature_calls_local_fraction_helper(self):
        row = self._successful_analytic_row()
        expected_E = 1000.0 * 0.02 * np.sqrt(1.0 + 0.02**2)
        expected_Pi = 1000.0 * 0.02**2 + 40.0
        captured = {}

        def fake_local_helper(
            a_target,
            E,
            Pi,
            jB,
            B_one_forth,
            *,
            ms,
            upB,
            initial_guess,
        ):
            captured.update(
                {
                    "a_target": a_target,
                    "E": E,
                    "Pi": Pi,
                    "jB": jB,
                    "B_one_forth": B_one_forth,
                    "ms": ms,
                    "upB": upB,
                    "initial_guess": initial_guess,
                }
            )
            return {
                "muB": 1120.0,
                "muK": 90.0,
                "T": 12.0,
                "nB": 10.0,
                "nK": 4.0,
                "u": 0.3,
                "E": expected_E,
                "Pi": expected_Pi,
            }

        with (
            self._patch_exact_isobar_for_current_fraction(row),
            patch.object(
                self.runner.phase_velocity,
                "_solve_interface_state_from_local_a_E_and_Pi",
                side_effect=fake_local_helper,
            ),
        ):
            enriched = self.runner.add_lte_temperature_fields(
                row,
                T_N_MeV=25.0,
                B14=180.0,
                ms=0.0,
                upB=5000,
            )

        self.assertIs(enriched, row)
        self.assertAlmostEqual(captured["a_target"], 0.4)
        self.assertAlmostEqual(captured["E"], expected_E)
        self.assertAlmostEqual(captured["Pi"], expected_Pi)
        self.assertAlmostEqual(captured["jB"], 3.0)
        self.assertAlmostEqual(captured["B_one_forth"], 180.0)
        self.assertEqual(captured["ms"], 0.0)
        self.assertEqual(captured["upB"], 5000)
        self.assertEqual(row["T_Qstar_LTE_MeV"], 12.0)
        self.assertEqual(row["delta_T_interface_MeV"], 13.0)
        self.assertTrue(row["lte_jump_closure_success"])
        self.assertEqual(row["lte_jump_closure_status"], self.runner.LTE_JUMP_SOLVED)
        self.assertAlmostEqual(row["a_Qstar_LTE_derived"], 0.4)

    def test_exact_lte_interface_temperature_never_calls_obsolete_a_helper(self):
        row = self._successful_analytic_row()
        expected_E = 1000.0 * 0.02 * np.sqrt(1.0 + 0.02**2)
        expected_Pi = 1000.0 * 0.02**2 + 40.0
        state = {
            "muB": 1120.0,
            "muK": 90.0,
            "T": 12.0,
            "nB": 10.0,
            "nK": 4.0,
            "u": 0.3,
            "E": expected_E,
            "Pi": expected_Pi,
        }

        # The obsolete a+E helper has been removed entirely, so the LTE path
        # can no longer call it; assert its absence and exercise the kept helper.
        self.assertFalse(
            hasattr(
                self.runner.phase_velocity,
                "_solve_local_quark_state_from_a_E_and_Pi",
            ),
            "obsolete a+E helper must no longer exist",
        )
        with (
            self._patch_exact_isobar_for_current_fraction(row),
            patch.object(
                self.runner.phase_velocity,
                "_solve_interface_state_from_local_a_E_and_Pi",
                return_value=state,
            ),
        ):
            self.runner.add_lte_temperature_fields(
                row,
                T_N_MeV=25.0,
                B14=180.0,
                ms=0.0,
                upB=5000,
            )

        self.assertTrue(row["lte_jump_closure_success"])

    def test_exact_lte_interface_root_failure_is_marked_and_returns_nan(self):
        row = self._successful_analytic_row()
        with (
            self._patch_exact_isobar_for_current_fraction(row),
            patch.object(
                self.runner.phase_velocity,
                "_solve_interface_state_from_local_a_E_and_Pi",
                side_effect=RuntimeError("no physical local-a interface root"),
            ),
        ):
            self.runner.add_lte_temperature_fields(
                row,
                T_N_MeV=25.0,
                B14=180.0,
                ms=0.0,
                upB=5000,
            )

        self.assertTrue(np.isnan(row["T_Qstar_LTE_MeV"]))
        self.assertTrue(np.isnan(row["delta_T_interface_MeV"]))
        self.assertFalse(row["lte_jump_closure_success"])
        self.assertEqual(
            row["lte_jump_closure_status"],
            self.runner.LTE_JUMP_ROOT_FAILURE,
        )
        self.assertIn(
            "no physical local-a interface root",
            row["lte_jump_closure_message"],
        )
        self.assertAlmostEqual(row["beta_LTE"], 5.0 * 0.75 * 0.02)

    def test_exact_lte_interface_postcheck_rejects_nonlocal_fraction(self):
        row = self._successful_analytic_row()
        expected_E = 1000.0 * 0.02 * np.sqrt(1.0 + 0.02**2)
        expected_Pi = 1000.0 * 0.02**2 + 40.0
        state = {
            "muB": 1120.0,
            "muK": 90.0,
            "T": 12.0,
            "nB": 10.0,
            "nK": 3.0,
            "u": 0.3,
            "E": expected_E,
            "Pi": expected_Pi,
        }

        with (
            self._patch_exact_isobar_for_current_fraction(row),
            patch.object(
                self.runner.phase_velocity,
                "_solve_interface_state_from_local_a_E_and_Pi",
                return_value=state,
            ),
        ):
            self.runner.add_lte_temperature_fields(
                row,
                T_N_MeV=25.0,
                B14=180.0,
                ms=0.0,
                upB=5000,
            )

        self.assertTrue(np.isnan(row["T_Qstar_LTE_MeV"]))
        self.assertFalse(row["lte_jump_closure_success"])
        self.assertEqual(
            row["lte_jump_closure_status"],
            self.runner.LTE_JUMP_POSTCHECK_FAILURE,
        )

    def test_signed_temperature_difference_is_upstream_minus_downstream(self):
        self.assertEqual(
            self.runner.signed_interface_temperature_difference(
                T_N_MeV=25.0,
                T_Qstar_LTE_MeV=20.0,
            ),
            5.0,
        )
        self.assertEqual(
            self.runner.signed_interface_temperature_difference(
                T_N_MeV=15.0,
                T_Qstar_LTE_MeV=20.0,
            ),
            -5.0,
        )

    def test_runner_defaults_to_lte_linear_linear_two_block_scan(self):
        args = self.runner.build_parser().parse_args([])
        self.assertEqual(
            args.run_tag,
            "lte_temperature_jump_two_block_linear_beta5_exact_local_a_v3",
        )
        self.assertEqual(args.interface_fraction_mode, "LTE")
        self.assertEqual(args.temperature_mesh, "adaptive_linear")
        self.assertEqual(args.split_temperature, 80.0)
        self.assertEqual(args.t_min, 1.0e-2)
        self.assertEqual(args.t_max, 100.0)
        self.assertEqual(args.t_count, 30)
        self.assertEqual(args.high_t_count, 12)
        self.assertEqual(args.m_min, 0.0)
        self.assertEqual(args.m_max, 0.9)
        self.assertEqual(args.m_count, 20)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.runner.build_parser().parse_args(["--temperature-mesh", "adaptive_log"])

    def test_two_blocks_use_linear_axes_and_share_80_MeV_seam(self):
        low = self.runner.build_low_temperature_axis(
            temperature_mesh="adaptive_linear",
            temperature_min=0.01,
            split_temperature=80.0,
            temperature_count=5,
        )
        high = self.runner.build_high_temperature_column(
            temperature_mesh="adaptive_linear",
            split_temperature=80.0,
            A_temperature_boundary=95.0,
            boundary_eps=0.005,
            temperature_count=4,
        )
        np.testing.assert_allclose(np.diff(low), np.diff(low)[0])
        np.testing.assert_allclose(np.diff(high), np.diff(high)[0])
        self.assertEqual(low[-1], 80.0)
        self.assertEqual(high[0], 80.0)
        self.assertLess(high[-1], 95.0)

    def test_successful_row_gets_lte_temperature_beta_and_signed_jump(self):
        row = self._successful_analytic_row()
        expected_E = 1000.0 * 0.02 * np.sqrt(1.0 + 0.02**2)
        expected_Pi = 1000.0 * 0.02**2 + 40.0
        state = {
            "muB": 1120.0,
            "muK": 90.0,
            "T": 12.0,
            "nB": 10.0,
            "nK": 4.0,
            "u": 0.3,
            "E": expected_E,
            "Pi": expected_Pi,
        }
        with (
            self._patch_exact_isobar_for_current_fraction(row),
            patch.object(
                self.runner.phase_velocity,
                "_solve_interface_state_from_local_a_E_and_Pi",
                return_value=state,
            ),
        ):
            enriched = self.runner.add_lte_temperature_fields(
                row,
                T_N_MeV=25.0,
                B14=180.0,
                ms=0.0,
                upB=5000,
            )
        self.assertIs(enriched, row)
        self.assertAlmostEqual(row["T_Qstar_LTE_MeV"], 12.0)
        self.assertAlmostEqual(row["delta_T_interface_MeV"], 13.0)
        self.assertAlmostEqual(row["beta_LTE"], 5.0 * 0.75 * 0.02)

    def test_payload_identifies_temperature_jump_and_exact_beta_formula(self):
        args = self.runner.build_parser().parse_args([])
        arrays = {
            "low_temperature_grid": np.array([[80.0]]),
            "low_metastability_grid": np.array([[0.2]]),
            "low_delta_T_interface_MeV_grid": np.array([[1.0]]),
            "low_T_Qstar_LTE_MeV_grid": np.array([[79.0]]),
            "low_beta_LTE_grid": np.array([[1.0e-5]]),
            "low_A_boundary_exact_local_grid": np.array([[0.7]]),
            "low_muK_A_exact_local_MeV_grid": np.array([[75.0]]),
            "low_lte_isobar_pressure_residual_norm_grid": np.array([[0.0]]),
            "low_lte_isobar_closure_status_grid": np.array([[0]], dtype=int),
            "low_lte_jump_closure_status_grid": np.array([[0]], dtype=int),
            "low_lte_jump_a_residual_grid": np.array([[0.0]]),
            "low_lte_jump_energy_residual_norm_grid": np.array([[0.0]]),
            "low_lte_jump_momentum_residual_norm_grid": np.array([[0.0]]),
            "low_muB_Qstar_LTE_MeV_grid": np.array([[1100.0]]),
            "low_muK_Qstar_LTE_MeV_grid": np.array([[80.0]]),
            "high_temperature_grid": np.array([[80.0]]),
            "high_metastability_grid": np.array([[0.2]]),
            "high_delta_T_interface_MeV_grid": np.array([[1.0]]),
            "high_T_Qstar_LTE_MeV_grid": np.array([[79.0]]),
            "high_beta_LTE_grid": np.array([[1.0e-5]]),
            "high_A_boundary_exact_local_grid": np.array([[0.7]]),
            "high_muK_A_exact_local_MeV_grid": np.array([[75.0]]),
            "high_lte_isobar_pressure_residual_norm_grid": np.array([[0.0]]),
            "high_lte_isobar_closure_status_grid": np.array([[0]], dtype=int),
            "high_lte_jump_closure_status_grid": np.array([[0]], dtype=int),
            "high_lte_jump_a_residual_grid": np.array([[0.0]]),
            "high_lte_jump_energy_residual_norm_grid": np.array([[0.0]]),
            "high_lte_jump_momentum_residual_norm_grid": np.array([[0.0]]),
            "high_muB_Qstar_LTE_MeV_grid": np.array([[1100.0]]),
            "high_muK_Qstar_LTE_MeV_grid": np.array([[80.0]]),
        }
        payload = self.runner._payload(
            args=args,
            calibration=(189.0, 1190.0, 42.0, 1.0, 3.0),
            arrays=arrays,
            records=[],
            A_boundary_records=[],
            gap_boundary_records=[],
        )
        self.assertEqual(payload["primary_observable"], "delta_T_interface_MeV")
        self.assertEqual(
            payload["temperature_jump_definition"],
            "delta_T_interface_MeV = T_N_MeV - T_Qstar_LTE_MeV",
        )
        self.assertEqual(
            payload["beta_LTE_definition"],
            "beta_LTE = 5*lambda_n*u_N",
        )
        self.assertEqual(
            payload["A_boundary_exact_local_definition"],
            (
                "nK/nB at fixed muB_Q and T=0 where "
                "PQM(muB_Q,muK,B14,0)=P_Q"
            ),
        )
        self.assertEqual(
            payload["A_boundary_grid_definition"],
            "analytic velocity-closure boundary retained for mesh compatibility",
        )
        self.assertEqual(
            payload["T_Qstar_LTE_definition"],
            "exact local-a interface solve of baryon, energy, and momentum jumps",
        )
        self.assertEqual(
            payload["interface_state_helper"],
            "phase_velocity._solve_interface_state_from_local_a_E_and_Pi",
        )
        self.assertEqual(
            payload["jump_failure_policy"],
            "mark closure status and return NaN temperature",
        )
        self.assertEqual(payload["T_Qstar_LTE_bound_type"], "lower_bound")
        self.assertEqual(
            payload["delta_T_interface_bound_type"],
            "upper_bound_at_fixed_T_N",
        )
        self.assertIn("low_delta_T_interface_MeV_grid", payload)
        self.assertIn("high_delta_T_interface_MeV_grid", payload)
        self.assertIn("low_lte_jump_closure_status_grid", payload)
        self.assertIn("high_lte_jump_closure_status_grid", payload)
        self.assertIn("low_A_boundary_exact_local_grid", payload)
        self.assertIn("high_A_boundary_exact_local_grid", payload)
        self.assertIn("low_lte_isobar_closure_status_grid", payload)
        self.assertIn("high_lte_isobar_closure_status_grid", payload)

    def test_runner_uses_analytic_jump_solver_not_numerical_front_bvp(self):
        source = inspect.getsource(self.runner)
        self.assertIn("analytic_velocity_bound", source)
        self.assertIn("_solve_interface_state_from_local_a_E_and_Pi", source)
        self.assertNotIn("_solve_local_quark_state_from_a_E_and_Pi(", source)
        self.assertNotIn("solve_front_energy_conserving_uNmax", source)

    def test_copy_low_seam_to_high_copies_every_point_field_exactly(self):
        arrays = {}
        field_names = (
            "delta_T_interface_MeV_grid",
            "T_Qstar_LTE_MeV_grid",
            "beta_LTE_grid",
            "u_N_grid",
            "A_boundary_exact_local_grid",
            "muK_A_exact_local_MeV_grid",
            "lte_isobar_pressure_residual_norm_grid",
            "lte_jump_a_residual_grid",
            "lte_jump_energy_residual_norm_grid",
            "lte_jump_momentum_residual_norm_grid",
            "muB_Qstar_LTE_MeV_grid",
            "muK_Qstar_LTE_MeV_grid",
        )
        for offset, name in enumerate(("status_grid", *field_names)):
            arrays[f"low_{name}"] = np.array(
                [[-1.0, -2.0], [10.0 + offset, 20.0 + offset]]
            )
            arrays[f"high_{name}"] = np.full((3, 2), np.nan)
        arrays["low_lte_jump_closure_status_grid"] = np.array(
            [[1, 1], [0, 2]],
            dtype=int,
        )
        arrays["high_lte_jump_closure_status_grid"] = np.full(
            (3, 2),
            self.runner.LTE_JUMP_NOT_ATTEMPTED,
            dtype=int,
        )
        arrays["low_lte_isobar_closure_status_grid"] = np.array(
            [[1, 1], [0, 2]],
            dtype=int,
        )
        arrays["high_lte_isobar_closure_status_grid"] = np.full(
            (3, 2),
            self.runner.LTE_ISOBAR_NOT_ATTEMPTED,
            dtype=int,
        )
        self.runner.copy_low_seam_to_high(
            arrays=arrays,
            low_seam_row=1,
            high_columns=[0, 1],
            field_names=field_names,
            integer_field_names=(
                "lte_isobar_closure_status_grid",
                "lte_jump_closure_status_grid",
            ),
        )
        for name in ("status_grid", *field_names):
            np.testing.assert_array_equal(
                arrays[f"high_{name}"][0],
                arrays[f"low_{name}"][1],
            )
        np.testing.assert_array_equal(
            arrays["high_lte_jump_closure_status_grid"][0],
            arrays["low_lte_jump_closure_status_grid"][1],
        )
        np.testing.assert_array_equal(
            arrays["high_lte_isobar_closure_status_grid"][0],
            arrays["low_lte_isobar_closure_status_grid"][1],
        )

if __name__ == "__main__":
    unittest.main()
