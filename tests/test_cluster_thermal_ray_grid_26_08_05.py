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
    / "26-08-05"
    / "run_thermal_conducting_uN_ray_grid.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_thermal_conducting_uN_ray_grid_26_08_05",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ThermalConductingRayGridRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_defaults_use_exact_zero_interface_and_600_points(self):
        args = self.runner.build_parser().parse_args([])
        self.assertEqual(args.T_0plus, 0.0)
        self.assertEqual(args.theta_count, 30)
        self.assertEqual(args.radial_count, 20)
        self.assertEqual(
            args.output,
            "thermal-conducting-uN-T-nB-ray-grid.npy",
        )
        self.assertEqual(
            self.runner.NUMERICAL_CLOSURE_LABEL,
            "thermal_conducting_nK_jK_T",
        )

    def test_T_0plus_cli_sets_interface_temperature(self):
        args = self.runner.build_parser().parse_args(
            ["--T_0plus", "0.25"]
        )
        self.assertEqual(args.T_0plus, 0.25)

    def test_point_adapter_calls_thermal_solver_and_stores_u0minus(self):
        task = self.runner.NumericalRayPointTask(
            row_index=2,
            col_index=3,
            theta_rad=0.5,
            rho=0.7,
            radial_fraction=0.5,
            nB_over_n0=2.5,
            T_0minus=10.0,
            n0=2.0,
            B14=189.0,
            ms=0.0,
            nm_type="PNM",
            upB=5000,
            T_0plus=0.0,
            tail_eps=1.0e-8,
            n_mesh=40,
            tol_bvp=1.0e-3,
            max_nodes=1000,
            jB_guess=2.5,
            jB_bounds=None,
            auto_jB_bounds=True,
            shooting_u_max=1.0e-4,
            shooting_jB_lower=1.0e-12,
            point_timeout=1.0,
            jB_guess_source="analytic_20_seed",
            continuation_direction="seed",
        )
        solved = {
            "success": True,
            "message": "converged",
            "uN": 1.0e-6,
            "jB": 1.0,
            "muB_inf": 1050.0,
            "T_inf": 20.0,
            "nB_inf": 6.0,
            "nK_inf": 0.3,
            "lambda_n": 5.0 / 6.0,
            "a_0plus": 0.5,
            "nK_0plus": 2.5,
            "T_0plus": 0.0,
            "Tprime_0plus": 0.2,
            "q_th_0plus": -3.0,
            "F_E": 4.0,
            "solver_variant": "thermal_conducting_nK_jK_T",
            "conductivity_model": "heiselberg_low_y",
            "bvp_status": 0,
            "bvp_message": "converged",
            "bvp_iterations": 4,
            "bvp_nodes": 80,
            "stats": {
                "bvp_ode_calls": 20,
                "bvp_bc_calls": 8,
                "local_root_calls": 50,
            },
        }
        with mock.patch.object(
            self.runner.phase_velocity,
            "solve_front_thermal_conducting",
            return_value=solved,
        ) as calculate:
            row = self.runner.calculate_numerical_point(task)

        self.assertTrue(row["success"])
        self.assertEqual(row["u_0minus"], 1.0e-6)
        self.assertAlmostEqual(
            row["u_0minus_m_s"],
            1.0e-6 * self.runner.const.m_per_sec,
        )
        self.assertEqual(row["bvp_parameter_count"], 1)
        self.assertEqual(row["T_0plus_target_MeV"], 0.0)
        self.assertEqual(row["Tprime_0plus"], 0.2)
        self.assertEqual(row["q_th_0plus"], -3.0)
        self.assertEqual(row["conductivity_model"], "heiselberg_low_y")
        self.assertEqual(calculate.call_args.kwargs["T_0plus"], 0.0)
        self.assertEqual(calculate.call_args.kwargs["jB_guess"], 2.5)
        self.assertEqual(
            calculate.call_args.kwargs["jB_bounds"],
            (1.0e-12, 5.0e-4),
        )

    def test_successful_row_populates_contour_arrays(self):
        mesh = {
            "nB_grid_over_n0": np.array([[2.0]]),
            "temperature_grid_MeV": np.array([[10.0]]),
        }
        arrays = self.runner._empty_result_arrays(mesh, n0=3.0)
        self.runner._record_result(
            arrays,
            {
                "row_index": 0,
                "col_index": 0,
                "success": True,
                "u_0minus": 2.0e-6,
                "u_0minus_m_s": 600.0,
                "jB": 4.0,
                "lambda_n": 0.8,
                "A_boundary": np.nan,
                "a_local_0plus": 0.4,
                "entropy_jump_per_baryon": np.nan,
                "point_elapsed_s": 1.0,
            },
        )
        self.assertEqual(arrays["velocity_grid_m_s"][0, 0], 600.0)
        self.assertEqual(arrays["u_0minus_grid"][0, 0], 2.0e-6)
        self.assertEqual(arrays["jB_grid"][0, 0], 4.0)
        self.assertEqual(
            arrays["status_grid"][0, 0],
            self.runner.STATUS_FINITE,
        )

    def test_payload_keeps_contour_contract_and_thermal_metadata(self):
        args = self.runner.build_parser().parse_args([])
        arrays = {
            "theta_axis_rad": np.array([0.5]),
            "nB_grid_over_n0": np.array([[2.0]]),
            "temperature_grid_MeV": np.array([[10.0]]),
            "velocity_grid_m_s": np.array([[300.0]]),
            "u_0minus_grid": np.array([[1.0e-6]]),
            "seed_rho_by_theta": np.array([0.7]),
            "seed_radial_index_by_theta": np.array([0]),
            "seed_analytic_jB_by_theta": np.array([3.0]),
        }
        payload = self.runner.build_numerical_payload(
            args=args,
            transform=self.runner.RayTransform(1.0, 4.5, 120.0),
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
            "thermal_conducting_nK_jK_T",
        )
        self.assertEqual(payload["T_0plus_target_MeV"], 0.0)
        self.assertEqual(
            payload["energy_closure"],
            "F_E = h*gamma*u - kappa_th*dT/dx",
        )
        self.assertEqual(payload["neutrino_source"], "Q_nu=0")
        np.testing.assert_array_equal(
            payload["velocity_grid_m_s"],
            arrays["velocity_grid_m_s"],
        )


if __name__ == "__main__":
    unittest.main()
