import argparse
import importlib.util
import queue
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "new_paper_calculations" / "26-08-07"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "run_T0plus_method_benchmark_test",
    SCRIPT_DIR / "run_T0plus_method_benchmark.py",
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def successful_numerical_result(jB):
    axis = np.array([0.0, 1.0])
    return {
        "success": True,
        "message": "ok",
        "u_0minus": 1.0e-6,
        "jB": float(jB),
        "a_0plus": 0.4,
        "a_inf": 0.0,
        "lambda_n": 0.8,
        "T_inf": 10.0,
        "bvp_status": 0,
        "bvp_message": "converged",
        "bvp_nodes": 12,
        "bvp_niter": 2,
        "bvp_parameters": np.array([float(jB)]),
        "bvp_max_rms_residual": 1.0e-6,
        "bvp_collocation_converged": True,
        "E_right_residual_norm": 1.0e-8,
        "Pi_right_residual_norm": 1.0e-8,
        "tail_residual_norm": 1.0e-8,
        "kaon_equation_residual_norm": 1.0e-6,
        "energy_equation_residual_norm": 1.0e-8,
        "momentum_equation_residual_norm": 1.0e-8,
        "T_0plus_residual": 0.0,
        "solver_variant": benchmark.SOLVER_VARIANT,
        "_root_method": benchmark.ROOT_METHOD,
        "x": axis,
        "s_coord": axis,
        "nK": axis + 1.0,
        "jK": axis + 2.0,
        "nB": axis + 3.0,
        "a": axis + 0.1,
        "u": axis + 0.2,
        "muB": axis + 900.0,
        "muK": axis + 10.0,
        "T_profile": axis + 0.5,
    }


def numerical_task(T_values=(4.0, 2.0, 1.0)):
    return {
        "point_index": 0,
        "sweep_direction": "downward",
        "T_0minus_MeV": 20.0,
        "nB_0minus_MeV3": 10.0,
        "B14_MeV": 189.0,
        "ms_MeV": 0.0,
        "nm_type": "PNM",
        "T_0plus_values": tuple(T_values),
        "semi_jB_by_T": {float(T): 2.0 + float(T) for T in T_values},
        "tail_eps": 1.0e-8,
        "n_mesh": 20,
        "tol_bvp": 1.0e-4,
        "max_nodes": 100,
        "jB_lower_bound": 1.0e-12,
        "jB_upper_factor": 1.0e-4,
        "point_timeout_s": 10.0,
    }


class BenchmarkGeometryTests(unittest.TestCase):
    def test_boundary_defaults_restore_26_08_02_refinement(self):
        self.assertEqual(benchmark.DEFAULT_RAY_PROBE_COUNT, 15)
        self.assertEqual(benchmark.DEFAULT_BISECTION_STEPS, 10)

    def test_semi_boundary_geometry_uses_only_the_semi_domain(self):
        semi = {
            "theta_axis_rad": np.array([0.2, 0.6, 1.0]),
            "rho_inner": np.array([1.0, 1.5, 2.0]),
            "rho_outer": np.array([4.0, 5.0, 6.0]),
        }
        geometry = benchmark.semi_boundary_geometry(
            semi,
            ray_count=3,
            angular_buffer_fraction=0.1,
        )
        np.testing.assert_allclose(geometry["theta_axis_rad"], [0.28, 0.6, 0.92])
        np.testing.assert_allclose(geometry["rho_inner"], [1.1, 1.5, 1.9])
        np.testing.assert_allclose(geometry["rho_outer"], [4.2, 5.0, 5.8])
        self.assertEqual(
            geometry["selection_rule"],
            "semi_analytic_safe_domain",
        )

    def test_selected_analytical_worker_calls_only_its_assigned_method(self):
        task = {
            "point_index": 0,
            "method": "semi_analytic",
            "T_0minus_MeV": 20.0,
            "nB_0minus_MeV3": 10.0,
            "T_0plus_values": (4.0, 2.0),
            "B14_MeV": 189.0,
            "xi": -0.5,
            "ms_MeV": 0.0,
            "nm_type": "PNM",
            "upB": 5000,
            "density_match_rtol": 1.0e-6,
        }
        semi_result = {
            "u_0minus": 2.0e-6,
            "jB": 3.0,
            "a_0plus": 0.4,
            "lambda_n": 0.8,
            "T_inf": 30.0,
            "A_boundary": 0.9,
        }
        with mock.patch.object(
            benchmark, "upstream_muB", return_value=(1100.0, 1.0e-10)
        ), mock.patch.object(
            benchmark.phase_velocity, "analytic_velocity_bound"
        ) as full, mock.patch.object(
            benchmark.phase_velocity,
            "semi_analytic_velocity_bound",
            return_value=semi_result,
        ) as semi:
            output = benchmark.evaluate_selected_analytical_point(task)
        full.assert_not_called()
        self.assertEqual(semi.call_count, 2)
        self.assertEqual(output["method"], "semi_analytic")
        self.assertTrue(all(row["success"] for row in output["rows"]))

    def test_manifest_validation_requires_semi_analytic_safe_domain(self):
        states = [
            {
                "T_0plus_MeV": value,
                "T_0minus_MeV": 20.0,
                "nB_0minus_over_n0": 3.0,
                "rho": 1.5,
                "rho_inner_boundary": 1.0,
                "rho_outer_boundary": 2.0,
            }
            for value in (4.0, 2.0)
        ]
        manifest = {
            "manifest_type": benchmark.DOMAIN_MANIFEST_TYPE,
            "selection_rule": "semi_analytic_safe_domain",
            "points": [{"point_index": 0, "states": states}],
            "T_0plus_values_MeV": np.array([4.0, 2.0]),
            "calibration": {"n0_MeV3": 10.0, "B14_MeV": 189.0},
        }
        self.assertIs(benchmark.validate_domain_manifest(manifest), manifest)
        broken = dict(manifest, selection_rule="intersection")
        with self.assertRaisesRegex(ValueError, "semi-analytical"):
            benchmark.validate_domain_manifest(broken)

    def test_boundary_engine_dispatches_the_requested_velocity_method(self):
        calls = []

        def result(label):
            calls.append(label)
            return {
                "A_boundary": 0.8,
                "muB_inf": 1000.0,
                "T_inf": 20.0,
                "velocity_closure": label,
            }

        with mock.patch.object(
            benchmark.ray_grid, "muB_from_nB", return_value=1100.0
        ), mock.patch.object(
            benchmark.ray_grid.phase_velocity,
            "analytic_velocity_bound",
            side_effect=lambda *args, **kwargs: result("closed_form"),
        ), mock.patch.object(
            benchmark.ray_grid.phase_velocity,
            "semi_analytic_velocity_bound",
            side_effect=lambda *args, **kwargs: result("numerical_I2"),
        ):
            for closure in ("closed_form", "numerical_I2"):
                row = benchmark.ray_grid.evaluate_A_state(
                    nB_over_n0=3.0,
                    temperature_MeV=20.0,
                    n0=1.0,
                    B14=189.0,
                    xi=-0.5,
                    ms=0.0,
                    nm_type="PNM",
                    upB=5000,
                    interface_fraction_mode="fixed_T_0plus",
                    velocity_closure=closure,
                )
                self.assertEqual(row["status"], "success")
        self.assertEqual(calls, ["closed_form", "numerical_I2"])

    def test_boundary_reference_temperature_is_passed_explicitly(self):
        result = {
            "A_boundary": 0.8,
            "muB_inf": 1000.0,
            "T_inf": 20.0,
            "velocity_closure": "numerical_I2",
        }
        with mock.patch.object(
            benchmark.ray_grid, "muB_from_nB", return_value=1100.0
        ), mock.patch.object(
            benchmark.ray_grid.phase_velocity,
            "semi_analytic_velocity_bound",
            return_value=result,
        ) as semi:
            benchmark.ray_grid.evaluate_A_state(
                nB_over_n0=3.0,
                temperature_MeV=20.0,
                n0=1.0,
                B14=189.0,
                xi=-0.5,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
                interface_fraction_mode="fixed_T_0plus",
                velocity_closure="numerical_I2",
                T_0plus_MeV=0.01,
            )
        self.assertEqual(semi.call_args.kwargs["T_0plus"], 0.01)

    def test_extreme_endothermic_A_state_uses_cold_numerical_I2_closure(self):
        result = {
            "A_boundary": 0.8,
            "muB_inf": 1000.0,
            "T_inf": 20.0,
            "velocity_closure": "numerical_I2",
        }
        with mock.patch.object(
            benchmark.ray_grid, "muB_from_nB", return_value=1100.0
        ), mock.patch.object(
            benchmark.ray_grid,
            "semi_analytic_extreme_endothermic_bound",
            return_value=result,
        ) as cold_bound, mock.patch.object(
            benchmark.ray_grid.phase_velocity, "semi_analytic_velocity_bound"
        ) as finite_temperature_bound:
            row = benchmark.ray_grid.evaluate_A_state(
                nB_over_n0=3.0,
                temperature_MeV=20.0,
                n0=1.0,
                B14=189.0,
                xi=-0.5,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
                interface_fraction_mode="extreme_endothermic",
                velocity_closure="numerical_I2",
                T_0plus_MeV=0.0,
            )
        self.assertEqual(row["status"], "success")
        self.assertEqual(cold_bound.call_count, 1)
        finite_temperature_bound.assert_not_called()

    def test_reachability_edge_is_a_valid_semi_analytic_outer_boundary(self):
        def fake_state(**kwargs):
            density = float(kwargs["nB_over_n0"])
            if density < 5.5:
                return {
                    "status": "success",
                    "A_boundary": 0.9 + 0.01 * density,
                    "endpoint_solution": (1000.0, 20.0),
                    "muB_0minus_MeV": 1100.0,
                    "message": "",
                }
            return {
                "status": "interface_temperature_unreachable",
                "A_boundary": np.nan,
                "endpoint_solution": None,
                "muB_0minus_MeV": 1100.0,
                "message": "T_0plus lies below the trajectory",
            }

        with mock.patch.object(
            benchmark.ray_grid, "evaluate_A_state", side_effect=fake_state
        ):
            record = benchmark.ray_grid.solve_A_transition_on_path(
                coordinates=np.array([4.0, 5.0, 6.0]),
                physical_coordinates=lambda value: (value, 20.0),
                n0=1.0,
                B14=189.0,
                xi=-0.5,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
                interface_fraction_mode="fixed_T_0plus",
                velocity_closure="numerical_I2",
                label="test",
                bisection_steps=3,
            )
        self.assertEqual(record["boundary_kind"], "T_0plus_reachability_edge")
        self.assertEqual(record["outer_forbidden_status"], "interface_temperature_unreachable")

    def test_ray_A_solver_recovers_success_interval_between_coarse_failures(self):
        def fake_state(**kwargs):
            density = float(kwargs["nB_over_n0"])
            if 3.4 < density < 3.7:
                return {
                    "status": "success",
                    "A_boundary": 0.9,
                    "endpoint_solution": (1000.0, 20.0),
                    "muB_0minus_MeV": 1100.0,
                    "message": "",
                }
            status = (
                "interface_temperature_unreachable"
                if density >= 3.7 else "gap_forbidden"
            )
            return {
                "status": status,
                "A_boundary": np.nan,
                "endpoint_solution": None,
                "muB_0minus_MeV": 1100.0,
                "message": status,
            }

        with mock.patch.object(
            benchmark.ray_grid,
            "evaluate_A_state",
            side_effect=fake_state,
        ):
            record = benchmark.ray_grid.solve_A_transition_on_path(
                coordinates=np.array([3.25, 3.8125]),
                physical_coordinates=lambda value: (value, 0.01),
                n0=1.0,
                B14=189.0,
                xi=-0.5,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
                interface_fraction_mode="fixed_T_0plus",
                velocity_closure="numerical_I2",
                T_0plus_MeV=0.01,
                label="test",
                bisection_steps=2,
            )
        self.assertEqual(record["boundary_kind"], "T_0plus_reachability_edge")
        self.assertGreater(record["coordinate"], 3.4)
        self.assertLess(record["coordinate"], 3.8)
        self.assertEqual(record["bisection_steps"], 2)
        self.assertLess(record["bracket_lower_coordinate"], record["bracket_upper_coordinate"])
        self.assertIn("success", record["probe_status_counts"])

    def test_endpoint_recovery_finds_admissible_interval_skipped_by_coarse_grid(self):
        rows_by_edge = {
            "A_bottom_endpoint": [
                {
                    "coordinate": 3.25,
                    "nB_over_n0": 3.25,
                    "T_MeV": 0.01,
                    "status": "numerical_failure",
                    "muB_0minus_MeV": 1000.0,
                },
                {
                    "coordinate": 3.8125,
                    "nB_over_n0": 3.8125,
                    "T_MeV": 0.01,
                    "status": "interface_temperature_unreachable",
                    "muB_0minus_MeV": 1050.0,
                },
            ],
            "A_left_endpoint": [
                {
                    "coordinate": 0.01,
                    "nB_over_n0": 1.0,
                    "T_MeV": 0.01,
                    "status": "success",
                    "A_boundary": 0.5,
                }
            ],
        }
        base = {
            "edge_kind": "A_bottom_endpoint",
            "coordinate": 3.25,
            "nB_over_n0": 3.25,
            "T_MeV": 0.01,
            "phase": "coarse",
        }

        def fake_probe(task):
            coordinate = float(task["coordinate"])
            status = (
                "success"
                if 3.4 < coordinate < 3.7
                else (
                    "interface_temperature_unreachable"
                    if coordinate >= 3.7
                    else "gap_forbidden"
                )
            )
            return {
                "edge_kind": task["edge_kind"],
                "phase": task["phase"],
                "record": {
                    "coordinate": coordinate,
                    "nB_over_n0": coordinate,
                    "T_MeV": 0.01,
                    "status": status,
                    "A_boundary": 0.9 if status == "success" else np.nan,
                    "endpoint_solution": (1000.0, 10.0)
                    if status == "success" else None,
                    "muB_0minus_MeV": 1000.0,
                },
            }

        with mock.patch.object(
            benchmark.ray_grid,
            "_solve_endpoint_probe_task",
            side_effect=fake_probe,
        ), ThreadPoolExecutor(max_workers=3) as executor:
            benchmark.ray_grid._recover_missing_A_endpoint_successes(
                rows_by_edge=rows_by_edge,
                base_tasks_by_edge={
                    "A_bottom_endpoint": base,
                    "A_left_endpoint": {
                        **base,
                        "edge_kind": "A_left_endpoint",
                    },
                },
                executor=executor,
                progress_factory=None,
            )

        state, record = benchmark.ray_grid._A_endpoint_state_or_record(
            "A_bottom_endpoint",
            rows_by_edge["A_bottom_endpoint"],
            bisection_steps=3,
        )
        self.assertIsNotNone(state)
        self.assertIsNone(record)
        self.assertEqual(state["lower"]["status"], "success")
        self.assertEqual(
            state["upper"]["status"],
            "interface_temperature_unreachable",
        )

    def test_gap_endpoint_refinement_uses_requested_bisection_count(self):
        rows = [
            {"coordinate": 1.0, "status": "success", "gap_MeV": -1.0},
            {"coordinate": 2.0, "status": "success", "gap_MeV": 1.0},
        ]
        state = benchmark.ray_grid._gap_endpoint_state(
            "gap_bottom_endpoint", rows, bisection_steps=3
        )
        self.assertEqual(state["remaining"], 3)

    def test_common_boundary_uses_intersection(self):
        full = {
            "theta_axis_rad": np.array([0.2, 0.8]),
            "rho_inner": np.array([1.0, 2.0]),
            "rho_outer": np.array([5.0, 6.0]),
        }
        semi = {
            "theta_axis_rad": np.array([0.3, 0.7]),
            "rho_inner": np.array([1.5, 2.5]),
            "rho_outer": np.array([4.0, 7.0]),
        }
        common = benchmark.common_boundary_geometry(
            full,
            semi,
            ray_count=2,
            angular_buffer_fraction=0.0,
        )
        np.testing.assert_allclose(common["theta_axis_rad"], [0.3, 0.7])
        np.testing.assert_allclose(common["rho_outer"], [4.0, 5.833333333333333])
        self.assertTrue(np.all(common["rho_inner"] < common["rho_outer"]))

    def test_default_selection_is_four_rays_by_three_fractions(self):
        transform = benchmark.ray_grid.RayTransform(1.0, 9.0, 200.0)
        points = benchmark.select_benchmark_points(
            theta_axis_rad=np.linspace(0.2, 1.2, 4),
            radial_fractions=np.array([0.2, 0.5, 0.8]),
            rho_inner=np.ones(4),
            rho_outer=np.full(4, 2.0),
            transform=transform,
            radial_buffer_fraction=0.02,
        )
        self.assertEqual(len(points), 12)
        self.assertTrue(all(p["rho_inner_boundary"] < p["rho"] for p in points))
        self.assertTrue(all(p["rho"] < p["rho_outer_boundary"] for p in points))


class ContinuationTests(unittest.TestCase):
    def test_continuation_uses_the_temperature_indexed_upstream_state(self):
        calls = []

        def fake_solver(**kwargs):
            calls.append(
                (
                    float(kwargs["T_0plus"]),
                    float(kwargs["T_0minus"]),
                    float(kwargs["nB_0minus"]),
                    tuple(float(value) for value in kwargs["jB_bounds"]),
                )
            )
            return successful_numerical_result(kwargs.get("jB_guess", 1.0))

        task = numerical_task(T_values=(4.0, 2.0))
        task["states_by_T0plus"] = {
            4.0: {
                "T_0minus_MeV": 20.0,
                "nB_0minus_MeV3": 10.0,
                "nB_0minus_over_n0": 3.0,
            },
            2.0: {
                "T_0minus_MeV": 25.0,
                "nB_0minus_MeV3": 12.0,
                "nB_0minus_over_n0": 3.5,
            },
        }
        with mock.patch.object(
            benchmark.phase_velocity,
            "solve_front_energy_conserving_uNmax",
            side_effect=fake_solver,
        ):
            result = benchmark.run_numerical_trajectory(task)

        self.assertEqual([call[:3] for call in calls], [(4.0, 20.0, 10.0), (2.0, 25.0, 12.0)])
        self.assertAlmostEqual(calls[0][3][1], 10.0e-4)
        self.assertAlmostEqual(calls[1][3][1], 12.0e-4)
        successful = [row for row in result["records"] if row["success"]]
        self.assertEqual(successful[1]["T_0minus_MeV"], 25.0)
        self.assertEqual(successful[1]["nB_0minus_MeV3"], 12.0)

    def test_continuation_failure_retries_with_current_semi_seed(self):
        calls = []

        def fake_solver(**kwargs):
            T = float(kwargs["T_0plus"])
            uses_profile = "continuation_guess" in kwargs
            calls.append((T, uses_profile, kwargs.get("jB_guess")))
            if T == 2.0 and uses_profile:
                raise RuntimeError("continuation failed")
            return successful_numerical_result(kwargs.get("jB_guess", 1.0))

        progress = queue.Queue()
        task = numerical_task()
        task["progress_queue"] = progress
        with mock.patch.object(
            benchmark.phase_velocity,
            "solve_front_energy_conserving_uNmax",
            side_effect=fake_solver,
        ):
            result = benchmark.run_numerical_trajectory(task)

        self.assertEqual([row["success"] for row in result["records"]], [True, False, True, True])
        self.assertEqual(calls[0][:2], (4.0, False))
        self.assertEqual(calls[1][:2], (2.0, True))
        self.assertEqual(calls[2], (2.0, False, 4.0))
        self.assertEqual(calls[3][:2], (1.0, True))
        events = []
        while not progress.empty():
            events.append(progress.get_nowait())
        self.assertEqual(sum(e["event"] == "attempt_completed" for e in events), 4)
        self.assertEqual(sum(e["event"] == "retry_scheduled" for e in events), 1)

    def test_failed_adjacent_temperature_blocks_distant_profile(self):
        calls = []

        def fake_solver(**kwargs):
            T = float(kwargs["T_0plus"])
            uses_profile = "continuation_guess" in kwargs
            calls.append((T, uses_profile))
            if T == 2.0:
                raise RuntimeError("both attempts fail")
            return successful_numerical_result(kwargs.get("jB_guess", 1.0))

        with mock.patch.object(
            benchmark.phase_velocity,
            "solve_front_energy_conserving_uNmax",
            side_effect=fake_solver,
        ):
            benchmark.run_numerical_trajectory(numerical_task())
        calls_at_one = [uses_profile for T, uses_profile in calls if T == 1.0]
        self.assertEqual(calls_at_one, [False])


class PayloadTests(unittest.TestCase):
    def test_payload_uses_canonical_schema_and_keeps_profiles(self):
        args = argparse.Namespace(
            run_tag="test",
            xi=-0.5,
            ms=0.0,
            nm_type="PNM",
            upB=5000,
            boundary_T0plus=np.nan,
        )
        point = {
            "point_index": 0,
            "theta_rad": 0.5,
            "theta_deg": 28.0,
            "radial_fraction": 0.5,
            "radial_fraction_used": 0.5,
            "rho": 1.5,
            "rho_inner_boundary": 1.0,
            "rho_outer_boundary": 2.0,
            "rho_outer_boundary_full_analytic": 2.2,
            "rho_outer_boundary_semi_analytic": 2.0,
            "nB_0minus_over_n0": 3.0,
            "T_0minus_MeV": 20.0,
        }
        analytical = [{
            "point_index": 0,
            "T_0plus_MeV": 4.0,
            "muB_0minus_MeV": 1000.0,
            "density_relative_error": 1.0e-10,
            "analytic": {"success": True, "u_0minus": 1.0e-6},
            "semi_analytic": {"success": True, "u_0minus": 2.0e-6},
        }]
        record = {
            "point_index": 0,
            "sweep_direction": "downward",
            "T_0plus_MeV": 4.0,
            "attempt_index": 0,
            "guess_source": benchmark.GUESS_SOURCE_SEMI,
            "success": True,
            "status": "success",
            "u_0minus": 3.0e-6,
        }
        payload = benchmark.build_payload(
            args=args,
            calibration={"n0_MeV3": 10.0, "B14_MeV": 189.0},
            points=[point],
            T_values=(4.0,),
            analytical_rows=analytical,
            downward_records=[record],
            downward_profiles={0: {4.0: {"x": np.array([0.0, 1.0])}}},
            upward_records=[],
            upward_profiles={},
            boundary_payload={"selection_rule": "intersection"},
            started_at="start",
        )
        self.assertIn("analytic_u_0minus_m_s", payload)
        self.assertIn("semi_analytic_u_0minus_m_s", payload)
        self.assertIn("numerical_downward_u_0minus_m_s", payload)
        self.assertIn(0, payload["numerical_downward_profiles"])
        self.assertEqual(payload["solver_variant"], benchmark.SOLVER_VARIANT)
        self.assertEqual(payload["root_method"], benchmark.ROOT_METHOD)
        banned = {
            "u_0minus_max_candidate",
            "a_0plus_derived",
            "q",
            "Qstar",
            "TQstar",
            "aQstar",
        }
        self.assertTrue(banned.isdisjoint(payload))


if __name__ == "__main__":
    unittest.main()
