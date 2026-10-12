import argparse
import importlib.util
import sys
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "new_paper_calculations" / "26-08-07"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "T0plus_split_runner_test",
    SCRIPT_DIR / "_T0plus_split_runner.py",
)
split = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(split)


def sample_manifest():
    states = [
        {
            "T_0plus_MeV": 4.0,
            "T_0minus_MeV": 20.0,
            "nB_0minus_over_n0": 3.0,
            "rho": 1.5,
            "rho_inner_boundary": 1.0,
            "rho_outer_boundary": 2.0,
        },
        {
            "T_0plus_MeV": 2.0,
            "T_0minus_MeV": 21.0,
            "nB_0minus_over_n0": 3.1,
            "rho": 1.6,
            "rho_inner_boundary": 1.1,
            "rho_outer_boundary": 2.0,
        },
    ]
    return {
        "manifest_type": split.benchmark.DOMAIN_MANIFEST_TYPE,
        "selection_rule": "semi_analytic_safe_domain",
        "T_0plus_values_MeV": np.array([4.0, 2.0]),
        "boundary_reference_T_0plus_MeV": 0.0,
        "calibration": {"n0_MeV3": 10.0, "B14_MeV": 189.0},
        "points": [
            {
                "point_index": 0,
                "nB_0minus_over_n0": 3.0,
                "T_0minus_MeV": 20.0,
                "theta_rad": 0.5,
                "theta_deg": 28.6,
                "radial_fraction": 0.5,
                "radial_fraction_used": 0.5,
                "rho": 1.5,
                "rho_inner_boundary": 1.0,
                "rho_outer_boundary": 2.0,
                "rho_outer_boundary_semi_analytic": 2.0,
                "states": states,
            }
        ],
        "boundary_diagnostics": {"method": "semi_analytic_velocity_bound"},
    }


class SplitPayloadTests(unittest.TestCase):
    def test_boundary_preparation_uses_the_unique_cold_A_boundary(self):
        args = split._prepare_parser().parse_args([])
        self.assertEqual(args.boundary_T0plus, 0.0)
        self.assertFalse(hasattr(args, "boundary_T0plus_values"))
        self.assertEqual(args.boundary_theta_count, 4)
        self.assertEqual(args.ray_probe_count, 15)
        self.assertEqual(args.bisection_steps, 10)

    def test_A_boundary_acceptance_is_not_loosened(self):
        self.assertEqual(split.benchmark.ray_grid.A_BOUNDARY_RESIDUAL_TOLERANCE, 5.0e-3)

    def test_default_interface_temperatures_are_the_requested_four(self):
        self.assertEqual(
            split.benchmark.DEFAULT_T_0PLUS_VALUES_MEV,
            (3.0, 1.0, 0.5, 0.01),
        )

    def test_26_08_02_A_boundary_reference_anchors_are_enforced(self):
        boundary = {
            "A_records": [
                {"theta_deg": theta, "nB_over_n0": density, "T_MeV": temperature}
                for theta, density, temperature in split.A_BOUNDARY_REFERENCE_ANCHORS
            ]
        }
        diagnostic = split.validate_A_boundary_reference(boundary)
        self.assertEqual(diagnostic["max_density_error_over_n0"], 0.0)
        broken = {"A_records": [dict(row) for row in boundary["A_records"]]}
        broken["A_records"][0]["nB_over_n0"] += 0.1
        with self.assertRaisesRegex(RuntimeError, "26-08-02 density anchors"):
            split.validate_A_boundary_reference(broken)

    def test_direct_boundary_uses_one_cold_A_root_per_ray(self):
        theta_axis = np.array([0.3, 0.8])
        T_values = (0.01, 0.5)
        results = []
        for theta_index, theta in enumerate(theta_axis):
            for T_index, T_0plus in enumerate(T_values):
                gap = {
                    "coordinate": 1.0 + theta_index + 0.1 * T_index,
                    "nB_over_n0": 2.0,
                    "T_MeV": 10.0,
                    "boundary_kind": "Delta_h_equals_zero",
                }
                results.append({
                    "success": True,
                    "task": {
                        "task_kind": "gap_ray",
                        "theta_index": theta_index,
                        "theta_rad": theta,
                        "T_0plus_MeV": T_0plus,
                    },
                    "completed": {"record": gap},
                    "message": "",
                })
            results.append({
                "success": True,
                "task": {
                    "task_kind": "A_ray",
                    "theta_index": theta_index,
                    "theta_rad": theta,
                    "T_0plus_MeV": 0.0,
                },
                "completed": {"record": {
                    "coordinate": 3.0 + theta_index,
                    "nB_over_n0": 4.0,
                    "T_MeV": 20.0,
                    "boundary_kind": "A_equals_one",
                }},
                "message": "",
            })
        boundary = split.assemble_direct_common_boundary(
            theta_axis=theta_axis,
            T_values=T_values,
            task_results=results,
        )
        np.testing.assert_allclose(
            boundary["rho_inner_by_T0plus"],
            [[1.0, 2.0], [1.1, 2.1]],
        )
        np.testing.assert_allclose(boundary["rho_outer"], [3.0, 4.0])
        self.assertEqual(
            boundary["construction"],
            "direct_rho_roots_at_fixed_theta",
        )
        self.assertEqual(boundary["A_boundary_T_0plus_MeV"], 0.0)
        self.assertEqual(
            boundary["selection_rule"],
            "semi_analytic_extreme_endothermic_A_boundary",
        )
        self.assertEqual(boundary["enthalpy_boundary_definition"], "fixed_T0plus_local_isobar")

    def test_direct_boundary_excludes_failed_edge_rays_instead_of_aborting(self):
        theta_axis = np.array([0.2, 0.5, 0.8])
        results = []
        for theta_index, theta in enumerate(theta_axis):
            if theta_index == 0:
                results.append({
                    "success": False,
                    "task": {
                        "task_kind": "gap_ray",
                        "theta_index": theta_index,
                        "theta_rad": theta,
                    },
                    "completed": None,
                    "message": "edge ray misses boundary",
                })
                continue
            for task_kind, rho in (("gap_ray", 1.0), ("A_ray", 2.0)):
                task = {
                    "task_kind": task_kind,
                    "theta_index": theta_index,
                    "theta_rad": theta,
                }
                if task_kind == "A_ray":
                    task["T_0plus_MeV"] = 0.0
                else:
                    task["T_0plus_MeV"] = 0.01
                results.append({
                    "success": True,
                    "task": task,
                    "completed": {"record": {
                        "coordinate": rho,
                        "nB_over_n0": 3.0,
                        "T_MeV": 20.0,
                        "boundary_kind": "root",
                    }},
                    "message": "",
                })
        boundary = split.assemble_direct_common_boundary(
            theta_axis=theta_axis,
            T_values=(0.01,),
            task_results=results,
        )
        np.testing.assert_allclose(boundary["theta_axis_rad"], [0.5, 0.8])
        self.assertEqual(len(boundary["failed_ray_tasks"]), 1)

    def test_direct_boundary_submits_each_finite_gap_and_one_cold_A_task_per_ray(self):
        submitted = []

        class ImmediateExecutor:
            def submit(self, function, task):
                submitted.append(dict(task))
                future = Future()
                future.set_result(function(task))
                return future

        def fake_task(task):
            coordinates = np.asarray(task["coordinates"], dtype=float)
            is_A = task["task_kind"] == "A_ray"
            coordinate = float(coordinates[-2] if is_A else coordinates[1])
            return {
                "success": True,
                "task": task,
                "completed": {
                    "record": {
                        "coordinate": coordinate,
                        "nB_over_n0": 3.0,
                        "T_MeV": 20.0,
                        "boundary_kind": "A_equals_one" if is_A else "Delta_h_equals_zero",
                    }
                },
                "message": "",
            }

        transform = split.benchmark.ray_grid.RayTransform(1.0, 9.0, 200.0)
        with mock.patch.object(
            split, "_solve_direct_boundary_task", side_effect=fake_task
        ):
            boundary = split.solve_direct_common_boundary(
                transform=transform,
                theta_count=2,
                density_max_over_n0=10.0,
                temperature_min_MeV=0.01,
                temperature_max_MeV=200.0,
                n0=1.0,
                B14=189.0,
                xi=-0.5,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
                ray_probe_count=5,
                bisection_steps=10,
                T_values=(0.01, 0.5),
                executor=ImmediateExecutor(),
            )

        self.assertEqual(len(submitted), 6)
        A_tasks = [task for task in submitted if task["task_kind"] == "A_ray"]
        self.assertEqual(len(A_tasks), 2)
        self.assertTrue(all(task["T_0plus_MeV"] == 0.0 for task in A_tasks))
        self.assertTrue(
            all(task["interface_fraction_mode"] == "extreme_endothermic" for task in A_tasks)
        )
        self.assertTrue(all(task["velocity_closure"] == "numerical_I2" for task in A_tasks))
        self.assertTrue(all(task["xi"] == 0.0 for task in A_tasks))
        self.assertEqual(boundary["A_boundary_T_0plus_MeV"], 0.0)

    def test_temperature_indexed_mesh_uses_one_outer_A_curve(self):
        transform = split.benchmark.ray_grid.RayTransform(1.0, 9.0, 200.0)
        geometry = {
            "theta_axis_rad": np.array([0.3, 0.8]),
            "T_0plus_values_MeV": np.array([0.01, 0.5]),
            "rho_inner_by_T0plus": np.array([[1.0, 1.2], [1.1, 1.3]]),
            "rho_outer": np.array([2.0, 2.2]),
        }
        points = split.build_temperature_indexed_points(
            geometry=geometry,
            radial_fractions=np.array([0.5]),
            transform=transform,
            radial_buffer_fraction=0.02,
        )
        self.assertEqual(len(points), 2)
        self.assertTrue(all(len(point["states"]) == 2 for point in points))
        for point in points:
            outer = {state["rho_outer_boundary"] for state in point["states"]}
            self.assertEqual(len(outer), 1)
            self.assertNotEqual(
                point["states"][0]["rho_inner_boundary"],
                point["states"][1]["rho_inner_boundary"],
            )

    def test_direct_boundary_diagnostics_are_saved_before_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary-debug.npy"
            payload = {
                "payload_type": "T0plus_cold_A_boundary_task_diagnostics_v2",
                "task_results": [{"success": False, "message": "failure"}],
            }
            split.benchmark.atomic_save(path, payload)
            loaded = np.load(path, allow_pickle=True).item()
        self.assertEqual(
            loaded["payload_type"],
            "T0plus_cold_A_boundary_task_diagnostics_v2",
        )
        self.assertEqual(loaded["task_results"][0]["message"], "failure")

    def test_domain_payload_records_only_semi_analytic_boundary(self):
        payload = split.build_domain_payload(
            args=argparse.Namespace(
                run_tag="domain",
                xi=-0.5,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
                density_match_rtol=1.0e-6,
            ),
            calibration={"n0_MeV3": 10.0, "B14_MeV": 189.0},
            points=sample_manifest()["points"],
            T_values=(4.0, 2.0),
            boundary_T_0plus=0.0,
            transform={"density_origin_over_n0": 1.0},
            semi_boundary={"rho_outer": np.array([2.0])},
            geometry={"rho_outer": np.array([2.0])},
            started_at="start",
        )
        self.assertEqual(payload["selection_rule"], "semi_analytic_safe_domain")
        self.assertEqual(
            payload["outer_boundary_method"],
            "semi_analytic_extreme_endothermic_bound",
        )
        self.assertEqual(
            payload["outer_boundary_interface_control"], "extreme_endothermic"
        )
        self.assertEqual(payload["outer_boundary_definition"], "A_equals_one")
        self.assertEqual(
            payload["outer_boundary_temperature_strategy"],
            "T_0plus_equals_zero_extreme_endothermic",
        )
        self.assertEqual(payload["boundary_reference_T_0plus_MeV"], 0.0)
        self.assertNotIn("full_analytic", payload["boundary_diagnostics"])

    def test_single_method_payload_does_not_emit_other_method_arrays(self):
        rows = [{
            "point_index": 0,
            "T_0plus_MeV": 4.0,
            "muB_0minus_MeV": 1000.0,
            "density_relative_error": 1.0e-10,
            "success": True,
            "message": "",
            "u_0minus": 1.0e-6,
            "jB": 2.0,
            "runtime_s": 0.1,
        }]
        payload = split.build_analytical_payload(
            method="analytic",
            args=argparse.Namespace(run_tag="full"),
            manifest=sample_manifest(),
            rows=rows,
            started_at="start",
            run_complete=True,
        )
        self.assertIn("u_0minus", payload)
        self.assertIn("u_0minus_m_s", payload)
        self.assertNotIn("semi_analytic_u_0minus", payload)
        self.assertNotIn("numerical_downward_u_0minus", payload)

    def test_seed_map_uses_only_successful_finite_semi_rows(self):
        rows = [
            {"point_index": 0, "T_0plus_MeV": 4.0, "success": True, "jB": 3.0},
            {"point_index": 0, "T_0plus_MeV": 2.0, "success": False, "jB": 4.0},
            {"point_index": 1, "T_0plus_MeV": 4.0, "success": True, "jB": np.nan},
        ]
        self.assertEqual(split.semi_seed_map(rows), {0: {4.0: 3.0}})

    def test_numerical_payload_keeps_attempts_profiles_and_hysteresis(self):
        down = [{
            "point_index": 0,
            "T_0plus_MeV": 4.0,
            "success": True,
            "status": "success",
            "u_0minus": 3.0e-6,
            "jB": 4.0,
            "attempt_index": 0,
        }]
        up = [dict(down[0], sweep_direction="upward", u_0minus=3.3e-6)]
        payload = split.build_numerical_payload(
            args=argparse.Namespace(run_tag="numerical"),
            manifest=sample_manifest(),
            seed_rows=[{
                "point_index": 0,
                "T_0plus_MeV": 4.0,
                "success": True,
                "jB": 3.0,
            }],
            downward_records=down,
            downward_profiles={0: {4.0: {"x": np.array([0.0, 1.0])}}},
            upward_records=up,
            upward_profiles={},
            started_at="start",
            run_complete=True,
        )
        self.assertIn("numerical_downward_profiles", payload)
        self.assertEqual(len(payload["numerical_downward_records"]), 1)
        self.assertTrue(payload["numerical_downward_success"][0, 0])
        self.assertAlmostEqual(payload["hysteresis_relative_difference"][0, 0], 0.3 / 3.3)

    def test_entry_scripts_are_method_specific(self):
        expected = {
            "run_T0plus_full_analytic.py": "analytic",
            "run_T0plus_semi_analytic.py": "semi_analytic",
            "run_T0plus_full_numerical.py": "numerical",
        }
        for filename, method in expected.items():
            path = SCRIPT_DIR / filename
            self.assertTrue(path.exists(), filename)
            self.assertEqual(split.entry_method(path), method)


if __name__ == "__main__":
    unittest.main()
