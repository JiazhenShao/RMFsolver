from __future__ import annotations

import importlib.util
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = (
    REPO_ROOT
    / "new_paper_calculations"
    / "26-08-02"
    / "_analytic_ray_grid.py"
)
RUNNER_PATH = (
    REPO_ROOT
    / "new_paper_calculations"
    / "26-08-02"
    / "run_analytic_uN_ray_grid.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnalyticRayGridAugustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = _load_module(ENGINE_PATH, "analytic_ray_grid_20260802_test")
        cls.runner = _load_module(RUNNER_PATH, "analytic_ray_runner_20260802_test")

    def test_runner_has_distinct_current_analytic_identity(self):
        args = self.runner.build_parser().parse_args([])
        self.assertEqual(
            args.output,
            "analytic-uN-T-nB-ray-grid-numerical-I2.npy",
        )
        self.assertEqual(
            args.run_tag,
            "analytic_uN_T_nB_numerical_I2_20260802_v3",
        )
        self.assertEqual(args.interface_fraction_mode, "extreme_endothermic")
        self.assertEqual(self.runner.DEFAULT_VELOCITY_CLOSURE, "numerical_I2")
        self.assertEqual(args.velocity_closure, "numerical_I2")

    def test_A_state_forwards_numerical_I2_closure(self):
        with (
            patch.object(self.engine, "muB_from_nB", return_value=1200.0),
            patch.object(
                self.engine.phase_velocity,
                "analytic_velocity_bound",
                return_value={
                    "A_boundary": 0.75,
                    "muB_inf": 1100.0,
                    "T_inf": 20.0,
                },
            ) as analytic_bound,
        ):
            row = self.engine.evaluate_A_state(
                nB_over_n0=4.0,
                temperature_MeV=10.0,
                n0=1.0,
                B14=189.0,
                xi=0.0,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
                interface_fraction_mode="extreme_endothermic",
                velocity_closure="numerical_I2",
            )

        self.assertEqual(row["status"], "success")
        self.assertEqual(
            analytic_bound.call_args.kwargs["velocity_closure"],
            "numerical_I2",
        )

    def test_PNM_density_inversion_uses_validated_physical_branch(self):
        with patch.object(
            self.engine.phase_velocity,
            "muB_from_nB_physical",
            return_value=1234.5,
        ) as physical_inverse:
            result = self.engine.muB_from_nB(
                nB_target=4.0,
                temperature_MeV=20.0,
                nm_type="PNM",
                prefer=1100.0,
            )

        self.assertEqual(result, 1234.5)
        self.assertEqual(physical_inverse.call_count, 1)
        self.assertEqual(physical_inverse.call_args.args[:2], (4.0, 20.0))
        self.assertEqual(physical_inverse.call_args.kwargs.get("scan_points"), 16)
        self.assertEqual(physical_inverse.call_args.kwargs.get("muB_lo"), 850.0)
        self.assertEqual(physical_inverse.call_args.kwargs.get("muB_hi"), 1450.0)
        self.assertIs(physical_inverse.call_args.kwargs.get("auto_expand"), True)

    def test_payload_identifies_numerical_I2_closure(self):
        args = self.runner.build_parser().parse_args([])
        payload = self.engine.build_payload(
            args=args,
            transform=self.engine.RayTransform(1.0, 9.0, 200.0),
            arrays={"status_grid": np.zeros((1, 1), dtype=np.int8)},
            records=[],
            boundary_payload={},
            calibration={},
        )

        self.assertEqual(payload["velocity_closure"], "numerical_I2")
        self.assertEqual(
            payload["transport_integral_definition"],
            "numerical_I2_exact_rate_full_diffusion",
        )
        self.assertEqual(payload["arguments"]["velocity_closure"], "numerical_I2")

    def test_parallel_endpoint_scheduler_tracks_individual_probe_work(self):
        self.assertTrue(
            hasattr(self.engine, "_solve_endpoint_probe_task"),
            "probe-level endpoint worker is missing",
        )
        self.assertTrue(
            hasattr(self.engine, "solve_endpoint_boundaries_parallel"),
            "parallel endpoint scheduler is missing",
        )
        progress_bars = []
        submitted_tasks = []

        class FakeProgress:
            def __init__(self, *, total, desc, unit, **_kwargs):
                self.total = int(total)
                self.desc = str(desc)
                self.unit = str(unit)
                self.completed = 0
                progress_bars.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def update(self, count=1):
                self.completed += int(count)

            def set_postfix_str(self, _value):
                return None

        class ImmediateExecutor:
            def submit(self, function, *args, **kwargs):
                submitted_tasks.append(dict(args[0]))
                future = Future()
                try:
                    future.set_result(function(*args, **kwargs))
                except Exception as exc:
                    future.set_exception(exc)
                return future

        gap_roots = {
            "gap_bottom_endpoint": 3.2,
            "gap_left_endpoint": 52.0,
        }
        A_roots = {
            "A_bottom_endpoint": 4.0,
            "A_left_endpoint": 80.0,
        }

        def fake_probe(task):
            edge = str(task["edge_kind"])
            coordinate = float(task["coordinate"])
            row = {
                "coordinate": coordinate,
                "nB_over_n0": float(task["nB_over_n0"]),
                "T_MeV": float(task["T_MeV"]),
                "muB_0minus_MeV": 1100.0 + coordinate,
                "message": "",
            }
            if edge.startswith("gap_"):
                row.update(
                    {
                        "status": "success",
                        "gap_MeV": coordinate - gap_roots[edge],
                    }
                )
            elif coordinate < A_roots[edge]:
                row.update(
                    {
                        "status": "success",
                        "A_boundary": 0.999,
                        "endpoint_solution": (1200.0, 20.0),
                    }
                )
            else:
                row.update(
                    {
                        "status": "A_forbidden",
                        "A_boundary": np.nan,
                        "endpoint_solution": None,
                    }
                )
            return {"edge_kind": edge, "record": row}

        with patch.object(
            self.engine,
            "_solve_endpoint_probe_task",
            side_effect=fake_probe,
        ):
            results = self.engine.solve_endpoint_boundaries_parallel(
                density_probes=np.linspace(1.0, 5.0, 5),
                temperature_probes=np.linspace(0.0, 100.0, 5),
                temperature_min_MeV=0.0,
                density_min_over_n0=1.0,
                n0=1.0,
                B14=189.0,
                xi=0.0,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
                interface_fraction_mode="extreme_endothermic",
                velocity_closure="numerical_I2",
                bisection_steps=2,
                progress_factory=FakeProgress,
                executor=ImmediateExecutor(),
            )

        coarse = [
            task for task in submitted_tasks if task["phase"] == "coarse"
        ]
        self.assertEqual(len(coarse), 20)
        self.assertEqual(
            {task["edge_kind"] for task in coarse},
            {
                "gap_bottom_endpoint",
                "gap_left_endpoint",
                "A_bottom_endpoint",
                "A_left_endpoint",
            },
        )
        completed = {
            progress.desc: (progress.completed, progress.total)
            for progress in progress_bars
        }
        self.assertEqual(completed["Boundary endpoint probes"], (20, 20))
        self.assertEqual(completed["Boundary endpoint refinement"], (44, 44))
        self.assertEqual(results["gap_bottom_endpoint"]["boundary_kind"], "Delta_h_equals_zero")
        self.assertEqual(results["A_left_endpoint"]["boundary_kind"], "A_equals_one")

    def test_runner_defaults_cover_expanded_exact_A_window(self):
        args = self.runner.build_parser().parse_args([])
        self.assertEqual(args.density_min_over_n0, 1.0)
        self.assertEqual(args.density_max_over_n0, 10.0)
        self.assertEqual(args.t_min, 1.0e-2)
        self.assertEqual(args.t_max, 200.0)
        self.assertEqual(args.theta_count, 45)
        self.assertEqual(args.radial_count, 30)

    def test_missing_A_boundary_clips_mesh_to_physical_window(self):
        A_calls = []

        def fake_gap_solver(**kwargs):
            label = kwargs["label"]
            if label == "gap_fixed_T_minimum":
                density, temperature, coordinate = 3.0, 0.01, None
            elif label == "gap_fixed_density_minimum":
                density, temperature, coordinate = 1.0, 60.0, None
            else:
                coordinate = 0.40
                density, temperature = kwargs["physical_coordinates"](coordinate)
            row = {
                "status": "success",
                "boundary_kind": "Delta_h_equals_zero",
                "solve_kind": label,
                "nB_over_n0": float(density),
                "T_MeV": float(temperature),
                "gap_MeV": 0.0,
                "muB_0minus_MeV": 1100.0,
            }
            if coordinate is not None:
                row["coordinate"] = coordinate
            return row

        def fake_A_solver(**kwargs):
            A_calls.append(kwargs)
            label = kwargs["label"]
            coordinate = float(np.asarray(kwargs["coordinates"])[-1])
            density, temperature = kwargs["physical_coordinates"](coordinate)
            row = {
                "status": "clipped",
                "boundary_kind": "physical_window_outer",
                "solve_kind": label,
                "nB_over_n0": float(density),
                "T_MeV": float(temperature),
                "A_below_boundary": 0.9,
                "endpoint_solution": (1200.0, 50.0),
                "muB_0minus_MeV": 1100.0,
                "coordinate": coordinate,
            }
            return row

        with (
            patch.object(self.engine, "solve_gap_on_path", fake_gap_solver),
            patch.object(self.engine, "solve_A_transition_on_path", fake_A_solver),
        ):
            boundaries = self.engine.solve_boundary_curves(
                transform=self.engine.RayTransform(1.0, 4.0, 100.0),
                theta_count=5,
                density_min_over_n0=1.0,
                density_max_over_n0=5.0,
                temperature_min_MeV=0.01,
                temperature_max_MeV=100.0,
                n0=1.0,
                B14=189.0,
                xi=0.0,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
                interface_fraction_mode="extreme_endothermic",
                endpoint_probe_count=5,
                ray_probe_count=5,
                bisection_steps=2,
                angular_buffer_fraction=1.0e-4,
                progress_factory=None,
                executor=None,
            )

        endpoints = boundaries["endpoint_records"]
        self.assertIn("A_bottom", endpoints)
        self.assertTrue(A_calls)
        self.assertTrue(all(call.get("allow_missing") is True for call in A_calls))
        self.assertTrue(
            all(
                row["boundary_kind"] == "physical_window_outer"
                for row in boundaries["A_records"]
            )
        )
        self.assertEqual(boundaries["A_curve_records"], [])
        self.assertTrue(np.all(boundaries["rho_outer"] > boundaries["rho_inner"]))

    def test_missing_A_boundary_does_not_mask_numerical_failure(self):
        def fake_state(**kwargs):
            density = float(kwargs["nB_over_n0"])
            if np.isclose(density, 1.5):
                return {
                    "status": "numerical_failure",
                    "A_boundary": np.nan,
                    "endpoint_solution": None,
                    "muB_0minus_MeV": np.nan,
                    "message": "synthetic failure",
                }
            return {
                "status": "success",
                "A_boundary": 0.5 + 0.1 * density,
                "endpoint_solution": (1200.0, 20.0),
                "muB_0minus_MeV": 1100.0,
                "message": "",
            }

        with patch.object(self.engine, "evaluate_A_state", fake_state):
            with self.assertRaisesRegex(RuntimeError, "numerical_failure"):
                self.engine.solve_A_transition_on_path(
                    coordinates=np.asarray([1.0, 1.5, 2.0]),
                    physical_coordinates=lambda density: (density, 10.0),
                    n0=1.0,
                    B14=189.0,
                    xi=0.0,
                    ms=0.0,
                    nm_type="PNM",
                    upB=5000,
                    interface_fraction_mode="extreme_endothermic",
                    label="synthetic_A_path",
                    allow_missing=True,
                )

    def test_missing_A_boundary_allows_pre_success_numerical_hole(self):
        statuses = {
            1.0: "gap_forbidden",
            1.5: "numerical_failure",
            2.0: "gap_forbidden",
            2.5: "success",
            3.0: "success",
        }

        def fake_state(**kwargs):
            density = float(kwargs["nB_over_n0"])
            status = statuses[density]
            if status == "success":
                return {
                    "status": status,
                    "A_boundary": 0.5 + 0.1 * density,
                    "endpoint_solution": (1200.0, 20.0),
                    "muB_0minus_MeV": 1100.0,
                    "message": "",
                }
            return {
                "status": status,
                "A_boundary": np.nan,
                "endpoint_solution": None,
                "muB_0minus_MeV": np.nan,
                "message": "synthetic pre-success state",
            }

        with patch.object(self.engine, "evaluate_A_state", fake_state):
            result = self.engine.solve_A_transition_on_path(
                coordinates=np.asarray(sorted(statuses)),
                physical_coordinates=lambda density: (density, 10.0),
                n0=1.0,
                B14=189.0,
                xi=0.0,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
                interface_fraction_mode="extreme_endothermic",
                label="synthetic_A_prefix_hole",
                allow_missing=True,
            )

        self.assertEqual(result["boundary_kind"], "physical_window_outer")
        self.assertEqual(result["coordinate"], 3.0)
        self.assertEqual(result["prefix_numerical_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
