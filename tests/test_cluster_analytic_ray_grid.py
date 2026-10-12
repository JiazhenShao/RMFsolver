from __future__ import annotations

import importlib.util
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = (
    REPO_ROOT
    / "new_paper_calculations"
    / "26-07-24"
    / "_analytic_ray_grid.py"
)
RUNNER_PATH = (
    REPO_ROOT
    / "new_paper_calculations"
    / "26-07-24"
    / "run_analytic_uN_ray_grid.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnalyticRayGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = _load_module(ENGINE_PATH, "analytic_ray_grid_engine_test")
        cls.runner = _load_module(RUNNER_PATH, "analytic_ray_grid_runner_test")

    def test_transform_round_trip_preserves_physical_units(self):
        transform = self.engine.RayTransform(
            density_origin_over_n0=1.0,
            density_scale_over_n0=4.0,
            temperature_scale_MeV=100.0,
        )
        density = np.asarray([1.01, 2.4, 4.95])
        temperature = np.asarray([0.01, 18.0, 99.0])

        rho, theta = transform.to_hidden(density, temperature)
        recovered_density, recovered_temperature = transform.to_physical(rho, theta)

        np.testing.assert_allclose(recovered_density, density, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            recovered_temperature,
            temperature,
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertTrue(np.all(rho >= 0.0))
        self.assertTrue(np.all((theta >= 0.0) & (theta <= 0.5 * np.pi)))

    def test_safe_ray_mesh_is_strictly_between_boundaries(self):
        transform = self.engine.RayTransform(1.0, 4.0, 100.0)
        theta = np.radians(np.asarray([5.0, 45.0, 90.0]))
        eta = self.engine.chebyshev_open_axis(5, buffer_fraction=1.0e-3)
        rho_gap = np.asarray([0.50, 0.55, 0.60])
        rho_A = np.asarray([0.90, 0.95, 0.98])

        mesh = self.engine.build_safe_ray_mesh(
            transform=transform,
            theta_axis_rad=theta,
            radial_fraction_axis=eta,
            rho_inner=rho_gap,
            rho_outer=rho_A,
        )

        self.assertEqual(mesh["rho_grid"].shape, (3, 5))
        self.assertTrue(np.all(mesh["rho_grid"] > rho_gap[:, None]))
        self.assertTrue(np.all(mesh["rho_grid"] < rho_A[:, None]))
        np.testing.assert_allclose(
            mesh["nB_grid_over_n0"],
            1.0 + 4.0 * mesh["rho_grid"] * np.cos(mesh["theta_grid_rad"]),
        )
        np.testing.assert_allclose(
            mesh["temperature_grid_MeV"],
            100.0 * mesh["rho_grid"] * np.sin(mesh["theta_grid_rad"]),
        )

    def test_radial_order_starts_near_10_mev_and_uses_adjacent_points(self):
        temperatures = np.asarray([2.0, 6.0, 11.0, 18.0, 28.0])
        order = self.engine.radial_solve_order(temperatures, seed_temperature_MeV=10.0)

        self.assertEqual(order[0], (2, "seed"))
        self.assertEqual(order[1], (3, "up"))
        self.assertEqual(order[2], (4, "up"))
        self.assertEqual(order[3], (1, "down"))
        self.assertEqual(order[4], (0, "down"))

    def test_muB_from_nB_rejects_discontinuous_false_density_root(self):
        n0 = float(0.16 * self.engine.const.MeV_fm**3)
        density_over_n0 = 1.000901255
        temperature_MeV = 0.01
        target_density = density_over_n0 * n0

        muB_N = self.engine.muB_from_nB(
            target_density,
            temperature_MeV,
            "PNM",
            prefer=1162.399971484,
        )
        recovered_density = self.engine.phase_velocity.nB_NM(
            muB_N,
            temperature_MeV,
            param=self.engine.para.paraQMCRMF3,
            NM_type="PNM",
        )

        self.assertLessEqual(
            abs(recovered_density - target_density),
            1.0e-8 * max(abs(target_density), 1.0),
        )
        self.assertAlmostEqual(muB_N, 973.784926, places=4)

    def test_payload_uses_cartesian_primary_coordinates_without_metastability(self):
        shape = (3, 4)
        arrays = {
            "nB_grid_over_n0": np.full(shape, 2.0),
            "nB_grid_MeV3": np.full(shape, 3.0),
            "temperature_grid_MeV": np.full(shape, 10.0),
            "velocity_grid_m_s": np.full(shape, 100.0),
            "u_N_grid": np.full(shape, 1.0e-6),
            "status_grid": np.zeros(shape, dtype=np.int8),
        }
        args = SimpleNamespace(run_tag="unit-test", theta_count=3, radial_count=4)
        payload = self.engine.build_payload(
            args=args,
            transform=self.engine.RayTransform(1.0, 4.0, 100.0),
            arrays=arrays,
            records=[],
            boundary_payload={"gap_records": [], "A_records": []},
            calibration={
                "B14_calibrated_MeV": 189.0,
                "n0_MeV3": 1.0,
                "nB_crit0_MeV3": 3.0,
            },
        )

        self.assertEqual(
            payload["primary_coordinate_system"],
            "physical_cartesian_curvilinear_grid",
        )
        self.assertFalse(payload["metastability_used"])
        self.assertEqual(payload["nB_grid_over_n0"].shape, shape)
        self.assertEqual(payload["temperature_grid_MeV"].shape, shape)
        self.assertNotIn("metastability_grid", payload)
        self.assertNotIn("metastability_definition", payload)

    def test_runner_defaults_to_30_by_20_extreme_endothermic_grid(self):
        parser = self.runner.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.theta_count, 30)
        self.assertEqual(args.radial_count, 20)
        self.assertEqual(args.interface_fraction_mode, "extreme_endothermic")
        self.assertEqual(
            self.runner.DEFAULT_RUN_TAG,
            "analytic_uN_elliptical_ray_grid_v1",
        )

    def test_boundary_solver_reports_endpoint_and_ray_progress(self):
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
                submitted_tasks.append(args[0])
                future = Future()
                try:
                    future.set_result(function(*args, **kwargs))
                except Exception as exc:
                    future.set_exception(exc)
                return future

        def fake_gap_solver(**kwargs):
            label = kwargs["label"]
            if label == "gap_fixed_T_minimum":
                density, temperature, rho = 3.0, 0.01, None
            elif label == "gap_fixed_density_minimum":
                density, temperature, rho = 1.0, 60.0, None
            else:
                rho = 0.40
                density, temperature = kwargs["physical_coordinates"](rho)
            row = {
                "status": "success",
                "boundary_kind": "Delta_h_equals_zero",
                "solve_kind": label,
                "nB_over_n0": float(density),
                "T_MeV": float(temperature),
                "gap_MeV": 0.0,
                "muB_N_MeV": 1100.0,
            }
            if rho is not None:
                row["coordinate"] = rho
            return row

        def fake_A_solver(**kwargs):
            label = kwargs["label"]
            if label == "A_fixed_T_minimum":
                density, temperature, rho = 4.6, 0.01, None
            elif label == "A_fixed_density_minimum":
                density, temperature, rho = 1.0, 98.0, None
            else:
                rho = 0.80
                density, temperature = kwargs["physical_coordinates"](rho)
            row = {
                "status": "root",
                "boundary_kind": "A_equals_one",
                "solve_kind": label,
                "nB_over_n0": float(density),
                "T_MeV": float(temperature),
                "A_below_boundary": 1.0,
                "endpoint_solution": (1200.0, 50.0),
                "muB_N_MeV": 1100.0,
            }
            if rho is not None:
                row["coordinate"] = rho
            return row

        with (
            patch.object(self.engine, "solve_gap_on_path", fake_gap_solver),
            patch.object(
                self.engine,
                "solve_A_transition_on_path",
                fake_A_solver,
            ),
        ):
            self.engine.solve_boundary_curves(
                transform=self.engine.RayTransform(1.0, 4.0, 100.0),
                theta_count=3,
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
                progress_factory=FakeProgress,
                executor=ImmediateExecutor(),
            )

        completed = {
            progress.desc: (progress.completed, progress.total)
            for progress in progress_bars
        }
        self.assertEqual(completed["Boundary endpoints"], (4, 4))
        self.assertEqual(completed["Boundary curves"], (6, 6))
        endpoint_kinds = {
            task["task_kind"]
            for task in submitted_tasks
            if task["task_kind"].endswith("_endpoint")
        }
        self.assertEqual(
            endpoint_kinds,
            {"gap_bottom_endpoint", "gap_left_endpoint",
             "A_bottom_endpoint", "A_left_endpoint"},
        )
        ray_kinds = [
            task["task_kind"]
            for task in submitted_tasks
            if task["task_kind"].endswith("_ray")
        ]
        self.assertIn("gap_ray", ray_kinds)
        self.assertIn("A_ray", ray_kinds)

    def test_point_front_scheduler_preserves_immediate_neighbor_continuation(self):
        args = self.runner.build_parser().parse_args([])
        args.endpoint_guess = (1200.0, 20.0)
        arrays = {
            "theta_axis_rad": np.asarray([0.5]),
            "theta_grid_rad": np.full((1, 5), 0.5),
            "rho_grid": np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            "radial_fraction_grid": np.asarray(
                [[0.1, 0.3, 0.5, 0.7, 0.9]]
            ),
            "nB_grid_over_n0": np.asarray([[1.2, 1.4, 1.6, 1.8, 2.0]]),
            "temperature_grid_MeV": np.asarray([[2.0, 6.0, 11.0, 18.0, 28.0]]),
        }

        seeds = self.runner._build_seed_point_tasks(
            args=args,
            arrays=arrays,
            n0=1.0,
            B14=189.0,
        )
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["col_index"], 2)
        self.assertEqual(seeds[0]["continuation_direction"], "seed")

        seed_result = {
            **seeds[0],
            "success": True,
            "endpoint_solution": (1180.0, 10.0),
            "muB_N_MeV": 1110.0,
        }
        successors = self.runner._continuation_successor_tasks(
            completed_task=seeds[0],
            completed_row=seed_result,
            args=args,
            arrays=arrays,
            n0=1.0,
            B14=189.0,
        )
        self.assertEqual(
            [(task["col_index"], task["continuation_direction"])
             for task in successors],
            [(3, "up"), (1, "down")],
        )
        self.assertTrue(
            all(task["endpoint_guess"] == (1180.0, 10.0) for task in successors)
        )

        failed_down = {
            **successors[1],
            "success": False,
            "endpoint_solution": None,
            "muB_N_MeV": np.nan,
        }
        fallback = self.runner._continuation_successor_tasks(
            completed_task=successors[1],
            completed_row=failed_down,
            args=args,
            arrays=arrays,
            n0=1.0,
            B14=189.0,
        )
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]["col_index"], 0)
        self.assertEqual(fallback[0]["endpoint_guess"], (1200.0, 20.0))
        self.assertEqual(fallback[0]["endpoint_guess_source"], "default")


if __name__ == "__main__":
    unittest.main()
