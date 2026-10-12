import importlib.util
import multiprocessing as mp
import queue
import sys
import unittest
import warnings
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "new_paper_calculations" / "26-08-10"
MODULE_PATH = SCRIPT_DIR / "_radial_continuation.py"
PREPARE_PATH = SCRIPT_DIR / "prepare_T0plus5_domain.py"
RUNNER_PATH = SCRIPT_DIR / "_T0plus5_runner.py"
NUMERICAL_PATH = SCRIPT_DIR / "run_T0plus5_numerical.py"


def load_module():
    if not MODULE_PATH.exists():
        raise AssertionError(f"missing radial continuation module: {MODULE_PATH}")
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "T0plus5_radial_continuation_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_importable_module(name):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    sys.modules.pop(name, None)
    return __import__(name)


def run_nested_hard_call(connection, script_dir):
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    radial = __import__("_radial_continuation")
    try:
        value = radial.run_module_call_with_hard_timeout(
            "builtins",
            "pow",
            args=(3, 4),
            timeout_s=5.0,
        )
        connection.send(("result", value))
    except Exception as exc:
        connection.send(("error", str(exc)))
    finally:
        connection.close()


def successful_result(jB):
    axis = np.array([0.0, 1.0])
    return {
        "success": True,
        "message": "ok",
        "u_0minus": 1.0e-6,
        "jB": float(jB),
        "a_0plus": 0.4,
        "a_inf": 0.0,
        "lambda_n": 0.8,
        "T_inf": 20.0,
        "bvp_status": 0,
        "bvp_message": "converged",
        "bvp_nodes": 20,
        "bvp_niter": 2,
        "bvp_parameters": np.array([float(jB)]),
        "bvp_max_rms_residual": 1.0e-6,
        "bvp_collocation_converged": True,
        "E_right_residual_norm": 1.0e-8,
        "Pi_right_residual_norm": 1.0e-8,
        "T_0plus_residual": 0.0,
        "tail_residual_norm": 1.0e-8,
        "kaon_equation_residual_norm": 1.0e-6,
        "energy_equation_residual_norm": 1.0e-8,
        "momentum_equation_residual_norm": 1.0e-8,
        "solver_variant": "energy_conserving_nK_fixed_T_0plus",
        "_root_method": "solve_bvp_nK_jK_parameter_jB",
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


def radial_task(fractions=(0.02, 0.5, 0.9)):
    points = []
    for radial_index, fraction in enumerate(fractions):
        points.append(
            {
                "point_index": radial_index,
                "theta_index": 0,
                "theta_rad": 0.0,
                "radial_index": radial_index,
                "radial_fraction": float(fraction),
                "radial_fraction_used": float(fraction),
                "rho": 1.0 + float(fraction),
                "rho_inner_boundary": 1.0,
                "rho_outer_boundary": 2.0,
                "T_0minus_MeV": 10.0 + float(fraction),
                "nB_0minus_over_n0": 3.0 + float(fraction),
                "nB_0minus_MeV3": 30.0 + float(fraction),
                "semi_jB": 2.0 + float(fraction),
            }
        )
    return {
        "theta_index": 0,
        "theta_rad": 0.0,
        "points": points,
        "T_0plus_MeV": 5.0,
        "B14_MeV": 189.0,
        "ms_MeV": 0.0,
        "nm_type": "PNM",
        "tail_eps": 1.0e-8,
        "n_mesh": 20,
        "tol_bvp": 1.0e-4,
        "max_nodes": 100,
        "jB_lower_bound": 1.0e-12,
        "jB_upper_factor": 1.0e-4,
        "point_timeout_s": 10.0,
        "target_timeout_s": 30.0,
        "max_bridge_depth": 3,
        "min_bridge_fraction_step": 1.0e-3,
    }


class GeometryTests(unittest.TestCase):
    def test_cluster_defaults_make_a_600_cell_endpoint_inclusive_grid(self):
        if not PREPARE_PATH.exists():
            self.fail(f"missing domain launcher: {PREPARE_PATH}")
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        spec = importlib.util.spec_from_file_location(
            "prepare_T0plus5_domain_test",
            PREPARE_PATH,
        )
        prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)
        self.assertEqual(prepare.DEFAULT_T0PLUS_VALUES_MEV, (5.0,))
        self.assertEqual(prepare.DEFAULT_CONTOUR_RAY_COUNT, 30)
        self.assertEqual(len(prepare.DEFAULT_RADIAL_FRACTIONS), 20)
        self.assertEqual(prepare.DEFAULT_ANGULAR_BUFFER_FRACTION, 0.0)
        self.assertEqual(prepare.DEFAULT_RADIAL_BUFFER_FRACTION, 0.0)

    def test_numerical_defaults_bound_trials_and_complete_cells(self):
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        spec = importlib.util.spec_from_file_location(
            "run_T0plus5_numerical_test",
            NUMERICAL_PATH,
        )
        numerical = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(numerical)
        self.assertEqual(numerical.DEFAULT_POINT_TIMEOUT_S, 600.0)
        self.assertEqual(numerical.DEFAULT_TARGET_TIMEOUT_S, 1800.0)
        self.assertEqual(numerical.DEFAULT_SAVE_EVERY, 1)

    def test_theta_axis_includes_both_endpoints(self):
        radial = load_module()
        axis = radial.inclusive_theta_axis(30)
        self.assertEqual(axis.size, 30)
        self.assertEqual(axis[0], 0.0)
        self.assertEqual(axis[-1], 0.5 * np.pi)

    def test_grouped_rays_sort_points_from_inner_to_outer(self):
        radial = load_module()
        points = radial_task((0.9, 0.02, 0.5))["points"]
        grouped = radial.group_points_by_theta(points)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(
            [point["radial_fraction_used"] for point in grouped[0]],
            [0.02, 0.5, 0.9],
        )


class RadialContinuationTests(unittest.TestCase):
    def test_hard_timeout_terminates_a_stuck_child_process(self):
        radial = load_importable_module("_radial_continuation")
        with self.assertRaises(radial.BVPTrialTimeoutError):
            radial.run_module_call_with_hard_timeout(
                "time",
                "sleep",
                args=(0.5,),
                timeout_s=0.05,
            )

    def test_hard_timeout_returns_a_completed_child_result(self):
        radial = load_importable_module("_radial_continuation")
        result = radial.run_module_call_with_hard_timeout(
            "builtins",
            "pow",
            args=(2, 5),
            timeout_s=5.0,
        )
        self.assertEqual(result, 32)

    def test_hard_timeout_child_can_run_inside_a_ray_worker(self):
        if "fork" not in mp.get_all_start_methods():
            self.skipTest("nested-process smoke test requires fork for its outer worker")
        context = mp.get_context("fork")
        receive_connection, send_connection = context.Pipe(duplex=False)
        worker = context.Process(
            target=run_nested_hard_call,
            args=(send_connection, SCRIPT_DIR),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            worker.start()
        send_connection.close()
        self.assertTrue(receive_connection.poll(10.0))
        self.assertEqual(receive_connection.recv(), ("result", 81))
        worker.join(timeout=2.0)
        self.assertEqual(worker.exitcode, 0)
        receive_connection.close()

    def test_adjacent_successful_profile_is_primary_seed(self):
        radial = load_module()
        calls = []

        def fake_solver(**kwargs):
            calls.append(
                {
                    "continuation": kwargs.get("continuation_guess"),
                    "jB_guess": kwargs.get("jB_guess"),
                }
            )
            return successful_result(kwargs.get("jB_guess", 1.0))

        with mock.patch.object(
            radial,
            "_solve_uNmax_with_hard_timeout",
            side_effect=lambda kwargs, timeout_s: fake_solver(**kwargs),
        ):
            output = radial.run_radial_trajectory(radial_task())

        self.assertEqual(len(output["target_records"]), 3)
        self.assertTrue(all(row["success"] for row in output["target_records"]))
        self.assertIsNone(calls[0]["continuation"])
        self.assertIsNotNone(calls[1]["continuation"])
        self.assertIsNotNone(calls[2]["continuation"])
        self.assertEqual(
            [row["guess_source"] for row in output["target_records"]],
            ["semi_analytic_jB", "adjacent_numerical_profile", "adjacent_numerical_profile"],
        )

    def test_failed_continuation_retries_with_current_semi_seed(self):
        radial = load_module()
        calls = []

        def fake_solver(**kwargs):
            uses_profile = "continuation_guess" in kwargs
            calls.append((uses_profile, kwargs.get("jB_guess")))
            if len(calls) == 2:
                raise RuntimeError("profile continuation failed")
            return successful_result(kwargs.get("jB_guess", 1.0))

        with mock.patch.object(
            radial,
            "_solve_uNmax_with_hard_timeout",
            side_effect=lambda kwargs, timeout_s: fake_solver(**kwargs),
        ):
            output = radial.run_radial_trajectory(radial_task((0.02, 0.5)))

        target = output["target_records"][1]
        self.assertTrue(target["success"])
        self.assertEqual(target["guess_source"], "continuation_profile_with_semi_jB")
        self.assertEqual(calls[1][0], True)
        self.assertEqual(calls[2], (True, 2.5))

    def test_failed_large_step_is_bridged_before_target_retry(self):
        radial = load_module()
        accepted = [0.02]
        attempted = []

        def fake_solver(**kwargs):
            fraction = float(kwargs["nB_0minus"] - 30.0)
            attempted.append(fraction)
            previous = accepted[-1]
            if fraction - previous > 0.3:
                raise RuntimeError("step too large")
            accepted.append(fraction)
            return successful_result(kwargs.get("jB_guess", 1.0))

        task = radial_task((0.02, 0.9))
        task["progress_queue"] = queue.Queue()
        with mock.patch.object(
            radial,
            "_solve_uNmax_with_hard_timeout",
            side_effect=lambda kwargs, timeout_s: fake_solver(**kwargs),
        ):
            output = radial.run_radial_trajectory(task)

        self.assertTrue(output["target_records"][-1]["success"])
        self.assertGreater(len(output["bridge_records"]), 0)
        self.assertTrue(any(0.02 < value < 0.9 for value in attempted))
        self.assertAlmostEqual(attempted[-1], 0.9)

    def test_target_event_contains_checkpoint_records(self):
        radial = load_module()
        task = radial_task((0.02,))
        task["progress_queue"] = queue.Queue()

        with mock.patch.object(
            radial,
            "_solve_uNmax_with_hard_timeout",
            side_effect=lambda kwargs, timeout_s: successful_result(kwargs["jB_guess"]),
        ):
            radial.run_radial_trajectory(task)

        events = []
        while not task["progress_queue"].empty():
            events.append(task["progress_queue"].get_nowait())
        target_event = next(event for event in events if event["event"] == "target_completed")
        self.assertEqual(target_event["target_record"]["point_index"], 0)
        self.assertEqual(len(target_event["attempt_records"]), 1)
        self.assertEqual(target_event["bridge_records"], [])

    def test_collector_invokes_checkpoint_callback_for_each_target(self):
        runner = load_importable_module("_T0plus5_runner")
        progress_events = queue.Queue()
        progress_events.put(
            {
                "event": "target_completed",
                "point_index": 7,
                "theta_index": 2,
                "radial_fraction": 0.5,
                "success": True,
                "attempt_count": 1,
                "bridge_count": 0,
                "target_record": {"point_index": 7, "success": True},
                "attempt_records": [{"point_index": 7, "success": True}],
                "bridge_records": [],
            }
        )
        future = Future()
        future.set_result({"theta_index": 2})
        checkpointed = []

        runner.collect_radial_results(
            {future},
            progress_events,
            total_targets=1,
            target_completed_callback=checkpointed.append,
        )

        self.assertEqual(len(checkpointed), 1)
        self.assertEqual(checkpointed[0]["target_record"]["point_index"], 7)


if __name__ == "__main__":
    unittest.main()
