import importlib.util
import time
import sys
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "new_paper_calculations" / "26-08-18"
PREPARE_PATH = SCRIPT_DIR / "prepare_T0plus0p1_domain.py"
RADIAL_PATH = SCRIPT_DIR / "_radial_shell_continuation.py"
NUMERICAL_PATH = SCRIPT_DIR / "run_T0plus0p1_numerical.py"
RUNNER_PATH = SCRIPT_DIR / "_T0plus0p1_runner.py"


def load_module(name, path):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_importable_module(name):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    sys.modules.pop(name, None)
    return __import__(name)


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
        "solver_variant": "uNmax_direct_T0plus_nK",
        "_root_method": "solve_bvp_uNmax_nK_parameter_1d",
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


def numerical_task():
    return {
        "point": {
            "point_index": 7,
            "theta_index": 3,
            "radial_index": 2,
            "radial_fraction": 0.10,
            "radial_fraction_used": 0.10,
            "T_0minus_MeV": 20.0,
            "nB_0minus_over_n0": 3.2,
            "nB_0minus_MeV3": 40.0,
        },
        "T_0plus_MeV": 0.1,
        "B14_MeV": 189.0,
        "ms_MeV": 0.0,
        "nm_type": "PNM",
        "tail_eps": 1.0e-8,
        "n_mesh": 20,
        "tol_bvp": 1.0e-4,
        "max_nodes": 100,
        "jB_lower_bound": 1.0e-12,
        "jB_upper_factor": 1.0e-4,
        "trial_timeout_s": 0.1,
        "cell_timeout_s": 10.0,
    }


class DomainContractTests(unittest.TestCase):
    def test_domain_defaults_define_fixed_temperature_600_cell_grid(self):
        self.assertTrue(PREPARE_PATH.exists(), f"missing {PREPARE_PATH}")
        prepare = load_module("T0plus0p1_prepare_test", PREPARE_PATH)
        self.assertEqual(prepare.DEFAULT_T0PLUS_MEV, 0.1)
        self.assertEqual(prepare.DEFAULT_CONTOUR_RAY_COUNT, 30)
        self.assertEqual(len(prepare.DEFAULT_RADIAL_FRACTIONS), 20)
        self.assertEqual(prepare.DEFAULT_RADIAL_FRACTIONS[0], 0.02)
        self.assertEqual(prepare.DEFAULT_RADIAL_FRACTIONS[-1], 0.99)

    def test_theta_axis_contains_both_exact_endpoints(self):
        self.assertTrue(PREPARE_PATH.exists(), f"missing {PREPARE_PATH}")
        prepare = load_module("T0plus0p1_prepare_theta_test", PREPARE_PATH)
        theta = prepare.contour_theta_axis(prepare.DEFAULT_CONTOUR_RAY_COUNT)
        np.testing.assert_allclose(theta[[0, -1]], [0.0, 0.5 * np.pi])


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.radial = load_module("T0plus0p1_radial_candidates_test", RADIAL_PATH)

    def test_empirical_candidate_ladders(self):
        self.assertEqual(self.radial.first_shell_candidates(), (0.8, 0.6, 1.0, 1.2, 1.4))
        self.assertEqual(
            self.radial.previous_shell_multipliers(),
            (1.15, 1.25, 1.35, 1.50),
        )
        self.assertEqual(
            self.radial.same_shell_multipliers(),
            (1.00, 0.90, 1.10, 1.20),
        )

    def test_nearby_theta_uses_robust_local_median(self):
        values = {
            0: 9.0,
            1: 10.0,
            2: np.nan,
            4: 12.0,
            5: 1000.0,
            6: 14.0,
        }
        result = self.radial.nearby_theta_median(values, target_theta_index=3, radius=2)
        self.assertEqual(result, 11.0)

    def test_same_ray_extrapolation_is_bounded(self):
        predicted = self.radial.extrapolated_same_ray_jB(
            previous_jB=10.0,
            earlier_jB=6.0,
            previous_rho=0.50,
            earlier_rho=0.40,
            target_rho=0.60,
        )
        self.assertEqual(predicted, 15.5)

    def test_candidate_deduplication_uses_three_percent_tolerance(self):
        self.assertEqual(
            self.radial.deduplicate_jB_candidates([10.0, 10.1, 11.0, np.nan, -1.0]),
            (10.0, 11.0),
        )

    def test_primary_candidate_builder_keeps_complete_ordered_ladders(self):
        first_shell = self.radial.primary_candidates(
            {"radial_index": 0, "radial_fraction_used": 0.02},
            previous=None,
            earlier=None,
        )
        self.assertEqual(
            tuple(candidate["jB_guess"] for candidate in first_shell),
            self.radial.first_shell_candidates(),
        )
        previous = {
            "point": {"radial_fraction_used": 0.10},
            "record": {"jB": 2.0, "success": True},
            "profile": {"x": np.array([0.0, 1.0])},
        }
        earlier = {
            "point": {"radial_fraction_used": 0.06},
            "record": {"jB": 1.5, "success": True},
            "profile": {"x": np.array([0.0, 1.0])},
        }
        later_shell = self.radial.primary_candidates(
            {"radial_index": 2, "radial_fraction_used": 0.15},
            previous=previous,
            earlier=earlier,
        )
        self.assertEqual(len(later_shell), 5)
        self.assertEqual(
            tuple(candidate["source"] for candidate in later_shell[1:]),
            (self.radial.GUESS_SOURCE_PROFILE_MULTIPLIER,) * 4,
        )


class TimeoutTests(unittest.TestCase):
    def setUp(self):
        self.radial = load_importable_module("_radial_shell_continuation")

    def test_numerical_defaults_bound_each_guess_and_each_cell(self):
        self.assertTrue(NUMERICAL_PATH.exists(), f"missing {NUMERICAL_PATH}")
        numerical = load_module("T0plus0p1_numerical_defaults_test", NUMERICAL_PATH)
        self.assertEqual(numerical.DEFAULT_TRIAL_TIMEOUT_S, 300.0)
        self.assertEqual(numerical.DEFAULT_CELL_TIMEOUT_S, 1200.0)
        self.assertEqual(numerical.DEFAULT_SAVE_EVERY, 1)

    def test_hard_timeout_terminates_stuck_child(self):
        with self.assertRaises(self.radial.BVPTrialTimeoutError):
            self.radial.run_module_call_with_hard_timeout(
                "time", "sleep", args=(0.5,), timeout_s=0.05
            )

    def test_timeout_advances_to_next_jB_candidate(self):
        calls = []

        def fake_solver(kwargs, timeout_s):
            calls.append((kwargs["jB_guess"], timeout_s))
            if len(calls) == 1:
                raise self.radial.BVPTrialTimeoutError("slow guess")
            return successful_result(kwargs["jB_guess"])

        candidates = (
            {"jB_guess": 1.0, "source": "first", "use_profile": False},
            {"jB_guess": 1.2, "source": "second", "use_profile": False},
        )
        with mock.patch.object(
            self.radial, "_solve_uNmax_with_hard_timeout", side_effect=fake_solver
        ):
            output = self.radial.attempt_jB_candidates(
                numerical_task(), candidates, continuation_profile=None
            )

        self.assertTrue(output["selected_record"]["success"])
        self.assertEqual(output["selected_record"]["jB_guess"], 1.2)
        self.assertEqual(len(output["attempt_records"]), 2)
        self.assertEqual(output["attempt_records"][0]["status"], "trial_timeout")

    def test_expired_cell_budget_starts_no_new_candidate(self):
        task = numerical_task()
        task["cell_deadline"] = time.monotonic() - 1.0
        with mock.patch.object(self.radial, "_solve_uNmax_with_hard_timeout") as solver:
            output = self.radial.attempt_jB_candidates(
                task,
                ({"jB_guess": 1.0, "source": "first", "use_profile": False},),
                continuation_profile=None,
            )
        solver.assert_not_called()
        self.assertEqual(output["selected_record"]["status"], "cell_timeout")


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.radial = load_module("T0plus0p1_radial_scheduler_test", RADIAL_PATH)

    def test_points_are_grouped_by_radial_shell_then_theta(self):
        points = [
            {"theta_index": 1, "radial_index": 1, "radial_fraction_used": 0.2},
            {"theta_index": 1, "radial_index": 0, "radial_fraction_used": 0.1},
            {"theta_index": 0, "radial_index": 1, "radial_fraction_used": 0.2},
            {"theta_index": 0, "radial_index": 0, "radial_fraction_used": 0.1},
        ]
        shells = self.radial.group_points_by_radial(points)
        self.assertEqual(list(shells), [0, 1])
        self.assertEqual([p["theta_index"] for p in shells[0]], [0, 1])
        self.assertEqual([p["theta_index"] for p in shells[1]], [0, 1])

    def test_primary_candidates_use_immediate_same_theta_profile(self):
        previous = {
            "point": {"radial_fraction_used": 0.10},
            "record": {"jB": 2.0, "success": True},
            "profile": {"x": np.array([0.0, 1.0])},
        }
        earlier = {
            "point": {"radial_fraction_used": 0.06},
            "record": {"jB": 1.5, "success": True},
            "profile": {"x": np.array([0.0, 1.0])},
        }
        candidates = self.radial.primary_candidates(
            {"radial_index": 2, "radial_fraction_used": 0.15},
            previous=previous,
            earlier=earlier,
        )
        self.assertTrue(candidates)
        self.assertTrue(all(candidate["use_profile"] for candidate in candidates))
        self.assertEqual(candidates[0]["source"], "adjacent_profile_extrapolated_jB")
        self.assertGreater(candidates[0]["jB_guess"], 2.0)

    def test_failed_immediate_predecessor_does_not_reuse_older_profile(self):
        candidates = self.radial.primary_candidates(
            {"radial_index": 4, "radial_fraction_used": 0.30},
            previous=None,
            earlier={
                "point": {"radial_fraction_used": 0.20},
                "record": {"jB": 3.0, "success": True},
                "profile": {"x": np.array([0.0, 1.0])},
            },
        )
        self.assertEqual(candidates, ())

    def test_theta_retry_perturbs_same_shell_median(self):
        candidates = self.radial.theta_retry_candidates(10.0, use_profile=True)
        self.assertEqual(
            tuple(candidate["jB_guess"] for candidate in candidates),
            (10.0, 9.0, 11.0, 12.0),
        )
        self.assertTrue(all(candidate["use_profile"] for candidate in candidates))

    def test_bridge_recovery_uses_midpoint_before_retrying_target(self):
        task = numerical_task()
        task["point"].update(
            {
                "radial_index": 2,
                "radial_fraction": 0.20,
                "radial_fraction_used": 0.20,
                "rho": 1.20,
                "rho_inner_boundary": 1.0,
                "rho_outer_boundary": 2.0,
                "theta_rad": 0.3,
                "theta_deg": np.degrees(0.3),
            }
        )
        task.update(
            {
                "max_bridge_depth": 2,
                "min_bridge_fraction_step": 1.0e-3,
                "previous_entry": {
                    "point": {
                        **task["point"],
                        "radial_index": 1,
                        "radial_fraction": 0.10,
                        "radial_fraction_used": 0.10,
                        "rho": 1.10,
                        "T_0minus_MeV": 18.0,
                        "nB_0minus_over_n0": 3.1,
                        "nB_0minus_MeV3": 39.0,
                    },
                    "record": {"jB": 2.0, "success": True},
                    "profile": {"x": np.array([0.0, 1.0])},
                },
            }
        )
        attempted_temperatures = []

        def fake_solver(kwargs, timeout_s):
            del timeout_s
            attempted_temperatures.append(kwargs["T_0minus"])
            return successful_result(kwargs["jB_guess"])

        with mock.patch.object(
            self.radial, "_solve_uNmax_with_hard_timeout", side_effect=fake_solver
        ):
            output = self.radial.run_bridge_task(task)

        self.assertTrue(output["selected_record"]["success"])
        self.assertGreater(len(output["bridge_records"]), 0)
        self.assertAlmostEqual(attempted_temperatures[0], 19.0)
        self.assertAlmostEqual(attempted_temperatures[-1], 20.0)


class ProgressStreamingTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_module("T0plus0p1_progress_runner_test", RUNNER_PATH)

    def test_successful_cells_finalize_during_stage_collection(self):
        successful_future = Future()
        failed_future = Future()
        successful_future.set_result(
            {"selected_record": {"success": True}, "attempt_records": []}
        )
        failed_future.set_result(
            {"selected_record": {"success": False}, "attempt_records": []}
        )
        successful_point = {"point_index": 1}
        failed_point = {"point_index": 2}
        outputs = {1: [], 2: []}
        finalized = []

        unresolved = self.runner.collect_stage_futures(
            {
                successful_future: successful_point,
                failed_future: failed_point,
            },
            outputs_by_point=outputs,
            finalize_successes=True,
            finalize_all=False,
            on_terminal=lambda point, point_outputs: finalized.append(
                (point["point_index"], len(point_outputs))
            ),
        )

        self.assertEqual(finalized, [(1, 1)])
        self.assertEqual([point["point_index"] for point in unresolved], [2])

    def test_final_stage_finalizes_successes_and_failures_individually(self):
        futures = []
        future_map = {}
        outputs = {}
        for point_index, success in ((3, True), (4, False)):
            future = Future()
            future.set_result(
                {"selected_record": {"success": success}, "attempt_records": []}
            )
            point = {"point_index": point_index}
            futures.append(future)
            future_map[future] = point
            outputs[point_index] = []
        finalized = []

        unresolved = self.runner.collect_stage_futures(
            future_map,
            outputs_by_point=outputs,
            finalize_successes=False,
            finalize_all=True,
            on_terminal=lambda point, point_outputs: finalized.append(
                (point["point_index"], len(point_outputs))
            ),
        )

        self.assertEqual(sorted(finalized), [(3, 1), (4, 1)])
        self.assertEqual(unresolved, [])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_module("T0plus0p1_checkpoint_runner_test", RUNNER_PATH)

    def test_resume_state_preserves_terminal_records_and_profiles(self):
        payload = {
            "radial_shell_records": [
                {"point_index": 3, "success": True},
                {"point_index": 4, "success": False},
            ],
            "radial_shell_attempt_records": [{"point_index": 3, "attempt_index": 0}],
            "radial_shell_bridge_records": [],
            "radial_shell_profiles": {3: {"x": np.array([0.0, 1.0])}},
        }
        state = self.runner.resume_state_from_payload(payload)
        self.assertEqual(set(state["records"]), {3, 4})
        self.assertEqual(set(state["profiles"]), {3})
        self.assertEqual(len(state["attempt_records"]), 1)

    def test_atomic_checkpoint_is_immediately_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npy"
            self.runner.save_payload_checkpoint(
                path,
                {"run_complete": False, "completed_cell_count": 3},
            )
            loaded = np.load(path, allow_pickle=True).item()
        self.assertFalse(loaded["run_complete"])
        self.assertEqual(loaded["completed_cell_count"], 3)

    def test_checkpoint_metadata_declares_profile_granularity(self):
        metadata = self.runner.shell_checkpoint_metadata(
            records={0: {"success": True}, 1: {"success": False}},
            profiles={0: {"x": np.array([0.0, 1.0])}},
            completed_radial_indices={0},
        )
        self.assertEqual(metadata["checkpoint_granularity"], "completed_cell_with_profile")
        self.assertEqual(metadata["completed_cell_count"], 2)
        self.assertEqual(metadata["successful_profile_count"], 1)


if __name__ == "__main__":
    unittest.main()
