from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = ROOT / "new_paper_calculations" / "26-08-21"
RADIAL_PATH = RUNNER_DIR / "_radial_shell_continuation.py"
RUNNER_PATH = RUNNER_DIR / "_T0plus0p1_runner.py"


def load_module(name: str, path: Path):
    runner_string = str(RUNNER_DIR)
    if runner_string not in sys.path:
        sys.path.insert(0, runner_string)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AngularSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.radial = load_module("t0plus0p1_angular_wavefront", RADIAL_PATH)

    def test_one_sided_seed_uses_only_immediate_neighbor(self):
        seed = self.radial.immediate_theta_seed(
            {2: 10.0, 5: 99.0}, target_theta_index=3
        )
        self.assertEqual(seed["theta_indices"], (2,))
        self.assertEqual(seed["mode"], "one_sided_left")
        self.assertAlmostEqual(seed["jB"], 10.0)

    def test_seed_does_not_jump_across_failed_neighbor(self):
        seed = self.radial.immediate_theta_seed(
            {1: 10.0, 5: 20.0}, target_theta_index=3
        )
        self.assertIsNone(seed)

    def test_two_sided_seed_interpolates_neighbor_currents(self):
        seed = self.radial.immediate_theta_seed(
            {2: 10.0, 4: 14.0}, target_theta_index=3
        )
        self.assertEqual(seed["theta_indices"], (2, 4))
        self.assertEqual(seed["mode"], "two_sided_interpolated")
        self.assertAlmostEqual(seed["jB"], 12.0)

    def test_new_success_exposes_next_wave(self):
        successful = {1: 10.0}
        self.assertIsNotNone(
            self.radial.immediate_theta_seed(successful, target_theta_index=2)
        )
        self.assertIsNone(
            self.radial.immediate_theta_seed(successful, target_theta_index=3)
        )
        successful[2] = 11.0
        seed = self.radial.immediate_theta_seed(successful, target_theta_index=3)
        self.assertEqual(seed["theta_indices"], (2,))

    def test_mock_wavefront_propagates_from_sparse_shell_anchors(self):
        successful = {1: 10.0, 6: 20.0}
        unresolved = set(range(8)) - set(successful)
        wave_count = 0
        while unresolved:
            recovered = {}
            for theta_index in sorted(unresolved):
                seed = self.radial.immediate_theta_seed(
                    successful, target_theta_index=theta_index
                )
                if seed is not None:
                    recovered[theta_index] = float(seed["jB"])
            self.assertTrue(recovered)
            successful.update(recovered)
            unresolved.difference_update(recovered)
            wave_count += 1
        self.assertEqual(set(successful), set(range(8)))
        self.assertGreaterEqual(wave_count, 2)

    def test_angular_candidate_carries_diagnostics(self):
        seed = {
            "jB": 12.0,
            "theta_indices": (2, 4),
            "mode": "two_sided_interpolated",
        }
        candidate = self.radial.angular_wave_candidates(
            seed,
            use_profile=True,
            wave_index=3,
            multipliers=(1.0,),
        )[0]
        self.assertEqual(candidate["source"], "same_shell_theta_interpolated_jB")
        self.assertEqual(candidate["angular_wave_index"], 3)
        self.assertEqual(candidate["angular_seed_theta_indices"], (2, 4))
        self.assertAlmostEqual(candidate["angular_seed_jB"], 12.0)
        self.assertTrue(candidate["use_profile"])

    def test_primary_and_fallback_candidates_are_separated(self):
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
        primary, fallback = self.radial.split_radial_candidates(candidates)
        self.assertEqual(len(primary), 1)
        self.assertEqual(primary[0]["source"], "adjacent_profile_extrapolated_jB")
        self.assertTrue(fallback)
        self.assertTrue(
            all(item["source"] == "adjacent_profile_multiplier_jB" for item in fallback)
        )

    def test_attempt_record_contains_angular_and_budget_diagnostics(self):
        task = {
            "point": {
                "point_index": 1,
                "theta_index": 2,
                "radial_index": 3,
                "radial_fraction": 0.4,
                "radial_fraction_used": 0.4,
                "T_0minus_MeV": 20.0,
                "nB_0minus_MeV3": 30.0,
                "nB_0minus_over_n0": 3.0,
            },
            "T_0plus_MeV": 0.1,
            "trial_timeout_s": 300.0,
            "cell_timeout_s": 300.0,
            "radial_primary_budget_s": 300.0,
            "angular_retry_budget_s": 900.0,
        }
        candidate = self.radial.angular_wave_candidates(
            {
                "jB": 12.0,
                "theta_indices": (1,),
                "mode": "one_sided_left",
            },
            use_profile=True,
            wave_index=4,
        )[0]
        record = self.radial._candidate_record(
            task,
            candidate,
            attempt_index=0,
            started=time.monotonic(),
            status="failure",
            message="mock",
        )
        self.assertEqual(record["angular_wave_index"], 4)
        self.assertEqual(record["angular_seed_theta_indices"], (1,))
        self.assertEqual(record["radial_primary_budget_s"], 300.0)
        self.assertEqual(record["angular_retry_budget_s"], 900.0)


class SchedulerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_module("t0plus0p1_angular_runner", RUNNER_PATH)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("_radial_shell_continuation", None)

    def test_stage_order_prioritizes_angular_recovery(self):
        order = self.runner.CONTINUATION_STAGE_ORDER
        self.assertLess(order.index("angular_direct_wavefront"), order.index("radial_multiplier_fallback"))
        self.assertLess(order.index("angular_perturbed_wavefront"), order.index("radial_bridge_fallback"))

    def test_stage_collection_finalizes_each_success_immediately(self):
        future = Future()
        future.set_result({"selected_record": {"success": True}, "attempt_records": []})
        outputs = {5: []}
        finalized = []
        unresolved = self.runner.collect_stage_futures(
            {future: {"point_index": 5}},
            outputs_by_point=outputs,
            finalize_successes=True,
            finalize_all=False,
            on_terminal=lambda point, point_outputs: finalized.append(point["point_index"]),
        )
        self.assertEqual(finalized, [5])
        self.assertEqual(unresolved, [])

    def test_atomic_checkpoint_is_immediately_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npy"
            self.runner.save_payload_checkpoint(
                path, {"run_complete": False, "completed_cell_count": 1}
            )
            payload = np.load(path, allow_pickle=True).item()
        self.assertFalse(payload["run_complete"])
        self.assertEqual(payload["completed_cell_count"], 1)


class TimeoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.radial = load_module("_radial_shell_continuation", RADIAL_PATH)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("_radial_shell_continuation", None)

    def test_hard_timeout_terminates_stuck_child(self):
        with self.assertRaises(self.radial.BVPTrialTimeoutError):
            self.radial.run_module_call_with_hard_timeout(
                "time", "sleep", args=(0.5,), timeout_s=0.05
            )

    def test_expired_cell_budget_starts_no_new_trial(self):
        task = {
            "point": {
                "point_index": 1,
                "theta_index": 2,
                "radial_index": 3,
                "radial_fraction": 0.4,
                "radial_fraction_used": 0.4,
                "T_0minus_MeV": 20.0,
                "nB_0minus_MeV3": 30.0,
                "nB_0minus_over_n0": 3.0,
            },
            "T_0plus_MeV": 0.1,
            "trial_timeout_s": 300.0,
            "cell_timeout_s": 1200.0,
            "cell_deadline": time.monotonic() - 1.0,
        }
        candidate = {"jB_guess": 12.0, "source": "mock", "use_profile": False}
        with mock.patch.object(self.radial, "_solve_uNmax_with_hard_timeout") as solver:
            output = self.radial.attempt_jB_candidates(
                task, (candidate,), continuation_profile=None
            )
        solver.assert_not_called()
        self.assertEqual(output["selected_record"]["status"], "cell_timeout")


if __name__ == "__main__":
    unittest.main()
