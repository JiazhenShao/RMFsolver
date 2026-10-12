import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "new_paper_calculations" / "26-08-27"
RUNNER_PATH = MODULE_DIR / "run_isothermal_numerical.py"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "isothermal_numeric_repair_0827_runner", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    private_module_names = ("_isothermal_domain", "_isothermal_execution")
    saved_private_modules = {
        name: sys.modules.pop(name, None) for name in private_module_names
    }
    sys.path.insert(0, str(MODULE_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(MODULE_DIR))
        for name in private_module_names:
            sys.modules.pop(name, None)
            if saved_private_modules[name] is not None:
                sys.modules[name] = saved_private_modules[name]
    return module


class IsothermalNumericRepairRunnerTests(unittest.TestCase):
    def test_payload_compatibility_ignores_only_absolute_solver_path(self):
        runner = load_runner()
        reference = {
            "run_tag": "repair",
            "B_one_forth_MeV": 189.1565957288247,
            "xi": -0.5,
            "ms_MeV": 0.0,
            "NM_type": "PNM",
            "upB": 5000,
            "coordinate_system": "dimensionless_elliptical_polar",
            "theta_axis_rad": np.asarray([0.0, 1.0]),
            "radial_fraction_axis": np.asarray([0.0, 1.0]),
            "phase_velocity_signatures": {
                "phase_velocity_file": "/cluster/RMFsolver/phase_velocity.py",
                "analytic_velocity_isothermal": "(T, nB)",
                "solve_front_isothermal": "(T, nB, *, scaled=True)",
            },
            "domain_fingerprint": "same-domain",
        }
        candidate = dict(reference)
        candidate["phase_velocity_signatures"] = dict(
            reference["phase_velocity_signatures"]
        )
        candidate["phase_velocity_signatures"]["phase_velocity_file"] = (
            "/local/RMFsolver/phase_velocity.py"
        )

        runner.domain.assert_compatible_payload(reference, candidate)

        candidate["phase_velocity_signatures"]["solve_front_isothermal"] = (
            "(T, nB, *, scaled=False)"
        )
        with self.assertRaisesRegex(RuntimeError, "live API signature"):
            runner.domain.assert_compatible_payload(reference, candidate)

    def test_default_command_reuses_local_completed_payload(self):
        runner = load_runner()
        args = runner.build_parser().parse_args([])

        self.assertEqual(runner.REPO_ROOT, ROOT)
        self.assertEqual(args.domain, MODULE_DIR / "isothermal-domain.npy")
        self.assertEqual(
            args.analytic_seed,
            MODULE_DIR / "isothermal-analytic.npy",
        )
        self.assertEqual(args.output, MODULE_DIR / "isothermal-numerical.npy")

    def test_main_requests_non_success_only_repair(self):
        runner = load_runner()
        domain_payload = {
            "payload_kind": "isothermal_domain",
            "ms_MeV": 0.0,
            "NM_type": "PNM",
            "upB": 5000,
        }
        analytic_payload = {"payload_kind": "isothermal_analytic"}
        numerical_payload = {
            "task_status": np.asarray([["success", "failure"]], dtype=object)
        }

        with (
            patch.object(
                runner.domain,
                "load_payload",
                side_effect=[domain_payload, analytic_payload],
            ),
            patch.object(runner.domain, "assert_domain_fingerprint"),
            patch.object(runner.domain, "validate_cluster_physics"),
            patch.object(
                runner.execution,
                "run_numerical_stage",
                return_value=numerical_payload,
            ) as run_stage,
        ):
            result = runner.main(["--workers", "7"])

        self.assertEqual(result, 0)
        self.assertTrue(run_stage.call_args.kwargs["resume"])
        self.assertTrue(run_stage.call_args.kwargs["retry_non_success"])

    def test_retry_reset_preserves_successes_and_clears_only_failed_cells(self):
        runner = load_runner()
        success_record = {"task_status": "success", "jB": 2.0}
        failed_record = {"task_status": "cell_timeout", "jB": np.nan}
        payload = {
            "task_status": np.asarray(
                [["success", "cell_timeout"], ["failure", "domain_masked"]],
                dtype=object,
            ),
            "records": np.asarray(
                [[success_record, failed_record], [failed_record, failed_record]],
                dtype=object,
            ),
            "u_0minus_grid": np.asarray([[1.0, 9.0], [8.0, 7.0]]),
            "velocity_m_s_grid": np.asarray([[2.0, 9.0], [8.0, 7.0]]),
            "jB_grid": np.asarray([[3.0, 9.0], [8.0, 7.0]]),
            "a_0plus_max_grid": np.asarray([[0.5, 0.9], [0.8, 0.7]]),
            "momentum_flux_ratio_grid": np.asarray(
                [[1.0e-6, 9.0], [8.0, 7.0]]
            ),
            "slow_front_consistent_grid": np.asarray(
                [[True, False], [False, False]], dtype=object
            ),
            "a_0plus_max_status_grid": np.asarray(
                [["interior", "bad"], ["bad", "masked"]], dtype=object
            ),
            "run_complete": True,
            "completed_radial_shells": 20,
        }
        domain_status = np.asarray(
            [["success", "success"], ["success", "failure"]], dtype=object
        )

        reset_count = runner.execution.reset_non_success_cells_for_repair(
            payload, domain_status
        )

        self.assertEqual(reset_count, 2)
        self.assertEqual(payload["task_status"][0, 0], "success")
        self.assertIs(payload["records"][0, 0], success_record)
        self.assertEqual(payload["jB_grid"][0, 0], 3.0)
        self.assertEqual(payload["task_status"][0, 1], "pending")
        self.assertEqual(payload["task_status"][1, 0], "pending")
        self.assertIsNone(payload["records"][0, 1])
        self.assertTrue(np.isnan(payload["jB_grid"][1, 0]))
        self.assertIsNone(payload["slow_front_consistent_grid"][0, 1])
        self.assertIsNone(payload["a_0plus_max_status_grid"][1, 0])
        self.assertEqual(payload["task_status"][1, 1], "domain_masked")
        self.assertFalse(payload["run_complete"])
        self.assertEqual(payload["completed_radial_shells"], 0)


if __name__ == "__main__":
    unittest.main()
