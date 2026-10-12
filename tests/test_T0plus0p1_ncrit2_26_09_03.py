import importlib.util
import sys
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).parents[1]
RUNNER_DIR = ROOT / "new_paper_calculations" / "26-09-03" / "Full_scan"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NcritTwoContourRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(RUNNER_DIR))
        cls.engine = load_module("ncrit2_method_engine", RUNNER_DIR / "_method_engine.py")
        cls.prepare = load_module("ncrit2_prepare", RUNNER_DIR / "prepare_T0plus0p1_domain.py")
        cls.runner = load_module(
            "ncrit2_runner", RUNNER_DIR / "_T0plus0p1_runner.py"
        )
        cls.numerical = load_module(
            "ncrit2_numerical", RUNNER_DIR / "run_T0plus0p1_numerical.py"
        )

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(RUNNER_DIR))

    def test_default_calibration_is_two_n0_and_explicitly_forwarded(self):
        self.assertEqual(self.engine.DEFAULT_NCRIT_OVER_N0, 2.0)
        self.assertEqual(self.prepare.DEFAULT_NCRIT_OVER_N0, 2.0)
        arguments = self.prepare.default_arguments()
        index = arguments.index("--ncrit-over-n0")
        self.assertEqual(float(arguments[index + 1]), 2.0)

    def test_new_domain_and_numerical_outputs_do_not_reuse_three_n0_artifacts(self):
        expected_domain = "T0plus0p1-ncrit2-domain.npy"
        expected_numerical = "T0plus0p1-ncrit2-uNmax-angular-wavefront.npy"
        self.assertEqual(self.prepare.DEFAULT_OUTPUT, expected_domain)
        self.assertEqual(self.numerical.DEFAULT_DOMAIN_FILE, expected_domain)
        self.assertEqual(self.numerical.DEFAULT_OUTPUT, expected_numerical)
        self.assertIn("ncrit2", self.prepare.DEFAULT_RUN_TAG)
        self.assertIn("ncrit2", self.numerical.DEFAULT_RUN_TAG)

    def test_boundary_defaults_use_dense_endpoint_inclusive_angles(self):
        self.assertEqual(self.prepare.DEFAULT_BOUNDARY_THETA_COUNT, 30)
        theta = self.prepare.contour_theta_axis(
            self.prepare.DEFAULT_BOUNDARY_THETA_COUNT
        )
        self.assertEqual(theta.size, 30)
        self.assertEqual(theta[0], 0.0)
        self.assertEqual(theta[-1], 0.5 * np.pi)

    def test_boundary_tasks_use_FL26_3_coupled_roots(self):
        submitted = []

        class ImmediateExecutor:
            def submit(self, function, task):
                submitted.append(dict(task))
                future = Future()
                future.set_result(function(task))
                return future

        def fake_task(task):
            rho = 0.25 if task["condition"] == "muK" else 0.75
            nB_over_n0, temperature = task["transform"].to_physical(
                rho, task["theta_rad"]
            )
            return {
                "success": True,
                "task": task,
                "completed": {
                    "record": {
                        "rho": rho,
                        "nB_over_n0": float(nB_over_n0),
                        "T_MeV": float(temperature),
                        "boundary_kind": (
                            "Delta_h_equals_zero"
                            if task["condition"] == "muK"
                            else "A_equals_one"
                        ),
                    }
                },
                "message": "",
            }

        transform = self.runner.benchmark.ray_grid.RayTransform(1.0, 4.5, 120.0)
        with mock.patch.object(
            self.runner, "_solve_direct_boundary_task", side_effect=fake_task
        ):
            boundary = self.runner.solve_direct_common_boundary(
                transform=transform,
                theta_count=3,
                density_min_over_n0=1.0,
                density_max_over_n0=5.5,
                temperature_min_MeV=0.01,
                temperature_max_MeV=120.0,
                n0=1.0,
                B14=176.5,
                xi=-0.5,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
                root_tolerance=2.0e-7,
                max_nfev=220,
                T_values=(0.1,),
                executor=ImmediateExecutor(),
            )

        self.assertEqual(len(submitted), 6)
        self.assertTrue(all("coordinates" not in task for task in submitted))
        stable_tasks = [task for task in submitted if task["condition"] == "muK"]
        A_tasks = [task for task in submitted if task["condition"] == "ns"]
        self.assertEqual(len(stable_tasks), 3)
        self.assertEqual(len(A_tasks), 3)
        self.assertTrue(all(task["T_0plus_MeV"] == 0.1 for task in stable_tasks))
        self.assertTrue(
            all(task["T_0plus_MeV"] == 1.0e-3 for task in A_tasks)
        )
        self.assertEqual(boundary["construction"], "FL26-3_coupled_interface_roots")
        np.testing.assert_allclose(boundary["rho_inner"], 0.25)
        np.testing.assert_allclose(boundary["rho_outer"], 0.75)

    def test_boundary_worker_calls_direct_interface_root_not_scan_classifier(self):
        expected = {
            "rho": 0.4,
            "nB_over_n0": 2.0,
            "T_MeV": 30.0,
            "boundary_kind": "Delta_h_equals_zero",
        }
        with mock.patch.object(
            self.runner,
            "_solve_direct_interface_root",
            return_value=expected,
            create=True,
        ) as direct, mock.patch.object(
            self.runner.benchmark.ray_grid,
            "_solve_boundary_task",
            side_effect=AssertionError("scan classifier must not be called"),
        ):
            result = self.runner._solve_direct_boundary_task(
                {"condition": "muK", "theta_rad": 0.2}
            )
        self.assertTrue(result["success"])
        self.assertEqual(result["completed"]["record"], expected)
        direct.assert_called_once()


if __name__ == "__main__":
    unittest.main()
