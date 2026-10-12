import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "new_paper_calculations" / "26-08-12"
RUNNER_PATH = SCRIPT_DIR / "_T0plus5_runner.py"
PREPARE_PATH = SCRIPT_DIR / "prepare_T0plus5_domain.py"
ALL_PATH = SCRIPT_DIR / "run_T0plus5_all.py"


def load_module(name, path):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ZeroUpstreamRayTests(unittest.TestCase):
    def test_contour_theta_axis_includes_zero_temperature_ray(self):
        radial = load_module("T0plus5_zero_upstream_radial", SCRIPT_DIR / "_radial_continuation.py")
        theta = radial.inclusive_theta_axis(30)
        self.assertEqual(theta[0], 0.0)
        self.assertEqual(theta[-1], 0.5 * np.pi)

    def test_boundary_theta_zero_uses_exact_zero_temperature(self):
        runner = load_module("T0plus5_zero_upstream_runner", RUNNER_PATH)
        theta_axis = np.linspace(0.0, 0.5 * np.pi, 12)
        path = runner._direct_boundary_path(
            theta_index=0,
            theta_axis=theta_axis,
            density_min_over_n0=1.0,
            density_max_over_n0=5.5,
            temperature_min_MeV=1.0e-2,
            temperature_max_MeV=120.0,
            ray_probe_count=15,
            transform=None,
        )
        self.assertEqual(path["path_mode"], "fixed_temperature")
        self.assertEqual(path["fixed_value"], 0.0)

    def test_launchers_use_fresh_zero_upstream_tags(self):
        prepare = load_module("T0plus5_zero_upstream_prepare", PREPARE_PATH)
        self.assertIn("zero_upstream", prepare.DEFAULT_RUN_TAG)
        self.assertEqual(prepare.DEFAULT_CONTOUR_RAY_COUNT, 30)
        self.assertEqual(len(prepare.DEFAULT_RADIAL_FRACTIONS), 20)


class IntegratedLauncherTests(unittest.TestCase):
    def test_integrated_launcher_runs_the_four_stages_sequentially(self):
        launcher = load_module("T0plus5_integrated_launcher", ALL_PATH)
        calls = []

        def output_argument(argv):
            return Path(argv[argv.index("--output") + 1])

        def fake_prepare(argv):
            calls.append(("domain", list(argv)))
            output_argument(argv).touch()
            return 0

        def fake_analytical(method, argv):
            calls.append((method, list(argv)))
            output_argument(argv).touch()
            return 0

        def fake_numerical(argv):
            calls.append(("numerical", list(argv)))
            output_argument(argv).touch()
            return 0

        launcher.prepare_domain_main = fake_prepare
        launcher.analytical_main = fake_analytical
        launcher.numerical_main = fake_numerical
        with tempfile.TemporaryDirectory() as directory:
            result = launcher.main(
                ["--output-directory", directory, "--workers", "7"]
            )

        self.assertEqual(result, 0)
        self.assertEqual([entry[0] for entry in calls], [
            "domain", "analytic", "semi_analytic", "numerical"
        ])
        for _, argv in calls:
            self.assertEqual(argv[argv.index("--workers") + 1], "7")
        semi_output = calls[2][1][calls[2][1].index("--output") + 1]
        numerical_args = calls[3][1]
        self.assertEqual(
            numerical_args[numerical_args.index("--semi-results") + 1],
            semi_output,
        )

    def test_integrated_output_bundle_uses_one_shared_rerun_suffix(self):
        launcher = load_module("T0plus5_integrated_output_bundle", ALL_PATH)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            initial = launcher.output_bundle(directory, "T0plus5-all")
            initial["domain"].touch()
            rerun = launcher.output_bundle(directory, "T0plus5-all")

        self.assertTrue(all("__run01" in path.name for path in rerun.values()))
        self.assertEqual(len({path.parent for path in rerun.values()}), 1)


if __name__ == "__main__":
    unittest.main()
