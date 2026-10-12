import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "new_paper_calculations" / "26-08-25" / "isothermal_analytic"


def load_domain():
    path = MODULE_DIR / "_isothermal_domain.py"
    spec = importlib.util.spec_from_file_location("isothermal_analytic_0825_domain", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class IsothermalAnalyticAngularGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain = load_domain()

    def test_boundary_axis_has_twice_the_baseline_ray_count(self):
        domain = self.domain
        theta_deg = np.degrees(domain.default_boundary_theta_axis_rad())

        self.assertEqual(domain.DEFAULT_BOUNDARY_THETA_COUNT, 24)
        self.assertEqual(theta_deg.shape, (24,))
        self.assertEqual(theta_deg[0], 0.0)
        self.assertAlmostEqual(theta_deg[-1], 90.0)
        self.assertTrue(np.all(np.diff(theta_deg) > 0.0))

        parameter = np.linspace(0.0, 1.0, 24)
        expected = np.degrees(
            0.25 * np.pi * (1.0 - np.cos(np.pi * parameter))
        )
        np.testing.assert_allclose(theta_deg, expected)

    def test_contour_axis_is_logarithmic_then_linear_with_one_seam(self):
        domain = self.domain
        theta_deg = np.degrees(domain.default_theta_axis_rad())

        self.assertEqual(theta_deg.shape, (30,))
        self.assertTrue(np.all(np.diff(theta_deg) > 0.0))
        np.testing.assert_allclose(
            theta_deg[[0, 1, 15, -1]],
            [0.0, 0.01, 20.0, 90.0],
        )
        self.assertEqual(np.count_nonzero(np.isclose(theta_deg, 20.0)), 1)
        np.testing.assert_allclose(
            np.diff(np.log(theta_deg[1:16])),
            np.diff(np.log(theta_deg[1:16]))[0],
        )
        np.testing.assert_allclose(
            np.diff(theta_deg[15:]),
            np.diff(theta_deg[15:])[0],
        )

    def test_piecewise_axis_keeps_the_600_cell_mesh(self):
        domain = self.domain
        theta = domain.default_theta_axis_rad()
        radial = domain.default_radial_fraction_axis()
        mesh = domain.build_safe_ray_mesh(
            transform=domain.RayTransform(),
            theta_axis_rad=theta,
            radial_fraction_axis=radial,
            rho_inner=np.full(theta.shape, 0.2),
            rho_outer=np.full(theta.shape, 0.8),
        )

        self.assertEqual(mesh["rho_grid"].shape, (30, 20))
        self.assertEqual(mesh["rho_grid"].size, 600)

    def test_schema_rejects_pre_piecewise_payloads(self):
        self.assertEqual(self.domain.SCHEMA_VERSION, 4)
        self.assertIn("piecewise-theta", self.domain.RUN_TAG)


if __name__ == "__main__":
    unittest.main()
