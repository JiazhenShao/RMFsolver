import json
import unittest
from pathlib import Path


class BoundaryCalibrationNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).parents[1] / "26-09-03.ipynb"
        cls.notebook = json.loads(cls.path.read_text())
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_notebook_exists_and_is_compact(self):
        self.assertTrue(self.path.exists())
        notebook = self.notebook
        self.assertEqual(notebook["nbformat"], 4)
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        self.assertLessEqual(len(sources), 5)
        self.assertLess(sum(len(source.splitlines()) for source in sources), 250)

    def test_bag_calibration_is_tunable_and_defaults_to_two_n0(self):
        source = self.source
        self.assertIn("NB_CRIT_0_OVER_N0 = 2.0", source)
        self.assertIn("calibrate_b14", source)
        self.assertIn("B14_MeV", source)

    def test_boundaries_are_scalar_roots_on_hidden_rays(self):
        source = self.source
        self.assertIn("THETA_AXIS = np.linspace(0.0, 0.5*np.pi", source)
        self.assertIn("def physical_point", source)
        self.assertIn("def solve_ray_root", source)
        self.assertIn('condition="muK"', source)
        self.assertIn('condition="ns"', source)
        self.assertIn("least_squares", source)

    def test_A_boundary_uses_exact_local_fraction_and_reports_ns(self):
        source = self.source
        self.assertIn('"n_s_over_nB"', source)
        self.assertIn('"a_0plus"', source)
        self.assertIn("nK_0plus/nB_0plus", source)

    def test_no_velocity_or_bvp_solve_is_used(self):
        source = self.source
        self.assertNotIn("solve_front", source)
        self.assertNotIn("solve_bvp", source)
        self.assertNotIn("semi_analytic_velocity_bound", source)

    def test_visualization_has_tunable_parameter_block(self):
        notebook = self.notebook
        plot_source = "".join(notebook["cells"][-1]["source"])
        self.assertTrue(plot_source.startswith("# Tunable plotting parameters"))
        for name in (
            "FIGURE_SIZE",
            "TITLE_FONT_SIZE",
            "AXIS_LABEL_FONT_SIZE",
            "TICK_LABEL_FONT_SIZE",
            "LINE_WIDTH",
            "MARKER_SIZE",
        ):
            self.assertIn(name, plot_source)


if __name__ == "__main__":
    unittest.main()
