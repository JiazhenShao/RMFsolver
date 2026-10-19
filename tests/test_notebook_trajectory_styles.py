"""Presentation contracts for the manuscript's two trajectory figures."""

import ast
from pathlib import Path
import unittest

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def settings(cell):
    result = {}
    for node in ast.parse(cell.source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


class TrajectoryStyleTests(unittest.TestCase):
    def setUp(self):
        self.notebook = nbformat.read(ROOT / "All_Plots_new.ipynb", as_version=4)
        self.adiabatic = next(
            cell for cell in self.notebook.cells
            if cell.cell_type == "code" and 'OUTPUT_PDF = "Latex_writting/Hydrodynamical combustion/Plots/T_a_constant_pressure.pdf"' in cell.source
        )

    def test_adiabatic_labels(self):
        config = settings(self.adiabatic)
        self.assertEqual(config["ELLIPSE_LABEL"], "Analytic approximation")
        self.assertEqual(config["PROFILE_LABEL"], "Numerical solution")
        self.assertEqual(config["ENDPOINT_LABEL"], r"$A_{\rm adb}$")

    def test_new_isothermal_cell_follows_original(self):
        index = next(i for i, cell in enumerate(self.notebook.cells)
                     if cell.get("id") == "isothermal-matching-trajectory-style")
        self.assertIn("Crossing phase boundary", self.notebook.cells[index - 1].source)
        config = settings(self.notebook.cells[index])
        first = settings(self.adiabatic)
        for name in ("FIGURE_SIZE", "AXIS_LABEL_FONT_SIZE", "TICK_LABEL_FONT_SIZE",
                     "LEGEND_FONT_SIZE", "ANNOTATION_FONT_SIZE", "PROFILE_COLOR",
                     "PROFILE_LINE_WIDTH", "TICK_DIRECTION"):
            self.assertEqual(config[name], first[name], name)
        self.assertEqual(config["ISOBAR_LINE_STYLE"], "--")
        self.assertEqual(config["ISOBAR_COLOR"], first["ELLIPSE_COLOR"])
        self.assertEqual(config["OUTPUT_PDF"],
                         "Latex_writting/Hydrodynamical combustion/Plots/isothermal_muB_muK_plane.pdf")

    def test_figure_text_uses_bag_energy_notation(self):
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                self.assertNotIn('rf"$B^{{1/4}}', cell.source)
        styled = next(cell for cell in self.notebook.cells
                      if cell.get("id") == "isothermal-matching-trajectory-style")
        self.assertIn(r"U_{{\rm bag}}^{{1/4}}", styled.source)


if __name__ == "__main__":
    unittest.main()
