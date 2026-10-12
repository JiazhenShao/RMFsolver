from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "26-07-23.ipynb"


class Notebook26SU9Tests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK_PATH.read_text())
        self.code_sources = [
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
        ]

    def test_loads_the_dated_lte_temperature_jump_payload(self):
        source = "\n".join(self.code_sources)
        self.assertIn("lte-temperature-jump-two-block-linear.npy", source)
        self.assertIn("low_delta_T_interface_MeV_grid", source)
        self.assertIn("high_delta_T_interface_MeV_grid", source)
        self.assertIn("low_lte_jump_closure_status_grid", source)
        self.assertIn("high_lte_jump_closure_status_grid", source)
        self.assertIn("low_lte_isobar_closure_status_grid", source)
        self.assertIn("high_lte_isobar_closure_status_grid", source)
        self.assertIn("A_boundary_exact_local_grid", source)

    def test_diagnostics_check_seam_temperature_and_beta_identities(self):
        source = "\n".join(self.code_sources)
        self.assertIn("seam_max_abs_MeV", source)
        self.assertIn("temperature_identity_max_abs_MeV", source)
        self.assertIn("beta_identity_max_abs", source)
        self.assertIn("composition_residual_max_abs", source)
        self.assertIn("root_failure_points", source)
        self.assertIn("assert", source)

    def test_visualization_cell_exposes_tunable_parameters(self):
        plotting_cells = [
            source
            for source in self.code_sources
            if source.lstrip().startswith("# Tunable plotting parameters")
            and "ax.contour(" in source
        ]
        self.assertEqual(len(plotting_cells), 2)
        for plotting_source in plotting_cells:
            self.assertTrue(
                plotting_source.lstrip().startswith("# Tunable plotting parameters")
            )
            for name in (
                "FIGURE_SIZE",
                "FIGURE_DPI",
                "TITLE_FONT_SIZE",
                "AXIS_LABEL_FONT_SIZE",
                "TICK_LABEL_FONT_SIZE",
                "CONTOUR_LEVELS_MEV",
                "CONTOUR_CMAP",
                "CONTOUR_LINE_WIDTH",
                "CONTOUR_LABEL_FONT_SIZE",
                "ROOT_FAILURE_MARKER_SIZE",
                "X_LIMITS",
                "Y_LIMITS",
                "EXPORT_PDF",
                "EXPORT_PNG",
            ):
                self.assertIn(name, plotting_source)

    def test_notebook_has_delta_T_and_T_plus_contour_plots(self):
        contour_cells = [
            source
            for source in self.code_sources
            if source.lstrip().startswith("# Tunable plotting parameters")
            and "ax.contour(" in source
        ]
        self.assertTrue(
            any(
                "low_delta_T" in source and "high_delta_T" in source
                for source in contour_cells
            )
        )
        self.assertTrue(
            any(
                "low_T_Qstar" in source and "high_T_Qstar" in source
                for source in contour_cells
            )
        )
        self.assertTrue(
            any("T(0^-)-T(0^+)" in source for source in contour_cells)
        )
        self.assertTrue(
            any("T(0^+)" in source for source in contour_cells)
        )

    def test_visualization_uses_native_blocks_and_saved_boundaries(self):
        source = "\n".join(self.code_sources)
        self.assertNotIn("griddata", source)
        self.assertNotIn("triang", source.lower())
        self.assertIn("A_boundary_temperature_values", source)
        self.assertIn("gap_boundary_temperature_values", source)
        self.assertIn("root_failure_mask", source)
        self.assertIn("np.ma.masked_invalid", source)
        self.assertNotIn("ax.contourf", source)
        self.assertIn("ax.contour", source)
        self.assertIn("ax.clabel", source)
        self.assertIn("fill_between", source)
        self.assertIn("fill_betweenx", source)


if __name__ == "__main__":
    unittest.main()
