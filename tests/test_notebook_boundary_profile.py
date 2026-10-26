"""The boundary overview must plot the saved numerical profile, not a sketch."""

import ast
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CELL_ID = "adiabatic-boundary-profile"


class BoundaryProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = nbformat.read(ROOT / "All_Plots_new.ipynb", as_version=4)
        matches = [cell for cell in notebook.cells if cell.get("id") == CELL_ID]
        assert len(matches) == 1, "Missing numerical boundary-profile plotting cell"
        cls.source = matches[0].source
        cls.directory = tempfile.TemporaryDirectory()
        tree = ast.parse(cls.source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ("OUTPUT_PDF", "OUTPUT_PNG"):
                    suffix = ".pdf" if name == "OUTPUT_PDF" else ".png"
                    node.value = ast.Constant(str(Path(cls.directory.name) / ("profile" + suffix)))
                elif name == "PROFILE_CSV":
                    node.value = ast.Constant(str(ROOT / ast.literal_eval(node.value)))
        cls.scope = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.fix_missing_locations(tree), CELL_ID, "exec"), cls.scope)

    @classmethod
    def tearDownClass(cls):
        plt.close(cls.scope["fig"])
        cls.directory.cleanup()

    def test_four_balanced_open_axes(self):
        axes = self.scope["axes"]
        self.assertEqual(len(axes), 4)
        for axis in axes:
            self.assertFalse(axis.spines["top"].get_visible())
            self.assertFalse(axis.spines["right"].get_visible())
            self.assertGreater(axis.bbox.width / axis.bbox.height, 3)
            self.assertLess(axis.bbox.width / axis.bbox.height, 6)

    def test_density_and_speed_include_zero(self):
        for axis in self.scope["axes"][:2]:
            self.assertLessEqual(axis.get_ylim()[0], 0.0)
            self.assertIn(0.0, axis.get_yticks())

    def test_forty_sixty_split_and_negative_ticks(self):
        for axis in self.scope["axes"]:
            left, right = axis.get_xlim()
            self.assertAlmostEqual(-left / (right - left), 0.4)
            self.assertTrue(np.any(axis.get_xticks() < 0.0))
        labels = [label.get_text() for label in self.scope["axes"][-1].get_xticklabels()]
        self.assertTrue(any(label.startswith(("-", "\N{MINUS SIGN}")) for label in labels))

    def test_phase_headers_stay_in_their_regions(self):
        self.scope["fig"].canvas.draw()
        axis = self.scope["axes"][0]
        renderer = self.scope["fig"].canvas.get_renderer()
        interface_x = axis.transData.transform((0.0, 0.0))[0]
        headings = {text.get_text(): text for text in axis.texts}
        neutron = headings["Neutron matter"].get_window_extent(renderer)
        quark = headings["Quark matter"].get_window_extent(renderer)
        self.assertLess(neutron.x1, interface_x)
        self.assertGreater(quark.x0, interface_x)
        self.assertGreaterEqual(neutron.x0, axis.bbox.x0)
        self.assertLessEqual(quark.x1, axis.bbox.x1)

    def test_curves_equal_saved_values_without_rescaling_shapes(self):
        s = self.scope
        for field, scale, line, jump in zip(
            s["PROFILE_FIELDS"], s["field_scales"], s["profile_lines"], s["jump_lines"]
        ):
            np.testing.assert_array_equal(line.get_xdata(), s["x_nm"])
            np.testing.assert_array_equal(line.get_ydata(), s["profile"][field] * scale)
            np.testing.assert_array_equal(jump.get_xdata(), [0.0, 0.0])
            np.testing.assert_array_equal(jump.get_ydata(),
                                          [s["upstream"][field] * scale,
                                           s["profile"][field][0] * scale])
        self.assertGreater(s["profile"]["a"][-1], 0.0)

    def test_tuning_block_and_no_legacy_endpoint_identifiers(self):
        self.assertTrue(self.source.startswith("# Tunable plotting parameters"))
        self.assertNotIn("RMFsolver", self.source)
        for node in ast.walk(ast.parse(self.source)):
            if isinstance(node, ast.Name):
                self.assertFalse(node.id.endswith(("_N", "_Qstar", "_Q")), node.id)

    def test_exports(self):
        for name in ("OUTPUT_PDF", "OUTPUT_PNG"):
            self.assertGreater(Path(self.scope[name]).stat().st_size, 1000)

    def test_fonts_match_ten_point_manuscript_at_column_width(self):
        s = self.scope
        s["fig"].canvas.draw()
        bounds = s["fig"].get_tightbbox(s["fig"].canvas.get_renderer())
        exported_width_points = (bounds.width + 2 * s["EXPORT_PAD_INCHES"]) * 72
        column_width_points = 245.09
        for name in ("AXIS_LABEL_FONT_SIZE", "TICK_LABEL_FONT_SIZE",
                     "ANNOTATION_FONT_SIZE", "PHASE_LABEL_FONT_SIZE"):
            self.assertGreaterEqual(s[name] * column_width_points / exported_width_points,
                                    10.0, name)
        self.assertIn("Plots/boundary_schematic_real.pdf", self.source)


if __name__ == "__main__":
    unittest.main()
