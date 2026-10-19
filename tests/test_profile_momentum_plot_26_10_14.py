"""The historical momentum-flux figure remains reproducible and clearly sourced."""

import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CELL_ID = "constant-momentum-profile-reproduction"
CACHE = ROOT / "local_analysis_results/26-10-14/constant-momentum-profile-data.json"


class ProfileMomentumPlotTests(unittest.TestCase):
    def cell(self):
        notebook = nbformat.read(ROOT / "26-10-14.ipynb", as_version=4)
        cells = [cell for cell in notebook.cells if cell.get("id") == CELL_ID]
        self.assertEqual(len(cells), 1, "The reproducible plotting cell is missing")
        return cells[0]

    def test_cached_original_geometry(self):
        self.assertTrue(CACHE.is_file(), "The recovered vector coordinates are missing")
        data = json.loads(CACHE.read_text())
        self.assertEqual(data["provenance"]["original_notebook"], "26-04-09.ipynb")
        self.assertEqual(data["provenance"]["original_cell_index"], 9)
        self.assertEqual(data["provenance"]["recovery_method"], "original PDF vector paths; no raster digitization or new BVP")
        np.testing.assert_array_equal(data["sample_x_nm"], [200, 400, 600, 800])
        self.assertEqual(len(data["contours_sample"]), 4)
        np.testing.assert_allclose(data["profile_path"], data["sample_points"], atol=2.e-5, rtol=0.)
        self.assertAlmostEqual(data["point_inf"][1], 0.)
        self.assertAlmostEqual(data["point_0minus"][1], 0.)
        self.assertLess(data["point_0plus"][0], data["point_inf"][0])
        self.assertLess(data["point_inf"][0], data["point_0minus"][0])

    def test_location_labels_and_local_parameters(self):
        cell = self.cell()
        self.assertTrue(cell.source.startswith("# Tunable plotting parameters"))
        self.assertNotIn("RMFsolver", cell.source)
        self.assertNotIn("rcParams.update", cell.source)
        for label in (r"x=0^+", r"x=\infty", r"x=0^-", r"T(\infty)", r"\Pi(0^-)"):
            self.assertIn(label, cell.source)
        for legacy in ("Qstar", "QSTAR", "Q_star", "Pi_N", "muB_N", "Q\\star", "^N", "{N}"):
            self.assertNotIn(legacy, cell.source)
        config = {}
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                try:
                    config[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        for name in ("TITLE_FONT_SIZE", "AXIS_LABEL_FONT_SIZE", "TICK_LABEL_FONT_SIZE",
                     "LEGEND_FONT_SIZE", "ANNOTATION_FONT_SIZE"):
            self.assertGreaterEqual(config[name], 14)

    def test_cell_renders_from_cache_only(self):
        cell = self.cell()
        tree = ast.parse(cell.source)
        with tempfile.TemporaryDirectory() as temporary:
            exports = {name: str(Path(temporary) / filename) for name, filename in (
                ("OUTPUT_PNG", "figure.png"), ("OUTPUT_PDF", "figure.pdf"),
                ("MEETING_PNG", "meeting.png"))}
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in exports:
                        node.value = ast.Constant(exports[name])
                    elif name == "DATA_FILE":
                        node.value = ast.Constant(str(CACHE))
            with patch.object(plt, "show"), redirect_stdout(io.StringIO()):
                namespace = {}
                exec(compile(ast.fix_missing_locations(tree), "meeting_plot_cell", "exec"), namespace)
            for path in exports.values():
                self.assertGreater(Path(path).stat().st_size, 10000)
            ax = namespace["ax"]
            self.assertTrue(all(tick.get_fontsize() >= 14 for tick in ax.get_xticklabels()))
            plt.close("all")

    def test_meeting_notes_embed_the_result(self):
        text = (ROOT / "26-10-14.md").read_text()
        self.assertIn("26-10-14.assets/constant-momentum-profile.png", text)
        self.assertIn("26-04-09.ipynb", text)
        self.assertIn("vector", text.lower())


if __name__ == "__main__":
    unittest.main()
