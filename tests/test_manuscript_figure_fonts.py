"""One font standard for every manuscript figure made by All_Plots_new.ipynb (user, 2026-10-30).

Main.tex (revtex4-1, 10pt, twocolumn) includes each figure at \\linewidth = 246 pt, so a font prints at
size * 246 / (72 * exported width in inches). The standard, stated at the 7.2 in width of the contour
figures: 22 pt text (ticks, legends, region text, annotations), 24 pt axis labels and titles, and 13 pt
contour labels (the adiabatic contour cell's 11 pt plus 2). Other widths scale in proportion.
"""
import ast
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "All_Plots_new.ipynb"
PLOTS = ROOT / "Latex_writting/Hydrodynamical combustion/Plots"
SCIPY_RUNTIME = ROOT / "local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64"
COLUMN_PT = 246.0
REFERENCE_WIDTH_IN = 7.2
TEXT_PT, LABEL_PT, CONTOUR_PT = 22, 24, 13   # at REFERENCE_WIDTH_IN
X_LABEL = "Nuclear baryon number density [$n_0$]"
Y_LABEL = "Nuclear phase temperature [MeV]"
ISOTHERMAL_TAG = "isothermal-contours-26-10-27"
ADIABATIC_CONTOUR_IDS = ("a1d69a14", "t0plus0p1-three-method")


def cells():
    import nbformat
    return nbformat.read(NOTEBOOK, as_version=4).cells


def constants(source):
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return found


def contour_cells():
    """Every contour figure cell: the two adiabatic ones and the tagged isothermal figure cells."""
    chosen = [c for c in cells() if c.cell_type == "code" and (c.get("id") in ADIABATIC_CONTOUR_IDS or (
        ISOTHERMAL_TAG in c.metadata.get("tags", []) and "FIGURE_SIZE" in c.source))]
    assert len(chosen) == 6, len(chosen)
    return chosen


def font_role(name):
    if name.endswith("CONTOUR_LABEL_FONT_SIZE"):
        return CONTOUR_PT
    if name.endswith(("AXIS_LABEL_FONT_SIZE", "TITLE_FONT_SIZE")):
        return LABEL_PT
    return TEXT_PT


def printed_sizes(pdf):
    """Printed font sizes (pt) of a figure PDF placed at the manuscript column width."""
    data = pdf.read_bytes()
    box = [float(v) for v in re.search(rb"/MediaBox\s*\[\s*([\d.\s-]+)\]", data).group(1).split()]
    scale = COLUMN_PT / (box[2] - box[0])
    sizes = set()
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        try:
            text = zlib.decompress(match.group(1))
        except zlib.error:
            continue
        sizes |= {round(float(size) * scale, 2) for size in re.findall(rb"/\w+\s+([\d.]+)\s+Tf", text)}
    return sizes


def prepare_rendering():
    import matplotlib
    matplotlib.use("Agg")
    if str(SCIPY_RUNTIME) not in sys.path and SCIPY_RUNTIME.is_dir():
        sys.path.insert(0, str(SCIPY_RUNTIME))


def render(*sources):
    """Run notebook cells in one namespace with savefig and show disabled; the current figure after each."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    figures = []
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        with patch.object(Figure, "savefig"), patch.object(plt, "show"):
            scope = {"__name__": "__figure_test__"}
            for source in sources:
                exec(compile(source, "<cell>", "exec"), scope)
                figures.append(plt.gcf())
    finally:
        os.chdir(cwd)
    return figures


def region_gray_fractions(figure):
    """Fraction of each gray-region name's box that lies on gray background, drawn with all texts hidden."""
    axis = figure.axes[0]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    regions = {t.get_text(): t.get_window_extent(renderer) for t in axis.texts if "favored" in t.get_text()}
    visible = [t for t in axis.texts if t.get_visible()]
    for text in visible:
        text.set_visible(False)
    figure.canvas.draw()
    pixels = np.asarray(figure.canvas.buffer_rgba())[..., :3].astype(int)
    for text in visible:
        text.set_visible(True)
    fractions = {}
    for name, box in regions.items():
        patch_ = pixels[int(pixels.shape[0] - box.y1):int(pixels.shape[0] - box.y0) + 1,
                        int(box.x0):int(box.x1) + 1]
        gray = (np.ptp(patch_, axis=-1) < 4) & (patch_[..., 0] > 150) & (patch_[..., 0] < 215)
        fractions[name] = gray.mean()
    return fractions


def isothermal_cells():
    return [c for c in cells() if c.cell_type == "code" and ISOTHERMAL_TAG in c.metadata.get("tags", [])]


def isothermal_load_source(tagged):
    return next(c.source for c in tagged if "iso_payloads" in c.source and "FIGURE_SIZE" not in c.source)


def allowed_printed_sizes():
    scale = COLUMN_PT / (72.0 * REFERENCE_WIDTH_IN)
    base = [TEXT_PT * scale, LABEL_PT * scale]
    return base + [CONTOUR_PT * scale] + [0.7 * size for size in base]   # 0.7: mathtext scripts


class ContourCellTests(unittest.TestCase):
    def test_contour_cells_share_the_font_standard(self):
        checked = 0
        for cell in contour_cells():
            found = constants(cell.source)
            width = next(v for k, v in found.items() if k.endswith("FIGURE_SIZE"))[0]
            for name, value in found.items():
                if name.endswith("FONT_SIZE"):
                    checked += 1
                    with self.subTest(font=name):
                        self.assertAlmostEqual(value * REFERENCE_WIDTH_IN / width, font_role(name), places=6)
        self.assertGreaterEqual(checked, 30)

    def test_contour_cells_use_nuclear_phase_axis_labels(self):
        for cell in contour_cells():
            found = constants(cell.source)
            for suffix, expected in (("X_LABEL", X_LABEL), ("Y_LABEL", Y_LABEL)):
                names = [k for k in found if k.endswith(suffix)]
                self.assertEqual(len(names), 1, (suffix, names))
                with self.subTest(name=names[0]):
                    self.assertEqual(found[names[0]], expected)


class ManuscriptPdfTests(unittest.TestCase):
    STANDARD = ("Numerical-contour.pdf", "Contour-diff-meth-0.1-band.pdf", "isothermal-numerical-contour.pdf",
                "isothermal-numerical-contour-logT.pdf", "isothermal-analytic-numerical-logT.pdf",
                "isothermal-analytical-numerical-comparison.pdf", "T_a_constant_pressure.pdf",
                "isothermal_muB_muK_plane.pdf")

    def assert_standard(self, name):
        allowed = allowed_printed_sizes()
        sizes = printed_sizes(PLOTS / name)
        self.assertTrue(sizes, name)
        for size in sorted(sizes):
            with self.subTest(figure=name, printed=size):
                self.assertLess(min(abs(size - a) for a in allowed), 0.15, allowed)

    def test_manuscript_pdfs_print_at_the_standard_sizes(self):
        for name in self.STANDARD:
            self.assert_standard(name)

    @unittest.expectedFailure
    def test_boundary_profile_pdf_prints_at_the_standard_sizes(self):
        # Prints 11.1 pt throughout: its cell refuses to run until the matched-profile solver cell is
        # rerun with the current RMFsolver (provenance guard), so it has not been resized yet.
        self.assert_standard("boundary_schematic_real.pdf")


class RegionLabelTests(unittest.TestCase):
    """The gray-region names of the linear contour figures sit wholly on gray background."""

    @classmethod
    def setUpClass(cls):
        prepare_rendering()
        cls.cells = cells()

    def assert_on_gray(self, *sources):
        import matplotlib.pyplot as plt
        fractions = region_gray_fractions(render(*sources)[-1])
        plt.close("all")
        self.assertEqual(len(fractions), 2)
        for name, fraction in fractions.items():
            with self.subTest(text=name):
                self.assertEqual(fraction, 1.0)

    def test_adiabatic_numerical_region_labels(self):
        self.assert_on_gray(next(c.source for c in self.cells if c.get("id") == ADIABATIC_CONTOUR_IDS[0]))

    def test_adiabatic_comparison_region_labels(self):
        self.assert_on_gray(next(c.source for c in self.cells if c.get("id") == ADIABATIC_CONTOUR_IDS[1]))

    def test_isothermal_linear_region_labels(self):
        tagged = isothermal_cells()
        linear = next(c.source for c in tagged if "ISO_LIN_FIGURE_SIZE" in c.source)
        self.assert_on_gray(isothermal_load_source(tagged), linear)


class IsothermalLogPanelTests(unittest.TestCase):
    """Fig. isothermal-contour-compare takes the shape of the numerical log-T panel (user, 2026-10-30)."""

    @classmethod
    def setUpClass(cls):
        prepare_rendering()
        tagged = isothermal_cells()
        panels = [next(c.source for c in tagged if f"{prefix}FIGURE_SIZE" in c.source)
                  for prefix in ("ISO_LOGN_", "ISO_CMP_")]
        cls.numerical, cls.comparison = render(isothermal_load_source(tagged), *panels)[1:]
        for figure in (cls.numerical, cls.comparison):
            figure.canvas.draw()

    @classmethod
    def tearDownClass(cls):
        import matplotlib.pyplot as plt
        plt.close("all")

    @staticmethod
    def inches(figure, artist):
        box = artist.get_window_extent(figure.canvas.get_renderer())
        return box.transformed(figure.dpi_scale_trans.inverted())

    def test_comparison_axes_box_matches_the_numerical_panel(self):
        numerical, comparison = self.numerical.axes[0], self.comparison.axes[0]
        expected, actual = self.inches(self.numerical, numerical), self.inches(self.comparison, comparison)
        self.assertEqual(self.comparison.get_size_inches()[0], self.numerical.get_size_inches()[0])
        self.assertAlmostEqual(actual.width, expected.width, delta=0.01)
        self.assertAlmostEqual(actual.height, expected.height, delta=0.01)
        self.assertEqual(comparison.get_xlim(), numerical.get_xlim())
        self.assertEqual(list(comparison.get_xticks()), list(numerical.get_xticks()))

    def test_comparison_legend_sits_between_the_axes_and_the_title(self):
        axis = self.comparison.axes[0]
        legend, box = self.inches(self.comparison, axis.get_legend()), self.inches(self.comparison, axis)
        self.assertGreaterEqual(legend.y0, box.y1)
        self.assertLessEqual(legend.y1, self.inches(self.comparison, axis.title).y0)
        self.assertGreaterEqual(legend.x0, 0.0)
        self.assertLessEqual(legend.x1, self.comparison.get_size_inches()[0])

    def test_log_panel_region_labels_on_gray(self):
        for name, figure in (("numerical", self.numerical), ("comparison", self.comparison)):
            fractions = region_gray_fractions(figure)
            self.assertEqual(len(fractions), 2, name)
            for text, fraction in fractions.items():
                with self.subTest(figure=name, text=text):
                    self.assertEqual(fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
