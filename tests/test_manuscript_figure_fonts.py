"""One font standard for every manuscript figure made by All_Plots_new.ipynb (user, 2026-10-30, 2026-11-03).

Main.tex (revtex4-1, 10pt, twocolumn) includes each figure at \\linewidth = 246 pt, so a font prints at
size * 246 / (72 * exported width in inches). The standard, stated at the 7.2 in width of the contour
figures: 22 pt text (ticks, legends, region text, annotations) and 13 pt contour labels (the adiabatic
contour cell's 11 pt plus 2); other widths scale in proportion. Axis labels, titles, the gray-region
names and the contour figures' legends print at the 10 pt body size in the body font, Computer Modern Roman
(cmr10 text, mathtext cm math):
21.0732 pt at 7.2 in, and on bbox-tight figures whatever the exported page width requires. Every
analytical-versus-numerical comparison uses xi = 0 (user, 2026-11-03).
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
TEXT_PT, CONTOUR_PT = 22, 13   # at REFERENCE_WIDTH_IN
BODY_PT = 10.0   # Main.tex body text; axis labels and titles print at this size
LABEL_FONT_FAMILY = "cmr10"   # the body font, Computer Modern Roman
COMPUTER_MODERN = ("Cmr10", "Cmmi10", "Cmsy10", "Cmex10")
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


def is_body_font(name):
    return name.endswith(("AXIS_LABEL_FONT_SIZE", "TITLE_FONT_SIZE", "REGION_TEXT_FONT_SIZE", "LEGEND_FONT_SIZE"))


def font_role(name):
    return CONTOUR_PT if name.endswith("CONTOUR_LABEL_FONT_SIZE") else TEXT_PT


def printed_runs(pdf):
    """(font, printed size in pt) of every text run of a figure PDF placed at the manuscript column width."""
    data = pdf.read_bytes()
    box = [float(v) for v in re.search(rb"/MediaBox\s*\[\s*([\d.\s-]+)\]", data).group(1).split()]
    scale = COLUMN_PT / (box[2] - box[0])
    objects = {int(number): body for number, body in re.findall(rb"(\d+) 0 obj(.*?)endobj", data, re.S)}
    fonts = {}
    for name, number in re.findall(rb"/(F\d+) (\d+) 0 R", data):
        base = re.search(rb"/BaseFont\s*/(?:[A-Z]{6}\+)?([\w-]+)", objects.get(int(number), b""))
        if base:
            fonts[name] = base.group(1).decode()
    runs = set()
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        try:
            text = zlib.decompress(match.group(1))
        except zlib.error:
            continue
        runs |= {(fonts.get(name), round(float(size) * scale, 3))
                 for name, size in re.findall(rb"/(F\d+)\s+([\d.]+)\s+Tf", text)}
    return runs


def printed_sizes(pdf):
    return {round(size, 2) for _, size in printed_runs(pdf)}


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
    base = [TEXT_PT * scale, BODY_PT]
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
                        if is_body_font(name):
                            self.assertAlmostEqual(value * COLUMN_PT / (72.0 * width), BODY_PT, delta=0.005)
                        else:
                            self.assertAlmostEqual(value * REFERENCE_WIDTH_IN / width, font_role(name), places=6)
        self.assertGreaterEqual(checked, 30)

    def test_contour_cells_set_labels_and_region_names_in_the_body_font(self):
        for cell in contour_cells():
            found = constants(cell.source)
            suffixes = ("LABEL_FONT_FAMILY", "REGION_TEXT_FONT_FAMILY") + (
                ("LEGEND_FONT_FAMILY",) if any(k.endswith("LEGEND_FONT_SIZE") for k in found) else ())
            for suffix in suffixes:
                names = [k for k in found if k.endswith(suffix) and not k.endswith("SPEED_" + suffix)]
                with self.subTest(suffix=suffix):
                    self.assertEqual(len(names), 1, names)
                    self.assertEqual(found[names[0]], LABEL_FONT_FAMILY)

    def test_comparisons_use_xi_zero(self):
        values = {k: v for cell in contour_cells() for k, v in constants(cell.source).items()
                  if k == "XI" or k.endswith("_XI")}
        self.assertEqual(len(values), 3, values)   # adiabatic, isothermal linear and isothermal log-T
        for name, value in values.items():
            with self.subTest(name=name):
                self.assertEqual(value, 0.0)

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
                "isothermal-analytic-numerical-linear.pdf", "T_a_constant_pressure.pdf",
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
        # Its ticks, panel names and phase names still print at about 11 pt; only the axis labels follow
        # the standard (test_labels_and_titles_print_at_the_body_size).
        self.assert_standard("boundary_schematic_real.pdf")

    LABELLED = STANDARD + ("boundary_schematic_real.pdf", "boundary_schematic_isothermal_real.pdf",
                           "T_muK_constant_pressure.pdf")

    def test_labels_and_titles_print_at_the_body_size_in_computer_modern(self):
        # Labels, titles, region names and contour legends are the only text near 10 pt (ticks: 10.4 pt).
        for name in self.LABELLED:
            runs = {(font, size) for font, size in printed_runs(PLOTS / name) if abs(size - BODY_PT) < 0.2}
            with self.subTest(figure=name):
                self.assertTrue(runs)
                for font, size in runs:
                    self.assertAlmostEqual(size, BODY_PT, delta=0.02, msg=(font, size))
                    self.assertIn(font, COMPUTER_MODERN)


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
