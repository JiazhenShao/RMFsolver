"""Geometry checks for the notebook's numerical-to-analytical contour bands."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np


class ContourBandTests(unittest.TestCase):
    def setUp(self):
        notebook = nbformat.read(Path(__file__).resolve().parents[1] / 'All_Plots_new.ipynb', as_version=4)
        cell = next(c for c in notebook.cells if c.get('id') == 't0plus0p1-three-method')
        self.cell_source = cell.source
        functions = [node for node in ast.parse(cell.source).body
                     if isinstance(node, ast.FunctionDef) and node.name in
                     ('comparison_band_polygon', 'comparison_band_paths')]
        self.assertEqual(len(functions), 2, 'Missing contour band geometry or selection helper')
        namespace = dict(np=np, BAND_ANGLE_TOLERANCE=1e-10, BAND_ALLOW_SEMI_ANALYTIC_FALLBACK=True,
                         transform=dict(density_origin_over_n0=1., density_scale_over_n0=1.,
                                        temperature_scale_MeV=1.))
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<notebook-band>', 'exec'), namespace)
        self.polygon = namespace['comparison_band_polygon']
        self.select = namespace['comparison_band_paths']
        self.namespace = namespace

    def test_requested_two_plot_variants(self):
        variants = [ast.literal_eval(node.value) for node in ast.parse(self.cell_source).body
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == 'COMPARISON_VARIANTS'
                        for target in node.targets)]
        self.assertEqual(variants, [((5000.0, False, ''), (1000.0, True, '-bands-to-1000'))])

    def test_banded_labels_have_larger_type_and_local_translucent_background(self):
        from matplotlib.text import Text
        tree = ast.parse(self.cell_source)
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'comparison_label_background']
        self.assertEqual(len(helpers), 1)
        settings = [node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id.startswith(
                        ('BANDED_CONTOUR_LABEL_', 'CONTOUR_LABEL_')) for target in node.targets)]
        namespace = {}
        exec(compile(ast.Module(body=settings+helpers, type_ignores=[]), '<label-style>', 'exec'), namespace)
        self.assertGreater(namespace['BANDED_CONTOUR_LABEL_FONT_SIZE'], namespace['CONTOUR_LABEL_FONT_SIZE'])
        label = Text(text='500')
        namespace['comparison_label_background']([label], True)
        box = label.get_bbox_patch()
        self.assertIsNotNone(box)
        self.assertAlmostEqual(box.get_alpha(), .85)
        self.assertEqual(box.get_linewidth(), 0.)
        self.assertGreater(label.get_zorder(), 2)
        plain = Text(text='500')
        namespace['comparison_label_background']([plain], False)
        self.assertIsNone(plain.get_bbox_patch())

    def test_full_analytic_preferred_with_semi_for_missing_levels(self):
        full = SimpleNamespace(levels=[50.], allsegs=['full50'])
        semi = SimpleNamespace(levels=[50., 2000.], allsegs=['semi50', 'semi2000'])
        refined = SimpleNamespace(levels=[5000.], allsegs=['semi5000'])
        paths, sources = self.select(full, semi, refined)
        self.assertEqual(paths, {50.: 'full50', 2000.: 'semi2000', 5000.: 'semi5000'})
        self.assertEqual(sources, {50.: 'analytic', 2000.: 'semi_analytic', 5000.: 'semi_analytic'})

    def test_fallback_can_be_disabled(self):
        self.namespace['BAND_ALLOW_SEMI_ANALYTIC_FALLBACK'] = False
        full = SimpleNamespace(levels=[50.], allsegs=['full50'])
        semi = SimpleNamespace(levels=[2000.], allsegs=['semi2000'])
        paths, sources = self.select(full, semi, None)
        self.assertEqual(paths, {50.: 'full50'})
        self.assertEqual(sources, {50.: 'analytic'})

    def test_complete_curves_keep_their_actual_vertices(self):
        first = np.array([[3., 0.], [2., 1.], [1., 2.]])
        second = np.array([[4., 0.], [2.5, 1.5], [1., 3.]])
        np.testing.assert_allclose(self.polygon(first, second), np.vstack((first, second[::-1])))

    def test_reversed_path_has_same_polygon(self):
        first = np.array([[3., 0.], [2., 1.], [1., 2.]])
        second = np.array([[4., 0.], [2.5, 1.5], [1., 3.]])
        np.testing.assert_allclose(self.polygon(first[::-1], second), self.polygon(first, second))

    def test_partial_path_clips_other_curve_without_extrapolation(self):
        first = np.array([[3., 0.], [2., 1.]])
        second = np.array([[4., 0.], [1., 3.]])
        polygon = self.polygon(first, second)
        np.testing.assert_allclose(polygon, [[3., 0.], [2., 1.], [2.5, 1.5], [4., 0.]])
        self.assertLessEqual(np.max(np.arctan2(polygon[:, 1], polygon[:, 0] - 1.)), np.pi / 4)

    def test_disjoint_or_empty_paths_produce_no_band(self):
        first = np.array([[3., 0.], [2., 1.]])
        second = np.array([[1.5, 2.], [1., 3.]])
        self.assertIsNone(self.polygon(first, second))
        self.assertIsNone(self.polygon(first, np.empty((0, 2))))


if __name__ == '__main__':
    unittest.main()
