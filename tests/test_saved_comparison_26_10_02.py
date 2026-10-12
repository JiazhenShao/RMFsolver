"""Saved-data selection only: no EOS or BVP calls."""
import importlib.util
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'new_paper_calculations/26-10-02/_saved_contour_data.py'


def target(temperature, level=50., status='success', validated=True, speed=None):
    return dict(level=level, T_0minus=temperature, kind='base', status=status,
                root=dict(nB_0minus_over_n0=3.1, T_0minus=temperature,
                          velocity_m_s=level if speed is None else speed,
                          result=dict(success=True, validated=validated)))


class SavedContourTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('saved_contour_test', MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_analytic_gap_not_joined(self):
        rows = [target(t) for t in [.01, .1, 1., 10., 20.]]
        rows[2]['status'] = 'unresolved'
        payload = dict(temperatures=[.01, .1, 1., 10., 20.],
                       targets=dict(enumerate(rows)))
        segments = self.module.analytical_segments(payload, 50.)
        self.assertEqual([len(segment) for segment in segments], [2, 2])
        np.testing.assert_equal(segments[1][:, 1], [10., 20.])

    def test_success_requires_validated_speed_and_matching_temperature(self):
        self.assertIsNone(self.module.accepted_target(target(1., validated=False)))
        self.assertIsNone(self.module.accepted_target(target(1., speed=51.)))
        row = target(1.)
        row['root']['T_0minus'] = 2.
        self.assertIsNone(self.module.accepted_target(row))

    def test_cold_dots_use_roots_not_smoothed_vertices_and_deduplicate(self):
        warm = dict(levels_m_s=np.array([50.]), roots_0=np.array([[3.1, 1.], [3.2, 25.]]),
                    curve_vertices_0=np.array([[3.05, .5], [3.1, 1.], [3.2, 25.]]))
        payload = dict(targets={'same': target(1.), 'new': target(2.),
                                'join': target(20.), 'bad': target(3., speed=51.)})
        dots = self.module.cold_numerical_points(warm, payload, {}, 50.)
        np.testing.assert_equal(dots, [[3.1, 1.], [3.1, 2.]])

    def test_warm_curve_clipped_at_twenty_no_cold_line(self):
        curve = np.array([[3., 0.], [3.1, 10.], [3.3, 30.], [2., 60.]])
        clipped = self.module.warm_curve(curve)
        np.testing.assert_allclose(clipped[0], [3.2, 20.])
        self.assertTrue(np.all(clipped[:, 1] >= 20.))
        np.testing.assert_equal(clipped[1:], curve[2:])

    def test_missing_target_remains_gap(self):
        payload = dict(temperatures=[1., 2., 3.], targets={'a': target(1.), 'b': target(3.)})
        segments = self.module.analytical_segments(payload, 50.)
        self.assertEqual([len(segment) for segment in segments], [1, 1])

    def test_legacy_cold_endpoint_acceptance_without_validation_flag(self):
        warm = dict(levels_m_s=np.array([50.]), roots_0=np.zeros((0, 2)))
        root = target(1.)['root']
        del root['result']['validated']
        cold = dict(rays={'accepted': dict(status='success', level=50., roots=[root])})
        dots = self.module.cold_numerical_points(warm, {}, cold, 50.)
        np.testing.assert_equal(dots, [[3.1, 1.]])

    def test_upper_shading_reaches_entire_top(self):
        geometry = dict(inner_boundary=[[3., 0.], [1., 65.]],
                        outer_boundary=[[5.3, 0.], [1., 95.]])
        temperature, density, edge = self.module.shading_boundaries(
            geometry, {}, (.01, 120.), (1., 5.5))['outer']
        self.assertEqual((temperature[-1], density[-1], edge), (120., 1., 5.5))

    def test_real_notebook_plot_labels_and_temperature_split(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import nbformat
        nb = nbformat.read(ROOT / '26-10-02.ipynb', as_version=4)
        namespace = {'__name__': '__main__'}
        with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory(dir=ROOT / 'local_analysis_results') as temporary:
            # Plot-only mode verifies that isothermal cells need no adiabatic state.
            source=nb.cells[4].source.replace('CALCULATE_ANALYTICAL = True','CALCULATE_ANALYTICAL = False')
            exec(compile(source,'isothermal-load','exec'),namespace)
            namespace['OUTPUT_DIRECTORY'] = Path(temporary)
            exec(compile(nb.cells[5].source, 'linear-plot-cell', 'exec'), namespace)
            axis = namespace['axis']
            labels = [text.get_text() for text in axis.texts if text.get_text().isdigit()]
            self.assertCountEqual(labels, ['50', '100', '200', '500', '1000', '2000', '5000'])
            self.assertTrue(all(text.get_bbox_patch() is not None for text in axis.texts
                                if text.get_text().isdigit()))
            for artist in axis.collections:
                if str(artist.get_gid()).startswith('numerical-cold:'):
                    self.assertTrue(np.all(artist.get_offsets()[:, 1] < 20.))
            for parts in namespace['numerical_contours'].allsegs:
                for part in parts:
                    self.assertTrue(np.all(part[:, 1] >= 20.))
            np.testing.assert_equal(namespace['figure'].get_size_inches(), [7.2, 7.2])
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
