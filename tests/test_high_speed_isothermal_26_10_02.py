"""High-speed analytical extension: immutable baselines and no invented BVP data."""
import importlib.util
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'new_paper_calculations/26-10-02'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HighSpeedTests(unittest.TestCase):
    def test_missing_numerical_level_is_empty(self):
        saved = load('_saved_contour_data')
        dots = saved.cold_numerical_points({'levels_m_s': [50.], 'roots_0': [[3.1, 1.]]}, {}, {}, 10000.)
        self.assertEqual(dots.shape, (0, 2))

    def test_extension_preserves_baseline_and_checks_axes(self):
        saved = load('_saved_contour_data')
        self.assertTrue(hasattr(saved, 'merge_analytical_extension'))
        baseline = dict(temperatures=[.01, 20.], levels=[50.], targets={'old': {'status': 'success'}},
                        domain_fingerprint='same', identity={'physics': {'xi': -.85}})
        extra = dict(temperatures=[.01, 20.], levels=[10000.], targets={'new': {'status': 'success'}},
                     domain_fingerprint='same', physics={'xi': -.85})
        merged = saved.merge_analytical_extension(baseline, extra)
        self.assertEqual(merged['levels'], [50., 10000.])
        self.assertEqual(merged['targets']['old'], baseline['targets']['old'])
        self.assertNotIn('new', baseline['targets'])
        extra['temperatures'] = [.01, 19.]
        with self.assertRaises(ValueError):
            saved.merge_analytical_extension(baseline, extra)

    def test_root_predictor_stays_inside_actual_bracket(self):
        path = SOURCE / 'extend_isothermal_levels.py'
        self.assertTrue(path.exists(), 'bounded high-speed extension runner is missing')
        runner = load('extend_isothermal_levels')
        rows = [dict(nB_0minus_over_n0=3., velocity_m_s=1000.),
                dict(nB_0minus_over_n0=4., velocity_m_s=100000.)]
        self.assertAlmostEqual(runner.next_density(rows, 10000., (2., 5.)), 3.5)
        self.assertAlmostEqual(runner.next_density(rows[:1], 10000., (2., 5.)), 4.)

    def test_target_accepts_only_computed_validated_speed(self):
        path = SOURCE / 'extend_isothermal_levels.py'
        self.assertTrue(path.exists(), 'bounded high-speed extension runner is missing')
        runner = load('extend_isothermal_levels')
        rows = [dict(nB_0minus_over_n0=3.4, velocity_m_s=10030., result={'validated': False, 'success': True}),
                dict(nB_0minus_over_n0=3.5, velocity_m_s=10100., result={'validated': True, 'success': True})]
        self.assertIsNone(runner.accepted_root(rows, 10000.))
        rows[0]['result']['validated'] = True
        self.assertIs(runner.accepted_root(rows, 10000.), rows[0])

    def test_log_plot_adds_four_levels_but_not_numerical_curves(self):
        template = load('build_comparison_notebook')
        source = template.notebook().cells[-1].source
        self.assertIn('10000., 20000., 50000., 100000.', source)
        self.assertIn("if level not in warm_data['levels_m_s']:", source)
        self.assertIn('ContourSet(axis, numerical_levels,', source)

    def test_outer_seed_cache_preserves_original_points(self):
        path = SOURCE / 'extend_isothermal_outer_levels.py'
        self.assertTrue(path.exists(), 'outer-level cache runner is missing')
        runner = load('extend_isothermal_outer_levels')
        original = {'points': {'original': {'T_0minus': 1., 'velocity_m_s': 50.}}}
        good = dict(T_0minus=1., nB_0minus_over_n0=4., velocity_m_s=100000.,
                    result={'success': True, 'validated': True})
        extension = {'evaluations': [{'root': good}, {'root': None}]}
        cached = runner.seed_cache(original, extension)
        self.assertEqual(len(cached['points']), 2)
        self.assertEqual(original['points'], {'original': {'T_0minus': 1., 'velocity_m_s': 50.}})
        self.assertIn(good, cached['points'].values())

    def test_real_log_plot_keeps_seven_numerical_levels_and_58_cold_dots(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import nbformat
        notebook = nbformat.read(ROOT / '26-10-02.ipynb', as_version=4)
        namespace = {'__name__': '__main__'}
        with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory(dir=ROOT / 'local_analysis_results') as temporary:
            source=notebook.cells[4].source.replace('CALCULATE_ANALYTICAL = True','CALCULATE_ANALYTICAL = False')
            exec(compile(source,'isothermal-load','exec'),namespace)
            namespace['OUTPUT_DIRECTORY'] = Path(temporary)
            exec(compile(notebook.cells[-1].source, 'log-plot-cell', 'exec'), namespace)
            self.assertEqual(len(namespace['CONTOUR_LEVELS']), 14)
            np.testing.assert_array_equal(namespace['numerical_contours'].levels, [50,100,200,500,1000,2000,5000])
            dots = [artist for artist in namespace['axis'].collections
                    if str(artist.get_gid()).startswith('numerical-cold:')]
            self.assertEqual(sum(len(artist.get_offsets()) for artist in dots), 58)
            self.assertTrue(all(np.all(artist.get_offsets()[:, 1] < 20.) for artist in dots))
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
