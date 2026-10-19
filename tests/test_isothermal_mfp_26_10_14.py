"""Lightweight tests: no live RMF imports, contour solves, or BVP calls."""
import copy
import importlib
import json
import hashlib
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'new_paper_calculations/26-10-14'))


class MfpTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('isothermal_mfp')

    def inputs(self):
        return dict(n0=1., fingerprint='fixture', source_hashes={},
                    physics=dict(B_one_forth=189.1565957288247, NM_type='PNM', ms=0., upB=5000, xi=-.85),
                    boundary_cache={float(t).hex():dict(T_0minus=t, status='success', lower=1., upper=5.)
                                    for t in (.01, 1., 20.)})

    def evaluator(self, point):
        # Small positive physical-like ratio increasing strictly with density.
        return dict(success=True, muB_inf=1200., T_0minus=point['T_0minus'],
                    D_full=point['nB_0minus_over_n0']**2, D_landau=2*point['nB_0minus_over_n0']**2,
                    gamma_K=1.e-14, validated=True)

    def test_formula_units_identity_and_landau_scaling(self):
        m = self.module()
        for T in (.01, 1., 20.):
            row = m.transport_lengths(1200., T)
            self.assertAlmostEqual(row['lambda_mfp_MeV_inv'], 3*row['D_K'])
            self.assertAlmostEqual(row['L_conv_MeV_inv'], 1200/(9*np.pi*T)*np.sqrt(row['D_K']/row['gamma_K']))
            self.assertAlmostEqual(row['ratio'], 27*np.pi*T/1200*np.sqrt(row['D_K']*row['gamma_K']))
            self.assertAlmostEqual(row['lambda_mfp_m']/row['lambda_mfp_MeV_inv'], 197.326960e-15)
            self.assertAlmostEqual(row['L_conv_m']/row['L_conv_MeV_inv'], 197.326960e-15)
        cold = m.transport_lengths(1200., .01, 'landau')
        warm = m.transport_lengths(1200., .1, 'landau')
        self.assertAlmostEqual(warm['ratio']/cold['ratio'], 10**(1/6))
        self.assertLess(m.transport_lengths(1200., 20.)['ratio'], m.transport_lengths(1200., 20., 'landau')['ratio'])
        with self.assertRaises(ValueError): m.transport_lengths(1200., 0.)

    def test_no_crossing_fraction_independent_cache_resume_and_ordering(self):
        m = self.module(); calls = []; events = []
        with tempfile.TemporaryDirectory() as directory:
            def evaluate(point):
                calls.append(point)
                return self.evaluator(point)
            def event(kind, record):
                path = Path(directory) / ('thermodynamics.npy' if kind == 'evaluation' else 'latest-overlay.npy')
                saved = np.load(path, allow_pickle=True).item()
                self.assertIn(record['key'], saved['evaluations'] if kind == 'evaluation' else saved['rows'])
                events.append(kind)
            kwargs = dict(evaluator=evaluate, progress=event, show_progress=False)
            first = m.calculate(self.inputs(), directory, fraction=1., **kwargs)
            self.assertEqual({r['status'] for r in first['rows'].values()}, {'no_crossing'})
            self.assertEqual(len(calls), 6)
            second = m.calculate(self.inputs(), directory, fraction=.1, **kwargs)
            m.calculate(self.inputs(), directory, fraction=.1, **kwargs)
            self.assertEqual(len(calls), 6)
            self.assertTrue(second['work_complete'])
            self.assertEqual(m.root_segments(second), [])
            self.assertIn('evaluation', events)
            # Model changes also reuse the endpoint thermodynamics.
            m.calculate(self.inputs(), directory, fraction=.1, diffusion='landau', **kwargs)
            self.assertEqual(len(calls), 6)

    def test_bracketed_root_accuracy(self):
        m = self.module(); inputs = self.inputs()
        fraction = 27*np.pi*.01/1200*np.sqrt(9.e-14)
        with tempfile.TemporaryDirectory() as directory:
            result = m.calculate(inputs, directory, temperatures=(.01,), fraction=fraction,
                                 evaluator=self.evaluator, show_progress=False)
        row = next(iter(result['rows'].values()))
        self.assertEqual(row['status'], 'success')
        self.assertLessEqual(abs(row['ratio']/fraction-1), 1.e-6)
        self.assertAlmostEqual(row['nB_0minus_over_n0'], 3., places=5)

    def test_failures_outside_domain_budget_and_no_gap_connections(self):
        m = self.module(); inputs = self.inputs()
        inputs['boundary_cache'][float(20.).hex()]['status'] = 'outside_window'
        def fail(point):
            raise RuntimeError('EOS did not converge')
        with tempfile.TemporaryDirectory() as directory:
            result = m.calculate(inputs, directory, evaluator=fail, show_progress=False)
            self.assertEqual([r['status'] for r in result['rows'].values()], ['failed_evaluation', 'failed_evaluation', 'outside_domain'])
            self.assertFalse(result['work_complete'])
            self.assertEqual(m.root_segments(result), [])
            # Failed calls are retained, not silently retried on resume.
            m.calculate(inputs, directory, evaluator=lambda _: self.fail('Unexpected retry'), show_progress=False)
        result['rows'] = {str(i):dict(T_0minus=t, status=s, nB_0minus_over_n0=3., ratio=1.)
                          for i, (t,s) in enumerate(zip((.01,1.,20.), ('success','failed_evaluation','success')))}
        self.assertEqual([len(s) for s in m.root_segments(result)], [1,1])
        with tempfile.TemporaryDirectory() as directory:
            fraction = 27*np.pi*.01/1200*np.sqrt(9.e-14)
            result = m.calculate(self.inputs(), directory, temperatures=(.01,), fraction=fraction,
                                 max_evaluations=2, evaluator=self.evaluator, show_progress=False)
            self.assertEqual(next(iter(result['rows'].values()))['status'], 'evaluation_budget')

    def test_identity_and_input_validation(self):
        m = self.module(); inputs = self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            m.calculate(inputs, directory, evaluator=self.evaluator, show_progress=False)
            changed = copy.deepcopy(inputs); changed['physics']['B_one_forth'] += 1
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                m.calculate(changed, directory, evaluator=self.evaluator, show_progress=False)
        for f in (0., -1., float('nan')):
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                m.calculate(inputs, directory, fraction=f, evaluator=self.evaluator, show_progress=False)

    def test_interrupted_reservation_and_missing_root_are_not_connected(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            try:
                m.calculate(self.inputs(), directory, evaluator=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()), show_progress=False)
            except KeyboardInterrupt:
                pass
            cache = np.load(Path(directory)/'thermodynamics.npy', allow_pickle=True).item()
            self.assertEqual(next(iter(cache['evaluations'].values()))['status'], 'running')
            result = m.calculate(self.inputs(), directory, evaluator=self.evaluator, show_progress=False)
            self.assertEqual(result['rows'][float(.01).hex()]['status'], 'failed_evaluation')
            self.assertFalse(result['work_complete'])
        result['rows'] = {float(t).hex():dict(T_0minus=t, status='success', nB_0minus_over_n0=3., ratio=1.)
                          for t in (.01,20.)}
        self.assertEqual([len(s) for s in m.root_segments(result)], [1,1])

    def test_saved_curves_dots_and_shading_match_original_plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        old_path = ROOT/'26-10-02.ipynb'
        digest = hashlib.sha256(old_path.read_bytes()).hexdigest()
        old = json.loads(old_path.read_text())
        new = json.loads((ROOT/'26-10-14.ipynb').read_text())
        namespace = {}
        def data_artists(axis):
            out=[]
            for artist in axis.get_children():
                gid=artist.get_gid()
                if not gid: continue
                if hasattr(artist, 'get_xydata'): values=[artist.get_xydata()]
                elif hasattr(artist, 'get_offsets') and 'cold' in gid: values=[artist.get_offsets()]
                elif hasattr(artist, 'get_paths'): values=[p.vertices for p in artist.get_paths()]
                else: continue
                out.append((gid,values))
            return out
        with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory(dir=ROOT) as directory:
            exec(''.join(new['cells'][1]['source']), namespace)
            namespace['OUTPUT_DIRECTORY'] = Path(directory)
            namespace['plt'].show = lambda: None
            exec(''.join(old['cells'][8]['source']), namespace)
            expected = data_artists(namespace['axis'])
            plt.close(namespace['figure'])
            exec(''.join(new['cells'][5]['source']), namespace)
            actual = data_artists(namespace['axis'])
            self.assertFalse(any('mfp-tail' in a[0] for a in actual))
            self.assertEqual(len(expected),len(actual))
            for (name,values), (new_name,new_values) in zip(expected,actual):
                self.assertEqual(name,new_name)
                self.assertEqual(len(values),len(new_values))
                for a,b in zip(values,new_values): np.testing.assert_array_equal(a,b)
            plt.close(namespace['figure'])
        self.assertEqual(hashlib.sha256(old_path.read_bytes()).hexdigest(), digest)

    def test_notebook_loading_only_and_plot_parameters(self):
        book = json.loads((ROOT/'26-10-14.ipynb').read_text())
        cells = [''.join(c['source']) for c in book['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells), 5)  # Original tail section plus the standalone core section.
        load, calculate, plot = cells[:3]
        self.assertNotIn('.calculate(', load)
        self.assertIn('MFP_FRACTION = 1.0', calculate)
        self.assertIn('# Tunable plotting parameters', plot.splitlines()[0])
        self.assertIn('MFP_LINE_STYLE', plot)
        self.assertIn('root_segments', plot)
        for forbidden in ('solve_front', '.calculate(', 'analytic_velocity'):
            self.assertNotIn(forbidden, plot)


if __name__ == '__main__':
    unittest.main()
