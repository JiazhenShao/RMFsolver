"""Regression checks for non-destructive local analytical recovery."""
import copy
import importlib.util
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import run_T0plus0p1_analytical as scan


class LocalRepairTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('repair_analytical_local'),
                             'The local repair implementation is missing')
        import repair_analytical_local
        self.repair = repair_analytical_local

    def test_merge_preserves_success_and_archives_failed_result(self):
        saved = dict(analytic=dict(success=True, u_0minus=1., jB=2., message='cluster'),
                     semi_analytic=dict(success=False, message='old timeout'))
        original = copy.deepcopy(saved['analytic'])
        self.repair.merge_results(saved, dict(
            analytic=dict(success=True, u_0minus=9., jB=9.),
            semi_analytic=dict(success=True, u_0minus=3., jB=4.)))
        self.assertEqual(saved['analytic'], original)
        self.assertTrue(scan.good(saved['semi_analytic']))
        self.assertEqual(saved['local_repair_history'][0]['previous']['semi_analytic']['message'],
                         'old timeout')

    def test_semi_timeout_retains_full_result(self):
        point = dict(theta_index=0, radial_fraction=.02)
        full = dict(success=True, u_0minus=1., jB=2.)
        with patch.object(self.repair, 'run_module_call_with_hard_timeout',
                          side_effect=[full, TimeoutError('test cap')]):
            results = self.repair.evaluate_point(point, {}, scan.METHODS, 1.)
        self.assertEqual(results['analytic'], full)
        self.assertFalse(results['semi_analytic']['success'])
        self.assertIn('test cap', results['semi_analytic']['message'])

    def test_checkpoint_refreshes_grids_and_keeps_reference(self):
        payload = np.load(scan.DEFAULT_OUTPUT, allow_pickle=True).item()
        numerical = copy.deepcopy(payload['numerical_reference_grid'])
        key = payload['original_domain_grid']['point_keys'][0, 0]
        for method in scan.METHODS:
            payload['records'][key][method] = dict(success=False, message='fixture failure')
        self.repair.merge_results(payload['records'][key], {
            m: dict(success=True, u_0minus=100. / scan.SPEED_OF_LIGHT, jB=1.)
            for m in scan.METHODS})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'checkpoint.npy'
            self.repair.checkpoint(payload, output)
            reread = np.load(output, allow_pickle=True).item()
        mask = reread['original_domain_grid']['point_keys'] == key
        self.assertTrue(np.all(reread['original_domain_grid']['analytic_success'][mask]))
        for field, value in numerical.items():
            np.testing.assert_array_equal(reread['numerical_reference_grid'][field], value)
        self.assertEqual(reread['planned_point_count'], len(payload['points']))

    def test_each_repaired_point_is_checkpointed_and_counted(self):
        payload = np.load(scan.DEFAULT_OUTPUT, allow_pickle=True).item()
        keys = payload['original_domain_grid']['point_keys'][0, :2]
        for key in keys:
            for method in scan.METHODS:
                payload['records'][key][method] = dict(success=False, message='fixture failure')
        readable = []
        real_checkpoint = self.repair.checkpoint
        def save_and_read(data, output):
            real_checkpoint(data, output)
            readable.append(np.load(output, allow_pickle=True).item())
        result = {m: dict(success=True, u_0minus=1., jB=2.) for m in scan.METHODS}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(self.repair, 'evaluate_point', return_value=result), \
                patch.object(self.repair, 'checkpoint', side_effect=save_and_read), \
                patch.object(self.repair, 'tqdm') as progress:
            args = SimpleNamespace(workers=2, timeout=1., output=Path(directory) / 'result.npy')
            self.repair.run_points([payload['points'][k] for k in keys], payload, {}, args)
        self.assertEqual(len(readable), 2)
        self.assertEqual(progress.return_value.__enter__.return_value.update.call_count, 2)
        self.assertTrue(all(scan.good(readable[-1]['records'][k]['analytic']) for k in keys))

    def test_backup_is_byte_identical_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.npy'
            np.save(output, {'stage': 'cluster'})
            backup = self.repair.backup_once(output)
            self.assertEqual(backup.read_bytes(), output.read_bytes())
            np.save(output, {'stage': 'local'})
            self.assertEqual(self.repair.backup_once(output), backup)
            self.assertEqual(np.load(backup, allow_pickle=True).item(), {'stage': 'cluster'})


if __name__ == '__main__':
    unittest.main()
