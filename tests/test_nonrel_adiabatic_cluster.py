"""Migration checks for the fixed-interface nonrelativistic cluster runner."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SCAN = Path(__file__).resolve().parents[1] / 'new_paper_calculations/26-10-24'
sys.path.insert(0, str(SCAN))
import _slow_front_adapter as adapter
import _radial_shell_continuation as radial
from prepare_T0plus0p1_domain import prepare_domain


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = prepare_domain()

    def task(self):
        return dict(point=dict(T_0minus_MeV=10., nB_0minus_MeV3=4e6),
                    B14_MeV=189.1565957288247, T_0plus_MeV=.1,
                    ms_MeV=0., nm_type='PNM', tail_eps=1e-8, n_mesh=300,
                    tol_bvp=1e-4, max_nodes=50000, path_rtol=1e-6, kappa_factor=1.)

    def result(self):
        return dict(success=True, status='converged', hydrodynamic_model=adapter.MODEL,
                    closure='adiabatic', jB=1., u_0minus=1e-6, bvp_status=0,
                    bvp_max_rms_residual=1e-5, closure_error_max=1e-9,
                    conservation_errors=dict(baryon_flux=0., pressure=1e-9, enthalpy_flux=1e-9),
                    boundary_residuals=np.zeros(3), constitutive_residual_norm=1e-5,
                    reaction_residual_norm=1e-5, diffusion_integral_residual_norm=1e-5,
                    T_0plus=.1, continuation_state=dict(model='slow_front_v1',
                    closure='adiabatic', ms=0., jB=1., s=np.array([0., .99]),
                    r_fraction=np.array([1., .01]), current_excess_fraction=np.array([1., .01])))

    def test_exact_reference_grid_and_seed(self):
        source = np.load(SCAN / 'inputs/relativistic-state-of-art.npy', allow_pickle=True).item()
        self.assertEqual(len(self.manifest['points']), 540)
        for p in self.manifest['points']:
            i, k = p['theta_index'], p['radial_index']
            self.assertEqual(p['relativistic_jB_guess'], source['plot_jB_grid'][i, k])
            self.assertEqual(p['T_0minus_MeV'], source['plot_T_0minus_MeV_grid'][i, k])
            self.assertEqual(p['nB_0minus_over_n0'], source['plot_nB_0minus_over_n0_grid'][i, k])
        self.assertEqual(self.manifest['calibration']['ncrit_over_n0'], 3.)

    def test_solver_kwargs_and_continuation_flux_override(self):
        profile = adapter.extract_profile(self.result())
        before = copy.deepcopy(profile)
        kw = radial._solver_kwargs_for_candidate(self.task(),
                    dict(jB_guess=2., use_profile=True), profile)
        self.assertNotIn('jB_bounds', kw)
        self.assertTrue(kw['return_continuation'])
        self.assertEqual(kw['continuation_guess']['jB'], 2.)
        self.assertEqual(profile['continuation_state']['jB'], before['continuation_state']['jB'])
        self.assertEqual(kw['T_0plus'], .1)

    def test_old_profiles_are_not_accepted(self):
        with self.assertRaises(ValueError):
            radial._solver_kwargs_for_candidate(self.task(), dict(jB_guess=2., use_profile=True),
                                                dict(nK=np.ones(2), jK=np.ones(2)))

    def test_validation_requires_actual_slow_success(self):
        self.assertIsNone(radial._validation_error(self.result(), self.task()))
        for key, value in [('success', False), ('status', 'bvp_failed'),
                           ('hydrodynamic_model', 'relativistic'), ('jB', -1.),
                           ('closure_error_max', np.nan), ('continuation_state', None),
                           ('reaction_residual_norm', .01), ('T_0plus', 1.)]:
            with self.subTest(key=key):
                bad = self.result(); bad[key] = value
                self.assertIsNotNone(radial._validation_error(bad, self.task()))
        bad = self.result(); del bad['success']
        self.assertIsNotNone(radial._validation_error(bad, self.task()))

    def test_historical_seed_available_after_failed_predecessor(self):
        p = self.manifest['points'][8]
        candidates = radial.primary_candidates(p, previous=None, earlier=None,
                                                reference_jB=p['relativistic_jB_guess'])
        self.assertEqual(candidates[0]['jB_guess'], p['relativistic_jB_guess'])
        self.assertFalse(candidates[0]['use_profile'])

    def test_hidden_bridge_without_historical_seed(self):
        p = dict(radial_index=1, radial_fraction_used=.2)
        previous = dict(point=dict(radial_fraction_used=.1), record=dict(jB=1.))
        candidates = radial.primary_candidates(p, previous=previous, earlier=None)
        self.assertTrue(candidates)
        self.assertTrue(all(c['use_profile'] for c in candidates))

    def test_refinement_retry_is_tighter_and_specific(self):
        from RMFsolver import phase_velocity
        failure = dict(status='thermodynamic_path_failed',
                       message='Slow-front EOS/diffusion table did not converge')
        with patch.object(phase_velocity, 'solve_front_adiabatic_slow',
                          side_effect=[failure, self.result()]) as call:
            result = adapter.solve_with_path_refinement(path_rtol=1e-6)
            self.assertEqual([c.kwargs['path_rtol'] for c in call.call_args_list], [1e-6, 5e-7])
            self.assertEqual(len(result['path_refinement_attempts']), 2)
        with patch.object(phase_velocity, 'solve_front_adiabatic_slow',
                          return_value=dict(status='thermodynamic_path_failed', message='invalid composition')) as call:
            adapter.solve_with_path_refinement(path_rtol=1e-6)
            self.assertEqual(call.call_count, 1)

    def test_nuclear_seed_extension_preserves_and_restores_branch_gate(self):
        from RMFsolver import phase_velocity
        seeds = phase_velocity._PNM_BRANCH_SEEDS
        seen = []
        def fake(**kwargs):
            seen.append(phase_velocity._PNM_BRANCH_SEEDS)
            if len(seen) == 1:
                return dict(status='thermodynamic_path_failed',
                            message='PNM density solve: no two physical seeds agreed')
            return self.result()
        with patch.object(phase_velocity, 'solve_front_adiabatic_slow', side_effect=fake):
            result = adapter.solve_with_path_refinement(path_rtol=1e-6, NM_type='PNM')
        self.assertEqual(seen, [seeds, seeds + ((35., 25., -3.), (40., 30., -3.))])
        self.assertEqual(phase_velocity._PNM_BRANCH_SEEDS, seeds)
        self.assertEqual(len(result['upstream_seed_extension']), 2)

    def test_resume_rejects_wrong_model_even_when_complete(self):
        identity = {'model': adapter.MODEL, 'source_sha256': 'abc'}
        with self.assertRaises(ValueError):
            adapter.validate_resume({'run_complete': True}, identity)
        adapter.validate_resume({'run_identity': identity}, identity)
        with self.assertRaises(ValueError):
            adapter.validate_resume({'run_identity': dict(identity, source_sha256='other')}, identity)

    def test_resume_identity_covers_authoritative_nested_state(self):
        args = SimpleNamespace(domain_file='T0plus0p1-domain.npy', tail_eps=1e-8,
                               n_mesh=300, tol_bvp=1e-4, max_nodes=50000, path_rtol=1e-6,
                               kappa_factor=1., reference_jB_scale=1., max_bridge_depth=2,
                               min_bridge_fraction_step=1e-3)
        original = adapter.run_identity(self.manifest, args)
        changed = copy.deepcopy(self.manifest)
        changed['points'][0]['states'][0]['T_0minus_MeV'] += 1.
        self.assertNotEqual(adapter.run_identity(changed, args), original)


if __name__ == '__main__':
    unittest.main()
