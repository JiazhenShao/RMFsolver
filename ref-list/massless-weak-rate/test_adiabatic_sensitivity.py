"""Physical invariants of the cold-interface conserved EOS calculation."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np


class AdiabaticTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('adiabatic_sensitivity.py')
        cls.module_exists = path.exists()
        if cls.module_exists:
            import adiabatic_sensitivity
            cls.model = adiabatic_sensitivity

    def test_conserved_trajectory_and_endpoints(self):
        self.assertTrue(self.module_exists, 'adiabatic trajectory module is required')
        trajectory = self.model.ColdInterfaceTrajectory(0.5, cold_mu=400.)
        self.assertAlmostEqual(trajectory.state(0.)['a'], .5)
        self.assertEqual(trajectory.state(1.)['a'], 0.)
        for z in (.001, .1, .5, .9, 1.):
            state = trajectory.state(z)
            self.assertLess(state['pressure_relative_residual'], 1e-10)
            densities = self.model.number_density(np.array(state['chemical']), state['T'])
            np.testing.assert_allclose(densities, trajectory.nB*np.array([1, 1+state['a'], 1-state['a']]), rtol=2e-12)
        self.assertGreater(trajectory.T_inf, 0.)

    def test_regular_cold_endpoint_and_scaling(self):
        self.assertTrue(self.module_exists, 'adiabatic trajectory module is required')
        one = self.model.ColdInterfaceTrajectory(.5, cold_mu=400.)
        two = self.model.ColdInterfaceTrajectory(.5, cold_mu=800.)
        near = self.model.weighted_integrand(one, 1e-4, phase_order=8)
        np.testing.assert_allclose(near, self.model.cold_endpoint_integrand(one), rtol=3e-4)
        base = self.model.integrated_estimates(one, outer_order=8, phase_order=8)
        scaled = self.model.integrated_estimates(two, outer_order=8, phase_order=8)
        np.testing.assert_allclose(scaled['integrals'], 16*np.array(base['integrals']), rtol=2e-12)
        np.testing.assert_allclose(scaled['ratios'], base['ratios'], rtol=2e-12)
        self.assertEqual(base['ratios'][0], 1.)

    def test_jacobian_matches_finite_difference(self):
        self.assertTrue(self.module_exists, 'adiabatic trajectory module is required')
        trajectory = self.model.ColdInterfaceTrajectory(.9, cold_mu=400.)
        z, step = .6, 1e-5
        numerical = -(trajectory.state(z+step)['a']-trajectory.state(z-step)['a'])/(2*step)
        self.assertAlmostEqual(trajectory.state(z)['minus_da_dz']/numerical, 1., places=7)

    def test_common_diffusion_scale_cancels(self):
        self.assertTrue(self.module_exists, 'adiabatic trajectory module is required')
        trajectory = self.model.ColdInterfaceTrajectory(.146)
        one = self.model.integrated_estimates(trajectory, outer_order=8, phase_order=8)
        two = self.model.integrated_estimates(trajectory, outer_order=8, phase_order=8, diffusion_scale=7.)
        np.testing.assert_allclose(two['ratios'], one['ratios'], rtol=2e-14)
        np.testing.assert_allclose(two['integrals'], 7*np.array(one['integrals']), rtol=2e-14)

    def test_roundoff_cold_pressure_does_not_lose_root(self):
        self.assertTrue(self.module_exists, 'adiabatic trajectory module is required')
        trajectory = self.model.ColdInterfaceTrajectory(.005)
        state = trajectory.state(.002481883968903989)
        self.assertGreater(state['T'], 0.)
        self.assertLess(state['pressure_relative_residual'], 2e-14)
        self.assertAlmostEqual(state['a'], .005, places=10)

    def test_very_cold_states_preserve_density_and_pressure(self):
        for a, z in ((.5,3.981071705534969e-6),(.146,2.5118864315095823e-7)):
            trajectory=self.model.ColdInterfaceTrajectory(a)
            state=trajectory.state(z)
            densities=self.model.number_density(np.array(state['chemical']),state['T'])
            np.testing.assert_allclose(densities,trajectory.nB*np.array([1,1+a,1-a]),rtol=3e-14)
            self.assertLess(state['pressure_relative_residual'],3e-14)


if __name__ == '__main__':
    unittest.main()
