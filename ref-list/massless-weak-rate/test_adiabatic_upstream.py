"""Physical-state and discovery checks independent of the expensive RMF solve."""

import math
import unittest

import adiabatic_upstream as upstream


class ColdInterfaceTests(unittest.TestCase):
    def test_recovers_composition_and_conserved_density(self):
        density = 4.2e6
        for composition in (0.005, 0.146, 0.5, 0.9, 0.995):
            pressure = ((math.pi**2 * density)**(4 / 3) / (4 * math.pi**2)
                        * (1 + (1 + composition)**(4 / 3)
                           + (1 - composition)**(4 / 3)))
            enthalpy_per_baryon = 4 * pressure / density
            state = upstream.cold_interface(pressure, enthalpy_per_baryon)
            self.assertEqual(state["status"], "physical")
            self.assertAlmostEqual(state["nB"] / density, 1, places=14)
            self.assertAlmostEqual(state["a_0plus"], composition, delta=2e-11)
            self.assertLess(abs(state["pressure_relative_residual"]), 1e-13)
            self.assertLess(abs(state["enthalpy_relative_residual"]), 1e-14)

    def test_classifies_unreachable_states_without_extrapolation(self):
        density = 4e6
        cold_equilibrium = upstream.cold_kinetic_pressure(density, 0)
        cold_neutron_content = upstream.cold_kinetic_pressure(density, 1)
        for pressure, status in (
            (0.99 * cold_equilibrium, "no_front"),
            (cold_equilibrium, "equilibrium_boundary"),
            (cold_neutron_content, "flavor_boundary"),
            (1.01 * cold_neutron_content, "no_cold_interface"),
        ):
            result = upstream.cold_interface(pressure, 4 * pressure / density)
            self.assertEqual(result["status"], status)
            if status in ("no_front", "no_cold_interface"):
                self.assertIsNone(result["a_0plus"])

    def test_rejects_invalid_thermodynamics(self):
        for pressure, enthalpy in ((0, 1000), (1e9, -1), (math.nan, 1000)):
            with self.assertRaises(ValueError):
                upstream.cold_interface(pressure, enthalpy)


class DensityDiscoveryTests(unittest.TestCase):
    @staticmethod
    def synthetic_state(temperature, density_ratio):
        # Two physical intervals separated by a composition-above-one domain.
        prefactor = math.pi**(2 / 3) / 4
        pressure = prefactor * (3 + 0.7 * (1 - ((density_ratio - 3) / 2)**2))
        result = upstream.cold_interface(pressure, 4 * pressure)
        result.update(T_0minus=temperature, nB_0minus_over_n0=density_ratio)
        return result

    def test_discovers_two_intervals_and_samples_each_composition(self):
        result, states = upstream.discover_slice(
            20, self.synthetic_state, density_min=1, density_max=5,
            scan_points=17, scan_refinements=1, composition_targets=(0.02, 0.5, 0.99),
        )
        self.assertEqual(result["status"], "physical_intervals")
        self.assertEqual(len(result["intervals"]), 2)
        self.assertEqual(len(states), 6)
        for interval in result["intervals"]:
            self.assertEqual(interval["selected_state_count"], 3)
        for state in states:
            self.assertLess(state["a_0plus"], 1)
            self.assertGreater(state["a_0plus"], 0)
            self.assertLess(abs(state["a_0plus"] - state["target_a_0plus"]), 2e-7)

    def test_no_interval_is_an_explicit_physical_result(self):
        def no_front(temperature, density_ratio):
            pressure = 0.99 * upstream.cold_kinetic_pressure(1, 0)
            state = upstream.cold_interface(pressure, 4 * pressure)
            state.update(T_0minus=temperature, nB_0minus_over_n0=density_ratio)
            return state

        result, states = upstream.discover_slice(
            100, no_front, scan_points=9, scan_refinements=1,
        )
        self.assertEqual(result["status"], "no_physical_interval")
        self.assertEqual(result["intervals"], [])
        self.assertEqual(states, [])
        self.assertEqual(result["classification_counts"], {"no_front": 17})

    def test_clipped_nonmonotone_interval_retains_both_roots_and_minimum(self):
        def clipped_state(temperature, density_ratio):
            composition = 0.2 + 0.5 * ((density_ratio - 3) / 2)**2
            pressure = upstream.cold_kinetic_pressure(1, composition)
            state = upstream.cold_interface(pressure, 4 * pressure)
            state.update(T_0minus=temperature, nB_0minus_over_n0=density_ratio)
            return state

        summary, states = upstream.discover_slice(
            100, clipped_state, density_min=1, density_max=5, scan_points=17,
            composition_targets=(0.3, 0.5, 0.8),
        )
        interval = summary["intervals"][0]
        self.assertTrue("reachable_a_0plus_bounds" in interval)
        self.assertAlmostEqual(interval["reachable_a_0plus_bounds"][0], 0.2, places=12)
        self.assertAlmostEqual(interval["reachable_a_0plus_bounds"][1], 0.7, places=12)
        self.assertEqual(interval["density_bound_inclusive"], [True, True])
        self.assertEqual(len(states), 7)
        self.assertEqual(sum(state["sample_role"] == "composition_extremum" for state in states), 1)
        self.assertEqual(sum(state["sample_role"] == "density_domain_edge" for state in states), 2)
        self.assertEqual(interval["inaccessible_targets"], [0.8])


class UpstreamRecoveryTests(unittest.TestCase):
    def test_high_density_recovery_preserves_branch_checks_and_global_seeds(self):
        self.assertTrue(hasattr(upstream, "validated_state_with_recovery"))
        from RMFsolver import phase_velocity

        original_seeds = phase_velocity._PNM_BRANCH_SEEDS
        state, provenance = upstream.validated_state_with_recovery(10 * upstream.N0, 0)
        self.assertIs(phase_velocity._PNM_BRANCH_SEEDS, original_seeds)
        self.assertTrue(provenance["recovery_used"])
        self.assertIn("could not pin the physical branch", provenance["original_error"])
        self.assertGreaterEqual(state["upstream_seed_count"], 2)
        self.assertLess(state["upstream_rmf_scaled_residual"], 1e-6)
        self.assertLess(abs(state["upstream_density_relative_residual"]), 2e-6)
        self.assertAlmostEqual(state["muB_0minus"], 2641.43378395, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
