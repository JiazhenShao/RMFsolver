"""Focused regression tests for the standalone four-quark reference."""

import unittest

import numpy as np

import validate_rate as rate


class PlotComparisonTests(unittest.TestCase):
    def test_only_three_requested_models_share_plot_styles(self):
        styles = rate.RATE_MODEL_STYLES
        self.assertEqual([style[0] for style in styles],
                         ["exact_rate", "eq23_rate", "sommerfeld_T2_rate"])
        self.assertEqual(len({style[2] for style in styles}), 3)
        self.assertIn("phase-space integral", styles[0][1])
        self.assertIn("Madsen", styles[1][1])
        self.assertIn("T^0+T^2", styles[2][1])

    def test_saved_grid_reader_preserves_models_and_paths(self):
        rows = rate.read_grid(rate.HERE / "results" / "rate-comparison-grid.csv")
        self.assertEqual(len(rows), 192)
        self.assertEqual({row["path"] for row in rows},
                         {"fixed_muB_a", "fixed_muB_muK"})
        for row in rows:
            for key in ("exact_rate", "eq23_rate", "sommerfeld_T2_rate", "T", "muB"):
                self.assertIsInstance(row[key], float)


class PairTests(unittest.TestCase):
    def test_pair_analytic_against_direct(self):
        nodes, weights = np.polynomial.legendre.leggauss(64)
        for energy, momentum, ma, mb, temperature in [
            (2.0, 1.5, 0.7, 1.3, 0.1),
            (2.000000001, 1.5, 0.7, 1.3, 0.1),
            (1.0, 0.8, 0.7, 1.3, 0.1),
            (3.0, 2.5, 0.7, 1.3, 0.1),
            (0.01, 0.009, 0.0, 0.0, 0.1),
            (2.0, 1.5, 0.7, 1.3, 0.001),
        ]:
            lo, hi = (energy - momentum) / 2, (energy + momentum) / 2
            edges = np.unique(np.clip([lo, hi, ma - 32 * temperature, ma,
                                       ma + 32 * temperature, energy - mb], lo, hi))
            half, mid = np.diff(edges) / 2, (edges[:-1] + edges[1:]) / 2
            p = mid[:, None] + half[:, None] * nodes
            first = rate.fermi(p, ma, temperature)
            second = rate.fermi(energy - p, mb, temperature)
            occupied = np.sum(weights * first * second * half[:, None])
            holes = np.sum(weights * (1 - first) * (1 - second) * half[:, None])
            actual = rate.pair_integrals(energy, momentum, ma, mb, temperature)
            np.testing.assert_allclose(actual, [occupied, holes], rtol=2e-11, atol=2e-14)

    def test_extreme_pair_is_finite(self):
        for energy in [0.01, 200.0, 600.0, 1000.0]:
            actual = rate.pair_integrals(energy, energy * 0.9, 300, 300, 0.001)
            self.assertTrue(np.all(np.isfinite(actual)))
            self.assertTrue(np.all(np.asarray(actual) >= 0))


class EosAndRateTests(unittest.TestCase):
    def test_eq24_printed_quartic_coefficient(self):
        # Madsen Eq. (24), p. 329: the v**4 coefficient is -5*z/64.
        for chemical, expected in [((1, 2, 0), 2),
                                   ((1, 1.5, 0.5), 459 / 256)]:
            with self.subTest(chemical=chemical):
                self.assertAlmostEqual(rate.eq24_factor(*chemical), expected, places=14)

    def test_neutral_fixed_composition(self):
        for a in [0, 0.001, 0.146, 0.9, 1]:
            for temperature in [0, 10, 100]:
                chemical = rate.neutral_fixed_a(900, a, temperature)
                density = rate.number_density(np.array(chemical), temperature)
                self.assertAlmostEqual(sum(chemical), 900, places=9)
                self.assertAlmostEqual((2 * density[0] - density[1] - density[2]) / density[0], 0, places=11)
                self.assertAlmostEqual((density[1] - density[2]) / (2 * density[0]), a, places=11)

    def test_fixed_muK_reproduces_fixed_a_state(self):
        chemical = rate.neutral_fixed_a(1200, 0.9, 20)
        actual = rate.neutral_fixed_muK(1200, chemical[1] - chemical[2], 20)
        np.testing.assert_allclose(actual, chemical, rtol=1e-12)

    def test_zero_muK_preserves_exact_flavor_equality(self):
        from thermal_sympy import phase_space_t2, phase_space_t3

        for muB in (900, 1000, 1100, 1200):
            for temperature in (0, 0.01, 10, 100):
                chemical = rate.neutral_fixed_muK(muB, 0, temperature)
                self.assertEqual(chemical, (muB / 3,) * 3)
                self.assertEqual(phase_space_t2(*chemical), 0)
                self.assertEqual(rate.exact_rate(*chemical, temperature), 0)
                # The forward cubic asymptotic is not uniform at equilibrium.
                with self.assertRaises(ValueError):
                    phase_space_t3(*chemical)

    def test_eos_quadratic_shifts(self):
        for a in (0.146, 0.99, 1):
            cold = np.array(rate.neutral_fixed_a(900, a, 0))
            temperature = 0.02
            for path in ("fixed_muB_a", "fixed_muB_muK"):
                warm = (rate.neutral_fixed_a(900, a, temperature) if path == "fixed_muB_a"
                        else rate.neutral_fixed_muK(900, cold[1]-cold[2], temperature))
                actual = (np.array(warm)-cold) / temperature**2
                np.testing.assert_allclose(actual, rate.eos_quadratic_shifts(*cold, path), rtol=2e-6, atol=2e-9)

    def test_cold_known_ratios(self):
        for a, expected in [(0.146, 0.93249051), (0.9, 0.35942364), (1, 0.1096583030099788)]:
            chemical = rate.neutral_fixed_a(900, a, 0)
            actual = rate.exact_rate(*chemical, 0, n=16) / rate.eq23_rate(*chemical, 0)
            self.assertAlmostEqual(actual, expected, places=7)

    def test_sign_equilibrium_scaling(self):
        chemical = (300, 350, 250)
        forward = rate.exact_rate(*chemical, 10, n=12)
        reverse = rate.exact_rate(300, 250, 350, 10, n=12)
        self.assertAlmostEqual(reverse / forward, -1, places=12)
        self.assertEqual(rate.exact_rate(300, 300, 300, 10), 0)
        scaled = rate.exact_rate(600, 700, 500, 20, n=12)
        self.assertAlmostEqual(scaled / forward, 256, places=8)

    def test_independent_detailed_balance(self):
        forward = rate.phase_space_forward(1, 1.2, 0.8, 0.12, n=16)
        reverse = rate.phase_space_forward(1, 0.8, 1.2, 0.12, n=16)
        self.assertAlmostEqual(reverse / forward, np.exp(-0.4 / 0.12), places=13)

    def test_independent_finite_three_dimensional_integral(self):
        forward = rate.phase_space_forward(1, 1.2, 0.8, 0.12, n=16)
        independent = rate.finite_phase_space_3d(1, 1.2, 0.8, 0.12, n=12)
        self.assertAlmostEqual(independent / forward, 1, places=11)


if __name__ == "__main__":
    unittest.main()
