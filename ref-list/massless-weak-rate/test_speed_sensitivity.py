"""Regression checks for rate-integral speed estimates, not front solutions."""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import speed_sensitivity as sensitivity


class SpeedSensitivityTests(unittest.TestCase):
    @staticmethod
    def plotting_rows():
        return [dict(T=temperature, a0plus=composition, exact_speed_ratio=1.0,
                     eq23_speed_ratio=1 + composition,
                     sommerfeld_T2_speed_ratio=(1 if temperature == 0 else
                                                0.5 + 2 * composition),
                     eq23_muB_speed_ratio=1 + composition / 2)
                for temperature in (0, 20, 50, 100)
                for composition in (0.146, 0.5, 0.9, 1)]

    def test_all_four_isothermal_temperatures_are_configured(self):
        self.assertEqual(getattr(sensitivity, "TEMPERATURES", ()), (0, 20, 50, 100))

    def test_four_panels_show_every_model_without_clipping(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = self.plotting_rows()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plt, "close"), patch.object(plt.Figure, "savefig"):
                sensitivity.make_plot(Path(directory), rows)
                figure = plt.gcf()
            try:
                self.assertEqual(len(figure.axes), 4)
                for axis, temperature in zip(figure.axes, (0, 20, 50, 100)):
                    self.assertEqual(axis.get_subplotspec().get_gridspec().get_geometry(), (2, 2))
                    self.assertEqual(len(axis.get_legend_handles_labels()[0]), 4)
                    lower, upper = axis.get_ylim()
                    for row in rows:
                        if row["T"] == temperature:
                            for key in ("exact_speed_ratio", "eq23_speed_ratio",
                                        "sommerfeld_T2_speed_ratio", "eq23_muB_speed_ratio"):
                                self.assertLess(lower, row[key])
                                self.assertGreater(upper, row[key])
            finally:
                plt.close(figure)

    def test_generated_table_contains_hot_temperature_rows(self):
        summary = dict(max_exact_integral_medium_to_fine=1e-8,
                       max_eq23_integral_medium_to_fine=1e-8,
                       max_T2_integral_medium_to_fine=1e-8,
                       max_independent_speed_ratio_relative_change=1e-8)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sensitivity, "HERE", Path(directory)):
                sensitivity.write_tex_summary(self.plotting_rows(), summary)
            generated = (Path(directory) / "speed-sensitivity-results.tex").read_text()
        for temperature in (0, 20, 50, 100):
            for composition in (0.146, 0.5, 0.9):
                self.assertIn(f"{temperature:g} & {composition:g} &", generated)
        self.assertIn(r"T=100\,\mathrm{MeV}", generated)

    def test_three_rates_use_one_eos_state(self):
        from cold_sympy import cold_phase_space
        from thermal_sympy import phase_space_t2
        from validate_rate import PREF, eq23_rate, exact_rate, neutral_fixed_a

        for temperature in (20, 50, 100):
            with self.subTest(temperature=temperature):
                chemical = neutral_fixed_a(1200, 0.5, temperature)
                actual = sensitivity.rates_at_composition(1200, 0.5, temperature, phase_order=12)
                expected = [exact_rate(*chemical, temperature, n=12),
                            eq23_rate(*chemical, temperature),
                            PREF * (cold_phase_space(*chemical)
                                    + temperature**2 * phase_space_t2(*chemical))]
                np.testing.assert_allclose(actual, expected, rtol=1e-14)

    def test_closed_form_quadrature_reproduces_saved_eq23_integrals(self):
        from validate_rate import eq23_rate, read_grid
        rows = read_grid(sensitivity.HERE / "results" / "speed-sensitivity.csv")
        for temperature in (0, 20, 50, 100):
            selected = sorted((row for row in rows if row["T"] == temperature),
                              key=lambda row: row["a0plus"])
            totals = sensitivity.closed_form_integrals(
                eq23_rate, 1200, temperature, [row["a0plus"] for row in selected])
            np.testing.assert_allclose(totals, [row["eq23_integral"] for row in selected],
                                       rtol=1e-13)

    def test_cold_integral_estimates_match_recorded_benchmarks(self):
        rows = sensitivity.integrated_estimates(1200, 0, [0.146, 0.5, 0.9],
                                                outer_order=16, phase_order=16)
        np.testing.assert_allclose([1 / row["eq23_speed_ratio"] for row in rows],
                                   [0.97291787, 0.88606401, 0.70220980], rtol=5e-7)
        for row in rows:
            self.assertEqual(row["exact_speed_ratio"], 1)
            self.assertAlmostEqual(row["sommerfeld_T2_speed_ratio"] / row["exact_speed_ratio"],
                                   1, places=11)

    def test_warm_integral_estimates_match_recorded_benchmarks(self):
        rows = sensitivity.integrated_estimates(1200, 20, [0.146, 0.5, 0.9],
                                                outer_order=12, phase_order=12)
        np.testing.assert_allclose([1 / row["eq23_speed_ratio"] for row in rows],
                                   [0.91318648, 0.86805296, 0.70972689], rtol=5e-7)

    def test_hot_integral_estimates_match_phase_refined_benchmarks(self):
        expected = {
            50: [[1.15610967, 1.11695403], [1.19610071, 1.04953144], [1.39349749, 1.00513768]],
            100: [[1.05141117, 1.00812157], [1.08762181, 0.92941397], [1.25887709, 0.88417623]],
        }
        for temperature, reference in expected.items():
            with self.subTest(temperature=temperature):
                rows = sensitivity.integrated_estimates(1200, temperature, [0.146, 0.5, 0.9],
                                                       outer_order=12, phase_order=12)
                np.testing.assert_allclose(
                    [[row["eq23_speed_ratio"], row["sommerfeld_T2_speed_ratio"]] for row in rows],
                    reference, rtol=1e-6)
                self.assertTrue(all(row["exact_speed_ratio"] == 1 for row in rows))

    def test_rejects_zero_or_unordered_endpoints(self):
        for compositions in ([0], [0.5, 0.146], [1.01]):
            with self.assertRaises(ValueError):
                sensitivity.integrated_estimates(1200, 0, compositions)


if __name__ == "__main__":
    unittest.main()
