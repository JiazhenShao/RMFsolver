"""Temperature-strip xi fits of the isothermal contours (new_paper_calculations/26-10-27/isothermal/xi_fit)."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "new_paper_calculations/26-10-27/isothermal"
SCIPY_RUNTIME = ROOT / "local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64"
for path in (SCIPY_RUNTIME, WORKFLOW / "xi_fit", WORKFLOW):
    if str(path) not in sys.path and path.is_dir():
        sys.path.insert(0, str(path))


class StripFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fit_xi_strips as strips
        cls.strips = strips
        cls.tables = {tag: strips.tabulate(*strips.load_panel(tag), strips.PANEL_LEVELS[tag])
                      for tag in strips.PANEL_LEVELS}
        cls.panel_fits = json.loads((WORKFLOW / "xi_fit/xi_fit.json").read_text())["fits"]

    def test_whole_panel_strip_reproduces_the_panel_contour_fit(self):
        for tag, table in self.tables.items():
            with self.subTest(panel=tag):
                fit = self.strips.strip_fit(table, (-np.inf, np.inf))
                self.assertFalse(fit["at_bound"])
                self.assertAlmostEqual(fit["xi"], self.panel_fits[tag]["contour"]["xi"], delta=0.005)

    def test_strip_fit_matches_direct_minimization_on_its_crossings(self):
        from scipy.optimize import minimize_scalar
        import _iso_xi as xi_cache
        strips = self.strips
        cache, numerical = strips.load_panel("lowT")
        levels = strips.PANEL_LEVELS["lowT"]
        table = self.tables["lowT"]
        T_range = (0.1, 10 ** -0.5)
        inside = (table["T_cross"] >= T_range[0]) & (table["T_cross"] <= T_range[1])

        def misfit(xi):
            offsets = xi_cache.contour_offsets(cache, numerical, xi, levels)
            return float(np.sqrt(np.nanmean(offsets[inside] ** 2)))
        direct = minimize_scalar(misfit, bounds=(-0.99, 4.0), method="bounded", options=dict(xatol=1e-6)).x
        self.assertAlmostEqual(strips.strip_fit(table, T_range)["xi"], direct, delta=0.005)

    def test_bootstrap_interval_brackets_the_fit(self):
        fit = self.strips.strip_fit(self.tables["highT"], (20.0, 30.0), resamples=200,
                                    rng=np.random.default_rng(1))
        self.assertGreater(fit["n_crossings"], 20)
        self.assertLessEqual(fit["xi_16"], fit["xi"])
        self.assertGreaterEqual(fit["xi_84"], fit["xi"])

    def test_strips_tile_each_panel_without_overlap(self):
        for tag, edges in self.strips.STRIPS.items():
            with self.subTest(panel=tag):
                for (low, high), (next_low, _) in zip(edges, edges[1:]):
                    self.assertLess(low, high)
                    self.assertEqual(high, next_low)
        self.assertEqual(self.strips.STRIPS["lowT"][-1][1], self.strips.STRIPS["highT"][0][0])   # seam at 10 MeV


if __name__ == "__main__":
    unittest.main()
