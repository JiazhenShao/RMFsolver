import json
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class InterfaceNotebookTests(unittest.TestCase):
    def setUp(self):
        nb = json.loads((ROOT / "26-08-31.ipynb").read_text())
        self.sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        self.ns = {}
        exec(self.sources[0], self.ns)

    def test_short_valid_notebook(self):
        self.assertLess(sum(len(s.splitlines()) for s in self.sources), 240)
        self.assertNotIn("class ", self.sources[0])
        for source in self.sources:
            compile(source, "notebook_cell", "exec")
            if "plt.subplots(" in source:
                self.assertTrue(source.startswith("# Tunable plotting parameters"))

    def test_matching_comparison_is_on_muK_temperature_plane(self):
        comparison = self.sources[1]
        self.assertIn("T_pressure", comparison)
        self.assertIn("T_enthalpy", comparison)
        self.assertIn("zero_temperature_muK", comparison)
        self.assertIn("TEMPERATURE_AXIS_LIMIT_MEV", comparison)
        self.assertIn("axes[0].set_ylim", comparison)
        self.assertNotIn('np.linspace(0., intersection["muK_0plus"]', comparison)
        self.assertIn(r'$\mu_K(0^+)$ [MeV]', comparison)
        self.assertIn(r'$T^{h/n_B}-T^P$ [MeV]', comparison)

    def test_boundary_comparison_includes_collaborator_style(self):
        calculation, plot = self.sources[2:4]
        self.assertIn("COLLABORATOR_STABLE_SOURCE", calculation)
        self.assertIn("COLLABORATOR_A_SOURCE", calculation)
        self.assertIn("collaborator_stable", calculation)
        self.assertIn("collaborator_A", calculation)
        self.assertIn("local_analysis_results/FL26-2/", calculation)
        self.assertIn("DIRECT_POINT_DENSITY_FACTOR = 1.5", calculation)
        self.assertIn('CLUSTER_COLOR = "black"', plot)
        self.assertIn('OURS_COLOR = "#E69F00"', plot)
        self.assertIn('COLLABORATOR_COLOR = "#0072B2"', plot)
        self.assertIn("ax.scatter", plot)

    def test_fixed_upstream_interface_state_returns_muK_and_ns(self):
        self.ns["matching_residuals"] = lambda muB, muK, T, n, Tp: np.array([
            (muB + muK - 820) / 1000, (muB + 2 * muK - 840) / 1000])
        self.ns["quark_outputs"] = lambda muB, muK, T: {
            "n_s": muK - 20, "nB": 4, "nK": 2, "a": 0.5, "mu_s": muK - 20}
        result = self.ns["solve_0plus_state"](7, 2, 0.1, guess=(800, 20))
        self.assertAlmostEqual(result["muB_0plus"], 800, places=6)
        self.assertAlmostEqual(result["muK_0plus"], 20, places=6)
        self.assertAlmostEqual(result["n_s_0plus"], 0, places=6)

    def test_boundaries_vary_only_the_external_density_coordinate(self):
        self.ns["N0"] = 1.0
        self.ns["matching_residuals"] = lambda muB, muK, T, n, Tp: np.array([
            (muB + n + muK - 1000) / 1000,
            (muB + 2 * n + muK - (996 + T)) / 1000])
        self.ns["quark_outputs"] = lambda muB, muK, T: {
            "n_s": muK - 5, "nB": 4, "nK": 4, "a": 1, "mu_s": muK - 5}
        stable = self.ns["stable_boundary_at_T"](7, 0.1, (3, 997))
        ceiling = self.ns["A_boundary_at_T"](7, 0.001, (3, 992, 5))
        self.assertAlmostEqual(stable["nB_0minus_over_n0"], 3, places=6)
        self.assertAlmostEqual(ceiling["nB_0minus_over_n0"], 3, places=6)
        self.assertAlmostEqual(ceiling["n_s_over_nB"], 0, places=6)


if __name__ == "__main__":
    unittest.main()
