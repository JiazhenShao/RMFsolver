import inspect
import unittest
from unittest.mock import patch

import numpy as np

from RMFsolver import RMFparameter as para
from RMFsolver import constants as const
from RMFsolver import phase_velocity as pv


N0 = 0.16 * const.MeV_fm**3


class IsothermalNumericalLocalFractionTests(unittest.TestCase):
    @staticmethod
    def _warm_profile(**overrides):
        kwargs = {
            "T": 10.0,
            "nB_0minus": 2.0 * N0,
            "B_one_forth": 180.0,
            "a_0plus": 0.6,
            "ms": 0.0,
            "param": para.paraQMCRMF3,
            "NM_type": "PNM",
            "tail_eps": 1.0e-6,
            "n_mesh": 80,
            "tol_bvp": 1.0e-3,
            "max_nodes": 3000,
            "jB_guess": 0.5,
            "jB_bounds": (1.0e-4, 5.0),
            "kappa_factor": 1.0,
            "return_profile": True,
            "verb": False,
        }
        kwargs.update(overrides)
        return pv.solve_front_isothermal(**kwargs)

    def test_public_signature_keeps_existing_defaults(self):
        signature = inspect.signature(pv.solve_front_isothermal)

        self.assertEqual(signature.parameters["ms"].default, 0.0)
        self.assertIs(signature.parameters["param"].default, para.paraQMCRMF3)
        self.assertEqual(signature.parameters["NM_type"].default, "PNM")

    def test_warm_profile_uses_physical_nK_jK_and_local_fraction(self):
        result = self._warm_profile()

        self.assertTrue(result.get("success"), result.get("message"))
        self.assertEqual(result["composition_definition"], "nK_over_local_nB")
        self.assertEqual(
            result["current_definition"],
            "u_nK_minus_D_K_dnK_dx",
        )
        self.assertEqual(result["rate_model"], "exact_nonleptonic")
        self.assertEqual(result["diffusion_model"], "local_muB_fixed_T")
        self.assertEqual(result["a_0minus"], 1.0)
        self.assertAlmostEqual(result["a_0plus_derived"], 0.6, places=8)
        self.assertNotIn("q", result)
        self.assertNotIn("q_end", result)

        for key in ("a", "nK", "jK", "nB", "u", "muB", "muK", "D_K", "Gamma_K"):
            values = np.asarray(result[key], dtype=float)
            self.assertGreater(values.size, 0, key)
            self.assertTrue(np.all(np.isfinite(values)), key)

        np.testing.assert_allclose(
            np.asarray(result["a"]),
            np.asarray(result["nK"]) / np.asarray(result["nB"]),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(result["jK"][0] / result["jB"], 1.0, places=8)
        self.assertGreater(np.ptp(np.asarray(result["D_K"], dtype=float)), 0.0)
        self.assertTrue(result["a_monotone_nonincreasing"])
        self.assertLess(result["constitutive_residual_norm"], 1.0e-8)
        self.assertLess(result["reaction_residual_norm"], 1.0e-8)
        self.assertLess(result["closure_error_max"], 1.0e-8)
        self.assertLess(
            np.max(np.abs(np.asarray(result["boundary_residuals"]))),
            1.0e-8,
        )

        interface = pv._quark_thermo_state(
            result["muB_0plus"],
            result["muK_0plus"],
            180.0,
            10.0,
            result["jB"],
            ms=0.0,
        )
        downstream = pv._quark_thermo_state(
            result["muB_inf"],
            result["muK_inf"],
            180.0,
            10.0,
            result["jB"],
            ms=0.0,
        )
        for endpoint in (interface, downstream):
            self.assertAlmostEqual(
                endpoint["Pi"],
                result["Pi"],
                delta=1.0e-8 * max(abs(result["Pi"]), 1.0),
            )
            self.assertAlmostEqual(
                endpoint["nB"] * endpoint["u"],
                result["jB"],
                delta=1.0e-12 * max(abs(result["jB"]), 1.0),
            )
        self.assertAlmostEqual(
            interface["nK"] / interface["nB"],
            result["a_0plus"],
            places=8,
        )
        self.assertAlmostEqual(downstream["muK"], 0.0)

        exact_rates = np.array(
            [
                pv._exact_kaon_transport_rate(muB, muK, 10.0, ms=0.0)[
                    "Gamma_K"
                ]
                for muB, muK in zip(result["muB"], result["muK"])
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            result["Gamma_K"],
            exact_rates,
            rtol=1.0e-12,
            atol=0.0,
        )

    def test_finite_strange_mass_uses_shifted_equilibrium_tail(self):
        result = self._warm_profile(
            ms=100.0,
            tail_eps=1.0e-2,
            n_mesh=12,
            tol_bvp=5.0e-2,
            max_nodes=100,
        )

        self.assertTrue(result.get("success"), result.get("message"))
        self.assertGreater(result["nK_inf"], 0.0)
        self.assertGreater(result["a_inf"], 0.0)
        self.assertLess(result["a_inf"], result["a_0plus"])
        self.assertAlmostEqual(result["jK_inf"], result["u_inf"] * result["nK_inf"])
        self.assertAlmostEqual(result["muK_inf"], 0.0)
        self.assertAlmostEqual(
            pv._exact_kaon_transport_rate(
                result["muB_inf"],
                result["muK_inf"],
                10.0,
                ms=100.0,
            )["Gamma_K"],
            0.0,
        )
        np.testing.assert_allclose(
            np.asarray(result["a"]),
            np.asarray(result["nK"]) / np.asarray(result["nB"]),
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    def test_upstream_fraction_tracks_nuclear_proton_fraction(self):
        pnm = pv._isothermal_upstream_nuclear_state(
            10.0,
            2.0 * N0,
            para.paraQMCRMF3,
            "PNM",
        )
        self.assertEqual(pnm["proton_fraction_0minus"], 0.0)
        self.assertEqual(pnm["a_0minus"], 1.0)

        with (
            patch.object(pv, "RMFsolveSYM", return_value=object()),
            patch.object(pv, "pressure_RMF", return_value=20.0),
            patch.object(pv, "edens_RMF", return_value=(None, 80.0, None)),
        ):
            symmetric = pv._isothermal_upstream_nuclear_state(
                10.0,
                2.0,
                para.paraQMCRMF3,
                "SYM",
            )
        self.assertEqual(symmetric["proton_fraction_0minus"], 0.5)
        self.assertEqual(symmetric["a_0minus"], 0.75)

        with (
            patch.object(
                pv,
                "RMFsolve",
                autospec=True,
                return_value=object(),
            ),
            patch.object(
                pv,
                "baryon_density_RMF",
                return_value=(2.0, {"n_n": 1.6, "n_p": 0.4}),
            ),
            patch.object(pv, "pressure_RMF", return_value=20.0),
            patch.object(pv, "edens_RMF", return_value=(None, 80.0, None)),
        ):
            beta_equilibrated = pv._isothermal_upstream_nuclear_state(
                10.0,
                2.0,
                para.paraQMCRMF3,
                "Beta_eq",
            )
        self.assertAlmostEqual(beta_equilibrated["proton_fraction_0minus"], 0.2)
        self.assertAlmostEqual(beta_equilibrated["a_0minus"], 0.9)

    def test_local_fraction_and_forward_branch_domains_are_enforced(self):
        for a_0plus in (0.0, 1.0):
            with self.assertRaisesRegex(RuntimeError, "a_0plus"):
                self._warm_profile(a_0plus=a_0plus, return_profile=False)

        with patch.object(
            pv,
            "_isothermal_upstream_nuclear_state",
            return_value={
                "P_0minus": 20.0,
                "e_0minus": 80.0,
                "h_0minus": 100.0,
                "nB_0minus": 2.0,
                "proton_fraction_0minus": 0.5,
                "a_0minus": 0.75,
                "nK_0minus": 1.5,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "a_0plus < a_0minus"):
                pv.solve_front_isothermal(
                    10.0,
                    2.0,
                    180.0,
                    0.8,
                    NM_type="SYM",
                )


if __name__ == "__main__":
    unittest.main()
