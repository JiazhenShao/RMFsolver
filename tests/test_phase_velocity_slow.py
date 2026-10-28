"""Spatial BVP regressions for the isobaric slow-front model."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicSpline

from RMFsolver import phase_velocity as pv
from RMFsolver import constants as const


N0 = 0.16 * const.MeV_fm**3


class SlowFrontTests(unittest.TestCase):
    def solve(self, closure, ms=0.0, **kwargs):
        name = "solve_front_" + closure + "_slow"
        self.assertTrue(callable(getattr(pv, name, None)), name)
        options = dict(nB_0minus=3 * N0, B_one_forth=180.0, ms=ms,
                       n_mesh=120, tol_bvp=1e-4, return_profile=True)
        if closure == "isothermal":
            options.update(T=10.0, a_0plus=0.6)
        else:
            options.update(T_0minus=10.0, T_0plus=5.0)
        options.update(kwargs)
        result = getattr(pv, name)(**options)
        self.assertTrue(result["success"], result.get("message"))
        return result

    def check_profile(self, result, closure, ms):
        self.assertEqual(result["hydrodynamic_model"], "nonrelativistic_isobaric")
        self.assertEqual(result["closure"], closure)
        self.assertEqual(result["rate_approximation"], "madsen_muB_over_3")
        self.assertEqual(result["rate_model"], "exact_nonleptonic")
        self.assertGreater(result["jB"], 0)
        np.testing.assert_allclose(result["nB"] * result["u"], result["jB"], rtol=1e-12)
        np.testing.assert_allclose(result["P"], result["P_0minus"], rtol=1e-8)
        np.testing.assert_allclose(result["a"], result["nK"] / result["nB"], rtol=1e-12)
        self.assertAlmostEqual(result["jK"][0] / result["jB"], result["a_0minus"], places=7)
        self.assertLess(result["closure_error_max"], 1e-8)
        self.assertLess(max(result["conservation_errors"].values()), 1e-8)
        self.assertLess(result["bvp_max_rms_residual"], 1e-4)
        self.assertLess(np.max(np.abs(result["boundary_residuals"])), 1e-7)
        self.assertLess(result["constitutive_residual_norm"], 2e-4)
        self.assertLess(result["reaction_residual_norm"], 2e-4)
        self.assertTrue(np.all(np.diff(result["r"]) <= 1e-8 * result["r"][0]))
        if closure == "adiabatic":
            np.testing.assert_allclose(result["h"] * result["u"],
                                       result["h_0minus"] * result["u_0minus"], rtol=1e-8)
        else:
            np.testing.assert_allclose(result["T"], 10.0, atol=1e-12)
        for i in np.linspace(0, len(result["T"]) - 1, 7, dtype=int):
            rate = pv._exact_kaon_transport_rate(result["muB"][i], result["muK"][i],
                                                result["T"][i], ms=ms)
            self.assertAlmostEqual(result["Gamma_K"][i] / rate["Gamma_K"], 1, places=7)
            micro = pv._microphysics_from_quark_state_energy(result["muB"][i],
                                                             result["T"][i], allow_zero_temperature=True)
            np.testing.assert_allclose(result["invD_K"][i], micro["invD"], rtol=1e-10)

    def test_isothermal_massless_and_massive(self):
        for ms in (0.0, 100.0):
            with self.subTest(ms=ms):
                result = self.solve("isothermal", ms)
                self.check_profile(result, "isothermal", ms)
                if ms:
                    self.assertGreater(result["nK_inf"], 0)
                    self.assertGreater(result["jK_inf"], 0)

    def test_adiabatic_massless_and_massive(self):
        for ms in (0.0, 100.0):
            with self.subTest(ms=ms):
                result = self.solve("adiabatic", ms)
                self.check_profile(result, "adiabatic", ms)
                if ms == 0:
                    self.assertLess(np.ptp(result["nB"]) / np.mean(result["nB"]), 1e-8)
                else:
                    self.assertGreater(np.ptp(result["nB"]) / np.mean(result["nB"]), 1e-4)
                prescribed_a = self.solve("adiabatic", ms, T_0plus=None,
                                          a_0plus=result["a_0plus"])
                self.assertAlmostEqual(prescribed_a["jB"] / result["jB"], 1, places=4)

    def test_cold_adiabatic_is_a_regular_seed_independent_limit(self):
        for ms in (0.0, 100.0):
            with self.subTest(ms=ms):
                cold = self.solve("adiabatic", ms, T_0plus=0.0)
                self.check_profile(cold, "adiabatic", ms)
                self.assertEqual(cold["T"][0], 0)
                self.assertEqual(cold["invD_K"][0], 0)
                self.assertTrue(np.isfinite(cold["r"][0]))
                self.assertTrue(np.all(cold["T"][1:] > 0))
                other = self.solve("adiabatic", ms, T_0plus=0.0,
                                   jB_guess=2 * cold["jB"])
                self.assertAlmostEqual(other["jB"] / cold["jB"], 1, places=4)
                errors = [abs(self.solve("adiabatic", ms, T_0plus=T)["jB"] / cold["jB"] - 1)
                          for T in (0.01, 1e-5)]
                self.assertLess(errors[1], errors[0])
                self.assertLess(errors[1], 0.01)

    def test_interface_selector_validation(self):
        self.assertTrue(callable(getattr(pv, "solve_front_adiabatic_slow", None)))
        for options in ({}, dict(a_0plus=0.3, T_0plus=2.0), dict(T_0plus=-1.0)):
            with self.assertRaises(ValueError):
                pv.solve_front_adiabatic_slow(10, 3 * N0, 180, **options)

    def test_physical_density_gradient_resolves_diffusion(self):
        result = self.solve("adiabatic", n_mesh=600)
        s = result["s_coord"]
        dnK_dx = CubicSpline(s, result["nK"])(s, 1) * (
            result["lambda_inf"] * (1 - s) / result["kappa_factor"])
        expected = result["invD_K"] * (result["jB"] * result["a"] - result["jK"])
        self.assertLess(np.max(np.abs(dnK_dx - expected)) / np.max(np.abs(expected)), 1e-4)

    def test_small_imbalance_adiabatic_paths_use_chemical_continuation(self):
        for fraction in (.01, .001, .0001):
            result = self.solve("adiabatic", a_0plus=fraction, T_0plus=None)
            self.assertAlmostEqual(result["a_0plus"], fraction, places=9)
        # The finite-ms isobar has a shallow temperature turn near equilibrium.
        massive = self.solve("adiabatic", 100, a_0plus=.0502713137744769, T_0plus=None)
        self.assertGreater(massive["T_0plus"], massive["T_inf"])

    def test_massive_temperature_turn_selects_larger_imbalance_root(self):
        by_fraction = self.solve("adiabatic", 100, a_0plus=.0503213137744769,
                                 T_0plus=None)
        self.assertGreater(by_fraction["T_0plus"], by_fraction["T_inf"])
        by_temperature = self.solve("adiabatic", 100, T_0plus=by_fraction["T_0plus"])
        self.assertLess(abs(by_temperature["jB"] / by_fraction["jB"] - 1), 1e-4)
        self.assertLess(abs(by_temperature["muK_0plus"] / by_fraction["muK_0plus"] - 1), 1e-4)

    def test_invalid_mesh_controls(self):
        for options in (dict(n_mesh=10), dict(tail_eps=1e-30)):
            with self.assertRaises(ValueError):
                pv.solve_front_isothermal_slow(10, 3 * N0, 180, a_0plus=.6, **options)

    def test_automatic_ceiling_and_continuation(self):
        automatic = self.solve("isothermal", a_0plus=np.nan, return_continuation=True)
        self.assertEqual(automatic["a_0plus_source"], "maximum")
        explicit = self.solve("isothermal", a_0plus=automatic["a_0plus"],
                              continuation_guess=automatic["continuation_state"])
        self.assertAlmostEqual(explicit["jB"] / automatic["jB"], 1, places=5)
        with self.assertRaises(ValueError):
            pv.solve_front_isothermal_slow(10, 3 * N0, 180, NM_type="SYM")
        with self.assertRaises(ValueError):
            pv.solve_front_adiabatic_slow(10, 3 * N0, 180, T_0plus=5,
                                         continuation_guess=automatic["continuation_state"])

    def test_massive_automatic_ceiling_starts_above_equilibrium_fraction(self):
        automatic = self.solve("isothermal", 100, a_0plus=np.nan)
        self.check_profile(automatic, "isothermal", 100)
        self.assertGreater(automatic["a_0plus"], automatic["a_inf"])
        self.assertAlmostEqual(automatic["muB_0plus"] +
                               automatic["a_0plus"] * automatic["muK_0plus"],
                               automatic["muB_0minus"], places=6)
        explicit = self.solve("isothermal", 100, a_0plus=automatic["a_0plus"])
        self.assertLess(abs(explicit["jB"] / automatic["jB"] - 1), 1e-5)

    def test_table_mesh_and_tail_convergence(self):
        for closure, options in (("isothermal", {}), ("adiabatic", dict(T_0plus=0.0))):
            base = self.solve(closure, **options)
            for changed in (dict(n_mesh=240, tol_bvp=2e-5), dict(tail_eps=1e-6),
                            dict(kappa_factor=2.0), dict(path_rtol=2e-8)):
                with self.subTest(closure=closure, changed=changed):
                    refined = self.solve(closure, **options, **changed)
                    self.assertLess(abs(refined["jB"] / base["jB"] - 1), 2e-4)

    def test_massive_cold_table_mesh_and_tail_convergence(self):
        base = self.solve("adiabatic", 100, T_0plus=0.0)
        for changed in (dict(n_mesh=240, tol_bvp=2e-5), dict(tail_eps=1e-6),
                        dict(path_rtol=5e-7)):
            with self.subTest(changed=changed):
                refined = self.solve("adiabatic", 100, T_0plus=0.0, **changed)
                self.check_profile(refined, "adiabatic", 100)
                self.assertLess(abs(refined["jB"] / base["jB"] - 1), 2e-4)
                # Compare the physical fields on the shared compact domain.
                grid = np.linspace(0, .999, 101)
                for key in ("T", "nK", "jK"):
                    # Linear resampling introduces an O(ds^2) error larger
                    # than the fourth-order collocation/profile error.
                    expected = CubicSpline(base["s_coord"], base[key])(grid)
                    actual = CubicSpline(refined["s_coord"], refined[key])(grid)
                    self.assertLess(np.max(np.abs(actual - expected)) /
                                    np.max(np.abs(expected)), 3e-4)

    def test_collocation_callbacks_do_not_call_the_eos(self):
        original = pv.solve_bvp
        def guarded(*args, **kwargs):
            with patch.object(pv, "_slow_quark_state", side_effect=AssertionError("EOS in BVP")):
                return original(*args, **kwargs)
        with patch.object(pv, "solve_bvp", side_effect=guarded):
            self.solve("adiabatic", T_0plus=0)

    def test_failure_results_keep_model_metadata(self):
        failures = [pv.solve_front_adiabatic_slow(10, 3 * N0, 180, T_0plus=999),
                    pv.solve_front_isothermal_slow(10, 3 * N0, 180, a_0plus=.6,
                                                   n_mesh=20, max_nodes=20, tol_bvp=1e-10)]
        for result in failures:
            self.assertFalse(result["success"])
            self.assertEqual(result["rate_model"], "exact_nonleptonic")
            self.assertEqual(result["rate_approximation"], "madsen_muB_over_3")
            self.assertEqual(result["hydrodynamic_model"], "nonrelativistic_isobaric")
            self.assertEqual(result["alpha_s"], .4)

    def test_profile_failure_and_corrupt_continuation_are_reported(self):
        with patch.object(pv._SlowFrontPath, "coordinate", side_effect=RuntimeError("invalid inverse")):
            failed = pv.solve_front_isothermal_slow(10, 3 * N0, 180, a_0plus=.6)
        self.assertFalse(failed["success"])
        self.assertEqual(failed["status"], "profile_rejected")
        self.assertEqual(failed["rate_approximation"], "madsen_muB_over_3")
        result = self.solve("isothermal", return_continuation=True)
        bad = result["continuation_state"].copy()
        bad["r_fraction"] = np.full_like(bad["r_fraction"], np.nan)
        with self.assertRaises(ValueError):
            pv.solve_front_isothermal_slow(10, 3 * N0, 180, a_0plus=.6,
                                          continuation_guess=bad)

    def test_warm_slow_limit_matches_existing_relativistic_solvers(self):
        iso = self.solve("isothermal")
        old_iso = pv.solve_front_isothermal(10, 3 * N0, 180, a_0plus=.6,
                                           n_mesh=120, tail_eps=1e-8, tol_bvp=1e-4)
        self.assertTrue(old_iso["success"], old_iso.get("message"))
        self.assertLess(abs(old_iso["jB"] / iso["jB"] - 1), 2e-4)
        adia = self.solve("adiabatic")
        old_adia = pv.solve_front_energy_conserving_uNmax(10, 3 * N0, 180,
                    T_0plus=5, n_mesh=120, tail_eps=1e-8, tol_bvp=1e-4)
        self.assertTrue(old_adia["success"], old_adia.get("message"))
        self.assertLess(abs(old_adia["jB"] / adia["jB"] - 1), 2e-4)


class ColdIsothermalSlowTests(unittest.TestCase):
    """At 0.01 MeV the conversion layer is 1e4-1e7 times thinner than the equilibrium tail."""
    B14 = 189.1565957288247
    SPEED = const.c_km * 1e3

    def cold(self, nB_ratio, mode, **kwargs):
        return pv.solve_front_isothermal_slow(0.01, nB_ratio * N0, self.B14,
                                              initial_guess_mode=mode, **kwargs)

    def check_accepted(self, result):
        self.assertTrue(result["success"], result.get("message"))
        self.assertLessEqual(result["closure_error_max"], 1e-8)
        self.assertLessEqual(max(result["conservation_errors"].values()), 1e-8)

    def test_default_initialization_still_fails_cold(self):
        result = self.cold(3.05, "default")
        self.assertFalse(result["success"])
        self.assertEqual(result["initial_guess_mode"], "default")

    def test_shooting_seed_resolves_the_layer_by_collocation(self):
        result = self.cold(3.05, "shooting")
        self.check_accepted(result)
        self.assertEqual(result["solution_method"], "collocation")
        # Brute-force default guess with n_mesh=3000 gave 3249.94 m/s on 201,775 nodes.
        self.assertAlmostEqual(result["u_0minus"] * self.SPEED, 3249.93, delta=0.02)
        self.assertLess(result["bvp_nodes"], 20000)

    def test_auto_shoots_dense_cold_states(self):
        # Layer half-points at s = 6.6e-7 and 2.4e-8; the default first interval is 3.4e-4.
        for nB_ratio, speed in ((3.5, 46578.8), (4.8, 520113.0)):
            with self.subTest(nB_ratio=nB_ratio):
                result = self.cold(nB_ratio, "auto")
                self.check_accepted(result)
                self.assertEqual(result["initial_guess_used"], "shooting")
                self.assertLess(abs(result["u_0minus"] * self.SPEED / speed - 1), 1e-4)
                self.assertLess(result["shooting_flux_change"], 1e-6)

    def test_validated_shooting_fallback_when_collocation_fails(self):
        with patch.object(pv, "solve_bvp", side_effect=ValueError("singular Jacobian")):
            result = self.cold(3.05, "shooting", return_profile=True)
        self.check_accepted(result)
        self.assertEqual(result["solution_method"], "shooting")
        self.assertEqual(result["collocation_message"], "singular Jacobian")
        self.assertAlmostEqual(result["u_0minus"] * self.SPEED, 3249.93, delta=0.02)
        self.assertLessEqual(np.max(np.diff(result["a"])), 1e-8)
        self.assertAlmostEqual(result["jK"][0] / result["jB"], result["a_0minus"], places=7)

    def test_auto_keeps_default_initialization_when_layer_is_resolved(self):
        default = pv.solve_front_isothermal_slow(10.0, 3.5 * N0, self.B14)
        auto = pv.solve_front_isothermal_slow(10.0, 3.5 * N0, self.B14, initial_guess_mode="auto")
        self.assertTrue(auto["success"], auto.get("message"))
        self.assertEqual(auto["initial_guess_used"], "default")
        # Only the tail condition differs: direct instead of tabulated rate slope.
        self.assertLess(abs(auto["jB"] / default["jB"] - 1), 1e-10)

    def test_exact_equilibrium_rate_slope(self):
        failed = self.cold(3.5, "default")
        path = pv._slow_front_path("isothermal", failed["P_0minus"],
                                   failed["h_0minus"] / failed["nB_0minus"], 0.01, self.B14,
                                   0.0, failed["a_0plus"], None, 1e-6)
        w = 1e-6   # muK = 1.5e-4 MeV, deep in the linear regime muK << 2*pi*T
        state = path.isobar(w * path.interface["muK"], 0.01, path.inf["muB"])
        rate = pv._exact_kaon_transport_rate(state["muB"], state["muK"], 0.01)["Gamma_K"]
        independent = rate / (path.primitive(w) - path.primitive(0))
        self.assertLess(abs(path.equilibrium_rate_slope() / independent - 1), 1e-4)
        self.assertGreater(path.drate_dr / independent - 1, 0.1)   # tabulated slope misses muK < 2 pi T

    def test_default_mode_is_unchanged(self):
        result = pv.solve_front_isothermal_slow(10.0, 3 * N0, 180.0, a_0plus=0.6, n_mesh=120)
        self.assertEqual(result["jB"], 2.0174252130852697)   # pre-fix value, same machine

    def test_failed_tight_check_never_validates_a_shooting_fallback(self):
        original = pv._slow_front_shooting_flux
        def tight_fails(*args, **kwargs):   # both tight self-check settings fail
            if kwargs.get("bracket") is not None:
                raise RuntimeError("Backward shooting found no interface flux")
            return original(*args, **kwargs)
        with patch.object(pv, "_slow_front_shooting_flux", side_effect=tight_fails), \
                patch.object(pv, "solve_bvp", side_effect=ValueError("singular Jacobian")):
            result = self.cold(3.05, "shooting")
        self.assertFalse(result["success"])
        self.assertTrue(np.isnan(result["shooting_flux_change"]))

    def test_shooting_brackets_flux_next_to_udd_boundary(self):
        # a(0+) = 0.99997: the flux scales as (1-a)^(-1/2), far below an advection-free (1-a)^(-1) centre.
        result = pv.solve_front_isothermal_slow(29.07, 5.0686 * N0, self.B14, initial_guess_mode="auto")
        self.assertGreater(result["a_0plus"], 0.9999)
        self.check_accepted(result)
        self.assertEqual(result["initial_guess_used"], "shooting")

    def test_invalid_initial_guess_mode(self):
        with self.assertRaises(ValueError):
            pv.solve_front_isothermal_slow(10, 3 * N0, 180, a_0plus=.6, initial_guess_mode="layer")


if __name__ == "__main__":
    unittest.main()
