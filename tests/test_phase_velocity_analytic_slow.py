"""Independent formula and EOS checks for the slow analytical branches."""
import inspect
import unittest
from functools import lru_cache
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad

from RMFsolver import phase_velocity as pv
from RMFsolver import constants as const


N0 = 0.16 * const.MeV_fm**3
NAMES = ("analytic_velocity_isothermal_slow", "analytic_velocity_adiabatic_slow",
         "semi_analytic_velocity_adiabatic_slow")


def weak_gamma(muB):
    return (128 / (135 * np.pi**3) * const.G_Fermi**2
            * np.cos(const.Cabibbo_angle_rad)**2 * np.sin(const.Cabibbo_angle_rad)**2
            * (muB / 3)**5)


def leading_diffusion(muB, T):
    qD = muB / 3 * np.sqrt(6 * 0.4 / np.pi)
    return np.pi * qD**(2 / 3) / (24 * 0.4**2 * 1.8131745180482931 * T**(5 / 3))


def independent_quadratic_endpoint(pressure, H, bag):
    """Find all polynomial roots, independently of the endpoint's scalar solve."""
    C = 108 * np.pi**2 * (pressure + bag**4)
    roots = np.roots([1.0, -2.0, 0.0, 0.0, C / H**4])
    physical = [root.real * H for root in roots
                if abs(root.imag) < 1e-12 and 0 < root.real * H < C**0.25]
    if len(physical) != 1 or C**0.25 >= H:
        raise ValueError("No unique positive-temperature quadratic endpoint")
    muB = physical[0]
    T_inf = np.sqrt(muB * (H - muB) / (9 * np.pi**2))
    A = 3 / np.sqrt(2) * np.sqrt((H - muB) / muB)
    return muB, T_inf, A


def independent_quadratic_integral(muB, T_inf, A, T_0plus):
    """Integrate in T^(1/3), with the cold and equilibrium factors cancelled.

    T=T_inf*t^3 and a=A*sqrt(1-t^6), so D(T)=D(T_inf)*t^-5.
    Cancelling t^5 and sqrt(1-t^6) against the Jacobian gives this regular
    integrand even at t=0 and t=1, without using the closed-I2 expression.
    """
    prefactor = 3 * A**2 * leading_diffusion(muB, T_inf) * weak_gamma(muB)
    integral = quad(lambda t: A**2 * (1 - t**6)
                    + 81 * np.pi**2 * T_inf**2 / muB**2 * t**6,
                    (T_0plus / T_inf)**(1 / 3), 1, epsabs=1e-14, epsrel=1e-13)[0]
    return prefactor * integral


@lru_cache(None)
def solve(name, temp=5.0, xi=-0.5, path_rtol=1e-6):
    function = getattr(pv, name)
    if name == NAMES[0]:
        return function(10.0, 3 * N0, 180.0, a_0plus=0.6, xi=xi)
    options = dict(T_0plus=temp, xi=xi)
    if name == NAMES[2]:
        options["path_rtol"] = path_rtol
    return function(10.0, 3 * N0, 180.0, **options)


class AnalyticSlowTests(unittest.TestCase):
    def api(self, name):
        self.assertTrue(callable(getattr(pv, name, None)), name + " is missing")
        return getattr(pv, name)

    def result(self, name, **kwargs):
        self.api(name)
        result = solve(name, **kwargs)
        self.assertTrue(result["success"], result.get("message"))
        self.assertTrue(result["front_exists"])
        self.assertEqual(result["hydrodynamic_model"], "nonrelativistic_isobaric")
        self.assertEqual(result["rate_model"], "exact_nonleptonic")
        self.assertEqual(result["rate_approximation"], "madsen_muB_over_3")
        self.assertEqual(result["alpha_s"], 0.4)
        self.assertGreater(result["I2"], 0)
        self.assertLess(max(result["conservation_errors"].values()), 1e-8)
        self.assertAlmostEqual(result["velocity_m_s"] / result["u_0minus"],
                               const.c_km * 1000, places=5)
        for suffix in ("0minus", "0plus", "inf"):
            np.testing.assert_allclose(result["nB_" + suffix] * result["u_" + suffix],
                                       result["jB"], rtol=2e-15)
        return result

    def test_public_interfaces_and_domains(self):
        for name in NAMES:
            function = self.api(name)
            self.assertIn(name, pv.__all__)
            signature = inspect.signature(function)
            self.assertEqual(signature.parameters["xi"].default, 0.0)
            self.assertEqual(signature.parameters["ms"].default, 0.0)
            self.assertEqual(signature.parameters["NM_type"].default, "PNM")
            options = {} if name == NAMES[0] else {"T_0plus": 5.0}
            for invalid in ({"ms": 100}, {"NM_type": "SYM"}, {"xi": np.nan}):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    function(10, 3 * N0, 180, **options, **invalid)
            with self.assertRaises(ValueError):
                function(-1, 3 * N0, 180, **options)
        with self.assertRaises(ValueError):
            self.api(NAMES[0])(0, 3 * N0, 180, a_0plus=0.6)
        for a in (-0.1, 1, np.inf):
            with self.assertRaises(ValueError):
                self.api(NAMES[0])(10, 3 * N0, 180, a_0plus=a)

    def test_isothermal_manuscript_integral_and_scaling(self):
        result = self.result(NAMES[0])
        muB, T, a = result["muB_0plus"], 10.0, 0.6
        D = leading_diffusion(muB, T)
        integral = quad(lambda fraction: D * weak_gamma(muB) *
                        (fraction**3 + 81 * np.pi**2 * T*T / muB**2 * fraction), 0, a,
                        epsabs=1e-30, epsrel=1e-12)[0]
        np.testing.assert_allclose(result["I2"], integral, rtol=3e-14)
        expected_speed_squared = (2*integral / ((result["nB_0minus"]/result["nB_inf"])**2
                                                * (1-a)*(1-0.5*a)))
        np.testing.assert_allclose(result["u_0minus"]**2, expected_speed_squared, rtol=3e-14)
        np.testing.assert_allclose(result["D_K"], D, rtol=3e-14)
        full = pv._microphysics_from_quark_state_isothermal_baseline(muB, T)
        self.assertGreater(D, full["D"])
        qD = muB / 3 * np.sqrt(6 * 0.4 / np.pi)
        np.testing.assert_allclose(D / full["D"],
            1 + np.pi**3 / (12 * result["h_D"]) * (T / qD)**(1 / 3), rtol=3e-14)
        speed = (98 * result["nB_inf"] / result["nB_0minus"] * (muB / 1200)**(17 / 6)
                 * (10 / T)**(5 / 6) * a * np.sqrt(a*a + 162*np.pi**2*T*T/muB**2)
                 / (0.4**(5 / 6) * np.sqrt((1-a)*(1-0.5*a))))
        # The manuscript prints a rounded 98 m/s, rather than 98.297... .
        np.testing.assert_allclose(result["velocity_m_s"], speed, rtol=0.0031)
        self.assertEqual(result["diffusion_approximation"], "leading_landau")
        self.assertEqual(result["density_reference"], "nB_inf")
        self.assertAlmostEqual(result["gamma_K"], weak_gamma(muB), delta=1e-25)

    def test_isothermal_exact_ceiling_and_explicit_equivalence(self):
        function = self.api(NAMES[0])
        automatic = function(10, 3*N0, 180, xi=-0.5)
        self.assertTrue(automatic["success"], automatic["message"])
        a = automatic["a_0plus"]
        self.assertAlmostEqual(automatic["muB_0plus"] + a*automatic["muK_0plus"],
                               automatic["muB_0minus"], places=6)
        self.assertAlmostEqual(a, automatic["nK_0plus"] / automatic["nB_0plus"], places=10)
        explicit = function(10, 3*N0, 180, a, xi=-0.5)
        self.assertTrue(explicit["success"], explicit["message"])
        np.testing.assert_allclose(automatic["jB"], explicit["jB"], rtol=1e-9)
        self.assertTrue(np.isnan(explicit["a_0plus_max"]))

    def test_quadratic_adiabatic_endpoints_and_independent_integral(self):
        for temp in (5.0, 1.0, 0.0):
            result = self.result(NAMES[1], temp=temp)
            H = result["h_0minus"] / result["nB_0minus"]
            nB = 4*(result["P_0minus"] + 180**4)/H
            muB, T_inf, A = independent_quadratic_endpoint(result["P_0minus"], H, 180)
            np.testing.assert_allclose(result["A_adb"], A, rtol=2e-13)
            np.testing.assert_allclose(result["T_inf"], T_inf, rtol=2e-13)
            for suffix in ("0plus", "inf"):
                np.testing.assert_allclose(result["nB_"+suffix], nB, rtol=2e-13)
                np.testing.assert_allclose(result["muB_"+suffix], muB, rtol=2e-13)
            a = result["a_0plus"]
            np.testing.assert_allclose(a*a + 81*np.pi**2*temp**2/(2*muB**2), A*A, rtol=2e-13)
            integral = independent_quadratic_integral(muB, T_inf, A, temp)
            np.testing.assert_allclose(result["I2"], integral, rtol=1e-12)
            self.assertEqual(result["eos_approximation"], "quadratic_pressure_second_order_density")
            self.assertEqual(result["conservation_eos"], result["eos_approximation"])
            np.testing.assert_allclose(result["gamma_K"], weak_gamma(muB), rtol=3e-14)
            if temp > 0:
                np.testing.assert_allclose(result["D_K_0plus"],
                                           leading_diffusion(muB, temp), rtol=3e-14)
            expected_speed_squared = (2*integral / ((result["nB_0minus"]/nB)**2
                                                     * (1-a)*(1-0.5*a)))
            np.testing.assert_allclose(result["u_0minus"]**2, expected_speed_squared, rtol=1e-12)
            if temp == 0:
                self.assertEqual(result["T_0plus"], 0)
                self.assertTrue(np.isinf(result["D_K_0plus"]))

    def test_quadratic_second_order_validation_points(self):
        # Unrounded independent targets saved before the implementation change.
        # The source report is endpoint-before-and-independent-targets.json in
        # local_analysis_results/26-10-26-second-order; rounded display values
        # cannot support the requested 1e-6 endpoint tolerance.
        cases = (
            (8.860992751744538, 3.0268624251597, 1190.7985710901767,
             9.306583545171428, 0.15625335814259514, 39.65471174992098),
            (27.83373842080244, 2.96875, 1181.4015087858008,
             17.327404486828094, 0.2932333599801995, 96.37487114556207),
            (70.6735943750199, 1.434488038438507, 1089.2432888101353,
             25.555560332224726, 0.46907026064504015, 304.4747205308038),
            (10.15113868209788, 4.500181704610503, 1265.3114878377821,
             54.77453968957801, 0.8654832551285321, 1372.3869971302497),
        )
        rounded_rows = (
            ("1190.7986", "9.3066", "0.1563", "39.65"),
            ("1181.4015", "17.3274", "0.2932", "96.37"),
            ("1089.2433", "25.5556", "0.4691", "304.47"),
            ("1265.3115", "54.7745", "0.8655", "1372.39"),
        )
        bag = 189.1565957288247
        for index, (temp, density, muB, T_inf, A, speed) in enumerate(cases):
            with self.subTest(index=index):
                result = self.api(NAMES[1])(temp, density*N0, bag, T_0plus=0, xi=-0.85)
                self.assertTrue(result["success"], result["message"])
                np.testing.assert_allclose([result["muB_inf"], result["T_inf"], result["A_adb"]],
                                           [muB, T_inf, A], rtol=1e-6)
                np.testing.assert_allclose(result["velocity_m_s"], speed, rtol=1e-4)
                self.assertEqual((f"{result['muB_inf']:.4f}", f"{result['T_inf']:.4f}",
                                  f"{result['A_adb']:.4f}", f"{result['velocity_m_s']:.2f}"),
                                 rounded_rows[index])
                self.assert_quadratic_endpoint_diagnostics(result, bag)
                if index < 3:
                    semi = self.api(NAMES[2])(temp, density*N0, bag, T_0plus=0, xi=-0.85)
                    self.assertTrue(semi["success"], semi["message"])
                    np.testing.assert_allclose(result["A_adb"], semi["a_0plus"], rtol=0.01)

    def assert_quadratic_endpoint_diagnostics(self, result, bag):
        muB, T_inf, A = result["muB_inf"], result["T_inf"], result["A_adb"]
        H = result["h_0minus"] / result["nB_0minus"]
        C = 108 * np.pi**2 * (result["P_0minus"] + bag**4)
        self.assertLess(muB, C**0.25)
        self.assertLess(C**0.25, H)
        residuals = result["quadratic_endpoint_residuals"]
        self.assertEqual(set(residuals), {"quartic", "pressure", "heat", "A_adb"})
        for name, residual in residuals.items():
            self.assertLessEqual(abs(residual), 1e-10, name)
        independent_residuals = {
            "quartic": (muB**4 - 2*H*muB**3 + C) / C,
            "pressure": (C - muB**4 - 18*np.pi**2*muB**2*T_inf**2) / C,
            "heat": (H - muB - 9*np.pi**2*T_inf**2/muB) / H,
            "A_adb": A / (1.5*np.sqrt(C-muB**4)/muB**2) - 1,
        }
        for name, residual in independent_residuals.items():
            self.assertLessEqual(abs(residual), 1e-10, name)
            self.assertAlmostEqual(residuals[name], residual, delta=2e-15, msg=name)
        nB_second_order = muB**3/(27*np.pi**2) + muB*T_inf**2/3
        np.testing.assert_allclose(result["nB_inf_second_order"], nB_second_order, rtol=2e-13)
        relative_difference = nB_second_order/result["nB_inf"] - 1
        self.assertAlmostEqual(result["nB_inf_second_order_relative_difference"],
                               relative_difference, delta=2e-13)
        exact_ratio_residual = result["h_inf"]/nB_second_order/H - 1
        self.assertAlmostEqual(result["unexpanded_enthalpy_per_baryon_relative_residual"],
                               exact_ratio_residual, delta=2e-13)
        self.assertGreater(relative_difference, 1e-10)
        self.assertLess(exact_ratio_residual, -1e-10)
        self.assertEqual(result["conservation_eos"], "quadratic_pressure_second_order_density")

    def test_quadratic_small_temperature_ceiling_ratio(self):
        H, bag, density = 1200.0, 180.0, 3*N0
        errors = []
        for offset in (1e-3, 1e-4, 1e-5):
            C = (H/(1 + offset))**4
            pressure = C/(108*np.pi**2) - bag**4
            upstream = {"P_0minus": pressure, "h_0minus": H*density}
            with patch.object(pv, "_isothermal_upstream_nuclear_state", return_value=upstream):
                result = self.api(NAMES[1])(0, density, bag, T_0plus=0)
            self.assertTrue(result["success"], result["message"])
            muB_old = np.cbrt(C/H)
            A_old = 1.5*np.sqrt((H-muB_old)/muB_old)
            ratio = result["A_adb"]/A_old
            np.testing.assert_allclose(ratio, np.sqrt(3), rtol=0.02)
            errors.append(abs(ratio/np.sqrt(3) - 1))
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[1], errors[0])

    def test_quadratic_rejects_nonpositive_temperature_root(self):
        density, bag = 3*N0, 180.0
        pressure = 0.0
        C = 108*np.pi**2*(pressure + bag**4)
        for H in (C**0.25, 0.99*C**0.25):
            with self.subTest(H=H):
                upstream = {"P_0minus": pressure, "h_0minus": H*density}
                with patch.object(pv, "_isothermal_upstream_nuclear_state", return_value=upstream):
                    result = self.api(NAMES[1])(0, density, bag, T_0plus=0)
                self.assertFalse(result["success"])
                self.assertFalse(result["front_exists"])
                self.assertEqual(result["status"], "no_adiabatic_equilibrium")
                self.assertEqual(result["eos_approximation"],
                                 "quadratic_pressure_second_order_density")

    def test_semi_analytical_full_eos_path_and_integral(self):
        for temp in (5.0, 0.0):
            result = self.result(NAMES[2], temp=temp)
            path = result["thermodynamic_path"]
            self.assertEqual(result["diffusion_approximation"], "full_two_term")
            np.testing.assert_allclose(path["nB"], result["nB_inf"], rtol=1e-8)
            np.testing.assert_allclose(path["P"], result["P_0minus"], rtol=1e-8)
            np.testing.assert_allclose(path["h"] / path["nB"],
                result["h_0minus"]/result["nB_0minus"], rtol=1e-8)
            for i in (0, len(path["T"])//2, -1):
                rate = pv._exact_kaon_transport_rate(path["muB"][i], path["muK"][i], path["T"][i])
                micro = pv._microphysics_from_quark_state_energy(path["muB"][i], path["T"][i],
                                                                allow_zero_temperature=True)
                np.testing.assert_allclose(path["Gamma_K"][i], rate["Gamma_K"], rtol=1e-12)
                np.testing.assert_allclose(path["invD_K"][i], micro["invD"], rtol=1e-12)
            # Independent chart quadrature, avoiding the stored rate spline in r.
            table = pv._slow_front_path("adiabatic", result["P_0minus"],
                result["h_0minus"]/result["nB_0minus"], 10.0, 180.0, 0.0, None, temp, 1e-6)
            def integrand(w):
                state = table.point(w)
                rate = pv._exact_kaon_transport_rate(state["muB"], state["muK"], state["T"])["Gamma_K"]
                return rate*table.primitive(w, 1)/result["nB_inf"]**2
            integral = quad(integrand, 0, 1, points=[table.core_join],
                            epsabs=1e-25, epsrel=2e-7)[0]
            np.testing.assert_allclose(result["I2"], integral, rtol=3e-7)

    def test_cold_limit_and_path_refinement(self):
        for name in NAMES[1:]:
            cold = self.result(name, temp=0.0)
            errors = [abs(self.result(name, temp=T)["jB"]/cold["jB"]-1) for T in (0.01, 1e-5)]
            self.assertLess(errors[1], errors[0])
            self.assertLess(errors[1], 0.01)
        refined = self.result(NAMES[2], temp=0.0, path_rtol=2.5e-7)
        np.testing.assert_allclose(refined["jB"], solve(NAMES[2], 0.0)["jB"], rtol=1e-6)

    def test_failure_metadata_and_no_moving_endpoint_calls(self):
        for name in NAMES:
            function = self.api(name)
            options = {"a_0plus": 0.6} if name == NAMES[0] else {"T_0plus": 5.0}
            with patch.object(pv, "_solve_analytic_inf_endpoint_for_u_0minus", side_effect=AssertionError), \
                 patch.object(pv, "solve_bvp", side_effect=AssertionError):
                result = function(10, 3*N0, 180, **options)
            self.assertTrue(result["success"], result["message"])
            failure = function(10, 3*N0, 180, xi=-10, **options)
            self.assertFalse(failure["success"])
            self.assertEqual(failure["status"], "invalid_speed_denominator")
            self.assertEqual(failure["rate_approximation"], "madsen_muB_over_3")
            for key in ("eos_approximation", "diffusion_approximation", "integral_approximation"):
                self.assertIn(key, failure)
        stable = self.api(NAMES[0])(20, N0, 180)
        self.assertEqual(stable["status"], "stable_neutron_matter")
        self.assertEqual(stable["u_0minus"], 0)
        unavailable = self.api(NAMES[1])(10, N0, 180, T_0plus=0)
        self.assertFalse(unavailable["success"])
        too_hot = self.api(NAMES[2])(10, 3*N0, 180, T_0plus=100)
        self.assertFalse(too_hot["success"])

    def test_quadratic_warm_interface_survives_ceiling_above_one(self):
        function = self.api(NAMES[1])
        upstream = pv._isothermal_upstream_nuclear_state(10, 10*N0, pv.para.paraQMCRMF3, "PNM")
        H = upstream["h_0minus"]/(10*N0)
        _, T_inf, _ = independent_quadratic_endpoint(upstream["P_0minus"], H, 180)
        warm = function(10, 10*N0, 180, T_0plus=0.95*T_inf)
        self.assertTrue(warm["success"], warm["message"])
        self.assertGreater(warm["A_adb"], 1)
        self.assertLess(warm["a_0plus"], 1)
        cold = function(10, 10*N0, 180, T_0plus=0)
        self.assertFalse(cold["success"])

    def test_quadratic_hot_limit_avoids_subtractive_cancellation(self):
        function = self.api(NAMES[1])
        normal = self.result(NAMES[1])
        T_inf = normal["T_inf"]
        almost_hot = function(10, 3*N0, 180, T_0plus=T_inf*(1-1e-12))
        self.assertTrue(almost_hot["success"], almost_hot["message"])
        muB, a = almost_hot["muB_inf"], almost_hot["a_0plus"]
        leading_small_a = (leading_diffusion(muB, T_inf) * weak_gamma(muB)
                           * 81*np.pi**2*T_inf**2/muB**2 * a*a/2)
        np.testing.assert_allclose(almost_hot["I2"], leading_small_a, rtol=1e-10)
        tiny_positive = function(10, 3*N0, 180, T_0plus=T_inf*1e-100)
        self.assertTrue(tiny_positive["success"], tiny_positive["message"])
        np.testing.assert_allclose(tiny_positive["I2"], self.result(NAMES[1], temp=0)["I2"], rtol=1e-14)

    def test_exact_cold_upstream_and_rounded_adiabatic_scaling(self):
        for name in NAMES[1:]:
            result = self.api(name)(0, 3*N0, 180, T_0plus=0, xi=-0.5)
            self.assertTrue(result["success"], result["message"])
            self.assertEqual(result["T_0minus"], 0)
            self.assertEqual(result["T_0plus"], 0)
            self.assertLess(max(result["conservation_errors"].values()), 1e-8)
        result = self.result(NAMES[1], temp=0)
        A = result["A_adb"]
        rounded_speed = (82*result["nB_inf"]/result["nB_0minus"]
                         * (result["muB_inf"]/1200)**2 * A**(7/6)
                         / (0.4**(5/6)*np.sqrt((1-A)*(1-0.5*A))))
        np.testing.assert_allclose(result["velocity_m_s"], rounded_speed, rtol=0.003)

    def test_isothermal_zero_composition_and_saturated_ceiling(self):
        function = self.api(NAMES[0])
        zero = function(10, 3*N0, 180, a_0plus=0)
        self.assertTrue(zero["success"], zero["message"])
        self.assertFalse(zero["front_exists"])
        self.assertEqual(zero["jB"], 0)
        saturated = function(0.01, 5.45*N0, 189.1566)
        self.assertFalse(saturated["success"])
        self.assertEqual(saturated["status"], "composition_ceiling_saturated")
        self.assertEqual(saturated["rate_approximation"], "madsen_muB_over_3")


if __name__ == "__main__":
    unittest.main()
