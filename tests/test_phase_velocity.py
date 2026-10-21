import ast
import sys
import warnings
from pathlib import Path
import inspect

import numpy as np
from scipy.optimize import root_scalar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver import SQMsolver as sqm
from RMFsolver import phase_velocity as pv


PARAM = para.paraQMCRMF3
N0 = 0.16 * const.MeV_fm**3


def _assert_close(actual, expected, rtol=1e-9, atol=0.0):
    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


def _weak_rate_expected(mu_q):
    gamma = (
        (128.0 / 27.0)
        * (1.0 / (5.0 * np.pi**3))
        * const.G_Fermi**2
        * np.cos(const.Cabibbo_angle_rad) ** 2
        * np.sin(const.Cabibbo_angle_rad) ** 2
        * mu_q**5
    )
    return gamma, 1.0 / gamma


def test_muB_from_nB_physical_auto_expands_downward():
    original_validated_state = pv._validated_pnm_state

    def fake_validated_state(muB, _temperature, _param):
        density = float(muB)
        return 1.0, 1.0, density, 25.0

    pv._validated_pnm_state = fake_validated_state
    try:
        recovered = pv.muB_from_nB_physical(
            500.0,
            100.0,
            param=PARAM,
            NM_type="PNM",
            muB_lo=850.0,
            muB_hi=1200.0,
            scan_points=8,
            auto_expand=True,
        )
    finally:
        pv._validated_pnm_state = original_validated_state

    _assert_close(recovered, 500.0, rtol=0.0, atol=1.0e-8)


def test_muB_from_nB_physical_recovers_high_temperature_n0_states():
    for temperature in (100.005, 133.337, 200.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            recovered = pv.muB_from_nB_physical(
                N0,
                temperature,
                param=PARAM,
                NM_type="PNM",
                muB_lo=850.0,
                muB_hi=1450.0,
                scan_points=16,
                auto_expand=True,
            )
            _, _, density, sigma = pv._validated_pnm_state(
                recovered,
                temperature,
                PARAM,
            )

        if not recovered < 850.0:
            raise AssertionError(
                f"expected the T={temperature:g} MeV n0 root below 850 MeV, "
                f"got muB={recovered:.6f} MeV"
            )
        _assert_close(density, N0, rtol=1.0e-6)
        if not sigma > 0.0:
            raise AssertionError(
                f"expanded inversion returned an unphysical sigma={sigma:.6e}"
            )


def test_neutral_quark_mu_triplet_equilibrated_massless_case():
    muB = 1100.0
    muK = 0.0
    temp = 10.0

    mu_u, mu_d, mu_s = sqm._quark_mu_triplet_neutral(muB, muK, temp, ms=0.0)

    _assert_close(mu_u, muB / 3.0, rtol=1.0e-12)
    _assert_close(mu_d, muB / 3.0, rtol=1.0e-12)
    _assert_close(mu_s, muB / 3.0, rtol=1.0e-12)


def test_neutral_quark_mu_triplet_starts_from_positive_muQ_guess():
    default = inspect.signature(sqm._quark_mu_triplet_neutral).parameters["muQ_init"].default

    _assert_close(float(default), 100.0, rtol=0.0, atol=0.0)


def test_quark_only_eos_enforces_color_included_charge_neutrality():
    muB = 1100.0
    muK = 40.0
    bag = 180.0
    temp = 10.0

    _, species = sqm.nB_QM(muB, muK, bag, temp, ms=0.0, return_species=True)
    residual = 2.0 * species["n_u"] - species["n_d"] - species["n_s"]
    scale = max(abs(species["n_u"]), abs(species["n_d"]), abs(species["n_s"]), 1.0)

    if abs(residual / scale) > 1.0e-12:
        raise AssertionError(f"quark charge neutrality residual too large: {residual:.6e}")


def test_massless_neutral_pqm_matches_color_included_analytic_pressure():
    muB = 1100.0
    muK = 40.0
    bag = 180.0
    temp = 10.0

    mu_u, mu_d, mu_s = sqm._quark_mu_triplet_neutral(muB, muK, temp, ms=0.0)

    def pressure_i(mu):
        return mu**4 / (4.0 * np.pi**2) + mu * mu * temp * temp / 2.0 + 7.0 * np.pi**2 * temp**4 / 60.0

    expected = (
        -bag**4
        + 8.0 * np.pi**2 * temp**4 / 45.0
        + pressure_i(mu_u)
        + pressure_i(mu_d)
        + pressure_i(mu_s)
    )

    _assert_close(sqm.PQM(muB, muK, bag, temp, ms=0.0), expected, rtol=1.0e-12)


def _synthetic_nuclear_state_for_quark_endpoint(muB_bar, T_inf, bag, nB_factor=0.5, ms=0.0):
    P_inf = pv.PQM(muB_bar, 0.0, bag, T_inf, ms=ms)
    e_inf = pv.edensQM(muB_bar, 0.0, bag, T_inf, ms=ms)
    nB_inf = pv.nB_QM(muB_bar, 0.0, bag, T_inf, ms=ms)
    h_inf = P_inf + e_inf
    nB_0minus = float(nB_factor * nB_inf)
    h_0minus = float(nB_0minus * h_inf / nB_inf)
    return {
        "P_0minus": float(P_inf),
        "h_0minus": h_0minus,
        "e_0minus": float(h_0minus - P_inf),
        "nB_0minus": nB_0minus,
        "nB_inf": float(nB_inf),
        "h_inf": float(h_inf),
    }


def _run_analytic_velocity_with_synthetic_nuclear_state(muB_bar, T_inf, bag, nB_factor=0.5, **kwargs):
    method = kwargs.pop("method", "full")
    T_0plus = kwargs.pop("T_0plus", 1.0e-3)
    state = _synthetic_nuclear_state_for_quark_endpoint(
        muB_bar,
        T_inf,
        bag,
        nB_factor=nB_factor,
        ms=float(kwargs.get("ms", 0.0)),
    )
    calls = {"nB_NM": 0}
    original_PNM = pv.PNM
    original_edensNM = pv.edensNM
    original_nB_NM = pv.nB_NM

    def fake_PNM(mu_B, Temp, param=PARAM, NM_type="PNM"):
        return state["P_0minus"]

    def fake_edensNM(mu_B, Temp, param=PARAM):
        return state["e_0minus"]

    def fake_nB_NM(mu_B, Temp, param=PARAM, NM_type="PNM"):
        calls["nB_NM"] += 1
        return state["nB_0minus"]

    pv.PNM = fake_PNM
    pv.edensNM = fake_edensNM
    pv.nB_NM = fake_nB_NM
    try:
        solver = (
            pv.analytic_velocity_bound_lte
            if method == "LTE"
            else pv.analytic_velocity_bound
        )
        call_kwargs = dict(kwargs)
        if method != "LTE":
            call_kwargs["T_0plus"] = T_0plus
        result = solver(
            1000.0,
            5.0,
            bag,
            initial_guess=(muB_bar, T_inf),
            **call_kwargs,
        )
    finally:
        pv.PNM = original_PNM
        pv.edensNM = original_edensNM
        pv.nB_NM = original_nB_NM

    return result, state, calls


def test_massless_quark_eos_regression():
    muB = 1100.0
    muK = 40.0
    bag = 180.0
    temp = 10.0

    _assert_close(pv.PQM(muB, muK, bag, temp, ms=0.0), 360395950.73063135, rtol=1e-12)
    _assert_close(pv.nB_QM(muB, muK, bag, temp, ms=0.0), 5061146.517399819, rtol=1e-12)
    _assert_close(pv.nK_QM(muB, muK, bag, temp, ms=0.0), 818521.5873314054, rtol=1e-12)
    _assert_close(pv.edensQM(muB, muK, bag, temp, ms=0.0), 5280227852.191894, rtol=1e-12)
    _assert_close(pv.entropyQM(muB, muK, bag, temp, ms=0.0), 4062177.0289468467, rtol=1e-12)


def test_sqmsolver_massless_quark_eos_regression():
    muB = 1100.0
    muK = 40.0
    bag = 180.0
    temp = 10.0

    _assert_close(sqm.PQM(muB, muK, bag, temp, ms=0.0), 360395950.73063135, rtol=1e-12)
    _assert_close(sqm.nB_QM(muB, muK, bag, temp, ms=0.0), 5061146.517399819, rtol=1e-12)
    _assert_close(sqm.nK_QM(muB, muK, bag, temp, ms=0.0), 818521.5873314054, rtol=1e-12)
    _assert_close(sqm.edensQM(muB, muK, bag, temp, ms=0.0), 5280227852.191894, rtol=1e-12)
    _assert_close(sqm.entropyQM(muB, muK, bag, temp, ms=0.0), 4062177.0289468467, rtol=1e-12)


def test_legacy_pqm_bag_helper_is_removed():
    if "PQM_bag" in getattr(sqm, "__all__", ()):
        raise AssertionError("SQMsolver should not export legacy PQM_bag")
    if hasattr(sqm, "PQM_bag"):
        raise AssertionError("SQMsolver should not expose legacy PQM_bag")

    from RMFsolver_Pub.RMFsolver import SQMsolver as pub_sqm

    if "PQM_bag" in getattr(pub_sqm, "__all__", ()):
        raise AssertionError("published SQMsolver should not export legacy PQM_bag")
    if hasattr(pub_sqm, "PQM_bag"):
        raise AssertionError("published SQMsolver should not expose legacy PQM_bag")


def test_phase_velocity_reexports_match_sqmsolver():
    muB = 1100.0
    muK = 20.0
    bag = 180.0
    temp = 10.0
    ms = 120.0

    if pv.PQM is not sqm.PQM:
        raise AssertionError("phase_velocity.PQM should be re-exported from SQMsolver")
    if pv.nB_QM is not sqm.nB_QM:
        raise AssertionError("phase_velocity.nB_QM should be re-exported from SQMsolver")
    if pv.nK_QM is not sqm.nK_QM:
        raise AssertionError("phase_velocity.nK_QM should be re-exported from SQMsolver")
    if pv.edensQM is not sqm.edensQM:
        raise AssertionError("phase_velocity.edensQM should be re-exported from SQMsolver")
    if pv.entropyQM is not sqm.entropyQM:
        raise AssertionError("phase_velocity.entropyQM should be re-exported from SQMsolver")

    _assert_close(pv.PQM(muB, muK, bag, temp, ms=ms), sqm.PQM(muB, muK, bag, temp, ms=ms), rtol=1e-12)
    _assert_close(pv.nB_QM(muB, muK, bag, temp, ms=ms), sqm.nB_QM(muB, muK, bag, temp, ms=ms), rtol=1e-12)
    _assert_close(pv.nK_QM(muB, muK, bag, temp, ms=ms), sqm.nK_QM(muB, muK, bag, temp, ms=ms), rtol=1e-12)
    _assert_close(pv.edensQM(muB, muK, bag, temp, ms=ms), sqm.edensQM(muB, muK, bag, temp, ms=ms), rtol=1e-12)
    _assert_close(pv.entropyQM(muB, muK, bag, temp, ms=ms), sqm.entropyQM(muB, muK, bag, temp, ms=ms), rtol=1e-12)


def test_quark_entropy_default_uses_integral_method():
    entropy_default = inspect.signature(sqm.entropyQM).parameters["use_thermal"].default
    if entropy_default is not False:
        raise AssertionError(f"SQMsolver.entropyQM should default to integral entropy, got use_thermal={entropy_default!r}")

    pv_entropy_default = inspect.signature(pv.entropyQM).parameters["use_thermal"].default
    if pv_entropy_default is not False:
        raise AssertionError(
            f"phase_velocity.entropyQM should default to integral entropy, got use_thermal={pv_entropy_default!r}"
        )

    muB = 1100.0
    muK = 20.0
    bag = 180.0
    temp = 10.0
    ms = 120.0
    _assert_close(
        sqm.entropyQM(muB, muK, bag, temp, ms=ms),
        sqm.entropyQM(muB, muK, bag, temp, ms=ms, use_thermal=False),
        rtol=1.0e-14,
    )


def test_pnm_entropy_default_satisfies_thermodynamic_identity():
    temp = 10.0
    nB_0minus = 3.0 * N0

    p_n = pv.PNM_n(nB_0minus, temp, param=PARAM, NM_type="PNM")
    e_n = pv.edensNM_n(nB_0minus, temp, param=PARAM)
    s_n = pv.sNM_n(nB_0minus, temp, param=PARAM, NM_type="PNM")
    muB_n = pv._solve_scalar_root(lambda muB: pv.nB_NM(muB, temp, param=PARAM, NM_type="PNM") - nB_0minus, 1050.0)

    residual = float(-p_n - e_n + temp * s_n + muB_n * nB_0minus)
    scale = max(abs(float(e_n)), 1.0)
    if abs(residual / scale) > 1.0e-10:
        raise AssertionError(f"PNM entropy default violates Euler identity: residual={residual:.6e}")


def test_nonzero_ms_consistency_helpers():
    muB = 1100.0
    muK = 20.0
    bag = 180.0
    temp = 10.0
    jB = 0.02 * const.MeV_fm**3

    h_ms0 = pv.hQM(muB, muK, bag, temp, ms=0.0)
    h_ms120 = pv.hQM(muB, muK, bag, temp, ms=120.0)
    if np.isclose(h_ms0, h_ms120, rtol=1e-8, atol=1e-8):
        raise AssertionError("hQM should change when the strange-quark mass changes")

    pi_direct = pv.Pi_QM(muB, muK, bag, temp, jB, ms=120.0)
    pi_state = pv._Pi_QM_state(muB, muK, bag, temp, jB, ms=120.0)
    _assert_close(pi_direct, pi_state, rtol=1e-12)

    pi_equil = pv.Pi_QM(muB, 0.0, bag, temp, jB, ms=120.0)
    mu_wrapper = pv._solve_muB_inf_at_muK0_for_given_Pi(pi_equil, jB, bag, temp, ms=120.0)
    mu_ms = pv._solve_muB_inf_at_muK0_for_given_Pi_ms(pi_equil, jB, bag, temp, ms=120.0)
    _assert_close(mu_wrapper, mu_ms, rtol=1e-12)

    p_ms0 = pv.PQM(muB, muK, bag, temp, ms=0.0)
    p_ms120 = pv.PQM(muB, muK, bag, temp, ms=120.0)
    if np.isclose(p_ms0, p_ms120, rtol=1e-8, atol=1e-8):
        raise AssertionError("PQM should change when the strange-quark mass changes")


def test_quark_derivative_identities_for_nonzero_ms():
    muB = 1100.0
    muK = 20.0
    bag = 180.0
    temp = 10.0
    ms = 120.0
    step = 1.0e-3

    fd_nB = (pv.PQM(muB + step, muK, bag, temp, ms=ms) - pv.PQM(muB - step, muK, bag, temp, ms=ms)) / (2.0 * step)
    fd_nK = (pv.PQM(muB, muK + step, bag, temp, ms=ms) - pv.PQM(muB, muK - step, bag, temp, ms=ms)) / (2.0 * step)
    fd_chi = (pv.nK_QM(muB, muK + step, bag, temp, ms=ms) - pv.nK_QM(muB, muK - step, bag, temp, ms=ms)) / (2.0 * step)

    _assert_close(pv.nB_QM(muB, muK, bag, temp, ms=ms), fd_nB, rtol=1e-7)
    _assert_close(pv.nK_QM(muB, muK, bag, temp, ms=ms), fd_nK, rtol=1e-7)
    _assert_close(pv._chiK_QM(muB, muK, bag, temp, ms=ms), fd_chi, rtol=1e-7)


def test_phase_velocity_public_api_surface():
    expected_public = [
        "analytic_velocity_bound",
        "analytic_velocity_isothermal",
        "solve_front_isothermal",
        "solve_front_energy_conserving_nK",
        "solve_front_energy_conserving_uNmax",
        "solve_front_thermal_conducting",
        "z_time_evolution",
        "solve_front_isothermal_slow",
        "solve_front_adiabatic_slow",
    ]
    if pv.__all__ != expected_public:
        raise AssertionError(f"expected __all__={expected_public!r}, got {pv.__all__!r}")

    for name in expected_public:
        if not hasattr(pv, name):
            raise AssertionError(f"phase_velocity is missing public solver {name!r}")

    removed_names = (
        "solve_steady_front_2d",
        "solve_steady_front_2d_simple",
        "solve_steady_front_1d_aQstar",
        "solve_steady_front_1d_aQstar_rescaled",
        "solve_steady_front_1d_aQstar_rescale_bvp",
        "solve_steady_front_entropy",
        "solve_steady_front_entropy_TQguess",
        "solve_steady_front_entropy_TQguess_uagrad",
        "extract_jB_curve_vs_aQstar",
        "vNtoQ_nc",
        "Get_Two_as_dens",
        "z_time_evolution2",
        "solve_front_adiabatic",
        "solve_front_energy_conserving",
    )
    for name in removed_names:
        if hasattr(pv, name):
            raise AssertionError(f"phase_velocity should not expose removed name {name!r}")


def test_z_time_evolution_uses_uniform_density_slope_temperature_and_warm_starts():
    original_solver = pv.solve_front_energy_conserving_uNmax
    calls = []
    u_0minus = 1.0e-4

    def fake_solver(T, nB_0minus, B_one_forth, **kwargs):
        index = len(calls)
        calls.append(
            {
                "T": float(T),
                "nB_0minus": float(nB_0minus),
                "B_one_forth": float(B_one_forth),
                "jB_guess": kwargs.get("jB_guess"),
                "T_inf_guess": kwargs.get("T_inf_guess"),
            }
        )
        return {
            "success": True,
            "message": "synthetic success",
            "u_0minus": u_0minus,
            "u_0minus": u_0minus,
            "jB": 0.5 + index,
            "T_inf": 20.0 + index,
            "T_0plus": float(kwargs["T_0plus"]),
        }

    pv.solve_front_energy_conserving_uNmax = fake_solver
    try:
        result = pv.z_time_evolution(
            nB_target=3.0 * N0,
            temperature=lambda z_km: 5.0 + z_km,
            B_one_forth=180.0,
            z0=1.0,
            density_slope_n0_per_km=0.3,
            z_stop=0.1,
            t_max=1.0,
            sample_count=4,
            output_count=20,
            jB_guess=0.25,
            T_inf_guess=12.0,
            return_solver_results=True,
        )
    finally:
        pv.solve_front_energy_conserving_uNmax = original_solver

    z_samples = np.asarray(result["z_samples_km"], dtype=float)
    expected_density = 3.0 * N0 + 0.3 * N0 * z_samples
    np.testing.assert_allclose(result["nB_N_samples_MeV3"], expected_density, rtol=1.0e-14)
    np.testing.assert_allclose(result["T_N_samples_MeV"], 5.0 + z_samples, rtol=0.0, atol=0.0)

    expected_velocity = const.c_km * u_0minus / np.sqrt(1.0 + u_0minus**2)
    np.testing.assert_allclose(result["velocity_samples_km_s"], expected_velocity, rtol=1.0e-14)
    expected_time = (1.0 - 0.1) / expected_velocity
    _assert_close(result["catchup_time_s"], expected_time, rtol=2.0e-6)
    _assert_close(result["integral_catchup_time_s"], expected_time, rtol=1.0e-12)
    _assert_close(result["z_km"][0], 1.0, rtol=0.0, atol=1.0e-14)
    _assert_close(result["z_km"][-1], 0.1, rtol=0.0, atol=1.0e-14)
    if not bool(result["caught_up"]):
        raise AssertionError("constant positive synthetic velocity should reach z_stop")

    _assert_close(calls[0]["nB_0minus"], 3.3 * N0, rtol=1.0e-14)
    _assert_close(calls[0]["T"], 6.0, rtol=0.0, atol=0.0)
    _assert_close(calls[0]["jB_guess"], 0.25, rtol=0.0, atol=0.0)
    _assert_close(calls[0]["T_inf_guess"], 12.0, rtol=0.0, atol=0.0)
    _assert_close(calls[1]["jB_guess"], 0.5, rtol=0.0, atol=0.0)
    _assert_close(calls[1]["T_inf_guess"], 20.0, rtol=0.0, atol=0.0)
    if len(result["solver_results"]) != 4:
        raise AssertionError("expected one retained solver result per velocity sample")


def test_z_time_evolution_propagates_complete_uNmax_continuation_state():
    original_solver = pv.solve_front_energy_conserving_uNmax
    returned_results = []
    received_guesses = []

    def fake_solver(T, nB_0minus, B_one_forth, **kwargs):
        received_guesses.append(kwargs.get("continuation_guess"))
        index = len(returned_results)
        result = {
            "success": True,
            "message": "synthetic profile success",
            "u_0minus": 1.0e-4,
            "u_0minus": 1.0e-4,
            "jB": 0.5 + index,
            "T_inf": 20.0 + index,
            "T_0plus": float(kwargs["T_0plus"]),
            "muB_inf": 1100.0 + index,
            "muB_0plus": 1090.0 + index,
            "muK_0plus": 10.0 + index,
            "s_coord": np.array([0.0, 0.5, 0.9]),
            "nK": np.array([3.0, 2.0, 1.0]) + index,
            "jK": np.array([0.5, 0.3, 0.1]) + index,
            "nK_inf": 1.0 + index,
        }
        returned_results.append(result)
        return result

    pv.solve_front_energy_conserving_uNmax = fake_solver
    try:
        pv.z_time_evolution(
            nB_target=3.0 * N0,
            temperature=8.0,
            B_one_forth=180.0,
            z0=1.0,
            z_stop=0.1,
            sample_count=3,
            output_count=10,
        )
    finally:
        pv.solve_front_energy_conserving_uNmax = original_solver

    if received_guesses[0] is not None:
        raise AssertionError("first hydrodynamic point must not have a continuation guess")
    if received_guesses[1] is not returned_results[0]:
        raise AssertionError("second point did not receive the complete previous solver result")


def test_uNmax_exact_zero_endpoint_does_not_accept_unconverged_collocation_status():
    if pv._uNmax_collocation_status_is_acceptable(
        solver_success=False,
        solver_status=1,
        exact_zero_left=True,
    ):
        raise AssertionError("max-node collocation status must remain a failure")
    if pv._uNmax_collocation_status_is_acceptable(
        solver_success=False,
        solver_status=1,
        exact_zero_left=False,
    ):
        raise AssertionError("finite-temperature max-node solutions must remain failures")
    if pv._uNmax_collocation_status_is_acceptable(
        solver_success=False,
        solver_status=2,
        exact_zero_left=True,
    ):
        raise AssertionError("singular-Jacobian solutions must remain failures")


def test_z_time_evolution_adaptively_inserts_log_midpoints_after_failed_step():
    original_solver = pv.solve_front_energy_conserving_uNmax
    last_success_z = {"value": None}
    attempted_z = []

    def fake_solver(T, nB_0minus, B_one_forth, **kwargs):
        z_value = float((nB_0minus - 3.0 * N0) / (0.3 * N0))
        attempted_z.append(z_value)
        previous_z = last_success_z["value"]
        if previous_z is not None and previous_z / z_value > 2.0 + 1.0e-12:
            return {"success": False, "message": "synthetic step too large"}
        last_success_z["value"] = z_value
        return {
            "success": True,
            "message": "synthetic adaptive success",
            "u_0minus": 1.0e-4,
            "u_0minus": 1.0e-4,
            "jB": 0.5,
            "T_inf": 20.0,
            "T_0plus": float(kwargs["T_0plus"]),
            "muB_inf": 1100.0,
            "muB_0plus": 1090.0,
            "muK_0plus": 10.0,
            "s_coord": np.array([0.0, 0.5, 0.9]),
            "nK": np.array([3.0, 2.0, 1.0]),
            "jK": np.array([0.5, 0.3, 0.1]),
            "nK_inf": 1.0,
        }

    pv.solve_front_energy_conserving_uNmax = fake_solver
    try:
        result = pv.z_time_evolution(
            nB_target=3.0 * N0,
            temperature=8.0,
            B_one_forth=180.0,
            z0=1.0,
            z_stop=0.1,
            sample_count=2,
            output_count=10,
            adaptive_continuation=True,
            max_continuation_subdivisions=12,
        )
    finally:
        pv.solve_front_energy_conserving_uNmax = original_solver

    z_samples = np.asarray(result["z_samples_km"], dtype=float)
    if z_samples.size <= 2:
        raise AssertionError("adaptive continuation did not retain inserted midpoint samples")
    if not bool(result["adaptive_continuation_used"]):
        raise AssertionError("adaptive continuation usage was not reported")
    if int(result["continuation_inserted_point_count"]) != z_samples.size - 2:
        raise AssertionError("inserted-point diagnostic does not match the returned sample grid")
    if np.any(np.diff(z_samples) <= 0.0):
        raise AssertionError("adaptive z samples must be strictly increasing")
    if not any(not np.isclose(value, 1.0) and not np.isclose(value, 0.1) for value in attempted_z):
        raise AssertionError("no logarithmic midpoint was attempted")


def test_z_time_evolution_reports_exact_zero_terminal_branch_as_stalled_front():
    original_solver = pv.solve_front_energy_conserving_uNmax

    def fake_solver(T, nB_0minus, B_one_forth, **kwargs):
        z_value = float((nB_0minus - 3.0 * N0) / (0.3 * N0))
        if z_value < 0.3:
            return {"success": False, "message": "synthetic terminal branch"}
        return {
            "success": True,
            "message": "synthetic pre-terminal success",
            "u_0minus": 1.0e-4,
            "u_0minus": 1.0e-4,
            "jB": 0.5,
            "T_inf": 20.0,
            "T_0plus": 0.0,
        }

    pv.solve_front_energy_conserving_uNmax = fake_solver
    try:
        result = pv.z_time_evolution(
            nB_target=3.0 * N0,
            temperature=8.0,
            B_one_forth=180.0,
            z0=1.0,
            z_stop=0.1,
            sample_count=2,
            output_count=10,
            T_0plus=0.0,
            adaptive_continuation=True,
            max_continuation_subdivisions=4,
            stall_on_terminal_branch=True,
            return_solver_results=True,
        )
    finally:
        pv.solve_front_energy_conserving_uNmax = original_solver

    if not bool(result["stalled"]):
        raise AssertionError("terminal exact-zero branch should be reported as stalled")
    if bool(result["success"]) or not bool(result["continuation_exhausted"]):
        raise AssertionError("numerical stall closure must not be reported as solver success")
    if not result["velocity_samples_km_s"][0] == 0.0:
        raise AssertionError("terminal branch sample must impose zero front velocity")
    if bool(result["caught_up"]) or not np.isnan(result["catchup_time_s"]):
        raise AssertionError("stalled front must not report finite catch-up")
    if not np.isinf(result["integral_catchup_time_s"]):
        raise AssertionError("stalled front must report infinite integral travel time")
    if len(result["solver_results"]) != len(result["z_samples_km"]):
        raise AssertionError("stalled samples and retained solver diagnostics must align")


def test_z_time_evolution_can_stall_at_initial_exact_zero_sample():
    original_solver = pv.solve_front_energy_conserving_uNmax

    def fake_solver(T, nB_0minus, B_one_forth, **kwargs):
        return {"success": False, "message": "synthetic absent branch"}

    pv.solve_front_energy_conserving_uNmax = fake_solver
    try:
        result = pv.z_time_evolution(
            nB_target=3.0 * N0,
            temperature=8.0,
            B_one_forth=180.0,
            z0=1.0,
            z_stop=0.1,
            sample_count=2,
            output_count=10,
            T_0plus=0.0,
            stall_on_terminal_branch=True,
        )
    finally:
        pv.solve_front_energy_conserving_uNmax = original_solver

    if not bool(result["stalled"]):
        raise AssertionError("absent exact-zero branch should stall at the initial position")
    if bool(result["success"]) or not bool(result["continuation_exhausted"]):
        raise AssertionError("absent branch closure must remain an unsuccessful solve")
    _assert_close(result["stall_z_km"], 1.0, rtol=0.0, atol=1.0e-14)
    if not np.all(np.asarray(result["velocity_samples_km_s"]) == 0.0):
        raise AssertionError("initially stalled branch must have a zero velocity table")


def test_z_time_evolution_reports_time_limit_for_slow_front():
    original_solver = pv.solve_front_energy_conserving_uNmax

    def fake_solver(T, nB_0minus, B_one_forth, **kwargs):
        return {
            "success": True,
            "message": "synthetic slow success",
            "u_0minus": 1.0e-12,
            "u_0minus": 1.0e-12,
            "jB": 0.5,
            "T_inf": 20.0,
            "T_0plus": float(kwargs["T_0plus"]),
        }

    pv.solve_front_energy_conserving_uNmax = fake_solver
    try:
        result = pv.z_time_evolution(
            nB_target=3.0 * N0,
            temperature=8.0,
            B_one_forth=180.0,
            z0=1.0,
            z_stop=0.1,
            t_max=0.01,
            sample_count=2,
            output_count=5,
        )
    finally:
        pv.solve_front_energy_conserving_uNmax = original_solver

    if bool(result["caught_up"]):
        raise AssertionError("synthetic slow front should not reach z_stop before t_max")
    if not np.isnan(result["catchup_time_s"]):
        raise AssertionError("uncaught evolution should report catchup_time_s=nan")
    _assert_close(result["t_s"][-1], 0.01, rtol=0.0, atol=1.0e-14)
    if not result["z_km"][-1] < result["z_km"][0]:
        raise AssertionError("positive velocity should decrease z even when t_max is reached")


def test_z_time_evolution_propagates_uNmax_sample_failure():
    original_solver = pv.solve_front_energy_conserving_uNmax
    call_count = {"value": 0}

    def fake_solver(T, nB_0minus, B_one_forth, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 2:
            return {"success": False, "message": "synthetic BVP failure"}
        return {
            "success": True,
            "message": "synthetic success",
            "u_0minus": 1.0e-4,
            "jB": 0.5,
            "T_inf": 20.0,
            "T_0plus": float(kwargs["T_0plus"]),
        }

    pv.solve_front_energy_conserving_uNmax = fake_solver
    try:
        try:
            pv.z_time_evolution(
                nB_target=3.0 * N0,
                temperature=8.0,
                B_one_forth=180.0,
                sample_count=3,
                adaptive_continuation=False,
            )
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected failed hydrodynamic sample to raise RuntimeError")
    finally:
        pv.solve_front_energy_conserving_uNmax = original_solver

    if "synthetic BVP failure" not in message or "z=" not in message:
        raise AssertionError(f"missing sample failure diagnostics: {message!r}")


def test_analytic_velocity_recovers_synthetic_equilibrium_and_uses_eos_density():
    muB_bar = 1100.0
    T_inf = 10.0
    bag = 180.0
    result, state, calls = _run_analytic_velocity_with_synthetic_nuclear_state(
        muB_bar,
        T_inf,
        bag,
        T_0plus=1.0e-3,
    )

    _assert_close(result["muB_bar"], muB_bar, rtol=1.0e-9)
    _assert_close(result["T_inf"], T_inf, rtol=1.0e-9)
    _assert_close(result["nB_0minus"], state["nB_0minus"], rtol=1.0e-12)
    _assert_close(result["nB_inf"], state["nB_inf"], rtol=1.0e-12)
    expected_lambda_n = state["nB_0minus"] / state["nB_inf"]
    _assert_close(result["lambda_n"], expected_lambda_n, rtol=1.0e-12)
    if "a_N" in result or "a_N_squared" in result:
        raise AssertionError("analytical results must not expose the old density-ratio aliases")
    _assert_close(result["lambda_n_squared"], result["lambda_n"] ** 2, rtol=1.0e-12)
    _assert_close(result["T_0plus"], 1.0e-3, rtol=1.0e-12)
    _assert_close(result["a_0plus"], result["nK_0plus"] / result["nB_0plus"], rtol=1.0e-12)
    if np.isclose(result["lambda_n"], 1.0, rtol=1.0e-12):
        raise AssertionError("lambda_n must not be replaced by the pure-neutron local fraction lambda_n=1")
    if calls["nB_NM"] <= 0:
        raise AssertionError("analytic_velocity_bound should read nB_0minus through nB_NM")

    leading_density = muB_bar**3 / (27.0 * np.pi**2)
    leading_lambda_n = state["nB_0minus"] / leading_density
    if np.isclose(result["lambda_n"], leading_lambda_n, rtol=1.0e-8):
        raise AssertionError("analytic_velocity_bound should use nB_QM, not the leading muB^3 density")

    numerator = (
        54.0
        * np.pi ** (7.0 / 3.0)
        * result["gamma"]
        * result["A_boundary"] ** (7.0 / 3.0)
    )
    denominator = (
        7.0
        * np.sqrt(2.0)
        * result["h_D"]
        * result["alpha_s"] ** (5.0 / 3.0)
        * result["muB_bar"]
        * result["lambda_n_squared"]
        * result["one_minus_a_0plus"]
        * result["one_plus_xi_a_0plus"]
    )
    _assert_close(result["u_0minus_squared"], numerator / denominator, rtol=1.0e-12)
    _assert_close(result["analytic_denominator"], denominator / (7.0 * np.sqrt(2.0) * result["h_D"] * result["alpha_s"] ** (5.0 / 3.0)), rtol=1.0e-12)
    if result["interface_control"] != "fixed_T_0plus":
        raise AssertionError("analytic formula should report fixed-T interface control")
    _assert_close(result["a_0plus"], result["A_boundary"], rtol=1.0e-12)
    _assert_close(result["a_0plus_used"], result["a_0plus"], rtol=1.0e-12)
    _assert_close(result["a_0plus_max"], result["A_boundary"], rtol=1.0e-12)
    if result["density_ratio_definition"] != "lambda_n_equals_nB_0minus_over_nB_inf":
        raise AssertionError("analytic formula must report the lambda_n density-ratio definition")
    if result["composition_definition"] != "a_local_equals_nK_over_nB":
        raise AssertionError("analytic formula must report the local K-fraction definition")
    if result["analytic_formula_variant"] != "piecewise_constant_fixed_T_0plus":
        raise AssertionError("analytic formula must report the fixed-T variant")


def test_analytic_velocity_lte_formula_and_default_mode():
    signature = inspect.signature(pv.analytic_velocity_bound_lte)
    if "T_0plus" in signature.parameters:
        raise AssertionError("the preserved LTE API should not accept T_0plus")

    result, _, _ = _run_analytic_velocity_with_synthetic_nuclear_state(
        1100.0,
        10.0,
        180.0,
        method="LTE",
    )
    if result["a_0plus_max_mode"] != "LTE":
        raise AssertionError("LTE mode should be normalized to its canonical name")
    if not (0.0 < result["a_0plus_LTE"] <= result["A_boundary"] < 1.0):
        raise AssertionError("LTE abundance should satisfy 0 < a_0plus_LTE <= A < 1")
    _assert_close(result["a_0plus"], result["a_0plus_LTE"], rtol=1.0e-12)
    _assert_close(result["a_0plus_used"], result["a_0plus"], rtol=1.0e-12)

    prefactor = (
        9.0
        * np.pi ** (7.0 / 3.0)
        * result["gamma"]
        / (
            4.0
            * np.sqrt(2.0)
            * result["h_D"]
            * result["alpha_s"] ** (5.0 / 3.0)
        )
    )
    denominator = (
        result["muB_bar"]
        * result["lambda_n_squared"]
        * result["one_minus_a_0plus"]
        * result["one_plus_xi_a_0plus"]
    )
    expected = (
        prefactor
        * result["A_boundary"] ** (7.0 / 3.0)
        * result["lte_correction"]
        / denominator
    )
    _assert_close(result["u_0minus_squared"], expected, rtol=5.0e-12)
    _assert_close(result["u_0minus_squared"], result["u_0minus_formula_squared"], rtol=1.0e-8)


def test_analytic_lte_a_0plus_stable_expression_and_slow_front_limit():
    A = 0.4
    lambda_n = 0.5
    u_0minus = 0.01
    data = pv._analytic_a_0plus_lte(A, u_0minus, lambda_n)
    beta = 5.0 * u_0minus * lambda_n
    direct = (
        -beta + np.sqrt(beta**2 + 4.0 * (1.0 - beta) * A**2)
    ) / (2.0 * (1.0 - beta))
    _assert_close(data["a_0plus_LTE"], direct, rtol=1.0e-12)
    _assert_close(data["lambda_n"], lambda_n, rtol=1.0e-12)

    beta_one = pv._analytic_a_0plus_lte(A, 1.0 / (5.0 * lambda_n), lambda_n)
    _assert_close(beta_one["beta_LTE"], 1.0, rtol=1.0e-12)
    _assert_close(beta_one["a_0plus_LTE"], A**2, rtol=1.0e-12)

    tiny_u = 1.0e-9
    exact = pv._analytic_a_0plus_lte(A, tiny_u, lambda_n)["a_0plus_LTE"]
    slow_front = A - 2.5 * tiny_u * lambda_n * (1.0 - A)
    _assert_close(exact, slow_front, rtol=1.0e-10, atol=1.0e-14)


def test_analytic_lte_domain_is_checked_with_a_0plus_lte_not_A():
    # A is now read from the exact EOS as nK/nB at the T -> 0 end of the isobar.
    # Charge neutrality with n_s >= 0 caps it at A = 1 (reached when mu_s = 0,
    # the neutron flavor content), so A > 1 is unreachable. The LTE saturation
    # condition A^2 - a^2 = 5 lambda_n u a (1 - a) has a positive right side at
    # a = A, so its root always lies below A. The LTE branch is therefore
    # bounded by 0 < a_0plus_LTE < A <= 1, and it is a_0plus_LTE, not A,
    # that the LTE branch reports as the interface fraction.
    result, _, _ = _run_analytic_velocity_with_synthetic_nuclear_state(
        1100.0,
        10.0,
        180.0,
        method="LTE",
    )
    A_boundary = float(result["A_boundary"])
    a_lte = float(result["a_0plus_LTE"])
    if not 0.0 < a_lte < A_boundary <= 1.0:
        raise AssertionError(
            f"expected 0 < a_0plus_LTE < A_boundary <= 1, "
            f"got a_0plus_LTE={a_lte}, A_boundary={A_boundary}"
        )
    _assert_close(result["a_0plus"], a_lte, rtol=1.0e-12)


def test_analytic_A_matches_exact_eos_nK_over_nB():
    # A must be the exact composition nK/nB at the T -> 0 end of the isobar
    # through the downstream state, not a quadratic-expansion closed form.
    result, _, _ = _run_analytic_velocity_with_synthetic_nuclear_state(
        1100.0,
        10.0,
        180.0,
        T_0plus=1.0e-3,
    )
    muB_inf = float(result["muB_inf"])
    P_inf = float(result["P_inf"])
    A_boundary = float(result["A_boundary"])
    if not 0.0 < A_boundary <= 1.0:
        raise AssertionError(f"A_boundary must satisfy 0 < A <= 1, got {A_boundary}")

    # Reproduce A independently: solve the T -> 0 isobar for muK, then read the
    # composition straight off the EOS.
    T_cold = 1.0e-3
    muK = root_scalar(
        lambda mu: float(sqm.PQM(muB_inf, mu, 180.0, T_cold, ms=0.0)) - P_inf,
        bracket=(0.0, 3.0 * muB_inf),
        method="brentq",
        xtol=1.0e-8,
    ).root
    expected = float(sqm.nK_QM(muB_inf, muK, 180.0, T_cold, ms=0.0)) / float(
        sqm.nB_QM(muB_inf, muK, 180.0, T_cold, ms=0.0)
    )
    _assert_close(A_boundary, expected, rtol=1.0e-6)
    _assert_close(result["muK_0plus_max"], muK, rtol=1.0e-10, atol=1.0e-8)

    # The discarded closed form disagrees; it assumes a non-neutral EOS.
    U_bag = 180.0**4
    closed_form = 1.5 * np.sqrt(
        max(108.0 * np.pi**2 * (P_inf + U_bag) - muB_inf**4, 0.0)
    ) / muB_inf**2
    if abs(closed_form - A_boundary) <= 1.0e-6 * A_boundary:
        raise AssertionError(
            "closed-form A unexpectedly agrees with the exact EOS value; "
            "the regression this test guards may have been reverted"
        )


def test_analytic_velocity_rejects_removed_interface_mode_selector():
    try:
        _run_analytic_velocity_with_synthetic_nuclear_state(
            1100.0,
            10.0,
            180.0,
            interface_fraction_mode="not-a-mode",
        )
    except TypeError:
        pass
    else:
        raise AssertionError("removed interface_fraction_mode should raise TypeError")


def test_analytic_velocity_solves_hydro_jump_internally_without_jB_input():
    signature = inspect.signature(pv.analytic_velocity_bound)
    if "jB" in signature.parameters:
        raise AssertionError("analytic_velocity_bound should not expose jB as a public input")

    result = pv.analytic_velocity_bound(
        1100.0,
        10.0,
        180.0,
        T_0plus=0.5,
        param=PARAM,
        NM_type="PNM",
    )

    _assert_close(result["jB"], result["nB_0minus"] * result["u_0minus"], rtol=1.0e-10)
    _assert_close(result["jB"], result["nB_inf"] * result["u_inf"], rtol=1.0e-10)

    energy_scale = max(abs(result["energy_flux_0minus"]), abs(result["energy_flux_inf"]), 1.0)
    momentum_scale = max(abs(result["momentum_flux_0minus"]), abs(result["momentum_flux_inf"]), 1.0)
    if abs(result["energy_flux_residual"]) > 1.0e-8 * energy_scale:
        raise AssertionError("analytic endpoint should satisfy exact energy flux matching")
    if abs(result["momentum_flux_residual"]) > 1.0e-8 * momentum_scale:
        raise AssertionError("analytic endpoint should satisfy exact momentum flux matching")

    _assert_close(result["pressure_jump"], result["P_inf"] - result["P_0minus"], rtol=1.0e-12, atol=1.0e-8)
    _assert_close(
        result["pressure_jump_balance"],
        result["h_0minus"] * result["u_0minus"] ** 2 - result["h_inf"] * result["u_inf"] ** 2,
        rtol=1.0e-12,
        atol=1.0e-8,
    )
    _assert_close(result["pressure_jump"], result["pressure_jump_balance"], rtol=2.0e-2, atol=1.0e-8)
    _assert_close(result["u_0minus_squared"], result["u_0minus_formula_squared"], rtol=1.0e-8)


def test_analytic_methods_accept_zero_upstream_temperature():
    class NuclearStateReached(RuntimeError):
        pass

    original_nuclear_state = pv._analytic_nuclear_state

    def zero_temperature_probe(muB_0minus, T_0minus, param=PARAM, NM_type="PNM"):
        _assert_close(float(T_0minus), 0.0, rtol=0.0, atol=0.0)
        raise NuclearStateReached("zero-temperature nuclear state reached")

    pv._analytic_nuclear_state = zero_temperature_probe
    try:
        for solver in (pv.analytic_velocity_bound, pv.semi_analytic_velocity_bound):
            try:
                solver(
                    1232.36560617,
                    0.0,
                    180.0,
                    T_0plus=5.0,
                    xi=-0.5,
                    param=PARAM,
                    NM_type="PNM",
                )
            except NuclearStateReached:
                pass
            else:
                raise AssertionError(f"{solver.__name__} did not evaluate the T(0-)=0 nuclear state")
    finally:
        pv._analytic_nuclear_state = original_nuclear_state


def test_analytic_methods_reject_negative_upstream_temperature():
    for solver in (pv.analytic_velocity_bound, pv.semi_analytic_velocity_bound):
        try:
            solver(
                1232.36560617,
                -1.0,
                180.0,
                T_0plus=5.0,
                xi=-0.5,
                param=PARAM,
                NM_type="PNM",
            )
        except RuntimeError as exc:
            if "T_0minus" not in str(exc):
                raise AssertionError(f"unexpected negative-temperature error: {exc}") from exc
        else:
            raise AssertionError(f"{solver.__name__} accepted negative T_0minus")


def test_analytic_velocity_gamma_matches_weak_rate_and_seconds_shortcut():
    result, _, _ = _run_analytic_velocity_with_synthetic_nuclear_state(
        900.0,
        10.0,
        145.0,
    )
    mu_q = result["mu_q"]
    _assert_close(result["alpha_s"], 0.4, rtol=0.0)
    gamma_expected, tau_expected = _weak_rate_expected(mu_q)
    _assert_close(result["gamma"], gamma_expected, rtol=1.0e-12)
    _assert_close(result["tau"], tau_expected, rtol=1.0e-12)
    _assert_close(result["tau_seconds"], result["tau"] * const.MeV_sec, rtol=1.0e-12)

    shortcut_tau_seconds = 1.3e-9 * (300.0 / mu_q) ** 5
    if abs(result["tau_seconds"] - shortcut_tau_seconds) / shortcut_tau_seconds > 0.05:
        raise AssertionError("direct weak-rate tau should match the quoted seconds shortcut to within 5%")


def test_microphysics_helpers_use_precise_weak_rate():
    muB = 1100.0
    T = 10.0
    muQ = muB / 3.0
    gamma_expected, tau_expected = _weak_rate_expected(muQ)

    helpers = (
        pv._microphysics_from_quark_state_energy(muB, T),
        pv._microphysics_from_quark_state_isothermal_baseline(muB, T),
    )
    for micro in helpers:
        _assert_close(micro["gamma"], gamma_expected, rtol=1.0e-12)
        _assert_close(micro["tau"], tau_expected, rtol=1.0e-12)


def _kaon_rate_expected(muB, muK, temp):
    return (
        16.0
        / (5.0 * np.pi**5)
        * const.G_Fermi**2
        * np.sin(const.Cabibbo_angle_rad) ** 2
        * np.cos(const.Cabibbo_angle_rad) ** 2
        * (muB / 3.0)**5
        * muK
        * (muK**2 + 4.0 * np.pi**2 * temp**2)
    )


def test_kaon_transport_rate_uses_baryon_prefactor_and_preserves_eos_diagnostics():
    muB = 1100.0
    for ms in (0.0, 100.0):
        for temp in (0.0, 10.0):
            for muK in (-80.0, 0.0, 80.0):
                state = sqm._quark_uds_state(muB, muK, temp, ms=ms)
                rate = pv._exact_kaon_transport_rate(muB, muK, temp, ms=ms)
                expected = _kaon_rate_expected(muB, muK, temp)
                _assert_close(rate["Gamma_K"], expected, rtol=1.0e-12)
                _assert_close(np.sign(rate["Gamma_K"]), np.sign(muK), atol=0.0)
                for key in ("mu_u", "mu_d", "mu_s"):
                    _assert_close(rate[key], state["chemical_potentials"][key], rtol=1.0e-12)
                if muK == 80.0:
                    old_rate = expected * (rate["mu_u"] / (muB / 3.0))**5
                    if np.isclose(old_rate, expected, rtol=1.0e-4, atol=0.0):
                        raise AssertionError("unequal-flavor test must distinguish the old prefactor")


def test_microphysics_uses_shared_alpha_s_0p4_and_consistent_diffusion():
    muB = 1100.0
    expected_qD = np.sqrt(6.0 * 0.4 / np.pi) * (muB / 3.0)
    # HP93 Eq. (25), Gamma(8/3)*zeta(5/3)*(2*pi)**(2/3)/6, evaluated with mpmath.
    hp93_h_D = 1.8131745180482931
    _assert_close(pv._TRANSPORT_H_CONST, hp93_h_D, rtol=1.0e-15)
    for temp in (1.0e-3, 10.0, 100.0):
        energy = pv._microphysics_from_quark_state_energy(muB, temp)
        isothermal = pv._microphysics_from_quark_state_isothermal_baseline(muB, temp)
        expected_invD = 24.0 * 0.4**2 / np.pi * (
            hp93_h_D * temp**(5.0 / 3.0) / expected_qD**(2.0 / 3.0)
            + np.pi**3 * temp**2 / (12.0 * expected_qD)
        )
        for micro in (energy, isothermal):
            _assert_close(micro["alpha_s"], 0.4, rtol=0.0)
            _assert_close(micro["qD"], expected_qD, rtol=1.0e-12)
        _assert_close(energy["invD"], expected_invD, rtol=1.0e-12)
        _assert_close(isothermal["D"] * energy["invD"], 1.0, rtol=1.0e-12)


def test_all_rate_tagged_result_paths_identify_the_approximation():
    # Cover every tagged success/failure dictionary without triggering expensive
    # nonlinear failures in each solver. Runtime cases also check the public output.
    tree = ast.parse(inspect.getsource(pv))
    tagged_paths = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        entries = {
            key.value: value for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "rate_model" not in entries:
            continue
        tagged_paths += 1
        if ast.literal_eval(entries["rate_model"]) != "exact_nonleptonic":
            raise AssertionError("legacy rate_model compatibility label changed")
        if "rate_approximation" not in entries:
            raise AssertionError(f"missing rate approximation at line {node.lineno}")
        if ast.literal_eval(entries["rate_approximation"]) != "madsen_muB_over_3":
            raise AssertionError(f"incorrect rate approximation at line {node.lineno}")
    if not tagged_paths:
        raise AssertionError("no rate-tagged result paths were checked")


def test_local_energy_closure_uses_absolute_nK_target():
    muB = 1100.0
    muK = 40.0
    temp = 10.0
    bag = 180.0
    jB = 0.02 * const.MeV_fm**3
    target = pv._quark_thermo_state(muB, muK, bag, temp, jB, ms=0.0)
    E = target["h"] * target["u"] * pv._relativistic_gamma_from_u(target["u"])

    solved = pv._solve_local_quark_state_from_nK_E_and_Pi(
        target["nK"],
        E,
        target["Pi"],
        jB,
        bag,
        ms=0.0,
        initial_guess=(muB, muK, temp),
        T_ref=temp,
    )

    _assert_close(solved["nK"], target["nK"], rtol=1.0e-10)
    _assert_close(solved["nK"] / solved["nB"], target["nK"] / target["nB"], rtol=1.0e-10)
    _assert_close(solved["E"], E, rtol=1.0e-10)


def test_new_nK_energy_solver_public_signature():
    if "solve_front_energy_conserving_nK" not in pv.__all__:
        raise AssertionError("new nK energy solver must be public")
    signature = inspect.signature(pv.solve_front_energy_conserving_nK)
    if "T_inf_guess" in signature.parameters:
        raise AssertionError("new nK solver must not expose T_inf_guess")
    if signature.parameters["ms"].default != 0.0:
        raise AssertionError("new nK solver must keep ms tunable with default 0.0")
    if "compact_scale" not in signature.parameters:
        raise AssertionError("new nK solver must expose compact_scale")


def test_solve_front_energy_conserving_nK_regression():
    result = pv.solve_front_energy_conserving_nK(
        T_0minus=10.0,
        nB_0minus=3.0 * N0,
        B_one_forth=180.0,
        a_0plus=0.3,
        ms=0.0,
        tail_eps=1.0e-5,
        n_mesh=40,
        tol_bvp=2.0e-3,
        max_nodes=1200,
        jB_guess=0.02,
        jB_bounds=(1.0e-5, 2.0),
        return_profile=True,
        return_raw_bvp_grid=True,
        verb=False,
    )
    if not bool(result.get("success")):
        raise AssertionError(f"absolute-nK energy solver failed: {result.get('message')}")
    if np.asarray(result["bvp_parameters"]).size != 1:
        raise AssertionError("absolute-nK solver must use only jB as a BVP parameter")
    if result.get("current_definition") != "u_nK_minus_D_dnK_dx":
        raise AssertionError("absolute-nK solver must report the physical kaon current")
    if result.get("rate_model") != "exact_nonleptonic":
        raise AssertionError("absolute-nK solver must retain the legacy rate label")
    if result.get("rate_approximation") != "madsen_muB_over_3":
        raise AssertionError("absolute-nK solver must identify the modified Madsen rate")
    for key in ("nK", "jK", "a", "nB", "muB", "muK", "T_profile", "Gamma_K_profile"):
        if key not in result:
            raise AssertionError(f"missing profile key {key!r}")
    if "q" in result or "bvp_q" in result:
        raise AssertionError("absolute-nK solver must not return q aliases")
    np.testing.assert_allclose(
        result["Gamma_K_profile"],
        _kaon_rate_expected(
            np.asarray(result["muB"]), np.asarray(result["muK"]),
            np.asarray(result["T_profile"]),
        ),
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(result["a"], dtype=float),
        np.asarray(result["nK"], dtype=float) / np.asarray(result["nB"], dtype=float),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    _assert_close(float(result["a_0plus_derived"]), 0.3, rtol=2.0e-4, atol=2.0e-6)
    _assert_close(float(result["jK"][0]), float(result["jB"]), rtol=2.0e-4, atol=2.0e-6)
    if abs(float(result["kaon_equation_residual_norm"])) > 5.0 * 2.0e-3:
        raise AssertionError("absolute-nK profile does not satisfy jK' = -Gamma_K")
    if int(result["downstream_root_calls"]) >= int(result["bvp_ode_calls"]):
        raise AssertionError("downstream closure must not be evaluated at local ODE points")


def test_nK_tail_residual_retains_massive_equilibrium_offset():
    thermo = pv._quark_thermo_state(1100.0, 0.0, 180.0, 10.0, 0.5, ms=50.0)
    if thermo["nK"] <= 0.0:
        raise AssertionError("nonzero ms must produce a nonzero equilibrium nK")
    target = {
        "nK": float(thermo["nK"]),
        "u": float(thermo["u"]),
        "jK": float(thermo["u"] * thermo["nK"]),
        "D": 3.0,
        "lambda": 0.2,
    }
    delta_nK = 7.0
    yb = np.array(
        [
            target["nK"] + delta_nK,
            target["jK"] + (target["D"] * target["lambda"] + target["u"]) * delta_nK,
        ]
    )
    _assert_close(pv._nK_tail_residual(yb, target), 0.0, atol=1.0e-10)


def test_solve_front_energy_conserving_nK_compact_scale_stability():
    common = dict(
        T_0minus=10.0,
        nB_0minus=3.0 * N0,
        B_one_forth=180.0,
        a_0plus=0.3,
        ms=0.0,
        tail_eps=1.0e-5,
        n_mesh=40,
        tol_bvp=3.0e-3,
        max_nodes=1200,
        jB_guess=0.5,
        jB_bounds=(1.0e-5, 2.0),
        verb=False,
    )
    baseline = pv.solve_front_energy_conserving_nK(**common)
    shifted = pv.solve_front_energy_conserving_nK(
        **common,
        compact_scale=0.7 * float(baseline["compact_scale"]),
    )
    if not (baseline.get("success") and shifted.get("success")):
        raise AssertionError("compact-scale stability solves must both converge")
    _assert_close(float(shifted["jB"]), float(baseline["jB"]), rtol=2.0e-3)


def test_analytic_velocity_validates_xi_endpoint_and_density_domain():
    for xi in (-1.0, 1.0):
        try:
            _run_analytic_velocity_with_synthetic_nuclear_state(1100.0, 10.0, 180.0, xi=xi)
        except RuntimeError as exc:
            if "xi" not in str(exc):
                raise AssertionError(f"expected xi validation error, got {exc}") from exc
        else:
            raise AssertionError("expected invalid xi to raise RuntimeError")

    try:
        _run_analytic_velocity_with_synthetic_nuclear_state(
            900.0,
            120.0,
            145.0,
            T_0plus=1.0e-3,
        )
    except RuntimeError as exc:
        if "A_boundary" not in str(exc) and "a_0plus" not in str(exc):
            raise AssertionError(f"expected interface-domain validation error, got {exc}") from exc
    else:
        raise AssertionError("expected invalid A_boundary to raise RuntimeError")

    try:
        _run_analytic_velocity_with_synthetic_nuclear_state(1100.0, 10.0, 180.0, nB_factor=0.0)
    except RuntimeError as exc:
        if "nB_0minus" not in str(exc):
            raise AssertionError(f"expected nB_0minus validation error, got {exc}") from exc
    else:
        raise AssertionError("expected non-positive nB_0minus to raise RuntimeError")


def test_solve_front_isothermal_regression():
    result = pv.solve_front_isothermal(
        T=10.0,
        nB_0minus=2.0 * N0,
        B_one_forth=180.0,
        a_0plus=0.6,
        ms=0.0,
        param=PARAM,
        NM_type="PNM",
        tail_eps=1.0e-6,
        n_mesh=80,
        tol_bvp=1.0e-3,
        max_nodes=3000,
        jB_guess=0.5,
        jB_bounds=(1.0e-4, 5.0),
        kappa_factor=1.0,
        return_profile=False,
        verb=False,
    )

    if not bool(result.get("success")):
        raise AssertionError(f"isothermal solver failed: {result.get('message')}")
    if not np.isfinite(result.get("jB", np.nan)) or result["jB"] <= 0.0:
        raise AssertionError(f"expected positive finite jB, got {result.get('jB')!r}")
    if not np.isfinite(result.get("tail_residual_norm", np.nan)):
        raise AssertionError("expected finite isothermal tail residual")


def test_solve_front_isothermal_warm_case_remains_finite():
    kwargs = dict(
        T=10.0,
        nB_0minus=2.0 * N0,
        B_one_forth=180.0,
        a_0plus=0.6,
        ms=0.0,
        param=PARAM,
        NM_type="PNM",
        tail_eps=1.0e-6,
        n_mesh=80,
        tol_bvp=1.0e-3,
        max_nodes=3000,
        jB_guess=0.5,
        jB_bounds=(1.0e-4, 5.0),
        kappa_factor=1.0,
        return_profile=False,
        verb=False,
    )
    current = pv.solve_front_isothermal(**kwargs)
    if not bool(current.get("success")):
        raise AssertionError(f"isothermal warm solver failed: {current.get('message')}")
    if not np.isfinite(current.get("jB", np.nan)) or current["jB"] <= 0.0:
        raise AssertionError(f"expected positive finite jB, got {current.get('jB')!r}")
    if not np.isfinite(current.get("tail_residual_norm", np.nan)):
        raise AssertionError("expected finite warm-case tail residual")


def test_solve_front_isothermal_low_temperature_case_remains_finite():
    kwargs = dict(
        T=4.5,
        nB_0minus=3.0 * N0,
        B_one_forth=180.0,
        a_0plus=0.3,
        ms=0.0,
        param=PARAM,
        NM_type="PNM",
        tail_eps=1.0e-6,
        n_mesh=60,
        tol_bvp=1.0e-3,
        max_nodes=2000,
        jB_guess=0.5,
        jB_bounds=(1.0e-5, 5.0),
        kappa_factor=1.0,
        return_profile=False,
        verb=False,
    )
    current = pv.solve_front_isothermal(**kwargs)
    if not bool(current.get("success")):
        raise AssertionError(f"isothermal low-temperature solver failed: {current.get('message')}")
    if not np.isfinite(current.get("jB", np.nan)) or current["jB"] <= 0.0:
        raise AssertionError(f"expected positive finite jB, got {current.get('jB')!r}")
    if not np.isfinite(current.get("tail_residual_norm", np.nan)):
        raise AssertionError("expected finite low-temperature tail residual")


def test_solve_front_isothermal_profile_remains_finite():
    kwargs = dict(
        T=10.0,
        nB_0minus=2.0 * N0,
        B_one_forth=180.0,
        a_0plus=0.6,
        ms=0.0,
        param=PARAM,
        NM_type="PNM",
        tail_eps=1.0e-6,
        n_mesh=80,
        tol_bvp=1.0e-3,
        max_nodes=3000,
        jB_guess=0.5,
        jB_bounds=(1.0e-4, 5.0),
        kappa_factor=1.0,
        return_profile=True,
        verb=False,
    )
    current = pv.solve_front_isothermal(**kwargs)
    if not bool(current.get("success")):
        raise AssertionError(f"isothermal profile solver failed: {current.get('message')}")
    for key in ("x", "a", "q", "u", "muB", "muK", "nB", "closure_error"):
        values = np.asarray(current[key], dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise AssertionError(f"expected finite nonempty profile array for {key!r}")


def test_solve_front_isothermal_strict_retry_regression():
    result = pv.solve_front_isothermal(
        T=10.0,
        nB_0minus=2.0 * N0,
        B_one_forth=180.0,
        a_0plus=0.18,
        ms=0.0,
        param=PARAM,
        NM_type="PNM",
        tail_eps=1.0e-6,
        n_mesh=80,
        tol_bvp=1.0e-6,
        max_nodes=3000,
        jB_guess=0.5,
        jB_bounds=(1.0e-4, 5.0),
        kappa_factor=1.0,
        return_profile=False,
        verb=False,
    )

    if not bool(result.get("success")):
        raise AssertionError(f"strict isothermal retry case failed: {result.get('message')}")
    if result.get("retry_source") != "coarse_isothermal_seed":
        raise AssertionError("expected the strict isothermal retry case to use the coarse jB seed recovery path")
    if not np.isfinite(result.get("retry_seed_jB", np.nan)) or result["retry_seed_jB"] <= 0.0:
        raise AssertionError(f"expected positive retry_seed_jB, got {result.get('retry_seed_jB')!r}")
    _assert_close(float(result["jB"]), 0.20069404233893348, rtol=1.0e-8, atol=1.0e-10)
    if abs(float(result["tail_residual_norm"])) > 1.0e-8:
        raise AssertionError(f"expected a tiny tail_residual_norm, got {result.get('tail_residual_norm')!r}")


def test_solve_front_isothermal_strict_profile_retry_regression():
    result = pv.solve_front_isothermal(
        T=10.0,
        nB_0minus=2.0 * N0,
        B_one_forth=180.0,
        a_0plus=0.18,
        ms=0.0,
        param=PARAM,
        NM_type="PNM",
        tail_eps=1.0e-6,
        n_mesh=80,
        tol_bvp=1.0e-6,
        max_nodes=3000,
        jB_guess=0.5,
        jB_bounds=(1.0e-4, 5.0),
        kappa_factor=1.0,
        return_profile=True,
        verb=False,
    )

    if not bool(result.get("success")):
        raise AssertionError(f"strict isothermal profile retry case failed: {result.get('message')}")
    if result.get("retry_source") != "coarse_isothermal_profile":
        raise AssertionError("expected the strict isothermal profile case to use the coarse profile recovery path")
    for key in ("x", "a", "q", "u", "muB", "muK", "nB", "closure_error"):
        if key not in result:
            raise AssertionError(f"expected profile key {key!r} in strict isothermal profile retry case")
    if not np.isfinite(result.get("retry_seed_jB", np.nan)) or result["retry_seed_jB"] <= 0.0:
        raise AssertionError(f"expected positive retry_seed_jB, got {result.get('retry_seed_jB')!r}")


def test_solve_front_energy_conserving_uNmax_regression():
    result = pv.solve_front_energy_conserving_uNmax(
        T_0minus=10.0,
        nB_0minus=3.0 * N0,
        B_one_forth=180.0,
        T_0plus=4.0,
        ms=0.0,
        param=PARAM,
        NM_type="PNM",
        tail_eps=1.0e-5,
        n_mesh=40,
        tol_bvp=1.0e-4,
        max_nodes=1200,
        jB_guess=0.6,
        T_inf_guess=20.0,
        jB_bounds=(1.0e-4, 5.0),
        return_profile=False,
        verb=False,
    )

    if not bool(result.get("success")):
        raise AssertionError(f"uNmax solver failed: {result.get('message')}")
    if result.get("solver_variant") != "energy_conserving_nK_fixed_T_0plus":
        raise AssertionError(f"unexpected solver variant {result.get('solver_variant')!r}")
    if result.get("energy_flux_equation") != "E_rgamma_const":
        raise AssertionError("uNmax solver should keep the E_rgamma_const energy flux")
    if result.get("composition_definition") != "nK_over_local_nB":
        raise AssertionError("uNmax solver must use local nK/nB composition")
    if result.get("current_definition") != "u_nK_minus_D_dnK_dx":
        raise AssertionError("uNmax solver must report the physical kaon current")
    if result.get("rate_model") != "exact_nonleptonic":
        raise AssertionError("uNmax solver must retain the legacy rate label")
    if result.get("rate_approximation") != "madsen_muB_over_3":
        raise AssertionError("uNmax solver must identify the modified Madsen rate")
    if result.get("_root_method") != "solve_bvp_nK_jK_parameter_jB":
        raise AssertionError("uNmax solver must report the nK/jK BVP method")
    if np.asarray(result["bvp_parameters"]).size != 1:
        raise AssertionError("uNmax solver must use only jB as a BVP parameter")
    for key in ("jB", "u_0minus", "u_0plus", "u_inf", "T_inf", "a_0plus", "E", "E_inf", "E_0plus"):
        if not np.isfinite(result.get(key, np.nan)) or result[key] <= 0.0:
            raise AssertionError(f"expected positive finite {key}, got {result.get(key)!r}")
    if not np.isfinite(result.get("a_inf", np.nan)) or result["a_inf"] < 0.0:
        raise AssertionError(f"expected non-negative finite a_inf, got {result.get('a_inf')!r}")
    for removed_key in ("u_0minus_max_candidate", "a_0plus_derived"):
        if removed_key in result:
            raise AssertionError(f"uNmax result must not expose redundant alias {removed_key!r}")
    _assert_close(result["a_0plus"], result["nK_0plus"] / result["nB_0plus"], rtol=1.0e-10, atol=1.0e-12)
    _assert_close(result["a_inf"], result["nK_inf"] / result["nB_inf"], rtol=1.0e-12, atol=1.0e-12)
    _assert_close(result["jK_0plus"], result["jB"], rtol=2.0e-4, atol=2.0e-6)
    _assert_close(result["T_0plus"], 4.0, rtol=1.0e-10, atol=1.0e-8)
    if abs(float(result.get("T_0plus_residual", np.nan))) > 1.0e-8:
        raise AssertionError(f"expected tiny T_0plus residual, got {result.get('T_0plus_residual')!r}")
    if abs(float(result.get("E_right_residual_norm", np.nan))) > 1.0e-4:
        raise AssertionError(f"expected small E right residual, got {result.get('E_right_residual_norm')!r}")


def test_uNmax_accepts_zero_upstream_temperature():
    class NuclearPressureReached(RuntimeError):
        pass

    original_pressure = pv.PNM_n

    def zero_temperature_probe(nB_0minus, T_0minus, param=PARAM, NM_type="PNM"):
        _assert_close(float(T_0minus), 0.0, rtol=0.0, atol=0.0)
        raise NuclearPressureReached("zero-temperature nuclear pressure reached")

    pv.PNM_n = zero_temperature_probe
    try:
        try:
            pv.solve_front_energy_conserving_uNmax(
                T_0minus=0.0,
                nB_0minus=3.24182532 * N0,
                B_one_forth=180.0,
                T_0plus=5.0,
                param=PARAM,
                NM_type="PNM",
            )
        except NuclearPressureReached:
            pass
        else:
            raise AssertionError("uNmax did not evaluate the T(0-)=0 nuclear state")
    finally:
        pv.PNM_n = original_pressure


def test_uNmax_rejects_negative_upstream_temperature():
    try:
        pv.solve_front_energy_conserving_uNmax(
            T_0minus=-1.0,
            nB_0minus=3.24182532 * N0,
            B_one_forth=180.0,
            T_0plus=5.0,
            param=PARAM,
            NM_type="PNM",
        )
    except RuntimeError as exc:
        if "T" not in str(exc):
            raise AssertionError(f"unexpected negative-temperature error: {exc}") from exc
    else:
        raise AssertionError("uNmax accepted negative T_0minus")


def test_solve_front_energy_conserving_uNmax_does_not_use_fixed_a_0plus_solver():
    # The fixed-a_0plus energy-conserving solver has been removed entirely, so the
    # uNmax path can no longer call it; assert its absence, then exercise uNmax.
    assert not hasattr(pv, "_solve_front_energy_conserving_once")
    result = pv.solve_front_energy_conserving_uNmax(
        T_0minus=10.0,
        nB_0minus=3.0 * N0,
        B_one_forth=180.0,
        T_0plus=4.0,
        ms=0.0,
        param=PARAM,
        NM_type="PNM",
        tail_eps=1.0e-5,
        n_mesh=40,
        tol_bvp=1.0e-4,
        max_nodes=1200,
        jB_guess=0.6,
        T_inf_guess=20.0,
        jB_bounds=(1.0e-4, 5.0),
        return_profile=False,
        verb=False,
    )

    if not bool(result.get("success")):
        raise AssertionError(f"dedicated uNmax solver failed: {result.get('message')}")
    if result.get("solver_variant") != "energy_conserving_nK_fixed_T_0plus":
        raise AssertionError(f"unexpected solver variant {result.get('solver_variant')!r}")


def test_solve_front_energy_conserving_uNmax_zero_temperature_limit_remains_finite():
    result = pv.solve_front_energy_conserving_uNmax(
        T_0minus=10.0,
        nB_0minus=3.0 * N0,
        B_one_forth=180.0,
        T_0plus=0.0,
        ms=0.0,
        param=PARAM,
        NM_type="PNM",
        tail_eps=1.0e-4,
        n_mesh=30,
        tol_bvp=5.0e-2,
        max_nodes=800,
        jB_guess=0.6,
        T_inf_guess=20.0,
        jB_bounds=(1.0e-4, 5.0),
        return_profile=False,
        verb=False,
    )

    if not bool(result.get("success")):
        raise AssertionError(f"exact zero-temperature uNmax failed: {result.get('message')}")
    _assert_close(result["T_0plus"], 0.0, rtol=0.0, atol=0.0)
    _assert_close(result["T_0plus_residual"], 0.0, rtol=0.0, atol=0.0)
    if "T_0plus_numerical_floor" in result or "zero_temperature_limit_used" in result:
        raise AssertionError("exact-zero uNmax result should not expose floor/limit bookkeeping")
    for key in ("jB", "u_0minus", "a_0plus", "E", "Pi", "invD_inf"):
        if not np.isfinite(result.get(key, np.nan)) or result[key] <= 0.0:
            raise AssertionError(f"expected positive finite {key}, got {result.get(key)!r}")
    if "D_inf" in result:
        raise AssertionError("uNmax result should expose invD_inf, not D_inf")
    if result["u_0minus"] >= 1.0:
        raise AssertionError(f"exact-zero solve should not blow up u_0minus, got {result['u_0minus']!r}")


def test_solve_front_energy_conserving_uNmax_bad_seed_fails_directly_without_continuation():
    result = pv.solve_front_energy_conserving_uNmax(
        T_0minus=10.0,
        nB_0minus=3.0 * N0,
        B_one_forth=180.0,
        T_0plus=0.0,
        ms=0.0,
        param=PARAM,
        NM_type="PNM",
        tail_eps=1.0e-5,
        n_mesh=30,
        tol_bvp=1.0e-3,
        max_nodes=800,
        jB_guess=1.0e-12,
        T_inf_guess=20.0,
        jB_bounds=(1.0e-4, 5.0),
        return_profile=False,
        verb=False,
    )

    if bool(result.get("success")):
        raise AssertionError("bad-seed direct uNmax solve unexpectedly succeeded")
    if any(key in result for key in ("T_0plus_continuation_steps", "T_0plus_continuation_used", "last_success_T_0plus")):
        raise AssertionError("direct uNmax solve should not expose continuation bookkeeping")
    if "continuation" in str(result.get("message", "")).lower():
        raise AssertionError(f"direct uNmax failure should not mention continuation: {result.get('message')!r}")


def test_solve_front_energy_conserving_uNmax_profile():
    result = pv.solve_front_energy_conserving_uNmax(
        T_0minus=10.0,
        nB_0minus=3.0 * N0,
        B_one_forth=180.0,
        T_0plus=4.0,
        ms=0.0,
        param=PARAM,
        NM_type="PNM",
        tail_eps=1.0e-5,
        n_mesh=40,
        tol_bvp=1.0e-4,
        max_nodes=1200,
        jB_guess=0.6,
        T_inf_guess=20.0,
        jB_bounds=(1.0e-4, 5.0),
        return_profile=True,
        verb=False,
    )

    if not bool(result.get("success")):
        raise AssertionError(f"uNmax profile solver failed: {result.get('message')}")
    for key in (
        "x",
        "nK",
        "jK",
        "a",
        "u",
        "nB",
        "muB",
        "muK",
        "T_profile",
        "P_profile",
        "h_profile",
        "r_gamma_profile",
        "invD_profile",
        "Gamma_K_profile",
        "jK_prime_profile",
        "kaon_equation_residual_profile",
    ):
        if key not in result:
            raise AssertionError(f"expected profile key {key!r}")
    for key in ("q", "q_end", "q_left_residual", "q_left_residual_norm"):
        if key in result:
            raise AssertionError(f"uNmax result must not expose old q alias {key!r}")
    if "D_profile" in result:
        raise AssertionError("uNmax profile should expose invD_profile, not D_profile")

    T_profile = np.asarray(result["T_profile"], dtype=float)
    np.testing.assert_allclose(
        result["Gamma_K_profile"],
        _kaon_rate_expected(
            np.asarray(result["muB"]), np.asarray(result["muK"]), T_profile,
        ),
        rtol=1.0e-12,
        atol=0.0,
    )
    E_profile = np.asarray(result["h_profile"], dtype=float) * np.asarray(result["u"], dtype=float) * np.asarray(result["r_gamma_profile"], dtype=float)
    if not np.all(np.isfinite(T_profile)) or np.any(T_profile <= 0.0):
        raise AssertionError("expected positive finite uNmax temperature profile")
    if np.ptp(T_profile) <= 1.0e-6 * max(1.0, abs(float(np.mean(T_profile)))):
        raise AssertionError("expected non-trivial x dependence in the uNmax temperature profile")
    _assert_close(float(np.asarray(result["a"], dtype=float)[0]), float(result["a_0plus"]), rtol=1.0e-8, atol=1.0e-10)
    np.testing.assert_allclose(
        np.asarray(result["a"], dtype=float),
        np.asarray(result["nK"], dtype=float) / np.asarray(result["nB"], dtype=float),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    _assert_close(float(np.asarray(result["jK"], dtype=float)[0]), float(result["jB"]), rtol=2.0e-4, atol=2.0e-6)
    _assert_close(float(result["nK_0plus"]), float(np.asarray(result["nK"], dtype=float)[0]), rtol=1.0e-8, atol=1.0e-8)
    _assert_close(float(result["nK_end"]), float(np.asarray(result["nK"], dtype=float)[-1]), rtol=1.0e-12, atol=1.0e-12)
    _assert_close(float(result["jK_end"]), float(np.asarray(result["jK"], dtype=float)[-1]), rtol=1.0e-12, atol=1.0e-12)

    E_scale = max(abs(float(result["E"])), 1.0)
    if float(np.ptp(E_profile)) / E_scale > 1.0e-6:
        raise AssertionError("expected h*u*r_gamma to remain constant across the uNmax profile")
    if float(np.max(np.abs(E_profile - float(result["E"])))) / E_scale > 1.0e-6:
        raise AssertionError("expected reconstructed uNmax h*u*r_gamma profile to match scalar E")
    momentum_profile = (
        np.asarray(result["h_profile"], dtype=float) * np.asarray(result["u"], dtype=float) ** 2
        + np.asarray(result["P_profile"], dtype=float)
    )
    momentum_scale = max(abs(float(result["Pi"])), 1.0)
    if float(np.max(np.abs(momentum_profile - float(result["Pi"])))) / momentum_scale > 1.0e-6:
        raise AssertionError("expected reconstructed uNmax momentum flux to match scalar Pi")
    if abs(float(result.get("kaon_equation_residual_norm", np.inf))) > 5.0e-4:
        raise AssertionError("uNmax profile does not satisfy jK' = -Gamma_K")


def test_solve_front_energy_conserving_uNmax_ignores_bad_T_inf_seed_as_bvp_parameter():
    result = pv.solve_front_energy_conserving_uNmax(
        T_0minus=10.0,
        nB_0minus=3.0 * N0,
        B_one_forth=180.0,
        T_0plus=4.0,
        ms=0.0,
        param=PARAM,
        NM_type="PNM",
        tail_eps=1.0e-5,
        n_mesh=40,
        tol_bvp=1.0e-4,
        max_nodes=1200,
        jB_guess=0.6,
        T_inf_guess=1.0e-300,
        jB_bounds=(1.0e-4, 5.0),
        return_profile=False,
        verb=False,
    )

    if not bool(result.get("success")):
        raise AssertionError(f"uNmax should not depend on T_inf_guess as a BVP seed: {result.get('message')}")
    if np.asarray(result["bvp_parameters"]).size != 1:
        raise AssertionError("uNmax solver must use only jB as a BVP parameter")


def test_uNmax_default_jB_guess_matches_energy_nK_policy():
    nK_signature = inspect.signature(pv.solve_front_energy_conserving_nK)
    uNmax_signature = inspect.signature(pv.solve_front_energy_conserving_uNmax)
    for name in ("ms", "NM_type", "tail_eps", "n_mesh", "tol_bvp", "max_nodes", "jB_bounds"):
        if nK_signature.parameters[name].default != uNmax_signature.parameters[name].default:
            raise AssertionError(f"uNmax default {name} has drifted from energy_conserving_nK")

    captured = {}
    original_solver = pv._solve_front_energy_conserving_uNmax_once

    def fake_solver(*args, **kwargs):
        captured["jB_guess"] = float(kwargs["jB_guess"])
        return {"success": False, "message": "default-seed probe"}

    pv._solve_front_energy_conserving_uNmax_once = fake_solver
    try:
        nB_0minus = 3.0 * N0
        pv.solve_front_energy_conserving_uNmax(
            T_0minus=10.0,
            nB_0minus=nB_0minus,
            B_one_forth=180.0,
            T_0plus=0.5,
        )
    finally:
        pv._solve_front_energy_conserving_uNmax_once = original_solver

    expected = max(1.0e-12, 1.0e-8 * nB_0minus)
    _assert_close(captured["jB_guess"], expected, rtol=0.0, atol=0.0)


def test_energy_microphysics_zero_temperature_uses_inverse_diffusion_limit():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        micro = pv._microphysics_from_quark_state_energy(
            1100.0,
            0.0,
            allow_zero_temperature=True,
        )

    runtime_warnings = [item for item in caught if issubclass(item.category, RuntimeWarning)]
    if runtime_warnings:
        raise AssertionError(f"unexpected runtime warnings: {[str(item.message) for item in runtime_warnings]}")
    _assert_close(micro["invD"], 0.0, rtol=0.0, atol=0.0)
    _assert_close(micro["eta"], 0.0, rtol=0.0, atol=0.0)
    if not np.isfinite(micro["gamma"]) or micro["gamma"] <= 0.0:
        raise AssertionError(f"expected finite positive gamma, got {micro['gamma']!r}")
    if "D" in micro:
        raise AssertionError("energy microphysics helper should expose invD, not D")


def test_energy_microphysics_positive_temperature_computes_inverse_diffusion_directly():
    micro = pv._microphysics_from_quark_state_energy(1100.0, 1.0e-3)
    if not np.isfinite(micro["invD"]) or micro["invD"] < 0.0:
        raise AssertionError(f"expected finite nonnegative invD, got {micro['invD']!r}")
    if "D" in micro:
        raise AssertionError("energy microphysics helper should not compute or return D")


def test_low_temperature_massive_0plus_entropy_stays_positive():
    temp = 0.1
    nB_0minus = 3.0 * N0
    bag = 180.0
    ms = 50.0
    a_0plus = 0.01
    jB = 0.008
    T_inf = temp + 50.0

    p_n = pv.PNM_n(nB_0minus, temp, param=PARAM, NM_type="PNM")
    e_n = pv.edensNM_n(nB_0minus, temp, param=PARAM)
    h_n = p_n + e_n
    pi = h_n * (jB / nB_0minus) ** 2 + p_n

    muB_inf = pv._solve_muB_inf_at_muK0_for_given_Pi_ms(pi, jB, bag, T_inf, ms=ms)
    thermo_inf = pv._quark_thermo_state(muB_inf, 0.0, bag, T_inf, jB, ms=ms)
    muB_0plus, muK_0plus = pv._solve_interface_0plus_from_a_0plus_and_Pi(
        a_0plus,
        pi,
        jB,
        thermo_inf["nB"],
        thermo_inf["nK"],
        bag,
        temp,
        ms=ms,
    )
    thermo_0plus = pv._quark_thermo_state(muB_0plus, muK_0plus, bag, temp, jB, ms=ms)

    if thermo_0plus["s"] <= 0.0:
        raise AssertionError(f"expected positive low-T x = 0+ state entropy, got {thermo_0plus['s']!r}")
    if thermo_0plus["w"] <= 0.0:
        raise AssertionError(f"expected positive low-T x = 0+ state entropy flux, got {thermo_0plus['w']!r}")


def _low_temperature_closure_reference():
    """A representative (E, Pi, jB) together with its exact T=0 endpoint."""
    T, nB_0minus, B_one_forth, jB = 10.0, 3.0 * N0, 180.0, 0.6
    P_0minus = float(pv.PNM_n(nB_0minus, T, param=PARAM, NM_type="PNM"))
    e_0minus = float(pv.edensNM_n(nB_0minus, T, param=PARAM))
    h_0minus = P_0minus + e_0minus
    u_0minus = jB / nB_0minus
    E = float(h_0minus * u_0minus * pv._relativistic_gamma_from_u(u_0minus))
    Pi = float(h_0minus * u_0minus * u_0minus + P_0minus)
    nuclear_state = {
        "P_0minus": P_0minus,
        "e_0minus": e_0minus,
        "h_0minus": h_0minus,
        "nB_0minus": nB_0minus,
        "h_over_nB_0minus": h_0minus / nB_0minus,
        "T_0minus": T,
    }
    endpoint = pv._solve_analytic_inf_endpoint_for_u_0minus(u_0minus, nuclear_state, B_one_forth)
    zero_state = pv._solve_interface_0plus_from_T_0plus_E_and_Pi(
        0.0, E, Pi, jB, 1.0, 0.0, B_one_forth,
        initial_guess=(float(endpoint["muB_inf"]), 1.0e-3),
    )
    return {
        "E": E,
        "Pi": Pi,
        "jB": jB,
        "B_one_forth": B_one_forth,
        "T_inf": float(endpoint["T_inf"]),
        "nK_zero": float(zero_state["nK"]),
        "muB_zero": float(zero_state["muB"]),
        "muK_zero": float(zero_state["muK"]),
    }


def test_local_nK_closure_rejects_nK_above_the_zero_temperature_limit():
    """
    At fixed (E, Pi) the local quark state exists only up to nK(T=0). A larger
    nK must fail loudly instead of returning a fabricated near-zero temperature.
    """
    ref = _low_temperature_closure_reference()
    accepted = []
    for excess in (1.0e-3, 1.0e-1, 1.0):
        try:
            state = pv._solve_local_quark_state_from_nK_E_and_Pi(
                ref["nK_zero"] + excess,
                ref["E"],
                ref["Pi"],
                ref["jB"],
                ref["B_one_forth"],
                initial_guess=(ref["muB_zero"], ref["muK_zero"], 1.0e-2),
                T_ref=ref["T_inf"],
            )
        except RuntimeError:
            continue
        accepted.append((excess, float(state["T"])))
    if accepted:
        raise AssertionError(
            "local closure accepted nK above its T=0 limit, returning "
            + ", ".join(f"nK_zero+{e:g} -> T={t:.3e}" for e, t in accepted)
        )


def test_local_nK_closure_temperature_is_independent_of_the_initial_guess():
    """
    E and Pi depend on T only at O(T**2), so a closure parameterised in log T
    stops constraining T at low temperature and returns its own seed. The
    recovered temperature must be a function of nK alone.
    """
    ref = _low_temperature_closure_reference()
    for gap in (1.0, 1.0e-2, 1.0e-3):
        recovered = []
        for seed in (10.0, 1.0, 0.1, 0.01, 1.0e-4):
            state = pv._solve_local_quark_state_from_nK_E_and_Pi(
                ref["nK_zero"] - gap,
                ref["E"],
                ref["Pi"],
                ref["jB"],
                ref["B_one_forth"],
                initial_guess=(ref["muB_zero"], ref["muK_zero"], seed),
                T_ref=ref["T_inf"],
            )
            recovered.append(float(state["T"]))
        recovered = np.asarray(recovered, dtype=float)
        spread = float(recovered.max() - recovered.min()) / float(recovered.mean())
        if not spread <= 1.0e-3:
            raise AssertionError(
                f"recovered T depends on the initial guess at nK_zero-{gap:g}: "
                f"{recovered!r} (relative spread {spread:.3e})"
            )


def test_local_nK_closure_accepts_exact_zero_temperature_seed():
    """The T**2 closure must accept w=0 as an ordinary initial root seed."""
    ref = _low_temperature_closure_reference()
    state = pv._solve_local_quark_state_from_nK_E_and_Pi(
        ref["nK_zero"],
        ref["E"],
        ref["Pi"],
        ref["jB"],
        ref["B_one_forth"],
        initial_guess=(ref["muB_zero"], ref["muK_zero"], 0.0),
        T_ref=ref["T_inf"],
    )
    if not (np.isfinite(state["T"]) and 0.0 <= float(state["T"]) <= 1.0e-4):
        raise AssertionError(f"expected a numerical T=0 state, got T={state['T']!r}")


def test_local_nK_closure_follows_the_square_root_low_temperature_law():
    """
    Near the T=0 edge the closure must follow T = c*sqrt(nK_zero - nK), the
    Sommerfeld statement that E and Pi are analytic in T**2.
    """
    ref = _low_temperature_closure_reference()
    gaps = np.array([1.0e2, 1.0, 1.0e-2, 1.0e-4])
    temperatures = []
    for gap in gaps:
        state = pv._solve_local_quark_state_from_nK_E_and_Pi(
            ref["nK_zero"] - float(gap),
            ref["E"],
            ref["Pi"],
            ref["jB"],
            ref["B_one_forth"],
            initial_guess=(ref["muB_zero"], ref["muK_zero"], 1.0e-2),
            T_ref=ref["T_inf"],
        )
        temperatures.append(float(state["T"]))
    temperatures = np.asarray(temperatures, dtype=float)
    slope = temperatures / np.sqrt(gaps)
    spread = float(slope.max() - slope.min()) / float(slope.mean())
    if not spread <= 1.0e-2:
        raise AssertionError(
            f"T/sqrt(nK_zero-nK) is not constant near the T=0 edge: c={slope!r} "
            f"(relative spread {spread:.3e}), T={temperatures!r}"
        )


def _cold_interface_reference():
    """Reference (E, Pi, jB) plus the composition a = nK/nB of its T=0 state."""
    ref = _low_temperature_closure_reference()
    zero_state = pv._solve_interface_0plus_from_T_0plus_E_and_Pi(
        0.0, ref["E"], ref["Pi"], ref["jB"], 1.0, 0.0, ref["B_one_forth"],
        initial_guess=(ref["muB_zero"], ref["muK_zero"]),
    )
    ref["a_max"] = float(zero_state["nK"]) / float(zero_state["nB"])
    return ref


def test_local_a_interface_temperature_is_independent_of_the_initial_guess():
    """
    The a_0plus interface closure shares the O(T**2) degeneracy of the absolute-nK
    closure, so it must also recover a seed-independent temperature as the
    interface composition approaches its T=0 value.
    """
    ref = _cold_interface_reference()
    for offset in (1.0e-3, 1.0e-5, 1.0e-6):
        recovered = []
        for seed in (30.0, 10.0, 1.0, 0.1, 0.01):
            state = pv._solve_interface_state_from_local_a_E_and_Pi(
                ref["a_max"] - offset,
                ref["E"],
                ref["Pi"],
                ref["jB"],
                ref["B_one_forth"],
                initial_guess=(ref["muB_zero"], ref["muK_zero"], seed),
            )
            recovered.append(float(state["T"]))
        recovered = np.asarray(recovered, dtype=float)
        spread = float(recovered.max() - recovered.min()) / float(recovered.mean())
        if not spread <= 1.0e-3:
            raise AssertionError(
                f"recovered interface T depends on the initial guess at "
                f"a_max-{offset:g}: {recovered!r} (relative spread {spread:.3e})"
            )


def test_local_a_interface_rejects_compositions_above_the_zero_temperature_value():
    """A composition richer than the T=0 interface state is unreachable."""
    ref = _cold_interface_reference()
    accepted = []
    for offset in (1.0e-4, 1.0e-2):
        a_target = min(ref["a_max"] + offset, 1.0 - 1.0e-9)
        try:
            state = pv._solve_interface_state_from_local_a_E_and_Pi(
                a_target,
                ref["E"],
                ref["Pi"],
                ref["jB"],
                ref["B_one_forth"],
                initial_guess=(ref["muB_zero"], ref["muK_zero"], 1.0),
            )
        except RuntimeError:
            continue
        accepted.append((offset, float(state["T"])))
    if accepted:
        raise AssertionError(
            "local-a interface closure accepted an unreachable composition, returning "
            + ", ".join(f"a_max+{o:g} -> T={t:.3e}" for o, t in accepted)
        )


# ---------------------------------------------------------------------------
# Thermal-conducting front solver (solve_front_thermal_conducting)
# ---------------------------------------------------------------------------

_TC_Z3 = 1.2020569031595943


def _tc_reference_kappa(muB):
    """kappa_th(T->0+) = pi^3 vF^2 qD^2 / (48 zeta(3) alpha_s^2), vF = 1."""
    qD = np.sqrt(6.0 * 0.4 / np.pi) * (float(muB) / 3.0)
    return float(np.pi**3 * qD**2 / (48.0 * _TC_Z3 * 0.4**2))


def test_thermal_conductivity_zero_temperature_limit_is_finite_and_positive():
    """(1) kappa_th at exactly T=0 must equal the analytic limit, not 0/0."""
    for muB in (900.0, 1100.0, 1400.0):
        out = pv._thermal_conductivity(muB, 0.0)
        kappa = float(out["kappa_th"])
        ref = _tc_reference_kappa(muB)
        if not np.isfinite(kappa) or kappa <= 0.0:
            raise AssertionError(f"kappa_th(T=0, muB={muB}) = {kappa!r} is not finite positive")
        if abs(kappa / ref - 1.0) > 1.0e-12:
            raise AssertionError(
                f"kappa_th(T=0, muB={muB}) = {kappa:.10e} != analytic limit {ref:.10e}"
            )
        # continuity: approaching zero must reproduce the same limit
        near = float(pv._thermal_conductivity(muB, 1.0e-6)["kappa_th"])
        if abs(near / ref - 1.0) > 1.0e-6:
            raise AssertionError(
                f"kappa_th is discontinuous at T->0+: {near:.10e} vs {ref:.10e}"
            )


def test_thermal_conductivity_low_y_asymptote_and_domain_guard():
    """(2) I_kappa reproduces its low-y asymptote; outside the validated band it raises."""
    for y in (1.0e-4, 1.0e-3, 1.0e-2, 0.1):
        got = pv._ikappa_low_y(y)
        want = 2.0 * _TC_Z3 * y * y
        if abs(got / want - 1.0) > 1.0e-12:
            raise AssertionError(f"I_kappa({y}) = {got!r}, expected {want!r}")

    muB = 1100.0
    qD = pv._TRANSPORT_QD_COEFF * (muB / 3.0)
    bad_T = 5.0 * pv._THERMAL_Y_MAX * qD
    try:
        pv._thermal_conductivity(muB, bad_T)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            f"kappa_th silently extrapolated to y = {bad_T / qD:.3f} > y_max"
        )

    # an injected model must actually be used
    calls = {"n": 0}

    def fake_I(y):
        calls["n"] += 1
        return 2.0 * _TC_Z3 * y * y

    model = {"name": "injected", "I_kappa": fake_I, "low_y_coeff": 2.0 * _TC_Z3, "y_max": 0.3}
    out = pv._thermal_conductivity(muB, 5.0, kappa_model=model)
    if calls["n"] == 0:
        raise AssertionError("injected I_kappa callback was never called")
    if out["model"] != "injected":
        raise AssertionError(f"model metadata not propagated: {out['model']!r}")


def test_thermal_ikappa_full_integral_reproduces_both_published_asymptotes():
    """
    (2) The full HP93 Eq. (59) integral must reproduce Eq. (60) in both limits.

    Low y: I_kappa -> 2*zeta(3)*y**2, approached as y**(2/3) because the paper
    states the leading correction enters at order (T/qD)**(8/3).
    High y: I_kappa -> ln(y)/3 + const, i.e. the SLOPE in ln(y) must be 1/3.
    """
    # low-y limit
    for y, tol in ((1.0e-4, 0.02), (1.0e-3, 0.06), (1.0e-2, 0.16)):
        full = pv._ikappa_quadrature(y)
        ratio = full / (2.0 * _TC_Z3 * y * y)
        if abs(ratio - 1.0) > tol:
            raise AssertionError(
                f"I_kappa({y}) / (2 zeta3 y^2) = {ratio:.6f}, outside 1 +/- {tol}"
            )
    # the approach must be monotone toward 1 as y decreases
    r_small = pv._ikappa_quadrature(1.0e-4) / (2.0 * _TC_Z3 * 1.0e-8)
    r_large = pv._ikappa_quadrature(1.0e-3) / (2.0 * _TC_Z3 * 1.0e-6)
    if not (abs(r_small - 1.0) < abs(r_large - 1.0)):
        raise AssertionError("I_kappa does not approach its low-y asymptote as y decreases")

    # high-y limit: slope in ln(y) must be 1/3
    y_hi = np.array([30.0, 100.0, 300.0])
    vals = np.array([pv._ikappa_quadrature(float(v)) for v in y_hi])
    slope = float(np.polyfit(np.log(y_hi), vals, 1)[0])
    if abs(slope - 1.0 / 3.0) > 5.0e-3:
        raise AssertionError(f"high-y slope in ln(y) is {slope:.6f}, expected 1/3")

    # quadrature must be converged in its order
    coarse = pv._ikappa_quadrature(1.0, nx=120, nt=120)
    fine = pv._ikappa_quadrature(1.0, nx=400, nt=400)
    if abs(coarse / fine - 1.0) > 1.0e-6:
        raise AssertionError(f"I_kappa quadrature is not converged: {coarse!r} vs {fine!r}")

    # the interpolated table must track the direct quadrature
    for y in (2.0e-5, 1.0e-3, 0.03, 0.5, 5.0, 40.0):
        table = pv._ikappa_full(y)
        direct = pv._ikappa_quadrature(y)
        if abs(table / direct - 1.0) > 1.0e-4:
            raise AssertionError(
                f"I_kappa table disagrees with quadrature at y={y}: {table!r} vs {direct!r}"
            )


def test_thermal_conductivity_is_dimensionally_consistent_with_energy_flux():
    """(3) kappa_th * dT/dx must carry the same MeV^4 units as h*gamma*u."""
    muB, T = 1100.0, 20.0
    kappa = float(pv._thermal_conductivity(muB, T)["kappa_th"])
    # kappa ~ qD^2 ~ muq^2 in natural MeV units => kappa/muq^2 is O(1)-ish, dimensionless
    ratio = kappa / (muB / 3.0) ** 2
    if not (0.1 < ratio < 100.0):
        raise AssertionError(
            f"kappa_th/muq^2 = {ratio:.3e} is not O(1); natural-MeV convention broken"
        )
    # scaling check: kappa must be exactly quadratic in muB at fixed y
    k1 = float(pv._thermal_conductivity(1000.0, 0.0)["kappa_th"])
    k2 = float(pv._thermal_conductivity(2000.0, 0.0)["kappa_th"])
    if abs(k2 / k1 - 4.0) > 1.0e-12:
        raise AssertionError(f"kappa_th is not quadratic in muB: ratio {k2 / k1!r}")


def _tc_reference_case(T_0minus=10.0, nB_over_n0=3.0, B14=180.0):
    nB_0minus = nB_over_n0 * N0
    P_0minus = float(pv.PNM_n(nB_0minus, T_0minus, param=PARAM, NM_type="PNM"))
    e_0minus = float(pv.edensNM_n(nB_0minus, T_0minus, param=PARAM))
    h_0minus = P_0minus + e_0minus
    return {
        "T_0minus": T_0minus,
        "B14": B14,
        "nuclear_state": {
            "P_0minus": P_0minus,
            "e_0minus": e_0minus,
            "h_0minus": h_0minus,
            "nB_0minus": nB_0minus,
            "h_over_nB_0minus": h_0minus / nB_0minus,
            "T_0minus": T_0minus,
        },
    }


def test_thermal_local_closure_satisfies_nK_and_momentum_equations():
    """(4) The 2x2 closure must reproduce the requested nK and Pi at prescribed T."""
    case = _tc_reference_case()
    ns = case["nuclear_state"]
    jB = float(pv._default_energy_jB_guess(ns["nB_0minus"]))
    u_0minus = jB / ns["nB_0minus"]
    Pi = ns["P_0minus"] + ns["h_0minus"] * u_0minus * u_0minus
    endpoint = pv._solve_analytic_inf_endpoint_for_u_0minus(u_0minus, ns, case["B14"])
    muB_inf, T_inf = float(endpoint["muB_inf"]), float(endpoint["T_inf"])
    thermo_inf = pv._quark_thermo_state(muB_inf, 0.0, case["B14"], T_inf, jB)
    nK_inf = float(thermo_inf["nK"])

    for nK_target, T_value in (
        (nK_inf, T_inf),
        (nK_inf * 1.05, T_inf),
        (nK_inf * 1.10, 0.5 * T_inf),
    ):
        state = pv._solve_local_quark_state_from_nK_T_and_Pi(
            nK_target, T_value, Pi, jB, case["B14"], initial_guess=(muB_inf, 1.0e-3)
        )
        if abs(float(state["nK"]) - nK_target) > 1.0e-6 * max(abs(nK_target), 1.0):
            raise AssertionError(
                f"closure nK residual too large: {state['nK']!r} vs {nK_target!r}"
            )
        if abs(float(state["Pi"]) - Pi) > 1.0e-6 * max(abs(Pi), 1.0):
            raise AssertionError(f"closure Pi residual too large: {state['Pi']!r} vs {Pi!r}")
        if abs(float(state["T"]) - T_value) > 1.0e-12 * max(T_value, 1.0):
            raise AssertionError("closure must not alter the prescribed temperature")


def test_thermal_local_closure_accepts_exact_zero_temperature():
    """The closure must work at exactly T=0 without a w=T^2 trick or a floor."""
    case = _tc_reference_case()
    ns = case["nuclear_state"]
    jB = float(pv._default_energy_jB_guess(ns["nB_0minus"]))
    u_0minus = jB / ns["nB_0minus"]
    Pi = ns["P_0minus"] + ns["h_0minus"] * u_0minus * u_0minus
    endpoint = pv._solve_analytic_inf_endpoint_for_u_0minus(u_0minus, ns, case["B14"])
    muB_inf = float(endpoint["muB_inf"])
    thermo_inf = pv._quark_thermo_state(muB_inf, 0.0, case["B14"], float(endpoint["T_inf"]), jB)
    state = pv._solve_local_quark_state_from_nK_T_and_Pi(
        float(thermo_inf["nK"]), 0.0, Pi, jB, case["B14"], initial_guess=(muB_inf, 1.0e-3)
    )
    if float(state["T"]) != 0.0:
        raise AssertionError("exact T=0 closure did not return T=0")
    if not np.isfinite(float(state["nB"])) or float(state["nB"]) <= 0.0:
        raise AssertionError("exact T=0 closure returned a non-physical density")


def test_thermal_downstream_jacobian_has_one_stable_direction():
    """(5) GO/NO-GO gate: the BC set is valid only if k = 1 (c*f > d*e, a+f > 0)."""
    case = _tc_reference_case()
    ns = case["nuclear_state"]
    jB = float(pv._default_energy_jB_guess(ns["nB_0minus"]))
    info = pv._thermal_downstream_analysis(jB, ns, case["B14"])
    if int(info["stable_dimension"]) != 1:
        raise AssertionError(
            f"downstream stable manifold has dimension {info['stable_dimension']}, "
            f"expected 1; eigenvalues={info['eigenvalues']!r}"
        )
    if not (float(info["cf_minus_de"]) > 0.0):
        raise AssertionError(f"c*f - d*e = {info['cf_minus_de']!r} must be positive")
    if not (float(info["trace"]) > 0.0):
        raise AssertionError(f"trace J = {info['trace']!r} must be positive")
    if not (float(info["lambda_stable"]) > 0.0):
        raise AssertionError("lambda_stable must be positive")
    if np.asarray(info["growing_left_vectors"]).shape != (2, 3):
        raise AssertionError("expected exactly two growing left eigenvectors of length 3")


def test_thermal_conducting_solver_is_registered_and_leaves_uNmax_untouched():
    """(14) Regression: the new path is additive."""
    if not hasattr(pv, "solve_front_thermal_conducting"):
        raise AssertionError("solve_front_thermal_conducting is missing")
    if "solve_front_thermal_conducting" not in pv.__all__:
        raise AssertionError("solve_front_thermal_conducting is not exported in __all__")
    for name in (
        "solve_front_energy_conserving_uNmax",
        "solve_front_energy_conserving_nK",
        "solve_front_isothermal",
        "analytic_velocity_bound",
    ):
        if not hasattr(pv, name):
            raise AssertionError(f"pre-existing public solver {name} disappeared")
    sig = inspect.signature(pv.solve_front_energy_conserving_uNmax)
    expected = [
        "T_0minus", "nB_0minus", "B_one_forth", "T_0plus", "ms", "param", "NM_type", "tail_eps",
        "n_mesh", "tol_bvp", "max_nodes", "jB_guess", "T_inf_guess", "jB_bounds",
        "return_profile", "verb", "continuation_guess",
    ]
    if list(sig.parameters) != expected:
        raise AssertionError(
            f"solve_front_energy_conserving_uNmax signature changed: {list(sig.parameters)}"
        )


def test_thermal_compact_mesh_has_exact_endpoints_and_monotone_tail():
    """The compact BVP mesh must reach exactly s=1-tail_eps and resolve its tail."""
    for tail_eps in (1.0e-3, 1.0e-6, 1.0e-8):
        mesh = pv._thermal_compact_mesh(200, tail_eps)
        if mesh[0] != 0.0 or mesh[-1] != 1.0 - tail_eps:
            raise AssertionError(f"wrong compact endpoints for tail_eps={tail_eps:g}")
        if np.any(np.diff(mesh) <= 0.0):
            raise AssertionError("compact mesh is not strictly increasing")
        if not np.any(mesh > 1.0 - 10.0 * tail_eps):
            raise AssertionError("compact mesh does not resolve the requested tail")


def test_thermal_tail_schedule_reaches_requested_depth():
    if pv._thermal_tail_schedule(1.0e-8) != [1.0e-4, 1.0e-6, 1.0e-8]:
        raise AssertionError("default thermal tail schedule is not geometric")
    if pv._thermal_tail_schedule(1.0e-3) != [1.0e-3]:
        raise AssertionError("coarse requested tail should not be deepened implicitly")


def test_thermal_profile_guess_extends_with_stable_one_minus_s_tail():
    previous_s = np.array([0.0, 0.5, 0.9])
    equilibrium = np.array([2.0, 3.0, 4.0])
    amplitudes = np.array([5.0, -2.0, 1.0])
    previous_y = equilibrium[:, None] + amplitudes[:, None] * (1.0 - previous_s)[None, :]
    profile = {
        "s": previous_s,
        "physical_y": previous_y,
        "equilibrium_y": equilibrium,
    }
    new_s = np.array([0.0, 0.5, 0.9, 0.99])
    got = pv._thermal_interpolate_profile_guess(profile, new_s, equilibrium, np.ones(3))
    want = equilibrium[:, None] + amplitudes[:, None] * (1.0 - new_s)[None, :]
    if not np.allclose(got, want, rtol=0.0, atol=1.0e-12):
        raise AssertionError(f"stable compact-tail extension is wrong:\n{got!r}\n!=\n{want!r}")


_TC_SOLVE_CACHE = {}


def _tc_reference_solution():
    """One converged T(0)=0 solve, shared by the slow end-to-end tests."""
    if "base" not in _TC_SOLVE_CACHE:
        _TC_SOLVE_CACHE["base"] = pv.solve_front_thermal_conducting(
            10.0, 3.0 * N0, 180.0, T_0plus=0.0, return_profile=True
        )
    return _TC_SOLVE_CACHE["base"]


def test_thermal_conducting_uses_requested_compact_endpoint():
    """The BVP itself must end at s=1-tail_eps, not post-compute a cosmetic s."""
    r = _tc_reference_solution()
    if not r.get("success"):
        raise AssertionError(f"reference thermal solve failed: {r.get('message')!r}")
    expected_s_end = 1.0 - float(r["tail_eps"])
    if not np.isclose(float(r["s_end"]), expected_s_end, rtol=0.0, atol=1.0e-14):
        raise AssertionError(f"s_end={r['s_end']!r}, expected {expected_s_end!r}")
    if not np.isclose(float(r["s"][-1]), expected_s_end, rtol=0.0, atol=1.0e-14):
        raise AssertionError("returned profile does not reach the solved compact endpoint")
    expected_efolds = -np.log(float(r["tail_eps"]))
    if abs(float(r["compact_tail_e_folds"]) - expected_efolds) > 1.0e-10:
        raise AssertionError(
            f"compact tail has {r['compact_tail_e_folds']!r} e-folds, "
            f"expected {expected_efolds!r}"
        )
    if r.get("coordinate") != "s=1-exp(-lambda_compact*x)":
        raise AssertionError(f"unexpected coordinate metadata: {r.get('coordinate')!r}")


def test_thermal_conducting_reaches_downstream_equilibrium():
    """Success requires actual proximity to muK=q_th=nK=jK=0 downstream."""
    r = _tc_reference_solution()
    if not r.get("success"):
        raise AssertionError(f"reference thermal solve failed: {r.get('message')!r}")
    limit = max(10.0 * float(r["tol_bvp"]), 100.0 * float(r["tail_eps"]))
    for key in (
        "tail_state_residual_norm",
        "tail_muK_residual_norm",
        "tail_q_residual_norm",
    ):
        value = float(r[key])
        if not np.isfinite(value) or value > limit:
            raise AssertionError(f"{key}={value!r} exceeds compact-tail limit {limit!r}")
    if abs(float(r["muK_profile"][-1])) > limit * max(abs(float(r["muB_inf"])), 1.0):
        raise AssertionError("downstream muK is not close to zero")
    if abs(float(r["q_th_profile"][-1])) > limit * max(abs(float(r["F_E"])), 1.0):
        raise AssertionError("downstream conductive heat flux is not close to zero")


def test_thermal_tail_continuation_converges_physical_outputs():
    """The physical branch must stop moving as the compact tail is deepened."""
    r = _tc_reference_solution()
    if not r.get("success"):
        raise AssertionError(f"reference thermal solve failed: {r.get('message')!r}")
    successful = [
        entry for entry in r["tail_continuation_history"] if entry["success"]
    ]
    if len(successful) < 2:
        raise AssertionError("expected at least two successful compact-tail depths")
    coarse, fine = successful[-2], successful[-1]
    if abs(float(fine["u_0minus"]) / float(coarse["u_0minus"]) - 1.0) > 1.0e-3:
        raise AssertionError(f"u_0minus is not tail-converged: {coarse!r} -> {fine!r}")
    if abs(float(fine["a_0plus"]) - float(coarse["a_0plus"])) > 1.0e-3:
        raise AssertionError(f"a_0plus is not tail-converged: {coarse!r} -> {fine!r}")


def test_thermal_conducting_validates_spline_derivatives_against_rhs():
    """ODE residuals must use spline derivatives, not identities built from the RHS."""
    r = _tc_reference_solution()
    if not r.get("success"):
        raise AssertionError(f"reference thermal solve failed: {r.get('message')!r}")
    limit = 10.0 * float(r["tol_bvp"])
    for key in (
        "nK_ode_residual_norm",
        "jK_ode_residual_norm",
        "T_ode_residual_norm",
    ):
        value = float(r[key])
        if not np.isfinite(value) or value > limit:
            raise AssertionError(f"{key}={value!r} exceeds {limit!r}")


def _tc_fit_exponent(x, y, lo_frac=1.0e-4, hi_frac=1.0e-2):
    L = float(x[-1])
    mask = (x > lo_frac * L) & (x < hi_frac * L) & np.isfinite(y) & (y > 0.0)
    if int(mask.sum()) < 5:
        raise AssertionError(f"not enough near-interface points to fit ({int(mask.sum())})")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def test_thermal_conducting_converges_with_conserved_fluxes_and_boundary_conditions():
    """(6)(7)(8) BCs, conserved fluxes, and interface signs on a converged solve."""
    r = _tc_reference_solution()
    if not r.get("success"):
        raise AssertionError(f"reference thermal solve failed: {r.get('message')!r}")
    if int(r["bvp_status"]) != 0:
        raise AssertionError(f"bvp status {r['bvp_status']} accepted as success")

    residuals = np.asarray(r["bc_residuals"], dtype=float)
    if residuals.size != 4:
        raise AssertionError(f"expected 4 boundary conditions, got {residuals.size}")
    if float(np.max(np.abs(residuals))) > 1.0e-4:
        raise AssertionError(f"boundary-condition residuals too large: {residuals!r}")

    for key in ("max_flux_error_jB", "max_flux_error_Pi", "max_flux_error_F_E"):
        value = float(r[key])
        if not np.isfinite(value) or value > 1.0e-6:
            raise AssertionError(f"{key} = {value!r} exceeds tolerance")
    if float(r["max_closure_residual"]) > 1.0e-7:
        raise AssertionError(f"local closure residual {r['max_closure_residual']!r} too large")

    # interface: T(0) exactly zero, T'(0+) NOT forced to zero, q_th consistent
    if float(r["T_profile"][0]) != 0.0:
        raise AssertionError(f"T(0) = {r['T_profile'][0]!r}, expected exactly 0")
    if not (float(r["Tprime_0plus"]) > 0.0):
        raise AssertionError(f"T'(0+) = {r['Tprime_0plus']!r} must be positive and non-zero")
    if not (float(r["q_th_0plus"]) < 0.0):
        raise AssertionError(f"q_th(0+) = {r['q_th_0plus']!r} must be negative")
    # q_th(0+) = F_E - h*gamma*u|0+ by construction
    flux_gap = float(r["F_E"]) - float(
        r["kappa_th_profile"][0] * r["T_prime_profile"][0] + r["F_E"]
    )
    if abs(flux_gap - float(r["q_th_0plus"])) > 1.0e-6 * max(abs(float(r["q_th_0plus"])), 1.0):
        raise AssertionError(
            f"q_th(0+) inconsistent: reported {r['q_th_0plus']!r} vs reconstructed {flux_gap!r}"
        )
    # a(0+) is an OUTPUT of this formulation
    if not (0.0 < float(r["a_0plus"]) < 1.0):
        raise AssertionError(f"a(0+) = {r['a_0plus']!r} outside (0, 1)")


def test_thermal_conducting_near_interface_power_laws():
    """(9) T ~ x and nK(0)-nK(x) ~ x^(8/3) near the interface."""
    r = _tc_reference_solution()
    if not r.get("success"):
        raise AssertionError(f"reference thermal solve failed: {r.get('message')!r}")
    x = np.asarray(r["x"], dtype=float)
    T = np.asarray(r["T_profile"], dtype=float)
    nK = np.asarray(r["nK_profile"], dtype=float)

    p_T = _tc_fit_exponent(x, T)
    if abs(p_T - 1.0) > 0.15:
        raise AssertionError(f"T exponent {p_T:.4f}, expected ~1 (linear lift-off from T=0)")

    p_nK = _tc_fit_exponent(x, nK[0] - nK)
    if abs(p_nK - 8.0 / 3.0) > 0.35:
        raise AssertionError(
            f"nK departure exponent {p_nK:.4f}, expected ~8/3 = 2.6667; a value near 0 "
            "would indicate the old non-Lipschitz plateau has returned"
        )


def test_thermal_conducting_reports_knudsen_validity_limit():
    """(11) Fourier conduction at exactly T=0 must self-report its LTE breakdown."""
    r = _tc_reference_solution()
    if not r.get("success"):
        raise AssertionError(f"reference thermal solve failed: {r.get('message')!r}")
    if not np.isfinite(float(r["max_knudsen"])):
        raise AssertionError("max_knudsen was not computed")
    if float(r["max_knudsen"]) <= 1.0:
        raise AssertionError(
            "expected the mean free path to diverge at the T=0 interface "
            f"(got max_knudsen = {r['max_knudsen']!r}); the diagnostic is not wired up"
        )
    text = " ".join(r.get("warnings", []))
    if "Knudsen" not in text:
        raise AssertionError(f"Knudsen breakdown not surfaced in warnings: {r.get('warnings')!r}")
    if float(r["max_kappa_y"]) > float(r["conductivity_y_max"]):
        raise AssertionError("solution used the conductivity model outside its validated band")


def test_thermal_conducting_is_independent_of_the_interface_composition_seed():
    """(10) a(0+) is selected by the BVP, so the seed must not change the answer."""
    base = _tc_reference_solution()
    if not base.get("success"):
        raise AssertionError(f"reference thermal solve failed: {base.get('message')!r}")
    other = pv.solve_front_thermal_conducting(
        10.0, 3.0 * N0, 180.0, T_0plus=0.0, a_0plus_guess=0.95
    )
    if not other.get("success"):
        raise AssertionError(f"seed-variant solve failed: {other.get('message')!r}")
    if abs(other["u_0minus"] / base["u_0minus"] - 1.0) > 1.0e-3:
        raise AssertionError(
            "u_0minus depends on the seed: "
            f"{base['u_0minus']!r} vs {other['u_0minus']!r}"
        )
    if abs(float(other["a_0plus"]) - float(base["a_0plus"])) > 1.0e-3:
        raise AssertionError(
            f"a(0+) depends on the seed: {base['a_0plus']!r} vs {other['a_0plus']!r}"
        )


def test_thermal_conducting_rejects_invalid_inputs_explicitly():
    """(11) Bad inputs must fail loudly, not return a plausible-looking profile."""
    for kwargs in (
        {"T_0minus": -1.0, "nB_0minus": 3.0 * N0, "B_one_forth": 180.0},
        {"T_0minus": 10.0, "nB_0minus": -1.0, "B_one_forth": 180.0},
        {"T_0minus": 10.0, "nB_0minus": 3.0 * N0, "B_one_forth": 180.0, "n_mesh": 2},
        {"T_0minus": 10.0, "nB_0minus": 3.0 * N0, "B_one_forth": 180.0, "tail_eps": 2.0},
    ):
        try:
            pv.solve_front_thermal_conducting(**kwargs)
        except RuntimeError:
            continue
        raise AssertionError(f"solve_front_thermal_conducting accepted invalid input {kwargs}")


if __name__ == "__main__":
    tests = sorted(name for name in globals() if name.startswith("test_"))
    for name in tests:
        globals()[name]()
        print(f"{name}: ok")
