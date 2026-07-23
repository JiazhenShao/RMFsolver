import numpy as np
import time
import warnings
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import fsolve, root_scalar, root
import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver.SQMsolver import (
    # Legacy re-exports kept for repo-wide notebook/script compatibility.
    E_f,
    P_f,
    PQM,
    PQM_em,
    edensQM,
    entropyQM,
    n_B,
    nB_QM,
    nK_QM,
    _chiK_QM,
    _quark_uds_state,
    _solve_scalar_root,
)
from RMFsolver.Solver import RMFsolve, RMFsolve_mu, RMFpressureSYM, RMFpressurePNM, pressure_RMF
from RMFsolver.Solver import RMFedensPNM, RMFentropyPNM, RMFbaryon_densityPNM, RMFbaryon_densitySYM, RMFbaryon_density

__all__ = [
    "analytic_velocity_bound",
    "solve_front_isothermal",
    "solve_front_energy_conserving_nK",
    "solve_front_energy_conserving_uNmax",
    "z_time_evolution",
]

_TRANSPORT_ALPHA_S = 0.3
_TRANSPORT_G_S = np.sqrt(4.0 * np.pi * _TRANSPORT_ALPHA_S)
_TRANSPORT_QD_COEFF = np.sqrt(3.0 * _TRANSPORT_G_S**2 / (2.0 * np.pi**2))
_TRANSPORT_D_PREFACTOR = 24.0 * _TRANSPORT_ALPHA_S**2 / np.pi
_TRANSPORT_H_CONST = 1.81317
_FLOAT_TINY = np.finfo(float).tiny
_ISOTHERMAL_RETRY_ACTIVE = 0


class SlowFrontNoSolution(RuntimeError):
    """
    No steadily moving conversion front exists for the given upstream state.

    Raised by analytic_velocity_bound when the metastable neutron matter has a
    lower enthalpy per baryon than the coldest equilibrated (mu_K = 0) quark
    matter reachable on the isobar P = P_N. Energy-plus-baryon flux conservation
    then forces h_Q/n_Q = h_N/n_N with no root, so a finite-speed front cannot
    bridge the gap even though the static u_N = 0 coexistence still exists. This
    is a physical outcome (typically at small metastability and finite
    temperature), not a numerical failure. Callers can inspect ``status``,
    ``gap``, ``muB_cold``, and ``h_over_nB_N``.
    """

    def __init__(self, message, *, gap, muB_cold, h_over_nB_N):
        super().__init__(message)
        self.status = "no_slow_front_solution"
        self.gap = float(gap)
        self.muB_cold = float(muB_cold)
        self.h_over_nB_N = float(h_over_nB_N)


def _slow_front_enthalpy_gap(nuclear_state, B_one_forth):
    """
    Enthalpy-per-baryon gap that blocks a slow conversion front.

    Returns (gap, muB_cold) where muB_cold is the baryon chemical potential of
    cold (T = 0) equilibrated quark matter at pressure P_N, and
    gap = muB_cold - h_N/n_B,N. The cold state minimizes h_Q/n_B,Q along the
    P = P_N isobar (there h_Q/n_B,Q = mu_B), so gap > 0 means no equilibrated
    downstream state satisfies energy-per-baryon continuity and no moving front
    exists.
    """
    P_N = float(nuclear_state["P_N"])
    h_over_nB_N = float(nuclear_state["h_over_nB_N"])
    U_bag = float(B_one_forth) ** 4
    radicand = 108.0 * np.pi**2 * (P_N + U_bag)
    if (not np.isfinite(radicand)) or radicand <= 0.0:
        return float("nan"), float("nan")
    muB_cold = float(radicand**0.25)
    return float(muB_cold - h_over_nB_N), muB_cold


def _raise_scan_failure(nuclear_state, B_one_forth, numerical_message):
    """
    Classify an eigenvalue-scan failure as physical or numerical.

    If the closed-form enthalpy gap is positive the upstream state admits no
    moving front, and a SlowFrontNoSolution is raised with a clear report.
    Otherwise the failure is genuinely numerical and ``numerical_message`` is
    raised as a plain RuntimeError.
    """
    gap, muB_cold = _slow_front_enthalpy_gap(nuclear_state, B_one_forth)
    if np.isfinite(gap) and gap > 0.0:
        h_over_nB_N = float(nuclear_state["h_over_nB_N"])
        raise SlowFrontNoSolution(
            (
                "No steadily moving conversion front exists: the metastable neutron "
                f"matter has enthalpy per baryon {h_over_nB_N:.4f} MeV, below the "
                f"coldest equilibrated quark matter on the P_N isobar at "
                f"{muB_cold:.4f} MeV (gap {gap:.4f} MeV). Energy-plus-baryon flux "
                "conservation has no root, so only the static u_N = 0 coexistence "
                "exists at this metastability and temperature."
            ),
            gap=gap,
            muB_cold=muB_cold,
            h_over_nB_N=h_over_nB_N,
        )
    raise RuntimeError(numerical_message)


# Nuclear and endpoint thermodynamics
def PNM(mu_B, Temp, param = para.paraQMCRMF3, NM_type = "PNM"):
    """
    Universal nuclear-matter pressure helper as a function of mu_B.
    """
    if NM_type == "Beta_eq":
        rmf_sol = RMFsolve_mu(mub=mu_B, Trmf=Temp, para=param, sigma_init=30, w0_init=20, r03_init=-3, mu_e_init=50, verb=False,)
        return float(np.asarray(pressure_RMF(rmf_sol)).item())

    if NM_type == "PNM":
        pre = RMFpressurePNM(input_num=mu_B, input_type="muB", Trmf=Temp, para=param, sigma_init=30, w0_init=20, r03_init=-3, mub_init=990, verb=False,)
        return float(np.asarray(pre).item())

    if NM_type == "SYM":
        pre = RMFpressureSYM(input_num=mu_B,input_type="muB", Trmf=Temp, para=param, sigma_init=30, w0_init=20, mub_init=990, verb=False,)
        return float(np.asarray(pre).item())

    raise ValueError("Nuclear matter type not defined.")

def PNM_n(nB, Temp, param = para.paraQMCRMF3, NM_type = "PNM"):
    """
    Universal nuclear-matter pressure helper as a function of n_B.
    """
    if NM_type == "Beta_eq":
        rmf_sol = RMFsolve(nbext=nB, Trmf=Temp, para=param, sigma_init=30, w0_init=20, r03_init=-3, mu_e_init=50, verb=False,)
        return float(np.asarray(pressure_RMF(rmf_sol)).item())

    if NM_type == "PNM":
        pre = RMFpressurePNM(input_num=nB, input_type="nB", Trmf=Temp, para=param, sigma_init=30, w0_init=20, r03_init=-3, mub_init=990, verb=False,)
        return float(np.asarray(pre).item())

    if NM_type == "SYM":
        pre = RMFpressureSYM(input_num=nB, input_type="nB", Trmf=Temp, para=param, sigma_init=30, w0_init=20, mub_init=990, verb=False,)
        return float(np.asarray(pre).item())

    raise ValueError("Nuclear matter type not defined.")

def edensNM(mu_B, Temp, param = para.paraQMCRMF3, ):
    """
    Return the nuclear-matter energy density at fixed mu_B.
    """
    edens = RMFedensPNM(input_num = mu_B, input_type = "muB", Trmf = Temp, para = param, 
        sigma_init = 30, w0_init = 20, r03_init = -3, mub_init = 990, verb = False
        )
    return float(edens.item())

def edensNM_n(nB, Temp, param = para.paraQMCRMF3, ):
    """
    Return the nuclear-matter energy density at fixed n_B.
    """
    edens = RMFedensPNM(input_num = nB, input_type = "nB", Trmf = Temp, para = param, 
        sigma_init = 30, w0_init = 20, r03_init = -3, mub_init = 990, verb = False
        )
    return float(edens.item())

def sNM_n(nB, Temp, param = para.paraQMCRMF3, NM_type = "PNM"):
    """
    Return the nuclear-matter entropy density at fixed n_B.
    """
    if NM_type != "PNM":
        raise RuntimeError("sNM_n currently supports NM_type='PNM' only")
    entropy = RMFentropyPNM(input_num=nB, input_type="nB", Trmf=Temp, para=param,
        sigma_init=30, w0_init=20, r03_init=-3, mub_init=990, verb=False,
        electrons=False, neutrinos=False,
        )
    return float(np.asarray(entropy).item())

def hNM(mu_B, Temp):
    """
    Return the nuclear-matter enthalpy density at fixed mu_B.
    """
    return PNM(mu_B, Temp) + edensNM(mu_B, Temp)

def hNM_n(nB, Temp):
    """
    Return the nuclear-matter enthalpy density at fixed n_B.
    """
    return PNM_n(nB, Temp) + edensNM_n(nB, Temp)

def nB_NM(mu_B, Temp, param = para.paraQMCRMF3, NM_type = "PNM"):
    """
    Universal nuclear-matter baryon density helper as a function of mu_B.
    """
    if NM_type == "Beta_eq":
        nB = RMFbaryon_density(input_num=mu_B, input_type="muB", Trmf=Temp, para=param,
            sigma_init=30, w0_init=20, r03_init=-3, mub_init=990, mue_init=50, verb=False,
        )
        return float(np.asarray(nB).item())

    if NM_type == "PNM":
        nB = RMFbaryon_densityPNM(input_num=mu_B, input_type="muB", Trmf=Temp, para=param,
            sigma_init=30, w0_init=20, r03_init=-3, mub_init=990, verb=False,
        )
        return float(np.asarray(nB).item())

    if NM_type == "SYM":
        nB = RMFbaryon_densitySYM(input_num=mu_B, input_type="muB", Trmf=Temp, para=param,
            sigma_init=30, w0_init=20, mub_init=990, verb=False,
        )
        return float(np.asarray(nB).item())

    raise ValueError("Nuclear matter type not defined.")

def _weak_rate_from_mu_q(mu_q):
    """
    Return the nonleptonic weak-rate gamma and corresponding tau.

    gamma is evaluated in natural units (MeV), so tau is returned in MeV^-1.
    tau_seconds is included as a diagnostic conversion using hbar = MeV_sec.
    """
    mu_q = float(mu_q)
    if (not np.isfinite(mu_q)) or mu_q <= 0.0:
        raise RuntimeError("Weak-rate coefficient requires mu_q > 0")

    gamma = (
        (128.0 / 27.0)
        * (1.0 / (5.0 * np.pi**3))
        * const.G_Fermi**2
        * np.cos(const.Cabibbo_angle_rad) ** 2
        * np.sin(const.Cabibbo_angle_rad) ** 2
        * mu_q**5
    )
    if (not np.isfinite(gamma)) or gamma <= 0.0:
        raise RuntimeError("Weak-rate coefficient is non-physical")

    tau = 1.0 / gamma
    tau_seconds = tau * const.MeV_sec
    if (
        (not np.isfinite(tau))
        or (not np.isfinite(tau_seconds))
        or tau <= 0.0
        or tau_seconds <= 0.0
    ):
        raise RuntimeError("Weak-rate timescale is non-physical")

    return {
        "gamma": float(gamma),
        "tau": float(tau),
        "tau_seconds": float(tau_seconds),
    }


def _analytic_weak_rate_from_mu_q(mu_q):
    """
    Backward-compatible alias for the canonical nonleptonic weak-rate helper.
    """
    return _weak_rate_from_mu_q(mu_q)


def _exact_kaon_transport_rate(muB, muK, T, ms=0.0, upB=5000):
    """Return the positive kaon relaxation rate Gamma_K."""
    muB = float(muB)
    muK = float(muK)
    T = float(T)
    if (not np.isfinite(muB)) or muB <= 0.0:
        raise RuntimeError("Exact kaon rate requires muB > 0")
    if (not np.isfinite(muK)) or (not np.isfinite(T)) or T < 0.0:
        raise RuntimeError("Exact kaon rate requires finite muK and T >= 0")

    quark_state = _quark_uds_state(muB, muK, T, ms=ms, upB=upB)
    chemical_potentials = quark_state["chemical_potentials"]
    mu_u = float(chemical_potentials["mu_u"])
    mu_d = float(chemical_potentials["mu_d"])
    mu_s = float(chemical_potentials["mu_s"])
    if not np.isclose(mu_d - mu_s, muK, rtol=1.0e-12, atol=1.0e-10):
        raise RuntimeError("Quark EOS chemical potentials are inconsistent with muK")

    prefactor = (
        16.0
        / (5.0 * np.pi**5)
        * const.G_Fermi**2
        * np.sin(const.Cabibbo_angle_rad) ** 2
        * np.cos(const.Cabibbo_angle_rad) ** 2
    )
    Gamma_K = float(prefactor * mu_u**5 * muK * (muK**2 + 4.0 * np.pi**2 * T**2))
    if not np.isfinite(Gamma_K):
        raise RuntimeError("Exact kaon rate is non-finite")
    return {
        "Gamma_K": Gamma_K,
        "mu_u": mu_u,
        "mu_d": mu_d,
        "mu_s": mu_s,
    }


def _normalize_interface_fraction_mode(mode_value, parameter_name="interface_fraction_mode"):
    """
    Normalize the analytic interface-fraction mode.
    """
    mode = str(mode_value).strip().lower()
    if mode == "lte":
        return "LTE"
    if mode == "extreme_endothermic":
        return "extreme_endothermic"
    raise ValueError(f"{parameter_name} must be 'LTE' or 'extreme_endothermic'")


def _analytic_aqstar_lte(A_boundary, u_N, lambda_n):
    """
    Return the exact LTE-limited interface fraction and its quadratic diagnostics.
    """
    A_boundary = float(A_boundary)
    u_N = float(u_N)
    lambda_n = float(lambda_n)
    if (not np.isfinite(A_boundary)) or A_boundary <= 0.0:
        raise RuntimeError("A_boundary must be positive and finite")
    if (not np.isfinite(u_N)) or u_N < 0.0:
        raise RuntimeError("LTE a_interface requires finite u_N >= 0")
    if (not np.isfinite(lambda_n)) or lambda_n <= 0.0:
        raise RuntimeError("LTE a_interface requires finite lambda_n > 0")

    beta = float(5.0 * u_N * lambda_n)
    discriminant = float(
        beta * beta + 4.0 * (1.0 - beta) * A_boundary * A_boundary
    )
    discriminant_scale = max(
        beta * beta,
        4.0 * abs(1.0 - beta) * A_boundary * A_boundary,
        1.0,
    )
    if discriminant < -1.0e-14 * discriminant_scale:
        raise RuntimeError("LTE a_interface discriminant is negative")
    discriminant = max(discriminant, 0.0)
    stable_denominator = float(np.sqrt(discriminant) + beta)
    if (not np.isfinite(stable_denominator)) or stable_denominator <= 0.0:
        raise RuntimeError("LTE a_interface denominator is non-physical")

    a_interface_LTE = float(
        2.0 * A_boundary * A_boundary / stable_denominator
    )
    if (not np.isfinite(a_interface_LTE)) or a_interface_LTE <= 0.0:
        raise RuntimeError("a_interface_LTE must be positive and finite")

    return {
        "A_boundary": A_boundary,
        "a_interface_LTE": a_interface_LTE,
        "aQstar_LTE": a_interface_LTE,
        "lambda_n": lambda_n,
        "beta_LTE": beta,
        "lte_discriminant": discriminant,
    }


def _analytic_nuclear_state(muB_N, T_N, param=para.paraQMCRMF3, NM_type="PNM"):
    """
    Return the upstream nuclear state used by analytic_velocity_bound.
    """
    P_N = float(PNM(muB_N, T_N, param=param, NM_type=NM_type))
    e_N = float(edensNM(muB_N, T_N, param=param))
    h_N = float(P_N + e_N)
    nB_N = float(nB_NM(muB_N, T_N, param=param, NM_type=NM_type))
    if (not np.isfinite(P_N)) or (not np.isfinite(e_N)) or (not np.isfinite(h_N)):
        raise RuntimeError("Nuclear EOS returned non-finite pressure or enthalpy")
    if (not np.isfinite(nB_N)) or nB_N <= 0.0:
        raise RuntimeError("nB_N must be positive and finite")
    if h_N <= 0.0:
        raise RuntimeError("h_N must be positive")
    h_over_nB_N = float(h_N / nB_N)
    if (not np.isfinite(h_over_nB_N)) or h_over_nB_N <= 0.0:
        raise RuntimeError("Nuclear h_N/nB_N must be positive and finite")
    return {
        "P_N": P_N,
        "e_N": e_N,
        "h_N": h_N,
        "nB_N": nB_N,
        "h_over_nB_N": h_over_nB_N,
    }


def _append_analytic_endpoint_guess(guesses, muB_guess, T_guess):
    try:
        muB_guess = float(muB_guess)
        T_guess = float(T_guess)
    except Exception:
        return
    if (
        np.isfinite(muB_guess)
        and np.isfinite(T_guess)
        and muB_guess > 0.0
        and T_guess > 0.0
    ):
        candidate = (muB_guess, T_guess)
        if candidate not in guesses:
            guesses.append(candidate)


def _solve_analytic_downstream_endpoint_for_uN(
    u_N,
    nuclear_state,
    B_one_forth,
    ms=0.0,
    upB=5000,
    initial_guess=None,
):
    """
    Solve the muK=0 downstream quark endpoint from exact hydro jump conditions.
    """
    u_N = float(u_N)
    if (not np.isfinite(u_N)) or u_N <= 0.0:
        raise RuntimeError("Trial u_N must be positive and finite")

    P_N = float(nuclear_state["P_N"])
    h_N = float(nuclear_state["h_N"])
    nB_N = float(nuclear_state["nB_N"])
    h_over_nB_N = float(nuclear_state["h_over_nB_N"])
    jB = float(nB_N * u_N)
    gamma_N = float(np.sqrt(1.0 + u_N * u_N))
    energy_flux_N = float(h_N * u_N * gamma_N)
    momentum_flux_N = float(P_N + h_N * u_N * u_N)
    energy_target = float(h_over_nB_N * gamma_N)

    guesses = []
    if initial_guess is not None:
        guess_arr = np.asarray(initial_guess, dtype=float).ravel()
        if guess_arr.size >= 2:
            _append_analytic_endpoint_guess(guesses, guess_arr[0], guess_arr[1])
    for muB_guess in (
        float(nuclear_state.get("muB_N", 0.0)),
        900.0,
        1100.0,
        1300.0,
        1500.0,
        700.0,
    ):
        for T_guess in (
            float(nuclear_state.get("T_N", 0.0)),
            max(float(nuclear_state.get("T_N", 0.0)), 1.0),
            10.0,
            30.0,
            60.0,
            100.0,
        ):
            _append_analytic_endpoint_guess(guesses, muB_guess, T_guess)

    def equations(vec):
        muB = float(vec[0])
        logT = float(vec[1])
        if (
            (not np.isfinite(muB))
            or muB <= 0.0
            or (not np.isfinite(logT))
            or abs(logT) > 700.0
        ):
            return np.array([1.0e30, 1.0e30], dtype=float)
        T_Q = float(np.exp(logT))
        try:
            P_Q = float(PQM(muB, 0.0, B_one_forth, T_Q, ms=ms, upB=upB))
            e_Q = float(
                edensQM(
                    muB,
                    0.0,
                    B_one_forth,
                    T_Q,
                    ms=ms,
                    include_em=False,
                    upB=upB,
                )
            )
            nB_Q = float(nB_QM(muB, 0.0, B_one_forth, T_Q, ms=ms, upB=upB))
        except Exception:
            return np.array([1.0e30, 1.0e30], dtype=float)
        if (
            (not np.isfinite(P_Q))
            or (not np.isfinite(e_Q))
            or (not np.isfinite(nB_Q))
            or nB_Q <= 0.0
        ):
            return np.array([1.0e30, 1.0e30], dtype=float)
        h_Q = float(P_Q + e_Q)
        u_Q = float(jB / nB_Q)
        gamma_Q = float(np.sqrt(1.0 + u_Q * u_Q))
        energy_flux_Q = float(h_Q * u_Q * gamma_Q)
        momentum_flux_Q = float(P_Q + h_Q * u_Q * u_Q)
        if (
            (not np.isfinite(h_Q))
            or h_Q <= 0.0
            or (not np.isfinite(energy_flux_Q))
            or (not np.isfinite(momentum_flux_Q))
        ):
            return np.array([1.0e30, 1.0e30], dtype=float)
        energy_residual = float(h_Q * gamma_Q / nB_Q - energy_target)
        pressure_jump = float(P_Q - P_N)
        pressure_jump_balance = float(h_N * u_N * u_N - h_Q * u_Q * u_Q)
        # Scale the momentum residual by the pressure, not by the momentum flux
        # h*u^2. For slow fronts h*u^2 -> 0 and flooring at 1.0 left this
        # residual as an absolute value in MeV^4 (~1e7), mis-scaled against the
        # relative energy residual, which stalled the hybr solve. The pressure
        # scale keeps both residuals relative and comparable.
        momentum_scale = max(abs(P_N), abs(P_Q), 1.0)
        return np.array(
            [
                energy_residual / max(abs(energy_target), _FLOAT_TINY),
                (pressure_jump - pressure_jump_balance) / momentum_scale,
            ],
            dtype=float,
        )

    best = None
    best_norm = np.inf
    best_message = "analytic hydro endpoint solve did not converge"
    for muB_guess, T_guess in guesses:
        sol = root(
            equations,
            np.array([float(muB_guess), float(np.log(T_guess))], dtype=float),
            method="hybr",
            options={"maxfev": 1600, "xtol": 1.0e-10},
        )
        if np.all(np.isfinite(sol.x)):
            residual = equations(sol.x)
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            if residual_norm < best_norm:
                best_norm = residual_norm
                best = sol.x.copy()
        if sol.success and np.all(np.isfinite(sol.x)) and best_norm <= 1.0e-6:
            break
        best_message = str(sol.message)

    if best is None or best_norm > 1.0e-6:
        raise RuntimeError(f"{best_message}; best scaled residual={best_norm:.3e}")

    muB_Q = float(best[0])
    T_Q = float(np.exp(float(best[1])))
    if (not np.isfinite(muB_Q)) or muB_Q <= 0.0 or (not np.isfinite(T_Q)) or T_Q <= 0.0:
        raise RuntimeError("Analytic hydro endpoint solve returned a non-physical root")

    P_Q = float(PQM(muB_Q, 0.0, B_one_forth, T_Q, ms=ms, upB=upB))
    e_Q = float(
        edensQM(
            muB_Q,
            0.0,
            B_one_forth,
            T_Q,
            ms=ms,
            include_em=False,
            upB=upB,
        )
    )
    h_Q = float(P_Q + e_Q)
    nB_Q = float(nB_QM(muB_Q, 0.0, B_one_forth, T_Q, ms=ms, upB=upB))
    if (
        (not np.isfinite(P_Q))
        or (not np.isfinite(e_Q))
        or (not np.isfinite(h_Q))
        or h_Q <= 0.0
        or (not np.isfinite(nB_Q))
        or nB_Q <= 0.0
    ):
        raise RuntimeError("Hydro endpoint quark EOS returned non-physical quantities")

    u_Q = float(jB / nB_Q)
    gamma_Q = float(np.sqrt(1.0 + u_Q * u_Q))
    energy_flux_Q = float(h_Q * u_Q * gamma_Q)
    momentum_flux_Q = float(P_Q + h_Q * u_Q * u_Q)
    energy_flux_residual = float(energy_flux_Q - energy_flux_N)
    momentum_flux_residual = float(momentum_flux_Q - momentum_flux_N)
    pressure_jump = float(P_Q - P_N)
    pressure_jump_balance = float(h_N * u_N * u_N - h_Q * u_Q * u_Q)

    return {
        "muB_Q": muB_Q,
        "T_Q": T_Q,
        "P_Q": P_Q,
        "e_Q": e_Q,
        "h_Q": h_Q,
        "nB_Q": nB_Q,
        "h_over_nB_Q": float(h_Q / nB_Q),
        "u_N": u_N,
        "u_Q": u_Q,
        "jB": jB,
        "gamma_N": gamma_N,
        "gamma_Q": gamma_Q,
        "energy_flux_N": energy_flux_N,
        "energy_flux_Q": energy_flux_Q,
        "momentum_flux_N": momentum_flux_N,
        "momentum_flux_Q": momentum_flux_Q,
        "energy_flux_residual": energy_flux_residual,
        "momentum_flux_residual": momentum_flux_residual,
        "pressure_jump": pressure_jump,
        "pressure_jump_balance": pressure_jump_balance,
        "pressure_jump_residual": float(pressure_jump - pressure_jump_balance),
        "endpoint_scaled_residual": best_norm,
        "endpoint_initial_guess": (muB_Q, T_Q),
        "h_over_nB_N": h_over_nB_N,
        "U_bag": float(B_one_forth) ** 4,
    }


def _analytic_velocity_formula_from_endpoint(
    endpoint,
    nuclear_state,
    xi,
    aQstar_max_mode="LTE",
    interface_fraction_mode=None,
):
    """
    Evaluate the selected analytic velocity formula from a hydro endpoint.
    """
    xi = float(xi)
    if interface_fraction_mode is None:
        interface_fraction_mode = _normalize_interface_fraction_mode(
            aQstar_max_mode,
            parameter_name="aQstar_max",
        )
    else:
        interface_fraction_mode = _normalize_interface_fraction_mode(
            interface_fraction_mode,
            parameter_name="interface_fraction_mode",
        )
    muB_Q = float(endpoint["muB_Q"])
    T_Q = float(endpoint["T_Q"])
    nB_Q = float(endpoint["nB_Q"])
    u_N = float(endpoint["u_N"])
    nB_N = float(nuclear_state["nB_N"])

    mu_q = float(muB_Q / 3.0)
    weak_rate = _analytic_weak_rate_from_mu_q(mu_q)
    gamma = float(weak_rate["gamma"])
    tau = float(weak_rate["tau"])
    tau_seconds = float(weak_rate["tau_seconds"])

    lambda_n = float(nB_N / nB_Q)
    if (not np.isfinite(lambda_n)) or lambda_n <= 0.0:
        raise RuntimeError("lambda_n must be positive and finite")

    # Interface-fraction ceiling A, the T = 0 endpoint of the constant-pressure
    # isobar defined by P_QM(A, T = 0) = P_Q. Using the quadratic quark EoS with
    # mu_K = 2 mu_B a / 9 this is the exact closed form
    #   A = (3/2) sqrt(108 pi^2 (P_Q + U_bag) - mu_B^4) / mu_B^2 ,
    # which matches the full-EoS root to <1%. The older leading-order form
    # A = 9 pi T_Q / (sqrt(2) mu_B_Q) is 3/2 too small relative to the intended
    # definition and is kept only as a fallback when U_bag is unavailable.
    U_bag = endpoint.get("U_bag", None)
    P_Q = float(endpoint.get("P_Q", np.nan))
    if U_bag is not None and np.isfinite(float(U_bag)) and np.isfinite(P_Q):
        A_radicand = float(108.0 * np.pi**2 * (P_Q + float(U_bag)) - muB_Q**4)
        A_boundary = float(1.5 * np.sqrt(max(A_radicand, 0.0)) / (muB_Q * muB_Q))
    else:
        A_boundary = float((9.0 * np.pi / np.sqrt(2.0)) * (T_Q / muB_Q))
    muKstar_max = float(np.sqrt(2.0) * np.pi * T_Q)
    one_minus_A_boundary = float(1.0 - A_boundary)
    one_plus_xi_A_boundary = float(1.0 + xi * A_boundary)
    lambda_n_squared = float(lambda_n * lambda_n)
    if (not np.isfinite(A_boundary)) or A_boundary <= 0.0:
        raise RuntimeError("A_boundary must be positive and finite")

    alpha_s = float(_TRANSPORT_ALPHA_S)
    h_D = float(_TRANSPORT_H_CONST)
    beta_LTE = np.nan
    a_interface_LTE = np.nan
    lte_correction = np.nan

    if interface_fraction_mode == "extreme_endothermic":
        if A_boundary >= 1.0:
            raise RuntimeError("A_boundary must satisfy 0 < A_boundary < 1")
        if (not np.isfinite(one_plus_xi_A_boundary)) or one_plus_xi_A_boundary <= 0.0:
            raise RuntimeError("1 + xi*A_boundary must be positive")
        a_interface = A_boundary
        one_minus_a_interface = one_minus_A_boundary
        one_plus_xi_a_interface = one_plus_xi_A_boundary
        denominator = float(
            muB_Q
            * lambda_n_squared
            * one_minus_a_interface
            * one_plus_xi_a_interface
        )
        if (not np.isfinite(denominator)) or denominator <= 0.0:
            raise RuntimeError("Analytic velocity denominator is non-physical")
        prefactor = float(
            (54.0 * np.pi ** (7.0 / 3.0) * gamma)
            / (7.0 * np.sqrt(2.0) * h_D * alpha_s ** (5.0 / 3.0))
        )
        u_N_formula_squared = float(
            prefactor * A_boundary ** (7.0 / 3.0) / denominator
        )
    else:
        lte_data = _analytic_aqstar_lte(A_boundary, u_N, lambda_n)
        beta_LTE = float(lte_data["beta_LTE"])
        a_interface_LTE = float(lte_data["a_interface_LTE"])
        if a_interface_LTE >= 1.0:
            raise RuntimeError(
                "a_interface_LTE must satisfy 0 < a_interface_LTE < 1"
            )
        a_interface = a_interface_LTE
        one_minus_a_interface = float(1.0 - a_interface)
        one_plus_xi_a_interface = float(1.0 + xi * a_interface)
        if (
            (not np.isfinite(one_minus_a_interface))
            or one_minus_a_interface <= 0.0
            or (not np.isfinite(one_plus_xi_a_interface))
            or one_plus_xi_a_interface <= 0.0
        ):
            raise RuntimeError("LTE analytic velocity denominator is non-physical")

        z_raw = float(1.0 - (a_interface_LTE / A_boundary) ** 2)
        if z_raw < -1.0e-12 or z_raw > 1.0 + 1.0e-12:
            raise RuntimeError("LTE correction argument must satisfy 0 <= z <= 1")
        z = float(np.clip(z_raw, 0.0, 1.0))
        lte_correction = float(
            24.0 / 7.0
            - 3.0 * z ** (1.0 / 6.0)
            - (3.0 / 7.0) * z ** (7.0 / 6.0)
        )
        if (not np.isfinite(lte_correction)) or lte_correction < 0.0:
            raise RuntimeError("LTE velocity correction is non-physical")

        denominator = float(
            muB_Q
            * lambda_n_squared
            * one_minus_a_interface
            * one_plus_xi_a_interface
        )
        if (not np.isfinite(denominator)) or denominator <= 0.0:
            raise RuntimeError("LTE analytic velocity denominator is non-physical")
        prefactor = float(
            (9.0 * np.pi ** (7.0 / 3.0) * gamma)
            / (4.0 * np.sqrt(2.0) * h_D * alpha_s ** (5.0 / 3.0))
        )
        u_N_formula_squared = float(
            prefactor
            * A_boundary ** (7.0 / 3.0)
            * lte_correction
            / denominator
        )

    if (not np.isfinite(u_N_formula_squared)) or u_N_formula_squared < 0.0:
        raise RuntimeError("Analytic velocity bound produced non-physical u_N^2")
    return {
        "u_N_formula_squared": u_N_formula_squared,
        "mu_q": mu_q,
        "lambda_n": lambda_n,
        "lambda_n_squared": lambda_n_squared,
        "A_boundary": A_boundary,
        "a_interface": a_interface,
        "a_interface_LTE": a_interface_LTE,
        "interface_fraction_mode": interface_fraction_mode,
        "a_N": lambda_n,
        "aQstar_max": A_boundary,
        "aQstar_max_mode": interface_fraction_mode,
        "A_extreme_endothermic": A_boundary,
        "aQstar_LTE": a_interface_LTE,
        "aQstar_used": a_interface,
        "beta_LTE": beta_LTE,
        "lte_correction": lte_correction,
        "muKstar_max": muKstar_max,
        "one_minus_A_boundary": one_minus_A_boundary,
        "one_plus_xi_A_boundary": one_plus_xi_A_boundary,
        "one_minus_a_interface": one_minus_a_interface,
        "one_plus_xi_a_interface": one_plus_xi_a_interface,
        "one_minus_aQstar": one_minus_A_boundary,
        "one_plus_xi_aQstar": one_plus_xi_A_boundary,
        "one_minus_aQstar_used": one_minus_a_interface,
        "one_plus_xi_aQstar_used": one_plus_xi_a_interface,
        "a_N_squared": lambda_n_squared,
        "alpha_s": alpha_s,
        "h_D": h_D,
        "gamma": gamma,
        "tau": tau,
        "tau_seconds": tau_seconds,
        "prefactor": prefactor,
        "analytic_denominator": denominator,
    }


def analytic_velocity_bound(
    muB_N,
    T_N,
    B_one_forth,
    xi=0.0,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    upB=5000,
    initial_guess=None,
    aQstar_max="extreme_endothermic",
    interface_fraction_mode=None,
):
    """
    Evaluate the hydro-consistent analytic upper bound for u_N.

    The nuclear state is read from the EOS at (muB_N, T_N). The downstream
    quark endpoint is solved at muK = 0 from exact energy and momentum flux
    jumps for each trial u_N, with jB = nB_N*u_N derived internally. Equation
    51 then supplies a scalar eigenvalue condition for u_N. The default
    extreme_endothermic mode sets the interface fraction to its ceiling
    a(0+) = A, giving the maximum-speed bound directly; pass
    interface_fraction_mode="LTE" (or aQstar_max="LTE") for the LTE-limited
    interface fraction. The legacy aQstar_max keyword still selects the mode
    when interface_fraction_mode is not supplied.

    Raises SlowFrontNoSolution when no steadily moving front exists (the
    small-metastability, finite-temperature gap): the message and its gap
    attribute report the enthalpy-per-baryon deficit that blocks the front.
    """
    muB_N = float(muB_N)
    T_N = float(T_N)
    B_one_forth = float(B_one_forth)
    xi = float(xi)
    if interface_fraction_mode is None:
        interface_fraction_mode = _normalize_interface_fraction_mode(
            aQstar_max,
            parameter_name="aQstar_max",
        )
    else:
        interface_fraction_mode = _normalize_interface_fraction_mode(
            interface_fraction_mode,
            parameter_name="interface_fraction_mode",
        )

    if (not np.isfinite(muB_N)) or muB_N <= 0.0:
        raise RuntimeError("muB_N must be positive and finite")
    if (not np.isfinite(T_N)) or T_N <= 0.0:
        raise RuntimeError("T_N must be positive and finite")
    if (not np.isfinite(B_one_forth)) or B_one_forth <= 0.0:
        raise RuntimeError("B_one_forth must be positive and finite")
    if (not np.isfinite(xi)) or not (-1.0 < xi < 1.0):
        raise RuntimeError("xi must satisfy -1 < xi < 1")

    nuclear_state = _analytic_nuclear_state(muB_N, T_N, param=param, NM_type=NM_type)
    nuclear_state["muB_N"] = muB_N
    nuclear_state["T_N"] = T_N

    endpoint_guess_cache = {"value": initial_guess}
    best_eval = {
        "theta": np.nan,
        "residual": np.inf,
        "data": None,
        "message": "",
        "endpoint_domain_message": "",
    }

    def evaluate_log_u(theta):
        theta = float(theta)
        u_N = float(np.exp(np.clip(theta, -700.0, np.log(0.999999))))
        endpoint = _solve_analytic_downstream_endpoint_for_uN(
            u_N,
            nuclear_state,
            B_one_forth,
            ms=ms,
            upB=upB,
            initial_guess=endpoint_guess_cache["value"],
        )
        endpoint_guess_cache["value"] = endpoint["endpoint_initial_guess"]
        formula = _analytic_velocity_formula_from_endpoint(
            endpoint,
            nuclear_state,
            xi,
            interface_fraction_mode=interface_fraction_mode,
        )
        residual = float(formula["u_N_formula_squared"] - u_N * u_N)
        data = {**endpoint, **formula, "residual": residual, "theta": theta}
        abs_residual = abs(residual)
        if abs_residual < abs(float(best_eval["residual"])):
            best_eval.update({"theta": theta, "residual": residual, "data": data, "message": ""})
        return residual, data

    valid_scan = []
    scan_thetas = np.linspace(np.log(1.0e-12), np.log(0.99), 72)
    for theta in scan_thetas:
        try:
            residual, data = evaluate_log_u(theta)
        except Exception as exc:
            message = str(exc)
            best_eval["message"] = message
            if (
                "a_interface_LTE must satisfy 0 < a_interface_LTE < 1" in message
                or "A_boundary must satisfy 0 < A_boundary < 1" in message
            ):
                best_eval["endpoint_domain_message"] = message
            continue
        if np.isfinite(residual):
            valid_scan.append((float(theta), float(residual), data))

    if not valid_scan:
        message = (
            best_eval["endpoint_domain_message"]
            or best_eval["message"]
            or "no valid hydro endpoint evaluations"
        )
        _raise_scan_failure(
            nuclear_state,
            B_one_forth,
            f"Analytic velocity eigenvalue scan failed: {message}",
        )

    bracket = None
    exact_data = None
    residual_tol = 1.0e-18
    prev_theta, prev_residual, prev_data = valid_scan[0]
    if abs(prev_residual) <= residual_tol:
        exact_data = prev_data
    else:
        for theta, residual, data in valid_scan[1:]:
            if abs(residual) <= residual_tol:
                exact_data = data
                break
            if prev_residual * residual < 0.0:
                bracket = (prev_theta, theta)
                break
            prev_theta, prev_residual, prev_data = theta, residual, data

    if exact_data is not None:
        final_data = exact_data
    elif bracket is not None:
        def scalar_residual(theta):
            residual, _ = evaluate_log_u(theta)
            return residual

        sol = root_scalar(scalar_residual, bracket=bracket, method="brentq", xtol=1.0e-12, rtol=1.0e-12)
        if not sol.converged:
            raise RuntimeError(f"Analytic velocity eigenvalue solve did not converge: {sol.flag}")
        _, final_data = evaluate_log_u(float(sol.root))
    else:
        best = best_eval["data"]
        best_msg = (
            f"; best theta={best_eval['theta']:.6g}, best residual={best_eval['residual']:.6e}"
            if best is not None
            else ""
        )
        _raise_scan_failure(
            nuclear_state,
            B_one_forth,
            f"Analytic velocity eigenvalue solve found no sign change{best_msg}",
        )

    u_N_max = float(final_data["u_N"])
    u_N_squared = float(u_N_max * u_N_max)
    formula_residual = float(final_data["u_N_formula_squared"] - u_N_squared)
    formula_scale = max(abs(float(final_data["u_N_formula_squared"])), abs(u_N_squared), 1.0)
    if abs(formula_residual) > 1.0e-8 * formula_scale:
        raise RuntimeError(f"Analytic velocity closure residual is too large: {formula_residual:.6e}")

    return {
        "success": True,
        "message": "hydro-consistent analytic velocity bound evaluated",
        "u_N_max": u_N_max,
        "u_N": u_N_max,
        "u_N_squared": u_N_squared,
        "u_N_formula_squared": float(final_data["u_N_formula_squared"]),
        "analytic_velocity_residual": formula_residual,
        "jB": float(final_data["jB"]),
        "muB_N": muB_N,
        "T_N": T_N,
        "P_N": float(nuclear_state["P_N"]),
        "e_N": float(nuclear_state["e_N"]),
        "h_N": float(nuclear_state["h_N"]),
        "nB_N": float(nuclear_state["nB_N"]),
        "h_over_nB_N": float(nuclear_state["h_over_nB_N"]),
        "muB_bar": float(final_data["muB_Q"]),
        "muB_Q": float(final_data["muB_Q"]),
        "T_Q": float(final_data["T_Q"]),
        "P_Q": float(final_data["P_Q"]),
        "e_Q": float(final_data["e_Q"]),
        "h_Q": float(final_data["h_Q"]),
        "nB_Q": float(final_data["nB_Q"]),
        "h_over_nB_Q": float(final_data["h_over_nB_Q"]),
        "u_Q": float(final_data["u_Q"]),
        "gamma_N": float(final_data["gamma_N"]),
        "gamma_Q": float(final_data["gamma_Q"]),
        "energy_flux_N": float(final_data["energy_flux_N"]),
        "energy_flux_Q": float(final_data["energy_flux_Q"]),
        "momentum_flux_N": float(final_data["momentum_flux_N"]),
        "momentum_flux_Q": float(final_data["momentum_flux_Q"]),
        "energy_flux_residual": float(final_data["energy_flux_residual"]),
        "momentum_flux_residual": float(final_data["momentum_flux_residual"]),
        "pressure_jump": float(final_data["pressure_jump"]),
        "pressure_jump_balance": float(final_data["pressure_jump_balance"]),
        "pressure_jump_residual": float(final_data["pressure_jump_residual"]),
        "endpoint_scaled_residual": float(final_data["endpoint_scaled_residual"]),
        "mu_q": float(final_data["mu_q"]),
        "lambda_n": float(final_data["lambda_n"]),
        "lambda_n_squared": float(final_data["lambda_n_squared"]),
        "A_boundary": float(final_data["A_boundary"]),
        "a_interface": float(final_data["a_interface"]),
        "a_interface_LTE": float(final_data["a_interface_LTE"]),
        "interface_fraction_mode": str(final_data["interface_fraction_mode"]),
        "a_N": float(final_data["a_N"]),
        "aQstar_max": float(final_data["aQstar_max"]),
        "aQstar_max_mode": str(final_data["aQstar_max_mode"]),
        "A_extreme_endothermic": float(final_data["A_extreme_endothermic"]),
        "aQstar_LTE": float(final_data["aQstar_LTE"]),
        "aQstar_used": float(final_data["aQstar_used"]),
        "beta_LTE": float(final_data["beta_LTE"]),
        "lte_correction": float(final_data["lte_correction"]),
        "muKstar_max": float(final_data["muKstar_max"]),
        "one_minus_A_boundary": float(final_data["one_minus_A_boundary"]),
        "one_plus_xi_A_boundary": float(final_data["one_plus_xi_A_boundary"]),
        "one_minus_a_interface": float(final_data["one_minus_a_interface"]),
        "one_plus_xi_a_interface": float(final_data["one_plus_xi_a_interface"]),
        "one_minus_aQstar": float(final_data["one_minus_aQstar"]),
        "one_plus_xi_aQstar": float(final_data["one_plus_xi_aQstar"]),
        "one_minus_aQstar_used": float(final_data["one_minus_aQstar_used"]),
        "one_plus_xi_aQstar_used": float(final_data["one_plus_xi_aQstar_used"]),
        "a_N_squared": float(final_data["a_N_squared"]),
        "alpha_s": float(final_data["alpha_s"]),
        "h_D": float(final_data["h_D"]),
        "gamma": float(final_data["gamma"]),
        "tau": float(final_data["tau"]),
        "tau_seconds": float(final_data["tau_seconds"]),
        "prefactor": float(final_data["prefactor"]),
        "analytic_denominator": float(final_data["analytic_denominator"]),
        "xi": xi,
        "composition_definition": "a_local_equals_nK_over_nB",
        "density_ratio_definition": "lambda_n_equals_nB_N_over_nB_Q",
        "analytic_formula_variant": "piecewise_constant_lambda_n",
        "slow_front_consistent": bool(u_N_max < 1.0),
    }


def Pi_NM(mu_B, Temp, j_B):
    """
    Return the nuclear-matter momentum flux Pi = h*u^2 + P.
    """
    nB = nB_NM(mu_B, Temp)
    if nB <= 0:
        return np.nan
    uN = j_B / nB
    return hNM(mu_B, Temp) * uN * uN + PNM(mu_B, Temp)
    
def hQM(muB, muK, B_one_forth, Temp, ms=0.0, upB=5000):
    """
    Return the quark-matter enthalpy density at fixed (muB, muK, T).
    """
    quark_state = _quark_uds_state(muB, muK, Temp, ms=ms, upB=upB)
    return float(quark_state["pressure"] + quark_state["energy"])

def Pi_QM(mu_B, mu_K, B_one_forth, Temp, j_B, ms=0.0, upB=5000):
    """
    Return the quark-matter momentum flux Pi = h*u^2 + P.
    """
    return _Pi_QM_state(mu_B, mu_K, B_one_forth, Temp, j_B, ms=ms, upB=upB)


def _solve_muB_inf_at_muK0_from_nB(nB_inf, B_one_forth, T, ms=0, upB=5000):
    """
    Solve for the equilibrated QM endpoint chemical potential muB_inf at muK=0
    from the target baryon density nB_inf.
    """
    if nB_inf <= 0.0:
        raise RuntimeError("nB_inf must be positive when solving for muB_inf")

    def density_residual(muB):
        return float(nB_QM(muB, 0.0, B_one_forth, T, ms=ms, upB=upB) - nB_inf)

    mu_lo = 1.0e-8
    mu_hi = 1500.0
    f_lo = density_residual(mu_lo)
    f_hi = density_residual(mu_hi)

    for _ in range(40):
        if abs(f_lo) == 0.0:
            return float(mu_lo)
        if abs(f_hi) == 0.0:
            return float(mu_hi)
        if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi < 0.0:
            break
        if not np.isfinite(f_hi) or f_hi < 0.0:
            mu_hi *= 1.5
            f_hi = density_residual(mu_hi)
        else:
            mu_lo *= 0.5
            if mu_lo <= 0.0:
                mu_lo = np.nextafter(0.0, 1.0)
            f_lo = density_residual(mu_lo)
    else:
        raise RuntimeError(
            f"Failed to bracket muB_inf for nB_inf={nB_inf:.6g} at T={T:.6g}"
        )

    sol = root_scalar(density_residual, bracket=[mu_lo, mu_hi], method="brentq")
    if not sol.converged:
        raise RuntimeError("Root solve for muB_inf failed to converge")

    return float(sol.root)


def uN(T, nB_N, Delta_n, B_one_forth, param=para.paraQMCRMF3, ms=0, upB=5000, return_more=False):
    """
    Compute the phase-boundary flux data from endpoint states.

    Parameters
    ----------
    T : float
        Temperature.
    nB_N : float
        Upstream nuclear baryon density n_B(0^-).
    Delta_n : float
        Density jump defined by n_B(infty) = nB_N + Delta_n.
    B_one_forth : float
        Bag constant parameter B^(1/4) for the quark EOS.
    param : sequence, optional
        RMF parameter set for the nuclear endpoint.
    ms : float, optional
        Strange-quark mass used on the QM side.
    upB : float, optional
        Upper integration bound passed to the quark EOS helpers.
    return_more : bool, optional
        If True, return a dict with endpoint thermodynamic details.

    Notes
    -----
    - The far-right quark endpoint is assumed fully equilibrated: muK = 0.
    - The returned values are Pi, jB, uN, where uN is in natural units.
    """
    if nB_N <= 0.0:
        raise RuntimeError("nB_N must be positive")

    nB_inf = nB_N + Delta_n
    if nB_inf <= 0.0:
        raise RuntimeError("nB_inf = nB_N + Delta_n must be positive")

    # Upstream nuclear endpoint from (T, nB_N).
    P_N = float(PNM_n(nB_N, T, param=param, NM_type="PNM"))
    h_N = float(P_N + edensNM_n(nB_N, T, param=param))

    # Fully equilibrated QM endpoint at muK = 0 and nB_inf.
    muB_inf = _solve_muB_inf_at_muK0_from_nB(nB_inf, B_one_forth, T, ms=ms, upB=upB)
    P_inf = float(PQM(muB_inf, 0.0, B_one_forth, T, ms=ms, upB=upB))
    h_inf = float(
        P_inf
        + edensQM(muB_inf, 0.0, B_one_forth, T, ms=ms, include_em=False, upB=upB)
    )

    # Momentum-flux matching fixes jB from the two endpoint states.
    term_N = h_N / (nB_N * nB_N)
    term_inf = h_inf / (nB_inf * nB_inf)
    denom = term_N - term_inf
    denom_scale = max(abs(term_N), abs(term_inf), 1.0)
    if abs(denom) <= 1.0e-12 * denom_scale:
        raise RuntimeError("Endpoint momentum-flux denominator is too close to zero")

    jB_sq = (P_inf - P_N) / denom
    if (not np.isfinite(jB_sq)) or jB_sq <= 0.0:
        raise RuntimeError(f"Endpoint matching gives non-physical jB^2={jB_sq}")

    jB = float(np.sqrt(jB_sq))
    uN = float(jB / nB_N)
    Pi = float(h_N * uN * uN + P_N)

    # Numerical consistency check on the equilibrated QM endpoint.
    uQ_inf = float(jB / nB_inf)
    Pi_inf = float(h_inf * uQ_inf * uQ_inf + P_inf)
    Pi_scale = max(abs(Pi), abs(Pi_inf), 1.0)
    if not np.isclose(Pi, Pi_inf, rtol=1.0e-8, atol=1.0e-10 * Pi_scale):
        raise RuntimeError(
            f"Endpoint momentum-flux mismatch: Pi_N={Pi:.12g}, Pi_inf={Pi_inf:.12g}"
        )

    if return_more:
        return {
            "Pi": Pi,
            "jB": jB,
            "uN": uN,
            "nB_inf": float(nB_inf),
            "P_N": P_N,
            "h_N": h_N,
            "P_inf": P_inf,
            "h_inf": h_inf,
            "muB_inf": muB_inf,
            "uQ_inf": uQ_inf,
        }

    return Pi, jB, uN


# Shared quark-state helpers
def _Pi_QM_state(muB, muK, B_one_forth, T, jB, ms=0.0, upB=5000):
    """
    Momentum flux Pi = h*u^2 + P for a quark state at fixed (muB, muK).
    """
    nB = nB_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB)
    if nB <= 0.0:
        return np.nan
    P = PQM(muB, muK, B_one_forth, T, ms=ms, upB=upB)
    h = P + edensQM(muB, muK, B_one_forth, T, ms=ms, include_em=False, upB=upB)
    u = jB / nB
    return float(h * u * u + P)


def _quark_thermo_state(muB, muK, B_one_forth, T, jB, ms=0.0, upB=5000, allow_zero_temperature=False):
    """
    Build a fully ms-consistent local quark thermodynamic state.

    The entropy density is reconstructed from the thermal relation
        s = (e + P - muB * nB - muK * nK) / T
    using the existing EOS helpers.
    """
    T = float(T)
    if (not np.isfinite(T)) or T < 0.0 or (T == 0.0 and not allow_zero_temperature):
        raise RuntimeError("Quark thermodynamic state requires T > 0")

    muB = float(muB)
    muK = float(muK)
    nB = float(nB_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
    if (not np.isfinite(nB)) or nB <= 0.0:
        raise RuntimeError("Quark thermodynamic state has non-positive baryon density")

    nK = float(nK_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
    P = float(PQM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
    e = float(edensQM(muB, muK, B_one_forth, T, ms=ms, include_em=False, upB=upB))
    if (not np.isfinite(nK)) or (not np.isfinite(P)) or (not np.isfinite(e)):
        raise RuntimeError("Quark thermodynamic state returned non-finite EOS quantities")

    u = float(jB / nB)
    h = float(P + e)
    if T == 0.0:
        s_density = 0.0
    else:
        s_density = float((e + P - muB * nB - muK * nK) / T)
    if not np.isfinite(s_density):
        raise RuntimeError("Quark thermodynamic state returned a non-finite entropy density")

    Pi = float(h * u * u + P)
    entropy_flux = float(s_density * u)
    if (not np.isfinite(Pi)) or (not np.isfinite(entropy_flux)):
        raise RuntimeError("Quark thermodynamic state returned non-finite fluxes")

    return {
        "muB": muB,
        "muK": muK,
        "T": T,
        "nB": nB,
        "nK": nK,
        "P": P,
        "e": e,
        "h": h,
        "u": u,
        "s": s_density,
        "w": entropy_flux,
        "Pi": Pi,
    }


def _solve_muB_Q_at_muK0_for_given_Pi(Pi, jB, B_one_forth, T, ms=0.0, upB=5000, stats=None):
    """
    Solve for the equilibrated QM endpoint muB_Q at muK=0 for a given Pi.
    """
    return _solve_muB_Q_at_muK0_for_given_Pi_ms(
        Pi,
        jB,
        B_one_forth,
        T,
        ms=ms,
        upB=upB,
        stats=stats,
    )


def _solve_muB_Q_at_muK0_for_given_Pi_ms(
    Pi,
    jB,
    B_one_forth,
    T,
    ms=0.0,
    upB=5000,
    stats=None,
    stats_key="q_root_calls",
    initial_guess=None,
):
    """
    Solve for the equilibrated QM endpoint muB_Q at muK=0 using the ms-aware
    quark momentum-flux helper throughout.
    """
    if stats is not None:
        stats[stats_key] = stats.get(stats_key, 0) + 1

    def equation(muB_in):
        muB = float(np.atleast_1d(muB_in)[0])
        return float(_Pi_QM_state(muB, 0.0, B_one_forth, T, jB, ms=ms, upB=upB) - Pi)

    guesses = []
    if initial_guess is not None:
        try:
            guesses.append(float(initial_guess))
        except Exception:
            pass
    guesses.extend([1100.0, 900.0, 1300.0, 700.0, 1500.0])

    tol = 1.0e-8 * max(abs(Pi), 1.0)
    best_muB = None
    best_metric = np.inf
    last_error = "fsolve did not produce a physical root"

    for muB_guess in guesses:
        if (not np.isfinite(muB_guess)) or muB_guess <= 0.0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                muB_arr, info, ier, mesg = fsolve(equation, muB_guess, full_output=True)
            muB_Q = float(np.atleast_1d(muB_arr)[0])
            Pi_residual = float(_Pi_QM_state(muB_Q, 0.0, B_one_forth, T, jB, ms=ms, upB=upB) - Pi)
            nB = float(nB_QM(muB_Q, 0.0, B_one_forth, T, ms=ms, upB=upB))
            if (not np.isfinite(muB_Q)) or (not np.isfinite(Pi_residual)) or (not np.isfinite(nB)) or nB <= 0.0:
                last_error = "Solved muB_Q lies on a non-physical density branch"
                continue
            metric = abs(Pi_residual)
            if metric < best_metric:
                best_metric = metric
                best_muB = muB_Q
            if ier == 1 and metric <= tol:
                return muB_Q
            last_error = str(mesg)
        except Exception as exc:
            last_error = str(exc)
            continue

    if best_muB is not None and np.isfinite(best_metric) and best_metric <= tol:
        return float(best_muB)
    raise RuntimeError(last_error)


def _branch_muK_seed(a_like):
    """
    Positive muK seed used by the steady-front quark-state solves.
    """
    return float(max(1.0, 20.0, 250.0 * abs(float(a_like))))


def _quark_state_residual(muB, muK, a_target, Pi, jB, nB_Q, nK_Q, B_one_forth, T, ms=0.0, upB=5000):
    """
    Return the local quark-state residuals at fixed (a_target, Pi, jB).
    """
    return np.array(
        [
            _Pi_QM_state(muB, muK, B_one_forth, T, jB, ms=ms, upB=upB) - Pi,
            (nK_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB) - nK_Q) / nB_Q - a_target,
        ],
        dtype=float,
    )


def _quark_state_residual_ok(residual, Pi, a_target):
    """
    Check whether a local quark-state residual is acceptable.
    """
    if not np.all(np.isfinite(residual)):
        return False
    pi_tol = 1.0e-8 * max(abs(Pi), 1.0)
    a_tol = 1.0e-8 * max(abs(a_target), 1.0)
    return bool(abs(float(residual[0])) <= pi_tol and abs(float(residual[1])) <= a_tol)


def _solve_quark_state_once_from_guess(a_target, Pi, jB, nB_Q, nK_Q, B_one_forth, T, ms=0.0, upB=5000, initial_guess=None, stats=None, stats_key="quark_state_root_calls"):
    """
    Try one local quark-state root solve from a single continuation guess.
    This is the fast path used during IVP integration.
    """
    if initial_guess is None:
        raise RuntimeError("initial_guess is required for single-guess quark-state solve")
    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving for a quark state")

    guess = np.asarray(initial_guess, dtype=float)
    if guess.shape[0] != 2 or not np.all(np.isfinite(guess)):
        raise RuntimeError("initial_guess must contain finite (muB, muK)")

    def equations(vec):
        muB, muK = map(float, vec)
        return _quark_state_residual(
            muB,
            muK,
            a_target,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            T,
            ms=ms,
            upB=upB,
        )

    if stats is not None:
        stats[stats_key] = stats.get(stats_key, 0) + 1
    sol = root(equations, guess, method="hybr", options={"maxfev": 120, "xtol": 1.0e-10})
    if not (sol.success and np.all(np.isfinite(sol.x))):
        raise RuntimeError(f"single-guess quark-state solve failed: {sol.message}")

    muB = float(sol.x[0])
    muK = float(sol.x[1])
    if muK < -1.0e-8:
        raise RuntimeError("single-guess quark-state solve returned negative muK")
    if muK < 0.0:
        muK = 0.0

    residual = _quark_state_residual(
        muB,
        muK,
        a_target,
        Pi,
        jB,
        nB_Q,
        nK_Q,
        B_one_forth,
        T,
        ms=ms,
        upB=upB,
    )
    if not _quark_state_residual_ok(residual, Pi, a_target):
        raise RuntimeError(
            "single-guess quark-state solve returned an unacceptable residual "
            f"({residual[0]:.3e}, {residual[1]:.3e})"
        )

    nB = float(nB_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
    if nB <= 0.0:
        raise RuntimeError("single-guess quark-state solve returned non-positive density")
    return muB, muK


def _solve_interface_Qstar_from_aQstar_and_Pi(aQstar, Pi, jB, nB_Q, nK_Q, B_one_forth, T, ms=0.0, upB=5000, initial_guess=None, stats=None, stats_key="qstar_root_calls"):
    """
    Solve for the interface state Qstar from Pi and aQstar.
    """
    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving for Qstar")

    guesses = []
    muK_seed = _branch_muK_seed(aQstar)
    muK_seed_strong = float(max(muK_seed, 400.0 * abs(float(aQstar))))
    if initial_guess is not None:
        guess0 = np.asarray(initial_guess, dtype=float)
        guesses.append(guess0)
        muB_seed = float(guess0[0])
    else:
        muB_seed = 1200.0

    guesses.append(np.array([muB_seed, muK_seed], dtype=float))
    guesses.append(np.array([1200.0, muK_seed], dtype=float))
    guesses.append(np.array([1500.0, max(muK_seed, 100.0 * abs(float(aQstar)))], dtype=float))
    guesses.append(np.array([muB_seed, muK_seed_strong], dtype=float))
    guesses.append(np.array([1500.0, muK_seed_strong], dtype=float))

    def equations(vec):
        muB, muK = map(float, vec)
        return _quark_state_residual(
            muB,
            muK,
            aQstar,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            T,
            ms=ms,
            upB=upB,
        )

    best_message = "Qstar solve did not converge"
    candidates = []
    candidate_tol = 1.0e-8
    nonneg_tol = 1.0e-8
    for guess in guesses:
        if stats is not None:
            stats[stats_key] = stats.get(stats_key, 0) + 1
        sol = root(equations, guess, method="hybr", options={"maxfev": 6000, "xtol": 1.0e-10})
        if sol.success and np.all(np.isfinite(sol.x)):
            muB_Qstar = float(sol.x[0])
            muK_Qstar = float(sol.x[1])
            residual = _quark_state_residual(
                muB_Qstar,
                max(muK_Qstar, 0.0),
                aQstar,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                T,
                ms=ms,
                upB=upB,
            )
            nB_Qstar = nB_QM(muB_Qstar, muK_Qstar, B_one_forth, T, ms=ms, upB=upB)
            if nB_Qstar > 0.0 and muK_Qstar >= -nonneg_tol and _quark_state_residual_ok(residual, Pi, aQstar):
                if muK_Qstar < 0.0:
                    muK_Qstar = 0.0
                is_new = True
                for cand in candidates:
                    if (
                        abs(muB_Qstar - cand["muB"]) <= candidate_tol * max(1.0, abs(cand["muB"]))
                        and abs(muK_Qstar - cand["muK"]) <= candidate_tol * max(1.0, abs(cand["muK"]), 1.0)
                    ):
                        is_new = False
                        break
                if is_new:
                    candidates.append({"muB": muB_Qstar, "muK": muK_Qstar})
        best_message = sol.message

    if candidates:
        if initial_guess is not None:
            muB_ref = float(initial_guess[0])
            muK_ref = max(0.0, float(initial_guess[1]))
        else:
            muB_ref = muB_seed
            muK_ref = 0.0

        muK_pref = max(muK_seed, muK_ref)
        candidates.sort(
            key=lambda cand: (
                abs(cand["muK"] - muK_pref),
                abs(cand["muB"] - muB_ref),
                -cand["muK"],
            )
        )
        return candidates[0]["muB"], candidates[0]["muK"]

    raise RuntimeError(f"Qstar state solve failed: {best_message}")


def _solve_local_quark_state_from_a_and_Pi(a, Pi, jB, nB_Q, nK_Q, B_one_forth, T, ms=0.0, upB=5000, initial_guess=None, stats=None):
    """
    Solve the local quark state (muB, muK, nB, u) at fixed a, Pi, jB.
    """
    if stats is not None:
        stats["local_state_calls"] = stats.get("local_state_calls", 0) + 1

    if initial_guess is not None:
        try:
            muB, muK = _solve_quark_state_once_from_guess(
                a,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                T,
                ms=ms,
                upB=upB,
                initial_guess=initial_guess,
                stats=stats,
                stats_key="local_root_calls",
            )
            nB = float(nB_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
            if nB <= 0.0:
                raise RuntimeError("Local quark state has non-positive density")
            u = float(jB / nB)
            return muB, muK, nB, u
        except Exception:
            # Fall back to the broader branch-aware candidate search only when
            # nearest-neighbor continuation fails.
            if stats is not None:
                stats["local_fast_failures"] = stats.get("local_fast_failures", 0) + 1

    muB, muK = _solve_interface_Qstar_from_aQstar_and_Pi(
        a,
        Pi,
        jB,
        nB_Q,
        nK_Q,
        B_one_forth,
        T,
        ms=ms,
        upB=upB,
        initial_guess=initial_guess,
        stats=stats,
        stats_key="local_root_calls",
    )
    nB = float(nB_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
    if nB <= 0.0:
        raise RuntimeError("Local quark state has non-positive density")
    u = float(jB / nB)
    return muB, muK, nB, u


# Shared transport and microphysics helpers


def _microphysics_at_Qstar_isothermal_baseline(muB_Qstar, T):
    """
    Isothermal BVP microphysics that matches the baseline steady-front solver.
    """
    return _microphysics_from_quark_state_isothermal_baseline(muB_Qstar, T)


def _microphysics_from_quark_state_energy(muB, T, allow_zero_temperature=False):
    """
    Energy-conserving transport coefficients.

    This path treats invD = 1/D as the primary coefficient. The T=0
    endpoint is an exact boundary state with invD=0, so no D value is needed
    by the ODE.
    """
    T = float(T)
    if (not np.isfinite(T)) or T < 0.0 or (T == 0.0 and not allow_zero_temperature):
        raise RuntimeError("Local microphysics requires T > 0")

    muQ = float(muB) / 3.0
    if (not np.isfinite(muQ)) or muQ <= 0.0:
        raise RuntimeError("Local microphysics requires muQ > 0")

    qD = _TRANSPORT_QD_COEFF * muQ
    if (not np.isfinite(qD)) or qD <= 0.0:
        raise RuntimeError("Local microphysics requires a positive finite screening scale")

    if T == 0.0:
        invD = 0.0
        eta = 0.0
    else:
        part1 = _TRANSPORT_H_CONST * T ** (5.0 / 3.0) / qD ** (2.0 / 3.0)
        part2 = np.pi**3 * T**2 / (12.0 * qD)
        denom_terms = part1 + part2
        if (
            (not np.isfinite(part1))
            or (not np.isfinite(part2))
            or (not np.isfinite(denom_terms))
            or denom_terms < 0.0
        ):
            raise RuntimeError("Energy microphysics returned a non-physical diffusion denominator")
        invD = float(_TRANSPORT_D_PREFACTOR * denom_terms)
        eta = float(9.0 * np.pi**2 * T**2 / muQ**2)

    weak_rate = _weak_rate_from_mu_q(muQ)
    tau = float(weak_rate["tau"])
    gamma = float(weak_rate["gamma"])
    if (
        (not np.isfinite(invD))
        or invD < 0.0
        or (not np.isfinite(eta))
        or eta < 0.0
        or (not np.isfinite(tau))
        or tau <= 0.0
        or (not np.isfinite(gamma))
        or gamma <= 0.0
    ):
        raise RuntimeError("Energy microphysics returned non-physical coefficients")

    return {
        "alpha_s": float(_TRANSPORT_ALPHA_S),
        "muQ": float(muQ),
        "qD": float(qD),
        "invD": float(invD),
        "eta": float(eta),
        "gamma": float(gamma),
        "tau": float(tau),
    }


def _microphysics_from_quark_state_isothermal_baseline(muB, T):
    """
    Baseline diffusion/reaction coefficients for the isothermal BVP path.
    """
    T = float(T)
    if (not np.isfinite(T)) or T <= 0.0:
        raise RuntimeError("Local microphysics requires T > 0")

    muQ = float(muB) / 3.0
    if (not np.isfinite(muQ)) or muQ <= 0.0:
        raise RuntimeError("Local microphysics requires muQ > 0")

    alpha_s = 0.3
    g_s = np.sqrt(4.0 * np.pi * alpha_s)
    qD = np.sqrt(3.0 * g_s**2 * muQ**2 / (2.0 * np.pi**2))
    h_const = 1.81317
    part1 = h_const * T**(5.0 / 3.0) / qD**(2.0 / 3.0)
    part2 = np.pi**3 * T**2 / (12.0 * qD)
    D = 1.0 / (24.0 * alpha_s**2 / np.pi * (part1 + part2))
    eta = 9.0 * np.pi**2 * T**2 / muQ**2
    weak_rate = _weak_rate_from_mu_q(muQ)
    tau = float(weak_rate["tau"])
    gamma = float(weak_rate["gamma"])

    if (
        (not np.isfinite(qD))
        or (not np.isfinite(D))
        or (not np.isfinite(eta))
        or (not np.isfinite(tau))
        or (not np.isfinite(gamma))
        or D <= 0.0
        or eta < 0.0
        or tau <= 0.0
        or gamma <= 0.0
    ):
        raise RuntimeError("Local microphysics returned non-physical coefficients")

    return {
        "alpha_s": float(alpha_s),
        "muQ": float(muQ),
        "qD": float(qD),
        "D": float(D),
        "eta": float(eta),
        "gamma": float(gamma),
        "tau": float(tau),
    }


# Entropy-enabled local closure helpers


# Energy-conserving local closure helpers


def _bvp_dense_derivative(sol, s_eval):
    """
    Evaluate dy/ds from a solve_bvp solution, falling back to finite differences.
    """
    s_eval = np.asarray(s_eval, dtype=float)
    try:
        dy_ds = np.asarray(sol.sol.derivative(1)(s_eval), dtype=float)
    except Exception:
        y_eval = np.asarray(sol.sol(s_eval), dtype=float)
        dy_ds = np.vstack(
            [
                np.gradient(np.asarray(y_eval[row], dtype=float), s_eval, edge_order=1)
                for row in range(y_eval.shape[0])
            ]
        )
    if dy_ds.shape[0] != sol.y.shape[0] and dy_ds.shape[-1] == sol.y.shape[0]:
        dy_ds = np.moveaxis(dy_ds, -1, 0)
    if dy_ds.shape[0] != sol.y.shape[0] or dy_ds.shape[-1] != s_eval.size:
        raise RuntimeError("invalid BVP dense derivative shape")
    return dy_ds


def _relativistic_gamma_from_u(u):
    """
    Relativistic gamma factor used by the energy-conserving solver.
    """
    u = float(u)
    if not np.isfinite(u):
        raise RuntimeError("Relativistic gamma factor requires finite u")
    r_gamma = float(np.sqrt(1.0 + u * u))
    if (not np.isfinite(r_gamma)) or r_gamma <= 0.0:
        raise RuntimeError("Relativistic gamma factor is non-physical")
    return r_gamma


def _fixed_TQstar_E_residual(muB, muK, T_Qstar, E_target, Pi, jB, B_one_forth, ms=0.0, upB=5000):
    """
    Residual for the fixed-T_Qstar interface solve. The unknowns are
    (muB_Qstar, muK_Qstar); the temperature is prescribed.
    """
    if E_target <= 0.0:
        raise RuntimeError("Fixed-T_Qstar interface closure requires E = h*u*r_gamma > 0")
    T_Qstar = float(T_Qstar)
    if (not np.isfinite(T_Qstar)) or T_Qstar < 0.0:
        raise RuntimeError("Fixed-T_Qstar interface closure requires T_Qstar >= 0")
    try:
        thermo = _quark_thermo_state(
            muB,
            muK,
            B_one_forth,
            T_Qstar,
            jB,
            ms=ms,
            upB=upB,
            allow_zero_temperature=True,
        )
    except Exception:
        return np.array([1.0e12, 1.0e12], dtype=float)
    E_loc = float(thermo["h"] * thermo["u"] * _relativistic_gamma_from_u(thermo["u"]))
    return np.array([thermo["Pi"] - Pi, E_loc - E_target], dtype=float)


def _fixed_TQstar_E_residual_ok(residual, Pi, E_target):
    if not np.all(np.isfinite(residual)):
        return False
    pi_tol = 1.0e-8 * max(abs(Pi), 1.0)
    E_tol = 1.0e-8 * max(abs(E_target), 1.0)
    return bool(abs(float(residual[0])) <= pi_tol and abs(float(residual[1])) <= E_tol)


def _solve_interface_Qstar_from_TQstar_E_and_Pi(
    T_Qstar,
    E_target,
    Pi,
    jB,
    nB_Q,
    nK_Q,
    B_one_forth,
    ms=0.0,
    upB=5000,
    initial_guess=None,
    stats=None,
    stats_key="qstar_root_calls",
):
    """
    Solve the Qstar interface state at prescribed T_Qstar.

    The two unknowns are (muB_Qstar, muK_Qstar). The constraints are momentum
    flux conservation and relativistic enthalpy-flux conservation. The
    interface composition aQstar is derived after the thermodynamic state is
    found.
    """
    T_Qstar = float(T_Qstar)
    if (not np.isfinite(T_Qstar)) or T_Qstar < 0.0:
        raise RuntimeError("Fixed-T_Qstar interface solve requires T_Qstar >= 0")
    if E_target <= 0.0:
        raise RuntimeError("Fixed-T_Qstar interface solve requires E = h*u*r_gamma > 0")
    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when deriving aQstar")

    guesses = []
    if initial_guess is not None:
        guess0 = np.asarray(initial_guess, dtype=float)
        if guess0.shape[0] >= 2 and np.all(np.isfinite(guess0[:2])):
            guesses.append(np.array([float(guess0[0]), max(0.0, float(guess0[1]))], dtype=float))
    guesses.extend(
        [
            np.array([1200.0, 120.0], dtype=float),
            np.array([1100.0, 80.0], dtype=float),
            np.array([1400.0, 160.0], dtype=float),
            np.array([1500.0, 240.0], dtype=float),
            np.array([1000.0, 40.0], dtype=float),
        ]
    )

    pi_scale = max(abs(Pi), 1.0)
    E_scale = max(abs(E_target), 1.0)

    def equations(vec):
        residual = _fixed_TQstar_E_residual(
            float(vec[0]),
            float(vec[1]),
            T_Qstar,
            E_target,
            Pi,
            jB,
            B_one_forth,
            ms=ms,
            upB=upB,
        )
        return np.array([residual[0] / pi_scale, residual[1] / E_scale], dtype=float)

    best_message = "Fixed-T_Qstar interface solve did not converge"
    candidates = []
    nonneg_tol = 1.0e-8
    for guess in guesses:
        try:
            if stats is not None:
                stats[stats_key] = stats.get(stats_key, 0) + 1
            sol = root(equations, guess, method="hybr", options={"maxfev": 160, "xtol": 1.0e-10})
            if not (sol.success and np.all(np.isfinite(sol.x))):
                best_message = str(sol.message)
                continue
            muB = float(sol.x[0])
            muK = float(sol.x[1])
            if muK < -nonneg_tol or muB <= 0.0:
                best_message = "Fixed-T_Qstar interface solve returned a non-physical chemical potential"
                continue
            if muK < 0.0:
                muK = 0.0
            residual = _fixed_TQstar_E_residual(
                muB,
                muK,
                T_Qstar,
                E_target,
                Pi,
                jB,
                B_one_forth,
                ms=ms,
                upB=upB,
            )
            if not _fixed_TQstar_E_residual_ok(residual, Pi, E_target):
                best_message = (
                    "Fixed-T_Qstar interface solve returned an unacceptable residual "
                    f"({residual[0]:.3e}, {residual[1]:.3e})"
                )
                continue
            thermo = _quark_thermo_state(
                muB,
                muK,
                B_one_forth,
                T_Qstar,
                jB,
                ms=ms,
                upB=upB,
                allow_zero_temperature=True,
            )
            r_gamma = _relativistic_gamma_from_u(thermo["u"])
            E_loc = float(thermo["h"] * thermo["u"] * r_gamma)
            aQstar = float((thermo["nK"] - nK_Q) / nB_Q)
            if thermo["h"] <= 0.0 or E_loc <= 0.0 or not np.isfinite(aQstar):
                best_message = "Fixed-T_Qstar interface solve returned a non-physical state"
                continue
            thermo["r_gamma"] = r_gamma
            thermo["E"] = E_loc
            thermo["aQstar"] = aQstar
            candidates.append(thermo)
        except Exception as exc:
            best_message = str(exc)

    if not candidates:
        raise RuntimeError(f"Fixed-T_Qstar interface solve failed: {best_message}")

    if initial_guess is not None:
        guess0 = np.asarray(initial_guess, dtype=float)
        muB_ref = float(guess0[0]) if guess0.shape[0] >= 1 and np.isfinite(guess0[0]) else 1200.0
        muK_ref = float(guess0[1]) if guess0.shape[0] >= 2 and np.isfinite(guess0[1]) else 120.0
    else:
        muB_ref = 1200.0
        muK_ref = 120.0
    candidates.sort(key=lambda cand: (abs(cand["muK"] - muK_ref), abs(cand["muB"] - muB_ref), -cand["muK"]))
    return candidates[0]


def _solve_local_quark_state_from_nK_E_and_Pi(
    nK_target,
    E,
    Pi,
    jB,
    B_one_forth,
    ms=0.0,
    upB=5000,
    initial_guess=None,
    T_ref=None,
    stats=None,
):
    """Solve the energy-conserving local EOS closure at an absolute nK."""
    if stats is not None:
        stats["local_state_calls"] = stats.get("local_state_calls", 0) + 1
    nK_target = float(nK_target)
    E = float(E)
    Pi = float(Pi)
    jB = float(jB)
    if not np.all(np.isfinite([nK_target, E, Pi, jB])) or E <= 0.0 or jB <= 0.0:
        raise RuntimeError("Absolute-nK energy closure requires finite nK, Pi and positive E, jB")

    guesses = []
    if initial_guess is not None:
        guess = np.asarray(initial_guess, dtype=float).ravel()
        if guess.size < 3 or not np.all(np.isfinite(guess[:3])) or guess[2] <= 0.0:
            raise RuntimeError("initial_guess must contain finite (muB, muK, T) with T > 0")
        guesses.append((float(guess[0]), float(guess[1]), float(guess[2])))
    if T_ref is None or (not np.isfinite(T_ref)) or T_ref <= 0.0:
        T_ref = 10.0
    muK_seed = max(0.0, guesses[0][1]) if guesses else 20.0
    guesses.extend(
        [
            (1100.0, muK_seed, float(T_ref)),
            (1300.0, max(muK_seed, 50.0), float(T_ref)),
            (900.0, max(muK_seed, 10.0), float(T_ref)),
        ]
    )

    pi_scale = max(abs(Pi), 1.0)
    E_scale = max(abs(E), 1.0)
    nK_scale = max(abs(nK_target), 1.0)
    best_message = "Absolute-nK energy-conserving local state did not converge"
    best = None
    best_norm = np.inf

    def equations(vec):
        muB_val = float(vec[0])
        muK_val = float(vec[1])
        logT_val = float(vec[2])
        if muB_val <= 0.0 or (not np.isfinite(logT_val)) or abs(logT_val) > 700.0:
            return np.full(3, 1.0e12, dtype=float)
        T_val = float(np.exp(logT_val))
        try:
            thermo = _quark_thermo_state(
                muB_val,
                muK_val,
                B_one_forth,
                T_val,
                jB,
                ms=ms,
                upB=upB,
            )
            E_val = float(thermo["h"] * thermo["u"] * _relativistic_gamma_from_u(thermo["u"]))
        except Exception:
            return np.full(3, 1.0e12, dtype=float)
        return np.array(
            [
                (thermo["Pi"] - Pi) / pi_scale,
                (thermo["nK"] - nK_target) / nK_scale,
                (E_val - E) / E_scale,
            ],
            dtype=float,
        )

    for muB_guess, muK_guess, T_guess in guesses:
        try:
            if stats is not None:
                stats["local_root_calls"] = stats.get("local_root_calls", 0) + 1
            sol = root(
                equations,
                np.array([muB_guess, muK_guess, np.log(T_guess)], dtype=float),
                method="hybr",
                options={"maxfev": 240, "xtol": 1.0e-10},
            )
            if not np.all(np.isfinite(sol.x)):
                continue
            residual_norm = float(np.linalg.norm(equations(sol.x), ord=np.inf))
            if residual_norm < best_norm:
                best_norm = residual_norm
                best = sol.x.copy()
            if sol.success and residual_norm <= 1.0e-8:
                break
            best_message = str(sol.message)
        except Exception as exc:
            best_message = str(exc)

    if best is None or best_norm > 1.0e-7:
        raise RuntimeError(f"{best_message}; best scaled residual={best_norm:.3e}")

    thermo = _quark_thermo_state(
        float(best[0]),
        float(best[1]),
        B_one_forth,
        float(np.exp(float(best[2]))),
        jB,
        ms=ms,
        upB=upB,
    )
    thermo["r_gamma"] = _relativistic_gamma_from_u(thermo["u"])
    thermo["E"] = float(thermo["h"] * thermo["u"] * thermo["r_gamma"])
    return thermo


# Public isothermal front solver
def solve_front_isothermal(
    T,
    nB_N,
    B_one_forth,
    aQstar,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1.0e-3,
    n_mesh=300,
    tol_bvp=1.0e-4,
    max_nodes=50000,
    jB_guess=None,
    jB_bounds=None,
    kappa_factor=1.0,
    return_profile=False,
    verb=False,
):
    """
    Solve the fixed-aQstar steady-front problem as a compact-coordinate BVP.

    This solver keeps the same hydro + diffusion + reaction equations as the
    1D IVP shooting solvers, but uses solve_bvp with jB as an unknown BVP
    parameter. The compact coordinate is s = 1 - exp(-lambda*x/kappa_factor), integrated on
    s in [0, 1 - tail_eps]. The endpoint truncation is controlled only by
    tail_eps; no compact_tail_lengths cutoff is used in this BVP path.
    """
    if nB_N <= 0.0:
        raise RuntimeError("nB_N must be positive")
    if tail_eps <= 0.0 or tail_eps >= 1.0:
        raise RuntimeError("tail_eps must satisfy 0 < tail_eps < 1")
    if int(n_mesh) < 5:
        raise RuntimeError("n_mesh must be at least 5")
    if tol_bvp <= 0.0:
        raise RuntimeError("tol_bvp must be positive")
    if max_nodes <= int(n_mesh):
        raise RuntimeError("max_nodes must be larger than n_mesh")
    if kappa_factor <= 0.0:
        raise RuntimeError("kappa_factor must be positive")

    upB = 5000
    t_start = time.perf_counter()
    if isinstance(verb, str):
        verb_mode = "full" if verb.lower() == "full" else ("simple" if verb else "off")
    else:
        verb_mode = "simple" if verb else "off"
    full_diag = verb_mode == "full"
    simple_diag = verb_mode in ("simple", "full")

    def _diag(msg):
        if full_diag:
            dt = time.perf_counter() - t_start
            print(f"[steady_front_bvp +{dt:8.2f}s] {msg}", flush=True)

    if jB_guess is None:
        jB_guess = 1.0e-6 * nB_N
    jB_guess = float(jB_guess)
    if jB_guess <= 0.0:
        raise RuntimeError("jB_guess must be positive")

    bounded_jB = jB_bounds is not None
    if bounded_jB:
        if len(jB_bounds) != 2:
            raise RuntimeError("jB_bounds must be a 2-tuple (jB_min, jB_max)")
        jB_lower_bound = float(jB_bounds[0])
        jB_upper_bound = float(jB_bounds[1])
        if jB_lower_bound <= 0.0 or jB_upper_bound <= 0.0 or jB_upper_bound <= jB_lower_bound:
            raise RuntimeError("jB_bounds must satisfy 0 < jB_min < jB_max")
        if not (jB_lower_bound < jB_guess < jB_upper_bound):
            jB_guess = min(max(jB_guess, jB_lower_bound * (1.0 + 1.0e-6)), jB_upper_bound * (1.0 - 1.0e-6))
    else:
        jB_lower_bound = 0.0
        jB_upper_bound = np.inf

    def _param_from_jB(jB):
        jB = float(jB)
        if bounded_jB:
            frac = (jB - jB_lower_bound) / (jB_upper_bound - jB_lower_bound)
            frac = float(np.clip(frac, 1.0e-12, 1.0 - 1.0e-12))
            return float(np.log(frac / (1.0 - frac)))
        return float(np.log(jB))

    def _jB_from_param(theta):
        theta = float(theta)
        if bounded_jB:
            theta_clip = float(np.clip(theta, -60.0, 60.0))
            sig = 1.0 / (1.0 + np.exp(-theta_clip))
            return float(jB_lower_bound + (jB_upper_bound - jB_lower_bound) * sig)
        return float(np.exp(np.clip(theta, -700.0, 700.0)))

    stats = {
        "bvp_ode_calls": 0,
        "bvp_bc_calls": 0,
        "q_root_calls": 0,
        "qstar_root_calls": 0,
        "local_state_calls": 0,
        "local_root_calls": 0,
        "local_fast_failures": 0,
        "profile_state_calls": 0,
        "global_state_builds": 0,
        "global_state_failures": 0,
        "local_state_failures": 0,
    }
    state_cache = {}
    last_failure = {"message": ""}
    s_end = float(1.0 - tail_eps)

    def _build_global_state(theta):
        key = round(float(theta), 12)
        if key in state_cache:
            return state_cache[key]

        jB = _jB_from_param(theta)
        stats["global_state_builds"] += 1

        # Upstream nuclear state N at x = 0^-.
        P_N = float(PNM_n(nB_N, T, param=param, NM_type=NM_type))
        e_N = float(edensNM_n(nB_N, T, param=param))
        h_N = float(P_N + e_N)
        u_N = float(jB / nB_N)
        Pi = float(h_N * u_N * u_N + P_N)

        # Far-right equilibrated quark state Q with muK = 0.
        muB_Q = _solve_muB_Q_at_muK0_for_given_Pi(Pi, jB, B_one_forth, T, ms=ms, upB=upB, stats=stats)
        nB_Q = float(nB_QM(muB_Q, 0.0, B_one_forth, T, ms=ms, upB=upB))
        if nB_Q <= 0.0:
            raise RuntimeError("Equilibrated Q state has non-positive density")
        if abs(ms) <= 1.0e-12:
            nK_Q = 0.0
        else:
            nK_Q = float(nK_QM(muB_Q, 0.0, B_one_forth, T, ms=ms, upB=upB))

        # Pure neutron matter implies nK_N = nB_N.
        a_N = float((nB_N - nK_Q) / nB_Q)

        # Interface state Qstar at x = 0+.
        muK_Qstar_seed = _branch_muK_seed(aQstar)
        muB_Qstar, muK_Qstar = _solve_interface_Qstar_from_aQstar_and_Pi(
            aQstar,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            T,
            ms=ms,
            upB=upB,
            initial_guess=(muB_Q, muK_Qstar_seed),
            stats=stats,
            stats_key="qstar_root_calls",
        )
        nB_Qstar = float(nB_QM(muB_Qstar, muK_Qstar, B_one_forth, T, ms=ms, upB=upB))
        if nB_Qstar <= 0.0:
            raise RuntimeError("Qstar state has non-positive density")

        micro = _microphysics_at_Qstar_isothermal_baseline(muB_Qstar, T)
        D = float(micro["D"])
        eta = float(micro["eta"])
        gamma = float(micro["gamma"])
        tau = float(micro["tau"])

        u_Q = float(jB / nB_Q)
        disc = float(u_Q * u_Q + 4.0 * D * gamma * eta)
        if (not np.isfinite(disc)) or disc <= 0.0:
            raise RuntimeError("Tail discriminant is non-positive")
        lam = float((-u_Q + np.sqrt(disc)) / (2.0 * D))
        if (not np.isfinite(lam)) or lam <= 0.0:
            raise RuntimeError("Tail decay lambda must be positive")
        q0 = float(-a_N * u_N)
        x_end = float(-float(kappa_factor) * np.log1p(-s_end) / lam)
        tail_coeff = float(D * lam + u_Q)
        state = {
            "jB": jB,
            "P_N": P_N,
            "e_N": e_N,
            "h_N": h_N,
            "u_N": u_N,
            "Pi": Pi,
            "muB_Q": float(muB_Q),
            "nB_Q": nB_Q,
            "nK_Q": float(nK_Q),
            "a_N": a_N,
            "muB_Qstar": float(muB_Qstar),
            "muK_Qstar": float(muK_Qstar),
            "nB_Qstar": nB_Qstar,
            "D": D,
            "eta": eta,
            "gamma": gamma,
            "tau": tau,
            "lambda": lam,
            "u_Q": u_Q,
            "q0": q0,
            "tail_coeff": tail_coeff,
            "x_end": x_end,
        }
        state_cache[key] = state
        return state

    def _state_or_none(theta):
        try:
            return _build_global_state(theta)
        except Exception as exc:
            stats["global_state_failures"] += 1
            last_failure["message"] = str(exc)
            return None

    def _ode(s, y, p):
        stats["bvp_ode_calls"] += 1
        state = _state_or_none(float(p[0]))
        if state is None:
            return np.zeros_like(y) + 1.0e12

        dyds = np.empty_like(y)
        guess = (state["muB_Qstar"], state["muK_Qstar"])
        for i in range(y.shape[1]):
            a_val = float(y[0, i])
            q_val = float(y[1, i])
            if (not np.isfinite(a_val)) or (not np.isfinite(q_val)):
                stats["local_state_failures"] += 1
                dyds[:, i] = 1.0e12
                continue
            try:
                muB_loc, muK_loc, nB_loc, u_loc = _solve_local_quark_state_from_a_and_Pi(
                    a_val,
                    state["Pi"],
                    state["jB"],
                    state["nB_Q"],
                    state["nK_Q"],
                    B_one_forth,
                    T,
                    ms=ms,
                    upB=upB,
                    initial_guess=guess,
                        stats=stats,
                )
                guess = (muB_loc, muK_loc)
                one_minus_s = max(1.0 - float(s[i]), np.finfo(float).tiny)
                dx_ds = float(kappa_factor) / (state["lambda"] * one_minus_s)
                dyds[0, i] = ((q_val + u_loc * a_val) / state["D"]) * dx_ds
                dyds[1, i] = (state["gamma"] * (a_val**3 + state["eta"] * a_val)) * dx_ds
            except Exception as exc:
                stats["local_state_failures"] += 1
                last_failure["message"] = str(exc)
                dyds[:, i] = 1.0e12
        return dyds

    def _bc(ya, yb, p):
        stats["bvp_bc_calls"] += 1
        state = _state_or_none(float(p[0]))
        if state is None:
            return np.array([1.0e12, 1.0e12, 1.0e12], dtype=float)
        return np.array(
            [
                ya[0] - float(aQstar),
                ya[1] - state["q0"],
                yb[1] + state["tail_coeff"] * yb[0],
            ],
            dtype=float,
        )

    theta_guess = _param_from_jB(jB_guess)
    state0 = _build_global_state(theta_guess)
    s_mesh = np.linspace(0.0, s_end, int(n_mesh))
    tail_shape = np.maximum(1.0 - s_mesh, tail_eps) ** max(float(kappa_factor), 1.0e-12)
    a_guess = float(aQstar) * tail_shape
    q_tail_guess = -state0["tail_coeff"] * a_guess
    blend = s_mesh / max(s_end, np.finfo(float).tiny)
    q_guess = (1.0 - blend) * state0["q0"] + blend * q_tail_guess
    y_guess = np.vstack((a_guess, q_guess))

    _diag(
        f"starting compact BVP with jB_guess={jB_guess:.6g}, aQstar={aQstar:.6g}, "
        f"tail_eps={tail_eps:.3g}, branch=muK-rich"
    )

    try:
        sol = solve_bvp(
            _ode,
            _bc,
            s_mesh,
            y_guess,
            p=np.array([theta_guess], dtype=float),
            tol=tol_bvp,
            max_nodes=max_nodes,
            verbose=2 if full_diag else 0,
        )
    except Exception as exc:
        return {
            "success": False,
            "message": f"solve_bvp raised: {exc}; last failure: {last_failure['message']}",
            "aQstar": float(aQstar),
            "jB": np.nan,
            "branch_label": "muK-rich",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "bvp_status": -999,
            **{k: int(v) for k, v in stats.items()},
        }

    theta_sol = float(sol.p[0])
    state = _state_or_none(theta_sol)
    if state is None:
        return {
            "success": False,
            "message": f"BVP final state construction failed: {last_failure['message']}",
            "aQstar": float(aQstar),
            "jB": np.nan,
            "branch_label": "muK-rich",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "bvp_status": int(sol.status),
            **{k: int(v) for k, v in stats.items()},
        }

    a_end = float(sol.y[0, -1])
    q_end = float(sol.y[1, -1])
    tail_drive = float(state["tail_coeff"] * a_end)
    tail_residual = float(q_end + tail_drive)
    tail_scale = float(max(abs(q_end), abs(tail_drive), np.finfo(float).tiny))
    tail_residual_norm = float(tail_residual / tail_scale)

    success = bool(sol.success and np.isfinite(tail_residual_norm) and abs(tail_residual_norm) <= max(tol_bvp, 10.0 * np.finfo(float).eps))
    result = {
        "success": success,
        "message": "Compact BVP steady-front solve converged" if success else f"{sol.message}; last failure: {last_failure['message']}",
        "jB": float(state["jB"]),
        "aQstar": float(aQstar),
        "branch_label": "muK-rich",
        "coordinate": "BVP: s in [0, 1-tail_eps], s=1-exp(-lambda*x/kappa_factor)",
        "tail_eps": float(tail_eps),
        "kappa_factor": float(kappa_factor),
        "u_N": float(state["u_N"]),
        "u_Q": float(state["u_Q"]),
        "a_N": float(state["a_N"]),
        "Pi": float(state["Pi"]),
        "muB_Qstar": float(state["muB_Qstar"]),
        "muK_Qstar": float(state["muK_Qstar"]),
        "nB_Qstar": float(state["nB_Qstar"]),
        "muB_Q": float(state["muB_Q"]),
        "nB_Q": float(state["nB_Q"]),
        "nK_Q": float(state["nK_Q"]),
        "D": float(state["D"]),
        "eta": float(state["eta"]),
        "gamma": float(state["gamma"]),
        "tau": float(state["tau"]),
        "lambda": float(state["lambda"]),
        "kappa": float(kappa_factor / state["lambda"]),
        "s_end": float(s_end),
        "x_end": float(state["x_end"]),
        "a_end": a_end,
        "q_end": q_end,
        "tail_residual": tail_residual,
        "tail_residual_norm": tail_residual_norm,
        "tail_scale": tail_scale,
        "_residual": np.array([tail_residual_norm], dtype=float),
        "_root_method": "solve_bvp_parameter",
        "bvp_status": int(sol.status),
        "bvp_message": str(sol.message),
        "bvp_niter": int(getattr(sol, "niter", -1)),
        "bvp_nodes": int(sol.x.size),
        **{k: int(v) for k, v in stats.items()},
    }

    coarse_tol_bvp = max(float(tol_bvp), 1.0e-3)
    coarse_n_mesh = int(n_mesh) if int(n_mesh) <= 60 else 60
    coarse_retry_available = (
        (coarse_tol_bvp > float(tol_bvp))
        or (coarse_n_mesh != int(n_mesh))
    )

    if return_profile:
        try:
            s_prof = np.asarray(sol.x, dtype=float)
            a_prof = np.asarray(sol.y[0], dtype=float)
            q_prof = np.asarray(sol.y[1], dtype=float)
            x_prof = -float(kappa_factor) * np.log1p(-s_prof) / float(state["lambda"])
            muB_prof = np.empty_like(s_prof)
            muK_prof = np.empty_like(s_prof)
            nB_prof = np.empty_like(s_prof)
            u_prof = np.empty_like(s_prof)
            guess = (state["muB_Qstar"], state["muK_Qstar"])
            for i, a_val in enumerate(a_prof):
                stats["profile_state_calls"] += 1
                muB_loc, muK_loc, nB_loc, u_loc = _solve_local_quark_state_from_a_and_Pi(
                    float(a_val),
                    state["Pi"],
                    state["jB"],
                    state["nB_Q"],
                    state["nK_Q"],
                    B_one_forth,
                    T,
                    ms=ms,
                    upB=upB,
                    initial_guess=guess,
                    stats=stats,
                )
                guess = (muB_loc, muK_loc)
                muB_prof[i] = muB_loc
                muK_prof[i] = muK_loc
                nB_prof[i] = nB_loc
                u_prof[i] = u_loc
            closure_prof = np.abs(
                np.array(
                    [
                        (nK_QM(float(muB_prof[i]), float(muK_prof[i]), B_one_forth, T, ms=ms, upB=upB) - state["nK_Q"])
                        / state["nB_Q"]
                        - a_prof[i]
                        for i in range(len(a_prof))
                    ],
                    dtype=float,
                )
            )
            closure_error_max = float(np.max(closure_prof)) if len(closure_prof) else 0.0
            result.update(
                {
                    "s": s_prof,
                    "x": x_prof,
                    "a": a_prof,
                    "q": q_prof,
                    "u": u_prof,
                    "nB": nB_prof,
                    "muB": muB_prof,
                    "muK": muK_prof,
                    "closure_error": closure_prof,
                    "closure_error_max": closure_error_max,
                    "profile_state_calls": int(stats["profile_state_calls"]),
                    "local_state_calls": int(stats["local_state_calls"]),
                    "local_root_calls": int(stats["local_root_calls"]),
                    "local_fast_failures": int(stats["local_fast_failures"]),
                }
            )
        except Exception as exc:
            global _ISOTHERMAL_RETRY_ACTIVE
            if coarse_retry_available and _ISOTHERMAL_RETRY_ACTIVE == 0:
                _ISOTHERMAL_RETRY_ACTIVE += 1
                try:
                    coarse_profile = solve_front_isothermal(
                        T=T,
                        nB_N=nB_N,
                        B_one_forth=B_one_forth,
                        aQstar=aQstar,
                        ms=ms,
                        param=param,
                        NM_type=NM_type,
                        tail_eps=tail_eps,
                        n_mesh=coarse_n_mesh,
                        tol_bvp=coarse_tol_bvp,
                        max_nodes=max(max_nodes, coarse_n_mesh + 1),
                        jB_guess=(
                            float(jB_guess)
                            if jB_guess is not None and np.isfinite(float(jB_guess)) and float(jB_guess) > 0.0
                            else float(state["jB"])
                        ),
                        jB_bounds=jB_bounds,
                        kappa_factor=kappa_factor,
                        return_profile=True,
                        verb=False,
                    )
                    if bool(coarse_profile.get("success")) and "x" in coarse_profile:
                        coarse_profile = dict(coarse_profile)
                        coarse_profile["retry_source"] = "coarse_isothermal_profile"
                        coarse_profile["retry_seed_jB"] = float(state["jB"])
                        coarse_profile["message"] = (
                            f"{coarse_profile.get('message', '')}; returned coarse profile after strict profile reconstruction failed: {exc}"
                        )
                        return coarse_profile
                finally:
                    _ISOTHERMAL_RETRY_ACTIVE -= 1
            raise

    retryable_isothermal_failure = (
        (not bool(result.get("success")))
        and int(result.get("bvp_status", -999)) == 2
        and "singular jacobian" in str(result.get("bvp_message", "")).lower()
    )

    if retryable_isothermal_failure and coarse_retry_available and _ISOTHERMAL_RETRY_ACTIVE == 0:
        _ISOTHERMAL_RETRY_ACTIVE += 1
        try:
            coarse_result = solve_front_isothermal(
                T=T,
                nB_N=nB_N,
                B_one_forth=B_one_forth,
                aQstar=aQstar,
                ms=ms,
                param=param,
                NM_type=NM_type,
                tail_eps=tail_eps,
                n_mesh=coarse_n_mesh,
                tol_bvp=coarse_tol_bvp,
                max_nodes=max(max_nodes, coarse_n_mesh + 1),
                jB_guess=jB_guess,
                jB_bounds=jB_bounds,
                kappa_factor=kappa_factor,
                return_profile=False,
                verb=False,
            )
            coarse_jB = float(coarse_result.get("jB", np.nan))
            if bool(coarse_result.get("success")) and np.isfinite(coarse_jB) and coarse_jB > 0.0:
                refined_result = solve_front_isothermal(
                    T=T,
                    nB_N=nB_N,
                    B_one_forth=B_one_forth,
                    aQstar=aQstar,
                    ms=ms,
                    param=param,
                    NM_type=NM_type,
                    tail_eps=tail_eps,
                    n_mesh=n_mesh,
                    tol_bvp=tol_bvp,
                    max_nodes=max_nodes,
                    jB_guess=coarse_jB,
                    jB_bounds=jB_bounds,
                    kappa_factor=kappa_factor,
                    return_profile=return_profile,
                    verb=verb,
                )
                if bool(refined_result.get("success")):
                    refined_result = dict(refined_result)
                    refined_result["retry_seed_jB"] = coarse_jB
                    refined_result["retry_source"] = "coarse_isothermal_seed"
                    refined_result["message"] = (
                        f"{refined_result.get('message', '')}; recovered via coarse jB-seeded retry"
                    )
                    return refined_result
        finally:
            _ISOTHERMAL_RETRY_ACTIVE -= 1

    if simple_diag:
        print(
            f"bvp jB={result['jB']:.6g}, aQstar={aQstar:.6g}, "
            f"tail_norm={tail_residual_norm:.6g}, status={sol.status}, success={success}"
        )
    return result


# Adiabatic solver support helpers


def _strip_energy_profile_fields(result):
    """
    Return a shallow copy of an energy-solver result without profile arrays.
    """
    result_out = dict(result)
    for key in (
        "s_coord",
        "bvp_s_coord",
        "bvp_x",
        "bvp_a",
        "bvp_q",
        "bvp_u",
        "bvp_nB",
        "bvp_muB",
        "bvp_muK",
        "bvp_T_profile",
        "bvp_h_profile",
        "bvp_r_gamma_profile",
        "bvp_D_profile",
        "bvp_invD_profile",
        "bvp_eta_profile",
        "bvp_gamma_profile",
        "bvp_q_prime_profile",
        "bvp_R_kaon_profile",
        "bvp_kaon_equation_residual_profile",
        "x",
        "a",
        "q",
        "u",
        "nB",
        "muB",
        "muK",
        "T_profile",
        "h_profile",
        "D_profile",
        "invD_profile",
        "eta_profile",
        "gamma_profile",
        "q_prime_profile",
        "R_kaon_profile",
        "kaon_equation_residual_profile",
    ):
        result_out.pop(key, None)
    return result_out


# Public adiabatic front solver


def _solve_interface_state_from_local_a_E_and_Pi(
    a_target,
    E,
    Pi,
    jB,
    B_one_forth,
    ms=0.0,
    upB=5000,
    initial_guess=None,
    stats=None,
):
    """Solve the interface EOS state at a local fraction nK/nB."""
    a_target = float(a_target)
    if not (0.0 < a_target < 1.0):
        raise RuntimeError("aQstar must satisfy 0 < aQstar < 1")
    guesses = []
    if initial_guess is not None:
        guess = np.asarray(initial_guess, dtype=float).ravel()
        if guess.size >= 3 and np.all(np.isfinite(guess[:3])) and guess[2] > 0.0:
            guesses.append((float(guess[0]), float(guess[1]), float(guess[2])))
    guesses.extend([(1200.0, 100.0, 10.0), (1500.0, 200.0, 20.0), (1000.0, 40.0, 5.0)])
    pi_scale = max(abs(float(Pi)), 1.0)
    E_scale = max(abs(float(E)), 1.0)
    best = None
    best_norm = np.inf
    best_message = "local-a interface closure did not converge"

    def equations(vec):
        muB_val, muK_val, logT_val = map(float, vec)
        if muB_val <= 0.0 or abs(logT_val) > 700.0:
            return np.full(3, 1.0e12, dtype=float)
        try:
            thermo = _quark_thermo_state(
                muB_val,
                muK_val,
                B_one_forth,
                float(np.exp(logT_val)),
                jB,
                ms=ms,
                upB=upB,
            )
            E_val = thermo["h"] * thermo["u"] * _relativistic_gamma_from_u(thermo["u"])
            return np.array(
                [
                    (thermo["Pi"] - Pi) / pi_scale,
                    E_val / E_scale - E / E_scale,
                    thermo["nK"] / thermo["nB"] - a_target,
                ],
                dtype=float,
            )
        except Exception:
            return np.full(3, 1.0e12, dtype=float)

    for muB_guess, muK_guess, T_guess in guesses:
        try:
            if stats is not None:
                stats["qstar_root_calls"] = stats.get("qstar_root_calls", 0) + 1
            sol = root(
                equations,
                np.array([muB_guess, muK_guess, np.log(T_guess)], dtype=float),
                method="hybr",
                options={"maxfev": 400, "xtol": 1.0e-10},
            )
            if not np.all(np.isfinite(sol.x)):
                continue
            norm = float(np.linalg.norm(equations(sol.x), ord=np.inf))
            if norm < best_norm:
                best_norm = norm
                best = sol.x.copy()
            if sol.success and norm <= 1.0e-8:
                break
            best_message = str(sol.message)
        except Exception as exc:
            best_message = str(exc)
    if best is None or best_norm > 1.0e-7:
        raise RuntimeError(f"{best_message}; best scaled residual={best_norm:.3e}")
    thermo = _quark_thermo_state(
        float(best[0]),
        float(best[1]),
        B_one_forth,
        float(np.exp(float(best[2]))),
        jB,
        ms=ms,
        upB=upB,
    )
    thermo["r_gamma"] = _relativistic_gamma_from_u(thermo["u"])
    thermo["E"] = float(thermo["h"] * thermo["u"] * thermo["r_gamma"])
    thermo["a"] = float(thermo["nK"] / thermo["nB"])
    return thermo


def _nK_tail_residual(yb, target):
    """Return the shifted downstream Robin residual for absolute nK."""
    return float(
        float(yb[1])
        - float(target["jK"])
        - (float(target["D"]) * float(target["lambda"]) + float(target["u"]))
        * (float(yb[0]) - float(target["nK"]))
    )


def _solve_front_energy_conserving_nK_once(
    T,
    nB_N,
    B_one_forth,
    aQstar,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    compact_scale=None,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    jB_bounds=None,
    return_profile=False,
    return_raw_bvp_grid=False,
    verb=False,
    profile_guess=None,
):
    """Solve the energy-conserving front using absolute nK and the exact rate."""
    T = float(T)
    nB_N = float(nB_N)
    aQstar = float(aQstar)
    if NM_type != "PNM":
        raise RuntimeError("solve_front_energy_conserving_nK currently requires NM_type='PNM'")
    if T <= 0.0 or nB_N <= 0.0:
        raise RuntimeError("T and nB_N must be positive")
    if not (0.0 < aQstar < 1.0):
        raise RuntimeError("aQstar must satisfy 0 < aQstar < 1")
    if not (0.0 < float(tail_eps) < 1.0):
        raise RuntimeError("tail_eps must satisfy 0 < tail_eps < 1")
    if int(n_mesh) < 5 or int(max_nodes) <= int(n_mesh) or float(tol_bvp) <= 0.0:
        raise RuntimeError("invalid BVP mesh or tolerance settings")
    if compact_scale is not None and ((not np.isfinite(compact_scale)) or compact_scale <= 0.0):
        raise RuntimeError("compact_scale must be positive and finite")

    upB = 5000
    P_N = float(PNM_n(nB_N, T, param=param, NM_type=NM_type))
    e_N = float(edensNM_n(nB_N, T, param=param))
    h_N = float(P_N + e_N)
    nuclear_state = {
        "P_N": P_N,
        "e_N": e_N,
        "h_N": h_N,
        "nB_N": nB_N,
        "h_over_nB_N": float(h_N / nB_N),
        "T_N": T,
    }
    if jB_guess is None:
        jB_guess = max(1.0e-12, 1.0e-8 * nB_N)
    jB_guess = float(jB_guess)
    if jB_guess <= 0.0:
        raise RuntimeError("jB_guess must be positive")

    bounded_jB = jB_bounds is not None
    if bounded_jB:
        if len(jB_bounds) != 2:
            raise RuntimeError("jB_bounds must be a 2-tuple")
        jB_lo, jB_hi = map(float, jB_bounds)
        if not (0.0 < jB_lo < jB_hi):
            raise RuntimeError("jB_bounds must satisfy 0 < lower < upper")
        jB_guess = float(np.clip(jB_guess, jB_lo * (1.0 + 1.0e-8), jB_hi * (1.0 - 1.0e-8)))
    else:
        jB_lo, jB_hi = 0.0, np.inf

    def param_from_jB(value):
        if bounded_jB:
            frac = np.clip((value - jB_lo) / (jB_hi - jB_lo), 1.0e-12, 1.0 - 1.0e-12)
            return float(np.log(frac / (1.0 - frac)))
        return float(np.log(value))

    def jB_from_param(theta):
        if bounded_jB:
            sig = 1.0 / (1.0 + np.exp(-np.clip(float(theta), -60.0, 60.0)))
            return float(jB_lo + (jB_hi - jB_lo) * sig)
        return float(np.exp(np.clip(float(theta), -700.0, 700.0)))

    stats = {
        "bvp_ode_calls": 0,
        "bvp_bc_calls": 0,
        "local_state_calls": 0,
        "local_root_calls": 0,
        "qstar_root_calls": 0,
        "downstream_root_calls": 0,
    }
    state_cache = {}
    downstream_cache = {}
    last_failure = {"message": ""}

    def build_state(theta):
        key = round(float(theta), 12)
        if key in state_cache:
            return state_cache[key]
        jB = jB_from_param(theta)
        u_N = float(jB / nB_N)
        r_gamma_N = _relativistic_gamma_from_u(u_N)
        state = {
            "jB": jB,
            "u_N": u_N,
            "r_gamma_N": r_gamma_N,
            "Pi": float(h_N * u_N * u_N + P_N),
            "E": float(h_N * u_N * r_gamma_N),
        }
        state_cache[key] = state
        return state

    def downstream_target(theta):
        key = round(float(theta), 12)
        if key in downstream_cache:
            return downstream_cache[key]
        state = build_state(theta)
        stats["downstream_root_calls"] += 1
        endpoint = _solve_analytic_downstream_endpoint_for_uN(
            state["u_N"],
            nuclear_state,
            B_one_forth,
            ms=ms,
            upB=upB,
        )
        thermo = _quark_thermo_state(
            endpoint["muB_Q"],
            0.0,
            B_one_forth,
            endpoint["T_Q"],
            state["jB"],
            ms=ms,
            upB=upB,
        )
        micro = _microphysics_from_quark_state_energy(thermo["muB"], thermo["T"])
        if micro["invD"] <= 0.0:
            raise RuntimeError("downstream inverse diffusion coefficient must be positive")
        D_Q = float(1.0 / micro["invD"])
        delta_nK = float(max(1.0e-5 * max(abs(thermo["nK"]), thermo["nB"]), 1.0e-2))
        probe = _solve_local_quark_state_from_nK_E_and_Pi(
            thermo["nK"] + delta_nK,
            state["E"],
            state["Pi"],
            state["jB"],
            B_one_forth,
            ms=ms,
            upB=upB,
            initial_guess=(thermo["muB"], 1.0e-3, thermo["T"]),
            T_ref=thermo["T"],
            stats=stats,
        )
        rate_probe = _exact_kaon_transport_rate(
            probe["muB"], probe["muK"], probe["T"], ms=ms, upB=upB
        )["Gamma_K"]
        rate_slope = float(rate_probe / delta_nK)
        if (not np.isfinite(rate_slope)) or rate_slope <= 0.0:
            raise RuntimeError("downstream exact-rate slope must be positive")
        u_Q = float(thermo["u"])
        lam = float((-u_Q + np.sqrt(u_Q * u_Q + 4.0 * D_Q * rate_slope)) / (2.0 * D_Q))
        target = {
            **thermo,
            "D": D_Q,
            "invD": float(micro["invD"]),
            "rate_slope": rate_slope,
            "lambda": lam,
            "jK": float(u_Q * thermo["nK"]),
            "a": float(thermo["nK"] / thermo["nB"]),
        }
        downstream_cache[key] = target
        return target

    theta0 = param_from_jB(jB_guess)
    state0 = build_state(theta0)
    qstar0 = _solve_interface_state_from_local_a_E_and_Pi(
        aQstar,
        state0["E"],
        state0["Pi"],
        state0["jB"],
        B_one_forth,
        ms=ms,
        upB=upB,
        initial_guess=(1200.0, max(40.0, 300.0 * aQstar), T),
        stats=stats,
    )
    micro_qstar0 = _microphysics_from_quark_state_energy(qstar0["muB"], qstar0["T"])
    rate_qstar0 = _exact_kaon_transport_rate(
        qstar0["muB"], qstar0["muK"], qstar0["T"], ms=ms, upB=upB
    )["Gamma_K"]
    if compact_scale is None:
        D_qstar0 = 1.0 / micro_qstar0["invD"]
        compact_scale_used = float(np.sqrt(D_qstar0 * max(abs(qstar0["nK"]), 1.0) / max(abs(rate_qstar0), _FLOAT_TINY)))
    else:
        compact_scale_used = float(compact_scale)
    if (not np.isfinite(compact_scale_used)) or compact_scale_used <= 0.0:
        raise RuntimeError("failed to construct a positive compact scale")

    s_end = float(1.0 - tail_eps)
    s_mesh = np.linspace(0.0, s_end, int(n_mesh))
    blend = s_mesh / s_end
    tail_shape = np.maximum(1.0 - blend, 0.0)
    nK_guess = qstar0["nK"] * tail_shape
    jK_guess_profile = state0["jB"] * tail_shape
    if isinstance(profile_guess, dict):
        try:
            prev_s = np.asarray(profile_guess["s_coord"], dtype=float)
            prev_nK = np.asarray(profile_guess["nK"], dtype=float)
            prev_jK = np.asarray(profile_guess["jK"], dtype=float)
            prev_nK_Q = float(profile_guess.get("nK_Q", 0.0))
            if prev_s.ndim == 1 and prev_nK.shape == prev_s.shape and prev_jK.shape == prev_s.shape:
                prev_delta0 = float(prev_nK[0] - prev_nK_Q)
                scale = float((qstar0["nK"] - prev_nK_Q) / prev_delta0) if abs(prev_delta0) > 1.0e-12 else 1.0
                nK_guess = prev_nK_Q + scale * np.interp(s_mesh, prev_s, prev_nK - prev_nK_Q)
                jK_guess_profile = np.interp(s_mesh, prev_s, prev_jK)
                jK_guess_profile += (state0["jB"] - jK_guess_profile[0]) * (1.0 - blend)
                nK_guess[0] = qstar0["nK"]
                jK_guess_profile[0] = state0["jB"]
        except Exception:
            nK_guess = qstar0["nK"] * tail_shape
            jK_guess_profile = state0["jB"] * tail_shape
    y_guess = np.vstack((nK_guess, jK_guess_profile))

    def ode(s_coord, y, p):
        stats["bvp_ode_calls"] += 1
        state = build_state(float(p[0]))
        dyds = np.empty_like(y)
        guess = (qstar0["muB"], qstar0["muK"], qstar0["T"])
        for i in range(y.shape[1]):
            try:
                thermo = _solve_local_quark_state_from_nK_E_and_Pi(
                    float(y[0, i]),
                    state["E"],
                    state["Pi"],
                    state["jB"],
                    B_one_forth,
                    ms=ms,
                    upB=upB,
                    initial_guess=guess,
                    T_ref=T,
                    stats=stats,
                )
                guess = (thermo["muB"], thermo["muK"], thermo["T"])
                micro = _microphysics_from_quark_state_energy(thermo["muB"], thermo["T"])
                rate = _exact_kaon_transport_rate(
                    thermo["muB"], thermo["muK"], thermo["T"], ms=ms, upB=upB
                )["Gamma_K"]
                dx_ds = compact_scale_used / max(1.0 - float(s_coord[i]), np.finfo(float).tiny)
                dyds[0, i] = (thermo["u"] * float(y[0, i]) - float(y[1, i])) * micro["invD"] * dx_ds
                dyds[1, i] = -rate * dx_ds
            except Exception as exc:
                last_failure["message"] = str(exc)
                dyds[:, i] = 1.0e12
        return dyds

    def bc(ya, yb, p):
        stats["bvp_bc_calls"] += 1
        try:
            state = build_state(float(p[0]))
            left = _solve_local_quark_state_from_nK_E_and_Pi(
                float(ya[0]),
                state["E"],
                state["Pi"],
                state["jB"],
                B_one_forth,
                ms=ms,
                upB=upB,
                initial_guess=(qstar0["muB"], qstar0["muK"], qstar0["T"]),
                T_ref=qstar0["T"],
                stats=stats,
            )
            target = downstream_target(float(p[0]))
            tail = _nK_tail_residual(yb, target)
            j_scale = max(abs(state["jB"]), abs(target["jK"]), 1.0)
            nK_scale = max(abs(left["nK"]), abs(target["nK"]), 1.0)
            return np.array(
                [
                    (left["nK"] / left["nB"] - aQstar) / max(abs(aQstar), 1.0e-6),
                    (float(ya[1]) - state["jB"]) / j_scale,
                    tail / max(j_scale, (target["D"] * target["lambda"] + abs(target["u"])) * nK_scale),
                ],
                dtype=float,
            )
        except Exception as exc:
            last_failure["message"] = str(exc)
            return np.full(3, 1.0e12, dtype=float)

    try:
        sol = solve_bvp(
            ode,
            bc,
            s_mesh,
            y_guess,
            p=np.array([theta0], dtype=float),
            tol=tol_bvp,
            bc_tol=tol_bvp,
            max_nodes=max_nodes,
            verbose=2 if verb == "full" else 0,
        )
        state = build_state(float(sol.p[0]))
        target = downstream_target(float(sol.p[0]))
        bc_residual = bc(sol.y[:, 0], sol.y[:, -1], sol.p)
    except Exception as exc:
        return {
            "success": False,
            "message": f"absolute-nK energy BVP failed: {exc}; last failure: {last_failure['message']}",
            "aQstar": aQstar,
            "rate_model": "exact_nonleptonic",
            "composition_definition": "nK_over_local_nB",
            "current_definition": "u_nK_minus_D_dnK_dx",
            "_root_method": "solve_bvp_nK_parameter_1d",
        }

    s_profile = np.linspace(0.0, s_end, max(int(n_mesh), 200))
    y_profile = np.asarray(sol.sol(s_profile), dtype=float)
    x_profile = -compact_scale_used * np.log1p(-s_profile)
    profile = {key: np.empty_like(s_profile) for key in (
        "nB", "u", "muB", "muK", "T", "P", "h", "r_gamma", "invD", "mu_u", "mu_d", "mu_s", "Gamma_K"
    )}
    guess = (qstar0["muB"], qstar0["muK"], qstar0["T"])
    for i, nK_val in enumerate(y_profile[0]):
        thermo = _solve_local_quark_state_from_nK_E_and_Pi(
            float(nK_val),
            state["E"],
            state["Pi"],
            state["jB"],
            B_one_forth,
            ms=ms,
            upB=upB,
            initial_guess=guess,
            T_ref=target["T"],
            stats=stats,
        )
        guess = (thermo["muB"], thermo["muK"], thermo["T"])
        micro = _microphysics_from_quark_state_energy(thermo["muB"], thermo["T"])
        rate = _exact_kaon_transport_rate(thermo["muB"], thermo["muK"], thermo["T"], ms=ms, upB=upB)
        profile["nB"][i] = thermo["nB"]
        profile["u"][i] = thermo["u"]
        profile["muB"][i] = thermo["muB"]
        profile["muK"][i] = thermo["muK"]
        profile["T"][i] = thermo["T"]
        profile["P"][i] = thermo["P"]
        profile["h"][i] = thermo["h"]
        profile["r_gamma"][i] = _relativistic_gamma_from_u(thermo["u"])
        profile["invD"][i] = micro["invD"]
        profile["mu_u"][i] = rate["mu_u"]
        profile["mu_d"][i] = rate["mu_d"]
        profile["mu_s"][i] = rate["mu_s"]
        profile["Gamma_K"][i] = rate["Gamma_K"]

    dy_ds = np.asarray(sol.sol.derivative(1)(s_profile), dtype=float)
    dx_ds = compact_scale_used / np.maximum(1.0 - s_profile, np.finfo(float).tiny)
    jK_prime = dy_ds[1] / dx_ds
    reaction_scale = max(float(np.mean(np.abs(profile["Gamma_K"]))), _FLOAT_TINY)
    kaon_residual = jK_prime + profile["Gamma_K"]
    kaon_residual_norm = float(np.mean(np.abs(kaon_residual)) / reaction_scale)
    a_profile = y_profile[0] / profile["nB"]
    energy_profile = profile["h"] * profile["u"] * profile["r_gamma"]
    momentum_profile = profile["h"] * profile["u"] ** 2 + profile["P"]
    energy_residual_norm = float(np.max(np.abs(energy_profile - state["E"])) / max(abs(state["E"]), 1.0))
    momentum_residual_norm = float(np.max(np.abs(momentum_profile - state["Pi"])) / max(abs(state["Pi"]), 1.0))
    success = bool(sol.success and np.max(np.abs(bc_residual)) <= max(float(tol_bvp), 1.0e-10) and abs(kaon_residual_norm) <= 5.0 * float(tol_bvp))
    result = {
        "success": success,
        "message": "Absolute-nK energy-conserving BVP converged" if success else f"{sol.message}; last failure: {last_failure['message']}",
        "aQstar": aQstar,
        "aQstar_derived": float(a_profile[0]),
        "jB": float(state["jB"]),
        "u_N": float(state["u_N"]),
        "Pi": float(state["Pi"]),
        "E": float(state["E"]),
        "T_Q": float(target["T"]),
        "nB_Q": float(target["nB"]),
        "nK_Q": float(target["nK"]),
        "a_Q": float(target["a"]),
        "lambda_Q": float(target["lambda"]),
        "compact_scale": compact_scale_used,
        "tail_eps": float(tail_eps),
        "boundary_residuals": np.asarray(bc_residual, dtype=float),
        "tail_residual_norm": float(bc_residual[2]),
        "kaon_equation_residual_norm": kaon_residual_norm,
        "energy_equation_residual_norm": energy_residual_norm,
        "momentum_equation_residual_norm": momentum_residual_norm,
        "bvp_status": int(sol.status),
        "bvp_message": str(sol.message),
        "bvp_niter": int(sol.niter),
        "bvp_nodes": int(sol.x.size),
        "bvp_parameters": np.asarray(sol.p, dtype=float),
        "rate_model": "exact_nonleptonic",
        "composition_definition": "nK_over_local_nB",
        "current_definition": "u_nK_minus_D_dnK_dx",
        "_root_method": "solve_bvp_nK_parameter_1d",
        **{key: int(value) for key, value in stats.items()},
    }
    if return_profile:
        result.update(
            {
                "x": x_profile,
                "s_coord": s_profile,
                "nK": y_profile[0],
                "jK": y_profile[1],
                "a": a_profile,
                "nB": profile["nB"],
                "u": profile["u"],
                "muB": profile["muB"],
                "muK": profile["muK"],
                "T_profile": profile["T"],
                "P_profile": profile["P"],
                "h_profile": profile["h"],
                "r_gamma_profile": profile["r_gamma"],
                "invD_profile": profile["invD"],
                "mu_u_profile": profile["mu_u"],
                "mu_d_profile": profile["mu_d"],
                "mu_s_profile": profile["mu_s"],
                "Gamma_K_profile": profile["Gamma_K"],
                "jK_prime_profile": jK_prime,
                "kaon_equation_residual_profile": kaon_residual,
            }
        )
    if return_raw_bvp_grid:
        result.update(
            {
                "bvp_s_coord": np.asarray(sol.x, dtype=float),
                "bvp_x": -compact_scale_used * np.log1p(-np.asarray(sol.x, dtype=float)),
                "bvp_nK": np.asarray(sol.y[0], dtype=float),
                "bvp_jK": np.asarray(sol.y[1], dtype=float),
                "bvp_a": np.interp(np.asarray(sol.x, dtype=float), s_profile, a_profile),
                "bvp_nB": np.interp(np.asarray(sol.x, dtype=float), s_profile, profile["nB"]),
                "bvp_u": np.interp(np.asarray(sol.x, dtype=float), s_profile, profile["u"]),
                "bvp_muB": np.interp(np.asarray(sol.x, dtype=float), s_profile, profile["muB"]),
                "bvp_muK": np.interp(np.asarray(sol.x, dtype=float), s_profile, profile["muK"]),
                "bvp_T_profile": np.interp(np.asarray(sol.x, dtype=float), s_profile, profile["T"]),
                "bvp_mu_u_profile": np.interp(np.asarray(sol.x, dtype=float), s_profile, profile["mu_u"]),
                "bvp_mu_d_profile": np.interp(np.asarray(sol.x, dtype=float), s_profile, profile["mu_d"]),
                "bvp_mu_s_profile": np.interp(np.asarray(sol.x, dtype=float), s_profile, profile["mu_s"]),
                "bvp_Gamma_K_profile": np.interp(np.asarray(sol.x, dtype=float), s_profile, profile["Gamma_K"]),
            }
        )
    return result


def solve_front_energy_conserving_nK(
    T,
    nB_N,
    B_one_forth,
    aQstar,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    compact_scale=None,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    jB_bounds=None,
    return_profile=False,
    return_raw_bvp_grid=False,
    verb=False,
):
    """Public absolute-nK solver with staged aQstar continuation fallback."""
    direct = _solve_front_energy_conserving_nK_once(
        T,
        nB_N,
        B_one_forth,
        aQstar,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        compact_scale=compact_scale,
        n_mesh=n_mesh,
        tol_bvp=tol_bvp,
        max_nodes=max_nodes,
        jB_guess=jB_guess,
        jB_bounds=jB_bounds,
        return_profile=return_profile,
        return_raw_bvp_grid=return_raw_bvp_grid,
        verb=verb,
    )
    if bool(direct.get("success")) or float(aQstar) <= 2.0e-2:
        direct["continuation_used"] = False
        direct["continuation_steps"] = 0
        return direct

    stage_targets = np.linspace(2.0e-2, float(aQstar), int(np.ceil((float(aQstar) - 2.0e-2) / 2.0e-2)) + 1)
    current = None
    for index, stage_a in enumerate(stage_targets):
        stage = _solve_front_energy_conserving_nK_once(
            T,
            nB_N,
            B_one_forth,
            float(stage_a),
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            compact_scale=compact_scale if current is None else current["compact_scale"],
            n_mesh=n_mesh,
            tol_bvp=tol_bvp,
            max_nodes=max_nodes,
            jB_guess=jB_guess if current is None else current["jB"],
            jB_bounds=jB_bounds,
            return_profile=True,
            return_raw_bvp_grid=bool(return_raw_bvp_grid and index == len(stage_targets) - 1),
            verb=False,
            profile_guess=current,
        )
        if not bool(stage.get("success")):
            failed = dict(stage)
            failed["message"] = f"{direct.get('message')}; continuation failed at aQstar={stage_a:.6g}: {stage.get('message')}"
            failed["continuation_used"] = True
            failed["continuation_steps"] = index
            return failed
        current = stage

    current["continuation_used"] = True
    current["continuation_steps"] = len(stage_targets)
    if not return_profile:
        for key in (
            "x", "s_coord", "nK", "jK", "a", "nB", "u", "muB", "muK",
            "T_profile", "P_profile", "h_profile", "r_gamma_profile", "invD_profile",
            "mu_u_profile", "mu_d_profile", "mu_s_profile", "Gamma_K_profile",
            "jK_prime_profile", "kaon_equation_residual_profile",
        ):
            current.pop(key, None)
    return current


def _uNmax_collocation_status_is_acceptable(
    *, solver_success, solver_status, exact_zero_left
):
    """Require SciPy's collocation solve itself to have converged."""
    return bool(solver_success)


def _solve_front_energy_conserving_uNmax_once(
    T,
    nB_N,
    B_one_forth,
    TQstar,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    TQ_guess=None,
    jB_bounds=None,
    return_profile=False,
    verb=False,
    continuation_guess=None,
):
    """Solve the fixed-TQstar energy front using absolute nK and physical jK."""
    T = float(T)
    nB_N = float(nB_N)
    TQstar = float(TQstar)
    if NM_type != "PNM":
        raise RuntimeError("solve_front_energy_conserving_uNmax currently requires NM_type='PNM'")
    if (not np.isfinite(T)) or T <= 0.0 or (not np.isfinite(nB_N)) or nB_N <= 0.0:
        raise RuntimeError("solve_front_energy_conserving_uNmax requires positive T and nB_N")
    if (not np.isfinite(TQstar)) or TQstar < 0.0:
        raise RuntimeError("TQstar must be non-negative")
    if not (0.0 < float(tail_eps) < 1.0):
        raise RuntimeError("tail_eps must satisfy 0 < tail_eps < 1")
    if int(n_mesh) < 5 or int(max_nodes) <= int(n_mesh) or float(tol_bvp) <= 0.0:
        raise RuntimeError("invalid BVP mesh or tolerance settings")

    upB = 5000
    P_N = float(PNM_n(nB_N, T, param=param, NM_type=NM_type))
    e_N = float(edensNM_n(nB_N, T, param=param))
    h_N = float(P_N + e_N)
    nuclear_state = {
        "P_N": P_N,
        "e_N": e_N,
        "h_N": h_N,
        "nB_N": nB_N,
        "h_over_nB_N": float(h_N / nB_N),
        "T_N": T,
    }
    if jB_guess is None:
        jB_guess = float(max(1.0e-12, 1.0e-8 * nB_N, 1.0))
    jB_guess = float(jB_guess)
    if (not np.isfinite(jB_guess)) or jB_guess <= 0.0:
        raise RuntimeError("jB_guess must be positive")
    endpoint_initial_guess = None
    if isinstance(continuation_guess, dict):
        try:
            continued_muB_Q = float(continuation_guess.get("muB_Q", np.nan))
            continued_T_Q = float(continuation_guess.get("T_Q", np.nan))
        except Exception:
            continued_muB_Q = np.nan
            continued_T_Q = np.nan
        if (
            np.isfinite(continued_muB_Q)
            and continued_muB_Q > 0.0
            and np.isfinite(continued_T_Q)
            and continued_T_Q > 0.0
        ):
            endpoint_initial_guess = (continued_muB_Q, continued_T_Q)
    if endpoint_initial_guess is None and TQ_guess is not None:
        try:
            TQ_guess_value = float(TQ_guess)
        except Exception:
            TQ_guess_value = np.nan
        if np.isfinite(TQ_guess_value) and TQ_guess_value > 0.0:
            endpoint_initial_guess = (1100.0, TQ_guess_value)

    bounded_jB = jB_bounds is not None
    if bounded_jB:
        if len(jB_bounds) != 2:
            raise RuntimeError("jB_bounds must be a 2-tuple")
        jB_lo, jB_hi = map(float, jB_bounds)
        if not (0.0 < jB_lo < jB_hi):
            raise RuntimeError("jB_bounds must satisfy 0 < lower < upper")
        jB_guess = float(np.clip(jB_guess, jB_lo * (1.0 + 1.0e-8), jB_hi * (1.0 - 1.0e-8)))
    else:
        jB_lo, jB_hi = 0.0, np.inf

    def param_from_jB(value):
        if bounded_jB:
            frac = np.clip((float(value) - jB_lo) / (jB_hi - jB_lo), 1.0e-12, 1.0 - 1.0e-12)
            return float(np.log(frac / (1.0 - frac)))
        return float(np.log(float(value)))

    def jB_from_param(theta):
        if bounded_jB:
            sig = 1.0 / (1.0 + np.exp(-np.clip(float(theta), -60.0, 60.0)))
            return float(jB_lo + (jB_hi - jB_lo) * sig)
        return float(np.exp(np.clip(float(theta), -700.0, 700.0)))

    TQ_min = 1.0e-6

    stats = {
        "bvp_ode_calls": 0,
        "bvp_bc_calls": 0,
        "q_root_calls": 0,
        "qstar_root_calls": 0,
        "downstream_root_calls": 0,
        "local_state_calls": 0,
        "local_root_calls": 0,
        "profile_state_calls": 0,
        "global_state_builds": 0,
        "global_state_failures": 0,
        "local_state_failures": 0,
    }
    state_cache = {}
    last_failure = {"message": ""}
    endpoint_guess_cache = {"value": endpoint_initial_guess}
    qstar_initial_guess = None
    if isinstance(continuation_guess, dict):
        try:
            continued_muB_Qstar = float(continuation_guess.get("muB_Qstar", np.nan))
            continued_muK_Qstar = float(continuation_guess.get("muK_Qstar", np.nan))
            continued_T_Qstar = float(continuation_guess.get("T_Qstar", TQstar))
        except Exception:
            continued_muB_Qstar = np.nan
            continued_muK_Qstar = np.nan
            continued_T_Qstar = np.nan
        if (
            np.isfinite(continued_muB_Qstar)
            and continued_muB_Qstar > 0.0
            and np.isfinite(continued_muK_Qstar)
            and continued_muK_Qstar >= 0.0
            and np.isfinite(continued_T_Qstar)
            and continued_T_Qstar >= 0.0
        ):
            qstar_initial_guess = (
                continued_muB_Qstar,
                continued_muK_Qstar,
                max(continued_T_Qstar, TQ_min),
            )
    qstar_guess_cache = {"value": qstar_initial_guess}
    exact_zero_left = bool(TQstar == 0.0)
    s_end = float(1.0 - tail_eps)

    def build_state(theta):
        key = round(float(theta), 12)
        if key in state_cache:
            return state_cache[key]
        stats["global_state_builds"] += 1
        jB = jB_from_param(theta)
        u_N = float(jB / nB_N)
        r_gamma_N = _relativistic_gamma_from_u(u_N)
        E = float(h_N * u_N * r_gamma_N)
        Pi = float(h_N * u_N * u_N + P_N)

        stats["downstream_root_calls"] += 1
        endpoint = _solve_analytic_downstream_endpoint_for_uN(
            u_N,
            nuclear_state,
            B_one_forth,
            ms=ms,
            upB=upB,
            initial_guess=endpoint_guess_cache["value"],
        )
        endpoint_guess_cache["value"] = endpoint.get("endpoint_initial_guess", (endpoint["muB_Q"], endpoint["T_Q"]))
        T_Q = float(endpoint["T_Q"])
        thermo_Q = _quark_thermo_state(endpoint["muB_Q"], 0.0, B_one_forth, T_Q, jB, ms=ms, upB=upB)
        E_Q = float(thermo_Q["h"] * thermo_Q["u"] * _relativistic_gamma_from_u(thermo_Q["u"]))
        Pi_Q = float(thermo_Q["h"] * thermo_Q["u"] * thermo_Q["u"] + thermo_Q["P"])

        qstar_guess = qstar_guess_cache["value"]
        if qstar_guess is None:
            qstar_guess = (endpoint["muB_Q"], _branch_muK_seed((nB_N - thermo_Q["nK"]) / thermo_Q["nB"]), max(TQstar, T_Q))
        thermo_Qstar = _solve_interface_Qstar_from_TQstar_E_and_Pi(
            TQstar,
            E,
            Pi,
            jB,
            thermo_Q["nB"],
            thermo_Q["nK"],
            B_one_forth,
            ms=ms,
            upB=upB,
            initial_guess=qstar_guess,
            stats=stats,
        )
        qstar_guess_cache["value"] = (thermo_Qstar["muB"], thermo_Qstar["muK"], max(TQstar, TQ_min))
        aQstar = float(thermo_Qstar["nK"] / thermo_Qstar["nB"])
        if not (0.0 < aQstar < 1.0):
            raise RuntimeError("fixed-TQstar interface requires 0 < nK_Qstar/nB_Qstar < 1")

        micro_Q = _microphysics_from_quark_state_energy(thermo_Q["muB"], thermo_Q["T"])
        if micro_Q["invD"] <= 0.0:
            raise RuntimeError("uNmax downstream inverse diffusion coefficient must be positive")
        D_Q = float(1.0 / micro_Q["invD"])
        delta_nK = float(max(1.0e-5 * max(abs(thermo_Q["nK"]), thermo_Q["nB"]), 1.0e-2))
        probe = _solve_local_quark_state_from_nK_E_and_Pi(
            thermo_Q["nK"] + delta_nK,
            E,
            Pi,
            jB,
            B_one_forth,
            ms=ms,
            upB=upB,
            initial_guess=(thermo_Q["muB"], 1.0e-3, thermo_Q["T"]),
            T_ref=thermo_Q["T"],
            stats=stats,
        )
        rate_probe = _exact_kaon_transport_rate(
            probe["muB"], probe["muK"], probe["T"], ms=ms, upB=upB
        )["Gamma_K"]
        rate_slope = float(rate_probe / delta_nK)
        if (not np.isfinite(rate_slope)) or rate_slope <= 0.0:
            raise RuntimeError("uNmax downstream exact-rate slope must be positive")
        u_Q = float(thermo_Q["u"])
        lam = float((-u_Q + np.sqrt(u_Q * u_Q + 4.0 * D_Q * rate_slope)) / (2.0 * D_Q))
        if (not np.isfinite(lam)) or lam <= 0.0:
            raise RuntimeError("uNmax downstream tail decay must be positive")

        state = {
            "jB": jB,
            "u_N": u_N,
            "r_gamma_N": r_gamma_N,
            "E": E,
            "Pi": Pi,
            "T_Q": T_Q,
            "P_N": P_N,
            "e_N": e_N,
            "h_N": h_N,
            "h_over_nB_N": float(h_N / nB_N),
            "E_N": E,
            "thermo_Q": thermo_Q,
            "thermo_Qstar": thermo_Qstar,
            "E_Q": E_Q,
            "Pi_Q": Pi_Q,
            "endpoint": endpoint,
            "lambda_n": float(nB_N / thermo_Q["nB"]),
            "aQstar": aQstar,
            "D_Q": D_Q,
            "invD_Q": float(micro_Q["invD"]),
            "rate_slope_Q": rate_slope,
            "lambda": lam,
            "jK_Q": float(u_Q * thermo_Q["nK"]),
            "x_end": float(-np.log1p(-s_end) / lam),
        }
        state_cache[key] = state
        return state

    def state_or_none(theta):
        try:
            return build_state(theta)
        except Exception as exc:
            stats["global_state_failures"] += 1
            last_failure["message"] = str(exc)
            return None

    def local_state(nK_value, state, s_value=None, initial_guess=None):
        if exact_zero_left and (
            (s_value is not None and abs(float(s_value)) <= 1.0e-14)
            or float(nK_value) >= float(state["thermo_Qstar"]["nK"])
        ):
            return state["thermo_Qstar"]
        if exact_zero_left and float(nK_value) <= float(state["thermo_Q"]["nK"]):
            return state["thermo_Q"]
        return _solve_local_quark_state_from_nK_E_and_Pi(
            float(nK_value),
            state["E"],
            state["Pi"],
            state["jB"],
            B_one_forth,
            ms=ms,
            upB=upB,
            initial_guess=initial_guess,
            T_ref=state["T_Q"],
            stats=stats,
        )

    def ode(s_coord, y, p):
        stats["bvp_ode_calls"] += 1
        state = state_or_none(float(p[0]))
        if state is None:
            return np.full_like(y, 1.0e12)
        dyds = np.empty_like(y)
        interface_T_seed = max(TQstar, TQ_min) if exact_zero_left else max(TQstar, state["T_Q"])
        guess = (state["thermo_Qstar"]["muB"], state["thermo_Qstar"]["muK"], interface_T_seed)
        for i in range(y.shape[1]):
            try:
                nK_value = float(y[0, i]) * bvp_nK_scale
                jK_value = float(y[1, i]) * bvp_jK_scale
                thermo = local_state(nK_value, state, s_value=s_coord[i], initial_guess=guess)
                if float(thermo["T"]) > 0.0:
                    guess = (thermo["muB"], thermo["muK"], thermo["T"])
                micro = _microphysics_from_quark_state_energy(
                    thermo["muB"],
                    thermo["T"],
                    allow_zero_temperature=exact_zero_left and float(thermo["T"]) == 0.0,
                )
                rate = _exact_kaon_transport_rate(
                    thermo["muB"], thermo["muK"], thermo["T"], ms=ms, upB=upB
                )["Gamma_K"]
                dx_ds = 1.0 / (state["lambda"] * max(1.0 - float(s_coord[i]), np.finfo(float).tiny))
                dyds[0, i] = (thermo["u"] * nK_value - jK_value) * micro["invD"] * dx_ds / bvp_nK_scale
                dyds[1, i] = -rate * dx_ds / bvp_jK_scale
            except Exception as exc:
                stats["local_state_failures"] += 1
                last_failure["message"] = str(exc)
                dyds[:, i] = 1.0e12
        return dyds

    def bc(ya, yb, p):
        stats["bvp_bc_calls"] += 1
        state = state_or_none(float(p[0]))
        if state is None:
            return np.full(3, 1.0e12, dtype=float)
        qstar = state["thermo_Qstar"]
        target = state["thermo_Q"]
        ya_physical = np.array([float(ya[0]) * bvp_nK_scale, float(ya[1]) * bvp_jK_scale])
        yb_physical = np.array([float(yb[0]) * bvp_nK_scale, float(yb[1]) * bvp_jK_scale])
        nK_scale = max(abs(qstar["nK"]), abs(target["nK"]), 1.0)
        jK_scale = max(abs(state["jB"]), abs(state["jK_Q"]), 1.0)
        tail_scale = max(jK_scale, (state["D_Q"] * state["lambda"] + abs(target["u"])) * nK_scale)
        return np.array(
            [
                (float(ya_physical[0]) - qstar["nK"]) / nK_scale,
                (float(ya_physical[1]) - state["jB"]) / jK_scale,
                _nK_tail_residual(
                    yb_physical,
                    {
                        "nK": target["nK"],
                        "jK": state["jK_Q"],
                        "D": state["D_Q"],
                        "lambda": state["lambda"],
                        "u": target["u"],
                    },
                )
                / tail_scale,
            ],
            dtype=float,
        )

    theta0 = param_from_jB(jB_guess)
    state0 = state_or_none(theta0)
    if state0 is None:
        return {
            "success": False,
            "message": f"Initial uNmax nK state construction failed: {last_failure['message']}",
            "T_Qstar": TQstar,
            "solver_variant": "uNmax_direct_TQstar_nK",
            "rate_model": "exact_nonleptonic",
            "composition_definition": "nK_over_local_nB",
            "current_definition": "u_nK_minus_D_dnK_dx",
            "density_ratio_definition": "lambda_n_equals_nB_N_over_nB_Q",
            "_root_method": "solve_bvp_uNmax_nK_parameter_1d",
        }

    bvp_nK_scale = max(abs(state0["thermo_Qstar"]["nK"]), abs(state0["thermo_Q"]["nK"]), 1.0)
    bvp_jK_scale = max(abs(state0["jB"]), abs(state0["jK_Q"]), 1.0)

    continuation_profile = None
    if isinstance(continuation_guess, dict):
        try:
            previous_s = np.asarray(continuation_guess["s_coord"], dtype=float)
            previous_nK = np.asarray(continuation_guess["nK"], dtype=float)
            previous_jK = np.asarray(continuation_guess["jK"], dtype=float)
            previous_nK_Q = float(continuation_guess["nK_Q"])
            if (
                previous_s.ndim == 1
                and previous_s.size >= 2
                and previous_nK.shape == previous_s.shape
                and previous_jK.shape == previous_s.shape
                and np.all(np.isfinite(previous_nK))
                and np.all(np.isfinite(previous_jK))
                and np.all(np.diff(previous_s) > 0.0)
            ):
                continuation_profile = (
                    previous_s,
                    previous_nK,
                    previous_jK,
                    previous_nK_Q,
                )
        except Exception:
            continuation_profile = None

    s_mesh = np.linspace(0.0, s_end, int(n_mesh))
    blend = s_mesh / s_end
    if exact_zero_left:
        tail_weight = tail_eps + (1.0 - tail_eps) * (1.0 - 3.0 * blend**2 + 2.0 * blend**3)
    else:
        tail_weight = np.maximum(1.0 - s_mesh, tail_eps)
    qstar0 = state0["thermo_Qstar"]
    target0 = state0["thermo_Q"]
    nK_guess = target0["nK"] + (qstar0["nK"] - target0["nK"]) * tail_weight
    jK_tail_weight = np.maximum(1.0 - s_mesh, tail_eps)
    jK_guess = state0["jK_Q"] + (state0["jB"] - state0["jK_Q"]) * jK_tail_weight
    nK_guess[0] = qstar0["nK"]
    jK_guess[0] = state0["jB"]
    if continuation_profile is not None:
        previous_s, previous_nK, previous_jK, previous_nK_Q = continuation_profile
        previous_delta = previous_nK - previous_nK_Q
        previous_left_delta = float(previous_delta[0])
        new_left_delta = float(qstar0["nK"] - target0["nK"])
        profile_scale = (
            new_left_delta / previous_left_delta
            if abs(previous_left_delta) > 1.0e-12
            else 1.0
        )
        nK_guess = target0["nK"] + profile_scale * np.interp(
            s_mesh,
            previous_s,
            previous_delta,
        )
        jK_guess = np.interp(s_mesh, previous_s, previous_jK)
        jK_guess += (state0["jB"] - jK_guess[0]) * (1.0 - blend)
        nK_guess[0] = qstar0["nK"]
        jK_guess[0] = state0["jB"]

    try:
        sol = solve_bvp(
            ode,
            bc,
            s_mesh,
            np.vstack((nK_guess / bvp_nK_scale, jK_guess / bvp_jK_scale)),
            p=np.array([theta0], dtype=float),
            tol=tol_bvp,
            bc_tol=tol_bvp,
            max_nodes=max_nodes,
            verbose=2 if verb == "full" else 0,
        )
        state = build_state(float(sol.p[0]))
        bc_residuals = bc(sol.y[:, 0], sol.y[:, -1], sol.p)
    except Exception as exc:
        return {
            "success": False,
            "message": f"absolute-nK uNmax BVP failed: {exc}; last failure: {last_failure['message']}",
            "T_Qstar": TQstar,
            "solver_variant": "uNmax_direct_TQstar_nK",
            "rate_model": "exact_nonleptonic",
            "composition_definition": "nK_over_local_nB",
            "current_definition": "u_nK_minus_D_dnK_dx",
            "density_ratio_definition": "lambda_n_equals_nB_N_over_nB_Q",
            "_root_method": "solve_bvp_uNmax_nK_parameter_1d",
        }

    s_profile = np.asarray(sol.x, dtype=float)
    nK_profile = np.asarray(sol.y[0], dtype=float) * bvp_nK_scale
    jK_profile = np.asarray(sol.y[1], dtype=float) * bvp_jK_scale
    x_profile = -np.log1p(-s_profile) / state["lambda"]
    profile = {key: np.empty_like(s_profile) for key in (
        "nB", "u", "muB", "muK", "T", "P", "h", "r_gamma", "invD", "Gamma_K"
    )}
    interface_T_seed = max(TQstar, TQ_min) if exact_zero_left else max(TQstar, state["T_Q"])
    guess = (state["thermo_Qstar"]["muB"], state["thermo_Qstar"]["muK"], interface_T_seed)
    try:
        for i, nK_value in enumerate(nK_profile):
            stats["profile_state_calls"] += 1
            thermo = local_state(nK_value, state, s_value=s_profile[i], initial_guess=guess)
            if float(thermo["T"]) > 0.0:
                guess = (thermo["muB"], thermo["muK"], thermo["T"])
            micro = _microphysics_from_quark_state_energy(
                thermo["muB"], thermo["T"], allow_zero_temperature=exact_zero_left and float(thermo["T"]) == 0.0
            )
            rate = _exact_kaon_transport_rate(thermo["muB"], thermo["muK"], thermo["T"], ms=ms, upB=upB)
            profile["nB"][i] = thermo["nB"]
            profile["u"][i] = thermo["u"]
            profile["muB"][i] = thermo["muB"]
            profile["muK"][i] = thermo["muK"]
            profile["T"][i] = thermo["T"]
            profile["P"][i] = thermo["P"]
            profile["h"][i] = thermo["h"]
            profile["r_gamma"][i] = _relativistic_gamma_from_u(thermo["u"])
            profile["invD"][i] = micro["invD"]
            profile["Gamma_K"][i] = rate["Gamma_K"]
    except Exception as exc:
        return {
            "success": False,
            "message": f"uNmax nK profile reconstruction failed: {exc}",
            "T_Qstar": TQstar,
            "solver_variant": "uNmax_direct_TQstar_nK",
            "rate_model": "exact_nonleptonic",
            "composition_definition": "nK_over_local_nB",
            "current_definition": "u_nK_minus_D_dnK_dx",
            "density_ratio_definition": "lambda_n_equals_nB_N_over_nB_Q",
            "_root_method": "solve_bvp_uNmax_nK_parameter_1d",
        }

    dy_ds = _bvp_dense_derivative(sol, s_profile)
    dx_ds = 1.0 / (state["lambda"] * np.maximum(1.0 - s_profile, np.finfo(float).tiny))
    jK_prime = np.asarray(dy_ds[1], dtype=float) * bvp_jK_scale / dx_ds
    kaon_residual = jK_prime + profile["Gamma_K"]
    reaction_scale = max(float(np.mean(np.abs(profile["Gamma_K"]))), _FLOAT_TINY)
    kaon_residual_norm = float(np.mean(np.abs(kaon_residual)) / reaction_scale)
    a_profile = nK_profile / profile["nB"]
    energy_profile = profile["h"] * profile["u"] * profile["r_gamma"]
    momentum_profile = profile["h"] * profile["u"] ** 2 + profile["P"]
    energy_residual_norm = float(np.max(np.abs(energy_profile - state["E"])) / max(abs(state["E"]), 1.0))
    momentum_residual_norm = float(np.max(np.abs(momentum_profile - state["Pi"])) / max(abs(state["Pi"]), 1.0))
    TQstar_residual = float(state["thermo_Qstar"]["T"] - TQstar)
    E_right_residual = float(state["E_Q"] - state["E"])
    E_right_residual_norm = float(E_right_residual / max(abs(state["E"]), abs(state["E_Q"]), 1.0))
    Pi_right_residual = float(state["Pi_Q"] - state["Pi"])
    Pi_right_residual_norm = float(Pi_right_residual / max(abs(state["Pi"]), abs(state["Pi_Q"]), 1.0))
    collocation_status_acceptable = _uNmax_collocation_status_is_acceptable(
        solver_success=sol.success,
        solver_status=sol.status,
        exact_zero_left=exact_zero_left,
    )
    success = bool(
        collocation_status_acceptable
        and np.max(np.abs(bc_residuals)) <= max(float(tol_bvp), 1.0e-10)
        and kaon_residual_norm <= 5.0 * float(tol_bvp)
        and abs(E_right_residual_norm) <= max(float(tol_bvp), 1.0e-8)
        and abs(Pi_right_residual_norm) <= max(float(tol_bvp), 1.0e-8)
        and abs(TQstar_residual) <= max(1.0e-10, 1.0e-8 * max(1.0, abs(TQstar)))
    )

    target = state["thermo_Q"]
    qstar = state["thermo_Qstar"]
    jK_left_residual = float(jK_profile[0] - state["jB"])
    jK_left_scale = max(abs(state["jB"]), abs(state["jK_Q"]), 1.0)
    tail_residual = _nK_tail_residual(
        np.array([nK_profile[-1], jK_profile[-1]], dtype=float),
        {
            "nK": target["nK"],
            "jK": state["jK_Q"],
            "D": state["D_Q"],
            "lambda": state["lambda"],
            "u": target["u"],
        },
    )
    accepted_max_nodes = bool(success and not sol.success and int(sol.status) == 1)
    if success and accepted_max_nodes:
        result_message = (
            "Absolute-nK uNmax BVP reached max_nodes at the exact TQstar=0 "
            "endpoint and passed all physical residual checks"
        )
    elif success:
        result_message = "Absolute-nK uNmax BVP converged"
    else:
        result_message = f"{sol.message}; last failure: {last_failure['message']}"
    result = {
        "success": success,
        "message": result_message,
        "jB": float(state["jB"]),
        "u_N": float(state["u_N"]),
        "u_N_max_candidate": float(state["u_N"]),
        "T_N": T,
        "T_Q": float(state["T_Q"]),
        "T_Qstar": float(qstar["T"]),
        "T_Qstar_target": TQstar,
        "T_Qstar_residual": TQstar_residual,
        "E": float(state["E"]),
        "Pi": float(state["Pi"]),
        "E_N": float(state["E_N"]),
        "E_Q": float(state["E_Q"]),
        "E_Qstar": float(qstar["E"]),
        "E_right_residual": E_right_residual,
        "E_right_residual_norm": E_right_residual_norm,
        "Pi_Q": float(state["Pi_Q"]),
        "Pi_right_residual": Pi_right_residual,
        "Pi_right_residual_norm": Pi_right_residual_norm,
        "aQstar": float(a_profile[0]),
        "aQstar_derived": float(a_profile[0]),
        "lambda_n": float(state["lambda_n"]),
        "u_Q": float(target["u"]),
        "r_gamma_N": float(state["r_gamma_N"]),
        "r_gamma_Q": float(_relativistic_gamma_from_u(target["u"])),
        "r_gamma_Qstar": float(_relativistic_gamma_from_u(qstar["u"])),
        "muB_Q": float(target["muB"]),
        "nB_Q": float(target["nB"]),
        "nK_Q": float(target["nK"]),
        "muB_Qstar": float(qstar["muB"]),
        "muK_Qstar": float(qstar["muK"]),
        "nB_Qstar": float(qstar["nB"]),
        "nK_Qstar": float(qstar["nK"]),
        "h_N": h_N,
        "h_Q": float(target["h"]),
        "h_Qstar": float(qstar["h"]),
        "h_over_nB_N": float(h_N / nB_N),
        "h_over_nB_Qstar": float(qstar["h"] / qstar["nB"]),
        "h_over_nB_jump_residual": float(qstar["h"] / qstar["nB"] - h_N / nB_N),
        "invD_Q": float(state["invD_Q"]),
        "rate_slope_Q": float(state["rate_slope_Q"]),
        "lambda": float(state["lambda"]),
        "kappa": float(1.0 / state["lambda"]),
        "s_end": s_end,
        "x_end": float(state["x_end"]),
        "nK_end": float(nK_profile[-1]),
        "jK_end": float(jK_profile[-1]),
        "a_end": float(a_profile[-1]),
        "jK_left_residual": jK_left_residual,
        "jK_left_residual_norm": float(jK_left_residual / jK_left_scale),
        "jK_left_scale": float(jK_left_scale),
        "tail_residual": float(tail_residual),
        "tail_residual_norm": float(bc_residuals[2]),
        "boundary_residuals": np.asarray(bc_residuals, dtype=float),
        "kaon_equation_residual_norm": kaon_residual_norm,
        "energy_equation_residual_norm": energy_residual_norm,
        "momentum_equation_residual_norm": momentum_residual_norm,
        "branch_label": "muK-rich",
        "energy_flux_equation": "E_rgamma_const",
        "rate_model": "exact_nonleptonic",
        "composition_definition": "nK_over_local_nB",
        "current_definition": "u_nK_minus_D_dnK_dx",
        "density_ratio_definition": "lambda_n_equals_nB_N_over_nB_Q",
        "solver_variant": "uNmax_direct_TQstar_nK",
        "coordinate": "BVP: s_coord in [0, 1-tail_eps], s_coord=1-exp(-lambda*x)",
        "tail_eps": float(tail_eps),
        "_root_method": "solve_bvp_uNmax_nK_parameter_1d",
        "bvp_status": int(sol.status),
        "bvp_message": str(sol.message),
        "bvp_collocation_converged": bool(sol.success),
        "bvp_max_nodes_postvalidated": accepted_max_nodes,
        "bvp_max_rms_residual": float(
            np.max(np.asarray(getattr(sol, "rms_residuals", [np.nan]), dtype=float))
        ),
        "bvp_niter": int(getattr(sol, "niter", -1)),
        "bvp_nodes": int(sol.x.size),
        "bvp_parameters": np.asarray(sol.p, dtype=float),
        **{key: int(value) for key, value in stats.items()},
    }
    if return_profile:
        result.update(
            {
                "s_coord": s_profile,
                "x": x_profile,
                "nK": nK_profile,
                "jK": jK_profile,
                "a": a_profile,
                "u": profile["u"],
                "nB": profile["nB"],
                "muB": profile["muB"],
                "muK": profile["muK"],
                "T_profile": profile["T"],
                "P_profile": profile["P"],
                "h_profile": profile["h"],
                "r_gamma_profile": profile["r_gamma"],
                "invD_profile": profile["invD"],
                "Gamma_K_profile": profile["Gamma_K"],
                "jK_prime_profile": jK_prime,
                "kaon_equation_residual_profile": kaon_residual,
            }
        )
    if verb:
        print(
            f"uNmax-nK jB={result['jB']:.6g}, T_Q={result['T_Q']:.6g}, "
            f"TQstar={TQstar:.6g}, aQstar={result['aQstar']:.6g}, "
            f"tail_norm={result['tail_residual_norm']:.6g}, "
            f"kaon_eq_norm={kaon_residual_norm:.6g}, status={sol.status}, success={success}"
        )
    return result


def _annotate_energy_uNmax_result(
    result,
    TQstar_target,
):
    out = dict(result)
    TQstar_target = float(TQstar_target)
    T_Qstar = float(out.get("T_Qstar", np.nan))
    aQstar_out = float(out.get("aQstar", np.nan))
    u_N_out = float(out.get("u_N", np.nan))
    out["solver_variant"] = "uNmax_direct_TQstar_nK"
    out["T_Qstar_target"] = TQstar_target
    out["T_Qstar_residual"] = (
        float(T_Qstar - TQstar_target)
        if np.isfinite(T_Qstar) and np.isfinite(TQstar_target)
        else np.nan
    )
    out["aQstar_derived"] = aQstar_out
    out["u_N_max_candidate"] = u_N_out
    return out


def solve_front_energy_conserving_uNmax(
    T,
    nB_N,
    B_one_forth,
    TQstar=0.5,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    TQ_guess=None,
    jB_bounds=None,
    return_profile=False,
    verb=False,
    continuation_guess=None,
):
    """
    Solve the energy-conserving front with the interface temperature fixed.

    This variant treats T_Qstar as the input and derives a_Qstar from the
    quark EOS, which is useful for probing the upper end of the branch where
    T_Qstar approaches zero. It is not a separate optimizer: the returned
    u_N_max_candidate is the upstream velocity for the requested fixed
    T_Qstar.
    """
    TQstar = float(TQstar)
    if (not np.isfinite(TQstar)) or TQstar < 0.0:
        raise RuntimeError("TQstar must be non-negative")

    direct_jB_guess = jB_guess
    if direct_jB_guess is None:
        direct_jB_guess = float(max(1.0e-12, 1.0e-8 * float(nB_N), 1.0))

    direct_result = _solve_front_energy_conserving_uNmax_once(
        T,
        nB_N,
        B_one_forth,
        TQstar,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=n_mesh,
        tol_bvp=tol_bvp,
        max_nodes=max_nodes,
        jB_guess=direct_jB_guess,
        TQ_guess=TQ_guess,
        jB_bounds=jB_bounds,
        return_profile=return_profile,
        verb=verb,
        continuation_guess=continuation_guess,
    )
    result_out = _annotate_energy_uNmax_result(direct_result, TQstar)
    if return_profile:
        return result_out
    return _strip_energy_profile_fields(result_out)


def _piecewise_linear_travel_time(z_samples, velocity_samples):
    """Return the exact travel time for a positive piecewise-linear velocity."""
    z_samples = np.asarray(z_samples, dtype=float)
    velocity_samples = np.asarray(velocity_samples, dtype=float)
    if z_samples.ndim != 1 or velocity_samples.ndim != 1:
        raise RuntimeError("Travel-time samples must be one-dimensional")
    if z_samples.size != velocity_samples.size or z_samples.size < 2:
        raise RuntimeError("Travel-time samples must have matching lengths of at least two")
    if np.any(np.diff(z_samples) <= 0.0):
        raise RuntimeError("Travel-time z samples must be strictly increasing")
    if np.any(~np.isfinite(velocity_samples)) or np.any(velocity_samples < 0.0):
        raise RuntimeError("Travel-time velocity samples must be finite and non-negative")
    if np.any(velocity_samples == 0.0):
        return np.inf

    total = 0.0
    for index in range(z_samples.size - 1):
        dz = float(z_samples[index + 1] - z_samples[index])
        v_left = float(velocity_samples[index])
        v_right = float(velocity_samples[index + 1])
        dv = float(v_right - v_left)
        if abs(dv) <= 1.0e-12 * max(v_left, v_right):
            total += dz / (0.5 * (v_left + v_right))
        else:
            total += dz * np.log1p(dv / v_left) / dv
    return float(total)


def z_time_evolution(
    nB_target,
    temperature,
    B_one_forth,
    *,
    z0=1.0,
    density_slope_n0_per_km=0.3,
    z_stop=1.0e-6,
    t_max=20.0,
    sample_count=24,
    output_count=500,
    TQstar=0.5,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    TQ_guess=None,
    jB_bounds=None,
    adaptive_continuation=True,
    max_continuation_subdivisions=12,
    stall_on_terminal_branch=True,
    return_solver_results=False,
    verb=False,
):
    """
    Evolve the remaining distance between a phase front and its target.

    The front begins ``z0`` km inside the target position. The magnitude of
    the outward density gradient is prescribed in units of n0/km, where
    n0 = 0.16 fm^-3, so the upstream density used by the hydrodynamic solver
    is

        nB_N(z) = nB_target + density_slope_n0_per_km * n0 * z.

    ``temperature`` may be a positive scalar in MeV or a callable accepting
    ``z`` in km and returning the local upstream temperature in MeV. Explicit
    time-dependent cooling is not supported because it would require a
    two-dimensional velocity surrogate v(z, t).

    The hydrodynamic solver returns the spatial four-velocity u_N. It is
    converted to ordinary speed with beta_N = u_N/sqrt(1 + u_N**2), and the
    ODE dz/dt = -v(z) is integrated until ``z_stop`` or ``t_max``.

    Parameters
    ----------
    nB_target : float
        Nuclear baryon density at the target position, in MeV^3.
    temperature : float or callable
        Upstream temperature in MeV, either constant or a function of z_km.
    B_one_forth : float
        Bag-model parameter B^(1/4), in MeV.
    z0, z_stop : float, optional
        Initial and terminal distances in km, satisfying 0 < z_stop < z0.
    density_slope_n0_per_km : float, optional
        Positive density-gradient magnitude in units of n0/km.
    t_max : float, optional
        Maximum integration time in seconds.
    sample_count : int, optional
        Number of nominal logarithmically spaced hydrodynamic velocity
        samples. Additional samples may be inserted to continue the nonlinear
        hydrodynamic solution through difficult density intervals.
    output_count : int, optional
        Number of points returned on the z(t) curve.
    adaptive_continuation : bool, optional
        If true, bisect a failed logarithmic z interval and retry using the
        complete last successful uNmax solution as a continuation state.
    max_continuation_subdivisions : int, optional
        Maximum number of logarithmic interval subdivisions for one nominal
        sample before reporting the hydrodynamic failure.
    stall_on_terminal_branch : bool, optional
        At exact ``TQstar=0``, return an explicitly unsuccessful numerical
        continuation result with a zero-velocity closure when retries are
        exhausted. This does not classify the failure as a physical terminal
        branch.

    Returns
    -------
    dict
        Evolution curve, catch-up estimates, sampled upstream states, and
        hydrodynamic diagnostics.
    """
    nB_target = float(nB_target)
    B_one_forth = float(B_one_forth)
    z0 = float(z0)
    density_slope_n0_per_km = float(density_slope_n0_per_km)
    z_stop = float(z_stop)
    t_max = float(t_max)
    TQstar = float(TQstar)
    ms = float(ms)

    if (not np.isfinite(nB_target)) or nB_target <= 0.0:
        raise ValueError("nB_target must be positive and finite")
    if (not np.isfinite(B_one_forth)) or B_one_forth <= 0.0:
        raise ValueError("B_one_forth must be positive and finite")
    if (not np.isfinite(z0)) or (not np.isfinite(z_stop)) or not (0.0 < z_stop < z0):
        raise ValueError("z0 and z_stop must satisfy 0 < z_stop < z0")
    if (not np.isfinite(density_slope_n0_per_km)) or density_slope_n0_per_km <= 0.0:
        raise ValueError("density_slope_n0_per_km must be positive and finite")
    if (not np.isfinite(t_max)) or t_max <= 0.0:
        raise ValueError("t_max must be positive and finite")
    if int(sample_count) != sample_count or int(sample_count) < 2:
        raise ValueError("sample_count must be an integer of at least two")
    if int(output_count) != output_count or int(output_count) < 2:
        raise ValueError("output_count must be an integer of at least two")
    if (
        int(max_continuation_subdivisions) != max_continuation_subdivisions
        or int(max_continuation_subdivisions) < 0
    ):
        raise ValueError("max_continuation_subdivisions must be a non-negative integer")
    sample_count = int(sample_count)
    output_count = int(output_count)
    max_continuation_subdivisions = int(max_continuation_subdivisions)
    adaptive_continuation = bool(adaptive_continuation)
    stall_on_terminal_branch = bool(stall_on_terminal_branch)

    if callable(temperature):
        temperature_of_z = temperature
        temperature_description = "callable_z"
    else:
        constant_temperature = float(temperature)
        if (not np.isfinite(constant_temperature)) or constant_temperature <= 0.0:
            raise ValueError("temperature must be positive and finite")

        def temperature_of_z(_z):
            return constant_temperature

        temperature_description = "constant"

    n0 = float(0.16 * const.MeV_fm**3)
    nominal_z_descending = np.geomspace(z0, z_stop, sample_count)
    successful_samples = []
    solver_results = []
    current_continuation = None
    inserted_point_count = 0
    stalled = False
    stall_z_km = np.nan
    stall_reason = ""

    def solve_velocity_sample(z_value, continuation_guess):
        z_value = float(z_value)
        nB_N = float(nB_target + density_slope_n0_per_km * n0 * z_value)
        try:
            T_N = float(temperature_of_z(z_value))
        except Exception as exc:
            raise RuntimeError(
                f"temperature(z) failed at z={z_value:.9g} km: {exc}"
            ) from exc
        if (not np.isfinite(T_N)) or T_N <= 0.0:
            raise RuntimeError(
                f"temperature(z) must be positive and finite at z={z_value:.9g} km"
            )

        point_jB_guess = jB_guess
        point_TQ_guess = TQ_guess
        if isinstance(continuation_guess, dict):
            point_jB_guess = continuation_guess.get("jB", point_jB_guess)
            point_TQ_guess = continuation_guess.get("T_Q", point_TQ_guess)

        result = solve_front_energy_conserving_uNmax(
            T_N,
            float(nB_N),
            B_one_forth,
            TQstar=TQstar,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=n_mesh,
            tol_bvp=tol_bvp,
            max_nodes=max_nodes,
            jB_guess=point_jB_guess,
            TQ_guess=point_TQ_guess,
            jB_bounds=jB_bounds,
            continuation_guess=continuation_guess,
            return_profile=True,
            verb=verb,
        )
        if not bool(result.get("success", False)):
            return None, result, nB_N, T_N

        u_N = float(result.get("u_N_max_candidate", result.get("u_N", np.nan)))
        if (not np.isfinite(u_N)) or u_N < 0.0:
            raise RuntimeError(
                f"uNmax returned a non-physical u_N={u_N!r} at z={z_value:.9g} km"
            )
        beta_N = float(u_N / np.sqrt(1.0 + u_N * u_N))
        velocity_km_s = float(const.c_km * beta_N)
        if (not np.isfinite(velocity_km_s)) or velocity_km_s < 0.0:
            raise RuntimeError(
                f"uNmax produced a non-physical velocity at z={z_value:.9g} km"
            )

        sample = {
            "z": z_value,
            "nB_N": nB_N,
            "T_N": T_N,
            "u_N": u_N,
            "velocity_km_s": velocity_km_s,
            "T_Q": float(result.get("T_Q", np.nan)),
            "T_Qstar": float(result.get("T_Qstar", np.nan)),
        }
        return sample, result, nB_N, T_N

    def failed_sample_message(z_value, nB_N, T_N, failure, subdivisions):
        last_success = successful_samples[-1]["z"] if successful_samples else np.nan
        return (
            "uNmax velocity solve failed at "
            f"z={float(z_value):.9g} km, nB_N={float(nB_N):.9g} MeV^3, "
            f"T_N={float(T_N):.9g} MeV after {int(subdivisions)} adaptive "
            f"subdivision(s); last successful z={float(last_success):.9g} km: "
            f"{failure.get('message', 'unknown failure')}"
        )

    for nominal_index, nominal_z in enumerate(nominal_z_descending):
        pending = [(float(nominal_z), 0, False)]
        while pending:
            z_value, subdivisions, is_inserted = pending.pop()
            sample, result, nB_N, T_N = solve_velocity_sample(
                z_value,
                current_continuation,
            )
            if sample is not None:
                successful_samples.append(sample)
                current_continuation = result
                if return_solver_results:
                    solver_results.append(result)
                if is_inserted:
                    inserted_point_count += 1
                continue

            can_subdivide = (
                adaptive_continuation
                and bool(successful_samples)
                and subdivisions < max_continuation_subdivisions
            )
            if not can_subdivide:
                if (
                    stall_on_terminal_branch
                    and TQstar == 0.0
                ):
                    had_successful_sample = bool(successful_samples)
                    successful_samples.append(
                        {
                            "z": float(z_value),
                            "nB_N": float(nB_N),
                            "T_N": float(T_N),
                            "u_N": 0.0,
                            "velocity_km_s": 0.0,
                            "T_Q": np.nan,
                            "T_Qstar": 0.0,
                        }
                    )
                    if return_solver_results:
                        solver_results.append(dict(result))
                    if not had_successful_sample and z_value > z_stop:
                        try:
                            T_stop = float(temperature_of_z(z_stop))
                        except Exception:
                            T_stop = float(T_N)
                        successful_samples.append(
                            {
                                "z": float(z_stop),
                                "nB_N": float(
                                    nB_target
                                    + density_slope_n0_per_km * n0 * z_stop
                                ),
                                "T_N": T_stop,
                                "u_N": 0.0,
                                "velocity_km_s": 0.0,
                                "T_Q": np.nan,
                                "T_Qstar": 0.0,
                            }
                        )
                        if return_solver_results:
                            solver_results.append(
                                {
                                    "success": False,
                                    "message": "Synthetic zero-velocity support point below exhausted continuation",
                                    "synthetic_stall_support": True,
                                    "T_Qstar": 0.0,
                                }
                            )
                    stalled = True
                    stall_z_km = float(z_value)
                    stall_reason = failed_sample_message(
                        z_value, nB_N, T_N, result, subdivisions
                    )
                    pending.clear()
                    break
                raise RuntimeError(
                    failed_sample_message(z_value, nB_N, T_N, result, subdivisions)
                )

            last_success_z = float(successful_samples[-1]["z"])
            midpoint_z = float(np.sqrt(last_success_z * z_value))
            if not (z_value < midpoint_z < last_success_z):
                raise RuntimeError(
                    failed_sample_message(z_value, nB_N, T_N, result, subdivisions)
                )
            pending.append((z_value, subdivisions + 1, is_inserted))
            pending.append((midpoint_z, subdivisions + 1, True))
        if stalled:
            break

    z_descending = np.asarray([sample["z"] for sample in successful_samples], dtype=float)
    density_descending = np.asarray(
        [sample["nB_N"] for sample in successful_samples], dtype=float
    )
    temperature_descending = np.asarray(
        [sample["T_N"] for sample in successful_samples], dtype=float
    )
    u_N_descending = np.asarray(
        [sample["u_N"] for sample in successful_samples], dtype=float
    )
    velocity_descending = np.asarray(
        [sample["velocity_km_s"] for sample in successful_samples], dtype=float
    )
    T_Q_descending = np.asarray(
        [sample["T_Q"] for sample in successful_samples], dtype=float
    )
    T_Qstar_descending = np.asarray(
        [sample["T_Qstar"] for sample in successful_samples], dtype=float
    )

    z_samples = z_descending[::-1].copy()
    density_samples = density_descending[::-1].copy()
    temperature_samples = temperature_descending[::-1].copy()
    u_N_samples = u_N_descending[::-1].copy()
    velocity_samples = velocity_descending[::-1].copy()
    T_Q_samples = T_Q_descending[::-1].copy()
    T_Qstar_samples = T_Qstar_descending[::-1].copy()

    def velocity_of_z(z_value):
        z_value = float(z_value)
        # RK stages may temporarily cross an event boundary. Clamp those
        # probes to the sampled endpoint instead of extrapolating v(z).
        return float(np.interp(np.clip(z_value, z_stop, z0), z_samples, velocity_samples))

    def dzdt(_t, y):
        return np.array([-velocity_of_z(float(y[0]))], dtype=float)

    def hit_z_stop(_t, y):
        return float(y[0] - z_stop)

    hit_z_stop.terminal = True
    hit_z_stop.direction = -1

    sol = solve_ivp(
        dzdt,
        (0.0, t_max),
        np.array([z0], dtype=float),
        method="RK45",
        events=hit_z_stop,
        dense_output=True,
        rtol=1.0e-7,
        atol=min(1.0e-9, 1.0e-3 * z_stop),
    )
    if not sol.success:
        raise RuntimeError(f"z(t) integration failed: {sol.message}")

    caught_up = bool(sol.t_events[0].size > 0)
    catchup_time_s = float(sol.t_events[0][0]) if caught_up else np.nan
    evolution_end_time = catchup_time_s if caught_up else float(sol.t[-1])
    t_s = np.linspace(0.0, evolution_end_time, output_count)
    z_km = np.asarray(sol.sol(t_s)[0], dtype=float)
    if caught_up:
        z_km[-1] = z_stop

    integral_catchup_time_s = _piecewise_linear_travel_time(
        z_samples,
        velocity_samples,
    )
    result = {
        "success": bool(not stalled),
        "message": (
            "Hydrodynamic continuation exhausted; zero-velocity closure applied"
            if stalled
            else "Phase-boundary distance evolution integrated"
        ),
        "caught_up": caught_up,
        "catchup_time_s": catchup_time_s,
        "integral_catchup_time_s": integral_catchup_time_s,
        "t_s": t_s,
        "z_km": z_km,
        "z0_km": z0,
        "z_stop_km": z_stop,
        "t_max_s": t_max,
        "n0_MeV3": n0,
        "nB_target_MeV3": nB_target,
        "density_slope_n0_per_km": density_slope_n0_per_km,
        "temperature_mode": temperature_description,
        "z_samples_km": z_samples,
        "nB_N_samples_MeV3": density_samples,
        "T_N_samples_MeV": temperature_samples,
        "u_N_samples": u_N_samples,
        "velocity_samples_km_s": velocity_samples,
        "T_Q_samples_MeV": T_Q_samples,
        "T_Qstar_samples_MeV": T_Qstar_samples,
        "nominal_sample_count": sample_count,
        "adaptive_continuation_used": bool(inserted_point_count > 0),
        "continuation_inserted_point_count": int(inserted_point_count),
        "stalled": stalled,
        "continuation_exhausted": stalled,
        "stall_is_physical_terminal": False,
        "stall_z_km": stall_z_km,
        "stall_reason": stall_reason,
    }
    if return_solver_results:
        result["solver_results"] = list(reversed(solver_results))
    return result
