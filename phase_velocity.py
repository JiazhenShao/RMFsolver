import numpy as np
import time
import warnings
from scipy.integrate import solve_bvp
from scipy.optimize import fsolve, root_scalar, root, least_squares
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
    "solve_front_adiabatic",
    "solve_front_energy_conserving",
    "solve_front_energy_conserving_uNmax",
]

_ADIABATIC_LOW_T_THRESHOLD = 5.0
_ADIABATIC_HOT_START_OFFSET = 50.0
_ADIABATIC_LOCAL_T_FLOOR = 1.0e-6
_ADIABATIC_LOCAL_LOGT_FLOOR = float(np.log(_ADIABATIC_LOCAL_T_FLOOR))
_TRANSPORT_ALPHA_S = 0.3
_TRANSPORT_G_S = np.sqrt(4.0 * np.pi * _TRANSPORT_ALPHA_S)
_TRANSPORT_QD_COEFF = np.sqrt(3.0 * _TRANSPORT_G_S**2 / (2.0 * np.pi**2))
_TRANSPORT_D_PREFACTOR = 24.0 * _TRANSPORT_ALPHA_S**2 / np.pi
_TRANSPORT_H_CONST = 1.81317
_FLOAT_TINY = np.finfo(float).tiny
_ISOTHERMAL_RETRY_ACTIVE = 0


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

def _analytic_weak_rate_from_mu_q(mu_q):
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
):
    """
    Evaluate the analytic upper bound for the upstream nuclear velocity u_N.

    The nuclear state is read from the existing EOS helpers at (muB_N, T_N).
    The downstream quark endpoint is solved at muK = 0 from pressure
    continuity and h/nB matching. Densities use nB_NM and nB_QM directly;
    the leading massless nB ~ muB^3 approximation is not used in the bound.
    """
    muB_N = float(muB_N)
    T_N = float(T_N)
    B_one_forth = float(B_one_forth)
    xi = float(xi)

    if (not np.isfinite(muB_N)) or muB_N <= 0.0:
        raise RuntimeError("muB_N must be positive and finite")
    if (not np.isfinite(T_N)) or T_N <= 0.0:
        raise RuntimeError("T_N must be positive and finite")
    if (not np.isfinite(B_one_forth)) or B_one_forth <= 0.0:
        raise RuntimeError("B_one_forth must be positive and finite")
    if (not np.isfinite(xi)) or not (-1.0 < xi < 1.0):
        raise RuntimeError("xi must satisfy -1 < xi < 1")

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

    def _append_guess(guesses, muB_guess, T_guess):
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

    guesses = []
    if initial_guess is not None:
        guess_arr = np.asarray(initial_guess, dtype=float).ravel()
        if guess_arr.size >= 2:
            _append_guess(guesses, guess_arr[0], guess_arr[1])
    _append_guess(guesses, muB_N, T_N)
    _append_guess(guesses, max(muB_N, 900.0), max(T_N, 1.0))
    _append_guess(guesses, 1100.0, max(T_N, 10.0))
    _append_guess(guesses, 900.0, 10.0)
    _append_guess(guesses, 1300.0, 10.0)
    _append_guess(guesses, 1100.0, 50.0)

    p_scale = max(abs(P_N), 1.0)
    h_scale = max(abs(h_over_nB_N), 1.0)

    def equations(vec):
        muB = float(vec[0])
        logT = float(vec[1])
        if (not np.isfinite(muB)) or muB <= 0.0 or (not np.isfinite(logT)) or abs(logT) > 700.0:
            return np.array([1.0e30, 1.0e30], dtype=float)
        T_Q = float(np.exp(logT))
        try:
            P_Q = float(PQM(muB, 0.0, B_one_forth, T_Q, ms=ms, upB=upB))
            e_Q = float(edensQM(muB, 0.0, B_one_forth, T_Q, ms=ms, include_em=False, upB=upB))
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
        h_over_nB_Q = float((P_Q + e_Q) / nB_Q)
        if (not np.isfinite(h_over_nB_Q)) or h_over_nB_Q <= 0.0:
            return np.array([1.0e30, 1.0e30], dtype=float)
        return np.array(
            [
                (P_Q - P_N) / p_scale,
                (h_over_nB_Q - h_over_nB_N) / h_scale,
            ],
            dtype=float,
        )

    best = None
    best_norm = np.inf
    best_message = "analytic quark-equilibrium solve did not converge"
    for muB_guess, T_guess in guesses:
        sol = root(
            equations,
            np.array([float(muB_guess), float(np.log(T_guess))], dtype=float),
            method="hybr",
            options={"maxfev": 1200, "xtol": 1.0e-10},
        )
        if not (sol.success and np.all(np.isfinite(sol.x))):
            best_message = str(sol.message)
            continue
        residual = equations(sol.x)
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        if residual_norm < best_norm:
            best_norm = residual_norm
            best = sol.x.copy()
        if residual_norm <= 1.0e-8:
            break

    if best is None or best_norm > 1.0e-8:
        raise RuntimeError(f"{best_message}; best scaled residual={best_norm:.3e}")

    muB_bar = float(best[0])
    T_Q = float(np.exp(float(best[1])))
    if (not np.isfinite(muB_bar)) or muB_bar <= 0.0 or (not np.isfinite(T_Q)) or T_Q <= 0.0:
        raise RuntimeError("Analytic quark-equilibrium solve returned a non-physical root")

    P_Q = float(PQM(muB_bar, 0.0, B_one_forth, T_Q, ms=ms, upB=upB))
    e_Q = float(edensQM(muB_bar, 0.0, B_one_forth, T_Q, ms=ms, include_em=False, upB=upB))
    h_Q = float(P_Q + e_Q)
    nB_Q = float(nB_QM(muB_bar, 0.0, B_one_forth, T_Q, ms=ms, upB=upB))
    if (
        (not np.isfinite(P_Q))
        or (not np.isfinite(e_Q))
        or (not np.isfinite(h_Q))
        or (not np.isfinite(nB_Q))
        or nB_Q <= 0.0
    ):
        raise RuntimeError("Equilibrated quark EOS returned non-physical quantities")

    mu_q = float(muB_bar / 3.0)
    weak_rate = _analytic_weak_rate_from_mu_q(mu_q)
    gamma = float(weak_rate["gamma"])
    tau = float(weak_rate["tau"])
    tau_seconds = float(weak_rate["tau_seconds"])

    a_N = float(nB_N / nB_Q)
    if (not np.isfinite(a_N)) or a_N <= 0.0:
        raise RuntimeError("a_N must be positive and finite")

    aQstar_max = float((9.0 * np.pi / np.sqrt(2.0)) * (T_Q / muB_bar))
    muKstar_max = float(np.sqrt(2.0) * np.pi * T_Q)
    one_minus_aQstar = float(1.0 - aQstar_max)
    one_plus_xi_aQstar = float(1.0 + xi * aQstar_max)
    a_N_squared = float(a_N * a_N)

    if (not np.isfinite(aQstar_max)) or not (0.0 < aQstar_max < 1.0):
        raise RuntimeError("aQstar_max must satisfy 0 < aQstar_max < 1")
    if (not np.isfinite(one_plus_xi_aQstar)) or one_plus_xi_aQstar <= 0.0:
        raise RuntimeError("1 + xi*aQstar_max must be positive")

    alpha_s = float(_TRANSPORT_ALPHA_S)
    h_D = float(_TRANSPORT_H_CONST)
    denominator = float(muB_bar * a_N_squared * one_minus_aQstar * one_plus_xi_aQstar)
    if (not np.isfinite(denominator)) or denominator <= 0.0:
        raise RuntimeError("Analytic velocity denominator is non-physical")

    prefactor = float(
        (54.0 * np.pi ** (7.0 / 3.0) * gamma)
        / (7.0 * np.sqrt(2.0) * h_D * alpha_s ** (5.0 / 3.0))
    )
    u_N_squared = float(prefactor * aQstar_max ** (7.0 / 3.0) / denominator)
    if (not np.isfinite(u_N_squared)) or u_N_squared < 0.0:
        raise RuntimeError("Analytic velocity bound produced non-physical u_N^2")
    u_N_max = float(np.sqrt(u_N_squared))

    return {
        "success": True,
        "message": "analytic velocity bound evaluated",
        "u_N_max": u_N_max,
        "u_N_squared": u_N_squared,
        "muB_N": muB_N,
        "T_N": T_N,
        "P_N": P_N,
        "e_N": e_N,
        "h_N": h_N,
        "nB_N": nB_N,
        "h_over_nB_N": h_over_nB_N,
        "muB_bar": muB_bar,
        "T_Q": T_Q,
        "P_Q": P_Q,
        "e_Q": e_Q,
        "h_Q": h_Q,
        "nB_Q": nB_Q,
        "h_over_nB_Q": float(h_Q / nB_Q),
        "mu_q": mu_q,
        "a_N": a_N,
        "aQstar_max": aQstar_max,
        "muKstar_max": muKstar_max,
        "one_minus_aQstar": one_minus_aQstar,
        "one_plus_xi_aQstar": one_plus_xi_aQstar,
        "a_N_squared": a_N_squared,
        "alpha_s": alpha_s,
        "h_D": h_D,
        "gamma": gamma,
        "tau": tau,
        "tau_seconds": tau_seconds,
        "xi": xi,
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


def _pi_target_state(T, B_one_forth, jB, Pi_over_crit, ms=0.0, upB=5000):
    """
    Convert the dimensionless Pi_over_crit input into the absolute Pi target and
    solve the corresponding mu_B roots on the mu_K = 0 reference surface.
    """
    muB_crit = _solve_scalar_root(
        lambda x: Pi_NM(x, T, jB) - Pi_QM(x, 0.0, B_one_forth, T, jB, ms=ms, upB=upB),
        1050.0,
    )
    Pi_crit = Pi_QM(muB_crit, 0.0, B_one_forth, T, jB, ms=ms, upB=upB)
    Pi_target = Pi_over_crit * Pi_crit
    muB_N = _solve_scalar_root(lambda x: Pi_NM(x, T, jB) - Pi_target, 1050.0)
    muB_Q = _solve_scalar_root(
        lambda x: Pi_QM(x, 0.0, B_one_forth, T, jB, ms=ms, upB=upB) - Pi_target,
        1050.0,
    )
    return muB_crit, Pi_crit, Pi_target, muB_N, muB_Q


def _find_Qstar_on_target(T, B_one_forth, lambda_val, jB, Pi_target, muB_N, muB_Q, ms=0.0, upB=5000, return_aux=False):
    """
    Solve the Q* system once the absolute Pi target and the associated mu_B
    roots are already known.
    """
    T2 = T * T

    quark_Q = _quark_uds_state(muB_Q, 0.0, T, ms=ms, upB=upB)
    nB_Q = quark_Q["nB"]
    nK_Q = quark_Q["nK"]

    nB_N_diff = nB_NM(muB_N, T)
    aN = (nB_N_diff - nK_Q) / nB_Q
    nB_N_surface = nB_NM(muB_N, T)
    uN = jB / nB_N_surface

    def system(vec):
        muBstar, muKstar = map(float, vec)
        muQ = muBstar / 3.0

        if muBstar <= 0.0 or muKstar < 0.0 or muQ <= 0.0:
            return np.array([1e30, 1e30], dtype=float)

        quark_Qstar = _quark_uds_state(muBstar, muKstar, T, ms=ms, upB=upB)
        nK_Qstar = quark_Qstar["nK"]
        aQstar = (nK_Qstar - nK_Q) / nB_Q

        try:
            _, D = _quark_diffusion_coefficient(muQ, T)
        except RuntimeError:
            return np.array([1e30, 1e30], dtype=float)
        eta = (9.0 * np.pi**2 * T2) / (muQ * muQ)
        gamma = 1.0 / (1.98e12 * ((300.0 / muQ) ** 5))

        eq1 = Pi_QM(muBstar, muKstar, B_one_forth, T, jB, ms=ms, upB=upB) - Pi_target
        uQ = jB / quark_Qstar["nB"]
        delta_u = uQ * aQstar - uN * aN

        eq2 = (
            0.5 * delta_u**2
            - lambda_val * uQ * aQstar * delta_u
            - 0.25 * D * gamma * (aQstar**4 + 2.0 * eta * aQstar**2)
        )

        return np.array([eq1, eq2], dtype=float)

    initial_guess = [1020, 40]
    res = least_squares(system, initial_guess, bounds=([0, 0], [np.inf, np.inf]))

    if not res.success:
        raise RuntimeError(f"Root finding failed: {res.message}")

    muB_star = float(res.x[0])
    muK_star = float(res.x[1])

    if muK_star < 0:
        raise RuntimeError("muK_star < 0")

    if return_aux:
        return muB_star, muK_star, nB_N_surface

    return muB_star, muK_star


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


def _quark_thermo_state(muB, muK, B_one_forth, T, jB, ms=0.0, upB=5000):
    """
    Build a fully ms-consistent local quark thermodynamic state.

    The entropy density is reconstructed from the thermal relation
        s = (e + P - muB * nB - muK * nK) / T
    using the existing EOS helpers.
    """
    T = float(T)
    if (not np.isfinite(T)) or T <= 0.0:
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


def _normalize_interface_temperature_mode(mode):
    mode = str(mode)
    valid_modes = ("hu_const", "T_const")
    if mode not in valid_modes:
        raise RuntimeError(
            "interface_temperature_mode must be one of "
            + ", ".join(repr(item) for item in valid_modes)
        )
    return mode


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


def _qstar_enthalpy_jump_residual(muB, muK, logT, aQstar, Pi, jB, nB_Q, nK_Q, B_one_forth, h_over_nB_N, ms=0.0, upB=5000):
    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving enthalpy-jump Qstar")
    if (not np.isfinite(h_over_nB_N)) or h_over_nB_N <= 0.0:
        raise RuntimeError("Nuclear h/nB must be positive when solving enthalpy-jump Qstar")

    logT = float(logT)
    if (not np.isfinite(logT)) or abs(logT) > 700.0 or logT < _ADIABATIC_LOCAL_LOGT_FLOOR:
        return np.array([1.0e12, 1.0e12, 1.0e12], dtype=float)

    try:
        thermo = _quark_thermo_state(muB, muK, B_one_forth, float(np.exp(logT)), jB, ms=ms, upB=upB)
    except Exception:
        return np.array([1.0e12, 1.0e12, 1.0e12], dtype=float)

    return np.array(
        [
            thermo["Pi"] - Pi,
            (thermo["nK"] - nK_Q) / nB_Q - aQstar,
            thermo["h"] / thermo["nB"] - h_over_nB_N,
        ],
        dtype=float,
    )


def _qstar_enthalpy_jump_residual_ok(residual, Pi, aQstar, h_over_nB_N):
    if not np.all(np.isfinite(residual)):
        return False
    pi_tol = 1.0e-8 * max(abs(Pi), 1.0)
    a_tol = 1.0e-8 * max(abs(aQstar), 1.0)
    h_tol = 1.0e-8 * max(abs(h_over_nB_N), 1.0)
    return bool(
        abs(float(residual[0])) <= pi_tol
        and abs(float(residual[1])) <= a_tol
        and abs(float(residual[2])) <= h_tol
    )


def _append_unique_qstar_guess(guesses, guess, min_temperature=_ADIABATIC_LOCAL_T_FLOOR):
    try:
        arr = np.asarray(guess, dtype=float)
    except Exception:
        return
    if arr.shape[0] < 2 or not np.all(np.isfinite(arr[:2])):
        return
    muB = float(arr[0])
    muK = max(0.0, float(arr[1]))
    if muB <= 0.0:
        return
    if arr.shape[0] >= 3 and np.isfinite(float(arr[2])) and float(arr[2]) > 0.0:
        temp = float(arr[2])
    else:
        return
    temp = max(float(temp), float(min_temperature))
    candidate = np.array([muB, muK, temp], dtype=float)
    for item in guesses:
        if (
            abs(candidate[0] - item[0]) <= 1.0e-10 * max(1.0, abs(item[0]))
            and abs(candidate[1] - item[1]) <= 1.0e-10 * max(1.0, abs(item[1]), 1.0)
            and abs(candidate[2] - item[2]) <= 1.0e-10 * max(1.0, abs(item[2]))
        ):
            return
    guesses.append(candidate)


def _solve_interface_Qstar_enthalpy_jump(
    aQstar,
    Pi,
    jB,
    nB_Q,
    nK_Q,
    B_one_forth,
    h_over_nB_N,
    T_N,
    T_Q,
    ms=0.0,
    upB=5000,
    initial_guess=None,
    stats=None,
    stats_key="qstar_root_calls",
):
    """
    Solve the Qstar interface state with temperature determined by h/nB continuity.

    Unknowns are (muB, muK, logT_Qstar). The constraints are momentum flux,
    the fixed aQstar composition coordinate, and h_Qstar/nB_Qstar = h_N/nB_N.
    """
    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving enthalpy-jump Qstar")
    T_N = float(T_N)
    T_Q = float(T_Q)
    if (not np.isfinite(T_N)) or T_N <= 0.0:
        raise RuntimeError("T_N must be positive when solving enthalpy-jump Qstar")
    if (not np.isfinite(T_Q)) or T_Q <= 0.0:
        raise RuntimeError("T_Q must be positive when solving enthalpy-jump Qstar")

    guesses = []
    muK_seed = _branch_muK_seed(aQstar)
    muK_seed_strong = float(max(muK_seed, 400.0 * abs(float(aQstar))))

    temp_seeds = [
        T_N,
        T_Q,
        float(np.sqrt(T_N * T_Q)),
        0.5 * (T_N + T_Q),
        T_N + _ADIABATIC_HOT_START_OFFSET,
        max(T_Q, T_N + _ADIABATIC_HOT_START_OFFSET),
        10.0,
        50.0,
    ]
    temp_seeds = [float(t) for t in temp_seeds if np.isfinite(t) and t > 0.0]

    if initial_guess is not None:
        guess0 = np.asarray(initial_guess, dtype=float)
        if guess0.shape[0] >= 3:
            _append_unique_qstar_guess(guesses, guess0)
            muB_seed = float(guess0[0])
            muK_ref = max(0.0, float(guess0[1]))
        elif guess0.shape[0] >= 2:
            muB_seed = float(guess0[0])
            muK_ref = max(0.0, float(guess0[1]))
            for temp_seed in temp_seeds:
                _append_unique_qstar_guess(guesses, (muB_seed, muK_ref, temp_seed))
        else:
            muB_seed = 1200.0
            muK_ref = muK_seed
    else:
        muB_seed = 1200.0
        muK_ref = muK_seed

    for temp_seed in temp_seeds:
        _append_unique_qstar_guess(guesses, (muB_seed, muK_ref, temp_seed))
        _append_unique_qstar_guess(guesses, (muB_seed, muK_seed, temp_seed))
        _append_unique_qstar_guess(guesses, (1200.0, muK_seed, temp_seed))
        _append_unique_qstar_guess(guesses, (1500.0, max(muK_seed, 100.0 * abs(float(aQstar))), temp_seed))
        _append_unique_qstar_guess(guesses, (muB_seed, muK_seed_strong, temp_seed))
        _append_unique_qstar_guess(guesses, (1500.0, muK_seed_strong, temp_seed))

    # The legacy fixed-temperature Qstar solve often provides an excellent
    # chemical-potential seed even when the final interface temperature jumps.
    try:
        legacy_muB, legacy_muK = _solve_interface_Qstar_from_aQstar_and_Pi(
            aQstar,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            T_N,
            ms=ms,
            upB=upB,
            initial_guess=(muB_seed, muK_ref),
            stats=stats,
            stats_key=stats_key,
        )
        for temp_seed in temp_seeds:
            _append_unique_qstar_guess(guesses, (legacy_muB, legacy_muK, temp_seed))
    except Exception:
        pass

    pi_scale = max(abs(Pi), 1.0)
    h_scale = max(abs(h_over_nB_N), 1.0)

    def equations(vec):
        residual = _qstar_enthalpy_jump_residual(
            float(vec[0]),
            float(vec[1]),
            float(vec[2]),
            aQstar,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            h_over_nB_N,
            ms=ms,
            upB=upB,
        )
        return np.array([residual[0] / pi_scale, residual[1], residual[2] / h_scale], dtype=float)

    best_message = "enthalpy-jump Qstar solve did not converge"
    candidates = []
    for guess in guesses:
        if stats is not None:
            stats[stats_key] = stats.get(stats_key, 0) + 1
        sol = root(
            equations,
            np.array([float(guess[0]), float(guess[1]), float(np.log(float(guess[2])))], dtype=float),
            method="hybr",
            options={"maxfev": 1200, "xtol": 1.0e-10},
        )
        if not (sol.success and np.all(np.isfinite(sol.x))):
            best_message = str(sol.message)
            continue

        muB = float(sol.x[0])
        muK = float(sol.x[1])
        logT = float(sol.x[2])
        if muB <= 0.0 or muK < -1.0e-8 or (not np.isfinite(logT)) or abs(logT) > 700.0:
            best_message = "enthalpy-jump Qstar solve returned a non-physical root"
            continue
        muK = max(0.0, muK)
        T_Qstar = float(np.exp(logT))
        residual = _qstar_enthalpy_jump_residual(
            muB,
            muK,
            np.log(T_Qstar),
            aQstar,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            h_over_nB_N,
            ms=ms,
            upB=upB,
        )
        if not _qstar_enthalpy_jump_residual_ok(residual, Pi, aQstar, h_over_nB_N):
            best_message = (
                "enthalpy-jump Qstar solve returned an unacceptable residual "
                f"({residual[0]:.3e}, {residual[1]:.3e}, {residual[2]:.3e})"
            )
            continue

        try:
            thermo = _quark_thermo_state(muB, muK, B_one_forth, T_Qstar, jB, ms=ms, upB=upB)
        except Exception as exc:
            best_message = str(exc)
            continue
        if thermo["nB"] <= 0.0 or thermo["s"] <= 0.0 or thermo["w"] <= 0.0:
            best_message = "enthalpy-jump Qstar solve returned a non-physical thermodynamic state"
            continue

        is_new = True
        for cand in candidates:
            if (
                abs(thermo["muB"] - cand["muB"]) <= 1.0e-8 * max(1.0, abs(cand["muB"]))
                and abs(thermo["muK"] - cand["muK"]) <= 1.0e-8 * max(1.0, abs(cand["muK"]), 1.0)
                and abs(thermo["T"] - cand["T"]) <= 1.0e-8 * max(1.0, abs(cand["T"]))
            ):
                is_new = False
                break
        if is_new:
            candidates.append(thermo)

    if candidates:
        candidates.sort(
            key=lambda cand: (
                abs(cand["muK"] - max(muK_seed, muK_ref)),
                abs(np.log(cand["T"] / max(T_Q, _ADIABATIC_LOCAL_T_FLOOR))),
                abs(cand["muB"] - muB_seed),
                -cand["T"],
            )
        )
        return candidates[0]

    raise RuntimeError(f"enthalpy-jump Qstar state solve failed: {best_message}")


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
def _quark_diffusion_coefficient(muQ, T):
    """
    Return the Debye scale and diffusion coefficient for a local quark state.
    """
    muQ = float(muQ)
    T = float(T)
    qD = _TRANSPORT_QD_COEFF * muQ
    if (not np.isfinite(qD)) or qD <= 0.0:
        raise RuntimeError("Quark diffusion coefficient requires a positive finite screening scale")

    part1 = _TRANSPORT_H_CONST * T ** (5.0 / 3.0) / qD ** (2.0 / 3.0)
    part2 = np.pi**3 * T**2 / (12.0 * qD)
    denom_terms = part1 + part2
    if (
        (not np.isfinite(part1))
        or (not np.isfinite(part2))
        or (not np.isfinite(denom_terms))
        or denom_terms <= _FLOAT_TINY
    ):
        raise RuntimeError("Quark diffusion coefficient denominator is non-physical")

    D = 1.0 / (_TRANSPORT_D_PREFACTOR * denom_terms)
    if (not np.isfinite(D)) or D <= 0.0:
        raise RuntimeError("Quark diffusion coefficient is non-physical")
    return float(qD), float(D)


def _microphysics_at_Qstar(muB_Qstar, T):
    """
    Frozen diffusion/reaction coefficients evaluated at Qstar.
    """
    return _microphysics_from_quark_state(muB_Qstar, T)


def _microphysics_at_Qstar_isothermal_baseline(muB_Qstar, T):
    """
    Isothermal BVP microphysics that matches the baseline steady-front solver.
    """
    return _microphysics_from_quark_state_isothermal_baseline(muB_Qstar, T)


def _microphysics_from_quark_state(muB, T):
    """
    Compute diffusion/reaction coefficients from a local quark thermodynamic
    state. In the current convention the coefficients depend on muQ = muB / 3
    and the local temperature.
    """
    T = float(T)
    if (not np.isfinite(T)) or T <= 0.0:
        raise RuntimeError("Local microphysics requires T > 0")

    muQ = float(muB) / 3.0
    if (not np.isfinite(muQ)) or muQ <= 0.0:
        raise RuntimeError("Local microphysics requires muQ > 0")

    try:
        qD, D = _quark_diffusion_coefficient(muQ, T)
    except RuntimeError as exc:
        raise RuntimeError(f"Local microphysics returned non-physical coefficients: {exc}") from exc
    eta = 9.0 * np.pi**2 * T**2 / muQ**2
    tau = 1.98e12 * (300.0 / muQ) ** 5
    gamma = 1.0 / tau

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
        "alpha_s": float(_TRANSPORT_ALPHA_S),
        "muQ": float(muQ),
        "qD": float(qD),
        "D": float(D),
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
    tau = 1.98e12 * (300.0 / muQ) ** 5
    gamma = 1.0 / tau

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
def _quark_state_entropy_residual(muB, muK, logT, a_target, w_target, Pi, jB, nB_Q, nK_Q, B_one_forth, ms=0.0, upB=5000):
    """
    Unscaled residual for the entropy-enabled local quark-state closure.
    """
    if w_target <= 0.0:
        raise RuntimeError("Entropy-enabled closure requires w > 0")
    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving an entropy-enabled quark state")

    logT = float(logT)
    if (not np.isfinite(logT)) or abs(logT) > 700.0 or logT < _ADIABATIC_LOCAL_LOGT_FLOOR:
        return np.array([1.0e12, 1.0e12, 1.0e12], dtype=float)

    T = float(np.exp(logT))
    try:
        thermo = _quark_thermo_state(muB, muK, B_one_forth, T, jB, ms=ms, upB=upB)
    except Exception:
        return np.array([1.0e12, 1.0e12, 1.0e12], dtype=float)

    return np.array(
        [
            thermo["Pi"] - Pi,
            (thermo["nK"] - nK_Q) / nB_Q - a_target,
            thermo["w"] - w_target,
        ],
        dtype=float,
    )


def _quark_state_entropy_residual_ok(residual, Pi, a_target, w_target):
    """
    Accept/reject an entropy-enabled local quark-state solve using relative
    tolerances matched to the three closure equations.
    """
    if not np.all(np.isfinite(residual)):
        return False
    pi_tol = 1.0e-8 * max(abs(Pi), 1.0)
    a_tol = 1.0e-8 * max(abs(a_target), 1.0)
    w_tol = 1.0e-8 * max(abs(w_target), 1.0)
    return bool(
        abs(float(residual[0])) <= pi_tol
        and abs(float(residual[1])) <= a_tol
        and abs(float(residual[2])) <= w_tol
    )


def _solve_quark_entropy_state_once_from_guess(a_target, w_target, Pi, jB, nB_Q, nK_Q, B_one_forth, ms=0.0, upB=5000, initial_guess=None, stats=None, stats_key="quark_state_entropy_root_calls"):
    """
    Try one entropy-enabled local quark-state root solve from a single
    continuation guess. The nonlinear solve is carried out in (muB, muK, logT).
    """
    if initial_guess is None:
        raise RuntimeError("initial_guess is required for single-guess entropy-enabled quark-state solve")
    if w_target <= 0.0:
        raise RuntimeError("Entropy-enabled closure requires w > 0")
    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving an entropy-enabled quark state")

    guess = np.asarray(initial_guess, dtype=float)
    if guess.shape[0] != 3 or not np.all(np.isfinite(guess)):
        raise RuntimeError("initial_guess must contain finite (muB, muK, T)")
    if guess[2] <= 0.0:
        raise RuntimeError("initial_guess temperature must be positive")

    pi_scale = max(abs(Pi), 1.0)
    w_scale = max(abs(w_target), 1.0)

    def equations(vec):
        residual = _quark_state_entropy_residual(
            float(vec[0]),
            float(vec[1]),
            float(vec[2]),
            a_target,
            w_target,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            ms=ms,
            upB=upB,
        )
        return np.array([residual[0] / pi_scale, residual[1], residual[2] / w_scale], dtype=float)

    if stats is not None:
        stats[stats_key] = stats.get(stats_key, 0) + 1
    sol = root(
        equations,
        np.array([float(guess[0]), float(guess[1]), float(np.log(float(guess[2])))], dtype=float),
        method="hybr",
        options={"maxfev": 180, "xtol": 1.0e-10},
    )
    if not (sol.success and np.all(np.isfinite(sol.x))):
        raise RuntimeError(f"single-guess entropy-enabled quark-state solve failed: {sol.message}")

    muB = float(sol.x[0])
    muK = float(sol.x[1])
    logT = float(sol.x[2])
    if muK < -1.0e-8:
        raise RuntimeError("single-guess entropy-enabled quark-state solve returned negative muK")
    if (not np.isfinite(logT)) or abs(logT) > 700.0:
        raise RuntimeError("single-guess entropy-enabled quark-state solve returned an invalid logT")

    if muK < 0.0:
        muK = 0.0
    T_loc = float(np.exp(logT))
    residual = _quark_state_entropy_residual(
        muB,
        muK,
        np.log(T_loc),
        a_target,
        w_target,
        Pi,
        jB,
        nB_Q,
        nK_Q,
        B_one_forth,
        ms=ms,
        upB=upB,
    )
    if not _quark_state_entropy_residual_ok(residual, Pi, a_target, w_target):
        raise RuntimeError(
            "single-guess entropy-enabled quark-state solve returned an unacceptable residual "
            f"({residual[0]:.3e}, {residual[1]:.3e}, {residual[2]:.3e})"
        )

    thermo = _quark_thermo_state(muB, muK, B_one_forth, T_loc, jB, ms=ms, upB=upB)
    if thermo["s"] <= 0.0:
        raise RuntimeError("single-guess entropy-enabled quark-state solve returned non-positive entropy density")
    if thermo["w"] <= 0.0:
        raise RuntimeError("single-guess entropy-enabled quark-state solve returned non-positive entropy flux")
    return thermo


def _solve_local_quark_state_from_a_w_and_Pi(a, w, Pi, jB, nB_Q, nK_Q, B_one_forth, ms=0.0, upB=5000, initial_guess=None, T_ref=None, stats=None):
    """
    Solve the local quark state (muB, muK, T, nB, u, s) at fixed (a, w, Pi, jB)
    using the entropy-enabled closure equations.
    """
    if stats is not None:
        stats["local_state_calls"] = stats.get("local_state_calls", 0) + 1
    if w <= 0.0:
        raise RuntimeError("Local entropy-enabled closure requires w > 0")

    if initial_guess is not None:
        try:
            return _solve_quark_entropy_state_once_from_guess(
                a,
                w,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                ms=ms,
                upB=upB,
                initial_guess=initial_guess,
                stats=stats,
                stats_key="local_root_calls",
            )
        except Exception:
            if stats is not None:
                stats["local_fast_failures"] = stats.get("local_fast_failures", 0) + 1

    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving an entropy-enabled quark state")

    guesses = []
    muK_seed = _branch_muK_seed(a)
    muK_seed_strong = float(max(muK_seed, 400.0 * abs(float(a))))
    if initial_guess is not None:
        guess0 = np.asarray(initial_guess, dtype=float)
        if guess0.shape[0] != 3 or not np.all(np.isfinite(guess0)):
            raise RuntimeError("initial_guess must contain finite (muB, muK, T)")
        if guess0[2] <= 0.0:
            raise RuntimeError("initial_guess temperature must be positive")
        guesses.append(guess0)
        muB_seed = float(guess0[0])
        muK_ref = max(0.0, float(guess0[1]))
        T_seed = float(guess0[2])
    else:
        if T_ref is None or (not np.isfinite(T_ref)) or T_ref <= 0.0:
            raise RuntimeError("T_ref must be positive when no entropy-closure initial_guess is provided")
        muB_seed = 1200.0
        muK_ref = muK_seed
        T_seed = float(T_ref)

    guesses.append(np.array([muB_seed, muK_seed, T_seed], dtype=float))
    guesses.append(np.array([1200.0, muK_seed, T_seed], dtype=float))
    guesses.append(np.array([1500.0, max(muK_seed, 100.0 * abs(float(a))), T_seed], dtype=float))
    guesses.append(np.array([muB_seed, muK_seed_strong, T_seed], dtype=float))
    guesses.append(np.array([1500.0, muK_seed_strong, T_seed], dtype=float))
    if T_ref is not None and np.isfinite(T_ref) and T_ref > 0.0:
        guesses.append(np.array([muB_seed, max(muK_seed, muK_ref), float(T_ref)], dtype=float))

    best_message = "Entropy-enabled local quark-state solve did not converge"
    candidates = []
    candidate_tol = 1.0e-8
    nonneg_tol = 1.0e-8
    for guess in guesses:
        try:
            thermo = _solve_quark_entropy_state_once_from_guess(
                a,
                w,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                ms=ms,
                upB=upB,
                initial_guess=guess,
                stats=stats,
                stats_key="local_root_calls",
            )
            if thermo["muK"] < -nonneg_tol or thermo["s"] <= 0.0 or thermo["w"] <= 0.0:
                continue
            is_new = True
            for cand in candidates:
                if (
                    abs(thermo["muB"] - cand["muB"]) <= candidate_tol * max(1.0, abs(cand["muB"]))
                    and abs(thermo["muK"] - cand["muK"]) <= candidate_tol * max(1.0, abs(cand["muK"]), 1.0)
                    and abs(thermo["T"] - cand["T"]) <= candidate_tol * max(1.0, abs(cand["T"]))
                ):
                    is_new = False
                    break
            if is_new:
                candidates.append(thermo)
        except Exception as exc:
            best_message = str(exc)

    if candidates:
        if initial_guess is not None:
            muB_ref = float(initial_guess[0])
            muK_ref = max(0.0, float(initial_guess[1]))
            T_pref = float(initial_guess[2])
        else:
            muB_ref = muB_seed
            muK_ref = muK_seed
            T_pref = T_seed

        muK_pref = max(muK_seed, muK_ref)
        candidates.sort(
            key=lambda cand: (
                abs(cand["muK"] - muK_pref),
                abs(np.log(cand["T"] / T_pref)),
                abs(cand["muB"] - muB_ref),
                -cand["muK"],
            )
        )
        return candidates[0]

    raise RuntimeError(f"Entropy-enabled local quark-state solve failed: {best_message}")


# Energy-conserving local closure helpers
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


def _quark_state_E_residual(muB, muK, logT, a_target, E_target, Pi, jB, nB_Q, nK_Q, B_one_forth, ms=0.0, upB=5000):
    """
    Unscaled residual for the energy-conserving local quark-state closure.
    """
    if E_target <= 0.0:
        raise RuntimeError("Energy-conserving closure requires E = h*u*r_gamma > 0")
    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving an energy-conserving quark state")

    logT = float(logT)
    if (not np.isfinite(logT)) or abs(logT) > 700.0 or logT < _ADIABATIC_LOCAL_LOGT_FLOOR:
        return np.array([1.0e12, 1.0e12, 1.0e12], dtype=float)

    T = float(np.exp(logT))
    try:
        thermo = _quark_thermo_state(muB, muK, B_one_forth, T, jB, ms=ms, upB=upB)
    except Exception:
        return np.array([1.0e12, 1.0e12, 1.0e12], dtype=float)

    return np.array(
        [
            thermo["Pi"] - Pi,
            (thermo["nK"] - nK_Q) / nB_Q - a_target,
            thermo["h"] * thermo["u"] * _relativistic_gamma_from_u(thermo["u"]) - E_target,
        ],
        dtype=float,
    )


def _quark_state_E_residual_ok(residual, Pi, a_target, E_target):
    """
    Accept/reject an energy-conserving local quark-state solve.
    """
    if not np.all(np.isfinite(residual)):
        return False
    pi_tol = 1.0e-8 * max(abs(Pi), 1.0)
    a_tol = 1.0e-8 * max(abs(a_target), 1.0)
    E_tol = 1.0e-8 * max(abs(E_target), 1.0)
    return bool(
        abs(float(residual[0])) <= pi_tol
        and abs(float(residual[1])) <= a_tol
        and abs(float(residual[2])) <= E_tol
    )


def _solve_quark_E_state_once_from_guess(a_target, E_target, Pi, jB, nB_Q, nK_Q, B_one_forth, ms=0.0, upB=5000, initial_guess=None, stats=None, stats_key="quark_state_E_root_calls"):
    """
    Try one energy-conserving local quark-state root solve from a single
    continuation guess. The nonlinear solve is carried out in (muB, muK, logT).
    """
    if initial_guess is None:
        raise RuntimeError("initial_guess is required for single-guess energy-conserving quark-state solve")
    if E_target <= 0.0:
        raise RuntimeError("Energy-conserving closure requires E = h*u*r_gamma > 0")
    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving an energy-conserving quark state")

    guess = np.asarray(initial_guess, dtype=float)
    if guess.shape[0] != 3 or not np.all(np.isfinite(guess)):
        raise RuntimeError("initial_guess must contain finite (muB, muK, T)")
    if guess[2] <= 0.0:
        raise RuntimeError("initial_guess temperature must be positive")

    pi_scale = max(abs(Pi), 1.0)
    E_scale = max(abs(E_target), 1.0)

    def equations(vec):
        residual = _quark_state_E_residual(
            float(vec[0]),
            float(vec[1]),
            float(vec[2]),
            a_target,
            E_target,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            ms=ms,
            upB=upB,
        )
        return np.array([residual[0] / pi_scale, residual[1], residual[2] / E_scale], dtype=float)

    if stats is not None:
        stats[stats_key] = stats.get(stats_key, 0) + 1
    sol = root(
        equations,
        np.array([float(guess[0]), float(guess[1]), float(np.log(float(guess[2])))], dtype=float),
        method="hybr",
        options={"maxfev": 180, "xtol": 1.0e-10},
    )
    if not (sol.success and np.all(np.isfinite(sol.x))):
        raise RuntimeError(f"single-guess energy-conserving quark-state solve failed: {sol.message}")

    muB = float(sol.x[0])
    muK = float(sol.x[1])
    logT = float(sol.x[2])
    if muK < -1.0e-8:
        raise RuntimeError("single-guess energy-conserving quark-state solve returned negative muK")
    if (not np.isfinite(logT)) or abs(logT) > 700.0:
        raise RuntimeError("single-guess energy-conserving quark-state solve returned an invalid logT")

    if muK < 0.0:
        muK = 0.0
    T_loc = float(np.exp(logT))
    residual = _quark_state_E_residual(
        muB,
        muK,
        np.log(T_loc),
        a_target,
        E_target,
        Pi,
        jB,
        nB_Q,
        nK_Q,
        B_one_forth,
        ms=ms,
        upB=upB,
    )
    if not _quark_state_E_residual_ok(residual, Pi, a_target, E_target):
        raise RuntimeError(
            "single-guess energy-conserving quark-state solve returned an unacceptable residual "
            f"({residual[0]:.3e}, {residual[1]:.3e}, {residual[2]:.3e})"
        )

    thermo = _quark_thermo_state(muB, muK, B_one_forth, T_loc, jB, ms=ms, upB=upB)
    r_gamma = _relativistic_gamma_from_u(thermo["u"])
    E = float(thermo["h"] * thermo["u"] * r_gamma)
    if thermo["h"] <= 0.0 or E <= 0.0:
        raise RuntimeError("single-guess energy-conserving quark-state solve returned non-positive E = h*u*r_gamma")
    thermo["r_gamma"] = r_gamma
    thermo["E"] = E
    return thermo


def _fixed_TQstar_E_residual(muB, muK, T_Qstar, E_target, Pi, jB, B_one_forth, ms=0.0, upB=5000):
    """
    Residual for the fixed-T_Qstar interface solve. The unknowns are
    (muB_Qstar, muK_Qstar); the temperature is prescribed.
    """
    if E_target <= 0.0:
        raise RuntimeError("Fixed-T_Qstar interface closure requires E = h*u*r_gamma > 0")
    T_Qstar = float(T_Qstar)
    if (not np.isfinite(T_Qstar)) or T_Qstar <= 0.0:
        raise RuntimeError("Fixed-T_Qstar interface closure requires T_Qstar > 0")
    try:
        thermo = _quark_thermo_state(muB, muK, B_one_forth, T_Qstar, jB, ms=ms, upB=upB)
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
    if (not np.isfinite(T_Qstar)) or T_Qstar <= 0.0:
        raise RuntimeError("Fixed-T_Qstar interface solve requires T_Qstar > 0")
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
            thermo = _quark_thermo_state(muB, muK, B_one_forth, T_Qstar, jB, ms=ms, upB=upB)
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


def _solve_local_quark_state_from_a_E_and_Pi(a, E, Pi, jB, nB_Q, nK_Q, B_one_forth, ms=0.0, upB=5000, initial_guess=None, T_ref=None, stats=None):
    """
    Solve the local quark state (muB, muK, T, nB, u, h) at fixed
    (a, E=h*u*r_gamma, Pi, jB) using the energy-conserving closure equations.
    """
    if stats is not None:
        stats["local_state_calls"] = stats.get("local_state_calls", 0) + 1
    if E <= 0.0:
        raise RuntimeError("Local energy-conserving closure requires E = h*u*r_gamma > 0")

    if initial_guess is not None:
        try:
            return _solve_quark_E_state_once_from_guess(
                a,
                E,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                ms=ms,
                upB=upB,
                initial_guess=initial_guess,
                stats=stats,
                stats_key="local_root_calls",
            )
        except Exception:
            if stats is not None:
                stats["local_fast_failures"] = stats.get("local_fast_failures", 0) + 1

    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving an energy-conserving quark state")

    guesses = []
    muK_seed = _branch_muK_seed(a)
    muK_seed_strong = float(max(muK_seed, 400.0 * abs(float(a))))
    if initial_guess is not None:
        guess0 = np.asarray(initial_guess, dtype=float)
        if guess0.shape[0] != 3 or not np.all(np.isfinite(guess0)):
            raise RuntimeError("initial_guess must contain finite (muB, muK, T)")
        if guess0[2] <= 0.0:
            raise RuntimeError("initial_guess temperature must be positive")
        guesses.append(guess0)
        muB_seed = float(guess0[0])
        muK_ref = max(0.0, float(guess0[1]))
        T_seed = float(guess0[2])
    else:
        if T_ref is None or (not np.isfinite(T_ref)) or T_ref <= 0.0:
            raise RuntimeError("T_ref must be positive when no energy-closure initial_guess is provided")
        muB_seed = 1200.0
        muK_ref = muK_seed
        T_seed = float(T_ref)

    guesses.append(np.array([muB_seed, muK_seed, T_seed], dtype=float))
    guesses.append(np.array([1200.0, muK_seed, T_seed], dtype=float))
    guesses.append(np.array([1500.0, max(muK_seed, 100.0 * abs(float(a))), T_seed], dtype=float))
    guesses.append(np.array([muB_seed, muK_seed_strong, T_seed], dtype=float))
    guesses.append(np.array([1500.0, muK_seed_strong, T_seed], dtype=float))
    if T_ref is not None and np.isfinite(T_ref) and T_ref > 0.0:
        guesses.append(np.array([muB_seed, max(muK_seed, muK_ref), float(T_ref)], dtype=float))

    best_message = "Energy-conserving local quark-state solve did not converge"
    candidates = []
    candidate_tol = 1.0e-8
    nonneg_tol = 1.0e-8
    for guess in guesses:
        try:
            thermo = _solve_quark_E_state_once_from_guess(
                a,
                E,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                ms=ms,
                upB=upB,
                initial_guess=guess,
                stats=stats,
                stats_key="local_root_calls",
            )
            if thermo["muK"] < -nonneg_tol or thermo["h"] <= 0.0 or thermo["E"] <= 0.0:
                continue
            is_new = True
            for cand in candidates:
                if (
                    abs(thermo["muB"] - cand["muB"]) <= candidate_tol * max(1.0, abs(cand["muB"]))
                    and abs(thermo["muK"] - cand["muK"]) <= candidate_tol * max(1.0, abs(cand["muK"]), 1.0)
                    and abs(thermo["T"] - cand["T"]) <= candidate_tol * max(1.0, abs(cand["T"]))
                ):
                    is_new = False
                    break
            if is_new:
                candidates.append(thermo)
        except Exception as exc:
            best_message = str(exc)

    if candidates:
        if initial_guess is not None:
            muB_ref = float(initial_guess[0])
            muK_ref = max(0.0, float(initial_guess[1]))
            T_pref = float(initial_guess[2])
        else:
            muB_ref = muB_seed
            muK_ref = muK_seed
            T_pref = T_seed

        muK_pref = max(muK_seed, muK_ref)
        candidates.sort(
            key=lambda cand: (
                abs(cand["muK"] - muK_pref),
                abs(np.log(cand["T"] / T_pref)),
                abs(cand["muB"] - muB_ref),
                -cand["muK"],
            )
        )
        return candidates[0]

    raise RuntimeError(f"Energy-conserving local quark-state solve failed: {best_message}")


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
def _strip_entropy_profile_fields(result):
    """
    Return a shallow copy of an entropy-solver result without profile arrays.
    """
    result_out = dict(result)
    for key in (
        "s_coord",
        "x",
        "a",
        "q",
        "w",
        "u",
        "nB",
        "muB",
        "muK",
        "T_profile",
        "s_profile",
        "D_profile",
        "eta_profile",
        "gamma_profile",
    ):
        result_out.pop(key, None)
    return result_out


def _strip_energy_profile_fields(result):
    """
    Return a shallow copy of an energy-solver result without profile arrays.
    """
    result_out = dict(result)
    for key in (
        "s_coord",
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
        "eta_profile",
        "gamma_profile",
    ):
        result_out.pop(key, None)
    return result_out


def _adiabatic_default_TQ_guesses(T):
    """
    Choose the default downstream T_Q seeds for low-temperature adiabatic solves.
    """
    T = float(T)
    if (not np.isfinite(T)) or T <= 0.0:
        raise RuntimeError("Adiabatic T_Q seed requires T > 0")
    if T < _ADIABATIC_LOW_T_THRESHOLD:
        hot = float(T + _ADIABATIC_HOT_START_OFFSET)
        return (hot, float(hot + _ADIABATIC_HOT_START_OFFSET))
    return (T,)


def _adiabatic_initial_state_failed(result):
    """
    Detect the adiabatic failure mode where no usable initial state was built.
    """
    return str(result.get("bvp_message", "")) == "initial_state_failure"


def _seed_adiabatic_jB_guess(
    T,
    nB_N,
    B_one_forth,
    aQstar_seed,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    verb=False,
):
    """
    Build a simple isothermal j_B seed for the adiabatic continuation solve.
    """
    heuristic_guess = float(max(1.0e-12, 1.0e-8 * float(nB_N)))
    seed_tol = float(max(1.0e-3, tol_bvp))
    seed_mesh = int(min(max(int(n_mesh), 60), 120))
    seed_nodes = int(min(max(int(max_nodes), 500), 2000))
    try:
        res_seed = solve_front_isothermal(
            T,
            nB_N,
            B_one_forth,
            aQstar_seed,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=max(float(tail_eps), 1.0e-6),
            n_mesh=seed_mesh,
            tol_bvp=seed_tol,
            max_nodes=seed_nodes,
            jB_guess=heuristic_guess,
            return_profile=False,
            verb=verb,
        )
        jB_seed = float(res_seed.get("jB", np.nan))
        if bool(res_seed.get("success")) and np.isfinite(jB_seed) and jB_seed > 0.0:
            return jB_seed
    except Exception:
        pass
    return heuristic_guess


def _solve_front_adiabatic_once(
    T,
    nB_N,
    B_one_forth,
    aQstar,
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
    interface_temperature_mode="hu_const",
    return_profile=False,
    verb=False,
    profile_guess=None,
    seed_profile=False,
):
    """
    Solve the entropy-enabled steady-front problem as a one-shot compact-coordinate BVP
    with downstream temperature T_Q promoted to a global unknown.

    The ODE unknowns are (a, q, w) with
        w = s * u
    and the BVP parameter vector is (jB, T_Q).
    """
    T = float(T)
    if (not np.isfinite(T)) or T <= 0.0:
        raise RuntimeError("solve_front_adiabatic requires T > 0")
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
            print(f"[front_adiabatic +{dt:8.2f}s] {msg}", flush=True)

    if jB_guess is None:
        jB_guess = 1.0e-8 * nB_N
    jB_guess = float(jB_guess)
    if jB_guess <= 0.0:
        raise RuntimeError("jB_guess must be positive")

    if TQ_guess is None:
        TQ_guess = _adiabatic_default_TQ_guesses(T)[0]
    TQ_guess = float(TQ_guess)
    if (not np.isfinite(TQ_guess)) or TQ_guess <= 0.0:
        raise RuntimeError("TQ_guess must be positive")
    interface_temperature_mode = _normalize_interface_temperature_mode(interface_temperature_mode)

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

    def _param_from_TQ(TQ):
        TQ = float(TQ)
        if (not np.isfinite(TQ)) or TQ <= 0.0:
            raise RuntimeError("T_Q must be positive")
        return float(np.log(TQ))

    def _TQ_from_param(phi):
        return float(np.exp(np.clip(float(phi), -700.0, 700.0)))

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
    muB_Q_last_guess = {"value": None}
    qstar_last_guess = {"value": None}
    last_failure = {"message": ""}
    s_coord_end = float(1.0 - tail_eps)

    def _build_global_state(theta, phi):
        key = (round(float(theta), 12), round(float(phi), 12))
        if key in state_cache:
            return state_cache[key]

        jB = _jB_from_param(theta)
        T_Q = _TQ_from_param(phi)
        if (not np.isfinite(T_Q)) or T_Q <= 0.0:
            raise RuntimeError("Adiabatic solver requires T_Q > 0")

        stats["global_state_builds"] += 1

        # Upstream nuclear state N at x = 0^-.
        P_N = float(PNM_n(nB_N, T, param=param, NM_type=NM_type))
        e_N = float(edensNM_n(nB_N, T, param=param))
        h_N = float(P_N + e_N)
        h_over_nB_N = float(h_N / nB_N)
        u_N = float(jB / nB_N)
        Pi = float(h_N * u_N * u_N + P_N)

        # Far-right equilibrated quark state Q with muK = 0 and solved T_Q.
        muB_Q_guess = muB_Q_last_guess["value"]
        if muB_Q_guess is None and isinstance(profile_guess, dict):
            muB_Q_guess = profile_guess.get("muB_Q", None)
        muB_Q = _solve_muB_Q_at_muK0_for_given_Pi_ms(
            Pi,
            jB,
            B_one_forth,
            T_Q,
            ms=ms,
            upB=upB,
            stats=stats,
            initial_guess=muB_Q_guess,
        )
        muB_Q_last_guess["value"] = float(muB_Q)
        thermo_Q = _quark_thermo_state(muB_Q, 0.0, B_one_forth, T_Q, jB, ms=ms, upB=upB)
        nB_Q = float(thermo_Q["nB"])
        if abs(ms) <= 1.0e-12:
            nK_Q = 0.0
        else:
            nK_Q = float(thermo_Q["nK"])

        a_N = float((nB_N - nK_Q) / nB_Q)

        # Interface state Qstar at x = 0+. In hu_const mode the temperature is
        # solved from h/nB continuity; in the fixed-temperature modes it is T.
        muK_Qstar_seed = _branch_muK_seed(aQstar)
        qstar_initial_guess = (muB_Q, muK_Qstar_seed, T_Q)
        if qstar_last_guess["value"] is not None:
            qstar_initial_guess = qstar_last_guess["value"]
        elif isinstance(profile_guess, dict):
            muB_Qstar_guess = profile_guess.get("muB_Qstar", None)
            muK_Qstar_guess = profile_guess.get("muK_Qstar", None)
            if muB_Qstar_guess is not None and muK_Qstar_guess is not None:
                try:
                    muB_Qstar_guess = float(muB_Qstar_guess)
                    muK_Qstar_guess = float(muK_Qstar_guess)
                except Exception:
                    muB_Qstar_guess = None
                    muK_Qstar_guess = None
                if (
                    muB_Qstar_guess is not None
                    and muK_Qstar_guess is not None
                    and np.isfinite(muB_Qstar_guess)
                    and np.isfinite(muK_Qstar_guess)
                    and muB_Qstar_guess > 0.0
                    and muK_Qstar_guess >= 0.0
                ):
                    T_Qstar_guess = profile_guess.get("T_Qstar", T_Q)
                    try:
                        T_Qstar_guess = float(T_Qstar_guess)
                    except Exception:
                        T_Qstar_guess = T_Q
                    if (not np.isfinite(T_Qstar_guess)) or T_Qstar_guess <= 0.0:
                        T_Qstar_guess = T_Q
                    qstar_initial_guess = (muB_Qstar_guess, muK_Qstar_guess, T_Qstar_guess)

        if interface_temperature_mode == "hu_const":
            thermo_Qstar = _solve_interface_Qstar_enthalpy_jump(
                aQstar,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                h_over_nB_N,
                T,
                T_Q,
                ms=ms,
                upB=upB,
                initial_guess=qstar_initial_guess,
                stats=stats,
                stats_key="qstar_root_calls",
            )
            muB_Qstar = float(thermo_Qstar["muB"])
            muK_Qstar = float(thermo_Qstar["muK"])
            qstar_last_guess["value"] = (muB_Qstar, muK_Qstar, float(thermo_Qstar["T"]))
        else:
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
                initial_guess=np.asarray(qstar_initial_guess, dtype=float)[:2],
                stats=stats,
                stats_key="qstar_root_calls",
            )
            qstar_last_guess["value"] = (float(muB_Qstar), float(muK_Qstar))
            thermo_Qstar = _quark_thermo_state(muB_Qstar, muK_Qstar, B_one_forth, T, jB, ms=ms, upB=upB)

        if thermo_Qstar["s"] <= 0.0:
            raise RuntimeError("Interface Qstar state has non-positive entropy density")
        if thermo_Qstar["w"] <= 0.0:
            raise RuntimeError("Interface Qstar state has non-positive entropy flux")
        h_over_nB_Qstar = float(thermo_Qstar["h"] / thermo_Qstar["nB"])
        h_over_nB_jump_residual = float(h_over_nB_Qstar - h_over_nB_N)
        w0 = float(thermo_Qstar["w"])

        micro_Q = _microphysics_from_quark_state(muB_Q, T_Q)
        D_Q = float(micro_Q["D"])
        eta_Q = float(micro_Q["eta"])
        gamma_Q = float(micro_Q["gamma"])
        tau_Q = float(micro_Q["tau"])

        u_Q = float(thermo_Q["u"])
        disc = float(u_Q * u_Q + 4.0 * D_Q * gamma_Q * eta_Q)
        if (not np.isfinite(disc)) or disc <= 0.0:
            raise RuntimeError("Adiabatic solver tail discriminant is non-positive")
        lam = float((-u_Q + np.sqrt(disc)) / (2.0 * D_Q))
        if (not np.isfinite(lam)) or lam <= 0.0:
            raise RuntimeError("Adiabatic solver tail decay lambda must be positive")
        q0 = float(-a_N * u_N)
        w_Q = float(thermo_Q["w"])
        x_end = float(-np.log1p(-s_coord_end) / lam)
        tail_coeff_Q = float(D_Q * lam + u_Q)
        state = {
            "jB": float(jB),
            "T_Q": float(T_Q),
            "P_N": P_N,
            "e_N": e_N,
            "h_N": h_N,
            "h_over_nB_N": h_over_nB_N,
            "u_N": u_N,
            "Pi": Pi,
            "muB_Q": float(muB_Q),
            "nB_Q": float(nB_Q),
            "nK_Q": float(nK_Q),
            "a_N": float(a_N),
            "muB_Qstar": float(muB_Qstar),
            "muK_Qstar": float(muK_Qstar),
            "nB_Qstar": float(thermo_Qstar["nB"]),
            "s_Q": float(thermo_Q["s"]),
            "w_Q": float(w_Q),
            "s_Qstar": float(thermo_Qstar["s"]),
            "u_Qstar": float(thermo_Qstar["u"]),
            "T_Qstar": float(thermo_Qstar["T"]),
            "h_over_nB_Qstar": h_over_nB_Qstar,
            "h_over_nB_jump_residual": h_over_nB_jump_residual,
            "D_Q": D_Q,
            "eta_Q": eta_Q,
            "gamma_Q": gamma_Q,
            "tau_Q": tau_Q,
            "lambda": float(lam),
            "u_Q": float(u_Q),
            "q0": float(q0),
            "w0": float(w0),
            "tail_coeff_Q": float(tail_coeff_Q),
            "x_end": float(x_end),
        }
        state_cache[key] = state
        return state

    def _state_or_none(theta, phi):
        try:
            return _build_global_state(theta, phi)
        except Exception as exc:
            stats["global_state_failures"] += 1
            last_failure["message"] = str(exc)
            return None

    def _finite_difference_ua_gradient(a_center, w_center, state, thermo_center, B_one_forth, ms, upB, stats):
        """
        Return partial derivatives of u(a,w)*a at fixed global state.

        The replacement entropy equation contains d(u*a)/dx. Since u is
        obtained from the local thermodynamic closure, approximate its local
        Jacobian with centered finite differences and reuse the current
        thermodynamic state as the root-solve seed.
        """
        a_center = float(a_center)
        w_center = float(w_center)
        if (not np.isfinite(a_center)) or (not np.isfinite(w_center)) or w_center <= 0.0:
            raise RuntimeError("ua-gradient entropy derivative requires finite a and positive w")

        root_guess = (thermo_center["muB"], thermo_center["muK"], thermo_center["T"])
        step_scale = np.sqrt(np.finfo(float).eps)
        h_a = float(max(step_scale * max(abs(a_center), 1.0), 1.0e-7))
        h_w = float(max(step_scale * max(abs(w_center), 1.0), 1.0e-7))

        def ua_value(a_probe, w_probe):
            thermo_probe = _solve_local_quark_state_from_a_w_and_Pi(
                float(a_probe),
                float(w_probe),
                state["Pi"],
                state["jB"],
                state["nB_Q"],
                state["nK_Q"],
                B_one_forth,
                ms=ms,
                upB=upB,
                initial_guess=root_guess,
                T_ref=state["T_Q"],
                stats=stats,
            )
            return float(thermo_probe["u"] * float(a_probe))

        ua_a_plus = ua_value(a_center + h_a, w_center)
        ua_a_minus = ua_value(a_center - h_a, w_center)
        d_ua_da = float((ua_a_plus - ua_a_minus) / (2.0 * h_a))

        if w_center > h_w:
            ua_w_plus = ua_value(a_center, w_center + h_w)
            ua_w_minus = ua_value(a_center, w_center - h_w)
            d_ua_dw = float((ua_w_plus - ua_w_minus) / (2.0 * h_w))
        else:
            ua_center = float(thermo_center["u"] * a_center)
            ua_w_plus = ua_value(a_center, w_center + h_w)
            d_ua_dw = float((ua_w_plus - ua_center) / h_w)

        if (not np.isfinite(d_ua_da)) or (not np.isfinite(d_ua_dw)):
            raise RuntimeError("ua-gradient entropy derivative returned non-finite values")
        return {"d_ua_da": d_ua_da, "d_ua_dw": d_ua_dw}

    def _ode(s_coord, y, p):
        stats["bvp_ode_calls"] += 1
        state = _state_or_none(float(p[0]), float(p[1]))
        if state is None:
            return np.zeros_like(y) + 1.0e12

        dyds = np.empty_like(y)
        guess = (state["muB_Qstar"], state["muK_Qstar"], state["T_Qstar"])
        for i in range(y.shape[1]):
            a_val = float(y[0, i])
            q_val = float(y[1, i])
            w_val = float(y[2, i])
            if (not np.isfinite(a_val)) or (not np.isfinite(q_val)) or (not np.isfinite(w_val)):
                stats["local_state_failures"] += 1
                dyds[:, i] = 1.0e12
                continue
            try:
                thermo_loc = _solve_local_quark_state_from_a_w_and_Pi(
                    a_val,
                    w_val,
                    state["Pi"],
                    state["jB"],
                    state["nB_Q"],
                    state["nK_Q"],
                    B_one_forth,
                    ms=ms,
                    upB=upB,
                    initial_guess=guess,
                    T_ref=state["T_Q"],
                    stats=stats,
                )
                guess = (thermo_loc["muB"], thermo_loc["muK"], thermo_loc["T"])
                micro_loc = _microphysics_from_quark_state(thermo_loc["muB"], thermo_loc["T"])
                reaction = float(micro_loc["gamma"] * (a_val**3 + micro_loc["eta"] * a_val))
                one_minus_s = max(1.0 - float(s_coord[i]), np.finfo(float).tiny)
                dx_ds = 1.0 / (state["lambda"] * one_minus_s)
                da_dx = float((q_val + thermo_loc["u"] * a_val) / micro_loc["D"])
                dyds[0, i] = da_dx * dx_ds
                dyds[1, i] = reaction * dx_ds
                ua_grad = _finite_difference_ua_gradient(
                    a_val,
                    w_val,
                    state,
                    thermo_loc,
                    B_one_forth,
                    ms,
                    upB,
                    stats,
                )
                coeff = float((state["nB_Q"] / thermo_loc["T"]) * thermo_loc["muK"])
                denom = float(1.0 + coeff * ua_grad["d_ua_dw"])
                if (not np.isfinite(denom)) or abs(denom) <= 1.0e-12:
                    raise RuntimeError("ua-gradient entropy equation has singular local denominator")
                dw_dx = float(-coeff * ua_grad["d_ua_da"] * da_dx / denom)
                dyds[2, i] = dw_dx * dx_ds
            except Exception as exc:
                stats["local_state_failures"] += 1
                last_failure["message"] = str(exc)
                dyds[:, i] = 1.0e12
        return dyds

    def _bc(ya, yb, p):
        stats["bvp_bc_calls"] += 1
        state = _state_or_none(float(p[0]), float(p[1]))
        if state is None:
            return np.array([1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12], dtype=float)
        return np.array(
            [
                ya[0] - float(aQstar),
                ya[1] - state["q0"],
                ya[2] - state["w0"],
                yb[1] + state["tail_coeff_Q"] * yb[0],
                yb[2] - state["w_Q"],
            ],
            dtype=float,
        )

    theta_guess = _param_from_jB(jB_guess)
    phi_guess = _param_from_TQ(TQ_guess)
    state0 = _state_or_none(theta_guess, phi_guess)
    if state0 is None:
        return {
            "success": False,
            "message": f"Initial adiabatic state construction failed: {last_failure['message']}",
            "aQstar": float(aQstar),
            "jB": np.nan,
            "T_Q": np.nan,
            "T_N": float(T),
            "T_Qstar": np.nan,
            "w_Q": np.nan,
            "interface_temperature_mode": interface_temperature_mode,
            "h_over_nB_N": np.nan,
            "h_over_nB_Qstar": np.nan,
            "h_over_nB_jump_residual": np.nan,
            "branch_label": "muK-rich",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "entropy_tail_residual": np.nan,
            "entropy_tail_residual_norm": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "w_end": np.nan,
            "bvp_status": -999,
            "bvp_message": "initial_state_failure",
            "bvp_niter": -1,
            "bvp_nodes": 0,
            **{k: int(v) for k, v in stats.items()},
        }

    s_coord_mesh = np.linspace(0.0, s_coord_end, int(n_mesh))
    blend = s_coord_mesh / max(s_coord_end, np.finfo(float).tiny)
    tail_shape = np.maximum(1.0 - s_coord_mesh, tail_eps)

    def _default_initial_guess():
        a_guess_loc = float(aQstar) * tail_shape
        q_tail_guess_loc = -state0["tail_coeff_Q"] * a_guess_loc
        q_guess_loc = (1.0 - blend) * state0["q0"] + blend * q_tail_guess_loc
        w_guess_loc = (1.0 - blend) * state0["w0"] + blend * state0["w_Q"]
        return a_guess_loc, q_guess_loc, w_guess_loc

    def _profile_initial_guess(profile):
        seed_input = profile
        try:
            s_prev = np.asarray(seed_input["s_coord"], dtype=float)
            a_prev = np.asarray(seed_input["a"], dtype=float)
            q_prev = np.asarray(seed_input["q"], dtype=float)
            w_prev = np.asarray(seed_input["w"], dtype=float)
        except Exception:
            return _default_initial_guess()

        if (
            s_prev.ndim != 1
            or a_prev.shape != s_prev.shape
            or q_prev.shape != s_prev.shape
            or w_prev.shape != s_prev.shape
            or s_prev.size < 5
            or not np.all(np.isfinite(s_prev))
            or not np.all(np.isfinite(a_prev))
            or not np.all(np.isfinite(q_prev))
            or not np.all(np.isfinite(w_prev))
        ):
            return _default_initial_guess()

        a_prev0 = float(a_prev[0])
        if abs(a_prev0) > 1.0e-12:
            a_scale = float(aQstar) / a_prev0
        else:
            a_scale = 1.0

        a_guess_loc = np.interp(s_coord_mesh, s_prev, a_prev) * a_scale
        a_guess_loc[0] = float(aQstar)

        q_base = np.interp(s_coord_mesh, s_prev, q_prev) * a_scale
        q_base0 = float(q_base[0])
        q_base_end = float(q_base[-1])
        q_target0 = float(state0["q0"])
        q_target_end = float(-state0["tail_coeff_Q"] * a_guess_loc[-1])
        q_span = q_base_end - q_base0
        if np.isfinite(q_span) and abs(q_span) > 1.0e-12 * max(1.0, abs(q_base0), abs(q_base_end)):
            q_frac = (q_base - q_base0) / q_span
            q_guess_loc = q_target0 + q_frac * (q_target_end - q_target0)
        else:
            q_guess_loc = (1.0 - blend) * q_target0 + blend * q_target_end
        q_guess_loc[0] = q_target0

        w_base = np.interp(s_coord_mesh, s_prev, w_prev)
        w_base0 = float(w_base[0])
        w_base_end = float(w_base[-1])
        w_span = w_base_end - w_base0
        if (
            np.all(np.isfinite(w_base))
            and w_base0 > 0.0
            and w_base_end > 0.0
            and abs(w_span) > 1.0e-12 * max(1.0, abs(w_base0), abs(w_base_end))
        ):
            w_frac = (w_base - w_base0) / w_span
            w_guess_loc = state0["w0"] + w_frac * (state0["w_Q"] - state0["w0"])
        else:
            w_guess_loc = (1.0 - blend) * state0["w0"] + blend * state0["w_Q"]

        if (not np.all(np.isfinite(a_guess_loc))) or (not np.all(np.isfinite(q_guess_loc))) or (not np.all(np.isfinite(w_guess_loc))) or np.any(w_guess_loc <= 0.0):
            return _default_initial_guess()

        w_guess_loc[0] = float(state0["w0"])
        return a_guess_loc, q_guess_loc, w_guess_loc

    if profile_guess is None:
        a_guess, q_guess, w_guess = _default_initial_guess()
    else:
        a_guess, q_guess, w_guess = _profile_initial_guess(profile_guess)
    y_guess = np.vstack((a_guess, q_guess, w_guess))

    _diag(
        f"starting adiabatic compact BVP with jB_guess={jB_guess:.6g}, "
        f"TQ_guess={TQ_guess:.6g}, aQstar={aQstar:.6g}, tail_eps={tail_eps:.3g}, branch=muK-rich"
    )

    try:
        sol = solve_bvp(
            _ode,
            _bc,
            s_coord_mesh,
            y_guess,
            p=np.array([theta_guess, phi_guess], dtype=float),
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
            "T_Q": np.nan,
            "T_N": float(T),
            "T_Qstar": np.nan,
            "w_Q": np.nan,
            "interface_temperature_mode": interface_temperature_mode,
            "h_over_nB_N": np.nan,
            "h_over_nB_Qstar": np.nan,
            "h_over_nB_jump_residual": np.nan,
            "branch_label": "muK-rich",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "entropy_tail_residual": np.nan,
            "entropy_tail_residual_norm": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "w_end": np.nan,
            "bvp_status": -999,
            "bvp_message": f"solve_bvp raised: {exc}",
            "bvp_niter": -1,
            "bvp_nodes": 0,
            **{k: int(v) for k, v in stats.items()},
        }

    theta_sol = float(sol.p[0])
    phi_sol = float(sol.p[1])
    state = _state_or_none(theta_sol, phi_sol)
    if state is None:
        return {
            "success": False,
            "message": f"BVP final state construction failed: {last_failure['message']}",
            "aQstar": float(aQstar),
            "jB": np.nan,
            "T_Q": np.nan,
            "T_N": float(T),
            "T_Qstar": np.nan,
            "w_Q": np.nan,
            "interface_temperature_mode": interface_temperature_mode,
            "h_over_nB_N": np.nan,
            "h_over_nB_Qstar": np.nan,
            "h_over_nB_jump_residual": np.nan,
            "branch_label": "muK-rich",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "entropy_tail_residual": np.nan,
            "entropy_tail_residual_norm": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "w_end": np.nan,
            "bvp_status": int(sol.status),
            "bvp_message": str(sol.message),
            "bvp_niter": int(getattr(sol, "niter", -1)),
            "bvp_nodes": int(sol.x.size),
            **{k: int(v) for k, v in stats.items()},
        }

    a_end = float(sol.y[0, -1])
    q_end = float(sol.y[1, -1])
    w_end = float(sol.y[2, -1])
    tail_drive = float(state["tail_coeff_Q"] * a_end)
    tail_residual = float(q_end + tail_drive)
    tail_scale = float(max(abs(q_end), abs(tail_drive), np.finfo(float).tiny))
    tail_residual_norm = float(tail_residual / tail_scale)
    entropy_tail_residual = float(w_end - state["w_Q"])
    entropy_tail_scale = float(max(abs(w_end), abs(state["w_Q"]), np.finfo(float).tiny))
    entropy_tail_residual_norm = float(entropy_tail_residual / entropy_tail_scale)

    success = bool(
        sol.success
        and np.isfinite(tail_residual_norm)
        and np.isfinite(entropy_tail_residual_norm)
        and abs(tail_residual_norm) <= max(tol_bvp, 10.0 * np.finfo(float).eps)
        and abs(entropy_tail_residual_norm) <= max(tol_bvp, 10.0 * np.finfo(float).eps)
    )
    result = {
        "success": success,
        "message": "Adiabatic compact BVP steady-front solve converged" if success else f"{sol.message}; last failure: {last_failure['message']}",
        "jB": float(state["jB"]),
        "T_Q": float(state["T_Q"]),
        "T_N": float(T),
        "T_Qstar": float(state["T_Qstar"]),
        "w_Q": float(state["w_Q"]),
        "aQstar": float(aQstar),
        "branch_label": "muK-rich",
        "interface_temperature_mode": interface_temperature_mode,
        "entropy_flux_equation": "ua_gradient",
        "coordinate": "BVP: s_coord in [0, 1-tail_eps], s_coord=1-exp(-lambda*x)",
        "tail_eps": float(tail_eps),
        "u_N": float(state["u_N"]),
        "u_Q": float(state["u_Q"]),
        "a_N": float(state["a_N"]),
        "Pi": float(state["Pi"]),
        "muB_Q": float(state["muB_Q"]),
        "nB_Q": float(state["nB_Q"]),
        "nK_Q": float(state["nK_Q"]),
        "s_Q": float(state["s_Q"]),
        "muB_Qstar": float(state["muB_Qstar"]),
        "muK_Qstar": float(state["muK_Qstar"]),
        "nB_Qstar": float(state["nB_Qstar"]),
        "s_Qstar": float(state["s_Qstar"]),
        "h_over_nB_N": float(state["h_over_nB_N"]),
        "h_over_nB_Qstar": float(state["h_over_nB_Qstar"]),
        "h_over_nB_jump_residual": float(state["h_over_nB_jump_residual"]),
        "w0": float(state["w0"]),
        "D_Q": float(state["D_Q"]),
        "eta_Q": float(state["eta_Q"]),
        "gamma_Q": float(state["gamma_Q"]),
        "tau_Q": float(state["tau_Q"]),
        "lambda": float(state["lambda"]),
        "kappa": float(1.0 / state["lambda"]),
        "s_end": float(s_coord_end),
        "x_end": float(state["x_end"]),
        "a_end": a_end,
        "q_end": q_end,
        "w_end": w_end,
        "tail_residual": tail_residual,
        "tail_residual_norm": tail_residual_norm,
        "tail_scale": tail_scale,
        "entropy_tail_residual": entropy_tail_residual,
        "entropy_tail_residual_norm": entropy_tail_residual_norm,
        "entropy_tail_scale": entropy_tail_scale,
        "_residual": np.array([tail_residual_norm, entropy_tail_residual_norm], dtype=float),
        "_root_method": "solve_bvp_parameter_2d",
        "bvp_status": int(sol.status),
        "bvp_message": str(sol.message),
        "bvp_niter": int(getattr(sol, "niter", -1)),
        "bvp_nodes": int(sol.x.size),
        **{k: int(v) for k, v in stats.items()},
    }

    if return_profile or seed_profile:
        s_coord_prof = np.asarray(sol.x, dtype=float)
        a_prof = np.asarray(sol.y[0], dtype=float)
        q_prof = np.asarray(sol.y[1], dtype=float)
        w_prof = np.asarray(sol.y[2], dtype=float)
        x_prof = -np.log1p(-s_coord_prof) / float(state["lambda"])
        result.update(
            {
                "s_coord": s_coord_prof,
                "x": x_prof,
                "a": a_prof,
                "q": q_prof,
                "w": w_prof,
            }
        )
    if return_profile and success:
        muB_prof = np.empty_like(s_coord_prof)
        muK_prof = np.empty_like(s_coord_prof)
        T_prof = np.empty_like(s_coord_prof)
        nB_prof = np.empty_like(s_coord_prof)
        u_prof = np.empty_like(s_coord_prof)
        entropy_prof = np.empty_like(s_coord_prof)
        D_prof = np.empty_like(s_coord_prof)
        eta_prof = np.empty_like(s_coord_prof)
        gamma_prof = np.empty_like(s_coord_prof)
        guess = (state["muB_Qstar"], state["muK_Qstar"], state["T_Qstar"])
        try:
            for i, a_val in enumerate(a_prof):
                stats["profile_state_calls"] += 1
                thermo_loc = _solve_local_quark_state_from_a_w_and_Pi(
                    float(a_val),
                    float(w_prof[i]),
                    state["Pi"],
                    state["jB"],
                    state["nB_Q"],
                    state["nK_Q"],
                    B_one_forth,
                    ms=ms,
                    upB=upB,
                    initial_guess=guess,
                    T_ref=state["T_Q"],
                    stats=stats,
                )
                guess = (thermo_loc["muB"], thermo_loc["muK"], thermo_loc["T"])
                micro_loc = _microphysics_from_quark_state(thermo_loc["muB"], thermo_loc["T"])
                muB_prof[i] = thermo_loc["muB"]
                muK_prof[i] = thermo_loc["muK"]
                T_prof[i] = thermo_loc["T"]
                nB_prof[i] = thermo_loc["nB"]
                u_prof[i] = thermo_loc["u"]
                entropy_prof[i] = thermo_loc["s"]
                D_prof[i] = micro_loc["D"]
                eta_prof[i] = micro_loc["eta"]
                gamma_prof[i] = micro_loc["gamma"]
        except Exception as exc:
            result["success"] = False
            result["message"] = f"{result['message']}; profile reconstruction failed: {exc}"
            result["profile_reconstruction_failed"] = True
            result["profile_reconstruction_message"] = str(exc)
            result["profile_state_calls"] = int(stats["profile_state_calls"])
            result["local_state_calls"] = int(stats["local_state_calls"])
            result["local_root_calls"] = int(stats["local_root_calls"])
            result["local_fast_failures"] = int(stats["local_fast_failures"])
            if simple_diag:
                print(
            f"adiabatic-bvp profile reconstruction failed for aQstar={aQstar:.6g}: {exc}"
                )
            return result

        result.update(
            {
                "u": u_prof,
                "nB": nB_prof,
                "muB": muB_prof,
                "muK": muK_prof,
                "T_profile": T_prof,
                "s_profile": entropy_prof,
                "D_profile": D_prof,
                "eta_profile": eta_prof,
                "gamma_profile": gamma_prof,
                "profile_state_calls": int(stats["profile_state_calls"]),
                "local_state_calls": int(stats["local_state_calls"]),
                "local_root_calls": int(stats["local_root_calls"]),
                "local_fast_failures": int(stats["local_fast_failures"]),
            }
        )

    if simple_diag:
        print(
            f"adiabatic-bvp jB={result['jB']:.6g}, T_Q={result['T_Q']:.6g}, "
            f"aQstar={aQstar:.6g}, tail_norm={tail_residual_norm:.6g}, "
            f"w_tail_norm={entropy_tail_residual_norm:.6g}, status={sol.status}, success={success}"
        )
    return result


# Public adiabatic front solver
def solve_front_adiabatic(
    T,
    nB_N,
    B_one_forth,
    aQstar,
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
    interface_temperature_mode="hu_const",
    return_profile=False,
    verb=False,
):
    """
    Solve the adiabatic steady-front problem as a compact-coordinate BVP
    with downstream asymptotic temperature T_Q treated as a global unknown.

    The solver first attempts a direct one-shot solve. If that fails, it
    automatically falls back to continuation in aQstar, reusing the last
    converged entropy profile and (jB, T_Q) pair as the next BVP initial guess.
    """
    aQstar = float(aQstar)
    abs_target = abs(aQstar)
    sign = -1.0 if aQstar < 0.0 else 1.0

    if isinstance(verb, str):
        verb_mode = "full" if verb.lower() == "full" else ("simple" if verb else "off")
    else:
        verb_mode = "simple" if verb else "off"
    simple_diag = verb_mode in ("simple", "full")
    interface_temperature_mode = _normalize_interface_temperature_mode(interface_temperature_mode)

    direct_jB_guess = jB_guess
    if direct_jB_guess is None:
        # Keep the first direct attempt cheap, but seed it on the rough flux
        # scale implied by both the upstream density and the interface aQstar.
        direct_jB_guess = float(max(1.0e-12, 1.0e-8 * float(nB_N), 0.8 * abs_target))

    if TQ_guess is None:
        TQ_guess_candidates = _adiabatic_default_TQ_guesses(T)
    else:
        TQ_guess_candidates = (float(TQ_guess),)
    direct_TQ_guess = float(TQ_guess_candidates[0])
    if (not np.isfinite(direct_TQ_guess)) or direct_TQ_guess <= 0.0:
        raise RuntimeError("TQ_guess must be positive")

    direct_result = _solve_front_adiabatic_once(
        T,
        nB_N,
        B_one_forth,
        aQstar,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=n_mesh,
        tol_bvp=tol_bvp,
        max_nodes=max_nodes,
        jB_guess=direct_jB_guess,
        TQ_guess=direct_TQ_guess,
        jB_bounds=jB_bounds,
        interface_temperature_mode=interface_temperature_mode,
        return_profile=return_profile,
        verb=verb,
        profile_guess=None,
    )
    if (
        TQ_guess is None
        and len(TQ_guess_candidates) > 1
        and _adiabatic_initial_state_failed(direct_result)
    ):
        retry_TQ_guess = float(TQ_guess_candidates[1])
        if simple_diag:
            print(
                f"adiabatic-bvp low-T hot-start retry: initial state failed at "
                f"TQ_guess={direct_TQ_guess:.6g}; retrying with TQ_guess={retry_TQ_guess:.6g}"
            )
        direct_TQ_guess = retry_TQ_guess
        direct_result = _solve_front_adiabatic_once(
            T,
            nB_N,
            B_one_forth,
            aQstar,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=n_mesh,
            tol_bvp=tol_bvp,
            max_nodes=max_nodes,
            jB_guess=direct_jB_guess,
            TQ_guess=direct_TQ_guess,
            jB_bounds=jB_bounds,
            interface_temperature_mode=interface_temperature_mode,
            return_profile=return_profile,
            verb=verb,
            profile_guess=None,
        )
    if bool(direct_result.get("success")) or abs_target <= 1.0e-2:
        result_out = dict(direct_result)
        result_out["continuation_used"] = False
        result_out["continuation_steps"] = 0
        if return_profile:
            return result_out
        return _strip_entropy_profile_fields(result_out)

    if simple_diag:
        print(
            f"adiabatic-bvp continuation: direct solve failed at aQstar={aQstar:.6g}; "
            f"retrying with staged continuation"
        )

    stage_n_mesh = int(min(max(max(int(n_mesh) // 2, 25), 25), 40))
    stage_tol = float(max(float(tol_bvp), 2.0e-2))
    stage_max_nodes = int(max(stage_n_mesh + 5, min(int(max_nodes), 250)))
    base_step_abs = 1.0e-3
    min_step_abs = 1.0e-4
    step_abs = float(min(base_step_abs, max(abs_target - min(abs_target, 1.0e-2), 0.0)))
    current_abs = min(abs_target, 1.0e-2)
    stage_jB_guess = direct_jB_guess
    if not (np.isfinite(stage_jB_guess) and stage_jB_guess > 0.0):
        stage_jB_guess = None
    direct_jB_out = direct_result.get("jB", np.nan)
    if np.isfinite(direct_jB_out) and direct_jB_out > 0.0:
        stage_jB_guess = float(direct_jB_out)
    if stage_jB_guess is None:
        seed_abs = current_abs if current_abs > 0.0 else 1.0e-2
        stage_jB_guess = _seed_adiabatic_jB_guess(
            T,
            nB_N,
            B_one_forth,
            sign * seed_abs,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=n_mesh,
            tol_bvp=tol_bvp,
            max_nodes=max_nodes,
            verb=False,
        )
    current = _solve_front_adiabatic_once(
        T,
        nB_N,
        B_one_forth,
        sign * current_abs,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=stage_n_mesh,
        tol_bvp=stage_tol,
        max_nodes=stage_max_nodes,
        jB_guess=stage_jB_guess,
        TQ_guess=direct_TQ_guess,
        jB_bounds=jB_bounds,
        interface_temperature_mode=interface_temperature_mode,
        return_profile=False,
        verb=False,
        profile_guess=None,
        seed_profile=True,
    )
    if not bool(current.get("success")):
        failed = dict(current)
        failed["message"] = (
            f"{direct_result['message']}; continuation seed at aQstar={sign * current_abs:.6g} failed: "
            f"{current['message']}"
        )
        failed["continuation_used"] = True
        failed["continuation_steps"] = 0
        failed["continuation_seed_aQstar"] = float(sign * current_abs)
        if return_profile:
            return failed
        return _strip_entropy_profile_fields(failed)

    steps_taken = 0
    while current_abs < abs_target - 1.0e-12:
        next_abs = min(abs_target, current_abs + step_abs)
        next_a = sign * next_abs
        trial = _solve_front_adiabatic_once(
            T,
            nB_N,
            B_one_forth,
            next_a,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=stage_n_mesh,
            tol_bvp=stage_tol,
            max_nodes=stage_max_nodes,
            jB_guess=current["jB"],
            TQ_guess=current["T_Q"],
            jB_bounds=jB_bounds,
            interface_temperature_mode=interface_temperature_mode,
            return_profile=False,
            verb=False,
            profile_guess=current,
            seed_profile=True,
        )
        if bool(trial.get("success")):
            current = trial
            current_abs = next_abs
            steps_taken += 1
            if simple_diag and (steps_taken % 25 == 0 or current_abs >= abs_target - 1.0e-12):
                print(
                    f"adiabatic-bvp continuation: reached aQstar={next_a:.6g} "
                    f"with jB={current['jB']:.6g}, T_Q={current['T_Q']:.6g}"
                )
            step_abs = min(base_step_abs, max(abs_target - current_abs, 0.0))
            continue

        step_abs *= 0.5
        if step_abs < min_step_abs:
            failed = dict(trial)
            failed["message"] = (
                f"Adiabatic continuation failed after reaching aQstar={sign * current_abs:.6g}; "
                f"last attempted aQstar={next_a:.6g}. {trial['message']}"
            )
            failed["continuation_used"] = True
            failed["continuation_steps"] = steps_taken
            failed["continuation_seed_aQstar"] = float(sign * min(abs_target, 1.0e-2))
            if return_profile:
                return failed
            return _strip_entropy_profile_fields(failed)

    refined = _solve_front_adiabatic_once(
        T,
        nB_N,
        B_one_forth,
        aQstar,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=n_mesh,
        tol_bvp=tol_bvp,
        max_nodes=max_nodes,
        jB_guess=current["jB"],
        TQ_guess=current["T_Q"],
        jB_bounds=jB_bounds,
        interface_temperature_mode=interface_temperature_mode,
        return_profile=return_profile,
        verb=False,
        profile_guess=current,
    )
    if bool(refined.get("success")):
        result_out = dict(refined)
        result_out["continuation_refined"] = True
    else:
        if return_profile:
            coarse_profile = _solve_front_adiabatic_once(
                T,
                nB_N,
                B_one_forth,
                aQstar,
                ms=ms,
                param=param,
                NM_type=NM_type,
                tail_eps=tail_eps,
                n_mesh=stage_n_mesh,
                tol_bvp=stage_tol,
                max_nodes=stage_max_nodes,
                jB_guess=current["jB"],
                TQ_guess=current["T_Q"],
                jB_bounds=jB_bounds,
                interface_temperature_mode=interface_temperature_mode,
                return_profile=True,
                verb=False,
                profile_guess=current,
            )
            result_out = dict(coarse_profile if bool(coarse_profile.get("success")) else current)
        else:
            result_out = dict(current)
        result_out["message"] = (
            f"{result_out['message']}; final refinement failed: {refined['message']}"
        )
        result_out["continuation_refined"] = False

    result_out["continuation_used"] = True
    result_out["continuation_steps"] = steps_taken
    result_out["continuation_seed_aQstar"] = float(sign * min(abs_target, 1.0e-2))
    if return_profile:
        return result_out
    return _strip_entropy_profile_fields(result_out)


def _solve_front_energy_conserving_once(
    T,
    nB_N,
    B_one_forth,
    aQstar,
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
    profile_guess=None,
    seed_profile=False,
    TQstar_target=None,
):
    """
    Solve the energy-conserving steady-front problem as a compact-coordinate
    BVP with downstream temperature T_Q promoted to a global unknown.

    The ODE unknowns are (a, q). The conserved enthalpy flux
    E = h*u*sqrt(1+u^2) is derived from the upstream trial jB, like Pi.
    """
    fixed_TQstar_mode = TQstar_target is not None
    T = float(T)
    if (not np.isfinite(T)) or T <= 0.0:
        raise RuntimeError("solve_front_energy_conserving requires T > 0")
    if fixed_TQstar_mode:
        TQstar_target = float(TQstar_target)
        if (not np.isfinite(TQstar_target)) or TQstar_target <= 0.0:
            raise RuntimeError("TQstar_target must be positive")
        aQstar = np.nan if aQstar is None else float(aQstar)
    else:
        aQstar = float(aQstar)
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
            print(f"[front_energy +{dt:8.2f}s] {msg}", flush=True)

    if jB_guess is None:
        jB_guess = 1.0e-8 * nB_N
    jB_guess = float(jB_guess)
    if jB_guess <= 0.0:
        raise RuntimeError("jB_guess must be positive")

    if TQ_guess is None:
        TQ_guess = _adiabatic_default_TQ_guesses(T)[0]
    TQ_guess = float(TQ_guess)
    if (not np.isfinite(TQ_guess)) or TQ_guess <= 0.0:
        raise RuntimeError("TQ_guess must be positive")

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

    def _param_from_TQ(TQ):
        TQ = float(TQ)
        if (not np.isfinite(TQ)) or TQ <= 0.0:
            raise RuntimeError("T_Q must be positive")
        return float(np.log(TQ))

    def _TQ_from_param(phi):
        return float(np.exp(np.clip(float(phi), -700.0, 700.0)))

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
    muB_Q_last_guess = {"value": None}
    qstar_last_guess = {"value": None}
    last_failure = {"message": ""}
    s_coord_end = float(1.0 - tail_eps)

    def _nan_result(message, bvp_status=-999, bvp_message="initial_state_failure", sol=None):
        return {
            "success": False,
            "message": message,
            "aQstar": float(aQstar) if np.isfinite(aQstar) else np.nan,
            "jB": np.nan,
            "E": np.nan,
            "T_Q": np.nan,
            "T_N": float(T),
            "T_Qstar": np.nan,
            "E_N": np.nan,
            "E_Q": np.nan,
            "E_Qstar": np.nan,
            "r_gamma_N": np.nan,
            "r_gamma_Q": np.nan,
            "r_gamma_Qstar": np.nan,
            "branch_label": "muK-rich",
            "energy_flux_equation": "E_rgamma_const",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "E_right_residual": np.nan,
            "E_right_residual_norm": np.nan,
            "E_scale": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "bvp_status": int(bvp_status),
            "bvp_message": str(bvp_message),
            "bvp_niter": int(getattr(sol, "niter", -1)) if sol is not None else -1,
            "bvp_nodes": int(sol.x.size) if sol is not None else 0,
            "_root_method": "solve_bvp_parameter_2d",
            **{k: int(v) for k, v in stats.items()},
        }

    def _build_global_state(theta, phi):
        key = (round(float(theta), 12), round(float(phi), 12))
        if key in state_cache:
            return state_cache[key]

        jB = _jB_from_param(theta)
        T_Q = _TQ_from_param(phi)
        if (not np.isfinite(T_Q)) or T_Q <= 0.0:
            raise RuntimeError("Energy-conserving solver requires T_Q > 0")

        stats["global_state_builds"] += 1

        P_N = float(PNM_n(nB_N, T, param=param, NM_type=NM_type))
        e_N = float(edensNM_n(nB_N, T, param=param))
        h_N = float(P_N + e_N)
        h_over_nB_N = float(h_N / nB_N)
        u_N = float(jB / nB_N)
        r_gamma_N = _relativistic_gamma_from_u(u_N)
        E_N = float(h_N * u_N * r_gamma_N)
        E = E_N
        Pi = float(h_N * u_N * u_N + P_N)

        muB_Q_guess = muB_Q_last_guess["value"]
        if muB_Q_guess is None and isinstance(profile_guess, dict):
            muB_Q_guess = profile_guess.get("muB_Q", None)
        muB_Q = _solve_muB_Q_at_muK0_for_given_Pi_ms(
            Pi,
            jB,
            B_one_forth,
            T_Q,
            ms=ms,
            upB=upB,
            stats=stats,
            initial_guess=muB_Q_guess,
        )
        muB_Q_last_guess["value"] = float(muB_Q)
        thermo_Q = _quark_thermo_state(muB_Q, 0.0, B_one_forth, T_Q, jB, ms=ms, upB=upB)
        nB_Q = float(thermo_Q["nB"])
        if abs(ms) <= 1.0e-12:
            nK_Q = 0.0
        else:
            nK_Q = float(thermo_Q["nK"])

        a_N = float((nB_N - nK_Q) / nB_Q)
        r_gamma_Q = _relativistic_gamma_from_u(thermo_Q["u"])
        E_Q = float(thermo_Q["h"] * thermo_Q["u"] * r_gamma_Q)

        muK_Qstar_seed = _branch_muK_seed(aQstar if np.isfinite(aQstar) else a_N)
        qstar_initial_guess = (muB_Q, muK_Qstar_seed, TQstar_target if fixed_TQstar_mode else T_Q)
        if qstar_last_guess["value"] is not None:
            qstar_initial_guess = qstar_last_guess["value"]
        elif isinstance(profile_guess, dict):
            muB_Qstar_guess = profile_guess.get("muB_Qstar", None)
            muK_Qstar_guess = profile_guess.get("muK_Qstar", None)
            if muB_Qstar_guess is not None and muK_Qstar_guess is not None:
                try:
                    muB_Qstar_guess = float(muB_Qstar_guess)
                    muK_Qstar_guess = float(muK_Qstar_guess)
                    T_Qstar_guess = float(profile_guess.get("T_Qstar", T_Q))
                except Exception:
                    muB_Qstar_guess = None
                    muK_Qstar_guess = None
                    T_Qstar_guess = T_Q
                if (
                    muB_Qstar_guess is not None
                    and muK_Qstar_guess is not None
                    and np.isfinite(muB_Qstar_guess)
                    and np.isfinite(muK_Qstar_guess)
                    and muB_Qstar_guess > 0.0
                    and muK_Qstar_guess >= 0.0
                ):
                    if (not np.isfinite(T_Qstar_guess)) or T_Qstar_guess <= 0.0:
                        T_Qstar_guess = T_Q
                    qstar_initial_guess = (muB_Qstar_guess, muK_Qstar_guess, T_Qstar_guess)

        if fixed_TQstar_mode:
            thermo_Qstar = _solve_interface_Qstar_from_TQstar_E_and_Pi(
                TQstar_target,
                E,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                ms=ms,
                upB=upB,
                initial_guess=qstar_initial_guess,
                stats=stats,
                stats_key="qstar_root_calls",
            )
            aQstar_state = float(thermo_Qstar["aQstar"])
        else:
            try:
                thermo_Qstar = _solve_quark_E_state_once_from_guess(
                    aQstar,
                    E,
                    Pi,
                    jB,
                    nB_Q,
                    nK_Q,
                    B_one_forth,
                    ms=ms,
                    upB=upB,
                    initial_guess=qstar_initial_guess,
                    stats=stats,
                    stats_key="qstar_root_calls",
                )
            except Exception:
                thermo_Qstar = _solve_local_quark_state_from_a_E_and_Pi(
                    aQstar,
                    E,
                    Pi,
                    jB,
                    nB_Q,
                    nK_Q,
                    B_one_forth,
                    ms=ms,
                    upB=upB,
                    initial_guess=qstar_initial_guess,
                    T_ref=T_Q,
                    stats=stats,
                )
            aQstar_state = float(aQstar)
        muB_Qstar = float(thermo_Qstar["muB"])
        muK_Qstar = float(thermo_Qstar["muK"])
        qstar_last_guess["value"] = (muB_Qstar, muK_Qstar, float(thermo_Qstar["T"]))
        r_gamma_Qstar = _relativistic_gamma_from_u(thermo_Qstar["u"])
        E_Qstar = float(thermo_Qstar["h"] * thermo_Qstar["u"] * r_gamma_Qstar)
        h_over_nB_Qstar = float(thermo_Qstar["h"] / thermo_Qstar["nB"])
        h_over_nB_jump_residual = float(h_over_nB_Qstar - h_over_nB_N)
        E_jump_residual = float(E_Qstar - E_N)

        micro_Q = _microphysics_from_quark_state(muB_Q, T_Q)
        D_Q = float(micro_Q["D"])
        eta_Q = float(micro_Q["eta"])
        gamma_Q = float(micro_Q["gamma"])
        tau_Q = float(micro_Q["tau"])

        u_Q = float(thermo_Q["u"])
        disc = float(u_Q * u_Q + 4.0 * D_Q * gamma_Q * eta_Q)
        if (not np.isfinite(disc)) or disc <= 0.0:
            raise RuntimeError("Energy-conserving solver tail discriminant is non-positive")
        lam = float((-u_Q + np.sqrt(disc)) / (2.0 * D_Q))
        if (not np.isfinite(lam)) or lam <= 0.0:
            raise RuntimeError("Energy-conserving solver tail decay lambda must be positive")
        q0 = float(-a_N * u_N)
        x_end = float(-np.log1p(-s_coord_end) / lam)
        tail_coeff_Q = float(D_Q * lam + u_Q)

        state = {
            "jB": float(jB),
            "E": float(E),
            "T_Q": float(T_Q),
            "P_N": P_N,
            "e_N": e_N,
            "h_N": h_N,
            "h_over_nB_N": h_over_nB_N,
            "u_N": u_N,
            "r_gamma_N": r_gamma_N,
            "E_N": E_N,
            "Pi": Pi,
            "muB_Q": float(muB_Q),
            "nB_Q": float(nB_Q),
            "nK_Q": float(nK_Q),
            "a_N": float(a_N),
            "aQstar": float(aQstar_state),
            "muB_Qstar": muB_Qstar,
            "muK_Qstar": muK_Qstar,
            "nB_Qstar": float(thermo_Qstar["nB"]),
            "u_Qstar": float(thermo_Qstar["u"]),
            "T_Qstar": float(thermo_Qstar["T"]),
            "h_Q": float(thermo_Q["h"]),
            "h_Qstar": float(thermo_Qstar["h"]),
            "r_gamma_Q": r_gamma_Q,
            "r_gamma_Qstar": r_gamma_Qstar,
            "E_Q": E_Q,
            "E_Qstar": E_Qstar,
            "E_jump_residual": E_jump_residual,
            "h_over_nB_Qstar": h_over_nB_Qstar,
            "h_over_nB_jump_residual": h_over_nB_jump_residual,
            "D_Q": D_Q,
            "eta_Q": eta_Q,
            "gamma_Q": gamma_Q,
            "tau_Q": tau_Q,
            "lambda": float(lam),
            "u_Q": float(u_Q),
            "q0": float(q0),
            "tail_coeff_Q": float(tail_coeff_Q),
            "x_end": float(x_end),
        }
        state_cache[key] = state
        return state

    def _state_or_none(theta, phi):
        try:
            return _build_global_state(theta, phi)
        except Exception as exc:
            stats["global_state_failures"] += 1
            last_failure["message"] = str(exc)
            return None

    def _ode(s_coord, y, p):
        stats["bvp_ode_calls"] += 1
        state = _state_or_none(float(p[0]), float(p[1]))
        if state is None:
            return np.zeros_like(y) + 1.0e12

        dyds = np.empty_like(y)
        guess = (state["muB_Qstar"], state["muK_Qstar"], state["T_Qstar"])
        for i in range(y.shape[1]):
            a_val = float(y[0, i])
            q_val = float(y[1, i])
            if (not np.isfinite(a_val)) or (not np.isfinite(q_val)):
                stats["local_state_failures"] += 1
                dyds[:, i] = 1.0e12
                continue
            try:
                thermo_loc = _solve_local_quark_state_from_a_E_and_Pi(
                    a_val,
                    state["E"],
                    state["Pi"],
                    state["jB"],
                    state["nB_Q"],
                    state["nK_Q"],
                    B_one_forth,
                    ms=ms,
                    upB=upB,
                    initial_guess=guess,
                    T_ref=state["T_Q"],
                    stats=stats,
                )
                guess = (thermo_loc["muB"], thermo_loc["muK"], thermo_loc["T"])
                micro_loc = _microphysics_from_quark_state(thermo_loc["muB"], thermo_loc["T"])
                reaction = float(micro_loc["gamma"] * (a_val**3 + micro_loc["eta"] * a_val))
                one_minus_s = max(1.0 - float(s_coord[i]), np.finfo(float).tiny)
                dx_ds = 1.0 / (state["lambda"] * one_minus_s)
                da_dx = float((q_val + thermo_loc["u"] * a_val) / micro_loc["D"])
                dyds[0, i] = da_dx * dx_ds
                dyds[1, i] = reaction * dx_ds
            except Exception as exc:
                stats["local_state_failures"] += 1
                last_failure["message"] = str(exc)
                dyds[:, i] = 1.0e12
        return dyds

    def _bc(ya, yb, p):
        stats["bvp_bc_calls"] += 1
        state = _state_or_none(float(p[0]), float(p[1]))
        if state is None:
            return np.array([1.0e12, 1.0e12, 1.0e12, 1.0e12], dtype=float)
        return np.array(
            [
                ya[0] - state["aQstar"],
                ya[1] - state["q0"],
                yb[1] + state["tail_coeff_Q"] * yb[0],
                state["E_Q"] - state["E"],
            ],
            dtype=float,
        )

    theta_guess = _param_from_jB(jB_guess)
    phi_guess = _param_from_TQ(TQ_guess)
    state0 = _state_or_none(theta_guess, phi_guess)
    if state0 is None:
        return _nan_result(
            f"Initial energy-conserving state construction failed: {last_failure['message']}",
            bvp_status=-999,
            bvp_message="initial_state_failure",
        )

    s_coord_mesh = np.linspace(0.0, s_coord_end, int(n_mesh))
    blend = s_coord_mesh / max(s_coord_end, np.finfo(float).tiny)
    tail_shape = np.maximum(1.0 - s_coord_mesh, tail_eps)

    def _default_initial_guess():
        a_guess_loc = float(state0["aQstar"]) * tail_shape
        q_tail_guess_loc = -state0["tail_coeff_Q"] * a_guess_loc
        q_guess_loc = (1.0 - blend) * state0["q0"] + blend * q_tail_guess_loc
        return a_guess_loc, q_guess_loc

    def _profile_initial_guess(profile):
        try:
            s_prev = np.asarray(profile["s_coord"], dtype=float)
            a_prev = np.asarray(profile["a"], dtype=float)
            q_prev = np.asarray(profile["q"], dtype=float)
        except Exception:
            return _default_initial_guess()

        if (
            s_prev.ndim != 1
            or a_prev.shape != s_prev.shape
            or q_prev.shape != s_prev.shape
            or s_prev.size < 5
            or not np.all(np.isfinite(s_prev))
            or not np.all(np.isfinite(a_prev))
            or not np.all(np.isfinite(q_prev))
        ):
            return _default_initial_guess()

        a_prev0 = float(a_prev[0])
        a_scale = float(state0["aQstar"]) / a_prev0 if abs(a_prev0) > 1.0e-12 else 1.0
        a_guess_loc = np.interp(s_coord_mesh, s_prev, a_prev) * a_scale
        a_guess_loc[0] = float(state0["aQstar"])

        q_base = np.interp(s_coord_mesh, s_prev, q_prev) * a_scale
        q_base0 = float(q_base[0])
        q_base_end = float(q_base[-1])
        q_target0 = float(state0["q0"])
        q_target_end = float(-state0["tail_coeff_Q"] * a_guess_loc[-1])
        q_span = q_base_end - q_base0
        if np.isfinite(q_span) and abs(q_span) > 1.0e-12 * max(1.0, abs(q_base0), abs(q_base_end)):
            q_frac = (q_base - q_base0) / q_span
            q_guess_loc = q_target0 + q_frac * (q_target_end - q_target0)
        else:
            q_guess_loc = (1.0 - blend) * q_target0 + blend * q_target_end
        q_guess_loc[0] = q_target0

        if (not np.all(np.isfinite(a_guess_loc))) or (not np.all(np.isfinite(q_guess_loc))):
            return _default_initial_guess()

        return a_guess_loc, q_guess_loc

    if profile_guess is None:
        a_guess, q_guess = _default_initial_guess()
    else:
        a_guess, q_guess = _profile_initial_guess(profile_guess)
    y_guess = np.vstack((a_guess, q_guess))

    _diag(
        f"starting energy compact BVP with jB_guess={jB_guess:.6g}, "
        f"TQ_guess={TQ_guess:.6g}, aQstar={state0['aQstar']:.6g}, tail_eps={tail_eps:.3g}, branch=muK-rich"
    )

    try:
        sol = solve_bvp(
            _ode,
            _bc,
            s_coord_mesh,
            y_guess,
            p=np.array([theta_guess, phi_guess], dtype=float),
            tol=tol_bvp,
            max_nodes=max_nodes,
            verbose=2 if full_diag else 0,
        )
    except Exception as exc:
        return _nan_result(
            f"solve_bvp raised: {exc}; last failure: {last_failure['message']}",
            bvp_status=-999,
            bvp_message=f"solve_bvp raised: {exc}",
        )

    theta_sol = float(sol.p[0])
    phi_sol = float(sol.p[1])
    state = _state_or_none(theta_sol, phi_sol)
    if state is None:
        return _nan_result(
            f"BVP final state construction failed: {last_failure['message']}",
            bvp_status=int(sol.status),
            bvp_message=str(sol.message),
            sol=sol,
        )

    a_end = float(sol.y[0, -1])
    q_end = float(sol.y[1, -1])
    tail_drive = float(state["tail_coeff_Q"] * a_end)
    tail_residual = float(q_end + tail_drive)
    tail_scale = float(max(abs(q_end), abs(tail_drive), np.finfo(float).tiny))
    tail_residual_norm = float(tail_residual / tail_scale)
    E_right_residual = float(state["E_Q"] - state["E"])
    E_scale = float(max(abs(state["E"]), abs(state["E_Q"]), np.finfo(float).tiny))
    E_right_residual_norm = float(E_right_residual / E_scale)

    success = bool(
        sol.success
        and np.isfinite(tail_residual_norm)
        and np.isfinite(E_right_residual_norm)
        and abs(tail_residual_norm) <= max(tol_bvp, 10.0 * np.finfo(float).eps)
        and abs(E_right_residual_norm) <= max(tol_bvp, 10.0 * np.finfo(float).eps)
    )
    result = {
        "success": success,
        "message": "Energy-conserving compact BVP steady-front solve converged" if success else f"{sol.message}; last failure: {last_failure['message']}",
        "jB": float(state["jB"]),
        "E": float(state["E"]),
        "T_Q": float(state["T_Q"]),
        "T_N": float(T),
        "T_Qstar": float(state["T_Qstar"]),
        "aQstar": float(state["aQstar"]),
        "branch_label": "muK-rich",
        "energy_flux_equation": "E_rgamma_const",
        "coordinate": "BVP: s_coord in [0, 1-tail_eps], s_coord=1-exp(-lambda*x)",
        "tail_eps": float(tail_eps),
        "u_N": float(state["u_N"]),
        "u_Q": float(state["u_Q"]),
        "r_gamma_N": float(state["r_gamma_N"]),
        "r_gamma_Q": float(state["r_gamma_Q"]),
        "r_gamma_Qstar": float(state["r_gamma_Qstar"]),
        "a_N": float(state["a_N"]),
        "Pi": float(state["Pi"]),
        "muB_Q": float(state["muB_Q"]),
        "nB_Q": float(state["nB_Q"]),
        "nK_Q": float(state["nK_Q"]),
        "muB_Qstar": float(state["muB_Qstar"]),
        "muK_Qstar": float(state["muK_Qstar"]),
        "nB_Qstar": float(state["nB_Qstar"]),
        "h_N": float(state["h_N"]),
        "h_Q": float(state["h_Q"]),
        "h_Qstar": float(state["h_Qstar"]),
        "h_over_nB_N": float(state["h_over_nB_N"]),
        "h_over_nB_Qstar": float(state["h_over_nB_Qstar"]),
        "h_over_nB_jump_residual": float(state["h_over_nB_jump_residual"]),
        "E_N": float(state["E_N"]),
        "E_Q": float(state["E_Q"]),
        "E_Qstar": float(state["E_Qstar"]),
        "E_jump_residual": float(state["E_jump_residual"]),
        "D_Q": float(state["D_Q"]),
        "eta_Q": float(state["eta_Q"]),
        "gamma_Q": float(state["gamma_Q"]),
        "tau_Q": float(state["tau_Q"]),
        "lambda": float(state["lambda"]),
        "kappa": float(1.0 / state["lambda"]),
        "s_end": float(s_coord_end),
        "x_end": float(state["x_end"]),
        "a_end": a_end,
        "q_end": q_end,
        "tail_residual": tail_residual,
        "tail_residual_norm": tail_residual_norm,
        "tail_scale": tail_scale,
        "E_right_residual": E_right_residual,
        "E_right_residual_norm": E_right_residual_norm,
        "E_scale": E_scale,
        "_residual": np.array([tail_residual_norm, E_right_residual_norm], dtype=float),
        "_root_method": "solve_bvp_parameter_2d",
        "bvp_status": int(sol.status),
        "bvp_message": str(sol.message),
        "bvp_niter": int(getattr(sol, "niter", -1)),
        "bvp_nodes": int(sol.x.size),
        **{k: int(v) for k, v in stats.items()},
    }

    if return_profile or seed_profile:
        s_coord_prof = np.asarray(sol.x, dtype=float)
        a_prof = np.asarray(sol.y[0], dtype=float)
        q_prof = np.asarray(sol.y[1], dtype=float)
        x_prof = -np.log1p(-s_coord_prof) / float(state["lambda"])
        result.update(
            {
                "s_coord": s_coord_prof,
                "x": x_prof,
                "a": a_prof,
                "q": q_prof,
            }
        )
    if return_profile and success:
        muB_prof = np.empty_like(s_coord_prof)
        muK_prof = np.empty_like(s_coord_prof)
        T_prof = np.empty_like(s_coord_prof)
        nB_prof = np.empty_like(s_coord_prof)
        u_prof = np.empty_like(s_coord_prof)
        h_prof = np.empty_like(s_coord_prof)
        r_gamma_prof = np.empty_like(s_coord_prof)
        D_prof = np.empty_like(s_coord_prof)
        eta_prof = np.empty_like(s_coord_prof)
        gamma_prof = np.empty_like(s_coord_prof)
        guess = (state["muB_Qstar"], state["muK_Qstar"], state["T_Qstar"])
        try:
            for i, a_val in enumerate(a_prof):
                stats["profile_state_calls"] += 1
                thermo_loc = _solve_local_quark_state_from_a_E_and_Pi(
                    float(a_val),
                    state["E"],
                    state["Pi"],
                    state["jB"],
                    state["nB_Q"],
                    state["nK_Q"],
                    B_one_forth,
                    ms=ms,
                    upB=upB,
                    initial_guess=guess,
                    T_ref=state["T_Q"],
                    stats=stats,
                )
                guess = (thermo_loc["muB"], thermo_loc["muK"], thermo_loc["T"])
                micro_loc = _microphysics_from_quark_state(thermo_loc["muB"], thermo_loc["T"])
                muB_prof[i] = thermo_loc["muB"]
                muK_prof[i] = thermo_loc["muK"]
                T_prof[i] = thermo_loc["T"]
                nB_prof[i] = thermo_loc["nB"]
                u_prof[i] = thermo_loc["u"]
                h_prof[i] = thermo_loc["h"]
                r_gamma_prof[i] = _relativistic_gamma_from_u(thermo_loc["u"])
                D_prof[i] = micro_loc["D"]
                eta_prof[i] = micro_loc["eta"]
                gamma_prof[i] = micro_loc["gamma"]
        except Exception as exc:
            result["success"] = False
            result["message"] = f"{result['message']}; profile reconstruction failed: {exc}"
            result["profile_reconstruction_failed"] = True
            result["profile_reconstruction_message"] = str(exc)
            result["profile_state_calls"] = int(stats["profile_state_calls"])
            result["local_state_calls"] = int(stats["local_state_calls"])
            result["local_root_calls"] = int(stats["local_root_calls"])
            result["local_fast_failures"] = int(stats["local_fast_failures"])
            if simple_diag:
                print(f"energy-bvp profile reconstruction failed for aQstar={state['aQstar']:.6g}: {exc}")
            return result

        result.update(
            {
                "u": u_prof,
                "nB": nB_prof,
                "muB": muB_prof,
                "muK": muK_prof,
                "T_profile": T_prof,
                "h_profile": h_prof,
                "r_gamma_profile": r_gamma_prof,
                "D_profile": D_prof,
                "eta_profile": eta_prof,
                "gamma_profile": gamma_prof,
                "profile_state_calls": int(stats["profile_state_calls"]),
                "local_state_calls": int(stats["local_state_calls"]),
                "local_root_calls": int(stats["local_root_calls"]),
                "local_fast_failures": int(stats["local_fast_failures"]),
            }
        )

    if simple_diag:
        print(
            f"energy-bvp jB={result['jB']:.6g}, T_Q={result['T_Q']:.6g}, "
            f"aQstar={state['aQstar']:.6g}, tail_norm={tail_residual_norm:.6g}, "
            f"E_right_norm={E_right_residual_norm:.6g}, status={sol.status}, success={success}"
        )
    return result


def solve_front_energy_conserving(
    T,
    nB_N,
    B_one_forth,
    aQstar,
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
):
    """
    Solve the steady-front problem with h(x)*u(x)*sqrt(1+u(x)^2) conserved
    across the front.

    The solver follows the adiabatic compact-BVP structure, but it derives
    E = h*u*sqrt(1+u^2) from the upstream trial jB, just like the momentum
    flux Pi, instead of solving for E as an independent BVP parameter.
    """
    aQstar = float(aQstar)
    abs_target = abs(aQstar)
    sign = -1.0 if aQstar < 0.0 else 1.0

    if isinstance(verb, str):
        verb_mode = "full" if verb.lower() == "full" else ("simple" if verb else "off")
    else:
        verb_mode = "simple" if verb else "off"
    simple_diag = verb_mode in ("simple", "full")

    direct_jB_guess = jB_guess
    if direct_jB_guess is None:
        direct_jB_guess = float(max(1.0e-12, 1.0e-8 * float(nB_N), 0.8 * abs_target))

    if TQ_guess is None:
        TQ_guess_candidates = _adiabatic_default_TQ_guesses(T)
    else:
        TQ_guess_candidates = (float(TQ_guess),)
    direct_TQ_guess = float(TQ_guess_candidates[0])
    if (not np.isfinite(direct_TQ_guess)) or direct_TQ_guess <= 0.0:
        raise RuntimeError("TQ_guess must be positive")

    direct_result = _solve_front_energy_conserving_once(
        T,
        nB_N,
        B_one_forth,
        aQstar,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=n_mesh,
        tol_bvp=tol_bvp,
        max_nodes=max_nodes,
        jB_guess=direct_jB_guess,
        TQ_guess=direct_TQ_guess,
        jB_bounds=jB_bounds,
        return_profile=return_profile,
        verb=verb,
        profile_guess=None,
    )
    if (
        TQ_guess is None
        and len(TQ_guess_candidates) > 1
        and _adiabatic_initial_state_failed(direct_result)
    ):
        retry_TQ_guess = float(TQ_guess_candidates[1])
        if simple_diag:
            print(
                f"energy-bvp low-T hot-start retry: initial state failed at "
                f"TQ_guess={direct_TQ_guess:.6g}; retrying with TQ_guess={retry_TQ_guess:.6g}"
            )
        direct_TQ_guess = retry_TQ_guess
        direct_result = _solve_front_energy_conserving_once(
            T,
            nB_N,
            B_one_forth,
            aQstar,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=n_mesh,
            tol_bvp=tol_bvp,
            max_nodes=max_nodes,
            jB_guess=direct_jB_guess,
            TQ_guess=direct_TQ_guess,
            jB_bounds=jB_bounds,
            return_profile=return_profile,
            verb=verb,
            profile_guess=None,
        )
    if bool(direct_result.get("success")) or abs_target <= 1.0e-2:
        result_out = dict(direct_result)
        result_out["continuation_used"] = False
        result_out["continuation_steps"] = 0
        if return_profile:
            return result_out
        return _strip_energy_profile_fields(result_out)

    if simple_diag:
        print(
            f"energy-bvp continuation: direct solve failed at aQstar={aQstar:.6g}; "
            f"retrying with staged continuation"
        )

    stage_n_mesh = int(min(max(max(int(n_mesh) // 2, 25), 25), 40))
    stage_tol = float(max(float(tol_bvp), 2.0e-2))
    stage_max_nodes = int(max(stage_n_mesh + 5, min(int(max_nodes), 250)))
    base_step_abs = 1.0e-3
    min_step_abs = 1.0e-4
    step_abs = float(min(base_step_abs, max(abs_target - min(abs_target, 1.0e-2), 0.0)))
    current_abs = min(abs_target, 1.0e-2)
    stage_jB_guess = direct_jB_guess
    if not (np.isfinite(stage_jB_guess) and stage_jB_guess > 0.0):
        stage_jB_guess = None
    direct_jB_out = direct_result.get("jB", np.nan)
    if np.isfinite(direct_jB_out) and direct_jB_out > 0.0:
        stage_jB_guess = float(direct_jB_out)
    if stage_jB_guess is None:
        seed_abs = current_abs if current_abs > 0.0 else 1.0e-2
        stage_jB_guess = _seed_adiabatic_jB_guess(
            T,
            nB_N,
            B_one_forth,
            sign * seed_abs,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=n_mesh,
            tol_bvp=tol_bvp,
            max_nodes=max_nodes,
            verb=False,
        )

    current = _solve_front_energy_conserving_once(
        T,
        nB_N,
        B_one_forth,
        sign * current_abs,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=stage_n_mesh,
        tol_bvp=stage_tol,
        max_nodes=stage_max_nodes,
        jB_guess=stage_jB_guess,
        TQ_guess=direct_TQ_guess,
        jB_bounds=jB_bounds,
        return_profile=False,
        verb=False,
        profile_guess=None,
        seed_profile=True,
    )
    if not bool(current.get("success")):
        failed = dict(current)
        failed["message"] = (
            f"{direct_result['message']}; continuation seed at aQstar={sign * current_abs:.6g} failed: "
            f"{current['message']}"
        )
        failed["continuation_used"] = True
        failed["continuation_steps"] = 0
        failed["continuation_seed_aQstar"] = float(sign * current_abs)
        if return_profile:
            return failed
        return _strip_energy_profile_fields(failed)

    steps_taken = 0
    while current_abs < abs_target - 1.0e-12:
        next_abs = min(abs_target, current_abs + step_abs)
        next_a = sign * next_abs
        trial = _solve_front_energy_conserving_once(
            T,
            nB_N,
            B_one_forth,
            next_a,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=stage_n_mesh,
            tol_bvp=stage_tol,
            max_nodes=stage_max_nodes,
            jB_guess=current["jB"],
            TQ_guess=current["T_Q"],
            jB_bounds=jB_bounds,
            return_profile=False,
            verb=False,
            profile_guess=current,
            seed_profile=True,
        )
        if bool(trial.get("success")):
            current = trial
            current_abs = next_abs
            steps_taken += 1
            if simple_diag and (steps_taken % 25 == 0 or current_abs >= abs_target - 1.0e-12):
                print(
                    f"energy-bvp continuation: reached aQstar={next_a:.6g} "
                    f"with jB={current['jB']:.6g}, T_Q={current['T_Q']:.6g}"
                )
            step_abs = min(base_step_abs, max(abs_target - current_abs, 0.0))
            continue

        step_abs *= 0.5
        if step_abs < min_step_abs:
            failed = dict(trial)
            failed["message"] = (
                f"Energy-conserving continuation failed after reaching aQstar={sign * current_abs:.6g}; "
                f"last attempted aQstar={next_a:.6g}. {trial['message']}"
            )
            failed["continuation_used"] = True
            failed["continuation_steps"] = steps_taken
            failed["continuation_seed_aQstar"] = float(sign * min(abs_target, 1.0e-2))
            if return_profile:
                return failed
            return _strip_energy_profile_fields(failed)

    refined = _solve_front_energy_conserving_once(
        T,
        nB_N,
        B_one_forth,
        aQstar,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=n_mesh,
        tol_bvp=tol_bvp,
        max_nodes=max_nodes,
        jB_guess=current["jB"],
        TQ_guess=current["T_Q"],
        jB_bounds=jB_bounds,
        return_profile=return_profile,
        verb=False,
        profile_guess=current,
    )
    if bool(refined.get("success")):
        result_out = dict(refined)
        result_out["continuation_refined"] = True
    else:
        if return_profile:
            coarse_profile = _solve_front_energy_conserving_once(
                T,
                nB_N,
                B_one_forth,
                aQstar,
                ms=ms,
                param=param,
                NM_type=NM_type,
                tail_eps=tail_eps,
                n_mesh=stage_n_mesh,
                tol_bvp=stage_tol,
                max_nodes=stage_max_nodes,
                jB_guess=current["jB"],
                TQ_guess=current["T_Q"],
                jB_bounds=jB_bounds,
                return_profile=True,
                verb=False,
                profile_guess=current,
            )
            result_out = dict(coarse_profile if bool(coarse_profile.get("success")) else current)
        else:
            result_out = dict(current)
        result_out["message"] = (
            f"{result_out['message']}; final refinement failed: {refined['message']}"
        )
        result_out["continuation_refined"] = False

    result_out["continuation_used"] = True
    result_out["continuation_steps"] = steps_taken
    result_out["continuation_seed_aQstar"] = float(sign * min(abs_target, 1.0e-2))
    if return_profile:
        return result_out
    return _strip_energy_profile_fields(result_out)


def _annotate_energy_uNmax_result(
    result,
    TQstar_target,
    continuation_used=False,
    continuation_steps=0,
    last_success=None,
):
    out = dict(result)
    TQstar_target = float(TQstar_target)
    T_Qstar = float(out.get("T_Qstar", np.nan))
    aQstar_out = float(out.get("aQstar", np.nan))
    u_N_out = float(out.get("u_N", np.nan))
    out["solver_variant"] = "uNmax_fixed_TQstar"
    out["T_Qstar_target"] = TQstar_target
    out["T_Qstar_residual"] = (
        float(T_Qstar - TQstar_target)
        if np.isfinite(T_Qstar) and np.isfinite(TQstar_target)
        else np.nan
    )
    out["aQstar_derived"] = aQstar_out
    out["u_N_max_candidate"] = u_N_out
    out["TQstar_continuation_used"] = bool(continuation_used)
    out["TQstar_continuation_steps"] = int(continuation_steps)
    if last_success is not None:
        out["last_success_T_Qstar"] = float(last_success.get("T_Qstar", np.nan))
        out["last_success_aQstar"] = float(last_success.get("aQstar", np.nan))
        out["last_success_u_N"] = float(last_success.get("u_N", np.nan))
    return out


def solve_front_energy_conserving_uNmax(
    T,
    nB_N,
    B_one_forth,
    TQstar=0.01,
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
    TQstar_start=5.0,
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
    if (not np.isfinite(TQstar)) or TQstar <= 0.0:
        raise RuntimeError("TQstar must be positive")

    if isinstance(verb, str):
        verb_mode = "full" if verb.lower() == "full" else ("simple" if verb else "off")
    else:
        verb_mode = "simple" if verb else "off"
    simple_diag = verb_mode in ("simple", "full")

    direct_jB_guess = jB_guess
    if direct_jB_guess is None:
        direct_jB_guess = float(max(1.0e-12, 1.0e-8 * float(nB_N), 1.0))
    continuation_jB_guess = float(direct_jB_guess)

    if TQ_guess is None:
        TQ_guess_candidates = _adiabatic_default_TQ_guesses(T)
    else:
        TQ_guess_candidates = (float(TQ_guess),)
    direct_TQ_guess = float(TQ_guess_candidates[0])
    if (not np.isfinite(direct_TQ_guess)) or direct_TQ_guess <= 0.0:
        raise RuntimeError("TQ_guess must be positive")

    direct_result = _solve_front_energy_conserving_once(
        T,
        nB_N,
        B_one_forth,
        None,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=n_mesh,
        tol_bvp=tol_bvp,
        max_nodes=max_nodes,
        jB_guess=direct_jB_guess,
        TQ_guess=direct_TQ_guess,
        jB_bounds=jB_bounds,
        return_profile=return_profile,
        verb=verb,
        profile_guess=None,
        TQstar_target=TQstar,
    )
    if (
        TQ_guess is None
        and len(TQ_guess_candidates) > 1
        and _adiabatic_initial_state_failed(direct_result)
    ):
        retry_TQ_guess = float(TQ_guess_candidates[1])
        if simple_diag:
            print(
                f"energy-uNmax low-T hot-start retry: initial state failed at "
                f"TQ_guess={direct_TQ_guess:.6g}; retrying with TQ_guess={retry_TQ_guess:.6g}"
            )
        direct_TQ_guess = retry_TQ_guess
        direct_result = _solve_front_energy_conserving_once(
            T,
            nB_N,
            B_one_forth,
            None,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=n_mesh,
            tol_bvp=tol_bvp,
            max_nodes=max_nodes,
            jB_guess=direct_jB_guess,
            TQ_guess=direct_TQ_guess,
            jB_bounds=jB_bounds,
            return_profile=return_profile,
            verb=verb,
            profile_guess=None,
            TQstar_target=TQstar,
        )

    if not bool(direct_result.get("success")):
        retry_jB_guesses = [
            0.75 * float(direct_jB_guess),
            0.5 * float(direct_jB_guess),
            1.0,
            0.6,
        ]
        if jB_bounds is not None:
            jB_min = float(jB_bounds[0])
            jB_max = float(jB_bounds[1])
            retry_jB_guesses = [
                min(max(float(g), jB_min * (1.0 + 1.0e-6)), jB_max * (1.0 - 1.0e-6))
                for g in retry_jB_guesses
            ]
        seen_jB_guesses = {round(float(direct_jB_guess), 12)}
        for retry_jB_guess in retry_jB_guesses:
            retry_jB_guess = float(retry_jB_guess)
            if (
                (not np.isfinite(retry_jB_guess))
                or retry_jB_guess <= 0.0
                or round(retry_jB_guess, 12) in seen_jB_guesses
            ):
                continue
            if round(retry_jB_guess, 12) == round(1.0, 12):
                continuation_jB_guess = retry_jB_guess
            seen_jB_guesses.add(round(retry_jB_guess, 12))
            if simple_diag:
                print(
                    f"energy-uNmax seed retry: direct solve failed at "
                    f"jB_guess={direct_jB_guess:.6g}; retrying jB_guess={retry_jB_guess:.6g}"
                )
            trial = _solve_front_energy_conserving_once(
                T,
                nB_N,
                B_one_forth,
                None,
                ms=ms,
                param=param,
                NM_type=NM_type,
                tail_eps=tail_eps,
                n_mesh=n_mesh,
                tol_bvp=tol_bvp,
                max_nodes=max_nodes,
                jB_guess=retry_jB_guess,
                TQ_guess=direct_TQ_guess,
                jB_bounds=jB_bounds,
                return_profile=return_profile,
                verb=False,
                profile_guess=None,
                TQstar_target=TQstar,
            )
            if bool(trial.get("success")):
                direct_result = trial
                break

    if bool(direct_result.get("success")):
        result_out = _annotate_energy_uNmax_result(
            direct_result,
            TQstar,
            continuation_used=False,
            continuation_steps=0,
        )
        if return_profile:
            return result_out
        return _strip_energy_profile_fields(result_out)

    TQstar_start = float(TQstar_start)
    if (not np.isfinite(TQstar_start)) or TQstar_start <= 0.0 or TQstar_start <= TQstar:
        failed = _annotate_energy_uNmax_result(
            direct_result,
            TQstar,
            continuation_used=False,
            continuation_steps=0,
        )
        if return_profile:
            return failed
        return _strip_energy_profile_fields(failed)

    if simple_diag:
        print(
            f"energy-uNmax continuation: direct solve failed at T_Qstar={TQstar:.6g}; "
            f"starting from T_Qstar={TQstar_start:.6g}"
        )

    stage_n_mesh = int(min(max(max(int(n_mesh) // 2, 25), 25), 40))
    stage_tol = float(max(float(tol_bvp), 2.0e-2))
    stage_max_nodes = int(max(stage_n_mesh + 5, min(int(max_nodes), 250)))
    stage_jB_guess = continuation_jB_guess
    direct_jB_out = direct_result.get("jB", np.nan)
    if np.isfinite(direct_jB_out) and direct_jB_out > 0.0:
        stage_jB_guess = float(direct_jB_out)

    current = _solve_front_energy_conserving_once(
        T,
        nB_N,
        B_one_forth,
        None,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=stage_n_mesh,
        tol_bvp=stage_tol,
        max_nodes=stage_max_nodes,
        jB_guess=stage_jB_guess,
        TQ_guess=direct_TQ_guess,
        jB_bounds=jB_bounds,
        return_profile=False,
        verb=False,
        profile_guess=None,
        seed_profile=True,
        TQstar_target=TQstar_start,
    )
    if not bool(current.get("success")):
        failed = _annotate_energy_uNmax_result(
            current,
            TQstar,
            continuation_used=True,
            continuation_steps=0,
        )
        failed["message"] = (
            f"{direct_result['message']}; T_Qstar continuation seed at "
            f"{TQstar_start:.6g} failed: {current['message']}"
        )
        if return_profile:
            return failed
        return _strip_energy_profile_fields(failed)

    steps_taken = 0
    targets = np.linspace(TQstar_start, TQstar, max(2, int(np.ceil((TQstar_start - TQstar) / 0.5)) + 1))[1:]
    last_success = current
    for target in targets:
        target = float(target)
        trial = _solve_front_energy_conserving_once(
            T,
            nB_N,
            B_one_forth,
            None,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=stage_n_mesh,
            tol_bvp=stage_tol,
            max_nodes=stage_max_nodes,
            jB_guess=current["jB"],
            TQ_guess=current["T_Q"],
            jB_bounds=jB_bounds,
            return_profile=False,
            verb=False,
            profile_guess=current,
            seed_profile=True,
            TQstar_target=target,
        )
        if not bool(trial.get("success")):
            failed = _annotate_energy_uNmax_result(
                trial,
                TQstar,
                continuation_used=True,
                continuation_steps=steps_taken,
                last_success=last_success,
            )
            failed["message"] = (
                f"Energy-uNmax T_Qstar continuation failed after reaching "
                f"T_Qstar={last_success.get('T_Qstar', np.nan):.6g}; "
                f"last attempted T_Qstar={target:.6g}. {trial['message']}"
            )
            if return_profile:
                return failed
            return _strip_energy_profile_fields(failed)
        current = trial
        last_success = current
        steps_taken += 1
        if simple_diag:
            print(
                f"energy-uNmax continuation: reached T_Qstar={target:.6g} "
                f"with aQstar={current['aQstar']:.6g}, u_N={current['u_N']:.6g}"
            )

    refined = _solve_front_energy_conserving_once(
        T,
        nB_N,
        B_one_forth,
        None,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=n_mesh,
        tol_bvp=tol_bvp,
        max_nodes=max_nodes,
        jB_guess=current["jB"],
        TQ_guess=current["T_Q"],
        jB_bounds=jB_bounds,
        return_profile=return_profile,
        verb=False,
        profile_guess=current,
        TQstar_target=TQstar,
    )
    if bool(refined.get("success")):
        result_out = _annotate_energy_uNmax_result(
            refined,
            TQstar,
            continuation_used=True,
            continuation_steps=steps_taken,
            last_success=last_success,
        )
        result_out["TQstar_continuation_refined"] = True
    else:
        if return_profile and abs(float(current.get("T_Qstar", np.nan)) - TQstar) <= 1.0e-10:
            coarse_profile = _solve_front_energy_conserving_once(
                T,
                nB_N,
                B_one_forth,
                None,
                ms=ms,
                param=param,
                NM_type=NM_type,
                tail_eps=tail_eps,
                n_mesh=stage_n_mesh,
                tol_bvp=stage_tol,
                max_nodes=stage_max_nodes,
                jB_guess=current["jB"],
                TQ_guess=current["T_Q"],
                jB_bounds=jB_bounds,
                return_profile=True,
                verb=False,
                profile_guess=current,
                TQstar_target=TQstar,
            )
            result_base = coarse_profile if bool(coarse_profile.get("success")) else current
        else:
            result_base = current
        result_out = _annotate_energy_uNmax_result(
            result_base,
            TQstar,
            continuation_used=True,
            continuation_steps=steps_taken,
            last_success=last_success,
        )
        result_out["message"] = (
            f"{result_out['message']}; final refinement failed: {refined['message']}"
        )
        result_out["TQstar_continuation_refined"] = False

    if return_profile:
        return result_out
    return _strip_energy_profile_fields(result_out)
