import numpy as np
import time
import warnings
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import fsolve, root_scalar, root
from scipy.interpolate import CubicSpline
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
from RMFsolver.Solver import (
    RMFsolve,
    RMFsolve_mu,
    RMFsolveSYM,
    RMFpressureSYM,
    RMFpressurePNM,
    pressure_RMF,
    edens_RMF,
    baryon_density_RMF,
    RMFsolvePNM_mu,
)
from RMFsolver.Solver import RMFedensPNM, RMFentropyPNM, RMFbaryon_densityPNM, RMFbaryon_densitySYM, RMFbaryon_density

__all__ = [
    "analytic_velocity_bound",
    "analytic_velocity_isothermal",
    "solve_front_isothermal",
    "solve_front_energy_conserving_nK",
    "solve_front_energy_conserving_uNmax",
    "solve_front_thermal_conducting",
    "z_time_evolution",
]

_TRANSPORT_ALPHA_S = 0.3
_TRANSPORT_G_S = np.sqrt(4.0 * np.pi * _TRANSPORT_ALPHA_S)
_TRANSPORT_QD_COEFF = np.sqrt(3.0 * _TRANSPORT_G_S**2 / (2.0 * np.pi**2))
_TRANSPORT_D_PREFACTOR = 24.0 * _TRANSPORT_ALPHA_S**2 / np.pi
_TRANSPORT_H_CONST = 1.81317
_FLOAT_TINY = np.finfo(float).tiny
_ISOTHERMAL_RETRY_ACTIVE = 0
# Continuation resolution for the conserved-(P, h/nB) conversion-layer sweep.
# That system has multiple roots at low T, so the steps must stay fine enough
# for the continuation to keep its branch; _LAYER_TRAJECTORY_MAX_DMUB guards it.
# I2 converges at first order in this step count: measured against a 640-step
# reference, 160 steps carry about 0.5% error in I2 (0.25% in u_0minus) and 320 steps
# about 0.12% (0.06% in u_0minus).
_LAYER_TRAJECTORY_STEPS = 320
_LAYER_TRAJECTORY_MAX_DMUB = 0.05

# Above this fraction the static-isobar reduction used by the isothermal
# interface construction is no longer trustworthy: (Pi - P)/P = w*u**2/P is the
# leading error of treating the interface as a static isobar.  Measured
# 2026-08-19, it reaches 10% only at 1 - a_0plus ~ 1e-9 to 1e-12, far outside
# the realized range a_0plus_max in (0.01, 0.99), so this bound never binds in
# practice; it exists so the failure is loud if it ever does.
MOMENTUM_FLUX_RATIO_TOLERANCE = 1.0e-3


class SlowFrontNoSolution(RuntimeError):
    """
    No steadily moving conversion front exists for the given upstream state.

    Raised by analytic_velocity_bound when the metastable neutron matter has a
    lower enthalpy per baryon than the coldest equilibrated (mu_K = 0) quark
    matter reachable on the isobar P = P_0minus. Energy-plus-baryon flux conservation
    then forces h_inf/n_inf = h_0minus/n_0minus with no root, so a finite-speed front cannot
    bridge the gap even though the static u_0minus = 0 coexistence still exists. This
    is a physical outcome (typically at small metastability and finite
    temperature), not a numerical failure. Callers can inspect ``status``,
    ``gap``, ``muB_cold``, and ``h_over_nB_0minus``.
    """

    def __init__(self, message, *, gap, muB_cold, h_over_nB_0minus):
        super().__init__(message)
        self.status = "no_slow_front_solution"
        self.gap = float(gap)
        self.muB_cold = float(muB_cold)
        self.h_over_nB_0minus = float(h_over_nB_0minus)


def _slow_front_enthalpy_gap(nuclear_state, B_one_forth):
    """
    Enthalpy-per-baryon gap that blocks a slow conversion front.

    Returns (gap, muB_cold) where muB_cold is the baryon chemical potential of
    cold (T = 0) equilibrated quark matter at pressure P_0minus, and
    gap = muB_cold - h_0minus/n_B,N. The cold state minimizes h_inf/n_B,Q along the
    P = P_0minus isobar (there h_inf/n_B,Q = mu_B), so gap > 0 means no equilibrated
    downstream state satisfies energy-per-baryon continuity and no moving front
    exists.
    """
    P_0minus = float(nuclear_state["P_0minus"])
    h_over_nB_0minus = float(nuclear_state["h_over_nB_0minus"])
    U_bag = float(B_one_forth) ** 4
    radicand = 108.0 * np.pi**2 * (P_0minus + U_bag)
    if (not np.isfinite(radicand)) or radicand <= 0.0:
        return float("nan"), float("nan")
    muB_cold = float(radicand**0.25)
    return float(muB_cold - h_over_nB_0minus), muB_cold


def _analytic_A_from_isobar(
    muB_inf,
    P_inf,
    B_one_forth,
    ms=0.0,
    upB=5000,
    T_cold=1.0e-3,
    return_muK=False,
):
    """
    Return the interface-fraction ceiling A = nK/nB at the T -> 0 end of the
    isobar, optionally together with its exact muK.

    A is the composition reached when the constant-pressure trajectory through
    the downstream state is extrapolated to zero temperature, defined by
    P_QM(muB_inf, muK, T -> 0) = P_inf. It is evaluated directly from the quark EOS
    (nK_QM / nB_QM) instead of a quadratic-expansion closed form: the EOS used
    here enforces charge neutrality (2 n_u = n_d + n_s), whereas the closed
    forms A = (3/2) sqrt(108 pi^2 (P+U) - muB^4)/muB^2 and its leading-order
    cousin A = 9 pi T_inf / (sqrt(2) muB_inf) are both derived from the
    non-neutral parametrization and overestimate A badly as A -> 1.

    Charge neutrality together with n_s >= 0 caps the composition at A = 1,
    attained when mu_s reaches zero. There n_u : n_d : n_s = 1 : 2 : 0, the
    flavor content of neutron matter, and the composition is frozen for any
    larger muK, so A = 1 is returned for pressures at or above that point.
    """
    muB_inf = float(muB_inf)
    P_inf = float(P_inf)
    B_one_forth = float(B_one_forth)
    if (not np.isfinite(muB_inf)) or muB_inf <= 0.0:
        raise RuntimeError("muB_inf must be positive and finite for the A solve")
    if not np.isfinite(P_inf):
        raise RuntimeError("P_inf must be finite for the A solve")
    T_cold = float(T_cold)

    def result(A_boundary, muK_boundary):
        if return_muK:
            return float(A_boundary), float(muK_boundary)
        return float(A_boundary)

    def mu_s_of(muK):
        state = _quark_uds_state(muB_inf, float(muK), T_cold, ms=ms, upB=upB)
        return float(state["chemical_potentials"]["mu_s"])

    # muK at which mu_s vanishes: the saturation point beyond which n_s = 0.
    muK_hi = 3.0 * muB_inf
    if mu_s_of(0.0) <= 0.0:
        return result(1.0, 0.0)
    if mu_s_of(muK_hi) > 0.0:
        raise RuntimeError("could not bracket the mu_s = 0 saturation point")
    muK_sat = float(
        root_scalar(
            mu_s_of, bracket=(0.0, muK_hi), method="brentq", xtol=1.0e-8, rtol=1.0e-12
        ).root
    )

    P_sat = float(PQM(muB_inf, muK_sat, B_one_forth, T_cold, ms=ms, upB=upB))
    if P_inf >= P_sat:
        return result(1.0, muK_sat)
    P_zero = float(PQM(muB_inf, 0.0, B_one_forth, T_cold, ms=ms, upB=upB))
    if P_inf <= P_zero:
        return result(0.0, 0.0)

    muK_A = float(
        root_scalar(
            lambda muK: float(PQM(muB_inf, muK, B_one_forth, T_cold, ms=ms, upB=upB)) - P_inf,
            bracket=(0.0, muK_sat),
            method="brentq",
            xtol=1.0e-8,
            rtol=1.0e-12,
        ).root
    )
    nK = float(nK_QM(muB_inf, muK_A, B_one_forth, T_cold, ms=ms, upB=upB))
    nB = float(nB_QM(muB_inf, muK_A, B_one_forth, T_cold, ms=ms, upB=upB))
    if (not np.isfinite(nB)) or nB <= 0.0:
        raise RuntimeError("nB_QM must be positive and finite for the A solve")
    A_boundary = float(nK / nB)
    if not np.isfinite(A_boundary):
        raise RuntimeError("A_boundary from the exact EOS is non-finite")
    return result(min(max(A_boundary, 0.0), 1.0), muK_A)


def _analytic_fixed_muB_interface_state(
    muB_inf,
    P_inf,
    T_0plus,
    B_one_forth,
    ms=0.0,
    upB=5000,
):
    """Return the exact fixed-muB isobar state at the prescribed T(0+)."""
    muB_inf = float(muB_inf)
    P_inf = float(P_inf)
    T_0plus = float(T_0plus)
    B_one_forth = float(B_one_forth)
    if (not np.isfinite(T_0plus)) or T_0plus < 0.0:
        raise RuntimeError("T_0plus must be finite and non-negative")

    A_boundary, muK_0plus_max = _analytic_A_from_isobar(
        muB_inf,
        P_inf,
        B_one_forth,
        ms=ms,
        upB=upB,
        return_muK=True,
    )
    a_0plus, muK_0plus = _analytic_A_from_isobar(
        muB_inf,
        P_inf,
        B_one_forth,
        ms=ms,
        upB=upB,
        T_cold=T_0plus,
        return_muK=True,
    )
    nB_0plus = float(
        nB_QM(muB_inf, muK_0plus, B_one_forth, T_0plus, ms=ms, upB=upB)
    )
    nK_0plus = float(
        nK_QM(muB_inf, muK_0plus, B_one_forth, T_0plus, ms=ms, upB=upB)
    )
    if (not np.isfinite(nB_0plus)) or nB_0plus <= 0.0:
        raise RuntimeError("fixed-muB interface state has non-physical nB_0plus")
    a_from_definition = float(nK_0plus / nB_0plus)
    if not np.isclose(a_from_definition, a_0plus, rtol=1.0e-10, atol=1.0e-12):
        raise RuntimeError("fixed-muB interface composition is inconsistent with nK/nB")
    if (not np.isfinite(a_from_definition)) or a_from_definition <= 0.0:
        raise RuntimeError(
            "prescribed T_0plus does not produce a positive interface fraction"
        )
    if a_from_definition >= 1.0:
        raise RuntimeError("a_0plus must satisfy 0 < a_0plus < 1")
    if a_from_definition > A_boundary + 1.0e-8:
        raise RuntimeError("exact a_0plus exceeds the cold A_boundary")
    return {
        "T_0plus": T_0plus,
        "muB_0plus": muB_inf,
        "muK_0plus": float(muK_0plus),
        "nB_0plus": nB_0plus,
        "nK_0plus": nK_0plus,
        "a_0plus": a_from_definition,
        "A_boundary": float(A_boundary),
        "muK_0plus_max": float(muK_0plus_max),
    }


def _normalize_velocity_closure(value):
    """
    Normalize the analytic velocity closure selector.

    "closed_form" reproduces the published piecewise-constant result: A is read
    off the isobar at fixed mu_B and I2 is the closed-form bracket of
    Eq. (speed_modified). "numerical_I2" traces the conversion layer along the
    conserved pair (P, h/nB) and integrates I2 numerically along it.
    """
    mode = str(value).strip().lower()
    if mode in ("closed_form", "closed-form", "analytic", "eq30"):
        return "closed_form"
    if mode in ("numerical_i2", "numerical", "numeric_i2", "exact_i2"):
        return "numerical_I2"
    raise ValueError("velocity_closure must be 'closed_form' or 'numerical_I2'")


def _analytic_layer_trajectory(
    P_plus,
    h_over_nB_plus,
    muB_inf,
    T_inf,
    B_one_forth,
    ms=0.0,
    upB=5000,
    n_steps=_LAYER_TRAJECTORY_STEPS,
    T_cold=1.0e-3,
):
    """
    Trace the conversion layer along the conserved pair (P, h/nB).

    The steady-front conservation laws hold two combinations fixed across the
    layer: the momentum flux, which for slow fronts reduces to P, and the
    energy-per-baryon flux ratio E/J = h*gamma/nB -> h/nB. Constancy of mu_B is
    an *additional* assumption; it is implied by the other two only in the
    quadratic equation-of-state expansion, and with the exact charge-neutral
    EOS it fails (mu_B drifts by tens of percent when the layer spans a wide
    temperature range). This routine therefore holds (P, h/nB) fixed and lets
    mu_B float.

    For massless quarks h = 4*(P + U_bag) identically, so fixing P fixes h and
    fixing h/nB then fixes nB: the baryon density really is constant along the
    layer, which is the only constancy the I2 reduction requires.

    The trajectory is followed by continuation in T from the equilibrated end
    (muK = 0 at T_inf, where a = 0) down toward T_cold, each step seeded from
    the previous one. The (P, h/nB) system admits more than one root at low T,
    so a monotonicity guard rejects continuation steps that jump branches.

    Returns a dict of arrays ordered by increasing a: "a", "muB", "muK", "T",
    plus "nB_plus", "A_boundary" and "saturated" (True when the composition
    reaches the n_s = 0 ceiling a = 1 before T reaches T_cold).
    """
    P_plus = float(P_plus)
    h_over_nB_plus = float(h_over_nB_plus)
    T_inf = float(T_inf)
    T_cold = float(T_cold)
    if (not np.isfinite(T_inf)) or T_inf <= T_cold:
        raise RuntimeError("layer trajectory requires T_inf > T_cold")
    if (not np.isfinite(h_over_nB_plus)) or h_over_nB_plus <= 0.0:
        raise RuntimeError("layer trajectory requires a positive h/nB")

    def quark_point(muB, muK, T):
        P = float(PQM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
        e = float(
            edensQM(muB, muK, B_one_forth, T, ms=ms, include_em=False, upB=upB)
        )
        nB = float(nB_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
        nK = float(nK_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
        if (not np.isfinite(nB)) or nB <= 0.0:
            raise RuntimeError("layer trajectory hit a non-physical nB")
        return P, (P + e) / nB, nB, nK / nB

    def residual(vec, T):
        muB, muK = float(vec[0]), float(vec[1])
        if (not np.isfinite(muB)) or muB <= 0.0 or (not np.isfinite(muK)) or muK < 0.0:
            return np.array([1.0e6, 1.0e6], dtype=float)
        try:
            P, h_over_nB, _, _ = quark_point(muB, muK, T)
        except Exception:
            return np.array([1.0e6, 1.0e6], dtype=float)
        return np.array(
            [
                (P - P_plus) / max(abs(P_plus), _FLOAT_TINY),
                (h_over_nB - h_over_nB_plus) / h_over_nB_plus,
            ],
            dtype=float,
        )

    # Sample uniformly in T^(1/3), not in T. The I2 integrand carries the
    # D_K ~ T^(-5/3) divergence of the cold end, which is integrable but which a
    # T-uniform grid resolves badly; in tau = T^(1/3) the integrand is regular
    # (see _analytic_I2_numerical), so this spacing both tracks the branch and
    # makes the quadrature converge.
    temperatures = (
        np.linspace(float(T_inf) ** (1.0 / 3.0), T_cold ** (1.0 / 3.0), int(n_steps))
    ) ** 3
    a_list, muB_list, muK_list, T_list = [], [], [], []
    guess = np.array([float(muB_inf), 0.0], dtype=float)
    nB_plus = np.nan
    saturated = False

    for index, T in enumerate(temperatures):
        # muK = 0 is a degenerate seed: it sits on the muK >= 0 domain edge, so
        # hybr can report success while never leaving it. Nudge it positive once
        # the equilibrated end has been recorded, and retry perturbed seeds
        # before declaring the step failed.
        seeds = [guess]
        if index > 0:
            nudged = guess.copy()
            nudged[1] = max(float(nudged[1]), 1.0)
            if nudged[1] != guess[1]:
                seeds.append(nudged)
            for factor in (2.0, 0.5):
                extra = nudged.copy()
                extra[1] = max(float(nudged[1]) * factor, 0.1)
                seeds.append(extra)
        solution = None
        for seed in seeds:
            candidate = root(residual, seed, args=(float(T),), method="hybr")
            if np.linalg.norm(residual(candidate.x, float(T)), ord=np.inf) <= 1.0e-8:
                solution = candidate
                break
        if solution is None and index > 0:
            # The unconstrained solve can step through muK = 0 and become
            # trapped on the artificial negative-muK penalty. Retry in log(muK)
            # so every trial remains on the physical positive-composition
            # branch while retaining muB as an unconstrained root variable.
            def positive_muK_residual(vec):
                muB_trial, log_muK = float(vec[0]), float(vec[1])
                if (not np.isfinite(log_muK)) or not (-50.0 < log_muK < 20.0):
                    return np.array([1.0e6, 1.0e6], dtype=float)
                return residual(
                    np.array([muB_trial, float(np.exp(log_muK))], dtype=float),
                    float(T),
                )

            base_muK = max(float(guess[1]), 1.0)
            for muK_seed in (base_muK, 0.5 * base_muK, 2.0 * base_muK, 0.1, 5.0):
                transformed = root(
                    positive_muK_residual,
                    np.array([float(guess[0]), np.log(muK_seed)], dtype=float),
                    method="hybr",
                )
                if (
                    (not np.isfinite(transformed.x[1]))
                    or not (-50.0 < float(transformed.x[1]) < 20.0)
                ):
                    continue
                mapped = np.array(
                    [float(transformed.x[0]), float(np.exp(transformed.x[1]))],
                    dtype=float,
                )
                if np.linalg.norm(residual(mapped, float(T)), ord=np.inf) <= 1.0e-8:
                    transformed.x = mapped
                    solution = transformed
                    break
        if solution is None:
            if index == 0:
                raise RuntimeError("layer trajectory failed at the equilibrated end")
            break
        muB, muK = float(solution.x[0]), float(max(solution.x[1], 0.0))
        _, _, nB, a = quark_point(muB, muK, float(T))
        if index == 0:
            nB_plus = nB
        elif a < a_list[-1] - 1.0e-9 or abs(muB - muB_list[-1]) > _LAYER_TRAJECTORY_MAX_DMUB * max(
            abs(muB_list[-1]), 1.0
        ):
            raise RuntimeError(
                "layer trajectory lost its branch: the (P, h/nB) system has "
                "multiple roots at low T and the continuation step was too "
                "coarse; increase n_steps"
            )
        a_clamped = float(min(max(a, 0.0), 1.0))
        a_list.append(a_clamped)
        muB_list.append(muB)
        muK_list.append(muK)
        T_list.append(float(T))
        guess = np.array([muB, muK], dtype=float)
        if a >= 1.0 - 1.0e-9:
            saturated = True
            break

    if len(a_list) < 2:
        raise RuntimeError("layer trajectory produced too few points to integrate")

    return {
        "a": np.asarray(a_list, dtype=float),
        "muB": np.asarray(muB_list, dtype=float),
        "muK": np.asarray(muK_list, dtype=float),
        "T": np.asarray(T_list, dtype=float),
        "P_plus": P_plus,
        "h_over_nB_plus": h_over_nB_plus,
        "B_one_forth": float(B_one_forth),
        "ms": float(ms),
        "upB": int(upB),
        "nB_plus": float(nB_plus),
        "A_boundary": float(a_list[-1]),
        "saturated": bool(saturated),
    }


def _analytic_trajectory_interface_state(trajectory, T_0plus):
    """Solve the conserved-layer state at a prescribed interface temperature."""
    T_0plus = float(T_0plus)
    if (not np.isfinite(T_0plus)) or T_0plus < 0.0:
        raise RuntimeError("T_0plus must be finite and non-negative")

    T_values = np.asarray(trajectory["T"], dtype=float)
    muB_values = np.asarray(trajectory["muB"], dtype=float)
    muK_values = np.asarray(trajectory["muK"], dtype=float)
    if T_0plus > T_values[0] + 1.0e-10:
        raise RuntimeError("T_0plus must not exceed the equilibrated T_inf")
    if T_0plus < T_values[-1] - 1.0e-10:
        raise RuntimeError(
            "T_0plus lies below the resolved conserved-layer trajectory"
        )

    muB_seed = float(np.interp(T_0plus, T_values[::-1], muB_values[::-1]))
    muK_seed = float(np.interp(T_0plus, T_values[::-1], muK_values[::-1]))
    P_plus = float(trajectory["P_plus"])
    h_over_nB_plus = float(trajectory["h_over_nB_plus"])
    B_one_forth = float(trajectory["B_one_forth"])
    ms = float(trajectory["ms"])
    upB = int(trajectory["upB"])

    def residual(vec):
        muB, muK = float(vec[0]), float(vec[1])
        if (not np.isfinite(muB)) or muB <= 0.0 or (not np.isfinite(muK)) or muK < 0.0:
            return np.array([1.0e6, 1.0e6], dtype=float)
        P = float(PQM(muB, muK, B_one_forth, T_0plus, ms=ms, upB=upB))
        e = float(
            edensQM(
                muB,
                muK,
                B_one_forth,
                T_0plus,
                ms=ms,
                include_em=False,
                upB=upB,
            )
        )
        nB = float(nB_QM(muB, muK, B_one_forth, T_0plus, ms=ms, upB=upB))
        if (not np.isfinite(nB)) or nB <= 0.0:
            return np.array([1.0e6, 1.0e6], dtype=float)
        return np.array(
            [
                (P - P_plus) / max(abs(P_plus), _FLOAT_TINY),
                ((P + e) / nB - h_over_nB_plus) / h_over_nB_plus,
            ],
            dtype=float,
        )

    solution = root(residual, np.array([muB_seed, muK_seed]), method="hybr")
    residual_norm = float(np.linalg.norm(residual(solution.x), ord=np.inf))
    if residual_norm > 1.0e-8:
        raise RuntimeError(
            "conserved-layer interface solve failed at prescribed T_0plus"
        )
    muB_0plus, muK_0plus = float(solution.x[0]), float(solution.x[1])
    nB_0plus = float(
        nB_QM(muB_0plus, muK_0plus, B_one_forth, T_0plus, ms=ms, upB=upB)
    )
    nK_0plus = float(
        nK_QM(muB_0plus, muK_0plus, B_one_forth, T_0plus, ms=ms, upB=upB)
    )
    if (not np.isfinite(nB_0plus)) or nB_0plus <= 0.0:
        raise RuntimeError("conserved-layer interface state has non-physical nB_0plus")
    a_0plus = float(nK_0plus / nB_0plus)
    A_boundary = float(trajectory["A_boundary"])
    if (not np.isfinite(a_0plus)) or not (0.0 < a_0plus < 1.0):
        raise RuntimeError("a_0plus must satisfy 0 < a_0plus < 1")
    if a_0plus > A_boundary + 1.0e-8:
        raise RuntimeError("exact a_0plus exceeds the trajectory A_boundary")
    return {
        "T_0plus": T_0plus,
        "muB_0plus": muB_0plus,
        "muK_0plus": muK_0plus,
        "nB_0plus": nB_0plus,
        "nK_0plus": nK_0plus,
        "a_0plus": a_0plus,
        "A_boundary": A_boundary,
        "muK_0plus_max": float(muK_values[-1]),
        "interface_scaled_residual": residual_norm,
    }


def _analytic_I2_numerical(trajectory, a_0plus, B_one_forth, ms=0.0, upB=5000):
    """
    Integrate I2 = (1/nB) * int_0^{a(0+)} D_K * Gamma_K da along the layer.

    D_K and Gamma_K are evaluated on the states of the conserved-(P, h/nB)
    trajectory, using the full diffusion coefficient (both the Landau-damped
    and Debye-screened terms) and the exact non-leptonic rate, rather than the
    Landau-only and cubic-rate approximations behind the closed form.
    """
    a_values = np.asarray(trajectory["a"], dtype=float)
    muB_values = np.asarray(trajectory["muB"], dtype=float)
    muK_values = np.asarray(trajectory["muK"], dtype=float)
    T_values = np.asarray(trajectory["T"], dtype=float)
    nB_plus = float(trajectory["nB_plus"])
    a_0plus = float(a_0plus)
    if (not np.isfinite(a_0plus)) or a_0plus <= 0.0:
        raise RuntimeError("I2 integration requires a positive interface fraction")
    if (not np.isfinite(nB_plus)) or nB_plus <= 0.0:
        raise RuntimeError("I2 integration requires a positive nB")

    integrand = np.empty(a_values.size, dtype=float)
    for index in range(a_values.size):
        if T_values[index] <= 0.0:
            raise RuntimeError("I2 integration requires T > 0 along the trajectory")
        micro = _microphysics_from_quark_state_energy(
            muB_values[index], T_values[index], allow_zero_temperature=False
        )
        invD = float(micro["invD"])
        if (not np.isfinite(invD)) or invD <= 0.0:
            raise RuntimeError("I2 integration requires a positive 1/D_K")
        Gamma_K = float(
            _exact_kaon_transport_rate(
                muB_values[index], muK_values[index], T_values[index], ms=ms, upB=upB
            )["Gamma_K"]
        )
        integrand[index] = Gamma_K / invD
    if not np.all(np.isfinite(integrand)):
        raise RuntimeError("I2 integrand is non-finite along the layer trajectory")

    # D_K ~ T^(-5/3) diverges at the cold end while da/dT ~ T there, so the
    # integrand is ~T^(-2/3) in T: integrable, but singular. Substituting
    # tau = T^(1/3) gives D_K*Gamma_K*(da/dtau) ~ const, so the trapezoid is
    # applied in tau rather than in a or T.
    tau = np.cbrt(T_values)
    a_max = float(min(a_0plus, a_values[-1]))
    mask = a_values <= a_max + 1.0e-15
    if int(np.count_nonzero(mask)) < 2:
        raise RuntimeError("I2 integration needs at least two trajectory samples")
    tau_sub = tau[mask]
    a_sub = a_values[mask]
    y_sub = integrand[mask]
    if a_sub[-1] < a_max - 1.0e-12:
        # Close the interval on a(0+) when it falls between two samples.
        tau_sub = np.append(tau_sub, float(np.interp(a_max, a_values, tau)))
        a_sub = np.append(a_sub, a_max)
        y_sub = np.append(y_sub, float(np.interp(a_max, a_values, integrand)))
    da_dtau = np.gradient(a_sub, tau_sub)
    integral = float(abs(np.trapezoid(y_sub * da_dtau, tau_sub)))
    if (not np.isfinite(integral)) or integral < 0.0:
        raise RuntimeError("I2 integral is non-physical")
    return float(integral / nB_plus)


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
        h_over_nB_0minus = float(nuclear_state["h_over_nB_0minus"])
        raise SlowFrontNoSolution(
            (
                "No steadily moving conversion front exists: the metastable neutron "
                f"matter has enthalpy per baryon {h_over_nB_0minus:.4f} MeV, below the "
                f"coldest equilibrated quark matter on the P_0minus isobar at "
                f"{muB_cold:.4f} MeV (gap {gap:.4f} MeV). Energy-plus-baryon flux "
                "conservation has no root, so only the static u_0minus = 0 coexistence "
                "exists at this metastability and temperature."
            ),
            gap=gap,
            muB_cold=muB_cold,
            h_over_nB_0minus=h_over_nB_0minus,
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

def muB_from_nB_physical(
    nB_target,
    Temp,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    muB_lo=950.0,
    muB_hi=2200.0,
    rtol=1.0e-6,
    scan_points=48,
    auto_expand=False,
    muB_floor=0.0,
    muB_ceiling=5000.0,
    max_expansions=6,
):
    """
    Invert nB -> muB on the physical mean-field branch.

    nB_NM is not reliably single-valued in muB: the spurious sigma < 0 RMF root
    makes it collapse to sub-saturation values at isolated muB, so a plain
    bracketed solve can return a muB whose true density is far from the request.
    This helper solves against the branch-validated density and then verifies
    that the density at the answer reproduces nB_target to rtol, raising instead
    of returning a silently wrong muB. When auto_expand is true, muB_lo and
    muB_hi define the initial scan window: a window whose validated densities
    all lie above (below) the target is expanded downward (upward) until a
    bracket is found or the configured limit is reached. The default rtol sits
    above the RMF solver's own convergence floor (a few times 1e-8) and far
    below the percent-level error the spurious branch produces.
    """
    nB_target = float(nB_target)
    Temp = float(Temp)
    if (not np.isfinite(nB_target)) or nB_target <= 0.0:
        raise RuntimeError("nB_target must be positive and finite")
    muB_lo = float(muB_lo)
    muB_hi = float(muB_hi)
    muB_floor = float(muB_floor)
    muB_ceiling = float(muB_ceiling)
    scan_points = int(scan_points)
    max_expansions = int(max_expansions)
    if not np.all(np.isfinite([muB_lo, muB_hi, muB_floor, muB_ceiling])):
        raise RuntimeError("muB inversion bounds must be finite")
    if muB_lo >= muB_hi:
        raise RuntimeError("muB_lo must be smaller than muB_hi")
    if scan_points < 2:
        raise RuntimeError("scan_points must be at least 2")
    if max_expansions < 0:
        raise RuntimeError("max_expansions must be non-negative")
    if bool(auto_expand) and (muB_floor > muB_lo or muB_ceiling < muB_hi):
        raise RuntimeError(
            "automatic muB expansion requires muB_floor <= muB_lo < muB_hi "
            "<= muB_ceiling"
        )

    def density(muB):
        if str(NM_type) == "PNM":
            return _validated_pnm_state(float(muB), Temp, param)[2]
        return float(nB_NM(float(muB), Temp, param=param, NM_type=NM_type))

    def residual(muB):
        return density(muB) - nB_target

    # Scan only states accepted by the physical-branch validator. At finite
    # temperature the requested density can occur far below its cold-matter
    # chemical potential, so optionally extend an initially inadequate window.
    sample_cache = {}

    def sample_interval(lower, upper):
        for muB in np.linspace(float(lower), float(upper), scan_points):
            coordinate = float(muB)
            if coordinate in sample_cache:
                continue
            try:
                sample_cache[coordinate] = float(residual(coordinate))
            except Exception:
                continue

    def ordered_samples():
        return sorted(sample_cache.items())

    def find_bracket(samples):
        for (mu_a, f_a), (mu_b, f_b) in zip(samples[:-1], samples[1:]):
            if f_a == 0.0:
                return mu_a, mu_a
            if f_a * f_b < 0.0:
                return mu_a, mu_b
        if samples and samples[-1][1] == 0.0:
            return samples[-1][0], samples[-1][0]
        return None

    current_lo = muB_lo
    current_hi = muB_hi
    sample_interval(current_lo, current_hi)
    expansion_count = 0
    bracket = find_bracket(ordered_samples())
    while bracket is None and bool(auto_expand) and expansion_count < max_expansions:
        samples = ordered_samples()
        residuals = np.asarray([value for _, value in samples], dtype=float)
        span = max(current_hi - current_lo, 1.0)
        expanded = False
        if residuals.size < 2 or np.all(residuals > 0.0):
            new_lo = max(muB_floor, current_lo - span)
            if new_lo < current_lo:
                sample_interval(new_lo, current_lo)
                current_lo = new_lo
                expanded = True
        if residuals.size < 2 or np.all(residuals < 0.0):
            new_hi = min(muB_ceiling, current_hi + span)
            if new_hi > current_hi:
                sample_interval(current_hi, new_hi)
                current_hi = new_hi
                expanded = True
        if not expanded:
            break
        expansion_count += 1
        bracket = find_bracket(ordered_samples())

    samples = ordered_samples()
    if len(samples) < 2:
        raise RuntimeError(
            f"no physical PNM branch found in [{current_lo:.1f}, {current_hi:.1f}] "
            f"MeV "
            f"at T={Temp:.4g}"
        )
    if bracket is None:
        densities = np.asarray(
            [sample_residual + nB_target for _, sample_residual in samples],
            dtype=float,
        )
        raise RuntimeError(
            f"nB={nB_target:.6e} is not bracketed on the physical branch: the scan "
            f"window [{current_lo:.1f}, {current_hi:.1f}] MeV spans validated nB "
            f"in [{densities.min():.6e}, {densities.max():.6e}] at T={Temp:.4g} "
            f"after {expansion_count} expansion(s)"
        )
    if bracket[0] == bracket[1]:
        muB = float(bracket[0])
    else:
        muB = float(
            root_scalar(
                residual,
                bracket=bracket,
                method="brentq",
                xtol=1.0e-10,
                rtol=1.0e-12,
            ).root
        )
    achieved = density(muB)
    if abs(achieved - nB_target) > rtol * abs(nB_target):
        raise RuntimeError(
            f"nB -> muB inversion did not converge onto the physical branch: "
            f"requested nB={nB_target:.6e}, muB={muB:.4f} gives nB={achieved:.6e} "
            f"(relative error {abs(achieved - nB_target) / abs(nB_target):.3e})"
        )
    return muB


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


def _analytic_a_0plus_lte(A_boundary, u_0minus, lambda_n):
    """
    Return the exact LTE-limited interface fraction and its quadratic diagnostics.
    """
    A_boundary = float(A_boundary)
    u_0minus = float(u_0minus)
    lambda_n = float(lambda_n)
    if (not np.isfinite(A_boundary)) or A_boundary <= 0.0:
        raise RuntimeError("A_boundary must be positive and finite")
    if (not np.isfinite(u_0minus)) or u_0minus < 0.0:
        raise RuntimeError("LTE a_0plus requires finite u_0minus >= 0")
    if (not np.isfinite(lambda_n)) or lambda_n <= 0.0:
        raise RuntimeError("LTE a_0plus requires finite lambda_n > 0")

    beta = float(5.0 * u_0minus * lambda_n)
    discriminant = float(
        beta * beta + 4.0 * (1.0 - beta) * A_boundary * A_boundary
    )
    discriminant_scale = max(
        beta * beta,
        4.0 * abs(1.0 - beta) * A_boundary * A_boundary,
        1.0,
    )
    if discriminant < -1.0e-14 * discriminant_scale:
        raise RuntimeError("LTE a_0plus discriminant is negative")
    discriminant = max(discriminant, 0.0)
    stable_denominator = float(np.sqrt(discriminant) + beta)
    if (not np.isfinite(stable_denominator)) or stable_denominator <= 0.0:
        raise RuntimeError("LTE a_0plus denominator is non-physical")

    a_0plus_LTE = float(
        2.0 * A_boundary * A_boundary / stable_denominator
    )
    if (not np.isfinite(a_0plus_LTE)) or a_0plus_LTE <= 0.0:
        raise RuntimeError("a_0plus_LTE must be positive and finite")

    return {
        "A_boundary": A_boundary,
        "a_0plus_LTE": a_0plus_LTE,
        "lambda_n": lambda_n,
        "beta_LTE": beta,
        "lte_discriminant": discriminant,
    }


# Mean-field seeds for the PNM branch retry. The scalar gap equation is cubic in
# sigma, so it admits a second root with sigma < 0 (effective mass above the
# vacuum mass) carrying negative pressure at sub-saturation density. That root is
# a real solution of the model but is unphysical: it lies inside the spinodal,
# where uniform neutron matter is mechanically unstable and real matter breaks up
# into crust. The default seed occasionally falls into its basin at isolated muB,
# so these alternatives re-seed the solve near the physical (sigma > 0) root.
# Seeds that land on the same branch reproduce sigma to well under this, and the
# off-branch roots differ from it by many percent, so the exact value is not
# delicate.
_PNM_BRANCH_AGREEMENT_RTOL = 1.0e-6

_PNM_BRANCH_SEEDS = (
    (30.0, 20.0, -3.0),
    (70.0, 70.0, -3.0),
    (90.0, 95.0, -3.0),
    (50.0, 45.0, -3.0),
    (110.0, 120.0, -3.0),
)


def _pnm_sigma_field(muB, T, param, seed):
    """
    Return the scalar mean field sigma for PNM at (muB, T) from a given seed.

    sigma > 0 is the physical branch: it lowers the effective nucleon mass
    M* = M - g_sigma*sigma below its vacuum value. sigma < 0 marks the spurious
    root.
    """
    sigma_init, w0_init, r03_init = seed
    solution = RMFsolvePNM_mu(
        float(muB),
        float(T),
        param,
        sigma_init=float(sigma_init),
        w0_init=float(w0_init),
        r03_init=float(r03_init),
        verb=False,
    )
    return float(np.asarray(solution[0][0]).ravel()[0])


def _pnm_state_at_seed(muB, T, param, seed):
    """
    Evaluate (P, e, nB, sigma) for PNM at (muB, T) from one mean-field seed.
    """
    sigma_init, w0_init, r03_init = seed
    common = dict(
        input_num=float(muB),
        input_type="muB",
        Trmf=float(T),
        para=param,
        sigma_init=float(sigma_init),
        w0_init=float(w0_init),
        r03_init=float(r03_init),
        mub_init=990,
        verb=False,
    )
    P = float(np.asarray(RMFpressurePNM(**common)).item())
    e = float(np.asarray(RMFedensPNM(**common)).item())
    nB = float(np.asarray(RMFbaryon_densityPNM(**common)).item())
    sigma = _pnm_sigma_field(muB, T, param, seed)
    return P, e, nB, sigma


def _pnm_branch_rejection(P, e, nB, sigma):
    """
    Return a reason string when a PNM solve landed off the physical branch.

    The tests are, in order of directness: sigma > 0 (the branch signature),
    P > 0 (pure neutron matter has no self-bound state, so its pressure is
    positive at every density, whereas the spurious root carries P < 0), and
    ordinary finiteness/positivity of the density and enthalpy.
    """
    if not np.all(np.isfinite([P, e, nB, sigma])):
        return "nuclear EOS returned non-finite values"
    if sigma <= 0.0:
        return (
            f"scalar field sigma={sigma:.4f} <= 0, i.e. an effective nucleon mass "
            "above the vacuum mass: the solve landed on the spurious RMF branch"
        )
    if P <= 0.0:
        return f"pressure P={P:.6e} <= 0 is unphysical for pure neutron matter"
    if nB <= 0.0:
        return f"baryon density nB={nB:.6e} is not positive"
    if P + e <= 0.0:
        return "enthalpy density is not positive"
    return ""


def _pnm_state_is_off_branch(muB, T, param, P, e, nB):
    """
    Decide whether a PNM solve landed on the spurious sigma < 0 branch.

    Observables alone are not enough: the spurious root usually carries a
    negative pressure, but not always (near muB ~ 1670 MeV it comes out positive
    while the density is still an order of magnitude too low). The scalar field
    is the exact signature, so it is checked directly, with the cheap
    finiteness/positivity screen kept as a first pass.

    The scalar-field probe uses the same default seed as the PNM helpers, so it
    reports the branch those helpers actually reached. If that probe cannot be
    solved the state is accepted rather than rejected: an unconfirmed suspicion
    should not turn a working call into a failure.

    Only branch selection is treated here. Non-finite values, a non-positive
    density or a non-positive enthalpy are not branch symptoms (the spurious
    root is finite and has nB > 0); they are left for the caller's own domain
    validation so that genuinely invalid input still raises its proper error
    instead of being silently replaced by a re-solved state.
    """
    if not np.all(np.isfinite([P, e, nB])) or nB <= 0.0 or (P + e) <= 0.0:
        return False
    if P <= 0.0:
        return True
    try:
        sigma = _pnm_sigma_field(muB, T, param, _PNM_BRANCH_SEEDS[0])
    except Exception:
        return False
    return bool(np.isfinite(sigma) and sigma <= 0.0)


def _validated_pnm_state(muB, T, param):
    """
    Solve PNM at (muB, T) on the physical mean-field branch.

    Seeds are tried in turn and the state is accepted once two of them agree on
    sigma, which is what pins the vacuum-connected branch; if every seed lands
    off the physical branch a RuntimeError is raised rather than a silently
    wrong state.

    _pnm_branch_rejection alone is not enough. It catches the sigma < 0 root,
    but two further ways of leaving the branch carry sigma > 0 and P > 0 and so
    slip past it, each over a muB window only a fraction of an MeV wide:

      * the collapsed large-sigma root of the scalar gap equation (near
        muB = 2450 MeV at T = 0.01 the first seed gives sigma = 108.31 and a
        density 7.7x the physical one), and
      * a solve that never moves off its own seed (near muB = 2705 the
        (50, 45, -3) seed returns sigma = 50.0 exactly, 28x too dense).

    Either one puts an isolated spike in nB(muB), and a spike is
    indistinguishable from a root to any bracketing solve, so muB_from_nB
    converges onto the jump instead of the density it was asked for. Both are
    minority outcomes -- the remaining seeds agree on the physical root -- so
    agreement between two seeds is the discriminator. Note that neither the
    smallest sigma nor the largest pressure identifies the physical branch: the
    collapsed root carries the larger pressure at the same muB, and the
    stalled seed carries the smaller sigma.
    """
    reasons = []
    accepted = []
    for seed in _PNM_BRANCH_SEEDS:
        try:
            P, e, nB, sigma = _pnm_state_at_seed(muB, T, param, seed)
        except Exception as exc:
            reasons.append(f"seed {seed}: solve failed ({str(exc)[:80]})")
            continue
        reason = _pnm_branch_rejection(P, e, nB, sigma)
        if reason:
            reasons.append(f"seed {seed}: {reason}")
            continue
        for other in accepted:
            if abs(sigma - other[3]) <= _PNM_BRANCH_AGREEMENT_RTOL * max(
                abs(sigma), abs(other[3]), 1.0
            ):
                return P, e, nB, sigma
        accepted.append((P, e, nB, sigma))
    if len(accepted) == 1:
        # Only one seed reached a physically acceptable state, so there is
        # nothing to corroborate it against; the single state is still the best
        # available answer and matches the pre-agreement behaviour.
        return accepted[0]
    if accepted:
        sigmas = ", ".join(f"{state[3]:.6f}" for state in accepted)
        raise RuntimeError(
            f"PNM solve at muB={float(muB):.4f}, T={float(T):.4g} did not settle on "
            f"one mean-field branch: no two seeds agreed on sigma (got {sigmas})"
        )
    raise RuntimeError(
        f"PNM solve at muB={float(muB):.4f}, T={float(T):.4g} could not reach the "
        "physical mean-field branch from any seed: " + "; ".join(reasons)
    )


def _analytic_nuclear_state(muB_0minus, T_0minus, param=para.paraQMCRMF3, NM_type="PNM"):
    """
    Return the upstream nuclear state used by analytic_velocity_bound.

    For PNM the state always comes from _validated_pnm_state, which pins the
    vacuum-connected branch by agreement between seeds. Screening the
    default-seed solve first is not enough: the screen only sees the sigma < 0
    root, so the collapsed large-sigma root and a solve that never left its seed
    both pass it and the wrong branch reaches the caller silently. The validator
    costs one extra mean-field solve, against the 72-point eigenvalue scan that
    follows.
    """
    if str(NM_type) == "PNM":
        P_0minus, e_0minus, nB_0minus, _sigma_0minus = _validated_pnm_state(
            muB_0minus, T_0minus, param
        )
    else:
        P_0minus = float(PNM(muB_0minus, T_0minus, param=param, NM_type=NM_type))
        e_0minus = float(edensNM(muB_0minus, T_0minus, param=param))
        nB_0minus = float(nB_NM(muB_0minus, T_0minus, param=param, NM_type=NM_type))
    h_0minus = float(P_0minus + e_0minus)
    if (not np.isfinite(P_0minus)) or (not np.isfinite(e_0minus)) or (not np.isfinite(h_0minus)):
        raise RuntimeError("Nuclear EOS returned non-finite pressure or enthalpy")
    if (not np.isfinite(nB_0minus)) or nB_0minus <= 0.0:
        raise RuntimeError("nB_0minus must be positive and finite")
    if h_0minus <= 0.0:
        raise RuntimeError("h_0minus must be positive")
    h_over_nB_0minus = float(h_0minus / nB_0minus)
    if (not np.isfinite(h_over_nB_0minus)) or h_over_nB_0minus <= 0.0:
        raise RuntimeError("Nuclear h_0minus/nB_0minus must be positive and finite")
    return {
        "P_0minus": P_0minus,
        "e_0minus": e_0minus,
        "h_0minus": h_0minus,
        "nB_0minus": nB_0minus,
        "h_over_nB_0minus": h_over_nB_0minus,
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


def _solve_analytic_inf_endpoint_for_u_0minus(
    u_0minus,
    nuclear_state,
    B_one_forth,
    ms=0.0,
    upB=5000,
    initial_guess=None,
):
    """
    Solve the muK=0 downstream quark endpoint from exact hydro jump conditions.
    """
    u_0minus = float(u_0minus)
    if (not np.isfinite(u_0minus)) or u_0minus <= 0.0:
        raise RuntimeError("Trial u_0minus must be positive and finite")

    P_0minus = float(nuclear_state["P_0minus"])
    h_0minus = float(nuclear_state["h_0minus"])
    nB_0minus = float(nuclear_state["nB_0minus"])
    h_over_nB_0minus = float(nuclear_state["h_over_nB_0minus"])
    jB = float(nB_0minus * u_0minus)
    gamma_0minus = float(np.sqrt(1.0 + u_0minus * u_0minus))
    energy_flux_0minus = float(h_0minus * u_0minus * gamma_0minus)
    momentum_flux_0minus = float(P_0minus + h_0minus * u_0minus * u_0minus)
    energy_target = float(h_over_nB_0minus * gamma_0minus)

    guesses = []
    if initial_guess is not None:
        guess_arr = np.asarray(initial_guess, dtype=float).ravel()
        if guess_arr.size >= 2:
            _append_analytic_endpoint_guess(guesses, guess_arr[0], guess_arr[1])
    for muB_guess in (
        float(nuclear_state.get("muB_0minus", 0.0)),
        900.0,
        1100.0,
        1300.0,
        1500.0,
        700.0,
    ):
        for T_guess in (
            float(nuclear_state.get("T_0minus", 0.0)),
            max(float(nuclear_state.get("T_0minus", 0.0)), 1.0),
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
        T_inf = float(np.exp(logT))
        try:
            P_inf = float(PQM(muB, 0.0, B_one_forth, T_inf, ms=ms, upB=upB))
            e_inf = float(
                edensQM(
                    muB,
                    0.0,
                    B_one_forth,
                    T_inf,
                    ms=ms,
                    include_em=False,
                    upB=upB,
                )
            )
            nB_inf = float(nB_QM(muB, 0.0, B_one_forth, T_inf, ms=ms, upB=upB))
        except Exception:
            return np.array([1.0e30, 1.0e30], dtype=float)
        if (
            (not np.isfinite(P_inf))
            or (not np.isfinite(e_inf))
            or (not np.isfinite(nB_inf))
            or nB_inf <= 0.0
        ):
            return np.array([1.0e30, 1.0e30], dtype=float)
        h_inf = float(P_inf + e_inf)
        u_inf = float(jB / nB_inf)
        gamma_inf = float(np.sqrt(1.0 + u_inf * u_inf))
        energy_flux_inf = float(h_inf * u_inf * gamma_inf)
        momentum_flux_inf = float(P_inf + h_inf * u_inf * u_inf)
        if (
            (not np.isfinite(h_inf))
            or h_inf <= 0.0
            or (not np.isfinite(energy_flux_inf))
            or (not np.isfinite(momentum_flux_inf))
        ):
            return np.array([1.0e30, 1.0e30], dtype=float)
        energy_residual = float(h_inf * gamma_inf / nB_inf - energy_target)
        pressure_jump = float(P_inf - P_0minus)
        pressure_jump_balance = float(h_0minus * u_0minus * u_0minus - h_inf * u_inf * u_inf)
        # Scale the momentum residual by the pressure, not by the momentum flux
        # h*u^2. For slow fronts h*u^2 -> 0 and flooring at 1.0 left this
        # residual as an absolute value in MeV^4 (~1e7), mis-scaled against the
        # relative energy residual, which stalled the hybr solve. The pressure
        # scale keeps both residuals relative and comparable.
        momentum_scale = max(abs(P_0minus), abs(P_inf), 1.0)
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

    muB_inf = float(best[0])
    T_inf = float(np.exp(float(best[1])))
    if (not np.isfinite(muB_inf)) or muB_inf <= 0.0 or (not np.isfinite(T_inf)) or T_inf <= 0.0:
        raise RuntimeError("Analytic hydro endpoint solve returned a non-physical root")

    P_inf = float(PQM(muB_inf, 0.0, B_one_forth, T_inf, ms=ms, upB=upB))
    e_inf = float(
        edensQM(
            muB_inf,
            0.0,
            B_one_forth,
            T_inf,
            ms=ms,
            include_em=False,
            upB=upB,
        )
    )
    h_inf = float(P_inf + e_inf)
    nB_inf = float(nB_QM(muB_inf, 0.0, B_one_forth, T_inf, ms=ms, upB=upB))
    if (
        (not np.isfinite(P_inf))
        or (not np.isfinite(e_inf))
        or (not np.isfinite(h_inf))
        or h_inf <= 0.0
        or (not np.isfinite(nB_inf))
        or nB_inf <= 0.0
    ):
        raise RuntimeError("Hydro endpoint quark EOS returned non-physical quantities")

    u_inf = float(jB / nB_inf)
    gamma_inf = float(np.sqrt(1.0 + u_inf * u_inf))
    energy_flux_inf = float(h_inf * u_inf * gamma_inf)
    momentum_flux_inf = float(P_inf + h_inf * u_inf * u_inf)
    energy_flux_residual = float(energy_flux_inf - energy_flux_0minus)
    momentum_flux_residual = float(momentum_flux_inf - momentum_flux_0minus)
    pressure_jump = float(P_inf - P_0minus)
    pressure_jump_balance = float(h_0minus * u_0minus * u_0minus - h_inf * u_inf * u_inf)

    return {
        "muB_inf": muB_inf,
        "T_inf": T_inf,
        "P_inf": P_inf,
        "e_inf": e_inf,
        "h_inf": h_inf,
        "nB_inf": nB_inf,
        "h_over_nB_inf": float(h_inf / nB_inf),
        "u_0minus": u_0minus,
        "u_inf": u_inf,
        "jB": jB,
        "gamma_0minus": gamma_0minus,
        "gamma_inf": gamma_inf,
        "energy_flux_0minus": energy_flux_0minus,
        "energy_flux_inf": energy_flux_inf,
        "momentum_flux_0minus": momentum_flux_0minus,
        "momentum_flux_inf": momentum_flux_inf,
        "energy_flux_residual": energy_flux_residual,
        "momentum_flux_residual": momentum_flux_residual,
        "pressure_jump": pressure_jump,
        "pressure_jump_balance": pressure_jump_balance,
        "pressure_jump_residual": float(pressure_jump - pressure_jump_balance),
        "endpoint_scaled_residual": best_norm,
        "endpoint_initial_guess": (muB_inf, T_inf),
        "h_over_nB_0minus": h_over_nB_0minus,
        "U_bag": float(B_one_forth) ** 4,
        "B_one_forth": float(B_one_forth),
        "ms": float(ms),
        "upB": int(upB),
    }


def _analytic_velocity_formula_from_endpoint(
    endpoint,
    nuclear_state,
    xi,
    T_0plus=None,
    interface_control="fixed_T_0plus",
    velocity_closure="closed_form",
):
    """Evaluate a fixed-temperature or LTE analytic closure at one endpoint."""
    velocity_closure = _normalize_velocity_closure(velocity_closure)
    xi = float(xi)
    if interface_control not in ("fixed_T_0plus", "LTE"):
        raise ValueError("interface_control must be 'fixed_T_0plus' or 'LTE'")
    if interface_control == "fixed_T_0plus":
        T_0plus = float(T_0plus)
    muB_inf = float(endpoint["muB_inf"])
    T_inf = float(endpoint["T_inf"])
    nB_inf = float(endpoint["nB_inf"])
    u_0minus = float(endpoint["u_0minus"])
    nB_0minus = float(nuclear_state["nB_0minus"])

    mu_q = float(muB_inf / 3.0)
    weak_rate = _analytic_weak_rate_from_mu_q(mu_q)
    gamma = float(weak_rate["gamma"])
    tau = float(weak_rate["tau"])
    tau_seconds = float(weak_rate["tau_seconds"])

    lambda_n = float(nB_0minus / nB_inf)
    if (not np.isfinite(lambda_n)) or lambda_n <= 0.0:
        raise RuntimeError("lambda_n must be positive and finite")

    P_inf = float(endpoint.get("P_inf", np.nan))
    B_one_forth_inf = endpoint.get("B_one_forth", None)
    if B_one_forth_inf is None:
        U_bag_inf = endpoint.get("U_bag", None)
        B_one_forth_inf = float(U_bag_inf) ** 0.25 if U_bag_inf is not None else None
    if B_one_forth_inf is None or not np.isfinite(P_inf):
        raise RuntimeError(
            "endpoint must carry B_one_forth (or U_bag) and P_inf to evaluate A"
        )
    ms_inf = float(endpoint.get("ms", 0.0))
    upB_inf = int(endpoint.get("upB", 5000))
    layer_trajectory = None
    if velocity_closure == "numerical_I2":
        layer_trajectory = _analytic_layer_trajectory(
            P_inf,
            float(endpoint["h_over_nB_inf"]),
            muB_inf,
            T_inf,
            float(B_one_forth_inf),
            ms=ms_inf,
            upB=upB_inf,
        )
        A_boundary = float(layer_trajectory["A_boundary"])
        muK_0plus_max = float(layer_trajectory["muK"][-1])
        if interface_control == "fixed_T_0plus":
            interface_state = _analytic_trajectory_interface_state(
                layer_trajectory,
                T_0plus,
            )
    else:
        A_boundary, muK_0plus_max = _analytic_A_from_isobar(
            muB_inf,
            P_inf,
            float(B_one_forth_inf),
            ms=ms_inf,
            upB=upB_inf,
            return_muK=True,
        )
        if interface_control == "fixed_T_0plus":
            interface_state = _analytic_fixed_muB_interface_state(
                muB_inf,
                P_inf,
                T_0plus,
                float(B_one_forth_inf),
                ms=ms_inf,
                upB=upB_inf,
            )
    one_minus_A_boundary = float(1.0 - A_boundary)
    one_plus_xi_A_boundary = float(1.0 + xi * A_boundary)
    lambda_n_squared = float(lambda_n * lambda_n)
    if (not np.isfinite(A_boundary)) or A_boundary <= 0.0:
        raise RuntimeError("A_boundary must be positive and finite")

    alpha_s = float(_TRANSPORT_ALPHA_S)
    h_D = float(_TRANSPORT_H_CONST)
    beta_LTE = np.nan
    a_0plus_LTE = np.nan
    lte_correction = np.nan

    if interface_control == "LTE":
        lte_data = _analytic_a_0plus_lte(A_boundary, u_0minus, lambda_n)
        beta_LTE = float(lte_data["beta_LTE"])
        a_0plus_LTE = float(lte_data["a_0plus_LTE"])
        if a_0plus_LTE >= 1.0:
            raise RuntimeError(
                "a_0plus_LTE must satisfy 0 < a_0plus_LTE < 1"
            )
        a_0plus = a_0plus_LTE
        T_0plus_result = np.nan
        muB_0plus = np.nan
        muK_0plus = np.nan
        nB_0plus = np.nan
        nK_0plus = np.nan
    else:
        a_0plus = float(interface_state["a_0plus"])
        T_0plus_result = float(interface_state["T_0plus"])
        muB_0plus = float(interface_state["muB_0plus"])
        muK_0plus = float(interface_state["muK_0plus"])
        nB_0plus = float(interface_state["nB_0plus"])
        nK_0plus = float(interface_state["nK_0plus"])

    if not (0.0 < a_0plus < min(A_boundary + 1.0e-10, 1.0)):
        raise RuntimeError("a_0plus must satisfy 0 < a_0plus <= A_boundary < 1")
    if A_boundary >= 1.0:
        raise RuntimeError("A_boundary must satisfy 0 < A_boundary < 1")
    one_minus_a_0plus = float(1.0 - a_0plus)
    one_plus_xi_a_0plus = float(1.0 + xi * a_0plus)
    if one_minus_a_0plus <= 0.0 or one_plus_xi_a_0plus <= 0.0:
        raise RuntimeError("analytic velocity denominator is non-physical")

    # This bracket is the closed-form I2 approximation. Its derivation uses the
    # quadratic relation between T^2 and A^2-a^2, but the physical a(0+) above
    # is always evaluated from its exact EOS definition nK/nB.
    z_raw = float(1.0 - (a_0plus / A_boundary) ** 2)
    if z_raw < -1.0e-12 or z_raw > 1.0 + 1.0e-12:
        raise RuntimeError("closed-form I2 argument must satisfy 0 <= z <= 1")
    z = float(np.clip(z_raw, 0.0, 1.0))
    lte_correction = float(
        24.0 / 7.0
        - 3.0 * z ** (1.0 / 6.0)
        - (3.0 / 7.0) * z ** (7.0 / 6.0)
    )
    if (not np.isfinite(lte_correction)) or lte_correction < 0.0:
        raise RuntimeError("closed-form I2 correction is non-physical")
    denominator = float(
        muB_inf * lambda_n_squared * one_minus_a_0plus * one_plus_xi_a_0plus
    )
    prefactor = float(
        (9.0 * np.pi ** (7.0 / 3.0) * gamma)
        / (4.0 * np.sqrt(2.0) * h_D * alpha_s ** (5.0 / 3.0))
    )
    u_0minus_formula_squared = float(
        prefactor * A_boundary ** (7.0 / 3.0) * lte_correction / denominator
    )

    # The closed-form branch above supplies I2 through the published bracket.
    # The numerical branch replaces it with the quadrature along the conserved
    # trajectory, keeping the same structural closure
    #   u^2 = 2 I2 / [lambda_n^2 (1 - a(0+)) (1 + xi a(0+))] ,
    # which stays exact here because nB is constant along the layer.
    I2_numerical = np.nan
    if velocity_closure == "numerical_I2":
        I2_numerical = _analytic_I2_numerical(
            layer_trajectory,
            a_0plus,
            float(B_one_forth_inf),
            ms=ms_inf,
            upB=upB_inf,
        )
        closure_denominator = float(
            lambda_n_squared * one_minus_a_0plus * one_plus_xi_a_0plus
        )
        if (not np.isfinite(closure_denominator)) or closure_denominator <= 0.0:
            raise RuntimeError("numerical-I2 velocity denominator is non-physical")
        u_0minus_formula_squared = float(2.0 * I2_numerical / closure_denominator)

    if (not np.isfinite(u_0minus_formula_squared)) or u_0minus_formula_squared < 0.0:
        raise RuntimeError("Analytic velocity bound produced non-physical u_0minus^2")
    return {
        "u_0minus_formula_squared": u_0minus_formula_squared,
        "velocity_closure": velocity_closure,
        "I2_numerical": float(I2_numerical),
        "layer_trajectory_saturated": bool(
            layer_trajectory["saturated"] if layer_trajectory is not None else False
        ),
        "layer_trajectory_points": int(
            layer_trajectory["a"].size if layer_trajectory is not None else 0
        ),
        "mu_q": mu_q,
        "lambda_n": lambda_n,
        "lambda_n_squared": lambda_n_squared,
        "A_boundary": A_boundary,
        "a_0plus": a_0plus,
        "a_0plus_LTE": a_0plus_LTE,
        "T_0plus": float(T_0plus_result),
        "muB_0plus": float(muB_0plus),
        "muK_0plus": float(muK_0plus),
        "nB_0plus": float(nB_0plus),
        "nK_0plus": float(nK_0plus),
        "interface_control": interface_control,
        "interface_fraction_mode": interface_control,
        "a_0plus_max": A_boundary,
        "a_0plus_max_mode": interface_control,
        "A_extreme_endothermic": A_boundary,
        "a_0plus_used": a_0plus,
        "beta_LTE": beta_LTE,
        "lte_correction": lte_correction,
        "closed_form_I2_correction": lte_correction,
        "muK_0plus_max": muK_0plus_max,
        "one_minus_A_boundary": one_minus_A_boundary,
        "one_plus_xi_A_boundary": one_plus_xi_A_boundary,
        "one_minus_a_0plus": one_minus_a_0plus,
        "one_plus_xi_a_0plus": one_plus_xi_a_0plus,
        "one_minus_a_0plus_used": one_minus_a_0plus,
        "one_plus_xi_a_0plus_used": one_plus_xi_a_0plus,
        "alpha_s": alpha_s,
        "h_D": h_D,
        "gamma": gamma,
        "tau": tau,
        "tau_seconds": tau_seconds,
        "prefactor": prefactor,
        "analytic_denominator": denominator,
    }


def _solve_analytic_velocity_bound(
    muB_0minus,
    T_0minus,
    B_one_forth,
    T_0plus,
    xi,
    velocity_closure,
    interface_control,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    upB=5000,
    initial_guess=None,
):
    """Shared scalar eigenvalue solve for the public analytical methods."""
    muB_0minus = float(muB_0minus)
    T_0minus = float(T_0minus)
    B_one_forth = float(B_one_forth)
    xi = float(xi)
    velocity_closure = _normalize_velocity_closure(velocity_closure)
    if interface_control == "fixed_T_0plus":
        T_0plus = float(T_0plus)

    if (not np.isfinite(muB_0minus)) or muB_0minus <= 0.0:
        raise RuntimeError("muB_0minus must be positive and finite")
    if (not np.isfinite(T_0minus)) or T_0minus < 0.0:
        raise RuntimeError("T_0minus must be finite and non-negative")
    if (not np.isfinite(B_one_forth)) or B_one_forth <= 0.0:
        raise RuntimeError("B_one_forth must be positive and finite")
    if (not np.isfinite(xi)) or not (-1.0 < xi < 1.0):
        raise RuntimeError("xi must satisfy -1 < xi < 1")
    if interface_control == "fixed_T_0plus" and (
        (not np.isfinite(T_0plus)) or T_0plus < 0.0
    ):
        raise RuntimeError("T_0plus must be finite and non-negative")

    nuclear_state = _analytic_nuclear_state(muB_0minus, T_0minus, param=param, NM_type=NM_type)
    nuclear_state["muB_0minus"] = muB_0minus
    nuclear_state["T_0minus"] = T_0minus

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
        u_0minus = float(np.exp(np.clip(theta, -700.0, np.log(0.999999))))
        endpoint = _solve_analytic_inf_endpoint_for_u_0minus(
            u_0minus,
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
            T_0plus=T_0plus,
            interface_control=interface_control,
            velocity_closure=velocity_closure,
        )
        residual = float(formula["u_0minus_formula_squared"] - u_0minus * u_0minus)
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
                "a_0plus_LTE must satisfy 0 < a_0plus_LTE < 1" in message
                or "a_0plus must satisfy 0 < a_0plus" in message
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

    u_0minus_max = float(final_data["u_0minus"])
    u_0minus_squared = float(u_0minus_max * u_0minus_max)
    formula_residual = float(final_data["u_0minus_formula_squared"] - u_0minus_squared)
    formula_scale = max(abs(float(final_data["u_0minus_formula_squared"])), abs(u_0minus_squared), 1.0)
    if abs(formula_residual) > 1.0e-8 * formula_scale:
        raise RuntimeError(f"Analytic velocity closure residual is too large: {formula_residual:.6e}")

    return {
        "success": True,
        "message": "hydro-consistent analytic velocity bound evaluated",
        "u_0minus_max": u_0minus_max,
        "u_0minus": u_0minus_max,
        "u_0minus_squared": u_0minus_squared,
        "u_0minus_formula_squared": float(final_data["u_0minus_formula_squared"]),
        "analytic_velocity_residual": formula_residual,
        "jB": float(final_data["jB"]),
        "muB_0minus": muB_0minus,
        "T_0minus": T_0minus,
        "P_0minus": float(nuclear_state["P_0minus"]),
        "e_0minus": float(nuclear_state["e_0minus"]),
        "h_0minus": float(nuclear_state["h_0minus"]),
        "nB_0minus": float(nuclear_state["nB_0minus"]),
        "h_over_nB_0minus": float(nuclear_state["h_over_nB_0minus"]),
        "muB_bar": float(final_data["muB_inf"]),
        "muB_inf": float(final_data["muB_inf"]),
        "T_inf": float(final_data["T_inf"]),
        "P_inf": float(final_data["P_inf"]),
        "e_inf": float(final_data["e_inf"]),
        "h_inf": float(final_data["h_inf"]),
        "nB_inf": float(final_data["nB_inf"]),
        "h_over_nB_inf": float(final_data["h_over_nB_inf"]),
        "u_inf": float(final_data["u_inf"]),
        "gamma_0minus": float(final_data["gamma_0minus"]),
        "gamma_inf": float(final_data["gamma_inf"]),
        "energy_flux_0minus": float(final_data["energy_flux_0minus"]),
        "energy_flux_inf": float(final_data["energy_flux_inf"]),
        "momentum_flux_0minus": float(final_data["momentum_flux_0minus"]),
        "momentum_flux_inf": float(final_data["momentum_flux_inf"]),
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
        "T_0plus": float(final_data["T_0plus"]),
        "muB_0plus": float(final_data["muB_0plus"]),
        "muK_0plus": float(final_data["muK_0plus"]),
        "nB_0plus": float(final_data["nB_0plus"]),
        "nK_0plus": float(final_data["nK_0plus"]),
        "a_0plus": float(final_data["a_0plus"]),
        "a_0plus_LTE": float(final_data["a_0plus_LTE"]),
        "interface_control": str(final_data["interface_control"]),
        "interface_fraction_mode": str(final_data["interface_fraction_mode"]),
        "a_0plus_max": float(final_data["a_0plus_max"]),
        "a_0plus_max_mode": str(final_data["a_0plus_max_mode"]),
        "A_extreme_endothermic": float(final_data["A_extreme_endothermic"]),
        "a_0plus_used": float(final_data["a_0plus_used"]),
        "beta_LTE": float(final_data["beta_LTE"]),
        "lte_correction": float(final_data["lte_correction"]),
        "closed_form_I2_correction": float(final_data["closed_form_I2_correction"]),
        "muK_0plus_max": float(final_data["muK_0plus_max"]),
        "one_minus_A_boundary": float(final_data["one_minus_A_boundary"]),
        "one_plus_xi_A_boundary": float(final_data["one_plus_xi_A_boundary"]),
        "one_minus_a_0plus": float(final_data["one_minus_a_0plus"]),
        "one_plus_xi_a_0plus": float(final_data["one_plus_xi_a_0plus"]),
        "one_minus_a_0plus_used": float(final_data["one_minus_a_0plus_used"]),
        "one_plus_xi_a_0plus_used": float(final_data["one_plus_xi_a_0plus_used"]),
        "alpha_s": float(final_data["alpha_s"]),
        "h_D": float(final_data["h_D"]),
        "gamma": float(final_data["gamma"]),
        "tau": float(final_data["tau"]),
        "tau_seconds": float(final_data["tau_seconds"]),
        "prefactor": float(final_data["prefactor"]),
        "analytic_denominator": float(final_data["analytic_denominator"]),
        "velocity_closure": str(final_data.get("velocity_closure", velocity_closure)),
        "I2_numerical": float(final_data.get("I2_numerical", np.nan)),
        "layer_trajectory_saturated": bool(
            final_data.get("layer_trajectory_saturated", False)
        ),
        "layer_trajectory_points": int(final_data.get("layer_trajectory_points", 0)),
        "xi": xi,
        "composition_definition": "a_local_equals_nK_over_nB",
        "density_ratio_definition": "lambda_n_equals_nB_0minus_over_nB_inf",
        "analytic_formula_variant": "piecewise_constant_lambda_n",
        "slow_front_consistent": bool(u_0minus_max < 1.0),
    }


class _AnalyticIsothermalSlowFrontInvalid(RuntimeError):
    """The formal isothermal root lies outside the slow-front regime."""

    def __init__(self, limit_data):
        super().__init__(
            "No analytical isothermal eigenvalue exists within u_0minus < 1; "
            "the slow-front approximation is invalid for this state"
        )
        self.limit_data = limit_data


def _solve_analytic_isothermal_log_root(
    evaluate_log_u,
    *,
    max_u_0minus=None,
):
    """Bracket and solve the positive isothermal velocity without a unit cap."""
    evaluation_cache = {}

    def cached_evaluate(theta):
        theta = float(theta)
        if theta not in evaluation_cache:
            evaluation_cache[theta] = evaluate_log_u(theta)
        return evaluation_cache[theta]

    seed_u = 1.0e-12
    seed_theta = float(np.log(seed_u))
    seed_residual, seed_data = cached_evaluate(seed_theta)
    seed_residual = float(seed_residual)
    if not np.isfinite(seed_residual):
        raise RuntimeError("Initial analytical isothermal residual is non-finite")
    if seed_residual == 0.0:
        return seed_data

    direction = 4.0 if seed_residual > 0.0 else 0.25
    previous_u = seed_u
    previous_residual = seed_residual
    bracket = None
    for _ in range(512):
        next_u = float(previous_u * direction)
        reached_upper_limit = False
        if (
            direction > 1.0
            and max_u_0minus is not None
            and next_u >= float(max_u_0minus)
        ):
            next_u = float(max_u_0minus)
            reached_upper_limit = True
        if (not np.isfinite(next_u)) or next_u <= 0.0 or next_u == previous_u:
            break
        next_residual, next_data = cached_evaluate(float(np.log(next_u)))
        next_residual = float(next_residual)
        if not np.isfinite(next_residual):
            raise RuntimeError("Analytical isothermal residual is non-finite")
        if reached_upper_limit:
            raise _AnalyticIsothermalSlowFrontInvalid(next_data)
        if next_residual == 0.0:
            return next_data
        if previous_residual * next_residual < 0.0:
            bracket = (float(np.log(previous_u)), float(np.log(next_u)))
            break
        previous_u = next_u
        previous_residual = next_residual

    if bracket is None:
        direction_label = "above" if direction > 1.0 else "below"
        raise RuntimeError(
            "Analytical isothermal eigenvalue scan found no sign change "
            f"{direction_label} u_0minus={seed_u:.1e}"
        )

    def scalar_residual(theta):
        residual, _ = cached_evaluate(theta)
        return float(residual)

    root_result = root_scalar(
        scalar_residual,
        bracket=bracket,
        method="brentq",
        xtol=1.0e-12,
        rtol=1.0e-12,
    )
    if not root_result.converged:
        raise RuntimeError(
            "Analytical isothermal eigenvalue solve did not converge: "
            f"{root_result.flag}"
        )
    _, final_data = cached_evaluate(float(root_result.root))
    return final_data


def _solve_interface_0plus_from_local_a_and_Pi(
    a_0plus,
    Pi,
    jB,
    B_one_forth,
    T,
    ms=0.0,
    upB=5000,
    initial_guess=None,
):
    """Solve the fixed-T interface state at local a(0+) = nK(0+)/nB(0+)."""
    a_0plus = float(a_0plus)
    if not (0.0 < a_0plus < 1.0):
        raise RuntimeError("a_0plus must satisfy 0 < a_0plus < 1")

    pi_scale = max(abs(float(Pi)), 1.0)

    def equations(vec):
        muB_0plus, muK_0plus = map(float, vec)
        if (
            (not np.isfinite(muB_0plus))
            or (not np.isfinite(muK_0plus))
            or muB_0plus <= 0.0
        ):
            return np.array([1.0e12, 1.0e12], dtype=float)
        try:
            nB_0plus = float(
                nB_QM(
                    muB_0plus,
                    muK_0plus,
                    B_one_forth,
                    T,
                    ms=ms,
                    upB=upB,
                )
            )
            nK_0plus = float(
                nK_QM(
                    muB_0plus,
                    muK_0plus,
                    B_one_forth,
                    T,
                    ms=ms,
                    upB=upB,
                )
            )
            if nB_0plus <= 0.0:
                return np.array([1.0e12, 1.0e12], dtype=float)
            return np.array(
                [
                    (
                        _Pi_QM_state(
                            muB_0plus,
                            muK_0plus,
                            B_one_forth,
                            T,
                            jB,
                            ms=ms,
                            upB=upB,
                        )
                        - Pi
                    )
                    / pi_scale,
                    nK_0plus / nB_0plus - a_0plus,
                ],
                dtype=float,
            )
        except Exception:
            return np.array([1.0e12, 1.0e12], dtype=float)

    muK_seed = _branch_muK_seed(a_0plus)
    guesses = []
    if initial_guess is not None:
        guess = np.asarray(initial_guess, dtype=float).ravel()
        if guess.size >= 2 and np.all(np.isfinite(guess[:2])):
            guesses.append(np.array(guess[:2], dtype=float))
            muB_seed = float(guess[0])
        else:
            muB_seed = 1200.0
    else:
        muB_seed = 1200.0
    guesses.extend(
        [
            np.array([muB_seed, muK_seed], dtype=float),
            np.array([1200.0, muK_seed], dtype=float),
            np.array([1500.0, max(muK_seed, 400.0 * a_0plus)], dtype=float),
            np.array([900.0, muK_seed], dtype=float),
        ]
    )

    best = None
    best_norm = np.inf
    best_message = "local-a interface state solve did not converge"
    for guess in guesses:
        try:
            sol = root(
                equations,
                guess,
                method="hybr",
                options={"maxfev": 3000, "xtol": 1.0e-10},
            )
            if not np.all(np.isfinite(sol.x)):
                best_message = str(sol.message)
                continue
            residual = equations(sol.x)
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            muB_0plus = float(sol.x[0])
            muK_0plus = float(sol.x[1])
            nB_0plus = float(
                nB_QM(
                    muB_0plus,
                    muK_0plus,
                    B_one_forth,
                    T,
                    ms=ms,
                    upB=upB,
                )
            )
            if (
                residual_norm < best_norm
                and muB_0plus > 0.0
                and muK_0plus >= -1.0e-8
                and nB_0plus > 0.0
            ):
                best = (muB_0plus, max(muK_0plus, 0.0))
                best_norm = residual_norm
            if sol.success and best is not None and best_norm <= 1.0e-8:
                return best
            best_message = str(sol.message)
        except Exception as exc:
            best_message = str(exc)

    if best is not None and best_norm <= 1.0e-8:
        return best
    raise RuntimeError(
        "x = 0+ local-a state solve failed: "
        f"{best_message}; best scaled residual={best_norm:.3e}"
    )


def _momentum_flux_diagnostics(P_0minus, w_0minus, u_0minus):
    """Report how far the front departs from the static-pressure limit.

    The isothermal solvers impose the gamma -> 1 junction conditions,
    jB = nB*u and Pi = P + w*u**2.  ``momentum_flux_ratio`` is the size of
    the flux term relative to the pressure, (Pi - P)/P, and is the leading
    error of treating the interface as a static isobar.
    ``relativistic_flux_ratio`` is the same quantity with the exact
    gamma**2 factor, so their difference isolates the relativistic part.
    """
    P_0minus = float(P_0minus)
    w_0minus = float(w_0minus)
    u_0minus = float(u_0minus)
    if (
        (not np.isfinite(P_0minus))
        or (not np.isfinite(w_0minus))
        or (not np.isfinite(u_0minus))
        or P_0minus <= 0.0
        or not (0.0 <= u_0minus < 1.0)
    ):
        return {
            "momentum_flux_ratio": np.nan,
            "relativistic_flux_ratio": np.nan,
            "gamma_minus_1": np.nan,
        }
    gamma_squared = 1.0 / (1.0 - u_0minus * u_0minus)
    flux = w_0minus * u_0minus * u_0minus / P_0minus
    return {
        "momentum_flux_ratio": float(flux),
        "relativistic_flux_ratio": float(flux * gamma_squared),
        "gamma_minus_1": float(np.sqrt(gamma_squared) - 1.0),
    }


def _relativistic_flux_pair(nB, w, P, u):
    """Return the exact (jB, Pi) junction fluxes for 3-velocity ``u``.

    jB = nB*gamma*u and Pi = P + w*gamma**2*u**2, the planar steady-front
    baryon and momentum fluxes.  The gamma -> 1 limit is the pair the
    isothermal solvers use by default.
    """
    nB = float(nB)
    w = float(w)
    P = float(P)
    u = float(u)
    if (not np.isfinite(u)) or not (0.0 <= u < 1.0):
        raise RuntimeError("Front-frame velocity must satisfy 0 <= u < 1")
    gamma_squared = 1.0 / (1.0 - u * u)
    gamma = float(np.sqrt(gamma_squared))
    return float(nB * gamma * u), float(P + w * gamma_squared * u * u)


def _solve_a_0plus_max(
    muB_0minus,
    P_0minus,
    T,
    B_one_forth,
    ms=0.0,
    upB=5000,
    a_floor=1.0e-8,
    a_ceiling=1.0 - 1.0e-9,
):
    """Return the largest interface fraction the fixed-T transition supports.

    Three unknowns -- muB(0+), muK(0+) and a(0+) -- are fixed by three
    conditions at the interface: the fixed-T pressure match to the upstream
    state, the local composition definition, and the average
    chemical-potential balance,

        PQM[muB(0+), muK(0+), T] = P(0-),
        nK(0+)/nB(0+)            = a(0+),
        muB(0-)                  = muB(0+) + a(0+)*muK(0+).

    Eliminating the first two leaves a scalar residual in a(0+) alone.  That
    residual is monotone, so it is bracketed and solved with brentq rather
    than by a simultaneous 3x3 Newton step, which is seed sensitive: a poor
    seed converges to a spurious point with a large residual instead of
    failing loudly.

    At a(0+) -> 0 the residual reduces identically to delta_muB, so the root
    runs to zero on the stable-neutron-matter boundary and to one where even
    a strangeness-free interface stays favorable.  ``status`` reports which:
    ``"interior"`` for a root in (0,1), ``"saturated"`` when a(0+) = 1 is
    still favorable, and ``"stable"`` when no positive a(0+) is.

    The interface state is evaluated in the static limit, at momentum flux
    P(0-) with jB = 0.  A moving front carries Pi = P(0-) + h(0-)*u(0-)**2,
    an O(u**2) ~ 1e-6 correction for the slow fronts this describes, so the
    bound is not iterated against the velocity it later constrains.
    """
    muB_0minus = float(muB_0minus)
    P_0minus = float(P_0minus)
    T = float(T)
    B_one_forth = float(B_one_forth)
    a_floor = float(a_floor)
    a_ceiling = float(a_ceiling)
    if not (0.0 < a_floor < a_ceiling < 1.0):
        raise RuntimeError("a_0plus bracket must satisfy 0 < a_floor < a_ceiling < 1")

    interface_cache = {}

    def gibbs_residual(a_0plus):
        a_0plus = float(a_0plus)
        muB_0plus, muK_0plus = _solve_interface_0plus_from_local_a_and_Pi(
            a_0plus,
            P_0minus,
            0.0,
            B_one_forth,
            T,
            ms=ms,
            upB=upB,
        )
        interface_cache[a_0plus] = (float(muB_0plus), float(muK_0plus))
        return float(muB_0plus + a_0plus * muK_0plus - muB_0minus)

    residual_ceiling = gibbs_residual(a_ceiling)
    if residual_ceiling <= 0.0:
        muB_0plus, muK_0plus = interface_cache[a_ceiling]
        return {
            "status": "saturated",
            "a_0plus_max": 1.0,
            "muB_0plus": muB_0plus,
            "muK_0plus": muK_0plus,
            "gibbs_residual": residual_ceiling,
        }

    residual_floor = gibbs_residual(a_floor)
    if residual_floor >= 0.0:
        muB_0plus, muK_0plus = interface_cache[a_floor]
        return {
            "status": "stable",
            "a_0plus_max": 0.0,
            "muB_0plus": muB_0plus,
            "muK_0plus": muK_0plus,
            "gibbs_residual": residual_floor,
        }

    root_result = root_scalar(
        gibbs_residual,
        bracket=(a_floor, a_ceiling),
        method="brentq",
        xtol=1.0e-12,
        rtol=1.0e-13,
    )
    if not root_result.converged:
        raise RuntimeError(
            f"a_0plus_max solve did not converge: {root_result.flag}"
        )
    a_0plus_max = float(root_result.root)
    muB_0plus, muK_0plus = interface_cache[a_0plus_max]
    return {
        "status": "interior",
        "a_0plus_max": a_0plus_max,
        "muB_0plus": muB_0plus,
        "muK_0plus": muK_0plus,
        "gibbs_residual": float(
            muB_0plus + a_0plus_max * muK_0plus - muB_0minus
        ),
    }


def analytic_velocity_isothermal(
    T_0minus,
    nB_0minus,
    B_one_forth,
    a_0plus=np.nan,
    *,
    xi=0.0,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    upB=5000,
    delta_muB_tol=1.0e-6,
    momentum_flux_tol=MOMENTUM_FLUX_RATIO_TOLERANCE,
):
    """Return the hydro-consistent analytical isothermal front velocity.

    ``a_0plus`` is the local interface fraction nK(0+)/nB(0+).  The current
    closed-form weak source is derived for massless quarks, for which the
    equilibrated endpoint has nK(inf) = 0.

    Passing ``a_0plus=nan`` (the default) requests the thermodynamic maximum
    instead of a prescribed value: the solver calls :func:`_solve_a_0plus_max`
    and evaluates the speed there, giving the fastest front the interface can
    support at this upstream state.  The resolved value is echoed in
    ``a_0plus``, with ``a_0plus_source`` set to ``"maximum"`` and the ceiling
    diagnostics in ``a_0plus_max``, ``a_0plus_max_status`` and
    ``a_0plus_max_residual``.  Where even a strangeness-free interface stays
    favorable the ceiling is the kinematic 1, at which the speed diverges;
    that returns ``status="composition_ceiling_saturated"`` and no velocity.
    """
    T_0minus = float(T_0minus)
    nB_0minus = float(nB_0minus)
    B_one_forth = float(B_one_forth)
    a_0plus = float(a_0plus)
    xi = float(xi)
    ms = float(ms)
    delta_muB_tol = float(delta_muB_tol)
    try:
        upB_float = float(upB)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("upB must be a positive finite integer") from exc

    if (not np.isfinite(T_0minus)) or T_0minus < 0.0:
        raise RuntimeError("T_0minus must be finite and non-negative")
    if (not np.isfinite(nB_0minus)) or nB_0minus <= 0.0:
        raise RuntimeError("nB_0minus must be positive and finite")
    if (not np.isfinite(B_one_forth)) or B_one_forth <= 0.0:
        raise RuntimeError("B_one_forth must be positive and finite")
    a_0plus_is_auto = bool(np.isnan(a_0plus))
    if (not a_0plus_is_auto) and (
        (not np.isfinite(a_0plus)) or not (0.0 <= a_0plus < 1.0)
    ):
        raise RuntimeError(
            "a_0plus must satisfy 0 <= a_0plus < 1, or be NaN to request the "
            "thermodynamic maximum"
        )
    if (not np.isfinite(xi)) or not (-1.0 < xi < 1.0):
        raise RuntimeError("xi must satisfy -1 < xi < 1")
    if (not np.isfinite(ms)) or abs(ms) > 1.0e-12:
        raise RuntimeError(
            "analytic_velocity_isothermal currently requires ms=0 because its "
            "weak source is written for nK(inf)=0"
        )
    if str(NM_type) != "PNM":
        raise RuntimeError(
            "analytic_velocity_isothermal currently requires NM_type='PNM'"
        )
    if (not np.isfinite(delta_muB_tol)) or delta_muB_tol < 0.0:
        raise RuntimeError("delta_muB_tol must be finite and non-negative")
    if (
        (not np.isfinite(upB_float))
        or upB_float <= 0.0
        or not upB_float.is_integer()
    ):
        raise RuntimeError("upB must be a positive finite integer")
    upB = int(upB_float)

    muB_0minus = float(
        muB_from_nB_physical(
            nB_0minus,
            T_0minus,
            param=param,
            NM_type=NM_type,
            auto_expand=True,
        )
    )
    nuclear_state = _analytic_nuclear_state(
        muB_0minus,
        T_0minus,
        param=param,
        NM_type=NM_type,
    )
    recovered_nB_0minus = float(nuclear_state["nB_0minus"])
    if not np.isclose(
        recovered_nB_0minus,
        nB_0minus,
        rtol=2.0e-6,
        atol=0.0,
    ):
        raise RuntimeError(
            "muB_from_nB_physical did not reproduce nB_0minus on the validated branch"
        )
    nuclear_state = dict(nuclear_state)
    nuclear_state["muB_0minus"] = muB_0minus
    nuclear_state["T_0minus"] = T_0minus

    P_0minus = float(nuclear_state["P_0minus"])
    muB_qm_candidate = float(
        _solve_muB_inf_at_muK0_for_given_Pi(
            P_0minus,
            0.0,
            B_one_forth,
            T_0minus,
            ms=ms,
            upB=upB,
        )
    )
    nB_qm_candidate = float(
        nB_QM(
            muB_qm_candidate,
            0.0,
            B_one_forth,
            T_0minus,
            ms=ms,
            upB=upB,
        )
    )
    nK_qm_candidate = float(
        nK_QM(
            muB_qm_candidate,
            0.0,
            B_one_forth,
            T_0minus,
            ms=ms,
            upB=upB,
        )
    )
    if (not np.isfinite(nB_qm_candidate)) or nB_qm_candidate <= 0.0:
        raise RuntimeError("Pressure-matched quark candidate has non-positive nB")
    if not np.isfinite(nK_qm_candidate):
        raise RuntimeError("Pressure-matched quark candidate has non-finite nK")

    delta_muB = float(muB_qm_candidate - muB_0minus)
    result = {
        "success": True,
        "status": "",
        "message": "",
        "phase_region": "",
        "front_exists": False,
        "u_0minus": np.nan,
        "u_0minus_squared": np.nan,
        "u_0minus_formula_squared": np.nan,
        "analytic_velocity_residual": np.nan,
        "jB": np.nan,
        "T_0minus": T_0minus,
        "T_0plus": np.nan,
        "T_inf": np.nan,
        "nB_0minus": nB_0minus,
        "muB_0minus": muB_0minus,
        "P_0minus": P_0minus,
        "e_0minus": float(nuclear_state["e_0minus"]),
        "h_0minus": float(nuclear_state["h_0minus"]),
        "h_over_nB_0minus": float(nuclear_state["h_over_nB_0minus"]),
        "B_one_forth": B_one_forth,
        "a_0minus": 1.0,
        "a_0plus": a_0plus,
        "a_0plus_source": "maximum" if a_0plus_is_auto else "input",
        "a_0plus_max": np.nan,
        "a_0plus_max_status": "",
        "a_0plus_max_residual": np.nan,
        "xi": xi,
        "ms": ms,
        "upB": int(upB),
        "delta_muB": delta_muB,
        "delta_muB_tol": delta_muB_tol,
        "muB_qm_candidate": muB_qm_candidate,
        "muK_qm_candidate": 0.0,
        "nB_qm_candidate": nB_qm_candidate,
        "nK_qm_candidate": nK_qm_candidate,
        "P_qm_candidate": P_0minus,
        "muB_0plus": np.nan,
        "muK_0plus": np.nan,
        "nB_0plus": np.nan,
        "nK_0plus": np.nan,
        "u_0plus": np.nan,
        "muB_inf": np.nan,
        "muK_inf": np.nan,
        "nB_inf": np.nan,
        "nK_inf": np.nan,
        "u_inf": np.nan,
        "Pi": np.nan,
        "lambda_n": np.nan,
        "lambda_n_squared": np.nan,
        "D": np.nan,
        "D_K": np.nan,
        "eta": np.nan,
        "gamma": np.nan,
        "gamma_K": np.nan,
        "tau": np.nan,
        "mu_q": np.nan,
        "qD": np.nan,
        "alpha_s": np.nan,
        "I2": np.nan,
        "analytic_denominator": np.nan,
        "composition_residual": np.nan,
        "momentum_flux_inf_residual": np.nan,
        "momentum_flux_0plus_residual": np.nan,
        "slow_front_consistent": False,
        "u_0minus_trial_limit": np.nan,
        "u_0minus_formula_squared_at_limit": np.nan,
        "composition_definition": "a_0plus_equals_nK_0plus_over_nB_0plus",
        "density_ratio_definition": "lambda_n_equals_nB_0minus_over_nB_inf",
        "velocity_method": "analytic_isothermal_closed_form_I2",
        "analytic_formula_variant": "isothermal_piecewise_constant_lambda_n",
    }

    if delta_muB > delta_muB_tol:
        result.update(
            {
                "status": "stable_neutron_matter",
                "message": (
                    "Neutron matter is thermodynamically stable at common P and T; "
                    "no forward conversion front exists"
                ),
                "phase_region": "stable_neutron_matter",
                "u_0minus": 0.0,
                "u_0minus_squared": 0.0,
                "jB": 0.0,
            }
        )
        return result

    if abs(delta_muB) <= delta_muB_tol:
        result.update(
            {
                "status": "isothermal_coexistence",
                "message": "Static isothermal coexistence at common P, T, and muB",
                "phase_region": "isothermal_coexistence",
                "u_0minus": 0.0,
                "u_0minus_squared": 0.0,
                "jB": 0.0,
                "T_0plus": T_0minus,
                "T_inf": T_0minus,
                "muB_0plus": muB_qm_candidate,
                "muK_0plus": 0.0,
                "nB_0plus": nB_qm_candidate,
                "nK_0plus": nK_qm_candidate,
                "u_0plus": 0.0,
                "muB_inf": muB_qm_candidate,
                "muK_inf": 0.0,
                "nB_inf": nB_qm_candidate,
                "nK_inf": nK_qm_candidate,
                "u_inf": 0.0,
                "Pi": P_0minus,
                "lambda_n": float(nB_0minus / nB_qm_candidate),
            }
        )
        return result

    result["phase_region"] = "quark_matter_favored"

    if a_0plus_is_auto:
        ceiling = _solve_a_0plus_max(
            muB_0minus,
            P_0minus,
            T_0minus,
            B_one_forth,
            ms=ms,
            upB=upB,
        )
        result.update(
            {
                "a_0plus_max": float(ceiling["a_0plus_max"]),
                "a_0plus_max_status": str(ceiling["status"]),
                "a_0plus_max_residual": float(ceiling["gibbs_residual"]),
            }
        )
        if ceiling["status"] == "saturated":
            result.update(
                {
                    "success": False,
                    "status": "composition_ceiling_saturated",
                    "message": (
                        "The strangeness-free interface is still favored, so "
                        "a_0plus is capped only by its kinematic ceiling of 1, "
                        "where the analytical speed diverges"
                    ),
                }
            )
            return result
        a_0plus = float(ceiling["a_0plus_max"])
        result["a_0plus"] = a_0plus

    if a_0plus == 0.0:
        result.update(
            {
                "status": "zero_interface_composition",
                "message": "The analytical reaction-diffusion speed vanishes at a_0plus=0",
                "u_0minus": 0.0,
                "u_0minus_squared": 0.0,
                "u_0minus_formula_squared": 0.0,
                "analytic_velocity_residual": 0.0,
                "jB": 0.0,
            }
        )
        return result

    if T_0minus == 0.0:
        result.update(
            {
                "success": False,
                "status": "zero_temperature_transport_invalid",
                "message": (
                    "The fixed-composition T -> 0+ formula diverges and the local "
                    "diffusion model is invalid at exactly T_0minus=0"
                ),
                "u_0minus": np.nan,
                "u_0minus_squared": np.nan,
                "jB": np.nan,
            }
        )
        return result

    evaluated_states = []

    def evaluate_log_u(theta):
        theta = float(theta)
        u_0minus = float(np.exp(theta))
        if (not np.isfinite(u_0minus)) or u_0minus <= 0.0:
            raise RuntimeError("Trial u_0minus must be positive and finite")
        jB = float(nB_0minus * u_0minus)
        Pi = float(P_0minus + float(nuclear_state["h_0minus"]) * u_0minus**2)

        if evaluated_states:
            _, nearest_state = min(
                evaluated_states,
                key=lambda item: abs(float(item[0]) - theta),
            )
            endpoint_initial_guess = float(nearest_state["muB_inf"])
            interface_initial_guess = (
                float(nearest_state["muB_0plus"]),
                float(nearest_state["muK_0plus"]),
            )
        else:
            endpoint_initial_guess = muB_qm_candidate
            interface_initial_guess = (
                muB_qm_candidate,
                _branch_muK_seed(a_0plus),
            )

        muB_inf = float(
            _solve_muB_inf_at_muK0_for_given_Pi_ms(
                Pi,
                jB,
                B_one_forth,
                T_0minus,
                ms=ms,
                upB=upB,
                initial_guess=endpoint_initial_guess,
            )
        )
        nB_inf = float(
            nB_QM(
                muB_inf,
                0.0,
                B_one_forth,
                T_0minus,
                ms=ms,
                upB=upB,
            )
        )
        nK_inf = float(
            nK_QM(
                muB_inf,
                0.0,
                B_one_forth,
                T_0minus,
                ms=ms,
                upB=upB,
            )
        )
        if (not np.isfinite(nB_inf)) or nB_inf <= 0.0:
            raise RuntimeError("Equilibrated quark endpoint has non-positive nB_inf")
        if not np.isfinite(nK_inf):
            raise RuntimeError("Equilibrated quark endpoint has non-finite nK_inf")

        muB_0plus, muK_0plus = _solve_interface_0plus_from_local_a_and_Pi(
            a_0plus,
            Pi,
            jB,
            B_one_forth,
            T_0minus,
            ms=ms,
            upB=upB,
            initial_guess=interface_initial_guess,
        )
        nB_0plus = float(
            nB_QM(
                muB_0plus,
                muK_0plus,
                B_one_forth,
                T_0minus,
                ms=ms,
                upB=upB,
            )
        )
        nK_0plus = float(
            nK_QM(
                muB_0plus,
                muK_0plus,
                B_one_forth,
                T_0minus,
                ms=ms,
                upB=upB,
            )
        )
        if (not np.isfinite(nB_0plus)) or nB_0plus <= 0.0:
            raise RuntimeError("x = 0+ quark state has non-positive nB_0plus")
        if not np.isfinite(nK_0plus):
            raise RuntimeError("x = 0+ quark state has non-finite nK_0plus")

        composition_residual = float(nK_0plus / nB_0plus - a_0plus)
        micro = _microphysics_from_quark_state_isothermal_baseline(
            muB_0plus,
            T_0minus,
        )
        D_K = float(micro["D"])
        eta = float(micro["eta"])
        gamma_K = float(micro["gamma"])
        lambda_n = float(nB_0minus / nB_inf)
        if (not np.isfinite(lambda_n)) or lambda_n <= 0.0:
            raise RuntimeError("lambda_n must be positive and finite")

        denominator = float(
            lambda_n**2
            * (1.0 - a_0plus)
            * (1.0 + xi * a_0plus)
        )
        if (not np.isfinite(denominator)) or denominator <= 0.0:
            raise RuntimeError("Analytical isothermal denominator is non-physical")
        I2 = float(
            D_K
            * gamma_K
            * a_0plus**2
            * (a_0plus**2 + 2.0 * eta)
            / 4.0
        )
        u_0minus_formula_squared = float(2.0 * I2 / denominator)
        if (
            (not np.isfinite(I2))
            or I2 < 0.0
            or (not np.isfinite(u_0minus_formula_squared))
            or u_0minus_formula_squared < 0.0
        ):
            raise RuntimeError("Analytical isothermal formula returned a non-physical value")

        momentum_flux_inf_residual = float(
            _Pi_QM_state(
                muB_inf,
                0.0,
                B_one_forth,
                T_0minus,
                jB,
                ms=ms,
                upB=upB,
            )
            - Pi
        )
        momentum_flux_0plus_residual = float(
            _Pi_QM_state(
                muB_0plus,
                muK_0plus,
                B_one_forth,
                T_0minus,
                jB,
                ms=ms,
                upB=upB,
            )
            - Pi
        )
        residual = float(u_0minus_formula_squared - u_0minus**2)
        data = {
            "u_0minus": u_0minus,
            "u_0minus_formula_squared": u_0minus_formula_squared,
            "jB": jB,
            "Pi": Pi,
            "muB_inf": muB_inf,
            "nB_inf": nB_inf,
            "nK_inf": nK_inf,
            "u_inf": float(jB / nB_inf),
            "muB_0plus": float(muB_0plus),
            "muK_0plus": float(muK_0plus),
            "nB_0plus": nB_0plus,
            "nK_0plus": nK_0plus,
            "u_0plus": float(jB / nB_0plus),
            "lambda_n": lambda_n,
            "D_K": D_K,
            "eta": eta,
            "gamma_K": gamma_K,
            "tau": float(micro["tau"]),
            "mu_q": float(micro["muQ"]),
            "qD": float(micro["qD"]),
            "alpha_s": float(micro["alpha_s"]),
            "I2": I2,
            "analytic_denominator": denominator,
            "composition_residual": composition_residual,
            "momentum_flux_inf_residual": momentum_flux_inf_residual,
            "momentum_flux_0plus_residual": momentum_flux_0plus_residual,
            "residual": residual,
        }
        evaluated_states.append((theta, data))
        return residual, data

    try:
        final_data = _solve_analytic_isothermal_log_root(
            evaluate_log_u,
            max_u_0minus=1.0,
        )
    except _AnalyticIsothermalSlowFrontInvalid as exc:
        limit_data = exc.limit_data
        result.update(
            {
                "success": False,
                "status": "slow_front_approximation_invalid",
                "message": str(exc),
                "phase_region": "quark_matter_favored",
                "front_exists": False,
                "u_0minus": np.nan,
                "u_0minus_squared": np.nan,
                "u_0minus_formula_squared": np.nan,
                "analytic_velocity_residual": np.nan,
                "jB": np.nan,
                "u_0minus_trial_limit": float(limit_data["u_0minus"]),
                "u_0minus_formula_squared_at_limit": float(
                    limit_data["u_0minus_formula_squared"]
                ),
                "slow_front_consistent": False,
            }
        )
        return result

    u_0minus = float(final_data["u_0minus"])
    u_0minus_squared = float(u_0minus**2)
    analytic_velocity_residual = float(
        final_data["u_0minus_formula_squared"] - u_0minus_squared
    )
    formula_scale = max(
        abs(float(final_data["u_0minus_formula_squared"])),
        abs(u_0minus_squared),
        1.0e-30,
    )
    Pi_scale = max(abs(float(final_data["Pi"])), 1.0)
    if abs(analytic_velocity_residual) > 1.0e-8 * formula_scale:
        raise RuntimeError(
            "Analytical isothermal velocity residual is too large: "
            f"{analytic_velocity_residual:.6e}"
        )
    if abs(float(final_data["momentum_flux_inf_residual"])) > 1.0e-8 * Pi_scale:
        raise RuntimeError("Downstream momentum-flux residual is too large")
    if abs(float(final_data["momentum_flux_0plus_residual"])) > 1.0e-8 * Pi_scale:
        raise RuntimeError("Interface momentum-flux residual is too large")
    if abs(float(final_data["composition_residual"])) > 1.0e-8:
        raise RuntimeError("Interface local-composition residual is too large")

    result.update(
        {
            "success": True,
            "status": "moving_front",
            "message": "Hydro-consistent analytical isothermal velocity evaluated",
            "phase_region": "quark_matter_favored",
            "front_exists": True,
            "u_0minus": u_0minus,
            "u_0minus_squared": u_0minus_squared,
            "u_0minus_formula_squared": float(
                final_data["u_0minus_formula_squared"]
            ),
            "analytic_velocity_residual": analytic_velocity_residual,
            "jB": float(final_data["jB"]),
            "T_0plus": T_0minus,
            "T_inf": T_0minus,
            "muB_0plus": float(final_data["muB_0plus"]),
            "muK_0plus": float(final_data["muK_0plus"]),
            "nB_0plus": float(final_data["nB_0plus"]),
            "nK_0plus": float(final_data["nK_0plus"]),
            "u_0plus": float(final_data["u_0plus"]),
            "muB_inf": float(final_data["muB_inf"]),
            "muK_inf": 0.0,
            "nB_inf": float(final_data["nB_inf"]),
            "nK_inf": float(final_data["nK_inf"]),
            "u_inf": float(final_data["u_inf"]),
            "Pi": float(final_data["Pi"]),
            "lambda_n": float(final_data["lambda_n"]),
            "lambda_n_squared": float(final_data["lambda_n"] ** 2),
            "D": float(final_data["D_K"]),
            "D_K": float(final_data["D_K"]),
            "eta": float(final_data["eta"]),
            "gamma": float(final_data["gamma_K"]),
            "gamma_K": float(final_data["gamma_K"]),
            "tau": float(final_data["tau"]),
            "mu_q": float(final_data["mu_q"]),
            "qD": float(final_data["qD"]),
            "alpha_s": float(final_data["alpha_s"]),
            "I2": float(final_data["I2"]),
            "analytic_denominator": float(final_data["analytic_denominator"]),
            "composition_residual": float(final_data["composition_residual"]),
            "momentum_flux_inf_residual": float(
                final_data["momentum_flux_inf_residual"]
            ),
            "momentum_flux_0plus_residual": float(
                final_data["momentum_flux_0plus_residual"]
            ),
            **_momentum_flux_diagnostics(
                P_0minus,
                float(nuclear_state["h_0minus"]),
                u_0minus,
            ),
            "slow_front_consistent": bool(u_0minus < 1.0),
        }
    )
    if result["momentum_flux_ratio"] > float(momentum_flux_tol):
        result.update(
            {
                "success": False,
                "status": "static_isobar_approximation_invalid",
                "message": (
                    "The momentum-flux term is "
                    f"{result['momentum_flux_ratio']:.3e} of the upstream "
                    "pressure, above momentum_flux_tol="
                    f"{float(momentum_flux_tol):.3e}; the static-isobar "
                    "interface reduction is not trustworthy here"
                ),
                "front_exists": False,
            }
        )
    return result


def analytic_velocity_bound(
    muB_0minus,
    T_0minus,
    B_one_forth,
    *,
    T_0plus,
    xi=0.0,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    upB=5000,
    initial_guess=None,
):
    """Return the full analytical velocity using a prescribed T(0+).

    The interface follows the fixed-muB isobar used by the closed-form model.
    Its composition is evaluated from the exact EOS definition a = nK/nB at
    T(0+). The closed-form I2 bracket retains the quadratic temperature relation
    used in its derivation; that approximation is not used to determine a(0+).
    """
    result = _solve_analytic_velocity_bound(
        muB_0minus,
        T_0minus,
        B_one_forth,
        T_0plus,
        xi,
        "closed_form",
        "fixed_T_0plus",
        ms=ms,
        param=param,
        NM_type=NM_type,
        upB=upB,
        initial_guess=initial_guess,
    )
    result["velocity_method"] = "full_analytic_closed_form_I2"
    result["analytic_formula_variant"] = "piecewise_constant_fixed_T_0plus"
    return result


def semi_analytic_velocity_bound(
    muB_0minus,
    T_0minus,
    B_one_forth,
    *,
    T_0plus,
    xi=0.0,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    upB=5000,
    initial_guess=None,
):
    """Return the semi-analytical velocity using a prescribed T(0+).

    The conversion layer follows the conserved (P, h/nB) trajectory. The exact
    EOS supplies a(0+) = nK/nB at the requested temperature, and I2 is integrated
    numerically with the full diffusion coefficient and exact weak rate.
    """
    result = _solve_analytic_velocity_bound(
        muB_0minus,
        T_0minus,
        B_one_forth,
        T_0plus,
        xi,
        "numerical_I2",
        "fixed_T_0plus",
        ms=ms,
        param=param,
        NM_type=NM_type,
        upB=upB,
        initial_guess=initial_guess,
    )
    result["velocity_method"] = "semi_analytic_numerical_I2"
    result["analytic_formula_variant"] = "conserved_layer_fixed_T_0plus"
    return result


def analytic_velocity_bound_lte(
    muB_0minus,
    T_0minus,
    B_one_forth,
    *,
    xi=0.0,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    upB=5000,
    initial_guess=None,
):
    """Return the preserved LTE-limited closed-form analytical velocity."""
    result = _solve_analytic_velocity_bound(
        muB_0minus,
        T_0minus,
        B_one_forth,
        None,
        xi,
        "closed_form",
        "LTE",
        ms=ms,
        param=param,
        NM_type=NM_type,
        upB=upB,
        initial_guess=initial_guess,
    )
    result["velocity_method"] = "full_analytic_LTE"
    result["analytic_formula_variant"] = "piecewise_constant_lambda_n_LTE"
    return result


def Pi_NM(mu_B, Temp, j_B):
    """
    Return the nuclear-matter momentum flux Pi = h*u^2 + P.
    """
    nB = nB_NM(mu_B, Temp)
    if nB <= 0:
        return np.nan
    u_0minus = j_B / nB
    return hNM(mu_B, Temp) * u_0minus * u_0minus + PNM(mu_B, Temp)
    
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


def u_0minus(T, nB_0minus, Delta_n, B_one_forth, param=para.paraQMCRMF3, ms=0, upB=5000, return_more=False):
    """
    Compute the phase-boundary flux data from endpoint states.

    Parameters
    ----------
    T : float
        Temperature.
    nB_0minus : float
        Upstream nuclear baryon density n_B(0^-).
    Delta_n : float
        Density jump defined by n_B(infty) = nB_0minus + Delta_n.
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
    - The returned values are Pi, jB, u_0minus, where u_0minus is in natural units.
    """
    if nB_0minus <= 0.0:
        raise RuntimeError("nB_0minus must be positive")

    nB_inf = nB_0minus + Delta_n
    if nB_inf <= 0.0:
        raise RuntimeError("nB_inf = nB_0minus + Delta_n must be positive")

    # Upstream nuclear endpoint from (T, nB_0minus).
    P_0minus = float(PNM_n(nB_0minus, T, param=param, NM_type="PNM"))
    h_0minus = float(P_0minus + edensNM_n(nB_0minus, T, param=param))

    # Fully equilibrated QM endpoint at muK = 0 and nB_inf.
    muB_inf = _solve_muB_inf_at_muK0_from_nB(nB_inf, B_one_forth, T, ms=ms, upB=upB)
    P_inf = float(PQM(muB_inf, 0.0, B_one_forth, T, ms=ms, upB=upB))
    h_inf = float(
        P_inf
        + edensQM(muB_inf, 0.0, B_one_forth, T, ms=ms, include_em=False, upB=upB)
    )

    # Momentum-flux matching fixes jB from the two endpoint states.
    term_0minus = h_0minus / (nB_0minus * nB_0minus)
    term_inf = h_inf / (nB_inf * nB_inf)
    denom = term_0minus - term_inf
    denom_scale = max(abs(term_0minus), abs(term_inf), 1.0)
    if abs(denom) <= 1.0e-12 * denom_scale:
        raise RuntimeError("Endpoint momentum-flux denominator is too close to zero")

    jB_sq = (P_inf - P_0minus) / denom
    if (not np.isfinite(jB_sq)) or jB_sq <= 0.0:
        raise RuntimeError(f"Endpoint matching gives non-physical jB^2={jB_sq}")

    jB = float(np.sqrt(jB_sq))
    u_0minus = float(jB / nB_0minus)
    Pi = float(h_0minus * u_0minus * u_0minus + P_0minus)

    # Numerical consistency check on the equilibrated QM endpoint.
    u_inf = float(jB / nB_inf)
    Pi_inf = float(h_inf * u_inf * u_inf + P_inf)
    Pi_scale = max(abs(Pi), abs(Pi_inf), 1.0)
    if not np.isclose(Pi, Pi_inf, rtol=1.0e-8, atol=1.0e-10 * Pi_scale):
        raise RuntimeError(
            f"Endpoint momentum-flux mismatch: Pi_0minus={Pi:.12g}, Pi_inf={Pi_inf:.12g}"
        )

    if return_more:
        return {
            "Pi": Pi,
            "jB": jB,
            "u_0minus": u_0minus,
            "nB_inf": float(nB_inf),
            "P_0minus": P_0minus,
            "h_0minus": h_0minus,
            "P_inf": P_inf,
            "h_inf": h_inf,
            "muB_inf": muB_inf,
            "u_inf": u_inf,
        }

    return Pi, jB, u_0minus


# Shared quark-state helpers
def _Pi_QM_state(muB, muK, B_one_forth, T, jB, ms=0.0, upB=5000, relativistic=False):
    """
    Momentum flux Pi = h*u^2 + P for a quark state at fixed (muB, muK).

    ``relativistic=True`` (opt in, off by default) routes the same flux through
    :func:`_relativistic_flux_pair`: the 3-velocity is recovered from
    jB = nB*gamma*v by the closed-form inverse v = x/sqrt(1 + x**2) with
    x = jB/nB, and the flux is P + h*gamma**2*v**2.

    That path is an *identity*, not a correction.  gamma**2 = 1 + x**2 and
    v**2 = x**2/(1 + x**2), so gamma**2*v**2 = x**2 exactly, and the default
    branch below already returns P + h*(jB/nB)**2.  In other words the local
    variable ``u = jB/nB`` is the proper velocity gamma*v, not the 3-velocity,
    and this closure has always been exactly relativistic at fixed (jB, nB).
    The flag is kept as an executable check of that equivalence; it moves the
    answer only by floating-point rounding (order 1e-16 relative).  Only
    :func:`solve_front_isothermal` passes it; every other caller keeps the
    default branch.
    """
    nB = nB_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB)
    if nB <= 0.0:
        return np.nan
    P = PQM(muB, muK, B_one_forth, T, ms=ms, upB=upB)
    h = P + edensQM(muB, muK, B_one_forth, T, ms=ms, include_em=False, upB=upB)
    if relativistic:
        x = float(jB / nB)
        if not np.isfinite(x):
            return np.nan
        # Pi is even in u, so the magnitude is all the inversion needs.
        u = float(abs(x) / np.sqrt(1.0 + x * x))
        return float(_relativistic_flux_pair(nB, h, P, u)[1])
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


def _solve_muB_inf_at_muK0_for_given_Pi(Pi, jB, B_one_forth, T, ms=0.0, upB=5000, stats=None, relativistic=False):
    """
    Solve for the equilibrated QM endpoint muB_inf at muK=0 for a given Pi.
    """
    return _solve_muB_inf_at_muK0_for_given_Pi_ms(
        Pi,
        jB,
        B_one_forth,
        T,
        ms=ms,
        upB=upB,
        stats=stats,
        relativistic=relativistic,
    )


def _solve_muB_inf_at_muK0_for_given_Pi_ms(
    Pi,
    jB,
    B_one_forth,
    T,
    ms=0.0,
    upB=5000,
    stats=None,
    stats_key="q_root_calls",
    initial_guess=None,
    relativistic=False,
):
    """
    Solve for the equilibrated QM endpoint muB_inf at muK=0 using the ms-aware
    quark momentum-flux helper throughout.
    """
    if stats is not None:
        stats[stats_key] = stats.get(stats_key, 0) + 1

    def equation(muB_in):
        muB = float(np.atleast_1d(muB_in)[0])
        return float(
            _Pi_QM_state(
                muB, 0.0, B_one_forth, T, jB, ms=ms, upB=upB, relativistic=relativistic
            )
            - Pi
        )

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
            muB_inf = float(np.atleast_1d(muB_arr)[0])
            Pi_residual = float(
                _Pi_QM_state(
                    muB_inf,
                    0.0,
                    B_one_forth,
                    T,
                    jB,
                    ms=ms,
                    upB=upB,
                    relativistic=relativistic,
                )
                - Pi
            )
            nB = float(nB_QM(muB_inf, 0.0, B_one_forth, T, ms=ms, upB=upB))
            if (not np.isfinite(muB_inf)) or (not np.isfinite(Pi_residual)) or (not np.isfinite(nB)) or nB <= 0.0:
                last_error = "Solved muB_inf lies on a non-physical density branch"
                continue
            metric = abs(Pi_residual)
            if metric < best_metric:
                best_metric = metric
                best_muB = muB_inf
            if ier == 1 and metric <= tol:
                return muB_inf
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


def _quark_state_residual(muB, muK, a_target, Pi, jB, nB_inf, nK_inf, B_one_forth, T, ms=0.0, upB=5000, relativistic=False):
    """
    Return the local quark-state residuals at fixed (a_target, Pi, jB).
    """
    return np.array(
        [
            _Pi_QM_state(
                muB, muK, B_one_forth, T, jB, ms=ms, upB=upB, relativistic=relativistic
            )
            - Pi,
            (nK_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB) - nK_inf) / nB_inf - a_target,
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


def _solve_quark_state_once_from_guess(a_target, Pi, jB, nB_inf, nK_inf, B_one_forth, T, ms=0.0, upB=5000, initial_guess=None, stats=None, stats_key="quark_state_root_calls"):
    """
    Try one local quark-state root solve from a single continuation guess.
    This is the fast path used during IVP integration.
    """
    if initial_guess is None:
        raise RuntimeError("initial_guess is required for single-guess quark-state solve")
    if nB_inf <= 0.0:
        raise RuntimeError("nB_inf must be positive when solving for a quark state")

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
            nB_inf,
            nK_inf,
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
        nB_inf,
        nK_inf,
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


def _solve_interface_0plus_from_a_0plus_and_Pi(a_0plus, Pi, jB, nB_inf, nK_inf, B_one_forth, T, ms=0.0, upB=5000, initial_guess=None, stats=None, stats_key="interface_0plus_root_calls"):
    """
    Solve for the interface state at x = 0+ from Pi and a_0plus.
    """
    if nB_inf <= 0.0:
        raise RuntimeError("nB_inf must be positive when solving the x = 0+ state")

    guesses = []
    muK_seed = _branch_muK_seed(a_0plus)
    muK_seed_strong = float(max(muK_seed, 400.0 * abs(float(a_0plus))))
    if initial_guess is not None:
        guess0 = np.asarray(initial_guess, dtype=float)
        guesses.append(guess0)
        muB_seed = float(guess0[0])
    else:
        muB_seed = 1200.0

    guesses.append(np.array([muB_seed, muK_seed], dtype=float))
    guesses.append(np.array([1200.0, muK_seed], dtype=float))
    guesses.append(np.array([1500.0, max(muK_seed, 100.0 * abs(float(a_0plus)))], dtype=float))
    guesses.append(np.array([muB_seed, muK_seed_strong], dtype=float))
    guesses.append(np.array([1500.0, muK_seed_strong], dtype=float))

    def equations(vec):
        muB, muK = map(float, vec)
        return _quark_state_residual(
            muB,
            muK,
            a_0plus,
            Pi,
            jB,
            nB_inf,
            nK_inf,
            B_one_forth,
            T,
            ms=ms,
            upB=upB,
        )

    best_message = "x = 0+ state solve did not converge"
    candidates = []
    candidate_tol = 1.0e-8
    nonneg_tol = 1.0e-8
    for guess in guesses:
        if stats is not None:
            stats[stats_key] = stats.get(stats_key, 0) + 1
        sol = root(equations, guess, method="hybr", options={"maxfev": 6000, "xtol": 1.0e-10})
        if sol.success and np.all(np.isfinite(sol.x)):
            muB_0plus = float(sol.x[0])
            muK_0plus = float(sol.x[1])
            residual = _quark_state_residual(
                muB_0plus,
                max(muK_0plus, 0.0),
                a_0plus,
                Pi,
                jB,
                nB_inf,
                nK_inf,
                B_one_forth,
                T,
                ms=ms,
                upB=upB,
            )
            nB_0plus = nB_QM(muB_0plus, muK_0plus, B_one_forth, T, ms=ms, upB=upB)
            if nB_0plus > 0.0 and muK_0plus >= -nonneg_tol and _quark_state_residual_ok(residual, Pi, a_0plus):
                if muK_0plus < 0.0:
                    muK_0plus = 0.0
                is_new = True
                for cand in candidates:
                    if (
                        abs(muB_0plus - cand["muB"]) <= candidate_tol * max(1.0, abs(cand["muB"]))
                        and abs(muK_0plus - cand["muK"]) <= candidate_tol * max(1.0, abs(cand["muK"]), 1.0)
                    ):
                        is_new = False
                        break
                if is_new:
                    candidates.append({"muB": muB_0plus, "muK": muK_0plus})
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

    raise RuntimeError(f"x = 0+ state solve failed: {best_message}")


def _solve_local_quark_state_from_a_and_Pi(a, Pi, jB, nB_inf, nK_inf, B_one_forth, T, ms=0.0, upB=5000, initial_guess=None, stats=None):
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
                nB_inf,
                nK_inf,
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

    muB, muK = _solve_interface_0plus_from_a_0plus_and_Pi(
        a,
        Pi,
        jB,
        nB_inf,
        nK_inf,
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


def _microphysics_at_0plus_isothermal_baseline(muB_0plus, T):
    """
    Isothermal BVP microphysics that matches the baseline steady-front solver.
    """
    return _microphysics_from_quark_state_isothermal_baseline(muB_0plus, T)


def _isothermal_upstream_nuclear_state(T, nB_0minus, param, NM_type):
    """Return fixed-T upstream thermodynamics and its local K fraction."""
    T = float(T)
    nB_0minus = float(nB_0minus)
    NM_type = str(NM_type)

    if NM_type == "PNM":
        P_0minus = float(PNM_n(nB_0minus, T, param=param, NM_type=NM_type))
        e_0minus = float(edensNM_n(nB_0minus, T, param=param))
        proton_fraction = 0.0
    elif NM_type in ("SYM", "Beta_eq"):
        if NM_type == "SYM":
            rmf_state = RMFsolveSYM(
                nB_0minus,
                T,
                param,
                sigma_init=30,
                w0_init=20,
                mub_init=990,
                verb=False,
            )
            electrons = False
            proton_fraction = 0.5
        else:
            rmf_state = RMFsolve(
                nB_0minus,
                T,
                param,
                sigma_init=30,
                w0_init=20,
                r03_init=-3,
                mub_init=990,
                mu_e_init=50,
                verb=False,
            )
            electrons = True
            recovered_nB, species = baryon_density_RMF(
                rmf_state,
                return_species=True,
            )
            if not np.isclose(
                float(recovered_nB),
                nB_0minus,
                rtol=2.0e-6,
                atol=0.0,
            ):
                raise RuntimeError(
                    "Beta-equilibrated RMF state did not reproduce nB_0minus"
                )
            proton_fraction = float(species["n_p"] / recovered_nB)

        P_0minus = float(
            pressure_RMF(
                rmf_state,
                electrons=electrons,
                neutrinos=False,
            )
        )
        e_0minus = float(
            edens_RMF(
                rmf_state,
                electrons=electrons,
                neutrinos=False,
            )[1]
        )
    else:
        raise RuntimeError(
            "NM_type must be one of 'PNM', 'SYM', or 'Beta_eq'"
        )

    a_0minus = float(1.0 - 0.5 * proton_fraction)
    values = np.array(
        [P_0minus, e_0minus, proton_fraction, a_0minus],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or e_0minus <= 0.0:
        raise RuntimeError("Upstream isothermal nuclear state is non-physical")
    if not (0.0 <= proton_fraction <= 1.0 and 0.0 < a_0minus <= 1.0):
        raise RuntimeError("Upstream nuclear composition is non-physical")
    return {
        "P_0minus": P_0minus,
        "e_0minus": e_0minus,
        "h_0minus": float(P_0minus + e_0minus),
        "nB_0minus": nB_0minus,
        "proton_fraction_0minus": proton_fraction,
        "a_0minus": a_0minus,
        "nK_0minus": float(a_0minus * nB_0minus),
    }


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


def _default_energy_jB_guess(nB_0minus):
    """Return the shared default shooting seed for energy-front solvers."""
    nB_0minus = float(nB_0minus)
    if (not np.isfinite(nB_0minus)) or nB_0minus <= 0.0:
        raise RuntimeError("The default energy-front jB seed requires positive finite nB_0minus")
    return float(max(1.0e-12, 1.0e-8 * nB_0minus))


def _fixed_T_0plus_E_residual(muB, muK, T_0plus, E_target, Pi, jB, B_one_forth, ms=0.0, upB=5000):
    """
    Residual for the fixed-T_0plus interface solve. The unknowns are
    (muB_0plus, muK_0plus); the temperature is prescribed.
    """
    if E_target <= 0.0:
        raise RuntimeError("Fixed-T_0plus interface closure requires E = h*u*r_gamma > 0")
    T_0plus = float(T_0plus)
    if (not np.isfinite(T_0plus)) or T_0plus < 0.0:
        raise RuntimeError("Fixed-T_0plus interface closure requires T_0plus >= 0")
    try:
        thermo = _quark_thermo_state(
            muB,
            muK,
            B_one_forth,
            T_0plus,
            jB,
            ms=ms,
            upB=upB,
            allow_zero_temperature=True,
        )
    except Exception:
        return np.array([1.0e12, 1.0e12], dtype=float)
    E_loc = float(thermo["h"] * thermo["u"] * _relativistic_gamma_from_u(thermo["u"]))
    return np.array([thermo["Pi"] - Pi, E_loc - E_target], dtype=float)


def _fixed_T_0plus_E_residual_ok(residual, Pi, E_target):
    if not np.all(np.isfinite(residual)):
        return False
    pi_tol = 1.0e-8 * max(abs(Pi), 1.0)
    E_tol = 1.0e-8 * max(abs(E_target), 1.0)
    return bool(abs(float(residual[0])) <= pi_tol and abs(float(residual[1])) <= E_tol)


def _solve_interface_0plus_from_T_0plus_E_and_Pi(
    T_0plus,
    E_target,
    Pi,
    jB,
    nB_inf,
    nK_inf,
    B_one_forth,
    ms=0.0,
    upB=5000,
    initial_guess=None,
    stats=None,
    stats_key="interface_0plus_root_calls",
):
    """
    Solve the x = 0+ interface state at prescribed T_0plus.

    The two unknowns are (muB_0plus, muK_0plus). The constraints are momentum
    flux conservation and relativistic enthalpy-flux conservation. The
    interface composition a_0plus is derived after the thermodynamic state is
    found.
    """
    T_0plus = float(T_0plus)
    if (not np.isfinite(T_0plus)) or T_0plus < 0.0:
        raise RuntimeError("Fixed-T_0plus interface solve requires T_0plus >= 0")
    if E_target <= 0.0:
        raise RuntimeError("Fixed-T_0plus interface solve requires E = h*u*r_gamma > 0")
    if nB_inf <= 0.0:
        raise RuntimeError("nB_inf must be positive when deriving a_0plus")

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
        residual = _fixed_T_0plus_E_residual(
            float(vec[0]),
            float(vec[1]),
            T_0plus,
            E_target,
            Pi,
            jB,
            B_one_forth,
            ms=ms,
            upB=upB,
        )
        return np.array([residual[0] / pi_scale, residual[1] / E_scale], dtype=float)

    best_message = "Fixed-T_0plus interface solve did not converge"
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
                best_message = "Fixed-T_0plus interface solve returned a non-physical chemical potential"
                continue
            if muK < 0.0:
                muK = 0.0
            residual = _fixed_T_0plus_E_residual(
                muB,
                muK,
                T_0plus,
                E_target,
                Pi,
                jB,
                B_one_forth,
                ms=ms,
                upB=upB,
            )
            if not _fixed_T_0plus_E_residual_ok(residual, Pi, E_target):
                best_message = (
                    "Fixed-T_0plus interface solve returned an unacceptable residual "
                    f"({residual[0]:.3e}, {residual[1]:.3e})"
                )
                continue
            thermo = _quark_thermo_state(
                muB,
                muK,
                B_one_forth,
                T_0plus,
                jB,
                ms=ms,
                upB=upB,
                allow_zero_temperature=True,
            )
            r_gamma = _relativistic_gamma_from_u(thermo["u"])
            E_loc = float(thermo["h"] * thermo["u"] * r_gamma)
            a_0plus = float((thermo["nK"] - nK_inf) / nB_inf)
            if thermo["h"] <= 0.0 or E_loc <= 0.0 or not np.isfinite(a_0plus):
                best_message = "Fixed-T_0plus interface solve returned a non-physical state"
                continue
            thermo["r_gamma"] = r_gamma
            thermo["E"] = E_loc
            thermo["a_0plus"] = a_0plus
            candidates.append(thermo)
        except Exception as exc:
            best_message = str(exc)

    if not candidates:
        raise RuntimeError(f"Fixed-T_0plus interface solve failed: {best_message}")

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
    """
    Solve the energy-conserving local EOS closure at an absolute nK.

    The third unknown is w = T**2 rather than T or log T. Every thermodynamic
    quantity is analytic in T**2 at low temperature, so E and Pi depend on T
    only at O(T**2); parameterising by log T makes the Jacobian singular as
    T -> 0, leaving the residual insensitive to the temperature and returning
    whatever T the initial guess happened to carry. In w the Jacobian stays
    regular down to w = 0, and w < 0 is the explicit statement that the
    requested nK lies above its zero-temperature limit for this (E, Pi).
    """
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
        if guess.size < 3 or not np.all(np.isfinite(guess[:3])) or guess[2] < 0.0:
            raise RuntimeError("initial_guess must contain finite (muB, muK, T) with T >= 0")
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

    w_slope_cache = {}

    def residual_at(muB_val, muK_val, w_val):
        thermo = _quark_thermo_state(
            muB_val,
            muK_val,
            B_one_forth,
            float(np.sqrt(max(w_val, 0.0))),
            jB,
            ms=ms,
            upB=upB,
            allow_zero_temperature=True,
        )
        E_val = float(thermo["h"] * thermo["u"] * _relativistic_gamma_from_u(thermo["u"]))
        return np.array(
            [
                (thermo["Pi"] - Pi) / pi_scale,
                (thermo["nK"] - nK_target) / nK_scale,
                (E_val - E) / E_scale,
            ],
            dtype=float,
        )

    def w_slope_at_zero(muB_val, muK_val):
        """dResidual/dw at w = 0, used to continue the residual to w < 0."""
        key = (round(muB_val, 6), round(muK_val, 6))
        if key not in w_slope_cache:
            w_probe = 1.0e-4
            w_slope_cache[key] = (
                residual_at(muB_val, muK_val, w_probe)
                - residual_at(muB_val, muK_val, 0.0)
            ) / w_probe
        return w_slope_cache[key]

    def equations(vec):
        muB_val = float(vec[0])
        muK_val = float(vec[1])
        w_val = float(vec[2])
        if muB_val <= 0.0 or (not np.isfinite(w_val)) or abs(w_val) > 1.0e8:
            return np.full(3, 1.0e12, dtype=float)
        try:
            if w_val >= 0.0:
                return residual_at(muB_val, muK_val, w_val)
            # Smooth linear continuation into w < 0 so the root finder can
            # converge there and report the state as infeasible.
            return residual_at(muB_val, muK_val, 0.0) + w_slope_at_zero(muB_val, muK_val) * w_val
        except Exception:
            return np.full(3, 1.0e12, dtype=float)

    for muB_guess, muK_guess, T_guess in guesses:
        try:
            if stats is not None:
                stats["local_root_calls"] = stats.get("local_root_calls", 0) + 1
            sol = root(
                equations,
                np.array([muB_guess, muK_guess, T_guess * T_guess], dtype=float),
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

    w_best = float(best[2])
    if w_best < -1.0e-12 * max(float(T_ref) ** 2, 1.0):
        raise RuntimeError(
            "Absolute-nK energy closure has no solution: the requested "
            f"nK={nK_target:.10g} lies above its zero-temperature limit at this "
            f"(E, Pi), the closure converging to T^2={w_best:.3e} < 0"
        )

    thermo = _quark_thermo_state(
        float(best[0]),
        float(best[1]),
        B_one_forth,
        float(np.sqrt(max(w_best, 0.0))),
        jB,
        ms=ms,
        upB=upB,
        allow_zero_temperature=True,
    )
    thermo["r_gamma"] = _relativistic_gamma_from_u(thermo["u"])
    thermo["E"] = float(thermo["h"] * thermo["u"] * thermo["r_gamma"])
    return thermo


# Public isothermal front solver
def _solve_front_isothermal_normalized_legacy(
    T,
    nB_0minus,
    B_one_forth,
    a_0plus,
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
    Historical normalized-composition implementation retained for comparison.

    This solver keeps the same hydro + diffusion + reaction equations as the
    1D IVP shooting solvers, but uses solve_bvp with jB as an unknown BVP
    parameter. The compact coordinate is s = 1 - exp(-lambda*x/kappa_factor), integrated on
    s in [0, 1 - tail_eps]. The endpoint truncation is controlled only by
    tail_eps; no compact_tail_lengths cutoff is used in this BVP path.
    """
    if nB_0minus <= 0.0:
        raise RuntimeError("nB_0minus must be positive")
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
        jB_guess = 1.0e-6 * nB_0minus
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
        "interface_0plus_root_calls": 0,
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
        P_0minus = float(PNM_n(nB_0minus, T, param=param, NM_type=NM_type))
        e_0minus = float(edensNM_n(nB_0minus, T, param=param))
        h_0minus = float(P_0minus + e_0minus)
        u_0minus = float(jB / nB_0minus)
        Pi = float(h_0minus * u_0minus * u_0minus + P_0minus)

        # Far-right equilibrated quark state Q with muK = 0.
        muB_inf = _solve_muB_inf_at_muK0_for_given_Pi(Pi, jB, B_one_forth, T, ms=ms, upB=upB, stats=stats)
        nB_inf = float(nB_QM(muB_inf, 0.0, B_one_forth, T, ms=ms, upB=upB))
        if nB_inf <= 0.0:
            raise RuntimeError("Equilibrated Q state has non-positive density")
        if abs(ms) <= 1.0e-12:
            nK_inf = 0.0
        else:
            nK_inf = float(nK_QM(muB_inf, 0.0, B_one_forth, T, ms=ms, upB=upB))

        # Pure neutron matter implies nK(0-) = nB_0minus.
        a_0minus = float((nB_0minus - nK_inf) / nB_inf)

        # Interface state at x = 0+.
        muK_0plus_seed = _branch_muK_seed(a_0plus)
        muB_0plus, muK_0plus = _solve_interface_0plus_from_a_0plus_and_Pi(
            a_0plus,
            Pi,
            jB,
            nB_inf,
            nK_inf,
            B_one_forth,
            T,
            ms=ms,
            upB=upB,
            initial_guess=(muB_inf, muK_0plus_seed),
            stats=stats,
            stats_key="interface_0plus_root_calls",
        )
        nB_0plus = float(nB_QM(muB_0plus, muK_0plus, B_one_forth, T, ms=ms, upB=upB))
        if nB_0plus <= 0.0:
            raise RuntimeError("x = 0+ state has non-positive density")

        micro = _microphysics_at_0plus_isothermal_baseline(muB_0plus, T)
        D = float(micro["D"])
        eta = float(micro["eta"])
        gamma = float(micro["gamma"])
        tau = float(micro["tau"])

        u_inf = float(jB / nB_inf)
        disc = float(u_inf * u_inf + 4.0 * D * gamma * eta)
        if (not np.isfinite(disc)) or disc <= 0.0:
            raise RuntimeError("Tail discriminant is non-positive")
        lam = float((-u_inf + np.sqrt(disc)) / (2.0 * D))
        if (not np.isfinite(lam)) or lam <= 0.0:
            raise RuntimeError("Tail decay lambda must be positive")
        q0 = float(-a_0minus * u_0minus)
        x_end = float(-float(kappa_factor) * np.log1p(-s_end) / lam)
        tail_coeff = float(D * lam + u_inf)
        state = {
            "jB": jB,
            "P_0minus": P_0minus,
            "e_0minus": e_0minus,
            "h_0minus": h_0minus,
            "u_0minus": u_0minus,
            "Pi": Pi,
            "muB_inf": float(muB_inf),
            "nB_inf": nB_inf,
            "nK_inf": float(nK_inf),
            "a_0minus": a_0minus,
            "muB_0plus": float(muB_0plus),
            "muK_0plus": float(muK_0plus),
            "nB_0plus": nB_0plus,
            "D": D,
            "eta": eta,
            "gamma": gamma,
            "tau": tau,
            "lambda": lam,
            "u_inf": u_inf,
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
        guess = (state["muB_0plus"], state["muK_0plus"])
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
                    state["nB_inf"],
                    state["nK_inf"],
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
                ya[0] - float(a_0plus),
                ya[1] - state["q0"],
                yb[1] + state["tail_coeff"] * yb[0],
            ],
            dtype=float,
        )

    theta_guess = _param_from_jB(jB_guess)
    state0 = _build_global_state(theta_guess)
    s_mesh = np.linspace(0.0, s_end, int(n_mesh))
    tail_shape = np.maximum(1.0 - s_mesh, tail_eps) ** max(float(kappa_factor), 1.0e-12)
    a_guess = float(a_0plus) * tail_shape
    q_tail_guess = -state0["tail_coeff"] * a_guess
    blend = s_mesh / max(s_end, np.finfo(float).tiny)
    q_guess = (1.0 - blend) * state0["q0"] + blend * q_tail_guess
    y_guess = np.vstack((a_guess, q_guess))

    _diag(
        f"starting compact BVP with jB_guess={jB_guess:.6g}, a_0plus={a_0plus:.6g}, "
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
            "a_0plus": float(a_0plus),
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
            "a_0plus": float(a_0plus),
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
        "a_0plus": float(a_0plus),
        "branch_label": "muK-rich",
        "coordinate": "BVP: s in [0, 1-tail_eps], s=1-exp(-lambda*x/kappa_factor)",
        "tail_eps": float(tail_eps),
        "kappa_factor": float(kappa_factor),
        "u_0minus": float(state["u_0minus"]),
        "u_inf": float(state["u_inf"]),
        "a_0minus": float(state["a_0minus"]),
        "Pi": float(state["Pi"]),
        "muB_0plus": float(state["muB_0plus"]),
        "muK_0plus": float(state["muK_0plus"]),
        "nB_0plus": float(state["nB_0plus"]),
        "muB_inf": float(state["muB_inf"]),
        "nB_inf": float(state["nB_inf"]),
        "nK_inf": float(state["nK_inf"]),
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
            guess = (state["muB_0plus"], state["muK_0plus"])
            for i, a_val in enumerate(a_prof):
                stats["profile_state_calls"] += 1
                muB_loc, muK_loc, nB_loc, u_loc = _solve_local_quark_state_from_a_and_Pi(
                    float(a_val),
                    state["Pi"],
                    state["jB"],
                    state["nB_inf"],
                    state["nK_inf"],
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
                        (nK_QM(float(muB_prof[i]), float(muK_prof[i]), B_one_forth, T, ms=ms, upB=upB) - state["nK_inf"])
                        / state["nB_inf"]
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
                    coarse_profile = _solve_front_isothermal_normalized_legacy(
                        T=T,
                        nB_0minus=nB_0minus,
                        B_one_forth=B_one_forth,
                        a_0plus=a_0plus,
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
            coarse_result = _solve_front_isothermal_normalized_legacy(
                T=T,
                nB_0minus=nB_0minus,
                B_one_forth=B_one_forth,
                a_0plus=a_0plus,
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
                refined_result = _solve_front_isothermal_normalized_legacy(
                    T=T,
                    nB_0minus=nB_0minus,
                    B_one_forth=B_one_forth,
                    a_0plus=a_0plus,
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
            f"bvp jB={result['jB']:.6g}, a_0plus={a_0plus:.6g}, "
            f"tail_norm={tail_residual_norm:.6g}, status={sol.status}, success={success}"
        )
    return result


def solve_front_isothermal(
    T,
    nB_0minus,
    B_one_forth,
    a_0plus=np.nan,
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
    relativistic=False,
):
    """Solve the fixed-T front as a compact BVP in physical nK and J_K.

    ``a_0plus`` is the local interface fraction nK(0+)/nB(0+).  The BVP
    transports the physical K density and current, reconstructs the local EOS
    state at fixed T and momentum flux, and evaluates both D_K and the exact
    nonleptonic rate at every node.

    Passing ``a_0plus=nan`` (the default) resolves it from
    :func:`_solve_a_0plus_max` instead of taking a prescribed value, giving the
    fastest front the interface can support at this upstream state.  That path
    needs muB(0-), so it adds one branch-validated density inversion.  The
    resolved value is echoed in ``a_0plus`` with ``a_0plus_source="maximum"``,
    alongside ``a_0plus_max`` and ``a_0plus_max_status``.  A ceiling that is
    not interior to (0,1) raises rather than returning, since the BVP has no
    admissible composition there.

    ``relativistic=True`` (default ``False``) opts the downstream quark
    momentum-flux closure into the explicit relativistic pair
    jB = nB*gamma*v, Pi = P + w*gamma**2*v**2 by way of
    :func:`_relativistic_flux_pair`.  It is a consistency switch, not a
    physics change: see :func:`_Pi_QM_state` for why gamma**2*v**2 equals
    (jB/nB)**2 identically, so the two branches agree to rounding.  No other
    solver family is affected -- the flag is passed only from inside this
    function.
    """
    T = float(T)
    nB_0minus = float(nB_0minus)
    B_one_forth = float(B_one_forth)
    a_0plus = float(a_0plus)
    ms = float(ms)
    tail_eps = float(tail_eps)
    tol_bvp = float(tol_bvp)
    kappa_factor = float(kappa_factor)
    NM_type = str(NM_type)
    relativistic = bool(relativistic)
    if (not np.isfinite(T)) or T <= 0.0:
        raise RuntimeError("T must be positive and finite")
    if (not np.isfinite(nB_0minus)) or nB_0minus <= 0.0:
        raise RuntimeError("nB_0minus must be positive and finite")
    if (not np.isfinite(B_one_forth)) or B_one_forth <= 0.0:
        raise RuntimeError("B_one_forth must be positive and finite")
    a_0plus_is_auto = bool(np.isnan(a_0plus))
    if (not a_0plus_is_auto) and (
        (not np.isfinite(a_0plus)) or not (0.0 < a_0plus < 1.0)
    ):
        raise RuntimeError(
            "a_0plus must satisfy 0 < a_0plus < 1, or be NaN to request the "
            "thermodynamic maximum"
        )
    if (not np.isfinite(ms)) or ms < 0.0:
        raise RuntimeError("ms must be finite and non-negative")
    if NM_type not in ("PNM", "SYM", "Beta_eq"):
        raise RuntimeError("NM_type must be one of 'PNM', 'SYM', or 'Beta_eq'")
    if not (0.0 < tail_eps < 1.0):
        raise RuntimeError("tail_eps must satisfy 0 < tail_eps < 1")
    if int(n_mesh) < 5:
        raise RuntimeError("n_mesh must be at least 5")
    if (not np.isfinite(tol_bvp)) or tol_bvp <= 0.0:
        raise RuntimeError("tol_bvp must be positive and finite")
    if int(max_nodes) <= int(n_mesh):
        raise RuntimeError("max_nodes must be larger than n_mesh")
    if (not np.isfinite(kappa_factor)) or kappa_factor <= 0.0:
        raise RuntimeError("kappa_factor must be positive and finite")

    upB = 5000
    t_start = time.perf_counter()
    if isinstance(verb, str):
        verb_mode = "full" if verb.lower() == "full" else ("simple" if verb else "off")
    else:
        verb_mode = "simple" if verb else "off"
    full_diag = verb_mode == "full"
    simple_diag = verb_mode in ("simple", "full")

    def diag(message):
        if full_diag:
            elapsed = time.perf_counter() - t_start
            print(f"[isothermal_nK_bvp +{elapsed:8.2f}s] {message}", flush=True)

    nuclear_state = _isothermal_upstream_nuclear_state(
        T,
        nB_0minus,
        param,
        NM_type,
    )
    a_0plus_max = np.nan
    a_0plus_max_status = ""
    if a_0plus_is_auto:
        muB_0minus_for_ceiling = float(
            muB_from_nB_physical(
                nB_0minus,
                T,
                param=param,
                NM_type=NM_type,
                auto_expand=True,
            )
        )
        ceiling = _solve_a_0plus_max(
            muB_0minus_for_ceiling,
            float(nuclear_state["P_0minus"]),
            T,
            B_one_forth,
            ms=ms,
            upB=upB,
        )
        a_0plus_max = float(ceiling["a_0plus_max"])
        a_0plus_max_status = str(ceiling["status"])
        if a_0plus_max_status != "interior":
            raise RuntimeError(
                "a_0plus=NaN requested the thermodynamic maximum, but the "
                f"ceiling solve returned status '{a_0plus_max_status}' "
                f"(a_0plus_max={a_0plus_max:.6g}); the BVP requires "
                "0 < a_0plus < 1"
            )
        a_0plus = a_0plus_max
        diag(
            f"resolved a_0plus from thermodynamic maximum: {a_0plus:.6f}"
        )

    if not (a_0plus < float(nuclear_state["a_0minus"])):
        raise RuntimeError(
            "Forward conversion requires a_0plus < a_0minus"
        )

    if jB_guess is None:
        jB_guess = 1.0e-6 * nB_0minus
    jB_guess = float(jB_guess)
    if (not np.isfinite(jB_guess)) or jB_guess <= 0.0:
        raise RuntimeError("jB_guess must be positive and finite")

    bounded_jB = jB_bounds is not None
    if bounded_jB:
        if len(jB_bounds) != 2:
            raise RuntimeError("jB_bounds must be a 2-tuple (jB_min, jB_max)")
        jB_lower_bound, jB_upper_bound = map(float, jB_bounds)
        if not (
            np.isfinite(jB_lower_bound)
            and np.isfinite(jB_upper_bound)
            and 0.0 < jB_lower_bound < jB_upper_bound
        ):
            raise RuntimeError("jB_bounds must satisfy 0 < jB_min < jB_max")
        jB_guess = float(
            np.clip(
                jB_guess,
                jB_lower_bound * (1.0 + 1.0e-8),
                jB_upper_bound * (1.0 - 1.0e-8),
            )
        )
    else:
        jB_lower_bound = 0.0
        jB_upper_bound = np.inf

    def parameter_from_jB(jB):
        jB = float(jB)
        if bounded_jB:
            fraction = (jB - jB_lower_bound) / (
                jB_upper_bound - jB_lower_bound
            )
            fraction = float(np.clip(fraction, 1.0e-12, 1.0 - 1.0e-12))
            return float(np.log(fraction / (1.0 - fraction)))
        return float(np.log(jB))

    def jB_from_parameter(theta):
        theta = float(theta)
        if bounded_jB:
            sigmoid = 1.0 / (1.0 + np.exp(-np.clip(theta, -60.0, 60.0)))
            return float(
                jB_lower_bound
                + (jB_upper_bound - jB_lower_bound) * sigmoid
            )
        return float(np.exp(np.clip(theta, -700.0, 700.0)))

    stats = {
        "bvp_ode_calls": 0,
        "bvp_bc_calls": 0,
        "global_state_builds": 0,
        "global_state_failures": 0,
        "local_state_failures": 0,
        "profile_state_calls": 0,
    }
    state_cache = {}
    last_failure = {"message": ""}
    s_end = float(1.0 - tail_eps)

    def build_global_state(theta):
        key = round(float(theta), 12)
        if key in state_cache:
            return state_cache[key]
        stats["global_state_builds"] += 1
        jB = jB_from_parameter(theta)
        u_0minus = float(jB / nB_0minus)
        Pi = float(
            nuclear_state["P_0minus"]
            + nuclear_state["h_0minus"] * u_0minus**2
        )

        muB_inf = float(
            _solve_muB_inf_at_muK0_for_given_Pi(
                Pi,
                jB,
                B_one_forth,
                T,
                ms=ms,
                upB=upB,
                relativistic=relativistic,
            )
        )
        thermo_inf = _quark_thermo_state(
            muB_inf,
            0.0,
            B_one_forth,
            T,
            jB,
            ms=ms,
            upB=upB,
        )
        micro_inf = _microphysics_from_quark_state_energy(muB_inf, T)
        if micro_inf["invD"] <= 0.0:
            raise RuntimeError("Downstream inverse diffusion coefficient is non-positive")
        D_K_inf = float(1.0 / micro_inf["invD"])
        nK_inf = float(thermo_inf["nK"])
        nB_inf = float(thermo_inf["nB"])
        u_inf = float(thermo_inf["u"])
        jK_inf = float(u_inf * nK_inf)
        a_inf = float(nK_inf / nB_inf)
        if not (a_inf < a_0plus):
            raise RuntimeError(
                "Forward conversion requires a_inf < a_0plus"
            )

        muB_0plus, muK_0plus = _solve_interface_0plus_from_local_a_and_Pi(
            a_0plus,
            Pi,
            jB,
            B_one_forth,
            T,
            ms=ms,
            upB=upB,
            initial_guess=(muB_inf, _branch_muK_seed(a_0plus)),
        )
        thermo_0plus = _quark_thermo_state(
            muB_0plus,
            muK_0plus,
            B_one_forth,
            T,
            jB,
            ms=ms,
            upB=upB,
        )
        micro_0plus = _microphysics_from_quark_state_energy(muB_0plus, T)
        D_K_0plus = float(1.0 / micro_0plus["invD"])
        rate_0plus = float(
            _exact_kaon_transport_rate(
                muB_0plus,
                muK_0plus,
                T,
                ms=ms,
                upB=upB,
            )["Gamma_K"]
        )

        delta_nK = float(
            max(1.0e-5 * max(abs(nK_inf), nB_inf), 1.0e-2)
        )
        probe = _solve_local_quark_state_from_nK_T_and_Pi(
            nK_inf + delta_nK,
            T,
            Pi,
            jB,
            B_one_forth,
            ms=ms,
            upB=upB,
            initial_guess=(muB_inf, max(1.0e-3, muK_0plus * 1.0e-3)),
            stats=stats,
        )
        rate_probe = float(
            _exact_kaon_transport_rate(
                probe["muB"],
                probe["muK"],
                T,
                ms=ms,
                upB=upB,
            )["Gamma_K"]
        )
        rate_slope_inf = float(rate_probe / delta_nK)
        advective_flux_slope_inf = float(
            (probe["u"] * probe["nK"] - jK_inf) / delta_nK
        )
        if (
            (not np.isfinite(rate_slope_inf))
            or rate_slope_inf <= 0.0
            or (not np.isfinite(advective_flux_slope_inf))
        ):
            raise RuntimeError("Downstream isothermal linearization is non-physical")
        discriminant = float(
            advective_flux_slope_inf**2
            + 4.0 * D_K_inf * rate_slope_inf
        )
        lambda_inf = float(
            (
                -advective_flux_slope_inf
                + np.sqrt(discriminant)
            )
            / (2.0 * D_K_inf)
        )
        if (not np.isfinite(lambda_inf)) or lambda_inf <= 0.0:
            raise RuntimeError("Downstream tail decay rate is non-positive")

        jK_0plus = float(jB * nuclear_state["a_0minus"])
        tail_coefficient = float(
            advective_flux_slope_inf + D_K_inf * lambda_inf
        )
        state = {
            "jB": jB,
            "u_0minus": u_0minus,
            "Pi": Pi,
            "muB_0plus": float(muB_0plus),
            "muK_0plus": float(muK_0plus),
            "nB_0plus": float(thermo_0plus["nB"]),
            "nK_0plus": float(thermo_0plus["nK"]),
            "u_0plus": float(thermo_0plus["u"]),
            "jK_0plus": jK_0plus,
            "D_K_0plus": D_K_0plus,
            "Gamma_K_0plus": rate_0plus,
            "muB_inf": muB_inf,
            "nB_inf": nB_inf,
            "nK_inf": nK_inf,
            "u_inf": u_inf,
            "jK_inf": jK_inf,
            "a_inf": a_inf,
            "D_K_inf": D_K_inf,
            "rate_slope_inf": rate_slope_inf,
            "advective_flux_slope_inf": advective_flux_slope_inf,
            "lambda_inf": lambda_inf,
            "tail_coefficient": tail_coefficient,
            "x_end": float(
                -kappa_factor * np.log1p(-s_end) / lambda_inf
            ),
        }
        state_cache[key] = state
        return state

    def state_or_none(theta):
        try:
            return build_global_state(theta)
        except Exception as exc:
            stats["global_state_failures"] += 1
            last_failure["message"] = str(exc)
            return None

    def ode(s, y, p):
        stats["bvp_ode_calls"] += 1
        state = state_or_none(float(p[0]))
        if state is None:
            return np.full_like(y, 1.0e12)
        dyds = np.empty_like(y)
        guess = (state["muB_0plus"], state["muK_0plus"])
        for index in range(y.shape[1]):
            nK_value = float(y[0, index])
            jK_value = float(y[1, index])
            try:
                thermo = _solve_local_quark_state_from_nK_T_and_Pi(
                    nK_value,
                    T,
                    state["Pi"],
                    state["jB"],
                    B_one_forth,
                    ms=ms,
                    upB=upB,
                    initial_guess=guess,
                    stats=stats,
                )
                guess = (thermo["muB"], thermo["muK"])
                micro = _microphysics_from_quark_state_energy(
                    thermo["muB"],
                    T,
                )
                rate = float(
                    _exact_kaon_transport_rate(
                        thermo["muB"],
                        thermo["muK"],
                        T,
                        ms=ms,
                        upB=upB,
                    )["Gamma_K"]
                )
                one_minus_s = max(
                    1.0 - float(s[index]),
                    np.finfo(float).tiny,
                )
                dx_ds = float(
                    kappa_factor
                    / (state["lambda_inf"] * one_minus_s)
                )
                dyds[0, index] = (
                    (thermo["u"] * nK_value - jK_value)
                    * micro["invD"]
                    * dx_ds
                )
                dyds[1, index] = -rate * dx_ds
            except Exception as exc:
                stats["local_state_failures"] += 1
                last_failure["message"] = str(exc)
                dyds[:, index] = 1.0e12
        return dyds

    def boundary_conditions(ya, yb, p):
        stats["bvp_bc_calls"] += 1
        state = state_or_none(float(p[0]))
        if state is None:
            return np.full(3, 1.0e12, dtype=float)
        nK_scale = max(
            abs(state["nK_0plus"] - state["nK_inf"]),
            abs(state["nK_0plus"]),
            1.0,
        )
        jK_scale = max(
            abs(state["jK_0plus"] - state["jK_inf"]),
            abs(state["jB"]),
            1.0,
        )
        tail_residual = float(
            yb[1]
            - state["jK_inf"]
            - state["tail_coefficient"]
            * (yb[0] - state["nK_inf"])
        )
        return np.array(
            [
                (ya[0] - state["nK_0plus"]) / nK_scale,
                (ya[1] - state["jK_0plus"]) / jK_scale,
                tail_residual / jK_scale,
            ],
            dtype=float,
        )

    theta_guess = parameter_from_jB(jB_guess)
    state0 = build_global_state(theta_guess)
    s_mesh = np.linspace(0.0, s_end, int(n_mesh))
    tail_shape = np.maximum(1.0 - s_mesh, tail_eps) ** kappa_factor
    nK_guess = state0["nK_inf"] + (
        state0["nK_0plus"] - state0["nK_inf"]
    ) * tail_shape
    stable_jK_guess = state0["jK_inf"] + state0["tail_coefficient"] * (
        nK_guess - state0["nK_inf"]
    )
    blend = s_mesh / max(s_end, np.finfo(float).tiny)
    jK_guess_profile = (
        (1.0 - blend) * state0["jK_0plus"]
        + blend * stable_jK_guess
    )
    y_guess = np.vstack((nK_guess, jK_guess_profile))
    diag(
        f"starting compact BVP with jB_guess={jB_guess:.6g}, "
        f"a_0plus={a_0plus:.6g}, tail_eps={tail_eps:.3g}"
    )

    try:
        sol = solve_bvp(
            ode,
            boundary_conditions,
            s_mesh,
            y_guess,
            p=np.array([theta_guess], dtype=float),
            tol=tol_bvp,
            bc_tol=tol_bvp,
            max_nodes=int(max_nodes),
            verbose=2 if full_diag else 0,
        )
        state = build_global_state(float(sol.p[0]))
        bc_residual = boundary_conditions(
            sol.y[:, 0],
            sol.y[:, -1],
            sol.p,
        )
    except Exception as exc:
        return {
            "success": False,
            "message": (
                f"Physical-nK isothermal BVP failed: {exc}; "
                f"last failure: {last_failure['message']}"
            ),
            "a_0plus": a_0plus,
            "a_0plus_source": "maximum" if a_0plus_is_auto else "input",
            "a_0plus_max": a_0plus_max,
            "a_0plus_max_status": a_0plus_max_status,
            "jB": np.nan,
            "composition_definition": "nK_over_local_nB",
            "current_definition": "u_nK_minus_D_K_dnK_dx",
            "rate_model": "exact_nonleptonic",
            "diffusion_model": "local_muB_fixed_T",
            "bvp_status": -999,
            **{key: int(value) for key, value in stats.items()},
        }

    nK_end = float(sol.y[0, -1])
    jK_end = float(sol.y[1, -1])
    end_thermo = _solve_local_quark_state_from_nK_T_and_Pi(
        nK_end,
        T,
        state["Pi"],
        state["jB"],
        B_one_forth,
        ms=ms,
        upB=upB,
        initial_guess=(state["muB_inf"], 0.0),
        stats=stats,
    )
    a_end = float(nK_end / end_thermo["nB"])
    tail_residual_norm = float(bc_residual[2])
    success = bool(
        sol.success
        and np.all(np.isfinite(bc_residual))
        and np.max(np.abs(bc_residual)) <= max(tol_bvp, 1.0e-10)
    )
    result = {
        "success": success,
        "message": (
            "Physical-nK isothermal BVP converged"
            if success
            else f"{sol.message}; last failure: {last_failure['message']}"
        ),
        "jB": float(state["jB"]),
        "u_0minus": float(state["u_0minus"]),
        "u_0plus": float(state["u_0plus"]),
        "u_inf": float(state["u_inf"]),
        "Pi": float(state["Pi"]),
        "T_0minus": T,
        "T_0plus": T,
        "T_inf": T,
        "nB_0minus": nB_0minus,
        "P_0minus": float(nuclear_state["P_0minus"]),
        "e_0minus": float(nuclear_state["e_0minus"]),
        "h_0minus": float(nuclear_state["h_0minus"]),
        "proton_fraction_0minus": float(
            nuclear_state["proton_fraction_0minus"]
        ),
        "a_0minus": float(nuclear_state["a_0minus"]),
        "nK_0minus": float(nuclear_state["nK_0minus"]),
        "a_0plus": a_0plus,
        "a_0plus_source": "maximum" if a_0plus_is_auto else "input",
        "a_0plus_max": a_0plus_max,
        "a_0plus_max_status": a_0plus_max_status,
        "a_0plus_derived": float(state["nK_0plus"] / state["nB_0plus"]),
        **_momentum_flux_diagnostics(
            float(nuclear_state["P_0minus"]),
            float(nuclear_state["h_0minus"]),
            float(state["u_0minus"]),
        ),
        "muB_0plus": float(state["muB_0plus"]),
        "muK_0plus": float(state["muK_0plus"]),
        "nB_0plus": float(state["nB_0plus"]),
        "nK_0plus": float(state["nK_0plus"]),
        "jK_0plus": float(state["jK_0plus"]),
        "muB_inf": float(state["muB_inf"]),
        "muK_inf": 0.0,
        "nB_inf": float(state["nB_inf"]),
        "nK_inf": float(state["nK_inf"]),
        "jK_inf": float(state["jK_inf"]),
        "a_inf": float(state["a_inf"]),
        "D_K_0plus": float(state["D_K_0plus"]),
        "D_K_inf": float(state["D_K_inf"]),
        "Gamma_K_0plus": float(state["Gamma_K_0plus"]),
        "rate_slope_inf": float(state["rate_slope_inf"]),
        "advective_flux_slope_inf": float(
            state["advective_flux_slope_inf"]
        ),
        "lambda_inf": float(state["lambda_inf"]),
        "coordinate": (
            "BVP: s in [0, 1-tail_eps], "
            "s=1-exp(-lambda_inf*x/kappa_factor)"
        ),
        "tail_eps": tail_eps,
        "kappa_factor": kappa_factor,
        "compact_scale": float(kappa_factor / state["lambda_inf"]),
        "s_end": s_end,
        "x_end": float(state["x_end"]),
        "nK_end": nK_end,
        "jK_end": jK_end,
        "a_end": a_end,
        "tail_residual": float(
            jK_end
            - state["jK_inf"]
            - state["tail_coefficient"] * (nK_end - state["nK_inf"])
        ),
        "tail_residual_norm": tail_residual_norm,
        "boundary_residuals": np.asarray(bc_residual, dtype=float),
        "composition_definition": "nK_over_local_nB",
        "current_definition": "u_nK_minus_D_K_dnK_dx",
        "rate_model": "exact_nonleptonic",
        "diffusion_model": "local_muB_fixed_T",
        "_residual": np.asarray(bc_residual, dtype=float),
        "_root_method": "solve_bvp_nK_parameter_1d",
        "bvp_status": int(sol.status),
        "bvp_message": str(sol.message),
        "bvp_niter": int(getattr(sol, "niter", -1)),
        "bvp_nodes": int(sol.x.size),
        **{key: int(value) for key, value in stats.items()},
    }

    if return_profile:
        try:
            s_profile = np.asarray(sol.x, dtype=float)
            nK_profile = np.asarray(sol.y[0], dtype=float)
            jK_profile = np.asarray(sol.y[1], dtype=float)
            x_profile = (
                -kappa_factor
                * np.log1p(-s_profile)
                / float(state["lambda_inf"])
            )
            fields = {
                key: np.empty_like(s_profile)
                for key in (
                    "nB",
                    "u",
                    "muB",
                    "muK",
                    "D_K",
                    "Gamma_K",
                    "closure_error",
                )
            }
            guess = (state["muB_0plus"], state["muK_0plus"])
            for index, nK_value in enumerate(nK_profile):
                stats["profile_state_calls"] += 1
                thermo = _solve_local_quark_state_from_nK_T_and_Pi(
                    float(nK_value),
                    T,
                    state["Pi"],
                    state["jB"],
                    B_one_forth,
                    ms=ms,
                    upB=upB,
                    initial_guess=guess,
                    stats=stats,
                )
                guess = (thermo["muB"], thermo["muK"])
                micro = _microphysics_from_quark_state_energy(
                    thermo["muB"],
                    T,
                )
                rate = _exact_kaon_transport_rate(
                    thermo["muB"],
                    thermo["muK"],
                    T,
                    ms=ms,
                    upB=upB,
                )
                fields["nB"][index] = thermo["nB"]
                fields["u"][index] = thermo["u"]
                fields["muB"][index] = thermo["muB"]
                fields["muK"][index] = thermo["muK"]
                fields["D_K"][index] = 1.0 / micro["invD"]
                fields["Gamma_K"][index] = rate["Gamma_K"]
                fields["closure_error"][index] = thermo["closure_residual"]
            a_profile = nK_profile / fields["nB"]
            dy_ds = np.asarray(sol.sol.derivative(1)(s_profile), dtype=float)
            dx_ds = (
                kappa_factor
                / (
                    state["lambda_inf"]
                    * np.maximum(1.0 - s_profile, np.finfo(float).tiny)
                )
            )
            dnK_dx = dy_ds[0] / dx_ds
            djK_dx = dy_ds[1] / dx_ds
            constitutive_residual = (
                fields["u"] * nK_profile
                - fields["D_K"] * dnK_dx
                - jK_profile
            )
            reaction_residual = djK_dx + fields["Gamma_K"]
            result.update(
                {
                    "s": s_profile,
                    "x": x_profile,
                    "a": a_profile,
                    "nK": nK_profile,
                    "jK": jK_profile,
                    "nB": fields["nB"],
                    "u": fields["u"],
                    "muB": fields["muB"],
                    "muK": fields["muK"],
                    "D_K": fields["D_K"],
                    "Gamma_K": fields["Gamma_K"],
                    "closure_error": fields["closure_error"],
                    "closure_error_max": float(
                        np.max(np.abs(fields["closure_error"]))
                    ),
                    "constitutive_residual": constitutive_residual,
                    "reaction_residual": reaction_residual,
                    "constitutive_residual_norm": float(
                        np.max(np.abs(constitutive_residual))
                        / max(abs(state["jB"]), 1.0)
                    ),
                    "reaction_residual_norm": float(
                        np.mean(np.abs(reaction_residual))
                        / max(np.mean(np.abs(fields["Gamma_K"])), np.finfo(float).tiny)
                    ),
                    "a_monotone_nonincreasing": bool(
                        np.max(np.diff(a_profile))
                        <= max(10.0 * tol_bvp, 1.0e-8)
                    ),
                    "profile_state_calls": int(stats["profile_state_calls"]),
                }
            )
        except Exception as exc:
            result["success"] = False
            result["message"] = (
                f"{result['message']}; profile reconstruction failed: {exc}"
            )

    if simple_diag:
        print(
            f"bvp jB={result['jB']:.6g}, a_0plus={a_0plus:.6g}, "
            f"tail_norm={tail_residual_norm:.6g}, status={sol.status}, "
            f"success={result['success']}"
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
    """
    Solve the interface EOS state at a local fraction nK/nB.

    As in _solve_local_quark_state_from_nK_E_and_Pi, the temperature unknown is
    w = T**2 rather than log T: E and Pi are analytic in T**2 and depend on T
    only at O(T**2), so a log T parameterisation leaves the Jacobian singular
    and the residual insensitive to T as the interface cools. w < 0 means the
    requested a_target is unreachable at this (E, Pi).
    """
    a_target = float(a_target)
    if not (0.0 < a_target < 1.0):
        raise RuntimeError("a_0plus must satisfy 0 < a_0plus < 1")
    guesses = []
    if initial_guess is not None:
        guess = np.asarray(initial_guess, dtype=float).ravel()
        if guess.size >= 3 and np.all(np.isfinite(guess[:3])) and guess[2] >= 0.0:
            guesses.append((float(guess[0]), float(guess[1]), float(guess[2])))
    guesses.extend([(1200.0, 100.0, 10.0), (1500.0, 200.0, 20.0), (1000.0, 40.0, 5.0)])
    pi_scale = max(abs(float(Pi)), 1.0)
    E_scale = max(abs(float(E)), 1.0)
    best = None
    best_norm = np.inf
    best_message = "local-a interface closure did not converge"

    w_slope_cache = {}

    def residual_at(muB_val, muK_val, w_val):
        thermo = _quark_thermo_state(
            muB_val,
            muK_val,
            B_one_forth,
            float(np.sqrt(max(w_val, 0.0))),
            jB,
            ms=ms,
            upB=upB,
            allow_zero_temperature=True,
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

    def w_slope_at_zero(muB_val, muK_val):
        """dResidual/dw at w = 0, used to continue the residual to w < 0."""
        key = (round(muB_val, 6), round(muK_val, 6))
        if key not in w_slope_cache:
            w_probe = 1.0e-4
            w_slope_cache[key] = (
                residual_at(muB_val, muK_val, w_probe)
                - residual_at(muB_val, muK_val, 0.0)
            ) / w_probe
        return w_slope_cache[key]

    def equations(vec):
        muB_val, muK_val, w_val = map(float, vec)
        if muB_val <= 0.0 or (not np.isfinite(w_val)) or abs(w_val) > 1.0e8:
            return np.full(3, 1.0e12, dtype=float)
        try:
            if w_val >= 0.0:
                return residual_at(muB_val, muK_val, w_val)
            # Smooth linear continuation into w < 0 so the root finder can
            # converge there and report the state as unreachable.
            return residual_at(muB_val, muK_val, 0.0) + w_slope_at_zero(muB_val, muK_val) * w_val
        except Exception:
            return np.full(3, 1.0e12, dtype=float)

    for muB_guess, muK_guess, T_guess in guesses:
        try:
            if stats is not None:
                stats["interface_0plus_root_calls"] = stats.get("interface_0plus_root_calls", 0) + 1
            sol = root(
                equations,
                np.array([muB_guess, muK_guess, T_guess * T_guess], dtype=float),
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
    w_best = float(best[2])
    if w_best < -1.0e-12:
        raise RuntimeError(
            "Local-a interface closure has no solution: the requested "
            f"a_0plus={a_target:.10g} is unreachable at this (E, Pi), the closure "
            f"converging to T^2={w_best:.3e} < 0"
        )
    thermo = _quark_thermo_state(
        float(best[0]),
        float(best[1]),
        B_one_forth,
        float(np.sqrt(max(w_best, 0.0))),
        jB,
        ms=ms,
        upB=upB,
        allow_zero_temperature=True,
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
    T_0minus,
    nB_0minus,
    B_one_forth,
    a_0plus,
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
    T_0minus = float(T_0minus)
    nB_0minus = float(nB_0minus)
    a_0plus = float(a_0plus)
    if NM_type != "PNM":
        raise RuntimeError("solve_front_energy_conserving_nK currently requires NM_type='PNM'")
    if T_0minus <= 0.0 or nB_0minus <= 0.0:
        raise RuntimeError("T and nB_0minus must be positive")
    if not (0.0 < a_0plus < 1.0):
        raise RuntimeError("a_0plus must satisfy 0 < a_0plus < 1")
    if not (0.0 < float(tail_eps) < 1.0):
        raise RuntimeError("tail_eps must satisfy 0 < tail_eps < 1")
    if int(n_mesh) < 5 or int(max_nodes) <= int(n_mesh) or float(tol_bvp) <= 0.0:
        raise RuntimeError("invalid BVP mesh or tolerance settings")
    if compact_scale is not None and ((not np.isfinite(compact_scale)) or compact_scale <= 0.0):
        raise RuntimeError("compact_scale must be positive and finite")

    upB = 5000
    P_0minus = float(PNM_n(nB_0minus, T_0minus, param=param, NM_type=NM_type))
    e_0minus = float(edensNM_n(nB_0minus, T_0minus, param=param))
    h_0minus = float(P_0minus + e_0minus)
    nuclear_state = {
        "P_0minus": P_0minus,
        "e_0minus": e_0minus,
        "h_0minus": h_0minus,
        "nB_0minus": nB_0minus,
        "h_over_nB_0minus": float(h_0minus / nB_0minus),
        "T_0minus": T_0minus,
    }
    if jB_guess is None:
        jB_guess = _default_energy_jB_guess(nB_0minus)
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
        "interface_0plus_root_calls": 0,
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
        u_0minus = float(jB / nB_0minus)
        r_gamma_0minus = _relativistic_gamma_from_u(u_0minus)
        state = {
            "jB": jB,
            "u_0minus": u_0minus,
            "r_gamma_0minus": r_gamma_0minus,
            "Pi": float(h_0minus * u_0minus * u_0minus + P_0minus),
            "E": float(h_0minus * u_0minus * r_gamma_0minus),
        }
        state_cache[key] = state
        return state

    def downstream_target(theta):
        key = round(float(theta), 12)
        if key in downstream_cache:
            return downstream_cache[key]
        state = build_state(theta)
        stats["downstream_root_calls"] += 1
        endpoint = _solve_analytic_inf_endpoint_for_u_0minus(
            state["u_0minus"],
            nuclear_state,
            B_one_forth,
            ms=ms,
            upB=upB,
        )
        thermo = _quark_thermo_state(
            endpoint["muB_inf"],
            0.0,
            B_one_forth,
            endpoint["T_inf"],
            state["jB"],
            ms=ms,
            upB=upB,
        )
        micro = _microphysics_from_quark_state_energy(thermo["muB"], thermo["T"])
        if micro["invD"] <= 0.0:
            raise RuntimeError("downstream inverse diffusion coefficient must be positive")
        D_inf = float(1.0 / micro["invD"])
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
        u_inf = float(thermo["u"])
        lam = float((-u_inf + np.sqrt(u_inf * u_inf + 4.0 * D_inf * rate_slope)) / (2.0 * D_inf))
        target = {
            **thermo,
            "D": D_inf,
            "invD": float(micro["invD"]),
            "rate_slope": rate_slope,
            "lambda": lam,
            "jK": float(u_inf * thermo["nK"]),
            "a": float(thermo["nK"] / thermo["nB"]),
        }
        downstream_cache[key] = target
        return target

    theta0 = param_from_jB(jB_guess)
    state0 = build_state(theta0)
    state_0plus = _solve_interface_state_from_local_a_E_and_Pi(
        a_0plus,
        state0["E"],
        state0["Pi"],
        state0["jB"],
        B_one_forth,
        ms=ms,
        upB=upB,
        initial_guess=(1200.0, max(40.0, 300.0 * a_0plus), T_0minus),
        stats=stats,
    )
    micro_0plus = _microphysics_from_quark_state_energy(state_0plus["muB"], state_0plus["T"])
    rate_0plus = _exact_kaon_transport_rate(
        state_0plus["muB"], state_0plus["muK"], state_0plus["T"], ms=ms, upB=upB
    )["Gamma_K"]
    if compact_scale is None:
        D_0plus = 1.0 / micro_0plus["invD"]
        compact_scale_used = float(np.sqrt(D_0plus * max(abs(state_0plus["nK"]), 1.0) / max(abs(rate_0plus), _FLOAT_TINY)))
    else:
        compact_scale_used = float(compact_scale)
    if (not np.isfinite(compact_scale_used)) or compact_scale_used <= 0.0:
        raise RuntimeError("failed to construct a positive compact scale")

    s_end = float(1.0 - tail_eps)
    s_mesh = np.linspace(0.0, s_end, int(n_mesh))
    blend = s_mesh / s_end
    tail_shape = np.maximum(1.0 - blend, 0.0)
    nK_guess = state_0plus["nK"] * tail_shape
    jK_guess_profile = state0["jB"] * tail_shape
    if isinstance(profile_guess, dict):
        try:
            prev_s = np.asarray(profile_guess["s_coord"], dtype=float)
            prev_nK = np.asarray(profile_guess["nK"], dtype=float)
            prev_jK = np.asarray(profile_guess["jK"], dtype=float)
            prev_nK_inf = float(profile_guess.get("nK_inf", 0.0))
            if prev_s.ndim == 1 and prev_nK.shape == prev_s.shape and prev_jK.shape == prev_s.shape:
                prev_delta0 = float(prev_nK[0] - prev_nK_inf)
                scale = float((state_0plus["nK"] - prev_nK_inf) / prev_delta0) if abs(prev_delta0) > 1.0e-12 else 1.0
                nK_guess = prev_nK_inf + scale * np.interp(s_mesh, prev_s, prev_nK - prev_nK_inf)
                jK_guess_profile = np.interp(s_mesh, prev_s, prev_jK)
                jK_guess_profile += (state0["jB"] - jK_guess_profile[0]) * (1.0 - blend)
                nK_guess[0] = state_0plus["nK"]
                jK_guess_profile[0] = state0["jB"]
        except Exception:
            nK_guess = state_0plus["nK"] * tail_shape
            jK_guess_profile = state0["jB"] * tail_shape
    y_guess = np.vstack((nK_guess, jK_guess_profile))

    def ode(s_coord, y, p):
        stats["bvp_ode_calls"] += 1
        state = build_state(float(p[0]))
        dyds = np.empty_like(y)
        guess = (state_0plus["muB"], state_0plus["muK"], state_0plus["T"])
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
                    T_ref=T_0minus,
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
                initial_guess=(state_0plus["muB"], state_0plus["muK"], state_0plus["T"]),
                T_ref=state_0plus["T"],
                stats=stats,
            )
            target = downstream_target(float(p[0]))
            tail = _nK_tail_residual(yb, target)
            j_scale = max(abs(state["jB"]), abs(target["jK"]), 1.0)
            nK_scale = max(abs(left["nK"]), abs(target["nK"]), 1.0)
            return np.array(
                [
                    (left["nK"] / left["nB"] - a_0plus) / max(abs(a_0plus), 1.0e-6),
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
            "a_0plus": a_0plus,
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
    guess = (state_0plus["muB"], state_0plus["muK"], state_0plus["T"])
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
        "a_0plus": a_0plus,
        "a_0plus_derived": float(a_profile[0]),
        "jB": float(state["jB"]),
        "u_0minus": float(state["u_0minus"]),
        "Pi": float(state["Pi"]),
        "E": float(state["E"]),
        "T_inf": float(target["T"]),
        "nB_inf": float(target["nB"]),
        "nK_inf": float(target["nK"]),
        "a_inf": float(target["a"]),
        "lambda_inf": float(target["lambda"]),
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
    T_0minus,
    nB_0minus,
    B_one_forth,
    a_0plus,
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
    """Public absolute-nK solver with staged a_0plus continuation fallback."""
    direct = _solve_front_energy_conserving_nK_once(
        T_0minus,
        nB_0minus,
        B_one_forth,
        a_0plus,
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
    if bool(direct.get("success")) or float(a_0plus) <= 2.0e-2:
        direct["continuation_used"] = False
        direct["continuation_steps"] = 0
        return direct

    stage_targets = np.linspace(2.0e-2, float(a_0plus), int(np.ceil((float(a_0plus) - 2.0e-2) / 2.0e-2)) + 1)
    current = None
    for index, stage_a in enumerate(stage_targets):
        stage = _solve_front_energy_conserving_nK_once(
            T_0minus,
            nB_0minus,
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
            failed["message"] = f"{direct.get('message')}; continuation failed at a_0plus={stage_a:.6g}: {stage.get('message')}"
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


def _u0minus_max_collocation_status_is_acceptable(
    *, solver_success, solver_status, exact_zero_left
):
    """Require SciPy's collocation solve itself to have converged."""
    return bool(solver_success)


def _solve_front_energy_conserving_uNmax_once(
    T_0minus,
    nB_0minus,
    B_one_forth,
    T_0plus,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    T_inf_guess=None,
    jB_bounds=None,
    return_profile=False,
    verb=False,
    continuation_guess=None,
):
    """
    Solve the fixed-T_0plus energy front using absolute nK and physical jK.

    The prescribed interface temperature is not a root unknown. Interior local
    states use the same T**2 closure as solve_front_energy_conserving_nK.
    """
    T_0minus = float(T_0minus)
    nB_0minus = float(nB_0minus)
    T_0plus = float(T_0plus)
    if NM_type != "PNM":
        raise RuntimeError("solve_front_energy_conserving_uNmax currently requires NM_type='PNM'")
    if (not np.isfinite(T_0minus)) or T_0minus < 0.0 or (not np.isfinite(nB_0minus)) or nB_0minus <= 0.0:
        raise RuntimeError(
            "solve_front_energy_conserving_uNmax requires non-negative T_0minus "
            "and positive nB_0minus"
        )
    if (not np.isfinite(T_0plus)) or T_0plus < 0.0:
        raise RuntimeError("T_0plus must be non-negative")
    if not (0.0 < float(tail_eps) < 1.0):
        raise RuntimeError("tail_eps must satisfy 0 < tail_eps < 1")
    if int(n_mesh) < 5 or int(max_nodes) <= int(n_mesh) or float(tol_bvp) <= 0.0:
        raise RuntimeError("invalid BVP mesh or tolerance settings")

    upB = 5000
    P_0minus = float(PNM_n(nB_0minus, T_0minus, param=param, NM_type=NM_type))
    e_0minus = float(edensNM_n(nB_0minus, T_0minus, param=param))
    h_0minus = float(P_0minus + e_0minus)
    nuclear_state = {
        "P_0minus": P_0minus,
        "e_0minus": e_0minus,
        "h_0minus": h_0minus,
        "nB_0minus": nB_0minus,
        "h_over_nB_0minus": float(h_0minus / nB_0minus),
        "T_0minus": T_0minus,
    }
    if jB_guess is None:
        jB_guess = _default_energy_jB_guess(nB_0minus)
    jB_guess = float(jB_guess)
    if (not np.isfinite(jB_guess)) or jB_guess <= 0.0:
        raise RuntimeError("jB_guess must be positive")
    endpoint_initial_guess = None
    if isinstance(continuation_guess, dict):
        try:
            continued_muB_inf = float(continuation_guess.get("muB_inf", np.nan))
            continued_T_inf = float(continuation_guess.get("T_inf", np.nan))
        except Exception:
            continued_muB_inf = np.nan
            continued_T_inf = np.nan
        if (
            np.isfinite(continued_muB_inf)
            and continued_muB_inf > 0.0
            and np.isfinite(continued_T_inf)
            and continued_T_inf > 0.0
        ):
            endpoint_initial_guess = (continued_muB_inf, continued_T_inf)
    if endpoint_initial_guess is None and T_inf_guess is not None:
        try:
            T_inf_guess_value = float(T_inf_guess)
        except Exception:
            T_inf_guess_value = np.nan
        if np.isfinite(T_inf_guess_value) and T_inf_guess_value > 0.0:
            endpoint_initial_guess = (1100.0, T_inf_guess_value)

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

    stats = {
        "bvp_ode_calls": 0,
        "bvp_bc_calls": 0,
        "q_root_calls": 0,
        "interface_0plus_root_calls": 0,
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
    state_0plus_initial_guess = None
    if isinstance(continuation_guess, dict):
        try:
            continued_muB_0plus = float(continuation_guess.get("muB_0plus", np.nan))
            continued_muK_0plus = float(continuation_guess.get("muK_0plus", np.nan))
            continued_T_0plus = float(continuation_guess.get("T_0plus", T_0plus))
        except Exception:
            continued_muB_0plus = np.nan
            continued_muK_0plus = np.nan
            continued_T_0plus = np.nan
        if (
            np.isfinite(continued_muB_0plus)
            and continued_muB_0plus > 0.0
            and np.isfinite(continued_muK_0plus)
            and continued_muK_0plus >= 0.0
            and np.isfinite(continued_T_0plus)
            and continued_T_0plus >= 0.0
        ):
            state_0plus_initial_guess = (
                continued_muB_0plus,
                continued_muK_0plus,
                continued_T_0plus,
            )
    state_0plus_guess_cache = {"value": state_0plus_initial_guess}
    exact_zero_left = bool(T_0plus == 0.0)
    s_end = float(1.0 - tail_eps)

    def build_state(theta):
        key = round(float(theta), 12)
        if key in state_cache:
            return state_cache[key]
        stats["global_state_builds"] += 1
        jB = jB_from_param(theta)
        u_0minus = float(jB / nB_0minus)
        r_gamma_0minus = _relativistic_gamma_from_u(u_0minus)
        E = float(h_0minus * u_0minus * r_gamma_0minus)
        Pi = float(h_0minus * u_0minus * u_0minus + P_0minus)

        stats["downstream_root_calls"] += 1
        endpoint = _solve_analytic_inf_endpoint_for_u_0minus(
            u_0minus,
            nuclear_state,
            B_one_forth,
            ms=ms,
            upB=upB,
            initial_guess=endpoint_guess_cache["value"],
        )
        endpoint_guess_cache["value"] = endpoint.get("endpoint_initial_guess", (endpoint["muB_inf"], endpoint["T_inf"]))
        T_inf = float(endpoint["T_inf"])
        thermo_inf = _quark_thermo_state(endpoint["muB_inf"], 0.0, B_one_forth, T_inf, jB, ms=ms, upB=upB)
        E_inf = float(thermo_inf["h"] * thermo_inf["u"] * _relativistic_gamma_from_u(thermo_inf["u"]))
        Pi_inf = float(thermo_inf["h"] * thermo_inf["u"] * thermo_inf["u"] + thermo_inf["P"])

        state_0plus_guess = state_0plus_guess_cache["value"]
        if state_0plus_guess is None:
            state_0plus_guess = (endpoint["muB_inf"], _branch_muK_seed((nB_0minus - thermo_inf["nK"]) / thermo_inf["nB"]), max(T_0plus, T_inf))
        thermo_0plus = _solve_interface_0plus_from_T_0plus_E_and_Pi(
            T_0plus,
            E,
            Pi,
            jB,
            thermo_inf["nB"],
            thermo_inf["nK"],
            B_one_forth,
            ms=ms,
            upB=upB,
            initial_guess=state_0plus_guess,
            stats=stats,
        )
        state_0plus_guess_cache["value"] = (thermo_0plus["muB"], thermo_0plus["muK"], T_0plus)
        a_0plus = float(thermo_0plus["nK"] / thermo_0plus["nB"])
        if not (0.0 < a_0plus < 1.0):
            raise RuntimeError("fixed-T_0plus interface requires 0 < nK_0plus/nB_0plus < 1")

        micro_inf = _microphysics_from_quark_state_energy(thermo_inf["muB"], thermo_inf["T"])
        if micro_inf["invD"] <= 0.0:
            raise RuntimeError("uNmax downstream inverse diffusion coefficient must be positive")
        D_inf = float(1.0 / micro_inf["invD"])
        delta_nK = float(max(1.0e-5 * max(abs(thermo_inf["nK"]), thermo_inf["nB"]), 1.0e-2))
        probe = _solve_local_quark_state_from_nK_E_and_Pi(
            thermo_inf["nK"] + delta_nK,
            E,
            Pi,
            jB,
            B_one_forth,
            ms=ms,
            upB=upB,
            initial_guess=(thermo_inf["muB"], 1.0e-3, thermo_inf["T"]),
            T_ref=thermo_inf["T"],
            stats=stats,
        )
        rate_probe = _exact_kaon_transport_rate(
            probe["muB"], probe["muK"], probe["T"], ms=ms, upB=upB
        )["Gamma_K"]
        rate_slope = float(rate_probe / delta_nK)
        if (not np.isfinite(rate_slope)) or rate_slope <= 0.0:
            raise RuntimeError("uNmax downstream exact-rate slope must be positive")
        u_inf = float(thermo_inf["u"])
        lam = float((-u_inf + np.sqrt(u_inf * u_inf + 4.0 * D_inf * rate_slope)) / (2.0 * D_inf))
        if (not np.isfinite(lam)) or lam <= 0.0:
            raise RuntimeError("uNmax downstream tail decay must be positive")

        state = {
            "jB": jB,
            "u_0minus": u_0minus,
            "r_gamma_0minus": r_gamma_0minus,
            "E": E,
            "Pi": Pi,
            "T_inf": T_inf,
            "P_0minus": P_0minus,
            "e_0minus": e_0minus,
            "h_0minus": h_0minus,
            "h_over_nB_0minus": float(h_0minus / nB_0minus),
            "E_0minus": E,
            "thermo_inf": thermo_inf,
            "thermo_0plus": thermo_0plus,
            "E_inf": E_inf,
            "Pi_inf": Pi_inf,
            "endpoint": endpoint,
            "lambda_n": float(nB_0minus / thermo_inf["nB"]),
            "a_0plus": a_0plus,
            "D_inf": D_inf,
            "invD_inf": float(micro_inf["invD"]),
            "rate_slope_inf": rate_slope,
            "lambda": lam,
            "jK_inf": float(u_inf * thermo_inf["nK"]),
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
        # At the singular T_0plus=0 endpoint invD vanishes. During collocation,
        # trial profiles can temporarily overshoot the physical nK interval by
        # many orders of magnitude; snap only those out-of-domain evaluations
        # to the already solved endpoint states. Interior in-domain states use
        # the shared T**2 closure below.
        if exact_zero_left and (
            (s_value is not None and abs(float(s_value)) <= 1.0e-14)
            or float(nK_value) >= float(state["thermo_0plus"]["nK"])
        ):
            return state["thermo_0plus"]
        if exact_zero_left and float(nK_value) <= float(state["thermo_inf"]["nK"]):
            return state["thermo_inf"]
        return _solve_local_quark_state_from_nK_E_and_Pi(
            float(nK_value),
            state["E"],
            state["Pi"],
            state["jB"],
            B_one_forth,
            ms=ms,
            upB=upB,
            initial_guess=initial_guess,
            T_ref=state["T_inf"],
            stats=stats,
        )

    def ode(s_coord, y, p):
        stats["bvp_ode_calls"] += 1
        state = state_or_none(float(p[0]))
        if state is None:
            return np.full_like(y, 1.0e12)
        dyds = np.empty_like(y)
        T_0plus_seed = T_0plus
        guess = (state["thermo_0plus"]["muB"], state["thermo_0plus"]["muK"], T_0plus_seed)
        for i in range(y.shape[1]):
            try:
                nK_value = float(y[0, i]) * bvp_nK_scale
                jK_value = float(y[1, i]) * bvp_jK_scale
                thermo = local_state(nK_value, state, s_value=s_coord[i], initial_guess=guess)
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
        state_0plus = state["thermo_0plus"]
        target = state["thermo_inf"]
        ya_physical = np.array([float(ya[0]) * bvp_nK_scale, float(ya[1]) * bvp_jK_scale])
        yb_physical = np.array([float(yb[0]) * bvp_nK_scale, float(yb[1]) * bvp_jK_scale])
        nK_scale = max(abs(state_0plus["nK"]), abs(target["nK"]), 1.0)
        jK_scale = max(abs(state["jB"]), abs(state["jK_inf"]), 1.0)
        tail_scale = max(jK_scale, (state["D_inf"] * state["lambda"] + abs(target["u"])) * nK_scale)
        return np.array(
            [
                (float(ya_physical[0]) - state_0plus["nK"]) / nK_scale,
                (float(ya_physical[1]) - state["jB"]) / jK_scale,
                _nK_tail_residual(
                    yb_physical,
                    {
                        "nK": target["nK"],
                        "jK": state["jK_inf"],
                        "D": state["D_inf"],
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
            "T_0plus": T_0plus,
            "solver_variant": "energy_conserving_nK_fixed_T_0plus",
            "rate_model": "exact_nonleptonic",
            "composition_definition": "nK_over_local_nB",
            "current_definition": "u_nK_minus_D_dnK_dx",
            "density_ratio_definition": "lambda_n_equals_nB_0minus_over_nB_inf",
            "_root_method": "solve_bvp_nK_jK_parameter_jB",
        }

    bvp_nK_scale = max(abs(state0["thermo_0plus"]["nK"]), abs(state0["thermo_inf"]["nK"]), 1.0)
    bvp_jK_scale = max(abs(state0["jB"]), abs(state0["jK_inf"]), 1.0)

    continuation_profile = None
    if isinstance(continuation_guess, dict):
        try:
            previous_s = np.asarray(continuation_guess["s_coord"], dtype=float)
            previous_nK = np.asarray(continuation_guess["nK"], dtype=float)
            previous_jK = np.asarray(continuation_guess["jK"], dtype=float)
            previous_nK_inf = float(continuation_guess["nK_inf"])
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
                    previous_nK_inf,
                )
        except Exception:
            continuation_profile = None

    s_mesh = np.linspace(0.0, s_end, int(n_mesh))
    blend = s_mesh / s_end
    if exact_zero_left:
        tail_weight = tail_eps + (1.0 - tail_eps) * (1.0 - 3.0 * blend**2 + 2.0 * blend**3)
    else:
        tail_weight = np.maximum(1.0 - s_mesh, tail_eps)
    state_0plus = state0["thermo_0plus"]
    target0 = state0["thermo_inf"]
    nK_guess = target0["nK"] + (state_0plus["nK"] - target0["nK"]) * tail_weight
    jK_tail_weight = np.maximum(1.0 - s_mesh, tail_eps)
    jK_guess = state0["jK_inf"] + (state0["jB"] - state0["jK_inf"]) * jK_tail_weight
    nK_guess[0] = state_0plus["nK"]
    jK_guess[0] = state0["jB"]
    if continuation_profile is not None:
        previous_s, previous_nK, previous_jK, previous_nK_inf = continuation_profile
        previous_delta = previous_nK - previous_nK_inf
        previous_left_delta = float(previous_delta[0])
        new_left_delta = float(state_0plus["nK"] - target0["nK"])
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
        nK_guess[0] = state_0plus["nK"]
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
            "T_0plus": T_0plus,
            "solver_variant": "energy_conserving_nK_fixed_T_0plus",
            "rate_model": "exact_nonleptonic",
            "composition_definition": "nK_over_local_nB",
            "current_definition": "u_nK_minus_D_dnK_dx",
            "density_ratio_definition": "lambda_n_equals_nB_0minus_over_nB_inf",
            "_root_method": "solve_bvp_nK_jK_parameter_jB",
        }

    s_profile = np.asarray(sol.x, dtype=float)
    nK_profile = np.asarray(sol.y[0], dtype=float) * bvp_nK_scale
    jK_profile = np.asarray(sol.y[1], dtype=float) * bvp_jK_scale
    x_profile = -np.log1p(-s_profile) / state["lambda"]
    profile = {key: np.empty_like(s_profile) for key in (
        "nB", "u", "muB", "muK", "T", "P", "h", "r_gamma", "invD", "Gamma_K"
    )}
    T_0plus_seed = T_0plus
    guess = (state["thermo_0plus"]["muB"], state["thermo_0plus"]["muK"], T_0plus_seed)
    try:
        for i, nK_value in enumerate(nK_profile):
            stats["profile_state_calls"] += 1
            thermo = local_state(nK_value, state, s_value=s_profile[i], initial_guess=guess)
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
            "T_0plus": T_0plus,
            "solver_variant": "energy_conserving_nK_fixed_T_0plus",
            "rate_model": "exact_nonleptonic",
            "composition_definition": "nK_over_local_nB",
            "current_definition": "u_nK_minus_D_dnK_dx",
            "density_ratio_definition": "lambda_n_equals_nB_0minus_over_nB_inf",
            "_root_method": "solve_bvp_nK_jK_parameter_jB",
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
    T_0plus_residual = float(state["thermo_0plus"]["T"] - T_0plus)
    E_right_residual = float(state["E_inf"] - state["E"])
    E_right_residual_norm = float(E_right_residual / max(abs(state["E"]), abs(state["E_inf"]), 1.0))
    Pi_right_residual = float(state["Pi_inf"] - state["Pi"])
    Pi_right_residual_norm = float(Pi_right_residual / max(abs(state["Pi"]), abs(state["Pi_inf"]), 1.0))
    collocation_status_acceptable = _u0minus_max_collocation_status_is_acceptable(
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
        and abs(T_0plus_residual) <= max(1.0e-10, 1.0e-8 * max(1.0, abs(T_0plus)))
    )

    target = state["thermo_inf"]
    state_0plus = state["thermo_0plus"]
    jK_left_residual = float(jK_profile[0] - state["jB"])
    jK_left_scale = max(abs(state["jB"]), abs(state["jK_inf"]), 1.0)
    tail_residual = _nK_tail_residual(
        np.array([nK_profile[-1], jK_profile[-1]], dtype=float),
        {
            "nK": target["nK"],
            "jK": state["jK_inf"],
            "D": state["D_inf"],
            "lambda": state["lambda"],
            "u": target["u"],
        },
    )
    accepted_max_nodes = bool(success and not sol.success and int(sol.status) == 1)
    if success and accepted_max_nodes:
        result_message = (
            "Absolute-nK uNmax BVP reached max_nodes at the exact T_0plus=0 "
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
        "u_0minus": float(state["u_0minus"]),
        "u_0plus": float(state_0plus["u"]),
        "u_inf": float(target["u"]),
        "T_0minus": T_0minus,
        "T_inf": float(state["T_inf"]),
        "T_0plus": float(state_0plus["T"]),
        "T_0plus_target": T_0plus,
        "T_0plus_residual": T_0plus_residual,
        "E": float(state["E"]),
        "Pi": float(state["Pi"]),
        "E_0minus": float(state["E_0minus"]),
        "E_inf": float(state["E_inf"]),
        "E_0plus": float(state_0plus["E"]),
        "E_right_residual": E_right_residual,
        "E_right_residual_norm": E_right_residual_norm,
        "Pi_inf": float(state["Pi_inf"]),
        "Pi_right_residual": Pi_right_residual,
        "Pi_right_residual_norm": Pi_right_residual_norm,
        "a_0plus": float(a_profile[0]),
        "a_inf": float(target["nK"] / target["nB"]),
        "lambda_n": float(state["lambda_n"]),
        "r_gamma_0minus": float(state["r_gamma_0minus"]),
        "r_gamma_inf": float(_relativistic_gamma_from_u(target["u"])),
        "r_gamma_0plus": float(_relativistic_gamma_from_u(state_0plus["u"])),
        "nB_0minus": nB_0minus,
        "P_0minus": P_0minus,
        "e_0minus": e_0minus,
        "muB_inf": float(target["muB"]),
        "muK_inf": float(target["muK"]),
        "nB_inf": float(target["nB"]),
        "nK_inf": float(target["nK"]),
        "P_inf": float(target["P"]),
        "e_inf": float(target["e"]),
        "jK_inf": float(state["jK_inf"]),
        "muB_0plus": float(state_0plus["muB"]),
        "muK_0plus": float(state_0plus["muK"]),
        "nB_0plus": float(state_0plus["nB"]),
        "nK_0plus": float(state_0plus["nK"]),
        "P_0plus": float(state_0plus["P"]),
        "e_0plus": float(state_0plus["e"]),
        "jK_0plus": float(jK_profile[0]),
        "h_0minus": h_0minus,
        "h_inf": float(target["h"]),
        "h_0plus": float(state_0plus["h"]),
        "h_over_nB_0minus": float(h_0minus / nB_0minus),
        "h_over_nB_0plus": float(state_0plus["h"] / state_0plus["nB"]),
        "h_over_nB_jump_residual": float(
            state_0plus["h"] / state_0plus["nB"] - h_0minus / nB_0minus
        ),
        "invD_inf": float(state["invD_inf"]),
        "rate_slope_inf": float(state["rate_slope_inf"]),
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
        "density_ratio_definition": "lambda_n_equals_nB_0minus_over_nB_inf",
        "solver_variant": "energy_conserving_nK_fixed_T_0plus",
        "coordinate": "BVP: s_coord in [0, 1-tail_eps], s_coord=1-exp(-lambda*x)",
        "tail_eps": float(tail_eps),
        "_root_method": "solve_bvp_nK_jK_parameter_jB",
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
            f"uNmax-nK jB={result['jB']:.6g}, T_inf={result['T_inf']:.6g}, "
            f"T_0plus={T_0plus:.6g}, a_0plus={result['a_0plus']:.6g}, "
            f"tail_norm={result['tail_residual_norm']:.6g}, "
            f"kaon_eq_norm={kaon_residual_norm:.6g}, status={sol.status}, success={success}"
        )
    return result


def _annotate_energy_uNmax_result(
    result,
    T_0plus_target,
):
    out = dict(result)
    T_0plus_target = float(T_0plus_target)
    T_0plus = float(out.get("T_0plus", np.nan))
    out["solver_variant"] = "energy_conserving_nK_fixed_T_0plus"
    out["_root_method"] = "solve_bvp_nK_jK_parameter_jB"
    out["T_0plus_target"] = T_0plus_target
    out["T_0plus_residual"] = (
        float(T_0plus - T_0plus_target)
        if np.isfinite(T_0plus) and np.isfinite(T_0plus_target)
        else np.nan
    )
    return out


def solve_front_energy_conserving_uNmax(
    T_0minus,
    nB_0minus,
    B_one_forth,
    T_0plus=0.5,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    T_inf_guess=None,
    jB_bounds=None,
    return_profile=False,
    verb=False,
    continuation_guess=None,
):
    """
    Solve the energy-conserving front with the interface temperature fixed.

    This variant treats T_0plus as the input and derives a_0plus from the
    quark EOS, which is useful for probing the upper end of the branch where
    T_0plus approaches zero. It is not a separate optimizer: u_0minus is the
    upstream velocity for the requested fixed T_0plus.
    """
    T_0plus = float(T_0plus)
    if (not np.isfinite(T_0plus)) or T_0plus < 0.0:
        raise RuntimeError("T_0plus must be non-negative")

    direct_jB_guess = jB_guess
    if direct_jB_guess is None:
        direct_jB_guess = _default_energy_jB_guess(nB_0minus)

    direct_result = _solve_front_energy_conserving_uNmax_once(
        T_0minus,
        nB_0minus,
        B_one_forth,
        T_0plus,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=n_mesh,
        tol_bvp=tol_bvp,
        max_nodes=max_nodes,
        jB_guess=direct_jB_guess,
        T_inf_guess=T_inf_guess,
        jB_bounds=jB_bounds,
        return_profile=return_profile,
        verb=verb,
        continuation_guess=continuation_guess,
    )
    result_out = _annotate_energy_uNmax_result(direct_result, T_0plus)
    if return_profile:
        return result_out
    return _strip_energy_profile_fields(result_out)


# ---------------------------------------------------------------------------
# Thermal-conducting steady front
#
# The ideal-fluid solvers above carry the energy flux as an algebraic first
# integral E = h*gamma*u, which lets them recover T pointwise from (nK, E, Pi).
# Once Fourier conduction is present the conserved quantity is instead
#
#     F_E = h*gamma*u - kappa_th*dT/dx = const,
#
# which contains dT/dx and is therefore no longer an algebraic relation for T:
# it *is* the ODE for T. So T is promoted from a reconstructed quantity to a
# propagated field, and the local closure drops from 3x3 (muB, muK, T) to 2x2
# (muB, muK) at prescribed T. Nothing in the ideal-fluid path is reused for the
# energy constraint; only the muK=0 downstream endpoint solve is shared, and
# that is exact because q_th(inf) = 0 makes h*gamma*u|_Q = F_E.
# ---------------------------------------------------------------------------

_THERMAL_ZETA3 = 1.2020569031595943
# The FULL Heiselberg-Pethick collision integral (PRD 48, 2916 (1993) Eq. (59),
# with the RPA susceptibilities of Eq. (7)) is implemented in
# _ikappa_quadrature below and used by default; the Eq. (60) asymptotes are
# kept only as validation targets. In the physical domain of this problem
# (mu_q ~ 300-450 MeV, T <~ 55 MeV, so y = T/qD <~ 0.19) the full integral runs
# up to ~16% ABOVE the low-y asymptote near y ~ 0.03, i.e. kappa_th is up to
# ~16% smaller than the asymptote would give -- not a negligible correction.
# Upper end of the tabulated I_kappa range. y = T/qD ~ 50 corresponds to
# T ~ 14 GeV, far outside anything physical here, so exceeding it is an error
# rather than something to extrapolate through.
_THERMAL_Y_MAX = 50.0
_THERMAL_LOW_Y_COEFF = 2.0 * _THERMAL_ZETA3
# Below this the exact low-y limit is used. At y = 1e-6 the converged
# quadrature exceeds 2*zeta(3)*y**2 by only 0.055%, so the seam is negligible;
# the relative correction falls off as y**(2/3), matching the (T/qD)**(8/3)
# term HP93 quotes below Eq. (60).
_THERMAL_IKAPPA_Y_MIN = 1.0e-6
_THERMAL_IKAPPA_TABLE_POINTS = 221
_THERMAL_IKAPPA_QUAD_NX = 400
_THERMAL_IKAPPA_QUAD_NT = 400
_THERMAL_IKAPPA_CACHE = {}
# Safety stop shared by the adaptive continuations in T_0plus and tail_eps.
# Failed temperature steps are bisected and failed tail steps are retried at a
# geometric midpoint, so this bounds total BVP solves rather than waypoints.
_THERMAL_MAX_CONTINUATION_ATTEMPTS = 40
# Below this downstream gap, the deviation is at the accuracy floor of the
# nonlinear EOS closure. Blend onto the downstream Jacobian used to formulate
# the asymptotic boundary conditions, then validate against the full RHS.
_THERMAL_LINEAR_TAIL_GAP = 1.0e-4


def _thermal_compact_mesh(n_mesh, tail_eps):
    """Return a fixed compact-s mesh resolving both the interface and tail."""
    n_mesh = int(n_mesh)
    tail_eps = float(tail_eps)
    if n_mesh < 5:
        raise RuntimeError("thermal compact mesh requires at least five points")
    if not (0.0 < tail_eps < 1.0):
        raise RuntimeError("thermal compact mesh requires 0 < tail_eps < 1")
    s_end = 1.0 - tail_eps
    if s_end <= 0.9:
        return np.linspace(0.0, s_end, n_mesh)
    n_tail = max(12, n_mesh // 3)
    n_core = n_mesh - n_tail + 1
    core_t = np.linspace(0.0, 1.0, n_core)
    core = 0.9 * np.expm1(4.0 * core_t) / np.expm1(4.0)
    tail = 1.0 - np.geomspace(0.1, tail_eps, n_tail)
    mesh = np.unique(np.concatenate((core, tail)))
    mesh[0] = 0.0
    mesh[-1] = s_end
    if np.any(np.diff(mesh) <= 0.0):
        raise RuntimeError("thermal compact mesh is not strictly increasing")
    return mesh


def _thermal_tail_schedule(tail_eps):
    """Continuation depths ending at the exact requested compact tail."""
    tail_eps = float(tail_eps)
    if not (0.0 < tail_eps < 1.0):
        raise RuntimeError("thermal tail schedule requires 0 < tail_eps < 1")
    first = max(tail_eps, 1.0e-4)
    schedule = [float(first)]
    while schedule[-1] > tail_eps:
        next_eps = max(tail_eps, float(f"{schedule[-1] * 1.0e-2:.15g}"))
        if next_eps == schedule[-1]:
            break
        schedule.append(float(next_eps))
    return schedule


def _thermal_interpolate_profile_guess(profile_guess, s_mesh, equilibrium_y, scales):
    """Map a physical compact profile to a new mesh and extend its stable tail."""
    previous_s = np.asarray(profile_guess["s"], dtype=float)
    previous_y = np.asarray(profile_guess["physical_y"], dtype=float)
    previous_eq = np.asarray(profile_guess["equilibrium_y"], dtype=float)
    s_mesh = np.asarray(s_mesh, dtype=float)
    equilibrium_y = np.asarray(equilibrium_y, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if previous_s.ndim != 1 or previous_s.size < 2 or np.any(np.diff(previous_s) <= 0.0):
        raise RuntimeError("thermal continuation profile has invalid compact coordinate")
    if previous_y.shape != (3, previous_s.size):
        raise RuntimeError("thermal continuation profile has invalid shape")
    if previous_eq.shape != (3,) or equilibrium_y.shape != (3,) or scales.shape != (3,):
        raise RuntimeError("thermal continuation equilibrium or scale has invalid shape")
    if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
        raise RuntimeError("thermal continuation scales must be positive and finite")
    old_end = float(previous_s[-1])
    old_gap = max(1.0 - old_end, np.finfo(float).tiny)
    physical_guess = np.empty((3, s_mesh.size), dtype=float)
    for row in range(3):
        old_delta = previous_y[row] - previous_eq[row]
        inside = s_mesh <= old_end
        physical_guess[row, inside] = equilibrium_y[row] + np.interp(
            s_mesh[inside], previous_s, old_delta
        )
        outside = ~inside
        physical_guess[row, outside] = equilibrium_y[row] + old_delta[-1] * (
            (1.0 - s_mesh[outside]) / old_gap
        )
    return physical_guess / scales[:, None]


def _ikappa_low_y(y):
    """
    Low-argument asymptote of the Heiselberg-Pethick thermal collision integral,
    I_kappa(y) -> 2*zeta(3)*y**2 as y -> 0, with y = T/qD (HP93 Eq. (60)).
    """
    y = float(y)
    if (not np.isfinite(y)) or y < 0.0:
        raise RuntimeError("I_kappa requires a finite non-negative argument")
    return float(_THERMAL_LOW_Y_COEFF * y * y)


def _ikappa_high_y(y):
    """High-argument asymptote of I_kappa, ln(y)/3 + 0.30 (HP93 Eq. (60))."""
    y = float(y)
    if (not np.isfinite(y)) or y <= 0.0:
        raise RuntimeError("I_kappa high-y asymptote requires a positive argument")
    return float(np.log(y) / 3.0 + 0.30)


def _ikappa_chi(x):
    """
    RPA longitudinal/transverse susceptibilities, HP93 Eq. (7):

        chi_l(x) = 1 - (x/2)*L(x)
        chi_t(x) = x**2/2 + (x*(1 - x**2)/4)*L(x)

    with x = omega/q and L(x) = ln((1+x)/(1-x)) + i*pi. The branch is fixed by
    HP93 Eq. (8), which requires chi_l -> 1 + O(x) and chi_t -> i*(pi/4)*x;
    the opposite branch flips the sign of Im(chi_t) and disagrees with the
    paper's own small-x reduction in Appendix B.
    """
    L = np.log((1.0 + x) / (1.0 - x)) + 1j * np.pi
    chi_l = 1.0 - 0.5 * x * L
    chi_t = 0.5 * x * x + 0.25 * x * (1.0 - x * x) * L
    return chi_l, chi_t


def _ikappa_quadrature(y, nx=_THERMAL_IKAPPA_QUAD_NX, nt=_THERMAL_IKAPPA_QUAD_NT):
    """
    Full HP93 Eq. (59) thermal collision integral by two-dimensional quadrature.

        I_k = Int_0^inf (dw/w) ((w/2T)/sinh(w/2T))**2 Int_0^1 dx
              Int_0^2pi (dphi/2pi) x**2 (1 - x**2)(1 - cos phi)
              * | A - B cos phi |**2

        A = 1/(1 + (x*qD/w)**2 * chi_l(x))
        B = 1/(1 + (x*qD/w)**2 * chi_t(x)/(1 - x**2))

    The azimuthal average is done analytically. With <1> = 1, <cos> = 0,
    <cos**2> = 1/2 and <cos**3> = 0,

        <(1 - cos phi)|A - B cos phi|**2> = |A|**2 + |B|**2/2 + Re(A conj(B)),

    which reduces to the momentum-relaxation integrand of Eq. (21) when the
    (1 - cos phi) weight is dropped, as it must.

    Substituting u = w/(2T) makes the integral a function of y = T/qD alone,
    because (x*qD/w)**2 = x**2/(4*u**2*y**2) and dw/w = du/u. The u integral is
    then done on a logarithmic grid, where dw/w = dt for u = exp(t).
    """
    y = float(y)
    if (not np.isfinite(y)) or y <= 0.0:
        raise RuntimeError("I_kappa quadrature requires y > 0")

    key = (int(nx), int(nt))
    if key not in _THERMAL_IKAPPA_CACHE:
        gx, gw = np.polynomial.legendre.leggauss(int(nx))
        x_nodes = np.clip(0.5 * (gx + 1.0), 1.0e-12, 1.0 - 1.0e-12)
        x_weights = 0.5 * gw
        chi_l, chi_t = _ikappa_chi(x_nodes)
        t_lo, t_hi = -22.0, 5.0
        gt, gwt = np.polynomial.legendre.leggauss(int(nt))
        t_nodes = 0.5 * (t_hi - t_lo) * gt + 0.5 * (t_hi + t_lo)
        t_weights = 0.5 * (t_hi - t_lo) * gwt
        u_nodes = np.exp(t_nodes)
        with np.errstate(over="ignore", invalid="ignore"):
            bose = np.where(
                u_nodes < 1.0e-8,
                1.0,
                (u_nodes / np.sinh(np.clip(u_nodes, 1.0e-300, 700.0))) ** 2,
            )
        bose = np.where(u_nodes > 700.0, 0.0, bose)
        _THERMAL_IKAPPA_CACHE[key] = {
            "x": x_nodes,
            "wx": x_weights * x_nodes**2 * (1.0 - x_nodes**2),
            "chi_l": chi_l,
            "chi_t_over": chi_t / (1.0 - x_nodes**2),
            "u2": u_nodes**2,
            "wt_bose": t_weights * bose,
        }
    grid = _THERMAL_IKAPPA_CACHE[key]

    # At small y and small omega the screening factor s is astronomically large,
    # so A and B underflow to zero. That is the physically correct answer (the
    # interaction is fully screened) but it generates spurious overflow/invalid
    # warnings on the way; suppress them and reject any genuinely non-finite
    # contribution instead.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        s = (grid["x"][None, :] ** 2) / (4.0 * grid["u2"][:, None] * y * y)
        A = 1.0 / (1.0 + s * grid["chi_l"][None, :])
        B = 1.0 / (1.0 + s * grid["chi_t_over"][None, :])
        phi_avg = np.abs(A) ** 2 + 0.5 * np.abs(B) ** 2 + np.real(A * np.conj(B))
        phi_avg = np.where(np.isfinite(phi_avg), phi_avg, 0.0)
        inner = phi_avg @ grid["wx"]
        value = float(grid["wt_bose"] @ inner)
    if (not np.isfinite(value)) or value <= 0.0:
        raise RuntimeError(f"I_kappa quadrature returned a non-physical value at y={y:.6g}")
    return value


def _ikappa_table():
    """Lazily built log-log interpolation table for the full Eq. (59) integral."""
    if "table" not in _THERMAL_IKAPPA_CACHE:
        log_y = np.linspace(
            np.log(_THERMAL_IKAPPA_Y_MIN),
            np.log(_THERMAL_Y_MAX),
            int(_THERMAL_IKAPPA_TABLE_POINTS),
        )
        # Tabulate I_kappa / y**2: that ratio tends to the finite constant
        # 2*zeta(3) as y -> 0, so the interpolation stays well conditioned and
        # joins the analytic low-y limit smoothly.
        ratio = np.array(
            [_ikappa_quadrature(float(np.exp(v))) / float(np.exp(v)) ** 2 for v in log_y]
        )
        _THERMAL_IKAPPA_CACHE["table"] = CubicSpline(log_y, np.log(ratio))
    return _THERMAL_IKAPPA_CACHE["table"]


def _ikappa_full(y):
    """
    Full HP93 Eq. (59) I_kappa(y), interpolated from cached quadrature.

    Below _THERMAL_IKAPPA_Y_MIN the exact low-y limit is used; there the
    quadrature and the asymptote agree to better than 0.2%.
    """
    y = float(y)
    if (not np.isfinite(y)) or y < 0.0:
        raise RuntimeError("I_kappa requires a finite non-negative argument")
    if y == 0.0:
        return 0.0
    if y <= _THERMAL_IKAPPA_Y_MIN:
        return float(_THERMAL_LOW_Y_COEFF * y * y)
    if y > _THERMAL_Y_MAX:
        raise RuntimeError(
            f"I_kappa argument y = {y:.6g} exceeds the tabulated range "
            f"y_max = {_THERMAL_Y_MAX:.6g}"
        )
    spline = _ikappa_table()
    value = float(np.exp(float(spline(np.log(y)))) * y * y)
    if (not np.isfinite(value)) or value <= 0.0:
        raise RuntimeError("I_kappa interpolation returned a non-physical value")
    return value


_THERMAL_DEFAULT_KAPPA_MODEL = {
    "name": "HP93_full_Eq59",
    "I_kappa": _ikappa_full,
    "low_y_coeff": _THERMAL_LOW_Y_COEFF,
    "y_max": _THERMAL_Y_MAX,
}


def _normalize_kappa_model(kappa_model):
    """Validate and fill in a thermal-conductivity model description."""
    if kappa_model is None:
        return _THERMAL_DEFAULT_KAPPA_MODEL
    if not isinstance(kappa_model, dict):
        raise RuntimeError("kappa_model must be a dict or None")
    model = dict(_THERMAL_DEFAULT_KAPPA_MODEL)
    model.update(kappa_model)
    if not callable(model.get("I_kappa")):
        raise RuntimeError("kappa_model['I_kappa'] must be callable")
    low_y = float(model.get("low_y_coeff", np.nan))
    if (not np.isfinite(low_y)) or low_y <= 0.0:
        raise RuntimeError(
            "kappa_model['low_y_coeff'] must be positive: it is the coefficient C in "
            "I_kappa(y) -> C*y**2, which supplies the exact T -> 0 limit and is the only "
            "way to evaluate kappa_th at T = 0 without forming 0/0"
        )
    y_max = float(model.get("y_max", np.nan))
    if (not np.isfinite(y_max)) or y_max <= 0.0:
        raise RuntimeError("kappa_model['y_max'] must be positive")
    model["low_y_coeff"] = low_y
    model["y_max"] = y_max
    return model


def _thermal_conductivity(muB, T, kappa_model=None, v_F=1.0):
    """
    Quark thermal conductivity in natural MeV**2.

        kappa_th = pi**3 * v_F**2 * T**2 / (24 * alpha_s**2 * I_kappa(T/qD))

    At exactly T = 0 that expression is 0/0. It is never evaluated there:
    with I_kappa(y) -> C*y**2 the limit is exact and finite,

        kappa_th(0+) = pi**3 * v_F**2 * qD**2 / (24 * alpha_s**2 * C),

    which for C = 2*zeta(3) is pi**3*v_F**2*qD**2/(48*zeta(3)*alpha_s**2).
    kappa_th(0) is finite and positive, never zero.
    """
    model = _normalize_kappa_model(kappa_model)
    T = float(T)
    if (not np.isfinite(T)) or T < 0.0:
        raise RuntimeError("Thermal conductivity requires finite T >= 0")
    muQ = float(muB) / 3.0
    if (not np.isfinite(muQ)) or muQ <= 0.0:
        raise RuntimeError("Thermal conductivity requires muQ > 0")

    qD = float(_TRANSPORT_QD_COEFF * muQ)
    if (not np.isfinite(qD)) or qD <= 0.0:
        raise RuntimeError("Thermal conductivity requires a positive screening scale")

    y = float(T / qD)
    if y > model["y_max"]:
        raise RuntimeError(
            f"Thermal conductivity argument y = T/qD = {y:.6g} exceeds the validated band "
            f"y_max = {model['y_max']:.6g}. The full Heiselberg-Pethick collision integral "
            "is not implemented (see _THERMAL_Y_MAX); refusing to extrapolate."
        )

    prefactor = np.pi**3 * float(v_F) ** 2 / (24.0 * _TRANSPORT_ALPHA_S**2)
    if T == 0.0:
        kappa = float(prefactor * qD * qD / model["low_y_coeff"])
    else:
        I_val = float(model["I_kappa"](y))
        if (not np.isfinite(I_val)) or I_val <= 0.0:
            raise RuntimeError("I_kappa returned a non-physical value")
        kappa = float(prefactor * T * T / I_val)
    if (not np.isfinite(kappa)) or kappa <= 0.0:
        raise RuntimeError("Thermal conductivity is non-physical")

    return {
        "kappa_th": kappa,
        "qD": qD,
        "muQ": muQ,
        "y": y,
        "model": model["name"],
        "y_max": model["y_max"],
    }


def _solve_local_quark_state_from_nK_T_and_Pi(
    nK_target,
    T,
    Pi,
    jB,
    B_one_forth,
    ms=0.0,
    upB=5000,
    initial_guess=None,
    stats=None,
):
    """
    Thermal-conducting local closure: solve (muB, muK) at a PRESCRIBED T.

    Two unknowns, two equations:

        nK_QM(muB, muK, T)                      = nK_target
        P_QM(muB, muK, T) + h*u**2              = Pi,    u = jB / nB_QM(muB, muK, T)

    T is an ODE variable here, not a root unknown. That is the whole point: the
    ideal-fluid closure needs a w = T**2 reparameterisation because its Jacobian
    degenerates in T as T -> 0, whereas this system stays regular at exactly
    T = 0 and needs no such trick.
    """
    if stats is not None:
        stats["thermal_local_state_calls"] = stats.get("thermal_local_state_calls", 0) + 1
    nK_target = float(nK_target)
    T = float(T)
    Pi = float(Pi)
    jB = float(jB)
    if not np.all(np.isfinite([nK_target, T, Pi, jB])):
        raise RuntimeError("Thermal local closure requires finite (nK, T, Pi, jB)")
    if T < 0.0:
        raise RuntimeError("Thermal local closure requires T >= 0")
    if jB <= 0.0:
        raise RuntimeError("Thermal local closure requires jB > 0")

    allow_zero = bool(T == 0.0)
    nK_scale = max(abs(nK_target), 1.0)
    pi_scale = max(abs(Pi), 1.0)

    guesses = []
    if initial_guess is not None:
        guess = np.asarray(initial_guess, dtype=float).ravel()
        if guess.size >= 2 and np.all(np.isfinite(guess[:2])):
            guesses.append((float(guess[0]), float(guess[1])))
    guesses.extend([(1100.0, 20.0), (1300.0, 60.0), (950.0, 5.0), (1500.0, 150.0)])

    def residual(vec):
        muB_val = float(vec[0])
        muK_val = float(vec[1])
        if muB_val <= 0.0 or (not np.isfinite(muB_val)) or (not np.isfinite(muK_val)):
            return np.array([1.0e12, 1.0e12], dtype=float)
        try:
            thermo = _quark_thermo_state(
                muB_val,
                muK_val,
                B_one_forth,
                T,
                jB,
                ms=ms,
                upB=upB,
                allow_zero_temperature=allow_zero,
            )
            return np.array(
                [
                    (thermo["nK"] - nK_target) / nK_scale,
                    (thermo["Pi"] - Pi) / pi_scale,
                ],
                dtype=float,
            )
        except Exception:
            return np.array([1.0e12, 1.0e12], dtype=float)

    best = None
    best_norm = np.inf
    best_message = "Thermal local closure did not converge"
    for muB_guess, muK_guess in guesses:
        try:
            if stats is not None:
                stats["thermal_local_root_calls"] = stats.get("thermal_local_root_calls", 0) + 1
            sol = root(
                residual,
                np.array([muB_guess, muK_guess], dtype=float),
                method="hybr",
                options={"maxfev": 300, "xtol": 1.0e-12},
            )
            if not np.all(np.isfinite(sol.x)):
                continue
            norm = float(np.linalg.norm(residual(sol.x), ord=np.inf))
            if norm < best_norm:
                best_norm = norm
                best = sol.x.copy()
            if sol.success and norm <= 1.0e-10:
                break
            best_message = str(sol.message)
        except Exception as exc:
            best_message = str(exc)

    if best is None or best_norm > 1.0e-8:
        raise RuntimeError(f"{best_message}; best scaled residual={best_norm:.3e}")

    thermo = _quark_thermo_state(
        float(best[0]),
        float(best[1]),
        B_one_forth,
        T,
        jB,
        ms=ms,
        upB=upB,
        allow_zero_temperature=allow_zero,
    )
    thermo["r_gamma"] = _relativistic_gamma_from_u(thermo["u"])
    thermo["E_flux"] = float(thermo["h"] * thermo["u"] * thermo["r_gamma"])
    thermo["closure_residual"] = float(best_norm)
    return thermo


def _thermal_downstream_analysis(
    jB,
    nuclear_state,
    B_one_forth,
    ms=0.0,
    upB=5000,
    kappa_model=None,
    endpoint_guess=None,
):
    """
    Linearise the coupled [nK, jK, T] system about the muK=0 downstream state.

    With a = invD*u, b = invD, c = dGamma_K/dnK, d = dGamma_K/dT,
    e = dE_flux/dnK / kappa, f = dE_flux/dT / kappa,

        J = [[ a, -b,  0],
             [-c,  0, -d],
             [ e,  0,  f]]      det J = b*(d*e - c*f),   trace J = a + f.

    The boundary-condition count closes only when the stable manifold has
    dimension 1, i.e. det J < 0 and trace J > 0 (equivalently c*f > d*e with
    f > 0). The caller must check ``stable_dimension``; this helper reports it
    rather than asserting, so the failure is visible instead of silent.

    The partial derivatives are taken *through the same 2x2 closure* the ODE
    uses, so the linearisation can never drift from the transported system.
    """
    jB = float(jB)
    nB_0minus = float(nuclear_state["nB_0minus"])
    u_0minus = float(jB / nB_0minus)
    r_gamma_0minus = _relativistic_gamma_from_u(u_0minus)
    Pi = float(nuclear_state["P_0minus"] + nuclear_state["h_0minus"] * u_0minus * u_0minus)
    F_E = float(nuclear_state["h_0minus"] * u_0minus * r_gamma_0minus)

    endpoint = _solve_analytic_inf_endpoint_for_u_0minus(
        u_0minus, nuclear_state, B_one_forth, ms=ms, upB=upB, initial_guess=endpoint_guess
    )
    muB_inf = float(endpoint["muB_inf"])
    T_inf = float(endpoint["T_inf"])
    if (not np.isfinite(T_inf)) or T_inf <= 0.0:
        raise RuntimeError("Thermal downstream endpoint returned a non-physical temperature")

    thermo_inf = _quark_thermo_state(muB_inf, 0.0, B_one_forth, T_inf, jB, ms=ms, upB=upB)
    nK_inf = float(thermo_inf["nK"])
    u_inf = float(thermo_inf["u"])
    micro_inf = _microphysics_from_quark_state_energy(muB_inf, T_inf)
    invD_inf = float(micro_inf["invD"])
    if invD_inf <= 0.0:
        raise RuntimeError("Thermal downstream inverse diffusion coefficient must be positive")
    kappa_info = _thermal_conductivity(muB_inf, T_inf, kappa_model=kappa_model)
    kappa_inf = float(kappa_info["kappa_th"])

    guess = (muB_inf, 1.0e-4)

    def rate_and_flux(nK_value, T_value):
        state = _solve_local_quark_state_from_nK_T_and_Pi(
            nK_value, T_value, Pi, jB, B_one_forth, ms=ms, upB=upB, initial_guess=guess
        )
        rate = float(
            _exact_kaon_transport_rate(
                state["muB"], state["muK"], state["T"], ms=ms, upB=upB
            )["Gamma_K"]
        )
        return rate, float(state["E_flux"])

    d_nK = max(1.0e-6 * abs(nK_inf), 1.0e-4)
    d_T = max(1.0e-6 * T_inf, 1.0e-6)
    rate_p, flux_p = rate_and_flux(nK_inf + d_nK, T_inf)
    rate_m, flux_m = rate_and_flux(nK_inf - d_nK, T_inf)
    c_coeff = float((rate_p - rate_m) / (2.0 * d_nK))
    e_coeff = float((flux_p - flux_m) / (2.0 * d_nK) / kappa_inf)
    rate_p, flux_p = rate_and_flux(nK_inf, T_inf + d_T)
    rate_m, flux_m = rate_and_flux(nK_inf, T_inf - d_T)
    d_coeff = float((rate_p - rate_m) / (2.0 * d_T))
    f_coeff = float((flux_p - flux_m) / (2.0 * d_T) / kappa_inf)

    a_coeff = float(invD_inf * u_inf)
    b_coeff = float(invD_inf)
    jacobian = np.array(
        [
            [a_coeff, -b_coeff, 0.0],
            [-c_coeff, 0.0, -d_coeff],
            [e_coeff, 0.0, f_coeff],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(jacobian)):
        raise RuntimeError("Thermal downstream Jacobian is non-finite")

    eigenvalues, right_vectors = np.linalg.eig(jacobian)
    stable_mask = eigenvalues.real < 0.0
    stable_dimension = int(np.sum(stable_mask))

    lambda_stable = np.nan
    if stable_dimension >= 1:
        lambda_stable = float(np.min(np.abs(eigenvalues.real[stable_mask])))

    # Left eigenvectors (rows of inv(V)) are what project a deviation onto the
    # eigen-directions: with J = V diag(lam) V^-1, the component along v_i is
    # (V^-1 delta)_i. Killing the growing components is (V^-1 delta)_i = 0.
    growing_left = np.zeros((0, 3), dtype=float)
    try:
        v_inverse = np.linalg.inv(right_vectors)
        growing_rows = v_inverse[~stable_mask, :]
        if growing_rows.shape[0] == 2:
            if np.max(np.abs(growing_rows.imag)) > 1.0e-10 * max(
                1.0, float(np.max(np.abs(growing_rows.real)))
            ):
                # Complex-conjugate growing pair: the two real conditions are the
                # real and imaginary parts of a single complex projection.
                growing_left = np.vstack([growing_rows[0].real, growing_rows[0].imag])
            else:
                growing_left = growing_rows.real.copy()
    except np.linalg.LinAlgError:
        growing_left = np.zeros((0, 3), dtype=float)

    return {
        "jB": jB,
        "u_0minus": u_0minus,
        "r_gamma_0minus": r_gamma_0minus,
        "Pi": Pi,
        "F_E": F_E,
        "endpoint": endpoint,
        "muB_inf": muB_inf,
        "T_inf": T_inf,
        "thermo_inf": thermo_inf,
        "nK_inf": nK_inf,
        "u_inf": u_inf,
        "jK_inf": float(u_inf * nK_inf),
        "invD_inf": invD_inf,
        "kappa_inf": kappa_inf,
        "kappa_info": kappa_info,
        "a": a_coeff,
        "b": b_coeff,
        "c": c_coeff,
        "d": d_coeff,
        "e": e_coeff,
        "f": f_coeff,
        "cf_minus_de": float(c_coeff * f_coeff - d_coeff * e_coeff),
        "jacobian": jacobian,
        "eigenvalues": eigenvalues,
        "determinant": float(np.linalg.det(jacobian)),
        "trace": float(np.trace(jacobian)),
        "stable_dimension": stable_dimension,
        "lambda_stable": lambda_stable,
        "growing_left_vectors": growing_left,
        "lambda_n": float(nB_0minus / float(thermo_inf["nB"])),
    }


def _thermal_scaled_growing_projections(jacobian, scales, stable_mask_hint=None):
    """
    Growing-mode left projections in SCALED coordinates.

    The three fields carry different units (nK ~ MeV**3, jK ~ MeV**3, T ~ MeV),
    so projecting a physical deviation onto raw eigenvectors is badly
    conditioned. Under y_i = phys_i / S_i the Jacobian maps to
    diag(1/S) J diag(S), a similarity transform that leaves the eigenvalues
    alone, and the projections then act on already-normalised deviations.
    """
    scales = np.asarray(scales, dtype=float)
    if scales.shape != (3,) or np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
        raise RuntimeError("Thermal BVP scales must be three positive finite numbers")
    scaled = (jacobian * scales[None, :]) / scales[:, None]
    eigenvalues, right_vectors = np.linalg.eig(scaled)
    stable_mask = eigenvalues.real < 0.0
    if int(np.sum(stable_mask)) != 1:
        raise RuntimeError(
            "Thermal downstream stable manifold must have dimension 1 for the boundary "
            f"conditions to close; got {int(np.sum(stable_mask))} with eigenvalues "
            f"{np.sort_complex(eigenvalues)!r}"
        )
    v_inverse = np.linalg.inv(right_vectors)
    growing_rows = v_inverse[~stable_mask, :]
    if growing_rows.shape[0] != 2:
        raise RuntimeError("Expected exactly two growing downstream modes")
    imag_scale = max(1.0, float(np.max(np.abs(growing_rows.real))))
    if np.max(np.abs(growing_rows.imag)) > 1.0e-10 * imag_scale:
        projections = np.vstack([growing_rows[0].real, growing_rows[0].imag])
    else:
        projections = growing_rows.real.copy()
    lambda_stable = float(np.min(np.abs(eigenvalues.real[stable_mask])))
    return np.ascontiguousarray(projections, dtype=float), lambda_stable


def _thermal_seed_from_ideal_solver(
    T, nB_0minus, B_one_forth, ms=0.0, param=para.paraQMCRMF3, NM_type="PNM", T_0plus_seed=1.0
):
    """
    INITIALISATION ONLY: borrow (jB, a_0plus) from the ideal-fluid uNmax solver.

    The ideal energy closure E = h*gamma*u is never used in the thermal solve;
    this call only places the first Newton iterate in the right basin. It is
    needed because _default_energy_jB_guess corresponds to u_0minus ~ 1e-8 while the
    physical value is ~1e-6, and the thermal BVP diverges from that placeholder
    rather than recovering from it.

    Returns None if the ideal solver fails; the caller then falls back.
    """
    try:
        seed = solve_front_energy_conserving_uNmax(
            T, nB_0minus, B_one_forth, T_0plus=float(T_0plus_seed), ms=ms, param=param, NM_type=NM_type
        )
    except Exception:
        return None
    if not seed.get("success", False):
        return None
    jB_seed = seed.get("jB", None)
    if jB_seed is None or (not np.isfinite(jB_seed)) or jB_seed <= 0.0:
        return None
    a_0plus_seed = seed.get("a_0plus", np.nan)
    return {
        "jB": float(jB_seed),
        "a_0plus": float(a_0plus_seed) if np.isfinite(a_0plus_seed) else np.nan,
        "T_0plus_seed": float(T_0plus_seed),
    }


def _thermal_interface_Tprime_sign_scan(
    jB, nuclear_state, B_one_forth, ms=0.0, upB=5000, kappa_model=None, a_values=None
):
    """
    Scan T'(0+) versus the interface composition a(0+) at fixed jB.

    T'(0+) = (h*gamma*u|_Q* - F_E)/kappa_th(0+) is positive only where the
    freshly deconfined quark matter carries MORE enthalpy flux than the incoming
    nuclear matter, which happens at large a(0+) (low strangeness). Starting the
    BVP below that threshold drives T negative immediately, so the sign change
    locates the physical branch. Reported, never imposed.
    """
    if a_values is None:
        a_values = np.linspace(0.05, 0.95, 19)
    info = _thermal_downstream_analysis(
        jB, nuclear_state, B_one_forth, ms=ms, upB=upB, kappa_model=kappa_model
    )
    nB_inf = float(info["thermo_inf"]["nB"])
    records = []
    for a_value in a_values:
        try:
            state = _solve_local_quark_state_from_nK_T_and_Pi(
                float(a_value) * nB_inf,
                0.0,
                info["Pi"],
                float(jB),
                B_one_forth,
                ms=ms,
                upB=upB,
                initial_guess=(info["muB_inf"], 50.0),
            )
            kappa = float(_thermal_conductivity(state["muB"], 0.0, kappa_model=kappa_model)["kappa_th"])
            records.append((float(a_value), float((state["E_flux"] - info["F_E"]) / kappa)))
        except Exception:
            records.append((float(a_value), np.nan))
    a_threshold = np.nan
    for (a_lo, t_lo), (a_hi, t_hi) in zip(records[:-1], records[1:]):
        if np.isfinite(t_lo) and np.isfinite(t_hi) and t_lo <= 0.0 < t_hi:
            a_threshold = float(a_lo + (a_hi - a_lo) * (-t_lo) / (t_hi - t_lo))
            break
    return {
        "a_values": np.array([r[0] for r in records], dtype=float),
        "Tprime_values": np.array([r[1] for r in records], dtype=float),
        "a_threshold": a_threshold,
        "downstream": info,
    }


def _thermal_upstream_nuclear_state(T_0minus, nB_0minus, param, NM_type):
    """Construct the fixed upstream state once for a thermal continuation run."""
    P_0minus = float(PNM_n(nB_0minus, T_0minus, param=param, NM_type=NM_type))
    e_0minus = float(edensNM_n(nB_0minus, T_0minus, param=param))
    h_0minus = float(P_0minus + e_0minus)
    return {
        "P_0minus": P_0minus,
        "e_0minus": e_0minus,
        "h_0minus": h_0minus,
        "nB_0minus": float(nB_0minus),
        "h_over_nB_0minus": float(h_0minus / nB_0minus),
        "T_0minus": float(T_0minus),
    }


def _solve_front_thermal_conducting_once(
    T_0minus,
    nB_0minus,
    B_one_forth,
    T_0plus=0.0,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    jB_bounds=None,
    kappa_model=None,
    a_0plus_guess=0.5,
    profile_guess=None,
    return_profile=False,
    verb=False,
    _nuclear_state=None,
):
    """
    One thermal-conducting front solve at a fixed interface temperature.

    Propagated fields: y = [nK, jK, T] (scaled). Scalar eigenvalue: jB.
    Four boundary conditions: T(0) = T_0plus, jK(0) = jB, and two
    downstream projections killing the growing modes. nK(0) is NOT prescribed;
    it is selected by the global problem, which is what removes the free
    interface composition of the ideal-fluid formulation.
    """
    T_0minus = float(T_0minus)
    nB_0minus = float(nB_0minus)
    T_0plus = float(T_0plus)
    if NM_type != "PNM":
        raise RuntimeError("solve_front_thermal_conducting currently requires NM_type='PNM'")
    if (not np.isfinite(T_0minus)) or T_0minus <= 0.0 or (not np.isfinite(nB_0minus)) or nB_0minus <= 0.0:
        raise RuntimeError("solve_front_thermal_conducting requires positive T and nB_0minus")
    if (not np.isfinite(T_0plus)) or T_0plus < 0.0:
        raise RuntimeError("T_0plus must be non-negative")
    if not (0.0 < float(tail_eps) < 1.0):
        raise RuntimeError("tail_eps must satisfy 0 < tail_eps < 1")
    if int(n_mesh) < 5 or int(max_nodes) <= int(n_mesh) or float(tol_bvp) <= 0.0:
        raise RuntimeError("invalid BVP mesh or tolerance settings")

    upB = 5000
    kappa_model = _normalize_kappa_model(kappa_model)
    if _nuclear_state is None:
        nuclear_state = _thermal_upstream_nuclear_state(T_0minus, nB_0minus, param, NM_type)
    else:
        nuclear_state = dict(_nuclear_state)

    if jB_guess is None:
        jB_guess = _default_energy_jB_guess(nB_0minus)
    jB_guess = float(jB_guess)
    if (not np.isfinite(jB_guess)) or jB_guess <= 0.0:
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
            frac = np.clip((float(value) - jB_lo) / (jB_hi - jB_lo), 1.0e-12, 1.0 - 1.0e-12)
            return float(np.log(frac / (1.0 - frac)))
        return float(np.log(float(value)))

    def jB_from_param(theta):
        if bounded_jB:
            sig = 1.0 / (1.0 + np.exp(-np.clip(float(theta), -60.0, 60.0)))
            return float(jB_lo + (jB_hi - jB_lo) * sig)
        return float(np.exp(np.clip(float(theta), -700.0, 700.0)))

    stats = {
        "bvp_ode_calls": 0,
        "bvp_bc_calls": 0,
        "thermal_local_state_calls": 0,
        "thermal_local_root_calls": 0,
        "thermal_local_cache_hits": 0,
        "linear_tail_ode_points": 0,
        "global_state_builds": 0,
        "global_state_failures": 0,
        "local_state_rejections": 0,
        "negative_temperature_rejections": 0,
        "closure_failure_rejections": 0,
        "conductivity_rejections": 0,
    }
    physical_state_cache = {}
    local_closure_cache = {}
    last_failure = {"message": ""}
    endpoint_guess_cache = {"value": None}

    def build_physical_state(theta):
        key = round(float(theta), 12)
        if key in physical_state_cache:
            return physical_state_cache[key]
        stats["global_state_builds"] += 1
        jB = jB_from_param(theta)
        info = _thermal_downstream_analysis(
            jB,
            nuclear_state,
            B_one_forth,
            ms=ms,
            upB=upB,
            kappa_model=kappa_model,
            endpoint_guess=endpoint_guess_cache["value"],
        )
        endpoint_guess_cache["value"] = info["endpoint"].get(
            "endpoint_initial_guess", (info["muB_inf"], info["T_inf"])
        )
        if int(info["stable_dimension"]) != 1:
            raise RuntimeError(
                "Thermal downstream stable manifold has dimension "
                f"{info['stable_dimension']}, expected 1 (c*f - d*e = {info['cf_minus_de']:.6e})"
            )
        lambda_compact = float(info["lambda_stable"])
        if (not np.isfinite(lambda_compact)) or lambda_compact <= 0.0:
            raise RuntimeError("Thermal compactification rate must be positive")
        info = dict(info)
        info["lambda_compact"] = lambda_compact
        info["lambda_growing_max"] = float(np.max(info["eigenvalues"].real))
        info["nB_inf"] = float(info["thermo_inf"]["nB"])
        physical_state_cache[key] = info
        return info

    theta0 = param_from_jB(jB_guess)
    if isinstance(profile_guess, dict) and np.isfinite(profile_guess.get("theta", np.nan)):
        theta0 = float(profile_guess["theta"])
    try:
        physical_state0 = build_physical_state(theta0)
    except Exception as exc:
        stats["global_state_failures"] += 1
        return {
            "success": False,
            "message": f"Initial thermal-conducting state construction failed: {exc}",
            "T_0plus": T_0plus,
            "tail_eps": float(tail_eps),
            "tol_bvp": float(tol_bvp),
            "s_end": float(1.0 - tail_eps),
            "coordinate": "s=1-exp(-lambda_compact*x)",
            "solver_variant": "thermal_conducting_nK_jK_T",
            "stats": stats,
        }

    nB_Q0 = float(physical_state0["thermo_inf"]["nB"])
    bvp_scales = np.array(
        [
            max(abs(float(physical_state0["nK_inf"])), nB_Q0, 1.0),
            max(
                abs(float(physical_state0["jB"])),
                abs(float(physical_state0["jK_inf"])),
                1.0,
            ),
            max(float(physical_state0["T_inf"]), 1.0),
        ],
        dtype=float,
    )
    state_cache = {}

    def build_state(theta):
        key = round(float(theta), 12)
        if key in state_cache:
            return state_cache[key]
        info = dict(build_physical_state(theta))
        projections, lambda_stable = _thermal_scaled_growing_projections(
            info["jacobian"], bvp_scales
        )
        info["scales"] = bvp_scales
        info["growing_projections"] = projections
        info["lambda_compact"] = float(lambda_stable)
        state_cache[key] = info
        return info

    def state_or_none(theta):
        try:
            return build_state(theta)
        except Exception as exc:
            stats["global_state_failures"] += 1
            last_failure["message"] = str(exc)
            return None

    def pointwise(nK_value, T_value, state, guess):
        """Local closure + full reconstruction. Raises on any invalid state."""
        if T_value < 0.0:
            stats["negative_temperature_rejections"] += 1
            raise RuntimeError(f"negative trial temperature T={T_value:.6e}")
        cache_key = (float(state["jB"]), float(nK_value), float(T_value))
        cached = local_closure_cache.get(cache_key)
        if cached is not None:
            stats["thermal_local_cache_hits"] += 1
            return cached
        thermo = _solve_local_quark_state_from_nK_T_and_Pi(
            nK_value,
            T_value,
            state["Pi"],
            state["jB"],
            B_one_forth,
            ms=ms,
            upB=upB,
            initial_guess=guess,
            stats=stats,
        )
        micro = _microphysics_from_quark_state_energy(
            thermo["muB"], thermo["T"], allow_zero_temperature=bool(thermo["T"] == 0.0)
        )
        kappa_info = _thermal_conductivity(thermo["muB"], thermo["T"], kappa_model=kappa_model)
        rate = float(
            _exact_kaon_transport_rate(
                thermo["muB"], thermo["muK"], thermo["T"], ms=ms, upB=upB
            )["Gamma_K"]
        )
        kappa = float(kappa_info["kappa_th"])
        E_flux = float(thermo["E_flux"])
        thermo["invD"] = float(micro["invD"])
        thermo["kappa_th"] = kappa
        thermo["kappa_y"] = float(kappa_info["y"])
        thermo["Gamma_K"] = rate
        thermo["T_prime"] = float((E_flux - state["F_E"]) / kappa)
        thermo["q_th"] = float(state["F_E"] - E_flux)
        if len(local_closure_cache) >= 50000:
            local_closure_cache.clear()
        local_closure_cache[cache_key] = thermo
        return thermo

    def ode(s_coord, y, p):
        stats["bvp_ode_calls"] += 1
        state = state_or_none(float(p[0]))
        if state is None:
            return np.full_like(y, 1.0e12)
        S = state["scales"]
        dyds = np.empty_like(y)
        guess = (state["muB_inf"], 1.0e-3)
        for i in range(y.shape[1]):
            try:
                nK_value = float(y[0, i]) * S[0]
                jK_value = float(y[1, i]) * S[1]
                T_value = float(y[2, i]) * S[2]
                if not np.all(np.isfinite([nK_value, jK_value, T_value])):
                    raise RuntimeError("non-finite thermal BVP trial state")
                if T_value < 0.0:
                    stats["negative_temperature_rejections"] += 1
                    raise RuntimeError(f"negative trial temperature T={T_value:.6e}")
                one_minus_s = max(1.0 - float(s_coord[i]), np.finfo(float).tiny)
                dx_ds = 1.0 / (state["lambda_compact"] * one_minus_s)
                linear_weight = 0.0
                if linear_tail_enabled:
                    if one_minus_s <= _THERMAL_LINEAR_TAIL_GAP:
                        linear_weight = 1.0
                    elif one_minus_s < 2.0 * _THERMAL_LINEAR_TAIL_GAP:
                        blend = (
                            2.0 * _THERMAL_LINEAR_TAIL_GAP - one_minus_s
                        ) / _THERMAL_LINEAR_TAIL_GAP
                        linear_weight = blend * blend * (3.0 - 2.0 * blend)
                if linear_weight > 0.0:
                    delta = np.array(
                        [
                            nK_value - state["nK_inf"],
                            jK_value - state["jK_inf"],
                            T_value - state["T_inf"],
                        ],
                        dtype=float,
                    )
                    linear_rhs = state["jacobian"] @ delta
                if linear_weight >= 1.0:
                    dyds[:, i] = linear_rhs * dx_ds / S
                    stats["linear_tail_ode_points"] += 1
                    continue
                thermo = pointwise(nK_value, T_value, state, guess)
                guess = (thermo["muB"], thermo["muK"])
                physical_rhs = np.array(
                    [
                        (thermo["u"] * nK_value - jK_value) * thermo["invD"],
                        -thermo["Gamma_K"],
                        thermo["T_prime"],
                    ],
                    dtype=float,
                )
                if linear_weight > 0.0:
                    physical_rhs = (
                        (1.0 - linear_weight) * physical_rhs
                        + linear_weight * linear_rhs
                    )
                    stats["linear_tail_ode_points"] += 1
                dyds[:, i] = physical_rhs * dx_ds / S
            except Exception as exc:
                stats["local_state_rejections"] += 1
                last_failure["message"] = str(exc)
                dyds[:, i] = 1.0e12
        return dyds

    def bc(ya, yb, p):
        stats["bvp_bc_calls"] += 1
        state = state_or_none(float(p[0]))
        if state is None:
            return np.full(4, 1.0e12, dtype=float)
        S = state["scales"]
        W = state["growing_projections"]
        delta = np.array(
            [
                float(yb[0]) - state["nK_inf"] / S[0],
                float(yb[1]) - state["jK_inf"] / S[1],
                float(yb[2]) - state["T_inf"] / S[2],
            ],
            dtype=float,
        )
        return np.array(
            [
                float(ya[2]) - T_0plus / S[2],
                float(ya[1]) - state["jB"] / S[1],
                float(W[0] @ delta),
                float(W[1] @ delta),
            ],
            dtype=float,
        )

    state0 = state_or_none(theta0)
    if state0 is None:
        return {
            "success": False,
            "message": f"Initial thermal-conducting state construction failed: {last_failure['message']}",
            "T_0plus": T_0plus,
            "tail_eps": float(tail_eps),
            "tol_bvp": float(tol_bvp),
            "s_end": float(1.0 - tail_eps),
            "coordinate": "s=1-exp(-lambda_compact*x)",
            "solver_variant": "thermal_conducting_nK_jK_T",
            "stats": stats,
        }

    S0 = state0["scales"]
    s_end = float(1.0 - tail_eps)
    linear_tail_enabled = bool(float(tail_eps) < 0.1 * _THERMAL_LINEAR_TAIL_GAP)
    s_mesh = _thermal_compact_mesh(n_mesh, tail_eps)
    equilibrium0 = np.array(
        [state0["nK_inf"], state0["jK_inf"], state0["T_inf"]], dtype=float
    )
    if profile_guess is not None:
        y_guess = _thermal_interpolate_profile_guess(
            profile_guess, s_mesh, equilibrium0, S0
        )
        left_shape = (1.0 - s_mesh) / max(1.0 - s_mesh[0], np.finfo(float).tiny)
        y_guess[1] += (state0["jB"] / S0[1] - y_guess[1, 0]) * left_shape
        y_guess[2] += (T_0plus / S0[2] - y_guess[2, 0]) * left_shape
        y_guess[1, 0] = state0["jB"] / S0[1]
        y_guess[2, 0] = T_0plus / S0[2]
    else:
        nK0_guess = float(a_0plus_guess) * state0["nB_inf"]
        stable_shape = 1.0 - s_mesh
        nK_line = state0["nK_inf"] + (nK0_guess - state0["nK_inf"]) * stable_shape
        jK_line = state0["jK_inf"] + (state0["jB"] - state0["jK_inf"]) * stable_shape
        T_line = state0["T_inf"] + (T_0plus - state0["T_inf"]) * stable_shape
        y_guess = np.vstack([nK_line / S0[0], jK_line / S0[1], T_line / S0[2]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sol = solve_bvp(
            ode,
            bc,
            s_mesh,
            y_guess,
            p=np.array([theta0], dtype=float),
            tol=float(tol_bvp),
            max_nodes=int(max_nodes),
            verbose=2 if verb else 0,
        )

    jB_final = jB_from_param(float(sol.p[0]))
    state_final = state_or_none(float(sol.p[0]))
    base = {
        "T_0plus": T_0plus,
        "tail_eps": float(tail_eps),
        "tol_bvp": float(tol_bvp),
        "s_end": s_end,
        "coordinate": "s=1-exp(-lambda_compact*x)",
        "compact_tail_e_folds": float(-np.log(tail_eps)),
        "linear_tail_gap": float(_THERMAL_LINEAR_TAIL_GAP),
        "linear_tail_enabled": linear_tail_enabled,
        "tail_ode_model": "full nonlinear with downstream-Jacobian asymptotic tail",
        "bvp_scales": np.asarray(bvp_scales, dtype=float),
        "solver_variant": "thermal_conducting_nK_jK_T",
        "rate_model": "exact_nonleptonic",
        "conductivity_model": kappa_model["name"],
        "conductivity_y_max": float(kappa_model["y_max"]),
        "energy_closure": "F_E = h*gamma*u - kappa_th*dT/dx",
        "bvp_status": int(sol.status),
        "bvp_message": str(sol.message),
        "bvp_iterations": int(getattr(sol, "niter", -1)),
        "bvp_nodes": int(np.asarray(sol.x).size),
        "stats": stats,
    }
    if state_final is None or sol.status != 0:
        base.update(
            {
                "success": False,
                "message": (
                    f"thermal-conducting BVP did not converge: {sol.message}"
                    + (f"; last local failure: {last_failure['message']}" if last_failure["message"] else "")
                ),
                "jB": jB_final,
                "u_0minus": float(jB_final / nB_0minus),
            }
        )
        return base

    # ---- post-solve reconstruction on a dense compact grid
    S = state_final["scales"]
    lam = float(state_final["lambda_compact"])
    s_dense = _thermal_compact_mesh(max(600, int(n_mesh) * 3), tail_eps)
    y_dense = sol.sol(s_dense)
    x_dense = -np.log1p(-s_dense) / lam
    x_end = float(-np.log(tail_eps) / lam)

    fields = {
        key: np.empty(s_dense.size)
        for key in (
            "nK",
            "jK",
            "T",
            "muB",
            "muK",
            "nB",
            "u",
            "kappa_th",
            "q_th",
            "q_th_closure",
            "Gamma_K",
            "invD",
            "T_prime",
            "T_prime_closure",
            "E_advective",
            "flux_B",
            "flux_Pi",
            "flux_E",
            "closure_residual",
            "kappa_y",
        )
    }
    rhs_nK = np.empty(s_dense.size)
    rhs_jK = np.empty(s_dense.size)
    rhs_T = np.empty(s_dense.size)
    guess = (state_final["muB_inf"], 1.0e-3)
    reconstruction_failures = 0
    for i in range(s_dense.size):
        nK_value = float(y_dense[0, i]) * S[0]
        jK_value = float(y_dense[1, i]) * S[1]
        T_value = float(y_dense[2, i]) * S[2]
        try:
            if T_value < 0.0:
                raise RuntimeError(f"negative reconstructed temperature T={T_value:.6e}")
            th = pointwise(nK_value, T_value, state_final, guess)
            guess = (th["muB"], th["muK"])
            fields["nK"][i] = nK_value
            fields["jK"][i] = jK_value
            fields["T"][i] = th["T"]
            fields["muB"][i] = th["muB"]
            fields["muK"][i] = th["muK"]
            fields["nB"][i] = th["nB"]
            fields["u"][i] = th["u"]
            fields["kappa_th"][i] = th["kappa_th"]
            fields["kappa_y"][i] = th["kappa_y"]
            fields["q_th_closure"][i] = th["q_th"]
            fields["Gamma_K"][i] = th["Gamma_K"]
            fields["invD"][i] = th["invD"]
            fields["T_prime_closure"][i] = th["T_prime"]
            fields["E_advective"][i] = th["E_flux"]
            fields["flux_B"][i] = th["nB"] * th["u"]
            fields["flux_Pi"][i] = th["Pi"]
            fields["closure_residual"][i] = th["closure_residual"]
            rhs_nK[i] = (th["u"] * nK_value - jK_value) * th["invD"]
            rhs_jK[i] = -th["Gamma_K"]
            rhs_T[i] = (th["E_flux"] - state_final["F_E"]) / th["kappa_th"]
        except Exception:
            reconstruction_failures += 1
            for key in fields:
                fields[key][i] = np.nan
            rhs_nK[i] = np.nan
            rhs_jK[i] = np.nan
            rhs_T[i] = np.nan

    try:
        dy_ds_scaled = _bvp_dense_derivative(sol, s_dense)
        dy_ds_physical = dy_ds_scaled * S[:, None]
        dx_ds = 1.0 / (lam * np.maximum(1.0 - s_dense, np.finfo(float).tiny))
        dy_dx_physical = dy_ds_physical / dx_ds[None, :]
        dnK_dx_spline = np.asarray(dy_dx_physical[0], dtype=float)
        djK_dx_spline = np.asarray(dy_dx_physical[1], dtype=float)
        dT_dx_spline = np.asarray(dy_dx_physical[2], dtype=float)
    except Exception as exc:
        reconstruction_failures += 1
        last_failure["message"] = f"compact spline derivative reconstruction failed: {exc}"
        dnK_dx_spline = np.full(s_dense.size, np.nan)
        djK_dx_spline = np.full(s_dense.size, np.nan)
        dT_dx_spline = np.full(s_dense.size, np.nan)

    fields["T_prime"] = dT_dx_spline
    fields["q_th"] = -fields["kappa_th"] * dT_dx_spline
    fields["flux_E"] = fields["E_advective"] + fields["q_th"]

    def mean_relative_residual(lhs, rhs):
        lhs = np.asarray(lhs, dtype=float)
        rhs = np.asarray(rhs, dtype=float)
        finite = np.isfinite(lhs) & np.isfinite(rhs)
        if not np.any(finite):
            return np.nan
        scale = max(float(np.mean(np.abs(rhs[finite]))), _FLOAT_TINY)
        return float(np.mean(np.abs(lhs[finite] - rhs[finite])) / scale)

    nK_ode_residual_norm = mean_relative_residual(dnK_dx_spline, rhs_nK)
    jK_ode_residual_norm = mean_relative_residual(djK_dx_spline, rhs_jK)
    T_ode_residual_norm = mean_relative_residual(dT_dx_spline, rhs_T)
    q_th_consistency_norm = mean_relative_residual(fields["q_th"], fields["q_th_closure"])

    def rel_error(arr, ref):
        finite = np.isfinite(arr)
        if not np.any(finite) or abs(ref) <= 0.0:
            return np.nan
        return float(np.max(np.abs(arr[finite] / ref - 1.0)))

    jB_error = rel_error(fields["flux_B"], jB_final)
    Pi_error = rel_error(fields["flux_Pi"], state_final["Pi"])
    FE_error = rel_error(fields["flux_E"], state_final["F_E"])
    bc_residual = bc(y_dense[:, 0], y_dense[:, -1], np.array([float(sol.p[0])]))

    Tprime_0plus = float(fields["T_prime"][0])
    q_th_0plus = float(fields["q_th"][0])
    nK_0plus = float(fields["nK"][0])
    nB_0plus = float(fields["nB"][0])
    a_0plus = float(nK_0plus / nB_0plus) if nB_0plus > 0.0 else np.nan
    closure_max = float(np.nanmax(fields["closure_residual"])) if s_dense.size else np.nan
    y_max_seen = float(np.nanmax(fields["kappa_y"])) if s_dense.size else np.nan
    equilibrium_y = np.array(
        [state_final["nK_inf"], state_final["jK_inf"], state_final["T_inf"]], dtype=float
    )
    tail_delta_scaled = (
        np.array([fields["nK"][-1], fields["jK"][-1], fields["T"][-1]])
        - equilibrium_y
    ) / S
    tail_state_residual_norm = float(np.max(np.abs(tail_delta_scaled)))
    tail_muK_residual_norm = float(
        abs(fields["muK"][-1]) / max(abs(state_final["muB_inf"]), 1.0)
    )
    tail_q_residual_norm = float(
        abs(fields["q_th"][-1]) / max(abs(state_final["F_E"]), 1.0)
    )

    warnings_list = []
    if reconstruction_failures:
        warnings_list.append(f"{reconstruction_failures} dense-grid reconstruction failures")
    if stats["local_state_rejections"]:
        warnings_list.append(
            f"{stats['local_state_rejections']} collocation states rejected during the solve"
        )
    if not (Tprime_0plus > 0.0):
        warnings_list.append(
            f"T'(0+) = {Tprime_0plus:.6e} is not positive: the interface is not heating "
            "downstream, so this branch may be unphysical"
        )
    if not (q_th_0plus < 0.0):
        warnings_list.append(f"q_th(0+) = {q_th_0plus:.6e} is not negative")
    if np.isfinite(y_max_seen) and y_max_seen > 0.5 * kappa_model["y_max"]:
        warnings_list.append(
            f"max T/qD = {y_max_seen:.4f} approaches the validated conductivity band "
            f"y_max = {kappa_model['y_max']:.2f}; the low-y collision integral is being "
            "used near its edge"
        )
    # Knudsen / LTE diagnostic from the transport model actually in use
    with np.errstate(divide="ignore", invalid="ignore"):
        mfp = 3.0 / np.where(fields["invD"] > 0.0, fields["invD"], np.nan)
        grad_scale = np.where(
            np.abs(fields["T_prime"]) > 0.0, fields["T"] / np.abs(fields["T_prime"]), np.inf
        )
        knudsen = mfp / grad_scale
    knudsen_max = float(np.nanmax(knudsen)) if np.any(np.isfinite(knudsen)) else np.nan
    if np.isfinite(knudsen_max) and knudsen_max > 1.0:
        warnings_list.append(
            f"max Knudsen number {knudsen_max:.3e} > 1: Fourier conduction is outside local "
            "thermal equilibrium somewhere in the layer"
        )

    ode_limit = 10.0 * float(tol_bvp)
    tail_limit = max(10.0 * float(tol_bvp), 100.0 * float(tail_eps))
    success = bool(
        sol.status == 0
        and reconstruction_failures == 0
        and np.isfinite(jB_error)
        and np.isfinite(Pi_error)
        and np.isfinite(FE_error)
        and max(jB_error, Pi_error, FE_error) < 1.0e-4
        and float(np.max(np.abs(bc_residual))) < 1.0e-4
        and np.isfinite(closure_max)
        and closure_max < 1.0e-7
        and np.all(
            np.isfinite(
                [
                    nK_ode_residual_norm,
                    jK_ode_residual_norm,
                    T_ode_residual_norm,
                    q_th_consistency_norm,
                ]
            )
        )
        and max(nK_ode_residual_norm, jK_ode_residual_norm, T_ode_residual_norm)
        <= ode_limit
        and q_th_consistency_norm <= ode_limit
        and np.all(
            np.isfinite(
                [
                    tail_state_residual_norm,
                    tail_muK_residual_norm,
                    tail_q_residual_norm,
                ]
            )
        )
        and tail_state_residual_norm <= tail_limit
        and tail_muK_residual_norm <= tail_limit
        and tail_q_residual_norm <= tail_limit
    )
    message = "converged" if success else "post-validation failed"
    if not success:
        detail = []
        if sol.status != 0:
            detail.append(f"bvp status {sol.status}")
        if reconstruction_failures:
            detail.append(f"{reconstruction_failures} reconstruction failures")
        for label, value in (("jB", jB_error), ("Pi", Pi_error), ("F_E", FE_error)):
            if not np.isfinite(value) or value >= 1.0e-4:
                detail.append(f"{label} flux error {value:.3e}")
        bc_worst = float(np.max(np.abs(bc_residual)))
        if bc_worst >= 1.0e-4:
            detail.append(f"max BC residual {bc_worst:.3e}")
        if not np.isfinite(closure_max) or closure_max >= 1.0e-7:
            detail.append(f"max closure residual {closure_max:.3e}")
        for label, value in (
            ("nK ODE", nK_ode_residual_norm),
            ("jK ODE", jK_ode_residual_norm),
            ("T ODE", T_ode_residual_norm),
            ("heat-flux consistency", q_th_consistency_norm),
        ):
            if not np.isfinite(value) or value > ode_limit:
                detail.append(f"{label} residual {value:.3e}")
        for label, value in (
            ("tail state", tail_state_residual_norm),
            ("tail muK", tail_muK_residual_norm),
            ("tail heat flux", tail_q_residual_norm),
        ):
            if not np.isfinite(value) or value > tail_limit:
                detail.append(f"{label} residual {value:.3e}")
        message = "post-validation failed: " + "; ".join(detail)

    base.update(
        {
            "success": success,
            "message": message,
            "jB": jB_final,
            "u_0minus": float(jB_final / nB_0minus),
            "Pi": float(state_final["Pi"]),
            "F_E": float(state_final["F_E"]),
            "T_inf": float(state_final["T_inf"]),
            "muB_inf": float(state_final["muB_inf"]),
            "nB_inf": float(state_final["nB_inf"]),
            "nK_inf": float(state_final["nK_inf"]),
            "jK_inf": float(state_final["jK_inf"]),
            "lambda_n": float(state_final["lambda_n"]),
            "lambda_compact": float(lam),
            "a_0plus": a_0plus,
            "nK_0plus": nK_0plus,
            "Tprime_0plus": Tprime_0plus,
            "q_th_0plus": q_th_0plus,
            "downstream_stable_dimension": int(state_final["stable_dimension"]),
            "downstream_cf_minus_de": float(state_final["cf_minus_de"]),
            "downstream_eigenvalues": np.asarray(state_final["eigenvalues"]),
            "bc_residuals": np.asarray(bc_residual, dtype=float),
            "max_closure_residual": closure_max,
            "max_flux_error_jB": jB_error,
            "max_flux_error_Pi": Pi_error,
            "max_flux_error_F_E": FE_error,
            "nK_ode_residual_norm": nK_ode_residual_norm,
            "jK_ode_residual_norm": jK_ode_residual_norm,
            "T_ode_residual_norm": T_ode_residual_norm,
            "q_th_consistency_norm": q_th_consistency_norm,
            "tail_state_residual_norm": tail_state_residual_norm,
            "tail_muK_residual_norm": tail_muK_residual_norm,
            "tail_q_residual_norm": tail_q_residual_norm,
            "max_knudsen": knudsen_max,
            "max_kappa_y": y_max_seen,
            "warnings": warnings_list,
            "s_end": s_end,
            "x_end": x_end,
            "L_domain": x_end,
            "domain_limited_by": "tail_eps",
            "lambda_growing_max": float(state_final["lambda_growing_max"]),
            "bvp_scales": np.asarray(S, dtype=float),
            "_continuation_state": {
                "s": np.asarray(sol.x, dtype=float),
                "physical_y": np.asarray(sol.sol(sol.x), dtype=float) * S[:, None],
                "equilibrium_y": equilibrium_y,
                "theta": float(sol.p[0]),
                "tail_eps": float(tail_eps),
            },
        }
    )
    if return_profile:
        base["s"] = s_dense
        base["x"] = x_dense
        for key, arr in fields.items():
            base[f"{key}_profile"] = arr
    return base


def solve_front_thermal_conducting(
    T_0minus,
    nB_0minus,
    B_one_forth,
    T_0plus=0.0,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    jB_bounds=None,
    kappa_model=None,
    a_0plus_guess=None,
    continuation_schedule=None,
    seed_from_ideal=True,
    return_profile=False,
    verb=False,
):
    """
    Steady conversion front including Fourier heat conduction in the quark phase.

    Solves

        d/dx(nB*u)                      = 0
        d/dx(P + h*u**2)                = 0
        d/dx(h*gamma*u - kappa_th*T')   = 0        (Q_nu = 0)
        d/dx(nK*u - D_K*nK')            = -Gamma_K

    as a boundary-value problem in the propagated fields [nK, jK, T] with the
    single scalar eigenvalue jB. The energy equation is no longer an algebraic
    closure for T, so T is transported rather than reconstructed, and the local
    EOS closure is a 2x2 solve for (muB, muK) at prescribed T.

    Boundary conditions (four, matching three ODEs plus one parameter):
        T(0) = T_0plus (default exactly 0), jK(0) = jB, and two downstream
        conditions killing the growing modes of the linearised system.

    Neither nK(0) nor T'(0+) is imposed: both are selected by the global
    problem. T'(0+) = (h*gamma*u|_Q* - F_E)/kappa_th(0+), so T(0) = 0 does not
    force T'(0+) = 0, and the interface is Lipschitz.

    The exact T(0) = 0 problem is reached by continuation in the interface
    temperature. That is a continuation waypoint only; the final solve uses the
    exact boundary condition and no temperature floor is introduced anywhere.
    At the first temperature waypoint, a separate compact-tail continuation
    deepens the domain to the exact requested tail_eps before T_0plus is
    continued. The numerically unresolved final tail is matched smoothly to
    the linear downstream system and independently checked against the full
    nonlinear RHS during reconstruction.
    """
    T_0minus = float(T_0minus)
    nB_0minus = float(nB_0minus)
    T_0plus = float(T_0plus)
    if (not np.isfinite(T_0plus)) or T_0plus < 0.0:
        raise RuntimeError("T_0plus must be non-negative")
    if NM_type != "PNM":
        raise RuntimeError("solve_front_thermal_conducting currently requires NM_type='PNM'")
    if (not np.isfinite(T_0minus)) or T_0minus <= 0.0 or (not np.isfinite(nB_0minus)) or nB_0minus <= 0.0:
        raise RuntimeError("solve_front_thermal_conducting requires positive T and nB_0minus")
    if not (0.0 < float(tail_eps) < 1.0):
        raise RuntimeError("tail_eps must satisfy 0 < tail_eps < 1")
    if int(n_mesh) < 5 or int(max_nodes) <= int(n_mesh) or float(tol_bvp) <= 0.0:
        raise RuntimeError("invalid BVP mesh or tolerance settings")
    nuclear_state = _thermal_upstream_nuclear_state(T_0minus, nB_0minus, param, NM_type)

    if continuation_schedule is None:
        if T_0plus == 0.0:
            # Geometric decrements. The measured choke point is a large first
            # drop (2.0 -> 0.5 MeV failed for most perturbed inputs); the
            # adaptive bisection below can recover from it, but each recovery
            # costs a wasted BVP solve, so the default path steps gently.
            schedule = [2.0, 1.0, 0.5, 0.25, 0.1, 0.04, 0.01, 0.0]
        else:
            schedule = [T_0plus]
    else:
        schedule = [float(v) for v in continuation_schedule]
        if not schedule:
            raise RuntimeError("continuation_schedule must be non-empty")
        if abs(schedule[-1] - T_0plus) > 1.0e-12:
            schedule = schedule + [T_0plus]
    if any((not np.isfinite(v)) or v < 0.0 for v in schedule):
        raise RuntimeError("continuation_schedule entries must be finite and non-negative")

    # Seeding. The physical jB sits orders of magnitude away from
    # _default_energy_jB_guess, and T'(0+) is positive only above a threshold
    # interface composition, so an unseeded start diverges. Both seeds are
    # initialisation only; neither enters the thermal closure.
    seed_info = None
    if jB_guess is None and seed_from_ideal:
        seed_info = _thermal_seed_from_ideal_solver(
            T_0minus, nB_0minus, B_one_forth, ms=ms, param=param, NM_type=NM_type
        )
        if seed_info is not None:
            jB_guess = seed_info["jB"]
            if verb:
                print(
                    f"[thermal_conducting] seeded from ideal solver: jB={jB_guess:.6e} "
                    f"a_0plus={seed_info['a_0plus']:.4f}",
                    flush=True,
                )
    if a_0plus_guess is None:
        # Start on the heating side of the T'(0+) sign change. Below the
        # threshold the interface cools into T < 0 and the solve cannot start.
        a_0plus_guess = 0.85
        if seed_info is not None and np.isfinite(seed_info.get("a_0plus", np.nan)):
            a_0plus_guess = float(min(0.95, max(0.85, seed_info["a_0plus"] + 0.30)))
    if jB_bounds is None and jB_guess is not None and np.isfinite(jB_guess) and jB_guess > 0.0:
        # Bounded logit parameterisation: without it the Newton step on
        # theta = log(jB) is unconstrained and can run away to overflow.
        jB_bounds = (float(jB_guess) / 100.0, float(jB_guess) * 100.0)

    tail_schedule = _thermal_tail_schedule(tail_eps)
    profile_guess = None
    jB_running = jB_guess
    history = []
    tail_history = []
    result = None
    attempts = 0

    def continuation_failure(message):
        failure = dict(result) if isinstance(result, dict) else {"success": False}
        failure["success"] = False
        failure["message"] = message
        failure["continuation_history"] = history
        failure["continuation_schedule"] = schedule
        failure["tail_continuation_history"] = tail_history
        failure["tail_continuation_schedule"] = tail_schedule
        failure["seed_info"] = seed_info
        return failure

    # Find the branch at the first temperature on a moderately truncated tail,
    # then deepen that same physical profile. This avoids asking a cold-start
    # Newton solve to resolve all -log(tail_eps) asymptotic e-folds at once.
    first_T = float(schedule[0])
    tail_queue = list(tail_schedule)
    last_good_eps = None
    while tail_queue:
        attempts += 1
        if attempts > _THERMAL_MAX_CONTINUATION_ATTEMPTS:
            return continuation_failure(
                "continuation exceeded "
                f"{_THERMAL_MAX_CONTINUATION_ATTEMPTS} attempts before reaching "
                f"tail_eps={tail_eps:g} at T_0plus={first_T:g} MeV"
            )

        eps_step = float(tail_queue[0])
        if profile_guess is None:
            seed_candidates = [float(a_0plus_guess)]
            for extra in (0.85, 0.92, 0.75, 0.95, 0.65):
                if all(abs(extra - existing) > 1.0e-9 for existing in seed_candidates):
                    seed_candidates.append(float(extra))
        else:
            seed_candidates = [float(a_0plus_guess)]

        result = None
        for seed_attempt, seed_value in enumerate(seed_candidates):
            result = _solve_front_thermal_conducting_once(
                T_0minus,
                nB_0minus,
                B_one_forth,
                T_0plus=first_T,
                ms=ms,
                param=param,
                NM_type=NM_type,
                tail_eps=eps_step,
                n_mesh=n_mesh,
                tol_bvp=tol_bvp,
                max_nodes=max_nodes,
                jB_guess=jB_running,
                jB_bounds=jB_bounds,
                kappa_model=kappa_model,
                a_0plus_guess=seed_value,
                profile_guess=profile_guess,
                return_profile=(
                    return_profile
                    and len(schedule) == 1
                    and eps_step == float(tail_eps)
                ),
                verb=verb,
                _nuclear_state=nuclear_state,
            )
            if result.get("success", False):
                if verb and seed_attempt:
                    print(
                        f"[thermal_conducting] cold start recovered with "
                        f"a_0plus_guess={seed_value:g}",
                        flush=True,
                    )
                break

        succeeded = bool(result.get("success", False))
        tail_history.append(
            {
                "tail_eps": eps_step,
                "success": succeeded,
                "u_0minus": result.get("u_0minus", np.nan),
                "jB": result.get("jB", np.nan),
                "a_0plus": result.get("a_0plus", np.nan),
                "tail_state_residual_norm": result.get(
                    "tail_state_residual_norm", np.nan
                ),
                "message": result.get("message", ""),
            }
        )
        if verb:
            print(
                f"[thermal_conducting] tail_eps={eps_step:g} at "
                f"T_0plus={first_T:g} MeV -> success={succeeded} "
                f"u_0minus={result.get('u_0minus', float('nan')):.6e} "
                f"a_0plus={result.get('a_0plus', float('nan')):.6f}",
                flush=True,
            )

        if succeeded:
            tail_queue.pop(0)
            last_good_eps = eps_step
            profile_guess = result.get("_continuation_state")
            jB_running = result.get("jB", jB_running)
            continue

        if last_good_eps is None:
            return continuation_failure(
                f"tail continuation failed at its first step "
                f"(tail_eps={eps_step:g}, T_0plus={first_T:g} MeV) after "
                f"{len(seed_candidates)} interface-composition seeds: "
                f"{result.get('message', '')}"
            )

        midpoint = float(np.sqrt(last_good_eps * eps_step))
        if (
            not np.isfinite(midpoint)
            or midpoint >= last_good_eps * (1.0 - 1.0e-10)
            or midpoint <= eps_step * (1.0 + 1.0e-10)
        ):
            return continuation_failure(
                f"tail continuation stalled between the last converged "
                f"tail_eps={last_good_eps:.10g} and target {eps_step:.10g}; "
                f"last solver message: {result.get('message', '')}"
            )
        if verb:
            print(
                f"[thermal_conducting] refining tail {last_good_eps:g} -> "
                f"{eps_step:g} via {midpoint:g}",
                flush=True,
            )
        tail_queue.insert(0, midpoint)

    history.append(
        {
            "T_0plus": first_T,
            "success": True,
            "u_0minus": result.get("u_0minus", np.nan),
            "jB": result.get("jB", np.nan),
            "a_0plus": result.get("a_0plus", np.nan),
            "message": result.get("message", ""),
        }
    )

    # With the deepest tail fixed, continue the left temperature boundary. A
    # failed decrement is retried through the arithmetic midpoint, preserving
    # the previously converged compact profile as the initial guess.
    queue = list(schedule[1:])
    last_good_T = first_T
    while queue:
        attempts += 1
        if attempts > _THERMAL_MAX_CONTINUATION_ATTEMPTS:
            return continuation_failure(
                "continuation exceeded "
                f"{_THERMAL_MAX_CONTINUATION_ATTEMPTS} attempts before reaching "
                f"T_0plus={T_0plus:g} MeV"
            )

        T_step = float(queue[0])
        is_final = bool(len(queue) == 1)
        result = _solve_front_thermal_conducting_once(
            T_0minus,
            nB_0minus,
            B_one_forth,
            T_0plus=T_step,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=n_mesh,
            tol_bvp=tol_bvp,
            max_nodes=max_nodes,
            jB_guess=jB_running,
            jB_bounds=jB_bounds,
            kappa_model=kappa_model,
            a_0plus_guess=a_0plus_guess,
            profile_guess=profile_guess,
            return_profile=return_profile and is_final,
            verb=verb,
            _nuclear_state=nuclear_state,
        )

        succeeded = bool(result.get("success", False))
        history.append(
            {
                "T_0plus": T_step,
                "success": succeeded,
                "u_0minus": result.get("u_0minus", np.nan),
                "jB": result.get("jB", np.nan),
                "a_0plus": result.get("a_0plus", np.nan),
                "message": result.get("message", ""),
            }
        )
        if verb:
            print(
                f"[thermal_conducting] T_0plus={T_step:g} MeV -> "
                f"success={succeeded} u_0minus={result.get('u_0minus', float('nan')):.6e} "
                f"a_0plus={result.get('a_0plus', float('nan')):.6f}",
                flush=True,
            )

        if succeeded:
            queue.pop(0)
            last_good_T = T_step
            profile_guess = result.get("_continuation_state")
            jB_running = result.get("jB", jB_running)
            continue

        midpoint = 0.5 * (last_good_T + T_step)
        if not np.isfinite(midpoint) or (last_good_T - midpoint) <= 1.0e-9 * max(
            abs(last_good_T), 1.0
        ):
            return continuation_failure(
                f"continuation stalled: repeated bisection from the last converged "
                f"T_0plus={last_good_T:.10g} MeV could not reach the next target "
                f"(stuck at {T_step:.10g} MeV, remaining schedule {queue[1:]!r}); "
                f"last solver message: {result.get('message', '')}"
            )
        if verb:
            print(
                f"[thermal_conducting] refining step {last_good_T:g} -> {T_step:g} MeV "
                f"via {midpoint:g} MeV",
                flush=True,
            )
        queue.insert(0, midpoint)

    result = dict(result)
    reached_requested_endpoint = bool(
        result.get("success", False)
        and float(result.get("T_0plus", np.nan)) == T_0plus
        and float(result.get("tail_eps", np.nan)) == float(tail_eps)
    )
    if not reached_requested_endpoint:
        return continuation_failure(
            "continuation terminated without reaching both the requested "
            f"T_0plus={T_0plus:g} MeV and tail_eps={tail_eps:g}"
        )
    result["continuation_history"] = history
    result["continuation_schedule"] = schedule
    result["tail_continuation_history"] = tail_history
    result["tail_continuation_schedule"] = tail_schedule
    result["seed_info"] = seed_info
    return result


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
    T_0plus=0.5,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    T_inf_guess=None,
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

        nB_0minus(z) = nB_target + density_slope_n0_per_km * n0 * z.

    ``temperature`` may be a positive scalar in MeV or a callable accepting
    ``z`` in km and returning the local upstream temperature in MeV. Explicit
    time-dependent cooling is not supported because it would require a
    two-dimensional velocity surrogate v(z, t).

    The hydrodynamic solver returns the spatial four-velocity u_0minus. It is
    converted to ordinary speed with beta_0minus = u_0minus/sqrt(1 + u_0minus**2), and the
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
        At exact ``T_0plus=0``, return an explicitly unsuccessful numerical
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
    T_0plus = float(T_0plus)
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
        nB_0minus = float(nB_target + density_slope_n0_per_km * n0 * z_value)
        try:
            T_0minus = float(temperature_of_z(z_value))
        except Exception as exc:
            raise RuntimeError(
                f"temperature(z) failed at z={z_value:.9g} km: {exc}"
            ) from exc
        if (not np.isfinite(T_0minus)) or T_0minus <= 0.0:
            raise RuntimeError(
                f"temperature(z) must be positive and finite at z={z_value:.9g} km"
            )

        point_jB_guess = jB_guess
        point_T_inf_guess = T_inf_guess
        if isinstance(continuation_guess, dict):
            point_jB_guess = continuation_guess.get("jB", point_jB_guess)
            point_T_inf_guess = continuation_guess.get("T_inf", point_T_inf_guess)

        result = solve_front_energy_conserving_uNmax(
            T_0minus,
            float(nB_0minus),
            B_one_forth,
            T_0plus=T_0plus,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=n_mesh,
            tol_bvp=tol_bvp,
            max_nodes=max_nodes,
            jB_guess=point_jB_guess,
            T_inf_guess=point_T_inf_guess,
            jB_bounds=jB_bounds,
            continuation_guess=continuation_guess,
            return_profile=True,
            verb=verb,
        )
        if not bool(result.get("success", False)):
            return None, result, nB_0minus, T_0minus

        u_0minus = float(result.get("u_0minus", np.nan))
        if (not np.isfinite(u_0minus)) or u_0minus < 0.0:
            raise RuntimeError(
                f"uNmax returned a non-physical u_0minus={u_0minus!r} at z={z_value:.9g} km"
            )
        beta_0minus = float(u_0minus / np.sqrt(1.0 + u_0minus * u_0minus))
        velocity_km_s = float(const.c_km * beta_0minus)
        if (not np.isfinite(velocity_km_s)) or velocity_km_s < 0.0:
            raise RuntimeError(
                f"uNmax produced a non-physical velocity at z={z_value:.9g} km"
            )

        sample = {
            "z": z_value,
            "nB_0minus": nB_0minus,
            "T_0minus": T_0minus,
            "u_0minus": u_0minus,
            "velocity_km_s": velocity_km_s,
            "T_inf": float(result.get("T_inf", np.nan)),
            "T_0plus": float(result.get("T_0plus", np.nan)),
        }
        return sample, result, nB_0minus, T_0minus

    def failed_sample_message(z_value, nB_0minus, T_0minus, failure, subdivisions):
        last_success = successful_samples[-1]["z"] if successful_samples else np.nan
        return (
            "uNmax velocity solve failed at "
            f"z={float(z_value):.9g} km, nB_0minus={float(nB_0minus):.9g} MeV^3, "
            f"T_0minus={float(T_0minus):.9g} MeV after {int(subdivisions)} adaptive "
            f"subdivision(s); last successful z={float(last_success):.9g} km: "
            f"{failure.get('message', 'unknown failure')}"
        )

    for nominal_index, nominal_z in enumerate(nominal_z_descending):
        pending = [(float(nominal_z), 0, False)]
        while pending:
            z_value, subdivisions, is_inserted = pending.pop()
            sample, result, nB_0minus, T_0minus = solve_velocity_sample(
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
                    and T_0plus == 0.0
                ):
                    had_successful_sample = bool(successful_samples)
                    successful_samples.append(
                        {
                            "z": float(z_value),
                            "nB_0minus": float(nB_0minus),
                            "T_0minus": float(T_0minus),
                            "u_0minus": 0.0,
                            "velocity_km_s": 0.0,
                            "T_inf": np.nan,
                            "T_0plus": 0.0,
                        }
                    )
                    if return_solver_results:
                        solver_results.append(dict(result))
                    if not had_successful_sample and z_value > z_stop:
                        try:
                            T_stop = float(temperature_of_z(z_stop))
                        except Exception:
                            T_stop = float(T_0minus)
                        successful_samples.append(
                            {
                                "z": float(z_stop),
                                "nB_0minus": float(
                                    nB_target
                                    + density_slope_n0_per_km * n0 * z_stop
                                ),
                                "T_0minus": T_stop,
                                "u_0minus": 0.0,
                                "velocity_km_s": 0.0,
                                "T_inf": np.nan,
                                "T_0plus": 0.0,
                            }
                        )
                        if return_solver_results:
                            solver_results.append(
                                {
                                    "success": False,
                                    "message": "Synthetic zero-velocity support point below exhausted continuation",
                                    "synthetic_stall_support": True,
                                    "T_0plus": 0.0,
                                }
                            )
                    stalled = True
                    stall_z_km = float(z_value)
                    stall_reason = failed_sample_message(
                        z_value, nB_0minus, T_0minus, result, subdivisions
                    )
                    pending.clear()
                    break
                raise RuntimeError(
                    failed_sample_message(z_value, nB_0minus, T_0minus, result, subdivisions)
                )

            last_success_z = float(successful_samples[-1]["z"])
            midpoint_z = float(np.sqrt(last_success_z * z_value))
            if not (z_value < midpoint_z < last_success_z):
                raise RuntimeError(
                    failed_sample_message(z_value, nB_0minus, T_0minus, result, subdivisions)
                )
            pending.append((z_value, subdivisions + 1, is_inserted))
            pending.append((midpoint_z, subdivisions + 1, True))
        if stalled:
            break

    z_descending = np.asarray([sample["z"] for sample in successful_samples], dtype=float)
    density_descending = np.asarray(
        [sample["nB_0minus"] for sample in successful_samples], dtype=float
    )
    temperature_descending = np.asarray(
        [sample["T_0minus"] for sample in successful_samples], dtype=float
    )
    u_0minus_descending = np.asarray(
        [sample["u_0minus"] for sample in successful_samples], dtype=float
    )
    velocity_descending = np.asarray(
        [sample["velocity_km_s"] for sample in successful_samples], dtype=float
    )
    T_inf_descending = np.asarray(
        [sample["T_inf"] for sample in successful_samples], dtype=float
    )
    T_0plus_descending = np.asarray(
        [sample["T_0plus"] for sample in successful_samples], dtype=float
    )

    z_samples = z_descending[::-1].copy()
    density_samples = density_descending[::-1].copy()
    temperature_samples = temperature_descending[::-1].copy()
    u_0minus_samples = u_0minus_descending[::-1].copy()
    velocity_samples = velocity_descending[::-1].copy()
    T_inf_samples = T_inf_descending[::-1].copy()
    T_0plus_samples = T_0plus_descending[::-1].copy()

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
        "nB_0minus_samples_MeV3": density_samples,
        "T_0minus_samples_MeV": temperature_samples,
        "u_0minus_samples": u_0minus_samples,
        "velocity_samples_km_s": velocity_samples,
        "T_inf_samples_MeV": T_inf_samples,
        "T_0plus_samples_MeV": T_0plus_samples,
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
