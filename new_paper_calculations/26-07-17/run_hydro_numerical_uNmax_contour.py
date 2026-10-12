#!/usr/bin/env python3
"""
Parallel adaptive-log hydro numerical-uNmax velocity contour calculation.

This mirrors run_hydro_analytic_contour.py: it first finds the explicit
A_boundary = 1 boundary with the hydro-analytic endpoint diagnostic, then
evaluates a logarithmic temperature mesh below that line with
phase_velocity.solve_front_energy_conserving_uNmax.  The numerical uNmax
composition is the local kaon fraction a = nK/nB; aQstar* fields are kept as
legacy aliases for old plotting and loading code.

The bag constant is calibrated once from nB_crit(T=0), then held fixed.  At
each temperature the runner solves finite-temperature phase coexistence and
defines metastability as (nB_N - nB_crit(T))/n0.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"matplotlib-cache-{os.getuid()}"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import brentq
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver import phase_velocity


# Tunable run parameters
DEFAULT_OUTPUT_TEMPLATE = "hydro-numerical-uNmax-adaptive-logT-contour__{run_tag}.npy"
DEFAULT_RUN_TAG = "numerical_uNmax_nK_logT_v2"
DEFAULT_TEMPERATURE_MIN_MEV = 1.0e-2
DEFAULT_TEMPERATURE_MAX_MEV = 100.0
DEFAULT_TEMPERATURE_COUNT = 30
DEFAULT_METASTABILITY_MIN = 0.01
DEFAULT_METASTABILITY_MAX = 0.9
DEFAULT_METASTABILITY_COUNT = 20
DEFAULT_NCRIT_OVER_N0 = 3.0
DEFAULT_B14_MEV = np.nan
DEFAULT_XI = 0.0
DEFAULT_MS_MEV = 0.0
DEFAULT_NM_TYPE = "PNM"
DEFAULT_UPB = 5000
DEFAULT_TQSTAR_MEV = 0.0
DEFAULT_TAIL_EPS = 1.0e-8
DEFAULT_N_MESH = 200
DEFAULT_TOL_BVP = 1.0e-4
DEFAULT_MAX_NODES = 10000
DEFAULT_JB_GUESS = np.nan
DEFAULT_TQ_GUESS = np.nan
DEFAULT_JB_BOUNDS = None
DEFAULT_BOUNDARY_PROBE_COUNT = 8
DEFAULT_BOUNDARY_BISECT_STEPS = 10
DEFAULT_BOUNDARY_EPS_MEV = 0.005
DEFAULT_SAVE_EVERY = 10
DEFAULT_SEED_TEMPERATURE_MEV = 10.0
DEFAULT_NEIGHBOR_RETRY = True
DEFAULT_SHOOTING_U_MAX = 1.0e-4
DEFAULT_SHOOTING_JB_LOWER = 1.0e-12

NUMERICAL_CLOSURE_LABEL = "uNmax_direct_TQstar_nK"
BOUNDARY_CLOSURE_LABEL = "hydro_jump_eigenvalue_A_boundary_1"
TEMPERATURE_MESH_LABEL = "adaptive_log"
TEMPERATURE_AXIS_SCALE = "log"
DENSITY_RATIO_DEFINITION = "lambda_n_equals_nB_N_over_nB_Q"


def _default_workers() -> int:
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name)
        if value:
            try:
                workers = int(value)
            except ValueError:
                continue
            if workers > 0:
                return workers
    return max(1, os.cpu_count() or 1)


def _resolve_output_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def _as_finite_float(value: Any, default: float = np.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _calibration_root(func, x_grid: np.ndarray, prefer: float) -> float:
    x_grid = np.asarray(x_grid, dtype=float)
    values = np.full_like(x_grid, np.nan, dtype=float)
    for i, x in enumerate(x_grid):
        try:
            values[i] = float(func(float(x)))
        except Exception:
            values[i] = np.nan

    finite = np.isfinite(values)
    x_grid = x_grid[finite]
    values = values[finite]
    if x_grid.size < 2:
        raise RuntimeError("Not enough finite samples to bracket the calibration root")

    zero_hits = np.where(np.isclose(values, 0.0, atol=1.0e-8, rtol=0.0))[0]
    if zero_hits.size:
        hit = zero_hits[np.argmin(np.abs(x_grid[zero_hits] - float(prefer)))]
        return float(x_grid[hit])

    sign_changes = np.where(values[:-1] * values[1:] < 0.0)[0]
    if sign_changes.size == 0:
        raise RuntimeError("No sign change found for the calibration root")

    midpoints = 0.5 * (x_grid[sign_changes] + x_grid[sign_changes + 1])
    idx = int(sign_changes[np.argmin(np.abs(midpoints - float(prefer)))])
    return float(
        brentq(
            func,
            float(x_grid[idx]),
            float(x_grid[idx + 1]),
            xtol=1.0e-6,
            rtol=1.0e-8,
            maxiter=200,
        )
    )


def _coexistence_root(func, prefer: float = 1150.0) -> float:
    """Bracket the finite-T coexistence root near the T=0 branch efficiently."""
    prefer = float(prefer)
    cached_values: dict[float, float] = {}

    def evaluate(value: float) -> float:
        value = float(value)
        if value not in cached_values:
            try:
                cached_values[value] = float(func(value))
            except Exception:
                cached_values[value] = np.nan
        return cached_values[value]

    for width in (25.0, 50.0, 100.0, 200.0, 350.0, 550.0):
        lower = max(1.0, prefer - width)
        upper = prefer + width
        f_lower = evaluate(lower)
        f_upper = evaluate(upper)
        if not (np.isfinite(f_lower) and np.isfinite(f_upper)):
            continue
        if np.isclose(f_lower, 0.0, atol=1.0e-8, rtol=0.0):
            return float(lower)
        if np.isclose(f_upper, 0.0, atol=1.0e-8, rtol=0.0):
            return float(upper)
        if f_lower * f_upper < 0.0:
            return float(
                brentq(func, lower, upper, xtol=1.0e-6, rtol=1.0e-8, maxiter=200)
            )

    return _calibration_root(
        func,
        np.linspace(500.0, 1800.0, 131),
        prefer=prefer,
    )


def calibrate_b14(nB_crit: float, ms: float, nm_type: str) -> tuple[float, float, float]:
    param = para.paraQMCRMF3
    P_crit0 = float(phase_velocity.PNM_n(nB_crit, 0.0, param=param, NM_type=nm_type))
    muB_crit0 = _calibration_root(
        lambda mu: phase_velocity.PNM(mu, 0.0, param=param, NM_type=nm_type) - P_crit0,
        np.linspace(900.0, 1500.0, 220),
        prefer=1050.0,
    )
    B14 = _calibration_root(
        lambda B: phase_velocity.PQM(muB_crit0, 0.0, B, 0.0, ms=ms) - P_crit0,
        np.linspace(120.0, 260.0, 281),
        prefer=180.0,
    )
    return float(B14), float(muB_crit0), float(P_crit0)


@lru_cache(maxsize=512)
def critical_state_for_temperature(
    T: float,
    B14: float,
    ms: float,
    nm_type: str,
    upB: int = DEFAULT_UPB,
) -> tuple[float, float, float]:
    """Return (muB_crit, P_crit, nB_crit) at fixed T and fixed bag constant."""
    T = float(T)
    B14 = float(B14)
    ms = float(ms)
    nm_type = str(nm_type)
    upB = int(upB)
    if not np.isfinite(T) or T < 0.0:
        raise ValueError("T must be non-negative and finite")
    if not np.isfinite(B14) or B14 <= 0.0:
        raise ValueError("B14 must be positive and finite")

    def pressure_difference(muB: float) -> float:
        return float(
            phase_velocity.PNM(muB, T, param=para.paraQMCRMF3, NM_type=nm_type)
            - phase_velocity.PQM(muB, 0.0, B14, T, ms=ms, upB=upB)
        )

    muB_crit = _coexistence_root(pressure_difference, prefer=1150.0)
    P_crit = float(
        phase_velocity.PQM(muB_crit, 0.0, B14, T, ms=ms, upB=upB)
    )
    nB_crit = float(
        phase_velocity.nB_NM(
            muB_crit,
            T,
            param=para.paraQMCRMF3,
            NM_type=nm_type,
        )
    )
    if not np.all(np.isfinite([muB_crit, P_crit, nB_crit])) or nB_crit <= 0.0:
        raise RuntimeError("Finite-temperature critical state is non-physical")
    return float(muB_crit), float(P_crit), float(nB_crit)


def upstream_density_from_metastability(
    T: float,
    metastability: float,
    n0: float,
    B14: float,
    ms: float,
    nm_type: str,
    upB: int = DEFAULT_UPB,
) -> tuple[float, float, float, float]:
    """Return (nB_N, nB_crit_T, muB_crit_T, P_crit_T) for the requested m."""
    muB_crit_T, P_crit_T, nB_crit_T = critical_state_for_temperature(
        float(T),
        float(B14),
        float(ms),
        str(nm_type),
        int(upB),
    )
    nB_N = float(nB_crit_T + float(metastability) * float(n0))
    if not np.isfinite(nB_N) or nB_N <= 0.0:
        raise RuntimeError("Metastability construction returned non-physical nB_N")
    return nB_N, nB_crit_T, muB_crit_T, P_crit_T


def _nB_residual(muB: float, T: float, nB_target: float, nm_type: str) -> float:
    return float(
        phase_velocity.nB_NM(
            float(muB),
            float(T),
            param=para.paraQMCRMF3,
            NM_type=nm_type,
        )
        - float(nB_target)
    )


def muB_from_nB(nB_target: float, T: float, nm_type: str, prefer: float = 1100.0) -> float:
    nB_target = float(nB_target)
    T = float(T)
    prefer = float(prefer) if np.isfinite(float(prefer)) else 1100.0

    def residual(mu: float) -> float:
        return _nB_residual(mu, T, nB_target, nm_type)

    for width in (20.0, 50.0, 100.0, 200.0, 400.0, 700.0):
        lo = max(1.0, prefer - width)
        hi = prefer + width
        try:
            f_lo = residual(lo)
            f_hi = residual(hi)
        except Exception:
            continue
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
            continue
        tolerance = 1.0e-8 * max(abs(nB_target), 1.0)
        if abs(f_lo) <= tolerance:
            return float(lo)
        if abs(f_hi) <= tolerance:
            return float(hi)
        if f_lo * f_hi < 0.0:
            return float(brentq(residual, lo, hi, xtol=1.0e-7, rtol=1.0e-10, maxiter=100))

    mu_grid = np.linspace(850.0, 1700.0, 80)
    values = np.full_like(mu_grid, np.nan, dtype=float)
    for i, mu_val in enumerate(mu_grid):
        try:
            values[i] = residual(mu_val)
        except Exception:
            values[i] = np.nan

    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        raise RuntimeError("Could not evaluate enough finite nB_NM samples to invert muB_N")

    mu_finite = mu_grid[finite]
    val_finite = values[finite]
    tolerance = 1.0e-8 * max(abs(nB_target), 1.0)
    exact_hits = np.where(np.isclose(val_finite, 0.0, rtol=0.0, atol=tolerance))[0]
    if exact_hits.size:
        hit = exact_hits[np.argmin(np.abs(mu_finite[exact_hits] - prefer))]
        return float(mu_finite[hit])

    sign_changes = np.where(val_finite[:-1] * val_finite[1:] < 0.0)[0]
    if sign_changes.size == 0:
        raise RuntimeError("Could not bracket muB_N for the requested nB_N and T_N")

    midpoints = 0.5 * (mu_finite[sign_changes] + mu_finite[sign_changes + 1])
    idx = int(sign_changes[np.argmin(np.abs(midpoints - prefer))])
    return float(
        brentq(
            residual,
            float(mu_finite[idx]),
            float(mu_finite[idx + 1]),
            xtol=1.0e-7,
            rtol=1.0e-10,
            maxiter=100,
        )
    )


def _parse_float_pair(value: str | None) -> tuple[float, float] | None:
    if value is None or str(value).strip() == "":
        return None
    parts = [item.strip() for item in str(value).split(",")]
    if len(parts) != 2:
        raise ValueError("Expected a comma-separated pair like 1e-5,5.0")
    lo, hi = (float(parts[0]), float(parts[1]))
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError("Pair bounds must satisfy 0 < lo < hi")
    return (lo, hi)


def resolve_jB_bounds_for_point(
    nB_N: float,
    explicit_jB_bounds: tuple[float, float] | None,
    auto_jB_bounds: bool,
    shooting_u_max: float,
    shooting_jB_lower: float,
) -> tuple[tuple[float, float] | None, dict[str, Any]]:
    if explicit_jB_bounds is not None:
        lo, hi = map(float, explicit_jB_bounds)
        diagnostics = {
            "jB_bounds_used": True,
            "jB_lower_bound": lo,
            "jB_upper_bound": hi,
            "shooting_u_max": np.nan,
            "auto_jB_bounds": False,
        }
        return (lo, hi), diagnostics
    if not bool(auto_jB_bounds):
        return None, {
            "jB_bounds_used": False,
            "jB_lower_bound": np.nan,
            "jB_upper_bound": np.nan,
            "shooting_u_max": np.nan,
            "auto_jB_bounds": False,
        }
    nB_N = float(nB_N)
    shooting_u_max = float(shooting_u_max)
    shooting_jB_lower = float(shooting_jB_lower)
    if not (np.isfinite(nB_N) and nB_N > 0.0):
        raise ValueError("nB_N must be positive and finite for automatic jB bounds")
    if not (np.isfinite(shooting_u_max) and shooting_u_max > 0.0):
        raise ValueError("shooting_u_max must be positive and finite")
    if not (np.isfinite(shooting_jB_lower) and shooting_jB_lower > 0.0):
        raise ValueError("shooting_jB_lower must be positive and finite")
    jB_upper = float(nB_N * shooting_u_max)
    if not (shooting_jB_lower < jB_upper):
        raise ValueError("automatic jB bounds require shooting_jB_lower < nB_N * shooting_u_max")
    diagnostics = {
        "jB_bounds_used": True,
        "jB_lower_bound": shooting_jB_lower,
        "jB_upper_bound": jB_upper,
        "shooting_u_max": shooting_u_max,
        "auto_jB_bounds": True,
    }
    return (shooting_jB_lower, jB_upper), diagnostics


def base_row(
    T_N: float,
    metastability: float,
    nB_N: float,
    nB_crit_T: float,
    muB_crit_T: float,
    P_crit_T: float,
    n0: float,
    B14: float,
    xi: float,
    ms: float,
    nm_type: str,
    TQstar: float,
) -> dict[str, Any]:
    return {
        "temperature_mesh": TEMPERATURE_MESH_LABEL,
        "T_MeV": float(T_N),
        "metastability": float(metastability),
        "nB_N_over_n0": float(nB_N / n0),
        "nB_N_MeV3": float(nB_N),
        "nB_crit_T_over_n0": float(nB_crit_T / n0),
        "nB_crit_T_MeV3": float(nB_crit_T),
        "muB_crit_T_MeV": float(muB_crit_T),
        "P_crit_T_MeV4": float(P_crit_T),
        "metastability_definition": "(nB_N - nB_crit(T))/n0",
        "B14_calibrated_MeV": float(B14),
        "xi": float(xi),
        "ms_MeV": float(ms),
        "NM_type": str(nm_type),
        "TQstar_target_MeV": float(TQstar),
        "endpoint_domain_failure": False,
        "analytic_closure": NUMERICAL_CLOSURE_LABEL,
        "boundary_closure": BOUNDARY_CLOSURE_LABEL,
        "solver_variant": NUMERICAL_CLOSURE_LABEL,
        "rate_model": "exact_nonleptonic",
        "composition_definition": "nK_over_local_nB",
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "current_definition": "u_nK_minus_D_dnK_dx",
        "root_method": "solve_bvp_uNmax_nK_parameter_1d",
    }


def failure_values(message: str) -> dict[str, Any]:
    endpoint_domain_failure = (
        "A_boundary must satisfy 0 < A_boundary < 1" in message
        or "aQstar_max must satisfy 0 < aQstar_max < 1" in message
    )
    values = {
        "success": False,
        "message": repr(RuntimeError(message)),
        "endpoint_domain_failure": bool(endpoint_domain_failure),
        "slow_front_consistent": False,
    }
    for key in (
        "muB_N_MeV",
        "u_N",
        "u_N_m_s",
        "u_N_max_candidate",
        "jB",
        "u_Q",
        "r_gamma_N",
        "r_gamma_Q",
        "lambda_n",
        "a_local_Qstar",
        "a_local_end",
        "aQstar",
        "aQstar_derived",
        "aQstar_max",
        "muB_Q_MeV",
        "T_Q_MeV",
        "T_Qstar_MeV",
        "T_Qstar_residual_MeV",
        "Pi_MeV4",
        "E_MeV4",
        "E_right_residual",
        "E_right_residual_norm",
        "kaon_equation_residual_norm",
        "jK_left_residual_norm",
        "tail_residual_norm",
        "nB_Q_MeV3",
        "nK_Q_MeV3",
        "muB_Qstar_MeV",
        "muK_Qstar_MeV",
        "nB_Qstar_MeV3",
        "nK_Qstar_MeV3",
        "nK_Qstar",
        "invD_Q",
        "eta_Q",
        "gamma_Q",
        "tau_Q",
        "lambda",
        "kappa",
        "s_end",
        "x_end",
        "a_end",
        "nK_end",
        "jK_end",
    ):
        values[key] = np.nan
    for key in (
        "bvp_status",
        "bvp_niter",
        "bvp_nodes",
        "bvp_ode_calls",
        "bvp_bc_calls",
        "bvp_parameter_count",
        "downstream_root_calls",
        "local_root_calls",
        "local_state_failures",
        "qstar_root_calls",
        "profile_state_calls",
        "global_state_builds",
        "global_state_failures",
    ):
        values[key] = -1
    values["bvp_message"] = ""
    values["solver_variant"] = NUMERICAL_CLOSURE_LABEL
    values["rate_model"] = "exact_nonleptonic"
    values["composition_definition"] = "nK_over_local_nB"
    values["density_ratio_definition"] = DENSITY_RATIO_DEFINITION
    values["current_definition"] = "u_nK_minus_D_dnK_dx"
    values["root_method"] = "solve_bvp_uNmax_nK_parameter_1d"
    return values


def _boundary_failure_values(message: str) -> dict[str, Any]:
    endpoint_domain_failure = (
        "A_boundary must satisfy 0 < A_boundary < 1" in message
        or "aQstar_max must satisfy 0 < aQstar_max < 1" in message
    )
    return {
        "success": False,
        "message": repr(RuntimeError(message)),
        "muB_N_MeV": np.nan,
        "u_N": np.nan,
        "u_N_m_s": np.nan,
        "u_N_squared": np.nan,
        "u_N_formula_squared": np.nan,
        "analytic_velocity_residual": np.nan,
        "jB": np.nan,
        "u_Q": np.nan,
        "gamma_N": np.nan,
        "gamma_Q": np.nan,
        "muB_bar_MeV": np.nan,
        "muB_Q_MeV": np.nan,
        "T_Q_MeV": np.nan,
        "mu_q_MeV": np.nan,
        "nB_Q_MeV3": np.nan,
        "lambda_n": np.nan,
        "a_N": np.nan,
        "A_boundary": np.nan,
        "interface_fraction_mode": "",
        "aQstar_max": np.nan,
        "muKstar_max_MeV": np.nan,
        "muK_Qstar_MeV": np.nan,
        "gamma_MeV": np.nan,
        "tau_MeV_inv": np.nan,
        "tau_seconds": np.nan,
        "slow_front_consistent": False,
        "endpoint_domain_failure": bool(endpoint_domain_failure),
    }


def calculate_boundary_point(task: tuple[float, float, float, float, float, float, float, str, int]) -> dict[str, Any]:
    T_N, metastability, n0, nB_crit, B14, xi, ms, nm_type, upB = task
    row = {
        "temperature_mesh": TEMPERATURE_MESH_LABEL,
        "T_MeV": float(T_N),
        "metastability": float(metastability),
        "nB_N_over_n0": np.nan,
        "nB_N_MeV3": np.nan,
        "nB_crit_T_over_n0": np.nan,
        "nB_crit_T_MeV3": np.nan,
        "muB_crit_T_MeV": np.nan,
        "P_crit_T_MeV4": np.nan,
        "metastability_definition": "(nB_N - nB_crit(T))/n0",
        "B14_calibrated_MeV": float(B14),
        "xi": float(xi),
        "ms_MeV": float(ms),
        "NM_type": str(nm_type),
        "endpoint_domain_failure": False,
        "analytic_closure": BOUNDARY_CLOSURE_LABEL,
    }

    try:
        nB_N, nB_crit_T, muB_crit_T, P_crit_T = upstream_density_from_metastability(
            T_N,
            metastability,
            n0,
            B14,
            ms,
            nm_type,
            upB,
        )
        row.update(
            {
                "nB_N_over_n0": float(nB_N / n0),
                "nB_N_MeV3": float(nB_N),
                "nB_crit_T_over_n0": float(nB_crit_T / n0),
                "nB_crit_T_MeV3": float(nB_crit_T),
                "muB_crit_T_MeV": float(muB_crit_T),
                "P_crit_T_MeV4": float(P_crit_T),
            }
        )
        muB_N = muB_from_nB(nB_N, float(T_N), str(nm_type), prefer=1100.0)
        result = phase_velocity.analytic_velocity_bound(
            muB_N,
            float(T_N),
            float(B14),
            xi=float(xi),
            ms=float(ms),
            param=para.paraQMCRMF3,
            NM_type=str(nm_type),
            upB=int(upB),
            initial_guess=None,
            interface_fraction_mode="extreme_endothermic",
        )
        row.update(
            {
                "success": True,
                "message": result.get("message", ""),
                "muB_N_MeV": float(muB_N),
                "u_N": float(result["u_N_max"]),
                "u_N_m_s": float(result["u_N_max"] * const.m_per_sec),
                "u_N_squared": float(result["u_N_squared"]),
                "u_N_formula_squared": float(result["u_N_formula_squared"]),
                "analytic_velocity_residual": float(result["analytic_velocity_residual"]),
                "jB": float(result["jB"]),
                "u_Q": float(result["u_Q"]),
                "gamma_N": float(result["gamma_N"]),
                "gamma_Q": float(result["gamma_Q"]),
                "muB_bar_MeV": float(result["muB_bar"]),
                "muB_Q_MeV": float(result["muB_Q"]),
                "T_Q_MeV": float(result["T_Q"]),
                "mu_q_MeV": float(result["mu_q"]),
                "nB_Q_MeV3": float(result["nB_Q"]),
                "lambda_n": _as_finite_float(result.get("lambda_n", result.get("a_N"))),
                "a_N": _as_finite_float(result.get("a_N", result.get("lambda_n"))),
                "A_boundary": _as_finite_float(result.get("A_boundary", result.get("aQstar_max"))),
                "interface_fraction_mode": str(result.get("interface_fraction_mode", "extreme_endothermic")),
                "aQstar_max": _as_finite_float(result.get("A_boundary", result.get("aQstar_max"))),
                "muKstar_max_MeV": float(result["muKstar_max"]),
                "muK_Qstar_MeV": float(result["muKstar_max"]),
                "gamma_MeV": float(result["gamma"]),
                "tau_MeV_inv": float(result["tau"]),
                "tau_seconds": float(result["tau_seconds"]),
                "slow_front_consistent": bool(result["slow_front_consistent"]),
                "endpoint_domain_failure": False,
            }
        )
    except Exception as exc:
        row.update(_boundary_failure_values(str(exc)))
    return row


def solve_boundary_for_metastability(
    task: tuple[float, float, float, float, float, float, str, int, float, float, int, int]
) -> dict[str, Any]:
    (
        metastability,
        n0,
        nB_crit,
        B14,
        xi,
        ms,
        nm_type,
        upB,
        temperature_min,
        temperature_max,
        probe_count,
        bisect_steps,
    ) = task

    probe_count = max(2, int(probe_count))
    bisect_steps = max(0, int(bisect_steps))
    probes = np.linspace(float(temperature_min), float(temperature_max), probe_count)
    evaluations: list[dict[str, Any]] = []

    for T_N in probes:
        row = calculate_boundary_point(
            (
                float(T_N),
                float(metastability),
                float(n0),
                float(nB_crit),
                float(B14),
                float(xi),
                float(ms),
                str(nm_type),
                int(upB),
            )
        )
        row["boundary_stage"] = "probe"
        evaluations.append(row)

    success_indices = [i for i, row in enumerate(evaluations) if bool(row.get("success", False))]
    endpoint_indices = [i for i, row in enumerate(evaluations) if bool(row.get("endpoint_domain_failure", False))]

    status = "no_endpoint_found"
    lower_T = np.nan
    upper_T = np.nan
    cut_T = float(temperature_max)

    if not success_indices:
        status = "no_valid_probe"
        cut_T = float(temperature_min)
    elif not endpoint_indices:
        status = "all_probes_valid"
        cut_T = float(temperature_max)
        lower_T = float(probes[success_indices[-1]])
    else:
        transition = None
        for idx in endpoint_indices:
            previous_successes = [j for j in success_indices if j < idx]
            if previous_successes:
                transition = (previous_successes[-1], idx)
                break

        if transition is None:
            status = "endpoint_below_first_valid_probe"
            cut_T = float(temperature_min)
            upper_T = float(probes[endpoint_indices[0]])
        else:
            lo_idx, hi_idx = transition
            lower_T = float(probes[lo_idx])
            upper_T = float(probes[hi_idx])
            status = "bracketed"
            for _ in range(bisect_steps):
                mid_T = 0.5 * (lower_T + upper_T)
                row = calculate_boundary_point(
                    (
                        float(mid_T),
                        float(metastability),
                        float(n0),
                        float(nB_crit),
                        float(B14),
                        float(xi),
                        float(ms),
                        str(nm_type),
                        int(upB),
                    )
                )
                row["boundary_stage"] = "bisect"
                evaluations.append(row)
                if bool(row.get("success", False)):
                    lower_T = float(mid_T)
                else:
                    upper_T = float(mid_T)
                    if not bool(row.get("endpoint_domain_failure", False)):
                        status = "bracketed_with_nonendpoint_failure"
            cut_T = float(lower_T)

    finite_A = [
        float(row["A_boundary"])
        for row in evaluations
        if np.isfinite(_as_finite_float(row.get("A_boundary")))
    ]
    return {
        "metastability": float(metastability),
        "temperature_cut_MeV": float(cut_T),
        "temperature_lower_bracket_MeV": float(lower_T) if np.isfinite(lower_T) else np.nan,
        "temperature_upper_bracket_MeV": float(upper_T) if np.isfinite(upper_T) else np.nan,
        "status": status,
        "success": bool(np.isfinite(cut_T) and cut_T > float(temperature_min)),
        "probe_count": int(probe_count),
        "bisect_steps": int(bisect_steps),
        "A_boundary_min": float(np.min(finite_A)) if finite_A else np.nan,
        "A_boundary_max": float(np.max(finite_A)) if finite_A else np.nan,
        "aQstar_min": float(np.min(finite_A)) if finite_A else np.nan,
        "aQstar_max": float(np.max(finite_A)) if finite_A else np.nan,
        "evaluations": evaluations,
    }


def calculate_point(
    task: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        str,
        int,
        float,
        float,
        int,
        float,
        int,
        float | None,
        float | None,
        tuple[float, float] | None,
    ]
) -> dict[str, Any]:
    (
        T_N,
        metastability,
        n0,
        nB_crit,
        B14,
        xi,
        ms,
        nm_type,
        upB,
        TQstar,
        tail_eps,
        n_mesh,
        tol_bvp,
        max_nodes,
        jB_guess,
        TQ_guess,
        jB_bounds,
    ) = task

    row = base_row(
        T_N,
        metastability,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        n0,
        B14,
        xi,
        ms,
        nm_type,
        TQstar,
    )
    try:
        nB_N, nB_crit_T, muB_crit_T, P_crit_T = upstream_density_from_metastability(
            T_N,
            metastability,
            n0,
            B14,
            ms,
            nm_type,
            upB,
        )
        row.update(
            {
                "nB_N_over_n0": float(nB_N / n0),
                "nB_N_MeV3": float(nB_N),
                "nB_crit_T_over_n0": float(nB_crit_T / n0),
                "nB_crit_T_MeV3": float(nB_crit_T),
                "muB_crit_T_MeV": float(muB_crit_T),
                "P_crit_T_MeV4": float(P_crit_T),
            }
        )
        muB_N = muB_from_nB(nB_N, float(T_N), str(nm_type), prefer=1100.0)
        result = phase_velocity.solve_front_energy_conserving_uNmax(
            float(T_N),
            nB_N,
            float(B14),
            TQstar=float(TQstar),
            ms=float(ms),
            param=para.paraQMCRMF3,
            NM_type=str(nm_type),
            tail_eps=float(tail_eps),
            n_mesh=int(n_mesh),
            tol_bvp=float(tol_bvp),
            max_nodes=int(max_nodes),
            jB_guess=jB_guess,
            TQ_guess=TQ_guess,
            jB_bounds=jB_bounds,
            return_profile=False,
            verb=False,
        )
        success = bool(result.get("success", False))
        u_N = _as_finite_float(result.get("u_N_max_candidate", result.get("u_N")))
        row.update(
            {
                "success": success,
                "message": result.get("message", ""),
                "muB_N_MeV": float(muB_N),
                "u_N": u_N,
                "u_N_m_s": float(u_N * const.m_per_sec) if np.isfinite(u_N) else np.nan,
                "u_N_max_candidate": u_N,
                "jB": _as_finite_float(result.get("jB")),
                "u_Q": _as_finite_float(result.get("u_Q")),
                "r_gamma_N": _as_finite_float(result.get("r_gamma_N")),
                "r_gamma_Q": _as_finite_float(result.get("r_gamma_Q")),
                "lambda_n": _as_finite_float(result.get("lambda_n")),
                "a_local_Qstar": _as_finite_float(result.get("aQstar_derived", result.get("aQstar"))),
                "a_local_end": _as_finite_float(result.get("a_end")),
                "aQstar": _as_finite_float(result.get("aQstar")),
                "aQstar_derived": _as_finite_float(result.get("aQstar_derived", result.get("aQstar"))),
                "aQstar_max": _as_finite_float(result.get("aQstar_derived", result.get("aQstar"))),
                "muB_Q_MeV": _as_finite_float(result.get("muB_Q")),
                "T_Q_MeV": _as_finite_float(result.get("T_Q")),
                "T_Qstar_MeV": _as_finite_float(result.get("T_Qstar")),
                "T_Qstar_residual_MeV": _as_finite_float(result.get("T_Qstar_residual")),
                "Pi_MeV4": _as_finite_float(result.get("Pi")),
                "E_MeV4": _as_finite_float(result.get("E")),
                "E_right_residual": _as_finite_float(result.get("E_right_residual")),
                "E_right_residual_norm": _as_finite_float(result.get("E_right_residual_norm")),
                "kaon_equation_residual_norm": _as_finite_float(result.get("kaon_equation_residual_norm")),
                "jK_left_residual_norm": _as_finite_float(result.get("jK_left_residual_norm")),
                "tail_residual_norm": _as_finite_float(result.get("tail_residual_norm")),
                "nB_Q_MeV3": _as_finite_float(result.get("nB_Q")),
                "nK_Q_MeV3": _as_finite_float(result.get("nK_Q")),
                "muB_Qstar_MeV": _as_finite_float(result.get("muB_Qstar")),
                "muK_Qstar_MeV": _as_finite_float(result.get("muK_Qstar")),
                "nB_Qstar_MeV3": _as_finite_float(result.get("nB_Qstar")),
                "nK_Qstar_MeV3": _as_finite_float(result.get("nK_Qstar")),
                "nK_Qstar": _as_finite_float(result.get("nK_Qstar")),
                "invD_Q": _as_finite_float(result.get("invD_Q")),
                "eta_Q": _as_finite_float(result.get("eta_Q")),
                "gamma_Q": _as_finite_float(result.get("gamma_Q")),
                "tau_Q": _as_finite_float(result.get("tau_Q")),
                "lambda": _as_finite_float(result.get("lambda")),
                "kappa": _as_finite_float(result.get("kappa")),
                "s_end": _as_finite_float(result.get("s_end")),
                "x_end": _as_finite_float(result.get("x_end")),
                "a_end": _as_finite_float(result.get("a_end")),
                "nK_end": _as_finite_float(result.get("nK_end")),
                "jK_end": _as_finite_float(result.get("jK_end")),
                "slow_front_consistent": bool(np.isfinite(u_N) and abs(u_N) < 1.0),
                "endpoint_domain_failure": False,
                "solver_variant": str(result.get("solver_variant", NUMERICAL_CLOSURE_LABEL)),
                "rate_model": str(result.get("rate_model", "")),
                "composition_definition": str(result.get("composition_definition", "")),
                "density_ratio_definition": str(result.get("density_ratio_definition", DENSITY_RATIO_DEFINITION)),
                "current_definition": str(result.get("current_definition", "")),
                "root_method": str(result.get("_root_method", "")),
                "bvp_status": int(result.get("bvp_status", -1)),
                "bvp_message": str(result.get("bvp_message", "")),
                "bvp_niter": int(result.get("bvp_niter", -1)),
                "bvp_nodes": int(result.get("bvp_nodes", -1)),
                "bvp_ode_calls": int(result.get("bvp_ode_calls", -1)),
                "bvp_bc_calls": int(result.get("bvp_bc_calls", -1)),
                "bvp_parameter_count": int(np.asarray(result.get("bvp_parameters", []), dtype=float).size),
                "downstream_root_calls": int(result.get("downstream_root_calls", -1)),
                "local_root_calls": int(result.get("local_root_calls", -1)),
                "local_state_failures": int(result.get("local_state_failures", -1)),
                "qstar_root_calls": int(result.get("qstar_root_calls", -1)),
                "profile_state_calls": int(result.get("profile_state_calls", -1)),
                "global_state_builds": int(result.get("global_state_builds", -1)),
                "global_state_failures": int(result.get("global_state_failures", -1)),
                "upB": int(upB),
            }
        )
    except Exception as exc:
        row.update(failure_values(str(exc)))
    return row


def calculate_velocity_point(
    task: tuple[
        int,
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        str,
        int,
        float,
        float,
        int,
        float,
        int,
        float | None,
        float | None,
        tuple[float, float] | None,
        float,
        float,
        str,
        str,
        float,
        int,
        tuple[int, int] | None,
        bool,
        float,
        float,
    ]
) -> dict[str, Any]:
    (
        row_index,
        col_index,
        T_N,
        metastability,
        n0,
        nB_crit,
        B14,
        xi,
        ms,
        nm_type,
        upB,
        TQstar,
        tail_eps,
        n_mesh,
        tol_bvp,
        max_nodes,
        jB_guess,
        TQ_guess,
        jB_bounds,
        temperature_cut,
        boundary_eps,
        jB_guess_source,
        continuation_direction,
        continuation_seed_temperature,
        retry_attempt,
        retry_of,
        auto_jB_bounds,
        shooting_u_max,
        shooting_jB_lower,
    ) = task

    try:
        nB_N = upstream_density_from_metastability(
            T_N,
            metastability,
            n0,
            B14,
            ms,
            nm_type,
            upB,
        )[0]
        point_jB_bounds, bounds_diagnostics = resolve_jB_bounds_for_point(
            nB_N,
            jB_bounds,
            bool(auto_jB_bounds),
            float(shooting_u_max),
            float(shooting_jB_lower),
        )
    except Exception as exc:
        point_jB_bounds = jB_bounds
        bounds_diagnostics = {
            "jB_bounds_used": bool(jB_bounds is not None),
            "jB_lower_bound": float(jB_bounds[0]) if jB_bounds is not None else np.nan,
            "jB_upper_bound": float(jB_bounds[1]) if jB_bounds is not None else np.nan,
            "shooting_u_max": np.nan,
            "auto_jB_bounds": False,
            "jB_bounds_error": str(exc),
        }
    row = calculate_point(
        (
            float(T_N),
            float(metastability),
            float(n0),
            float(nB_crit),
            float(B14),
            float(xi),
            float(ms),
            str(nm_type),
            int(upB),
            float(TQstar),
            float(tail_eps),
            int(n_mesh),
            float(tol_bvp),
            int(max_nodes),
            jB_guess,
            TQ_guess,
            point_jB_bounds,
        )
    )
    row["adaptive_row_index"] = int(row_index)
    row["adaptive_col_index"] = int(col_index)
    row["temperature_cut_MeV"] = float(temperature_cut)
    row["boundary_eps"] = float(boundary_eps)
    row["jB_guess_used"] = _as_finite_float(jB_guess)
    row["jB_guess_source"] = str(jB_guess_source)
    row["continuation_direction"] = str(continuation_direction)
    row["continuation_seed_temperature_MeV"] = float(continuation_seed_temperature)
    row["retry_attempt"] = int(retry_attempt)
    row["retry_of"] = retry_of
    row.update(bounds_diagnostics)
    if int(retry_attempt) > 0 and bool(row.get("success", False)):
        row["retry_replaced_failure"] = True
    return {
        "row_index": int(row_index),
        "col_index": int(col_index),
        "T_MeV": float(T_N),
        "metastability": float(metastability),
        "row": row,
    }


def _base_jB_guess_source(jB_guess: float | None) -> str:
    return "user" if jB_guess is not None and np.isfinite(float(jB_guess)) else "default"


def _successful_jB(point_result: dict[str, Any] | None) -> float | None:
    if not point_result:
        return None
    row = point_result.get("row", {})
    if not bool(row.get("success", False)):
        return None
    jB = _as_finite_float(row.get("jB"))
    return float(jB) if np.isfinite(jB) and jB > 0.0 else None


def choose_seed_row_index(column_temperature_values: np.ndarray, seed_temperature: float) -> int:
    values = np.asarray(column_temperature_values, dtype=float)
    if values.size == 0:
        raise ValueError("column_temperature_values must be non-empty")
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("column_temperature_values must contain a finite value")
    seed_temperature = float(seed_temperature)
    if not np.isfinite(seed_temperature):
        seed_temperature = DEFAULT_SEED_TEMPERATURE_MEV
    finite_indices = np.where(finite)[0]
    idx = finite_indices[np.argmin(np.abs(values[finite_indices] - seed_temperature))]
    return int(idx)


def build_adaptive_temperature_column(
    temperature_min: float,
    temperature_max: float,
    temperature_cut: float,
    boundary_eps: float,
    temperature_count: int,
) -> np.ndarray:
    temperature_min = float(temperature_min)
    temperature_max = float(temperature_max)
    temperature_cut = float(temperature_cut)
    boundary_eps = float(boundary_eps)
    temperature_count = int(temperature_count)

    if not np.isfinite(temperature_min) or temperature_min <= 0.0:
        raise ValueError("temperature_min must be positive and finite for a logarithmic mesh")
    if not np.isfinite(temperature_max) or temperature_max <= temperature_min:
        raise ValueError("temperature_max must be finite and larger than temperature_min")
    if not np.isfinite(temperature_cut):
        raise ValueError("temperature_cut must be finite")
    if not np.isfinite(boundary_eps) or boundary_eps < 0.0:
        raise ValueError("boundary_eps must be non-negative and finite")
    if temperature_count <= 0:
        raise ValueError("temperature_count must be positive")

    upper = min(temperature_max, temperature_cut - boundary_eps)
    if upper <= temperature_min:
        raise ValueError("A_boundary=1 leaves no safe logarithmic temperature interval")
    return np.geomspace(temperature_min, upper, temperature_count)


def _make_velocity_task(
    row_index: int,
    col_index: int,
    T_N: float,
    metastability: float,
    n0: float,
    nB_crit: float,
    B14: float,
    xi: float,
    ms: float,
    nm_type: str,
    upB: int,
    TQstar: float,
    tail_eps: float,
    n_mesh: int,
    tol_bvp: float,
    max_nodes: int,
    jB_guess: float | None,
    TQ_guess: float | None,
    jB_bounds: tuple[float, float] | None,
    temperature_cut: float,
    boundary_eps: float,
    jB_guess_source: str,
    continuation_direction: str,
    continuation_seed_temperature: float,
    retry_attempt: int = 0,
    retry_of: tuple[int, int] | None = None,
    auto_jB_bounds: bool = True,
    shooting_u_max: float = DEFAULT_SHOOTING_U_MAX,
    shooting_jB_lower: float = DEFAULT_SHOOTING_JB_LOWER,
) -> tuple:
    return (
        int(row_index),
        int(col_index),
        float(T_N),
        float(metastability),
        float(n0),
        float(nB_crit),
        float(B14),
        float(xi),
        float(ms),
        str(nm_type),
        int(upB),
        float(TQstar),
        float(tail_eps),
        int(n_mesh),
        float(tol_bvp),
        int(max_nodes),
        jB_guess,
        TQ_guess,
        jB_bounds,
        float(temperature_cut),
        float(boundary_eps),
        str(jB_guess_source),
        str(continuation_direction),
        float(continuation_seed_temperature),
        int(retry_attempt),
        retry_of,
        bool(auto_jB_bounds),
        float(shooting_u_max),
        float(shooting_jB_lower),
    )


def _column_solve_order(n_rows: int, seed_row_index: int) -> list[tuple[int, str]]:
    seed_row_index = int(np.clip(int(seed_row_index), 0, int(n_rows) - 1))
    order = [(seed_row_index, "seed")]
    order.extend((i, "up") for i in range(seed_row_index + 1, int(n_rows)))
    order.extend((i, "down") for i in range(seed_row_index - 1, -1, -1))
    return order


def _make_column_point_task(
    column_task: tuple,
    row_index: int,
    direction: str,
    solved_by_row: dict[int, dict[str, Any]],
) -> tuple:
    (
        col_index,
        metastability,
        column_temperature_values,
        n0,
        nB_crit,
        B14,
        xi,
        ms,
        nm_type,
        upB,
        TQstar,
        tail_eps,
        n_mesh,
        tol_bvp,
        max_nodes,
        base_jB_guess,
        TQ_guess,
        jB_bounds,
        temperature_cut,
        boundary_eps,
        seed_temperature,
        auto_jB_bounds,
        shooting_u_max,
        shooting_jB_lower,
    ) = column_task
    column_temperature_values = np.asarray(column_temperature_values, dtype=float)
    base_source = _base_jB_guess_source(base_jB_guess)
    guess = base_jB_guess
    source = base_source
    if direction == "up":
        neighbor_jB = _successful_jB(solved_by_row.get(int(row_index) - 1))
        if neighbor_jB is not None:
            guess = neighbor_jB
            source = "same_column_lower_neighbor"
        else:
            source = "fallback_after_failed_neighbor"
    elif direction == "down":
        neighbor_jB = _successful_jB(solved_by_row.get(int(row_index) + 1))
        if neighbor_jB is not None:
            guess = neighbor_jB
            source = "same_column_upper_neighbor"
        else:
            source = "fallback_after_failed_neighbor"

    return _make_velocity_task(
        int(row_index),
        int(col_index),
        float(column_temperature_values[int(row_index)]),
        float(metastability),
        float(n0),
        float(nB_crit),
        float(B14),
        float(xi),
        float(ms),
        str(nm_type),
        int(upB),
        float(TQstar),
        float(tail_eps),
        int(n_mesh),
        float(tol_bvp),
        int(max_nodes),
        guess,
        TQ_guess,
        jB_bounds,
        float(temperature_cut),
        float(boundary_eps),
        source,
        direction,
        float(seed_temperature),
        auto_jB_bounds=bool(auto_jB_bounds),
        shooting_u_max=float(shooting_u_max),
        shooting_jB_lower=float(shooting_jB_lower),
    )


def _column_solve_plan(column_task: tuple) -> dict[str, Any]:
    col_index = int(column_task[0])
    column_temperature_values = np.asarray(column_task[2], dtype=float)
    seed_temperature = float(column_task[20])
    seed_row = choose_seed_row_index(column_temperature_values, seed_temperature)
    return {
        "col_index": col_index,
        "task": column_task,
        "order": _column_solve_order(column_temperature_values.size, seed_row),
        "order_pos": 0,
        "solved_by_row": {},
    }


def _next_column_point_task(plan: dict[str, Any]) -> tuple | None:
    order = plan["order"]
    pos = int(plan["order_pos"])
    if pos >= len(order):
        return None
    row_index, direction = order[pos]
    return _make_column_point_task(plan["task"], row_index, direction, plan["solved_by_row"])


def _remember_column_point(plan: dict[str, Any], point_result: dict[str, Any]) -> None:
    plan["solved_by_row"][int(point_result["row_index"])] = point_result
    plan["order_pos"] = int(plan["order_pos"]) + 1


def solve_velocity_column(task: tuple) -> dict[str, Any]:
    plan = _column_solve_plan(task)
    results: list[dict[str, Any]] = []

    while True:
        point_task = _next_column_point_task(plan)
        if point_task is None:
            break
        point = calculate_velocity_point(point_task)
        _remember_column_point(plan, point)
        results.append(point)
    return {"col_index": int(plan["col_index"]), "results": results}


def _iter_velocity_points_serial(column_tasks: list[tuple]):
    for column_task in column_tasks:
        plan = _column_solve_plan(column_task)
        while True:
            point_task = _next_column_point_task(plan)
            if point_task is None:
                break
            point = calculate_velocity_point(point_task)
            _remember_column_point(plan, point)
            yield point


def _iter_velocity_points_parallel(column_tasks: list[tuple], workers: int):
    plans = [_column_solve_plan(task) for task in column_tasks]
    pending = {}
    max_workers = max(1, min(int(workers), len(plans)))

    def submit_next(pool, plan):
        point_task = _next_column_point_task(plan)
        if point_task is None:
            return
        pending[pool.submit(calculate_velocity_point, point_task)] = plan

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for plan in plans:
            submit_next(pool, plan)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                plan = pending.pop(future)
                point = future.result()
                _remember_column_point(plan, point)
                yield point
                submit_next(pool, plan)


def _immediate_retry_seed(
    row_index: int,
    col_index: int,
    successful_cells: dict[tuple[int, int], dict[str, Any]],
) -> tuple[float | None, str]:
    candidates = (
        ((row_index - 1, col_index), "retry_same_column_lower_neighbor"),
        ((row_index + 1, col_index), "retry_same_column_upper_neighbor"),
        ((row_index, col_index - 1), "retry_same_row_left_neighbor"),
        ((row_index, col_index + 1), "retry_same_row_right_neighbor"),
    )
    for key, source in candidates:
        jB = _successful_jB(successful_cells.get(key))
        if jB is not None:
            return jB, source
    return None, ""


def save_payload(
    path: Path,
    temperature_values: np.ndarray,
    boundary_temperature_values: np.ndarray,
    boundary_metastability_values: np.ndarray,
    temperature_grid: np.ndarray,
    metastability_grid: np.ndarray,
    velocity_grid_m_s: np.ndarray,
    u_N_grid: np.ndarray,
    a_local_grid: np.ndarray,
    records: list[dict[str, Any]],
    boundary_records: list[dict[str, Any]],
    B14: float,
    n0: float,
    nB_crit: float,
    xi: float,
    ms: float,
    nm_type: str,
    run_tag: str,
    boundary_eps: float,
    TQstar: float,
) -> None:
    payload = {
        "temperature_mesh": TEMPERATURE_MESH_LABEL,
        "temperature_axis_scale": TEMPERATURE_AXIS_SCALE,
        "temperature_values": np.asarray(temperature_values, dtype=float),
        "metastability_values": np.asarray(boundary_metastability_values, dtype=float),
        "boundary_temperature_values": np.asarray(boundary_temperature_values, dtype=float),
        "boundary_metastability_values": np.asarray(boundary_metastability_values, dtype=float),
        "temperature_grid": np.asarray(temperature_grid, dtype=float),
        "metastability_grid": np.asarray(metastability_grid, dtype=float),
        "velocity_grid_m_s": np.asarray(velocity_grid_m_s, dtype=float),
        "u_N_grid": np.asarray(u_N_grid, dtype=float),
        "a_local_grid": np.asarray(a_local_grid, dtype=float),
        "aQstar_grid": np.asarray(a_local_grid, dtype=float),
        "records": list(records),
        "boundary_records": list(boundary_records),
        "B14": float(B14),
        "n0": float(n0),
        "nB_crit": float(nB_crit),
        "nB_crit0": float(nB_crit),
        "nB_crit0_over_n0": float(nB_crit / n0),
        "metastability_definition": "(nB_N - nB_crit(T))/n0",
        "critical_state_definition": "P_N(muB,T) = P_Q(muB,muK=0,T;B_fixed)",
        "xi": float(xi),
        "ms_MeV": float(ms),
        "NM_type": str(nm_type),
        "analytic_closure": NUMERICAL_CLOSURE_LABEL,
        "boundary_closure": BOUNDARY_CLOSURE_LABEL,
        "solver_variant": NUMERICAL_CLOSURE_LABEL,
        "rate_model": "exact_nonleptonic",
        "composition_definition": "nK_over_local_nB",
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "current_definition": "u_nK_minus_D_dnK_dx",
        "root_method": "solve_bvp_uNmax_nK_parameter_1d",
        "TQstar_target_MeV": float(TQstar),
        "adaptive": True,
        "adaptive_orientation": "temperature_cut_by_metastability",
        "run_tag": str(run_tag),
        "boundary_eps": float(boundary_eps),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, payload, allow_pickle=True)
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="Output .npy path. Relative paths are resolved next to this script.")
    parser.add_argument("--run-tag", default=os.getenv("RUN_TAG", DEFAULT_RUN_TAG), help="Version tag used in the default output filename.")
    parser.add_argument("--workers", type=int, default=_default_workers(), help="Number of parallel worker processes.")
    parser.add_argument("--temperature-min", type=float, default=DEFAULT_TEMPERATURE_MIN_MEV, help="Minimum nuclear temperature in MeV.")
    parser.add_argument("--temperature-max", type=float, default=DEFAULT_TEMPERATURE_MAX_MEV, help="Maximum nuclear temperature in MeV.")
    parser.add_argument("--temperature-count", type=int, default=DEFAULT_TEMPERATURE_COUNT, help="Number of adaptive logarithmic temperature points per metastability column.")
    parser.add_argument("--metastability-min", type=float, default=DEFAULT_METASTABILITY_MIN, help="Minimum (nB_N - nB_crit(T))/n0 value.")
    parser.add_argument("--metastability-max", type=float, default=DEFAULT_METASTABILITY_MAX, help="Maximum (nB_N - nB_crit(T))/n0 value.")
    parser.add_argument("--metastability-count", type=int, default=DEFAULT_METASTABILITY_COUNT, help="Number of metastability grid points.")
    parser.add_argument("--ncrit-over-n0", type=float, default=DEFAULT_NCRIT_OVER_N0, help="Critical density at T=0 in units of n0; used only to calibrate B.")
    parser.add_argument("--B14", type=float, default=DEFAULT_B14_MEV, help="Optional B^(1/4) in MeV. If omitted, it is calibrated.")
    parser.add_argument("--xi", type=float, default=DEFAULT_XI, help="Boundary diagnostic parameter; retained for payload compatibility.")
    parser.add_argument("--ms", type=float, default=DEFAULT_MS_MEV, help="Strange-quark mass in MeV.")
    parser.add_argument("--nm-type", default=DEFAULT_NM_TYPE, help="Nuclear matter type passed to phase_velocity.")
    parser.add_argument("--upB", type=int, default=DEFAULT_UPB, help="Upper bound parameter used by the analytic boundary diagnostic.")
    parser.add_argument("--TQstar", type=float, default=DEFAULT_TQSTAR_MEV, help="Fixed interface temperature passed to numerical uNmax, in MeV.")
    parser.add_argument("--tail-eps", type=float, default=DEFAULT_TAIL_EPS, help="Compact-domain tail epsilon for uNmax.")
    parser.add_argument("--n-mesh", type=int, default=DEFAULT_N_MESH, help="Initial BVP mesh size for uNmax.")
    parser.add_argument("--tol-bvp", type=float, default=DEFAULT_TOL_BVP, help="BVP tolerance for uNmax.")
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES, help="Maximum BVP nodes for uNmax.")
    parser.add_argument("--jB-guess", type=float, default=DEFAULT_JB_GUESS, help="Optional jB seed for every uNmax grid point.")
    parser.add_argument("--TQ-guess", type=float, default=DEFAULT_TQ_GUESS, help="Optional downstream endpoint T_Q seed for every uNmax grid point; not a BVP parameter.")
    parser.add_argument("--jB-bounds", default=DEFAULT_JB_BOUNDS, help="Optional jB bounds as lo,hi.")
    parser.add_argument("--boundary-probe-count", type=int, default=DEFAULT_BOUNDARY_PROBE_COUNT, help="Sparse temperature samples used to bracket A_boundary=1 at each metastability.")
    parser.add_argument("--boundary-bisect-steps", type=int, default=DEFAULT_BOUNDARY_BISECT_STEPS, help="Bisection refinements after the endpoint boundary is bracketed.")
    parser.add_argument("--boundary-eps", type=float, default=DEFAULT_BOUNDARY_EPS_MEV, help="Temperature offset in MeV below the boundary for the adaptive velocity mesh.")
    parser.add_argument("--seed-temperature", type=float, default=DEFAULT_SEED_TEMPERATURE_MEV, help="Mild T_N anchor in MeV used to start each neighbor-seeded column continuation.")
    parser.add_argument("--no-neighbor-retry", action="store_false", dest="neighbor_retry", default=DEFAULT_NEIGHBOR_RETRY, help="Disable one-pass retry of failed cells using immediate successful neighbors.")
    parser.add_argument("--shooting-u-max", type=float, default=DEFAULT_SHOOTING_U_MAX, help="Automatic shooting upper bound on u_N; converted to jB_upper = nB_N * shooting_u_max.")
    parser.add_argument("--shooting-jB-lower", type=float, default=DEFAULT_SHOOTING_JB_LOWER, help="Automatic lower bound for jB when --jB-bounds is not supplied.")
    parser.add_argument("--no-auto-jB-bounds", action="store_false", dest="auto_jB_bounds", default=True, help="Disable automatic velocity-based jB bounds when --jB-bounds is not supplied.")
    parser.add_argument("--force", action="store_true", help="Compatibility flag; adaptive runs recompute and overwrite the selected output.")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Save partial velocity results after this many completed velocity grid points.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.temperature_count <= 0:
        raise ValueError("--temperature-count must be positive")
    if not np.isfinite(float(args.temperature_min)) or float(args.temperature_min) <= 0.0:
        raise ValueError("--temperature-min must be positive and finite for a logarithmic mesh")
    if not np.isfinite(float(args.temperature_max)) or float(args.temperature_max) <= float(args.temperature_min):
        raise ValueError("--temperature-max must be finite and larger than --temperature-min")
    if args.metastability_count <= 0:
        raise ValueError("--metastability-count must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not (-1.0 < float(args.xi) < 1.0):
        raise ValueError("--xi must satisfy -1 < xi < 1")
    if float(args.TQstar) < 0.0:
        raise ValueError("--TQstar must be non-negative")
    if not np.isfinite(float(args.seed_temperature)):
        raise ValueError("--seed-temperature must be finite")
    if float(args.shooting_u_max) <= 0.0 or not np.isfinite(float(args.shooting_u_max)):
        raise ValueError("--shooting-u-max must be positive and finite")
    if float(args.shooting_jB_lower) <= 0.0 or not np.isfinite(float(args.shooting_jB_lower)):
        raise ValueError("--shooting-jB-lower must be positive and finite")
    if int(args.max_nodes) <= int(args.n_mesh):
        raise ValueError("--max-nodes must be larger than --n-mesh")


def _save_current(
    output_path: Path,
    temperature_values: np.ndarray,
    boundary_temperature_values: np.ndarray,
    boundary_metastability_values: np.ndarray,
    temperature_grid: np.ndarray,
    metastability_grid: np.ndarray,
    velocity_grid_m_s: np.ndarray,
    u_N_grid: np.ndarray,
    a_local_grid: np.ndarray,
    records: list[dict[str, Any]],
    boundary_records: list[dict[str, Any]],
    B14: float,
    n0: float,
    nB_crit: float,
    args: argparse.Namespace,
) -> None:
    save_payload(
        output_path,
        temperature_values,
        boundary_temperature_values,
        boundary_metastability_values,
        temperature_grid,
        metastability_grid,
        velocity_grid_m_s,
        u_N_grid,
        a_local_grid,
        records,
        boundary_records,
        B14,
        n0,
        nB_crit,
        float(args.xi),
        float(args.ms),
        str(args.nm_type),
        str(args.run_tag),
        float(args.boundary_eps),
        float(args.TQstar),
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)

    output_name = args.output
    if output_name is None:
        output_name = DEFAULT_OUTPUT_TEMPLATE.format(run_tag=str(args.run_tag))
    output_path = _resolve_output_path(output_name)
    n0 = float(0.16 * const.MeV_fm**3)
    nB_crit = float(args.ncrit_over_n0 * n0)
    jB_guess = None if not np.isfinite(float(args.jB_guess)) else float(args.jB_guess)
    TQ_guess = None if not np.isfinite(float(args.TQ_guess)) else float(args.TQ_guess)
    jB_bounds = _parse_float_pair(args.jB_bounds)
    auto_jB_bounds = bool(args.auto_jB_bounds) and jB_bounds is None

    if np.isfinite(float(args.B14)) and float(args.B14) > 0.0:
        B14 = float(args.B14)
        muB_crit0 = np.nan
        P_crit0 = np.nan
    else:
        print(f"Calibrating B^(1/4) from nB_crit(T=0) = {args.ncrit_over_n0:g} n0 ...", flush=True)
        B14, muB_crit0, P_crit0 = calibrate_b14(nB_crit, float(args.ms), str(args.nm_type))
        print(f"Calibrated B^(1/4) = {B14:.10g} MeV", flush=True)

    temperature_values = np.geomspace(float(args.temperature_min), float(args.temperature_max), int(args.temperature_count))
    metastability_values = np.linspace(float(args.metastability_min), float(args.metastability_max), int(args.metastability_count))
    boundary_metastability_values = metastability_values.copy()
    boundary_temperature_values = np.full_like(metastability_values, np.nan, dtype=float)
    temperature_grid = np.full((len(temperature_values), len(metastability_values)), np.nan, dtype=float)
    metastability_grid = np.repeat(metastability_values[None, :], len(temperature_values), axis=0)
    velocity_grid_m_s = np.full_like(metastability_grid, np.nan, dtype=float)
    u_N_grid = np.full_like(metastability_grid, np.nan, dtype=float)
    a_local_grid = np.full_like(metastability_grid, np.nan, dtype=float)
    records: list[dict[str, Any]] = []
    boundary_records: list[dict[str, Any]] = []

    boundary_tasks = [
        (
            float(metastability),
            float(n0),
            float(nB_crit),
            float(B14),
            float(args.xi),
            float(args.ms),
            str(args.nm_type),
            int(args.upB),
            float(args.temperature_min),
            float(args.temperature_max),
            int(args.boundary_probe_count),
            int(args.boundary_bisect_steps),
        )
        for metastability in metastability_values
    ]

    print(f"Output: {output_path}", flush=True)
    print(f"Run tag: {args.run_tag}", flush=True)
    print(f"Numerical uNmax TQstar: {float(args.TQstar):.10g} MeV", flush=True)
    print(f"Adaptive log grid: {len(temperature_values)} local temperatures x {len(metastability_values)} metastability values", flush=True)
    print(f"Boundary probe count: {int(args.boundary_probe_count)}, bisection steps: {int(args.boundary_bisect_steps)}", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    if int(args.workers) == 1:
        progress = tqdm(boundary_tasks, total=len(boundary_tasks), desc="A_boundary=1 boundary", unit="col", dynamic_ncols=True, mininterval=0.5)
        for task in progress:
            boundary_row = solve_boundary_for_metastability(task)
            progress.set_postfix_str(f"m={float(boundary_row['metastability']):.3g}", refresh=False)
            boundary_records.append(boundary_row)
            idx = int(np.argmin(np.abs(metastability_values - float(boundary_row["metastability"]))))
            boundary_temperature_values[idx] = float(boundary_row["temperature_cut_MeV"])
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = [pool.submit(solve_boundary_for_metastability, task) for task in boundary_tasks]
            progress = tqdm(as_completed(futures), total=len(futures), desc="A_boundary=1 boundary", unit="col", dynamic_ncols=True, mininterval=0.5)
            for future in progress:
                boundary_row = future.result()
                progress.set_postfix_str(f"m={float(boundary_row['metastability']):.3g}", refresh=False)
                boundary_records.append(boundary_row)
                idx = int(np.argmin(np.abs(metastability_values - float(boundary_row["metastability"]))))
                boundary_temperature_values[idx] = float(boundary_row["temperature_cut_MeV"])

    _save_current(
        output_path,
        temperature_values,
        boundary_temperature_values,
        boundary_metastability_values,
        temperature_grid,
        metastability_grid,
        velocity_grid_m_s,
        u_N_grid,
        a_local_grid,
        records,
        boundary_records,
        B14,
        n0,
        nB_crit,
        args,
    )

    column_tasks = []
    skipped_velocity_columns = []
    total_velocity_points = 0
    for j, metastability in enumerate(metastability_values):
        temperature_cut = float(boundary_temperature_values[j])
        upper = temperature_cut - float(args.boundary_eps)
        if (not np.isfinite(upper)) or upper <= float(args.temperature_min):
            skipped_velocity_columns.append(
                {
                    "col_index": int(j),
                    "metastability": float(metastability),
                    "temperature_cut_MeV": temperature_cut,
                    "status": "no_safe_temperature_interval",
                }
            )
            continue
        column_temperature_values = build_adaptive_temperature_column(
            temperature_min=float(args.temperature_min),
            temperature_max=float(args.temperature_max),
            temperature_cut=temperature_cut,
            boundary_eps=float(args.boundary_eps),
            temperature_count=int(args.temperature_count),
        )
        total_velocity_points += int(column_temperature_values.size)
        column_tasks.append(
            (
                int(j),
                float(metastability),
                np.asarray(column_temperature_values, dtype=float),
                float(n0),
                float(nB_crit),
                float(B14),
                float(args.xi),
                float(args.ms),
                str(args.nm_type),
                int(args.upB),
                float(args.TQstar),
                float(args.tail_eps),
                int(args.n_mesh),
                float(args.tol_bvp),
                int(args.max_nodes),
                jB_guess,
                TQ_guess,
                jB_bounds,
                temperature_cut,
                float(args.boundary_eps),
                float(args.seed_temperature),
                bool(auto_jB_bounds),
                float(args.shooting_u_max),
                float(args.shooting_jB_lower),
            )
        )

    completed_since_save = 0
    print(
        f"Velocity grid tasks: {total_velocity_points} point solves "
        f"({len(skipped_velocity_columns)} skipped metastability columns)",
        flush=True,
    )

    def _store_velocity_point(point_result: dict[str, Any]) -> None:
        row = point_result["row"]
        i = int(point_result["row_index"])
        j = int(point_result["col_index"])
        temperature_grid[i, j] = float(point_result["T_MeV"])
        records.append(row)
        if bool(row.get("success", False)):
            velocity_grid_m_s[i, j] = float(row["u_N_m_s"])
            u_N_grid[i, j] = float(row["u_N"])
            a_local_grid[i, j] = float(row["a_local_Qstar"])

    latest_by_cell: dict[tuple[int, int], dict[str, Any]] = {}
    successful_cells: dict[tuple[int, int], dict[str, Any]] = {}
    first_pass_failures: set[tuple[int, int]] = set()

    def _accept_point_result(point_result: dict[str, Any], *, first_pass: bool) -> None:
        nonlocal completed_since_save
        i = int(point_result["row_index"])
        j = int(point_result["col_index"])
        key = (i, j)
        _store_velocity_point(point_result)
        latest_by_cell[key] = point_result
        if bool(point_result["row"].get("success", False)):
            successful_cells[key] = point_result
        elif first_pass:
            first_pass_failures.add(key)
        completed_since_save += 1
        if args.save_every > 0 and completed_since_save >= int(args.save_every):
            _save_current(output_path, temperature_values, boundary_temperature_values, boundary_metastability_values, temperature_grid, metastability_grid, velocity_grid_m_s, u_N_grid, a_local_grid, records, boundary_records, B14, n0, nB_crit, args)
            completed_since_save = 0

    progress = tqdm(total=total_velocity_points, desc="Numerical uNmax contour", unit="pt", dynamic_ncols=True, mininterval=0.5)
    try:
        if int(args.workers) == 1:
            for point_result in _iter_velocity_points_serial(column_tasks):
                progress.set_postfix_str(
                    f"m={float(point_result['metastability']):.3g}, T={float(point_result['T_MeV']):.3g}",
                    refresh=False,
                )
                _accept_point_result(point_result, first_pass=True)
                progress.update(1)
        elif total_velocity_points > 0:
            for point_result in _iter_velocity_points_parallel(column_tasks, int(args.workers)):
                progress.set_postfix_str(
                    f"m={float(point_result['metastability']):.3g}, T={float(point_result['T_MeV']):.3g}",
                    refresh=False,
                )
                _accept_point_result(point_result, first_pass=True)
                progress.update(1)
        else:
            print("No velocity grid tasks were generated; saving boundary-only payload.", flush=True)

        retry_tasks = []
        if bool(args.neighbor_retry) and first_pass_failures:
            retry_seed_cells = dict(successful_cells)
            for i, j in sorted(first_pass_failures):
                if (i, j) in retry_seed_cells:
                    continue
                seed_jB, seed_source = _immediate_retry_seed(i, j, retry_seed_cells)
                if seed_jB is None:
                    continue
                original = latest_by_cell[(i, j)]
                retry_tasks.append(
                    _make_velocity_task(
                        i,
                        j,
                        float(original["T_MeV"]),
                        float(original["metastability"]),
                        float(n0),
                        float(nB_crit),
                        float(B14),
                        float(args.xi),
                        float(args.ms),
                        str(args.nm_type),
                        int(args.upB),
                        float(args.TQstar),
                        float(args.tail_eps),
                        int(args.n_mesh),
                        float(args.tol_bvp),
                        int(args.max_nodes),
                        float(seed_jB),
                        TQ_guess,
                        jB_bounds,
                        float(original["row"].get("temperature_cut_MeV", original.get("T_MeV", np.nan))),
                        float(args.boundary_eps),
                        seed_source,
                        "retry",
                        float(args.seed_temperature),
                        1,
                        (i, j),
                        auto_jB_bounds=bool(auto_jB_bounds),
                        shooting_u_max=float(args.shooting_u_max),
                        shooting_jB_lower=float(args.shooting_jB_lower),
                    )
                )
        if retry_tasks:
            progress.total = int(progress.total or 0) + len(retry_tasks)
            progress.refresh()
            if int(args.workers) == 1:
                for task in retry_tasks:
                    point_result = calculate_velocity_point(task)
                    progress.set_postfix_str(
                        f"retry m={float(point_result['metastability']):.3g}, T={float(point_result['T_MeV']):.3g}",
                        refresh=False,
                    )
                    _accept_point_result(point_result, first_pass=False)
                    progress.update(1)
            else:
                with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
                    futures = [pool.submit(calculate_velocity_point, task) for task in retry_tasks]
                    for future in as_completed(futures):
                        point_result = future.result()
                        progress.set_postfix_str(
                            f"retry m={float(point_result['metastability']):.3g}, T={float(point_result['T_MeV']):.3g}",
                            refresh=False,
                        )
                        _accept_point_result(point_result, first_pass=False)
                        progress.update(1)
    finally:
        progress.close()
    _save_current(
        output_path,
        temperature_values,
        boundary_temperature_values,
        boundary_metastability_values,
        temperature_grid,
        metastability_grid,
        velocity_grid_m_s,
        u_N_grid,
        a_local_grid,
        records,
        boundary_records,
        B14,
        n0,
        nB_crit,
        args,
    )

    successes = sum(1 for row in records if bool(row.get("success", False)))
    other_failures = len(records) - successes
    print("Done.", flush=True)
    print(f"Saved rows: {len(records)}", flush=True)
    print(f"Saved boundary rows: {len(boundary_records)}", flush=True)
    print(f"success={successes}, failure={other_failures}", flush=True)
    print(f"finite boundary points={int(np.isfinite(boundary_temperature_values).sum())}", flush=True)
    print(f"B14={B14:.10g} MeV, muB_crit0={muB_crit0:.10g}, P_crit0={P_crit0:.10g}", flush=True)


if __name__ == "__main__":
    main()
