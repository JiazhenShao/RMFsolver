#!/usr/bin/env python3
"""
Parallel adaptive hydro numerical-uNmax velocity contour calculation.

This mirrors run_hydro_analytic_contour.py: it first finds the saved
aQstar_max = 1 boundary with the hydro-analytic endpoint diagnostic, then
evaluates the velocity mesh below that line with
phase_velocity.solve_front_energy_conserving_uNmax.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
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


DEFAULT_OUTPUT_TEMPLATE = "26SU-2-hydro-numerical-uNmax-adaptive-contour__{run_tag}.npy"
NUMERICAL_CLOSURE_LABEL = "uNmax_direct_TQstar_nK"
BOUNDARY_CLOSURE_LABEL = "hydro_jump_eigenvalue_aQstar_max_1"
TEMPERATURE_MESH_LABEL = "adaptive_linear"


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


def base_row(
    T_N: float,
    metastability: float,
    nB_N: float,
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
        "B14_calibrated_MeV": float(B14),
        "xi": float(xi),
        "ms_MeV": float(ms),
        "NM_type": str(nm_type),
        "TQstar_target_MeV": float(TQstar),
        "endpoint_domain_failure": False,
        "analytic_closure": NUMERICAL_CLOSURE_LABEL,
        "boundary_closure": BOUNDARY_CLOSURE_LABEL,
    }


def failure_values(message: str) -> dict[str, Any]:
    endpoint_domain_failure = "aQstar_max must satisfy 0 < aQstar_max < 1" in message
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
        "a_N",
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
        "global_state_builds",
        "global_state_failures",
    ):
        values[key] = -1
    values["bvp_message"] = ""
    values["solver_variant"] = NUMERICAL_CLOSURE_LABEL
    values["rate_model"] = "exact_nonleptonic"
    values["composition_definition"] = "nK_over_local_nB"
    values["current_definition"] = "u_nK_minus_D_dnK_dx"
    return values


def _boundary_failure_values(message: str) -> dict[str, Any]:
    endpoint_domain_failure = "aQstar_max must satisfy 0 < aQstar_max < 1" in message
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
        "a_N": np.nan,
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
    nB_N = float(nB_crit + float(metastability) * n0)
    row = {
        "temperature_mesh": TEMPERATURE_MESH_LABEL,
        "T_MeV": float(T_N),
        "metastability": float(metastability),
        "nB_N_over_n0": float(nB_N / n0),
        "nB_N_MeV3": float(nB_N),
        "B14_calibrated_MeV": float(B14),
        "xi": float(xi),
        "ms_MeV": float(ms),
        "NM_type": str(nm_type),
        "endpoint_domain_failure": False,
        "analytic_closure": BOUNDARY_CLOSURE_LABEL,
    }

    try:
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
                "a_N": float(result["a_N"]),
                "aQstar_max": float(result["aQstar_max"]),
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

    finite_a = [
        float(row["aQstar_max"])
        for row in evaluations
        if np.isfinite(_as_finite_float(row.get("aQstar_max")))
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
        "aQstar_min": float(np.min(finite_a)) if finite_a else np.nan,
        "aQstar_max": float(np.max(finite_a)) if finite_a else np.nan,
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

    nB_N = float(nB_crit + float(metastability) * n0)
    row = base_row(T_N, metastability, nB_N, n0, B14, xi, ms, nm_type, TQstar)
    try:
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
                "a_N": _as_finite_float(result.get("a_N")),
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
                "current_definition": str(result.get("current_definition", "")),
                "root_method": str(result.get("_root_method", "")),
                "bvp_status": int(result.get("bvp_status", -1)),
                "bvp_message": str(result.get("bvp_message", "")),
                "bvp_niter": int(result.get("bvp_niter", -1)),
                "bvp_nodes": int(result.get("bvp_nodes", -1)),
                "bvp_ode_calls": int(result.get("bvp_ode_calls", -1)),
                "bvp_bc_calls": int(result.get("bvp_bc_calls", -1)),
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
    ) = task

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
            jB_bounds,
        )
    )
    row["adaptive_row_index"] = int(row_index)
    row["adaptive_col_index"] = int(col_index)
    row["temperature_cut_MeV"] = float(temperature_cut)
    row["boundary_eps"] = float(boundary_eps)
    return {
        "row_index": int(row_index),
        "col_index": int(col_index),
        "T_MeV": float(T_N),
        "metastability": float(metastability),
        "row": row,
    }


def save_payload(
    path: Path,
    temperature_values: np.ndarray,
    boundary_temperature_values: np.ndarray,
    boundary_metastability_values: np.ndarray,
    temperature_grid: np.ndarray,
    metastability_grid: np.ndarray,
    velocity_grid_m_s: np.ndarray,
    u_N_grid: np.ndarray,
    aQstar_grid: np.ndarray,
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
        "temperature_axis_scale": TEMPERATURE_MESH_LABEL,
        "temperature_values": np.asarray(temperature_values, dtype=float),
        "metastability_values": np.asarray(boundary_metastability_values, dtype=float),
        "boundary_temperature_values": np.asarray(boundary_temperature_values, dtype=float),
        "boundary_metastability_values": np.asarray(boundary_metastability_values, dtype=float),
        "temperature_grid": np.asarray(temperature_grid, dtype=float),
        "metastability_grid": np.asarray(metastability_grid, dtype=float),
        "velocity_grid_m_s": np.asarray(velocity_grid_m_s, dtype=float),
        "u_N_grid": np.asarray(u_N_grid, dtype=float),
        "aQstar_grid": np.asarray(aQstar_grid, dtype=float),
        "records": list(records),
        "boundary_records": list(boundary_records),
        "B14": float(B14),
        "n0": float(n0),
        "nB_crit": float(nB_crit),
        "xi": float(xi),
        "ms_MeV": float(ms),
        "NM_type": str(nm_type),
        "analytic_closure": NUMERICAL_CLOSURE_LABEL,
        "boundary_closure": BOUNDARY_CLOSURE_LABEL,
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
    parser.add_argument("--run-tag", default=os.getenv("RUN_TAG", "numerical_uNmax_v1"), help="Version tag used in the default output filename.")
    parser.add_argument("--workers", type=int, default=_default_workers(), help="Number of parallel worker processes.")
    parser.add_argument("--temperature-min", type=float, default=0.1, help="Minimum nuclear temperature in MeV.")
    parser.add_argument("--temperature-max", type=float, default=100.0, help="Maximum nuclear temperature in MeV.")
    parser.add_argument("--temperature-count", type=int, default=150, help="Number of linear temperature grid points.")
    parser.add_argument("--metastability-min", type=float, default=0.01, help="Minimum (nB_N - nB_crit)/n0 value.")
    parser.add_argument("--metastability-max", type=float, default=0.9, help="Maximum (nB_N - nB_crit)/n0 value.")
    parser.add_argument("--metastability-count", type=int, default=35, help="Number of metastability grid points.")
    parser.add_argument("--ncrit-over-n0", type=float, default=3.0, help="Critical density at T=0 in units of n0.")
    parser.add_argument("--B14", type=float, default=np.nan, help="Optional B^(1/4) in MeV. If omitted, it is calibrated.")
    parser.add_argument("--xi", type=float, default=0.0, help="Boundary diagnostic parameter; retained for payload compatibility.")
    parser.add_argument("--ms", type=float, default=0.0, help="Strange-quark mass in MeV.")
    parser.add_argument("--nm-type", default="PNM", help="Nuclear matter type passed to phase_velocity.")
    parser.add_argument("--upB", type=int, default=5000, help="Upper bound parameter used by the analytic boundary diagnostic.")
    parser.add_argument("--TQstar", type=float, default=0.5, help="Fixed interface temperature passed to numerical uNmax, in MeV.")
    parser.add_argument("--tail-eps", type=float, default=1.0e-8, help="Compact-domain tail epsilon for uNmax.")
    parser.add_argument("--n-mesh", type=int, default=200, help="Initial BVP mesh size for uNmax.")
    parser.add_argument("--tol-bvp", type=float, default=1.0e-4, help="BVP tolerance for uNmax.")
    parser.add_argument("--max-nodes", type=int, default=10000, help="Maximum BVP nodes for uNmax.")
    parser.add_argument("--jB-guess", type=float, default=np.nan, help="Optional jB seed for every uNmax grid point.")
    parser.add_argument("--TQ-guess", type=float, default=np.nan, help="Optional downstream T_Q seed for every uNmax grid point.")
    parser.add_argument("--jB-bounds", default=None, help="Optional jB bounds as lo,hi.")
    parser.add_argument("--boundary-probe-count", type=int, default=10, help="Sparse temperature samples used to bracket aQstar_max=1 at each metastability.")
    parser.add_argument("--boundary-bisect-steps", type=int, default=12, help="Bisection refinements after the endpoint boundary is bracketed.")
    parser.add_argument("--boundary-eps", type=float, default=0.005, help="Temperature offset in MeV below the boundary for the adaptive velocity mesh.")
    parser.add_argument("--force", action="store_true", help="Compatibility flag; adaptive runs recompute and overwrite the selected output.")
    parser.add_argument("--save-every", type=int, default=10, help="Save partial velocity results after this many completed velocity grid points.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.temperature_count <= 0:
        raise ValueError("--temperature-count must be positive")
    if args.metastability_count <= 0:
        raise ValueError("--metastability-count must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not (-1.0 < float(args.xi) < 1.0):
        raise ValueError("--xi must satisfy -1 < xi < 1")
    if float(args.TQstar) < 0.0:
        raise ValueError("--TQstar must be non-negative")
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
    aQstar_grid: np.ndarray,
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
        aQstar_grid,
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

    if np.isfinite(float(args.B14)) and float(args.B14) > 0.0:
        B14 = float(args.B14)
        muB_crit0 = np.nan
        P_crit0 = np.nan
    else:
        print(f"Calibrating B^(1/4) from nB_crit(T=0) = {args.ncrit_over_n0:g} n0 ...", flush=True)
        B14, muB_crit0, P_crit0 = calibrate_b14(nB_crit, float(args.ms), str(args.nm_type))
        print(f"Calibrated B^(1/4) = {B14:.10g} MeV", flush=True)

    temperature_values = np.linspace(float(args.temperature_min), float(args.temperature_max), int(args.temperature_count))
    metastability_values = np.linspace(float(args.metastability_min), float(args.metastability_max), int(args.metastability_count))
    boundary_metastability_values = metastability_values.copy()
    boundary_temperature_values = np.full_like(metastability_values, np.nan, dtype=float)
    temperature_grid = np.full((len(temperature_values), len(metastability_values)), np.nan, dtype=float)
    metastability_grid = np.repeat(metastability_values[None, :], len(temperature_values), axis=0)
    velocity_grid_m_s = np.full_like(metastability_grid, np.nan, dtype=float)
    u_N_grid = np.full_like(metastability_grid, np.nan, dtype=float)
    aQstar_grid = np.full_like(metastability_grid, np.nan, dtype=float)
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
    print(f"Adaptive grid: {len(temperature_values)} local temperatures x {len(metastability_values)} metastability values", flush=True)
    print(f"Boundary probe count: {int(args.boundary_probe_count)}, bisection steps: {int(args.boundary_bisect_steps)}", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    if int(args.workers) == 1:
        progress = tqdm(boundary_tasks, total=len(boundary_tasks), desc="aQstar_max=1 boundary", unit="col", dynamic_ncols=True, mininterval=0.5)
        for task in progress:
            boundary_row = solve_boundary_for_metastability(task)
            progress.set_postfix_str(f"m={float(boundary_row['metastability']):.3g}", refresh=False)
            boundary_records.append(boundary_row)
            idx = int(np.argmin(np.abs(metastability_values - float(boundary_row["metastability"]))))
            boundary_temperature_values[idx] = float(boundary_row["temperature_cut_MeV"])
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = [pool.submit(solve_boundary_for_metastability, task) for task in boundary_tasks]
            progress = tqdm(as_completed(futures), total=len(futures), desc="aQstar_max=1 boundary", unit="col", dynamic_ncols=True, mininterval=0.5)
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
        aQstar_grid,
        records,
        boundary_records,
        B14,
        n0,
        nB_crit,
        args,
    )

    velocity_tasks = []
    skipped_velocity_columns = []
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
        column_temperature_values = np.linspace(float(args.temperature_min), upper, int(args.temperature_count))
        for i, T_N in enumerate(column_temperature_values):
            velocity_tasks.append(
                (
                    int(i),
                    int(j),
                    float(T_N),
                    float(metastability),
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
                )
            )

    completed_since_save = 0
    total_velocity_points = len(velocity_tasks)
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
            aQstar_grid[i, j] = float(row["aQstar_derived"])

    if int(args.workers) == 1:
        progress = tqdm(total=total_velocity_points, desc="Numerical uNmax contour", unit="pt", dynamic_ncols=True, mininterval=0.5)
        for task in velocity_tasks:
            point_result = calculate_velocity_point(task)
            progress.set_postfix_str(
                f"m={float(point_result['metastability']):.3g}, T={float(point_result['T_MeV']):.3g}",
                refresh=False,
            )
            _store_velocity_point(point_result)
            progress.update(1)
            completed_since_save += 1
            if args.save_every > 0 and completed_since_save >= int(args.save_every):
                _save_current(output_path, temperature_values, boundary_temperature_values, boundary_metastability_values, temperature_grid, metastability_grid, velocity_grid_m_s, u_N_grid, aQstar_grid, records, boundary_records, B14, n0, nB_crit, args)
                completed_since_save = 0
    elif total_velocity_points > 0:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = [pool.submit(calculate_velocity_point, task) for task in velocity_tasks]
            progress = tqdm(total=len(futures), desc="Numerical uNmax contour", unit="pt", dynamic_ncols=True, mininterval=0.5)
            for future in as_completed(futures):
                point_result = future.result()
                progress.set_postfix_str(
                    f"m={float(point_result['metastability']):.3g}, T={float(point_result['T_MeV']):.3g}",
                    refresh=False,
                )
                _store_velocity_point(point_result)
                progress.update(1)
                completed_since_save += 1
                if args.save_every > 0 and completed_since_save >= int(args.save_every):
                    _save_current(output_path, temperature_values, boundary_temperature_values, boundary_metastability_values, temperature_grid, metastability_grid, velocity_grid_m_s, u_N_grid, aQstar_grid, records, boundary_records, B14, n0, nB_crit, args)
                    completed_since_save = 0
    else:
        print("No velocity grid tasks were generated; saving boundary-only payload.", flush=True)
    _save_current(
        output_path,
        temperature_values,
        boundary_temperature_values,
        boundary_metastability_values,
        temperature_grid,
        metastability_grid,
        velocity_grid_m_s,
        u_N_grid,
        aQstar_grid,
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
