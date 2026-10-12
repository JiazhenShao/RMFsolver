#!/usr/bin/env python3
"""
Parallel adaptive LTE hydro-analytic velocity contour calculation.

This script first finds the selected analytic abundance boundary as T_cut at
fixed metastability. In LTE mode the boundary is a_interface_LTE = 1; in legacy
extreme-endothermic mode it is A_boundary = 1. It then computes an
adaptive velocity mesh below that boundary and writes a versioned .npy file in
the same directory as this script.
"""

from __future__ import annotations

import argparse
import inspect
import math
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
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if (PROJECT_ROOT / "RMFsolver").is_dir() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver import phase_velocity


DEFAULT_OUTPUT_TEMPLATE = "hydro-analytic-{mode}-adaptive-contour__{run_tag}.npy"
ANALYTIC_CLOSURE_LABEL = "hydro_jump_eigenvalue"
TEMPERATURE_MESH_LABEL = "adaptive_linear"
ANALYTIC_AQSTAR_MAX_MODE = "LTE"
DENSITY_RATIO_DEFINITION = "lambda_n_equals_nB_N_over_nB_Q"
DEFAULT_RUN_TAG = "analytic_LTE_v1"
DEFAULT_TEMPERATURE_MIN_MEV = 0.1
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
DEFAULT_BOUNDARY_PROBE_COUNT = 8
DEFAULT_BOUNDARY_BISECT_STEPS = 10
DEFAULT_BOUNDARY_EPS_MEV = 0.005
DEFAULT_SAVE_EVERY = 10


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


def _point_key(T: float, metastability: float) -> tuple[float, float]:
    return (round(float(T), 12), round(float(metastability), 12))


def _as_finite_float(value: Any, default: float = np.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _message_counts(rows: list[dict[str, Any]], limit: int = 5) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        message = str(row.get("message", ""))
        if not message:
            continue
        counts[message] = counts.get(message, 0) + 1
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]


def _normalize_aqstar_max_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode == "lte":
        return "LTE"
    if mode == "extreme_endothermic":
        return "extreme_endothermic"
    raise ValueError("--aQstar-max must be 'LTE' or 'extreme_endothermic'")


def _validate_phase_velocity_runtime(module=phase_velocity) -> str:
    signature = inspect.signature(module.analytic_velocity_bound)
    source_path = getattr(module, "__file__", None)
    if source_path is None:
        source_path = inspect.getfile(module.analytic_velocity_bound)
    source_path = str(source_path)
    if "aQstar_max" not in signature.parameters:
        raise RuntimeError(
            "Imported RMFsolver.phase_velocity does not support "
            "analytic_velocity_bound(..., aQstar_max=...). "
            f"Imported from {source_path}. Sync/update RMFsolver on the cluster "
            "or run from the source tree containing the LTE update."
        )
    return source_path


def _boundary_quantity_name(aQstar_max_mode: str) -> str:
    mode = _normalize_aqstar_max_mode(aQstar_max_mode)
    if mode == "LTE":
        return "a_interface_LTE"
    return "A_boundary"


def _boundary_semantics(aQstar_max_mode: str) -> str:
    return f"{_boundary_quantity_name(aQstar_max_mode)} = 1"


def _is_endpoint_domain_failure(message: str, aQstar_max_mode: str) -> bool:
    mode = _normalize_aqstar_max_mode(aQstar_max_mode)
    if mode == "LTE":
        return "a_interface_LTE must satisfy 0 < a_interface_LTE < 1" in str(message)
    return "A_boundary must satisfy 0 < A_boundary < 1" in str(message)


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
    """Return B^(1/4), muB_crit(T=0), and P_crit(T=0)."""
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


def base_row(
    T_N: float,
    metastability: float,
    nB_N: float,
    n0: float,
    B14: float,
    xi: float,
    ms: float,
    nm_type: str,
    aQstar_max_mode: str,
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
        "aQstar_max_mode": str(aQstar_max_mode),
        "endpoint_domain_failure": False,
        "analytic_closure": ANALYTIC_CLOSURE_LABEL,
        "solver_variant": ANALYTIC_CLOSURE_LABEL,
        "composition_definition": "a_local_equals_nK_over_nB",
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "root_method": "analytic_velocity_bound",
    }


def failure_values(message: str, aQstar_max_mode: str) -> dict[str, Any]:
    endpoint_domain_failure = _is_endpoint_domain_failure(
        message,
        aQstar_max_mode,
    )
    return {
        "success": False,
        "message": repr(RuntimeError(message)),
        "solver_variant": ANALYTIC_CLOSURE_LABEL,
        "composition_definition": "a_local_equals_nK_over_nB",
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "root_method": "analytic_velocity_bound",
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
        "P_N_MeV4": np.nan,
        "P_Q_MeV4": np.nan,
        "pressure_jump_MeV4": np.nan,
        "pressure_jump_balance_MeV4": np.nan,
        "pressure_jump_residual_MeV4": np.nan,
        "energy_flux_residual": np.nan,
        "momentum_flux_residual": np.nan,
        "endpoint_scaled_residual": np.nan,
        "mu_q_MeV": np.nan,
        "nB_Q_MeV3": np.nan,
        "h_N_MeV4": np.nan,
        "h_Q_MeV4": np.nan,
        "h_over_nB_N_MeV": np.nan,
        "h_over_nB_Q_MeV": np.nan,
        "lambda_n": np.nan,
        "A_boundary": np.nan,
        "a_interface": np.nan,
        "a_interface_LTE": np.nan,
        "interface_fraction_mode": str(aQstar_max_mode),
        "a_N": np.nan,
        "aQstar_max": np.nan,
        "A_extreme_endothermic": np.nan,
        "aQstar_LTE": np.nan,
        "aQstar_used": np.nan,
        "beta_LTE": np.nan,
        "lte_correction": np.nan,
        "one_minus_aQstar_used": np.nan,
        "one_plus_xi_aQstar_used": np.nan,
        "muKstar_max_MeV": np.nan,
        "muK_Qstar_MeV": np.nan,
        "gamma_MeV": np.nan,
        "tau_MeV_inv": np.nan,
        "tau_seconds": np.nan,
        "slow_front_consistent": False,
        "endpoint_domain_failure": bool(endpoint_domain_failure),
    }


def calculate_point(task: tuple[float, float, float, float, float, float, float, str, int, str]) -> dict[str, Any]:
    T_N, metastability, n0, nB_crit, B14, xi, ms, nm_type, upB, aQstar_max_mode = task
    nB_N = float(nB_crit + float(metastability) * n0)
    row = base_row(
        T_N,
        metastability,
        nB_N,
        n0,
        B14,
        xi,
        ms,
        nm_type,
        aQstar_max_mode,
    )

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
            aQstar_max=str(aQstar_max_mode),
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
                "P_N_MeV4": float(result["P_N"]),
                "P_Q_MeV4": float(result["P_Q"]),
                "pressure_jump_MeV4": float(result["pressure_jump"]),
                "pressure_jump_balance_MeV4": float(result["pressure_jump_balance"]),
                "pressure_jump_residual_MeV4": float(result["pressure_jump_residual"]),
                "energy_flux_residual": float(result["energy_flux_residual"]),
                "momentum_flux_residual": float(result["momentum_flux_residual"]),
                "endpoint_scaled_residual": float(result["endpoint_scaled_residual"]),
                "mu_q_MeV": float(result["mu_q"]),
                "nB_N_MeV3_check": float(result["nB_N"]),
                "nB_Q_MeV3": float(result["nB_Q"]),
                "h_N_MeV4": float(result["h_N"]),
                "h_Q_MeV4": float(result["h_Q"]),
                "h_over_nB_N_MeV": float(result["h_over_nB_N"]),
                "h_over_nB_Q_MeV": float(result["h_over_nB_Q"]),
                "lambda_n": float(result["lambda_n"]),
                "A_boundary": float(result["A_boundary"]),
                "a_interface": float(result["a_interface"]),
                "a_interface_LTE": float(result["a_interface_LTE"]),
                "interface_fraction_mode": str(result["interface_fraction_mode"]),
                "a_N": float(result["a_N"]),
                "aQstar_max": float(result["aQstar_max"]),
                "A_extreme_endothermic": float(result["A_extreme_endothermic"]),
                "aQstar_LTE": float(result["aQstar_LTE"]),
                "aQstar_used": float(result["aQstar_used"]),
                "beta_LTE": float(result["beta_LTE"]),
                "lte_correction": float(result["lte_correction"]),
                "one_minus_aQstar_used": float(result["one_minus_aQstar_used"]),
                "one_plus_xi_aQstar_used": float(result["one_plus_xi_aQstar_used"]),
                "aQstar_max_mode": str(result["aQstar_max_mode"]),
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
        row.update(failure_values(str(exc), str(aQstar_max_mode)))

    return row


def solve_boundary_for_metastability(
    task: tuple[float, float, float, float, float, float, str, int, float, float, int, int, str]
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
        aQstar_max_mode,
    ) = task

    probe_count = max(2, int(probe_count))
    bisect_steps = max(0, int(bisect_steps))
    boundary_quantity_name = _boundary_quantity_name(aQstar_max_mode)
    probes = np.linspace(float(temperature_min), float(temperature_max), probe_count)
    evaluations: list[dict[str, Any]] = []

    for T_N in probes:
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
                str(aQstar_max_mode),
            )
        )
        row["boundary_stage"] = "probe"
        evaluations.append(row)

    success_indices = [i for i, row in enumerate(evaluations) if bool(row.get("success", False))]
    endpoint_indices = [i for i, row in enumerate(evaluations) if bool(row.get("endpoint_domain_failure", False))]

    status = "no_endpoint_found"
    lower_T = np.nan
    upper_T = np.nan
    cut_T = np.nan

    if not success_indices:
        status = "no_valid_probe"
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
            upper_T = float(probes[endpoint_indices[0]])
        else:
            lo_idx, hi_idx = transition
            lower_T = float(probes[lo_idx])
            upper_T = float(probes[hi_idx])
            status = "bracketed"
            for _ in range(bisect_steps):
                mid_T = 0.5 * (lower_T + upper_T)
                row = calculate_point(
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
                        str(aQstar_max_mode),
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

    finite_boundary_quantity = [
        float(row[boundary_quantity_name])
        for row in evaluations
        if np.isfinite(_as_finite_float(row.get(boundary_quantity_name)))
    ]
    return {
        "metastability": float(metastability),
        "temperature_cut_MeV": float(cut_T),
        "temperature_lower_bracket_MeV": float(lower_T) if np.isfinite(lower_T) else np.nan,
        "temperature_upper_bracket_MeV": float(upper_T) if np.isfinite(upper_T) else np.nan,
        "status": status,
        "success": bool(
            status in {
                "bracketed",
                "bracketed_with_nonendpoint_failure",
                "all_probes_valid",
            }
            and np.isfinite(cut_T)
            and cut_T > float(temperature_min)
        ),
        "probe_count": int(probe_count),
        "bisect_steps": int(bisect_steps),
        "boundary_quantity_name": boundary_quantity_name,
        "boundary_target": 1.0,
        "boundary_quantity_value": 1.0,
        "boundary_semantics": _boundary_semantics(aQstar_max_mode),
        "boundary_quantity_min": (
            float(np.min(finite_boundary_quantity))
            if finite_boundary_quantity
            else np.nan
        ),
        "boundary_quantity_max": (
            float(np.max(finite_boundary_quantity))
            if finite_boundary_quantity
            else np.nan
        ),
        "evaluations": evaluations,
    }


def _validate_boundary_pass(
    boundary_records: list[dict[str, Any]],
    expected_count: int,
) -> None:
    if len(boundary_records) != int(expected_count):
        raise RuntimeError(
            f"Boundary pass returned {len(boundary_records)} of {int(expected_count)} columns"
        )

    invalid = [row for row in boundary_records if not bool(row.get("success", False))]
    if not invalid:
        return

    status_counts: dict[str, int] = {}
    evaluations: list[dict[str, Any]] = []
    for row in invalid:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        evaluations.extend(list(row.get("evaluations", [])))

    status_text = ", ".join(
        f"{status}={count}" for status, count in sorted(status_counts.items())
    )
    messages = _message_counts(evaluations, limit=3)
    message_text = "; ".join(
        f"{count}x {message}" for message, count in messages
    )
    if not message_text:
        message_text = "no worker error message was recorded"
    raise RuntimeError(
        "Boundary pass failed; velocity mesh was not started. "
        f"Invalid columns: {len(invalid)}/{int(expected_count)} "
        f"({status_text}). Most common worker errors: {message_text}"
    )


def calculate_velocity_column(
    task: tuple[int, float, float, float, float, float, float, float, str, int, float, float, int, str]
) -> dict[str, Any]:
    (
        col_index,
        metastability,
        temperature_cut,
        n0,
        nB_crit,
        B14,
        xi,
        ms,
        nm_type,
        upB,
        temperature_min,
        boundary_eps,
        temperature_count,
        aQstar_max_mode,
    ) = task

    temperature_count = int(temperature_count)
    temperature_values = np.full(temperature_count, np.nan, dtype=float)
    velocity_values = np.full(temperature_count, np.nan, dtype=float)
    u_N_values = np.full(temperature_count, np.nan, dtype=float)
    aQstar_used_values = np.full(temperature_count, np.nan, dtype=float)
    aQstar_LTE_values = np.full(temperature_count, np.nan, dtype=float)
    A_extreme_endothermic_values = np.full(temperature_count, np.nan, dtype=float)
    records: list[dict[str, Any]] = []

    upper = float(temperature_cut) - float(boundary_eps)
    if (not np.isfinite(upper)) or upper <= float(temperature_min):
        return {
            "col_index": int(col_index),
            "metastability": float(metastability),
            "temperature_values": temperature_values,
            "velocity_values_m_s": velocity_values,
            "u_N_values": u_N_values,
            "aQstar_used_values": aQstar_used_values,
            "aQstar_LTE_values": aQstar_LTE_values,
            "A_extreme_endothermic_values": A_extreme_endothermic_values,
            "records": records,
            "status": "no_safe_temperature_interval",
        }

    temperature_values = np.linspace(float(temperature_min), float(upper), temperature_count)
    for i, T_N in enumerate(temperature_values):
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
                str(aQstar_max_mode),
            )
        )
        row["adaptive_row_index"] = int(i)
        row["adaptive_col_index"] = int(col_index)
        row["temperature_cut_MeV"] = float(temperature_cut)
        row["boundary_eps"] = float(boundary_eps)
        records.append(row)
        if bool(row.get("success", False)):
            velocity_values[i] = float(row["u_N_m_s"])
            u_N_values[i] = float(row["u_N"])
            aQstar_used_values[i] = float(row["aQstar_used"])
            aQstar_LTE_values[i] = float(row["aQstar_LTE"])
            A_extreme_endothermic_values[i] = float(row["A_extreme_endothermic"])

    return {
        "col_index": int(col_index),
        "metastability": float(metastability),
        "temperature_values": temperature_values,
        "velocity_values_m_s": velocity_values,
        "u_N_values": u_N_values,
        "aQstar_used_values": aQstar_used_values,
        "aQstar_LTE_values": aQstar_LTE_values,
        "A_extreme_endothermic_values": A_extreme_endothermic_values,
        "records": records,
        "status": "ok",
    }


def _row_matches(row: dict[str, Any], B14: float, xi: float, ms: float, nm_type: str) -> bool:
    if str(row.get("temperature_mesh", TEMPERATURE_MESH_LABEL)).lower() != TEMPERATURE_MESH_LABEL:
        return False
    if str(row.get("NM_type", nm_type)) != str(nm_type):
        return False
    if str(row.get("analytic_closure", ANALYTIC_CLOSURE_LABEL)) != ANALYTIC_CLOSURE_LABEL:
        return False
    if not math.isclose(_as_finite_float(row.get("B14_calibrated_MeV")), float(B14), rel_tol=1.0e-10, abs_tol=1.0e-8):
        return False
    if not math.isclose(_as_finite_float(row.get("xi")), float(xi), rel_tol=0.0, abs_tol=1.0e-12):
        return False
    if not math.isclose(_as_finite_float(row.get("ms_MeV")), float(ms), rel_tol=0.0, abs_tol=1.0e-12):
        return False
    return True


def load_matching_records(path: Path, B14: float, xi: float, ms: float, nm_type: str, force: bool) -> list[dict[str, Any]]:
    if force or not path.exists():
        return []
    payload = np.load(path, allow_pickle=True).item()
    records = list(payload.get("records", []))
    return [dict(row) for row in records if _row_matches(dict(row), B14, xi, ms, nm_type)]


def latest_rows(records: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, Any]]:
    latest: dict[tuple[float, float], dict[str, Any]] = {}
    for row in records:
        if "T_MeV" in row and "metastability" in row:
            latest[_point_key(row["T_MeV"], row["metastability"])] = row
    return latest


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
    aQstar_LTE_grid: np.ndarray,
    A_extreme_endothermic_grid: np.ndarray,
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
    aQstar_max_mode: str,
) -> None:
    payload = {
        "temperature_mesh": TEMPERATURE_MESH_LABEL,
        "temperature_axis_scale": TEMPERATURE_MESH_LABEL,
        "temperature_values": np.asarray(temperature_values, dtype=float),
        "metastability_values": np.asarray(boundary_metastability_values, dtype=float),
        "boundary_temperature_values": np.asarray(boundary_temperature_values, dtype=float),
        "boundary_metastability_values": np.asarray(boundary_metastability_values, dtype=float),
        "boundary_quantity_values": np.ones_like(
            np.asarray(boundary_metastability_values, dtype=float)
        ),
        "temperature_grid": np.asarray(temperature_grid, dtype=float),
        "metastability_grid": np.asarray(metastability_grid, dtype=float),
        "velocity_grid_m_s": np.asarray(velocity_grid_m_s, dtype=float),
        "u_N_grid": np.asarray(u_N_grid, dtype=float),
        "aQstar_grid": np.asarray(aQstar_grid, dtype=float),
        "aQstar_used_grid": np.asarray(aQstar_grid, dtype=float),
        "aQstar_LTE_grid": np.asarray(aQstar_LTE_grid, dtype=float),
        "A_extreme_endothermic_grid": np.asarray(A_extreme_endothermic_grid, dtype=float),
        "aQstar_grid_semantics": "aQstar_used by selected analytic velocity mode",
        "boundary_quantity_name": _boundary_quantity_name(aQstar_max_mode),
        "boundary_target": 1.0,
        "boundary_semantics": _boundary_semantics(aQstar_max_mode),
        "records": list(records),
        "boundary_records": list(boundary_records),
        "B14": float(B14),
        "n0": float(n0),
        "nB_crit": float(nB_crit),
        "xi": float(xi),
        "ms_MeV": float(ms),
        "NM_type": str(nm_type),
        "analytic_closure": ANALYTIC_CLOSURE_LABEL,
        "solver_variant": ANALYTIC_CLOSURE_LABEL,
        "composition_definition": "a_local_equals_nK_over_nB",
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "root_method": "analytic_velocity_bound",
        "aQstar_max_mode": str(aQstar_max_mode),
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
    parser.add_argument("--temperature-count", type=int, default=DEFAULT_TEMPERATURE_COUNT, help="Number of adaptive temperature points per metastability column.")
    parser.add_argument("--metastability-min", type=float, default=DEFAULT_METASTABILITY_MIN, help="Minimum (nB_N - nB_crit)/n0 value.")
    parser.add_argument("--metastability-max", type=float, default=DEFAULT_METASTABILITY_MAX, help="Maximum (nB_N - nB_crit)/n0 value.")
    parser.add_argument("--metastability-count", type=int, default=DEFAULT_METASTABILITY_COUNT, help="Number of metastability grid points.")
    parser.add_argument("--ncrit-over-n0", type=float, default=DEFAULT_NCRIT_OVER_N0, help="Critical density at T=0 in units of n0.")
    parser.add_argument("--B14", type=float, default=DEFAULT_B14_MEV, help="Optional B^(1/4) in MeV. If omitted, it is calibrated.")
    parser.add_argument("--xi", type=float, default=DEFAULT_XI, help="Analytic profile parameter; must satisfy -1 < xi < 1.")
    parser.add_argument(
        "--aQstar-max",
        default=ANALYTIC_AQSTAR_MAX_MODE,
        help="Analytic abundance cutoff mode: LTE (default) or extreme_endothermic.",
    )
    parser.add_argument("--ms", type=float, default=DEFAULT_MS_MEV, help="Strange-quark mass in MeV.")
    parser.add_argument("--nm-type", default=DEFAULT_NM_TYPE, help="Nuclear matter type passed to phase_velocity.")
    parser.add_argument("--upB", type=int, default=DEFAULT_UPB, help="Upper bound parameter passed to nB_QM/EOS helpers.")
    parser.add_argument(
        "--boundary-probe-count",
        type=int,
        default=DEFAULT_BOUNDARY_PROBE_COUNT,
        help=(
            "Sparse temperature samples used to bracket aQstar_LTE=1 in LTE "
            "mode or A_extreme_endothermic=1 in legacy mode."
        ),
    )
    parser.add_argument("--boundary-bisect-steps", type=int, default=DEFAULT_BOUNDARY_BISECT_STEPS, help="Bisection refinements after the endpoint boundary is bracketed.")
    parser.add_argument("--boundary-eps", type=float, default=DEFAULT_BOUNDARY_EPS_MEV, help="Temperature offset in MeV below the boundary for the adaptive velocity mesh.")
    parser.add_argument("--force", action="store_true", help="Compatibility flag; adaptive runs recompute and overwrite the selected output.")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Save partial velocity results after this many completed temperature rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_velocity_source = _validate_phase_velocity_runtime()
    if args.temperature_count <= 0:
        raise ValueError("--temperature-count must be positive")
    if args.metastability_count <= 0:
        raise ValueError("--metastability-count must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not (-1.0 < float(args.xi) < 1.0):
        raise ValueError("--xi must satisfy -1 < xi < 1")
    aQstar_max_mode = _normalize_aqstar_max_mode(args.aQstar_max)

    output_name = args.output
    if output_name is None:
        output_name = DEFAULT_OUTPUT_TEMPLATE.format(
            mode=str(aQstar_max_mode),
            run_tag=str(args.run_tag),
        )
    output_path = _resolve_output_path(output_name)
    n0 = float(0.16 * const.MeV_fm**3)
    nB_crit = float(args.ncrit_over_n0 * n0)

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
    aQstar_LTE_grid = np.full_like(metastability_grid, np.nan, dtype=float)
    A_extreme_endothermic_grid = np.full_like(metastability_grid, np.nan, dtype=float)
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
            str(aQstar_max_mode),
        )
        for metastability in metastability_values
    ]

    print(f"Output: {output_path}", flush=True)
    print(f"RMFsolver.phase_velocity: {phase_velocity_source}", flush=True)
    print(f"Run tag: {args.run_tag}", flush=True)
    print(f"Analytic aQstar_max mode: {aQstar_max_mode}", flush=True)
    print(f"Adaptive boundary: {_boundary_semantics(aQstar_max_mode)}", flush=True)
    print(f"Adaptive grid: {len(temperature_values)} local temperatures x {len(metastability_values)} metastability values", flush=True)
    print(f"Boundary probe count: {int(args.boundary_probe_count)}, bisection steps: {int(args.boundary_bisect_steps)}", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    if int(args.workers) == 1:
        progress = tqdm(
            boundary_tasks,
            total=len(boundary_tasks),
            desc="Hydro boundary",
            unit="col",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for task in progress:
            boundary_row = solve_boundary_for_metastability(task)
            progress.set_postfix_str(f"m={float(boundary_row['metastability']):.3g}", refresh=False)
            boundary_records.append(boundary_row)
            idx = int(np.argmin(np.abs(metastability_values - float(boundary_row["metastability"]))))
            boundary_temperature_values[idx] = float(boundary_row["temperature_cut_MeV"])
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = [pool.submit(solve_boundary_for_metastability, task) for task in boundary_tasks]
            progress = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Hydro boundary",
                unit="col",
                dynamic_ncols=True,
                mininterval=0.5,
            )
            for future in progress:
                boundary_row = future.result()
                progress.set_postfix_str(f"m={float(boundary_row['metastability']):.3g}", refresh=False)
                boundary_records.append(boundary_row)
                idx = int(np.argmin(np.abs(metastability_values - float(boundary_row["metastability"]))))
                boundary_temperature_values[idx] = float(boundary_row["temperature_cut_MeV"])

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
        aQstar_LTE_grid,
        A_extreme_endothermic_grid,
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
        str(aQstar_max_mode),
    )
    _validate_boundary_pass(boundary_records, len(metastability_values))

    velocity_tasks = [
        (
            int(j),
            float(metastability),
            float(boundary_temperature_values[j]),
            float(n0),
            float(nB_crit),
            float(B14),
            float(args.xi),
            float(args.ms),
            str(args.nm_type),
            int(args.upB),
            float(args.temperature_min),
            float(args.boundary_eps),
            int(args.temperature_count),
            str(aQstar_max_mode),
        )
        for j, metastability in enumerate(metastability_values)
    ]

    completed_since_save = 0
    if int(args.workers) == 1:
        progress = tqdm(
            total=len(velocity_tasks) * int(args.temperature_count),
            desc="Hydro analytic contour",
            unit="pt",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for task in velocity_tasks:
            row_result = calculate_velocity_column(task)
            if row_result.get("status") != "ok":
                raise RuntimeError(
                    "Velocity column has no safe temperature interval: "
                    f"metastability={row_result['metastability']}, "
                    f"status={row_result.get('status')}"
                )
            finite_T = np.asarray(row_result["temperature_values"], dtype=float)
            finite_T = finite_T[np.isfinite(finite_T)]
            if finite_T.size:
                progress.set_postfix_str(f"T={float(finite_T[-1]):.3g} MeV", refresh=False)
            j = int(row_result["col_index"])
            temperature_grid[:, j] = row_result["temperature_values"]
            velocity_grid_m_s[:, j] = row_result["velocity_values_m_s"]
            u_N_grid[:, j] = row_result["u_N_values"]
            aQstar_grid[:, j] = row_result["aQstar_used_values"]
            aQstar_LTE_grid[:, j] = row_result["aQstar_LTE_values"]
            A_extreme_endothermic_grid[:, j] = row_result["A_extreme_endothermic_values"]
            records.extend(row_result["records"])
            progress.update(len(row_result["records"]))
            completed_since_save += 1
            if args.save_every > 0 and completed_since_save >= int(args.save_every):
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
                    aQstar_LTE_grid,
                    A_extreme_endothermic_grid,
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
                    str(aQstar_max_mode),
                )
                completed_since_save = 0
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = [pool.submit(calculate_velocity_column, task) for task in velocity_tasks]
            progress = tqdm(
                total=len(futures) * int(args.temperature_count),
                desc="Hydro analytic contour",
                unit="pt",
                dynamic_ncols=True,
                mininterval=0.5,
            )
            for future in as_completed(futures):
                row_result = future.result()
                if row_result.get("status") != "ok":
                    raise RuntimeError(
                        "Velocity column has no safe temperature interval: "
                        f"metastability={row_result['metastability']}, "
                        f"status={row_result.get('status')}"
                    )
                finite_T = np.asarray(row_result["temperature_values"], dtype=float)
                finite_T = finite_T[np.isfinite(finite_T)]
                if finite_T.size:
                    progress.set_postfix_str(f"T={float(finite_T[-1]):.3g} MeV", refresh=False)
                j = int(row_result["col_index"])
                temperature_grid[:, j] = row_result["temperature_values"]
                velocity_grid_m_s[:, j] = row_result["velocity_values_m_s"]
                u_N_grid[:, j] = row_result["u_N_values"]
                aQstar_grid[:, j] = row_result["aQstar_used_values"]
                aQstar_LTE_grid[:, j] = row_result["aQstar_LTE_values"]
                A_extreme_endothermic_grid[:, j] = row_result["A_extreme_endothermic_values"]
                records.extend(row_result["records"])
                progress.update(len(row_result["records"]))
                completed_since_save += 1
                if args.save_every > 0 and completed_since_save >= int(args.save_every):
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
                        aQstar_LTE_grid,
                        A_extreme_endothermic_grid,
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
                        str(aQstar_max_mode),
                    )
                    completed_since_save = 0

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
        aQstar_LTE_grid,
        A_extreme_endothermic_grid,
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
        str(aQstar_max_mode),
    )

    successes = sum(1 for row in records if bool(row.get("success", False)))
    endpoint_failures = sum(1 for row in records if bool(row.get("endpoint_domain_failure", False)))
    other_failures = len(records) - successes - endpoint_failures
    print("Done.", flush=True)
    print(f"Saved rows: {len(records)}", flush=True)
    print(f"Saved boundary rows: {len(boundary_records)}", flush=True)
    print(f"success={successes}, endpoint_domain_failure={endpoint_failures}, other_failure={other_failures}", flush=True)
    print(f"finite boundary points={int(np.isfinite(boundary_temperature_values).sum())}", flush=True)
    print(f"B14={B14:.10g} MeV, muB_crit0={muB_crit0:.10g}, P_crit0={P_crit0:.10g}", flush=True)


if __name__ == "__main__":
    main()
