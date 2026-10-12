#!/usr/bin/env python3
"""Shared engine for the 26-07-18 hydro-analytic cluster contours."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"matplotlib-cache-{os.getuid()}"),
)
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


METASTABILITY_DEFINITION = "(nB_N - nB_crit(T))/n0"
DENSITY_RATIO_DEFINITION = "lambda_n_equals_nB_N_over_nB_Q"
COMPOSITION_DEFINITION = "a_local_equals_nK_over_nB"
ANALYTIC_FORMULA_VARIANT = "piecewise_constant_lambda_n"
ANALYTIC_CLOSURE_LABEL = "hydro_jump_eigenvalue"
BOUNDARY_CLOSURE_LABEL = "hydro_jump_eigenvalue_A_boundary_1"


@dataclass(frozen=True)
class RunConfig:
    output_template: str
    run_tag: str
    temperature_mesh: str
    temperature_axis_scale: str
    temperature_min: float = 1.0e-2
    temperature_max: float = 100.0
    temperature_count: int = 30
    metastability_min: float = 0.01
    metastability_max: float = 0.9
    metastability_count: int = 20
    ncrit_over_n0: float = 3.0
    B14: float = np.nan
    xi: float = 0.0
    ms: float = 0.0
    nm_type: str = "PNM"
    upB: int = 5000
    interface_fraction_mode: str = "LTE"
    boundary_probe_count: int = 8
    boundary_bisect_steps: int = 10
    boundary_eps: float = 0.005
    save_every: int = 10
    seed_temperature: float = 10.0
    neighbor_retry: bool = True


@dataclass(frozen=True)
class AnalyticPointTask:
    T_N: float
    metastability: float
    n0: float
    nB_crit0: float
    B14: float
    xi: float
    ms: float
    nm_type: str
    upB: int
    interface_fraction_mode: str
    endpoint_initial_guess: tuple[float, float] | None = None


@dataclass(frozen=True)
class BoundaryTask:
    metastability: float
    n0: float
    nB_crit0: float
    B14: float
    xi: float
    ms: float
    nm_type: str
    upB: int
    interface_fraction_mode: str
    temperature_min: float
    temperature_max: float
    probe_count: int
    bisect_steps: int


@dataclass(frozen=True)
class ColumnTask:
    col_index: int
    metastability: float
    temperature_values: np.ndarray
    n0: float
    nB_crit0: float
    B14: float
    xi: float
    ms: float
    nm_type: str
    upB: int
    interface_fraction_mode: str
    temperature_cut: float
    boundary_eps: float
    seed_temperature: float
    default_endpoint_guess: tuple[float, float] | None


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


def _as_finite_float(value: Any, default: float = np.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _finite_pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        first, second = value
        pair = (float(first), float(second))
    except Exception:
        return None
    if not np.all(np.isfinite(pair)) or pair[0] <= 0.0 or pair[1] <= 0.0:
        return None
    return pair


def _parse_float_pair(value: str | None) -> tuple[float, float] | None:
    if value is None or not str(value).strip():
        return None
    fields = [item.strip() for item in str(value).split(",")]
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("endpoint guess must be MU_B_Q,T_Q")
    pair = _finite_pair((float(fields[0]), float(fields[1])))
    if pair is None:
        raise argparse.ArgumentTypeError("endpoint guess values must be positive and finite")
    return pair


def _normalize_interface_fraction_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode == "lte":
        return "LTE"
    if mode == "extreme_endothermic":
        return "extreme_endothermic"
    raise ValueError("interface fraction mode must be 'LTE' or 'extreme_endothermic'")


def _resolve_output_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else SCRIPT_DIR / path


def _calibration_root(func, x_grid: np.ndarray, prefer: float) -> float:
    x_grid = np.asarray(x_grid, dtype=float)
    values = np.full_like(x_grid, np.nan)
    for i, value in enumerate(x_grid):
        try:
            values[i] = float(func(float(value)))
        except Exception:
            pass
    finite = np.isfinite(values)
    x_grid = x_grid[finite]
    values = values[finite]
    if x_grid.size < 2:
        raise RuntimeError("Not enough finite samples to bracket the calibration root")
    exact = np.where(np.isclose(values, 0.0, atol=1.0e-8, rtol=0.0))[0]
    if exact.size:
        index = exact[np.argmin(np.abs(x_grid[exact] - float(prefer)))]
        return float(x_grid[index])
    changes = np.where(values[:-1] * values[1:] < 0.0)[0]
    if not changes.size:
        raise RuntimeError("No sign change found for the calibration root")
    midpoints = 0.5 * (x_grid[changes] + x_grid[changes + 1])
    index = int(changes[np.argmin(np.abs(midpoints - float(prefer)))])
    return float(
        brentq(
            func,
            float(x_grid[index]),
            float(x_grid[index + 1]),
            xtol=1.0e-6,
            rtol=1.0e-8,
            maxiter=200,
        )
    )


def _coexistence_root(func, prefer: float = 1150.0) -> float:
    cached: dict[float, float] = {}

    def evaluate(value: float) -> float:
        value = float(value)
        if value not in cached:
            try:
                cached[value] = float(func(value))
            except Exception:
                cached[value] = np.nan
        return cached[value]

    for width in (25.0, 50.0, 100.0, 200.0, 350.0, 550.0):
        lower = max(1.0, float(prefer) - width)
        upper = float(prefer) + width
        f_lower = evaluate(lower)
        f_upper = evaluate(upper)
        if not (np.isfinite(f_lower) and np.isfinite(f_upper)):
            continue
        if np.isclose(f_lower, 0.0, atol=1.0e-8, rtol=0.0):
            return lower
        if np.isclose(f_upper, 0.0, atol=1.0e-8, rtol=0.0):
            return upper
        if f_lower * f_upper < 0.0:
            return float(brentq(func, lower, upper, xtol=1.0e-6, rtol=1.0e-8, maxiter=200))
    return _calibration_root(func, np.linspace(500.0, 1800.0, 131), prefer=prefer)


def calibrate_b14(nB_crit0: float, ms: float, nm_type: str) -> tuple[float, float, float]:
    param = para.paraQMCRMF3
    P_crit0 = float(
        phase_velocity.PNM_n(nB_crit0, 0.0, param=param, NM_type=nm_type)
    )
    muB_crit0 = _calibration_root(
        lambda mu: phase_velocity.PNM(mu, 0.0, param=param, NM_type=nm_type) - P_crit0,
        np.linspace(900.0, 1500.0, 220),
        prefer=1050.0,
    )
    B14 = _calibration_root(
        lambda bag: phase_velocity.PQM(muB_crit0, 0.0, bag, 0.0, ms=ms) - P_crit0,
        np.linspace(120.0, 260.0, 281),
        prefer=180.0,
    )
    return float(B14), float(muB_crit0), float(P_crit0)


@lru_cache(maxsize=1024)
def critical_state_for_temperature(
    T: float,
    B14: float,
    ms: float,
    nm_type: str,
    upB: int = 5000,
) -> tuple[float, float, float]:
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
    P_crit = float(phase_velocity.PQM(muB_crit, 0.0, B14, T, ms=ms, upB=upB))
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
    upB: int = 5000,
) -> tuple[float, float, float, float]:
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


def muB_from_nB(
    nB_target: float,
    T: float,
    nm_type: str,
    prefer: float = 1100.0,
) -> float:
    nB_target = float(nB_target)
    T = float(T)

    def residual(muB: float) -> float:
        return float(
            phase_velocity.nB_NM(
                float(muB),
                T,
                param=para.paraQMCRMF3,
                NM_type=str(nm_type),
            )
            - nB_target
        )

    for width in (20.0, 50.0, 100.0, 200.0, 400.0, 700.0):
        lower = max(1.0, float(prefer) - width)
        upper = float(prefer) + width
        try:
            f_lower = residual(lower)
            f_upper = residual(upper)
        except Exception:
            continue
        if not (np.isfinite(f_lower) and np.isfinite(f_upper)):
            continue
        tolerance = 1.0e-8 * max(abs(nB_target), 1.0)
        if abs(f_lower) <= tolerance:
            return lower
        if abs(f_upper) <= tolerance:
            return upper
        if f_lower * f_upper < 0.0:
            return float(brentq(residual, lower, upper, xtol=1.0e-7, rtol=1.0e-10, maxiter=100))
    return _calibration_root(residual, np.linspace(850.0, 1700.0, 100), prefer=prefer)


def _base_row(task: AnalyticPointTask) -> dict[str, Any]:
    return {
        "T_MeV": float(task.T_N),
        "metastability": float(task.metastability),
        "metastability_definition": METASTABILITY_DEFINITION,
        "nB_N_over_n0": np.nan,
        "nB_N_MeV3": np.nan,
        "nB_crit_T_over_n0": np.nan,
        "nB_crit_T_MeV3": np.nan,
        "muB_crit_T_MeV": np.nan,
        "P_crit_T_MeV4": np.nan,
        "B14_calibrated_MeV": float(task.B14),
        "xi": float(task.xi),
        "ms_MeV": float(task.ms),
        "NM_type": str(task.nm_type),
        "interface_fraction_mode": str(task.interface_fraction_mode),
        "analytic_closure": ANALYTIC_CLOSURE_LABEL,
        "composition_definition": COMPOSITION_DEFINITION,
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "analytic_formula_variant": ANALYTIC_FORMULA_VARIANT,
        "endpoint_guess_used": _finite_pair(task.endpoint_initial_guess),
        "endpoint_solution": None,
        "endpoint_domain_failure": False,
    }


def _endpoint_domain_failure(message: str) -> bool:
    message = str(message)
    return (
        "A_boundary must satisfy 0 < A_boundary < 1" in message
        or "a_interface_LTE must satisfy 0 < a_interface_LTE < 1" in message
    )


def failure_values(message: str, mode: str) -> dict[str, Any]:
    return {
        "success": False,
        "message": str(message),
        "u_N": np.nan,
        "u_N_m_s": np.nan,
        "u_N_squared": np.nan,
        "u_N_formula_squared": np.nan,
        "analytic_velocity_residual": np.nan,
        "jB": np.nan,
        "muB_N_MeV": np.nan,
        "muB_Q_MeV": np.nan,
        "T_Q_MeV": np.nan,
        "lambda_n": np.nan,
        "A_boundary": np.nan,
        "a_interface": np.nan,
        "a_interface_LTE": np.nan,
        "interface_fraction_mode": str(mode),
        "endpoint_solution": None,
        "endpoint_domain_failure": _endpoint_domain_failure(message),
        "composition_definition": COMPOSITION_DEFINITION,
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "analytic_formula_variant": ANALYTIC_FORMULA_VARIANT,
    }


def calculate_point(task: AnalyticPointTask) -> dict[str, Any]:
    row = _base_row(task)
    try:
        nB_N, nB_crit_T, muB_crit_T, P_crit_T = upstream_density_from_metastability(
            task.T_N,
            task.metastability,
            task.n0,
            task.B14,
            task.ms,
            task.nm_type,
            task.upB,
        )
        row.update(
            {
                "nB_N_over_n0": float(nB_N / task.n0),
                "nB_N_MeV3": float(nB_N),
                "nB_crit_T_over_n0": float(nB_crit_T / task.n0),
                "nB_crit_T_MeV3": float(nB_crit_T),
                "muB_crit_T_MeV": float(muB_crit_T),
                "P_crit_T_MeV4": float(P_crit_T),
            }
        )
        muB_N = muB_from_nB(nB_N, task.T_N, task.nm_type, prefer=1100.0)
        result = phase_velocity.analytic_velocity_bound(
            muB_N,
            float(task.T_N),
            float(task.B14),
            xi=float(task.xi),
            ms=float(task.ms),
            param=para.paraQMCRMF3,
            NM_type=str(task.nm_type),
            upB=int(task.upB),
            initial_guess=_finite_pair(task.endpoint_initial_guess),
            interface_fraction_mode=_normalize_interface_fraction_mode(
                task.interface_fraction_mode
            ),
        )
        row.update(
            {
                "success": True,
                "message": str(result.get("message", "")),
                "muB_N_MeV": float(muB_N),
                "u_N": float(result["u_N_max"]),
                "u_N_m_s": float(result["u_N_max"] * const.m_per_sec),
                "u_N_squared": float(result["u_N_squared"]),
                "u_N_formula_squared": float(result["u_N_formula_squared"]),
                "analytic_velocity_residual": float(result["analytic_velocity_residual"]),
                "jB": float(result["jB"]),
                "u_Q": _as_finite_float(result.get("u_Q")),
                "gamma_N": _as_finite_float(result.get("gamma_N")),
                "gamma_Q": _as_finite_float(result.get("gamma_Q")),
                "muB_Q_MeV": float(result["muB_Q"]),
                "T_Q_MeV": float(result["T_Q"]),
                "P_N_MeV4": _as_finite_float(result.get("P_N")),
                "P_Q_MeV4": _as_finite_float(result.get("P_Q")),
                "pressure_jump_MeV4": _as_finite_float(result.get("pressure_jump")),
                "pressure_jump_balance_MeV4": _as_finite_float(result.get("pressure_jump_balance")),
                "pressure_jump_residual_MeV4": _as_finite_float(result.get("pressure_jump_residual")),
                "energy_flux_residual": _as_finite_float(result.get("energy_flux_residual")),
                "momentum_flux_residual": _as_finite_float(result.get("momentum_flux_residual")),
                "endpoint_scaled_residual": _as_finite_float(result.get("endpoint_scaled_residual")),
                "mu_q_MeV": _as_finite_float(result.get("mu_q")),
                "nB_N_MeV3_check": _as_finite_float(result.get("nB_N")),
                "nB_Q_MeV3": _as_finite_float(result.get("nB_Q")),
                "h_N_MeV4": _as_finite_float(result.get("h_N")),
                "h_Q_MeV4": _as_finite_float(result.get("h_Q")),
                "h_over_nB_N_MeV": _as_finite_float(result.get("h_over_nB_N")),
                "h_over_nB_Q_MeV": _as_finite_float(result.get("h_over_nB_Q")),
                "lambda_n": float(result["lambda_n"]),
                "A_boundary": float(result["A_boundary"]),
                "a_interface": float(result["a_interface"]),
                "a_interface_LTE": float(result["a_interface_LTE"]),
                "interface_fraction_mode": str(result["interface_fraction_mode"]),
                "beta_LTE": _as_finite_float(result.get("beta_LTE")),
                "lte_correction": _as_finite_float(result.get("lte_correction")),
                "muKstar_max_MeV": _as_finite_float(result.get("muKstar_max")),
                "gamma_MeV": _as_finite_float(result.get("gamma")),
                "tau_MeV_inv": _as_finite_float(result.get("tau")),
                "tau_seconds": _as_finite_float(result.get("tau_seconds")),
                "slow_front_consistent": bool(result.get("slow_front_consistent", False)),
                "composition_definition": str(result.get("composition_definition", COMPOSITION_DEFINITION)),
                "density_ratio_definition": str(result.get("density_ratio_definition", DENSITY_RATIO_DEFINITION)),
                "analytic_formula_variant": str(result.get("analytic_formula_variant", ANALYTIC_FORMULA_VARIANT)),
                "endpoint_solution": (float(result["muB_Q"]), float(result["T_Q"])),
                "endpoint_domain_failure": False,
            }
        )
    except Exception as exc:
        row.update(failure_values(str(exc), task.interface_fraction_mode))
    return row


def build_adaptive_temperature_column(
    *,
    temperature_mesh: str,
    temperature_min: float,
    temperature_max: float,
    temperature_cut: float,
    boundary_eps: float,
    temperature_count: int,
) -> np.ndarray:
    temperature_min = float(temperature_min)
    temperature_max = float(temperature_max)
    upper = min(float(temperature_max), float(temperature_cut) - float(boundary_eps))
    count = int(temperature_count)
    if temperature_min <= 0.0 or not np.isfinite(temperature_min):
        raise ValueError("temperature_min must be positive and finite")
    if not np.isfinite(upper) or upper <= temperature_min:
        raise ValueError("adaptive temperature column has no valid interval")
    if count < 2:
        raise ValueError("temperature_count must be at least 2")
    if temperature_mesh == "adaptive_log":
        return np.geomspace(temperature_min, upper, count)
    if temperature_mesh == "adaptive_linear":
        return np.linspace(temperature_min, upper, count)
    raise ValueError("temperature_mesh must be adaptive_log or adaptive_linear")


def choose_seed_row_index(values: np.ndarray, seed_temperature: float) -> int:
    values = np.asarray(values, dtype=float)
    finite = np.where(np.isfinite(values))[0]
    if not finite.size:
        raise ValueError("temperature column has no finite values")
    return int(finite[np.argmin(np.abs(values[finite] - float(seed_temperature)))])


def column_solve_order(n_rows: int, seed_row_index: int) -> list[tuple[int, str]]:
    seed = int(np.clip(int(seed_row_index), 0, int(n_rows) - 1))
    order = [(seed, "seed")]
    order.extend((i, "up") for i in range(seed + 1, int(n_rows)))
    order.extend((i, "down") for i in range(seed - 1, -1, -1))
    return order


def _successful_endpoint(row: dict[str, Any] | None) -> tuple[float, float] | None:
    if not row or not bool(row.get("success", False)):
        return None
    return _finite_pair(row.get("endpoint_solution"))


def neighbor_endpoint_guess(
    row_index: int,
    direction: str,
    solved_by_row: dict[int, dict[str, Any]],
    default: tuple[float, float] | None = None,
) -> tuple[tuple[float, float] | None, str]:
    if direction == "seed":
        return _finite_pair(default), "default"
    neighbor_index = int(row_index) - 1 if direction == "up" else int(row_index) + 1
    endpoint = _successful_endpoint(solved_by_row.get(neighbor_index))
    if endpoint is not None:
        source = "same_column_lower_neighbor" if direction == "up" else "same_column_upper_neighbor"
        return endpoint, source
    return _finite_pair(default), "fallback_after_failed_neighbor"


def _calculate_boundary_point(task: BoundaryTask, temperature: float) -> dict[str, Any]:
    row = calculate_point(
        AnalyticPointTask(
            T_N=float(temperature),
            metastability=task.metastability,
            n0=task.n0,
            nB_crit0=task.nB_crit0,
            B14=task.B14,
            xi=task.xi,
            ms=task.ms,
            nm_type=task.nm_type,
            upB=task.upB,
            interface_fraction_mode=task.interface_fraction_mode,
        )
    )
    row["analytic_closure"] = BOUNDARY_CLOSURE_LABEL
    return row


def solve_boundary_for_metastability(task: BoundaryTask) -> dict[str, Any]:
    probes = np.linspace(task.temperature_min, task.temperature_max, max(2, task.probe_count))
    evaluations: list[dict[str, Any]] = []
    for temperature in probes:
        row = _calculate_boundary_point(task, float(temperature))
        row["boundary_stage"] = "probe"
        evaluations.append(row)

    success_indices = [i for i, row in enumerate(evaluations) if row.get("success")]
    endpoint_indices = [i for i, row in enumerate(evaluations) if row.get("endpoint_domain_failure")]
    nonendpoint_failure_indices = [
        i
        for i, row in enumerate(evaluations)
        if not row.get("success") and not row.get("endpoint_domain_failure")
    ]
    status = "no_endpoint_found"
    lower = np.nan
    upper = np.nan
    cut = np.nan

    if nonendpoint_failure_indices:
        status = "nonendpoint_probe_failure"
    elif not success_indices:
        status = "no_valid_probe"
    elif not endpoint_indices:
        status = "all_probes_valid"
        lower = float(probes[success_indices[-1]])
        cut = float(task.temperature_max)
    else:
        transition = next(
            (
                (max(j for j in success_indices if j < i), i)
                for i in endpoint_indices
                if any(j < i for j in success_indices)
            ),
            None,
        )
        if transition is None:
            status = "endpoint_below_first_valid_probe"
            upper = float(probes[endpoint_indices[0]])
        else:
            lower = float(probes[transition[0]])
            upper = float(probes[transition[1]])
            status = "bracketed"
            for _ in range(max(0, int(task.bisect_steps))):
                middle = 0.5 * (lower + upper)
                row = _calculate_boundary_point(task, middle)
                row["boundary_stage"] = "bisect"
                evaluations.append(row)
                if row.get("success"):
                    lower = middle
                elif row.get("endpoint_domain_failure"):
                    upper = middle
                else:
                    status = "nonendpoint_failure_during_bisection"
                    break
            if status == "bracketed":
                cut = lower

    successful = [row for row in evaluations if row.get("success")]
    A_values = [_as_finite_float(row.get("A_boundary")) for row in successful]
    lte_values = [_as_finite_float(row.get("a_interface_LTE")) for row in successful]
    A_values = [value for value in A_values if np.isfinite(value)]
    lte_values = [value for value in lte_values if np.isfinite(value)]
    return {
        "success": bool(status in {"bracketed", "all_probes_valid"} and np.isfinite(cut)),
        "status": status,
        "metastability": float(task.metastability),
        "temperature_cut_MeV": float(cut),
        "temperature_lower_bracket_MeV": float(lower),
        "temperature_upper_bracket_MeV": float(upper),
        "boundary_target": 1.0,
        "boundary_quantity_name": "A_boundary",
        "boundary_semantics": "A_boundary = 1 (equivalent to a_interface_LTE = 1)",
        "interface_fraction_mode": str(task.interface_fraction_mode),
        "A_boundary_min": min(A_values) if A_values else np.nan,
        "A_boundary_max": max(A_values) if A_values else np.nan,
        "a_interface_LTE_min": min(lte_values) if lte_values else np.nan,
        "a_interface_LTE_max": max(lte_values) if lte_values else np.nan,
        "evaluations": evaluations,
    }


def _column_plan(task: ColumnTask) -> dict[str, Any]:
    temperatures = np.asarray(task.temperature_values, dtype=float)
    seed = choose_seed_row_index(temperatures, task.seed_temperature)
    return {
        "task": task,
        "order": column_solve_order(temperatures.size, seed),
        "position": 0,
        "solved": {},
    }


def _next_point_task(
    plan: dict[str, Any],
) -> tuple[
    int,
    int,
    AnalyticPointTask,
    str,
    str,
    float,
    int,
    tuple[int, int] | None,
] | None:
    if plan["position"] >= len(plan["order"]):
        return None
    row_index, direction = plan["order"][plan["position"]]
    column: ColumnTask = plan["task"]
    guess, source = neighbor_endpoint_guess(
        row_index,
        direction,
        plan["solved"],
        column.default_endpoint_guess,
    )
    point_task = AnalyticPointTask(
        T_N=float(column.temperature_values[row_index]),
        metastability=column.metastability,
        n0=column.n0,
        nB_crit0=column.nB_crit0,
        B14=column.B14,
        xi=column.xi,
        ms=column.ms,
        nm_type=column.nm_type,
        upB=column.upB,
        interface_fraction_mode=column.interface_fraction_mode,
        endpoint_initial_guess=guess,
    )
    return (
        row_index,
        column.col_index,
        point_task,
        direction,
        source,
        float(column.seed_temperature),
        0,
        None,
    )


def calculate_velocity_point(
    work: tuple[
        int,
        int,
        AnalyticPointTask,
        str,
        str,
        float,
        int,
        tuple[int, int] | None,
    ],
) -> dict[str, Any]:
    (
        row_index,
        col_index,
        point_task,
        direction,
        source,
        seed_temperature,
        retry_attempt,
        retry_of,
    ) = work
    row = calculate_point(point_task)
    row.update(
        {
            "row_index": int(row_index),
            "col_index": int(col_index),
            "endpoint_guess_source": str(source),
            "continuation_direction": str(direction),
            "continuation_seed_temperature_MeV": float(seed_temperature),
            "retry_attempt": int(retry_attempt),
            "retry_of": retry_of,
        }
    )
    return {
        "row_index": int(row_index),
        "col_index": int(col_index),
        "T_MeV": float(point_task.T_N),
        "metastability": float(point_task.metastability),
        "row": row,
    }


def _remember_point(plan: dict[str, Any], result: dict[str, Any]) -> None:
    plan["solved"][int(result["row_index"])] = result["row"]
    plan["position"] += 1


def _iter_points_serial(column_tasks: Iterable[ColumnTask]):
    for column_task in column_tasks:
        plan = _column_plan(column_task)
        while (work := _next_point_task(plan)) is not None:
            result = calculate_velocity_point(work)
            _remember_point(plan, result)
            yield result


def _iter_points_parallel(column_tasks: list[ColumnTask], workers: int):
    plans = [_column_plan(task) for task in column_tasks]
    pending: dict[Any, dict[str, Any]] = {}

    def submit_next(pool, plan):
        work = _next_point_task(plan)
        if work is not None:
            pending[pool.submit(calculate_velocity_point, work)] = plan

    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), len(plans)))) as pool:
        for plan in plans:
            submit_next(pool, plan)
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                plan = pending.pop(future)
                result = future.result()
                _remember_point(plan, result)
                yield result
                submit_next(pool, plan)


def _immediate_retry_endpoint(
    row_index: int,
    col_index: int,
    successful: dict[tuple[int, int], dict[str, Any]],
) -> tuple[tuple[float, float] | None, str]:
    candidates = (
        ((row_index - 1, col_index), "retry_same_column_lower_neighbor"),
        ((row_index + 1, col_index), "retry_same_column_upper_neighbor"),
        ((row_index, col_index - 1), "retry_same_row_left_neighbor"),
        ((row_index, col_index + 1), "retry_same_row_right_neighbor"),
    )
    for key, source in candidates:
        endpoint = _successful_endpoint(successful.get(key))
        if endpoint is not None:
            return endpoint, source
    return None, ""


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, payload, allow_pickle=True)
    temporary.replace(path)


def save_payload(
    *,
    path: Path,
    config: RunConfig,
    args: argparse.Namespace,
    B14: float,
    muB_crit0: float,
    P_crit0: float,
    n0: float,
    nB_crit0: float,
    temperature_grid: np.ndarray,
    metastability_grid: np.ndarray,
    velocity_grid_m_s: np.ndarray,
    u_N_grid: np.ndarray,
    lambda_n_grid: np.ndarray,
    A_boundary_grid: np.ndarray,
    a_interface_grid: np.ndarray,
    a_interface_LTE_grid: np.ndarray,
    nB_crit_T_grid: np.ndarray,
    muB_crit_T_grid: np.ndarray,
    boundary_temperature_values: np.ndarray,
    boundary_metastability_values: np.ndarray,
    records: list[dict[str, Any]],
    boundary_records: list[dict[str, Any]],
) -> None:
    payload = {
        "run_tag": str(args.run_tag),
        "temperature_mesh": config.temperature_mesh,
        "temperature_axis_scale": config.temperature_axis_scale,
        "analytic_closure": ANALYTIC_CLOSURE_LABEL,
        "boundary_closure": BOUNDARY_CLOSURE_LABEL,
        "interface_fraction_mode": str(args.interface_fraction_mode),
        "metastability_definition": METASTABILITY_DEFINITION,
        "composition_definition": COMPOSITION_DEFINITION,
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "analytic_formula_variant": ANALYTIC_FORMULA_VARIANT,
        "B14_calibrated_MeV": float(B14),
        "muB_crit0_MeV": float(muB_crit0),
        "P_crit0_MeV4": float(P_crit0),
        "n0_MeV3": float(n0),
        "nB_crit0_MeV3": float(nB_crit0),
        "nB_crit0_over_n0": float(nB_crit0 / n0),
        "temperature_grid": np.asarray(temperature_grid),
        "metastability_grid": np.asarray(metastability_grid),
        "velocity_grid_m_s": np.asarray(velocity_grid_m_s),
        "u_N_grid": np.asarray(u_N_grid),
        "lambda_n_grid": np.asarray(lambda_n_grid),
        "A_boundary_grid": np.asarray(A_boundary_grid),
        "a_interface_grid": np.asarray(a_interface_grid),
        "a_interface_LTE_grid": np.asarray(a_interface_LTE_grid),
        "nB_crit_T_grid": np.asarray(nB_crit_T_grid),
        "muB_crit_T_grid": np.asarray(muB_crit_T_grid),
        "boundary_temperature_values": np.asarray(boundary_temperature_values),
        "boundary_metastability_values": np.asarray(boundary_metastability_values),
        "records": list(records),
        "boundary_records": list(boundary_records),
        "run_config": asdict(config),
        "arguments": vars(args).copy(),
    }
    _atomic_save(Path(path), payload)


def parse_args(config: RunConfig, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel hydro-analytic velocity contour with finite-temperature "
            "critical density and an adaptive per-column temperature mesh."
        )
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-tag", default=os.getenv("RUN_TAG", config.run_tag))
    parser.add_argument("--temperature-min", type=float, default=config.temperature_min)
    parser.add_argument("--temperature-max", type=float, default=config.temperature_max)
    parser.add_argument("--temperature-count", type=int, default=config.temperature_count)
    parser.add_argument("--metastability-min", type=float, default=config.metastability_min)
    parser.add_argument("--metastability-max", type=float, default=config.metastability_max)
    parser.add_argument("--metastability-count", type=int, default=config.metastability_count)
    parser.add_argument("--ncrit-over-n0", type=float, default=config.ncrit_over_n0)
    parser.add_argument("--B14", type=float, default=config.B14)
    parser.add_argument("--xi", type=float, default=config.xi)
    parser.add_argument("--ms", type=float, default=config.ms)
    parser.add_argument("--nm-type", default=config.nm_type)
    parser.add_argument("--upB", type=int, default=config.upB)
    parser.add_argument(
        "--interface-fraction-mode",
        choices=("LTE", "extreme_endothermic"),
        default=config.interface_fraction_mode,
    )
    parser.add_argument("--boundary-probe-count", type=int, default=config.boundary_probe_count)
    parser.add_argument("--boundary-bisect-steps", type=int, default=config.boundary_bisect_steps)
    parser.add_argument("--boundary-eps", type=float, default=config.boundary_eps)
    parser.add_argument("--save-every", type=int, default=config.save_every)
    parser.add_argument("--seed-temperature", type=float, default=config.seed_temperature)
    parser.add_argument("--endpoint-guess", type=_parse_float_pair, default=None)
    parser.add_argument(
        "--no-neighbor-retry",
        action="store_false",
        dest="neighbor_retry",
        default=config.neighbor_retry,
    )
    parser.add_argument("--workers", type=int, default=_default_workers())
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.temperature_min <= 0.0 or args.temperature_max <= args.temperature_min:
        raise ValueError("temperature bounds must satisfy 0 < min < max")
    if args.temperature_count < 2 or args.metastability_count < 1:
        raise ValueError("temperature_count >= 2 and metastability_count >= 1 are required")
    if args.metastability_max < args.metastability_min:
        raise ValueError("metastability_max must be at least metastability_min")
    if args.ncrit_over_n0 <= 0.0:
        raise ValueError("ncrit_over_n0 must be positive")
    if args.boundary_probe_count < 2 or args.boundary_bisect_steps < 0:
        raise ValueError("boundary probe/bisection settings are invalid")
    if args.boundary_eps < 0.0 or args.save_every < 0 or args.workers < 1:
        raise ValueError("boundary_eps, save_every, and workers are invalid")
    _normalize_interface_fraction_mode(args.interface_fraction_mode)


def _save_current(
    path: Path,
    config: RunConfig,
    args: argparse.Namespace,
    calibration: tuple[float, float, float, float, float],
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    boundary_records: list[dict[str, Any]],
) -> None:
    B14, muB_crit0, P_crit0, n0, nB_crit0 = calibration
    save_payload(
        path=path,
        config=config,
        args=args,
        B14=B14,
        muB_crit0=muB_crit0,
        P_crit0=P_crit0,
        n0=n0,
        nB_crit0=nB_crit0,
        records=records,
        boundary_records=boundary_records,
        **arrays,
    )


def main(config: RunConfig, argv: list[str] | None = None) -> None:
    args = parse_args(config, argv=argv)
    _validate_args(args)
    output_name = args.output or config.output_template.format(run_tag=args.run_tag)
    output_path = _resolve_output_path(output_name)
    n0 = float(0.16 * const.MeV_fm**3)
    nB_crit0 = float(args.ncrit_over_n0 * n0)
    if np.isfinite(args.B14) and args.B14 > 0.0:
        B14 = float(args.B14)
        muB_crit0 = np.nan
        P_crit0 = np.nan
    else:
        print(
            f"Calibrating B^(1/4) from nB_crit(T=0) = {args.ncrit_over_n0:g} n0 ...",
            flush=True,
        )
        B14, muB_crit0, P_crit0 = calibrate_b14(nB_crit0, args.ms, args.nm_type)
        print(f"Calibrated B^(1/4) = {B14:.10g} MeV", flush=True)
    calibration = (B14, muB_crit0, P_crit0, n0, nB_crit0)

    metastability_values = np.linspace(
        args.metastability_min,
        args.metastability_max,
        args.metastability_count,
    )
    shape = (args.temperature_count, args.metastability_count)
    temperature_grid = np.full(shape, np.nan)
    metastability_grid = np.repeat(metastability_values[None, :], args.temperature_count, axis=0)
    arrays = {
        "temperature_grid": temperature_grid,
        "metastability_grid": metastability_grid,
        "velocity_grid_m_s": np.full(shape, np.nan),
        "u_N_grid": np.full(shape, np.nan),
        "lambda_n_grid": np.full(shape, np.nan),
        "A_boundary_grid": np.full(shape, np.nan),
        "a_interface_grid": np.full(shape, np.nan),
        "a_interface_LTE_grid": np.full(shape, np.nan),
        "nB_crit_T_grid": np.full(shape, np.nan),
        "muB_crit_T_grid": np.full(shape, np.nan),
        "boundary_temperature_values": np.full(args.metastability_count, np.nan),
        "boundary_metastability_values": metastability_values.copy(),
    }
    records: list[dict[str, Any]] = []
    boundary_records: list[dict[str, Any]] = []

    print(f"Output: {output_path}", flush=True)
    print(f"Run tag: {args.run_tag}", flush=True)
    print(
        f"Analytic {args.interface_fraction_mode} grid: {args.temperature_count} "
        f"{config.temperature_axis_scale}-T points x {args.metastability_count} metastabilities",
        flush=True,
    )
    print(f"Workers: {args.workers}", flush=True)

    boundary_tasks = [
        BoundaryTask(
            metastability=float(metastability),
            n0=n0,
            nB_crit0=nB_crit0,
            B14=B14,
            xi=args.xi,
            ms=args.ms,
            nm_type=args.nm_type,
            upB=args.upB,
            interface_fraction_mode=args.interface_fraction_mode,
            temperature_min=args.temperature_min,
            temperature_max=args.temperature_max,
            probe_count=args.boundary_probe_count,
            bisect_steps=args.boundary_bisect_steps,
        )
        for metastability in metastability_values
    ]
    if args.workers == 1:
        boundary_iterator = map(solve_boundary_for_metastability, boundary_tasks)
        for result in tqdm(boundary_iterator, total=len(boundary_tasks), desc="A_boundary=1 boundary", unit="col"):
            boundary_records.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(solve_boundary_for_metastability, task) for task in boundary_tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="A_boundary=1 boundary", unit="col"):
                boundary_records.append(future.result())
    for result in boundary_records:
        index = int(np.argmin(np.abs(metastability_values - result["metastability"])))
        arrays["boundary_temperature_values"][index] = result["temperature_cut_MeV"]
    _save_current(output_path, config, args, calibration, arrays, records, boundary_records)

    column_tasks: list[ColumnTask] = []
    for col_index, metastability in enumerate(metastability_values):
        cut = arrays["boundary_temperature_values"][col_index]
        if not np.isfinite(cut) or cut - args.boundary_eps <= args.temperature_min:
            continue
        temperatures = build_adaptive_temperature_column(
            temperature_mesh=config.temperature_mesh,
            temperature_min=args.temperature_min,
            temperature_max=args.temperature_max,
            temperature_cut=cut,
            boundary_eps=args.boundary_eps,
            temperature_count=args.temperature_count,
        )
        column_tasks.append(
            ColumnTask(
                col_index=col_index,
                metastability=float(metastability),
                temperature_values=temperatures,
                n0=n0,
                nB_crit0=nB_crit0,
                B14=B14,
                xi=args.xi,
                ms=args.ms,
                nm_type=args.nm_type,
                upB=args.upB,
                interface_fraction_mode=args.interface_fraction_mode,
                temperature_cut=float(cut),
                boundary_eps=args.boundary_eps,
                seed_temperature=args.seed_temperature,
                default_endpoint_guess=args.endpoint_guess,
            )
        )

    total = sum(task.temperature_values.size for task in column_tasks)
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    successful: dict[tuple[int, int], dict[str, Any]] = {}
    failures: set[tuple[int, int]] = set()
    since_save = 0

    def accept(result: dict[str, Any], first_pass: bool) -> None:
        nonlocal since_save
        row = result["row"]
        i = result["row_index"]
        j = result["col_index"]
        key = (i, j)
        arrays["temperature_grid"][i, j] = result["T_MeV"]
        records.append(row)
        latest[key] = result
        arrays["nB_crit_T_grid"][i, j] = _as_finite_float(row.get("nB_crit_T_MeV3"))
        arrays["muB_crit_T_grid"][i, j] = _as_finite_float(row.get("muB_crit_T_MeV"))
        if row.get("success"):
            successful[key] = row
            arrays["velocity_grid_m_s"][i, j] = row["u_N_m_s"]
            arrays["u_N_grid"][i, j] = row["u_N"]
            arrays["lambda_n_grid"][i, j] = row["lambda_n"]
            arrays["A_boundary_grid"][i, j] = row["A_boundary"]
            arrays["a_interface_grid"][i, j] = row["a_interface"]
            arrays["a_interface_LTE_grid"][i, j] = row["a_interface_LTE"]
        elif first_pass:
            failures.add(key)
        since_save += 1
        if args.save_every and since_save >= args.save_every:
            _save_current(output_path, config, args, calibration, arrays, records, boundary_records)
            since_save = 0

    progress = tqdm(total=total, desc="Analytic velocity contour", unit="pt", dynamic_ncols=True)
    try:
        iterator = (
            _iter_points_serial(column_tasks)
            if args.workers == 1
            else _iter_points_parallel(column_tasks, args.workers)
        )
        for result in iterator:
            progress.set_postfix_str(
                f"m={result['metastability']:.3g}, T={result['T_MeV']:.3g}",
                refresh=False,
            )
            accept(result, first_pass=True)
            progress.update(1)

        retries = []
        if args.neighbor_retry:
            for i, j in sorted(failures):
                if (i, j) in successful:
                    continue
                endpoint, source = _immediate_retry_endpoint(i, j, successful)
                if endpoint is None:
                    continue
                original = latest[(i, j)]
                task = AnalyticPointTask(
                    T_N=original["T_MeV"],
                    metastability=original["metastability"],
                    n0=n0,
                    nB_crit0=nB_crit0,
                    B14=B14,
                    xi=args.xi,
                    ms=args.ms,
                    nm_type=args.nm_type,
                    upB=args.upB,
                    interface_fraction_mode=args.interface_fraction_mode,
                    endpoint_initial_guess=endpoint,
                )
                retries.append(
                    (
                        i,
                        j,
                        task,
                        "retry",
                        source,
                        float(args.seed_temperature),
                        1,
                        (i, j),
                    )
                )
        if retries:
            progress.total = int(progress.total or 0) + len(retries)
            progress.refresh()
            retry_iterator = map(calculate_velocity_point, retries)
            if args.workers > 1:
                pool = ProcessPoolExecutor(max_workers=args.workers)
                futures = [pool.submit(calculate_velocity_point, task) for task in retries]
                retry_iterator = (future.result() for future in as_completed(futures))
            try:
                for result in retry_iterator:
                    result["row"]["retry_replaced_failure"] = bool(result["row"].get("success"))
                    accept(result, first_pass=False)
                    progress.update(1)
            finally:
                if args.workers > 1:
                    pool.shutdown()
    finally:
        progress.close()

    _save_current(output_path, config, args, calibration, arrays, records, boundary_records)
    successes = sum(bool(row.get("success")) for row in records)
    print("Done.", flush=True)
    print(f"Saved rows: {len(records)}; success={successes}; failure={len(records) - successes}", flush=True)
    print(f"Saved boundary rows: {len(boundary_records)}", flush=True)
