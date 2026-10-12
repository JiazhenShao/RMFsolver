#!/usr/bin/env python3
"""Numerical uNmax scan on an entropy-safe adaptive log-T/log-m mesh.

The scan covers 1e-2 <= T_N <= 50 MeV and 1e-5 <= m <= 1 with
m = (nB_N - nB_crit(T_N)) / n0.  Before any expensive BVP is submitted, the
runner solves the Delta h = 0 and Delta(s/nB) = 0 boundaries at every
temperature.  Each logarithmic metastability row begins on the side satisfying
Delta h <= 0 and Delta(s/nB) >= 0.  Numerical velocities use the one-parameter
energy-conserving nK/jK uNmax solver at fixed TQstar.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
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
for path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver import phase_velocity
from _hydro_analytic_contour_common import (
    AnalyticPointTask,
    calculate_point as calculate_analytic_point,
    calibrate_b14,
    critical_state_for_temperature,
    muB_from_nB,
)


# Tunable run parameters
DEFAULT_OUTPUT = "uNmax-logTm-entropy-safe.npy"
DEFAULT_RUN_TAG = "uNmax_logT_logm_entropy_safe_TQstar0p5_v1"
DEFAULT_T_MIN_MEV = 1.0e-2
DEFAULT_T_MAX_MEV = 50.0
DEFAULT_T_COUNT = 30
DEFAULT_M_MIN = 1.0e-5
DEFAULT_M_MAX = 1.0
DEFAULT_M_COUNT = 20
DEFAULT_GAP_BUFFER_FRACTION = 1.0e-3
DEFAULT_ENTROPY_BUFFER_FRACTION = 1.0e-3
DEFAULT_ENTROPY_PROBE_COUNT = 12
DEFAULT_SEED_TEMPERATURE_MEV = 10.0
DEFAULT_NEIGHBOR_RETRY = True
DEFAULT_SAVE_EVERY = 10
DEFAULT_INTERFACE_FRACTION_MODE = "extreme_endothermic"
DEFAULT_NCRIT_OVER_N0 = 3.0
DEFAULT_XI = 0.0
DEFAULT_MS_MEV = 0.0
DEFAULT_NM_TYPE = "PNM"
DEFAULT_UPB = 5000
DEFAULT_TQSTAR_MEV = 0.5
DEFAULT_TAIL_EPS = 1.0e-8
DEFAULT_N_MESH = 200
DEFAULT_TOL_BVP = 1.0e-4
DEFAULT_MAX_NODES = 10000
DEFAULT_JB_GUESS = np.nan
DEFAULT_TQ_GUESS = np.nan
DEFAULT_JB_BOUNDS = None
DEFAULT_AUTO_JB_BOUNDS = True
DEFAULT_SHOOTING_U_MAX = 1.0e-4
DEFAULT_SHOOTING_JB_LOWER = 1.0e-12
DEFAULT_POINT_TIMEOUT_S = 1800.0

METASTABILITY_DEFINITION = "(nB_N - nB_crit(T)) / n0"
GAP_DEFINITION = "muB_cold(P_N) - h_N/nB_N"
ENTROPY_JUMP_DEFINITION = "sQ_over_nB_Q_minus_sN_over_nB_N"
NUMERICAL_CLOSURE_LABEL = "uNmax_direct_TQstar_nK"
NUMERICAL_ROOT_METHOD = "solve_bvp_uNmax_nK_parameter_1d"
DENSITY_RATIO_DEFINITION = "lambda_n_equals_nB_N_over_nB_Q"
CURRENT_DEFINITION = "u_nK_minus_D_dnK_dx"
RATE_MODEL = "exact_nonleptonic"

STATUS_FINITE = 0
STATUS_NUMERICAL_FAILURE = 3
STATUS_UNSCANNED = 5
STATUS_TIMEOUT = 6

GAP_BOUNDARY_ROOT = 0
GAP_BOUNDARY_ALL_ALLOWED = 1
GAP_BOUNDARY_ALL_FORBIDDEN = 2
GAP_BOUNDARY_FAILURE = 3

ENTROPY_BOUNDARY_ROOT = "root"
ENTROPY_BOUNDARY_ALL_ALLOWED = "all_positive"
ENTROPY_BOUNDARY_ALL_FORBIDDEN = "all_nonpositive"
ENTROPY_BOUNDARY_MULTIPLE_ROOTS = "multiple_roots"
ENTROPY_BOUNDARY_INSUFFICIENT = "insufficient_data"


@dataclass(frozen=True)
class GapBoundaryTask:
    row_index: int
    T_N: float
    B14: float
    n0: float
    ms: float
    nm_type: str
    upB: int
    m_lo: float
    m_hi: float


@dataclass(frozen=True)
class EntropyBoundaryTask:
    row_index: int
    T_N: float
    B14: float
    n0: float
    nB_crit0: float
    xi: float
    ms: float
    nm_type: str
    upB: int
    m_lo: float
    m_hi: float
    probe_count: int
    interface_fraction_mode: str


@dataclass(frozen=True)
class NumericalPointTask:
    row_index: int
    col_index: int
    T_N: float
    metastability: float
    n0: float
    nB_crit0: float
    B14: float
    ms: float
    nm_type: str
    upB: int
    TQstar: float
    tail_eps: float
    n_mesh: int
    tol_bvp: float
    max_nodes: int
    jB_guess: float | None
    TQ_guess: float | None
    jB_bounds: tuple[float, float] | None
    auto_jB_bounds: bool
    shooting_u_max: float
    shooting_jB_lower: float
    point_timeout: float
    jB_guess_source: str
    continuation_direction: str
    continuation_seed_temperature: float
    retry_attempt: int = 0
    retry_of: tuple[int, int] | None = None


class PointTimeoutError(TimeoutError):
    """Raised when one numerical BVP exceeds its configured wall time."""


def _default_workers() -> int:
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


def _as_finite_float(value: Any, default: float = np.nan) -> float:
    try:
        converted = float(value)
    except Exception:
        return float(default)
    return converted if np.isfinite(converted) else float(default)


def _optional_finite_float(value: Any) -> float | None:
    converted = _as_finite_float(value)
    return float(converted) if np.isfinite(converted) else None


def _parse_float_pair(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    fields = [field.strip() for field in value.split(",")]
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("expected LOWER,UPPER")
    pair = (float(fields[0]), float(fields[1]))
    if not np.all(np.isfinite(pair)) or not 0.0 < pair[0] < pair[1]:
        raise argparse.ArgumentTypeError("bounds must satisfy 0 < LOWER < UPPER")
    return pair


def build_temperature_axis(t_min: float, t_max: float, count: int) -> np.ndarray:
    return np.geomspace(float(t_min), float(t_max), int(count))


def temperature_row_order(values: np.ndarray, seed_temperature: float) -> list[int]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0.0):
        raise ValueError("temperature values must be a positive one-dimensional array")
    seed = int(np.argmin(np.abs(np.log(values) - np.log(float(seed_temperature)))))
    return [seed, *range(seed + 1, values.size), *range(seed - 1, -1, -1)]


def temperature_row_schedule(
    values: np.ndarray, seed_temperature: float
) -> list[tuple[int, int | None, str]]:
    values = np.asarray(values, dtype=float)
    order = temperature_row_order(values, seed_temperature)
    seed = order[0]
    schedule: list[tuple[int, int | None, str]] = [(seed, None, "seed")]
    schedule.extend((index, index - 1, "up") for index in range(seed + 1, values.size))
    schedule.extend((index, index + 1, "down") for index in range(seed - 1, -1, -1))
    return schedule


def _boundary_lower(
    boundary: float,
    status: int | str,
    allowed_above: bool,
    all_allowed_status: int | str,
    root_status: int | str,
    m_min: float,
    buffer_fraction: float,
) -> float | None:
    if status == all_allowed_status:
        return float(m_min)
    if status == root_status and allowed_above and np.isfinite(boundary):
        return max(float(m_min), float(boundary) * (1.0 + float(buffer_fraction)))
    return None


def build_entropy_safe_metastability_row(
    *,
    gap_boundary: float,
    gap_status: int,
    gap_allowed_above: bool,
    entropy_boundary: float,
    entropy_status: str,
    entropy_allowed_above: bool,
    m_min: float,
    m_max: float,
    count: int,
    gap_buffer_fraction: float,
    entropy_buffer_fraction: float,
) -> np.ndarray | None:
    gap_lower = _boundary_lower(
        gap_boundary,
        gap_status,
        gap_allowed_above,
        GAP_BOUNDARY_ALL_ALLOWED,
        GAP_BOUNDARY_ROOT,
        m_min,
        gap_buffer_fraction,
    )
    entropy_lower = _boundary_lower(
        entropy_boundary,
        entropy_status,
        entropy_allowed_above,
        ENTROPY_BOUNDARY_ALL_ALLOWED,
        ENTROPY_BOUNDARY_ROOT,
        m_min,
        entropy_buffer_fraction,
    )
    if gap_lower is None or entropy_lower is None:
        return None
    lower = max(gap_lower, entropy_lower)
    if not 0.0 < lower < float(m_max):
        return None
    return np.geomspace(lower, float(m_max), int(count))


def _entropy_crossings(
    metastability: np.ndarray, entropy_jump: np.ndarray
) -> list[float]:
    metastability = np.asarray(metastability, dtype=float)
    entropy_jump = np.asarray(entropy_jump, dtype=float)
    finite = np.isfinite(metastability) & (metastability > 0.0) & np.isfinite(entropy_jump)
    metastability = metastability[finite]
    entropy_jump = entropy_jump[finite]
    if metastability.size < 2:
        return []
    order = np.argsort(metastability)
    metastability = metastability[order]
    entropy_jump = entropy_jump[order]
    roots: list[float] = []
    for index in range(metastability.size - 1):
        left = float(entropy_jump[index])
        right = float(entropy_jump[index + 1])
        if left == 0.0:
            roots.append(float(metastability[index]))
        elif left * right < 0.0:
            fraction = -left / (right - left)
            root = np.exp(
                np.log(metastability[index])
                + fraction
                * (np.log(metastability[index + 1]) - np.log(metastability[index]))
            )
            roots.append(float(root))
    if entropy_jump[-1] == 0.0:
        roots.append(float(metastability[-1]))
    return roots


def entropy_zero_crossing(
    metastability: np.ndarray, entropy_jump: np.ndarray
) -> float:
    roots = _entropy_crossings(metastability, entropy_jump)
    return float(roots[0]) if len(roots) == 1 else np.nan


def entropy_boundary_for_samples(
    metastability: np.ndarray, entropy_jump: np.ndarray
) -> tuple[float, str, bool, list[float]]:
    metastability = np.asarray(metastability, dtype=float)
    entropy_jump = np.asarray(entropy_jump, dtype=float)
    finite = np.isfinite(metastability) & (metastability > 0.0) & np.isfinite(entropy_jump)
    values = entropy_jump[finite]
    roots = _entropy_crossings(metastability, entropy_jump)
    if values.size < 2:
        return np.nan, ENTROPY_BOUNDARY_INSUFFICIENT, True, roots
    if len(roots) == 1:
        order = np.argsort(metastability[finite])
        allowed_above = bool(values[order][-1] >= 0.0)
        return float(roots[0]), ENTROPY_BOUNDARY_ROOT, allowed_above, roots
    if len(roots) > 1:
        return np.nan, ENTROPY_BOUNDARY_MULTIPLE_ROOTS, True, roots
    if np.all(values >= 0.0):
        return np.nan, ENTROPY_BOUNDARY_ALL_ALLOWED, True, []
    if np.all(values < 0.0):
        return np.nan, ENTROPY_BOUNDARY_ALL_FORBIDDEN, True, []
    return np.nan, ENTROPY_BOUNDARY_INSUFFICIENT, True, []


def entropy_jump_per_baryon(row: dict[str, Any]) -> float:
    values = np.asarray(
        [
            row.get("T_MeV", np.nan),
            row.get("T_Q_MeV", np.nan),
            row.get("h_over_nB_N_MeV", np.nan),
            row.get("muB_N_MeV", np.nan),
            row.get("h_over_nB_Q_MeV", np.nan),
            row.get("muB_Q_MeV", np.nan),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or values[0] <= 0.0 or values[1] <= 0.0:
        return np.nan
    T_N, T_Q, h_N_over_nB, muB_N, h_Q_over_nB, muB_Q = values
    return float((h_Q_over_nB - muB_Q) / T_Q - (h_N_over_nB - muB_N) / T_N)


def gap_at_metastability(
    metastability: float,
    T_N: float,
    B14: float,
    n0: float,
    nB_crit_T: float,
    nm_type: str,
) -> float:
    nB_N = float(nB_crit_T + float(metastability) * float(n0))
    muB_N = muB_from_nB(nB_N, float(T_N), str(nm_type), prefer=1100.0)
    state = phase_velocity._analytic_nuclear_state(
        float(muB_N),
        float(T_N),
        param=para.paraQMCRMF3,
        NM_type=str(nm_type),
    )
    gap, _ = phase_velocity._slow_front_enthalpy_gap(state, float(B14))
    return float(gap)


def solve_gap_boundary_at_temperature(task: GapBoundaryTask) -> dict[str, Any]:
    row = {
        "row_index": int(task.row_index),
        "T_MeV": float(task.T_N),
        "metastability_boundary": np.nan,
        "status": GAP_BOUNDARY_FAILURE,
        "allowed_above_boundary": True,
        "gap_at_m_min": np.nan,
        "gap_at_m_max": np.nan,
        "message": "",
    }
    try:
        _, _, nB_crit_T = critical_state_for_temperature(
            task.T_N, task.B14, task.ms, task.nm_type, task.upB
        )

        def gap(metastability: float) -> float:
            return gap_at_metastability(
                metastability,
                task.T_N,
                task.B14,
                task.n0,
                nB_crit_T,
                task.nm_type,
            )

        gap_lo = gap(task.m_lo)
        gap_hi = gap(task.m_hi)
        row.update(gap_at_m_min=float(gap_lo), gap_at_m_max=float(gap_hi))
        if not np.all(np.isfinite([gap_lo, gap_hi])):
            row["message"] = "non-finite enthalpy-gap endpoint"
        elif gap_lo <= 0.0 and gap_hi <= 0.0:
            row.update(status=GAP_BOUNDARY_ALL_ALLOWED, message="all allowed")
        elif gap_lo > 0.0 and gap_hi > 0.0:
            row.update(status=GAP_BOUNDARY_ALL_FORBIDDEN, message="all forbidden")
        else:
            root = brentq(gap, task.m_lo, task.m_hi, xtol=1.0e-8, rtol=1.0e-10)
            row.update(
                metastability_boundary=float(root),
                status=GAP_BOUNDARY_ROOT,
                allowed_above_boundary=bool(gap_hi <= 0.0),
                message="Delta h = 0 root",
            )
    except Exception as exc:
        row["message"] = str(exc)
    return row


def _analytic_entropy_row(
    task: EntropyBoundaryTask,
    metastability: float,
    endpoint_guess: tuple[float, float] | None,
) -> dict[str, Any]:
    row = calculate_analytic_point(
        AnalyticPointTask(
            T_N=float(task.T_N),
            metastability=float(metastability),
            n0=float(task.n0),
            nB_crit0=float(task.nB_crit0),
            B14=float(task.B14),
            xi=float(task.xi),
            ms=float(task.ms),
            nm_type=str(task.nm_type),
            upB=int(task.upB),
            interface_fraction_mode=str(task.interface_fraction_mode),
            endpoint_initial_guess=endpoint_guess,
        )
    )
    row["entropy_jump_per_baryon"] = entropy_jump_per_baryon(row)
    return row


def solve_entropy_boundary_at_temperature(task: EntropyBoundaryTask) -> dict[str, Any]:
    probes = np.geomspace(float(task.m_lo), float(task.m_hi), max(2, int(task.probe_count)))
    evaluations: list[dict[str, Any]] = []
    endpoint_guess: tuple[float, float] | None = None
    for metastability in probes[::-1]:
        row = _analytic_entropy_row(task, float(metastability), endpoint_guess)
        evaluations.append(row)
        if row.get("success"):
            endpoint = row.get("endpoint_solution")
            if endpoint is not None and np.all(np.isfinite(endpoint)):
                endpoint_guess = (float(endpoint[0]), float(endpoint[1]))
    evaluations.sort(key=lambda item: float(item.get("metastability", np.inf)))
    finite_rows = [
        row
        for row in evaluations
        if row.get("success") and np.isfinite(row.get("entropy_jump_per_baryon", np.nan))
    ]
    metastability = np.asarray([row["metastability"] for row in finite_rows], dtype=float)
    jumps = np.asarray([row["entropy_jump_per_baryon"] for row in finite_rows], dtype=float)
    boundary, status, allowed_above, roots = entropy_boundary_for_samples(
        metastability, jumps
    )
    if status == ENTROPY_BOUNDARY_ROOT:
        signs = np.sign(jumps)
        bracket = next(
            (
                (float(metastability[i]), float(metastability[i + 1]))
                for i in range(metastability.size - 1)
                if signs[i] == 0.0 or signs[i] * signs[i + 1] < 0.0
            ),
            None,
        )
        if bracket is not None and bracket[0] != bracket[1]:
            cache: dict[float, dict[str, Any]] = {
                float(row["metastability"]): row for row in finite_rows
            }

            def jump(metastability_value: float) -> float:
                nearest = min(
                    cache.values(),
                    key=lambda row: abs(
                        np.log(float(row["metastability"]))
                        - np.log(float(metastability_value))
                    ),
                )
                row = _analytic_entropy_row(
                    task,
                    float(metastability_value),
                    nearest.get("endpoint_solution"),
                )
                evaluations.append(row)
                value = float(row.get("entropy_jump_per_baryon", np.nan))
                if not row.get("success") or not np.isfinite(value):
                    raise RuntimeError("entropy-boundary refinement failed")
                cache[float(metastability_value)] = row
                return value

            try:
                boundary = float(
                    brentq(jump, bracket[0], bracket[1], xtol=1.0e-8, rtol=1.0e-10)
                )
                roots = [boundary]
            except Exception as exc:
                status = ENTROPY_BOUNDARY_INSUFFICIENT
                boundary = np.nan
                refinement_message = str(exc)
            else:
                refinement_message = ""
        else:
            refinement_message = ""
    else:
        refinement_message = ""
    A_values = np.asarray(
        [row.get("A_boundary", np.nan) for row in evaluations if row.get("success")],
        dtype=float,
    )
    finite_A = A_values[np.isfinite(A_values)]
    return {
        "row_index": int(task.row_index),
        "T_MeV": float(task.T_N),
        "metastability_boundary": float(boundary),
        "status": str(status),
        "allowed_above_boundary": bool(allowed_above),
        "roots": list(roots),
        "successful_probes": int(len(finite_rows)),
        "probe_count": int(len(probes)),
        "A_diagnostic_max": float(np.max(finite_A)) if finite_A.size else np.nan,
        "unexpected_A_ge_1": bool(finite_A.size and np.max(finite_A) >= 1.0),
        "message": refinement_message,
        "evaluations": evaluations,
    }


@contextmanager
def point_timeout(seconds: float | None):
    seconds = _as_finite_float(seconds, default=0.0)
    if seconds <= 0.0 or not hasattr(signal, "setitimer"):
        yield
        return
    old_handler = signal.getsignal(signal.SIGALRM)
    old_delay, old_interval = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def handle_timeout(signum, frame):
        del signum, frame
        raise PointTimeoutError(f"numerical point exceeded {seconds:g} s")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, old_delay, old_interval)
        signal.signal(signal.SIGALRM, old_handler)


def resolve_jB_bounds(
    nB_N: float,
    explicit_bounds: tuple[float, float] | None,
    auto_bounds: bool,
    shooting_u_max: float,
    shooting_jB_lower: float,
) -> tuple[tuple[float, float] | None, dict[str, Any]]:
    if explicit_bounds is not None:
        bounds = (float(explicit_bounds[0]), float(explicit_bounds[1]))
        return bounds, {
            "jB_bounds_used": True,
            "jB_lower_bound": bounds[0],
            "jB_upper_bound": bounds[1],
            "shooting_u_max": np.nan,
            "auto_jB_bounds": False,
        }
    if not auto_bounds:
        return None, {
            "jB_bounds_used": False,
            "jB_lower_bound": np.nan,
            "jB_upper_bound": np.nan,
            "shooting_u_max": np.nan,
            "auto_jB_bounds": False,
        }
    lower = float(shooting_jB_lower)
    upper = float(nB_N) * float(shooting_u_max)
    if not 0.0 < lower < upper:
        raise RuntimeError("automatic jB bounds require 0 < lower < nB_N*shooting_u_max")
    return (lower, upper), {
        "jB_bounds_used": True,
        "jB_lower_bound": lower,
        "jB_upper_bound": upper,
        "shooting_u_max": float(shooting_u_max),
        "auto_jB_bounds": True,
    }


def _numerical_entropy_jump(row: dict[str, Any]) -> float:
    values = np.asarray(
        [
            row.get("T_MeV", np.nan),
            row.get("T_Q_MeV", np.nan),
            row.get("h_N_MeV4", np.nan),
            row.get("nB_N_MeV3", np.nan),
            row.get("muB_N_MeV", np.nan),
            row.get("h_Q_MeV4", np.nan),
            row.get("nB_Q_MeV3", np.nan),
            row.get("muB_Q_MeV", np.nan),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or np.any(values[[0, 1, 3, 6]] <= 0.0):
        return np.nan
    T_N, T_Q, h_N, nB_N, muB_N, h_Q, nB_Q, muB_Q = values
    return float((h_Q / nB_Q - muB_Q) / T_Q - (h_N / nB_N - muB_N) / T_N)


def calculate_numerical_point(task: NumericalPointTask) -> dict[str, Any]:
    started = time.monotonic()
    row = {
        "success": False,
        "message": "",
        "row_index": int(task.row_index),
        "col_index": int(task.col_index),
        "T_MeV": float(task.T_N),
        "metastability": float(task.metastability),
        "metastability_definition": METASTABILITY_DEFINITION,
        "TQstar_target_MeV": float(task.TQstar),
        "u_N": np.nan,
        "u_N_m_s": np.nan,
        "jB": np.nan,
        "lambda_n": np.nan,
        "A_boundary": np.nan,
        "a_local_Qstar": np.nan,
        "a_local_end": np.nan,
        "aQstar": np.nan,
        "aQstar_derived": np.nan,
        "aQstar_max": np.nan,
        "entropy_jump_per_baryon": np.nan,
        "entropy_allowed": False,
        "point_timed_out": False,
        "point_timeout_s": float(task.point_timeout),
        "point_elapsed_s": np.nan,
        "jB_guess_used": _as_finite_float(task.jB_guess),
        "jB_guess_source": str(task.jB_guess_source),
        "continuation_direction": str(task.continuation_direction),
        "continuation_seed_temperature_MeV": float(
            task.continuation_seed_temperature
        ),
        "retry_attempt": int(task.retry_attempt),
        "retry_of": task.retry_of,
        "retry_replaced_failure": False,
        "solver_variant": NUMERICAL_CLOSURE_LABEL,
        "root_method": NUMERICAL_ROOT_METHOD,
        "bvp_parameter_count": 0,
        "composition_definition": "nK_over_local_nB",
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "current_definition": CURRENT_DEFINITION,
        "rate_model": RATE_MODEL,
    }
    try:
        muB_crit_T, P_crit_T, nB_crit_T = critical_state_for_temperature(
            task.T_N, task.B14, task.ms, task.nm_type, task.upB
        )
        nB_N = float(nB_crit_T + task.metastability * task.n0)
        point_bounds, bounds_diagnostics = resolve_jB_bounds(
            nB_N,
            task.jB_bounds,
            task.auto_jB_bounds,
            task.shooting_u_max,
            task.shooting_jB_lower,
        )
        with point_timeout(task.point_timeout):
            result = phase_velocity.solve_front_energy_conserving_uNmax(
                T=float(task.T_N),
                nB_N=nB_N,
                B_one_forth=float(task.B14),
                TQstar=float(task.TQstar),
                ms=float(task.ms),
                param=para.paraQMCRMF3,
                NM_type=str(task.nm_type),
                tail_eps=float(task.tail_eps),
                n_mesh=int(task.n_mesh),
                tol_bvp=float(task.tol_bvp),
                max_nodes=int(task.max_nodes),
                jB_guess=task.jB_guess,
                TQ_guess=task.TQ_guess,
                jB_bounds=point_bounds,
                return_profile=False,
                verb=False,
            )
        u_N = _as_finite_float(result.get("u_N_max_candidate", result.get("u_N")))
        muB_Q = _as_finite_float(result.get("muB_Q"))
        T_Q = _as_finite_float(result.get("T_Q"))
        A_boundary = (
            float(9.0 * np.pi / np.sqrt(2.0) * T_Q / muB_Q)
            if np.isfinite(muB_Q) and muB_Q > 0.0 and np.isfinite(T_Q)
            else np.nan
        )
        a_local_Qstar = _as_finite_float(
            result.get("aQstar_derived", result.get("aQstar"))
        )
        row.update(
            {
                "success": bool(result.get("success", False)),
                "message": str(result.get("message", "")),
                "nB_N_over_n0": float(nB_N / task.n0),
                "nB_N_MeV3": nB_N,
                "nB_crit_T_over_n0": float(nB_crit_T / task.n0),
                "nB_crit_T_MeV3": float(nB_crit_T),
                "muB_crit_T_MeV": float(muB_crit_T),
                "P_crit_T_MeV4": float(P_crit_T),
                "muB_N_MeV": _as_finite_float(result.get("muB_N")),
                "u_N": u_N,
                "u_N_m_s": float(u_N * const.m_per_sec) if np.isfinite(u_N) else np.nan,
                "jB": _as_finite_float(result.get("jB")),
                "u_Q": _as_finite_float(result.get("u_Q")),
                "lambda_n": _as_finite_float(result.get("lambda_n")),
                "A_boundary": A_boundary,
                "a_local_Qstar": a_local_Qstar,
                "a_local_end": _as_finite_float(result.get("a_end")),
                "aQstar": _as_finite_float(result.get("aQstar")),
                "aQstar_derived": a_local_Qstar,
                "aQstar_max": a_local_Qstar,
                "muB_Q_MeV": muB_Q,
                "T_Q_MeV": T_Q,
                "T_Qstar_MeV": _as_finite_float(result.get("T_Qstar")),
                "nB_Q_MeV3": _as_finite_float(result.get("nB_Q")),
                "nK_Q_MeV3": _as_finite_float(result.get("nK_Q")),
                "nB_Qstar_MeV3": _as_finite_float(result.get("nB_Qstar")),
                "nK_Qstar_MeV3": _as_finite_float(result.get("nK_Qstar")),
                "nK_end": _as_finite_float(result.get("nK_end")),
                "jK_end": _as_finite_float(result.get("jK_end")),
                "h_N_MeV4": _as_finite_float(result.get("h_N")),
                "h_Q_MeV4": _as_finite_float(result.get("h_Q")),
                "root_method": str(result.get("_root_method", NUMERICAL_ROOT_METHOD)),
                "solver_variant": str(
                    result.get("solver_variant", NUMERICAL_CLOSURE_LABEL)
                ),
                "bvp_parameter_count": int(
                    np.asarray(result.get("bvp_parameters", [])).size
                ),
                "bvp_status": int(result.get("bvp_status", -1)),
                "bvp_message": str(result.get("bvp_message", "")),
                "bvp_niter": int(result.get("bvp_niter", -1)),
                "bvp_nodes": int(result.get("bvp_nodes", -1)),
                "bvp_ode_calls": int(result.get("bvp_ode_calls", -1)),
                "downstream_root_calls": int(result.get("downstream_root_calls", -1)),
                "local_root_calls": int(result.get("local_root_calls", -1)),
                **bounds_diagnostics,
            }
        )
        row["entropy_jump_per_baryon"] = _numerical_entropy_jump(row)
        row["entropy_allowed"] = bool(
            np.isfinite(row["entropy_jump_per_baryon"])
            and row["entropy_jump_per_baryon"] >= 0.0
        )
    except PointTimeoutError as exc:
        row.update(
            message=str(exc),
            point_timed_out=True,
            bvp_status=STATUS_TIMEOUT,
        )
    except Exception as exc:
        row["message"] = str(exc)
    row["point_elapsed_s"] = float(time.monotonic() - started)
    return row


def _successful_jB(row: dict[str, Any] | None) -> float | None:
    if not row or not bool(row.get("success", False)):
        return None
    return _optional_finite_float(row.get("jB"))


def adjacent_row_jB_guess(
    *,
    metastability: float,
    adjacent_row: dict[int, dict[str, Any]],
    default: float | None,
) -> tuple[float | None, str]:
    if not adjacent_row:
        return _optional_finite_float(default), "default"
    candidates = [
        (
            abs(
                np.log(float(record.get("metastability")))
                - np.log(float(metastability))
            ),
            record,
        )
        for record in adjacent_row.values()
        if _as_finite_float(record.get("metastability")) > 0.0
    ]
    if not candidates:
        return _optional_finite_float(default), "default"
    nearest = min(candidates, key=lambda item: item[0])[1]
    guess = _successful_jB(nearest)
    if guess is not None:
        return guess, "adjacent_temperature_nearest_log_m"
    return _optional_finite_float(default), "fallback_after_failed_nearest_log_m"


def same_row_jB_guess(
    col_index: int,
    solved_row: dict[int, dict[str, Any]],
    default: float | None,
) -> tuple[float | None, str]:
    guess = _successful_jB(solved_row.get(int(col_index) + 1))
    if guess is not None:
        return guess, "same_row_higher_metastability_neighbor"
    return _optional_finite_float(default), "fallback_after_failed_neighbor"


def _parallel_map_unordered(function, tasks: Iterable[Any], workers: int):
    tasks = list(tasks)
    if not tasks:
        return
    if int(workers) == 1:
        for task in tasks:
            yield function(task)
        return
    with ProcessPoolExecutor(max_workers=min(int(workers), len(tasks))) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, payload, allow_pickle=True)
    temporary.replace(path)


def build_payload(
    *,
    args: argparse.Namespace,
    calibration: tuple[float, float, float, float, float],
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    gap_boundary_records: list[dict[str, Any]],
    entropy_boundary_records: list[dict[str, Any]],
) -> dict[str, Any]:
    B14, muB_crit0, P_crit0, n0, nB_crit0 = calibration
    A_grid = np.asarray(arrays.get("A_boundary_grid", np.array([])), dtype=float)
    finite_A = A_grid[np.isfinite(A_grid)]
    return {
        "run_tag": str(args.run_tag),
        "temperature_axis_scale": "log",
        "metastability_axis_scale": "log",
        "mesh_layout": "adaptive_logT_logm_entropy_safe",
        "numerical_closure": NUMERICAL_CLOSURE_LABEL,
        "solver_variant": NUMERICAL_CLOSURE_LABEL,
        "root_method": NUMERICAL_ROOT_METHOD,
        "composition_definition": "nK_over_local_nB",
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "current_definition": CURRENT_DEFINITION,
        "rate_model": RATE_MODEL,
        "metastability_definition": METASTABILITY_DEFINITION,
        "gap_definition": GAP_DEFINITION,
        "entropy_jump_definition": ENTROPY_JUMP_DEFINITION,
        "safe_region_definition": "Delta_h <= 0 and Delta_s_per_baryon >= 0",
        "entropy_affects_mesh": True,
        "entropy_boundary_closure": "analytic_extreme_endothermic",
        "A_boundary_handling": "diagnostic_only_T_max_50_MeV",
        "A_diagnostic_max": float(np.max(finite_A)) if finite_A.size else np.nan,
        "TQstar_target_MeV": float(args.TQstar),
        "B14_calibrated_MeV": float(B14),
        "muB_crit0_MeV": float(muB_crit0),
        "P_crit0_MeV4": float(P_crit0),
        "n0_MeV3": float(n0),
        "nB_crit0_MeV3": float(nB_crit0),
        "nB_crit0_over_n0": float(nB_crit0 / n0),
        "status_codes": {
            "finite": STATUS_FINITE,
            "numerical_failure": STATUS_NUMERICAL_FAILURE,
            "unscanned": STATUS_UNSCANNED,
            "timeout": STATUS_TIMEOUT,
        },
        **{name: np.asarray(value) for name, value in arrays.items()},
        "records": list(records),
        "gap_boundary_records": list(gap_boundary_records),
        "entropy_boundary_records": list(entropy_boundary_records),
        "arguments": vars(args).copy(),
    }


def _save_current(
    output: Path,
    args: argparse.Namespace,
    calibration: tuple[float, float, float, float, float],
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    gap_boundary_records: list[dict[str, Any]],
    entropy_boundary_records: list[dict[str, Any]],
) -> None:
    _atomic_save(
        output,
        build_payload(
            args=args,
            calibration=calibration,
            arrays=arrays,
            records=records,
            gap_boundary_records=gap_boundary_records,
            entropy_boundary_records=entropy_boundary_records,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--run-tag", default=os.getenv("RUN_TAG", DEFAULT_RUN_TAG))
    parser.add_argument("--t-min", type=float, default=DEFAULT_T_MIN_MEV)
    parser.add_argument("--t-max", type=float, default=DEFAULT_T_MAX_MEV)
    parser.add_argument("--t-count", type=int, default=DEFAULT_T_COUNT)
    parser.add_argument("--m-min", type=float, default=DEFAULT_M_MIN)
    parser.add_argument("--m-max", type=float, default=DEFAULT_M_MAX)
    parser.add_argument("--m-count", type=int, default=DEFAULT_M_COUNT)
    parser.add_argument(
        "--gap-buffer-fraction", type=float, default=DEFAULT_GAP_BUFFER_FRACTION
    )
    parser.add_argument(
        "--entropy-buffer-fraction",
        type=float,
        default=DEFAULT_ENTROPY_BUFFER_FRACTION,
    )
    parser.add_argument(
        "--entropy-probe-count", type=int, default=DEFAULT_ENTROPY_PROBE_COUNT
    )
    parser.add_argument(
        "--seed-temperature", type=float, default=DEFAULT_SEED_TEMPERATURE_MEV
    )
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--TQstar", type=float, default=DEFAULT_TQSTAR_MEV)
    parser.add_argument("--tail-eps", type=float, default=DEFAULT_TAIL_EPS)
    parser.add_argument("--n-mesh", type=int, default=DEFAULT_N_MESH)
    parser.add_argument("--tol-bvp", type=float, default=DEFAULT_TOL_BVP)
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument("--jB-guess", type=float, default=DEFAULT_JB_GUESS)
    parser.add_argument(
        "--TQ-guess",
        type=float,
        default=DEFAULT_TQ_GUESS,
        help="optional downstream endpoint seed; not a BVP parameter",
    )
    parser.add_argument("--jB-bounds", type=_parse_float_pair, default=DEFAULT_JB_BOUNDS)
    parser.add_argument(
        "--no-auto-jB-bounds",
        action="store_false",
        dest="auto_jB_bounds",
        default=DEFAULT_AUTO_JB_BOUNDS,
    )
    parser.add_argument("--shooting-u-max", type=float, default=DEFAULT_SHOOTING_U_MAX)
    parser.add_argument(
        "--shooting-jB-lower", type=float, default=DEFAULT_SHOOTING_JB_LOWER
    )
    parser.add_argument("--point-timeout", type=float, default=DEFAULT_POINT_TIMEOUT_S)
    parser.add_argument(
        "--no-point-timeout", action="store_const", const=0.0, dest="point_timeout"
    )
    parser.add_argument(
        "--no-neighbor-retry",
        action="store_false",
        dest="neighbor_retry",
        default=DEFAULT_NEIGHBOR_RETRY,
    )
    parser.add_argument(
        "--interface-fraction-mode",
        choices=("extreme_endothermic",),
        default=DEFAULT_INTERFACE_FRACTION_MODE,
    )
    parser.add_argument("--ncrit-over-n0", type=float, default=DEFAULT_NCRIT_OVER_N0)
    parser.add_argument("--xi", type=float, default=DEFAULT_XI)
    parser.add_argument("--ms", type=float, default=DEFAULT_MS_MEV)
    parser.add_argument("--nm-type", default=DEFAULT_NM_TYPE)
    parser.add_argument("--upB", type=int, default=DEFAULT_UPB)
    parser.add_argument("--B14", "--b14", dest="B14", type=float, default=np.nan)
    parser.add_argument("--workers", type=int, default=_default_workers())
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.t_min < args.t_max or args.t_count < 2:
        raise ValueError("temperature range must satisfy 0 < t_min < t_max")
    if not 0.0 < args.m_min < args.m_max or args.m_count < 2:
        raise ValueError("metastability range must satisfy 0 < m_min < m_max")
    if args.entropy_probe_count < 2:
        raise ValueError("entropy-probe-count must be at least 2")
    if args.gap_buffer_fraction < 0.0 or args.entropy_buffer_fraction < 0.0:
        raise ValueError("boundary buffers must be nonnegative")
    if args.seed_temperature <= 0.0 or args.workers < 1 or args.save_every < 0:
        raise ValueError("seed temperature/workers/save interval are invalid")
    if args.TQstar < 0.0 or args.tail_eps <= 0.0:
        raise ValueError("TQstar must be nonnegative and tail-eps positive")
    if args.n_mesh < 2 or args.max_nodes < args.n_mesh or args.tol_bvp <= 0.0:
        raise ValueError("BVP mesh settings are invalid")
    if args.point_timeout < 0.0:
        raise ValueError("point-timeout must be nonnegative")


def _initialize_arrays(shape: tuple[int, int], temperature_values: np.ndarray):
    arrays = {
        "temperature_values": np.asarray(temperature_values, dtype=float),
        "temperature_grid": np.full(shape, np.nan),
        "metastability_grid": np.full(shape, np.nan),
        "velocity_grid_m_s": np.full(shape, np.nan),
        "u_N_grid": np.full(shape, np.nan),
        "jB_grid": np.full(shape, np.nan),
        "lambda_n_grid": np.full(shape, np.nan),
        "a_local_Qstar_grid": np.full(shape, np.nan),
        "aQstar_grid": np.full(shape, np.nan),
        "entropy_jump_grid": np.full(shape, np.nan),
        "A_boundary_grid": np.full(shape, np.nan),
        "status_grid": np.full(shape, STATUS_UNSCANNED, dtype=int),
        "point_elapsed_s_grid": np.full(shape, np.nan),
        "point_timed_out_grid": np.zeros(shape, dtype=bool),
        "gap_boundary_metastability": np.full(shape[0], np.nan),
        "entropy_boundary_metastability": np.full(shape[0], np.nan),
        "gap_boundary_status": np.full(shape[0], GAP_BOUNDARY_FAILURE, dtype=int),
        "entropy_boundary_status": np.full(shape[0], "", dtype=object),
    }
    return arrays


def _record_result(arrays: dict[str, np.ndarray], row: dict[str, Any]) -> None:
    i = int(row["row_index"])
    j = int(row["col_index"])
    arrays["temperature_grid"][i, j] = _as_finite_float(row.get("T_MeV"))
    arrays["metastability_grid"][i, j] = _as_finite_float(row.get("metastability"))
    arrays["entropy_jump_grid"][i, j] = _as_finite_float(
        row.get("entropy_jump_per_baryon")
    )
    arrays["A_boundary_grid"][i, j] = _as_finite_float(row.get("A_boundary"))
    arrays["point_elapsed_s_grid"][i, j] = _as_finite_float(row.get("point_elapsed_s"))
    arrays["point_timed_out_grid"][i, j] = bool(row.get("point_timed_out", False))
    if row.get("success"):
        arrays["velocity_grid_m_s"][i, j] = _as_finite_float(row.get("u_N_m_s"))
        arrays["u_N_grid"][i, j] = _as_finite_float(row.get("u_N"))
        arrays["jB_grid"][i, j] = _as_finite_float(row.get("jB"))
        arrays["lambda_n_grid"][i, j] = _as_finite_float(row.get("lambda_n"))
        arrays["a_local_Qstar_grid"][i, j] = _as_finite_float(
            row.get("a_local_Qstar")
        )
        arrays["aQstar_grid"][i, j] = arrays["a_local_Qstar_grid"][i, j]
        arrays["status_grid"][i, j] = STATUS_FINITE
    elif row.get("point_timed_out"):
        arrays["status_grid"][i, j] = STATUS_TIMEOUT
    else:
        arrays["status_grid"][i, j] = STATUS_NUMERICAL_FAILURE


def _make_point_task(
    *,
    args: argparse.Namespace,
    calibration: tuple[float, float, float, float, float],
    row_index: int,
    col_index: int,
    T_N: float,
    metastability: float,
    jB_guess: float | None,
    source: str,
    direction: str,
    retry_attempt: int = 0,
    retry_of: tuple[int, int] | None = None,
) -> NumericalPointTask:
    B14, _, _, n0, nB_crit0 = calibration
    return NumericalPointTask(
        row_index=int(row_index),
        col_index=int(col_index),
        T_N=float(T_N),
        metastability=float(metastability),
        n0=float(n0),
        nB_crit0=float(nB_crit0),
        B14=float(B14),
        ms=float(args.ms),
        nm_type=str(args.nm_type),
        upB=int(args.upB),
        TQstar=float(args.TQstar),
        tail_eps=float(args.tail_eps),
        n_mesh=int(args.n_mesh),
        tol_bvp=float(args.tol_bvp),
        max_nodes=int(args.max_nodes),
        jB_guess=_optional_finite_float(jB_guess),
        TQ_guess=_optional_finite_float(args.TQ_guess),
        jB_bounds=args.jB_bounds,
        auto_jB_bounds=bool(args.auto_jB_bounds),
        shooting_u_max=float(args.shooting_u_max),
        shooting_jB_lower=float(args.shooting_jB_lower),
        point_timeout=float(args.point_timeout),
        jB_guess_source=str(source),
        continuation_direction=str(direction),
        continuation_seed_temperature=float(args.seed_temperature),
        retry_attempt=int(retry_attempt),
        retry_of=retry_of,
    )


def resolve_calibration(
    args: argparse.Namespace,
) -> tuple[float, float, float, float, float]:
    n0 = float(0.16 * const.MeV_fm**3)
    nB_crit0 = float(args.ncrit_over_n0 * n0)
    if np.isfinite(args.B14) and args.B14 > 0.0:
        B14 = float(args.B14)
        muB_crit0 = np.nan
        P_crit0 = np.nan
    else:
        B14, muB_crit0, P_crit0 = calibrate_b14(
            nB_crit0, args.ms, args.nm_type
        )
    return (
        float(B14),
        float(muB_crit0),
        float(P_crit0),
        n0,
        nB_crit0,
    )


def _retry_guess(
    row_index: int,
    col_index: int,
    solved_rows: dict[int, dict[int, dict[str, Any]]],
) -> tuple[float | None, str]:
    same_row = solved_rows.get(row_index, {})
    for neighbor_col in (col_index - 1, col_index + 1):
        guess = _successful_jB(same_row.get(neighbor_col))
        if guess is not None:
            return guess, "retry_immediate_same_row_neighbor"
    for neighbor_row in (row_index - 1, row_index + 1):
        adjacent = solved_rows.get(neighbor_row, {})
        if not adjacent:
            continue
        target = same_row.get(col_index, {})
        metastability = _as_finite_float(target.get("metastability"))
        if metastability <= 0.0:
            continue
        guess, source = adjacent_row_jB_guess(
            metastability=metastability, adjacent_row=adjacent, default=None
        )
        if guess is not None:
            return guess, f"retry_{source}"
    return None, "retry_no_immediate_success"


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    calibration = resolve_calibration(args)
    B14, muB_crit0, P_crit0, n0, nB_crit0 = calibration
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = SCRIPT_DIR / output
    output = output.resolve()
    temperature_values = build_temperature_axis(args.t_min, args.t_max, args.t_count)
    shape = (int(args.t_count), int(args.m_count))
    arrays = _initialize_arrays(shape, temperature_values)
    records: list[dict[str, Any]] = []

    print(f"Calibrated B^(1/4) = {B14:.6f} MeV")
    print(f"Output: {output}")
    print(f"Run tag: {args.run_tag}")
    print(
        f"Log grid: {args.t_count} temperatures x {args.m_count} metastabilities; "
        f"TQstar={args.TQstar:g} MeV"
    )
    print(f"Workers: {args.workers}; point timeout: {args.point_timeout:g} s")

    gap_tasks = [
        GapBoundaryTask(
            row_index=i,
            T_N=float(T_N),
            B14=B14,
            n0=n0,
            ms=float(args.ms),
            nm_type=str(args.nm_type),
            upB=int(args.upB),
            m_lo=float(args.m_min),
            m_hi=float(args.m_max),
        )
        for i, T_N in enumerate(temperature_values)
    ]
    gap_by_row: dict[int, dict[str, Any]] = {}
    with tqdm(total=len(gap_tasks), desc="Delta h=0 boundary", unit="row") as progress:
        for row in _parallel_map_unordered(
            solve_gap_boundary_at_temperature, gap_tasks, args.workers
        ):
            gap_by_row[int(row["row_index"])] = row
            progress.update(1)
    gap_boundary_records = [gap_by_row[i] for i in range(len(temperature_values))]
    for i, row in enumerate(gap_boundary_records):
        arrays["gap_boundary_metastability"][i] = _as_finite_float(
            row.get("metastability_boundary")
        )
        arrays["gap_boundary_status"][i] = int(row.get("status", GAP_BOUNDARY_FAILURE))

    entropy_tasks = []
    for i, T_N in enumerate(temperature_values):
        gap_row = gap_by_row[i]
        preliminary = _boundary_lower(
            gap_row.get("metastability_boundary", np.nan),
            gap_row.get("status", GAP_BOUNDARY_FAILURE),
            bool(gap_row.get("allowed_above_boundary", True)),
            GAP_BOUNDARY_ALL_ALLOWED,
            GAP_BOUNDARY_ROOT,
            args.m_min,
            args.gap_buffer_fraction,
        )
        if preliminary is None or preliminary >= args.m_max:
            continue
        entropy_tasks.append(
            EntropyBoundaryTask(
                row_index=i,
                T_N=float(T_N),
                B14=B14,
                n0=n0,
                nB_crit0=nB_crit0,
                xi=float(args.xi),
                ms=float(args.ms),
                nm_type=str(args.nm_type),
                upB=int(args.upB),
                m_lo=float(preliminary),
                m_hi=float(args.m_max),
                probe_count=int(args.entropy_probe_count),
                interface_fraction_mode=str(args.interface_fraction_mode),
            )
        )
    entropy_by_row: dict[int, dict[str, Any]] = {}
    with tqdm(
        total=len(entropy_tasks), desc="Delta(s/nB)=0 boundary", unit="row"
    ) as progress:
        for row in _parallel_map_unordered(
            solve_entropy_boundary_at_temperature, entropy_tasks, args.workers
        ):
            entropy_by_row[int(row["row_index"])] = row
            progress.update(1)
    for i in range(len(temperature_values)):
        entropy_by_row.setdefault(
            i,
            {
                "row_index": i,
                "T_MeV": float(temperature_values[i]),
                "metastability_boundary": np.nan,
                "status": ENTROPY_BOUNDARY_ALL_FORBIDDEN,
                "allowed_above_boundary": True,
                "roots": [],
                "message": "row excluded by Delta h boundary",
                "evaluations": [],
            },
        )
    entropy_boundary_records = [entropy_by_row[i] for i in range(len(temperature_values))]
    for i, row in enumerate(entropy_boundary_records):
        arrays["entropy_boundary_metastability"][i] = _as_finite_float(
            row.get("metastability_boundary")
        )
        arrays["entropy_boundary_status"][i] = str(row.get("status", ""))

    metastability_rows: dict[int, np.ndarray] = {}
    for i in range(len(temperature_values)):
        gap_row = gap_by_row[i]
        entropy_row = entropy_by_row[i]
        values = build_entropy_safe_metastability_row(
            gap_boundary=_as_finite_float(gap_row.get("metastability_boundary")),
            gap_status=int(gap_row.get("status", GAP_BOUNDARY_FAILURE)),
            gap_allowed_above=bool(gap_row.get("allowed_above_boundary", True)),
            entropy_boundary=_as_finite_float(
                entropy_row.get("metastability_boundary")
            ),
            entropy_status=str(entropy_row.get("status", "")),
            entropy_allowed_above=bool(
                entropy_row.get("allowed_above_boundary", True)
            ),
            m_min=float(args.m_min),
            m_max=float(args.m_max),
            count=int(args.m_count),
            gap_buffer_fraction=float(args.gap_buffer_fraction),
            entropy_buffer_fraction=float(args.entropy_buffer_fraction),
        )
        if values is not None:
            metastability_rows[i] = values
            arrays["temperature_grid"][i, :] = float(temperature_values[i])
            arrays["metastability_grid"][i, :] = values

    _save_current(
        output,
        args,
        calibration,
        arrays,
        records,
        gap_boundary_records,
        entropy_boundary_records,
    )

    eligible_indices = sorted(metastability_rows)
    eligible_temperatures = temperature_values[eligible_indices]
    local_schedule = temperature_row_schedule(
        eligible_temperatures, args.seed_temperature
    ) if eligible_indices else []
    row_schedule = [
        (
            eligible_indices[local_index],
            None if local_parent is None else eligible_indices[local_parent],
            direction,
        )
        for local_index, local_parent, direction in local_schedule
    ]
    total_points = len(row_schedule) * int(args.m_count)
    solved_rows: dict[int, dict[int, dict[str, Any]]] = {}
    default_jB = _optional_finite_float(args.jB_guess)
    completed = 0
    with tqdm(total=total_points, desc="Numerical entropy-safe uNmax", unit="pt") as progress:
        for order_position, (row_index, adjacent_index, direction) in enumerate(row_schedule):
            T_N = float(temperature_values[row_index])
            metastabilities = metastability_rows[row_index]
            solved_rows[row_index] = {}
            if order_position == 0:
                for col_index in range(args.m_count - 1, -1, -1):
                    guess, source = same_row_jB_guess(
                        col_index, solved_rows[row_index], default_jB
                    )
                    task = _make_point_task(
                        args=args,
                        calibration=calibration,
                        row_index=row_index,
                        col_index=col_index,
                        T_N=T_N,
                        metastability=float(metastabilities[col_index]),
                        jB_guess=guess,
                        source=source,
                        direction="seed_row_high_to_low",
                    )
                    row = calculate_numerical_point(task)
                    solved_rows[row_index][col_index] = row
                    records.append(row)
                    _record_result(arrays, row)
                    completed += 1
                    progress.update(1)
                    progress.set_postfix(m=f"{metastabilities[col_index]:.3g}", T=f"{T_N:.3g}")
                    if args.save_every and completed % args.save_every == 0:
                        _save_current(output, args, calibration, arrays, records, gap_boundary_records, entropy_boundary_records)
            else:
                assert adjacent_index is not None
                adjacent = solved_rows[adjacent_index]
                tasks = []
                for col_index, metastability in enumerate(metastabilities):
                    guess, source = adjacent_row_jB_guess(
                        metastability=float(metastability),
                        adjacent_row=adjacent,
                        default=default_jB,
                    )
                    tasks.append(
                        _make_point_task(
                            args=args,
                            calibration=calibration,
                            row_index=row_index,
                            col_index=col_index,
                            T_N=T_N,
                            metastability=float(metastability),
                            jB_guess=guess,
                            source=source,
                            direction=direction,
                        )
                    )
                for row in _parallel_map_unordered(
                    calculate_numerical_point, tasks, args.workers
                ):
                    col_index = int(row["col_index"])
                    solved_rows[row_index][col_index] = row
                    records.append(row)
                    _record_result(arrays, row)
                    completed += 1
                    progress.update(1)
                    progress.set_postfix(
                        m=f"{row['metastability']:.3g}", T=f"{T_N:.3g}"
                    )
                    if args.save_every and completed % args.save_every == 0:
                        _save_current(output, args, calibration, arrays, records, gap_boundary_records, entropy_boundary_records)
            _save_current(output, args, calibration, arrays, records, gap_boundary_records, entropy_boundary_records)

        if args.neighbor_retry:
            retry_tasks = []
            for row_index, _, _ in row_schedule:
                for col_index, failed in solved_rows[row_index].items():
                    if failed.get("success"):
                        continue
                    guess, source = _retry_guess(row_index, col_index, solved_rows)
                    if guess is None:
                        continue
                    retry_tasks.append(
                        _make_point_task(
                            args=args,
                            calibration=calibration,
                            row_index=row_index,
                            col_index=col_index,
                            T_N=float(temperature_values[row_index]),
                            metastability=float(
                                metastability_rows[row_index][col_index]
                            ),
                            jB_guess=guess,
                            source=source,
                            direction="retry",
                            retry_attempt=1,
                            retry_of=(row_index, col_index),
                        )
                    )
            if retry_tasks:
                progress.total += len(retry_tasks)
                progress.refresh()
                for row in _parallel_map_unordered(
                    calculate_numerical_point, retry_tasks, args.workers
                ):
                    row["retry_replaced_failure"] = bool(row.get("success"))
                    records.append(row)
                    if row.get("success"):
                        solved_rows[int(row["row_index"])][int(row["col_index"])] = row
                        _record_result(arrays, row)
                    progress.update(1)
                    progress.set_postfix(
                        retry="1",
                        m=f"{row['metastability']:.3g}",
                        T=f"{row['T_MeV']:.3g}",
                    )

    _save_current(
        output,
        args,
        calibration,
        arrays,
        records,
        gap_boundary_records,
        entropy_boundary_records,
    )
    success_count = int(np.count_nonzero(arrays["status_grid"] == STATUS_FINITE))
    timeout_count = int(np.count_nonzero(arrays["status_grid"] == STATUS_TIMEOUT))
    print(f"Done. finite={success_count}, timeout={timeout_count}, records={len(records)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
