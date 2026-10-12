#!/usr/bin/env python3
"""Parallel analytical velocity scan on an adaptive log-T/log-m grid.

The grid spans 1e-2 <= T <= 50 MeV and 1e-5 <= m <= 1.  Each
temperature row starts on the allowed side of Delta h = 0.  The entropy jump
is recorded after the velocity scan and never changes the mesh.  No A = 1
boundary search is performed in this low-temperature domain.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    ANALYTIC_CLOSURE_LABEL,
    ANALYTIC_FORMULA_VARIANT,
    COMPOSITION_DEFINITION,
    DENSITY_RATIO_DEFINITION,
    AnalyticPointTask,
    calculate_point as calculate_analytic_point,
    calibrate_b14,
    critical_state_for_temperature,
    muB_from_nB,
)


# Tunable run parameters
DEFAULT_OUTPUT = "analytic-uN-logTm.npy"
DEFAULT_RUN_TAG = "analytic_uN_logT_logm_extreme_v1"
DEFAULT_T_MIN_MEV = 1.0e-2
DEFAULT_T_MAX_MEV = 50.0
DEFAULT_T_COUNT = 30
DEFAULT_M_MIN = 1.0e-5
DEFAULT_M_MAX = 1.0
DEFAULT_M_COUNT = 20
DEFAULT_GAP_BUFFER_FRACTION = 1.0e-3
DEFAULT_SEED_TEMPERATURE_MEV = 10.0
DEFAULT_NEIGHBOR_RETRY = True
DEFAULT_SAVE_EVERY = 10
DEFAULT_INTERFACE_FRACTION_MODE = "extreme_endothermic"
DEFAULT_ENDPOINT_GUESS = None
DEFAULT_NCRIT_OVER_N0 = 3.0
DEFAULT_XI = 0.0
DEFAULT_MS_MEV = 0.0
DEFAULT_NM_TYPE = "PNM"
DEFAULT_UPB = 5000

METASTABILITY_DEFINITION = "(nB_N - nB_crit(T)) / n0"
GAP_DEFINITION = "muB_cold(P_N) - h_N/nB_N"
ENTROPY_JUMP_DEFINITION = "sQ_over_nB_Q_minus_sN_over_nB_N"

STATUS_FINITE = 0
STATUS_NUMERICAL_FAILURE = 3
STATUS_UNSCANNED = 5

GAP_BOUNDARY_ROOT = 0
GAP_BOUNDARY_ALL_ALLOWED = 1
GAP_BOUNDARY_ALL_FORBIDDEN = 2
GAP_BOUNDARY_FAILURE = 3

ENTROPY_BOUNDARY_ROOT = "root"
ENTROPY_ALL_ALLOWED = "all_positive"
ENTROPY_ALL_FORBIDDEN = "all_nonpositive"
ENTROPY_MULTIPLE_ROOTS = "multiple_roots"
ENTROPY_INSUFFICIENT = "insufficient_data"


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
class PointWork:
    row_index: int
    col_index: int
    T_N: float
    metastability: float
    n0: float
    nB_crit0: float
    B14: float
    xi: float
    ms: float
    nm_type: str
    upB: int
    endpoint_guess: tuple[float, float] | None
    endpoint_guess_source: str
    continuation_direction: str
    continuation_seed_temperature: float
    retry_attempt: int = 0
    retry_of: tuple[int, int] | None = None


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
        value = float(value)
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _finite_endpoint(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        pair = (float(value[0]), float(value[1]))
    except Exception:
        return None
    if not np.all(np.isfinite(pair)) or pair[0] <= 0.0 or pair[1] <= 0.0:
        return None
    return pair


def _parse_endpoint(value: str | None) -> tuple[float, float] | None:
    if value is None or not str(value).strip():
        return None
    fields = [field.strip() for field in str(value).split(",")]
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("endpoint guess must be MU_B_Q,T_Q")
    endpoint = _finite_endpoint((float(fields[0]), float(fields[1])))
    if endpoint is None:
        raise argparse.ArgumentTypeError("endpoint guess values must be positive and finite")
    return endpoint


def build_temperature_axis(t_min: float, t_max: float, count: int) -> np.ndarray:
    return np.geomspace(float(t_min), float(t_max), int(count))


def build_metastability_row(
    *,
    boundary: float,
    boundary_status: int,
    allowed_above_boundary: bool,
    m_min: float,
    m_max: float,
    count: int,
    gap_buffer_fraction: float,
) -> np.ndarray | None:
    if boundary_status == GAP_BOUNDARY_ALL_FORBIDDEN:
        return None
    if boundary_status == GAP_BOUNDARY_ALL_ALLOWED:
        lower = float(m_min)
    elif boundary_status == GAP_BOUNDARY_ROOT and allowed_above_boundary:
        lower = max(
            float(m_min),
            float(boundary) * (1.0 + float(gap_buffer_fraction)),
        )
    else:
        return None
    if not (0.0 < lower < float(m_max)):
        return None
    return np.geomspace(lower, float(m_max), int(count))


def temperature_row_order(values: np.ndarray, seed_temperature: float) -> list[int]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("temperature values must be a finite one-dimensional array")
    seed = int(np.argmin(np.abs(np.log(values) - np.log(float(seed_temperature)))))
    return [seed, *range(seed + 1, values.size), *range(seed - 1, -1, -1)]


def _upstream_state(muB_N: float, T_N: float, nm_type: str):
    return phase_velocity._analytic_nuclear_state(
        float(muB_N),
        float(T_N),
        param=para.paraQMCRMF3,
        NM_type=str(nm_type),
    )


def _enthalpy_gap(muB_N: float, T_N: float, B14: float, nm_type: str) -> float:
    state = _upstream_state(muB_N, T_N, nm_type)
    gap, _ = phase_velocity._slow_front_enthalpy_gap(state, float(B14))
    return float(gap)


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
    return _enthalpy_gap(muB_N, T_N, B14, nm_type)


def solve_gap_boundary_at_temperature(task: GapBoundaryTask) -> dict[str, Any]:
    row = {
        "row_index": int(task.row_index),
        "T_MeV": float(task.T_N),
        "metastability_boundary": np.nan,
        "status": GAP_BOUNDARY_FAILURE,
        "gap_at_m_min": np.nan,
        "gap_at_m_max": np.nan,
        "allowed_above_boundary": True,
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

        gap_lo = float(gap(task.m_lo))
        gap_hi = float(gap(task.m_hi))
        row.update({"gap_at_m_min": gap_lo, "gap_at_m_max": gap_hi})
        if not np.all(np.isfinite([gap_lo, gap_hi])):
            row["message"] = "non-finite enthalpy-gap endpoint"
        elif gap_lo <= 0.0 and gap_hi <= 0.0:
            row.update(
                status=GAP_BOUNDARY_ALL_ALLOWED,
                message="all metastabilities allowed",
            )
        elif gap_lo > 0.0 and gap_hi > 0.0:
            row.update(
                status=GAP_BOUNDARY_ALL_FORBIDDEN,
                message="all metastabilities forbidden",
            )
        else:
            root = brentq(
                gap,
                float(task.m_lo),
                float(task.m_hi),
                xtol=1.0e-8,
                rtol=1.0e-10,
                maxiter=100,
            )
            row.update(
                metastability_boundary=float(root),
                status=GAP_BOUNDARY_ROOT,
                allowed_above_boundary=bool(gap_hi <= 0.0),
                message="Delta h = 0 root",
            )
    except Exception as exc:
        row["message"] = str(exc)
    return row


def adjacent_row_endpoint_guess(
    *,
    metastability: float,
    adjacent_row: dict[int, dict[str, Any]],
    default: tuple[float, float] | None,
) -> tuple[tuple[float, float] | None, str]:
    if not adjacent_row:
        return _finite_endpoint(default), "default"
    candidates = []
    for record in adjacent_row.values():
        candidate_m = _as_finite_float(record.get("metastability"))
        if candidate_m > 0.0:
            candidates.append((abs(np.log(candidate_m) - np.log(float(metastability))), record))
    if not candidates:
        return _finite_endpoint(default), "default"
    nearest = min(candidates, key=lambda item: item[0])[1]
    endpoint = _finite_endpoint(nearest.get("endpoint_solution")) if nearest.get("success") else None
    if endpoint is not None:
        return endpoint, "adjacent_temperature_nearest_log_m"
    return _finite_endpoint(default), "fallback_after_failed_nearest_log_m"


def calculate_velocity_point(work: PointWork) -> dict[str, Any]:
    task = AnalyticPointTask(
        T_N=float(work.T_N),
        metastability=float(work.metastability),
        n0=float(work.n0),
        nB_crit0=float(work.nB_crit0),
        B14=float(work.B14),
        xi=float(work.xi),
        ms=float(work.ms),
        nm_type=str(work.nm_type),
        upB=int(work.upB),
        interface_fraction_mode=DEFAULT_INTERFACE_FRACTION_MODE,
        endpoint_initial_guess=_finite_endpoint(work.endpoint_guess),
    )
    row = calculate_analytic_point(task)
    row.update(
        {
            "row_index": int(work.row_index),
            "col_index": int(work.col_index),
            "endpoint_guess_source": str(work.endpoint_guess_source),
            "continuation_direction": str(work.continuation_direction),
            "continuation_seed_temperature_MeV": float(
                work.continuation_seed_temperature
            ),
            "retry_attempt": int(work.retry_attempt),
            "retry_of": work.retry_of,
        }
    )
    return {
        "row_index": int(work.row_index),
        "col_index": int(work.col_index),
        "T_MeV": float(work.T_N),
        "metastability": float(work.metastability),
        "row": row,
    }


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
        left, right = entropy_jump[index], entropy_jump[index + 1]
        if left == 0.0:
            roots.append(float(metastability[index]))
            continue
        if left * right > 0.0:
            continue
        fraction = -left / (right - left)
        log_root = np.log(metastability[index]) + fraction * (
            np.log(metastability[index + 1]) - np.log(metastability[index])
        )
        roots.append(float(np.exp(log_root)))
    if entropy_jump[-1] == 0.0:
        roots.append(float(metastability[-1]))
    return roots


def entropy_zero_crossing(
    metastability: np.ndarray, entropy_jump: np.ndarray
) -> float:
    roots = _entropy_crossings(metastability, entropy_jump)
    return float(roots[0]) if len(roots) == 1 else np.nan


def entropy_boundary_for_row(
    metastability: np.ndarray, entropy_jump: np.ndarray
) -> tuple[float, str, list[float]]:
    metastability = np.asarray(metastability, dtype=float)
    entropy_jump = np.asarray(entropy_jump, dtype=float)
    finite = np.isfinite(metastability) & (metastability > 0.0) & np.isfinite(entropy_jump)
    values = entropy_jump[finite]
    if values.size < 2:
        return np.nan, ENTROPY_INSUFFICIENT, []
    roots = _entropy_crossings(metastability, entropy_jump)
    if len(roots) == 1:
        return roots[0], ENTROPY_BOUNDARY_ROOT, roots
    if len(roots) > 1:
        return np.nan, ENTROPY_MULTIPLE_ROOTS, roots
    if np.all(values > 0.0):
        return np.nan, ENTROPY_ALL_ALLOWED, []
    if np.all(values <= 0.0):
        return np.nan, ENTROPY_ALL_FORBIDDEN, []
    return np.nan, ENTROPY_INSUFFICIENT, []


def _parallel_map_unordered(function, tasks, workers: int):
    tasks = list(tasks)
    if not tasks:
        return
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), len(tasks)))) as pool:
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
) -> dict[str, Any]:
    B14, muB_crit0, P_crit0, n0, nB_crit0 = calibration
    A_grid = np.asarray(arrays.get("A_boundary_grid", np.array([])), dtype=float)
    finite_A = A_grid[np.isfinite(A_grid)]
    return {
        "run_tag": str(args.run_tag),
        "temperature_axis_scale": "log",
        "metastability_axis_scale": "log",
        "mesh_layout": "single_block_logT_adaptive_logm",
        "analytic_closure": ANALYTIC_CLOSURE_LABEL,
        "interface_fraction_mode": str(args.interface_fraction_mode),
        "analytic_formula_variant": ANALYTIC_FORMULA_VARIANT,
        "composition_definition": COMPOSITION_DEFINITION,
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "metastability_definition": METASTABILITY_DEFINITION,
        "gap_definition": GAP_DEFINITION,
        "safe_region_definition": "Delta_h <= 0",
        "A_boundary_handling": "not_scanned_T_max_50_MeV",
        "A_diagnostic_max": float(np.max(finite_A)) if finite_A.size else np.nan,
        "entropy_jump_definition": ENTROPY_JUMP_DEFINITION,
        "entropy_forbidden_condition": "entropy_jump <= 0",
        "entropy_affects_mesh": False,
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
        },
        **{name: np.asarray(value) for name, value in arrays.items()},
        "records": list(records),
        "gap_boundary_records": list(gap_boundary_records),
        "arguments": vars(args).copy(),
    }


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
        "--seed-temperature", type=float, default=DEFAULT_SEED_TEMPERATURE_MEV
    )
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument(
        "--endpoint-guess", type=_parse_endpoint, default=DEFAULT_ENDPOINT_GUESS
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
    if not 0.0 < args.t_min < args.t_max <= 50.0:
        raise ValueError("require 0 < t-min < t-max <= 50 MeV")
    if not 0.0 < args.m_min < args.m_max:
        raise ValueError("log metastability bounds require 0 < m-min < m-max")
    if args.t_count < 2 or args.m_count < 2:
        raise ValueError("t-count and m-count must be at least 2")
    if args.gap_buffer_fraction < 0.0:
        raise ValueError("gap-buffer-fraction must be nonnegative")
    if args.save_every < 0 or args.workers < 1:
        raise ValueError("save-every must be nonnegative and workers must be positive")


def _make_work(
    *,
    args: argparse.Namespace,
    row_index: int,
    col_index: int,
    temperature: float,
    metastability: float,
    n0: float,
    nB_crit0: float,
    B14: float,
    endpoint: tuple[float, float] | None,
    source: str,
    direction: str,
    retry_attempt: int = 0,
    retry_of: tuple[int, int] | None = None,
) -> PointWork:
    return PointWork(
        row_index=row_index,
        col_index=col_index,
        T_N=float(temperature),
        metastability=float(metastability),
        n0=n0,
        nB_crit0=nB_crit0,
        B14=B14,
        xi=args.xi,
        ms=args.ms,
        nm_type=args.nm_type,
        upB=args.upB,
        endpoint_guess=endpoint,
        endpoint_guess_source=source,
        continuation_direction=direction,
        continuation_seed_temperature=args.seed_temperature,
        retry_attempt=retry_attempt,
        retry_of=retry_of,
    )


def _retry_endpoint(
    row_index: int,
    col_index: int,
    metastability: float,
    row_results: dict[int, dict[int, dict[str, Any]]],
    default: tuple[float, float] | None,
) -> tuple[tuple[float, float] | None, str]:
    same_row = row_results.get(row_index, {})
    for neighbor, source in (
        (col_index - 1, "retry_same_row_lower_m"),
        (col_index + 1, "retry_same_row_higher_m"),
    ):
        record = same_row.get(neighbor)
        endpoint = _finite_endpoint(record.get("endpoint_solution")) if record and record.get("success") else None
        if endpoint is not None:
            return endpoint, source
    for neighbor_row, source in (
        (row_index - 1, "retry_adjacent_lower_temperature"),
        (row_index + 1, "retry_adjacent_higher_temperature"),
    ):
        if neighbor_row not in row_results:
            continue
        endpoint, _ = adjacent_row_endpoint_guess(
            metastability=metastability,
            adjacent_row=row_results[neighbor_row],
            default=None,
        )
        if endpoint is not None:
            return endpoint, source
    return _finite_endpoint(default), "retry_default"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path

    n0 = float(0.16 * const.MeV_fm**3)
    nB_crit0 = float(args.ncrit_over_n0 * n0)
    if np.isfinite(args.B14) and args.B14 > 0.0:
        B14 = float(args.B14)
        muB_crit0 = P_crit0 = np.nan
    else:
        print(
            f"Calibrating B^(1/4) from nB_crit(T=0) = {args.ncrit_over_n0:g} n0 ...",
            flush=True,
        )
        B14, muB_crit0, P_crit0 = calibrate_b14(
            nB_crit0, args.ms, args.nm_type
        )
        print(f"Calibrated B^(1/4) = {B14:.10g} MeV", flush=True)
    calibration = (B14, muB_crit0, P_crit0, n0, nB_crit0)

    temperatures = build_temperature_axis(args.t_min, args.t_max, args.t_count)
    shape = (args.t_count, args.m_count)
    arrays: dict[str, np.ndarray] = {
        "temperature_values": temperatures.copy(),
        "temperature_grid": np.repeat(temperatures[:, None], args.m_count, axis=1),
        "metastability_grid": np.full(shape, np.nan),
        "status_grid": np.full(shape, STATUS_UNSCANNED, dtype=int),
        "velocity_grid_m_s": np.full(shape, np.nan),
        "u_N_grid": np.full(shape, np.nan),
        "jB_grid": np.full(shape, np.nan),
        "lambda_n_grid": np.full(shape, np.nan),
        "A_boundary_grid": np.full(shape, np.nan),
        "a_interface_grid": np.full(shape, np.nan),
        "a_interface_LTE_grid": np.full(shape, np.nan),
        "entropy_jump_grid": np.full(shape, np.nan),
        "nB_crit_T_grid": np.full(shape, np.nan),
        "muB_crit_T_grid": np.full(shape, np.nan),
        "gap_boundary_metastability_values": np.full(args.t_count, np.nan),
        "gap_boundary_status": np.full(args.t_count, GAP_BOUNDARY_FAILURE, dtype=int),
        "gap_boundary_allowed_above": np.zeros(args.t_count, dtype=bool),
        "entropy_boundary_metastability_values": np.full(args.t_count, np.nan),
        "entropy_boundary_status": np.full(args.t_count, ENTROPY_INSUFFICIENT, dtype=object),
        "entropy_boundary_crossings": np.empty(args.t_count, dtype=object),
    }
    arrays["entropy_boundary_crossings"][:] = [list() for _ in range(args.t_count)]
    records: list[dict[str, Any]] = []
    gap_records: list[dict[str, Any]] = []

    def save() -> None:
        _atomic_save(
            output_path,
            build_payload(
                args=args,
                calibration=calibration,
                arrays=arrays,
                records=records,
                gap_boundary_records=gap_records,
            ),
        )

    print(f"Output: {output_path}", flush=True)
    print(f"Run tag: {args.run_tag}", flush=True)
    print(
        f"Analytical log grid: {args.t_count} temperatures x {args.m_count} adaptive metastabilities; "
        f"workers={args.workers}",
        flush=True,
    )

    gap_tasks = [
        GapBoundaryTask(
            row_index=i,
            T_N=float(temperature),
            B14=B14,
            n0=n0,
            ms=args.ms,
            nm_type=args.nm_type,
            upB=args.upB,
            m_lo=args.m_min,
            m_hi=args.m_max,
        )
        for i, temperature in enumerate(temperatures)
    ]
    gap_iterator = (
        map(solve_gap_boundary_at_temperature, gap_tasks)
        if args.workers == 1
        else _parallel_map_unordered(
            solve_gap_boundary_at_temperature, gap_tasks, args.workers
        )
    )
    for completed, boundary in enumerate(
        tqdm(gap_iterator, total=len(gap_tasks), desc="Delta_h=0 boundary", unit="T"),
        start=1,
    ):
        gap_records.append(boundary)
        i = int(boundary["row_index"])
        arrays["gap_boundary_metastability_values"][i] = boundary[
            "metastability_boundary"
        ]
        arrays["gap_boundary_status"][i] = int(boundary["status"])
        arrays["gap_boundary_allowed_above"][i] = bool(
            boundary["allowed_above_boundary"]
        )
        metastabilities = build_metastability_row(
            boundary=boundary["metastability_boundary"],
            boundary_status=int(boundary["status"]),
            allowed_above_boundary=bool(boundary["allowed_above_boundary"]),
            m_min=args.m_min,
            m_max=args.m_max,
            count=args.m_count,
            gap_buffer_fraction=args.gap_buffer_fraction,
        )
        if metastabilities is not None:
            arrays["metastability_grid"][i] = metastabilities
        if args.save_every and completed % args.save_every == 0:
            save()
    gap_records.sort(key=lambda record: record["row_index"])
    save()

    valid_rows = [
        i
        for i in temperature_row_order(temperatures, args.seed_temperature)
        if np.all(np.isfinite(arrays["metastability_grid"][i]))
    ]
    total = len(valid_rows) * args.m_count
    row_results: dict[int, dict[int, dict[str, Any]]] = {}
    failures: set[tuple[int, int]] = set()
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    since_save = 0

    def accept(result: dict[str, Any], first_pass: bool) -> None:
        nonlocal since_save
        i, j = int(result["row_index"]), int(result["col_index"])
        row = result["row"]
        row["block_row_index"] = i
        row["block_col_index"] = j
        records.append(row)
        row_results.setdefault(i, {})[j] = row
        latest[(i, j)] = result
        arrays["status_grid"][i, j] = (
            STATUS_FINITE if row.get("success") else STATUS_NUMERICAL_FAILURE
        )
        arrays["nB_crit_T_grid"][i, j] = _as_finite_float(
            row.get("nB_crit_T_MeV3")
        )
        arrays["muB_crit_T_grid"][i, j] = _as_finite_float(
            row.get("muB_crit_T_MeV")
        )
        if row.get("success"):
            failures.discard((i, j))
            arrays["velocity_grid_m_s"][i, j] = _as_finite_float(row.get("u_N_m_s"))
            arrays["u_N_grid"][i, j] = _as_finite_float(row.get("u_N"))
            arrays["jB_grid"][i, j] = _as_finite_float(row.get("jB"))
            arrays["lambda_n_grid"][i, j] = _as_finite_float(row.get("lambda_n"))
            arrays["A_boundary_grid"][i, j] = _as_finite_float(row.get("A_boundary"))
            arrays["a_interface_grid"][i, j] = _as_finite_float(row.get("a_interface"))
            arrays["a_interface_LTE_grid"][i, j] = _as_finite_float(
                row.get("a_interface_LTE")
            )
            arrays["entropy_jump_grid"][i, j] = entropy_jump_per_baryon(row)
        elif first_pass:
            failures.add((i, j))
        since_save += 1
        if args.save_every and since_save >= args.save_every:
            save()
            since_save = 0

    progress = tqdm(
        total=total,
        desc="Analytical log-T/log-m contour",
        unit="pt",
        dynamic_ncols=True,
    )
    try:
        if valid_rows:
            seed_row = valid_rows[0]
            direction_by_row = {seed_row: "seed"}
            for row_index in valid_rows[1:]:
                direction_by_row[row_index] = "up" if row_index > seed_row else "down"

            for row_index in valid_rows:
                metastabilities = arrays["metastability_grid"][row_index]
                direction = direction_by_row[row_index]
                if row_index == seed_row:
                    previous: dict[str, Any] | None = None
                    for col_index in range(args.m_count - 1, -1, -1):
                        endpoint = None
                        source = "default"
                        if previous is not None and previous.get("success"):
                            endpoint = _finite_endpoint(previous.get("endpoint_solution"))
                            if endpoint is not None:
                                source = "same_row_higher_m_neighbor"
                        if endpoint is None:
                            endpoint = _finite_endpoint(args.endpoint_guess)
                            if previous is not None:
                                source = "fallback_after_failed_higher_m_neighbor"
                        work = _make_work(
                            args=args,
                            row_index=row_index,
                            col_index=col_index,
                            temperature=temperatures[row_index],
                            metastability=metastabilities[col_index],
                            n0=n0,
                            nB_crit0=nB_crit0,
                            B14=B14,
                            endpoint=endpoint,
                            source=source,
                            direction=direction,
                        )
                        result = calculate_velocity_point(work)
                        accept(result, first_pass=True)
                        previous = result["row"]
                        progress.set_postfix_str(
                            f"m={result['metastability']:.3g}, T={result['T_MeV']:.3g}",
                            refresh=False,
                        )
                        progress.update(1)
                else:
                    adjacent_index = row_index - 1 if direction == "up" else row_index + 1
                    adjacent = row_results.get(adjacent_index, {})
                    works = []
                    for col_index in range(args.m_count - 1, -1, -1):
                        endpoint, source = adjacent_row_endpoint_guess(
                            metastability=float(metastabilities[col_index]),
                            adjacent_row=adjacent,
                            default=args.endpoint_guess,
                        )
                        works.append(
                            _make_work(
                                args=args,
                                row_index=row_index,
                                col_index=col_index,
                                temperature=temperatures[row_index],
                                metastability=metastabilities[col_index],
                                n0=n0,
                                nB_crit0=nB_crit0,
                                B14=B14,
                                endpoint=endpoint,
                                source=source,
                                direction=direction,
                            )
                        )
                    iterator = (
                        map(calculate_velocity_point, works)
                        if args.workers == 1
                        else _parallel_map_unordered(
                            calculate_velocity_point, works, args.workers
                        )
                    )
                    for result in iterator:
                        accept(result, first_pass=True)
                        progress.set_postfix_str(
                            f"m={result['metastability']:.3g}, T={result['T_MeV']:.3g}",
                            refresh=False,
                        )
                        progress.update(1)

        retries: list[PointWork] = []
        if args.neighbor_retry:
            for row_index, col_index in sorted(failures):
                metastability = float(arrays["metastability_grid"][row_index, col_index])
                endpoint, source = _retry_endpoint(
                    row_index,
                    col_index,
                    metastability,
                    row_results,
                    args.endpoint_guess,
                )
                if endpoint is None:
                    continue
                retries.append(
                    _make_work(
                        args=args,
                        row_index=row_index,
                        col_index=col_index,
                        temperature=temperatures[row_index],
                        metastability=metastability,
                        n0=n0,
                        nB_crit0=nB_crit0,
                        B14=B14,
                        endpoint=endpoint,
                        source=source,
                        direction="retry",
                        retry_attempt=1,
                        retry_of=(row_index, col_index),
                    )
                )
        if retries:
            progress.total = int(progress.total or 0) + len(retries)
            progress.refresh()
            retry_iterator = (
                map(calculate_velocity_point, retries)
                if args.workers == 1
                else _parallel_map_unordered(
                    calculate_velocity_point, retries, args.workers
                )
            )
            for result in retry_iterator:
                result["row"]["retry_replaced_failure"] = bool(
                    result["row"].get("success")
                )
                accept(result, first_pass=False)
                progress.update(1)
    finally:
        progress.close()

    for row_index in range(args.t_count):
        boundary, status, crossings = entropy_boundary_for_row(
            arrays["metastability_grid"][row_index],
            arrays["entropy_jump_grid"][row_index],
        )
        arrays["entropy_boundary_metastability_values"][row_index] = boundary
        arrays["entropy_boundary_status"][row_index] = status
        arrays["entropy_boundary_crossings"][row_index] = crossings

    save()
    successes = sum(bool(record.get("success")) for record in records)
    finite_A = arrays["A_boundary_grid"][np.isfinite(arrays["A_boundary_grid"])]
    print("Done.", flush=True)
    print(
        f"Saved rows: {len(records)}; success={successes}; failure={len(records) - successes}",
        flush=True,
    )
    if finite_A.size and np.max(finite_A) >= 1.0:
        print(
            f"WARNING: A diagnostic reached {np.max(finite_A):.6g}; inspect the assumed low-T domain.",
            flush=True,
        )


if __name__ == "__main__":
    main()
