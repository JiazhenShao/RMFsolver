#!/usr/bin/env python3
"""Parallel numerical uNmax scan on a hidden elliptical-ray grid.

The physical mesh is constructed between the Delta h = 0 and A = 1
boundaries using the same hidden coordinate system as the 26-07-24
analytical run. Numerical continuation starts on the analytical
u(0-) = 500 m/s contour and propagates toward both radial boundaries.
All saved primary coordinate arrays remain physical nB(0-)/n0 and T.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import RMFsolver.RMFparameter as para
import RMFsolver.constants as const
from RMFsolver import phase_velocity
from _analytic_ray_grid import (
    STATUS_FINITE,
    STATUS_NUMERICAL_FAILURE,
    STATUS_UNSCANNED,
    RayTransform,
    atomic_save,
    build_payload,
    build_safe_ray_mesh,
    calibrate_b14,
    chebyshev_open_axis,
    solve_boundary_curves,
)


# Tunable run parameters
DEFAULT_OUTPUT = "uNmax-elliptical-ray-grid.npy"
DEFAULT_RUN_TAG = "numerical_uNmax_elliptical_ray_grid_TQstar0_v1"
DEFAULT_ANALYTIC_SEED_PAYLOAD = (
    "../26-07-24-2/analytic-uN-elliptical-ray-grid.npy"
)
DEFAULT_SEED_VELOCITY_M_S = 500.0
DEFAULT_DENSITY_MIN_OVER_N0 = 1.0
DEFAULT_DENSITY_MAX_OVER_N0 = 5.0
DEFAULT_TEMPERATURE_MIN_MEV = 1.0e-2
DEFAULT_TEMPERATURE_MAX_MEV = 100.0
DEFAULT_THETA_COUNT = 30
DEFAULT_RADIAL_COUNT = 20
DEFAULT_ANGULAR_BUFFER_FRACTION = 1.0e-4
DEFAULT_BOUNDARY_BUFFER_FRACTION = 1.0e-3
DEFAULT_ENDPOINT_PROBE_COUNT = 25
DEFAULT_RAY_PROBE_COUNT = 15
DEFAULT_BOUNDARY_BISECT_STEPS = 10
DEFAULT_NEIGHBOR_RETRY = True
DEFAULT_SAVE_EVERY = 10
DEFAULT_INTERFACE_FRACTION_MODE = "extreme_endothermic"
DEFAULT_NCRIT_OVER_N0 = 3.0
DEFAULT_XI = 0.0
DEFAULT_MS_MEV = 0.0
DEFAULT_NM_TYPE = "PNM"
DEFAULT_UPB = 5000
DEFAULT_TQSTAR_MEV = 0.0
DEFAULT_TAIL_EPS = 1.0e-8
DEFAULT_N_MESH = 200
DEFAULT_TOL_BVP = 1.0e-4
DEFAULT_MAX_NODES = 10000
DEFAULT_TQ_GUESS = np.nan
DEFAULT_JB_BOUNDS = None
DEFAULT_AUTO_JB_BOUNDS = True
DEFAULT_SHOOTING_U_MAX = 1.0e-4
DEFAULT_SHOOTING_JB_LOWER = 1.0e-12
DEFAULT_POINT_TIMEOUT_S = 1800.0

NUMERICAL_CLOSURE_LABEL = "uNmax_direct_TQstar_nK"
NUMERICAL_ROOT_METHOD = "solve_bvp_uNmax_nK_parameter_1d"
DENSITY_RATIO_DEFINITION = "lambda_n_equals_nB_N_over_nB_Q"
CURRENT_DEFINITION = "u_nK_minus_D_dnK_dx"
RATE_MODEL = "exact_nonleptonic"
ENTROPY_JUMP_DEFINITION = "sQ_over_nB_Q_minus_sN_over_nB_N"
STATUS_TIMEOUT = 6


@dataclass(frozen=True)
class NumericalRayPointTask:
    row_index: int
    col_index: int
    theta_rad: float
    rho: float
    radial_fraction: float
    nB_over_n0: float
    T_N: float
    n0: float
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
    retry_attempt: int = 0
    retry_of: tuple[int, int] | None = None


class PointTimeoutError(TimeoutError):
    """Raised when a numerical BVP exceeds its configured wall time."""


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
    fields = [field.strip() for field in str(value).split(",")]
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("expected LOWER,UPPER")
    pair = (float(fields[0]), float(fields[1]))
    if not np.all(np.isfinite(pair)) or not 0.0 < pair[0] < pair[1]:
        raise argparse.ArgumentTypeError(
            "bounds must satisfy 0 < LOWER < UPPER"
        )
    return pair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-tag",
        default=os.getenv("RUN_TAG", DEFAULT_RUN_TAG),
    )
    parser.add_argument(
        "--analytic-seed-payload",
        default=DEFAULT_ANALYTIC_SEED_PAYLOAD,
        help="26-07-24-2 analytical payload used for pointwise jB seeds",
    )
    parser.add_argument(
        "--seed-velocity",
        type=float,
        default=DEFAULT_SEED_VELOCITY_M_S,
        help="analytical velocity contour used as the numerical seed spine",
    )
    parser.add_argument(
        "--density-min-over-n0",
        type=float,
        default=DEFAULT_DENSITY_MIN_OVER_N0,
    )
    parser.add_argument(
        "--density-max-over-n0",
        type=float,
        default=DEFAULT_DENSITY_MAX_OVER_N0,
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=DEFAULT_TEMPERATURE_MIN_MEV,
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=DEFAULT_TEMPERATURE_MAX_MEV,
    )
    parser.add_argument(
        "--theta-count",
        type=int,
        default=DEFAULT_THETA_COUNT,
    )
    parser.add_argument(
        "--radial-count",
        type=int,
        default=DEFAULT_RADIAL_COUNT,
    )
    parser.add_argument(
        "--angular-buffer-fraction",
        type=float,
        default=DEFAULT_ANGULAR_BUFFER_FRACTION,
    )
    parser.add_argument(
        "--boundary-buffer-fraction",
        type=float,
        default=DEFAULT_BOUNDARY_BUFFER_FRACTION,
    )
    parser.add_argument(
        "--endpoint-probe-count",
        type=int,
        default=DEFAULT_ENDPOINT_PROBE_COUNT,
    )
    parser.add_argument(
        "--ray-probe-count",
        type=int,
        default=DEFAULT_RAY_PROBE_COUNT,
    )
    parser.add_argument(
        "--boundary-bisect-steps",
        type=int,
        default=DEFAULT_BOUNDARY_BISECT_STEPS,
    )
    parser.add_argument(
        "--no-neighbor-retry",
        action="store_false",
        dest="neighbor_retry",
        default=DEFAULT_NEIGHBOR_RETRY,
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
    )
    parser.add_argument(
        "--interface-fraction-mode",
        choices=("extreme_endothermic",),
        default=DEFAULT_INTERFACE_FRACTION_MODE,
        help="mode used only to locate the analytical A=1 mesh boundary",
    )
    parser.add_argument(
        "--ncrit-over-n0",
        type=float,
        default=DEFAULT_NCRIT_OVER_N0,
    )
    parser.add_argument("--xi", type=float, default=DEFAULT_XI)
    parser.add_argument("--ms", type=float, default=DEFAULT_MS_MEV)
    parser.add_argument("--nm-type", default=DEFAULT_NM_TYPE)
    parser.add_argument("--upB", type=int, default=DEFAULT_UPB)
    parser.add_argument("--B14", "--b14", dest="B14", type=float, default=np.nan)
    parser.add_argument("--TQstar", type=float, default=DEFAULT_TQSTAR_MEV)
    parser.add_argument("--tail-eps", type=float, default=DEFAULT_TAIL_EPS)
    parser.add_argument("--n-mesh", type=int, default=DEFAULT_N_MESH)
    parser.add_argument("--tol-bvp", type=float, default=DEFAULT_TOL_BVP)
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument("--TQ-guess", type=float, default=DEFAULT_TQ_GUESS)
    parser.add_argument(
        "--jB-bounds",
        type=_parse_float_pair,
        default=DEFAULT_JB_BOUNDS,
    )
    parser.add_argument(
        "--no-auto-jB-bounds",
        action="store_false",
        dest="auto_jB_bounds",
        default=DEFAULT_AUTO_JB_BOUNDS,
    )
    parser.add_argument(
        "--shooting-u-max",
        type=float,
        default=DEFAULT_SHOOTING_U_MAX,
    )
    parser.add_argument(
        "--shooting-jB-lower",
        type=float,
        default=DEFAULT_SHOOTING_JB_LOWER,
    )
    parser.add_argument(
        "--point-timeout",
        type=float,
        default=DEFAULT_POINT_TIMEOUT_S,
    )
    parser.add_argument("--workers", type=int, default=_default_workers())
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.density_min_over_n0 < args.density_max_over_n0:
        raise ValueError(
            "require 0 < density-min-over-n0 < density-max-over-n0"
        )
    if not 0.0 < args.t_min < args.t_max:
        raise ValueError("require 0 < t-min < t-max")
    if args.theta_count < 3 or args.radial_count < 2:
        raise ValueError("theta-count >= 3 and radial-count >= 2 are required")
    if not 0.0 < args.angular_buffer_fraction < 0.1:
        raise ValueError(
            "angular-buffer-fraction must satisfy 0 < value < 0.1"
        )
    if not 0.0 < args.boundary_buffer_fraction < 0.5:
        raise ValueError(
            "boundary-buffer-fraction must satisfy 0 < value < 0.5"
        )
    if args.endpoint_probe_count < 5 or args.ray_probe_count < 5:
        raise ValueError("boundary probe counts must be at least 5")
    if args.boundary_bisect_steps < 1:
        raise ValueError("boundary-bisect-steps must be positive")
    if args.seed_velocity <= 0.0:
        raise ValueError("seed-velocity must be positive")
    if args.TQstar < 0.0:
        raise ValueError("TQstar must be nonnegative")
    if args.tail_eps <= 0.0 or args.n_mesh < 10:
        raise ValueError("tail-eps must be positive and n-mesh >= 10")
    if args.tol_bvp <= 0.0 or args.max_nodes < args.n_mesh:
        raise ValueError(
            "tol-bvp must be positive and max-nodes >= n-mesh"
        )
    if args.shooting_u_max <= 0.0 or args.shooting_jB_lower <= 0.0:
        raise ValueError("automatic shooting bounds must be positive")
    if args.point_timeout <= 0.0:
        raise ValueError("point-timeout must be positive")
    if args.save_every < 0 or args.workers < 1:
        raise ValueError(
            "save-every must be nonnegative and workers must be positive"
        )


def _output_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


def _seed_payload_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def _crossing_on_ray(
    rho: np.ndarray,
    velocity: np.ndarray,
    jB: np.ndarray,
    target_velocity_m_s: float,
) -> tuple[float, float]:
    rho = np.asarray(rho, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    jB = np.asarray(jB, dtype=float)
    finite = np.isfinite(rho) & np.isfinite(velocity) & np.isfinite(jB)
    rho = rho[finite]
    velocity = velocity[finite]
    jB = jB[finite]
    if rho.size < 2:
        raise ValueError("analytical seed ray has fewer than two finite points")
    order = np.argsort(rho)
    rho = rho[order]
    velocity = velocity[order]
    jB = jB[order]
    exact = np.flatnonzero(velocity == float(target_velocity_m_s))
    if exact.size == 1:
        index = int(exact[0])
        return float(rho[index]), float(jB[index])
    if exact.size > 1:
        raise ValueError(
            f"analytical {target_velocity_m_s:g} m/s crossing is not unique"
        )
    brackets = [
        index
        for index in range(rho.size - 1)
        if (
            (velocity[index] - target_velocity_m_s)
            * (velocity[index + 1] - target_velocity_m_s)
            < 0.0
        )
    ]
    if len(brackets) != 1:
        raise ValueError(
            f"expected one analytical {target_velocity_m_s:g} m/s crossing, "
            f"found {len(brackets)}"
        )
    index = brackets[0]
    delta = velocity[index + 1] - velocity[index]
    fraction = (
        0.0
        if delta == 0.0
        else (target_velocity_m_s - velocity[index]) / delta
    )
    return (
        float(rho[index] + fraction * (rho[index + 1] - rho[index])),
        float(jB[index] + fraction * (jB[index + 1] - jB[index])),
    )


def extract_seed_spine(
    analytic_payload: dict[str, Any],
    *,
    theta_axis_rad: np.ndarray,
    rho_grid: np.ndarray,
    target_velocity_m_s: float,
) -> dict[str, np.ndarray]:
    source_theta = np.asarray(
        analytic_payload["theta_axis_rad"],
        dtype=float,
    )
    source_rho = np.asarray(analytic_payload["rho_grid"], dtype=float)
    source_velocity = np.asarray(
        analytic_payload["velocity_grid_m_s"],
        dtype=float,
    )
    source_jB = np.asarray(analytic_payload["jB_grid"], dtype=float)
    if not (
        source_rho.shape
        == source_velocity.shape
        == source_jB.shape
        == (source_theta.size, source_rho.shape[1])
    ):
        raise ValueError("analytical seed payload has incompatible grid shapes")
    order = np.argsort(source_theta)
    source_theta = source_theta[order]
    source_rho = source_rho[order]
    source_velocity = source_velocity[order]
    source_jB = source_jB[order]
    crossings = [
        _crossing_on_ray(
            source_rho[index],
            source_velocity[index],
            source_jB[index],
            float(target_velocity_m_s),
        )
        for index in range(source_theta.size)
    ]
    crossing_rho = np.asarray([item[0] for item in crossings], dtype=float)
    crossing_jB = np.asarray([item[1] for item in crossings], dtype=float)
    target_theta = np.asarray(theta_axis_rad, dtype=float)
    target_grid = np.asarray(rho_grid, dtype=float)
    if target_grid.shape[0] != target_theta.size:
        raise ValueError("target rho grid must have one row per theta")
    if (
        np.min(target_theta) < np.min(source_theta) - 1.0e-12
        or np.max(target_theta) > np.max(source_theta) + 1.0e-12
    ):
        raise ValueError("target theta axis lies outside analytical seed axis")
    seed_rho = np.interp(target_theta, source_theta, crossing_rho)
    seed_jB = np.interp(target_theta, source_theta, crossing_jB)
    seed_indices = np.asarray(
        [
            int(np.nanargmin(np.abs(target_grid[index] - seed_rho[index])))
            for index in range(target_theta.size)
        ],
        dtype=int,
    )
    return {
        "seed_rho_by_theta": seed_rho,
        "seed_radial_index_by_theta": seed_indices,
        "seed_analytic_jB_by_theta": seed_jB,
    }


def interpolate_analytic_jB_grid(
    analytic_payload: dict[str, Any],
    *,
    theta_axis_rad: np.ndarray,
    rho_grid: np.ndarray,
) -> np.ndarray:
    source_theta = np.asarray(
        analytic_payload["theta_axis_rad"],
        dtype=float,
    )
    source_rho = np.asarray(analytic_payload["rho_grid"], dtype=float)
    source_jB = np.asarray(analytic_payload["jB_grid"], dtype=float)
    order = np.argsort(source_theta)
    source_theta = source_theta[order]
    source_rho = source_rho[order]
    source_jB = source_jB[order]
    target_theta = np.asarray(theta_axis_rad, dtype=float)
    target_rho = np.asarray(rho_grid, dtype=float)
    output = np.full(target_rho.shape, np.nan)
    for row_index, theta in enumerate(target_theta):
        for col_index, rho in enumerate(target_rho[row_index]):
            ray_values = np.asarray(
                [
                    np.interp(
                        rho,
                        source_rho[source_index],
                        source_jB[source_index],
                    )
                    for source_index in range(source_theta.size)
                ],
                dtype=float,
            )
            output[row_index, col_index] = np.interp(
                theta,
                source_theta,
                ray_values,
            )
    if not np.all(np.isfinite(output)) or np.any(output <= 0.0):
        raise ValueError("analytical jB fallback grid is not positive and finite")
    return output


def validate_analytic_seed_payload(
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
    calibration: dict[str, float],
) -> None:
    required = {
        "theta_axis_rad",
        "rho_grid",
        "velocity_grid_m_s",
        "jB_grid",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            f"analytical seed payload is missing fields: {', '.join(missing)}"
        )
    source_arguments = payload.get("arguments", {})
    comparisons = (
        ("density_min_over_n0", args.density_min_over_n0),
        ("density_max_over_n0", args.density_max_over_n0),
        ("t_min", args.t_min),
        ("t_max", args.t_max),
    )
    for name, expected in comparisons:
        value = _as_finite_float(source_arguments.get(name))
        if np.isfinite(value) and not np.isclose(
            value,
            expected,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"analytical seed payload {name}={value:g} does not match "
                f"requested {expected:g}"
            )
    source_calibration = payload.get("calibration", {})
    source_B14 = _as_finite_float(
        source_calibration.get("B14_calibrated_MeV")
    )
    expected_B14 = float(calibration["B14_calibrated_MeV"])
    if np.isfinite(source_B14) and not np.isclose(
        source_B14,
        expected_B14,
        rtol=1.0e-8,
        atol=1.0e-8,
    ):
        raise ValueError(
            "analytical seed payload uses a different calibrated bag constant"
        )


@contextmanager
def point_timeout(seconds: float | None):
    if seconds is None or not np.isfinite(seconds) or seconds <= 0.0:
        yield
        return
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return
    old_handler = signal.getsignal(signal.SIGALRM)
    old_delay, old_interval = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def handle_timeout(signum, frame):
        del signum, frame
        raise PointTimeoutError(
            f"numerical point exceeded {float(seconds):g} s"
        )

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
        raise RuntimeError(
            "automatic jB bounds require "
            "0 < lower < nB_N*shooting_u_max"
        )
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
    if not np.all(np.isfinite(values)) or np.any(
        values[[0, 1, 3, 6]] <= 0.0
    ):
        return np.nan
    T_N, T_Q, h_N, nB_N, muB_N, h_Q, nB_Q, muB_Q = values
    return float(
        (h_Q / nB_Q - muB_Q) / T_Q
        - (h_N / nB_N - muB_N) / T_N
    )


def calculate_numerical_point(
    task: NumericalRayPointTask,
) -> dict[str, Any]:
    started = time.monotonic()
    nB_N = float(task.nB_over_n0) * float(task.n0)
    row = {
        **vars(task),
        "success": False,
        "message": "",
        "theta_deg": float(np.degrees(task.theta_rad)),
        "T_MeV": float(task.T_N),
        "nB_N_over_n0": float(task.nB_over_n0),
        "nB_N_MeV3": nB_N,
        "TQstar_target_MeV": float(task.TQstar),
        "u_N": np.nan,
        "u_N_m_s": np.nan,
        "jB": np.nan,
        "lambda_n": np.nan,
        "A_boundary": np.nan,
        "a_local_Qstar": np.nan,
        "a_local_end": np.nan,
        "entropy_jump_per_baryon": np.nan,
        "entropy_allowed": False,
        "point_timed_out": False,
        "point_timeout_s": float(task.point_timeout),
        "point_elapsed_s": np.nan,
        "jB_guess_used": _as_finite_float(task.jB_guess),
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
        u_N = _as_finite_float(
            result.get("u_N_max_candidate", result.get("u_N"))
        )
        muB_Q = _as_finite_float(result.get("muB_Q"))
        T_Q = _as_finite_float(result.get("T_Q"))
        A_boundary = _as_finite_float(result.get("A_boundary"))
        if (
            not np.isfinite(A_boundary)
            and np.isfinite(muB_Q)
            and muB_Q > 0.0
            and np.isfinite(T_Q)
        ):
            A_boundary = float(
                9.0 * np.pi / np.sqrt(2.0) * T_Q / muB_Q
            )
        a_local_Qstar = _as_finite_float(
            result.get("aQstar_derived", result.get("aQstar"))
        )
        row.update(
            {
                "success": bool(result.get("success", False)),
                "message": str(result.get("message", "")),
                "muB_N_MeV": _as_finite_float(result.get("muB_N")),
                "u_N": u_N,
                "u_N_m_s": (
                    float(u_N * const.m_per_sec)
                    if np.isfinite(u_N)
                    else np.nan
                ),
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
                "nB_Qstar_MeV3": _as_finite_float(
                    result.get("nB_Qstar")
                ),
                "nK_Qstar_MeV3": _as_finite_float(
                    result.get("nK_Qstar")
                ),
                "nK_end": _as_finite_float(result.get("nK_end")),
                "jK_end": _as_finite_float(result.get("jK_end")),
                "h_N_MeV4": _as_finite_float(result.get("h_N")),
                "h_Q_MeV4": _as_finite_float(result.get("h_Q")),
                "root_method": str(
                    result.get("_root_method", NUMERICAL_ROOT_METHOD)
                ),
                "solver_variant": str(
                    result.get(
                        "solver_variant",
                        NUMERICAL_CLOSURE_LABEL,
                    )
                ),
                "bvp_parameter_count": int(
                    np.asarray(result.get("bvp_parameters", [])).size
                ),
                "bvp_status": int(result.get("bvp_status", -1)),
                "bvp_message": str(result.get("bvp_message", "")),
                "bvp_niter": int(result.get("bvp_niter", -1)),
                "bvp_nodes": int(result.get("bvp_nodes", -1)),
                "bvp_ode_calls": int(result.get("bvp_ode_calls", -1)),
                "downstream_root_calls": int(
                    result.get("downstream_root_calls", -1)
                ),
                "local_root_calls": int(
                    result.get("local_root_calls", -1)
                ),
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


def _empty_result_arrays(
    mesh: dict[str, np.ndarray],
    n0: float,
) -> dict[str, np.ndarray]:
    shape = mesh["nB_grid_over_n0"].shape
    arrays = {name: np.asarray(value) for name, value in mesh.items()}
    arrays["nB_grid_MeV3"] = arrays["nB_grid_over_n0"] * float(n0)
    arrays.update(
        {
            "velocity_grid_m_s": np.full(shape, np.nan),
            "u_N_grid": np.full(shape, np.nan),
            "jB_grid": np.full(shape, np.nan),
            "lambda_n_grid": np.full(shape, np.nan),
            "A_boundary_grid": np.full(shape, np.nan),
            "a_local_Qstar_grid": np.full(shape, np.nan),
            "aQstar_grid": np.full(shape, np.nan),
            "entropy_jump_grid": np.full(shape, np.nan),
            "status_grid": np.full(
                shape,
                STATUS_UNSCANNED,
                dtype=np.int8,
            ),
            "point_elapsed_s_grid": np.full(shape, np.nan),
            "point_timed_out_grid": np.zeros(shape, dtype=bool),
        }
    )
    return arrays


def _grid_point_task(
    *,
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    row_index: int,
    col_index: int,
    direction: str,
    n0: float,
    B14: float,
    jB_guess: float | None,
    jB_guess_source: str,
    retry_attempt: int = 0,
    retry_of: tuple[int, int] | None = None,
) -> NumericalRayPointTask:
    return NumericalRayPointTask(
        row_index=int(row_index),
        col_index=int(col_index),
        theta_rad=float(
            arrays["theta_grid_rad"][row_index, col_index]
        ),
        rho=float(arrays["rho_grid"][row_index, col_index]),
        radial_fraction=float(
            arrays["radial_fraction_grid"][row_index, col_index]
        ),
        nB_over_n0=float(
            arrays["nB_grid_over_n0"][row_index, col_index]
        ),
        T_N=float(
            arrays["temperature_grid_MeV"][row_index, col_index]
        ),
        n0=float(n0),
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
        jB_guess_source=str(jB_guess_source),
        continuation_direction=str(direction),
        retry_attempt=int(retry_attempt),
        retry_of=retry_of,
    )


def _successful_jB(row: dict[str, Any] | None) -> float | None:
    if not row or not bool(row.get("success", False)):
        return None
    return _optional_finite_float(row.get("jB"))


def _build_seed_point_tasks(
    *,
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    n0: float,
    B14: float,
) -> list[NumericalRayPointTask]:
    return [
        _grid_point_task(
            args=args,
            arrays=arrays,
            row_index=row_index,
            col_index=int(
                arrays["seed_radial_index_by_theta"][row_index]
            ),
            direction="seed",
            n0=n0,
            B14=B14,
            jB_guess=float(
                arrays["seed_analytic_jB_by_theta"][row_index]
            ),
            jB_guess_source="analytic_500_seed",
        )
        for row_index in range(arrays["theta_axis_rad"].size)
    ]


def _continuation_successor_tasks(
    *,
    completed_task: dict[str, Any] | NumericalRayPointTask,
    completed_row: dict[str, Any],
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    n0: float,
    B14: float,
) -> list[NumericalRayPointTask]:
    task_data = (
        vars(completed_task)
        if isinstance(completed_task, NumericalRayPointTask)
        else completed_task
    )
    row_index = int(task_data["row_index"])
    col_index = int(task_data["col_index"])
    direction = str(task_data["continuation_direction"])
    column_count = arrays["rho_grid"].shape[1]
    if direction == "seed":
        candidates = (
            (col_index + 1, "rho_out"),
            (col_index - 1, "rho_in"),
        )
    elif direction == "rho_out":
        candidates = ((col_index + 1, "rho_out"),)
    elif direction == "rho_in":
        candidates = ((col_index - 1, "rho_in"),)
    else:
        return []
    numerical_guess = _successful_jB(completed_row)
    tasks = []
    for next_index, next_direction in candidates:
        if not 0 <= next_index < column_count:
            continue
        if numerical_guess is not None:
            guess = numerical_guess
            source = "immediate_same_ray_neighbor"
        else:
            guess = float(
                arrays["analytic_jB_guess_grid"][row_index, next_index]
            )
            source = "analytic_fallback_after_failed_immediate_neighbor"
        tasks.append(
            _grid_point_task(
                args=args,
                arrays=arrays,
                row_index=row_index,
                col_index=next_index,
                direction=next_direction,
                n0=n0,
                B14=B14,
                jB_guess=guess,
                jB_guess_source=source,
            )
        )
    return tasks


def _retry_task(
    *,
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    records_by_cell: dict[tuple[int, int], dict[str, Any]],
    row_index: int,
    col_index: int,
    n0: float,
    B14: float,
    seed_only: bool = False,
) -> NumericalRayPointTask | None:
    candidates = (
        (
            ((row_index - 1, col_index), "immediate_lower_theta_neighbor"),
            ((row_index + 1, col_index), "immediate_upper_theta_neighbor"),
        )
        if seed_only
        else (
            ((row_index, col_index - 1), "immediate_rho_in_neighbor"),
            ((row_index, col_index + 1), "immediate_rho_out_neighbor"),
            ((row_index - 1, col_index), "immediate_lower_theta_neighbor"),
            ((row_index + 1, col_index), "immediate_upper_theta_neighbor"),
        )
    )
    for key, source in candidates:
        neighbor_jB = _successful_jB(records_by_cell.get(key))
        if neighbor_jB is None:
            continue
        return _grid_point_task(
            args=args,
            arrays=arrays,
            row_index=row_index,
            col_index=col_index,
            direction="retry",
            n0=n0,
            B14=B14,
            jB_guess=neighbor_jB,
            jB_guess_source=f"retry_{source}",
            retry_attempt=1,
            retry_of=(int(row_index), int(col_index)),
        )
    return None


def _record_result(
    arrays: dict[str, np.ndarray],
    row: dict[str, Any],
) -> None:
    i = int(row["row_index"])
    j = int(row["col_index"])
    arrays["point_elapsed_s_grid"][i, j] = _as_finite_float(
        row.get("point_elapsed_s")
    )
    arrays["point_timed_out_grid"][i, j] = bool(
        row.get("point_timed_out", False)
    )
    if row.get("success"):
        arrays["status_grid"][i, j] = STATUS_FINITE
        arrays["velocity_grid_m_s"][i, j] = _as_finite_float(
            row.get("u_N_m_s")
        )
        arrays["u_N_grid"][i, j] = _as_finite_float(row.get("u_N"))
        arrays["jB_grid"][i, j] = _as_finite_float(row.get("jB"))
        arrays["lambda_n_grid"][i, j] = _as_finite_float(
            row.get("lambda_n")
        )
        arrays["A_boundary_grid"][i, j] = _as_finite_float(
            row.get("A_boundary")
        )
        arrays["a_local_Qstar_grid"][i, j] = _as_finite_float(
            row.get("a_local_Qstar")
        )
        arrays["aQstar_grid"][i, j] = arrays[
            "a_local_Qstar_grid"
        ][i, j]
        arrays["entropy_jump_grid"][i, j] = _as_finite_float(
            row.get("entropy_jump_per_baryon")
        )
    elif row.get("point_timed_out"):
        arrays["status_grid"][i, j] = STATUS_TIMEOUT
    else:
        arrays["status_grid"][i, j] = STATUS_NUMERICAL_FAILURE


def build_numerical_payload(
    *,
    args: argparse.Namespace,
    transform: RayTransform,
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    boundary_payload: dict[str, Any],
    calibration: dict[str, float],
    analytic_seed_path: Path,
) -> dict[str, Any]:
    payload = build_payload(
        args=args,
        transform=transform,
        arrays=arrays,
        records=records,
        boundary_payload=boundary_payload,
        calibration=calibration,
    )
    payload.pop("analytic_closure", None)
    payload.update(
        {
            "numerical_closure": NUMERICAL_CLOSURE_LABEL,
            "solver_variant": NUMERICAL_CLOSURE_LABEL,
            "root_method": NUMERICAL_ROOT_METHOD,
            "composition_definition": "nK_over_local_nB",
            "density_ratio_definition": DENSITY_RATIO_DEFINITION,
            "current_definition": CURRENT_DEFINITION,
            "rate_model": RATE_MODEL,
            "boundary_interface_fraction_mode": str(
                args.interface_fraction_mode
            ),
            "TQstar_target_MeV": float(args.TQstar),
            "seed_velocity_m_s": float(args.seed_velocity),
            "seed_strategy": (
                "analytic_velocity_spine_then_immediate_same_ray_"
                "bidirectional_jB_continuation"
            ),
            "analytic_seed_payload": str(analytic_seed_path),
            "entropy_jump_definition": ENTROPY_JUMP_DEFINITION,
            "entropy_affects_mesh": False,
            "status_codes": {
                "finite": STATUS_FINITE,
                "numerical_failure": STATUS_NUMERICAL_FAILURE,
                "unscanned": STATUS_UNSCANNED,
                "timeout": STATUS_TIMEOUT,
            },
        }
    )
    return payload


def resolve_calibration(
    args: argparse.Namespace,
) -> dict[str, float]:
    n0 = float(0.16 * const.MeV_fm**3)
    nB_crit0 = float(args.ncrit_over_n0 * n0)
    if np.isfinite(args.B14) and args.B14 > 0.0:
        B14 = float(args.B14)
        _, muB_crit0, pressure_crit0 = calibrate_b14(
            nB_crit0,
            args.ms,
            args.nm_type,
        )
    else:
        B14, muB_crit0, pressure_crit0 = calibrate_b14(
            nB_crit0,
            args.ms,
            args.nm_type,
        )
    return {
        "B14_calibrated_MeV": float(B14),
        "muB_crit0_MeV": float(muB_crit0),
        "P_crit0_MeV4": float(pressure_crit0),
        "n0_MeV3": float(n0),
        "nB_crit0_MeV3": float(nB_crit0),
        "nB_crit0_over_n0": float(args.ncrit_over_n0),
    }


def _run_unordered(
    function,
    tasks,
    executor: ProcessPoolExecutor | None,
):
    tasks = list(tasks)
    if executor is None:
        for task in tasks:
            yield task, function(task)
        return
    pending = {executor.submit(function, task): task for task in tasks}
    while pending:
        completed, _waiting = wait(
            tuple(pending),
            return_when=FIRST_COMPLETED,
        )
        for future in completed:
            task = pending.pop(future)
            yield task, future.result()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    output_path = _output_path(args.output).resolve()
    analytic_seed_path = _seed_payload_path(args.analytic_seed_payload)
    if not analytic_seed_path.exists():
        raise FileNotFoundError(
            f"analytical seed payload not found: {analytic_seed_path}"
        )
    analytic_seed_payload = np.load(
        analytic_seed_path,
        allow_pickle=True,
    ).item()
    calibration = resolve_calibration(args)
    validate_analytic_seed_payload(
        analytic_seed_payload,
        args=args,
        calibration=calibration,
    )
    B14 = float(calibration["B14_calibrated_MeV"])
    n0 = float(calibration["n0_MeV3"])
    transform = RayTransform(
        density_origin_over_n0=args.density_min_over_n0,
        density_scale_over_n0=(
            args.density_max_over_n0 - args.density_min_over_n0
        ),
        temperature_scale_MeV=args.t_max,
    )
    print(
        f"Calibrated B^(1/4) = {B14:.9f} MeV\n"
        f"Output: {output_path}\n"
        f"Run tag: {args.run_tag}\n"
        f"Hidden grid: {args.theta_count} theta rays x "
        f"{args.radial_count} safe radial fractions\n"
        f"Physical window: nB(0-)/n0={args.density_min_over_n0:g}-"
        f"{args.density_max_over_n0:g}, "
        f"T={args.t_min:g}-{args.t_max:g} MeV\n"
        f"Seed spine: analytical u={args.seed_velocity:g} m/s from "
        f"{analytic_seed_path}\n"
        f"TQstar={args.TQstar:g} MeV; workers={args.workers}; "
        f"point timeout={args.point_timeout:g} s"
    )
    executor = (
        None
        if args.workers == 1
        else ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("spawn"),
        )
    )
    try:
        boundary_payload = solve_boundary_curves(
            transform=transform,
            theta_count=args.theta_count,
            density_min_over_n0=args.density_min_over_n0,
            density_max_over_n0=args.density_max_over_n0,
            temperature_min_MeV=args.t_min,
            temperature_max_MeV=args.t_max,
            n0=n0,
            B14=B14,
            xi=args.xi,
            ms=args.ms,
            nm_type=args.nm_type,
            upB=args.upB,
            interface_fraction_mode=args.interface_fraction_mode,
            endpoint_probe_count=args.endpoint_probe_count,
            ray_probe_count=args.ray_probe_count,
            bisection_steps=args.boundary_bisect_steps,
            angular_buffer_fraction=args.angular_buffer_fraction,
            progress_factory=tqdm,
            executor=executor,
        )
    except BaseException:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        raise
    eta_axis = chebyshev_open_axis(
        args.radial_count,
        args.boundary_buffer_fraction,
    )
    mesh = build_safe_ray_mesh(
        transform=transform,
        theta_axis_rad=boundary_payload["theta_axis_rad"],
        radial_fraction_axis=eta_axis,
        rho_inner=boundary_payload["rho_inner"],
        rho_outer=boundary_payload["rho_outer"],
    )
    arrays = _empty_result_arrays(mesh, n0)
    seed_spine = extract_seed_spine(
        analytic_seed_payload,
        theta_axis_rad=arrays["theta_axis_rad"],
        rho_grid=arrays["rho_grid"],
        target_velocity_m_s=args.seed_velocity,
    )
    arrays.update(seed_spine)
    arrays["analytic_jB_guess_grid"] = interpolate_analytic_jB_grid(
        analytic_seed_payload,
        theta_axis_rad=arrays["theta_axis_rad"],
        rho_grid=arrays["rho_grid"],
    )
    records: list[dict[str, Any]] = []
    records_by_cell: dict[tuple[int, int], dict[str, Any]] = {}
    failures: set[tuple[int, int]] = set()
    completed_since_save = 0

    def save() -> None:
        atomic_save(
            output_path,
            build_numerical_payload(
                args=args,
                transform=transform,
                arrays=arrays,
                records=records,
                boundary_payload=boundary_payload,
                calibration=calibration,
                analytic_seed_path=analytic_seed_path,
            ),
        )

    def accept(row: dict[str, Any], first_pass: bool) -> None:
        nonlocal completed_since_save
        key = (int(row["row_index"]), int(row["col_index"]))
        records.append(row)
        records_by_cell[key] = row
        if row.get("success"):
            failures.discard(key)
        elif first_pass:
            failures.add(key)
        _record_result(arrays, row)
        completed_since_save += 1
        if args.save_every and completed_since_save >= args.save_every:
            save()
            completed_since_save = 0

    save()
    seed_tasks = _build_seed_point_tasks(
        args=args,
        arrays=arrays,
        n0=n0,
        B14=B14,
    )
    first_pass_total = args.theta_count * args.radial_count
    progress = tqdm(
        total=first_pass_total,
        desc="Numerical uNmax elliptical-ray contour",
        unit="pt",
        dynamic_ncols=True,
    )
    try:
        completed_seed_tasks: dict[int, NumericalRayPointTask] = {}
        for task, row in _run_unordered(
            calculate_numerical_point,
            seed_tasks,
            executor,
        ):
            accept(row, first_pass=True)
            completed_seed_tasks[int(task.row_index)] = task
            progress.update(1)
            progress.set_postfix_str(
                f"seed theta={np.degrees(task.theta_rad):.3g} deg, "
                f"rho={task.rho:.4g}"
            )

        seed_retry_tasks = []
        for row_index, task in completed_seed_tasks.items():
            key = (row_index, int(task.col_index))
            if records_by_cell[key].get("success"):
                continue
            retry = _retry_task(
                args=args,
                arrays=arrays,
                records_by_cell=records_by_cell,
                row_index=row_index,
                col_index=int(task.col_index),
                n0=n0,
                B14=B14,
                seed_only=True,
            )
            if retry is not None:
                seed_retry_tasks.append(retry)
        if seed_retry_tasks:
            progress.total += len(seed_retry_tasks)
            progress.refresh()
            for task, row in _run_unordered(
                calculate_numerical_point,
                seed_retry_tasks,
                executor,
            ):
                row["retry_replaced_failure"] = bool(row.get("success"))
                accept(row, first_pass=False)
                progress.update(1)
                progress.set_postfix_str(
                    f"seed retry theta={np.degrees(task.theta_rad):.3g} deg"
                )

        initial_successors = []
        for row_index, seed_task in completed_seed_tasks.items():
            seed_row = records_by_cell[
                (row_index, int(seed_task.col_index))
            ]
            initial_successors.extend(
                _continuation_successor_tasks(
                    completed_task=seed_task,
                    completed_row=seed_row,
                    args=args,
                    arrays=arrays,
                    n0=n0,
                    B14=B14,
                )
            )
        if executor is None:
            pending_tasks = list(initial_successors)
            while pending_tasks:
                task = pending_tasks.pop(0)
                row = calculate_numerical_point(task)
                accept(row, first_pass=True)
                pending_tasks.extend(
                    _continuation_successor_tasks(
                        completed_task=task,
                        completed_row=row,
                        args=args,
                        arrays=arrays,
                        n0=n0,
                        B14=B14,
                    )
                )
                progress.update(1)
                progress.set_postfix_str(
                    f"theta={np.degrees(task.theta_rad):.3g} deg, "
                    f"rho={task.rho:.4g}"
                )
        else:
            pending = {
                executor.submit(calculate_numerical_point, task): task
                for task in initial_successors
            }
            while pending:
                completed, _waiting = wait(
                    tuple(pending),
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    task = pending.pop(future)
                    row = future.result()
                    accept(row, first_pass=True)
                    for successor in _continuation_successor_tasks(
                        completed_task=task,
                        completed_row=row,
                        args=args,
                        arrays=arrays,
                        n0=n0,
                        B14=B14,
                    ):
                        pending[
                            executor.submit(
                                calculate_numerical_point,
                                successor,
                            )
                        ] = successor
                    progress.update(1)
                    progress.set_postfix_str(
                        f"theta={np.degrees(task.theta_rad):.3g} deg, "
                        f"rho={task.rho:.4g}"
                    )

        if args.neighbor_retry and failures:
            retry_tasks = []
            for row_index, col_index in sorted(failures):
                retry = _retry_task(
                    args=args,
                    arrays=arrays,
                    records_by_cell=records_by_cell,
                    row_index=row_index,
                    col_index=col_index,
                    n0=n0,
                    B14=B14,
                )
                if retry is not None:
                    retry_tasks.append(retry)
            if retry_tasks:
                progress.total += len(retry_tasks)
                progress.refresh()
                for task, row in _run_unordered(
                    calculate_numerical_point,
                    retry_tasks,
                    executor,
                ):
                    row["retry_replaced_failure"] = bool(
                        row.get("success")
                    )
                    accept(row, first_pass=False)
                    progress.update(1)
                    progress.set_postfix_str(
                        f"retry theta={np.degrees(task.theta_rad):.3g} deg, "
                        f"rho={task.rho:.4g}"
                    )
    finally:
        progress.close()
        save()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    finite = int(np.count_nonzero(arrays["status_grid"] == STATUS_FINITE))
    failed = int(
        np.count_nonzero(
            arrays["status_grid"] == STATUS_NUMERICAL_FAILURE
        )
    )
    timed_out = int(
        np.count_nonzero(arrays["status_grid"] == STATUS_TIMEOUT)
    )
    print(
        f"Done. finite={finite}, failure={failed}, timeout={timed_out}, "
        f"saved records={len(records)}\nSaved {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
