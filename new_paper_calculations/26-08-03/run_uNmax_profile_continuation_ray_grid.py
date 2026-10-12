#!/usr/bin/env python3
"""Full-profile-continuation uNmax scan in the T versus nB(0-) plane.

The hidden ray transform and physical boundaries come from the state-of-the-
art 26-08-02 numerical-I2 analytical payload. The denser analytical mesh is
interpolated in the physical (nB(0-), T) plane onto every numerical cell.
Full numerical uNmax solves start at the actual cell nearest 20 m/s on each
ray. The seed profile is handed once to each radial half-ray worker, which then
keeps the complete successful nK/jK profile in-process and propagates it only
to the immediately adjacent point. A biased analytical seed is used only when
no adjacent numerical profile exists or the full-profile attempt fails.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
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
    chebyshev_closed_axis,
    chebyshev_open_axis,
)


# Tunable run parameters
DEFAULT_OUTPUT = "uNmax-T-nB-ray-grid-full-profile-continuation.npy"
DEFAULT_RUN_TAG = "numerical_uNmax_T_nB_full_profile_continuation_20260803_v2"
DEFAULT_ANALYTIC_SEED_PAYLOAD = (
    "../26-08-02/analytic-uN-T-nB-ray-grid-numerical-I2.npy"
)
DEFAULT_SEED_VELOCITY_M_S = 20.0
DEFAULT_SAFETY_VELOCITY_M_S = 1700.0
DEFAULT_DENSITY_MIN_OVER_N0 = 1.0
DEFAULT_DENSITY_MAX_OVER_N0 = 5.5
DEFAULT_TEMPERATURE_MIN_MEV = 1.0e-2
DEFAULT_TEMPERATURE_MAX_MEV = 120.0
DEFAULT_THETA_COUNT = 30
DEFAULT_RADIAL_COUNT = 20
DEFAULT_BOUNDARY_BUFFER_FRACTION = 1.0e-3
DEFAULT_NEIGHBOR_RETRY = True
DEFAULT_SAVE_EVERY = 10
DEFAULT_INTERFACE_FRACTION_MODE = "extreme_endothermic"
DEFAULT_NCRIT_OVER_N0 = 3.0
DEFAULT_XI = 0.0
DEFAULT_MS_MEV = 0.0
DEFAULT_NM_TYPE = "PNM"
DEFAULT_UPB = 5000
DEFAULT_T0PLUS_MEV = 0.0
DEFAULT_TAIL_EPS = 1.0e-8
DEFAULT_N_MESH = 200
DEFAULT_TOL_BVP = 1.0e-4
DEFAULT_MAX_NODES = 10000
DEFAULT_T_INF_GUESS = np.nan
DEFAULT_JB_BOUNDS = None
DEFAULT_AUTO_JB_BOUNDS = True
DEFAULT_SHOOTING_U_MAX = 1.0e-4
DEFAULT_SHOOTING_JB_LOWER = 1.0e-12
DEFAULT_POINT_TIMEOUT_S = 1800.0
DEFAULT_CONTINUATION_PROFILE_POINTS = 200

NUMERICAL_CLOSURE_LABEL = "u0minus_max_direct_T0plus_nK"
NUMERICAL_ROOT_METHOD = "solve_bvp_u0minus_max_nK_parameter_1d"
DENSITY_RATIO_DEFINITION = "lambda_n_equals_nB_0minus_over_nB_inf"
CURRENT_DEFINITION = "u_nK_minus_D_dnK_dx"
RATE_MODEL = "exact_nonleptonic"
ENTROPY_JUMP_DEFINITION = "s_inf_over_nB_inf_minus_s_0minus_over_nB_0minus"
STATUS_TIMEOUT = 6


@dataclass(frozen=True)
class NumericalRayPointTask:
    row_index: int
    col_index: int
    theta_rad: float
    rho: float
    radial_fraction: float
    nB_over_n0: float
    T_0minus: float
    n0: float
    B14: float
    ms: float
    nm_type: str
    upB: int
    T_0plus: float
    tail_eps: float
    n_mesh: int
    tol_bvp: float
    max_nodes: int
    jB_guess: float | None
    T_inf_guess: float | None
    jB_bounds: tuple[float, float] | None
    auto_jB_bounds: bool
    shooting_u_max: float
    shooting_jB_lower: float
    point_timeout: float
    continuation_profile_points: int
    jB_guess_source: str
    continuation_direction: str
    analytic_jB_guess: float | None = None
    propagated_jB_guess: float | None = None
    attempt_kind: str = "primary"
    retry_attempt: int = 0
    retry_of: tuple[int, int] | None = None


@dataclass(frozen=True)
class RayDirectionJob:
    """One sequential radial half-ray and its seed continuation profile."""

    row_index: int
    direction: str
    tasks: tuple[Any, ...]
    initial_continuation: dict[str, Any] | None


@dataclass(frozen=True)
class ProfileRetryJob:
    task: NumericalRayPointTask
    continuation: dict[str, Any]
    attempt_kind: str
    guess_source: str


class PointTimeoutError(BaseException):
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
        help=(
            "26-08-02 numerical-I2 analytical payload supplying the ray "
            "geometry, boundaries, safety contour, and physical-2D-"
            "interpolated jB seeds"
        ),
    )
    parser.add_argument(
        "--seed-velocity",
        type=float,
        default=DEFAULT_SEED_VELOCITY_M_S,
        help=(
            "target numerical-I2 velocity used to select the nearest actual "
            "uNmax seed cell on each ray"
        ),
    )
    parser.add_argument(
        "--safety-velocity",
        type=float,
        default=DEFAULT_SAFETY_VELOCITY_M_S,
        help=(
            "analytical velocity contour used as the numerical outer "
            "boundary before A=1"
        ),
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
        "--boundary-buffer-fraction",
        type=float,
        default=DEFAULT_BOUNDARY_BUFFER_FRACTION,
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
        help="interface mode required to match the analytical seed payload",
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
    parser.add_argument("--T_0plus", type=float, default=DEFAULT_T0PLUS_MEV)
    parser.add_argument("--tail-eps", type=float, default=DEFAULT_TAIL_EPS)
    parser.add_argument("--n-mesh", type=int, default=DEFAULT_N_MESH)
    parser.add_argument("--tol-bvp", type=float, default=DEFAULT_TOL_BVP)
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument("--T-inf-guess", type=float, default=DEFAULT_T_INF_GUESS)
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
    parser.add_argument(
        "--continuation-profile-points",
        type=int,
        default=DEFAULT_CONTINUATION_PROFILE_POINTS,
        help=(
            "maximum number of nK/jK profile points retained in each "
            "in-worker continuation bundle"
        ),
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
    if not 0.0 < args.boundary_buffer_fraction < 0.5:
        raise ValueError(
            "boundary-buffer-fraction must satisfy 0 < value < 0.5"
        )
    if args.seed_velocity <= 0.0:
        raise ValueError("seed-velocity must be positive")
    if args.safety_velocity <= args.seed_velocity:
        raise ValueError(
            "safety-velocity must be greater than seed-velocity"
        )
    if args.T_0plus < 0.0:
        raise ValueError("T_0plus must be nonnegative")
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
    if args.continuation_profile_points < 5:
        raise ValueError("continuation-profile-points must be at least 5")
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
    contour = extract_analytic_velocity_contour(
        analytic_payload,
        theta_axis_rad=theta_axis_rad,
        target_velocity_m_s=target_velocity_m_s,
    )
    target_theta = np.asarray(theta_axis_rad, dtype=float)
    target_grid = np.asarray(rho_grid, dtype=float)
    if target_grid.shape[0] != target_theta.size:
        raise ValueError("target rho grid must have one row per theta")
    seed_rho = contour["rho_by_theta"]
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
        "seed_analytic_jB_by_theta": contour["jB_by_theta"],
    }


def extract_analytic_velocity_contour(
    analytic_payload: dict[str, Any],
    *,
    theta_axis_rad: np.ndarray,
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
    if (
        np.min(target_theta) < np.min(source_theta) - 1.0e-12
        or np.max(target_theta) > np.max(source_theta) + 1.0e-12
    ):
        raise ValueError("target theta axis lies outside analytical seed axis")
    return {
        "rho_by_theta": np.interp(
            target_theta,
            source_theta,
            crossing_rho,
        ),
        "jB_by_theta": np.interp(
            target_theta,
            source_theta,
            crossing_jB,
        ),
    }


def build_analytic_safety_boundary(
    analytic_payload: dict[str, Any],
    *,
    transform: RayTransform,
    theta_axis_rad: np.ndarray,
    rho_inner: np.ndarray,
    rho_A_boundary: np.ndarray,
    target_velocity_m_s: float,
) -> dict[str, np.ndarray | float]:
    theta = np.asarray(theta_axis_rad, dtype=float)
    inner = np.asarray(rho_inner, dtype=float)
    A_boundary = np.asarray(rho_A_boundary, dtype=float)
    if inner.shape != theta.shape or A_boundary.shape != theta.shape:
        raise ValueError("physical boundary arrays must match theta axis")
    contour = extract_analytic_velocity_contour(
        analytic_payload,
        theta_axis_rad=theta,
        target_velocity_m_s=target_velocity_m_s,
    )
    safety_rho = np.asarray(contour["rho_by_theta"], dtype=float)
    safety_jB = np.asarray(contour["jB_by_theta"], dtype=float)
    if not np.all(safety_rho > inner):
        bad = int(np.flatnonzero(safety_rho <= inner)[0])
        raise ValueError(
            "analytical safety contour is not outside the Delta h boundary "
            f"at theta={np.degrees(theta[bad]):.6g} deg"
        )
    if not np.all(safety_rho < A_boundary):
        bad = int(np.flatnonzero(safety_rho >= A_boundary)[0])
        raise ValueError(
            "analytical safety contour is not strictly inside the A=1 "
            f"boundary at theta={np.degrees(theta[bad]):.6g} deg"
        )
    density, temperature = transform.to_physical(safety_rho, theta)
    return {
        "velocity_m_s": float(target_velocity_m_s),
        "rho_by_theta": safety_rho,
        "jB_by_theta": safety_jB,
        "nB_over_n0": np.asarray(density, dtype=float),
        "temperature_MeV": np.asarray(temperature, dtype=float),
        "clearance_to_A_rho_by_theta": A_boundary - safety_rho,
    }


def _interpolate_analytic_field_hidden(
    analytic_payload: dict[str, Any],
    *,
    field_name: str,
    theta_axis_rad: np.ndarray,
    rho_grid: np.ndarray,
) -> np.ndarray:
    source_theta = np.asarray(
        analytic_payload["theta_axis_rad"],
        dtype=float,
    )
    source_rho = np.asarray(analytic_payload["rho_grid"], dtype=float)
    source_values = np.asarray(analytic_payload[field_name], dtype=float)
    order = np.argsort(source_theta)
    source_theta = source_theta[order]
    source_rho = source_rho[order]
    source_values = source_values[order]
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
                        source_values[source_index],
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
        raise ValueError(
            f"analytical {field_name} fallback grid is not positive and finite"
        )
    return output


def interpolate_analytic_jB_grid(
    analytic_payload: dict[str, Any],
    *,
    theta_axis_rad: np.ndarray,
    rho_grid: np.ndarray,
) -> np.ndarray:
    return _interpolate_analytic_field_hidden(
        analytic_payload,
        field_name="jB_grid",
        theta_axis_rad=theta_axis_rad,
        rho_grid=rho_grid,
    )


def interpolate_analytic_fields_2d(
    analytic_payload: dict[str, Any],
    *,
    nB_grid_over_n0: np.ndarray,
    temperature_grid_MeV: np.ndarray,
    theta_grid_rad: np.ndarray,
    rho_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate analytical velocity and jB at the numerical mesh points."""
    source_nB = np.asarray(
        analytic_payload["nB_grid_over_n0"],
        dtype=float,
    )
    source_temperature = np.asarray(
        analytic_payload["temperature_grid_MeV"],
        dtype=float,
    )
    target_nB = np.asarray(nB_grid_over_n0, dtype=float)
    target_temperature = np.asarray(temperature_grid_MeV, dtype=float)
    target_theta = np.asarray(theta_grid_rad, dtype=float)
    target_rho = np.asarray(rho_grid, dtype=float)
    if not (
        target_nB.shape
        == target_temperature.shape
        == target_theta.shape
        == target_rho.shape
    ):
        raise ValueError("target analytical interpolation grids must match")
    if source_nB.shape != source_temperature.shape:
        raise ValueError("source analytical physical-coordinate grids must match")
    source_points = np.column_stack(
        (source_nB.ravel(), source_temperature.ravel())
    )
    target_points = np.column_stack(
        (target_nB.ravel(), target_temperature.ravel())
    )
    output: dict[str, np.ndarray] = {}
    fallback_mask = np.zeros(target_nB.shape, dtype=bool)
    target_theta_axis = target_theta[:, 0]
    for field_name in ("jB_grid", "velocity_grid_m_s"):
        source_field = np.asarray(
            analytic_payload[field_name],
            dtype=float,
        )
        if source_field.shape != source_nB.shape:
            raise ValueError(
                f"source analytical {field_name} has an incompatible shape"
            )
        finite = (
            np.all(np.isfinite(source_points), axis=1)
            & np.isfinite(source_field.ravel())
        )
        interpolator = LinearNDInterpolator(
            source_points[finite],
            source_field.ravel()[finite],
            fill_value=np.nan,
            rescale=True,
        )
        values = np.asarray(
            interpolator(target_points),
            dtype=float,
        ).reshape(target_nB.shape)
        missing = ~np.isfinite(values)
        if np.any(missing):
            ray_values = _interpolate_analytic_field_hidden(
                analytic_payload,
                field_name=field_name,
                theta_axis_rad=target_theta_axis,
                rho_grid=target_rho,
            )
            values[missing] = ray_values[missing]
            fallback_mask |= missing
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(
                f"interpolated analytical {field_name} is not positive and finite"
            )
        output[field_name] = values
    output["fallback_mask"] = fallback_mask
    return output


def select_seed_cells_from_interpolated_grid(
    *,
    rho_grid: np.ndarray,
    velocity_grid_m_s: np.ndarray,
    jB_grid: np.ndarray,
    target_velocity_m_s: float,
) -> dict[str, np.ndarray]:
    """Select one actual numerical cell per ray near the target velocity."""
    rho = np.asarray(rho_grid, dtype=float)
    velocity = np.asarray(velocity_grid_m_s, dtype=float)
    jB = np.asarray(jB_grid, dtype=float)
    if rho.ndim != 2 or not (rho.shape == velocity.shape == jB.shape):
        raise ValueError("seed-selection grids must be matching 2D arrays")
    indices = np.argmin(
        np.abs(velocity - float(target_velocity_m_s)),
        axis=1,
    ).astype(int)
    rows = np.arange(rho.shape[0])
    return {
        "seed_rho_by_theta": rho[rows, indices],
        "seed_radial_index_by_theta": indices,
        "seed_analytic_jB_by_theta": jB[rows, indices],
        "seed_actual_velocity_m_s_by_theta": velocity[rows, indices],
    }


def validate_analytic_seed_payload(
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
    calibration: dict[str, float],
) -> None:
    required = {
        "theta_axis_rad",
        "rho_grid",
        "nB_grid_over_n0",
        "temperature_grid_MeV",
        "velocity_grid_m_s",
        "jB_grid",
        "rho_inner_boundary",
        "rho_outer_boundary",
        "transform",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            f"analytical seed payload is missing fields: {', '.join(missing)}"
        )
    if str(payload.get("velocity_closure", "")) != "numerical_I2":
        raise ValueError(
            "analytical seed payload must use velocity_closure=numerical_I2"
        )
    source_arguments = payload.get("arguments", {})
    source_window = {
        name: _as_finite_float(source_arguments.get(name))
        for name in (
            "density_min_over_n0",
            "density_max_over_n0",
            "t_min",
            "t_max",
        )
    }
    requested_window = {
        "density_min_over_n0": float(args.density_min_over_n0),
        "density_max_over_n0": float(args.density_max_over_n0),
        "t_min": float(args.t_min),
        "t_max": float(args.t_max),
    }
    coverage_ok = (
        source_window["density_min_over_n0"]
        <= requested_window["density_min_over_n0"]
        and source_window["density_max_over_n0"]
        >= requested_window["density_max_over_n0"]
        and source_window["t_min"] <= requested_window["t_min"]
        and source_window["t_max"] >= requested_window["t_max"]
    )
    if not coverage_ok:
        raise ValueError(
            "numerical-I2 analytical payload does not cover the requested "
            "physical window"
        )
    scalar_physics = (
        ("xi", float(args.xi)),
        ("ms", float(args.ms)),
        ("ncrit_over_n0", float(args.ncrit_over_n0)),
        ("upB", float(args.upB)),
    )
    for name, expected in scalar_physics:
        actual = _as_finite_float(source_arguments.get(name))
        if not np.isfinite(actual) or not np.isclose(
            actual,
            expected,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"analytical seed payload {name}={actual:g} does not match "
                f"requested {expected:g}"
            )
    if str(source_arguments.get("nm_type", "")) != str(args.nm_type):
        raise ValueError(
            "analytical seed payload uses a different nuclear-matter EoS"
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
    source_mode = payload.get("interface_fraction_mode")
    if (
        source_mode is not None
        and str(source_mode) != str(args.interface_fraction_mode)
    ):
        raise ValueError(
            "analytical seed payload uses a different interface-fraction mode"
        )


def source_geometry_from_payload(
    payload: dict[str, Any],
    *,
    theta_count: int,
) -> dict[str, Any]:
    """Resample saved numerical-I2 ray boundaries onto the requested rays."""
    metadata = dict(payload["transform"])
    transform = RayTransform(
        metadata["density_origin_over_n0"],
        metadata["density_scale_over_n0"],
        metadata["temperature_scale_MeV"],
    )
    source_theta = np.asarray(payload["theta_axis_rad"], dtype=float)
    source_inner = np.asarray(payload["rho_inner_boundary"], dtype=float)
    source_outer = np.asarray(payload["rho_outer_boundary"], dtype=float)
    if not (
        source_theta.ndim == 1
        and source_inner.shape == source_theta.shape
        and source_outer.shape == source_theta.shape
    ):
        raise ValueError("analytical payload ray boundaries have invalid shapes")
    order = np.argsort(source_theta)
    source_theta = source_theta[order]
    source_inner = source_inner[order]
    source_outer = source_outer[order]
    if np.any(np.diff(source_theta) <= 0.0):
        raise ValueError("analytical payload theta axis must be strictly increasing")
    theta_axis = chebyshev_closed_axis(
        float(source_theta[0]),
        float(source_theta[-1]),
        int(theta_count),
    )
    rho_inner = np.interp(theta_axis, source_theta, source_inner)
    rho_outer = np.interp(theta_axis, source_theta, source_outer)
    if not np.all(rho_outer > rho_inner):
        raise ValueError("resampled numerical-I2 safe interval collapsed")
    boundary_payload = {
        "theta_axis_rad": theta_axis,
        "rho_inner": rho_inner,
        "rho_outer": rho_outer,
        "gap_curve_records": list(payload.get("gap_boundary_records", [])),
        "A_curve_records": list(payload.get("A_boundary_records", [])),
        "gap_records": list(payload.get("gap_mesh_boundary_records", [])),
        "A_records": list(payload.get("A_mesh_boundary_records", [])),
        "endpoint_records": dict(payload.get("endpoint_records", {})),
    }
    return {
        "transform": transform,
        "theta_axis_rad": theta_axis,
        "rho_inner": rho_inner,
        "rho_A_boundary": rho_outer,
        "boundary_payload": boundary_payload,
    }


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
    nB_0minus: float,
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
    upper = float(nB_0minus) * float(shooting_u_max)
    if not 0.0 < lower < upper:
        raise RuntimeError(
            "automatic jB bounds require "
            "0 < lower < nB_0minus*shooting_u_max"
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
            row.get("T_inf_MeV", np.nan),
            row.get("h_0minus_MeV4", np.nan),
            row.get("nB_0minus_MeV3", np.nan),
            row.get("muB_0minus_MeV", np.nan),
            row.get("h_inf_MeV4", np.nan),
            row.get("nB_inf_MeV3", np.nan),
            row.get("muB_inf_MeV", np.nan),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or np.any(
        values[[0, 1, 3, 6]] <= 0.0
    ):
        return np.nan
    T_0minus, T_inf, h_0minus, nB_0minus, muB_0minus, h_inf, nB_inf, muB_inf = values
    return float(
        (h_inf / nB_inf - muB_inf) / T_inf
        - (h_0minus / nB_0minus - muB_0minus) / T_0minus
    )


def effective_xi_from_profile(result: dict[str, Any]) -> dict[str, float]:
    """Measure the semi-analytical I1 shape parameter from a BVP profile."""
    a = np.asarray(result.get("a", []), dtype=float)
    jK = np.asarray(result.get("jK", []), dtype=float)
    nB = np.asarray(result.get("nB", []), dtype=float)
    u = np.asarray(result.get("u", []), dtype=float)
    if a.ndim != 1 or a.size < 2 or not (
        a.shape == jK.shape == nB.shape == u.shape
    ):
        return {
            "xi_effective": np.nan,
            "I1_profile": np.nan,
            "interface_diffusive_flux": np.nan,
            "profile_nB_relative_span": np.nan,
        }
    finite = (
        np.isfinite(a)
        & np.isfinite(jK)
        & np.isfinite(nB)
        & np.isfinite(u)
        & (nB > 0.0)
    )
    if np.count_nonzero(finite) < 2:
        return {
            "xi_effective": np.nan,
            "I1_profile": np.nan,
            "interface_diffusive_flux": np.nan,
            "profile_nB_relative_span": np.nan,
        }
    a = a[finite]
    jK = jK[finite]
    nB = nB[finite]
    u = u[finite]
    diffusive_flux = jK / nB - u * a
    order = np.argsort(a)
    I1_profile = float(
        np.trapezoid(diffusive_flux[order], a[order])
    )
    a_0plus = float(a[0])
    lambda_n = _as_finite_float(result.get("lambda_n"))
    u_0minus = _as_finite_float(result.get("u_0minus"))
    interface_flux = float(lambda_n * u_0minus * (1.0 - a_0plus))
    denominator = float(interface_flux * a_0plus)
    xi_effective = (
        float(2.0 * I1_profile / denominator - 1.0)
        if np.isfinite(denominator) and denominator > 0.0
        else np.nan
    )
    nB_scale = max(abs(float(np.mean(nB))), np.finfo(float).tiny)
    return {
        "xi_effective": xi_effective,
        "I1_profile": I1_profile,
        "interface_diffusive_flux": interface_flux,
        "profile_nB_relative_span": float(np.ptp(nB) / nB_scale),
    }


def extract_continuation_bundle(
    result: dict[str, Any],
    *,
    max_points: int,
) -> dict[str, Any] | None:
    """Keep a compact solver-ready nK/jK profile for adjacent continuation."""
    if not bool(result.get("success", False)):
        return None
    s_coord = np.asarray(result.get("s_coord", []), dtype=float)
    nK = np.asarray(result.get("nK", []), dtype=float)
    jK = np.asarray(result.get("jK", []), dtype=float)
    if s_coord.ndim != 1 or s_coord.size < 2 or not (
        s_coord.shape == nK.shape == jK.shape
    ):
        return None
    if not (
        np.all(np.isfinite(s_coord))
        and np.all(np.isfinite(nK))
        and np.all(np.isfinite(jK))
        and np.all(np.diff(s_coord) > 0.0)
    ):
        return None
    point_count = max(5, int(max_points))
    if s_coord.size > point_count:
        compact_s = np.linspace(
            float(s_coord[0]),
            float(s_coord[-1]),
            point_count,
        )
        compact_nK = np.interp(compact_s, s_coord, nK)
        compact_jK = np.interp(compact_s, s_coord, jK)
    else:
        compact_s = s_coord.copy()
        compact_nK = nK.copy()
        compact_jK = jK.copy()
    scalar_keys = (
        "jB",
        "T_inf",
        "muB_inf",
        "muB_0plus",
        "muK_0plus",
        "T_0plus",
        "nK_inf",
    )
    bundle = {
        key: _as_finite_float(result.get(key))
        for key in scalar_keys
    }
    if not all(np.isfinite(bundle[key]) for key in scalar_keys):
        return None
    bundle.update(
        {
            "s_coord": compact_s,
            "nK": compact_nK,
            "jK": compact_jK,
            "continuation_profile_points": int(compact_s.size),
        }
    )
    return bundle


def calculate_numerical_point(
    task: NumericalRayPointTask,
    *,
    continuation_guess: dict[str, Any] | None = None,
    return_continuation: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
    started = time.monotonic()
    nB_0minus = float(task.nB_over_n0) * float(task.n0)
    row = {
        **vars(task),
        "success": False,
        "message": "",
        "theta_deg": float(np.degrees(task.theta_rad)),
        "T_MeV": float(task.T_0minus),
        "nB_0minus_over_n0": float(task.nB_over_n0),
        "nB_0minus_MeV3": nB_0minus,
        "T_0plus_target_MeV": float(task.T_0plus),
        "u_0minus": np.nan,
        "u_0minus_m_s": np.nan,
        "jB": np.nan,
        "lambda_n": np.nan,
        "A_boundary": np.nan,
        "a_local_0plus": np.nan,
        "a_local_end": np.nan,
        "entropy_jump_per_baryon": np.nan,
        "entropy_allowed": False,
        "point_timed_out": False,
        "point_timeout_s": float(task.point_timeout),
        "point_elapsed_s": np.nan,
        "jB_guess_used": _as_finite_float(task.jB_guess),
        "analytic_jB_guess_used": _as_finite_float(
            task.analytic_jB_guess
        ),
        "propagated_jB_guess_available": _as_finite_float(
            task.propagated_jB_guess
        ),
        "attempt_kind": str(task.attempt_kind),
        "primary_attempt_success": False,
        "fallback_attempted": bool(
            task.attempt_kind in ("propagated_fallback", "analytic_fallback")
        ),
        "fallback_success": False,
        "retry_replaced_failure": False,
        "continuation_profile_used": bool(continuation_guess is not None),
        "continuation_profile_points_used": int(
            np.asarray(continuation_guess.get("s_coord", [])).size
            if isinstance(continuation_guess, dict)
            else 0
        ),
        "xi_effective": np.nan,
        "I1_profile": np.nan,
        "interface_diffusive_flux": np.nan,
        "profile_nB_relative_span": np.nan,
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
            nB_0minus,
            task.jB_bounds,
            task.auto_jB_bounds,
            task.shooting_u_max,
            task.shooting_jB_lower,
        )
        with point_timeout(task.point_timeout):
            result = phase_velocity.solve_front_energy_conserving_uNmax(
                T_0minus=float(task.T_0minus),
                nB_0minus=nB_0minus,
                B_one_forth=float(task.B14),
                T_0plus=float(task.T_0plus),
                ms=float(task.ms),
                param=para.paraQMCRMF3,
                NM_type=str(task.nm_type),
                tail_eps=float(task.tail_eps),
                n_mesh=int(task.n_mesh),
                tol_bvp=float(task.tol_bvp),
                max_nodes=int(task.max_nodes),
                jB_guess=task.jB_guess,
                T_inf_guess=task.T_inf_guess,
                jB_bounds=point_bounds,
                return_profile=bool(return_continuation),
                verb=False,
                continuation_guess=continuation_guess,
            )
        u_0minus = _as_finite_float(
            result.get("u_0minus_max_candidate", result.get("u_0minus"))
        )
        muB_inf = _as_finite_float(result.get("muB_inf"))
        T_inf = _as_finite_float(result.get("T_inf"))
        a_local_0plus = _as_finite_float(
            result.get("a_0plus_derived", result.get("a_0plus"))
        )
        A_boundary = _as_finite_float(result.get("A_boundary"))
        if (
            not np.isfinite(A_boundary)
            and float(task.T_0plus) == 0.0
            and np.isfinite(a_local_0plus)
        ):
            A_boundary = a_local_0plus
        if (
            not np.isfinite(A_boundary)
            and np.isfinite(muB_inf)
            and muB_inf > 0.0
            and np.isfinite(T_inf)
        ):
            A_boundary = float(
                9.0 * np.pi / np.sqrt(2.0) * T_inf / muB_inf
            )
        row.update(
            {
                "success": bool(result.get("success", False)),
                "message": str(result.get("message", "")),
                "muB_0minus_MeV": _as_finite_float(result.get("muB_0minus")),
                "u_0minus": u_0minus,
                "u_0minus_m_s": (
                    float(u_0minus * const.m_per_sec)
                    if np.isfinite(u_0minus)
                    else np.nan
                ),
                "jB": _as_finite_float(result.get("jB")),
                "u_inf": _as_finite_float(result.get("u_inf")),
                "lambda_n": _as_finite_float(result.get("lambda_n")),
                "A_boundary": A_boundary,
                "a_local_0plus": a_local_0plus,
                "a_local_end": _as_finite_float(result.get("a_end")),
                "a_0plus": _as_finite_float(result.get("a_0plus")),
                "a_0plus_derived": a_local_0plus,
                "a_0plus_max": a_local_0plus,
                "muB_inf_MeV": muB_inf,
                "T_inf_MeV": T_inf,
                "T_0plus_MeV": _as_finite_float(result.get("T_0plus")),
                "nB_inf_MeV3": _as_finite_float(result.get("nB_inf")),
                "nK_inf_MeV3": _as_finite_float(result.get("nK_inf")),
                "nB_0plus_MeV3": _as_finite_float(
                    result.get("nB_0plus")
                ),
                "nK_0plus_MeV3": _as_finite_float(
                    result.get("nK_0plus")
                ),
                "nK_end": _as_finite_float(result.get("nK_end")),
                "jK_end": _as_finite_float(result.get("jK_end")),
                "h_0minus_MeV4": _as_finite_float(result.get("h_0minus")),
                "h_inf_MeV4": _as_finite_float(result.get("h_inf")),
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
        row["primary_attempt_success"] = bool(
            task.attempt_kind
            in (
                "primary",
                "analytic_seed_primary",
                "profile_continuation_primary",
                "analytic_primary_no_neighbor",
            )
            and row["success"]
        )
        row["fallback_success"] = bool(
            task.attempt_kind
            in ("propagated_fallback", "analytic_fallback")
            and row["success"]
        )
        if return_continuation and row["success"]:
            row.update(effective_xi_from_profile(result))
    except PointTimeoutError as exc:
        row.update(
            message=str(exc),
            point_timed_out=True,
            bvp_status=STATUS_TIMEOUT,
        )
    except Exception as exc:
        row["message"] = str(exc)
    row["point_elapsed_s"] = float(time.monotonic() - started)
    if not return_continuation:
        return row
    return row, extract_continuation_bundle(
        result if bool(row.get("success", False)) and "result" in locals() else {},
        max_points=int(task.continuation_profile_points),
    )


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
            "u_0minus_grid": np.full(shape, np.nan),
            "jB_grid": np.full(shape, np.nan),
            "lambda_n_grid": np.full(shape, np.nan),
            "A_boundary_grid": np.full(shape, np.nan),
            "a_local_0plus_grid": np.full(shape, np.nan),
            "a_0plus_grid": np.full(shape, np.nan),
            "entropy_jump_grid": np.full(shape, np.nan),
            "status_grid": np.full(
                shape,
                STATUS_UNSCANNED,
                dtype=np.int8,
            ),
            "point_elapsed_s_grid": np.full(shape, np.nan),
            "point_total_elapsed_s_grid": np.zeros(shape),
            "point_timed_out_grid": np.zeros(shape, dtype=bool),
            "attempt_count_grid": np.zeros(shape, dtype=np.int16),
            "primary_attempt_success_grid": np.zeros(shape, dtype=bool),
            "fallback_attempted_grid": np.zeros(shape, dtype=bool),
            "fallback_success_grid": np.zeros(shape, dtype=bool),
            "propagated_jB_guess_grid": np.full(shape, np.nan),
            "continuation_profile_used_grid": np.zeros(shape, dtype=bool),
            "continuation_profile_points_used_grid": np.zeros(
                shape,
                dtype=np.int32,
            ),
            "profile_continuation_success_grid": np.zeros(
                shape,
                dtype=bool,
            ),
            "analytic_fallback_success_grid": np.zeros(
                shape,
                dtype=bool,
            ),
            "xi_effective_grid": np.full(shape, np.nan),
            "I1_profile_grid": np.full(shape, np.nan),
            "interface_diffusive_flux_grid": np.full(shape, np.nan),
            "profile_nB_relative_span_grid": np.full(shape, np.nan),
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
    analytic_jB_guess: float | None = None,
    propagated_jB_guess: float | None = None,
    attempt_kind: str = "primary",
    retry_attempt: int = 0,
    retry_of: tuple[int, int] | None = None,
) -> NumericalRayPointTask:
    analytic_guess = _optional_finite_float(analytic_jB_guess)
    if analytic_guess is None and str(jB_guess_source).startswith("analytic_"):
        analytic_guess = _optional_finite_float(jB_guess)
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
        T_0minus=float(
            arrays["temperature_grid_MeV"][row_index, col_index]
        ),
        n0=float(n0),
        B14=float(B14),
        ms=float(args.ms),
        nm_type=str(args.nm_type),
        upB=int(args.upB),
        T_0plus=float(args.T_0plus),
        tail_eps=float(args.tail_eps),
        n_mesh=int(args.n_mesh),
        tol_bvp=float(args.tol_bvp),
        max_nodes=int(args.max_nodes),
        jB_guess=_optional_finite_float(jB_guess),
        T_inf_guess=_optional_finite_float(args.T_inf_guess),
        jB_bounds=args.jB_bounds,
        auto_jB_bounds=bool(args.auto_jB_bounds),
        shooting_u_max=float(args.shooting_u_max),
        shooting_jB_lower=float(args.shooting_jB_lower),
        point_timeout=float(args.point_timeout),
        continuation_profile_points=int(args.continuation_profile_points),
        jB_guess_source=str(jB_guess_source),
        continuation_direction=str(direction),
        analytic_jB_guess=analytic_guess,
        propagated_jB_guess=_optional_finite_float(propagated_jB_guess),
        attempt_kind=str(attempt_kind),
        retry_attempt=int(retry_attempt),
        retry_of=retry_of,
    )


def _solve_profile_attempt(
    task: NumericalRayPointTask,
    *,
    continuation_guess: dict[str, Any] | None,
    attempt_kind: str,
    guess_source: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    analytic_guess = _optional_finite_float(task.analytic_jB_guess)
    if continuation_guess is not None:
        propagated_guess = _optional_finite_float(
            continuation_guess.get("jB")
        )
        attempted_task = replace(
            task,
            jB_guess=propagated_guess,
            propagated_jB_guess=propagated_guess,
            jB_guess_source=str(guess_source),
            attempt_kind=str(attempt_kind),
        )
    else:
        attempted_task = replace(
            task,
            jB_guess=analytic_guess,
            jB_guess_source=str(guess_source),
            attempt_kind=str(attempt_kind),
        )
    outcome = calculate_numerical_point(
        attempted_task,
        continuation_guess=continuation_guess,
        return_continuation=True,
    )
    if not isinstance(outcome, tuple):
        raise RuntimeError("profile solve did not return a continuation tuple")
    return outcome


def calculate_seed_profile(
    task: NumericalRayPointTask,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _solve_profile_attempt(
        task,
        continuation_guess=None,
        attempt_kind="analytic_seed_primary",
        guess_source="analytic_2d_cell_nearest_seed_velocity",
    )


def calculate_profile_retry(
    job: ProfileRetryJob,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    row, bundle = _solve_profile_attempt(
        job.task,
        continuation_guess=job.continuation,
        attempt_kind=job.attempt_kind,
        guess_source=job.guess_source,
    )
    row["retry_attempt"] = max(1, int(row.get("retry_attempt", 0)))
    row["retry_of"] = (
        int(row.get("row_index", job.task.row_index)),
        int(row.get("col_index", job.task.col_index)),
    )
    row["retry_replaced_failure"] = bool(row.get("success", False))
    return row, bundle


def _emit_progress_row(
    progress_queue,
    row: dict[str, Any],
) -> None:
    if progress_queue is None:
        return
    progress_queue.put(
        {
            key: value
            for key, value in row.items()
            if not str(key).startswith("_")
        }
    )


def solve_ray_direction(
    job: RayDirectionJob,
    *,
    progress_queue=None,
    solve_point=None,
) -> list[dict[str, Any]]:
    """Walk one half-ray, using only the immediately preceding profile."""
    if solve_point is None:
        solve_point = _solve_profile_attempt
    continuation = job.initial_continuation
    outcomes: list[dict[str, Any]] = []
    for task in job.tasks:
        if continuation is not None:
            row, bundle = solve_point(
                task,
                continuation_guess=continuation,
                attempt_kind="profile_continuation_primary",
                guess_source="immediate_same_ray_full_profile",
            )
            outcomes.append(row)
            _emit_progress_row(progress_queue, row)
            if bool(row.get("success", False)) and bundle is not None:
                row["_continuation_bundle"] = bundle
                continuation = bundle
                continue
            row, bundle = solve_point(
                task,
                continuation_guess=None,
                attempt_kind="analytic_fallback",
                guess_source="analytic_2d_numerical_I2_fallback",
            )
            row["retry_attempt"] = max(
                1,
                int(row.get("retry_attempt", 0)),
            )
            row["retry_of"] = (
                int(row.get("row_index", job.row_index)),
                int(row.get("col_index", task.col_index)),
            )
            outcomes.append(row)
            _emit_progress_row(progress_queue, row)
        else:
            row, bundle = solve_point(
                task,
                continuation_guess=None,
                attempt_kind="analytic_primary_no_neighbor",
                guess_source="analytic_2d_numerical_I2_no_neighbor",
            )
            outcomes.append(row)
            _emit_progress_row(progress_queue, row)
        if bool(row.get("success", False)) and bundle is not None:
            row["_continuation_bundle"] = bundle
            continuation = bundle
        else:
            # Strict immediate-neighbor rule: never jump across a failed cell.
            continuation = None
    return outcomes


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
            jB_guess_source="analytic_2d_cell_nearest_seed_velocity",
            analytic_jB_guess=float(
                arrays["seed_analytic_jB_by_theta"][row_index]
            ),
        )
        for row_index in range(arrays["theta_axis_rad"].size)
    ]


def build_ray_direction_jobs(
    *,
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    n0: float,
    B14: float,
    seed_continuations: dict[int, dict[str, Any]],
) -> list[RayDirectionJob]:
    """Create adjacent half-ray walks so at most one profile crosses a process."""
    row_count, column_count = arrays["rho_grid"].shape
    jobs: list[RayDirectionJob] = []
    for row_index in range(row_count):
        seed_index = int(
            arrays["seed_radial_index_by_theta"][row_index]
        )
        index_groups = (
            ("rho_out", range(seed_index + 1, column_count)),
            ("rho_in", range(seed_index - 1, -1, -1)),
        )
        for direction, indices in index_groups:
            tasks = tuple(
                _grid_point_task(
                    args=args,
                    arrays=arrays,
                    row_index=row_index,
                    col_index=col_index,
                    direction=direction,
                    n0=n0,
                    B14=B14,
                    jB_guess=float(
                        arrays["analytic_jB_guess_grid"][
                            row_index,
                            col_index,
                        ]
                    ),
                    jB_guess_source="analytic_2d_numerical_I2_available",
                    analytic_jB_guess=float(
                        arrays["analytic_jB_guess_grid"][
                            row_index,
                            col_index,
                        ]
                    ),
                )
                for col_index in indices
            )
            if tasks:
                jobs.append(
                    RayDirectionJob(
                        row_index=row_index,
                        direction=direction,
                        tasks=tasks,
                        initial_continuation=seed_continuations.get(
                            row_index
                        ),
                    )
                )
    return jobs


def _record_result(
    arrays: dict[str, np.ndarray],
    row: dict[str, Any],
) -> None:
    i = int(row["row_index"])
    j = int(row["col_index"])
    arrays["point_elapsed_s_grid"][i, j] = _as_finite_float(
        row.get("point_elapsed_s")
    )
    elapsed = _as_finite_float(row.get("point_elapsed_s"), default=0.0)
    arrays["point_total_elapsed_s_grid"][i, j] += elapsed
    arrays["attempt_count_grid"][i, j] += 1
    arrays["primary_attempt_success_grid"][i, j] |= bool(
        row.get("primary_attempt_success", False)
    )
    arrays["fallback_attempted_grid"][i, j] |= bool(
        row.get("fallback_attempted", False)
    )
    arrays["fallback_success_grid"][i, j] |= bool(
        row.get("fallback_success", False)
    )
    arrays["continuation_profile_used_grid"][i, j] |= bool(
        row.get("continuation_profile_used", False)
    )
    arrays["continuation_profile_points_used_grid"][i, j] = max(
        int(arrays["continuation_profile_points_used_grid"][i, j]),
        int(row.get("continuation_profile_points_used", 0)),
    )
    arrays["profile_continuation_success_grid"][i, j] |= bool(
        row.get("success", False)
        and str(row.get("attempt_kind")) == "profile_continuation_primary"
    )
    arrays["analytic_fallback_success_grid"][i, j] |= bool(
        row.get("success", False)
        and str(row.get("attempt_kind")) == "analytic_fallback"
    )
    propagated_guess = _as_finite_float(
        row.get("propagated_jB_guess_available")
    )
    if np.isfinite(propagated_guess):
        arrays["propagated_jB_guess_grid"][i, j] = propagated_guess
    arrays["point_timed_out_grid"][i, j] = bool(
        row.get("point_timed_out", False)
    )
    if row.get("success"):
        arrays["status_grid"][i, j] = STATUS_FINITE
        arrays["velocity_grid_m_s"][i, j] = _as_finite_float(
            row.get("u_0minus_m_s")
        )
        arrays["u_0minus_grid"][i, j] = _as_finite_float(row.get("u_0minus"))
        arrays["jB_grid"][i, j] = _as_finite_float(row.get("jB"))
        arrays["lambda_n_grid"][i, j] = _as_finite_float(
            row.get("lambda_n")
        )
        arrays["A_boundary_grid"][i, j] = _as_finite_float(
            row.get("A_boundary")
        )
        arrays["a_local_0plus_grid"][i, j] = _as_finite_float(
            row.get("a_local_0plus")
        )
        arrays["a_0plus_grid"][i, j] = arrays[
            "a_local_0plus_grid"
        ][i, j]
        arrays["entropy_jump_grid"][i, j] = _as_finite_float(
            row.get("entropy_jump_per_baryon")
        )
        arrays["xi_effective_grid"][i, j] = _as_finite_float(
            row.get("xi_effective")
        )
        arrays["I1_profile_grid"][i, j] = _as_finite_float(
            row.get("I1_profile")
        )
        arrays["interface_diffusive_flux_grid"][i, j] = _as_finite_float(
            row.get("interface_diffusive_flux")
        )
        arrays["profile_nB_relative_span_grid"][i, j] = _as_finite_float(
            row.get("profile_nB_relative_span")
        )
    elif row.get("point_timed_out"):
        arrays["status_grid"][i, j] = STATUS_TIMEOUT
    else:
        arrays["status_grid"][i, j] = STATUS_NUMERICAL_FAILURE


def update_effective_xi_map(arrays: dict[str, np.ndarray]) -> None:
    """Interpolate profile-measured xi over the physical numerical mesh."""
    nB = np.asarray(arrays["nB_grid_over_n0"], dtype=float)
    temperature = np.asarray(arrays["temperature_grid_MeV"], dtype=float)
    xi = np.asarray(arrays["xi_effective_grid"], dtype=float)
    if not (nB.shape == temperature.shape == xi.shape):
        raise ValueError("effective-xi mapping grids must have matching shapes")
    points = np.column_stack((nB.ravel(), temperature.ravel()))
    values = xi.ravel()
    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(values)
    mapped = np.full(values.shape, np.nan)
    fallback = np.zeros(values.shape, dtype=bool)
    if np.count_nonzero(finite) >= 3:
        try:
            linear = LinearNDInterpolator(
                points[finite],
                values[finite],
                fill_value=np.nan,
                rescale=True,
            )
            mapped = np.asarray(linear(points), dtype=float)
        except Exception:
            mapped.fill(np.nan)
    missing = ~np.isfinite(mapped)
    if np.any(missing) and np.count_nonzero(finite):
        nearest = NearestNDInterpolator(
            points[finite],
            values[finite],
            rescale=True,
        )
        mapped[missing] = np.asarray(
            nearest(points[missing]),
            dtype=float,
        )
        fallback[missing] = True
    arrays["xi_effective_interpolated_grid"] = mapped.reshape(xi.shape)
    arrays["xi_interpolation_fallback_mask"] = fallback.reshape(xi.shape)


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
    payload.pop("velocity_closure", None)
    payload.pop("transport_integral_definition", None)
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
            "T_0plus_target_MeV": float(args.T_0plus),
            "seed_velocity_m_s": float(args.seed_velocity),
            "safety_velocity_m_s": float(args.safety_velocity),
            "seed_strategy": (
                "actual_cell_nearest_seed_velocity_then_immediate_same_ray_"
                "full_nK_jK_profile_primary_with_analytic_jB_fallback"
            ),
            "continuation_strategy": (
                "seed_once_then_parallel_inward_outward_half_rays_with_"
                "strict_immediate_neighbor_full_profile_continuation"
            ),
            "continuation_profile_points": int(
                args.continuation_profile_points
            ),
            "guess_priority": (
                "full_numerical_profile_then_analytic_numerical_I2"
            ),
            "xi_effective_definition": (
                "2_I1_profile_over_interface_flux_a_0plus_minus_1"
            ),
            "xi_effective_role": (
                "profile_measured_map_for_future_semi_analytical_correction"
            ),
            "analytic_seed_interpolation": (
                "physical_2d_linear_with_hidden_ray_edge_fallback"
            ),
            "analytic_seed_velocity_closure": "numerical_I2",
            "analytic_seed_role": "initial_guess_and_mesh_geometry_only",
            "boundary_provenance": "26-08-02_numerical_I2_payload",
            "numerical_mesh_outer_boundary": (
                "analytic_extreme_endothermic_velocity_cap"
            ),
            "safe_region_definition": (
                "Delta_h_le_0_and_analytic_extreme_endothermic_"
                "velocity_lt_safety_velocity_and_A_boundary_lt_1"
            ),
            "A_boundary_role": (
                "physical_exclusion_boundary_saved_outside_numerical_mesh"
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
    source_geometry = source_geometry_from_payload(
        analytic_seed_payload,
        theta_count=args.theta_count,
    )
    transform = source_geometry["transform"]
    boundary_payload = source_geometry["boundary_payload"]
    print(
        f"Calibrated B^(1/4) = {B14:.9f} MeV\n"
        f"Output: {output_path}\n"
        f"Run tag: {args.run_tag}\n"
        f"Hidden grid: {args.theta_count} theta rays x "
        f"{args.radial_count} safe radial fractions\n"
        f"Physical window: nB(0-)/n0={args.density_min_over_n0:g}-"
        f"{args.density_max_over_n0:g}, "
        f"T={args.t_min:g}-{args.t_max:g} MeV\n"
        f"Seed cells: nearest to interpolated analytical "
        f"u={args.seed_velocity:g} m/s on each numerical ray, from "
        f"{analytic_seed_path}\n"
        f"Safety boundary: analytical u={args.safety_velocity:g} m/s, "
        "strictly inside A=1\n"
        "Boundary geometry: reused from the numerical-I2 payload\n"
        "Guess order after each seed: immediate same-ray full nK/jK "
        "profile first, analytical numerical-I2 jB fallback\n"
        f"T_0plus={args.T_0plus:g} MeV; workers={args.workers}; "
        f"point timeout={args.point_timeout:g} s"
    )
    safety_boundary = build_analytic_safety_boundary(
        analytic_seed_payload,
        transform=transform,
        theta_axis_rad=source_geometry["theta_axis_rad"],
        rho_inner=source_geometry["rho_inner"],
        rho_A_boundary=source_geometry["rho_A_boundary"],
        target_velocity_m_s=args.safety_velocity,
    )
    print(
        "Analytical safety boundary clearance from A=1 in hidden rho: "
        f"{np.min(safety_boundary['clearance_to_A_rho_by_theta']):.6g}"
        "-"
        f"{np.max(safety_boundary['clearance_to_A_rho_by_theta']):.6g}"
    )
    eta_axis = chebyshev_open_axis(
        args.radial_count,
        args.boundary_buffer_fraction,
    )
    mesh = build_safe_ray_mesh(
        transform=transform,
        theta_axis_rad=source_geometry["theta_axis_rad"],
        radial_fraction_axis=eta_axis,
        rho_inner=source_geometry["rho_inner"],
        rho_outer=safety_boundary["rho_by_theta"],
    )
    arrays = _empty_result_arrays(mesh, n0)
    arrays.update(
        {
            "A_mesh_boundary_rho_by_theta": np.asarray(
                source_geometry["rho_A_boundary"],
                dtype=float,
            ),
            "numerical_safety_boundary_rho_by_theta": np.asarray(
                safety_boundary["rho_by_theta"],
                dtype=float,
            ),
            "numerical_safety_boundary_jB_by_theta": np.asarray(
                safety_boundary["jB_by_theta"],
                dtype=float,
            ),
            "numerical_safety_boundary_nB_over_n0": np.asarray(
                safety_boundary["nB_over_n0"],
                dtype=float,
            ),
            "numerical_safety_boundary_temperature_MeV": np.asarray(
                safety_boundary["temperature_MeV"],
                dtype=float,
            ),
            "safety_to_A_rho_clearance_by_theta": np.asarray(
                safety_boundary["clearance_to_A_rho_by_theta"],
                dtype=float,
            ),
        }
    )
    interpolated_analytic = interpolate_analytic_fields_2d(
        analytic_seed_payload,
        nB_grid_over_n0=arrays["nB_grid_over_n0"],
        temperature_grid_MeV=arrays["temperature_grid_MeV"],
        theta_grid_rad=arrays["theta_grid_rad"],
        rho_grid=arrays["rho_grid"],
    )
    arrays["analytic_jB_guess_grid"] = interpolated_analytic["jB_grid"]
    arrays["analytic_velocity_guess_grid_m_s"] = interpolated_analytic[
        "velocity_grid_m_s"
    ]
    arrays["analytic_2d_interpolation_fallback_mask"] = (
        interpolated_analytic["fallback_mask"]
    )
    seed_spine = select_seed_cells_from_interpolated_grid(
        rho_grid=arrays["rho_grid"],
        velocity_grid_m_s=arrays["analytic_velocity_guess_grid_m_s"],
        jB_grid=arrays["analytic_jB_guess_grid"],
        target_velocity_m_s=args.seed_velocity,
    )
    arrays.update(seed_spine)
    print(
        "Physical 2D analytical interpolation edge fallbacks: "
        f"{np.count_nonzero(arrays['analytic_2d_interpolation_fallback_mask'])}"
        f"/{arrays['analytic_2d_interpolation_fallback_mask'].size}\n"
        "Selected seed-cell analytical velocities: "
        f"{np.min(arrays['seed_actual_velocity_m_s_by_theta']):.6g}-"
        f"{np.max(arrays['seed_actual_velocity_m_s_by_theta']):.6g} m/s"
    )
    if (
        np.nanmin(arrays["nB_grid_over_n0"])
        < args.density_min_over_n0 - 1.0e-10
        or np.nanmax(arrays["nB_grid_over_n0"])
        > args.density_max_over_n0 + 1.0e-10
        or np.nanmin(arrays["temperature_grid_MeV"])
        < args.t_min - 1.0e-10
        or np.nanmax(arrays["temperature_grid_MeV"])
        > args.t_max + 1.0e-10
    ):
        raise ValueError("numerical-I2 safe mesh lies outside the requested window")
    process_context = mp.get_context("spawn")
    manager = None
    executor = (
        None
        if args.workers == 1
        else ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=process_context,
        )
    )
    records: list[dict[str, Any]] = []
    records_by_cell: dict[tuple[int, int], dict[str, Any]] = {}
    failures: set[tuple[int, int]] = set()
    completed_since_save = 0
    continuations_by_cell: dict[tuple[int, int], dict[str, Any]] = {}

    def save() -> None:
        update_effective_xi_map(arrays)
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
        public_row = {
            key: value
            for key, value in row.items()
            if not str(key).startswith("_")
        }
        key = (
            int(public_row["row_index"]),
            int(public_row["col_index"]),
        )
        records.append(public_row)
        records_by_cell[key] = public_row
        if public_row.get("success"):
            failures.discard(key)
        elif first_pass:
            failures.add(key)
        _record_result(arrays, public_row)
        completed_since_save += 1
        if args.save_every and completed_since_save >= args.save_every:
            save()
            completed_since_save = 0

    def remember_continuations(outcomes: list[dict[str, Any]]) -> None:
        for row in outcomes:
            bundle = row.get("_continuation_bundle")
            if bool(row.get("success", False)) and isinstance(bundle, dict):
                continuations_by_cell[
                    (int(row["row_index"]), int(row["col_index"]))
                ] = bundle

    def advance_progress(row: dict[str, Any], *, retry: bool = False) -> None:
        if retry:
            progress.total += 1
            progress.refresh()
        progress.update(1)
        progress.set_postfix_str(
            f"{row.get('attempt_kind', 'attempt')}, "
            f"theta={float(row.get('theta_deg', np.nan)):.3g} deg, "
            f"rho={float(row.get('rho', np.nan)):.4g}"
        )

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
        desc="Full-profile-continuation uNmax contour",
        unit="pt",
        dynamic_ncols=True,
    )
    try:
        seed_continuations: dict[int, dict[str, Any]] = {}
        completed_seed_tasks: dict[int, NumericalRayPointTask] = {}
        for task, outcome in _run_unordered(
            calculate_seed_profile,
            seed_tasks,
            executor,
        ):
            row, bundle = outcome
            accept(row, first_pass=True)
            completed_seed_tasks[int(task.row_index)] = task
            if bool(row.get("success", False)) and bundle is not None:
                seed_continuations[int(task.row_index)] = bundle
                continuations_by_cell[
                    (int(task.row_index), int(task.col_index))
                ] = bundle
            advance_progress(row)

        seed_retry_jobs: list[ProfileRetryJob] = []
        for row_index, task in completed_seed_tasks.items():
            key = (row_index, int(task.col_index))
            if records_by_cell[key].get("success"):
                continue
            neighbor_bundle = next(
                (
                    seed_continuations[neighbor]
                    for neighbor in (row_index - 1, row_index + 1)
                    if neighbor in seed_continuations
                ),
                None,
            )
            if neighbor_bundle is not None:
                seed_retry_jobs.append(
                    ProfileRetryJob(
                        task=task,
                        continuation=neighbor_bundle,
                        attempt_kind="seed_neighbor_full_profile_retry",
                        guess_source="immediate_theta_seed_full_profile",
                    )
                )
        if seed_retry_jobs:
            progress.total += len(seed_retry_jobs)
            progress.refresh()
            for job, outcome in _run_unordered(
                calculate_profile_retry,
                seed_retry_jobs,
                executor,
            ):
                row, bundle = outcome
                accept(row, first_pass=False)
                if bool(row.get("success", False)) and bundle is not None:
                    row_index = int(job.task.row_index)
                    seed_continuations[row_index] = bundle
                    continuations_by_cell[
                        (row_index, int(job.task.col_index))
                    ] = bundle
                advance_progress(row)

        direction_jobs = build_ray_direction_jobs(
            args=args,
            arrays=arrays,
            n0=n0,
            B14=B14,
            seed_continuations=seed_continuations,
        )
        if executor is None:
            for job in direction_jobs:
                outcomes = solve_ray_direction(job)
                for row in outcomes:
                    is_fallback = (
                        str(row.get("attempt_kind")) == "analytic_fallback"
                    )
                    accept(row, first_pass=not is_fallback)
                    advance_progress(row, retry=is_fallback)
                remember_continuations(outcomes)
        else:
            manager = process_context.Manager()
            progress_queue = manager.Queue()
            pending = {
                executor.submit(
                    solve_ray_direction,
                    job,
                    progress_queue=progress_queue,
                ): job
                for job in direction_jobs
            }
            while pending:
                try:
                    row = progress_queue.get(timeout=0.25)
                except queue.Empty:
                    row = None
                if row is not None:
                    is_fallback = (
                        str(row.get("attempt_kind")) == "analytic_fallback"
                    )
                    accept(row, first_pass=not is_fallback)
                    advance_progress(row, retry=is_fallback)
                completed = [future for future in pending if future.done()]
                for future in completed:
                    pending.pop(future)
                    remember_continuations(future.result())
            while True:
                try:
                    row = progress_queue.get_nowait()
                except queue.Empty:
                    break
                is_fallback = (
                    str(row.get("attempt_kind")) == "analytic_fallback"
                )
                accept(row, first_pass=not is_fallback)
                advance_progress(row, retry=is_fallback)

        if args.neighbor_retry and failures:
            retry_jobs: list[ProfileRetryJob] = []
            for row_index, col_index in sorted(failures):
                neighbor_candidates = (
                    (row_index, col_index - 1, "rho_in"),
                    (row_index, col_index + 1, "rho_out"),
                    (row_index - 1, col_index, "theta_lower"),
                    (row_index + 1, col_index, "theta_upper"),
                )
                neighbor = None
                for neighbor_i, neighbor_j, source in neighbor_candidates:
                    candidate = (neighbor_i, neighbor_j)
                    if candidate in continuations_by_cell:
                        neighbor = (candidate, source)
                        break
                if neighbor is None:
                    continue
                candidate, source = neighbor
                task = _grid_point_task(
                    args=args,
                    arrays=arrays,
                    row_index=row_index,
                    col_index=col_index,
                    direction="retry",
                    n0=n0,
                    B14=B14,
                    jB_guess=float(
                        continuations_by_cell[candidate]["jB"]
                    ),
                    jB_guess_source=f"retry_{source}_full_profile",
                    analytic_jB_guess=float(
                        arrays["analytic_jB_guess_grid"][
                            row_index,
                            col_index,
                        ]
                    ),
                )
                retry_jobs.append(
                    ProfileRetryJob(
                        task=task,
                        continuation=continuations_by_cell[candidate],
                        attempt_kind="neighbor_full_profile_retry",
                        guess_source=f"retry_{source}_full_profile",
                    )
                )
            if retry_jobs:
                progress.total += len(retry_jobs)
                progress.refresh()
                for job, outcome in _run_unordered(
                    calculate_profile_retry,
                    retry_jobs,
                    executor,
                ):
                    row, bundle = outcome
                    accept(row, first_pass=False)
                    if bool(row.get("success", False)) and bundle is not None:
                        continuations_by_cell[
                            (int(job.task.row_index), int(job.task.col_index))
                        ] = bundle
                    advance_progress(row)
    finally:
        progress.close()
        save()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if manager is not None:
            manager.shutdown()
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
