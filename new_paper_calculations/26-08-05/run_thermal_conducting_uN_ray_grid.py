#!/usr/bin/env python3
"""Thermal-conducting numerical scan in the T versus nB(0-) plane.

The hidden ray transform and physical boundaries come from the state-of-the-
art 26-08-02 numerical-I2 analytical payload. The denser analytical mesh is
interpolated in the physical (nB(0-), T) plane onto every numerical cell.
Thermal-conducting solves start at the actual cell nearest 20 m/s on each ray
and propagate radially. Every new point tries its interpolated analytical jB
first; a failed attempt is retried with the immediately adjacent successful
numerical jB. The solver evolves (nK, jK, T), so the exact T(0+)=0 boundary is
regularized physically by Fourier heat conduction rather than by a nonzero
interface-temperature floor.
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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.interpolate import LinearNDInterpolator
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
DEFAULT_OUTPUT = "thermal-conducting-uN-T-nB-ray-grid.npy"
DEFAULT_RUN_TAG = (
    "thermal_conducting_uN_T_nB_exact_Tinterface0_20260805_v1"
)
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
DEFAULT_JB_BOUNDS = None
DEFAULT_AUTO_JB_BOUNDS = True
DEFAULT_SHOOTING_U_MAX = 1.0e-4
DEFAULT_SHOOTING_JB_LOWER = 1.0e-12
DEFAULT_POINT_TIMEOUT_S = 1800.0

NUMERICAL_CLOSURE_LABEL = "thermal_conducting_nK_jK_T"
NUMERICAL_ROOT_METHOD = "solve_bvp_thermal_conducting_nK_jK_T_parameter_1d"
DENSITY_RATIO_DEFINITION = "lambda_n_equals_nB_0minus_over_nB_inf"
CURRENT_DEFINITION = "u_nK_minus_D_dnK_dx"
RATE_MODEL = "exact_nonleptonic"
ENERGY_CLOSURE = "F_E = h*gamma*u - kappa_th*dT/dx"
NEUTRINO_SOURCE = "Q_nu=0"
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
    jB_bounds: tuple[float, float] | None
    auto_jB_bounds: bool
    shooting_u_max: float
    shooting_jB_lower: float
    point_timeout: float
    jB_guess_source: str
    continuation_direction: str
    analytic_jB_guess: float | None = None
    propagated_jB_guess: float | None = None
    attempt_kind: str = "primary"
    retry_attempt: int = 0
    retry_of: tuple[int, int] | None = None


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
            "thermal-conducting seed cell on each ray"
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
    parser.add_argument(
        "--T_0plus",
        type=float,
        default=DEFAULT_T0PLUS_MEV,
        help="quark-side interface temperature T(0+) in MeV",
    )
    parser.add_argument("--tail-eps", type=float, default=DEFAULT_TAIL_EPS)
    parser.add_argument("--n-mesh", type=int, default=DEFAULT_N_MESH)
    parser.add_argument("--tol-bvp", type=float, default=DEFAULT_TOL_BVP)
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
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
        raise ValueError("T-interface must be nonnegative")
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


def calculate_numerical_point(
    task: NumericalRayPointTask,
) -> dict[str, Any]:
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
            task.attempt_kind == "propagated_fallback"
        ),
        "fallback_success": False,
        "retry_replaced_failure": False,
        "solver_variant": NUMERICAL_CLOSURE_LABEL,
        "root_method": NUMERICAL_ROOT_METHOD,
        "bvp_parameter_count": 0,
        "composition_definition": "nK_over_local_nB",
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "current_definition": CURRENT_DEFINITION,
        "rate_model": RATE_MODEL,
        "energy_closure": ENERGY_CLOSURE,
        "neutrino_source": NEUTRINO_SOURCE,
        "conductivity_model": "",
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
            result = phase_velocity.solve_front_thermal_conducting(
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
                jB_bounds=point_bounds,
                return_profile=False,
                verb=False,
            )
        u_0minus = _as_finite_float(result.get("uN"))
        muB_inf = _as_finite_float(result.get("muB_inf"))
        T_inf = _as_finite_float(result.get("T_inf"))
        A_boundary = _as_finite_float(result.get("A_boundary"))
        a_local_0plus = _as_finite_float(result.get("a_0plus"))
        nB_inf = _as_finite_float(result.get("nB_inf"))
        nK_inf = _as_finite_float(result.get("nK_inf"))
        a_local_end = (
            float(nK_inf / nB_inf)
            if np.isfinite(nK_inf) and np.isfinite(nB_inf) and nB_inf > 0.0
            else np.nan
        )
        stats = result.get("stats", {})
        if not isinstance(stats, dict):
            stats = {}
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
                "a_local_end": a_local_end,
                "a_0plus": _as_finite_float(result.get("a_0plus")),
                "a_0plus_derived": a_local_0plus,
                "a_0plus_max": a_local_0plus,
                "muB_inf_MeV": muB_inf,
                "T_inf_MeV": T_inf,
                "T_0plus_MeV": _as_finite_float(
                    result.get("T_0plus")
                ),
                "nB_inf_MeV3": nB_inf,
                "nK_inf_MeV3": nK_inf,
                "nB_0plus_MeV3": np.nan,
                "nK_0plus_MeV3": _as_finite_float(
                    result.get("nK_0plus")
                ),
                "Tprime_0plus": _as_finite_float(
                    result.get("Tprime_0plus")
                ),
                "q_th_0plus": _as_finite_float(result.get("q_th_0plus")),
                "F_E": _as_finite_float(result.get("F_E")),
                "conductivity_model": str(
                    result.get("conductivity_model", "")
                ),
                "max_knudsen": _as_finite_float(
                    result.get("max_knudsen")
                ),
                "max_kappa_y": _as_finite_float(
                    result.get("max_kappa_y")
                ),
                "max_flux_error_jB": _as_finite_float(
                    result.get("max_flux_error_jB")
                ),
                "max_flux_error_Pi": _as_finite_float(
                    result.get("max_flux_error_Pi")
                ),
                "max_flux_error_F_E": _as_finite_float(
                    result.get("max_flux_error_F_E")
                ),
                "nK_ode_residual_norm": _as_finite_float(
                    result.get("nK_ode_residual_norm")
                ),
                "jK_ode_residual_norm": _as_finite_float(
                    result.get("jK_ode_residual_norm")
                ),
                "T_ode_residual_norm": _as_finite_float(
                    result.get("T_ode_residual_norm")
                ),
                "tail_state_residual_norm": _as_finite_float(
                    result.get("tail_state_residual_norm")
                ),
                "continuation_schedule": result.get(
                    "continuation_schedule", []
                ),
                "continuation_history": result.get(
                    "continuation_history", []
                ),
                "solver_warnings": result.get("warnings", []),
                "root_method": str(
                    result.get("_root_method", NUMERICAL_ROOT_METHOD)
                ),
                "solver_variant": str(
                    result.get(
                        "solver_variant",
                        NUMERICAL_CLOSURE_LABEL,
                    )
                ),
                "bvp_parameter_count": 1,
                "bvp_status": int(result.get("bvp_status", -1)),
                "bvp_message": str(result.get("bvp_message", "")),
                "bvp_niter": int(result.get("bvp_iterations", -1)),
                "bvp_nodes": int(result.get("bvp_nodes", -1)),
                "bvp_ode_calls": int(stats.get("bvp_ode_calls", -1)),
                "bvp_bc_calls": int(stats.get("bvp_bc_calls", -1)),
                "local_root_calls": int(stats.get("local_root_calls", -1)),
                **bounds_diagnostics,
            }
        )
        row["entropy_jump_per_baryon"] = _numerical_entropy_jump(row)
        row["entropy_allowed"] = bool(
            np.isfinite(row["entropy_jump_per_baryon"])
            and row["entropy_jump_per_baryon"] >= 0.0
        )
        row["primary_attempt_success"] = bool(
            task.attempt_kind == "primary" and row["success"]
        )
        row["fallback_success"] = bool(
            task.attempt_kind == "propagated_fallback" and row["success"]
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
        jB_bounds=args.jB_bounds,
        auto_jB_bounds=bool(args.auto_jB_bounds),
        shooting_u_max=float(args.shooting_u_max),
        shooting_jB_lower=float(args.shooting_jB_lower),
        point_timeout=float(args.point_timeout),
        jB_guess_source=str(jB_guess_source),
        continuation_direction=str(direction),
        analytic_jB_guess=analytic_guess,
        propagated_jB_guess=_optional_finite_float(propagated_jB_guess),
        attempt_kind=str(attempt_kind),
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
            jB_guess_source="analytic_2d_cell_nearest_seed_velocity",
            analytic_jB_guess=float(
                arrays["seed_analytic_jB_by_theta"][row_index]
            ),
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
        guess = float(
            arrays["analytic_jB_guess_grid"][row_index, next_index]
        )
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
                jB_guess_source="analytic_2d_numerical_I2_primary",
                analytic_jB_guess=guess,
                propagated_jB_guess=numerical_guess,
            )
        )
    return tasks


def _propagated_fallback_task(
    task: NumericalRayPointTask,
    row: dict[str, Any],
) -> NumericalRayPointTask | None:
    """Retry a failed analytical primary with its immediate ray neighbor."""
    if bool(row.get("success", False)) or task.attempt_kind != "primary":
        return None
    propagated = _optional_finite_float(task.propagated_jB_guess)
    if propagated is None:
        return None
    return replace(
        task,
        jB_guess=propagated,
        jB_guess_source="immediate_same_ray_propagated_fallback",
        attempt_kind="propagated_fallback",
        retry_attempt=int(task.retry_attempt) + 1,
        retry_of=(int(task.row_index), int(task.col_index)),
    )


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
            analytic_jB_guess=float(
                arrays["analytic_jB_guess_grid"][row_index, col_index]
            ),
            propagated_jB_guess=neighbor_jB,
            attempt_kind="neighbor_retry",
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
            "thermal_interface_condition": "T(0+)=T_0plus",
            "energy_closure": ENERGY_CLOSURE,
            "neutrino_source": NEUTRINO_SOURCE,
            "seed_velocity_m_s": float(args.seed_velocity),
            "safety_velocity_m_s": float(args.safety_velocity),
            "seed_strategy": (
                "physical_2d_numerical_I2_analytic_jB_primary_then_"
                "immediate_same_ray_propagated_jB_fallback_from_actual_"
                "cell_nearest_seed_velocity"
            ),
            "analytic_seed_interpolation": (
                "physical_2d_linear_with_hidden_ray_edge_fallback"
            ),
            "analytic_seed_velocity_closure": "numerical_I2",
            "analytic_seed_role": "jB_initial_guess_and_mesh_geometry_only",
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
        "Guess order: physical-2D-interpolated numerical-I2 analytical "
        "jB first, "
        "immediate same-ray numerical jB fallback\n"
        f"T(0+)={args.T_0plus:g} MeV; workers={args.workers}; "
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
    executor = (
        None
        if args.workers == 1
        else ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("spawn"),
        )
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
        desc="Thermal-conducting numerical u(0-) contour",
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
                accept(row, first_pass=task.attempt_kind == "primary")
                fallback = _propagated_fallback_task(task, row)
                if fallback is not None:
                    progress.total += 1
                    progress.refresh()
                    pending_tasks.insert(0, fallback)
                else:
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
                    f"{task.attempt_kind}, "
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
                    accept(row, first_pass=task.attempt_kind == "primary")
                    fallback = _propagated_fallback_task(task, row)
                    if fallback is not None:
                        progress.total += 1
                        progress.refresh()
                        pending[
                            executor.submit(
                                calculate_numerical_point,
                                fallback,
                            )
                        ] = fallback
                    else:
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
                        f"{task.attempt_kind}, "
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
