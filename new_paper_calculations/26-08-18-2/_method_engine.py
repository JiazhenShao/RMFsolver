#!/usr/bin/env python3
"""Shared physics and multiprocessing engine for the T(0+) contour scans.

At identical upstream states this compares

  1. phase_velocity.analytic_velocity_bound          (analytical, xi-dependent)
  2. phase_velocity.semi_analytic_velocity_bound     (semi-analytical, xi-dependent)
  3. phase_velocity.solve_front_energy_conserving_uNmax  (full numerical BVP)

Upstream points come from the semi-analytical numerical-I2 hidden elliptical-
ray safe domain. A few interior ray angles are sampled at fixed radial
fractions, so every point keeps a finite buffer from both the Delta h = 0
inner boundary and the semi-analytical outer domain boundary.

The numerical solver is walked downward through the T_0plus list with
profile continuation, and the middle-radial points are additionally walked
upward so hysteresis between the two sweep directions can be tested.

Nothing is plotted: the run saves one .npy payload holding every raw result,
including successful profiles and all failed attempt records.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import platform
import queue
import signal
import sys
import tempfile
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from contextlib import contextmanager
from datetime import datetime, timezone
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
from scipy.integrate import IntegrationWarning
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    category=IntegrationWarning,
    module=r"RMFsolver\.Solver",
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver import phase_velocity

import _analytic_ray_grid as ray_grid


# --------------------------------------------------------------------------
# Tunable run parameters
# --------------------------------------------------------------------------
DEFAULT_OUTPUT = "T0plus-three-method-contour.npy"
DEFAULT_RUN_TAG = "T0plus_three_method_contour_v1"
DEFAULT_DOMAIN_OUTPUT = "T0plus-contour-domain.npy"
DEFAULT_FULL_ANALYTIC_OUTPUT = "T0plus-full-analytic-contour.npy"
DEFAULT_SEMI_ANALYTIC_OUTPUT = "T0plus-semi-analytic-contour.npy"
DEFAULT_NUMERICAL_OUTPUT = "T0plus-full-numerical-contour.npy"
DOMAIN_MANIFEST_TYPE = "T0plus_semi_analytic_contour_domain_v1"

# Interface temperatures, walked in the order given (downward sweep).
DEFAULT_T_0PLUS_VALUES_MEV = (50.0, 10.0, 1.0, 0.1)
DEFAULT_A_BOUNDARY_T_0PLUS_MEV = 0.0

# Benchmark geometry inside the hidden elliptical-ray safe domain.
DEFAULT_RAY_COUNT = 20
DEFAULT_RADIAL_FRACTIONS = (0.05, 0.15, 0.25, 0.35, 0.45,
                            0.55, 0.65, 0.75, 0.85, 0.95)
DEFAULT_BRANCH_CHECK_RADIAL_FRACTION = 0.55
DEFAULT_ANGULAR_BUFFER_FRACTION = 0.02
DEFAULT_RADIAL_BUFFER_FRACTION = 0.02

# Physics.
DEFAULT_NCRIT_OVER_N0 = 3.0
DEFAULT_XI = -0.5
DEFAULT_MS_MEV = 0.0
DEFAULT_NM_TYPE = "PNM"
DEFAULT_UPB = 5000
DEFAULT_DENSITY_MATCH_RTOL = 1.0e-6

# Safe-domain window used to build the boundary curves.
DEFAULT_DENSITY_MIN_OVER_N0 = 1.0
DEFAULT_DENSITY_MAX_OVER_N0 = 10.0
DEFAULT_TEMPERATURE_MIN_MEV = 1.0e-2
DEFAULT_TEMPERATURE_MAX_MEV = 200.0
DEFAULT_BOUNDARY_THETA_COUNT = 12
DEFAULT_ENDPOINT_PROBE_COUNT = 17
DEFAULT_RAY_PROBE_COUNT = 15
DEFAULT_BISECTION_STEPS = 10

# Numerical BVP defaults.
DEFAULT_TAIL_EPS = 1.0e-8
DEFAULT_N_MESH = 200
DEFAULT_TOL_BVP = 1.0e-4
DEFAULT_MAX_NODES = 10000
DEFAULT_JB_LOWER_BOUND = 1.0e-12
DEFAULT_JB_UPPER_FACTOR = 1.0e-4
DEFAULT_POINT_TIMEOUT_S = 1800.0
DEFAULT_SAVE_EVERY = 1

# Canonical labels (new naming only).
SOLVER_VARIANT = "energy_conserving_nK_fixed_T_0plus"
ROOT_METHOD = "solve_bvp_nK_jK_parameter_jB"
COMPOSITION_DEFINITION = "nK_over_local_nB"
CURRENT_DEFINITION = "u_nK_minus_D_dnK_dx"
METHOD_LABELS = (
    "analytic_velocity_bound",
    "semi_analytic_velocity_bound",
    "solve_front_energy_conserving_uNmax",
)
PROFILE_FIELDS = (
    "x", "s_coord", "nK", "jK", "nB", "a", "u", "muB", "muK", "T_profile",
)
GUESS_SOURCE_SEMI = "semi_analytic_jB"
GUESS_SOURCE_CONTINUATION = "adjacent_numerical_profile"


class PointTimeoutError(TimeoutError):
    """Raised when one numerical BVP attempt exceeds its wall-time limit."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def default_workers() -> int:
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_float_list(text: str) -> tuple[float, ...]:
    parts = [p.strip() for p in str(text).replace(",", " ").split()]
    values = tuple(float(p) for p in parts if p)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def _extract_profile(result: dict[str, Any]) -> dict[str, np.ndarray]:
    """Copy the canonical profile arrays out of a successful numerical result."""
    profile: dict[str, np.ndarray] = {}
    for field in PROFILE_FIELDS:
        value = result.get(field)
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        if array.ndim == 1 and array.size:
            profile[field] = array
    return profile


# --------------------------------------------------------------------------
# Benchmark point selection
# --------------------------------------------------------------------------
def select_benchmark_points(
    *,
    theta_axis_rad: np.ndarray,
    radial_fractions: np.ndarray,
    rho_inner: np.ndarray,
    rho_outer: np.ndarray,
    transform,
    radial_buffer_fraction: float = DEFAULT_RADIAL_BUFFER_FRACTION,
) -> list[dict[str, Any]]:
    """Build the upstream benchmark points on the safe elliptical-ray mesh.

    Every requested radial fraction is squeezed into
    [buffer, 1 - buffer] so each point keeps a finite margin from the
    Delta h = 0 inner boundary and from the outer A boundary.
    """
    theta_axis = np.asarray(theta_axis_rad, dtype=float)
    fractions = np.asarray(radial_fractions, dtype=float)
    if theta_axis.ndim != 1 or theta_axis.size < 1:
        raise ValueError("theta axis must be one-dimensional and nonempty")
    if fractions.ndim != 1 or fractions.size < 1:
        raise ValueError("radial fractions must be one-dimensional and nonempty")
    if not np.all((fractions > 0.0) & (fractions < 1.0)):
        raise ValueError("radial fractions must lie strictly inside (0, 1)")
    buffer = float(radial_buffer_fraction)
    if not 0.0 <= buffer < 0.5:
        raise ValueError("radial buffer fraction must lie in [0, 0.5)")

    squeezed = buffer + fractions * (1.0 - 2.0 * buffer)
    mesh = ray_grid.build_safe_ray_mesh(
        transform=transform,
        theta_axis_rad=theta_axis,
        radial_fraction_axis=squeezed,
        rho_inner=np.asarray(rho_inner, dtype=float),
        rho_outer=np.asarray(rho_outer, dtype=float),
    )
    points: list[dict[str, Any]] = []
    for i in range(theta_axis.size):
        for j in range(squeezed.size):
            points.append(
                {
                    "point_index": len(points),
                    "theta_index": int(i),
                    "radial_index": int(j),
                    "theta_rad": float(theta_axis[i]),
                    "theta_deg": float(np.degrees(theta_axis[i])),
                    "radial_fraction": float(fractions[j]),
                    "radial_fraction_used": float(squeezed[j]),
                    "rho": float(mesh["rho_grid"][i, j]),
                    "rho_inner_boundary": float(rho_inner[i]),
                    "rho_outer_boundary": float(rho_outer[i]),
                    "nB_0minus_over_n0": float(mesh["nB_grid_over_n0"][i, j]),
                    "T_0minus_MeV": float(mesh["temperature_grid_MeV"][i, j]),
                }
            )
    return points


def interior_theta_axis(
    boundary_payload: dict[str, Any],
    ray_count: int,
    angular_buffer_fraction: float,
) -> np.ndarray:
    """Pick interior ray angles spanning the allowed angular window.

    The window is taken from the solved boundary axis itself, and an angular
    buffer is trimmed off both ends so no benchmark ray sits on the edge where
    the inner and outer boundary curves are least reliable.
    """
    theta_axis = np.asarray(boundary_payload["theta_axis_rad"], dtype=float)
    finite = theta_axis[np.isfinite(theta_axis)]
    if finite.size < 2:
        raise RuntimeError("boundary payload has a degenerate angular window")
    theta_min = float(np.min(finite))
    theta_max = float(np.max(finite))
    if theta_max <= theta_min:
        raise RuntimeError("boundary payload has a degenerate angular window")
    span = theta_max - theta_min
    buffer = float(angular_buffer_fraction) * span
    lo, hi = theta_min + buffer, theta_max - buffer
    if hi <= lo:
        raise RuntimeError("angular buffer removed the whole window")
    count = int(ray_count)
    if count < 1:
        raise ValueError("ray count must be positive")
    if count == 1:
        return np.array([0.5 * (lo + hi)], dtype=float)
    return np.linspace(lo, hi, count)


def common_boundary_geometry(
    full_boundary: dict[str, Any],
    semi_boundary: dict[str, Any],
    *,
    ray_count: int,
    angular_buffer_fraction: float,
) -> dict[str, np.ndarray]:
    """Return the radial intersection of two method-specific safe domains."""
    full_theta = np.asarray(full_boundary["theta_axis_rad"], dtype=float)
    semi_theta = np.asarray(semi_boundary["theta_axis_rad"], dtype=float)
    theta_min = max(float(np.nanmin(full_theta)), float(np.nanmin(semi_theta)))
    theta_max = min(float(np.nanmax(full_theta)), float(np.nanmax(semi_theta)))
    if not np.isfinite(theta_min + theta_max) or theta_max <= theta_min:
        raise RuntimeError("analytical safe domains have no common angular interval")
    buffer = float(angular_buffer_fraction) * (theta_max - theta_min)
    lower = theta_min + buffer
    upper = theta_max - buffer
    if upper <= lower:
        raise RuntimeError("angular buffer removed the common safe domain")
    count = int(ray_count)
    if count < 1:
        raise ValueError("ray count must be positive")
    theta_axis = (
        np.array([0.5 * (lower + upper)], dtype=float)
        if count == 1
        else np.linspace(lower, upper, count)
    )
    full_inner = np.interp(
        theta_axis,
        full_theta,
        np.asarray(full_boundary["rho_inner"], dtype=float),
    )
    semi_inner = np.interp(
        theta_axis,
        semi_theta,
        np.asarray(semi_boundary["rho_inner"], dtype=float),
    )
    full_outer = np.interp(
        theta_axis,
        full_theta,
        np.asarray(full_boundary["rho_outer"], dtype=float),
    )
    semi_outer = np.interp(
        theta_axis,
        semi_theta,
        np.asarray(semi_boundary["rho_outer"], dtype=float),
    )
    common_inner = np.maximum(full_inner, semi_inner)
    common_outer = np.minimum(full_outer, semi_outer)
    if not np.all(np.isfinite(common_inner)) or not np.all(np.isfinite(common_outer)):
        raise RuntimeError("common radial boundary contains non-finite values")
    if np.any(common_outer <= common_inner):
        bad = np.flatnonzero(common_outer <= common_inner)
        raise RuntimeError(f"common safe radial interval collapsed on rays {bad.tolist()}")
    return {
        "theta_axis_rad": theta_axis,
        "rho_inner": common_inner,
        "rho_outer": common_outer,
        "rho_inner_full_analytic": full_inner,
        "rho_inner_semi_analytic": semi_inner,
        "rho_outer_full_analytic": full_outer,
        "rho_outer_semi_analytic": semi_outer,
    }


def semi_boundary_geometry(
    semi_boundary: dict[str, Any],
    *,
    ray_count: int,
    angular_buffer_fraction: float,
) -> dict[str, Any]:
    """Interpolate the semi-analytical safe domain onto interior rays."""
    source_theta = np.asarray(semi_boundary["theta_axis_rad"], dtype=float)
    theta_axis = interior_theta_axis(
        semi_boundary,
        ray_count=int(ray_count),
        angular_buffer_fraction=float(angular_buffer_fraction),
    )
    rho_inner = np.interp(
        theta_axis,
        source_theta,
        np.asarray(semi_boundary["rho_inner"], dtype=float),
    )
    rho_outer = np.interp(
        theta_axis,
        source_theta,
        np.asarray(semi_boundary["rho_outer"], dtype=float),
    )
    if not np.all(np.isfinite(rho_inner)) or not np.all(np.isfinite(rho_outer)):
        raise RuntimeError("semi-analytical radial boundary contains non-finite values")
    if np.any(rho_outer <= rho_inner):
        bad = np.flatnonzero(rho_outer <= rho_inner)
        raise RuntimeError(
            f"semi-analytical safe radial interval collapsed on rays {bad.tolist()}"
        )
    return {
        "selection_rule": "semi_analytic_safe_domain",
        "theta_axis_rad": theta_axis,
        "rho_inner": rho_inner,
        "rho_outer": rho_outer,
        "rho_outer_semi_analytic": rho_outer.copy(),
    }


def validate_domain_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate that all method runners share the intended safe-domain file."""
    if not isinstance(manifest, dict):
        raise ValueError("domain manifest must be a dictionary payload")
    if manifest.get("manifest_type") != DOMAIN_MANIFEST_TYPE:
        raise ValueError("domain manifest has an unsupported type")
    if manifest.get("selection_rule") != "semi_analytic_safe_domain":
        raise ValueError("domain manifest must use the semi-analytical safe domain")
    points = manifest.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("domain manifest contains no benchmark points")
    temperatures = np.asarray(manifest.get("T_0plus_values_MeV", []), dtype=float)
    if temperatures.ndim != 1 or temperatures.size < 1 or not np.all(np.isfinite(temperatures)):
        raise ValueError("domain manifest has invalid T_0plus values")
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("domain manifest is missing calibration metadata")
    for key in ("n0_MeV3", "B14_MeV"):
        if not np.isfinite(float(calibration.get(key, np.nan))):
            raise ValueError(f"domain manifest calibration is missing {key}")
    expected_temperatures = {float(value) for value in temperatures}
    for point in points:
        states = point.get("states")
        if not isinstance(states, list) or len(states) != temperatures.size:
            raise ValueError("each benchmark trajectory must contain one state per T_0plus")
        state_temperatures = {float(state["T_0plus_MeV"]) for state in states}
        if state_temperatures != expected_temperatures:
            raise ValueError("benchmark trajectory T_0plus states do not match the manifest")
        for state in states:
            values = np.asarray(
                [
                    state.get("T_0minus_MeV"),
                    state.get("nB_0minus_over_n0"),
                    state.get("rho"),
                    state.get("rho_inner_boundary"),
                    state.get("rho_outer_boundary"),
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(values)):
                raise ValueError("benchmark trajectory contains non-finite geometry")
            if not values[3] < values[2] < values[4]:
                raise ValueError("benchmark point is not strictly inside its safe interval")
    return manifest


# --------------------------------------------------------------------------
# Analytical / semi-analytical evaluation
# --------------------------------------------------------------------------
def upstream_muB(
    nB_0minus_MeV3: float,
    T_0minus_MeV: float,
    nm_type: str,
    density_match_rtol: float = DEFAULT_DENSITY_MATCH_RTOL,
) -> tuple[float, float]:
    """Invert nB -> muB on the physical branch and verify the density."""
    muB = float(
        phase_velocity.muB_from_nB_physical(
            float(nB_0minus_MeV3),
            float(T_0minus_MeV),
            param=para.paraQMCRMF3,
            NM_type=str(nm_type),
            auto_expand=True,
        )
    )
    achieved = float(
        phase_velocity.nB_NM(
            muB,
            float(T_0minus_MeV),
            param=para.paraQMCRMF3,
            NM_type=str(nm_type),
        )
    )
    rel = abs(achieved - float(nB_0minus_MeV3)) / max(abs(float(nB_0minus_MeV3)), 1.0e-300)
    if rel > float(density_match_rtol):
        raise RuntimeError(
            f"muB inversion missed the requested density: requested "
            f"nB={float(nB_0minus_MeV3):.6e}, muB={muB:.4f} gives {achieved:.6e} "
            f"(relative error {rel:.3e})"
        )
    return muB, rel


def evaluate_analytical_point(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate both analytical methods at one upstream point for all T_0plus."""
    T_0minus = float(task["T_0minus_MeV"])
    nB_MeV3 = float(task["nB_0minus_MeV3"])
    rows: list[dict[str, Any]] = []
    try:
        muB_0minus, density_rel = upstream_muB(
            nB_MeV3, T_0minus, str(task["nm_type"]), float(task["density_match_rtol"])
        )
    except Exception as exc:
        for T_0plus in task["T_0plus_values"]:
            rows.append(
                {
                    "point_index": int(task["point_index"]),
                    "T_0plus_MeV": float(T_0plus),
                    "muB_0minus_MeV": np.nan,
                    "density_relative_error": np.nan,
                    "analytic": {"success": False, "message": str(exc)},
                    "semi_analytic": {"success": False, "message": str(exc)},
                }
            )
        return {"point_index": int(task["point_index"]), "rows": rows}

    for T_0plus in task["T_0plus_values"]:
        row: dict[str, Any] = {
            "point_index": int(task["point_index"]),
            "T_0plus_MeV": float(T_0plus),
            "muB_0minus_MeV": float(muB_0minus),
            "density_relative_error": float(density_rel),
        }
        for label, fn in (
            ("analytic", phase_velocity.analytic_velocity_bound),
            ("semi_analytic", phase_velocity.semi_analytic_velocity_bound),
        ):
            started = time.monotonic()
            try:
                result = fn(
                    muB_0minus,
                    T_0minus,
                    float(task["B14_MeV"]),
                    T_0plus=float(T_0plus),
                    xi=float(task["xi"]),
                    ms=float(task["ms_MeV"]),
                    param=para.paraQMCRMF3,
                    NM_type=str(task["nm_type"]),
                    upB=int(task["upB"]),
                )
                row[label] = {
                    "success": True,
                    "message": "",
                    "u_0minus": _as_finite_float(result.get("u_0minus")),
                    "jB": _as_finite_float(result.get("jB")),
                    "a_0plus": _as_finite_float(result.get("a_0plus")),
                    "a_inf": _as_finite_float(result.get("a_inf")),
                    "lambda_n": _as_finite_float(result.get("lambda_n")),
                    "T_inf_MeV": _as_finite_float(result.get("T_inf")),
                    "A_boundary": _as_finite_float(result.get("A_boundary")),
                    "runtime_s": float(time.monotonic() - started),
                }
            except Exception as exc:
                row[label] = {
                    "success": False,
                    "message": str(exc),
                    "u_0minus": np.nan,
                    "jB": np.nan,
                    "a_0plus": np.nan,
                    "a_inf": np.nan,
                    "lambda_n": np.nan,
                    "T_inf_MeV": np.nan,
                    "A_boundary": np.nan,
                    "runtime_s": float(time.monotonic() - started),
                }
        rows.append(row)
    return {"point_index": int(task["point_index"]), "rows": rows}


def evaluate_selected_analytical_point(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate exactly one analytical method at one upstream point."""
    method = str(task["method"])
    methods = {
        "analytic": phase_velocity.analytic_velocity_bound,
        "semi_analytic": phase_velocity.semi_analytic_velocity_bound,
    }
    try:
        function = methods[method]
    except KeyError as exc:
        raise ValueError("method must be 'analytic' or 'semi_analytic'") from exc

    T_0minus = float(task["T_0minus_MeV"])
    nB_MeV3 = float(task["nB_0minus_MeV3"])
    rows: list[dict[str, Any]] = []
    try:
        muB_0minus, density_rel = upstream_muB(
            nB_MeV3,
            T_0minus,
            str(task["nm_type"]),
            float(task["density_match_rtol"]),
        )
    except Exception as exc:
        for T_0plus in task["T_0plus_values"]:
            rows.append(
                {
                    "point_index": int(task["point_index"]),
                    "T_0plus_MeV": float(T_0plus),
                    "muB_0minus_MeV": np.nan,
                    "density_relative_error": np.nan,
                    "success": False,
                    "message": str(exc),
                    "runtime_s": 0.0,
                }
            )
        return {"point_index": int(task["point_index"]), "method": method, "rows": rows}

    for T_0plus in task["T_0plus_values"]:
        started = time.monotonic()
        row = {
            "point_index": int(task["point_index"]),
            "T_0plus_MeV": float(T_0plus),
            "muB_0minus_MeV": float(muB_0minus),
            "density_relative_error": float(density_rel),
        }
        try:
            result = function(
                muB_0minus,
                T_0minus,
                float(task["B14_MeV"]),
                T_0plus=float(T_0plus),
                xi=float(task["xi"]),
                ms=float(task["ms_MeV"]),
                param=para.paraQMCRMF3,
                NM_type=str(task["nm_type"]),
                upB=int(task["upB"]),
            )
            row.update(
                {
                    "success": True,
                    "message": "",
                    "u_0minus": _as_finite_float(result.get("u_0minus")),
                    "jB": _as_finite_float(result.get("jB")),
                    "a_0plus": _as_finite_float(result.get("a_0plus")),
                    "a_inf": _as_finite_float(result.get("a_inf")),
                    "lambda_n": _as_finite_float(result.get("lambda_n")),
                    "T_inf_MeV": _as_finite_float(result.get("T_inf")),
                    "A_boundary": _as_finite_float(result.get("A_boundary")),
                    "runtime_s": float(time.monotonic() - started),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "success": False,
                    "message": str(exc),
                    "u_0minus": np.nan,
                    "jB": np.nan,
                    "a_0plus": np.nan,
                    "a_inf": np.nan,
                    "lambda_n": np.nan,
                    "T_inf_MeV": np.nan,
                    "A_boundary": np.nan,
                    "runtime_s": float(time.monotonic() - started),
                }
            )
        rows.append(row)
    return {"point_index": int(task["point_index"]), "method": method, "rows": rows}


# --------------------------------------------------------------------------
# Numerical trajectory with profile continuation
# --------------------------------------------------------------------------
def continuation_attempt_plan(
    *,
    step_index: int,
    previous_succeeded: bool,
    semi_jB: float,
) -> list[dict[str, Any]]:
    """Attempt order for one T_0plus step.

    The first temperature is seeded from the same-point semi-analytical jB.
    Later temperatures try the immediately previous successful numerical
    profile first, then fall back once to the current semi-analytical jB with
    no profile. If the immediately adjacent temperature failed there is no
    profile to continue from, so only the semi-analytical seed is used: a more
    distant profile is never reused.
    """
    semi_attempt = {
        "guess_source": GUESS_SOURCE_SEMI,
        "use_continuation": False,
        "jB_guess": float(semi_jB) if np.isfinite(semi_jB) else None,
    }
    if int(step_index) == 0 or not bool(previous_succeeded):
        return [semi_attempt]
    return [
        {
            "guess_source": GUESS_SOURCE_CONTINUATION,
            "use_continuation": True,
            "jB_guess": None,
        },
        semi_attempt,
    ]


def attempts_per_step(step_index: int, previous_succeeded: bool) -> int:
    """Maximum attempts a step may consume (used for honest progress totals)."""
    return len(
        continuation_attempt_plan(
            step_index=int(step_index),
            previous_succeeded=bool(previous_succeeded),
            semi_jB=1.0,
        )
    )


@contextmanager
def point_timeout(seconds: float | None):
    """Interrupt one stuck numerical attempt on POSIX worker processes."""
    if seconds is None or not np.isfinite(seconds) or seconds <= 0.0:
        yield
        return
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def handle_timeout(signum, frame):
        del signum, frame
        raise PointTimeoutError(f"numerical attempt exceeded {float(seconds):g} s")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, previous_delay, previous_interval)
        signal.signal(signal.SIGALRM, previous_handler)


def _emit_progress(task: dict[str, Any], event: str, record=None) -> None:
    progress_queue = task.get("progress_queue")
    if progress_queue is None:
        return
    payload = {
        "event": str(event),
        "point_index": int(task["point_index"]),
        "sweep_direction": str(task["sweep_direction"]),
    }
    if record is not None:
        payload["record"] = {
            key: value
            for key, value in dict(record).items()
            if key not in {"traceback"}
        }
    progress_queue.put(payload)


def run_numerical_trajectory(task: dict[str, Any]) -> dict[str, Any]:
    """Walk one upstream point through its T_0plus list, in one process.

    Keeping the whole trajectory in a single worker is what lets the full BVP
    profile be handed to the next temperature through continuation_guess.
    """
    B14 = float(task["B14_MeV"])
    timeout_s = float(task["point_timeout_s"])
    semi_jB_by_T = {float(k): float(v) for k, v in dict(task["semi_jB_by_T"]).items()}
    states_by_T = {
        float(key): dict(value)
        for key, value in dict(task.get("states_by_T0plus", {})).items()
    }

    records: list[dict[str, Any]] = []
    profiles: dict[float, dict[str, np.ndarray]] = {}
    previous_result: dict[str, Any] | None = None
    previous_succeeded = False
    started_point = time.monotonic()

    for step_index, T_0plus in enumerate(task["T_0plus_values"]):
        T_0plus = float(T_0plus)
        state = states_by_T.get(T_0plus, {})
        T_0minus = float(state.get("T_0minus_MeV", task["T_0minus_MeV"]))
        nB_0minus = float(state.get("nB_0minus_MeV3", task["nB_0minus_MeV3"]))
        jB_bounds = (
            float(task["jB_lower_bound"]),
            float(nB_0minus * float(task["jB_upper_factor"])),
        )
        semi_jB = semi_jB_by_T.get(T_0plus, np.nan)
        plan = continuation_attempt_plan(
            step_index=step_index,
            previous_succeeded=previous_succeeded,
            semi_jB=semi_jB,
        )
        step_succeeded = False
        for attempt_index, attempt in enumerate(plan):
            if attempt_index > 0:
                _emit_progress(task, "retry_scheduled")
            kwargs: dict[str, Any] = dict(
                T_0minus=T_0minus,
                nB_0minus=nB_0minus,
                B_one_forth=B14,
                T_0plus=T_0plus,
                ms=float(task["ms_MeV"]),
                param=para.paraQMCRMF3,
                NM_type=str(task["nm_type"]),
                tail_eps=float(task["tail_eps"]),
                n_mesh=int(task["n_mesh"]),
                tol_bvp=float(task["tol_bvp"]),
                max_nodes=int(task["max_nodes"]),
                jB_bounds=jB_bounds,
                return_profile=True,
                verb=False,
            )
            if attempt["use_continuation"] and previous_result is not None:
                kwargs["continuation_guess"] = previous_result
                kwargs["jB_guess"] = _as_finite_float(previous_result.get("jB"))
            elif attempt["jB_guess"] is not None:
                kwargs["jB_guess"] = float(attempt["jB_guess"])

            started = time.monotonic()
            record: dict[str, Any] = {
                "point_index": int(task["point_index"]),
                "sweep_direction": str(task["sweep_direction"]),
                "T_0plus_MeV": T_0plus,
                "T_0minus_MeV": T_0minus,
                "nB_0minus_MeV3": nB_0minus,
                "nB_0minus_over_n0": _as_finite_float(
                    state.get("nB_0minus_over_n0")
                ),
                "attempt_index": int(attempt_index),
                "attempt_count": int(attempt_index + 1),
                "guess_source": str(attempt["guess_source"]),
                "is_primary_attempt": bool(attempt_index == 0),
                "retry_attempt": int(attempt_index),
                "retry_of": (
                    (int(task["point_index"]), T_0plus)
                    if attempt_index > 0
                    else None
                ),
                "continuation_direction": str(task["sweep_direction"]),
                "continuation_profile_used": bool(attempt["use_continuation"]),
                "semi_analytic_jB_seed": _as_finite_float(semi_jB),
                "jB_bounds_used": jB_bounds,
            }
            try:
                with point_timeout(timeout_s):
                    result = phase_velocity.solve_front_energy_conserving_uNmax(**kwargs)
                runtime = float(time.monotonic() - started)
                success = bool(result.get("success", True))
                record.update(
                    {
                        "success": success,
                        "retry_replaced_failure": bool(attempt_index > 0 and success),
                        "status": "success" if success else "solver_reported_failure",
                        "message": str(result.get("message", "")),
                        "runtime_s": runtime,
                        "u_0minus": _as_finite_float(result.get("u_0minus")),
                        "jB": _as_finite_float(result.get("jB")),
                        "a_0plus": _as_finite_float(result.get("a_0plus")),
                        "a_inf": _as_finite_float(result.get("a_inf")),
                        "lambda_n": _as_finite_float(result.get("lambda_n")),
                        "T_inf_MeV": _as_finite_float(result.get("T_inf")),
                        "bvp_status": result.get("bvp_status"),
                        "bvp_message": str(result.get("bvp_message", "")),
                        "bvp_nodes": result.get("bvp_nodes"),
                        "bvp_niter": result.get("bvp_niter"),
                        "bvp_parameters": np.asarray(
                            result.get("bvp_parameters", []), dtype=float
                        ),
                        "bvp_max_rms_residual": _as_finite_float(
                            result.get("bvp_max_rms_residual")
                        ),
                        "bvp_collocation_converged": result.get(
                            "bvp_collocation_converged"
                        ),
                        "E_right_residual_norm": _as_finite_float(
                            result.get("E_right_residual_norm")
                        ),
                        "Pi_right_residual_norm": _as_finite_float(
                            result.get("Pi_right_residual_norm")
                        ),
                        "T_0plus_residual": _as_finite_float(
                            result.get("T_0plus_residual")
                        ),
                        "tail_residual_norm": _as_finite_float(
                            result.get("tail_residual_norm")
                        ),
                        "kaon_equation_residual_norm": _as_finite_float(
                            result.get("kaon_equation_residual_norm")
                        ),
                        "energy_equation_residual_norm": _as_finite_float(
                            result.get("energy_equation_residual_norm")
                        ),
                        "momentum_equation_residual_norm": _as_finite_float(
                            result.get("momentum_equation_residual_norm")
                        ),
                        "solver_variant": str(result.get("solver_variant", SOLVER_VARIANT)),
                        "root_method": str(result.get("_root_method", ROOT_METHOD)),
                    }
                )
                records.append(record)
                _emit_progress(task, "attempt_completed", record)
                if success:
                    profiles[T_0plus] = _extract_profile(result)
                    previous_result = result
                    previous_succeeded = True
                    step_succeeded = True
                    break
            except Exception as exc:
                record.update(
                    {
                        "success": False,
                        "retry_replaced_failure": False,
                        "status": (
                            "timeout" if isinstance(exc, PointTimeoutError) else "exception"
                        ),
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=4),
                        "runtime_s": float(time.monotonic() - started),
                    }
                )
                records.append(record)
                _emit_progress(task, "attempt_completed", record)

        if not step_succeeded:
            previous_succeeded = False
            previous_result = None

    return {
        "point_index": int(task["point_index"]),
        "sweep_direction": str(task["sweep_direction"]),
        "records": records,
        "profiles": profiles,
        "runtime_s": float(time.monotonic() - started_point),
    }


# --------------------------------------------------------------------------
# Payload assembly and atomic save
# --------------------------------------------------------------------------
def _method_arrays(
    analytical_rows: list[dict[str, Any]],
    numerical_records: list[dict[str, Any]],
    n_points: int,
    T_values: tuple[float, ...],
    sweep: str,
) -> dict[str, np.ndarray]:
    shape = (n_points, len(T_values))
    index_of_T = {float(T): i for i, T in enumerate(T_values)}
    out: dict[str, np.ndarray] = {}
    for label in ("analytic", "semi_analytic"):
        for field, key in (
            ("u_0minus", "u_0minus"), ("jB", "jB"), ("a_0plus", "a_0plus"),
            ("a_inf", "a_inf"), ("lambda_n", "lambda_n"), ("T_inf_MeV", "T_inf_MeV"),
            ("A_boundary", "A_boundary"), ("runtime_s", "runtime_s"),
        ):
            out[f"{label}_{field}"] = np.full(shape, np.nan)
        out[f"{label}_success"] = np.zeros(shape, dtype=bool)
        out[f"{label}_message"] = np.full(shape, "", dtype=object)
    for row in analytical_rows:
        i = int(row["point_index"])
        j = index_of_T.get(float(row["T_0plus_MeV"]))
        if j is None or i >= n_points:
            continue
        for label in ("analytic", "semi_analytic"):
            blob = row.get(label, {})
            out[f"{label}_success"][i, j] = bool(blob.get("success", False))
            out[f"{label}_message"][i, j] = str(blob.get("message", ""))
            for field in (
                "u_0minus", "jB", "a_0plus", "a_inf", "lambda_n",
                "T_inf_MeV", "A_boundary", "runtime_s",
            ):
                out[f"{label}_{field}"][i, j] = _as_finite_float(blob.get(field))

    num_fields = (
        "u_0minus",
        "jB",
        "a_0plus",
        "a_inf",
        "lambda_n",
        "T_inf_MeV",
        "bvp_max_rms_residual",
        "runtime_s",
        "T_0plus_residual",
        "E_right_residual_norm",
        "Pi_right_residual_norm",
        "tail_residual_norm",
        "kaon_equation_residual_norm",
        "energy_equation_residual_norm",
        "momentum_equation_residual_norm",
    )
    for field in num_fields:
        out[f"numerical_{sweep}_{field}"] = np.full(shape, np.nan)
    out[f"numerical_{sweep}_success"] = np.zeros(shape, dtype=bool)
    out[f"numerical_{sweep}_attempts"] = np.zeros(shape, dtype=int)
    out[f"numerical_{sweep}_bvp_nodes"] = np.full(shape, -1, dtype=int)
    out[f"numerical_{sweep}_bvp_niter"] = np.full(shape, -1, dtype=int)
    out[f"numerical_{sweep}_bvp_status"] = np.full(shape, -1, dtype=int)
    out[f"numerical_{sweep}_bvp_collocation_converged"] = np.zeros(
        shape, dtype=bool
    )
    bvp_parameters = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        bvp_parameters[index] = np.array([], dtype=float)
    status = np.empty(shape, dtype=object)
    message = np.empty(shape, dtype=object)
    guess_source = np.empty(shape, dtype=object)
    status[:] = "unscanned"
    message[:] = ""
    guess_source[:] = ""
    for record in numerical_records:
        i = int(record["point_index"])
        j = index_of_T.get(float(record["T_0plus_MeV"]))
        if j is None or i >= n_points:
            continue
        out[f"numerical_{sweep}_attempts"][i, j] += 1
        already = bool(out[f"numerical_{sweep}_success"][i, j])
        if already:
            continue
        status[i, j] = str(record.get("status", "unknown"))
        message[i, j] = str(record.get("message", ""))
        guess_source[i, j] = str(record.get("guess_source", ""))
        if record.get("success"):
            out[f"numerical_{sweep}_success"][i, j] = True
            for field in num_fields:
                out[f"numerical_{sweep}_{field}"][i, j] = _as_finite_float(record.get(field))
            for field in ("bvp_nodes", "bvp_niter"):
                value = record.get(field)
                try:
                    out[f"numerical_{sweep}_{field}"][i, j] = int(value)
                except Exception:
                    pass
            try:
                out[f"numerical_{sweep}_bvp_status"][i, j] = int(
                    record.get("bvp_status")
                )
            except Exception:
                pass
            out[f"numerical_{sweep}_bvp_collocation_converged"][i, j] = bool(
                record.get("bvp_collocation_converged", False)
            )
            bvp_parameters[i, j] = np.asarray(
                record.get("bvp_parameters", []), dtype=float
            )
    out[f"numerical_{sweep}_status"] = status
    out[f"numerical_{sweep}_message"] = message
    out[f"numerical_{sweep}_guess_source"] = guess_source
    out[f"numerical_{sweep}_bvp_parameters"] = bvp_parameters
    return out


def build_payload(
    *,
    args: argparse.Namespace,
    calibration: dict[str, Any],
    points: list[dict[str, Any]],
    T_values: tuple[float, ...],
    analytical_rows: list[dict[str, Any]],
    downward_records: list[dict[str, Any]],
    downward_profiles: dict[int, dict[float, dict[str, np.ndarray]]],
    upward_records: list[dict[str, Any]],
    upward_profiles: dict[int, dict[float, dict[str, np.ndarray]]],
    boundary_payload: dict[str, Any],
    started_at: str,
    run_complete: bool = False,
) -> dict[str, Any]:
    n_points = len(points)
    payload: dict[str, Any] = {
        "run_tag": str(args.run_tag),
        "script": Path(__file__).name,
        "started_at_utc": started_at,
        "checkpoint_at_utc": _utc_now(),
        "finished_at_utc": _utc_now() if run_complete else None,
        "run_complete": bool(run_complete),
        "completed_downward_point_count": int(len(downward_profiles)),
        "completed_upward_point_count": int(len(upward_profiles)),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "arguments": vars(args),
        "method_labels": list(METHOD_LABELS),
        "xi": float(args.xi),
        "ms_MeV": float(args.ms),
        "NM_type": str(args.nm_type),
        "upB": int(args.upB),
        "solver_variant": SOLVER_VARIANT,
        "root_method": ROOT_METHOD,
        "composition_definition": COMPOSITION_DEFINITION,
        "current_definition": CURRENT_DEFINITION,
        "T_0plus_values_MeV": np.asarray(T_values, dtype=float),
        "boundary_reference_T_0plus_MeV": DEFAULT_A_BOUNDARY_T_0PLUS_MEV,
        **{f"calibration_{k}": v for k, v in calibration.items()},
        # upstream coordinates
        "point_index": np.arange(n_points, dtype=int),
        "T_0minus_MeV": np.asarray([p["T_0minus_MeV"] for p in points], dtype=float),
        "nB_0minus_over_n0": np.asarray(
            [p["nB_0minus_over_n0"] for p in points], dtype=float
        ),
        "nB_0minus_MeV3": np.asarray(
            [p["nB_0minus_over_n0"] * calibration["n0_MeV3"] for p in points], dtype=float
        ),
        "muB_0minus_MeV": np.full(n_points, np.nan),
        "density_relative_error": np.full(n_points, np.nan),
        # ray diagnostics
        "ray_theta_rad": np.asarray([p["theta_rad"] for p in points], dtype=float),
        "ray_theta_deg": np.asarray([p["theta_deg"] for p in points], dtype=float),
        "ray_radial_fraction": np.asarray(
            [p["radial_fraction"] for p in points], dtype=float
        ),
        "ray_radial_fraction_used": np.asarray(
            [p["radial_fraction_used"] for p in points], dtype=float
        ),
        "ray_rho": np.asarray([p["rho"] for p in points], dtype=float),
        "ray_rho_inner_boundary": np.asarray(
            [p["rho_inner_boundary"] for p in points], dtype=float
        ),
        "ray_rho_outer_boundary": np.asarray(
            [p["rho_outer_boundary"] for p in points], dtype=float
        ),
        "ray_rho_outer_boundary_semi_analytic": np.asarray(
            [p["rho_outer_boundary_semi_analytic"] for p in points], dtype=float
        ),
        "boundary_diagnostics": boundary_payload,
        # raw records
        "analytical_rows": analytical_rows,
        "numerical_downward_records": downward_records,
        "numerical_upward_records": upward_records,
        "numerical_downward_profiles": downward_profiles,
        "numerical_upward_profiles": upward_profiles,
        "profile_fields": list(PROFILE_FIELDS),
    }
    for row in analytical_rows:
        i = int(row["point_index"])
        if i < n_points:
            payload["muB_0minus_MeV"][i] = _as_finite_float(row.get("muB_0minus_MeV"))
            payload["density_relative_error"][i] = _as_finite_float(
                row.get("density_relative_error")
            )
    payload.update(
        _method_arrays(analytical_rows, downward_records, n_points, T_values, "downward")
    )
    payload.update(
        _method_arrays([], upward_records, n_points, T_values, "upward")
    )
    # hysteresis diagnostics
    down = payload["numerical_downward_u_0minus"]
    up = payload["numerical_upward_u_0minus"]
    both = payload["numerical_downward_success"] & payload["numerical_upward_success"]
    delta = np.full(down.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(up - down) / np.maximum(np.abs(down), np.abs(up))
    delta[both] = rel[both]
    payload["hysteresis_relative_difference"] = delta
    payload["hysteresis_compared_count"] = int(np.count_nonzero(both))
    payload["hysteresis_max_relative_difference"] = (
        float(np.nanmax(delta)) if np.any(both) else np.nan
    )
    # unit conversion
    for sweep in ("downward", "upward"):
        payload[f"numerical_{sweep}_u_0minus_m_s"] = (
            payload[f"numerical_{sweep}_u_0minus"] * const.m_per_sec
        )
    for label in ("analytic", "semi_analytic"):
        payload[f"{label}_u_0minus_m_s"] = payload[f"{label}_u_0minus"] * const.m_per_sec
    return payload


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, payload, allow_pickle=True)
    temporary.replace(path)


def resolve_output_path(value: str | Path) -> Path:
    """Never overwrite older data: append a numeric suffix if needed."""
    path = Path(value)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    index = 1
    while True:
        candidate = path.with_name(f"{stem}__run{index:02d}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def collect_trajectory_results(
    futures,
    progress_queue,
    *,
    base_attempt_count: int,
    description: str,
    leave: bool,
    completed_callback=None,
) -> list[dict[str, Any]]:
    """Collect trajectory futures while streaming every attempt to one tqdm."""
    pending = set(futures)
    outputs: list[dict[str, Any]] = []
    with tqdm(
        total=int(base_attempt_count),
        desc=str(description),
        unit="attempt",
        leave=bool(leave),
        dynamic_ncols=True,
    ) as progress:
        while pending:
            try:
                event = progress_queue.get(timeout=0.25)
            except queue.Empty:
                event = None
            if event is not None:
                if event.get("event") == "retry_scheduled":
                    progress.total += 1
                    progress.refresh()
                elif event.get("event") == "attempt_completed":
                    progress.update(1)
                    record = event.get("record", {})
                    progress.set_postfix_str(
                        f"p={event.get('point_index')}, "
                        f"T0+={record.get('T_0plus_MeV', np.nan):g}, "
                        f"{record.get('status', 'unknown')}"
                    )
            completed, pending = wait(
                pending,
                timeout=0.0,
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                output = future.result()
                outputs.append(output)
                if completed_callback is not None:
                    completed_callback(output, len(outputs))
        while True:
            try:
                event = progress_queue.get_nowait()
            except queue.Empty:
                break
            if event.get("event") == "retry_scheduled":
                progress.total += 1
            elif event.get("event") == "attempt_completed":
                progress.update(1)
        progress.refresh()
    return outputs


# --------------------------------------------------------------------------
# CLI and driver
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--T0plus-values", type=_parse_float_list,
                        default=DEFAULT_T_0PLUS_VALUES_MEV,
                        help="descending list of interface temperatures in MeV")
    parser.add_argument("--ray-count", type=int, default=DEFAULT_RAY_COUNT)
    parser.add_argument("--radial-fractions", type=_parse_float_list,
                        default=DEFAULT_RADIAL_FRACTIONS)
    parser.add_argument("--branch-check-radial-fraction", type=float,
                        default=DEFAULT_BRANCH_CHECK_RADIAL_FRACTION)
    parser.add_argument("--angular-buffer-fraction", type=float,
                        default=DEFAULT_ANGULAR_BUFFER_FRACTION)
    parser.add_argument("--radial-buffer-fraction", type=float,
                        default=DEFAULT_RADIAL_BUFFER_FRACTION)
    parser.add_argument("--xi", type=float, default=DEFAULT_XI)
    parser.add_argument("--ms", type=float, default=DEFAULT_MS_MEV)
    parser.add_argument("--nm-type", default=DEFAULT_NM_TYPE)
    parser.add_argument("--upB", type=int, default=DEFAULT_UPB)
    parser.add_argument("--ncrit-over-n0", type=float, default=DEFAULT_NCRIT_OVER_N0)
    parser.add_argument("--b14", type=float, default=np.nan,
                        help="override the calibrated B^(1/4) in MeV")
    parser.add_argument("--density-min-over-n0", type=float,
                        default=DEFAULT_DENSITY_MIN_OVER_N0)
    parser.add_argument("--density-max-over-n0", type=float,
                        default=DEFAULT_DENSITY_MAX_OVER_N0)
    parser.add_argument("--temperature-min", type=float, default=DEFAULT_TEMPERATURE_MIN_MEV)
    parser.add_argument("--temperature-max", type=float, default=DEFAULT_TEMPERATURE_MAX_MEV)
    parser.add_argument("--boundary-theta-count", type=int, default=DEFAULT_BOUNDARY_THETA_COUNT)
    parser.add_argument("--endpoint-probe-count", type=int, default=DEFAULT_ENDPOINT_PROBE_COUNT)
    parser.add_argument("--ray-probe-count", type=int, default=DEFAULT_RAY_PROBE_COUNT)
    parser.add_argument("--bisection-steps", type=int, default=DEFAULT_BISECTION_STEPS)
    parser.add_argument("--tail-eps", type=float, default=DEFAULT_TAIL_EPS)
    parser.add_argument("--n-mesh", type=int, default=DEFAULT_N_MESH)
    parser.add_argument("--tol-bvp", type=float, default=DEFAULT_TOL_BVP)
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument("--jB-lower-bound", type=float, default=DEFAULT_JB_LOWER_BOUND)
    parser.add_argument("--jB-upper-factor", type=float, default=DEFAULT_JB_UPPER_FACTOR)
    parser.add_argument("--point-timeout", type=float, default=DEFAULT_POINT_TIMEOUT_S)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--density-match-rtol", type=float, default=DEFAULT_DENSITY_MATCH_RTOL)
    parser.set_defaults(boundary_T0plus=DEFAULT_A_BOUNDARY_T_0PLUS_MEV)
    parser.add_argument("--skip-branch-check", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    T_values = tuple(float(value) for value in args.T0plus_values)
    if not T_values or not all(np.isfinite(value) and value > 0.0 for value in T_values):
        raise ValueError("T_0plus values must be positive and finite")
    if any(left <= right for left, right in zip(T_values[:-1], T_values[1:])):
        raise ValueError("T_0plus values must be strictly descending")
    fractions = tuple(float(value) for value in args.radial_fractions)
    if not fractions or not all(0.0 < value < 1.0 for value in fractions):
        raise ValueError("radial fractions must lie strictly inside (0, 1)")
    if (
        not args.skip_branch_check
        and not any(
            np.isclose(value, float(args.branch_check_radial_fraction), atol=1.0e-12)
            for value in fractions
        )
    ):
        raise ValueError(
            "branch-check radial fraction must be present in radial fractions"
        )
    if int(args.ray_count) < 1:
        raise ValueError("ray count must be positive")
    if int(args.boundary_theta_count) < 3:
        raise ValueError("boundary theta count must include two axes and an interior ray")
    if int(args.endpoint_probe_count) < 3 or int(args.ray_probe_count) < 3:
        raise ValueError("boundary probe counts must be at least three")
    if int(args.bisection_steps) < 1:
        raise ValueError("bisection steps must be positive")
    if not float(args.density_min_over_n0) < float(args.density_max_over_n0):
        raise ValueError("density window is empty")
    if not 0.0 < float(args.temperature_min) < float(args.temperature_max):
        raise ValueError("temperature window must satisfy 0 < min < max")
    if int(args.workers) < 1:
        raise ValueError("workers must be positive")
    if int(args.save_every) < 0:
        raise ValueError("save frequency cannot be negative")
    if not np.isfinite(float(args.boundary_T0plus)) or float(args.boundary_T0plus) != 0.0:
        raise ValueError("the intrinsic A = 1 boundary requires T_0plus = 0")
    if not float(args.jB_lower_bound) > 0.0 or not float(args.jB_upper_factor) > 0.0:
        raise ValueError("jB bound controls must be positive")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_arguments(args)
    started_at = _utc_now()
    output_path = resolve_output_path(args.output)
    workers = max(1, int(args.workers))

    n0 = float(0.16 * const.MeV_fm**3)
    if np.isfinite(args.b14) and args.b14 > 0.0:
        B14, muB_crit0, P_crit0 = float(args.b14), np.nan, np.nan
    else:
        B14, muB_crit0, P_crit0 = ray_grid.calibrate_b14(
            float(args.ncrit_over_n0) * n0, float(args.ms), str(args.nm_type)
        )
    calibration = {
        "B14_MeV": float(B14),
        "muB_crit0_MeV": float(muB_crit0),
        "P_crit0_MeV4": float(P_crit0),
        "n0_MeV3": float(n0),
        "ncrit_over_n0": float(args.ncrit_over_n0),
    }
    T_values = tuple(float(t) for t in args.T0plus_values)
    boundary_T_0plus = DEFAULT_A_BOUNDARY_T_0PLUS_MEV
    ray_grid.BOUNDARY_T_0PLUS_MEV = boundary_T_0plus
    print(f"B14 = {B14:.6f} MeV; workers = {workers}; output = {output_path}", flush=True)
    print(f"T_0plus values (MeV): {list(T_values)}", flush=True)
    print("Safe domain uses the cold extreme-endothermic A = 1 boundary", flush=True)

    # ---- safe domain -----------------------------------------------------
    # The hidden coordinate measures temperature from zero, so the scale is the
    # window maximum rather than its span.
    transform = ray_grid.RayTransform(
        density_origin_over_n0=float(args.density_min_over_n0),
        density_scale_over_n0=float(args.density_max_over_n0 - args.density_min_over_n0),
        temperature_scale_MeV=float(args.temperature_max),
    )
    process_context = mp.get_context("spawn")
    with process_context.Manager() as manager, ProcessPoolExecutor(
        max_workers=workers,
        mp_context=process_context,
    ) as executor:
        print("Building shared semi-analytical safe boundary", flush=True)
        semi_boundary = ray_grid.solve_boundary_curves(
            transform=transform,
            theta_count=int(args.boundary_theta_count),
            density_min_over_n0=float(args.density_min_over_n0),
            density_max_over_n0=float(args.density_max_over_n0),
            temperature_min_MeV=float(args.temperature_min),
            temperature_max_MeV=float(args.temperature_max),
            n0=n0,
            B14=float(B14),
            xi=float(args.xi),
            ms=float(args.ms),
            nm_type=str(args.nm_type),
            upB=int(args.upB),
            interface_fraction_mode="extreme_endothermic",
            velocity_closure="numerical_I2",
            T_0plus_MeV=boundary_T_0plus,
            endpoint_probe_count=int(args.endpoint_probe_count),
            ray_probe_count=int(args.ray_probe_count),
            bisection_steps=int(args.bisection_steps),
            angular_buffer_fraction=float(args.angular_buffer_fraction),
            progress_factory=lambda **kw: tqdm(leave=False, **kw),
            executor=executor,
        )
        common_boundary = semi_boundary_geometry(
            semi_boundary,
            ray_count=int(args.ray_count),
            angular_buffer_fraction=float(args.angular_buffer_fraction),
        )
        theta_axis = common_boundary["theta_axis_rad"]
        rho_inner = common_boundary["rho_inner"]
        rho_outer = common_boundary["rho_outer"]
        points = select_benchmark_points(
            theta_axis_rad=theta_axis,
            radial_fractions=np.asarray(args.radial_fractions, dtype=float),
            rho_inner=rho_inner,
            rho_outer=rho_outer,
            transform=transform,
            radial_buffer_fraction=float(args.radial_buffer_fraction),
        )
        for point in points:
            theta_index = int(point["theta_index"])
            point["rho_outer_boundary_semi_analytic"] = float(
                common_boundary["rho_outer_semi_analytic"][theta_index]
            )
        boundary = {
            "selection_rule": "semi_analytic_safe_domain",
            "outer_boundary_method": "semi_analytic_extreme_endothermic_bound",
            "outer_boundary_closure": "numerical_I2",
            "outer_boundary_interface_control": "extreme_endothermic",
            "outer_boundary_definition": "A_equals_one",
            "reference_T_0plus_MeV": float(boundary_T_0plus),
            "semi_analytic": semi_boundary,
            "selected_geometry": common_boundary,
        }
        print(f"Benchmark points: {len(points)} "
              f"({theta_axis.size} rays x {len(args.radial_fractions)} radial fractions)",
              flush=True)

        common = {
            "B14_MeV": float(B14), "xi": float(args.xi), "ms_MeV": float(args.ms),
            "nm_type": str(args.nm_type), "upB": int(args.upB),
            "density_match_rtol": float(args.density_match_rtol),
            "T_0plus_values": T_values,
        }

        # ---- analytical + semi-analytical (parallel across points) -------
        analytical_tasks = [
            {**common, "point_index": p["point_index"],
             "T_0minus_MeV": p["T_0minus_MeV"],
             "nB_0minus_MeV3": p["nB_0minus_over_n0"] * n0}
            for p in points
        ]
        analytical_rows: list[dict[str, Any]] = []
        with tqdm(total=len(analytical_tasks), desc="Analytical points", leave=False) as bar:
            futures = {executor.submit(evaluate_analytical_point, t): t for t in analytical_tasks}
            for future in as_completed(futures):
                analytical_rows.extend(future.result()["rows"])
                bar.update(1)
        analytical_rows.sort(key=lambda r: (int(r["point_index"]), -float(r["T_0plus_MeV"])))

        semi_jB_by_point: dict[int, dict[float, float]] = {}
        for row in analytical_rows:
            semi_jB_by_point.setdefault(int(row["point_index"]), {})[
                float(row["T_0plus_MeV"])
            ] = _as_finite_float(row.get("semi_analytic", {}).get("jB"))
        atomic_save(
            output_path,
            build_payload(
                args=args,
                calibration=calibration,
                points=points,
                T_values=T_values,
                analytical_rows=analytical_rows,
                downward_records=[],
                downward_profiles={},
                upward_records=[],
                upward_profiles={},
                boundary_payload=boundary,
                started_at=started_at,
            ),
        )
        print("Checkpointed boundary and analytical results", flush=True)

        # ---- numerical downward sweep (one trajectory per worker) --------
        def numerical_tasks(
            values: tuple[float, ...],
            sweep: str,
            progress_queue,
        ) -> list[dict[str, Any]]:
            return [
                {**common, "point_index": p["point_index"], "sweep_direction": sweep,
                 "T_0minus_MeV": p["T_0minus_MeV"],
                 "nB_0minus_MeV3": p["nB_0minus_over_n0"] * n0,
                 "T_0plus_values": values,
                 "semi_jB_by_T": semi_jB_by_point.get(p["point_index"], {}),
                 "tail_eps": float(args.tail_eps), "n_mesh": int(args.n_mesh),
                 "tol_bvp": float(args.tol_bvp), "max_nodes": int(args.max_nodes),
                 "jB_lower_bound": float(args.jB_lower_bound),
                 "jB_upper_factor": float(args.jB_upper_factor),
                 "point_timeout_s": float(args.point_timeout),
                 "progress_queue": progress_queue}
                for p in points
                if sweep == "downward"
                or abs(p["radial_fraction"] - float(args.branch_check_radial_fraction)) < 1e-12
            ]

        downward_progress_queue = manager.Queue()
        down_tasks = numerical_tasks(T_values, "downward", downward_progress_queue)
        downward_records: list[dict[str, Any]] = []
        downward_profiles: dict[int, dict[float, dict[str, np.ndarray]]] = {}
        downward_futures = {
            executor.submit(run_numerical_trajectory, task): task
            for task in down_tasks
        }

        def accept_downward(output, completed_count):
            downward_records.extend(output["records"])
            downward_profiles[int(output["point_index"])] = output["profiles"]
            if (
                int(args.save_every) > 0
                and completed_count % int(args.save_every) == 0
            ):
                atomic_save(
                    output_path,
                    build_payload(
                        args=args,
                        calibration=calibration,
                        points=points,
                        T_values=T_values,
                        analytical_rows=analytical_rows,
                        downward_records=downward_records,
                        downward_profiles=downward_profiles,
                        upward_records=[],
                        upward_profiles={},
                        boundary_payload=boundary,
                        started_at=started_at,
                    ),
                )

        collect_trajectory_results(
            downward_futures,
            downward_progress_queue,
            base_attempt_count=len(down_tasks) * len(T_values),
            description="Numerical attempts (downward)",
            leave=True,
            completed_callback=accept_downward,
        )

        # ---- upward branch check ----------------------------------------
        upward_records: list[dict[str, Any]] = []
        upward_profiles: dict[int, dict[float, dict[str, np.ndarray]]] = {}
        if not args.skip_branch_check:
            up_values = tuple(reversed(T_values))
            upward_progress_queue = manager.Queue()
            up_tasks = numerical_tasks(up_values, "upward", upward_progress_queue)
            upward_futures = {
                executor.submit(run_numerical_trajectory, task): task
                for task in up_tasks
            }

            def accept_upward(output, completed_count):
                upward_records.extend(output["records"])
                upward_profiles[int(output["point_index"])] = output["profiles"]
                if (
                    int(args.save_every) > 0
                    and completed_count % int(args.save_every) == 0
                ):
                    atomic_save(
                        output_path,
                        build_payload(
                            args=args,
                            calibration=calibration,
                            points=points,
                            T_values=T_values,
                            analytical_rows=analytical_rows,
                            downward_records=downward_records,
                            downward_profiles=downward_profiles,
                            upward_records=upward_records,
                            upward_profiles=upward_profiles,
                            boundary_payload=boundary,
                            started_at=started_at,
                        ),
                    )

            collect_trajectory_results(
                upward_futures,
                upward_progress_queue,
                base_attempt_count=len(up_tasks) * len(up_values),
                description="Branch check (upward)",
                leave=False,
                completed_callback=accept_upward,
            )

    payload = build_payload(
        args=args, calibration=calibration, points=points, T_values=T_values,
        analytical_rows=analytical_rows,
        downward_records=downward_records, downward_profiles=downward_profiles,
        upward_records=upward_records, upward_profiles=upward_profiles,
        boundary_payload=boundary, started_at=started_at,
        run_complete=True,
    )
    atomic_save(output_path, payload)

    n_ok = int(np.count_nonzero(payload["numerical_downward_success"]))
    n_cells = payload["numerical_downward_success"].size
    print(f"\nDownward numerical successes: {n_ok}/{n_cells}", flush=True)
    print(f"Analytic successes: {int(np.count_nonzero(payload['analytic_success']))}"
          f"/{payload['analytic_success'].size}", flush=True)
    print(f"Semi-analytic successes: "
          f"{int(np.count_nonzero(payload['semi_analytic_success']))}"
          f"/{payload['semi_analytic_success'].size}", flush=True)
    if payload["hysteresis_compared_count"]:
        print(f"Hysteresis: compared {payload['hysteresis_compared_count']} cells, "
              f"max relative difference "
              f"{payload['hysteresis_max_relative_difference']:.3e}", flush=True)
    print(f"Saved {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "This is an internal module. Use prepare_T0plus_contour_domain.py or "
        "one of the three run_T0plus_*_contour.py launchers."
    )
