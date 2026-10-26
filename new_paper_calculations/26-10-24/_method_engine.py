#!/usr/bin/env python3
"""Numerical support copied from 26-08-21; slow-front adapter added 26-10-24."""
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

import _slow_front_adapter as slow


# --------------------------------------------------------------------------
# Tunable run parameters
# --------------------------------------------------------------------------
DEFAULT_OUTPUT = "T0plus-three-method-contour.npy"
DEFAULT_RUN_TAG = "T0plus_three_method_contour_v1"
DEFAULT_DOMAIN_OUTPUT = "T0plus-contour-domain.npy"
DEFAULT_FULL_ANALYTIC_OUTPUT = "T0plus-full-analytic-contour.npy"
DEFAULT_SEMI_ANALYTIC_OUTPUT = "T0plus-semi-analytic-contour.npy"
DEFAULT_NUMERICAL_OUTPUT = "T0plus-full-numerical-contour.npy"
DOMAIN_MANIFEST_TYPE = "T0plus_slow_numerical_reference_grid_v1"

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
DEFAULT_N_MESH = 300
DEFAULT_TOL_BVP = 1.0e-4
DEFAULT_MAX_NODES = 50000
DEFAULT_JB_LOWER_BOUND = 1.0e-12
DEFAULT_JB_UPPER_FACTOR = 1.0e-4
DEFAULT_POINT_TIMEOUT_S = 1800.0
DEFAULT_SAVE_EVERY = 1

# Canonical labels (new naming only).
SOLVER_VARIANT = "adiabatic_slow_fixed_T_0plus"
ROOT_METHOD = "solve_bvp_r_jK_parameter_jB"
COMPOSITION_DEFINITION = "nK_over_local_nB"
CURRENT_DEFINITION = "u_nK_minus_D_dnK_dx"
METHOD_LABELS = (
    "analytic_velocity_bound",
    "semi_analytic_velocity_bound",
    "solve_front_adiabatic_slow",
)
PROFILE_FIELDS = (
    "x", "s_coord", "r", "nK", "jK", "nB", "a", "u", "muB", "muK", "T",
    "P", "e", "h", "Gamma_K", "D_K", "invD_K", "dnK_dx",
)
GUESS_SOURCE_SEMI = "semi_analytic_jB"
GUESS_SOURCE_CONTINUATION = "adjacent_numerical_profile"




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


def _extract_profile(result: dict[str, Any]) -> dict[str, np.ndarray]:
    return slow.extract_profile(result)


def validate_domain_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate that all method runners share the intended safe-domain file."""
    if not isinstance(manifest, dict):
        raise ValueError("domain manifest must be a dictionary payload")
    if manifest.get("manifest_type") != DOMAIN_MANIFEST_TYPE:
        raise ValueError("domain manifest has an unsupported type")
    if manifest.get("selection_rule") != "saved_relativistic_numerical_grid":
        raise ValueError("domain manifest must use the saved numerical reference grid")
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
        "lambda_inf",
        "T_inf_MeV",
        "closure_error_max",
        "path_error",
        "constitutive_residual_norm",
        "reaction_residual_norm",
        "diffusion_integral_residual_norm",
        "bvp_max_rms_residual",
        "runtime_s",
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

