#!/usr/bin/env python3
"""Two-block cluster scan of the LTE-implied interface temperature jump.

For T <= 80 MeV, the runner uses common temperature rows and adapts each row's
metastability interval to lie to the right of the single Delta h = 0 boundary.
For T >= 80 MeV, it uses common metastability columns and adapts each column's
temperature interval to lie below A_boundary = 1. The blocks share the 80 MeV
edge and are saved separately so no folded boundary is inverted into a false
single-valued temperature cutoff. The primary field is
T_N - T_Qstar_LTE, where beta_LTE = 5*lambda_n*u_N and T_Qstar_LTE is
reconstructed from the exact local-a interface jump closure.
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
from RMFsolver.SQMsolver import PQM, nB_QM, nK_QM
from _hydro_analytic_contour_common import (
    AnalyticPointTask,
    ColumnTask,
    _immediate_retry_endpoint,
    _iter_points_parallel,
    _iter_points_serial,
    calculate_velocity_point,
    calibrate_b14,
    critical_state_for_temperature,
    muB_from_nB,
)


# Tunable run parameters
DEFAULT_OUTPUT = "lte-temperature-jump-two-block-linear.npy"
DEFAULT_RUN_TAG = "lte_temperature_jump_two_block_linear_beta5_exact_local_a_v3"
DEFAULT_T_MIN_MEV = 1.0e-2
DEFAULT_T_MAX_MEV = 100.0
DEFAULT_T_COUNT = 30
DEFAULT_HIGH_T_COUNT = 12
DEFAULT_SPLIT_TEMPERATURE_MEV = 80.0
DEFAULT_M_MIN = 0.0
DEFAULT_M_MAX = 0.9
DEFAULT_M_COUNT = 20
DEFAULT_BOUNDARY_T_COUNT = 160
DEFAULT_A_BOUNDARY_PROBE_COUNT = 8
DEFAULT_A_BOUNDARY_BISECT_STEPS = 10
DEFAULT_BOUNDARY_EPS_MEV = 0.005
DEFAULT_METASTABILITY_BOUNDARY_EPS = 1.0e-4
DEFAULT_TEMPERATURE_MESH = "adaptive_linear"
DEFAULT_SEED_TEMPERATURE_MEV = 10.0
DEFAULT_NEIGHBOR_RETRY = True
DEFAULT_SAVE_EVERY = 10
DEFAULT_INTERFACE_FRACTION_MODE = "LTE"
DEFAULT_NCRIT_OVER_N0 = 3.0
DEFAULT_XI = 0.0
DEFAULT_MS_MEV = 0.0
DEFAULT_NM_TYPE = "PNM"
DEFAULT_UPB = 5000

METASTABILITY_DEFINITION = "(nB_N - nB_crit(T)) / n0"
GAP_DEFINITION = "muB_cold(P_N) - h_N/nB_N"
DENSITY_RATIO_DEFINITION = "lambda_n_equals_nB_N_over_nB_Q"
COMPOSITION_DEFINITION = "a_local_equals_nK_over_nB"
ANALYTIC_FORMULA_VARIANT = "piecewise_constant_lambda_n"

STATUS_FINITE_U = 0
STATUS_NUMERICAL = 3
STATUS_UNSCANNED = 5

LTE_JUMP_SOLVED = 0
LTE_JUMP_NOT_ATTEMPTED = 1
LTE_JUMP_ROOT_FAILURE = 2
LTE_JUMP_POSTCHECK_FAILURE = 3

LTE_ISOBAR_SOLVED = 0
LTE_ISOBAR_NOT_ATTEMPTED = 1
LTE_ISOBAR_ROOT_FAILURE = 2
LTE_ISOBAR_POSTCHECK_FAILURE = 3

GAP_BOUNDARY_ROOT = 0
GAP_BOUNDARY_ALL_ALLOWED = 1
GAP_BOUNDARY_ALL_FORBIDDEN = 2
GAP_BOUNDARY_FAILURE = 3

A_BOUNDARY_SOLVED = "A_boundary_solved"
A_BOUNDARY_ALL_ALLOWED = "A_all_allowed"
A_BOUNDARY_MASKED_BY_GAP = "A_boundary_masked_by_gap"
A_BOUNDARY_FAILURE = "A_boundary_failure"


@dataclass(frozen=True)
class GapBoundaryTask:
    T_N: float
    B14: float
    n0: float
    ms: float
    nm_type: str
    upB: int
    m_lo: float
    m_hi: float


@dataclass(frozen=True)
class ABoundaryTask:
    metastability: float
    B14: float
    n0: float
    nB_crit0: float
    xi: float
    ms: float
    nm_type: str
    upB: int
    temperature_min: float
    temperature_max: float
    probe_count: int
    bisect_steps: int
    interface_fraction_mode: str


@dataclass(frozen=True)
class LowRowTask:
    row_index: int
    T_N: float
    metastability_values: np.ndarray
    n0: float
    nB_crit0: float
    B14: float
    xi: float
    ms: float
    nm_type: str
    upB: int
    interface_fraction_mode: str
    seed_temperature: float
    default_endpoint_guess: tuple[float, float] | None


def _default_workers() -> int:
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


def _parse_float_pair(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected MU_B_Q,T_Q")
    pair = (float(parts[0]), float(parts[1]))
    if not np.all(np.isfinite(pair)):
        raise argparse.ArgumentTypeError("endpoint guess must be finite")
    return pair


def _as_finite_float(value: Any, default: float = np.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _ellipse_temperature_seed(
    *,
    T_Q_MeV: float,
    A_boundary: float,
    a_interface_LTE: float,
    roundoff_tolerance: float = 1.0e-12,
) -> float:
    """Return the piecewise-constant ellipse estimate used only as a root seed."""
    T_Q_MeV = float(T_Q_MeV)
    A_boundary = float(A_boundary)
    a_interface_LTE = float(a_interface_LTE)
    roundoff_tolerance = float(roundoff_tolerance)
    if not np.all(np.isfinite([T_Q_MeV, A_boundary, a_interface_LTE])):
        raise ValueError("LTE interface-temperature inputs must be finite")
    if T_Q_MeV < 0.0:
        raise ValueError("T_Q_MeV must be non-negative")
    if A_boundary <= 0.0:
        raise ValueError("A_boundary must be positive")
    if a_interface_LTE < 0.0:
        raise ValueError("a_interface_LTE must be non-negative")
    radicand = 1.0 - (a_interface_LTE / A_boundary) ** 2
    if radicand < -roundoff_tolerance or radicand > 1.0 + roundoff_tolerance:
        raise ValueError("a_interface_LTE/A_boundary is outside the physical interval [0, 1]")
    return float(T_Q_MeV * np.sqrt(np.clip(radicand, 0.0, 1.0)))


def signed_interface_temperature_difference(
    *,
    T_N_MeV: float,
    T_Qstar_LTE_MeV: float,
) -> float:
    """Return the signed jump using the upstream-minus-downstream convention."""
    T_N_MeV = float(T_N_MeV)
    T_Qstar_LTE_MeV = float(T_Qstar_LTE_MeV)
    if not np.all(np.isfinite([T_N_MeV, T_Qstar_LTE_MeV])):
        raise ValueError("interface temperatures must be finite")
    return float(T_N_MeV - T_Qstar_LTE_MeV)


def _lte_jump_failure(
    *,
    status: int,
    message: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "status": int(status),
        "message": str(message),
        "state": None,
        "a_residual": np.nan,
        "energy_residual_norm": np.nan,
        "momentum_residual_norm": np.nan,
    }


def _lte_isobar_failure(
    *,
    status: int,
    message: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "status": int(status),
        "message": str(message),
        "muK_A_MeV": np.nan,
        "A_boundary_exact_local": np.nan,
        "pressure_residual_norm": np.nan,
    }


def _exact_local_A_on_zero_temperature_isobar(
    *,
    muB_Q: float,
    P_Q: float,
    B14: float,
    ms: float,
    upB: int,
    muK_probe: float,
) -> dict[str, Any]:
    """Return exact local nK/nB on the fixed-muB_Q, T=0 isobar."""
    try:
        muB_Q = float(muB_Q)
        P_Q = float(P_Q)
        B14 = float(B14)
        ms = float(ms)
        upB = int(upB)
        muK_probe = float(muK_probe)
    except Exception as exc:
        return _lte_isobar_failure(
            status=LTE_ISOBAR_POSTCHECK_FAILURE,
            message=f"invalid exact-isobar inputs: {exc}",
        )
    if (
        not np.all(np.isfinite([muB_Q, P_Q, B14, ms, muK_probe]))
        or muB_Q <= 0.0
        or B14 <= 0.0
        or upB <= 0
    ):
        return _lte_isobar_failure(
            status=LTE_ISOBAR_POSTCHECK_FAILURE,
            message="invalid or non-physical exact-isobar inputs",
        )

    pressure_scale = max(abs(P_Q), 1.0)

    def pressure_residual(muK: float) -> float:
        return float(
            PQM(
                muB_Q,
                float(muK),
                B14,
                0.0,
                ms=ms,
                upB=upB,
            )
            - P_Q
        )

    try:
        lower = 0.0
        residual_lower = pressure_residual(lower)
        upper = max(abs(muK_probe), 1.0)
        residual_upper = pressure_residual(upper)
        for _ in range(64):
            if (
                np.isfinite(residual_lower)
                and np.isfinite(residual_upper)
                and residual_lower * residual_upper <= 0.0
            ):
                break
            upper *= 2.0
            residual_upper = pressure_residual(upper)
        else:
            return _lte_isobar_failure(
                status=LTE_ISOBAR_ROOT_FAILURE,
                message=(
                    "exact T=0 isobar did not bracket a positive-muK root: "
                    f"f(0)={residual_lower:.6e}, "
                    f"f({upper:.6e})={residual_upper:.6e}"
                ),
            )
        muK_A = float(
            brentq(
                pressure_residual,
                lower,
                upper,
                xtol=1.0e-10,
                rtol=1.0e-12,
                maxiter=200,
            )
        )
        nB_A = float(
            nB_QM(muB_Q, muK_A, B14, 0.0, ms=ms, upB=upB)
        )
        nK_A = float(
            nK_QM(muB_Q, muK_A, B14, 0.0, ms=ms, upB=upB)
        )
        pressure_residual_norm = float(
            pressure_residual(muK_A) / pressure_scale
        )
        A_boundary_exact_local = float(nK_A / nB_A)
    except Exception as exc:
        return _lte_isobar_failure(
            status=LTE_ISOBAR_ROOT_FAILURE,
            message=str(exc),
        )

    if (
        not np.all(
            np.isfinite(
                [
                    muK_A,
                    nB_A,
                    nK_A,
                    pressure_residual_norm,
                    A_boundary_exact_local,
                ]
            )
        )
        or muK_A <= 0.0
        or nB_A <= 0.0
        or not (0.0 < A_boundary_exact_local < 1.0)
        or abs(pressure_residual_norm) > 1.0e-9
    ):
        return _lte_isobar_failure(
            status=LTE_ISOBAR_POSTCHECK_FAILURE,
            message=(
                "exact T=0 local isobar returned a non-physical state: "
                f"A={A_boundary_exact_local:.6e}, "
                f"pressure residual={pressure_residual_norm:.6e}"
            ),
        )

    return {
        "success": True,
        "status": LTE_ISOBAR_SOLVED,
        "message": "exact T=0 local-isobar closure converged",
        "muK_A_MeV": muK_A,
        "A_boundary_exact_local": A_boundary_exact_local,
        "pressure_residual_norm": pressure_residual_norm,
    }


def solve_lte_interface_state(
    row: dict[str, Any],
    *,
    B14: float,
    ms: float,
    upB: int,
) -> dict[str, Any]:
    """Solve the exact interface jumps at fixed local a=nK/nB."""
    try:
        u_N = float(row["u_N"])
        jB = float(row["jB"])
        P_N = float(row["P_N_MeV4"])
        h_N = float(row["h_N_MeV4"])
        a_target = float(row["a_interface_LTE"])
        muB_seed = float(row["muB_Q_MeV"])
        T_Q = float(row["T_Q_MeV"])
        A_boundary = float(row["A_boundary_exact_local"])
        B14 = float(B14)
        ms = float(ms)
        upB = int(upB)
    except Exception as exc:
        return _lte_jump_failure(
            status=LTE_JUMP_POSTCHECK_FAILURE,
            message=f"invalid exact-jump inputs: {exc}",
        )

    required = np.array(
        [u_N, jB, P_N, h_N, a_target, muB_seed, T_Q, A_boundary, B14, ms],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(required))
        or jB <= 0.0
        or h_N <= 0.0
        or not (0.0 < a_target < 1.0)
        or muB_seed <= 0.0
        or T_Q < 0.0
        or A_boundary <= 0.0
        or B14 <= 0.0
        or upB <= 0
    ):
        return _lte_jump_failure(
            status=LTE_JUMP_POSTCHECK_FAILURE,
            message="invalid or non-physical exact-jump inputs",
        )

    r_gamma_N = float(np.sqrt(1.0 + u_N * u_N))
    E = float(h_N * u_N * r_gamma_N)
    Pi = float(h_N * u_N * u_N + P_N)
    try:
        T_seed = _ellipse_temperature_seed(
            T_Q_MeV=T_Q,
            A_boundary=A_boundary,
            a_interface_LTE=a_target,
        )
    except Exception:
        T_seed = T_Q
    T_seed = float(max(T_seed, 1.0e-6))
    muK_seed = float(
        max(
            _as_finite_float(
                row.get("muK_A_exact_local_MeV"),
                2.0 * muB_seed * a_target / 9.0,
            ),
            1.0e-6,
        )
    )

    try:
        state = phase_velocity._solve_interface_state_from_local_a_E_and_Pi(
            a_target,
            E,
            Pi,
            jB,
            B14,
            ms=ms,
            upB=upB,
            initial_guess=(muB_seed, muK_seed, T_seed),
        )
    except Exception as exc:
        return _lte_jump_failure(
            status=LTE_JUMP_ROOT_FAILURE,
            message=str(exc),
        )

    try:
        muB_Qstar = float(state["muB"])
        muK_Qstar = float(state["muK"])
        T_Qstar = float(state["T"])
        nB_Qstar = float(state["nB"])
        nK_Qstar = float(state["nK"])
        u_Qstar = float(state["u"])
        if "E" in state:
            E_Qstar = float(state["E"])
        else:
            E_Qstar = float(
                float(state["h"])
                * u_Qstar
                * np.sqrt(1.0 + u_Qstar * u_Qstar)
            )
        if "Pi" in state:
            Pi_Qstar = float(state["Pi"])
        else:
            Pi_Qstar = float(
                float(state["h"]) * u_Qstar * u_Qstar + float(state["P"])
            )
    except Exception as exc:
        return _lte_jump_failure(
            status=LTE_JUMP_POSTCHECK_FAILURE,
            message=f"incomplete exact-jump state: {exc}",
        )

    physical_values = np.array(
        [
            muB_Qstar,
            muK_Qstar,
            T_Qstar,
            nB_Qstar,
            nK_Qstar,
            u_Qstar,
            E_Qstar,
            Pi_Qstar,
        ],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(physical_values))
        or muB_Qstar <= 0.0
        or muK_Qstar < 0.0
        or T_Qstar <= 0.0
        or nB_Qstar <= 0.0
    ):
        return _lte_jump_failure(
            status=LTE_JUMP_POSTCHECK_FAILURE,
            message="exact-jump helper returned a non-physical state",
        )

    a_residual = float(nK_Qstar / nB_Qstar - a_target)
    energy_residual_norm = float((E_Qstar - E) / max(abs(E), 1.0))
    momentum_residual_norm = float((Pi_Qstar - Pi) / max(abs(Pi), 1.0))
    if (
        abs(a_residual) > 1.0e-7
        or abs(energy_residual_norm) > 1.0e-7
        or abs(momentum_residual_norm) > 1.0e-7
    ):
        return _lte_jump_failure(
            status=LTE_JUMP_POSTCHECK_FAILURE,
            message=(
                "exact local-a jump postcheck failed: "
                f"a={a_residual:.3e}, E={energy_residual_norm:.3e}, "
                f"Pi={momentum_residual_norm:.3e}"
            ),
        )

    return {
        "success": True,
        "status": LTE_JUMP_SOLVED,
        "message": "exact local-a interface closure converged",
        "state": state,
        "a_residual": a_residual,
        "energy_residual_norm": energy_residual_norm,
        "momentum_residual_norm": momentum_residual_norm,
    }


def _clear_lte_interface_solution_fields(row: dict[str, Any]) -> None:
    for key in (
        "T_Qstar_LTE_MeV",
        "delta_T_interface_MeV",
        "muB_Qstar_LTE_MeV",
        "muK_Qstar_LTE_MeV",
        "nB_Qstar_LTE_MeV3",
        "nK_Qstar_LTE_MeV3",
        "u_Qstar_LTE",
        "a_Qstar_LTE_derived",
        "lte_jump_a_residual",
        "lte_jump_energy_residual_norm",
        "lte_jump_momentum_residual_norm",
    ):
        row[key] = np.nan


def _clear_lte_isobar_fields(row: dict[str, Any]) -> None:
    for key in (
        "A_boundary_exact_local",
        "muK_A_exact_local_MeV",
        "lte_isobar_pressure_residual_norm",
    ):
        row[key] = np.nan


def add_lte_temperature_fields(
    row: dict[str, Any],
    *,
    T_N_MeV: float,
    B14: float,
    ms: float,
    upB: int,
) -> dict[str, Any]:
    """Enrich one analytic row using the exact local-a interface jump closure."""
    if not row.get("success", False):
        _clear_lte_isobar_fields(row)
        _clear_lte_interface_solution_fields(row)
        row["beta_LTE"] = np.nan
        row["lte_isobar_closure_success"] = False
        row["lte_isobar_closure_status"] = LTE_ISOBAR_NOT_ATTEMPTED
        row["lte_isobar_closure_message"] = "analytic velocity closure failed"
        row["lte_jump_closure_success"] = False
        row["lte_jump_closure_status"] = LTE_JUMP_NOT_ATTEMPTED
        row["lte_jump_closure_message"] = "analytic velocity closure failed"
        return row

    row["A_boundary_analytic"] = _as_finite_float(row.get("A_boundary"))
    row["a_interface_LTE_solver"] = _as_finite_float(
        row.get("a_interface_LTE")
    )
    solver_beta = _as_finite_float(row.get("beta_LTE"))
    row["beta_LTE_solver"] = solver_beta
    row["beta_LTE"] = float(
        5.0 * float(row["lambda_n"]) * float(row["u_N"])
    )
    muB_Q = _as_finite_float(row.get("muB_Q_MeV"))
    analytic_A = _as_finite_float(row.get("A_boundary"))
    isobar = _exact_local_A_on_zero_temperature_isobar(
        muB_Q=muB_Q,
        P_Q=_as_finite_float(row.get("P_Q_MeV4")),
        B14=B14,
        ms=ms,
        upB=upB,
        muK_probe=2.0 * muB_Q * analytic_A / 9.0,
    )
    row["lte_isobar_closure_success"] = bool(isobar["success"])
    row["lte_isobar_closure_status"] = int(isobar["status"])
    row["lte_isobar_closure_message"] = str(isobar["message"])
    row["A_boundary_exact_local"] = float(
        isobar["A_boundary_exact_local"]
    )
    row["muK_A_exact_local_MeV"] = float(isobar["muK_A_MeV"])
    row["lte_isobar_pressure_residual_norm"] = float(
        isobar["pressure_residual_norm"]
    )
    if not isobar["success"]:
        _clear_lte_interface_solution_fields(row)
        row["lte_jump_closure_success"] = False
        row["lte_jump_closure_status"] = LTE_JUMP_NOT_ATTEMPTED
        row["lte_jump_closure_message"] = (
            "exact local-isobar closure failed; jump not attempted"
        )
        return row

    try:
        lte_data = phase_velocity._analytic_a_0plus_lte(
            row["A_boundary_exact_local"],
            float(row["u_N"]),
            float(row["lambda_n"]),
        )
        row["a_interface_LTE"] = float(lte_data["a_0plus_LTE"])
        row["beta_LTE"] = float(lte_data["beta_LTE"])
    except Exception as exc:
        _clear_lte_interface_solution_fields(row)
        row["lte_isobar_closure_success"] = False
        row["lte_isobar_closure_status"] = LTE_ISOBAR_POSTCHECK_FAILURE
        row["lte_isobar_closure_message"] = (
            f"exact local-isobar LTE fraction failed: {exc}"
        )
        row["lte_jump_closure_success"] = False
        row["lte_jump_closure_status"] = LTE_JUMP_NOT_ATTEMPTED
        row["lte_jump_closure_message"] = (
            "exact local-isobar LTE fraction failed; jump not attempted"
        )
        return row

    closure = solve_lte_interface_state(
        row,
        B14=B14,
        ms=ms,
        upB=upB,
    )
    row["lte_jump_closure_success"] = bool(closure["success"])
    row["lte_jump_closure_status"] = int(closure["status"])
    row["lte_jump_closure_message"] = str(closure["message"])
    if not closure["success"]:
        _clear_lte_interface_solution_fields(row)
        return row

    state = closure["state"]
    T_Qstar_LTE_MeV = float(state["T"])
    row["T_Qstar_LTE_MeV"] = float(T_Qstar_LTE_MeV)
    row["delta_T_interface_MeV"] = signed_interface_temperature_difference(
        T_N_MeV=float(T_N_MeV),
        T_Qstar_LTE_MeV=T_Qstar_LTE_MeV,
    )
    row["muB_Qstar_LTE_MeV"] = float(state["muB"])
    row["muK_Qstar_LTE_MeV"] = float(state["muK"])
    row["nB_Qstar_LTE_MeV3"] = float(state["nB"])
    row["nK_Qstar_LTE_MeV3"] = float(state["nK"])
    row["u_Qstar_LTE"] = float(state["u"])
    row["a_Qstar_LTE_derived"] = float(state["nK"] / state["nB"])
    row["lte_jump_a_residual"] = float(closure["a_residual"])
    row["lte_jump_energy_residual_norm"] = float(closure["energy_residual_norm"])
    row["lte_jump_momentum_residual_norm"] = float(
        closure["momentum_residual_norm"]
    )
    return row


def _upstream_nuclear_state(muB_N: float, T_N: float, nm_type: str):
    try:
        return phase_velocity._analytic_nuclear_state(
            float(muB_N),
            float(T_N),
            param=para.paraQMCRMF3,
            NM_type=str(nm_type),
        )
    except Exception:
        return None


def _enthalpy_gap(muB_N: float, T_N: float, B14: float, nm_type: str):
    state = _upstream_nuclear_state(muB_N, T_N, nm_type)
    if state is None:
        return float("nan"), float("nan"), None
    gap, muB_cold = phase_velocity._slow_front_enthalpy_gap(state, float(B14))
    return float(gap), float(muB_cold), state


def solve_analytic_velocity(
    *,
    muB_N: float,
    T_N: float,
    B14: float,
    xi: float,
    ms: float,
    nm_type: str,
    upB: int,
    endpoint_initial_guess: tuple[float, float] | None,
    interface_fraction_mode: str,
) -> dict[str, Any]:
    """Call the analytic solver with an explicit closure and endpoint seed."""
    try:
        result = phase_velocity.analytic_velocity_bound(
            float(muB_N),
            float(T_N),
            float(B14),
            xi=float(xi),
            ms=float(ms),
            param=para.paraQMCRMF3,
            NM_type=str(nm_type),
            upB=int(upB),
            initial_guess=endpoint_initial_guess,
            interface_fraction_mode=str(interface_fraction_mode),
        )
    except Exception as exc:
        message = str(exc)
        return {
            "success": False,
            "message": message,
            "endpoint_domain_failure": (
                "A_boundary must satisfy" in message
                or "a_interface" in message
            ),
            "result": None,
            "endpoint_solution": None,
        }
    endpoint = (
        float(result["muB_Q"]),
        float(result["T_Q"]),
    )
    return {
        "success": True,
        "message": str(result.get("message", "")),
        "endpoint_domain_failure": False,
        "result": result,
        "endpoint_solution": endpoint,
    }


def gap_at_metastability(
    metastability: float,
    T_N: float,
    B14: float,
    n0: float,
    nB_crit_T: float,
    nm_type: str,
) -> float:
    nB_N = float(nB_crit_T + float(metastability) * float(n0))
    try:
        muB_N = muB_from_nB(nB_N, float(T_N), str(nm_type), prefer=1100.0)
    except Exception:
        return float("nan")
    gap, _, _ = _enthalpy_gap(muB_N, T_N, B14, nm_type)
    return float(gap)


def solve_gap_boundary_at_temperature(task: GapBoundaryTask) -> dict[str, Any]:
    """Solve Delta h = 0 for metastability at one fixed temperature."""
    base = {
        "T_MeV": float(task.T_N),
        "metastability_boundary": np.nan,
        "status": GAP_BOUNDARY_FAILURE,
        "gap_at_m_min": np.nan,
        "gap_at_m_max": np.nan,
        "allowed_above_boundary": True,
    }
    try:
        _, _, nB_crit_T = critical_state_for_temperature(
            task.T_N, task.B14, task.ms, task.nm_type, task.upB
        )
    except Exception as exc:
        base["message"] = str(exc)
        return base

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
    base.update({"gap_at_m_min": gap_lo, "gap_at_m_max": gap_hi})
    if not (np.isfinite(gap_lo) and np.isfinite(gap_hi)):
        base["message"] = "non-finite enthalpy-gap endpoint"
        return base
    if gap_lo <= 0.0 and gap_hi <= 0.0:
        base.update({"status": GAP_BOUNDARY_ALL_ALLOWED, "message": "all metastabilities allowed"})
        return base
    if gap_lo > 0.0 and gap_hi > 0.0:
        base.update({"status": GAP_BOUNDARY_ALL_FORBIDDEN, "message": "all metastabilities forbidden"})
        return base
    try:
        root = brentq(gap, task.m_lo, task.m_hi, xtol=1.0e-6, rtol=1.0e-10, maxiter=100)
    except Exception as exc:
        base["message"] = str(exc)
        return base
    base.update(
        {
            "metastability_boundary": float(root),
            "status": GAP_BOUNDARY_ROOT,
            "message": "Delta h = 0 root",
            "allowed_above_boundary": bool(gap_hi <= 0.0),
        }
    )
    return base


def separation_boundary_metastability(
    T_N: float,
    B14: float,
    n0: float,
    ms: float,
    nm_type: str,
    upB: int,
    m_lo: float,
    m_hi: float,
) -> float:
    """Compatibility wrapper returning only the Delta h = 0 root."""
    result = solve_gap_boundary_at_temperature(
        GapBoundaryTask(T_N, B14, n0, ms, nm_type, upB, m_lo, m_hi)
    )
    return float(result["metastability_boundary"])


def _evaluate_A_boundary_point(
    task: ABoundaryTask,
    temperature: float,
    endpoint_guess: tuple[float, float] | None,
) -> dict[str, Any]:
    row = {
        "T_MeV": float(temperature),
        "metastability": float(task.metastability),
        "classification": "numerical_failure",
        "A_boundary": np.nan,
        "gap": np.nan,
        "endpoint_solution": None,
    }
    try:
        _, _, nB_crit_T = critical_state_for_temperature(
            temperature, task.B14, task.ms, task.nm_type, task.upB
        )
        nB_N = float(nB_crit_T + task.metastability * task.n0)
        muB_N = muB_from_nB(nB_N, temperature, task.nm_type, prefer=1100.0)
    except Exception as exc:
        row["message"] = str(exc)
        return row

    gap, _, state = _enthalpy_gap(muB_N, temperature, task.B14, task.nm_type)
    row.update({"gap": gap, "muB_N_MeV": float(muB_N)})
    if state is None or not np.isfinite(gap):
        row["message"] = "upstream or gap evaluation failed"
        return row
    if gap > 0.0:
        row.update({"classification": "gap_forbidden", "message": "Delta h > 0"})
        return row

    solved = solve_analytic_velocity(
        muB_N=muB_N,
        T_N=temperature,
        B14=task.B14,
        xi=task.xi,
        ms=task.ms,
        nm_type=task.nm_type,
        upB=task.upB,
        endpoint_initial_guess=endpoint_guess,
        interface_fraction_mode=task.interface_fraction_mode,
    )
    row["message"] = solved["message"]
    if not solved["success"]:
        row["classification"] = (
            "A_forbidden" if solved["endpoint_domain_failure"] else "numerical_failure"
        )
        return row
    result = solved["result"]
    row.update(
        {
            "classification": "allowed",
            "A_boundary": float(result["A_boundary"]),
            "endpoint_solution": solved["endpoint_solution"],
        }
    )
    return row


def solve_A_boundary_for_metastability(task: ABoundaryTask) -> dict[str, Any]:
    """Solve A_boundary = 1 above the two-block seam."""
    probes = np.linspace(
        task.temperature_min,
        task.temperature_max,
        max(2, int(task.probe_count)),
    )
    evaluations: list[dict[str, Any]] = []
    endpoint_guess = None
    lower_allowed = None
    upper_A = None
    gap_seen = False
    numerical_failure = False
    boundary_numerical_limit = False

    for temperature in probes:
        row = _evaluate_A_boundary_point(task, float(temperature), endpoint_guess)
        row["boundary_stage"] = "probe"
        evaluations.append(row)
        classification = row["classification"]
        if classification == "allowed":
            lower_allowed = float(temperature)
            endpoint_guess = row["endpoint_solution"]
        elif classification == "A_forbidden" and lower_allowed is not None:
            upper_A = float(temperature)
            break
        elif classification == "gap_forbidden":
            gap_seen = True
            break
        else:
            numerical_failure = True
            break

    status = A_BOUNDARY_FAILURE
    cut = np.nan
    lower = lower_allowed if lower_allowed is not None else np.nan
    upper = upper_A if upper_A is not None else np.nan
    if numerical_failure:
        status = A_BOUNDARY_FAILURE
    elif gap_seen:
        status = A_BOUNDARY_MASKED_BY_GAP
    elif upper_A is None and lower_allowed is not None:
        status = A_BOUNDARY_ALL_ALLOWED
        cut = float(task.temperature_max)
    elif lower_allowed is not None and upper_A is not None:
        status = A_BOUNDARY_SOLVED
        lo = float(lower_allowed)
        hi = float(upper_A)
        seed = endpoint_guess
        for _ in range(max(0, int(task.bisect_steps))):
            middle = 0.5 * (lo + hi)
            row = _evaluate_A_boundary_point(task, middle, seed)
            row["boundary_stage"] = "bisect"
            evaluations.append(row)
            if row["classification"] == "allowed":
                lo = middle
                seed = row["endpoint_solution"]
            elif row["classification"] == "A_forbidden":
                hi = middle
            elif row["classification"] == "gap_forbidden":
                status = A_BOUNDARY_MASKED_BY_GAP
                lo = hi = np.nan
                break
            else:
                # The extreme-endothermic velocity formula is singular as
                # A_boundary approaches 1 from below. Once allowed and
                # forbidden evaluations already bracket the boundary, a
                # failed interior eigenvalue evaluation cannot invalidate
                # that bracket; it only stops further refinement.
                boundary_numerical_limit = True
                break
        if status == A_BOUNDARY_SOLVED:
            lower, upper = lo, hi
            cut = 0.5 * (lo + hi)

    return {
        "success": bool(status in {A_BOUNDARY_SOLVED, A_BOUNDARY_ALL_ALLOWED}),
        "status": status,
        "metastability": float(task.metastability),
        "temperature_cut_MeV": float(cut),
        "temperature_lower_bracket_MeV": float(lower),
        "temperature_upper_bracket_MeV": float(upper),
        "boundary_target": 1.0,
        "boundary_quantity_name": "A_boundary",
        "boundary_numerical_limit": bool(boundary_numerical_limit),
        "interface_fraction_mode": str(task.interface_fraction_mode),
        "evaluations": evaluations,
    }


def build_low_temperature_axis(
    *,
    temperature_mesh: str,
    temperature_min: float,
    split_temperature: float,
    temperature_count: int,
) -> np.ndarray:
    """Build the shared temperature rows for the low-temperature block."""
    if not (0.0 < float(temperature_min) < float(split_temperature)):
        raise ValueError("low-block temperatures must satisfy 0 < min < split")
    if int(temperature_count) < 2:
        raise ValueError("low-block temperature_count must be at least 2")
    if temperature_mesh == "adaptive_log":
        return np.geomspace(float(temperature_min), float(split_temperature), int(temperature_count))
    if temperature_mesh == "adaptive_linear":
        return np.linspace(float(temperature_min), float(split_temperature), int(temperature_count))
    raise ValueError("temperature_mesh must be adaptive_log or adaptive_linear")


def build_low_metastability_row(
    *,
    metastability_boundary: float,
    boundary_active: bool,
    metastability_max: float,
    boundary_eps: float,
    metastability_count: int,
) -> np.ndarray:
    """Build one low-block row strictly to the right of the gap boundary."""
    lower = (
        float(metastability_boundary) + float(boundary_eps)
        if boundary_active
        else 0.0
    )
    upper = float(metastability_max)
    if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 <= lower < upper):
        raise ValueError("low-block metastability row has no valid interval")
    if int(metastability_count) < 2:
        raise ValueError("metastability_count must be at least 2")
    return np.linspace(lower, upper, int(metastability_count))


def build_high_temperature_column(
    *,
    temperature_mesh: str,
    split_temperature: float,
    A_temperature_boundary: float,
    boundary_eps: float,
    temperature_count: int,
) -> np.ndarray:
    """Build one high-block column from the seam to below A_boundary = 1."""
    upper = float(A_temperature_boundary) - float(boundary_eps)
    lower = float(split_temperature)
    if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 < lower < upper):
        raise ValueError("high-block temperature column has no valid interval")
    if int(temperature_count) < 2:
        raise ValueError("high-block temperature_count must be at least 2")
    if temperature_mesh == "adaptive_log":
        return np.geomspace(lower, upper, int(temperature_count))
    if temperature_mesh == "adaptive_linear":
        return np.linspace(lower, upper, int(temperature_count))
    raise ValueError("temperature_mesh must be adaptive_log or adaptive_linear")


def copy_low_seam_to_high(
    *,
    arrays: dict[str, np.ndarray],
    low_seam_row: int,
    high_columns: list[int],
    field_names: tuple[str, ...],
    integer_field_names: tuple[str, ...] = (),
) -> None:
    """Copy the shared split-temperature row exactly into the high block."""
    for j in high_columns:
        for name in ("status_grid", *integer_field_names, *field_names):
            arrays[f"high_{name}"][0, j] = arrays[f"low_{name}"][low_seam_row, j]


def low_temperature_row_order(values: np.ndarray, seed_temperature: float) -> list[int]:
    """Order low-block rows from the seed, then upward and downward."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("low-block temperature values must be a finite 1D array")
    seed = int(np.argmin(np.abs(values - float(seed_temperature))))
    return [
        seed,
        *range(seed + 1, values.size),
        *range(seed - 1, -1, -1),
    ]


def low_metastability_solve_order(metastability_count: int) -> list[int]:
    """Walk a low-block row from stable high metastability toward its boundary."""
    if int(metastability_count) < 1:
        raise ValueError("metastability_count must be positive")
    return list(range(int(metastability_count) - 1, -1, -1))


def _finite_endpoint(row: dict[str, Any] | None) -> tuple[float, float] | None:
    if not row or not bool(row.get("success", False)):
        return None
    try:
        endpoint = tuple(float(value) for value in row["endpoint_solution"])
    except Exception:
        return None
    return endpoint if len(endpoint) == 2 and np.all(np.isfinite(endpoint)) else None


def low_row_neighbor_endpoint_guess(
    col_index: int,
    solved_by_col: dict[int, dict[str, Any]],
    default: tuple[float, float] | None = None,
) -> tuple[tuple[float, float] | None, str]:
    """Use only the immediately adjacent higher-metastability solution."""
    endpoint = _finite_endpoint(solved_by_col.get(int(col_index) + 1))
    if endpoint is not None:
        return endpoint, "same_row_higher_metastability_neighbor"
    if default is not None:
        try:
            default = tuple(float(value) for value in default)
        except Exception:
            default = None
    return default, "fallback_after_failed_neighbor"


def _low_row_plan(task: LowRowTask) -> dict[str, Any]:
    return {
        "task": task,
        "order": low_metastability_solve_order(task.metastability_values.size),
        "position": 0,
        "solved": {},
    }


def _next_low_point_task(plan: dict[str, Any]):
    if plan["position"] >= len(plan["order"]):
        return None
    col_index = int(plan["order"][plan["position"]])
    row_task: LowRowTask = plan["task"]
    guess, source = low_row_neighbor_endpoint_guess(
        col_index,
        plan["solved"],
        row_task.default_endpoint_guess,
    )
    point_task = AnalyticPointTask(
        T_N=float(row_task.T_N),
        metastability=float(row_task.metastability_values[col_index]),
        n0=row_task.n0,
        nB_crit0=row_task.nB_crit0,
        B14=row_task.B14,
        xi=row_task.xi,
        ms=row_task.ms,
        nm_type=row_task.nm_type,
        upB=row_task.upB,
        interface_fraction_mode=row_task.interface_fraction_mode,
        endpoint_initial_guess=guess,
    )
    direction = "seed_high_metastability" if plan["position"] == 0 else "left"
    return (
        int(row_task.row_index),
        col_index,
        point_task,
        direction,
        source,
        float(row_task.seed_temperature),
        0,
        None,
    )


def _remember_low_point(plan: dict[str, Any], result: dict[str, Any]) -> None:
    plan["solved"][int(result["col_index"])] = result["row"]
    plan["position"] += 1


def _iter_low_points_serial(row_tasks: list[LowRowTask]):
    for row_task in row_tasks:
        plan = _low_row_plan(row_task)
        while (work := _next_low_point_task(plan)) is not None:
            result = calculate_velocity_point(work)
            _remember_low_point(plan, result)
            yield result


def _parallel_map_unordered(function, tasks, workers: int):
    """Yield independent task results as workers finish them."""
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--run-tag", default=os.getenv("RUN_TAG", DEFAULT_RUN_TAG))
    parser.add_argument("--t-min", type=float, default=DEFAULT_T_MIN_MEV)
    parser.add_argument("--t-max", type=float, default=DEFAULT_T_MAX_MEV)
    parser.add_argument("--t-count", type=int, default=DEFAULT_T_COUNT)
    parser.add_argument("--high-t-count", type=int, default=DEFAULT_HIGH_T_COUNT)
    parser.add_argument(
        "--split-temperature",
        type=float,
        default=DEFAULT_SPLIT_TEMPERATURE_MEV,
    )
    parser.add_argument(
        "--m-min",
        type=float,
        default=DEFAULT_M_MIN,
        help="compatibility option; the two-block topology requires m-min=0",
    )
    parser.add_argument("--m-max", type=float, default=DEFAULT_M_MAX)
    parser.add_argument("--m-count", type=int, default=DEFAULT_M_COUNT)
    parser.add_argument("--boundary-t-count", type=int, default=DEFAULT_BOUNDARY_T_COUNT)
    parser.add_argument("--A-boundary-probe-count", type=int, default=DEFAULT_A_BOUNDARY_PROBE_COUNT)
    parser.add_argument("--A-boundary-bisect-steps", type=int, default=DEFAULT_A_BOUNDARY_BISECT_STEPS)
    parser.add_argument("--boundary-eps", type=float, default=DEFAULT_BOUNDARY_EPS_MEV)
    parser.add_argument(
        "--metastability-boundary-eps",
        type=float,
        default=DEFAULT_METASTABILITY_BOUNDARY_EPS,
    )
    parser.add_argument(
        "--temperature-mesh",
        choices=("adaptive_linear",),
        default=DEFAULT_TEMPERATURE_MESH,
    )
    parser.add_argument("--seed-temperature", type=float, default=DEFAULT_SEED_TEMPERATURE_MEV)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--endpoint-guess", type=_parse_float_pair, default=None)
    parser.add_argument(
        "--no-neighbor-retry",
        action="store_false",
        dest="neighbor_retry",
        default=DEFAULT_NEIGHBOR_RETRY,
    )
    parser.add_argument(
        "--interface-fraction-mode",
        choices=("LTE",),
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
    if not (0.0 < args.t_min < args.split_temperature < args.t_max):
        raise ValueError("temperatures must satisfy 0 < t_min < split_temperature < t_max")
    if args.t_count < 2 or args.high_t_count < 2 or args.m_count < 2:
        raise ValueError("both temperature counts and m_count must be at least 2")
    if args.boundary_t_count < 2:
        raise ValueError("boundary_t_count must be at least 2")
    if args.m_min != 0.0 or args.m_max <= 0.0:
        raise ValueError("the two-block domain requires m_min = 0 and m_max > 0")
    if args.A_boundary_probe_count < 2 or args.A_boundary_bisect_steps < 0:
        raise ValueError("A boundary settings are invalid")
    if (
        args.boundary_eps < 0.0
        or args.metastability_boundary_eps < 0.0
        or args.save_every < 0
        or args.workers < 1
    ):
        raise ValueError("boundary buffers, save-every, and workers must be nonnegative/positive")


def _payload(
    *,
    args: argparse.Namespace,
    calibration: tuple[float, float, float, float, float],
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    A_boundary_records: list[dict[str, Any]],
    gap_boundary_records: list[dict[str, Any]],
) -> dict[str, Any]:
    B14, muB_crit0, P_crit0, n0, nB_crit0 = calibration
    return {
        "run_tag": str(args.run_tag),
        "temperature_mesh": str(args.temperature_mesh),
        "temperature_axis_scale": "log" if args.temperature_mesh == "adaptive_log" else "linear",
        "mesh_layout": "two_block_orthogonal_adaptive",
        "split_temperature_MeV": float(args.split_temperature),
        "primary_observable": "delta_T_interface_MeV",
        "temperature_jump_definition": (
            "delta_T_interface_MeV = T_N_MeV - T_Qstar_LTE_MeV"
        ),
        "T_Qstar_LTE_definition": (
            "exact local-a interface solve of baryon, energy, and momentum jumps"
        ),
        "T_Qstar_LTE_bound_type": "lower_bound",
        "delta_T_interface_bound_type": "upper_bound_at_fixed_T_N",
        "beta_LTE_definition": "beta_LTE = 5*lambda_n*u_N",
        "A_boundary_grid_definition": (
            "analytic velocity-closure boundary retained for mesh compatibility"
        ),
        "A_boundary_exact_local_definition": (
            "nK/nB at fixed muB_Q and T=0 where "
            "PQM(muB_Q,muK,B14,0)=P_Q"
        ),
        "a_interface_LTE_definition": (
            "LTE quadratic evaluated with A_boundary_exact_local"
        ),
        "interface_state_helper": (
            "phase_velocity._solve_interface_state_from_local_a_E_and_Pi"
        ),
        "exact_isobar_helper": (
            "cluster-private fixed-muB_Q T=0 pressure root followed by nK/nB"
        ),
        "isobar_failure_policy": (
            "mark isobar status, skip jump closure, and return NaN temperature"
        ),
        "jump_failure_policy": "mark closure status and return NaN temperature",
        "analytic_closure": "hydro_jump_eigenvalue",
        "A_boundary_closure": "hydro_jump_eigenvalue_A_boundary_1",
        "interface_fraction_mode": str(args.interface_fraction_mode),
        "metastability_definition": METASTABILITY_DEFINITION,
        "gap_definition": GAP_DEFINITION,
        "composition_definition": COMPOSITION_DEFINITION,
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "analytic_formula_variant": ANALYTIC_FORMULA_VARIANT,
        "safe_region_definition": "A_boundary <= 1 and Delta_h <= 0",
        "B14_calibrated_MeV": float(B14),
        "muB_crit0_MeV": float(muB_crit0),
        "P_crit0_MeV4": float(P_crit0),
        "n0_MeV3": float(n0),
        "nB_crit0_MeV3": float(nB_crit0),
        "nB_crit0_over_n0": float(nB_crit0 / n0),
        "status_codes": {
            "finite_u": STATUS_FINITE_U,
            "numerical_failure": STATUS_NUMERICAL,
            "unscanned_outside_safe_region": STATUS_UNSCANNED,
        },
        "lte_jump_closure_status_codes": {
            "solved": LTE_JUMP_SOLVED,
            "not_attempted": LTE_JUMP_NOT_ATTEMPTED,
            "root_failure": LTE_JUMP_ROOT_FAILURE,
            "postcheck_failure": LTE_JUMP_POSTCHECK_FAILURE,
        },
        "lte_isobar_closure_status_codes": {
            "solved": LTE_ISOBAR_SOLVED,
            "not_attempted": LTE_ISOBAR_NOT_ATTEMPTED,
            "root_failure": LTE_ISOBAR_ROOT_FAILURE,
            "postcheck_failure": LTE_ISOBAR_POSTCHECK_FAILURE,
        },
        **{name: np.asarray(value) for name, value in arrays.items()},
        "records": list(records),
        "A_boundary_records": list(A_boundary_records),
        "gap_boundary_records": list(gap_boundary_records),
        "arguments": vars(args).copy(),
    }


def _save_current(
    path: Path,
    args: argparse.Namespace,
    calibration: tuple[float, float, float, float, float],
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    A_boundary_records: list[dict[str, Any]],
    gap_boundary_records: list[dict[str, Any]],
) -> None:
    _atomic_save(
        path,
        _payload(
            args=args,
            calibration=calibration,
            arrays=arrays,
            records=records,
            A_boundary_records=A_boundary_records,
            gap_boundary_records=gap_boundary_records,
        ),
    )


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
        B14, muB_crit0, P_crit0 = calibrate_b14(nB_crit0, args.ms, args.nm_type)
        print(f"Calibrated B^(1/4) = {B14:.10g} MeV", flush=True)
    calibration = (B14, muB_crit0, P_crit0, n0, nB_crit0)

    low_temperatures = build_low_temperature_axis(
        temperature_mesh=args.temperature_mesh,
        temperature_min=args.t_min,
        split_temperature=args.split_temperature,
        temperature_count=args.t_count,
    )
    high_metastabilities = np.linspace(0.0, args.m_max, args.m_count)
    low_shape = (args.t_count, args.m_count)
    high_shape = (args.high_t_count, args.m_count)
    field_names = (
        "delta_T_interface_MeV_grid",
        "T_Qstar_LTE_MeV_grid",
        "beta_LTE_grid",
        "velocity_grid_m_s",
        "u_N_grid",
        "lambda_n_grid",
        "A_boundary_grid",
        "A_boundary_exact_local_grid",
        "muK_A_exact_local_MeV_grid",
        "lte_isobar_pressure_residual_norm_grid",
        "a_interface_grid",
        "a_interface_LTE_grid",
        "a_Qstar_LTE_derived_grid",
        "muB_Qstar_LTE_MeV_grid",
        "muK_Qstar_LTE_MeV_grid",
        "nB_Qstar_LTE_MeV3_grid",
        "nK_Qstar_LTE_MeV3_grid",
        "u_Qstar_LTE_grid",
        "lte_jump_a_residual_grid",
        "lte_jump_energy_residual_norm_grid",
        "lte_jump_momentum_residual_norm_grid",
        "nB_crit_T_grid",
        "muB_crit_T_grid",
    )
    arrays: dict[str, np.ndarray] = {
        "low_temperature_grid": np.repeat(low_temperatures[:, None], args.m_count, axis=1),
        "low_metastability_grid": np.full(low_shape, np.nan),
        "low_status_grid": np.full(low_shape, STATUS_UNSCANNED, dtype=int),
        "high_temperature_grid": np.full(high_shape, np.nan),
        "high_metastability_grid": np.repeat(
            high_metastabilities[None, :], args.high_t_count, axis=0
        ),
        "high_status_grid": np.full(high_shape, STATUS_UNSCANNED, dtype=int),
        "low_gap_boundary_metastability_values": np.full(args.t_count, np.nan),
        "low_gap_boundary_status": np.full(args.t_count, GAP_BOUNDARY_FAILURE, dtype=int),
        "low_gap_boundary_active": np.zeros(args.t_count, dtype=bool),
        "A_boundary_temperature_values": np.full(args.m_count, np.nan),
        "A_boundary_metastability_values": high_metastabilities.copy(),
        "A_boundary_status": np.full(args.m_count, A_BOUNDARY_FAILURE, dtype=object),
    }
    for prefix, shape in (("low", low_shape), ("high", high_shape)):
        for name in field_names:
            arrays[f"{prefix}_{name}"] = np.full(shape, np.nan)
        arrays[f"{prefix}_lte_jump_closure_status_grid"] = np.full(
            shape,
            LTE_JUMP_NOT_ATTEMPTED,
            dtype=int,
        )
        arrays[f"{prefix}_lte_isobar_closure_status_grid"] = np.full(
            shape,
            LTE_ISOBAR_NOT_ATTEMPTED,
            dtype=int,
        )

    records: list[dict[str, Any]] = []
    A_boundary_records: list[dict[str, Any]] = []
    gap_boundary_records: list[dict[str, Any]] = []

    print(f"Output: {output_path}", flush=True)
    print(f"Run tag: {args.run_tag}", flush=True)
    print(
        f"Two-block grid: low={args.t_count} T rows x {args.m_count} adaptive m; "
        f"high={args.m_count} m columns x {args.high_t_count} adaptive T; "
        f"seam={args.split_temperature:g} MeV; workers={args.workers}",
        flush=True,
    )

    gap_sample_temperatures = np.unique(
        np.concatenate(
            (
                low_temperatures,
                np.linspace(args.t_min, args.split_temperature, args.boundary_t_count),
            )
        )
    )
    gap_tasks = [
        GapBoundaryTask(
            T_N=float(temperature),
            B14=B14,
            n0=n0,
            ms=args.ms,
            nm_type=args.nm_type,
            upB=args.upB,
            m_lo=0.0,
            m_hi=args.m_max,
        )
        for temperature in gap_sample_temperatures
    ]
    gap_iterator = (
        map(solve_gap_boundary_at_temperature, gap_tasks)
        if args.workers == 1
        else _parallel_map_unordered(solve_gap_boundary_at_temperature, gap_tasks, args.workers)
    )
    for completed, result in enumerate(
        tqdm(gap_iterator, total=len(gap_tasks), desc="Delta_h=0 boundary", unit="T"),
        start=1,
    ):
        gap_boundary_records.append(result)
        if args.save_every and completed % args.save_every == 0:
            _save_current(
                output_path, args, calibration, arrays, records,
                A_boundary_records, gap_boundary_records,
            )
    gap_boundary_records.sort(key=lambda row: row["T_MeV"])

    def nearest_gap_result(temperature: float) -> dict[str, Any]:
        return min(
            gap_boundary_records,
            key=lambda row: abs(float(row["T_MeV"]) - float(temperature)),
        )

    for i, temperature in enumerate(low_temperatures):
        result = nearest_gap_result(float(temperature))
        status = int(result["status"])
        arrays["low_gap_boundary_status"][i] = status
        if status == GAP_BOUNDARY_ROOT and result.get("allowed_above_boundary", True):
            boundary = float(result["metastability_boundary"])
            arrays["low_gap_boundary_metastability_values"][i] = boundary
            arrays["low_gap_boundary_active"][i] = True
            arrays["low_metastability_grid"][i] = build_low_metastability_row(
                metastability_boundary=boundary,
                boundary_active=True,
                metastability_max=args.m_max,
                boundary_eps=args.metastability_boundary_eps,
                metastability_count=args.m_count,
            )
        elif status == GAP_BOUNDARY_ALL_ALLOWED:
            arrays["low_gap_boundary_metastability_values"][i] = 0.0
            arrays["low_metastability_grid"][i] = build_low_metastability_row(
                metastability_boundary=0.0,
                boundary_active=False,
                metastability_max=args.m_max,
                boundary_eps=args.metastability_boundary_eps,
                metastability_count=args.m_count,
            )

    root_rows = [
        row for row in gap_boundary_records
        if row["status"] == GAP_BOUNDARY_ROOT
        and np.isfinite(row["metastability_boundary"])
    ]
    curve_temperatures = [0.0, *[float(row["T_MeV"]) for row in root_rows]]
    curve_metastabilities = [0.0, *[float(row["metastability_boundary"]) for row in root_rows]]
    all_allowed_rows = [
        row for row in gap_boundary_records
        if row["status"] == GAP_BOUNDARY_ALL_ALLOWED and float(row["T_MeV"]) > curve_temperatures[-1]
    ]
    if root_rows and all_allowed_rows:
        lower = float(root_rows[-1]["T_MeV"])
        upper = float(all_allowed_rows[0]["T_MeV"])

        def gap_on_axis(temperature: float) -> float:
            muB_crit_T, _, _ = critical_state_for_temperature(
                temperature, B14, args.ms, args.nm_type, args.upB
            )
            return float(_enthalpy_gap(muB_crit_T, temperature, B14, args.nm_type)[0])

        try:
            closure_temperature = brentq(
                gap_on_axis, lower, upper, xtol=1.0e-7, rtol=1.0e-11, maxiter=100
            )
        except Exception:
            closure_temperature = upper
        curve_temperatures.append(float(closure_temperature))
        curve_metastabilities.append(0.0)
    arrays["gap_boundary_temperature_values"] = np.asarray(curve_temperatures)
    arrays["gap_boundary_metastability_values"] = np.asarray(curve_metastabilities)

    A_tasks = [
        ABoundaryTask(
            metastability=float(metastability),
            B14=B14,
            n0=n0,
            nB_crit0=nB_crit0,
            xi=args.xi,
            ms=args.ms,
            nm_type=args.nm_type,
            upB=args.upB,
            temperature_min=args.split_temperature,
            temperature_max=args.t_max,
            probe_count=args.A_boundary_probe_count,
            bisect_steps=args.A_boundary_bisect_steps,
            interface_fraction_mode=args.interface_fraction_mode,
        )
        for metastability in high_metastabilities
    ]
    A_iterator = (
        map(solve_A_boundary_for_metastability, A_tasks)
        if args.workers == 1
        else _parallel_map_unordered(solve_A_boundary_for_metastability, A_tasks, args.workers)
    )
    for completed, result in enumerate(
        tqdm(A_iterator, total=len(A_tasks), desc="A_boundary=1 boundary", unit="m"),
        start=1,
    ):
        A_boundary_records.append(result)
        j = int(np.argmin(np.abs(high_metastabilities - result["metastability"])))
        arrays["A_boundary_temperature_values"][j] = result["temperature_cut_MeV"]
        arrays["A_boundary_status"][j] = result["status"]
        if result["success"]:
            try:
                arrays["high_temperature_grid"][:, j] = build_high_temperature_column(
                    temperature_mesh=args.temperature_mesh,
                    split_temperature=args.split_temperature,
                    A_temperature_boundary=result["temperature_cut_MeV"],
                    boundary_eps=args.boundary_eps,
                    temperature_count=args.high_t_count,
                )
            except ValueError:
                pass
        if args.save_every and completed % args.save_every == 0:
            _save_current(
                output_path, args, calibration, arrays, records,
                A_boundary_records, gap_boundary_records,
            )
    A_boundary_records.sort(key=lambda row: row["metastability"])
    _save_current(
        output_path, args, calibration, arrays, records,
        A_boundary_records, gap_boundary_records,
    )

    low_rows = [
        i for i in low_temperature_row_order(low_temperatures, args.seed_temperature)
        if np.all(np.isfinite(arrays["low_metastability_grid"][i]))
    ]
    valid_high_columns = [
        j for j in range(args.m_count)
        if np.all(np.isfinite(arrays["high_temperature_grid"][:, j]))
    ]
    total = len(low_rows) * args.m_count + len(valid_high_columns) * (args.high_t_count - 1)
    latest = {"low": {}, "high": {}}
    successful = {"low": {}, "high": {}}
    failures = {"low": set(), "high": set()}
    since_save = 0

    def accept(block: str, result: dict[str, Any], first_pass: bool) -> None:
        nonlocal since_save
        row = result["row"]
        add_lte_temperature_fields(
            row,
            T_N_MeV=float(result["T_MeV"]),
            B14=B14,
            ms=args.ms,
            upB=args.upB,
        )
        i, j = int(result["row_index"]), int(result["col_index"])
        key = (i, j)
        prefix = f"{block}_"
        row["mesh_block"] = block
        row["block_row_index"] = i
        row["block_col_index"] = j
        arrays[f"{prefix}temperature_grid"][i, j] = result["T_MeV"]
        arrays[f"{prefix}metastability_grid"][i, j] = result["metastability"]
        arrays[f"{prefix}status_grid"][i, j] = (
            STATUS_FINITE_U if row.get("success") else STATUS_NUMERICAL
        )
        arrays[f"{prefix}lte_jump_closure_status_grid"][i, j] = int(
            row.get("lte_jump_closure_status", LTE_JUMP_NOT_ATTEMPTED)
        )
        arrays[f"{prefix}lte_isobar_closure_status_grid"][i, j] = int(
            row.get("lte_isobar_closure_status", LTE_ISOBAR_NOT_ATTEMPTED)
        )
        arrays[f"{prefix}nB_crit_T_grid"][i, j] = _as_finite_float(
            row.get("nB_crit_T_MeV3")
        )
        arrays[f"{prefix}muB_crit_T_grid"][i, j] = _as_finite_float(
            row.get("muB_crit_T_MeV")
        )
        records.append(row)
        latest[block][key] = result
        if row.get("success"):
            successful[block][key] = row
            failures[block].discard(key)
            arrays[f"{prefix}delta_T_interface_MeV_grid"][i, j] = row[
                "delta_T_interface_MeV"
            ]
            arrays[f"{prefix}T_Qstar_LTE_MeV_grid"][i, j] = row[
                "T_Qstar_LTE_MeV"
            ]
            arrays[f"{prefix}beta_LTE_grid"][i, j] = row["beta_LTE"]
            arrays[f"{prefix}velocity_grid_m_s"][i, j] = row["u_N_m_s"]
            arrays[f"{prefix}u_N_grid"][i, j] = row["u_N"]
            arrays[f"{prefix}lambda_n_grid"][i, j] = row["lambda_n"]
            arrays[f"{prefix}A_boundary_grid"][i, j] = row["A_boundary"]
            arrays[f"{prefix}A_boundary_exact_local_grid"][i, j] = (
                _as_finite_float(row.get("A_boundary_exact_local"))
            )
            arrays[f"{prefix}muK_A_exact_local_MeV_grid"][i, j] = (
                _as_finite_float(row.get("muK_A_exact_local_MeV"))
            )
            arrays[
                f"{prefix}lte_isobar_pressure_residual_norm_grid"
            ][i, j] = _as_finite_float(
                row.get("lte_isobar_pressure_residual_norm")
            )
            arrays[f"{prefix}a_interface_grid"][i, j] = row["a_interface"]
            arrays[f"{prefix}a_interface_LTE_grid"][i, j] = row["a_interface_LTE"]
            for field_name, row_key in (
                ("a_Qstar_LTE_derived_grid", "a_Qstar_LTE_derived"),
                ("muB_Qstar_LTE_MeV_grid", "muB_Qstar_LTE_MeV"),
                ("muK_Qstar_LTE_MeV_grid", "muK_Qstar_LTE_MeV"),
                ("nB_Qstar_LTE_MeV3_grid", "nB_Qstar_LTE_MeV3"),
                ("nK_Qstar_LTE_MeV3_grid", "nK_Qstar_LTE_MeV3"),
                ("u_Qstar_LTE_grid", "u_Qstar_LTE"),
                ("lte_jump_a_residual_grid", "lte_jump_a_residual"),
                (
                    "lte_jump_energy_residual_norm_grid",
                    "lte_jump_energy_residual_norm",
                ),
                (
                    "lte_jump_momentum_residual_norm_grid",
                    "lte_jump_momentum_residual_norm",
                ),
            ):
                arrays[f"{prefix}{field_name}"][i, j] = _as_finite_float(
                    row.get(row_key)
                )
        elif first_pass:
            failures[block].add(key)
        since_save += 1
        if args.save_every and since_save >= args.save_every:
            _save_current(
                output_path, args, calibration, arrays, records,
                A_boundary_records, gap_boundary_records,
            )
            since_save = 0

    progress = tqdm(
        total=total,
        desc="LTE temperature-jump contour",
        unit="pt",
        dynamic_ncols=True,
    )
    try:
        seed_row = low_rows[0] if low_rows else -1
        for row_index in low_rows:
            metastabilities = arrays["low_metastability_grid"][row_index]
            if row_index == seed_row:
                row_task = LowRowTask(
                    row_index=row_index,
                    T_N=float(low_temperatures[row_index]),
                    metastability_values=metastabilities,
                    n0=n0,
                    nB_crit0=nB_crit0,
                    B14=B14,
                    xi=args.xi,
                    ms=args.ms,
                    nm_type=args.nm_type,
                    upB=args.upB,
                    interface_fraction_mode=args.interface_fraction_mode,
                    seed_temperature=args.seed_temperature,
                    default_endpoint_guess=args.endpoint_guess,
                )
                iterator = _iter_low_points_serial([row_task])
            else:
                neighbor_row = row_index - 1 if row_index > seed_row else row_index + 1
                work = []
                for j in low_metastability_solve_order(args.m_count):
                    endpoint = _finite_endpoint(successful["low"].get((neighbor_row, j)))
                    source = (
                        "adjacent_temperature_same_adaptive_column"
                        if endpoint is not None
                        else "fallback_after_failed_temperature_neighbor"
                    )
                    task = AnalyticPointTask(
                        T_N=float(low_temperatures[row_index]),
                        metastability=float(metastabilities[j]),
                        n0=n0,
                        nB_crit0=nB_crit0,
                        B14=B14,
                        xi=args.xi,
                        ms=args.ms,
                        nm_type=args.nm_type,
                        upB=args.upB,
                        interface_fraction_mode=args.interface_fraction_mode,
                        endpoint_initial_guess=endpoint or args.endpoint_guess,
                    )
                    work.append(
                        (
                            row_index, j, task,
                            "up" if row_index > seed_row else "down",
                            source, args.seed_temperature, 0, None,
                        )
                    )
                iterator = (
                    map(calculate_velocity_point, work)
                    if args.workers == 1
                    else _parallel_map_unordered(calculate_velocity_point, work, args.workers)
                )
            for result in iterator:
                progress.set_postfix_str(
                    f"low: m={result['metastability']:.3g}, T={result['T_MeV']:.3g}",
                    refresh=False,
                )
                accept("low", result, first_pass=True)
                progress.update(1)

        seam_row = args.t_count - 1
        copy_low_seam_to_high(
            arrays=arrays,
            low_seam_row=seam_row,
            high_columns=valid_high_columns,
            field_names=field_names,
            integer_field_names=(
                "lte_isobar_closure_status_grid",
                "lte_jump_closure_status_grid",
            ),
        )
        for j in valid_high_columns:
            seam_row_data = successful["low"].get((seam_row, j))
            if seam_row_data is not None:
                successful["high"][(0, j)] = seam_row_data

        column_tasks = []
        for j in valid_high_columns:
            seam_endpoint = _finite_endpoint(successful["high"].get((0, j)))
            column_tasks.append(
                ColumnTask(
                    col_index=j,
                    metastability=float(high_metastabilities[j]),
                    temperature_values=arrays["high_temperature_grid"][1:, j],
                    n0=n0,
                    nB_crit0=nB_crit0,
                    B14=B14,
                    xi=args.xi,
                    ms=args.ms,
                    nm_type=args.nm_type,
                    upB=args.upB,
                    interface_fraction_mode=args.interface_fraction_mode,
                    temperature_cut=float(arrays["A_boundary_temperature_values"][j]),
                    boundary_eps=args.boundary_eps,
                    seed_temperature=args.split_temperature,
                    default_endpoint_guess=seam_endpoint or args.endpoint_guess,
                )
            )
        high_iterator = (
            _iter_points_serial(column_tasks)
            if args.workers == 1
            else _iter_points_parallel(column_tasks, args.workers)
        )
        for result in high_iterator:
            result["row_index"] += 1
            result["row"]["row_index"] += 1
            progress.set_postfix_str(
                f"high: m={result['metastability']:.3g}, T={result['T_MeV']:.3g}",
                refresh=False,
            )
            accept("high", result, first_pass=True)
            progress.update(1)

        retries: list[tuple[str, tuple[Any, ...]]] = []
        if args.neighbor_retry:
            for block in ("low", "high"):
                for i, j in sorted(failures[block]):
                    endpoint, source = _immediate_retry_endpoint(i, j, successful[block])
                    if endpoint is None:
                        continue
                    original = latest[block][(i, j)]
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
                            block,
                            (i, j, task, "retry", source, args.seed_temperature, 1, (i, j)),
                        )
                    )
        if retries:
            progress.total = int(progress.total or 0) + len(retries)
            progress.refresh()
            if args.workers == 1:
                retry_results = (
                    (block, calculate_velocity_point(work)) for block, work in retries
                )
            else:
                pool = ProcessPoolExecutor(max_workers=args.workers)
                futures = {
                    pool.submit(calculate_velocity_point, work): block
                    for block, work in retries
                }
                retry_results = ((futures[future], future.result()) for future in as_completed(futures))
            try:
                for block, result in retry_results:
                    result["row"]["retry_replaced_failure"] = bool(result["row"].get("success"))
                    accept(block, result, first_pass=False)
                    progress.update(1)
            finally:
                if args.workers != 1:
                    pool.shutdown()
    finally:
        progress.close()

    _save_current(
        output_path, args, calibration, arrays, records,
        A_boundary_records, gap_boundary_records,
    )
    successes = sum(bool(row.get("success")) for row in records)
    print("Done.", flush=True)
    print(
        f"Saved rows: {len(records)}; success={successes}; "
        f"failure={len(records) - successes}",
        flush=True,
    )
    print(f"Saved A-boundary rows: {len(A_boundary_records)}", flush=True)
    print(f"Saved gap-boundary rows: {len(gap_boundary_records)}", flush=True)


if __name__ == "__main__":
    main()
