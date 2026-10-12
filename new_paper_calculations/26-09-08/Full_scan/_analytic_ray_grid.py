#!/usr/bin/env python3
"""Shared engine for the absolute-density analytical elliptical-ray grid."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"matplotlib-cache-{os.getuid()}"),
)
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import brentq

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver import phase_velocity


ANALYTIC_CLOSURE_LABEL = "hydro_jump_eigenvalue"
ANALYTIC_FORMULA_VARIANT = "piecewise_constant_lambda_n"
COMPOSITION_DEFINITION = "a_local_equals_nK_over_nB"
DENSITY_RATIO_DEFINITION = "lambda_n_equals_nB_0minus_over_nB_inf"
GAP_DEFINITION = "muB_cold_at_P_0minus_minus_h_0minus_over_nB_0minus"
ENTROPY_JUMP_DEFINITION = "s_inf_over_nB_inf_minus_s_0minus_over_nB_0minus"
PRIMARY_COORDINATE_SYSTEM = "physical_cartesian_curvilinear_grid"
MESH_PARAMETERIZATION = "dimensionless_elliptical_theta_safe_radial_fraction"

STATUS_FINITE = 0
STATUS_NUMERICAL_FAILURE = 3
STATUS_UNSCANNED = 5
OUTER_FORBIDDEN_STATUSES = frozenset(
    {"A_forbidden", "interface_temperature_unreachable"}
)
# phase_velocity spells the interface-fraction ceiling several ways depending on
# which guard trips; mirror its own predicate (see the endpoint_domain_message
# scan in analytic_velocity_bound) so a genuine A -> 1 hit is never demoted to a
# numerical failure.
A_CEILING_MESSAGE_MARKERS = (
    "a_0plus must satisfy 0 < a_0plus",
    "a_0plus_LTE must satisfy 0 < a_0plus_LTE < 1",
    "A_boundary must satisfy 0 < A_boundary < 1",
)


def _outer_boundary_kind(status: str) -> str:
    if str(status) == "A_forbidden":
        return "A_equals_one"
    if str(status) == "interface_temperature_unreachable":
        return "T_0plus_reachability_edge"
    raise ValueError(f"unrecognized outer-boundary status {status!r}")


class _SilentProgress:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def update(self, _count: int = 1) -> None:
        return None

    def set_postfix_str(self, _value: str) -> None:
        return None


def _make_progress(progress_factory, **kwargs):
    if progress_factory is None:
        return _SilentProgress()
    return progress_factory(**kwargs)


class RayTransform:
    """Dimensionless elliptical coordinates used only for mesh construction."""

    def __init__(
        self,
        density_origin_over_n0: float,
        density_scale_over_n0: float,
        temperature_scale_MeV: float,
    ):
        self.density_origin_over_n0 = float(density_origin_over_n0)
        self.density_scale_over_n0 = float(density_scale_over_n0)
        self.temperature_scale_MeV = float(temperature_scale_MeV)
        if self.density_scale_over_n0 <= 0.0 or self.temperature_scale_MeV <= 0.0:
            raise ValueError("coordinate scales must be positive")

    def to_hidden(self, nB_over_n0, temperature_MeV):
        density = np.asarray(nB_over_n0, dtype=float)
        temperature = np.asarray(temperature_MeV, dtype=float)
        X = (
            density - self.density_origin_over_n0
        ) / self.density_scale_over_n0
        Y = temperature / self.temperature_scale_MeV
        return np.hypot(X, Y), np.arctan2(Y, X)

    def to_physical(self, rho, theta_rad):
        radius = np.asarray(rho, dtype=float)
        theta = np.asarray(theta_rad, dtype=float)
        density = (
            self.density_origin_over_n0
            + self.density_scale_over_n0 * radius * np.cos(theta)
        )
        temperature = self.temperature_scale_MeV * radius * np.sin(theta)
        return density, temperature

    def metadata(self) -> dict[str, Any]:
        return {
            "density_origin_over_n0": self.density_origin_over_n0,
            "density_scale_over_n0": self.density_scale_over_n0,
            "temperature_scale_MeV": self.temperature_scale_MeV,
            "rho_definition": (
                "sqrt(((nB_over_n0-density_origin_over_n0)/"
                "density_scale_over_n0)^2+(T_MeV/temperature_scale_MeV)^2)"
            ),
            "theta_definition": (
                "atan2(T_MeV/temperature_scale_MeV,"
                "(nB_over_n0-density_origin_over_n0)/density_scale_over_n0)"
            ),
        }


def chebyshev_closed_axis(lower: float, upper: float, count: int) -> np.ndarray:
    if int(count) < 2:
        raise ValueError("closed Chebyshev axis requires at least two points")
    parameter = np.linspace(0.0, 1.0, int(count))
    clustered = 0.5 * (1.0 - np.cos(np.pi * parameter))
    return float(lower) + (float(upper) - float(lower)) * clustered


def chebyshev_open_axis(count: int, buffer_fraction: float) -> np.ndarray:
    if int(count) < 1:
        raise ValueError("open Chebyshev axis requires at least one point")
    buffer = float(buffer_fraction)
    if not 0.0 < buffer < 0.5:
        raise ValueError("buffer_fraction must satisfy 0 < buffer_fraction < 0.5")
    parameter = (np.arange(int(count), dtype=float) + 0.5) / int(count)
    clustered = 0.5 * (1.0 - np.cos(np.pi * parameter))
    return buffer + (1.0 - 2.0 * buffer) * clustered


def build_safe_ray_mesh(
    *,
    transform: RayTransform,
    theta_axis_rad: np.ndarray,
    radial_fraction_axis: np.ndarray,
    rho_inner: np.ndarray,
    rho_outer: np.ndarray,
) -> dict[str, np.ndarray]:
    theta_axis = np.asarray(theta_axis_rad, dtype=float)
    eta_axis = np.asarray(radial_fraction_axis, dtype=float)
    inner = np.asarray(rho_inner, dtype=float)
    outer = np.asarray(rho_outer, dtype=float)
    if theta_axis.ndim != 1 or eta_axis.ndim != 1:
        raise ValueError("theta and radial-fraction axes must be one-dimensional")
    if inner.shape != theta_axis.shape or outer.shape != theta_axis.shape:
        raise ValueError("radial boundary arrays must match theta_axis")
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        raise ValueError("radial boundaries must be finite")
    if not np.all(outer > inner):
        raise ValueError("outer radial boundary must exceed inner boundary")
    if not np.all((eta_axis > 0.0) & (eta_axis < 1.0)):
        raise ValueError("radial fractions must lie strictly inside (0,1)")

    theta_grid = np.repeat(theta_axis[:, None], eta_axis.size, axis=1)
    eta_grid = np.repeat(eta_axis[None, :], theta_axis.size, axis=0)
    rho_grid = inner[:, None] + eta_grid * (outer - inner)[:, None]
    density_grid, temperature_grid = transform.to_physical(rho_grid, theta_grid)
    return {
        "theta_axis_rad": theta_axis,
        "theta_axis_deg": np.degrees(theta_axis),
        "radial_fraction_axis": eta_axis,
        "theta_grid_rad": theta_grid,
        "radial_fraction_grid": eta_grid,
        "rho_inner_boundary": inner,
        "rho_outer_boundary": outer,
        "rho_grid": rho_grid,
        "nB_grid_over_n0": np.asarray(density_grid, dtype=float),
        "temperature_grid_MeV": np.asarray(temperature_grid, dtype=float),
    }


def radial_solve_order(
    temperature_values_MeV: np.ndarray,
    seed_temperature_MeV: float,
) -> list[tuple[int, str]]:
    temperatures = np.asarray(temperature_values_MeV, dtype=float)
    if temperatures.ndim != 1 or temperatures.size < 1:
        raise ValueError("temperature row must be one-dimensional and nonempty")
    if not np.all(np.isfinite(temperatures)):
        raise ValueError("temperature row must be finite")
    seed = int(np.argmin(np.abs(temperatures - float(seed_temperature_MeV))))
    return [
        (seed, "seed"),
        *((index, "up") for index in range(seed + 1, temperatures.size)),
        *((index, "down") for index in range(seed - 1, -1, -1)),
    ]


def _finite_pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        pair = (float(value[0]), float(value[1]))
    except Exception:
        return None
    if not np.all(np.isfinite(pair)) or pair[0] <= 0.0 or pair[1] <= 0.0:
        return None
    return pair


def _as_finite_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    return number if np.isfinite(number) else float(default)


def _calibration_root(
    function: Callable[[float], float],
    grid: np.ndarray,
    prefer: float,
) -> float:
    coordinates = np.asarray(grid, dtype=float)
    values = np.asarray(
        [
            _as_finite_float(function(value))
            for value in coordinates
        ],
        dtype=float,
    )
    finite = np.isfinite(values)
    exact = np.where(finite & np.isclose(values, 0.0, atol=1.0e-8, rtol=0.0))[0]
    if exact.size:
        index = exact[np.argmin(np.abs(coordinates[exact] - float(prefer)))]
        return float(coordinates[index])
    changes = np.where(
        finite[:-1] & finite[1:] & (values[:-1] * values[1:] < 0.0)
    )[0]
    if not changes.size:
        raise RuntimeError("No sign change found for calibration root")
    midpoints = 0.5 * (coordinates[changes] + coordinates[changes + 1])
    index = int(changes[np.argmin(np.abs(midpoints - float(prefer)))])
    return float(
        brentq(
            function,
            float(coordinates[index]),
            float(coordinates[index + 1]),
            xtol=1.0e-7,
            rtol=1.0e-10,
            maxiter=200,
        )
    )


def calibrate_b14(
    nB_crit0: float,
    ms: float,
    nm_type: str,
) -> tuple[float, float, float]:
    parameter = para.paraQMCRMF3
    pressure_crit0 = float(
        phase_velocity.PNM_n(
            float(nB_crit0),
            0.0,
            param=parameter,
            NM_type=str(nm_type),
        )
    )
    muB_crit0 = _calibration_root(
        lambda muB: (
            phase_velocity.PNM(
                muB,
                0.0,
                param=parameter,
                NM_type=str(nm_type),
            )
            - pressure_crit0
        ),
        np.linspace(900.0, 1500.0, 220),
        prefer=1050.0,
    )
    B14 = _calibration_root(
        lambda bag: (
            phase_velocity.PQM(muB_crit0, 0.0, bag, 0.0, ms=float(ms))
            - pressure_crit0
        ),
        np.linspace(120.0, 260.0, 281),
        prefer=180.0,
    )
    return float(B14), float(muB_crit0), float(pressure_crit0)


def muB_from_nB(
    nB_target: float,
    temperature_MeV: float,
    nm_type: str,
    prefer: float = 1100.0,
) -> float:
    target = float(nB_target)
    temperature = float(temperature_MeV)
    if str(nm_type) == "PNM":
        center = float(prefer) if np.isfinite(prefer) else 1100.0
        try:
            return float(
                phase_velocity.muB_from_nB_physical(
                    target,
                    temperature,
                    param=para.paraQMCRMF3,
                    NM_type="PNM",
                    muB_lo=max(850.0, center - 350.0),
                    muB_hi=center + 350.0,
                    scan_points=16,
                    auto_expand=True,
                )
            )
        except Exception:
            return float(
                phase_velocity.muB_from_nB_physical(
                    target,
                    temperature,
                    param=para.paraQMCRMF3,
                    NM_type="PNM",
                    muB_lo=850.0,
                    muB_hi=3000.0,
                    scan_points=64,
                    auto_expand=True,
                )
            )

    def residual(muB: float) -> float:
        return float(
            phase_velocity.nB_NM(
                float(muB),
                temperature,
                param=para.paraQMCRMF3,
                NM_type=str(nm_type),
            )
            - target
        )

    tolerance = 1.0e-8 * max(abs(target), 1.0)

    def validated_candidate(candidate: float) -> float | None:
        try:
            candidate_residual = residual(float(candidate))
        except Exception:
            return None
        if (
            np.isfinite(candidate_residual)
            and abs(candidate_residual) <= tolerance
        ):
            return float(candidate)
        return None

    for width in (20.0, 50.0, 100.0, 200.0, 400.0, 700.0):
        lower = max(1.0, float(prefer) - width)
        upper = float(prefer) + width
        try:
            f_lower = residual(lower)
            f_upper = residual(upper)
        except Exception:
            continue
        if not np.all(np.isfinite((f_lower, f_upper))):
            continue
        if abs(f_lower) <= tolerance:
            return lower
        if abs(f_upper) <= tolerance:
            return upper
        if f_lower * f_upper < 0.0:
            try:
                candidate = brentq(
                    residual,
                    lower,
                    upper,
                    xtol=1.0e-7,
                    rtol=1.0e-10,
                    maxiter=100,
                )
            except Exception:
                continue
            validated = validated_candidate(candidate)
            if validated is not None:
                return validated

    coordinates = np.linspace(850.0, 1800.0, 100)
    values = np.full(coordinates.shape, np.nan, dtype=float)
    for index, coordinate in enumerate(coordinates):
        try:
            values[index] = residual(float(coordinate))
        except Exception:
            continue

    candidates: list[float] = []
    exact = np.where(np.isfinite(values) & (np.abs(values) <= tolerance))[0]
    candidates.extend(float(coordinates[index]) for index in exact)
    changes = np.where(
        np.isfinite(values[:-1])
        & np.isfinite(values[1:])
        & (values[:-1] * values[1:] < 0.0)
    )[0]
    for index in changes:
        try:
            candidate = brentq(
                residual,
                float(coordinates[index]),
                float(coordinates[index + 1]),
                xtol=1.0e-7,
                rtol=1.0e-10,
                maxiter=100,
            )
        except Exception:
            continue
        validated = validated_candidate(candidate)
        if validated is not None:
            candidates.append(validated)
    if not candidates:
        raise RuntimeError(
            "Could not find a residual-validated muB root for the requested "
            "nuclear-matter density"
        )
    return min(candidates, key=lambda candidate: abs(candidate - float(prefer)))


def enthalpy_gap_at_absolute_density(
    *,
    nB_over_n0: float,
    temperature_MeV: float,
    n0: float,
    B14: float,
    nm_type: str,
    muB_prefer: float = 1100.0,
) -> tuple[float, float, dict[str, float]]:
    muB_0minus = muB_from_nB(
        float(nB_over_n0) * float(n0),
        float(temperature_MeV),
        str(nm_type),
        prefer=float(muB_prefer),
    )
    state = phase_velocity._analytic_nuclear_state(
        muB_0minus,
        float(temperature_MeV),
        param=para.paraQMCRMF3,
        NM_type=str(nm_type),
    )
    gap, muB_cold = phase_velocity._slow_front_enthalpy_gap(state, float(B14))
    return float(gap), float(muB_0minus), {
        "muB_0minus_MeV": float(muB_0minus),
        "muB_cold_MeV": float(muB_cold),
    }


# A is the maximum interface fraction obtained at the cold endpoint of the
# conversion-layer trajectory. Finite interface temperatures select a < A and
# therefore do not define the A = 1 boundary.
BOUNDARY_T_0PLUS_MEV = 0.0
A_BOUNDARY_RESIDUAL_TOLERANCE = 5.0e-3
A_BOUNDARY_REFERENCE_XI = 0.0
A_BOUNDARY_SOURCE = "26-08-02_cold_extreme_endothermic_numerical_I2"


def enthalpy_gap_at_fixed_T0plus_absolute_density(
    *,
    nB_over_n0: float,
    temperature_MeV: float,
    T_0plus_MeV: float,
    n0: float,
    B14: float,
    nm_type: str,
    ms: float = 0.0,
    upB: int = 5000,
    muB_prefer: float = 1100.0,
) -> tuple[float, float, dict[str, float]]:
    """Return the static local-isobar enthalpy gap at prescribed T(0+)."""
    muB_0minus = muB_from_nB(
        float(nB_over_n0) * float(n0),
        float(temperature_MeV),
        str(nm_type),
        prefer=float(muB_prefer),
    )
    nuclear_state = phase_velocity._analytic_nuclear_state(
        muB_0minus,
        float(temperature_MeV),
        param=para.paraQMCRMF3,
        NM_type=str(nm_type),
    )
    pressure_target = float(nuclear_state["P_0minus"])
    T_0plus = float(T_0plus_MeV)

    def pressure_residual(muB: float) -> float:
        return float(
            phase_velocity.PQM(
                float(muB), 0.0, float(B14), T_0plus,
                ms=float(ms), upB=int(upB),
            ) - pressure_target
        )

    lower = 1.0e-6
    upper = max(1500.0, float(muB_0minus) + 500.0)
    f_lower = pressure_residual(lower)
    f_upper = pressure_residual(upper)
    for _ in range(12):
        if np.isfinite(f_lower) and np.isfinite(f_upper) and f_lower * f_upper <= 0.0:
            break
        upper *= 1.5
        f_upper = pressure_residual(upper)
    else:
        raise RuntimeError(
            "fixed-T(0+) quark isobar does not bracket the nuclear pressure"
        )
    muB_0plus = float(
        brentq(
            pressure_residual,
            lower,
            upper,
            xtol=1.0e-8,
            rtol=1.0e-11,
            maxiter=100,
        )
    )
    nB_0plus = float(
        phase_velocity.nB_QM(
            muB_0plus, 0.0, float(B14), T_0plus,
            ms=float(ms), upB=int(upB),
        )
    )
    h_0plus = float(
        phase_velocity.hQM(
            muB_0plus, 0.0, float(B14), T_0plus,
            ms=float(ms), upB=int(upB),
        )
    )
    if not np.isfinite(nB_0plus) or nB_0plus <= 0.0:
        raise RuntimeError("fixed-T(0+) isobar returned non-positive quark density")
    h_over_nB_0plus = float(h_0plus / nB_0plus)
    gap = float(h_over_nB_0plus - nuclear_state["h_over_nB_0minus"])
    return gap, float(muB_0minus), {
        "muB_0minus_MeV": float(muB_0minus),
        "muB_0plus_MeV": muB_0plus,
        "T_0plus_MeV": T_0plus,
        "nB_0plus_MeV3": nB_0plus,
        "h_over_nB_0plus_MeV": h_over_nB_0plus,
        "h_over_nB_0minus_MeV": float(nuclear_state["h_over_nB_0minus"]),
        "pressure_residual_MeV4": pressure_residual(muB_0plus),
    }


def semi_analytic_extreme_endothermic_bound(
    muB_0minus: float,
    T_0minus: float,
    B14: float,
    *,
    xi: float,
    ms: float,
    nm_type: str,
    upB: int,
    initial_guess: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Evaluate the cold a(0+) = A numerical-I2 eigenvalue closure.

    This is the boundary closure used by the successful 26-08-02 ray scan.
    The public semi-analytical function now selects a finite-T(0+) interface
    state, so the cold limiting closure is kept local to boundary construction.
    """
    muB_0minus = float(muB_0minus)
    T_0minus = float(T_0minus)
    B14 = float(B14)
    xi = float(xi)
    if not (-1.0 < xi < 1.0):
        raise RuntimeError("xi must satisfy -1 < xi < 1")

    nuclear_state = phase_velocity._analytic_nuclear_state(
        muB_0minus,
        T_0minus,
        param=para.paraQMCRMF3,
        NM_type=str(nm_type),
    )
    nuclear_state["muB_0minus"] = muB_0minus
    nuclear_state["T_0minus"] = T_0minus
    endpoint_cache = {"value": _finite_pair(initial_guess)}
    best = {"theta": np.nan, "residual": np.inf, "data": None, "message": ""}
    A_domain_message = ""

    def evaluate_log_u(theta: float) -> tuple[float, dict[str, Any]]:
        u_0minus = float(np.exp(np.clip(float(theta), -700.0, np.log(0.999999))))
        endpoint = phase_velocity._solve_analytic_inf_endpoint_for_u_0minus(
            u_0minus,
            nuclear_state,
            B14,
            ms=float(ms),
            upB=int(upB),
            initial_guess=endpoint_cache["value"],
        )
        endpoint_cache["value"] = _finite_pair(endpoint["endpoint_initial_guess"])
        trajectory = phase_velocity._analytic_layer_trajectory(
            float(endpoint["P_inf"]),
            float(endpoint["h_over_nB_inf"]),
            float(endpoint["muB_inf"]),
            float(endpoint["T_inf"]),
            B14,
            ms=float(ms),
            upB=int(upB),
            T_cold=1.0e-3,
        )
        A_boundary = float(trajectory["A_boundary"])
        if bool(trajectory["saturated"]) or A_boundary >= 1.0 - 1.0e-9:
            raise RuntimeError("A_boundary must satisfy 0 < A_boundary < 1")
        if A_boundary <= 0.0 or not np.isfinite(A_boundary):
            raise RuntimeError("A_boundary must be positive and finite")
        if float(trajectory["T"][-1]) > 1.001e-3:
            raise RuntimeError(
                "conserved-layer trajectory did not reach its cold endpoint"
            )

        I2 = phase_velocity._analytic_I2_numerical(
            trajectory,
            A_boundary,
            B14,
            ms=float(ms),
            upB=int(upB),
        )
        lambda_n = float(nuclear_state["nB_0minus"] / endpoint["nB_inf"])
        denominator = float(
            lambda_n**2 * (1.0 - A_boundary) * (1.0 + xi * A_boundary)
        )
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("numerical-I2 velocity denominator is non-physical")
        formula_squared = float(2.0 * I2 / denominator)
        residual = float(formula_squared - u_0minus**2)
        data = {
            **endpoint,
            "u_0minus": u_0minus,
            "A_boundary": A_boundary,
            "a_0plus": A_boundary,
            "T_0plus": 0.0,
            "I2_numerical": float(I2),
            "lambda_n": lambda_n,
            "velocity_closure": "numerical_I2",
            "interface_fraction_mode": "extreme_endothermic",
            "residual": residual,
        }
        if abs(residual) < abs(float(best["residual"])):
            best.update(
                {
                    "theta": float(theta),
                    "residual": residual,
                    "data": data,
                    "message": "",
                }
            )
        return residual, data

    valid_scan: list[tuple[float, float, dict[str, Any]]] = []
    for theta in np.linspace(np.log(1.0e-12), np.log(0.99), 72):
        try:
            residual, data = evaluate_log_u(float(theta))
        except Exception as exc:
            message = str(exc)
            best["message"] = message
            if "A_boundary must satisfy 0 < A_boundary < 1" in message:
                A_domain_message = message
            continue
        if np.isfinite(residual):
            valid_scan.append((float(theta), float(residual), data))

    if not valid_scan:
        phase_velocity._raise_scan_failure(
            nuclear_state,
            B14,
            "Semi-analytical extreme-endothermic eigenvalue scan failed: "
            + (A_domain_message or str(best["message"]) or "no valid evaluations"),
        )

    exact_data = None
    bracket = None
    previous_theta, previous_residual, _ = valid_scan[0]
    if abs(previous_residual) <= 1.0e-18:
        exact_data = valid_scan[0][2]
    else:
        for theta, residual, data in valid_scan[1:]:
            if abs(residual) <= 1.0e-18:
                exact_data = data
                break
            if previous_residual * residual < 0.0:
                bracket = (previous_theta, theta)
                break
            previous_theta, previous_residual = theta, residual

    if exact_data is None and bracket is not None:
        theta_root = float(
            brentq(
                lambda value: evaluate_log_u(float(value))[0],
                bracket[0],
                bracket[1],
                xtol=1.0e-12,
                rtol=1.0e-12,
                maxiter=100,
            )
        )
        _, exact_data = evaluate_log_u(theta_root)
    if exact_data is None:
        best_suffix = (
            f"; best theta={best['theta']:.6g}, best residual={best['residual']:.6e}"
            if best["data"] is not None
            else ""
        )
        phase_velocity._raise_scan_failure(
            nuclear_state,
            B14,
            "Semi-analytical extreme-endothermic eigenvalue solve found no sign change"
            + best_suffix,
        )
    return exact_data


def evaluate_A_state(
    *,
    nB_over_n0: float,
    temperature_MeV: float,
    n0: float,
    B14: float,
    xi: float,
    ms: float,
    nm_type: str,
    upB: int,
    interface_fraction_mode: str,
    velocity_closure: str = "closed_form",
    T_0plus_MeV: float | None = None,
    endpoint_guess: tuple[float, float] | None = None,
    muB_prefer: float = 1100.0,
) -> dict[str, Any]:
    method_by_closure = {
        "closed_form": phase_velocity.analytic_velocity_bound,
        "numerical_I2": phase_velocity.semi_analytic_velocity_bound,
    }
    try:
        velocity_method = method_by_closure[str(velocity_closure)]
    except KeyError as exc:
        raise ValueError(
            "velocity_closure must be 'closed_form' or 'numerical_I2'"
        ) from exc
    muB_0minus = np.nan
    try:
        muB_0minus = muB_from_nB(
            float(nB_over_n0) * float(n0),
            float(temperature_MeV),
            str(nm_type),
            prefer=float(muB_prefer),
        )
        boundary_T_0plus = (
            float(BOUNDARY_T_0PLUS_MEV)
            if T_0plus_MeV is None
            else float(T_0plus_MeV)
        )
        if (
            str(interface_fraction_mode) == "extreme_endothermic"
            and str(velocity_closure) == "numerical_I2"
        ):
            result = semi_analytic_extreme_endothermic_bound(
                muB_0minus,
                float(temperature_MeV),
                float(B14),
                xi=float(xi),
                ms=float(ms),
                nm_type=str(nm_type),
                upB=int(upB),
                initial_guess=_finite_pair(endpoint_guess),
            )
        else:
            result = velocity_method(
                muB_0minus,
                float(temperature_MeV),
                float(B14),
                xi=float(xi),
                ms=float(ms),
                param=para.paraQMCRMF3,
                NM_type=str(nm_type),
                upB=int(upB),
                initial_guess=_finite_pair(endpoint_guess),
                T_0plus=boundary_T_0plus,
            )
        return {
            "status": "success",
            "A_boundary": float(result["A_boundary"]),
            "endpoint_solution": (
                float(result["muB_inf"]),
                float(result["T_inf"]),
            ),
            "muB_0minus_MeV": float(muB_0minus),
            "velocity_closure": str(result.get("velocity_closure", velocity_closure)),
            "message": "",
        }
    except phase_velocity.SlowFrontNoSolution as exc:
        return {
            "status": "gap_forbidden",
            "A_boundary": np.nan,
            "endpoint_solution": None,
            "muB_0minus_MeV": _as_finite_float(muB_0minus),
            "message": str(exc),
        }
    except Exception as exc:
        message = str(exc)
        A_failure = any(
            marker in message for marker in A_CEILING_MESSAGE_MARKERS
        )
        interface_unreachable = (
            "T_0plus lies below the resolved conserved-layer trajectory" in message
        )
        return {
            "status": (
                "A_forbidden"
                if A_failure
                else (
                    "interface_temperature_unreachable"
                    if interface_unreachable
                    else "numerical_failure"
                )
            ),
            "A_boundary": np.nan,
            "endpoint_solution": None,
            "muB_0minus_MeV": _as_finite_float(muB_0minus),
            "message": message,
        }


def _sign_change_brackets(
    coordinates: np.ndarray,
    values: np.ndarray,
) -> list[tuple[float, float]]:
    x = np.asarray(coordinates, dtype=float)
    y = np.asarray(values, dtype=float)
    return [
        (float(x[index]), float(x[index + 1]))
        for index in range(x.size - 1)
        if np.all(np.isfinite((y[index], y[index + 1])))
        and y[index] * y[index + 1] <= 0.0
    ]


def solve_gap_on_path(
    *,
    coordinates: np.ndarray,
    physical_coordinates: Callable[[float], tuple[float, float]],
    n0: float,
    B14: float,
    nm_type: str,
    label: str,
    muB_hint: float = 1100.0,
    residual_tolerance_MeV: float = 1.0e-4,
    T_0plus_MeV: float | None = None,
    ms: float = 0.0,
    upB: int = 5000,
) -> dict[str, Any]:
    probes = np.asarray(coordinates, dtype=float)
    cache: dict[float, dict[str, Any]] = {}

    def evaluate(coordinate: float) -> dict[str, Any]:
        key = float(np.round(coordinate, 12))
        if key not in cache:
            nearest = min(
                cache.values(),
                key=lambda row: abs(row["coordinate"] - key),
                default=None,
            )
            prefer = (
                nearest["muB_0minus_MeV"]
                if nearest is not None and np.isfinite(nearest["muB_0minus_MeV"])
                else float(muB_hint)
            )
            density, temperature = physical_coordinates(key)
            try:
                if T_0plus_MeV is None:
                    gap, muB_0minus, details = enthalpy_gap_at_absolute_density(
                        nB_over_n0=density,
                        temperature_MeV=temperature,
                        n0=n0,
                        B14=B14,
                        nm_type=nm_type,
                        muB_prefer=prefer,
                    )
                else:
                    gap, muB_0minus, details = (
                        enthalpy_gap_at_fixed_T0plus_absolute_density(
                            nB_over_n0=density,
                            temperature_MeV=temperature,
                            T_0plus_MeV=float(T_0plus_MeV),
                            n0=n0,
                            B14=B14,
                            nm_type=nm_type,
                            ms=float(ms),
                            upB=int(upB),
                            muB_prefer=prefer,
                        )
                    )
                cache[key] = {
                    "status": "success",
                    "coordinate": key,
                    "nB_over_n0": density,
                    "T_MeV": temperature,
                    "gap_MeV": gap,
                    "muB_0minus_MeV": muB_0minus,
                    **details,
                }
            except Exception as exc:
                cache[key] = {
                    "status": "numerical_failure",
                    "coordinate": key,
                    "nB_over_n0": density,
                    "T_MeV": temperature,
                    "gap_MeV": np.nan,
                    "muB_0minus_MeV": np.nan,
                    "message": str(exc),
                }
        return cache[key]

    rows = [evaluate(value) for value in probes]
    brackets = _sign_change_brackets(
        probes,
        np.asarray([row["gap_MeV"] for row in rows]),
    )
    if len(brackets) != 1:
        raise RuntimeError(
            f"{label}: expected one Delta h sign change, found {len(brackets)}"
        )
    lower, upper = brackets[0]
    root = float(
        brentq(
            lambda value: evaluate(value)["gap_MeV"],
            lower,
            upper,
            xtol=1.0e-8,
            rtol=1.0e-11,
            maxiter=100,
        )
    )
    result = dict(evaluate(root))
    if abs(result["gap_MeV"]) > float(residual_tolerance_MeV):
        raise RuntimeError(f"{label}: Delta h residual did not close")
    result.update(
        {
            "boundary_kind": "Delta_h_equals_zero",
            "solve_kind": label,
            "evaluation_count": len(cache),
            "enthalpy_boundary_definition": (
                "cold_muB_at_P_0minus"
                if T_0plus_MeV is None
                else "fixed_T0plus_local_isobar"
            ),
            "T_0plus_MeV": (
                np.nan if T_0plus_MeV is None else float(T_0plus_MeV)
            ),
        }
    )
    return result


def solve_A_transition_on_path(
    *,
    coordinates: np.ndarray,
    physical_coordinates: Callable[[float], tuple[float, float]],
    n0: float,
    B14: float,
    xi: float,
    ms: float,
    nm_type: str,
    upB: int,
    interface_fraction_mode: str,
    label: str,
    velocity_closure: str = "closed_form",
    T_0plus_MeV: float | None = None,
    endpoint_hint: tuple[float, float] | None = None,
    muB_hint: float = 1100.0,
    bisection_steps: int = 10,
    residual_tolerance: float = A_BOUNDARY_RESIDUAL_TOLERANCE,
    allow_missing: bool = False,
) -> dict[str, Any]:
    probes = np.asarray(coordinates, dtype=float)
    cache: dict[float, dict[str, Any]] = {}

    def evaluate(
        coordinate: float,
        preferred_endpoint: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        key = float(np.round(coordinate, 12))
        if key not in cache:
            successful = [
                row
                for row in cache.values()
                if row.get("endpoint_solution") is not None
            ]
            nearest = min(
                successful,
                key=lambda row: abs(row["coordinate"] - key),
                default=None,
            )
            endpoint = (
                _finite_pair(preferred_endpoint)
                or (_finite_pair(nearest["endpoint_solution"]) if nearest else None)
                or _finite_pair(endpoint_hint)
            )
            prefer = (
                nearest["muB_0minus_MeV"]
                if nearest is not None and np.isfinite(nearest["muB_0minus_MeV"])
                else float(muB_hint)
            )
            density, temperature = physical_coordinates(key)
            row = evaluate_A_state(
                nB_over_n0=density,
                temperature_MeV=temperature,
                n0=n0,
                B14=B14,
                xi=xi,
                ms=ms,
                nm_type=nm_type,
                upB=upB,
                interface_fraction_mode=interface_fraction_mode,
                velocity_closure=velocity_closure,
                T_0plus_MeV=T_0plus_MeV,
                endpoint_guess=endpoint,
                muB_prefer=prefer,
            )
            if row["status"] == "numerical_failure" and endpoint is not None:
                row = evaluate_A_state(
                    nB_over_n0=density,
                    temperature_MeV=temperature,
                    n0=n0,
                    B14=B14,
                    xi=xi,
                    ms=ms,
                    nm_type=nm_type,
                    upB=upB,
                    interface_fraction_mode=interface_fraction_mode,
                    velocity_closure=velocity_closure,
                    T_0plus_MeV=T_0plus_MeV,
                    endpoint_guess=None,
                    muB_prefer=prefer,
                )
            row.update(
                {
                    "coordinate": key,
                    "nB_over_n0": density,
                    "T_MeV": temperature,
                }
            )
            cache[key] = row
        return cache[key]

    rows = [evaluate(value) for value in probes]

    def transition_brackets():
        ordered = sorted(rows, key=lambda row: float(row["coordinate"]))
        return [
            (
                float(lower_row["coordinate"]),
                float(upper_row["coordinate"]),
                lower_row,
                upper_row,
            )
            for lower_row, upper_row in zip(ordered[:-1], ordered[1:])
            if lower_row["status"] == "success"
            and upper_row["status"] in OUTER_FORBIDDEN_STATUSES
        ]

    transitions = transition_brackets()
    for _ in range(4):
        if transitions:
            break
        recovery_coordinates = _missing_A_recovery_coordinates(rows)
        if not recovery_coordinates:
            break
        rows.extend(evaluate(value) for value in recovery_coordinates)
        transitions = transition_brackets()
    success_indices = [
        index for index, row in enumerate(rows)
        if row["status"] == "success"
    ]
    first_success = success_indices[0] if success_indices else None
    clean_missing_boundary = bool(
        first_success is not None
        and all(
            row["status"] in {"gap_forbidden", "numerical_failure"}
            for row in rows[:first_success]
        )
        and all(row["status"] == "success" for row in rows[first_success:])
    )
    if not transitions and allow_missing and clean_missing_boundary:
        outer_row = rows[-1]
        if outer_row["status"] == "success":
            return {
                "status": "clipped",
                "boundary_kind": "physical_window_outer",
                "solve_kind": label,
                "coordinate": float(probes[-1]),
                "nB_over_n0": float(outer_row["nB_over_n0"]),
                "T_MeV": float(outer_row["T_MeV"]),
                "one_minus_A_residual": float(
                    1.0 - outer_row["A_boundary"]
                ),
                "A_below_boundary": float(outer_row["A_boundary"]),
                "endpoint_solution": outer_row.get("endpoint_solution"),
                "muB_0minus_MeV": _as_finite_float(
                    outer_row.get("muB_0minus_MeV")
                ),
                "prefix_numerical_failure_count": sum(
                    row["status"] == "numerical_failure"
                    for row in rows[:first_success]
                ),
                "evaluation_count": len(cache),
            }
    if len(transitions) != 1:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        raise RuntimeError(
            f"{label}: expected one success-to-outer-domain transition, found "
            f"{len(transitions)}; {counts}"
        )

    lower, upper, lower_row, upper_row = transitions[0]
    for _ in range(int(bisection_steps)):
        middle = 0.5 * (lower + upper)
        middle_row = evaluate(middle, lower_row.get("endpoint_solution"))
        if middle_row["status"] == "success":
            lower, lower_row = middle, middle_row
            continue
        if middle_row["status"] in OUTER_FORBIDDEN_STATUSES:
            upper, upper_row = middle, middle_row
            continue

        width = upper - lower
        recovery = [
            (
                lower + fraction * width,
                evaluate(
                    lower + fraction * width,
                    lower_row.get("endpoint_solution"),
                ),
            )
            for fraction in (0.45, 0.55, 0.35, 0.65)
        ]
        successes = [(lower, lower_row)] + [
            item for item in recovery if item[1]["status"] == "success"
        ]
        forbidden = [
            item
            for item in recovery
            if item[1]["status"] in OUTER_FORBIDDEN_STATUSES
        ] + [(upper, upper_row)]
        lower, lower_row = max(successes, key=lambda item: item[0])
        valid_forbidden = [item for item in forbidden if item[0] > lower]
        if not valid_forbidden:
            raise RuntimeError(
                f"{label}: could not recover A bisection: "
                f"{middle_row.get('message', '')}"
            )
        upper, upper_row = min(valid_forbidden, key=lambda item: item[0])

    boundary_coordinate = 0.5 * (lower + upper)
    density, temperature = physical_coordinates(boundary_coordinate)
    residual = float(1.0 - lower_row["A_boundary"])
    boundary_kind = _outer_boundary_kind(upper_row["status"])
    if boundary_kind == "A_equals_one" and residual > float(residual_tolerance):
        raise RuntimeError(
            f"{label}: one-sided 1-A residual did not close: {residual:.3e}"
        )
    return {
        "status": "root",
        "boundary_kind": boundary_kind,
        "outer_forbidden_status": str(upper_row["status"]),
        "solve_kind": label,
        "coordinate": boundary_coordinate,
        "nB_over_n0": float(density),
        "T_MeV": float(temperature),
        "one_minus_A_residual": residual,
        "A_below_boundary": float(lower_row["A_boundary"]),
        "endpoint_solution": lower_row.get("endpoint_solution"),
        "muB_0minus_MeV": _as_finite_float(lower_row.get("muB_0minus_MeV")),
        "evaluation_count": len(cache),
        "bracket_lower_coordinate": float(lower),
        "bracket_upper_coordinate": float(upper),
        "bisection_steps": int(bisection_steps),
        "probe_status_counts": {
            status: sum(row["status"] == status for row in cache.values())
            for status in sorted({row["status"] for row in cache.values()})
        },
    }


def _solve_endpoint_probe_task(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one coarse, retry, or refinement endpoint probe."""
    edge_kind = str(task["edge_kind"])
    coordinate = float(task["coordinate"])
    density = float(task["nB_over_n0"])
    temperature = float(task["T_MeV"])
    if edge_kind.startswith("gap_"):
        try:
            gap, muB_0minus, details = enthalpy_gap_at_absolute_density(
                nB_over_n0=density,
                temperature_MeV=temperature,
                n0=float(task["n0"]),
                B14=float(task["B14"]),
                nm_type=str(task["nm_type"]),
                muB_prefer=float(task.get("muB_hint", 1100.0)),
            )
            row = {
                "status": "success",
                "gap_MeV": float(gap),
                "muB_0minus_MeV": float(muB_0minus),
                **details,
            }
        except Exception as exc:
            row = {
                "status": "numerical_failure",
                "gap_MeV": np.nan,
                "muB_0minus_MeV": np.nan,
                "message": str(exc),
            }
    else:
        row = evaluate_A_state(
            nB_over_n0=density,
            temperature_MeV=temperature,
            n0=float(task["n0"]),
            B14=float(task["B14"]),
            xi=float(task["xi"]),
            ms=float(task["ms"]),
            nm_type=str(task["nm_type"]),
            upB=int(task["upB"]),
            interface_fraction_mode=str(task["interface_fraction_mode"]),
            velocity_closure=str(task["velocity_closure"]),
            T_0plus_MeV=task.get("T_0plus_MeV"),
            endpoint_guess=_finite_pair(task.get("endpoint_hint")),
            muB_prefer=float(task.get("muB_hint", 1100.0)),
        )
    row.update(
        {
            "coordinate": coordinate,
            "nB_over_n0": density,
            "T_MeV": temperature,
        }
    )
    return {
        "edge_kind": edge_kind,
        "phase": str(task["phase"]),
        "record": row,
    }


def _endpoint_task(
    *,
    edge_kind: str,
    coordinate: float,
    density_min_over_n0: float,
    temperature_min_MeV: float,
    common: dict[str, Any],
    phase: str,
    endpoint_hint=None,
    muB_hint: float = 1100.0,
) -> dict[str, Any]:
    if edge_kind.endswith("bottom_endpoint"):
        density = float(coordinate)
        temperature = float(temperature_min_MeV)
    else:
        density = float(density_min_over_n0)
        temperature = float(coordinate)
    return {
        **common,
        "edge_kind": str(edge_kind),
        "coordinate": float(coordinate),
        "nB_over_n0": density,
        "T_MeV": temperature,
        "phase": str(phase),
        "endpoint_hint": _finite_pair(endpoint_hint),
        "muB_hint": float(muB_hint),
    }


def _run_endpoint_probe_tasks(
    tasks: list[dict[str, Any]],
    *,
    executor,
    progress,
) -> list[dict[str, Any]]:
    futures = {
        executor.submit(_solve_endpoint_probe_task, task): task
        for task in tasks
    }
    completed = []
    for future in as_completed(futures):
        result = future.result()
        completed.append(result)
        progress.update(1)
        progress.set_postfix_str(
            f"{result['edge_kind']}, {result['record']['coordinate']:.4g}"
        )
    return completed


def _nearest_successful_endpoint_row(
    rows: list[dict[str, Any]],
    coordinate: float,
) -> dict[str, Any] | None:
    successful = [row for row in rows if row.get("status") == "success"]
    return min(
        successful,
        key=lambda row: abs(float(row["coordinate"]) - float(coordinate)),
        default=None,
    )


def _missing_A_recovery_coordinates(
    rows: list[dict[str, Any]],
) -> list[float]:
    """Return interior probes where a narrow admissible A interval may hide."""
    ordered = sorted(rows, key=lambda row: float(row["coordinate"]))
    if any(row.get("status") == "success" for row in ordered):
        return []
    coordinates: list[float] = []
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        if lower.get("status") in OUTER_FORBIDDEN_STATUSES:
            continue
        if upper.get("status") not in OUTER_FORBIDDEN_STATUSES:
            continue
        lower_coordinate = float(lower["coordinate"])
        width = float(upper["coordinate"]) - lower_coordinate
        if width <= 0.0:
            continue
        coordinates.extend(
            lower_coordinate + fraction * width
            for fraction in (0.25, 0.5, 0.75)
        )
    return coordinates


def _recover_missing_A_endpoint_successes(
    *,
    rows_by_edge: dict[str, list[dict[str, Any]]],
    base_tasks_by_edge: dict[str, dict[str, Any]],
    executor,
    progress_factory,
    max_rounds: int = 4,
) -> None:
    """Resolve thin allowed intervals skipped by uniform endpoint probes."""
    for round_index in range(max(0, int(max_rounds))):
        recovery_tasks = []
        for edge_kind in ("A_bottom_endpoint", "A_left_endpoint"):
            rows = rows_by_edge[edge_kind]
            coordinates = _missing_A_recovery_coordinates(rows)
            if not coordinates:
                continue
            base_task = base_tasks_by_edge[edge_kind]
            existing = {float(row["coordinate"]) for row in rows}
            for coordinate in coordinates:
                if float(coordinate) in existing:
                    continue
                task = dict(base_task)
                task.update(
                    {
                        "coordinate": float(coordinate),
                        "phase": f"admissible_recovery_{round_index + 1}",
                    }
                )
                if edge_kind.endswith("bottom_endpoint"):
                    task["nB_over_n0"] = float(coordinate)
                else:
                    task["T_MeV"] = float(coordinate)
                nearest = min(
                    rows,
                    key=lambda row: abs(
                        float(row["coordinate"]) - float(coordinate)
                    ),
                )
                task["muB_hint"] = _as_finite_float(
                    nearest.get("muB_0minus_MeV"),
                    1100.0,
                )
                task["endpoint_hint"] = _finite_pair(
                    nearest.get("endpoint_solution")
                )
                recovery_tasks.append(task)
        if not recovery_tasks:
            return
        with _make_progress(
            progress_factory,
            total=len(recovery_tasks),
            desc="Boundary admissible-interval recovery",
            unit="probe",
            dynamic_ncols=True,
        ) as progress:
            recovered = _run_endpoint_probe_tasks(
                recovery_tasks,
                executor=executor,
                progress=progress,
            )
        for completed in recovered:
            rows_by_edge[completed["edge_kind"]].append(completed["record"])
        if all(
            any(row.get("status") == "success" for row in rows_by_edge[edge_kind])
            or not _missing_A_recovery_coordinates(rows_by_edge[edge_kind])
            for edge_kind in ("A_bottom_endpoint", "A_left_endpoint")
        ):
            return


def _gap_endpoint_state(
    edge_kind: str,
    rows: list[dict[str, Any]],
    *,
    bisection_steps: int,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["coordinate"]))
    brackets = []
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        if lower.get("status") != "success" or upper.get("status") != "success":
            continue
        if float(lower["gap_MeV"]) * float(upper["gap_MeV"]) <= 0.0:
            brackets.append((lower, upper))
    if len(brackets) != 1:
        counts: dict[str, int] = {}
        for row in ordered:
            status = str(row.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        raise RuntimeError(
            f"{edge_kind}: expected one Delta h sign change, found "
            f"{len(brackets)}; {counts}"
        )
    lower, upper = brackets[0]
    return {
        "kind": "gap",
        "edge_kind": edge_kind,
        "lower": lower,
        "upper": upper,
        "best": min((lower, upper), key=lambda row: abs(float(row["gap_MeV"]))),
        "remaining": int(bisection_steps),
        "evaluation_count": len(rows),
    }


def _A_endpoint_state_or_record(
    edge_kind: str,
    rows: list[dict[str, Any]],
    *,
    bisection_steps: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    ordered = sorted(rows, key=lambda row: float(row["coordinate"]))
    transitions = [
        (lower, upper)
        for lower, upper in zip(ordered[:-1], ordered[1:])
        if lower.get("status") == "success"
        and upper.get("status") in OUTER_FORBIDDEN_STATUSES
    ]
    success_indices = [
        index for index, row in enumerate(ordered)
        if row.get("status") == "success"
    ]
    first_success = success_indices[0] if success_indices else None
    clean_missing = bool(
        first_success is not None
        and all(
            row.get("status") in {"gap_forbidden", "numerical_failure"}
            for row in ordered[:first_success]
        )
        and all(row.get("status") == "success" for row in ordered[first_success:])
    )
    if not transitions and clean_missing:
        outer = ordered[-1]
        return None, {
            "status": "clipped",
            "boundary_kind": "physical_window_outer",
            "solve_kind": edge_kind,
            "coordinate": float(outer["coordinate"]),
            "nB_over_n0": float(outer["nB_over_n0"]),
            "T_MeV": float(outer["T_MeV"]),
            "one_minus_A_residual": float(1.0 - outer["A_boundary"]),
            "A_below_boundary": float(outer["A_boundary"]),
            "endpoint_solution": outer.get("endpoint_solution"),
            "muB_0minus_MeV": _as_finite_float(outer.get("muB_0minus_MeV")),
            "evaluation_count": len(rows),
        }
    if len(transitions) != 1:
        counts: dict[str, int] = {}
        for row in ordered:
            status = str(row.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        raise RuntimeError(
            f"{edge_kind}: expected one success-to-outer-domain transition, found "
            f"{len(transitions)}; {counts}"
        )
    lower, upper = transitions[0]
    return {
        "kind": "A",
        "edge_kind": edge_kind,
        "lower": lower,
        "upper": upper,
        "remaining": int(bisection_steps),
        "evaluation_count": len(rows),
    }, None


def _recover_endpoint_refinement(
    state: dict[str, Any],
    base_task: dict[str, Any],
    *,
    executor,
) -> None:
    lower = state["lower"]
    upper = state["upper"]
    width = float(upper["coordinate"] - lower["coordinate"])
    tasks = []
    for fraction in (0.35, 0.45, 0.55, 0.65):
        coordinate = float(lower["coordinate"] + fraction * width)
        task = dict(base_task)
        task.update(
            {
                "coordinate": coordinate,
                "nB_over_n0": (
                    coordinate
                    if state["edge_kind"].endswith("bottom_endpoint")
                    else float(base_task["nB_over_n0"])
                ),
                "T_MeV": (
                    float(base_task["T_MeV"])
                    if state["edge_kind"].endswith("bottom_endpoint")
                    else coordinate
                ),
                "phase": "recovery",
            }
        )
        tasks.append(task)
    with _SilentProgress() as progress:
        recovered = _run_endpoint_probe_tasks(
            tasks,
            executor=executor,
            progress=progress,
        )
    rows = [lower] + [item["record"] for item in recovered] + [upper]
    rows.sort(key=lambda row: float(row["coordinate"]))
    if state["kind"] == "gap":
        brackets = [
            (left, right)
            for left, right in zip(rows[:-1], rows[1:])
            if left.get("status") == "success"
            and right.get("status") == "success"
            and float(left["gap_MeV"]) * float(right["gap_MeV"]) <= 0.0
        ]
        if not brackets:
            raise RuntimeError(f"{state['edge_kind']}: gap refinement recovery failed")
        state["lower"], state["upper"] = min(
            brackets,
            key=lambda pair: float(pair[1]["coordinate"] - pair[0]["coordinate"]),
        )
        successful = [row for row in rows if row.get("status") == "success"]
        state["best"] = min(
            [state["best"], *successful],
            key=lambda row: abs(float(row["gap_MeV"])),
        )
    else:
        successes = [row for row in rows if row.get("status") == "success"]
        forbidden = [
            row
            for row in rows
            if row.get("status") in OUTER_FORBIDDEN_STATUSES
        ]
        new_lower = max(successes, key=lambda row: float(row["coordinate"]))
        valid_upper = [
            row for row in forbidden
            if float(row["coordinate"]) > float(new_lower["coordinate"])
        ]
        if not valid_upper:
            raise RuntimeError(f"{state['edge_kind']}: A refinement recovery failed")
        state["lower"] = new_lower
        state["upper"] = min(valid_upper, key=lambda row: float(row["coordinate"]))
    state["evaluation_count"] += len(tasks)


def solve_endpoint_boundaries_parallel(
    *,
    density_probes: np.ndarray,
    temperature_probes: np.ndarray,
    temperature_min_MeV: float,
    density_min_over_n0: float,
    n0: float,
    B14: float,
    xi: float,
    ms: float,
    nm_type: str,
    upB: int,
    interface_fraction_mode: str,
    velocity_closure: str,
    T_0plus_MeV: float | None,
    bisection_steps: int,
    gap_bisection_steps: int | None = None,
    progress_factory,
    executor,
) -> dict[str, dict[str, Any]]:
    """Solve four endpoint edges with coarse probes distributed per worker.

    The Delta h edges need far more bisection depth than the A edges. The gap
    closes to 1e-4 MeV, which from an 0.9 n0 coarse bracket at about 70 MeV per
    n0 takes 19 halvings, while the A edges close at 1 - A <= 5e-3 and are
    already inside that at the coarse probe. gap_bisection_steps lets the caller
    buy the gap depth without spending the same number of much more expensive
    velocity-bound evaluations on every A edge; it defaults to bisection_steps.
    """
    common = {
        "n0": float(n0),
        "B14": float(B14),
        "xi": float(xi),
        "ms": float(ms),
        "nm_type": str(nm_type),
        "upB": int(upB),
        "interface_fraction_mode": str(interface_fraction_mode),
        "velocity_closure": str(velocity_closure),
        "T_0plus_MeV": T_0plus_MeV,
    }
    edge_coordinates = {
        "gap_bottom_endpoint": np.asarray(density_probes, dtype=float),
        "gap_left_endpoint": np.asarray(temperature_probes, dtype=float),
        "A_bottom_endpoint": np.asarray(density_probes, dtype=float),
        "A_left_endpoint": np.asarray(temperature_probes, dtype=float),
    }
    coarse_tasks = [
        _endpoint_task(
            edge_kind=edge_kind,
            coordinate=float(coordinate),
            density_min_over_n0=density_min_over_n0,
            temperature_min_MeV=temperature_min_MeV,
            common=common,
            phase="coarse",
        )
        for edge_kind, coordinates in edge_coordinates.items()
        for coordinate in coordinates
    ]
    with _make_progress(
        progress_factory,
        total=len(coarse_tasks),
        desc="Boundary endpoint probes",
        unit="probe",
        dynamic_ncols=True,
    ) as progress:
        coarse_results = _run_endpoint_probe_tasks(
            coarse_tasks,
            executor=executor,
            progress=progress,
        )
    rows_by_edge = {edge_kind: [] for edge_kind in edge_coordinates}
    tasks_by_key = {
        (task["edge_kind"], float(task["coordinate"])): task
        for task in coarse_tasks
    }
    for completed in coarse_results:
        rows_by_edge[completed["edge_kind"]].append(completed["record"])

    retry_tasks = []
    for edge_kind, rows in rows_by_edge.items():
        for row in rows:
            if row.get("status") != "numerical_failure":
                continue
            neighbor = _nearest_successful_endpoint_row(rows, row["coordinate"])
            if neighbor is None:
                continue
            retry = dict(tasks_by_key[(edge_kind, float(row["coordinate"]))])
            retry.update(
                {
                    "phase": "retry",
                    "muB_hint": _as_finite_float(neighbor.get("muB_0minus_MeV"), 1100.0),
                    "endpoint_hint": _finite_pair(neighbor.get("endpoint_solution")),
                }
            )
            retry_tasks.append(retry)
    if retry_tasks:
        with _make_progress(
            progress_factory,
            total=len(retry_tasks),
            desc="Boundary endpoint retries",
            unit="probe",
            dynamic_ncols=True,
        ) as progress:
            retry_results = _run_endpoint_probe_tasks(
                retry_tasks,
                executor=executor,
                progress=progress,
            )
        for completed in retry_results:
            rows = rows_by_edge[completed["edge_kind"]]
            coordinate = float(completed["record"]["coordinate"])
            for index, row in enumerate(rows):
                if float(row["coordinate"]) == coordinate:
                    rows[index] = completed["record"]
                    break

    base_tasks_by_edge = {
        edge_kind: next(
            task for task in coarse_tasks if task["edge_kind"] == edge_kind
        )
        for edge_kind in ("A_bottom_endpoint", "A_left_endpoint")
    }
    _recover_missing_A_endpoint_successes(
        rows_by_edge=rows_by_edge,
        base_tasks_by_edge=base_tasks_by_edge,
        executor=executor,
        progress_factory=progress_factory,
    )

    gap_steps = (
        int(bisection_steps)
        if gap_bisection_steps is None
        else int(gap_bisection_steps)
    )
    states = [
        _gap_endpoint_state(
            "gap_bottom_endpoint",
            rows_by_edge["gap_bottom_endpoint"],
            bisection_steps=gap_steps,
        ),
        _gap_endpoint_state(
            "gap_left_endpoint",
            rows_by_edge["gap_left_endpoint"],
            bisection_steps=gap_steps,
        ),
    ]
    endpoint_records: dict[str, dict[str, Any]] = {}
    for edge_kind in ("A_bottom_endpoint", "A_left_endpoint"):
        state, record = _A_endpoint_state_or_record(
            edge_kind,
            rows_by_edge[edge_kind],
            bisection_steps=bisection_steps,
        )
        if state is not None:
            states.append(state)
        else:
            endpoint_records[edge_kind] = record

    refinement_total = sum(int(state["remaining"]) for state in states)
    with _make_progress(
        progress_factory,
        total=refinement_total,
        desc="Boundary endpoint refinement",
        unit="step",
        dynamic_ncols=True,
    ) as progress:
        while any(int(state["remaining"]) > 0 for state in states):
            active = [state for state in states if int(state["remaining"]) > 0]
            refinement_tasks = []
            state_by_edge = {}
            for state in active:
                lower = state["lower"]
                upper = state["upper"]
                coordinate = 0.5 * (
                    float(lower["coordinate"]) + float(upper["coordinate"])
                )
                task = _endpoint_task(
                    edge_kind=state["edge_kind"],
                    coordinate=coordinate,
                    density_min_over_n0=density_min_over_n0,
                    temperature_min_MeV=temperature_min_MeV,
                    common=common,
                    phase="refinement",
                    endpoint_hint=lower.get("endpoint_solution"),
                    muB_hint=_as_finite_float(lower.get("muB_0minus_MeV"), 1100.0),
                )
                refinement_tasks.append(task)
                state_by_edge[state["edge_kind"]] = state
            with _SilentProgress() as silent:
                completed_round = _run_endpoint_probe_tasks(
                    refinement_tasks,
                    executor=executor,
                    progress=silent,
                )
            task_by_edge = {
                task["edge_kind"]: task for task in refinement_tasks
            }
            for completed in completed_round:
                task = task_by_edge[completed["edge_kind"]]
                state = state_by_edge[completed["edge_kind"]]
                row = completed["record"]
                state["evaluation_count"] += 1
                if state["kind"] == "gap" and row.get("status") == "success":
                    if abs(float(row["gap_MeV"])) < abs(float(state["best"]["gap_MeV"])):
                        state["best"] = row
                    if float(state["lower"]["gap_MeV"]) * float(row["gap_MeV"]) <= 0.0:
                        state["upper"] = row
                    else:
                        state["lower"] = row
                elif state["kind"] == "A" and (
                    row.get("status") == "success"
                    or row.get("status") in OUTER_FORBIDDEN_STATUSES
                ):
                    if row["status"] == "success":
                        state["lower"] = row
                    else:
                        state["upper"] = row
                else:
                    _recover_endpoint_refinement(
                        state,
                        task,
                        executor=executor,
                    )
                state["remaining"] -= 1
                progress.update(1)
                progress.set_postfix_str(state["edge_kind"])

    for state in states:
        edge_kind = state["edge_kind"]
        if state["kind"] == "gap":
            best = state["best"]
            if abs(float(best["gap_MeV"])) > 1.0e-4:
                raise RuntimeError(f"{edge_kind}: Delta h residual did not close")
            endpoint_records[edge_kind] = {
                **best,
                "boundary_kind": "Delta_h_equals_zero",
                "solve_kind": edge_kind,
                "evaluation_count": int(state["evaluation_count"]),
            }
        else:
            lower = state["lower"]
            upper = state["upper"]
            residual = float(1.0 - lower["A_boundary"])
            boundary_kind = _outer_boundary_kind(upper["status"])
            if boundary_kind == "A_equals_one" and residual > 5.0e-3:
                raise RuntimeError(
                    f"{edge_kind}: one-sided 1-A residual did not close: {residual:.3e}"
                )
            coordinate = 0.5 * (
                float(lower["coordinate"]) + float(upper["coordinate"])
            )
            task = _endpoint_task(
                edge_kind=edge_kind,
                coordinate=coordinate,
                density_min_over_n0=density_min_over_n0,
                temperature_min_MeV=temperature_min_MeV,
                common=common,
                phase="finalize",
            )
            endpoint_records[edge_kind] = {
                "status": "root",
                "boundary_kind": boundary_kind,
                "outer_forbidden_status": str(upper["status"]),
                "solve_kind": edge_kind,
                "coordinate": coordinate,
                "nB_over_n0": float(task["nB_over_n0"]),
                "T_MeV": float(task["T_MeV"]),
                "one_minus_A_residual": residual,
                "A_below_boundary": float(lower["A_boundary"]),
                "endpoint_solution": lower.get("endpoint_solution"),
                "muB_0minus_MeV": _as_finite_float(lower.get("muB_0minus_MeV")),
                "evaluation_count": int(state["evaluation_count"]),
            }
    return endpoint_records


def _ray_rho_bounds(
    *,
    transform: RayTransform,
    theta_rad: float,
    density_max_over_n0: float,
    temperature_min_MeV: float,
    temperature_max_MeV: float,
) -> tuple[float, float]:
    cosine = float(np.cos(theta_rad))
    sine = float(np.sin(theta_rad))
    if cosine <= 0.0 or sine <= 0.0:
        raise ValueError("interior ray angle must satisfy 0 < theta < pi/2")
    lower = float(temperature_min_MeV) / (
        transform.temperature_scale_MeV * sine
    )
    density_upper = (
        float(density_max_over_n0) - transform.density_origin_over_n0
    ) / (transform.density_scale_over_n0 * cosine)
    temperature_upper = float(temperature_max_MeV) / (
        transform.temperature_scale_MeV * sine
    )
    upper = min(density_upper, temperature_upper)
    if not lower < upper:
        raise RuntimeError("ray does not intersect the physical window")
    return float(lower), float(upper)


def _solve_boundary_task(task: dict[str, Any]) -> dict[str, Any]:
    mode = str(task["path_mode"])
    if mode == "fixed_temperature":
        fixed_temperature = float(task["fixed_value"])

        def physical_coordinates(coordinate):
            return float(coordinate), fixed_temperature

    elif mode == "fixed_density":
        fixed_density = float(task["fixed_value"])

        def physical_coordinates(coordinate):
            return fixed_density, float(coordinate)

    elif mode == "elliptical_ray":
        transform = RayTransform(**task["transform"])
        theta = float(task["theta_rad"])

        def physical_coordinates(coordinate):
            return tuple(
                float(value)
                for value in transform.to_physical(float(coordinate), theta)
            )

    else:
        raise ValueError(f"unknown boundary path mode {mode!r}")

    kind = str(task["task_kind"])
    if kind.startswith("gap_") or kind == "gap_ray":
        record = solve_gap_on_path(
            coordinates=np.asarray(task["coordinates"], dtype=float),
            physical_coordinates=physical_coordinates,
            n0=float(task["n0"]),
            B14=float(task["B14"]),
            nm_type=str(task["nm_type"]),
            label=str(task["label"]),
            muB_hint=float(task.get("muB_hint", 1100.0)),
            T_0plus_MeV=task.get("T_0plus_MeV"),
            ms=float(task.get("ms", 0.0)),
            upB=int(task.get("upB", 5000)),
        )
    elif kind.startswith("A_") or kind == "A_ray":
        record = solve_A_transition_on_path(
            coordinates=np.asarray(task["coordinates"], dtype=float),
            physical_coordinates=physical_coordinates,
            n0=float(task["n0"]),
            B14=float(task["B14"]),
            xi=float(task["xi"]),
            ms=float(task["ms"]),
            nm_type=str(task["nm_type"]),
            upB=int(task["upB"]),
            interface_fraction_mode=str(task["interface_fraction_mode"]),
            velocity_closure=str(task.get("velocity_closure", "closed_form")),
            T_0plus_MeV=task.get("T_0plus_MeV"),
            label=str(task["label"]),
            endpoint_hint=_finite_pair(task.get("endpoint_hint")),
            muB_hint=float(task.get("muB_hint", 1100.0)),
            bisection_steps=int(task["bisection_steps"]),
            allow_missing=bool(task.get("allow_missing", False)),
        )
    else:
        raise ValueError(f"unknown boundary task kind {kind!r}")
    return {
        "task_kind": kind,
        "index": task.get("index"),
        "record": record,
    }


def _interpolate_scalar_hint(
    coordinate: float,
    lower_coordinate: float,
    upper_coordinate: float,
    lower_value: Any,
    upper_value: Any,
    default: float,
) -> float:
    lower = _as_finite_float(lower_value, default)
    upper = _as_finite_float(upper_value, lower)
    if not upper_coordinate > lower_coordinate:
        return lower
    fraction = np.clip(
        (float(coordinate) - float(lower_coordinate))
        / (float(upper_coordinate) - float(lower_coordinate)),
        0.0,
        1.0,
    )
    return float(lower + fraction * (upper - lower))


def _interpolate_endpoint_hint(
    coordinate: float,
    lower_coordinate: float,
    upper_coordinate: float,
    lower_endpoint: Any,
    upper_endpoint: Any,
) -> tuple[float, float] | None:
    lower = _finite_pair(lower_endpoint)
    upper = _finite_pair(upper_endpoint)
    if lower is None:
        return upper
    if upper is None or not upper_coordinate > lower_coordinate:
        return lower
    fraction = np.clip(
        (float(coordinate) - float(lower_coordinate))
        / (float(upper_coordinate) - float(lower_coordinate)),
        0.0,
        1.0,
    )
    return tuple(
        float(left + fraction * (right - left))
        for left, right in zip(lower, upper)
    )


def solve_boundary_curves(
    *,
    transform: RayTransform,
    theta_count: int,
    density_min_over_n0: float,
    density_max_over_n0: float,
    temperature_min_MeV: float,
    temperature_max_MeV: float,
    n0: float,
    B14: float,
    xi: float,
    ms: float,
    nm_type: str,
    upB: int,
    interface_fraction_mode: str,
    velocity_closure: str = "closed_form",
    T_0plus_MeV: float | None = None,
    endpoint_probe_count: int,
    ray_probe_count: int,
    bisection_steps: int,
    gap_bisection_steps: int | None = None,
    angular_buffer_fraction: float,
    progress_factory=None,
    executor=None,
) -> dict[str, Any]:
    density_probes = np.linspace(
        density_min_over_n0,
        density_max_over_n0,
        int(endpoint_probe_count),
    )
    temperature_probes = np.linspace(
        temperature_min_MeV,
        temperature_max_MeV,
        int(endpoint_probe_count),
    )
    endpoint_common = {
        "n0": float(n0),
        "B14": float(B14),
        "nm_type": str(nm_type),
    }
    A_common = {
        **endpoint_common,
        "xi": float(xi),
        "ms": float(ms),
        "upB": int(upB),
        "interface_fraction_mode": str(interface_fraction_mode),
        "velocity_closure": str(velocity_closure),
        "T_0plus_MeV": T_0plus_MeV,
        "bisection_steps": int(bisection_steps),
        "allow_missing": True,
    }
    endpoint_tasks = [
        {
            **endpoint_common,
            "task_kind": "gap_bottom_endpoint",
            "label": "gap_fixed_T_minimum",
            "path_mode": "fixed_temperature",
            "fixed_value": float(temperature_min_MeV),
            "coordinates": density_probes,
        },
        {
            **endpoint_common,
            "task_kind": "gap_left_endpoint",
            "label": "gap_fixed_density_minimum",
            "path_mode": "fixed_density",
            "fixed_value": float(density_min_over_n0),
            "coordinates": temperature_probes,
        },
        {
            **A_common,
            "task_kind": "A_bottom_endpoint",
            "label": "A_fixed_T_minimum",
            "path_mode": "fixed_temperature",
            "fixed_value": float(temperature_min_MeV),
            "coordinates": density_probes,
            "endpoint_hint": None,
        },
        {
            **A_common,
            "task_kind": "A_left_endpoint",
            "label": "A_fixed_density_minimum",
            "path_mode": "fixed_density",
            "fixed_value": float(density_min_over_n0),
            "coordinates": temperature_probes,
            "endpoint_hint": None,
        },
    ]
    if executor is not None:
        endpoint_results = solve_endpoint_boundaries_parallel(
            density_probes=density_probes,
            temperature_probes=temperature_probes,
            temperature_min_MeV=temperature_min_MeV,
            density_min_over_n0=density_min_over_n0,
            n0=n0,
            B14=B14,
            xi=xi,
            ms=ms,
            nm_type=nm_type,
            upB=upB,
            interface_fraction_mode=interface_fraction_mode,
            velocity_closure=velocity_closure,
            T_0plus_MeV=T_0plus_MeV,
            bisection_steps=bisection_steps,
            gap_bisection_steps=gap_bisection_steps,
            progress_factory=progress_factory,
            executor=executor,
        )
    else:
        endpoint_results: dict[str, dict[str, Any]] = {}
        with _make_progress(
            progress_factory,
            total=len(endpoint_tasks),
            desc="Boundary endpoints",
            unit="edge",
            dynamic_ncols=True,
        ) as progress:
            for completed in map(_solve_boundary_task, endpoint_tasks):
                kind = str(completed["task_kind"])
                endpoint_results[kind] = completed["record"]
                progress.update(1)
                progress.set_postfix_str(kind)
    gap_bottom = endpoint_results["gap_bottom_endpoint"]
    gap_left = endpoint_results["gap_left_endpoint"]
    A_bottom = endpoint_results["A_bottom_endpoint"]
    A_left = endpoint_results["A_left_endpoint"]
    for record in (gap_bottom, gap_left, A_bottom, A_left):
        rho, theta = transform.to_hidden(
            record["nB_over_n0"],
            record["T_MeV"],
        )
        record["rho"] = float(rho)
        record["theta_rad"] = float(theta)
        record["theta_deg"] = float(np.degrees(theta))

    theta_lower = float(A_bottom["theta_rad"])
    theta_upper = 0.5 * np.pi
    theta_lower += float(angular_buffer_fraction) * (
        theta_upper - theta_lower
    )
    theta_axis = chebyshev_closed_axis(theta_lower, theta_upper, theta_count)
    gap_bottom_theta = float(gap_bottom["theta_rad"])
    order = radial_solve_order(
        np.degrees(theta_axis),
        seed_temperature_MeV=45.0,
    )
    directions = {index: direction for index, direction in order}
    gap_rows: dict[int, dict[str, Any]] = {}
    A_rows: dict[int, dict[str, Any]] = {}
    transform_args = {
        "density_origin_over_n0": transform.density_origin_over_n0,
        "density_scale_over_n0": transform.density_scale_over_n0,
        "temperature_scale_MeV": transform.temperature_scale_MeV,
    }
    ray_tasks: list[dict[str, Any]] = []
    special_count = 0
    for index in range(theta_axis.size):
        theta = float(theta_axis[index])
        rho_min, rho_max = _ray_rho_bounds(
            transform=transform,
            theta_rad=theta,
            density_max_over_n0=density_max_over_n0,
            temperature_min_MeV=temperature_min_MeV,
            temperature_max_MeV=temperature_max_MeV,
        )
        probes = np.linspace(rho_min, rho_max, int(ray_probe_count))
        if theta < gap_bottom_theta:
            gap_rows[index] = {
                "status": "gap_below_window",
                "boundary_kind": "physical_T_minimum",
                "nB_over_n0": float(
                    transform.to_physical(rho_min, theta)[0]
                ),
                "T_MeV": float(temperature_min_MeV),
                "rho": rho_min,
                "theta_rad": theta,
                "theta_deg": float(np.degrees(theta)),
                "gap_MeV": np.nan,
            }
            special_count += 1
        elif np.isclose(theta, 0.5 * np.pi):
            gap_rows[index] = dict(gap_left)
            special_count += 1
        else:
            ray_tasks.append(
                {
                    **endpoint_common,
                    "task_kind": "gap_ray",
                    "index": int(index),
                    "label": "gap_elliptical_ray",
                    "path_mode": "elliptical_ray",
                    "transform": transform_args,
                    "theta_rad": theta,
                    "coordinates": probes,
                    "muB_hint": _interpolate_scalar_hint(
                        theta,
                        float(gap_bottom["theta_rad"]),
                        float(gap_left["theta_rad"]),
                        gap_bottom.get("muB_0minus_MeV"),
                        gap_left.get("muB_0minus_MeV"),
                        1100.0,
                    ),
                }
            )
        if np.isclose(theta, 0.5 * np.pi):
            A_rows[index] = dict(A_left)
            special_count += 1
        else:
            ray_tasks.append(
                {
                    **A_common,
                    "task_kind": "A_ray",
                    "index": int(index),
                    "label": "A_elliptical_ray",
                    "path_mode": "elliptical_ray",
                    "transform": transform_args,
                    "theta_rad": theta,
                    "coordinates": probes,
                    "endpoint_hint": _interpolate_endpoint_hint(
                        theta,
                        float(A_bottom["theta_rad"]),
                        float(A_left["theta_rad"]),
                        A_bottom.get("endpoint_solution"),
                        A_left.get("endpoint_solution"),
                    ),
                    "muB_hint": _interpolate_scalar_hint(
                        theta,
                        float(A_bottom["theta_rad"]),
                        float(A_left["theta_rad"]),
                        A_bottom.get("muB_0minus_MeV"),
                        A_left.get("muB_0minus_MeV"),
                        1100.0,
                    ),
                }
            )
    with _make_progress(
        progress_factory,
        total=2 * theta_axis.size,
        desc="Boundary curves",
        unit="ray",
        dynamic_ncols=True,
    ) as progress:
        progress.update(special_count)
        if executor is None:
            completed_rays = map(_solve_boundary_task, ray_tasks)
        else:
            ray_futures = {
                executor.submit(_solve_boundary_task, task): task
                for task in ray_tasks
            }
            completed_rays = (
                future.result() for future in as_completed(ray_futures)
            )
        for completed in completed_rays:
            index = int(completed["index"])
            theta = float(theta_axis[index])
            record = completed["record"]
            record["rho"] = float(record["coordinate"])
            record["theta_rad"] = theta
            record["theta_deg"] = float(np.degrees(theta))
            if completed["task_kind"] == "gap_ray":
                gap_rows[index] = record
            else:
                A_rows[index] = record
            progress.update(1)
            progress.set_postfix_str(
                f"{completed['task_kind']}, theta={np.degrees(theta):.3g} deg"
            )

    ordered = []
    for index in range(theta_axis.size):
        inner_record = gap_rows[index]
        outer_record = A_rows[index]
        if not float(outer_record["rho"]) > float(inner_record["rho"]):
            raise RuntimeError("safe radial interval collapsed")
        ordered.append(
            {
                "theta_rad": float(theta_axis[index]),
                "direction": directions[index],
                "gap_record": inner_record,
                "A_record": outer_record,
            }
        )
    gap_mesh_records = [row["gap_record"] for row in ordered]
    A_mesh_records = [row["A_record"] for row in ordered]
    gap_curve_records = [gap_bottom] + [
        row
        for row in gap_mesh_records
        if row["boundary_kind"] == "Delta_h_equals_zero"
        and row["theta_rad"] > gap_bottom["theta_rad"]
    ]
    A_curve_records = [
        row
        for row in [A_bottom] + A_mesh_records
        if row["boundary_kind"] in {"A_equals_one", "T_0plus_reachability_edge"}
    ]
    return {
        "theta_axis_rad": theta_axis,
        "rho_inner": np.asarray(
            [row["rho"] for row in gap_mesh_records],
            dtype=float,
        ),
        "rho_outer": np.asarray(
            [row["rho"] for row in A_mesh_records],
            dtype=float,
        ),
        "gap_records": gap_mesh_records,
        "A_records": A_mesh_records,
        "gap_curve_records": gap_curve_records,
        "A_curve_records": A_curve_records,
        "endpoint_records": {
            "gap_bottom": gap_bottom,
            "gap_left": gap_left,
            "A_bottom": A_bottom,
            "A_left": A_left,
        },
    }


def _point_failure(message: str, task: dict[str, Any]) -> dict[str, Any]:
    return {
        **task,
        "success": False,
        "message": str(message),
        "u_0minus": np.nan,
        "u_0minus_m_s": np.nan,
        "jB": np.nan,
        "lambda_n": np.nan,
        "A_boundary": np.nan,
        "a_0plus": np.nan,
        "a_0plus_LTE": np.nan,
        "velocity_closure": str(task.get("velocity_closure", "closed_form")),
        "I2_numerical": np.nan,
        "layer_trajectory_points": 0,
        "layer_trajectory_saturated": False,
        "endpoint_solution": None,
        "entropy_jump_per_baryon": np.nan,
    }


def entropy_jump_per_baryon(row: dict[str, Any]) -> float:
    values = np.asarray(
        [
            row.get("T_MeV", np.nan),
            row.get("T_inf_MeV", np.nan),
            row.get("h_over_nB_0minus_MeV", np.nan),
            row.get("muB_0minus_MeV", np.nan),
            row.get("h_over_nB_inf_MeV", np.nan),
            row.get("muB_inf_MeV", np.nan),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or values[0] <= 0.0 or values[1] <= 0.0:
        return np.nan
    T_0minus, T_inf, h_0minus, muB_0minus, h_inf, muB_inf = values
    return float((h_inf - muB_inf) / T_inf - (h_0minus - muB_0minus) / T_0minus)


def calculate_absolute_density_point(task: dict[str, Any]) -> dict[str, Any]:
    row = dict(task)
    try:
        n0 = float(task["n0_MeV3"])
        nB_0minus = float(task["nB_over_n0"]) * n0
        muB_0minus = muB_from_nB(
            nB_0minus,
            float(task["T_MeV"]),
            str(task["nm_type"]),
            prefer=float(task.get("muB_guess_MeV", 1100.0)),
        )
        result = phase_velocity.analytic_velocity_bound(
            muB_0minus,
            float(task["T_MeV"]),
            float(task["B14_MeV"]),
            xi=float(task["xi"]),
            ms=float(task["ms_MeV"]),
            param=para.paraQMCRMF3,
            NM_type=str(task["nm_type"]),
            upB=int(task["upB"]),
            initial_guess=_finite_pair(task.get("endpoint_guess")),
            T_0plus=float(task.get("T_0plus_MeV", BOUNDARY_T_0PLUS_MEV)),
        )
        row.update(
            {
                "success": True,
                "message": str(result.get("message", "")),
                "nB_0minus_MeV3": nB_0minus,
                "muB_0minus_MeV": float(muB_0minus),
                "u_0minus": float(result["u_0minus_max"]),
                "u_0minus_m_s": float(result["u_0minus_max"] * const.m_per_sec),
                "u_0minus_squared": float(result["u_0minus_squared"]),
                "u_0minus_formula_squared": float(result["u_0minus_formula_squared"]),
                "analytic_velocity_residual": float(
                    result["analytic_velocity_residual"]
                ),
                "jB": float(result["jB"]),
                "muB_inf_MeV": float(result["muB_inf"]),
                "T_inf_MeV": float(result["T_inf"]),
                "h_over_nB_0minus_MeV": _as_finite_float(
                    result.get("h_over_nB_0minus")
                ),
                "h_over_nB_inf_MeV": _as_finite_float(
                    result.get("h_over_nB_inf")
                ),
                "lambda_n": float(result["lambda_n"]),
                "A_boundary": float(result["A_boundary"]),
                "a_0plus": float(result["a_0plus"]),
                "a_0plus_LTE": float(result["a_0plus_LTE"]),
                "velocity_closure": str(
                    result.get(
                        "velocity_closure",
                        task.get("velocity_closure", "closed_form"),
                    )
                ),
                "I2_numerical": _as_finite_float(result.get("I2_numerical")),
                "layer_trajectory_points": int(
                    result.get("layer_trajectory_points", 0)
                ),
                "layer_trajectory_saturated": bool(
                    result.get("layer_trajectory_saturated", False)
                ),
                "endpoint_scaled_residual": _as_finite_float(
                    result.get("endpoint_scaled_residual")
                ),
                "energy_flux_residual": _as_finite_float(
                    result.get("energy_flux_residual")
                ),
                "momentum_flux_residual": _as_finite_float(
                    result.get("momentum_flux_residual")
                ),
                "endpoint_solution": (
                    float(result["muB_inf"]),
                    float(result["T_inf"]),
                ),
                "composition_definition": str(
                    result.get(
                        "composition_definition",
                        COMPOSITION_DEFINITION,
                    )
                ),
                "density_ratio_definition": str(
                    result.get(
                        "density_ratio_definition",
                        DENSITY_RATIO_DEFINITION,
                    )
                ),
                "analytic_formula_variant": str(
                    result.get(
                        "analytic_formula_variant",
                        ANALYTIC_FORMULA_VARIANT,
                    )
                ),
            }
        )
        row["entropy_jump_per_baryon"] = entropy_jump_per_baryon(row)
        return row
    except Exception as exc:
        return _point_failure(str(exc), row)


def calculate_ray_row(task: dict[str, Any]) -> list[dict[str, Any]]:
    temperatures = np.asarray(task["temperature_values_MeV"], dtype=float)
    densities = np.asarray(task["density_values_over_n0"], dtype=float)
    rho_values = np.asarray(task["rho_values"], dtype=float)
    eta_values = np.asarray(task["radial_fraction_values"], dtype=float)
    order = radial_solve_order(
        temperatures,
        float(task["seed_temperature_MeV"]),
    )
    solved: dict[int, dict[str, Any]] = {}
    for col_index, direction in order:
        if direction == "up":
            neighbor = solved.get(col_index - 1)
        elif direction == "down":
            neighbor = solved.get(col_index + 1)
        else:
            neighbor = None
        endpoint = (
            _finite_pair(neighbor.get("endpoint_solution"))
            if neighbor and neighbor.get("success")
            else _finite_pair(task.get("default_endpoint_guess"))
        )
        muB_guess = (
            _as_finite_float(neighbor.get("muB_0minus_MeV"), 1100.0)
            if neighbor and neighbor.get("success")
            else 1100.0
        )
        source = (
            f"same_ray_{direction}_neighbor"
            if neighbor and neighbor.get("success")
            else "default"
        )
        point_task = {
            "row_index": int(task["row_index"]),
            "col_index": int(col_index),
            "theta_rad": float(task["theta_rad"]),
            "theta_deg": float(np.degrees(task["theta_rad"])),
            "rho": float(rho_values[col_index]),
            "radial_fraction": float(eta_values[col_index]),
            "nB_over_n0": float(densities[col_index]),
            "T_MeV": float(temperatures[col_index]),
            "n0_MeV3": float(task["n0_MeV3"]),
            "B14_MeV": float(task["B14_MeV"]),
            "xi": float(task["xi"]),
            "ms_MeV": float(task["ms_MeV"]),
            "nm_type": str(task["nm_type"]),
            "upB": int(task["upB"]),
            "interface_fraction_mode": str(
                task["interface_fraction_mode"]
            ),
            "velocity_closure": str(
                task.get("velocity_closure", "closed_form")
            ),
            "endpoint_guess": endpoint,
            "endpoint_guess_source": source,
            "continuation_direction": direction,
            "muB_guess_MeV": muB_guess,
            "retry_attempt": 0,
            "retry_of": None,
        }
        solved[col_index] = calculate_absolute_density_point(point_task)
    return [solved[index] for index in range(temperatures.size)]


def build_payload(
    *,
    args: argparse.Namespace | Any,
    transform: RayTransform,
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    boundary_payload: dict[str, Any],
    calibration: dict[str, float],
) -> dict[str, Any]:
    velocity_closure = str(getattr(args, "velocity_closure", "closed_form"))
    transport_integral_definition = (
        "numerical_I2_exact_rate_full_diffusion"
        if velocity_closure == "numerical_I2"
        else "closed_form_piecewise_constant"
    )
    payload = {
        "run_tag": str(args.run_tag),
        "primary_coordinate_system": PRIMARY_COORDINATE_SYSTEM,
        "mesh_parameterization": MESH_PARAMETERIZATION,
        "cartesian_coordinates_are_primary": True,
        "density_coordinate": "nB_at_0minus_over_n0",
        "temperature_coordinate": "T_0minus_MeV",
        "metastability_used": False,
        "analytic_closure": ANALYTIC_CLOSURE_LABEL,
        "interface_fraction_mode": str(
            getattr(args, "interface_fraction_mode", "extreme_endothermic")
        ),
        "velocity_closure": velocity_closure,
        "transport_integral_definition": transport_integral_definition,
        "analytic_formula_variant": ANALYTIC_FORMULA_VARIANT,
        "composition_definition": COMPOSITION_DEFINITION,
        "density_ratio_definition": DENSITY_RATIO_DEFINITION,
        "gap_definition": GAP_DEFINITION,
        "safe_region_definition": "Delta_h_le_0_and_A_boundary_lt_1",
        "entropy_jump_definition": ENTROPY_JUMP_DEFINITION,
        "entropy_affects_mesh": False,
        "transform": transform.metadata(),
        "status_codes": {
            "finite": STATUS_FINITE,
            "numerical_failure": STATUS_NUMERICAL_FAILURE,
            "unscanned": STATUS_UNSCANNED,
        },
        **{name: np.asarray(value) for name, value in arrays.items()},
        "records": list(records),
        "gap_boundary_records": list(
            boundary_payload.get("gap_curve_records", [])
        ),
        "A_boundary_records": list(
            boundary_payload.get("A_curve_records", [])
        ),
        "gap_mesh_boundary_records": list(
            boundary_payload.get("gap_records", [])
        ),
        "A_mesh_boundary_records": list(
            boundary_payload.get("A_records", [])
        ),
        "endpoint_records": dict(
            boundary_payload.get("endpoint_records", {})
        ),
        "calibration": dict(calibration),
        "arguments": vars(args).copy(),
    }
    gap_curve = payload["gap_boundary_records"]
    A_curve = payload["A_boundary_records"]
    payload["gap_boundary_nB_over_n0"] = np.asarray(
        [row["nB_over_n0"] for row in gap_curve],
        dtype=float,
    )
    payload["gap_boundary_temperature_MeV"] = np.asarray(
        [row["T_MeV"] for row in gap_curve],
        dtype=float,
    )
    payload["A_boundary_nB_over_n0"] = np.asarray(
        [row["nB_over_n0"] for row in A_curve],
        dtype=float,
    )
    payload["A_boundary_temperature_MeV"] = np.asarray(
        [row["T_MeV"] for row in A_curve],
        dtype=float,
    )
    return payload


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        suffix=".npy",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, payload, allow_pickle=True)
    temporary.replace(target)
