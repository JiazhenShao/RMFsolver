"""Shared domain and payload helpers for isothermal contour calculations."""

from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from scipy.optimize import brentq, root


SCHEMA_VERSION = 3
RUN_TAG = "isothermal-contour-26-08-20-polar-v1"
SPEED_OF_LIGHT_M_S = 299_792_458.0
DEFAULT_B_ONE_FORTH_MEV = 189.1565957288247
DEFAULT_XI = -0.5
DEFAULT_MS_MEV = 0.0
DEFAULT_NM_TYPE = "PNM"
DEFAULT_UPB = 5000
DEFAULT_DENSITY_MIN_OVER_N0 = 1.0
DEFAULT_DENSITY_MAX_OVER_N0 = 5.5
DEFAULT_TEMPERATURE_MIN_MEV = 1.0e-2
DEFAULT_TEMPERATURE_MAX_MEV = 120.0
DEFAULT_BOUNDARY_THETA_COUNT = 12
DEFAULT_CONTOUR_RAY_COUNT = 30
DEFAULT_BOUNDARY_PROBE_COUNT = 10
DEFAULT_RHO_LOWER = 0.02
DEFAULT_RHO_TOL = 1.0e-4
DEFAULT_RADIAL_FRACTIONS = (
    0.02, 0.06, 0.10, 0.15, 0.20,
    0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.66, 0.72,
    0.78, 0.84, 0.90, 0.95, 0.99,
)
BOUNDARY_KINDS = ("delta_muB_equals_zero", "a_0plus_equals_one")
A_0PLUS_CEILING = 1.0 - 1.0e-9
MUB_COLD_LO_MEV = 950.0
MUB_COLD_HI_MEV = 2200.0
MUB_COLD_SCAN_POINTS = 48
MUB_WARM_SCAN_POINTS = 6
MUB_WARM_MIN_HALF_WIDTH_MEV = 45.0
MUB_WARM_MAX_HALF_WIDTH_MEV = 260.0
MUB_WARM_SLOPE_SAFETY = 0.6
MUB_SEED_HALF_WIDTH_MEV = 200.0


class RayTransform:
    """Dimensionless elliptical-polar coordinates used to build the mesh."""

    def __init__(
        self,
        density_origin_over_n0: float = DEFAULT_DENSITY_MIN_OVER_N0,
        density_scale_over_n0: float = (
            DEFAULT_DENSITY_MAX_OVER_N0 - DEFAULT_DENSITY_MIN_OVER_N0
        ),
        temperature_scale_MeV: float = DEFAULT_TEMPERATURE_MAX_MEV,
        temperature_min_MeV: float = DEFAULT_TEMPERATURE_MIN_MEV,
    ):
        self.density_origin_over_n0 = float(density_origin_over_n0)
        self.density_scale_over_n0 = float(density_scale_over_n0)
        self.temperature_scale_MeV = float(temperature_scale_MeV)
        self.temperature_min_MeV = float(temperature_min_MeV)
        if self.density_scale_over_n0 <= 0.0 or self.temperature_scale_MeV <= 0.0:
            raise ValueError("coordinate scales must be positive")

    def to_hidden(self, nB_0minus_over_n0: Any, T_0minus_MeV: Any):
        density = np.asarray(nB_0minus_over_n0, dtype=float)
        temperature = np.asarray(T_0minus_MeV, dtype=float)
        x_coordinate = (
            density - self.density_origin_over_n0
        ) / self.density_scale_over_n0
        y_coordinate = temperature / self.temperature_scale_MeV
        return np.hypot(x_coordinate, y_coordinate), np.arctan2(
            y_coordinate, x_coordinate
        )

    def to_physical(self, rho: Any, theta_rad: Any):
        radius = np.asarray(rho, dtype=float)
        theta = np.asarray(theta_rad, dtype=float)
        density = (
            self.density_origin_over_n0
            + self.density_scale_over_n0 * radius * np.cos(theta)
        )
        temperature = np.maximum(
            self.temperature_scale_MeV * radius * np.sin(theta),
            self.temperature_min_MeV,
        )
        return density, temperature

    def rho_upper_bound(self, theta_rad: float) -> float:
        cosine = float(np.cos(theta_rad))
        sine = float(np.sin(theta_rad))
        limits: list[float] = []
        if cosine > 1.0e-12:
            limits.append(
                (DEFAULT_DENSITY_MAX_OVER_N0 - self.density_origin_over_n0)
                / (self.density_scale_over_n0 * cosine)
            )
        if sine > 1.0e-12:
            limits.append(
                DEFAULT_TEMPERATURE_MAX_MEV
                / (self.temperature_scale_MeV * sine)
            )
        if not limits:
            raise ValueError("ray does not intersect the physical window")
        return float(min(limits))

    def metadata(self) -> dict[str, float | str]:
        return {
            "density_origin_over_n0": self.density_origin_over_n0,
            "density_scale_over_n0": self.density_scale_over_n0,
            "temperature_scale_MeV": self.temperature_scale_MeV,
            "temperature_min_MeV": self.temperature_min_MeV,
            "rho_definition": (
                "sqrt(((nB_0minus_over_n0-density_origin_over_n0)/"
                "density_scale_over_n0)^2+(T_0minus_MeV/"
                "temperature_scale_MeV)^2)"
            ),
            "theta_definition": (
                "atan2(T_0minus_MeV/temperature_scale_MeV,"
                "(nB_0minus_over_n0-density_origin_over_n0)/"
                "density_scale_over_n0)"
            ),
        }


def default_theta_axis_rad() -> np.ndarray:
    """The same 30 endpoint-inclusive rays as the 26-08-18 workflow."""
    return np.linspace(0.0, 0.5 * np.pi, DEFAULT_CONTOUR_RAY_COUNT)


def default_boundary_theta_axis_rad() -> np.ndarray:
    """The same endpoint-clustered 12 boundary rays as the prior workflow."""
    parameter = np.linspace(0.0, 1.0, DEFAULT_BOUNDARY_THETA_COUNT)
    return 0.25 * np.pi * (1.0 - np.cos(np.pi * parameter))


def default_radial_fraction_axis() -> np.ndarray:
    return np.asarray(DEFAULT_RADIAL_FRACTIONS, dtype=float)


def build_safe_ray_mesh(
    *,
    transform: RayTransform,
    theta_axis_rad: np.ndarray,
    radial_fraction_axis: np.ndarray,
    rho_inner: np.ndarray,
    rho_outer: np.ndarray,
) -> dict[str, np.ndarray]:
    theta_axis = np.asarray(theta_axis_rad, dtype=float)
    radial_axis = np.asarray(radial_fraction_axis, dtype=float)
    inner = np.asarray(rho_inner, dtype=float)
    outer = np.asarray(rho_outer, dtype=float)
    if theta_axis.ndim != 1 or radial_axis.ndim != 1:
        raise ValueError("theta and radial-fraction axes must be one-dimensional")
    if inner.shape != theta_axis.shape or outer.shape != theta_axis.shape:
        raise ValueError("boundary radii must match the theta axis")
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        raise ValueError("boundary radii must be finite")
    if not np.all(outer > inner):
        raise ValueError("outer boundary must lie outside inner boundary")
    if not np.all((radial_axis > 0.0) & (radial_axis < 1.0)):
        raise ValueError("radial fractions must lie strictly inside (0, 1)")
    theta_grid = np.repeat(theta_axis[:, None], radial_axis.size, axis=1)
    radial_grid = np.repeat(radial_axis[None, :], theta_axis.size, axis=0)
    rho_grid = inner[:, None] + radial_grid * (outer - inner)[:, None]
    density_grid, temperature_grid = transform.to_physical(rho_grid, theta_grid)
    if (
        np.any(density_grid < DEFAULT_DENSITY_MIN_OVER_N0 - 1.0e-12)
        or np.any(density_grid > DEFAULT_DENSITY_MAX_OVER_N0 + 1.0e-12)
        or np.any(temperature_grid < DEFAULT_TEMPERATURE_MIN_MEV - 1.0e-12)
        or np.any(temperature_grid > DEFAULT_TEMPERATURE_MAX_MEV + 1.0e-12)
    ):
        raise RuntimeError("polar mesh escaped the validated physical window")
    return {
        "theta_axis_rad": theta_axis,
        "radial_fraction_axis": radial_axis,
        "theta_grid_rad": theta_grid,
        "radial_fraction_grid": radial_grid,
        "rho_inner_boundary": inner,
        "rho_outer_boundary": outer,
        "rho_grid": rho_grid,
        "nB_0minus_over_n0_grid": np.asarray(density_grid, dtype=float),
        "T_0minus_grid_MeV": np.asarray(temperature_grid, dtype=float),
    }


def _interface_boundary_residuals(
    nB_0minus_over_n0: float,
    T_0minus_MeV: float,
    *,
    B_one_forth: float,
    n0: float,
    muB_window: tuple[float, float, int] | None = None,
) -> dict[str, Any]:
    """Evaluate both ray-boundary residuals from one upstream EOS solve."""
    from RMFsolver import RMFparameter as para
    from RMFsolver import phase_velocity as pv

    nB_0minus = float(nB_0minus_over_n0) * float(n0)
    T_0minus_MeV = float(T_0minus_MeV)
    inversion_windows: list[dict[str, Any]] = []
    if muB_window is not None:
        lower, upper, points = (
            float(muB_window[0]),
            float(muB_window[1]),
            int(muB_window[2]),
        )
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            inversion_windows.append(
                {
                    "muB_lo": max(lower, 1.0),
                    "muB_hi": upper,
                    "scan_points": points,
                }
            )
    inversion_windows.append(
        {
            "muB_lo": MUB_COLD_LO_MEV,
            "muB_hi": MUB_COLD_HI_MEV,
            "scan_points": MUB_COLD_SCAN_POINTS,
            "auto_expand": True,
        }
    )
    muB_0minus = np.nan
    last_error = ""
    for window in inversion_windows:
        try:
            muB_0minus = float(
                pv.muB_from_nB_physical(
                    nB_0minus,
                    T_0minus_MeV,
                    param=para.paraQMCRMF3,
                    NM_type=DEFAULT_NM_TYPE,
                    **window,
                )
            )
            break
        except Exception as exc:
            last_error = str(exc)
    if not np.isfinite(muB_0minus):
        raise RuntimeError(f"muB inversion failed: {last_error}")

    upstream = pv._analytic_nuclear_state(
        muB_0minus,
        T_0minus_MeV,
        param=para.paraQMCRMF3,
        NM_type=DEFAULT_NM_TYPE,
    )
    P_0minus = float(upstream["P_0minus"])
    muB_inf = float(
        pv._solve_muB_inf_at_muK0_for_given_Pi(
            P_0minus,
            0.0,
            float(B_one_forth),
            T_0minus_MeV,
            ms=DEFAULT_MS_MEV,
            upB=DEFAULT_UPB,
        )
    )

    muB_0plus = np.nan
    muK_0plus = np.nan
    nK_0plus = np.nan
    nB_0plus = np.nan
    residual_a_0plus = np.nan
    try:
        muB_0plus, muK_0plus = (
            float(value)
            for value in pv._solve_interface_0plus_from_local_a_and_Pi(
                A_0PLUS_CEILING,
                P_0minus,
                0.0,
                float(B_one_forth),
                T_0minus_MeV,
                ms=DEFAULT_MS_MEV,
                upB=DEFAULT_UPB,
            )
        )
        nK_0plus = float(
            pv.nK_QM(
                muB_0plus,
                muK_0plus,
                float(B_one_forth),
                T_0minus_MeV,
                ms=DEFAULT_MS_MEV,
                upB=DEFAULT_UPB,
            )
        )
        nB_0plus = float(
            pv.nB_QM(
                muB_0plus,
                muK_0plus,
                float(B_one_forth),
                T_0minus_MeV,
                ms=DEFAULT_MS_MEV,
                upB=DEFAULT_UPB,
            )
        )
        residual_a_0plus = (
            muB_0plus + A_0PLUS_CEILING * muK_0plus - muB_0minus
        )
    except Exception:
        pass

    return {
        "nB_0minus_over_n0": float(nB_0minus_over_n0),
        "T_0minus_MeV": T_0minus_MeV,
        "muB_0minus_MeV": muB_0minus,
        "P_0minus_MeV4": P_0minus,
        "nB_0minus_MeV3": nB_0minus,
        "muB_inf_MeV": muB_inf,
        "delta_muB_MeV": muB_inf - muB_0minus,
        "a_0plus": A_0PLUS_CEILING,
        "muB_0plus_MeV": muB_0plus,
        "muK_0plus_MeV": muK_0plus,
        "nK_0plus_MeV3": nK_0plus,
        "nB_0plus_MeV3": nB_0plus,
        "nB_0plus_over_n0": nB_0plus / float(n0),
        "compression_ratio": nB_0plus / nB_0minus,
        "residual_a_0plus": residual_a_0plus,
    }


def _polar_muB_inversion_window(
    solved: list[tuple[float, float]], rho: float
) -> tuple[float, float, int] | None:
    """Predict a narrow baryon-chemical-potential bracket along one ray."""
    if not solved:
        return None
    ordered = sorted(solved, key=lambda point: abs(point[0] - rho))
    center = ordered[0][1]
    half_width = MUB_SEED_HALF_WIDTH_MEV
    if len(ordered) >= 2:
        (rho_a, muB_a), (rho_b, muB_b) = ordered[0], ordered[1]
        if abs(rho_b - rho_a) > 1.0e-12:
            slope = (muB_b - muB_a) / (rho_b - rho_a)
            center = muB_a + slope * (rho - rho_a)
            half_width = max(
                MUB_WARM_MIN_HALF_WIDTH_MEV,
                MUB_WARM_SLOPE_SAFETY * abs(slope) * abs(rho - rho_a),
            )
    half_width = min(half_width, MUB_WARM_MAX_HALF_WIDTH_MEV)
    return (
        center - half_width,
        center + half_width,
        MUB_WARM_SCAN_POINTS,
    )


def process_pool_available() -> bool:
    """Return whether Python can perform its process-pool semaphore check."""
    try:
        semaphore_limit = int(os.sysconf("SC_SEM_NSEMS_MAX"))
    except (AttributeError, OSError, PermissionError, TypeError, ValueError):
        return False
    return semaphore_limit == -1 or semaphore_limit >= 256


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_temperature_axis() -> np.ndarray:
    low = np.geomspace(1.0e-2, 1.0, 10)
    high = np.linspace(1.0, 120.0, 21)[1:]
    return np.concatenate((low, high))


def default_a_0plus_target_axis() -> np.ndarray:
    return np.linspace(0.01, 0.99, 20)


def phase_temperature_axis() -> np.ndarray:
    return np.concatenate(([0.0], default_temperature_axis()))


def proper_velocity_to_m_s(u_0minus: Any) -> Any:
    values = np.asarray(u_0minus, dtype=float)
    converted = SPEED_OF_LIGHT_M_S * values / np.sqrt(1.0 + values * values)
    if values.ndim == 0:
        return float(converted)
    return converted


def validate_cluster_physics(*, ms: float, NM_type: str, upB: int) -> None:
    if float(ms) != DEFAULT_MS_MEV:
        raise ValueError("The matched isothermal contour requires ms=0")
    if str(NM_type) != DEFAULT_NM_TYPE:
        raise ValueError("The matched isothermal contour requires NM_type='PNM'")
    if int(upB) != DEFAULT_UPB:
        raise ValueError(
            "The numerical isothermal API fixes upB=5000; matched contours require it"
        )


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def _live_signatures() -> dict[str, str]:
    try:
        from RMFsolver import phase_velocity as pv

        return {
            "phase_velocity_file": str(Path(pv.__file__).resolve()),
            "analytic_velocity_isothermal": str(
                inspect.signature(pv.analytic_velocity_isothermal)
            ),
            "solve_front_isothermal": str(
                inspect.signature(pv.solve_front_isothermal)
            ),
        }
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def polar_base_payload(
    *,
    kind: str,
    theta_axis_rad: np.ndarray,
    radial_fraction_axis: np.ndarray,
    B_one_forth: float,
    xi: float,
    ms: float,
    NM_type: str,
    upB: int,
) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": RUN_TAG,
        "payload_kind": str(kind),
        "run_complete": False,
        "created_at_utc": now,
        "updated_at_utc": now,
        "git_commit": _git_commit(),
        "phase_velocity_signatures": _live_signatures(),
        "coordinate_system": "dimensionless_elliptical_polar",
        "theta_axis_rad": np.asarray(theta_axis_rad, dtype=float),
        "radial_fraction_axis": np.asarray(radial_fraction_axis, dtype=float),
        "B_one_forth_MeV": float(B_one_forth),
        "xi": float(xi),
        "ms_MeV": float(ms),
        "NM_type": str(NM_type),
        "upB": int(upB),
        "density_window_over_n0": np.asarray(
            (DEFAULT_DENSITY_MIN_OVER_N0, DEFAULT_DENSITY_MAX_OVER_N0),
            dtype=float,
        ),
        "temperature_window_MeV": np.asarray(
            (DEFAULT_TEMPERATURE_MIN_MEV, DEFAULT_TEMPERATURE_MAX_MEV),
            dtype=float,
        ),
        "composition_definition": "a_0plus_equals_nK_0plus_over_nB_0plus",
        "composition_selection": "thermodynamic_maximum_weighted_muK_balance",
        "velocity_definition": "u_0minus_is_proper_velocity_gamma_times_v_over_c",
    }


def base_payload(
    *,
    kind: str,
    temperature_axis: np.ndarray,
    a_0plus_target_axis: np.ndarray,
    B_one_forth: float,
    xi: float,
    ms: float,
    NM_type: str,
    upB: int,
) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": RUN_TAG,
        "payload_kind": str(kind),
        "run_complete": False,
        "created_at_utc": now,
        "updated_at_utc": now,
        "git_commit": _git_commit(),
        "phase_velocity_signatures": _live_signatures(),
        "temperature_axis_MeV": np.asarray(temperature_axis, dtype=float),
        "a_0plus_target_axis": np.asarray(a_0plus_target_axis, dtype=float),
        "B_one_forth_MeV": float(B_one_forth),
        "xi": float(xi),
        "ms_MeV": float(ms),
        "NM_type": str(NM_type),
        "upB": int(upB),
        "composition_definition": "a_0plus_equals_nK_0plus_over_nB_0plus",
        "composition_selection": "thermodynamic_maximum_weighted_muK_balance",
        "velocity_definition": "u_0minus_is_proper_velocity_gamma_times_v_over_c",
    }


def atomic_save(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    materialized = dict(payload)
    materialized["updated_at_utc"] = utc_now()
    try:
        with temporary.open("wb") as stream:
            np.save(stream, materialized, allow_pickle=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_payload(path: str | Path, *, expected_kind: str | None = None) -> dict[str, Any]:
    source = Path(path)
    payload = np.load(source, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Payload at {source} is not a dictionary")
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(
            f"Payload schema mismatch: expected {SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}"
        )
    if expected_kind is not None and payload.get("payload_kind") != expected_kind:
        raise RuntimeError(
            f"Payload kind mismatch: expected {expected_kind!r}, "
            f"got {payload.get('payload_kind')!r}"
        )
    return payload


def assert_compatible_payload(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    scalar_keys = (
        "run_tag",
        "B_one_forth_MeV",
        "xi",
        "ms_MeV",
        "NM_type",
        "upB",
    )
    for key in scalar_keys:
        if candidate.get(key) != reference.get(key):
            raise RuntimeError(f"Resume payload parameter mismatch for {key}")
    if reference.get("coordinate_system") != candidate.get("coordinate_system"):
        raise RuntimeError("Resume payload coordinate-system mismatch")
    if reference.get("coordinate_system") == "dimensionless_elliptical_polar":
        axis_keys = ("theta_axis_rad", "radial_fraction_axis")
    else:
        axis_keys = ("temperature_axis_MeV", "a_0plus_target_axis")
    for key in axis_keys:
        if not np.array_equal(np.asarray(candidate.get(key)), np.asarray(reference.get(key))):
            raise RuntimeError(f"Resume payload axis mismatch for {key}")
    if candidate.get("phase_velocity_signatures") != reference.get(
        "phase_velocity_signatures"
    ):
        raise RuntimeError("Resume payload live API signature mismatch")
    expected_fingerprint = reference.get("domain_fingerprint")
    if expected_fingerprint is not None and candidate.get("domain_fingerprint") != expected_fingerprint:
        raise RuntimeError("Resume payload domain fingerprint mismatch")


def compute_domain_fingerprint(payload: dict[str, Any]) -> str:
    """Hash every coordinate-defining domain value used by later stages."""
    hasher = hashlib.sha256()
    scalar_keys = (
        "schema_version",
        "run_tag",
        "B_one_forth_MeV",
        "xi",
        "ms_MeV",
        "NM_type",
        "upB",
    )
    if payload.get("coordinate_system") == "dimensionless_elliptical_polar":
        array_keys = (
            "theta_axis_rad",
            "radial_fraction_axis",
            "boundary_theta_axis_rad",
            "rho_inner_boundary",
            "rho_outer_boundary",
            "lower_phase_nB_0minus",
            "lower_phase_T_0minus_MeV",
            "upper_phase_nB_0minus",
            "upper_phase_T_0minus_MeV",
            "cell_status",
            "nB_0minus_grid",
            "T_0minus_grid_MeV",
        )
    else:
        array_keys = (
            "temperature_axis_MeV",
            "a_0plus_target_axis",
            "phase_temperature_axis_MeV",
            "lower_phase_nB_0minus",
            "upper_phase_nB_0minus",
            "row_status",
            "cell_status",
            "nB_0minus_grid",
            "muB_0minus_grid",
            "P_0minus_grid",
            "a_0plus_max_grid",
            "a_0plus_residual_grid",
        )
    for key in scalar_keys:
        hasher.update(key.encode("utf-8"))
        hasher.update(json.dumps(payload.get(key), sort_keys=True).encode("utf-8"))
    for key in array_keys:
        if key not in payload:
            raise RuntimeError(f"Domain fingerprint requires {key}")
        values = np.asarray(payload[key])
        hasher.update(key.encode("utf-8"))
        hasher.update(str(values.shape).encode("ascii"))
        if values.dtype == object:
            encoded = json.dumps(values.tolist(), sort_keys=True).encode("utf-8")
            hasher.update(encoded)
        else:
            contiguous = np.ascontiguousarray(values)
            hasher.update(contiguous.dtype.str.encode("ascii"))
            hasher.update(contiguous.tobytes())
    hasher.update(
        json.dumps(payload.get("phase_velocity_signatures", {}), sort_keys=True).encode(
            "utf-8"
        )
    )
    return hasher.hexdigest()


def assert_domain_fingerprint(payload: dict[str, Any]) -> None:
    stored = payload.get("domain_fingerprint")
    computed = compute_domain_fingerprint(payload)
    if stored != computed:
        raise RuntimeError("Domain payload fingerprint does not match its contents")


def ceiling_at_upstream(
    *,
    T: float,
    nB_0minus: float,
    B_one_forth: float,
    ms: float = DEFAULT_MS_MEV,
    upB: int = DEFAULT_UPB,
) -> dict[str, Any]:
    from RMFsolver import RMFparameter as para
    from RMFsolver import phase_velocity as pv

    T = float(T)
    nB_0minus = float(nB_0minus)
    muB_0minus = float(
        pv.muB_from_nB_physical(
            nB_0minus,
            T,
            param=para.paraQMCRMF3,
            NM_type="PNM",
            auto_expand=True,
        )
    )
    upstream = pv._analytic_nuclear_state(
        muB_0minus,
        T,
        param=para.paraQMCRMF3,
        NM_type="PNM",
    )
    ceiling = dict(
        pv._solve_a_0plus_max(
            muB_0minus,
            float(upstream["P_0minus"]),
            T,
            float(B_one_forth),
            ms=float(ms),
            upB=int(upB),
        )
    )
    ceiling.update(
        {
            "T_0minus": T,
            "nB_0minus": nB_0minus,
            "muB_0minus": muB_0minus,
            "P_0minus": float(upstream["P_0minus"]),
            "weighted_mu_residual_MeV": float(
                ceiling["muB_0plus"]
                + ceiling["a_0plus_max"] * ceiling["muK_0plus"]
                - muB_0minus
            ),
        }
    )
    return ceiling


def ceiling_at_muB_0minus(
    *,
    T: float,
    muB_0minus: float,
    B_one_forth: float,
    ms: float = DEFAULT_MS_MEV,
    upB: int = DEFAULT_UPB,
    validate_nuclear_state: bool = False,
) -> dict[str, Any]:
    """Evaluate the live ceiling without repeating an nB-to-muB scan."""
    from RMFsolver import RMFparameter as para
    from RMFsolver import phase_velocity as pv

    T = float(T)
    muB_0minus = float(muB_0minus)
    if validate_nuclear_state:
        upstream = pv._analytic_nuclear_state(
            muB_0minus, T, param=para.paraQMCRMF3, NM_type="PNM"
        )
        P_0minus = float(upstream["P_0minus"])
        nB_0minus = float(upstream["nB_0minus"])
    else:
        P_0minus = float(
            pv.PNM(muB_0minus, T, param=para.paraQMCRMF3, NM_type="PNM")
        )
        nB_0minus = np.nan
    ceiling = dict(
        pv._solve_a_0plus_max(
            muB_0minus,
            P_0minus,
            T,
            float(B_one_forth),
            ms=float(ms),
            upB=int(upB),
        )
    )
    ceiling.update(
        {
            "T_0minus": T,
            "nB_0minus": nB_0minus,
            "muB_0minus": muB_0minus,
            "P_0minus": P_0minus,
            "weighted_mu_residual_MeV": float(
                ceiling["muB_0plus"]
                + ceiling["a_0plus_max"] * ceiling["muK_0plus"]
                - muB_0minus
            ),
        }
    )
    return ceiling


def _lower_pressure_residual(
    muB_0minus: float,
    T: float,
    B_one_forth: float,
    ms: float,
    upB: int,
) -> float:
    from RMFsolver import RMFparameter as para
    from RMFsolver import phase_velocity as pv

    return float(
        pv.PNM(muB_0minus, T, param=para.paraQMCRMF3, NM_type="PNM")
        - pv.PQM(muB_0minus, 0.0, B_one_forth, T, ms=ms, upB=upB)
    )


def _finite_brackets(coordinates: np.ndarray, residuals: np.ndarray) -> list[tuple[float, float]]:
    brackets: list[tuple[float, float]] = []
    for left, right, f_left, f_right in zip(
        coordinates[:-1], coordinates[1:], residuals[:-1], residuals[1:]
    ):
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if f_left == 0.0:
            brackets.append((float(left), float(left)))
        elif f_left * f_right < 0.0:
            brackets.append((float(left), float(right)))
    return brackets


def solve_lower_phase_boundary(
    T: float,
    B_one_forth: float,
    *,
    previous_muB_0minus: float | None = None,
    ms: float = DEFAULT_MS_MEV,
    upB: int = DEFAULT_UPB,
) -> dict[str, Any]:
    from RMFsolver import RMFparameter as para
    from RMFsolver import phase_velocity as pv

    T = float(T)
    center = 1100.0 if previous_muB_0minus is None else float(previous_muB_0minus)

    def residual(muB: float) -> float:
        try:
            return _lower_pressure_residual(
                float(muB), T, float(B_one_forth), float(ms), int(upB)
            )
        except Exception:
            return np.nan

    candidates: list[float] = []
    for half_width in (30.0, 60.0, 120.0, 240.0, 480.0):
        lower = max(300.0, center - half_width)
        upper = min(2000.0, center + half_width)
        f_lower, f_upper = residual(lower), residual(upper)
        if np.isfinite(f_lower) and np.isfinite(f_upper) and f_lower * f_upper <= 0.0:
            candidates.append(
                lower
                if f_lower == 0.0
                else float(brentq(residual, lower, upper, xtol=1.0e-8, rtol=1.0e-11))
            )
            break
    if not candidates:
        coordinates = np.linspace(300.0, 2000.0, 81)
        residuals = np.asarray([residual(value) for value in coordinates])
        for lower, upper in _finite_brackets(coordinates, residuals):
            candidates.append(
                lower
                if lower == upper
                else float(brentq(residual, lower, upper, xtol=1.0e-8, rtol=1.0e-11))
            )
    if not candidates:
        return {
            "status": "no_root",
            "T_MeV": T,
            "nB_0minus": np.nan,
            "muB_0minus": np.nan,
            "pressure_residual": np.nan,
        }
    muB_0minus = float(min(candidates, key=lambda value: abs(value - center)))
    try:
        state = pv._analytic_nuclear_state(
            muB_0minus, T, param=para.paraQMCRMF3, NM_type="PNM"
        )
    except Exception as exc:
        return {
            "status": "invalid_nuclear_branch",
            "message": str(exc),
            "T_MeV": T,
            "nB_0minus": np.nan,
            "muB_0minus": muB_0minus,
            "pressure_residual": residual(muB_0minus),
        }
    return {
        "status": "success",
        "T_MeV": T,
        "nB_0minus": float(state["nB_0minus"]),
        "muB_0minus": muB_0minus,
        "muB_0plus": muB_0minus,
        "muK_0plus": 0.0,
        "pressure_residual": residual(muB_0minus),
    }


def solve_upper_phase_boundary(
    T: float,
    B_one_forth: float,
    *,
    previous_state: tuple[float, float] | None = None,
    ms: float = DEFAULT_MS_MEV,
    upB: int = DEFAULT_UPB,
) -> dict[str, Any]:
    from RMFsolver import RMFparameter as para
    from RMFsolver import phase_velocity as pv

    T = float(T)
    seed = np.asarray(previous_state or (1640.0, 587.0), dtype=float)

    def scaled_residual(values: np.ndarray) -> np.ndarray:
        muB_0minus, muK_0plus = np.asarray(values, dtype=float)
        muB_0plus = muB_0minus - muK_0plus
        try:
            _, species = pv.nB_QM(
                muB_0plus,
                muK_0plus,
                float(B_one_forth),
                T,
                ms=float(ms),
                upB=int(upB),
                return_species=True,
            )
            pressure = pv.PNM(
                muB_0minus, T, param=para.paraQMCRMF3, NM_type="PNM"
            ) - pv.PQM(
                muB_0plus,
                muK_0plus,
                float(B_one_forth),
                T,
                ms=float(ms),
                upB=int(upB),
            )
            return np.asarray((pressure / 1.0e8, species["n_s"] / 1.0e6))
        except Exception:
            return np.asarray((1.0e12, 1.0e12))

    solution = root(
        scaled_residual,
        seed,
        method="hybr",
        options={"xtol": 1.0e-9},
    )
    closing = scaled_residual(solution.x)
    if (
        not solution.success
        or not np.all(np.isfinite(solution.x))
        or np.max(np.abs(closing)) > 1.0e-7
    ):
        return {
            "status": "no_root",
            "message": str(solution.message),
            "T_MeV": T,
            "nB_0minus": np.nan,
            "muB_0minus": np.nan,
            "muB_0plus": np.nan,
            "muK_0plus": np.nan,
            "scaled_residual": closing,
        }
    muB_0minus, muK_0plus = map(float, solution.x)
    muB_0plus = float(muB_0minus - muK_0plus)
    try:
        state = pv._analytic_nuclear_state(
            muB_0minus, T, param=para.paraQMCRMF3, NM_type="PNM"
        )
    except Exception as exc:
        return {
            "status": "invalid_nuclear_branch",
            "message": str(exc),
            "T_MeV": T,
            "nB_0minus": np.nan,
            "muB_0minus": muB_0minus,
            "muB_0plus": muB_0plus,
            "muK_0plus": muK_0plus,
            "scaled_residual": closing,
        }
    return {
        "status": "success",
        "T_MeV": T,
        "nB_0minus": float(state["nB_0minus"]),
        "muB_0minus": muB_0minus,
        "muB_0plus": muB_0plus,
        "muK_0plus": muK_0plus,
        "pressure_residual": float(closing[0] * 1.0e8),
        "strange_density_residual": float(closing[1] * 1.0e6),
    }


def solve_nB_0minus_for_a_0plus_max(
    *,
    T: float,
    a_0plus_target: float,
    lower_nB_0minus: float,
    upper_nB_0minus: float,
    lower_muB_0minus: float,
    upper_muB_0minus: float,
    B_one_forth: float,
    ms: float = DEFAULT_MS_MEV,
    upB: int = DEFAULT_UPB,
) -> dict[str, Any]:
    target = float(a_0plus_target)
    if not (0.0 < target < 1.0):
        raise ValueError("a_0plus_target must satisfy 0 < a_0plus_target < 1")
    lower = float(lower_nB_0minus)
    upper = float(upper_nB_0minus)
    lower_muB = float(lower_muB_0minus)
    upper_muB = float(upper_muB_0minus)
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or not lower < upper
        or not np.isfinite(lower_muB)
        or not np.isfinite(upper_muB)
        or not lower_muB < upper_muB
    ):
        return {"status": "no_allowed_band", "nB_0minus": np.nan}
    cache: dict[float, dict[str, Any]] = {}

    def evaluated(muB_value: float) -> dict[str, Any]:
        key = float(muB_value)
        if key not in cache:
            cache[key] = ceiling_at_muB_0minus(
                T=float(T),
                muB_0minus=key,
                B_one_forth=float(B_one_forth),
                ms=float(ms),
                upB=int(upB),
            )
        return cache[key]

    def residual(muB_value: float) -> float:
        return float(evaluated(muB_value)["a_0plus_max"] - target)

    try:
        f_lower, f_upper = residual(lower_muB), residual(upper_muB)
        if f_lower > 0.0 or f_upper < 0.0:
            raise RuntimeError(
                f"ceiling target is not bracketed: residuals {f_lower:.3e}, {f_upper:.3e}"
            )
        muB_0minus = float(
            brentq(
                residual,
                lower_muB,
                upper_muB,
                xtol=1.0e-8,
                rtol=1.0e-11,
                maxiter=100,
            )
        )
        ceiling = ceiling_at_muB_0minus(
            T=float(T),
            muB_0minus=muB_0minus,
            B_one_forth=float(B_one_forth),
            ms=float(ms),
            upB=int(upB),
            validate_nuclear_state=True,
        )
    except Exception as exc:
        return {
            "status": "ceiling_inversion_failed",
            "message": str(exc),
            "nB_0minus": np.nan,
            "a_0plus_target": target,
            "a_0plus_residual": np.nan,
        }
    achieved = float(ceiling["a_0plus_max"])
    result = dict(ceiling)
    result.update(
        {
            "status": "success" if ceiling["status"] == "interior" else "non_interior_ceiling",
            "ceiling_status": str(ceiling["status"]),
            "a_0plus_target": target,
            "a_0plus_residual": achieved - target,
        }
    )
    return result


def build_interior_row(
    *,
    T: float,
    a_0plus_target_axis: np.ndarray,
    lower_boundary: dict[str, Any],
    upper_boundary: dict[str, Any],
    B_one_forth: float,
    ms: float = DEFAULT_MS_MEV,
    upB: int = DEFAULT_UPB,
) -> dict[str, Any]:
    targets = np.asarray(a_0plus_target_axis, dtype=float)
    count = targets.size
    row = {
        "T_MeV": float(T),
        "row_status": "success",
        "cell_status": np.full(count, "pending", dtype=object),
        "nB_0minus": np.full(count, np.nan),
        "muB_0minus": np.full(count, np.nan),
        "P_0minus": np.full(count, np.nan),
        "a_0plus_max": np.full(count, np.nan),
        "a_0plus_residual": np.full(count, np.nan),
        "records": np.full(count, None, dtype=object),
    }
    lower = float(lower_boundary.get("nB_0minus", np.nan))
    upper = float(upper_boundary.get("nB_0minus", np.nan))
    if (
        lower_boundary.get("status") != "success"
        or upper_boundary.get("status") != "success"
        or not np.isfinite(lower)
        or not np.isfinite(upper)
        or not lower < upper
    ):
        row["row_status"] = "no_allowed_band"
        row["cell_status"][:] = "no_allowed_band"
        return row
    for index, target in enumerate(targets):
        record = solve_nB_0minus_for_a_0plus_max(
            T=float(T),
            a_0plus_target=float(target),
            lower_nB_0minus=lower,
            upper_nB_0minus=upper,
            lower_muB_0minus=float(lower_boundary.get("muB_0minus", np.nan)),
            upper_muB_0minus=float(upper_boundary.get("muB_0minus", np.nan)),
            B_one_forth=float(B_one_forth),
            ms=float(ms),
            upB=int(upB),
        )
        row["records"][index] = record
        row["cell_status"][index] = record["status"]
        if record["status"] == "success":
            for key in (
                "nB_0minus",
                "muB_0minus",
                "P_0minus",
                "a_0plus_max",
                "a_0plus_residual",
            ):
                row[key][index] = float(record[key])
    if not np.all(row["cell_status"] == "success"):
        row["row_status"] = "partial"
    return row


def _build_row_task(task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    index = int(task.pop("temperature_index"))
    return index, build_interior_row(**task)


def _open_progress(
    progress_factory: Callable[..., Any] | None,
    *,
    desc: str,
    total: int,
    initial: int = 0,
) -> Any | None:
    if progress_factory is None:
        return None
    return progress_factory(
        desc=desc,
        total=int(total),
        initial=int(initial),
        unit="point",
        dynamic_ncols=True,
        leave=True,
        miniters=1,
        mininterval=0.0,
    )


def _update_progress(progress: Any | None) -> None:
    if progress is not None:
        progress.update(1)


def _close_progress(progress: Any | None) -> None:
    if progress is not None:
        progress.close()


def _new_domain_payload(
    *,
    temperatures: np.ndarray,
    targets: np.ndarray,
    phase_temperatures: np.ndarray,
    B_one_forth: float,
    xi: float,
    ms: float,
    NM_type: str,
    upB: int,
) -> dict[str, Any]:
    payload = base_payload(
        kind="isothermal_domain",
        temperature_axis=temperatures,
        a_0plus_target_axis=targets,
        B_one_forth=float(B_one_forth),
        xi=float(xi),
        ms=float(ms),
        NM_type=str(NM_type),
        upB=int(upB),
    )
    shape = (temperatures.size, targets.size)
    payload.update(
        {
            "phase_temperature_axis_MeV": phase_temperatures,
            "lower_phase_records": np.full(phase_temperatures.size, None, dtype=object),
            "upper_phase_records": np.full(phase_temperatures.size, None, dtype=object),
            "lower_phase_nB_0minus": np.full(phase_temperatures.size, np.nan),
            "upper_phase_nB_0minus": np.full(phase_temperatures.size, np.nan),
            "row_status": np.full(temperatures.size, "pending", dtype=object),
            "cell_status": np.full(shape, "pending", dtype=object),
            "nB_0minus_grid": np.full(shape, np.nan),
            "muB_0minus_grid": np.full(shape, np.nan),
            "P_0minus_grid": np.full(shape, np.nan),
            "a_0plus_max_grid": np.full(shape, np.nan),
            "a_0plus_residual_grid": np.full(shape, np.nan),
            "cell_records": np.full(shape, None, dtype=object),
            "domain_fingerprint": None,
        }
    )
    return payload


def _boundary_is_complete(record: Any) -> bool:
    return isinstance(record, dict) and bool(record.get("status"))


def _build_cell_task(task: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    i = int(task.pop("temperature_index"))
    j = int(task.pop("composition_index"))
    lower_boundary = dict(task.pop("lower_boundary"))
    upper_boundary = dict(task.pop("upper_boundary"))
    lower = float(lower_boundary.get("nB_0minus", np.nan))
    upper = float(upper_boundary.get("nB_0minus", np.nan))
    target = float(task["a_0plus_target"])
    if (
        lower_boundary.get("status") != "success"
        or upper_boundary.get("status") != "success"
        or not np.isfinite(lower)
        or not np.isfinite(upper)
        or not lower < upper
    ):
        return i, j, {
            "status": "no_allowed_band",
            "T_MeV": float(task["T"]),
            "a_0plus_target": target,
            "nB_0minus": np.nan,
            "a_0plus_residual": np.nan,
        }
    record = solve_nB_0minus_for_a_0plus_max(
        T=float(task["T"]),
        a_0plus_target=target,
        lower_nB_0minus=lower,
        upper_nB_0minus=upper,
        lower_muB_0minus=float(lower_boundary.get("muB_0minus", np.nan)),
        upper_muB_0minus=float(upper_boundary.get("muB_0minus", np.nan)),
        B_one_forth=float(task["B_one_forth"]),
        ms=float(task["ms"]),
        upB=int(task["upB"]),
    )
    return i, j, record


def _build_cell_child(connection: Any, task: dict[str, Any]) -> None:
    try:
        connection.send(("result", _build_cell_task(task)))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _build_cell_in_spawned_process(
    task: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    """Run one CPU-bound domain inversion without process-pool semaphores."""
    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_build_cell_child,
        args=(send_connection, dict(task)),
    )
    process.start()
    send_connection.close()
    try:
        try:
            message = receive_connection.recv()
        except EOFError as exc:
            process.join(timeout=2.0)
            raise RuntimeError(
                f"Domain child exited with code {process.exitcode} without a result"
            ) from exc
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if message[0] == "error":
            raise RuntimeError(message[1])
        return message[1]
    finally:
        receive_connection.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)


def _store_domain_cell(
    payload: dict[str, Any], i: int, j: int, record: dict[str, Any]
) -> None:
    payload["cell_records"][i, j] = record
    payload["cell_status"][i, j] = str(record["status"])
    if record["status"] == "success":
        for record_key, grid_key in (
            ("nB_0minus", "nB_0minus_grid"),
            ("muB_0minus", "muB_0minus_grid"),
            ("P_0minus", "P_0minus_grid"),
            ("a_0plus_max", "a_0plus_max_grid"),
            ("a_0plus_residual", "a_0plus_residual_grid"),
        ):
            payload[grid_key][i, j] = float(record[record_key])
    row = np.asarray(payload["cell_status"], dtype=object)[i]
    if np.any(row == "pending"):
        payload["row_status"][i] = "pending"
    elif np.all(row == "success"):
        payload["row_status"][i] = "success"
    elif np.all(row == "no_allowed_band"):
        payload["row_status"][i] = "no_allowed_band"
    else:
        payload["row_status"][i] = "partial"


def _finalize_domain_if_complete(payload: dict[str, Any]) -> None:
    payload["run_complete"] = not np.any(
        np.asarray(payload["cell_status"], dtype=object) == "pending"
    )
    if payload["run_complete"]:
        payload["domain_fingerprint"] = compute_domain_fingerprint(payload)


def build_domain_payload(
    *,
    temperature_axis: np.ndarray,
    a_0plus_target_axis: np.ndarray,
    B_one_forth: float,
    xi: float,
    ms: float,
    NM_type: str,
    upB: int,
    workers: int = 1,
    output_path: str | Path | None = None,
    progress_factory: Callable[..., Any] | None = None,
    resume_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_cluster_physics(ms=ms, NM_type=NM_type, upB=upB)
    temperatures = np.asarray(temperature_axis, dtype=float)
    targets = np.asarray(a_0plus_target_axis, dtype=float)
    if np.any(temperatures <= 0.0) or not np.all(np.isfinite(temperatures)):
        raise ValueError("Moving-front temperature axis must be positive and finite")
    if np.any((targets <= 0.0) | (targets >= 1.0)):
        raise ValueError("a_0plus target axis must lie strictly inside (0, 1)")
    phase_temperatures = np.concatenate(([0.0], temperatures))
    reference = base_payload(
        kind="isothermal_domain",
        temperature_axis=temperatures,
        a_0plus_target_axis=targets,
        B_one_forth=float(B_one_forth),
        xi=float(xi),
        ms=float(ms),
        NM_type=str(NM_type),
        upB=int(upB),
    )
    if resume_payload is None:
        payload = _new_domain_payload(
            temperatures=temperatures,
            targets=targets,
            phase_temperatures=phase_temperatures,
            B_one_forth=float(B_one_forth),
            xi=float(xi),
            ms=float(ms),
            NM_type=str(NM_type),
            upB=int(upB),
        )
    else:
        assert_compatible_payload(reference, resume_payload)
        payload = resume_payload
        if bool(payload.get("run_complete")):
            assert_domain_fingerprint(payload)
            return payload

    checkpoint_path = Path(output_path) if output_path is not None else None

    lower_records = np.asarray(payload["lower_phase_records"], dtype=object)
    previous_lower_muB = 1100.0
    lower_progress = _open_progress(
        progress_factory,
        desc="Stable neutron matter boundary",
        total=phase_temperatures.size,
        initial=sum(_boundary_is_complete(record) for record in lower_records),
    )
    try:
        for phase_index, T in enumerate(phase_temperatures):
            existing = lower_records[phase_index]
            if _boundary_is_complete(existing):
                if existing.get("status") == "success":
                    previous_lower_muB = float(existing["muB_0minus"])
                continue
            lower = solve_lower_phase_boundary(
                float(T),
                float(B_one_forth),
                previous_muB_0minus=previous_lower_muB,
                ms=float(ms),
                upB=int(upB),
            )
            lower_records[phase_index] = lower
            payload["lower_phase_nB_0minus"][phase_index] = float(
                lower.get("nB_0minus", np.nan)
            )
            if lower.get("status") == "success":
                previous_lower_muB = float(lower["muB_0minus"])
            if checkpoint_path is not None:
                atomic_save(checkpoint_path, payload)
            _update_progress(lower_progress)
    finally:
        _close_progress(lower_progress)

    upper_records = np.asarray(payload["upper_phase_records"], dtype=object)
    previous_upper_state = (1640.0, 587.0)
    upper_progress = _open_progress(
        progress_factory,
        desc="a(0+)=1 boundary",
        total=phase_temperatures.size,
        initial=sum(_boundary_is_complete(record) for record in upper_records),
    )
    try:
        for phase_index, T in enumerate(phase_temperatures):
            existing = upper_records[phase_index]
            if _boundary_is_complete(existing):
                if existing.get("status") == "success":
                    previous_upper_state = (
                        float(existing["muB_0minus"]),
                        float(existing["muK_0plus"]),
                    )
                continue
            upper = solve_upper_phase_boundary(
                float(T),
                float(B_one_forth),
                previous_state=previous_upper_state,
                ms=float(ms),
                upB=int(upB),
            )
            upper_records[phase_index] = upper
            payload["upper_phase_nB_0minus"][phase_index] = float(
                upper.get("nB_0minus", np.nan)
            )
            if upper.get("status") == "success":
                previous_upper_state = (
                    float(upper["muB_0minus"]),
                    float(upper["muK_0plus"]),
                )
            if checkpoint_path is not None:
                atomic_save(checkpoint_path, payload)
            _update_progress(upper_progress)
    finally:
        _close_progress(upper_progress)

    tasks: list[dict[str, Any]] = []
    for temperature_index, T in enumerate(temperatures):
        phase_index = temperature_index + 1
        for composition_index, target in enumerate(targets):
            if payload["cell_status"][temperature_index, composition_index] != "pending":
                continue
            tasks.append(
                {
                    "temperature_index": temperature_index,
                    "composition_index": composition_index,
                    "T": float(T),
                    "a_0plus_target": float(target),
                    "lower_boundary": lower_records[phase_index],
                    "upper_boundary": upper_records[phase_index],
                    "B_one_forth": float(B_one_forth),
                    "ms": float(ms),
                    "upB": int(upB),
                }
            )

    terminal_count = int(np.count_nonzero(payload["cell_status"] != "pending"))
    grid_progress = _open_progress(
        progress_factory,
        desc="Domain grid",
        total=payload["cell_status"].size,
        initial=terminal_count,
    )

    def store_checkpoint(result: tuple[int, int, dict[str, Any]]) -> None:
        i, j, record = result
        _store_domain_cell(payload, i, j, record)
        _finalize_domain_if_complete(payload)
        if checkpoint_path is not None:
            atomic_save(checkpoint_path, payload)
        _update_progress(grid_progress)

    worker_count = max(1, min(int(workers), max(len(tasks), 1)))
    try:
        if worker_count == 1:
            for task in tasks:
                store_checkpoint(_build_cell_task(dict(task)))
        elif tasks:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(_build_cell_in_spawned_process, dict(task))
                    for task in tasks
                ]
                for future in as_completed(futures):
                    store_checkpoint(future.result())
    finally:
        _close_progress(grid_progress)

    if not tasks:
        was_complete = bool(payload.get("run_complete"))
        _finalize_domain_if_complete(payload)
        if checkpoint_path is not None and payload["run_complete"] and not was_complete:
            atomic_save(checkpoint_path, payload)
    return payload


def _solve_polar_boundary_ray(task: dict[str, Any]) -> dict[str, Any]:
    """Trace both physical boundaries along one independent polar ray."""
    import warnings

    from scipy.integrate import IntegrationWarning

    warnings.filterwarnings("ignore", category=IntegrationWarning)
    theta = float(task["theta_rad"])
    transform = RayTransform(**task["transform"])
    B_one_forth = float(task["B14"])
    n0 = float(task["n0"])
    probe_count = int(task["probe_count"])
    rho_tolerance = float(task["rho_tol"])
    rho_lower = float(task["rho_lower"])
    rho_upper = transform.rho_upper_bound(theta)
    cache: dict[float, dict[str, Any]] = {}

    def evaluate(rho: float) -> dict[str, Any]:
        key = float(np.round(rho, 12))
        if key in cache:
            return cache[key]
        density, temperature = (
            float(value) for value in transform.to_physical(key, theta)
        )
        solved = [
            (row["rho"], row["muB_0minus_MeV"])
            for row in cache.values()
            if np.isfinite(row.get("muB_0minus_MeV", np.nan))
        ]
        row = _interface_boundary_residuals(
            density,
            temperature,
            B_one_forth=B_one_forth,
            n0=n0,
            muB_window=_polar_muB_inversion_window(solved, key),
        )
        row["rho"] = key
        row["theta_rad"] = theta
        cache[key] = row
        return row

    probes = np.linspace(rho_lower, rho_upper, probe_count)
    for probe in probes:
        evaluate(float(probe))

    def solve_residual(field: str) -> dict[str, Any]:
        values = np.asarray(
            [evaluate(float(radius))[field] for radius in probes], dtype=float
        )
        brackets = [
            (float(probes[index]), float(probes[index + 1]))
            for index in range(probes.size - 1)
            if np.all(np.isfinite((values[index], values[index + 1])))
            and values[index] * values[index + 1] <= 0.0
        ]
        if len(brackets) != 1:
            return {
                "status": "no_unique_bracket",
                "theta_rad": theta,
                "bracket_count": len(brackets),
                "finite_probe_count": int(np.count_nonzero(np.isfinite(values))),
            }
        lower, upper = brackets[0]
        rho_root = float(
            brentq(
                lambda radius: evaluate(float(radius))[field],
                lower,
                upper,
                xtol=rho_tolerance,
                rtol=1.0e-12,
            )
        )
        record = dict(evaluate(rho_root))
        record.update(
            {
                "status": "root",
                "residual_field": field,
                "bracket_lower_rho": lower,
                "bracket_upper_rho": upper,
            }
        )
        return record

    return {
        "theta_rad": theta,
        "theta_deg": float(np.degrees(theta)),
        "rho_lower": rho_lower,
        "rho_upper": rho_upper,
        "delta_muB_equals_zero": solve_residual("delta_muB_MeV"),
        "a_0plus_equals_one": solve_residual("residual_a_0plus"),
        "evaluation_count": len(cache),
        "probe_rows": [dict(row) for row in cache.values()],
    }


def _polar_boundary_ray_child(connection: Any, task: dict[str, Any]) -> None:
    try:
        connection.send(("result", _solve_polar_boundary_ray(task)))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _solve_polar_boundary_ray_in_spawned_process(
    task: dict[str, Any],
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_polar_boundary_ray_child,
        args=(send_connection, dict(task)),
    )
    process.start()
    send_connection.close()
    try:
        try:
            message = receive_connection.recv()
        except EOFError as exc:
            process.join(timeout=2.0)
            raise RuntimeError(
                f"Boundary child exited with code {process.exitcode} without a result"
            ) from exc
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if message[0] == "error":
            raise RuntimeError(message[1])
        return dict(message[1])
    finally:
        receive_connection.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)


def _boundary_root(ray: dict[str, Any], kind: str) -> dict[str, Any]:
    record = ray.get(kind)
    if not isinstance(record, dict) or record.get("status") != "root":
        theta = float(ray.get("theta_rad", np.nan))
        status = record.get("status") if isinstance(record, dict) else "missing"
        raise RuntimeError(
            f"Incomplete {kind} boundary at theta={theta:g}: {status}"
        )
    rho = float(record.get("rho", np.nan))
    if not np.isfinite(rho):
        raise RuntimeError(f"Non-finite {kind} boundary radius")
    return record


def build_domain_payload_from_rays(
    *,
    boundary_rays: list[dict[str, Any]],
    theta_axis_rad: np.ndarray,
    radial_fraction_axis: np.ndarray,
    B_one_forth: float,
    xi: float,
    ms: float,
    NM_type: str,
    upB: int,
    transform: RayTransform | None = None,
) -> dict[str, Any]:
    """Interpolate two traced boundaries and construct the physical ray mesh."""
    validate_cluster_physics(ms=ms, NM_type=NM_type, upB=upB)
    if len(boundary_rays) < 2:
        raise RuntimeError("At least two complete boundary rays are required")
    transform = transform or RayTransform()
    ordered = sorted(boundary_rays, key=lambda row: float(row["theta_rad"]))
    boundary_theta = np.asarray(
        [float(ray["theta_rad"]) for ray in ordered], dtype=float
    )
    if not np.all(np.diff(boundary_theta) > 0.0):
        raise RuntimeError("Boundary-ray angles must be unique and increasing")
    inner_records = [
        _boundary_root(ray, "delta_muB_equals_zero") for ray in ordered
    ]
    outer_records = [
        _boundary_root(ray, "a_0plus_equals_one") for ray in ordered
    ]
    inner_boundary_rho = np.asarray(
        [float(record["rho"]) for record in inner_records], dtype=float
    )
    outer_boundary_rho = np.asarray(
        [float(record["rho"]) for record in outer_records], dtype=float
    )
    if not np.all(outer_boundary_rho > inner_boundary_rho):
        raise RuntimeError("The a(0+)=1 boundary is not outside the coexistence boundary")
    theta_axis = np.asarray(theta_axis_rad, dtype=float)
    if (
        theta_axis.ndim != 1
        or theta_axis.size < 2
        or theta_axis[0] < boundary_theta[0] - 1.0e-12
        or theta_axis[-1] > boundary_theta[-1] + 1.0e-12
    ):
        raise ValueError("Contour rays must lie inside the traced angular interval")
    rho_inner = np.interp(theta_axis, boundary_theta, inner_boundary_rho)
    rho_outer = np.interp(theta_axis, boundary_theta, outer_boundary_rho)
    mesh = build_safe_ray_mesh(
        transform=transform,
        theta_axis_rad=theta_axis,
        radial_fraction_axis=np.asarray(radial_fraction_axis, dtype=float),
        rho_inner=rho_inner,
        rho_outer=rho_outer,
    )
    from RMFsolver import constants as const

    n0 = 0.16 * float(const.MeV_fm) ** 3
    shape = mesh["rho_grid"].shape
    lower_density_over_n0, lower_temperature = transform.to_physical(
        inner_boundary_rho, boundary_theta
    )
    upper_density_over_n0, upper_temperature = transform.to_physical(
        outer_boundary_rho, boundary_theta
    )
    payload = polar_base_payload(
        kind="isothermal_domain",
        theta_axis_rad=theta_axis,
        radial_fraction_axis=np.asarray(radial_fraction_axis, dtype=float),
        B_one_forth=float(B_one_forth),
        xi=float(xi),
        ms=float(ms),
        NM_type=str(NM_type),
        upB=int(upB),
    )
    payload.update(
        {
            "ray_transform": transform.metadata(),
            "boundary_theta_axis_rad": boundary_theta,
            "boundary_ray_records": np.asarray(ordered, dtype=object),
            "rho_inner_boundary": rho_inner,
            "rho_outer_boundary": rho_outer,
            "rho_grid": mesh["rho_grid"],
            "theta_grid_rad": mesh["theta_grid_rad"],
            "radial_fraction_grid": mesh["radial_fraction_grid"],
            "lower_phase_nB_0minus": np.asarray(lower_density_over_n0) * n0,
            "lower_phase_T_0minus_MeV": np.asarray(lower_temperature),
            "upper_phase_nB_0minus": np.asarray(upper_density_over_n0) * n0,
            "upper_phase_T_0minus_MeV": np.asarray(upper_temperature),
            "nB_0minus_grid": mesh["nB_0minus_over_n0_grid"] * n0,
            "nB_0minus_over_n0_grid": mesh["nB_0minus_over_n0_grid"],
            "T_0minus_grid_MeV": mesh["T_0minus_grid_MeV"],
            "cell_status": np.full(shape, "success", dtype=object),
            "cell_records": np.full(shape, None, dtype=object),
            "domain_fingerprint": None,
        }
    )
    for i in range(shape[0]):
        for j in range(shape[1]):
            payload["cell_records"][i, j] = {
                "status": "success",
                "ray_index": i,
                "radial_index": j,
                "theta_rad": float(payload["theta_grid_rad"][i, j]),
                "radial_fraction": float(payload["radial_fraction_grid"][i, j]),
                "rho": float(payload["rho_grid"][i, j]),
                "nB_0minus": float(payload["nB_0minus_grid"][i, j]),
                "T_0minus_MeV": float(payload["T_0minus_grid_MeV"][i, j]),
            }
    payload["run_complete"] = True
    payload["domain_fingerprint"] = compute_domain_fingerprint(payload)
    return payload


def build_polar_domain_payload(
    *,
    theta_axis_rad: np.ndarray,
    radial_fraction_axis: np.ndarray,
    boundary_theta_axis_rad: np.ndarray,
    B_one_forth: float,
    xi: float,
    ms: float,
    NM_type: str,
    upB: int,
    workers: int = 1,
    probe_count: int = DEFAULT_BOUNDARY_PROBE_COUNT,
    rho_lower: float = DEFAULT_RHO_LOWER,
    rho_tol: float = DEFAULT_RHO_TOL,
    output_path: str | Path | None = None,
    progress_factory: Callable[..., Any] | None = None,
    resume_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Trace both boundaries, then checkpoint all 600 deterministic mesh cells."""
    validate_cluster_physics(ms=ms, NM_type=NM_type, upB=upB)
    theta_axis = np.asarray(theta_axis_rad, dtype=float)
    radial_axis = np.asarray(radial_fraction_axis, dtype=float)
    boundary_theta = np.asarray(boundary_theta_axis_rad, dtype=float)
    reference = polar_base_payload(
        kind="isothermal_domain",
        theta_axis_rad=theta_axis,
        radial_fraction_axis=radial_axis,
        B_one_forth=B_one_forth,
        xi=xi,
        ms=ms,
        NM_type=NM_type,
        upB=upB,
    )
    checkpoint_path = Path(output_path) if output_path is not None else None
    if resume_payload is None:
        payload = dict(reference)
        payload.update(
            {
                "boundary_theta_axis_rad": boundary_theta,
                "boundary_ray_records": np.full(boundary_theta.size, None, dtype=object),
                "domain_fingerprint": None,
            }
        )
    else:
        assert_compatible_payload(reference, resume_payload)
        payload = resume_payload
        if not np.array_equal(
            np.asarray(payload.get("boundary_theta_axis_rad")), boundary_theta
        ):
            raise RuntimeError("Resume payload boundary-ray axis mismatch")
        if bool(payload.get("run_complete")):
            assert_domain_fingerprint(payload)
            return payload

    transform = RayTransform()
    transform_kwargs = {
        "density_origin_over_n0": transform.density_origin_over_n0,
        "density_scale_over_n0": transform.density_scale_over_n0,
        "temperature_scale_MeV": transform.temperature_scale_MeV,
        "temperature_min_MeV": transform.temperature_min_MeV,
    }
    from RMFsolver import constants as const

    n0 = 0.16 * float(const.MeV_fm) ** 3
    ray_records = np.asarray(payload["boundary_ray_records"], dtype=object)
    tasks = [
        {
            "boundary_index": index,
            "theta_rad": float(theta),
            "transform": transform_kwargs,
            "B14": float(B_one_forth),
            "n0": n0,
            "probe_count": int(probe_count),
            "rho_lower": float(rho_lower),
            "rho_tol": float(rho_tol),
        }
        for index, theta in enumerate(boundary_theta)
        if not isinstance(ray_records[index], dict)
    ]
    completed_rays = int(sum(isinstance(record, dict) for record in ray_records))
    lower_progress = _open_progress(
        progress_factory,
        desc="Stable neutron matter boundary rays",
        total=boundary_theta.size,
        initial=completed_rays,
    )
    upper_progress = _open_progress(
        progress_factory,
        desc="a(0+)=1 boundary rays",
        total=boundary_theta.size,
        initial=completed_rays,
    )

    def store_ray(ray: dict[str, Any]) -> None:
        index = int(np.argmin(np.abs(boundary_theta - float(ray["theta_rad"]))))
        ray_records[index] = ray
        if checkpoint_path is not None:
            atomic_save(checkpoint_path, payload)
        _update_progress(lower_progress)
        _update_progress(upper_progress)

    worker_count = max(1, min(int(workers), max(len(tasks), 1)))
    try:
        if worker_count == 1:
            for task in tasks:
                store_ray(_solve_polar_boundary_ray(dict(task)))
        elif tasks:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _solve_polar_boundary_ray_in_spawned_process, dict(task)
                    )
                    for task in tasks
                ]
                for future in as_completed(futures):
                    store_ray(future.result())
    finally:
        _close_progress(lower_progress)
        _close_progress(upper_progress)

    built = build_domain_payload_from_rays(
        boundary_rays=[dict(record) for record in ray_records],
        theta_axis_rad=theta_axis,
        radial_fraction_axis=radial_axis,
        B_one_forth=B_one_forth,
        xi=xi,
        ms=ms,
        NM_type=NM_type,
        upB=upB,
        transform=transform,
    )
    created_at = payload.get("created_at_utc", built["created_at_utc"])
    if "cell_status" in payload:
        built["cell_status"] = np.asarray(payload["cell_status"], dtype=object)
        built["cell_records"] = np.asarray(payload["cell_records"], dtype=object)
    else:
        built["cell_status"][:] = "pending"
        built["cell_records"][:] = None
    built["created_at_utc"] = created_at
    built["run_complete"] = False
    built["domain_fingerprint"] = None
    payload = built
    total_cells = int(payload["cell_status"].size)
    initial_cells = int(np.count_nonzero(payload["cell_status"] != "pending"))
    grid_progress = _open_progress(
        progress_factory,
        desc="Polar domain mesh",
        total=total_cells,
        initial=initial_cells,
    )
    try:
        for i in range(theta_axis.size):
            for j in range(radial_axis.size):
                if payload["cell_status"][i, j] != "pending":
                    continue
                payload["cell_status"][i, j] = "success"
                payload["cell_records"][i, j] = {
                    "status": "success",
                    "ray_index": i,
                    "radial_index": j,
                    "theta_rad": float(payload["theta_grid_rad"][i, j]),
                    "radial_fraction": float(
                        payload["radial_fraction_grid"][i, j]
                    ),
                    "rho": float(payload["rho_grid"][i, j]),
                    "nB_0minus": float(payload["nB_0minus_grid"][i, j]),
                    "T_0minus_MeV": float(payload["T_0minus_grid_MeV"][i, j]),
                }
                if not np.any(payload["cell_status"] == "pending"):
                    payload["run_complete"] = True
                    payload["domain_fingerprint"] = compute_domain_fingerprint(payload)
                if checkpoint_path is not None:
                    atomic_save(checkpoint_path, payload)
                _update_progress(grid_progress)
    finally:
        _close_progress(grid_progress)
    if not bool(payload["run_complete"]):
        raise RuntimeError("Polar domain mesh did not complete")
    return payload
