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


SCHEMA_VERSION = 2
RUN_TAG = "isothermal-contour-26-08-19"
SPEED_OF_LIGHT_M_S = 299_792_458.0
DEFAULT_B_ONE_FORTH_MEV = 189.1565957288247
DEFAULT_XI = -0.5
DEFAULT_MS_MEV = 0.0
DEFAULT_NM_TYPE = "PNM"
DEFAULT_UPB = 5000
DEFAULT_DENSITY_MIN_OVER_N0 = 1.0
DEFAULT_DENSITY_MAX_OVER_N0 = 5.5


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
            "analytic_velocity_isothermal": str(
                inspect.signature(pv.analytic_velocity_isothermal)
            ),
            "solve_front_isothermal": str(
                inspect.signature(pv.solve_front_isothermal)
            ),
        }
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


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
    for key in ("temperature_axis_MeV", "a_0plus_target_axis"):
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
