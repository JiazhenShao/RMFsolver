#!/usr/bin/env python3
"""Shared implementation for the nB_crit(0)=2n0, T(0+)=0.1 MeV runners."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import platform
import queue
import sys
import tempfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import RMFsolver.constants as const
import _method_engine as benchmark
import _radial_shell_continuation as radial


DOMAIN_FILENAME = "T0plus0p1-ncrit2-domain.npy"
FULL_ANALYTIC_FILENAME = "T0plus0p1-ncrit2-full-analytic.npy"
SEMI_ANALYTIC_FILENAME = "T0plus0p1-ncrit2-semi-analytic.npy"
NUMERICAL_FILENAME = "T0plus0p1-ncrit2-uNmax-angular-wavefront.npy"
CONTINUATION_STAGE_ORDER = (
    "radial_primary",
    "angular_direct_wavefront",
    "angular_perturbed_wavefront",
    "radial_multiplier_fallback",
    "angular_recovery_after_radial_fallback",
    "radial_bridge_fallback",
)
DIRECT_BOUNDARY_METHOD = "FL26-3_coupled_interface_roots"
DIRECT_A_T_0PLUS_MEV = 1.0e-3
DIRECT_BOUNDARY_ROOT_TOLERANCE = 2.0e-7
DIRECT_BOUNDARY_MAX_NFEV = 220


def resume_state_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a partial radial-shell payload into mutable runner state."""
    return {
        "records": {
            int(record["point_index"]): dict(record)
            for record in payload.get("radial_shell_records", [])
        },
        "attempt_records": list(payload.get("radial_shell_attempt_records", [])),
        "bridge_records": list(payload.get("radial_shell_bridge_records", [])),
        "profiles": {
            int(index): dict(profile)
            for index, profile in payload.get("radial_shell_profiles", {}).items()
        },
        "completed_radial_indices": set(
            int(index) for index in payload.get("completed_radial_indices", [])
        ),
        "reference_records": {
            int(record["point_index"]): dict(record)
            for record in payload.get("matched_reference_records", [])
        },
        "reference_attempt_records": list(
            payload.get("matched_reference_attempt_records", [])
        ),
    }


def shell_checkpoint_metadata(
    *,
    records: dict[int, dict[str, Any]],
    profiles: dict[int, dict[str, np.ndarray]],
    completed_radial_indices: set[int],
) -> dict[str, Any]:
    """Return durable progress metadata for a partial shell scan."""
    return {
        "checkpoint_granularity": "completed_cell_with_profile",
        "completed_cell_count": int(len(records)),
        "successful_profile_count": int(len(profiles)),
        "completed_radial_indices": np.asarray(
            sorted(int(index) for index in completed_radial_indices), dtype=int
        ),
        "partial_active_shell_profiles_saved": True,
    }


def save_payload_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one readable numerical checkpoint."""
    benchmark.atomic_save(Path(path), payload)


def collect_stage_futures(
    future_to_point,
    *,
    outputs_by_point: dict[int, list[dict[str, Any]]],
    finalize_successes: bool,
    finalize_all: bool,
    on_terminal,
) -> list[dict[str, Any]]:
    """Consume one solver stage and immediately publish terminal cells."""
    unresolved: list[dict[str, Any]] = []
    for future in as_completed(future_to_point):
        point = future_to_point[future]
        point_outputs = outputs_by_point[int(point["point_index"])]
        point_outputs.append(future.result())
        success = bool(point_outputs[-1]["selected_record"].get("success"))
        if bool(finalize_all) or (bool(finalize_successes) and success):
            on_terminal(point, point_outputs)
        else:
            unresolved.append(point)
    return unresolved


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _args_dict(args: argparse.Namespace) -> dict[str, Any]:
    return dict(vars(args))


def _transform_payload(transform: Any) -> dict[str, float]:
    if isinstance(transform, dict):
        return dict(transform)
    return {
        "density_origin_over_n0": float(transform.density_origin_over_n0),
        "density_scale_over_n0": float(transform.density_scale_over_n0),
        "temperature_scale_MeV": float(transform.temperature_scale_MeV),
    }


def build_domain_payload(
    *,
    args: argparse.Namespace,
    calibration: dict[str, Any],
    points: list[dict[str, Any]],
    T_values: tuple[float, ...],
    boundary_T_0plus: float,
    transform: Any,
    semi_boundary: dict[str, Any],
    geometry: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    """Return the immutable point manifest shared by all three methods."""
    return {
        "manifest_type": benchmark.DOMAIN_MANIFEST_TYPE,
        "selection_rule": "semi_analytic_safe_domain",
        "boundary_root_method": DIRECT_BOUNDARY_METHOD,
        "outer_boundary_method": "direct_ns_0plus_equals_zero",
        "outer_boundary_closure": "none_boundary_is_thermodynamic",
        "outer_boundary_interface_control": "exact_nK_over_nB",
        "inner_boundary_definition": (
            "muK_0plus_equals_zero_with_P_and_h_over_nB_matching"
        ),
        "outer_boundary_definition": "ns_0plus_equals_zero_equivalent_to_A_equals_one",
        "outer_boundary_temperature_strategy": "cold_T_0plus_equals_1e-3_MeV",
        "run_tag": str(args.run_tag),
        "created_at_utc": started_at,
        "arguments": _args_dict(args),
        "calibration": dict(calibration),
        "xi": float(args.xi),
        "ms_MeV": float(args.ms),
        "NM_type": str(args.nm_type),
        "upB": int(args.upB),
        "density_match_rtol": float(args.density_match_rtol),
        "T_0plus_values_MeV": np.asarray(T_values, dtype=float),
        "boundary_reference_T_0plus_MeV": float(boundary_T_0plus),
        "A_boundary_source": "direct_ns_0plus_equals_zero",
        "A_boundary_T0plus_MeV": DIRECT_A_T_0PLUS_MEV,
        "A_boundary_radial_probes": 0,
        "A_boundary_bisection_steps": 0,
        "A_boundary_residual_tolerance": DIRECT_BOUNDARY_ROOT_TOLERANCE,
        "enthalpy_boundary_definition": (
            "muK_0plus_equals_zero_with_P_and_h_over_nB_matching"
        ),
        "transform": _transform_payload(transform),
        "points": list(points),
        "boundary_diagnostics": {
            "method": DIRECT_BOUNDARY_METHOD,
            "velocity_closure": "none_boundary_is_thermodynamic",
            "semi_analytic": semi_boundary,
            "selected_geometry": geometry,
        },
    }


def _base_method_payload(
    *,
    method: str,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    started_at: str,
    run_complete: bool,
) -> dict[str, Any]:
    benchmark.validate_domain_manifest(manifest)
    points = list(manifest["points"])
    T_values = tuple(float(value) for value in manifest["T_0plus_values_MeV"])
    states = [_states_for_point(point, T_values) for point in points]
    theta_indices = sorted({int(point["theta_index"]) for point in points})
    radial_indices = sorted({int(point["radial_index"]) for point in points})
    contour_shape = (len(theta_indices), len(radial_indices), len(T_values))
    if len(points) != contour_shape[0] * contour_shape[1]:
        raise ValueError("domain points do not form a complete theta/radial grid")
    payload = {
        "run_tag": str(args.run_tag),
        "method": str(method),
        "script": Path(sys.argv[0]).name,
        "started_at_utc": started_at,
        "checkpoint_at_utc": _utc_now(),
        "finished_at_utc": _utc_now() if run_complete else None,
        "run_complete": bool(run_complete),
        "arguments": _args_dict(args),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "domain_manifest_type": manifest["manifest_type"],
        "domain_selection_rule": manifest["selection_rule"],
        "boundary_reference_T_0plus_MeV": float(
            manifest["boundary_reference_T_0plus_MeV"]
        ),
        "boundary_diagnostics": manifest["boundary_diagnostics"],
        "calibration": manifest["calibration"],
        "T_0plus_values_MeV": np.asarray(
            manifest["T_0plus_values_MeV"], dtype=float
        ),
        "points": points,
        "point_index": np.asarray([point["point_index"] for point in points], dtype=int),
        "T_0minus_MeV": np.asarray(
            [[state["T_0minus_MeV"] for state in row] for row in states],
            dtype=float,
        ),
        "nB_0minus_over_n0": np.asarray(
            [[state["nB_0minus_over_n0"] for state in row] for row in states],
            dtype=float,
        ),
        "ray_rho": np.asarray(
            [[state.get("rho", np.nan) for state in row] for row in states],
            dtype=float,
        ),
        "ray_rho_inner_boundary": np.asarray(
            [
                [state.get("rho_inner_boundary", np.nan) for state in row]
                for row in states
            ],
            dtype=float,
        ),
        "ray_rho_outer_boundary": np.asarray(
            [
                [state.get("rho_outer_boundary", np.nan) for state in row]
                for row in states
            ],
            dtype=float,
        ),
        "ray_theta_rad": np.asarray([point["theta_rad"] for point in points], dtype=float),
        "ray_radial_fraction": np.asarray(
            [point["radial_fraction"] for point in points], dtype=float
        ),
        "contour_grid_shape": contour_shape,
        "contour_theta_axis_rad": np.asarray(
            [
                next(
                    point["theta_rad"]
                    for point in points
                    if int(point["theta_index"]) == theta_index
                )
                for theta_index in theta_indices
            ],
            dtype=float,
        ),
        "contour_radial_fraction_axis": np.asarray(
            [
                next(
                    point["radial_fraction"]
                    for point in points
                    if int(point["radial_index"]) == radial_index
                )
                for radial_index in radial_indices
            ],
            dtype=float,
        ),
    }
    for field in (
        "T_0minus_MeV",
        "nB_0minus_over_n0",
        "ray_rho",
        "ray_rho_inner_boundary",
        "ray_rho_outer_boundary",
    ):
        payload[f"{field}_grid"] = payload[field].reshape(contour_shape)
    return payload


def build_analytical_payload(
    *,
    method: str,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    started_at: str,
    run_complete: bool = False,
) -> dict[str, Any]:
    if method not in {"analytic", "semi_analytic"}:
        raise ValueError("analytical payload method is invalid")
    payload = _base_method_payload(
        method=method,
        args=args,
        manifest=manifest,
        started_at=started_at,
        run_complete=run_complete,
    )
    n_points = len(manifest["points"])
    T_values = tuple(float(value) for value in manifest["T_0plus_values_MeV"])
    shape = (n_points, len(T_values))
    T_index = {value: index for index, value in enumerate(T_values)}
    fields = (
        "u_0minus",
        "jB",
        "a_0plus",
        "a_inf",
        "lambda_n",
        "T_inf_MeV",
        "A_boundary",
        "runtime_s",
        "muB_0minus_MeV",
        "density_relative_error",
    )
    for field in fields:
        payload[field] = np.full(shape, np.nan)
    payload["success"] = np.zeros(shape, dtype=bool)
    payload["message"] = np.full(shape, "", dtype=object)
    for row in rows:
        i = int(row["point_index"])
        j = T_index.get(float(row["T_0plus_MeV"]))
        if j is None or not 0 <= i < n_points:
            continue
        payload["success"][i, j] = bool(row.get("success", False))
        payload["message"][i, j] = str(row.get("message", ""))
        for field in fields:
            payload[field][i, j] = benchmark._as_finite_float(row.get(field))
    payload["u_0minus_m_s"] = payload["u_0minus"] * const.m_per_sec
    contour_shape = tuple(payload["contour_grid_shape"])
    for field in (*fields, "u_0minus_m_s", "success", "message"):
        payload[f"{field}_grid"] = payload[field].reshape(contour_shape)
    payload["records"] = list(rows)
    payload["completed_cell_count"] = len(rows)
    return payload


def semi_seed_map(rows: list[dict[str, Any]]) -> dict[int, dict[float, float]]:
    """Extract finite successful jB seeds by upstream point and T(0+)."""
    seeds: dict[int, dict[float, float]] = {}
    for row in rows:
        jB = benchmark._as_finite_float(row.get("jB"))
        if not bool(row.get("success", False)) or not np.isfinite(jB):
            continue
        seeds.setdefault(int(row["point_index"]), {})[
            float(row["T_0plus_MeV"])
        ] = float(jB)
    return seeds


def build_numerical_payload(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    seed_rows: list[dict[str, Any]],
    downward_records: list[dict[str, Any]],
    downward_profiles: dict[int, dict[float, dict[str, np.ndarray]]],
    upward_records: list[dict[str, Any]],
    upward_profiles: dict[int, dict[float, dict[str, np.ndarray]]],
    started_at: str,
    run_complete: bool = False,
) -> dict[str, Any]:
    payload = _base_method_payload(
        method="numerical",
        args=args,
        manifest=manifest,
        started_at=started_at,
        run_complete=run_complete,
    )
    n_points = len(manifest["points"])
    T_values = tuple(float(value) for value in manifest["T_0plus_values_MeV"])
    for sweep, records in (
        ("downward", downward_records),
        ("upward", upward_records),
    ):
        arrays = benchmark._method_arrays([], records, n_points, T_values, sweep)
        payload.update(
            {
                key: value
                for key, value in arrays.items()
                if key.startswith(f"numerical_{sweep}_")
            }
        )
        payload[f"numerical_{sweep}_u_0minus_m_s"] = (
            payload[f"numerical_{sweep}_u_0minus"] * const.m_per_sec
        )
    down = payload["numerical_downward_u_0minus"]
    up = payload["numerical_upward_u_0minus"]
    both = (
        payload["numerical_downward_success"]
        & payload["numerical_upward_success"]
    )
    relative = np.full(down.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        candidate = np.abs(up - down) / np.maximum(np.abs(up), np.abs(down))
    relative[both] = candidate[both]
    payload.update(
        {
            "solver_variant": benchmark.SOLVER_VARIANT,
            "root_method": benchmark.ROOT_METHOD,
            "composition_definition": benchmark.COMPOSITION_DEFINITION,
            "current_definition": benchmark.CURRENT_DEFINITION,
            "semi_analytic_seed_records": list(seed_rows),
            "numerical_downward_records": list(downward_records),
            "numerical_upward_records": list(upward_records),
            "numerical_downward_profiles": downward_profiles,
            "numerical_upward_profiles": upward_profiles,
            "profile_fields": list(benchmark.PROFILE_FIELDS),
            "hysteresis_relative_difference": relative,
            "hysteresis_compared_count": int(np.count_nonzero(both)),
            "hysteresis_max_relative_difference": (
                float(np.nanmax(relative)) if np.any(both) else np.nan
            ),
            "completed_downward_point_count": int(len(downward_profiles)),
            "completed_upward_point_count": int(len(upward_profiles)),
        }
    )
    contour_shape = tuple(payload["contour_grid_shape"])
    for sweep in ("downward", "upward"):
        for field in (
            "u_0minus",
            "u_0minus_m_s",
            "jB",
            "success",
            "status",
            "message",
            "guess_source",
        ):
            key = f"numerical_{sweep}_{field}"
            payload[f"{key}_grid"] = payload[key].reshape(contour_shape)
    return payload


def build_radial_numerical_payload(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    seed_rows: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    attempt_records: list[dict[str, Any]],
    bridge_records: list[dict[str, Any]],
    profiles: dict[int, dict[str, np.ndarray]],
    completed_radial_indices: set[int],
    reference_records: list[dict[str, Any]],
    reference_attempt_records: list[dict[str, Any]],
    started_at: str,
    run_complete: bool = False,
) -> dict[str, Any]:
    """Build a compatible numerical payload with outward-continuation aliases."""
    T_0plus = float(np.asarray(manifest["T_0plus_values_MeV"], dtype=float)[0])
    compatible_profiles = {
        int(point_index): {T_0plus: profile}
        for point_index, profile in profiles.items()
    }
    payload = build_numerical_payload(
        args=args,
        manifest=manifest,
        seed_rows=seed_rows,
        downward_records=target_records,
        downward_profiles=compatible_profiles,
        upward_records=[],
        upward_profiles={},
        started_at=started_at,
        run_complete=run_complete,
    )
    for key, value in tuple(payload.items()):
        if key.startswith("numerical_downward_"):
            payload[key.replace("numerical_downward_", "numerical_outward_", 1)] = value
    payload.update(
        {
            "continuation_coordinate": "radial_fraction_from_Delta_h_zero_to_A_one",
            "continuation_direction": "outward_in_rho",
            "continuation_primary_seed": radial.GUESS_SOURCE_PROFILE_EXTRAPOLATED,
            "continuation_fallback_seed": "strict_immediate_theta_wavefront",
            "continuation_stage_order": CONTINUATION_STAGE_ORDER,
            "velocity_seed_model": "matched_T0plus1_then_numerical_propagation",
            "matched_reference_T_0plus_MeV": float(args.reference_T0plus),
            "matched_reference_jB_scale": float(args.reference_jB_scale),
            "matched_reference_records": list(reference_records),
            "matched_reference_attempt_records": list(reference_attempt_records),
            "radial_attempt_records": list(attempt_records),
            "radial_bridge_records": list(bridge_records),
            "radial_continuation_profiles": dict(profiles),
            "radial_shell_records": list(target_records),
            "radial_shell_attempt_records": list(attempt_records),
            "radial_shell_bridge_records": list(bridge_records),
            "radial_shell_profiles": dict(profiles),
            "completed_radial_indices": np.asarray(
                sorted(int(index) for index in completed_radial_indices), dtype=int
            ),
            "completed_radial_count": int(len(completed_radial_indices)),
            "completed_cell_count": int(len(target_records)),
            "successful_profile_count": int(len(profiles)),
            "bridge_attempt_count": int(len(bridge_records)),
            "checkpoint_granularity": "completed_cell_with_profile",
            "partial_active_shell_profiles_saved": True,
        }
    )
    return payload


def collect_radial_results(
    futures,
    progress_queue,
    *,
    total_targets: int,
    completed_callback=None,
    target_completed_callback=None,
) -> list[dict[str, Any]]:
    """Stream one fixed-size cell bar while reporting retry and bridge counters."""
    pending = set(futures)
    outputs: list[dict[str, Any]] = []
    attempts = retries = bridges = 0
    with tqdm(
        total=int(total_targets),
        desc="Numerical outward radial continuation",
        unit="cell",
        dynamic_ncols=True,
    ) as progress:
        def handle_event(event):
            nonlocal attempts, retries, bridges
            kind = event.get("event")
            if kind == "attempt_completed":
                attempts += 1
            elif kind == "retry_scheduled":
                retries += 1
            elif kind == "bridge_scheduled":
                bridges += 1
            elif kind == "target_completed":
                progress.update(1)
                progress.set_postfix_str(
                    f"theta={event.get('theta_index')}, "
                    f"s={event.get('radial_fraction', np.nan):.3f}, "
                    f"{'ok' if event.get('success') else 'failed'}, "
                    f"attempts={attempts}, retries={retries}, bridges={bridges}"
                )
                if target_completed_callback is not None:
                    target_completed_callback(dict(event))

        while pending:
            try:
                event = progress_queue.get(timeout=0.25)
            except queue.Empty:
                event = None
            if event is not None:
                handle_event(event)
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
            handle_event(event)
        progress.refresh()
    return outputs


def _input_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else SCRIPT_DIR / path


def _load_manifest(value: str | Path) -> dict[str, Any]:
    path = _input_path(value)
    if not path.exists():
        raise FileNotFoundError(
            f"shared domain manifest not found: {path}; run "
            "prepare_T0plus_contour_domain.py first"
        )
    return benchmark.validate_domain_manifest(
        np.load(path, allow_pickle=True).item()
    )


def _calibration(args: argparse.Namespace) -> dict[str, float]:
    n0 = float(0.16 * const.MeV_fm**3)
    if np.isfinite(args.b14) and float(args.b14) > 0.0:
        B14, muB_crit0, P_crit0 = float(args.b14), np.nan, np.nan
    else:
        B14, muB_crit0, P_crit0 = benchmark.ray_grid.calibrate_b14(
            float(args.ncrit_over_n0) * n0,
            float(args.ms),
            str(args.nm_type),
        )
    return {
        "B14_MeV": float(B14),
        "muB_crit0_MeV": float(muB_crit0),
        "P_crit0_MeV4": float(P_crit0),
        "n0_MeV3": n0,
        "ncrit_over_n0": float(args.ncrit_over_n0),
    }


def _prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the shared domain with direct coupled interface roots."
    )
    parser.add_argument("--output", default=DOMAIN_FILENAME)
    parser.add_argument("--run-tag", default="T0plus_contour_domain_v1")
    parser.add_argument("--workers", type=int, default=benchmark.default_workers())
    parser.add_argument(
        "--T0plus-values",
        type=benchmark._parse_float_list,
        default=benchmark.DEFAULT_T_0PLUS_VALUES_MEV,
        help="descending interface-temperature list in MeV",
    )
    parser.add_argument("--ray-count", type=int, default=benchmark.DEFAULT_RAY_COUNT)
    parser.add_argument(
        "--radial-fractions",
        type=benchmark._parse_float_list,
        default=benchmark.DEFAULT_RADIAL_FRACTIONS,
    )
    parser.add_argument(
        "--branch-check-radial-fraction",
        type=float,
        default=benchmark.DEFAULT_BRANCH_CHECK_RADIAL_FRACTION,
    )
    parser.add_argument(
        "--angular-buffer-fraction",
        type=float,
        default=benchmark.DEFAULT_ANGULAR_BUFFER_FRACTION,
    )
    parser.add_argument(
        "--radial-buffer-fraction",
        type=float,
        default=benchmark.DEFAULT_RADIAL_BUFFER_FRACTION,
    )
    parser.add_argument("--xi", type=float, default=benchmark.DEFAULT_XI)
    parser.add_argument("--ms", type=float, default=benchmark.DEFAULT_MS_MEV)
    parser.add_argument("--nm-type", default=benchmark.DEFAULT_NM_TYPE)
    parser.add_argument("--upB", type=int, default=benchmark.DEFAULT_UPB)
    parser.add_argument(
        "--ncrit-over-n0", type=float, default=benchmark.DEFAULT_NCRIT_OVER_N0
    )
    parser.add_argument("--b14", type=float, default=np.nan)
    parser.add_argument(
        "--density-min-over-n0",
        type=float,
        default=benchmark.DEFAULT_DENSITY_MIN_OVER_N0,
    )
    parser.add_argument(
        "--density-max-over-n0",
        type=float,
        default=benchmark.DEFAULT_DENSITY_MAX_OVER_N0,
    )
    parser.add_argument(
        "--temperature-min",
        type=float,
        default=benchmark.DEFAULT_TEMPERATURE_MIN_MEV,
    )
    parser.add_argument(
        "--temperature-max",
        type=float,
        default=benchmark.DEFAULT_TEMPERATURE_MAX_MEV,
    )
    parser.add_argument(
        "--boundary-theta-count",
        type=int,
        default=benchmark.DEFAULT_BOUNDARY_THETA_COUNT,
    )
    parser.add_argument(
        "--boundary-root-tolerance",
        type=float,
        default=DIRECT_BOUNDARY_ROOT_TOLERANCE,
    )
    parser.add_argument(
        "--boundary-max-nfev",
        type=int,
        default=DIRECT_BOUNDARY_MAX_NFEV,
    )
    parser.add_argument(
        "--density-match-rtol",
        type=float,
        default=benchmark.DEFAULT_DENSITY_MATCH_RTOL,
    )
    parser.set_defaults(boundary_T0plus=DIRECT_A_T_0PLUS_MEV)
    return parser


def _validate_prepare_args(args: argparse.Namespace) -> None:
    T_values = tuple(float(value) for value in args.T0plus_values)
    if not T_values or not all(np.isfinite(value) and value > 0.0 for value in T_values):
        raise ValueError("T_0plus values must be positive and finite")
    if any(left <= right for left, right in zip(T_values[:-1], T_values[1:])):
        raise ValueError("T_0plus values must be strictly descending")
    fractions = tuple(float(value) for value in args.radial_fractions)
    if not fractions or not all(0.0 < value < 1.0 for value in fractions):
        raise ValueError("radial fractions must lie strictly inside (0, 1)")
    if not any(
        np.isclose(value, float(args.branch_check_radial_fraction), atol=1.0e-12)
        for value in fractions
    ):
        raise ValueError("branch-check radial fraction must be in radial fractions")
    if int(args.ray_count) < 1:
        raise ValueError("contour ray count must be positive")
    if int(args.boundary_theta_count) < 3:
        raise ValueError("boundary angles must include both axes and an interior ray")
    if float(args.boundary_root_tolerance) <= 0.0 or int(args.boundary_max_nfev) < 1:
        raise ValueError("direct boundary root controls are invalid")
    if not float(args.density_min_over_n0) < float(args.density_max_over_n0):
        raise ValueError("density window is empty")
    if not 0.0 < float(args.temperature_min) < float(args.temperature_max):
        raise ValueError("temperature window must satisfy 0 < min < max")
    for name in ("angular_buffer_fraction", "radial_buffer_fraction"):
        value = float(getattr(args, name))
        if not 0.0 <= value < 0.5:
            raise ValueError(f"{name} must lie in [0, 0.5)")
    if int(args.workers) < 1 or int(args.upB) < 1:
        raise ValueError("workers and upB must be positive")
    if float(args.density_match_rtol) <= 0.0:
        raise ValueError("density matching tolerance must be positive")


def _method_parser(method: str) -> argparse.ArgumentParser:
    output = {
        "analytic": FULL_ANALYTIC_FILENAME,
        "semi_analytic": SEMI_ANALYTIC_FILENAME,
        "numerical": NUMERICAL_FILENAME,
    }[method]
    parser = argparse.ArgumentParser(
        description=f"Run the {method.replace('_', ' ')} T(0+) contour scan."
    )
    parser.add_argument("--domain-file", default=DOMAIN_FILENAME)
    parser.add_argument("--output", default=output)
    parser.add_argument("--run-tag", default=f"T0plus_{method}_contour_v1")
    parser.add_argument("--workers", type=int, default=benchmark.default_workers())
    parser.add_argument("--save-every", type=int, default=benchmark.DEFAULT_SAVE_EVERY)
    if method == "numerical":
        parser.add_argument("--tail-eps", type=float, default=benchmark.DEFAULT_TAIL_EPS)
        parser.add_argument("--n-mesh", type=int, default=benchmark.DEFAULT_N_MESH)
        parser.add_argument("--tol-bvp", type=float, default=benchmark.DEFAULT_TOL_BVP)
        parser.add_argument("--max-nodes", type=int, default=benchmark.DEFAULT_MAX_NODES)
        parser.add_argument(
            "--jB-lower-bound", type=float, default=benchmark.DEFAULT_JB_LOWER_BOUND
        )
        parser.add_argument(
            "--jB-upper-factor", type=float, default=benchmark.DEFAULT_JB_UPPER_FACTOR
        )
        parser.add_argument(
            "--point-timeout",
            type=float,
            default=benchmark.DEFAULT_POINT_TIMEOUT_S,
            help="hard wall-clock limit for one disposable BVP trial",
        )
        parser.add_argument(
            "--target-timeout", type=float, default=1800.0,
            help="hard wall-clock budget for all trials and bridges of one cell",
        )
        parser.add_argument(
            "--reference-T0plus",
            type=float,
            default=1.0,
            help="matched-grid interface temperature used to bootstrap the first shell",
        )
        parser.add_argument(
            "--reference-jB-scale",
            type=float,
            default=1.18,
            help="scale from the matched reference current to the target current guess",
        )
        parser.add_argument("--max-bridge-depth", type=int, default=5)
        parser.add_argument(
            "--min-bridge-fraction-step", type=float, default=1.0e-3
        )
        parser.add_argument(
            "--branch-check-radial-fraction",
            type=float,
            default=benchmark.DEFAULT_BRANCH_CHECK_RADIAL_FRACTION,
        )
        parser.add_argument("--skip-branch-check", action="store_true")
        parser.add_argument(
            "--restart",
            action="store_true",
            help="start a new suffixed output instead of resuming an incomplete checkpoint",
        )
    return parser


def _validate_method_args(args: argparse.Namespace, numerical: bool = False) -> None:
    if int(args.workers) < 1:
        raise ValueError("workers must be positive")
    if int(args.save_every) < 0:
        raise ValueError("save frequency cannot be negative")
    if numerical:
        if int(args.n_mesh) < 5 or int(args.max_nodes) <= int(args.n_mesh):
            raise ValueError("numerical mesh settings are invalid")
        if (
            float(args.tol_bvp) <= 0.0
            or float(args.point_timeout) <= 0.0
            or float(args.target_timeout) <= 0.0
        ):
            raise ValueError("numerical tolerance and timeout must be positive")
        if float(args.jB_lower_bound) <= 0.0 or float(args.jB_upper_factor) <= 0.0:
            raise ValueError("jB bounds must be positive")
        if float(args.reference_T0plus) <= 0.0:
            raise ValueError("reference T(0+) must be positive")
        if float(args.reference_jB_scale) <= 0.0:
            raise ValueError("reference jB scale must be positive")
        if int(args.max_bridge_depth) < 0:
            raise ValueError("max bridge depth cannot be negative")
        if float(args.min_bridge_fraction_step) <= 0.0:
            raise ValueError("minimum bridge step must be positive")


def _task_common(manifest: dict[str, Any]) -> dict[str, Any]:
    calibration = manifest["calibration"]
    return {
        "B14_MeV": float(calibration["B14_MeV"]),
        "xi": float(manifest["xi"]),
        "ms_MeV": float(manifest["ms_MeV"]),
        "nm_type": str(manifest["NM_type"]),
        "upB": int(manifest["upB"]),
        "density_match_rtol": float(manifest["density_match_rtol"]),
    }


def _states_for_point(
    point: dict[str, Any], T_values: tuple[float, ...]
) -> list[dict[str, Any]]:
    states = [dict(state) for state in point.get("states", [])]
    if states:
        by_T = {float(state["T_0plus_MeV"]): state for state in states}
        missing = [value for value in T_values if float(value) not in by_T]
        if missing:
            raise ValueError(
                f"point {point.get('point_index')} is missing T(0+) states {missing}"
            )
        return [by_T[float(value)] for value in T_values]
    return [
        {
            "T_0plus_MeV": float(value),
            "T_0minus_MeV": float(point["T_0minus_MeV"]),
            "nB_0minus_over_n0": float(point["nB_0minus_over_n0"]),
            "rho": float(point.get("rho", np.nan)),
            "rho_inner_boundary": float(point.get("rho_inner_boundary", np.nan)),
            "rho_outer_boundary": float(point.get("rho_outer_boundary", np.nan)),
        }
        for value in T_values
    ]


def _solve_direct_boundary_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run one independent ray root without letting one bad ray abort the scan."""
    try:
        record = _solve_direct_interface_root(task)
        return {
            "success": True,
            "task": task,
            "completed": {"record": record},
            "message": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "task": task,
            "completed": None,
            "message": str(exc),
        }


def _direct_boundary_rho_limit(task: dict[str, Any]) -> float:
    transform = task["transform"]
    theta = float(task["theta_rad"])
    cosine = float(np.cos(theta))
    sine = float(np.sin(theta))
    limits = []
    if cosine > 1.0e-12:
        limits.append(
            (
                float(task["density_max_over_n0"])
                - float(transform.density_origin_over_n0)
            )
            / (float(transform.density_scale_over_n0) * cosine)
        )
    if sine > 1.0e-12:
        limits.append(
            float(task["temperature_max_MeV"])
            / (float(transform.temperature_scale_MeV) * sine)
        )
    if not limits:
        raise RuntimeError("direct boundary ray has no finite physical extent")
    limit = float(min(limits))
    if not np.isfinite(limit) or limit <= 0.0:
        raise RuntimeError("direct boundary ray does not intersect the physical window")
    return limit


def _direct_upstream_state(
    task: dict[str, Any], rho: float
) -> tuple[float, float, float, float]:
    if str(task["nm_type"]) != "PNM":
        raise ValueError("the direct FL26-3 boundary construction supports PNM only")
    nB_over_n0, T_0minus = task["transform"].to_physical(
        float(rho), float(task["theta_rad"])
    )
    nB_0minus = float(nB_over_n0) * float(task["n0"])
    phase_velocity = benchmark.phase_velocity
    P_0minus = float(
        phase_velocity.PNM_n(nB_0minus, float(T_0minus), NM_type="PNM")
    )
    h_0minus = float(phase_velocity.hNM_n(nB_0minus, float(T_0minus)))
    values = np.asarray(
        [nB_over_n0, T_0minus, nB_0minus, P_0minus, h_0minus], dtype=float
    )
    if not np.all(np.isfinite(values)) or nB_0minus <= 0.0 or h_0minus <= 0.0:
        raise RuntimeError("direct boundary upstream state is not physical")
    return float(nB_over_n0), float(T_0minus), P_0minus, h_0minus / nB_0minus


def _direct_matching_residuals(
    task: dict[str, Any],
    *,
    rho: float,
    muB_0plus: float,
    muK_0plus: float,
) -> np.ndarray:
    _, _, P_0minus, h_over_nB_0minus = _direct_upstream_state(task, rho)
    phase_velocity = benchmark.phase_velocity
    T_0plus = float(task["T_0plus_MeV"])
    B14 = float(task["B14"])
    ms = float(task["ms"])
    upB = int(task["upB"])
    P_0plus = float(
        phase_velocity.PQM(muB_0plus, muK_0plus, B14, T_0plus, ms=ms, upB=upB)
    )
    h_0plus = float(
        phase_velocity.hQM(muB_0plus, muK_0plus, B14, T_0plus, ms=ms, upB=upB)
    )
    nB_0plus = float(
        phase_velocity.nB_QM(
            muB_0plus, muK_0plus, B14, T_0plus, ms=ms, upB=upB
        )
    )
    if not np.all(np.isfinite([P_0plus, h_0plus, nB_0plus])) or nB_0plus <= 0.0:
        return np.asarray([1.0e6, 1.0e6], dtype=float)
    return np.asarray(
        [
            (P_0plus - P_0minus) / max(abs(P_0minus), 1.0),
            (h_0plus / nB_0plus - h_over_nB_0minus)
            / max(abs(h_over_nB_0minus), 1.0),
        ],
        dtype=float,
    )


def _direct_quark_outputs(
    task: dict[str, Any], *, muB_0plus: float, muK_0plus: float
) -> dict[str, float]:
    state = benchmark.phase_velocity._quark_uds_state(
        float(muB_0plus),
        float(muK_0plus),
        float(task["T_0plus_MeV"]),
        ms=float(task["ms"]),
        upB=int(task["upB"]),
    )
    nB_0plus = float(state["nB"])
    nK_0plus = float(state["nK"])
    n_s_0plus = float(state["species"]["n_s"])
    return {
        "mu_s_0plus_MeV": float(state["chemical_potentials"]["mu_s"]),
        "n_s_0plus": n_s_0plus,
        "nB_0plus": nB_0plus,
        "nK_0plus": nK_0plus,
        "n_s_over_nB": n_s_0plus / nB_0plus,
        "a_0plus": nK_0plus / nB_0plus,
    }


def _solve_direct_interface_root(task: dict[str, Any]) -> dict[str, Any]:
    """Mirror FL26-3: solve the interface equations and boundary condition together."""
    condition = str(task["condition"])
    rho_limit = _direct_boundary_rho_limit(task)
    muB_hint = float(task.get("muB_hint", 1000.0)) / 1000.0
    if condition == "muK":
        def residual(vector):
            return _direct_matching_residuals(
                task,
                rho=float(vector[0]),
                muB_0plus=1000.0 * float(vector[1]),
                muK_0plus=0.0,
            )

        seeds = (
            (min(0.5, 0.8 * rho_limit), muB_hint),
            (min(0.8, 0.8 * rho_limit), 1.0),
        )
        bounds = ([0.0, 0.3], [rho_limit, 3.0])
    elif condition == "ns":
        def residual(vector):
            muB_0plus = 1000.0 * float(vector[1])
            muK_0plus = 100.0 * float(vector[2])
            matching = _direct_matching_residuals(
                task,
                rho=float(vector[0]),
                muB_0plus=muB_0plus,
                muK_0plus=muK_0plus,
            )
            outputs = _direct_quark_outputs(
                task, muB_0plus=muB_0plus, muK_0plus=muK_0plus
            )
            return np.r_[matching, outputs["mu_s_0plus_MeV"] / 1000.0]

        seeds = (
            (min(0.8, 0.8 * rho_limit), 1.0, 5.5),
            (min(0.55, 0.8 * rho_limit), muB_hint, 5.5),
        )
        bounds = ([0.0, 0.3, 0.0], [rho_limit, 3.0, 20.0])
    else:
        raise ValueError("direct boundary condition must be 'muK' or 'ns'")

    candidates = []
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    for seed in seeds:
        clipped = np.clip(np.asarray(seed, dtype=float), lower + 1.0e-8, upper - 1.0e-8)
        solution = least_squares(
            residual,
            clipped,
            bounds=bounds,
            diff_step=1.0e-4,
            xtol=1.0e-11,
            ftol=1.0e-11,
            gtol=1.0e-11,
            max_nfev=int(task.get("max_nfev", DIRECT_BOUNDARY_MAX_NFEV)),
        )
        residual_norm = float(np.linalg.norm(residual(solution.x), ord=np.inf))
        if np.all(np.isfinite(solution.x)) and np.isfinite(residual_norm):
            candidates.append((residual_norm, int(solution.nfev), solution.x.copy()))
    if not candidates:
        raise RuntimeError(f"{condition} direct boundary root returned no finite candidate")
    residual_norm, evaluation_count, solved = min(candidates, key=lambda item: item[0])
    tolerance = float(task.get("root_tolerance", DIRECT_BOUNDARY_ROOT_TOLERANCE))
    if residual_norm > tolerance:
        raise RuntimeError(
            f"{condition} direct boundary root failed at theta={task['theta_rad']:.6g}; "
            f"residual={residual_norm:.3e}"
        )

    rho = float(solved[0])
    muB_0plus = 1000.0 * float(solved[1])
    muK_0plus = 0.0 if condition == "muK" else 100.0 * float(solved[2])
    nB_over_n0, T_0minus, _, _ = _direct_upstream_state(task, rho)
    outputs = _direct_quark_outputs(
        task, muB_0plus=muB_0plus, muK_0plus=muK_0plus
    )
    return {
        "condition": condition,
        "boundary_kind": (
            "Delta_h_equals_zero" if condition == "muK" else "A_equals_one"
        ),
        "root_method": DIRECT_BOUNDARY_METHOD,
        "coordinate": rho,
        "rho": rho,
        "theta_rad": float(task["theta_rad"]),
        "theta_deg": float(np.degrees(task["theta_rad"])),
        "nB_over_n0": nB_over_n0,
        "T_MeV": T_0minus,
        "T_0plus_MeV": float(task["T_0plus_MeV"]),
        "muB_0plus_MeV": muB_0plus,
        "muK_0plus_MeV": muK_0plus,
        **outputs,
        "A_boundary": float(outputs["a_0plus"]),
        "one_minus_A_residual": float(1.0 - outputs["a_0plus"]),
        "residual_norm": residual_norm,
        "evaluation_count": evaluation_count,
        "solver_vector": solved,
    }


def assemble_direct_common_boundary(
    *,
    theta_axis: np.ndarray,
    T_values: tuple[float, ...],
    task_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble fixed-T(0+) Delta-h curves and one cold A = 1 curve."""
    theta_axis = np.asarray(theta_axis, dtype=float)
    T_values = tuple(float(value) for value in T_values)
    gap_by_temperature: dict[float, dict[int, dict[str, Any]]] = {
        value: {} for value in T_values
    }
    A_by_theta: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for result in task_results:
        task = result["task"]
        if not result["success"]:
            failures.append(
                {
                    "theta_index": int(task["theta_index"]),
                    "theta_rad": float(task["theta_rad"]),
                    "T_0plus_MeV": task.get("T_0plus_MeV"),
                    "task_kind": str(task["task_kind"]),
                    "message": str(result["message"]),
                }
            )
            continue
        record = dict(result["completed"]["record"])
        record["rho"] = float(record["rho"])
        record["path_mode"] = "elliptical_ray_direct_coupled_root"
        record["theta_rad"] = float(task["theta_rad"])
        record["theta_deg"] = float(np.degrees(task["theta_rad"]))
        theta_index = int(task["theta_index"])
        if task["condition"] == "muK":
            T_0plus = float(task["T_0plus_MeV"])
            gap_by_temperature[T_0plus][theta_index] = record
        else:
            A_by_theta[theta_index] = record

    valid_indices = []
    for theta_index in range(theta_axis.size):
        A_boundary = A_by_theta.get(theta_index)
        gaps = [gap_by_temperature[value].get(theta_index) for value in T_values]
        if A_boundary is None or any(gap is None for gap in gaps):
            continue
        if any(float(A_boundary["rho"]) <= float(gap["rho"]) for gap in gaps):
            failures.append(
                {
                    "theta_index": theta_index,
                    "theta_rad": float(theta_axis[theta_index]),
                    "task_kind": "common_safe_interval",
                    "message": "common outer boundary does not exceed Delta h boundary",
                }
            )
            continue
        valid_indices.append(theta_index)

    if len(valid_indices) != theta_axis.size:
        raise RuntimeError(
            "direct boundary construction did not resolve every requested ray; "
            f"valid={len(valid_indices)}/{theta_axis.size}, failures={len(failures)}"
        )
    valid_theta = theta_axis[np.asarray(valid_indices, dtype=int)]
    gap_records_by_T = {
        value: [gap_by_temperature[value][index] for index in valid_indices]
        for value in T_values
    }
    A_records = [A_by_theta[index] for index in valid_indices]
    rho_inner_by_T = np.asarray(
        [
            [record["rho"] for record in gap_records_by_T[value]]
            for value in T_values
        ],
        dtype=float,
    )
    return {
        "construction": DIRECT_BOUNDARY_METHOD,
        "selection_rule": "direct_interface_matching_safe_domain",
        "A_boundary_T_0plus_MeV": DIRECT_A_T_0PLUS_MEV,
        "A_boundary_interface_fraction_mode": "exact_nK_over_nB",
        "A_boundary_velocity_closure": "none_boundary_is_thermodynamic",
        "A_boundary_source": "direct_ns_0plus_equals_zero",
        "A_boundary_reference_xi": None,
        "A_boundary_residual_tolerance": DIRECT_BOUNDARY_ROOT_TOLERANCE,
        "enthalpy_boundary_definition": (
            "muK_0plus_equals_zero_with_P_and_h_over_nB_matching"
        ),
        "T_0plus_values_MeV": np.asarray(T_values, dtype=float),
        "theta_axis_rad": valid_theta,
        "rho_inner": np.max(rho_inner_by_T, axis=0),
        "rho_inner_by_T0plus": rho_inner_by_T,
        "rho_outer": np.asarray(
            [record["rho"] for record in A_records], dtype=float
        ),
        "gap_records": gap_records_by_T,
        "A_records": A_records,
        "gap_curve_records_by_T0plus": gap_records_by_T,
        "A_curve_records": A_records,
        "failed_ray_tasks": failures,
        "endpoint_records": {
            "A_theta_0": A_records[0],
            "A_theta_90": A_records[-1],
            "gap_theta_0_by_T0plus": {
                value: records[0] for value, records in gap_records_by_T.items()
            },
            "gap_theta_90_by_T0plus": {
                value: records[-1] for value, records in gap_records_by_T.items()
            },
        },
    }


def solve_direct_common_boundary(
    *,
    transform,
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
    root_tolerance: float,
    max_nfev: int,
    T_values: tuple[float, ...],
    executor,
) -> dict[str, Any]:
    """Solve the FL26-3 coupled interface roots on every requested ray."""
    theta_count = int(theta_count)
    if theta_count < 3:
        raise ValueError("boundary construction needs two endpoints and an interior ray")
    theta_axis = np.linspace(0.0, 0.5 * np.pi, theta_count)
    tasks: list[dict[str, Any]] = []
    for theta_index, theta in enumerate(theta_axis):
        common = {
            "index": int(theta_index),
            "theta_index": int(theta_index),
            "theta_rad": float(theta),
            "transform": transform,
            "density_max_over_n0": float(density_max_over_n0),
            "temperature_max_MeV": float(temperature_max_MeV),
            "n0": float(n0),
            "B14": float(B14),
            "nm_type": str(nm_type),
            "ms": float(ms),
            "upB": int(upB),
            "root_tolerance": float(root_tolerance),
            "max_nfev": int(max_nfev),
            "muB_hint": 1000.0,
        }
        for T_0plus in T_values:
            tasks.append(
                {
                    **common,
                    "task_kind": "gap_ray",
                    "condition": "muK",
                    "label": f"gap_ray_theta_{theta_index}_T0plus_{T_0plus:g}",
                    "T_0plus_MeV": float(T_0plus),
                }
            )
        tasks.append(
            {
                **common,
                "task_kind": "A_ray",
                "condition": "ns",
                "label": f"A_ray_theta_{theta_index}_cold",
                "T_0plus_MeV": DIRECT_A_T_0PLUS_MEV,
            }
        )
    futures = {
        executor.submit(_solve_direct_boundary_task, task): task for task in tasks
    }
    results = []
    progress_keys = [("A", 0.0)] + [
        ("gap", float(T_0plus)) for T_0plus in T_values
    ]
    progress_bars = {
        key: tqdm(
            total=theta_count,
            desc=(
                "A=1 boundary: n_s(0+)=0 (cold)"
                if key[0] == "A"
                else f"T(0+)={key[1]:g} MeV: muK(0+)=0"
            ),
            unit="ray",
            position=position,
            leave=True,
            dynamic_ncols=True,
        )
        for position, key in enumerate(progress_keys)
    }
    try:
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            task = result["task"]
            progress_key = (
                ("A", 0.0)
                if task["condition"] == "ns"
                else ("gap", float(task["T_0plus_MeV"]))
            )
            progress = progress_bars[progress_key]
            progress.update(1)
            progress.set_postfix_str(
                f"theta={np.degrees(task['theta_rad']):.3g} deg, "
                f"{'ok' if result['success'] else 'failed'}"
            )
    finally:
        for progress in progress_bars.values():
            progress.close()
    try:
        return assemble_direct_common_boundary(
            theta_axis=theta_axis,
            T_values=tuple(float(value) for value in T_values),
            task_results=results,
        )
    except Exception as exc:
        failed = [result for result in results if not result["success"]]
        kind_counts = Counter(
            str(result["task"]["task_kind"]) for result in failed
        )
        message_counts = Counter(str(result["message"]) for result in failed)
        print(f"Boundary task failure counts: {dict(kind_counts)}", flush=True)
        for message, count in message_counts.most_common(8):
            print(f"  {count} x {message}", flush=True)
        raise RuntimeError(
            f"direct boundary assembly failed after {len(results)} completed tasks"
        ) from exc


def temperature_boundary_geometry(
    boundary: dict[str, Any],
    *,
    ray_count: int,
    angular_buffer_fraction: float,
) -> dict[str, Any]:
    """Interpolate all inner curves and the shared A curve onto benchmark rays."""
    source_theta = np.asarray(boundary["theta_axis_rad"], dtype=float)
    theta_axis = benchmark.interior_theta_axis(
        boundary,
        ray_count=int(ray_count),
        angular_buffer_fraction=float(angular_buffer_fraction),
    )
    T_values = np.asarray(boundary["T_0plus_values_MeV"], dtype=float)
    source_inner = np.asarray(boundary["rho_inner_by_T0plus"], dtype=float)
    rho_inner = np.asarray(
        [np.interp(theta_axis, source_theta, row) for row in source_inner],
        dtype=float,
    )
    rho_outer = np.interp(
        theta_axis,
        source_theta,
        np.asarray(boundary["rho_outer"], dtype=float),
    )
    if not np.all(np.isfinite(rho_inner)) or not np.all(np.isfinite(rho_outer)):
        raise RuntimeError("temperature-indexed boundary contains non-finite values")
    invalid = rho_inner >= rho_outer[None, :]
    if np.any(invalid):
        cells = [tuple(int(value) for value in index) for index in np.argwhere(invalid)]
        raise RuntimeError(f"temperature-indexed safe interval collapsed at {cells}")
    return {
        "selection_rule": "semi_analytic_safe_domain",
        "theta_axis_rad": theta_axis,
        "T_0plus_values_MeV": T_values,
        "rho_inner_by_T0plus": rho_inner,
        "rho_outer": rho_outer,
    }


def build_temperature_indexed_points(
    *,
    geometry: dict[str, Any],
    radial_fractions: np.ndarray,
    transform,
    radial_buffer_fraction: float,
) -> list[dict[str, Any]]:
    """Build ray trajectories whose physical upstream state varies with T(0+)."""
    theta_axis = np.asarray(geometry["theta_axis_rad"], dtype=float)
    T_values = np.asarray(geometry["T_0plus_values_MeV"], dtype=float)
    rho_inner = np.asarray(geometry["rho_inner_by_T0plus"], dtype=float)
    rho_outer = np.asarray(geometry["rho_outer"], dtype=float)
    fractions = np.asarray(radial_fractions, dtype=float)
    buffer = float(radial_buffer_fraction)
    if not np.all((fractions > 0.0) & (fractions < 1.0)):
        raise ValueError("radial fractions must lie strictly inside (0, 1)")
    if not 0.0 <= buffer < 0.5:
        raise ValueError("radial buffer fraction must lie in [0, 0.5)")
    used = buffer + fractions * (1.0 - 2.0 * buffer)
    points: list[dict[str, Any]] = []
    for theta_index, theta in enumerate(theta_axis):
        for radial_index, (fraction, fraction_used) in enumerate(zip(fractions, used)):
            states = []
            for T_index, T_0plus in enumerate(T_values):
                inner = float(rho_inner[T_index, theta_index])
                outer = float(rho_outer[theta_index])
                rho = float(inner + fraction_used * (outer - inner))
                density, temperature = transform.to_physical(rho, float(theta))
                states.append(
                    {
                        "T_0plus_index": int(T_index),
                        "T_0plus_MeV": float(T_0plus),
                        "rho": rho,
                        "rho_inner_boundary": inner,
                        "rho_outer_boundary": outer,
                        "nB_0minus_over_n0": float(density),
                        "T_0minus_MeV": float(temperature),
                    }
                )
            representative = states[0]
            points.append(
                {
                    "point_index": len(points),
                    "trajectory_index": len(points),
                    "theta_index": int(theta_index),
                    "radial_index": int(radial_index),
                    "theta_rad": float(theta),
                    "theta_deg": float(np.degrees(theta)),
                    "radial_fraction": float(fraction),
                    "radial_fraction_used": float(fraction_used),
                    "states": states,
                    # Representative aliases keep older readers functional.
                    "rho": representative["rho"],
                    "rho_inner_boundary": representative["rho_inner_boundary"],
                    "rho_outer_boundary": representative["rho_outer_boundary"],
                    "rho_outer_boundary_semi_analytic": representative["rho_outer_boundary"],
                    "nB_0minus_over_n0": representative["nB_0minus_over_n0"],
                    "T_0minus_MeV": representative["T_0minus_MeV"],
                }
            )
    return points


def prepare_domain_main(argv: list[str] | None = None) -> int:
    args = _prepare_parser().parse_args(argv)
    _validate_prepare_args(args)
    started_at = _utc_now()
    output_path = benchmark.resolve_output_path(args.output)
    calibration = _calibration(args)
    T_values = tuple(float(value) for value in args.T0plus_values)
    boundary_T_0plus = float(args.boundary_T0plus)
    if not np.isfinite(boundary_T_0plus) or not np.isclose(
        boundary_T_0plus, DIRECT_A_T_0PLUS_MEV
    ):
        raise ValueError("the direct A = 1 boundary requires cold T_0plus = 1e-3 MeV")
    transform = benchmark.ray_grid.RayTransform(
        density_origin_over_n0=float(args.density_min_over_n0),
        density_scale_over_n0=float(
            args.density_max_over_n0 - args.density_min_over_n0
        ),
        temperature_scale_MeV=float(args.temperature_max),
    )
    workers = max(1, int(args.workers))
    print(
        "Building FL26-3 direct coupled-root domain from muK(0+) = 0 and the "
        f"cold n_s(0+) = 0 / A = 1 boundary with {workers} workers",
        flush=True,
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        semi_boundary = solve_direct_common_boundary(
            transform=transform,
            theta_count=int(args.boundary_theta_count),
            density_min_over_n0=float(args.density_min_over_n0),
            density_max_over_n0=float(args.density_max_over_n0),
            temperature_min_MeV=float(args.temperature_min),
            temperature_max_MeV=float(args.temperature_max),
            n0=float(calibration["n0_MeV3"]),
            B14=float(calibration["B14_MeV"]),
            xi=float(args.xi),
            ms=float(args.ms),
            nm_type=str(args.nm_type),
            upB=int(args.upB),
            root_tolerance=float(args.boundary_root_tolerance),
            max_nfev=int(args.boundary_max_nfev),
            T_values=T_values,
            executor=executor,
        )
    print(
        f"Direct safe rays: {len(semi_boundary['theta_axis_rad'])}/"
        f"{int(args.boundary_theta_count)}; failed ray tasks: "
        f"{len(semi_boundary['failed_ray_tasks'])}",
        flush=True,
    )
    geometry = temperature_boundary_geometry(
        semi_boundary,
        ray_count=int(args.ray_count),
        angular_buffer_fraction=float(args.angular_buffer_fraction),
    )
    points = build_temperature_indexed_points(
        geometry=geometry,
        radial_fractions=np.asarray(args.radial_fractions, dtype=float),
        transform=transform,
        radial_buffer_fraction=float(args.radial_buffer_fraction),
    )
    payload = build_domain_payload(
        args=args,
        calibration=calibration,
        points=points,
        T_values=T_values,
        boundary_T_0plus=boundary_T_0plus,
        transform=transform,
        semi_boundary=semi_boundary,
        geometry=geometry,
        started_at=started_at,
    )
    benchmark.atomic_save(output_path, payload)
    print(
        f"Saved {len(points)} contour trajectories to {output_path}",
        flush=True,
    )
    return 0


def analytical_main(method: str, argv: list[str] | None = None) -> int:
    if method not in {"analytic", "semi_analytic"}:
        raise ValueError("method must be analytic or semi_analytic")
    args = _method_parser(method).parse_args(argv)
    _validate_method_args(args)
    manifest = _load_manifest(args.domain_file)
    points = list(manifest["points"])
    T_values = tuple(float(value) for value in manifest["T_0plus_values_MeV"])
    calibration = manifest["calibration"]
    common = _task_common(manifest)
    output_path = benchmark.resolve_output_path(args.output)
    started_at = _utc_now()
    tasks = []
    for point in points:
        for state in _states_for_point(point, T_values):
            tasks.append(
                {
                    **common,
                    "method": method,
                    "point_index": int(point["point_index"]),
                    "T_0minus_MeV": float(state["T_0minus_MeV"]),
                    "nB_0minus_MeV3": float(
                        state["nB_0minus_over_n0"] * calibration["n0_MeV3"]
                    ),
                    "T_0plus_values": (float(state["T_0plus_MeV"]),),
                }
            )
    rows: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)), mp_context=context
    ) as executor, tqdm(
        total=len(tasks),
        desc=("Full analytical" if method == "analytic" else "Semi-analytical"),
        unit="cell",
        dynamic_ncols=True,
    ) as progress:
        futures = {executor.submit(benchmark.evaluate_selected_analytical_point, task): task for task in tasks}
        for completed_count, future in enumerate(as_completed(futures), start=1):
            output = future.result()
            rows.extend(output["rows"])
            progress.update(1)
            row = output["rows"][0]
            progress.set_postfix_str(
                f"p={row['point_index']}, T0+={row['T_0plus_MeV']:g}, "
                f"{'ok' if row['success'] else 'failed'}"
            )
            if int(args.save_every) > 0 and completed_count % int(args.save_every) == 0:
                benchmark.atomic_save(
                    output_path,
                    build_analytical_payload(
                        method=method,
                        args=args,
                        manifest=manifest,
                        rows=rows,
                        started_at=started_at,
                    ),
                )
    rows.sort(key=lambda row: (int(row["point_index"]), -float(row["T_0plus_MeV"])))
    payload = build_analytical_payload(
        method=method,
        args=args,
        manifest=manifest,
        rows=rows,
        started_at=started_at,
        run_complete=True,
    )
    benchmark.atomic_save(output_path, payload)
    print(
        f"Saved {sum(bool(row['success']) for row in rows)}/{len(rows)} successful "
        f"{method} cells to {output_path}",
        flush=True,
    )
    return 0


def numerical_main(argv: list[str] | None = None) -> int:
    args = _method_parser("numerical").parse_args(argv)
    _validate_method_args(args, numerical=True)
    manifest = _load_manifest(args.domain_file)
    points = list(manifest["points"])
    T_values = tuple(float(value) for value in manifest["T_0plus_values_MeV"])
    if len(T_values) != 1 or not np.isclose(T_values[0], 0.1):
        raise ValueError("the 26-09-03 Full_scan runner requires exactly T(0+)=0.1 MeV")
    T_0plus = T_values[0]
    calibration = manifest["calibration"]
    common = _task_common(manifest)
    requested_output = _input_path(args.output)
    resume_payload = None
    if requested_output.exists() and not bool(args.restart):
        resume_payload = np.load(requested_output, allow_pickle=True).item()
        if bool(resume_payload.get("run_complete", False)):
            print(f"Completed numerical output already exists: {requested_output}", flush=True)
            return 0
        output_path = requested_output
        started_at = str(resume_payload.get("started_at_utc") or _utc_now())
        print(f"Resuming incomplete checkpoint {output_path}", flush=True)
    else:
        output_path = benchmark.resolve_output_path(args.output)
        started_at = _utc_now()
    context = mp.get_context("spawn")

    enriched_points: list[dict[str, Any]] = []
    for point in points:
        state = _states_for_point(point, T_values)[0]
        enriched_points.append(
            {
                **point,
                **state,
                "nB_0minus_MeV3": float(
                    state["nB_0minus_over_n0"] * calibration["n0_MeV3"]
                ),
            }
        )
    points_by_id = {int(point["point_index"]): point for point in enriched_points}
    shells = radial.group_points_by_radial(enriched_points)
    if resume_payload is None:
        state = {
            "records": {},
            "attempt_records": [],
            "bridge_records": [],
            "profiles": {},
            "completed_radial_indices": set(),
            "reference_records": {},
            "reference_attempt_records": [],
        }
    else:
        state = resume_state_from_payload(resume_payload)
    records: dict[int, dict[str, Any]] = state["records"]
    attempt_records: list[dict[str, Any]] = state["attempt_records"]
    bridge_records: list[dict[str, Any]] = state["bridge_records"]
    profiles: dict[int, dict[str, np.ndarray]] = state["profiles"]
    completed_radial_indices: set[int] = state["completed_radial_indices"]
    reference_records: dict[int, dict[str, Any]] = state["reference_records"]
    reference_attempt_records: list[dict[str, Any]] = state[
        "reference_attempt_records"
    ]
    radial_primary_budget_s = min(
        float(args.point_timeout), float(args.target_timeout)
    )
    angular_retry_budget_s = max(
        0.0, float(args.target_timeout) - radial_primary_budget_s
    )

    def checkpoint(*, run_complete: bool = False) -> None:
        payload = build_radial_numerical_payload(
            args=args,
            manifest=manifest,
            seed_rows=[],
            target_records=[records[index] for index in sorted(records)],
            attempt_records=attempt_records,
            bridge_records=bridge_records,
            profiles=profiles,
            completed_radial_indices=completed_radial_indices,
            reference_records=[
                reference_records[index] for index in sorted(reference_records)
            ],
            reference_attempt_records=reference_attempt_records,
            started_at=started_at,
            run_complete=run_complete,
        )
        payload.update(
            shell_checkpoint_metadata(
                records=records,
                profiles=profiles,
                completed_radial_indices=completed_radial_indices,
            )
        )
        save_payload_checkpoint(output_path, payload)

    def history_entry(theta_index: int, radial_index: int) -> dict[str, Any] | None:
        point = next(
            (
                value
                for value in shells.get(int(radial_index), [])
                if int(value["theta_index"]) == int(theta_index)
            ),
            None,
        )
        if point is None:
            return None
        point_index = int(point["point_index"])
        record = records.get(point_index)
        profile = profiles.get(point_index)
        if record is None or not bool(record.get("success")) or profile is None:
            return None
        return {"point": point, "record": record, "profile": profile}

    def cell_task(
        point: dict[str, Any],
        candidates: tuple[dict[str, Any], ...],
        previous: dict[str, Any] | None,
        remaining_budget_s: float,
        T_0plus_override: float | None = None,
        stage_budget_s: float | None = None,
    ) -> dict[str, Any]:
        effective_budget_s = float(max(1.0e-6, remaining_budget_s))
        if stage_budget_s is not None:
            effective_budget_s = min(
                effective_budget_s, float(max(1.0e-6, stage_budget_s))
            )
        return {
            **common,
            "point": point,
            "T_0plus_MeV": (
                T_0plus
                if T_0plus_override is None
                else float(T_0plus_override)
            ),
            "tail_eps": float(args.tail_eps),
            "n_mesh": int(args.n_mesh),
            "tol_bvp": float(args.tol_bvp),
            "max_nodes": int(args.max_nodes),
            "jB_lower_bound": float(args.jB_lower_bound),
            "jB_upper_factor": float(args.jB_upper_factor),
            "trial_timeout_s": float(args.point_timeout),
            "cell_timeout_s": effective_budget_s,
            "radial_primary_budget_s": radial_primary_budget_s,
            "angular_retry_budget_s": angular_retry_budget_s,
            "candidates": candidates,
            "continuation_profile": previous["profile"] if previous is not None else None,
        }

    def consumed_time(outputs: list[dict[str, Any]]) -> float:
        return float(
            sum(
                float(record.get("runtime_s", 0.0))
                for output in outputs
                for record in output.get("attempt_records", [])
            )
        )

    checkpoint()
    worker_count = min(max(1, int(args.workers)), max(len(shell) for shell in shells.values()))
    first_shell_index = min(shells)
    first_shell = shells[first_shell_index]
    reference_pending = [
        point
        for point in first_shell
        if int(point["point_index"]) not in reference_records
    ]
    if reference_pending:
        print(
            f"Bootstrapping {len(reference_pending)} first-shell cells at "
            f"T(0+)={float(args.reference_T0plus):g} MeV",
            flush=True,
        )
        reference_futures = {}
        with ProcessPoolExecutor(
            max_workers=min(worker_count, len(reference_pending)),
            mp_context=context,
        ) as reference_executor, tqdm(
            total=len(first_shell),
            initial=len(first_shell) - len(reference_pending),
            desc=f"T(0+)={float(args.reference_T0plus):g} MeV first-shell bootstrap",
            unit="cell",
            dynamic_ncols=True,
        ) as reference_progress:
            for point in reference_pending:
                candidates = radial.matched_reference_candidates(
                    np.nan,
                    scale=float(args.reference_jB_scale),
                )
                task = cell_task(
                    point,
                    candidates,
                    None,
                    float(args.target_timeout),
                    T_0plus_override=float(args.reference_T0plus),
                )
                reference_futures[
                    reference_executor.submit(radial.run_cell_task, task)
                ] = point
            for future in as_completed(reference_futures):
                point = reference_futures[future]
                output = future.result()
                reference_attempt_records.extend(output.get("attempt_records", []))
                record = dict(output["selected_record"])
                reference_records[int(point["point_index"])] = record
                checkpoint()
                reference_progress.update(1)
                reference_progress.set_postfix_str(
                    f"theta={point['theta_deg']:.1f}, "
                    f"{'ok' if record.get('success') else 'failed'}",
                    refresh=False,
                )
                reference_progress.refresh()

    timeout_count = retry_count = 0
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor, tqdm(
        total=len(enriched_points),
        initial=len(records),
        desc="T(0+)=0.1 MeV numerical radial shells",
        unit="cell",
        dynamic_ncols=True,
    ) as progress:
        def finalize_cell(
            point: dict[str, Any], outputs: list[dict[str, Any]]
        ) -> None:
            nonlocal timeout_count
            point_index = int(point["point_index"])
            if point_index in records:
                return
            for output in outputs:
                attempt_records.extend(output.get("attempt_records", []))
                bridge_records.extend(output.get("bridge_records", []))
            selected_output = next(
                (
                    output
                    for output in reversed(outputs)
                    if bool(output["selected_record"].get("success"))
                ),
                outputs[-1],
            )
            record = dict(selected_output["selected_record"])
            records[point_index] = record
            if selected_output.get("profile") is not None and bool(record.get("success")):
                profiles[point_index] = selected_output["profile"]
            timeout_count += sum(
                attempt.get("status") in {"trial_timeout", "cell_timeout"}
                for output in outputs
                for attempt in output.get("attempt_records", [])
            )
            checkpoint()
            progress.update(1)
            progress.set_postfix_str(
                f"rho={point['radial_fraction_used']:.3f}, "
                f"theta={point['theta_deg']:.1f}, "
                f"{'ok' if record.get('success') else 'failed'}, "
                f"retries={retry_count}, timeouts={timeout_count}",
                refresh=False,
            )
            progress.refresh()

        for radial_index, shell in shells.items():
            pending_points = [
                point for point in shell if int(point["point_index"]) not in records
            ]
            if not pending_points:
                completed_radial_indices.add(int(radial_index))
                continue
            outputs_by_point: dict[int, list[dict[str, Any]]] = {
                int(point["point_index"]): [] for point in pending_points
            }
            previous_by_point: dict[int, dict[str, Any] | None] = {}
            radial_fallbacks_by_point: dict[int, tuple[dict[str, Any], ...]] = {}
            angular_signatures_by_point: dict[int, set[tuple[Any, ...]]] = {
                int(point["point_index"]): set() for point in pending_points
            }
            primary_futures = {}
            for point in pending_points:
                theta_index = int(point["theta_index"])
                previous = history_entry(theta_index, int(radial_index) - 1)
                earlier = (
                    history_entry(theta_index, int(radial_index) - 2)
                    if previous is not None
                    else None
                )
                previous_by_point[int(point["point_index"])] = previous
                reference_record = reference_records.get(int(point["point_index"]), {})
                reference_jB = (
                    float(reference_record["jB"])
                    if bool(reference_record.get("success"))
                    and np.isfinite(reference_record.get("jB", np.nan))
                    else np.nan
                )
                candidates = radial.primary_candidates(
                    point,
                    previous=previous,
                    earlier=earlier,
                    reference_jB=reference_jB,
                    reference_scale=float(args.reference_jB_scale),
                )
                primary_candidates, radial_fallbacks = radial.split_radial_candidates(
                    candidates
                )
                radial_fallbacks_by_point[int(point["point_index"])] = radial_fallbacks
                if primary_candidates:
                    task = cell_task(
                        point,
                        primary_candidates,
                        previous,
                        float(args.target_timeout),
                        stage_budget_s=radial_primary_budget_s,
                    )
                    primary_futures[executor.submit(radial.run_cell_task, task)] = point
            collect_stage_futures(
                primary_futures,
                outputs_by_point=outputs_by_point,
                finalize_successes=True,
                finalize_all=False,
                on_terminal=finalize_cell,
            )

            def successful_shell_currents() -> dict[int, float]:
                values: dict[int, float] = {}
                for shell_point in shell:
                    shell_point_index = int(shell_point["point_index"])
                    record = records.get(shell_point_index)
                    if record is not None and bool(record.get("success")):
                        jB = float(record.get("jB", np.nan))
                        if np.isfinite(jB) and jB > 0.0:
                            values[int(shell_point["theta_index"])] = jB
                return values

            angular_wave_index = 0

            def run_angular_waves(*, perturbed: bool) -> int:
                nonlocal angular_wave_index, retry_count
                recovered_total = 0
                while True:
                    successful_currents = successful_shell_currents()
                    wave_futures = {}
                    next_wave_index = angular_wave_index + 1
                    for wave_point in pending_points:
                        wave_point_index = int(wave_point["point_index"])
                        if wave_point_index in records:
                            continue
                        outputs = outputs_by_point[wave_point_index]
                        remaining = float(args.target_timeout) - consumed_time(outputs)
                        if remaining <= 0.0:
                            continue
                        seed = radial.immediate_theta_seed(
                            successful_currents,
                            target_theta_index=int(wave_point["theta_index"]),
                        )
                        if seed is None:
                            continue
                        signature = (
                            "perturbed" if perturbed else "direct",
                            tuple(seed["theta_indices"]),
                            str(seed["mode"]),
                        )
                        if signature in angular_signatures_by_point[wave_point_index]:
                            continue
                        angular_signatures_by_point[wave_point_index].add(signature)
                        previous = previous_by_point[wave_point_index]
                        candidates = radial.angular_wave_candidates(
                            seed,
                            use_profile=previous is not None,
                            wave_index=next_wave_index,
                            multipliers=(0.90, 1.10, 1.20) if perturbed else (1.0,),
                        )
                        if not candidates:
                            continue
                        retry_count += 1
                        task = cell_task(
                            wave_point,
                            candidates,
                            previous,
                            remaining,
                            stage_budget_s=(
                                remaining
                                if perturbed
                                else min(float(args.point_timeout), remaining)
                            ),
                        )
                        wave_futures[
                            executor.submit(radial.run_cell_task, task)
                        ] = wave_point
                    if not wave_futures:
                        break
                    angular_wave_index = next_wave_index
                    successful_before = len(records)
                    collect_stage_futures(
                        wave_futures,
                        outputs_by_point=outputs_by_point,
                        finalize_successes=True,
                        finalize_all=False,
                        on_terminal=finalize_cell,
                    )
                    recovered = len(records) - successful_before
                    recovered_total += recovered
                    if recovered == 0 or perturbed:
                        break
                return recovered_total

            # Give strict same-shell angular continuation the reserved budget
            # before spending time on the old radial multiplier ladder.
            run_angular_waves(perturbed=False)
            if run_angular_waves(perturbed=True):
                run_angular_waves(perturbed=False)

            radial_fallback_futures = {}
            for point in pending_points:
                point_index = int(point["point_index"])
                if point_index in records:
                    continue
                outputs = outputs_by_point[point_index]
                remaining = float(args.target_timeout) - consumed_time(outputs)
                candidates = radial_fallbacks_by_point.get(point_index, ())
                if remaining <= 0.0 or not candidates:
                    continue
                previous = previous_by_point[point_index]
                retry_count += 1
                task = cell_task(point, candidates, previous, remaining)
                radial_fallback_futures[
                    executor.submit(radial.run_cell_task, task)
                ] = point
            collect_stage_futures(
                radial_fallback_futures,
                outputs_by_point=outputs_by_point,
                finalize_successes=True,
                finalize_all=False,
                on_terminal=finalize_cell,
            )

            # A radial fallback can create a new angular anchor, so let that
            # success propagate before attempting expensive bridge solves.
            run_angular_waves(perturbed=False)
            if run_angular_waves(perturbed=True):
                run_angular_waves(perturbed=False)

            bridge_futures = {}
            for point in pending_points:
                point_index = int(point["point_index"])
                if point_index in records:
                    continue
                outputs = outputs_by_point[point_index]
                previous = previous_by_point[point_index]
                remaining = float(args.target_timeout) - consumed_time(outputs)
                if (
                    previous is None
                    or remaining <= 0.0
                    or int(args.max_bridge_depth) <= 0
                ):
                    if outputs:
                        finalize_cell(point, outputs)
                    continue
                task = cell_task(point, (), previous, remaining)
                task.update(
                    {
                        "previous_entry": previous,
                        "max_bridge_depth": int(args.max_bridge_depth),
                        "min_bridge_fraction_step": float(
                            args.min_bridge_fraction_step
                        ),
                    }
                )
                bridge_futures[executor.submit(radial.run_bridge_task, task)] = point
            collect_stage_futures(
                bridge_futures,
                outputs_by_point=outputs_by_point,
                finalize_successes=False,
                finalize_all=True,
                on_terminal=finalize_cell,
            )

            for point in pending_points:
                point_index = int(point["point_index"])
                if point_index in records:
                    continue
                outputs = outputs_by_point[point_index]
                if not outputs:
                    previous = previous_by_point[point_index]
                    fallback = radial.angular_wave_candidates(
                        radial.immediate_theta_seed(
                            successful_shell_currents(),
                            target_theta_index=int(point["theta_index"]),
                        ),
                        use_profile=previous is not None,
                        wave_index=angular_wave_index + 1,
                    )
                    task = cell_task(point, fallback, previous, 1.0e-6)
                    selected_output = radial.attempt_jB_candidates(
                        task,
                        fallback,
                        continuation_profile=(
                            previous["profile"] if previous is not None else None
                        ),
                    )
                    outputs.append(selected_output)
                finalize_cell(point, outputs)
            completed_radial_indices.add(int(radial_index))
            checkpoint()

    checkpoint(run_complete=True)
    payload = np.load(output_path, allow_pickle=True).item()
    successes = int(np.count_nonzero(payload["numerical_outward_success"]))
    total = int(payload["numerical_outward_success"].size)
    print(f"Saved {successes}/{total} numerical cells to {output_path}", flush=True)
    return 0
