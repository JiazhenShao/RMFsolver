#!/usr/bin/env python3
"""Shared implementation for the fixed T(0+)=0.1 MeV contour runners."""

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
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import RMFsolver.constants as const
import _method_engine as benchmark
import _radial_shell_continuation as radial
import _slow_front_adapter as slow


DOMAIN_FILENAME = "T0plus0p1-domain.npy"
FULL_ANALYTIC_FILENAME = "T0plus0p1-full-analytic.npy"
SEMI_ANALYTIC_FILENAME = "T0plus0p1-semi-analytic.npy"
NUMERICAL_FILENAME = "T0plus0p1-adiabatic-slow.npy"
CONTINUATION_STAGE_ORDER = (
    "radial_primary",
    "angular_direct_wavefront",
    "angular_perturbed_wavefront",
    "radial_multiplier_fallback",
    "angular_recovery_after_radial_fallback",
    "radial_bridge_fallback",
)




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
            "velocity_seed_model": "saved_relativistic_flux_then_slow_profile_continuation",
            "seed_reference_T_0plus_MeV": 0.1,
            "seed_reference_jB_scale": float(args.reference_jB_scale),
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
            "--reference-jB-scale",
            type=float,
            default=1.18,
            help="optional scale of the saved relativistic numerical current guess",
        )
        parser.add_argument("--max-bridge-depth", type=int, default=5)
        parser.add_argument(
            "--min-bridge-fraction-step", type=float, default=1.0e-3
        )
        parser.add_argument("--path-rtol", type=float, default=1e-6)
        parser.add_argument("--kappa-factor", type=float, default=1.0)
        parser.add_argument("--smoke", action="store_true", help="run a 3 by 2 subset into a separate output")
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
        if not 0 < args.tail_eps < 1:
            raise ValueError("tail-eps must lie strictly between zero and one")
        for name in ("path_rtol", "kappa_factor", "tol_bvp", "reference_jB_scale", "point_timeout", "target_timeout"):
            value = float(getattr(args, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive and finite")
        if int(args.n_mesh) < 13 or int(args.max_nodes) <= int(args.n_mesh):
            raise ValueError("numerical mesh settings are invalid")
        if (
            float(args.tol_bvp) <= 0.0
            or float(args.point_timeout) <= 0.0
            or float(args.target_timeout) <= 0.0
        ):
            raise ValueError("numerical tolerance and timeout must be positive")
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


def numerical_main(argv: list[str] | None = None) -> int:
    args = _method_parser("numerical").parse_args(argv)
    _validate_method_args(args, numerical=True)
    manifest = _load_manifest(args.domain_file)
    if args.smoke:
        manifest = slow.smoke_manifest(manifest)
        output = Path(args.output)
        args.output = str(output.with_name(output.stem + "-smoke" + output.suffix))
    identity = slow.run_identity(manifest, args)
    points = list(manifest["points"])
    T_values = tuple(float(value) for value in manifest["T_0plus_values_MeV"])
    if len(T_values) != 1 or not np.isclose(T_values[0], 0.1):
        raise ValueError("the slow-front runner requires exactly T(0+)=0.1 MeV")
    T_0plus = T_values[0]
    calibration = manifest["calibration"]
    common = _task_common(manifest)
    requested_output = _input_path(args.output)
    resume_payload = None
    if requested_output.exists() and not bool(args.restart):
        resume_payload = np.load(requested_output, allow_pickle=True).item()
        slow.validate_resume(resume_payload, identity)
        if bool(resume_payload.get("run_complete", False)):
            print(f"Completed numerical output already exists: {requested_output}", flush=True)
            return 0 if resume_payload.get("all_targets_successful") else 2
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
        payload.update(slow.payload_metadata(manifest, records, identity, run_complete))
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
            "path_rtol": float(args.path_rtol),
            "kappa_factor": float(args.kappa_factor),
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
                reference_jB = float(point["relativistic_jB_guess"])
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
                        if previous is None:
                            # A failed radial predecessor must not discard a
                            # newly solved immediate-angle slow-front profile.
                            previous = next((entry for index in seed["theta_indices"]
                                             if (entry := history_entry(index, int(radial_index)))
                                             is not None), None)
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
    return 0 if successes == total else 2
