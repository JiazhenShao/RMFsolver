#!/usr/bin/env python3
"""Recover the first five rho=0.90 rays through hidden-rho continuation."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
SOURCE_RUN_DIR = PROJECT_ROOT / "new_paper_calculations" / "26-08-21"
for path in (PROJECT_ROOT, SOURCE_RUN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _radial_shell_continuation as radial
import _T0plus0p1_runner as source_runner


# Tunable run parameters
DEFAULT_THETA_INDICES = (0, 1, 2, 3, 4)
DEFAULT_RHO_LADDER = (0.86, 0.875, 0.885, 0.8925, 0.90)
DEFAULT_JB_DIVERGENCE_EXPONENT = 1.26
DEFAULT_JB_PERTURBATIONS = (1.0, 0.95, 1.05, 0.90, 1.10)
DEFAULT_WORKERS = 5
DEFAULT_TRIAL_TIMEOUT_S = 120.0
DEFAULT_CELL_TIMEOUT_S = 600.0
DEFAULT_TAIL_EPS = 1.0e-8
DEFAULT_N_MESH = 200
DEFAULT_TOL_BVP = 1.0e-4
DEFAULT_MAX_NODES = 10000
DEFAULT_JB_LOWER_BOUND = 1.0e-12
DEFAULT_JB_UPPER_FACTOR = 1.0e-4
DEFAULT_T0PLUS_MEV = 0.1
DEFAULT_BASE_RHO = 0.84
DEFAULT_DOMAIN_PATH = SOURCE_RUN_DIR / "T0plus0p1-domain.npy"
DEFAULT_SOURCE_RESULT_PATH = (
    SOURCE_RUN_DIR / "T0plus0p1-uNmax-angular-wavefront.npy"
)
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "T0plus0p1-rho0p9-hidden-rho-pilot.npy"
SPEED_OF_LIGHT_M_S = 299_792_458.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_number_tuple(value: str, cast=float) -> tuple:
    values = tuple(cast(part.strip()) for part in str(value).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated nonempty list")
    return values


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npy", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, payload, allow_pickle=True)
    temporary.replace(path)


def point_map(domain_payload: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(point["theta_index"]), int(point["radial_index"])): dict(point)
        for point in domain_payload["points"]
    }


def record_map(source_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(record["point_index"]): dict(record)
        for record in source_payload["radial_shell_records"]
    }


def enrich_domain_point(
    point: dict[str, Any], domain_payload: dict[str, Any]
) -> dict[str, Any]:
    point = dict(point)
    state = source_runner._states_for_point(point, (DEFAULT_T0PLUS_MEV,))[0]
    point.update(state)
    point["nB_0minus_MeV3"] = float(
        point["nB_0minus_over_n0"]
        * domain_payload["calibration"]["n0_MeV3"]
    )
    return point


def interpolate_hidden_point(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    target_rho: float,
    synthetic_point_index: int,
    synthetic_radial_index: int,
) -> dict[str, Any]:
    """Interpolate one exact point on a fixed-theta ray."""
    left_rho = float(left["radial_fraction_used"])
    right_rho = float(right["radial_fraction_used"])
    target_rho = float(target_rho)
    if not left_rho < target_rho <= right_rho:
        raise ValueError(
            f"target rho={target_rho} is outside ({left_rho}, {right_rho}]"
        )
    weight = (target_rho - left_rho) / (right_rho - left_rho)
    point = dict(right)
    for field in (
        "T_0minus_MeV",
        "nB_0minus_over_n0",
        "nB_0minus_MeV3",
        "rho",
    ):
        point[field] = (1.0 - weight) * float(left[field]) + weight * float(
            right[field]
        )
    point.update(
        {
            "point_index": int(synthetic_point_index),
            "radial_index": int(synthetic_radial_index),
            "radial_fraction": target_rho,
            "radial_fraction_used": target_rho,
            "hidden_rho_continuation": True,
        }
    )
    return point


def predict_jB(
    previous_jB: float,
    *,
    previous_rho: float,
    target_rho: float,
    exponent: float,
) -> float:
    """Predict accelerating current growth as rho approaches one."""
    previous_jB = float(previous_jB)
    previous_rho = float(previous_rho)
    target_rho = float(target_rho)
    exponent = float(exponent)
    if previous_jB <= 0.0 or not 0.0 <= previous_rho < target_rho < 1.0:
        raise ValueError("jB must be positive and 0 <= previous_rho < target_rho < 1")
    return float(
        previous_jB
        * ((1.0 - previous_rho) / (1.0 - target_rho)) ** exponent
    )


def jB_candidates(
    predicted_jB: float, perturbations: tuple[float, ...]
) -> tuple[dict[str, Any], ...]:
    values = radial.deduplicate_jB_candidates(
        [float(predicted_jB) * float(factor) for factor in perturbations],
        relative_tolerance=0.02,
    )
    return tuple(
        {
            "jB_guess": value,
            "source": "hidden_rho_divergence_prediction",
            "use_profile": True,
        }
        for value in values
    )


def entry_key(theta_index: int, target_rho: float) -> str:
    return f"theta={int(theta_index):02d}|rho={float(target_rho):.8f}"


def failure_output(task: dict[str, Any], message: str) -> dict[str, Any]:
    point = task["point"]
    record = {
        "point_index": int(point["point_index"]),
        "theta_index": int(point["theta_index"]),
        "radial_index": int(point["radial_index"]),
        "radial_fraction": float(point["radial_fraction"]),
        "radial_fraction_used": float(point["radial_fraction_used"]),
        "T_0plus_MeV": float(task["T_0plus_MeV"]),
        "T_0minus_MeV": float(point["T_0minus_MeV"]),
        "nB_0minus_over_n0": float(point["nB_0minus_over_n0"]),
        "status": "worker_failure",
        "success": False,
        "message": str(message),
        "jB_guess": float(task["candidates"][0]["jB_guess"]),
        "jB_guess_source": "hidden_rho_divergence_prediction",
    }
    return {
        "selected_record": record,
        "attempt_records": [record],
        "profile": None,
        "result": None,
    }


def blocked_entry(
    point: dict[str, Any], *, previous_rho: float, target_rho: float
) -> dict[str, Any]:
    record = {
        "point_index": int(point["point_index"]),
        "theta_index": int(point["theta_index"]),
        "radial_index": int(point["radial_index"]),
        "radial_fraction": float(target_rho),
        "radial_fraction_used": float(target_rho),
        "T_0plus_MeV": DEFAULT_T0PLUS_MEV,
        "T_0minus_MeV": float(point["T_0minus_MeV"]),
        "nB_0minus_over_n0": float(point["nB_0minus_over_n0"]),
        "status": "blocked_by_previous_hidden_rho",
        "success": False,
        "message": f"rho={previous_rho:g} did not provide a successful profile",
    }
    return {
        "point": point,
        "record": record,
        "attempt_records": [],
        "profile": None,
    }


def build_payload(
    *,
    args: argparse.Namespace,
    domain_payload: dict[str, Any],
    entries: dict[str, dict[str, Any]],
    started_at: str,
    run_complete: bool,
) -> dict[str, Any]:
    theta_indices = tuple(int(value) for value in args.theta_indices)
    rho_ladder = tuple(float(value) for value in args.rho_ladder)
    success_grid = np.zeros((len(theta_indices), len(rho_ladder)), dtype=bool)
    jB_grid = np.full(success_grid.shape, np.nan)
    velocity_grid = np.full(success_grid.shape, np.nan)
    records = []
    profiles = {}
    attempt_records = []
    for theta_position, theta_index in enumerate(theta_indices):
        for rho_position, rho_value in enumerate(rho_ladder):
            key = entry_key(theta_index, rho_value)
            entry = entries.get(key)
            if entry is None:
                continue
            record = dict(entry["record"])
            records.append(record)
            attempt_records.extend(dict(item) for item in entry["attempt_records"])
            success = bool(record.get("success"))
            success_grid[theta_position, rho_position] = success
            if success:
                jB_grid[theta_position, rho_position] = float(record["jB"])
                velocity_grid[theta_position, rho_position] = (
                    float(record["u_0minus"]) * SPEED_OF_LIGHT_M_S
                )
                if entry.get("profile") is not None:
                    profiles[key] = entry["profile"]
    return {
        "run_tag": "T0plus0p1_rho0p9_hidden_rho_pilot",
        "created_at_utc": started_at,
        "checkpoint_at_utc": utc_now(),
        "run_complete": bool(run_complete),
        "checkpoint_granularity": "every_hidden_rho_point",
        "source_domain_path": str(args.domain),
        "source_result_path": str(args.source_result),
        "T_0plus_MeV": DEFAULT_T0PLUS_MEV,
        "theta_indices": np.asarray(theta_indices, dtype=int),
        "rho_base": float(args.base_rho),
        "rho_ladder": np.asarray(rho_ladder, dtype=float),
        "jB_divergence_exponent": float(args.jB_exponent),
        "jB_perturbations": np.asarray(args.jB_perturbations, dtype=float),
        "trial_timeout_s": float(args.trial_timeout),
        "cell_timeout_s": float(args.cell_timeout),
        "entries": dict(entries),
        "records": records,
        "attempt_records": attempt_records,
        "profiles": profiles,
        "success_grid": success_grid,
        "jB_grid": jB_grid,
        "u_0minus_m_s_grid": velocity_grid,
        "completed_step_count": int(len(entries)),
        "successful_step_count": int(np.count_nonzero(success_grid)),
        "domain_calibration": domain_payload["calibration"],
        "arguments": vars(args),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Pilot hidden-rho continuation for theta indices 0-4."
    )
    result.add_argument("--theta-indices", type=lambda value: parse_number_tuple(value, int), default=DEFAULT_THETA_INDICES)
    result.add_argument("--rho-ladder", type=parse_number_tuple, default=DEFAULT_RHO_LADDER)
    result.add_argument("--jB-exponent", type=float, default=DEFAULT_JB_DIVERGENCE_EXPONENT)
    result.add_argument("--jB-perturbations", type=parse_number_tuple, default=DEFAULT_JB_PERTURBATIONS)
    result.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    result.add_argument("--trial-timeout", type=float, default=DEFAULT_TRIAL_TIMEOUT_S)
    result.add_argument("--cell-timeout", type=float, default=DEFAULT_CELL_TIMEOUT_S)
    result.add_argument("--tail-eps", type=float, default=DEFAULT_TAIL_EPS)
    result.add_argument("--n-mesh", type=int, default=DEFAULT_N_MESH)
    result.add_argument("--tol-bvp", type=float, default=DEFAULT_TOL_BVP)
    result.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    result.add_argument("--jB-lower-bound", type=float, default=DEFAULT_JB_LOWER_BOUND)
    result.add_argument("--jB-upper-factor", type=float, default=DEFAULT_JB_UPPER_FACTOR)
    result.add_argument("--base-rho", type=float, default=DEFAULT_BASE_RHO)
    result.add_argument("--domain", type=Path, default=DEFAULT_DOMAIN_PATH)
    result.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT_PATH)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    result.add_argument("--restart", action="store_true")
    result.add_argument("--retry-failed", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def validate_args(args: argparse.Namespace) -> None:
    if not args.theta_indices or any(index < 0 or index >= 30 for index in args.theta_indices):
        raise ValueError("theta indices must lie in [0, 29]")
    if tuple(sorted(set(args.theta_indices))) != tuple(args.theta_indices):
        raise ValueError("theta indices must be unique and increasing")
    if not args.rho_ladder or any(not args.base_rho < rho < 1.0 for rho in args.rho_ladder):
        raise ValueError("every hidden rho must satisfy base_rho < rho < 1")
    if tuple(sorted(set(args.rho_ladder))) != tuple(args.rho_ladder):
        raise ValueError("rho ladder must be unique and increasing")
    if args.workers < 1 or args.trial_timeout <= 0.0 or args.cell_timeout <= 0.0:
        raise ValueError("workers and time limits must be positive")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    validate_args(args)
    domain_payload = np.load(args.domain, allow_pickle=True).item()
    source_payload = np.load(args.source_result, allow_pickle=True).item()
    points = point_map(domain_payload)
    source_records = record_map(source_payload)
    source_profiles = {
        int(index): dict(profile)
        for index, profile in source_payload["radial_shell_profiles"].items()
    }
    base_radial_index = next(
        int(point["radial_index"])
        for point in domain_payload["points"]
        if np.isclose(float(point["radial_fraction_used"]), float(args.base_rho))
    )
    target_radial_index = next(
        int(point["radial_index"])
        for point in domain_payload["points"]
        if np.isclose(float(point["radial_fraction_used"]), 0.90)
    )
    base_entries: dict[int, dict[str, Any]] = {}
    target_points: dict[int, dict[str, Any]] = {}
    for theta_index in args.theta_indices:
        base_point = enrich_domain_point(points[(theta_index, base_radial_index)], domain_payload)
        target_point = enrich_domain_point(points[(theta_index, target_radial_index)], domain_payload)
        base_record = source_records[int(base_point["point_index"])]
        base_profile = source_profiles.get(int(base_point["point_index"]))
        if not bool(base_record.get("success")) or base_profile is None:
            raise RuntimeError(f"theta_index={theta_index} has no successful rho={args.base_rho} base profile")
        base_entries[int(theta_index)] = {
            "point": base_point,
            "record": base_record,
            "profile": base_profile,
        }
        target_points[int(theta_index)] = target_point

    if args.output.exists() and not args.restart:
        existing_payload = np.load(args.output, allow_pickle=True).item()
        entries = dict(existing_payload.get("entries", {}))
        started_at = str(existing_payload.get("created_at_utc", utc_now()))
        print(f"Resuming {args.output}", flush=True)
    else:
        entries = {}
        started_at = utc_now()

    total_base_steps = len(args.theta_indices) * len(args.rho_ladder)
    pending_retry_count = 0
    if args.retry_failed:
        pending_retry_count = sum(
            1
            for entry in entries.values()
            if not bool(entry.get("record", {}).get("success"))
        )
    progress_total = total_base_steps + pending_retry_count
    progress_initial = len(entries)

    print(f"Theta indices: {tuple(args.theta_indices)}", flush=True)
    print(f"Hidden rho ladder: {tuple(args.rho_ladder)}", flush=True)
    print(f"jB growth exponent: {args.jB_exponent:g}", flush=True)
    print(f"Output: {args.output}", flush=True)
    if args.dry_run:
        for theta_index in args.theta_indices:
            base = base_entries[int(theta_index)]
            predicted = predict_jB(
                base["record"]["jB"],
                previous_rho=args.base_rho,
                target_rho=args.rho_ladder[0],
                exponent=args.jB_exponent,
            )
            print(
                f"theta={theta_index}: rho={args.base_rho:g} jB={base['record']['jB']:.8g} "
                f"-> rho={args.rho_ladder[0]:g} predicted jB={predicted:.8g}"
            )
        return 0

    common = source_runner._task_common(domain_payload)

    def checkpoint(run_complete: bool = False) -> None:
        atomic_save(
            args.output,
            build_payload(
                args=args,
                domain_payload=domain_payload,
                entries=entries,
                started_at=started_at,
                run_complete=run_complete,
            ),
        )

    checkpoint(False)
    context = mp.get_context("spawn")
    worker_count = min(int(args.workers), len(args.theta_indices))
    with tqdm(
        total=progress_total,
        initial=progress_initial,
        desc="Hidden-rho pilot",
        unit="point",
        dynamic_ncols=True,
    ) as progress, ProcessPoolExecutor(
        max_workers=worker_count, mp_context=context
    ) as executor:
        for ladder_index, target_rho in enumerate(args.rho_ladder):
            future_to_metadata = {}
            for theta_index in args.theta_indices:
                key = entry_key(theta_index, target_rho)
                existing = entries.get(key)
                if existing is not None and (
                    bool(existing["record"].get("success")) or not args.retry_failed
                ):
                    continue
                if ladder_index == 0:
                    previous = base_entries[int(theta_index)]
                    previous_rho = float(args.base_rho)
                else:
                    previous_rho = float(args.rho_ladder[ladder_index - 1])
                    previous = entries.get(entry_key(theta_index, previous_rho))
                target = interpolate_hidden_point(
                    base_entries[int(theta_index)]["point"],
                    target_points[int(theta_index)],
                    target_rho=float(target_rho),
                    synthetic_point_index=1_000_000 + int(theta_index) * 100 + ladder_index,
                    synthetic_radial_index=100 + ladder_index,
                )
                if previous is None or not bool(previous["record"].get("success")) or previous.get("profile") is None:
                    entries[key] = blocked_entry(
                        target, previous_rho=previous_rho, target_rho=target_rho
                    )
                    checkpoint(False)
                    progress.update(1)
                    progress.set_postfix(
                        theta=int(theta_index), rho=f"{target_rho:.4f}", status="blocked"
                    )
                    continue
                predicted = predict_jB(
                    previous["record"]["jB"],
                    previous_rho=previous_rho,
                    target_rho=target_rho,
                    exponent=args.jB_exponent,
                )
                candidates = jB_candidates(predicted, tuple(args.jB_perturbations))
                task = {
                    **common,
                    "point": target,
                    "T_0plus_MeV": DEFAULT_T0PLUS_MEV,
                    "tail_eps": float(args.tail_eps),
                    "n_mesh": int(args.n_mesh),
                    "tol_bvp": float(args.tol_bvp),
                    "max_nodes": int(args.max_nodes),
                    "jB_lower_bound": float(args.jB_lower_bound),
                    "jB_upper_factor": float(args.jB_upper_factor),
                    "trial_timeout_s": float(args.trial_timeout),
                    "cell_timeout_s": float(args.cell_timeout),
                    "candidates": candidates,
                    "continuation_profile": previous["profile"],
                }
                future = executor.submit(radial.run_cell_task, task)
                future_to_metadata[future] = (key, task, target, theta_index)
            for future in as_completed(future_to_metadata):
                key, task, target, theta_index = future_to_metadata[future]
                try:
                    output = future.result()
                except Exception as exc:
                    output = failure_output(task, f"{type(exc).__name__}: {exc}")
                record = dict(output["selected_record"])
                record.update(
                    {
                        "hidden_rho_target": float(target_rho),
                        "jB_divergence_exponent": float(args.jB_exponent),
                        "predicted_jB": float(task["candidates"][0]["jB_guess"]),
                    }
                )
                entries[key] = {
                    "point": target,
                    "record": record,
                    "attempt_records": [dict(item) for item in output["attempt_records"]],
                    "profile": output.get("profile"),
                }
                checkpoint(False)
                progress.update(1)
                progress.set_postfix(
                    theta=int(theta_index),
                    rho=f"{target_rho:.4f}",
                    status=str(record.get("status")),
                    success=sum(
                        bool(entry["record"].get("success")) for entry in entries.values()
                    ),
                )
    expected_keys = {
        entry_key(theta_index, rho)
        for theta_index in args.theta_indices
        for rho in args.rho_ladder
    }
    run_complete = expected_keys.issubset(entries)
    checkpoint(run_complete)
    successes = sum(bool(entry["record"].get("success")) for entry in entries.values())
    final_successes = sum(
        bool(entries.get(entry_key(theta_index, args.rho_ladder[-1]), {}).get("record", {}).get("success"))
        for theta_index in args.theta_indices
    )
    print(
        f"Done: {successes}/{len(entries)} hidden steps successful; "
        f"rho={args.rho_ladder[-1]:g} successes={final_successes}/{len(args.theta_indices)}",
        flush=True,
    )
    print(f"Saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
