#!/usr/bin/env python3
"""Build an adaptive numerical bracket for the 5000 m/s contour."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PILOT_ENGINE_DIR = SCRIPT_DIR.parent / "rho0p9_hidden_rho_pilot"
SOURCE_RUN_DIR = PROJECT_ROOT / "new_paper_calculations" / "26-08-21"
for path in (PROJECT_ROOT, PILOT_ENGINE_DIR, SOURCE_RUN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _radial_shell_continuation as radial
import _T0plus0p1_runner as source_runner
import run_hidden_rho_pilot as engine


# Tunable run parameters
DEFAULT_THETA_INDICES = tuple(range(7, 27))
DEFAULT_BASE_RHO = 0.86
DEFAULT_TARGET_VELOCITY_M_S = 5200.0
DEFAULT_ACCEPTANCE_VELOCITY_M_S = 5050.0
DEFAULT_MAXIMUM_RHO = 0.885
DEFAULT_MAXIMUM_HIDDEN_STEP = 0.005
DEFAULT_EXTENSION_STEP = 0.0025
DEFAULT_JB_DIVERGENCE_EXPONENT = 1.26
DEFAULT_JB_PERTURBATIONS = (1.0, 0.95, 1.05, 0.90, 1.10)
DEFAULT_WORKERS = 20
DEFAULT_TRIAL_TIMEOUT_S = 300.0
DEFAULT_CELL_TIMEOUT_S = 900.0
DEFAULT_TAIL_EPS = 1.0e-8
DEFAULT_N_MESH = 200
DEFAULT_TOL_BVP = 1.0e-4
DEFAULT_MAX_NODES = 10000
DEFAULT_JB_LOWER_BOUND = 1.0e-12
DEFAULT_JB_UPPER_FACTOR = 1.0e-4
DEFAULT_T0PLUS_MEV = 0.1
DEFAULT_DOMAIN_PATH = SOURCE_RUN_DIR / "T0plus0p1-domain.npy"
DEFAULT_BASE_RESULT_PATH = (
    SCRIPT_DIR.parent
    / "rho0p86_full_theta_shell"
    / "T0plus0p1-rho0p86-full-theta-shell.npy"
)
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "T0plus0p1-rho5k-adaptive-bracket.npy"
SPEED_OF_LIGHT_M_S = 299_792_458.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def predict_target_rho(
    velocity_m_s: float,
    *,
    base_rho: float,
    target_velocity_m_s: float,
    exponent: float,
    minimum_increment: float,
    maximum_rho: float,
) -> float:
    """Predict a radial target from the observed near-boundary growth law."""
    velocity_m_s = float(velocity_m_s)
    base_rho = float(base_rho)
    target_velocity_m_s = float(target_velocity_m_s)
    exponent = float(exponent)
    minimum_increment = float(minimum_increment)
    maximum_rho = float(maximum_rho)
    if velocity_m_s <= 0.0 or target_velocity_m_s <= 0.0 or exponent <= 0.0:
        raise ValueError("velocities and exponent must be positive")
    if not 0.0 <= base_rho < maximum_rho < 1.0:
        raise ValueError("expected 0 <= base_rho < maximum_rho < 1")
    raw_target = 1.0 - (1.0 - base_rho) * (
        velocity_m_s / target_velocity_m_s
    ) ** (1.0 / exponent)
    return float(
        np.clip(raw_target, base_rho + minimum_increment, maximum_rho)
    )


def build_hidden_ladder(
    *, base_rho: float, target_rho: float, maximum_step: float
) -> tuple[float, ...]:
    """Return increasing hidden points ending exactly at target_rho."""
    base_rho = float(base_rho)
    target_rho = float(target_rho)
    maximum_step = float(maximum_step)
    if not 0.0 <= base_rho < target_rho < 1.0 or maximum_step <= 0.0:
        raise ValueError("expected 0 <= base_rho < target_rho < 1 and step > 0")
    step_count = int(np.ceil((target_rho - base_rho) / maximum_step))
    return tuple(
        float(value)
        for value in np.linspace(base_rho, target_rho, step_count + 1)[1:]
    )


def next_extension_rho(
    current_rho: float, *, increment: float, maximum_rho: float
) -> float | None:
    current_rho = float(current_rho)
    increment = float(increment)
    maximum_rho = float(maximum_rho)
    if current_rho >= maximum_rho - 1.0e-12:
        return None
    return float(min(current_rho + increment, maximum_rho))


def build_jB_candidates(
    *,
    predicted_radial_jB: float,
    perturbations: tuple[float, ...],
    theta_index: int,
    target_rho: float,
    base_rho: float,
    exponent: float,
    base_jB_by_theta: dict[int, float],
) -> tuple[dict[str, Any], ...]:
    """Put radial-profile predictions before strictly adjacent-theta seeds."""
    radial_values = radial.deduplicate_jB_candidates(
        [float(predicted_radial_jB) * float(factor) for factor in perturbations],
        relative_tolerance=0.02,
    )
    candidates: list[dict[str, Any]] = [
        {
            "jB_guess": float(value),
            "source": "adaptive_radial_prediction",
            "use_profile": True,
            "angular_seed_theta_indices": (),
        }
        for value in radial_values
    ]
    neighbor_indices = tuple(
        index
        for index in (int(theta_index) - 1, int(theta_index) + 1)
        if index in base_jB_by_theta
    )
    if neighbor_indices:
        neighbor_predictions = [
            engine.predict_jB(
                base_jB_by_theta[index],
                previous_rho=base_rho,
                target_rho=target_rho,
                exponent=exponent,
            )
            for index in neighbor_indices
        ]
        source = (
            "adjacent_theta_two_sided"
            if len(neighbor_indices) == 2
            else "adjacent_theta_one_sided"
        )
        candidates.append(
            {
                "jB_guess": float(np.mean(neighbor_predictions)),
                "source": source,
                "use_profile": True,
                "angular_seed_theta_indices": neighbor_indices,
            }
        )
    return tuple(candidates)


def parse_number_tuple(value: str, cast=float) -> tuple:
    values = tuple(cast(part.strip()) for part in str(value).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated nonempty list")
    return values


def entry_key(theta_index: int, target_rho: float) -> str:
    return f"theta={int(theta_index):02d}|rho={float(target_rho):.8f}"


def terminal_failure_entry(
    *, theta_index: int, rho: float, previous: dict[str, Any], message: str
) -> dict[str, Any]:
    point = previous["point"]
    record = {
        "point_index": int(point["point_index"]),
        "theta_index": int(theta_index),
        "radial_index": int(point["radial_index"]),
        "radial_fraction": float(rho),
        "radial_fraction_used": float(rho),
        "T_0plus_MeV": DEFAULT_T0PLUS_MEV,
        "T_0minus_MeV": float(point["T_0minus_MeV"]),
        "nB_0minus_over_n0": float(point["nB_0minus_over_n0"]),
        "status": "maximum_rho_below_acceptance",
        "success": False,
        "message": str(message),
    }
    return {"point": point, "record": record, "attempt_records": [], "profile": None}


def build_payload(
    *,
    args: argparse.Namespace,
    domain_payload: dict[str, Any],
    plans: dict[int, dict[str, Any]],
    entries: dict[str, dict[str, Any]],
    terminal_entries: dict[int, dict[str, Any]],
    started_at: str,
    run_complete: bool,
) -> dict[str, Any]:
    theta_indices = tuple(int(value) for value in args.theta_indices)
    planned_target_rho = np.asarray(
        [plans[index]["predicted_target_rho"] for index in theta_indices], dtype=float
    )
    final_rho = np.full(len(theta_indices), np.nan)
    final_velocity = np.full(len(theta_indices), np.nan)
    accepted = np.zeros(len(theta_indices), dtype=bool)
    final_records = []
    profiles = {}
    attempt_records = []
    records = []
    for entry in entries.values():
        record = dict(entry["record"])
        records.append(record)
        attempt_records.extend(dict(item) for item in entry.get("attempt_records", []))
        if entry.get("profile") is not None:
            profiles[entry_key(record["theta_index"], record["radial_fraction_used"])] = entry["profile"]
    for position, theta_index in enumerate(theta_indices):
        terminal = terminal_entries.get(theta_index)
        if terminal is None:
            continue
        record = dict(terminal["record"])
        final_records.append(record)
        if bool(record.get("success")):
            final_rho[position] = float(record["radial_fraction_used"])
            final_velocity[position] = float(record["u_0minus"]) * SPEED_OF_LIGHT_M_S
            accepted[position] = final_velocity[position] >= float(
                args.acceptance_velocity
            )
    return {
        "run_tag": "T0plus0p1_rho5k_adaptive_bracket",
        "created_at_utc": started_at,
        "checkpoint_at_utc": utc_now(),
        "run_complete": bool(run_complete),
        "checkpoint_granularity": "every_hidden_rho_point",
        "source_domain_path": str(args.domain),
        "source_base_result_path": str(args.base_result),
        "T_0plus_MeV": DEFAULT_T0PLUS_MEV,
        "theta_indices": np.asarray(theta_indices, dtype=int),
        "rho_base": float(args.base_rho),
        "predicted_target_rho": planned_target_rho,
        "final_rho": final_rho,
        "final_u_0minus_m_s": final_velocity,
        "accepted_5000_bracket": accepted,
        "target_velocity_m_s": float(args.target_velocity),
        "acceptance_velocity_m_s": float(args.acceptance_velocity),
        "maximum_rho": float(args.maximum_rho),
        "maximum_hidden_step": float(args.maximum_hidden_step),
        "extension_step": float(args.extension_step),
        "jB_divergence_exponent": float(args.jB_exponent),
        "jB_perturbations": np.asarray(args.jB_perturbations, dtype=float),
        "trial_timeout_s": float(args.trial_timeout),
        "cell_timeout_s": float(args.cell_timeout),
        "plans": plans,
        "entries": dict(entries),
        "terminal_entries": dict(terminal_entries),
        "records": records,
        "final_records": final_records,
        "attempt_records": attempt_records,
        "profiles": profiles,
        "completed_hidden_point_count": int(len(entries)),
        "terminal_theta_count": int(len(terminal_entries)),
        "accepted_theta_count": int(np.count_nonzero(accepted)),
        "domain_calibration": domain_payload["calibration"],
        "arguments": vars(args),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Continue theta indices 7-26 beyond rho=0.86 only far enough to "
            "numerically bracket the 5000 m/s contour."
        )
    )
    result.add_argument("--theta-indices", type=lambda value: parse_number_tuple(value, int), default=DEFAULT_THETA_INDICES)
    result.add_argument("--base-rho", type=float, default=DEFAULT_BASE_RHO)
    result.add_argument("--target-velocity", type=float, default=DEFAULT_TARGET_VELOCITY_M_S)
    result.add_argument("--acceptance-velocity", type=float, default=DEFAULT_ACCEPTANCE_VELOCITY_M_S)
    result.add_argument("--maximum-rho", type=float, default=DEFAULT_MAXIMUM_RHO)
    result.add_argument("--maximum-hidden-step", type=float, default=DEFAULT_MAXIMUM_HIDDEN_STEP)
    result.add_argument("--extension-step", type=float, default=DEFAULT_EXTENSION_STEP)
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
    result.add_argument("--domain", type=Path, default=DEFAULT_DOMAIN_PATH)
    result.add_argument("--base-result", type=Path, default=DEFAULT_BASE_RESULT_PATH)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    result.add_argument("--restart", action="store_true")
    result.add_argument("--retry-failed", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def validate_args(args: argparse.Namespace) -> None:
    if tuple(sorted(set(args.theta_indices))) != tuple(args.theta_indices):
        raise ValueError("theta indices must be unique and increasing")
    if any(index < 0 or index >= 30 for index in args.theta_indices):
        raise ValueError("theta indices must lie in [0, 29]")
    if not 0.0 <= args.base_rho < args.maximum_rho < 1.0:
        raise ValueError("expected 0 <= base_rho < maximum_rho < 1")
    if args.target_velocity <= args.acceptance_velocity:
        raise ValueError("target velocity must exceed the acceptance velocity")
    if args.maximum_hidden_step <= 0.0 or args.extension_step <= 0.0:
        raise ValueError("radial steps must be positive")
    if args.workers < 1 or args.trial_timeout <= 0.0 or args.cell_timeout <= 0.0:
        raise ValueError("workers and time limits must be positive")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    validate_args(args)
    domain_payload = np.load(args.domain, allow_pickle=True).item()
    base_payload = np.load(args.base_result, allow_pickle=True).item()
    if not bool(base_payload.get("run_complete")):
        raise RuntimeError("the rho=0.86 base shell is incomplete")
    domain_points = engine.point_map(domain_payload)
    target_radial_index = next(
        int(point["radial_index"])
        for point in domain_payload["points"]
        if np.isclose(float(point["radial_fraction_used"]), 0.90)
    )
    base_entries: dict[int, dict[str, Any]] = {}
    target_points: dict[int, dict[str, Any]] = {}
    base_jB_by_theta: dict[int, float] = {}
    base_velocity_by_theta: dict[int, float] = {}
    for theta_index in range(30):
        key = engine.entry_key(theta_index, args.base_rho)
        base = base_payload["entries"].get(key)
        if base is None or not bool(base["record"].get("success")) or base.get("profile") is None:
            raise RuntimeError(
                f"theta_index={theta_index} lacks a successful rho={args.base_rho:g} profile"
            )
        base_entries[theta_index] = base
        base_jB_by_theta[theta_index] = float(base["record"]["jB"])
        base_velocity_by_theta[theta_index] = (
            float(base["record"]["u_0minus"]) * SPEED_OF_LIGHT_M_S
        )
        target_points[theta_index] = engine.enrich_domain_point(
            domain_points[(theta_index, target_radial_index)], domain_payload
        )

    plans: dict[int, dict[str, Any]] = {}
    for theta_index in args.theta_indices:
        predicted_target = predict_target_rho(
            base_velocity_by_theta[theta_index],
            base_rho=args.base_rho,
            target_velocity_m_s=args.target_velocity,
            exponent=args.jB_exponent,
            minimum_increment=args.extension_step,
            maximum_rho=args.maximum_rho,
        )
        plans[int(theta_index)] = {
            "theta_index": int(theta_index),
            "base_velocity_m_s": float(base_velocity_by_theta[theta_index]),
            "predicted_target_rho": float(predicted_target),
            "initial_ladder": build_hidden_ladder(
                base_rho=args.base_rho,
                target_rho=predicted_target,
                maximum_step=args.maximum_hidden_step,
            ),
        }

    if args.output.exists() and not args.restart:
        existing_payload = np.load(args.output, allow_pickle=True).item()
        entries = dict(existing_payload.get("entries", {}))
        terminal_entries = dict(existing_payload.get("terminal_entries", {}))
        terminal_entries = {int(key): value for key, value in terminal_entries.items()}
        started_at = str(existing_payload.get("created_at_utc", utc_now()))
        print(f"Resuming {args.output}", flush=True)
    else:
        entries = {}
        terminal_entries = {}
        started_at = utc_now()

    states: dict[int, dict[str, Any]] = {}
    for theta_index in args.theta_indices:
        existing_terminal = terminal_entries.get(theta_index)
        if existing_terminal is not None:
            terminal_succeeded = bool(existing_terminal["record"].get("success"))
            if terminal_succeeded or not args.retry_failed:
                continue
            terminal_entries.pop(theta_index)
        previous = base_entries[theta_index]
        pending = list(plans[theta_index]["initial_ladder"])
        existing_for_theta = sorted(
            (
                (
                    float(entry["record"]["radial_fraction_used"]),
                    key,
                    entry,
                )
                for key, entry in entries.items()
                if int(entry["record"]["theta_index"]) == theta_index
            ),
            key=lambda item: item[0],
        )
        for rho, key, entry in existing_for_theta:
            if not bool(entry["record"].get("success")):
                if args.retry_failed:
                    entries.pop(key, None)
                    pending = [rho] + [value for value in pending if value > rho + 1.0e-12]
                else:
                    terminal_entries[theta_index] = entry
                    pending = []
                break
            previous = entry
            pending = [value for value in pending if value > rho + 1.0e-12]
            velocity = float(entry["record"]["u_0minus"]) * SPEED_OF_LIGHT_M_S
            if velocity >= args.acceptance_velocity:
                terminal_entries[theta_index] = entry
                pending = []
                break
        if theta_index not in terminal_entries:
            if not pending:
                extension = next_extension_rho(
                    float(previous["record"]["radial_fraction_used"]),
                    increment=args.extension_step,
                    maximum_rho=args.maximum_rho,
                )
                if extension is not None:
                    pending.append(extension)
            states[theta_index] = {"previous": previous, "pending": pending}

    if args.dry_run:
        print("Adaptive rho targets:", flush=True)
        for theta_index in args.theta_indices:
            plan = plans[theta_index]
            print(
                f"theta={theta_index:02d}: u(rho=0.86)={plan['base_velocity_m_s']:.2f} m/s, "
                f"target rho={plan['predicted_target_rho']:.6f}, "
                f"ladder={tuple(round(value, 6) for value in plan['initial_ladder'])}",
                flush=True,
            )
        return 0

    common = source_runner._task_common(domain_payload)

    def checkpoint(run_complete: bool = False) -> None:
        engine.atomic_save(
            args.output,
            build_payload(
                args=args,
                domain_payload=domain_payload,
                plans=plans,
                entries=entries,
                terminal_entries=terminal_entries,
                started_at=started_at,
                run_complete=run_complete,
            ),
        )

    checkpoint(False)
    initial_total = sum(len(state["pending"]) for state in states.values())
    completed_initial = len(entries)
    context = mp.get_context("spawn")
    worker_count = min(int(args.workers), max(1, len(states)))
    with tqdm(
        total=completed_initial + initial_total,
        initial=completed_initial,
        desc="Adaptive 5000 m/s bracket",
        unit="point",
        dynamic_ncols=True,
    ) as progress, ProcessPoolExecutor(
        max_workers=worker_count, mp_context=context
    ) as executor:
        while states:
            future_to_metadata = {}
            for theta_index, state in list(states.items()):
                if not state["pending"]:
                    previous_rho = float(
                        state["previous"]["record"]["radial_fraction_used"]
                    )
                    extension = next_extension_rho(
                        previous_rho,
                        increment=args.extension_step,
                        maximum_rho=args.maximum_rho,
                    )
                    if extension is None:
                        terminal_entries[theta_index] = terminal_failure_entry(
                            theta_index=theta_index,
                            rho=previous_rho,
                            previous=state["previous"],
                            message=(
                                f"rho={previous_rho:g} reached the configured cap without "
                                f"u(0-) >= {args.acceptance_velocity:g} m/s"
                            ),
                        )
                        states.pop(theta_index)
                        checkpoint(False)
                        continue
                    state["pending"].append(extension)
                    progress.total += 1
                    progress.refresh()
                target_rho = float(state["pending"].pop(0))
                previous = state["previous"]
                previous_rho = float(previous["record"]["radial_fraction_used"])
                target = engine.interpolate_hidden_point(
                    base_entries[theta_index]["point"],
                    target_points[theta_index],
                    target_rho=target_rho,
                    synthetic_point_index=(
                        2_000_000 + theta_index * 10_000 + int(round(target_rho * 1_000_000))
                    ),
                    synthetic_radial_index=int(round(target_rho * 1_000_000)),
                )
                predicted = engine.predict_jB(
                    previous["record"]["jB"],
                    previous_rho=previous_rho,
                    target_rho=target_rho,
                    exponent=args.jB_exponent,
                )
                candidates = build_jB_candidates(
                    predicted_radial_jB=predicted,
                    perturbations=tuple(args.jB_perturbations),
                    theta_index=theta_index,
                    target_rho=target_rho,
                    base_rho=args.base_rho,
                    exponent=args.jB_exponent,
                    base_jB_by_theta=base_jB_by_theta,
                )
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
                future_to_metadata[future] = (
                    theta_index,
                    target_rho,
                    predicted,
                    target,
                    task,
                )
            if not future_to_metadata:
                continue
            for future in as_completed(future_to_metadata):
                theta_index, target_rho, predicted, target, task = future_to_metadata[future]
                key = entry_key(theta_index, target_rho)
                try:
                    output = future.result()
                except Exception as exc:
                    output = engine.failure_output(task, f"{type(exc).__name__}: {exc}")
                record = dict(output["selected_record"])
                selected_source = str(record.get("jB_guess_source", ""))
                selected_candidate = next(
                    (
                        candidate
                        for candidate in task["candidates"]
                        if np.isclose(
                            float(candidate["jB_guess"]),
                            float(record.get("jB_guess", np.nan)),
                            rtol=1.0e-8,
                            atol=0.0,
                        )
                    ),
                    None,
                )
                record.update(
                    {
                        "adaptive_rho_target": float(target_rho),
                        "predicted_jB": float(predicted),
                        "jB_divergence_exponent": float(args.jB_exponent),
                        "angular_seed_theta_indices": tuple(
                            selected_candidate.get("angular_seed_theta_indices", ())
                            if selected_candidate is not None
                            else ()
                        ),
                        "adaptive_seed_source": selected_source,
                    }
                )
                entry = {
                    "point": target,
                    "record": record,
                    "attempt_records": [dict(item) for item in output["attempt_records"]],
                    "profile": output.get("profile"),
                }
                entries[key] = entry
                progress.update(1)
                if bool(record.get("success")) and entry.get("profile") is not None:
                    states[theta_index]["previous"] = entry
                    velocity = float(record["u_0minus"]) * SPEED_OF_LIGHT_M_S
                    if velocity >= args.acceptance_velocity:
                        unused = len(states[theta_index]["pending"])
                        if unused:
                            progress.total -= unused
                            progress.refresh()
                        terminal_entries[theta_index] = entry
                        states.pop(theta_index)
                else:
                    unused = len(states[theta_index]["pending"])
                    if unused:
                        progress.total -= unused
                        progress.refresh()
                    terminal_entries[theta_index] = entry
                    states.pop(theta_index)
                checkpoint(False)
                progress.set_postfix(
                    theta=theta_index,
                    rho=f"{target_rho:.5f}",
                    status=str(record.get("status")),
                    accepted=sum(
                        bool(item["record"].get("success"))
                        and float(item["record"].get("u_0minus", 0.0))
                        * SPEED_OF_LIGHT_M_S
                        >= args.acceptance_velocity
                        for item in terminal_entries.values()
                    ),
                )
    run_complete = all(index in terminal_entries for index in args.theta_indices)
    checkpoint(run_complete)
    accepted_count = sum(
        bool(entry["record"].get("success"))
        and float(entry["record"].get("u_0minus", 0.0)) * SPEED_OF_LIGHT_M_S
        >= args.acceptance_velocity
        for entry in terminal_entries.values()
    )
    print(
        f"Done: accepted numerical brackets={accepted_count}/{len(args.theta_indices)}; "
        f"hidden points={len(entries)}",
        flush=True,
    )
    print(f"Saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
