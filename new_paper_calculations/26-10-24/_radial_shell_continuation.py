#!/usr/bin/env python3
"""Outward full-profile continuation for the fixed-T(0+) numerical scan."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import time
import traceback
from collections import defaultdict
from typing import Any

import numpy as np

import _method_engine as benchmark
import _slow_front_adapter as slow


GUESS_SOURCE_PROFILE = "adjacent_numerical_profile"
GUESS_SOURCE_PROFILE_EXTRAPOLATED = "adjacent_profile_extrapolated_jB"
GUESS_SOURCE_PROFILE_MULTIPLIER = "adjacent_profile_multiplier_jB"
GUESS_SOURCE_THETA_LEFT = "same_shell_theta_left_jB"
GUESS_SOURCE_THETA_RIGHT = "same_shell_theta_right_jB"
GUESS_SOURCE_THETA_INTERPOLATED = "same_shell_theta_interpolated_jB"
GUESS_SOURCE_FIRST_SHELL = "first_shell_numerical_bootstrap"
GUESS_SOURCE_REFERENCE_SCALED = "matched_T0plus1_numerical_jB_scaled"
GUESS_SOURCE_REFERENCE_RAW = "matched_T0plus1_numerical_jB"


def first_shell_candidates() -> tuple[float, ...]:
    """Bootstrap the first shell inside the measured 0.1 MeV branch."""
    return (0.4, 0.3, 0.5, 0.6, 0.8)


def matched_reference_candidates(
    reference_jB: float,
    *,
    scale: float,
) -> tuple[dict[str, Any], ...]:
    """Seed only from the same-state saved relativistic numerical flux."""
    if not np.isfinite(reference_jB) or reference_jB <= 0:
        raise ValueError("Missing positive relativistic reference flux")
    values = deduplicate_jB_candidates([scale * reference_jB, reference_jB,
                                        0.75 * reference_jB, 1.25 * reference_jB])
    return tuple(dict(jB_guess=value, source="saved_relativistic_numerical_flux",
                      use_profile=False) for value in values)



def previous_shell_multipliers() -> tuple[float, ...]:
    """Growth factors for a finer outward radial step."""
    return (1.15, 1.25, 1.35, 1.50)


def immediate_theta_seed(
    values_by_theta: dict[int, float],
    *,
    target_theta_index: int,
) -> dict[str, Any] | None:
    """Build a current seed from successful immediate angular neighbors only."""
    target = int(target_theta_index)
    left = _finite(values_by_theta.get(target - 1))
    right = _finite(values_by_theta.get(target + 1))
    left_ok = bool(np.isfinite(left) and left > 0.0)
    right_ok = bool(np.isfinite(right) and right > 0.0)
    if left_ok and right_ok:
        return {
            "jB": float(0.5 * (left + right)),
            "theta_indices": (target - 1, target + 1),
            "mode": "two_sided_interpolated",
        }
    if left_ok:
        return {
            "jB": float(left),
            "theta_indices": (target - 1,),
            "mode": "one_sided_left",
        }
    if right_ok:
        return {
            "jB": float(right),
            "theta_indices": (target + 1,),
            "mode": "one_sided_right",
        }
    return None


def extrapolated_same_ray_jB(
    *,
    previous_jB: float,
    earlier_jB: float | None,
    previous_rho: float,
    earlier_rho: float | None,
    target_rho: float,
) -> float:
    """Log-extrapolate the same-ray current and cap unstable growth."""
    previous = _finite(previous_jB)
    earlier = _finite(earlier_jB)
    previous_rho = _finite(previous_rho)
    earlier_rho = _finite(earlier_rho)
    target_rho = _finite(target_rho)
    if not np.isfinite(previous) or previous <= 0.0:
        return np.nan
    if (
        not np.isfinite(earlier)
        or earlier <= 0.0
        or not np.isfinite(earlier_rho)
        or previous_rho <= earlier_rho
        or target_rho <= previous_rho
    ):
        return previous
    log_slope = (np.log(previous) - np.log(earlier)) / (previous_rho - earlier_rho)
    predicted = previous * np.exp(log_slope * (target_rho - previous_rho))
    return float(np.clip(predicted, 1.05 * previous, 1.55 * previous))


def deduplicate_jB_candidates(
    candidates: list[float] | tuple[float, ...],
    *,
    relative_tolerance: float = 0.03,
) -> tuple[float, ...]:
    """Keep positive finite guesses that are materially distinct."""
    unique: list[float] = []
    for candidate in candidates:
        value = _finite(candidate)
        if not np.isfinite(value) or value <= 0.0:
            continue
        if any(abs(value - prior) / max(value, prior) < relative_tolerance for prior in unique):
            continue
        unique.append(float(value))
    return tuple(unique)


class BVPTrialTimeoutError(TimeoutError):
    """Raised after forcibly stopping one over-budget BVP child process."""


class TargetTimeBudgetError(TimeoutError):
    """Raised when retries and bridges exhaust one contour cell's budget."""


class RemoteBVPTrialError(RuntimeError):
    """Represent an exception raised inside the disposable BVP child."""


def _module_call_child(connection, module_name, function_name, args, kwargs):
    """Execute one importable call and serialize its outcome to the parent."""
    try:
        function = getattr(importlib.import_module(module_name), function_name)
        connection.send(("result", function(*args, **kwargs)))
    except BaseException as exc:
        connection.send(
            (
                "error",
                f"{type(exc).__module__}.{type(exc).__name__}",
                str(exc),
                traceback.format_exc(limit=8),
            )
        )
    finally:
        connection.close()


def _stop_process(process, grace_s=2.0):
    """Terminate a child and escalate to SIGKILL if it does not exit."""
    if not process.is_alive():
        process.join(timeout=0.0)
        return
    process.terminate()
    process.join(timeout=float(grace_s))
    if process.is_alive():
        process.kill()
        process.join(timeout=float(grace_s))


def run_module_call_with_hard_timeout(
    module_name,
    function_name,
    *,
    args=(),
    kwargs=None,
    timeout_s,
):
    """Run one call in a killable subprocess with a real wall-clock limit."""
    timeout_s = float(timeout_s)
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("hard timeout must be positive and finite")
    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_module_call_child,
        args=(
            send_connection,
            str(module_name),
            str(function_name),
            tuple(args),
            dict(kwargs or {}),
        ),
    )
    process.start()
    send_connection.close()
    try:
        if not receive_connection.poll(timeout_s):
            _stop_process(process)
            raise BVPTrialTimeoutError(
                f"BVP trial exceeded its hard {timeout_s:g} s wall-clock limit"
            )
        try:
            payload = receive_connection.recv()
        except EOFError as exc:
            process.join(timeout=2.0)
            raise RemoteBVPTrialError(
                f"BVP child exited with code {process.exitcode} without a result"
            ) from exc
        process.join(timeout=2.0)
        if process.is_alive():
            _stop_process(process)
        if payload[0] == "result":
            return payload[1]
        _, exception_type, message, remote_traceback = payload
        raise RemoteBVPTrialError(
            f"{exception_type}: {message}\nRemote traceback:\n{remote_traceback}"
        )
    finally:
        receive_connection.close()
        if process.is_alive():
            _stop_process(process)


def _solve_slow_with_hard_timeout(kwargs, timeout_s):
    return run_module_call_with_hard_timeout(
        slow.__name__,
        "solve_with_path_refinement",
        kwargs=kwargs,
        timeout_s=timeout_s,
    )


def inclusive_theta_axis(ray_count: int) -> np.ndarray:
    """Return equally spaced rays including the density and temperature axes."""
    count = int(ray_count)
    if count < 2:
        raise ValueError("ray_count must include both angular endpoints")
    return np.linspace(0.0, 0.5 * np.pi, count)


def group_points_by_theta(points: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Group regular cells into strictly outward radial trajectories."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        grouped[int(point["theta_index"])].append(dict(point))
    return {
        theta_index: sorted(
            ray,
            key=lambda point: (
                float(point["radial_fraction_used"]),
                int(point["radial_index"]),
            ),
        )
        for theta_index, ray in sorted(grouped.items())
    }


def group_points_by_radial(points: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Group cells into ordered radial shells and sort each shell by theta."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        grouped[int(point["radial_index"])].append(dict(point))
    return {
        radial_index: sorted(shell, key=lambda point: int(point["theta_index"]))
        for radial_index, shell in sorted(grouped.items())
    }


def primary_candidates(
    point: dict[str, Any],
    *,
    previous: dict[str, Any] | None,
    earlier: dict[str, Any] | None,
    reference_jB: float = np.nan,
    reference_scale: float = 1.0,
) -> tuple[dict[str, Any], ...]:
    """Build same-ray candidates without crossing a failed predecessor."""
    if previous is None:
        return matched_reference_candidates(
            reference_jB,
            scale=reference_scale,
        )
    previous_jB = _finite(previous.get("record", {}).get("jB"))
    earlier_jB = (
        _finite(earlier.get("record", {}).get("jB")) if earlier is not None else np.nan
    )
    predicted = extrapolated_same_ray_jB(
        previous_jB=previous_jB,
        earlier_jB=earlier_jB,
        previous_rho=float(previous["point"]["radial_fraction_used"]),
        earlier_rho=(
            float(earlier["point"]["radial_fraction_used"])
            if earlier is not None
            else None
        ),
        target_rho=float(point["radial_fraction_used"]),
    )
    raw = [
        (predicted, GUESS_SOURCE_PROFILE_EXTRAPOLATED),
        *(
            (previous_jB * multiplier, GUESS_SOURCE_PROFILE_MULTIPLIER)
            for multiplier in previous_shell_multipliers()
        ),
    ]
    values = deduplicate_jB_candidates([value for value, _ in raw])
    candidates: list[dict[str, Any]] = []
    for value in values:
        source = next(
            source
            for raw_value, source in raw
            if abs(value - raw_value) / max(value, raw_value) < 0.03
        )
        candidates.append(
            {"jB_guess": value, "source": source, "use_profile": True}
        )
    # Hidden bridge points have a numerical predecessor but no same-state
    # historical solve. Their continuation alone supplies the candidate flux.
    if np.isfinite(reference_jB) and reference_jB > 0:
        candidates.extend(matched_reference_candidates(reference_jB, scale=reference_scale))
    return tuple(candidates)


def split_radial_candidates(
    candidates: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Reserve the best radial candidate for the quick primary pass."""
    ordered = tuple(dict(candidate) for candidate in candidates)
    return ordered[:1], ordered[1:]


def angular_wave_candidates(
    seed: dict[str, Any] | None,
    *,
    use_profile: bool,
    wave_index: int,
    multipliers: tuple[float, ...] = (1.0,),
) -> tuple[dict[str, Any], ...]:
    """Build bounded candidates from one strict same-shell angular seed."""
    if not seed:
        return ()
    jB = _finite(seed.get("jB"))
    if not np.isfinite(jB) or jB <= 0.0:
        return ()
    mode = str(seed.get("mode", ""))
    source = {
        "one_sided_left": GUESS_SOURCE_THETA_LEFT,
        "one_sided_right": GUESS_SOURCE_THETA_RIGHT,
        "two_sided_interpolated": GUESS_SOURCE_THETA_INTERPOLATED,
    }.get(mode)
    if source is None:
        raise ValueError(f"unknown angular seed mode: {mode}")
    theta_indices = tuple(int(index) for index in seed.get("theta_indices", ()))
    return tuple(
        {
            "jB_guess": float(jB * multiplier),
            "source": source,
            "use_profile": bool(use_profile),
            "angular_wave_index": int(wave_index),
            "angular_seed_theta_indices": theta_indices,
            "angular_seed_jB": float(jB),
            "angular_seed_mode": mode,
        }
        for multiplier in multipliers
        if np.isfinite(multiplier) and multiplier > 0.0
    )


def run_cell_task(task: dict[str, Any]) -> dict[str, Any]:
    """Importable worker entry point for one bounded physical cell."""
    task = dict(task)
    task["cell_deadline"] = time.monotonic() + float(task["cell_timeout_s"])
    return attempt_jB_candidates(
        task,
        tuple(task.get("candidates", ())),
        continuation_profile=task.get("continuation_profile"),
    )


def _bridge_midpoint(
    left: dict[str, Any], right: dict[str, Any], *, depth: int
) -> dict[str, Any]:
    """Interpolate one hidden continuation point between physical cells."""
    midpoint = dict(right)
    for field in (
        "radial_fraction",
        "radial_fraction_used",
        "rho",
        "T_0minus_MeV",
        "nB_0minus_over_n0",
        "nB_0minus_MeV3",
    ):
        midpoint[field] = 0.5 * (float(left[field]) + float(right[field]))
    midpoint["radial_index"] = -1
    midpoint["bridge_depth"] = int(depth)
    return midpoint


def run_bridge_task(task: dict[str, Any]) -> dict[str, Any]:
    """Recover one failed target through bounded midpoint continuation."""
    task = dict(task)
    target = dict(task["point"])
    deadline = time.monotonic() + float(task["cell_timeout_s"])
    max_depth = int(task.get("max_bridge_depth", 0))
    minimum_step = float(task.get("min_bridge_fraction_step", 1.0e-3))
    attempt_records: list[dict[str, Any]] = []
    bridge_records: list[dict[str, Any]] = []

    def advance(
        left_entry: dict[str, Any],
        right_point: dict[str, Any],
        depth: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        step = float(
            right_point["radial_fraction_used"]
            - left_entry["point"]["radial_fraction_used"]
        )
        if (
            depth >= max_depth
            or step <= minimum_step
            or time.monotonic() >= deadline
        ):
            return None, None
        midpoint = _bridge_midpoint(left_entry["point"], right_point, depth=depth + 1)
        midpoint_candidates = primary_candidates(
            midpoint, previous=left_entry, earlier=None
        )[:2]
        midpoint_task = {**task, "point": midpoint, "cell_deadline": deadline}
        midpoint_output = attempt_jB_candidates(
            midpoint_task,
            midpoint_candidates,
            continuation_profile=left_entry["profile"],
        )
        attempt_records.extend(midpoint_output["attempt_records"])
        bridge_records.append(dict(midpoint_output["selected_record"]))
        if not bool(midpoint_output["selected_record"].get("success")):
            recovered_entry, _ = advance(left_entry, midpoint, depth + 1)
            if recovered_entry is None:
                return None, midpoint_output
            midpoint_candidates = primary_candidates(
                midpoint, previous=recovered_entry, earlier=left_entry
            )[:2]
            midpoint_task = {**task, "point": midpoint, "cell_deadline": deadline}
            midpoint_output = attempt_jB_candidates(
                midpoint_task,
                midpoint_candidates,
                continuation_profile=recovered_entry["profile"],
            )
            attempt_records.extend(midpoint_output["attempt_records"])
            bridge_records.append(dict(midpoint_output["selected_record"]))
            if not bool(midpoint_output["selected_record"].get("success")):
                return None, midpoint_output
        midpoint_entry = {
            "point": midpoint,
            "record": midpoint_output["selected_record"],
            "profile": midpoint_output["profile"],
        }
        target_candidates = primary_candidates(
            right_point, previous=midpoint_entry, earlier=left_entry
        )[:2]
        target_task = {**task, "point": right_point, "cell_deadline": deadline}
        target_output = attempt_jB_candidates(
            target_task,
            target_candidates,
            continuation_profile=midpoint_entry["profile"],
        )
        attempt_records.extend(target_output["attempt_records"])
        if bool(target_output["selected_record"].get("success")):
            return (
                {
                    "point": right_point,
                    "record": target_output["selected_record"],
                    "profile": target_output["profile"],
                },
                target_output,
            )
        recovered, recovered_output = advance(midpoint_entry, right_point, depth + 1)
        return recovered, recovered_output or target_output

    entry, selected_output = advance(dict(task["previous_entry"]), target, 0)
    if entry is not None and selected_output is not None:
        selected_record = dict(entry["record"])
        profile = entry["profile"]
        result = selected_output.get("result")
    else:
        candidate = {
            "jB_guess": _finite(task["previous_entry"]["record"].get("jB")),
            "source": "midpoint_bridge_exhausted",
            "use_profile": True,
        }
        failure_task = {**task, "point": target}
        selected_record = _candidate_record(
            failure_task,
            candidate,
            attempt_index=len(attempt_records),
            started=time.monotonic(),
            status=("cell_timeout" if time.monotonic() >= deadline else "failure"),
            message="bounded midpoint continuation did not reach the target",
        )
        profile = None
        result = None
    return {
        "selected_record": selected_record,
        "attempt_records": attempt_records,
        "bridge_records": bridge_records,
        "result": result,
        "profile": profile,
    }


def _emit(task: dict[str, Any], event: str, **values: Any) -> None:
    progress_queue = task.get("progress_queue")
    if progress_queue is not None:
        progress_queue.put({"event": str(event), **values})


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _validation_error(result: dict[str, Any], task: dict[str, Any]) -> str | None:
    return slow.validation_error(result, task)


def _solver_kwargs_for_candidate(task, candidate, continuation_profile):
    return slow.solver_kwargs(task, candidate, continuation_profile, benchmark.para.paraQMCRMF3)



def _candidate_record(
    task: dict[str, Any],
    candidate: dict[str, Any],
    *,
    attempt_index: int,
    started: float,
    status: str,
    message: str,
    result: dict[str, Any] | None = None,
    child_terminated: bool = False,
) -> dict[str, Any]:
    point = task["point"]
    record: dict[str, Any] = {
        "point_index": int(point["point_index"]),
        "theta_index": int(point["theta_index"]),
        "radial_index": int(point["radial_index"]),
        "radial_fraction": float(point["radial_fraction"]),
        "radial_fraction_used": float(point["radial_fraction_used"]),
        "T_0plus_MeV": float(task["T_0plus_MeV"]),
        "T_0minus_MeV": float(point["T_0minus_MeV"]),
        "nB_0minus_MeV3": float(point["nB_0minus_MeV3"]),
        "nB_0minus_over_n0": float(point["nB_0minus_over_n0"]),
        "attempt_index": int(attempt_index),
        "jB_guess": float(candidate.get("jB_guess", np.nan)),
        "jB_guess_source": str(candidate.get("source", "unknown")),
        "guess_source": str(candidate.get("source", "unknown")),
        "continuation_profile_used": bool(candidate.get("use_profile", False)),
        "trial_timeout_s": float(task["trial_timeout_s"]),
        "cell_timeout_s": float(task["cell_timeout_s"]),
        "runtime_s": float(time.monotonic() - started),
        "status": str(status),
        "success": bool(status == "success"),
        "message": str(message),
        "child_terminated": bool(child_terminated),
        "angular_wave_index": int(candidate.get("angular_wave_index", -1)),
        "angular_seed_theta_indices": tuple(
            int(index) for index in candidate.get("angular_seed_theta_indices", ())
        ),
        "angular_seed_jB": _finite(candidate.get("angular_seed_jB")),
        "angular_seed_mode": str(candidate.get("angular_seed_mode", "none")),
        "radial_primary_budget_s": float(task.get("radial_primary_budget_s", np.nan)),
        "angular_retry_budget_s": float(task.get("angular_retry_budget_s", np.nan)),
    }
    record.update(hydrodynamic_model=slow.MODEL, closure="adiabatic",
                  solver="solve_front_adiabatic_slow",
                  alpha_s=0.4, rate_model="exact_nonleptonic",
                  rate_approximation="madsen_muB_over_3",
                  relativistic_jB_guess=point.get("relativistic_jB_guess"),
                  source_grid_index=point.get("source_grid_index"))
    if result is not None:
        record.update({key: result.get(key) for key in slow.RESULT_METADATA})
        for field in (
            "u_0minus",
            "jB",
            "a_0plus",
            "a_inf",
            "lambda_inf",
            "closure_error_max",
            "path_nodes",
            "path_error",
            "constitutive_residual_norm",
            "reaction_residual_norm",
            "diffusion_integral_residual_norm",
            "bvp_max_rms_residual",
        ):
            record[field] = _finite(result.get(field))
        record.update(
            {
                "T_inf_MeV": _finite(result.get("T_inf")),
                "bvp_status": result.get("bvp_status"),
                "bvp_message": str(result.get("bvp_message", "")),
                "bvp_nodes": result.get("bvp_nodes"),
                "bvp_niter": result.get("bvp_niter"),
                "bvp_parameters": np.asarray(result.get("bvp_parameters", []), dtype=float),
                "bvp_collocation_converged": result.get("bvp_status") == 0,
                "solver_variant": str(result.get("solver_variant", benchmark.SOLVER_VARIANT)),
                "root_method": str(result.get("_root_method", benchmark.ROOT_METHOD)),
            }
        )
    return record


def attempt_jB_candidates(
    task: dict[str, Any],
    candidates: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    continuation_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """Try bounded numerical guesses until one validated BVP succeeds."""
    deadline = float(
        task.get("cell_deadline", time.monotonic() + float(task["cell_timeout_s"]))
    )
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_result: dict[str, Any] | None = None
    for attempt_index, candidate in enumerate(candidates):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            started = time.monotonic()
            selected = _candidate_record(
                task,
                candidate,
                attempt_index=attempt_index,
                started=started,
                status="cell_timeout",
                message=f"cell exhausted its hard {float(task['cell_timeout_s']):g} s budget",
            )
            attempts.append(selected)
            break
        started = time.monotonic()
        result = None
        try:
            result = _solve_slow_with_hard_timeout(
                _solver_kwargs_for_candidate(task, candidate, continuation_profile),
                min(float(task["trial_timeout_s"]), remaining),
            )
            validation_error = _validation_error(result, task)
            status = "success" if validation_error is None else "failure"
            message = str(result.get("message", "")) if validation_error is None else validation_error
            selected = _candidate_record(
                task,
                candidate,
                attempt_index=attempt_index,
                started=started,
                status=status,
                message=message,
                result=result,
            )
            attempts.append(selected)
            if validation_error is None:
                selected_result = result
                break
        except BVPTrialTimeoutError as exc:
            selected = _candidate_record(
                task,
                candidate,
                attempt_index=attempt_index,
                started=started,
                status="trial_timeout",
                message=str(exc),
                child_terminated=True,
            )
            attempts.append(selected)
        except Exception as exc:
            selected = _candidate_record(
                task,
                candidate,
                attempt_index=attempt_index,
                started=started,
                status="failure",
                message=str(exc),
            )
            attempts.append(selected)
    if selected is None:
        candidate = {"jB_guess": np.nan, "source": "no_candidate", "use_profile": False}
        started = time.monotonic()
        selected = _candidate_record(
            task,
            candidate,
            attempt_index=0,
            started=started,
            status="failure",
            message="no finite jB candidates were available",
        )
        attempts.append(selected)
    return {
        "selected_record": selected,
        "attempt_records": attempts,
        "result": selected_result,
        "profile": (
            benchmark._extract_profile(selected_result)
            if selected_result is not None
            else None
        ),
    }
