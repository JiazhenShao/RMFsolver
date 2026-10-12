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


GUESS_SOURCE_SEMI = "semi_analytic_jB"
GUESS_SOURCE_PROFILE = "adjacent_numerical_profile"
GUESS_SOURCE_PROFILE_SEMI = "continuation_profile_with_semi_jB"


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


def _solve_uNmax_with_hard_timeout(kwargs, timeout_s):
    return run_module_call_with_hard_timeout(
        benchmark.phase_velocity.__name__,
        "solve_front_energy_conserving_uNmax",
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
    if not bool(result.get("success", True)):
        return str(result.get("message", "solver reported failure"))
    if not np.isfinite(_finite(result.get("u_0minus"))):
        return "solver returned non-finite u_0minus"
    if not np.isfinite(_finite(result.get("jB"))):
        return "solver returned non-finite jB"
    if result.get("bvp_collocation_converged") is False:
        return "BVP collocation did not converge"
    try:
        if int(result.get("bvp_status", 0)) != 0:
            return f"BVP returned status {result.get('bvp_status')}"
    except (TypeError, ValueError):
        return "BVP returned an invalid status"

    bvp_limit = float(task.get("max_bvp_residual", max(1.0e-3, 10.0 * float(task["tol_bvp"]))))
    equation_limit = float(task.get("max_equation_residual", 1.0e-2))
    temperature_limit = float(task.get("max_temperature_residual", bvp_limit))
    limits = {
        "bvp_max_rms_residual": bvp_limit,
        "T_0plus_residual": temperature_limit,
        "E_right_residual_norm": equation_limit,
        "Pi_right_residual_norm": equation_limit,
        "tail_residual_norm": equation_limit,
        "kaon_equation_residual_norm": equation_limit,
        "energy_equation_residual_norm": equation_limit,
        "momentum_equation_residual_norm": equation_limit,
    }
    for field, limit in limits.items():
        if field not in result or result[field] is None:
            continue
        value = _finite(result[field])
        if not np.isfinite(value) or abs(value) > limit:
            return f"{field}={value!r} exceeds {limit:g}"
    return None


def _record_result(
    *,
    task: dict[str, Any],
    point: dict[str, Any],
    attempt_index: int,
    guess_source: str,
    continuation_used: bool,
    is_bridge: bool,
    bridge_depth: int,
    started: float,
    result: dict[str, Any] | None,
    error: Exception | None,
) -> dict[str, Any]:
    validation_error = None if result is None else _validation_error(result, task)
    success = result is not None and error is None and validation_error is None
    message = (
        str(error)
        if error is not None
        else validation_error
        if validation_error is not None
        else str(result.get("message", ""))
    )
    record: dict[str, Any] = {
        "point_index": int(point["point_index"]),
        "theta_index": int(task["theta_index"]),
        "radial_index": int(point.get("radial_index", -1)),
        "radial_fraction": float(point["radial_fraction"]),
        "radial_fraction_used": float(point["radial_fraction_used"]),
        "T_0plus_MeV": float(task["T_0plus_MeV"]),
        "T_0minus_MeV": float(point["T_0minus_MeV"]),
        "nB_0minus_MeV3": float(point["nB_0minus_MeV3"]),
        "nB_0minus_over_n0": float(point["nB_0minus_over_n0"]),
        "attempt_index": int(attempt_index),
        "attempt_count": int(attempt_index + 1),
        "guess_source": str(guess_source),
        "continuation_direction": "outward_in_rho",
        "continuation_profile_used": bool(continuation_used),
        "is_bridge": bool(is_bridge),
        "bridge_depth": int(bridge_depth),
        "success": bool(success),
        "status": (
            "success"
            if success
            else "target_timeout"
            if isinstance(error, TargetTimeBudgetError)
            else "trial_timeout"
            if isinstance(error, BVPTrialTimeoutError)
            else "exception"
            if error
            else "validation_failure"
        ),
        "message": str(message or ""),
        "runtime_s": float(time.monotonic() - started),
        "semi_analytic_jB_seed": _finite(point.get("semi_jB")),
    }
    if result is None:
        if error is not None:
            record["traceback"] = "".join(
                traceback.format_exception_only(type(error), error)
            ).strip()
        return record
    for field in (
        "u_0minus",
        "jB",
        "a_0plus",
        "a_inf",
        "lambda_n",
        "bvp_max_rms_residual",
        "E_right_residual_norm",
        "Pi_right_residual_norm",
        "T_0plus_residual",
        "tail_residual_norm",
        "kaon_equation_residual_norm",
        "energy_equation_residual_norm",
        "momentum_equation_residual_norm",
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
            "bvp_collocation_converged": result.get("bvp_collocation_converged"),
            "solver_variant": str(result.get("solver_variant", benchmark.SOLVER_VARIANT)),
            "root_method": str(result.get("_root_method", benchmark.ROOT_METHOD)),
        }
    )
    return record


def _attempt_point(
    task: dict[str, Any],
    point: dict[str, Any],
    previous_result: dict[str, Any] | None,
    *,
    is_bridge: bool,
    bridge_depth: int,
    deadline: float,
) -> tuple[dict[str, Any] | None, dict[str, Any], list[dict[str, Any]]]:
    semi_jB = _finite(point.get("semi_jB"))
    if previous_result is None:
        plan = [(GUESS_SOURCE_SEMI, False, semi_jB)]
    else:
        plan = [
            (GUESS_SOURCE_PROFILE, True, _finite(previous_result.get("jB"))),
            (GUESS_SOURCE_PROFILE_SEMI, True, semi_jB),
            (GUESS_SOURCE_SEMI, False, semi_jB),
        ]
    records: list[dict[str, Any]] = []
    for attempt_index, (source, use_profile, jB_guess) in enumerate(plan):
        remaining_s = float(deadline - time.monotonic())
        if remaining_s <= 0.0:
            started = time.monotonic()
            record = _record_result(
                task=task,
                point=point,
                attempt_index=attempt_index,
                guess_source=source,
                continuation_used=use_profile,
                is_bridge=is_bridge,
                bridge_depth=bridge_depth,
                started=started,
                result=None,
                error=TargetTimeBudgetError(
                    f"contour cell exhausted its hard {float(task['target_timeout_s']):g} s budget"
                ),
            )
            records.append(record)
            _emit(
                task,
                "attempt_completed",
                theta_index=int(task["theta_index"]),
                radial_fraction=float(point["radial_fraction_used"]),
                success=False,
                is_bridge=bool(is_bridge),
            )
            return None, record, records
        if attempt_index:
            _emit(task, "retry_scheduled", theta_index=int(task["theta_index"]))
        bounds = (
            float(task["jB_lower_bound"]),
            float(point["nB_0minus_MeV3"] * float(task["jB_upper_factor"])),
        )
        kwargs: dict[str, Any] = {
            "T_0minus": float(point["T_0minus_MeV"]),
            "nB_0minus": float(point["nB_0minus_MeV3"]),
            "B_one_forth": float(task["B14_MeV"]),
            "T_0plus": float(task["T_0plus_MeV"]),
            "ms": float(task["ms_MeV"]),
            "param": benchmark.para.paraQMCRMF3,
            "NM_type": str(task["nm_type"]),
            "tail_eps": float(task["tail_eps"]),
            "n_mesh": int(task["n_mesh"]),
            "tol_bvp": float(task["tol_bvp"]),
            "max_nodes": int(task["max_nodes"]),
            "jB_bounds": bounds,
            "return_profile": True,
            "verb": False,
        }
        if use_profile:
            kwargs["continuation_guess"] = previous_result
        if np.isfinite(jB_guess):
            kwargs["jB_guess"] = float(jB_guess)
        started = time.monotonic()
        result = None
        error = None
        try:
            result = _solve_uNmax_with_hard_timeout(
                kwargs,
                min(float(task["point_timeout_s"]), remaining_s),
            )
        except Exception as exc:
            error = exc
        record = _record_result(
            task=task,
            point=point,
            attempt_index=attempt_index,
            guess_source=source,
            continuation_used=use_profile,
            is_bridge=is_bridge,
            bridge_depth=bridge_depth,
            started=started,
            result=result,
            error=error,
        )
        records.append(record)
        _emit(
            task,
            "attempt_completed",
            theta_index=int(task["theta_index"]),
            radial_fraction=float(point["radial_fraction_used"]),
            success=bool(record["success"]),
            is_bridge=bool(is_bridge),
        )
        if record["success"]:
            return result, record, records
    return None, records[-1], records


def _interpolate_point(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    target_point_index: int,
    bridge_depth: int,
) -> dict[str, Any]:
    fraction = 0.5 * (
        float(left["radial_fraction_used"])
        + float(right["radial_fraction_used"])
    )
    left_fraction = float(left["radial_fraction_used"])
    right_fraction = float(right["radial_fraction_used"])
    weight = (fraction - left_fraction) / (right_fraction - left_fraction)

    def interpolate(field: str) -> float:
        return float(left[field] + weight * (right[field] - left[field]))

    return {
        "point_index": int(target_point_index),
        "theta_index": int(left["theta_index"]),
        "theta_rad": float(left["theta_rad"]),
        "radial_index": -1,
        "radial_fraction": fraction,
        "radial_fraction_used": fraction,
        "rho": interpolate("rho"),
        "rho_inner_boundary": float(left["rho_inner_boundary"]),
        "rho_outer_boundary": float(left["rho_outer_boundary"]),
        "T_0minus_MeV": interpolate("T_0minus_MeV"),
        "nB_0minus_over_n0": interpolate("nB_0minus_over_n0"),
        "nB_0minus_MeV3": interpolate("nB_0minus_MeV3"),
        "semi_jB": interpolate("semi_jB"),
        "bridge_depth": int(bridge_depth),
    }


def _advance_with_bridges(
    task: dict[str, Any],
    previous_point: dict[str, Any],
    previous_result: dict[str, Any],
    target: dict[str, Any],
    depth: int,
    deadline: float,
) -> tuple[dict[str, Any] | None, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    result, selected, attempts = _attempt_point(
        task,
        target,
        previous_result,
        is_bridge=bool(target.get("radial_index", -1) < 0),
        bridge_depth=depth,
        deadline=deadline,
    )
    bridge_records: list[dict[str, Any]] = []
    if result is not None:
        return result, selected, attempts, bridge_records

    step = float(target["radial_fraction_used"] - previous_point["radial_fraction_used"])
    if (
        time.monotonic() >= deadline
        or depth >= int(task["max_bridge_depth"])
        or step <= float(task["min_bridge_fraction_step"])
    ):
        return None, selected, attempts, bridge_records

    midpoint = _interpolate_point(
        previous_point,
        target,
        target_point_index=int(target["point_index"]),
        bridge_depth=depth + 1,
    )
    _emit(
        task,
        "bridge_scheduled",
        theta_index=int(task["theta_index"]),
        radial_fraction=float(midpoint["radial_fraction_used"]),
        bridge_depth=depth + 1,
    )
    mid_result, mid_selected, mid_attempts, nested = _advance_with_bridges(
        task,
        previous_point,
        previous_result,
        midpoint,
        depth + 1,
        deadline,
    )
    attempts.extend(mid_attempts)
    bridge_records.extend(nested)
    bridge_records.append(mid_selected)
    if mid_result is None:
        return None, selected, attempts, bridge_records

    final_result, final_selected, final_attempts, final_bridges = _advance_with_bridges(
        task,
        midpoint,
        mid_result,
        target,
        depth + 1,
        deadline,
    )
    attempts.extend(final_attempts)
    bridge_records.extend(final_bridges)
    return final_result, final_selected, attempts, bridge_records


def run_radial_trajectory(task: dict[str, Any]) -> dict[str, Any]:
    """Walk one theta ray from the enthalpy boundary toward the cold A=1 edge."""
    points = sorted(
        (dict(point) for point in task["points"]),
        key=lambda point: float(point["radial_fraction_used"]),
    )
    target_records: list[dict[str, Any]] = []
    attempt_records: list[dict[str, Any]] = []
    bridge_records: list[dict[str, Any]] = []
    profiles: dict[int, dict[str, np.ndarray]] = {}
    previous_point = None
    previous_result = None
    started = time.monotonic()

    for point in points:
        target_deadline = time.monotonic() + float(task["target_timeout_s"])
        if previous_result is None:
            result, selected, attempts = _attempt_point(
                task,
                point,
                None,
                is_bridge=False,
                bridge_depth=0,
                deadline=target_deadline,
            )
            bridges: list[dict[str, Any]] = []
        else:
            result, selected, attempts, bridges = _advance_with_bridges(
                task,
                previous_point,
                previous_result,
                point,
                0,
                target_deadline,
            )
        selected = dict(selected)
        selected["attempt_count"] = len(attempts)
        selected["retry_replaced_failure"] = bool(
            selected["success"] and len(attempts) > 1
        )
        target_records.append(selected)
        attempt_records.extend(attempts)
        bridge_records.extend(bridges)
        if result is not None:
            profiles[int(point["point_index"])] = benchmark._extract_profile(result)
            previous_point = point
            previous_result = result
        else:
            previous_point = None
            previous_result = None
        _emit(
            task,
            "target_completed",
            theta_index=int(task["theta_index"]),
            point_index=int(point["point_index"]),
            radial_fraction=float(point["radial_fraction_used"]),
            success=bool(selected["success"]),
            attempt_count=len(attempts),
            bridge_count=len(bridges),
            target_record=selected,
            attempt_records=attempts,
            bridge_records=bridges,
        )

    return {
        "theta_index": int(task["theta_index"]),
        "theta_rad": float(task["theta_rad"]),
        "target_records": target_records,
        "attempt_records": attempt_records,
        "bridge_records": bridge_records,
        "profiles": profiles,
        "runtime_s": float(time.monotonic() - started),
    }
