"""Cluster execution helpers for analytical and numerical isothermal contours."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
from typing import Any, Callable

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np

import _isothermal_domain as domain


DEFAULT_TRIAL_TIMEOUT_S = 180.0
DEFAULT_CELL_TIMEOUT_S = 900.0
DEFAULT_ANALYTIC_TIMEOUT_S = 300.0
DEFAULT_JB_LOWER_BOUND = 1.0e-12
DEFAULT_JB_UPPER_FACTOR = 1.0e-4

DEFAULT_TAIL_EPS = 1.0e-8
DEFAULT_N_MESH = 300
DEFAULT_BVP_TOL = 1.0e-4
DEFAULT_MAX_NODES = 50000
DEFAULT_KAPPA_FACTOR = 1.0


class BVPTrialTimeoutError(TimeoutError):
    pass


class RemoteBVPTrialError(RuntimeError):
    pass


def discover_workers(requested: int | None = None) -> int:
    if requested is not None:
        return max(1, int(requested))
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


def _module_call_child(connection, module_name: str, function_name: str, kwargs: dict[str, Any]) -> None:
    try:
        function = getattr(importlib.import_module(module_name), function_name)
        connection.send(("result", function(**kwargs)))
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


def _stop_process(process: mp.Process, grace_s: float = 2.0) -> bool:
    terminated = False
    if process.is_alive():
        terminated = True
        process.terminate()
        process.join(timeout=float(grace_s))
    if process.is_alive():
        process.kill()
        process.join(timeout=float(grace_s))
    else:
        process.join(timeout=0.0)
    return terminated


def run_module_call_with_hard_timeout(
    module_name: str,
    function_name: str,
    kwargs: dict[str, Any],
    *,
    timeout_s: float,
) -> Any:
    timeout_s = float(timeout_s)
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("hard timeout must be positive and finite")
    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_module_call_child,
        args=(send_connection, str(module_name), str(function_name), dict(kwargs)),
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


def _append_candidate(
    candidates: list[tuple[float, str]],
    value: float,
    source: str,
    lower_bound: float,
    upper_bound: float,
) -> None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if not np.isfinite(value) or value <= 0.0:
        return
    value = float(np.clip(value, lower_bound * (1.0 + 1.0e-8), upper_bound * (1.0 - 1.0e-8)))
    for existing, _ in candidates:
        if abs(value - existing) / max(abs(value), abs(existing)) < 0.03:
            return
    candidates.append((value, str(source)))


def build_jB_candidates(
    *,
    nB_0minus: float,
    previous_jB: float | None,
    earlier_jB: float | None,
    analytic_jB: float | None,
    nearby_shell_jB: tuple[float, ...] | list[float],
    lower_bound: float,
    upper_bound: float,
) -> list[tuple[float, str]]:
    candidates: list[tuple[float, str]] = []
    _append_candidate(candidates, previous_jB, "previous_composition_shell", lower_bound, upper_bound)
    extrapolated = np.nan
    if previous_jB is not None and earlier_jB is not None:
        try:
            extrapolated = float(previous_jB) ** 2 / float(earlier_jB)
        except (TypeError, ValueError, ZeroDivisionError):
            extrapolated = np.nan
        _append_candidate(candidates, extrapolated, "log_shell_extrapolation", lower_bound, upper_bound)
    _append_candidate(candidates, analytic_jB, "analytic_current_cell", lower_bound, upper_bound)
    bases = (
        (previous_jB, "previous_shell"),
        (extrapolated, "log_extrapolation"),
        (analytic_jB, "analytic"),
    )
    for base, base_source in bases:
        if base is None:
            continue
        for factor in (0.5, 0.8, 1.25, 2.0):
            _append_candidate(
                candidates,
                float(base) * factor,
                f"{base_source}_multiplier_{factor:g}",
                lower_bound,
                upper_bound,
            )
    nearby = np.asarray(nearby_shell_jB, dtype=float)
    nearby = nearby[np.isfinite(nearby) & (nearby > 0.0)]
    if nearby.size:
        median = float(np.median(nearby))
        for factor in (0.8, 1.0, 1.25):
            _append_candidate(
                candidates,
                median * factor,
                f"nearby_temperature_median_{factor:g}",
                lower_bound,
                upper_bound,
            )
    _append_candidate(
        candidates,
        1.0e-6 * float(nB_0minus),
        "default_nB_scaled",
        lower_bound,
        upper_bound,
    )
    return candidates


def run_numerical_trial(
    task: dict[str, Any],
    jB_guess: float,
    jB_guess_source: str,
    *,
    solver: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    from RMFsolver import RMFparameter as para
    if solver is None:
        from RMFsolver import phase_velocity as pv

        solver = pv.solve_front_isothermal
    base = {
        "temperature_index": int(task["temperature_index"]),
        "composition_index": int(task["composition_index"]),
        "a_0plus_target": float(task["a_0plus_target"]),
        "T_0minus": float(task["T"]),
        "nB_0minus": float(task["nB_0minus"]),
        "jB_guess": float(jB_guess),
        "jB_guess_source": str(jB_guess_source),
        "velocity_m_s": np.nan,
    }
    try:
        result = dict(
            solver(
                float(task["T"]),
                float(task["nB_0minus"]),
                float(task["B_one_forth"]),
                ms=float(task["ms"]),
                param=para.paraQMCRMF3,
                NM_type=str(task["NM_type"]),
                tail_eps=float(task["tail_eps"]),
                n_mesh=int(task["n_mesh"]),
                tol_bvp=float(task["tol_bvp"]),
                max_nodes=int(task["max_nodes"]),
                jB_guess=float(jB_guess),
                jB_bounds=(
                    float(task["jB_lower_bound"]),
                    float(task["jB_upper_bound"]),
                ),
                kappa_factor=float(task["kappa_factor"]),
                return_profile=False,
                verb=False,
            )
        )
    except Exception as exc:
        return {
            **base,
            "success": False,
            "task_status": "exception",
            "message": f"{type(exc).__name__}: {exc}",
            "runtime_s": time.perf_counter() - started,
        }
    task_status = "solver_failure"
    a_0plus_derived = np.nan
    if bool(result.get("success")):
        achieved = float(result.get("a_0plus_max", np.nan))
        a_0plus_derived = float(result.get("a_0plus_derived", np.nan))
        source_ok = result.get("a_0plus_source") == "maximum"
        status_ok = result.get("a_0plus_max_status") == "interior"
        composition_ok = np.isfinite(achieved) and abs(achieved - float(task["a_0plus_target"])) <= 1.0e-7
        derived_composition_ok = bool(
            np.isfinite(a_0plus_derived)
            and abs(a_0plus_derived - float(task["a_0plus_target"])) <= 1.0e-7
        )
        u_0minus = float(result.get("u_0minus", np.nan))
        jB = float(result.get("jB", np.nan))
        boundary_residuals = np.asarray(
            result.get("boundary_residuals", (np.nan,)), dtype=float
        )
        diagnostics_ok = bool(
            np.isfinite(float(result.get("Pi", np.nan)))
            and np.isfinite(float(result.get("momentum_flux_ratio", np.nan)))
            and np.isfinite(float(result.get("tail_residual_norm", np.nan)))
            and boundary_residuals.size > 0
            and np.all(np.isfinite(boundary_residuals))
            and np.max(np.abs(boundary_residuals))
            <= max(float(task["tol_bvp"]), 1.0e-10)
            and np.isfinite(float(result.get("compact_scale", np.nan)))
            and float(result.get("compact_scale", np.nan)) > 0.0
            and 0.0 < float(result.get("s_end", np.nan)) < 1.0
            and int(result.get("bvp_status", -1)) == 0
            and int(result.get("bvp_niter", 0)) > 0
            and int(result.get("bvp_nodes", 0)) >= 2
        )
        exact_model_ok = bool(
            result.get("rate_model") == "exact_nonleptonic"
            and result.get("diffusion_model") == "local_muB_fixed_T"
            and result.get("composition_definition") == "nK_over_local_nB"
            and result.get("current_definition") == "u_nK_minus_D_K_dnK_dx"
            and "BVP" in str(result.get("coordinate", ""))
        )
        if not source_ok or not status_ok:
            task_status = "composition_source_mismatch"
        elif not composition_ok:
            task_status = "composition_mismatch"
        elif not derived_composition_ok:
            task_status = "composition_closure_mismatch"
        elif not exact_model_ok:
            task_status = "exact_model_mismatch"
        elif not diagnostics_ok:
            task_status = "invalid_bvp_diagnostics"
        elif np.isfinite(u_0minus) and u_0minus > 0.0 and np.isfinite(jB) and jB > 0.0:
            task_status = "success"
            base["velocity_m_s"] = domain.proper_velocity_to_m_s(u_0minus)
        else:
            task_status = "nonphysical_result"
    merged = {
        **result,
        **base,
        "task_status": task_status,
        "composition_closure_error": float(
            a_0plus_derived - float(task["a_0plus_target"])
        )
        if bool(result.get("success")) and np.isfinite(a_0plus_derived)
        else np.nan,
        "runtime_s": time.perf_counter() - started,
    }
    if task_status != "success":
        merged["velocity_m_s"] = np.nan
    return merged


def _numerical_trial_entry(
    task: dict[str, Any], jB_guess: float, jB_guess_source: str
) -> dict[str, Any]:
    return run_numerical_trial(task, jB_guess, jB_guess_source)


def _hard_numerical_trial_runner(
    task: dict[str, Any],
    jB_guess: float,
    source: str,
    timeout_s: float,
) -> dict[str, Any]:
    return run_module_call_with_hard_timeout(
        __name__,
        "_numerical_trial_entry",
        {
            "task": task,
            "jB_guess": float(jB_guess),
            "jB_guess_source": str(source),
        },
        timeout_s=float(timeout_s),
    )


def solve_numerical_cell(
    task: dict[str, Any],
    candidates: list[tuple[float, str]],
    *,
    trial_timeout_s: float,
    cell_timeout_s: float,
    trial_runner: Callable[[dict[str, Any], float, str, float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    runner = _hard_numerical_trial_runner if trial_runner is None else trial_runner
    started = time.monotonic()
    deadline = started + float(cell_timeout_s)
    attempts: list[dict[str, Any]] = []
    last_record: dict[str, Any] | None = None
    for jB_guess, source in candidates:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        allowed = min(float(trial_timeout_s), remaining)
        attempt_started = time.monotonic()
        try:
            record = runner(task, float(jB_guess), str(source), allowed)
            last_record = dict(record)
            attempt_status = str(record.get("task_status", "failure"))
            attempts.append(
                {
                    "jB_guess": float(jB_guess),
                    "jB_guess_source": str(source),
                    "trial_timeout_s": allowed,
                    "cell_timeout_s": float(cell_timeout_s),
                    "runtime_s": time.monotonic() - attempt_started,
                    "status": attempt_status,
                    "message": str(record.get("message", "")),
                    "child_terminated": False,
                }
            )
            if attempt_status == "success":
                record = dict(record)
                record.setdefault("jB_guess", float(jB_guess))
                record.setdefault("jB_guess_source", str(source))
                record["attempts"] = attempts
                record["cell_runtime_s"] = time.monotonic() - started
                return record
        except BVPTrialTimeoutError as exc:
            attempts.append(
                {
                    "jB_guess": float(jB_guess),
                    "jB_guess_source": str(source),
                    "trial_timeout_s": allowed,
                    "cell_timeout_s": float(cell_timeout_s),
                    "runtime_s": time.monotonic() - attempt_started,
                    "status": "trial_timeout",
                    "message": str(exc),
                    "child_terminated": True,
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "jB_guess": float(jB_guess),
                    "jB_guess_source": str(source),
                    "trial_timeout_s": allowed,
                    "cell_timeout_s": float(cell_timeout_s),
                    "runtime_s": time.monotonic() - attempt_started,
                    "status": "exception",
                    "message": f"{type(exc).__name__}: {exc}",
                    "child_terminated": False,
                }
            )
    timed_out = time.monotonic() >= deadline
    final = dict(last_record or {})
    final.update(
        {
            "success": False,
            "task_status": "cell_timeout" if timed_out else "failure",
            "temperature_index": int(task["temperature_index"]),
            "composition_index": int(task["composition_index"]),
            "T_0minus": float(task["T"]),
            "nB_0minus": float(task["nB_0minus"]),
            "a_0plus_target": float(task["a_0plus_target"]),
            "jB": np.nan,
            "u_0minus": np.nan,
            "velocity_m_s": np.nan,
            "attempts": attempts,
            "cell_runtime_s": time.monotonic() - started,
            "message": (
                f"Cell exhausted its hard {float(cell_timeout_s):g} s budget"
                if timed_out
                else "All numerical jB candidates failed"
            ),
        }
    )
    return final


def run_analytic_task(
    task: dict[str, Any],
    *,
    solver: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if solver is None:
        from RMFsolver import RMFparameter as para
        from RMFsolver import phase_velocity as pv

        solver = pv.analytic_velocity_isothermal
        parameter = para.paraQMCRMF3
    else:
        from RMFsolver import RMFparameter as para

        parameter = para.paraQMCRMF3
    base = {
        "temperature_index": int(task["temperature_index"]),
        "composition_index": int(task["composition_index"]),
        "a_0plus_target": float(task["a_0plus_target"]),
        "T_0minus": float(task["T_0minus"]),
        "nB_0minus": float(task["nB_0minus"]),
        "velocity_m_s": np.nan,
    }
    try:
        result = dict(
            solver(
                float(task["T_0minus"]),
                float(task["nB_0minus"]),
                float(task["B_one_forth"]),
                xi=float(task["xi"]),
                ms=float(task["ms"]),
                param=parameter,
                NM_type=str(task["NM_type"]),
                upB=int(task["upB"]),
            )
        )
    except Exception as exc:
        return {
            **base,
            "success": False,
            "status": "exception",
            "task_status": "exception",
            "message": f"{type(exc).__name__}: {exc}",
            "runtime_s": time.perf_counter() - started,
        }

    status = str(result.get("status", "unknown"))
    task_status = status
    if status in {"stable_neutron_matter", "isothermal_coexistence"}:
        task_status = "domain_solver_disagreement"
    elif bool(result.get("success")) and status == "moving_front":
        if result.get("a_0plus_source") != "maximum":
            task_status = "composition_source_mismatch"
        else:
            achieved = float(result.get("a_0plus_max", np.nan))
            if (
                not np.isfinite(achieved)
                or abs(achieved - float(task["a_0plus_target"])) > 1.0e-7
            ):
                task_status = "composition_mismatch"
            else:
                u_0minus = float(result.get("u_0minus", np.nan))
                jB = float(result.get("jB", np.nan))
                if np.isfinite(u_0minus) and u_0minus > 0.0 and np.isfinite(jB) and jB > 0.0:
                    task_status = "success"
                    base["velocity_m_s"] = domain.proper_velocity_to_m_s(u_0minus)
                else:
                    task_status = "nonphysical_result"
    merged = {
        **result,
        **base,
        "task_status": task_status,
        "runtime_s": time.perf_counter() - started,
    }
    if task_status != "success":
        merged["velocity_m_s"] = np.nan
    return merged


def _analytic_task_entry(task: dict[str, Any]) -> dict[str, Any]:
    return run_analytic_task(task)


def _hard_analytic_task_runner(
    task: dict[str, Any], timeout_s: float
) -> dict[str, Any]:
    return run_module_call_with_hard_timeout(
        __name__, "_analytic_task_entry", {"task": task}, timeout_s=timeout_s
    )


def run_analytic_task_with_timeout(
    task: dict[str, Any],
    *,
    timeout_s: float,
    runner: Callable[[dict[str, Any], float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    execute = _hard_analytic_task_runner if runner is None else runner
    try:
        return execute(task, float(timeout_s))
    except BVPTrialTimeoutError as exc:
        return {
            "success": False,
            "status": "timeout",
            "task_status": "timeout",
            "temperature_index": int(task["temperature_index"]),
            "composition_index": int(task["composition_index"]),
            "a_0plus_target": float(task["a_0plus_target"]),
            "T_0minus": float(task["T_0minus"]),
            "nB_0minus": float(task["nB_0minus"]),
            "u_0minus": np.nan,
            "jB": np.nan,
            "velocity_m_s": np.nan,
            "timeout_s": float(timeout_s),
            "child_terminated": True,
            "message": str(exc),
        }
    except Exception as exc:
        return {
            "success": False,
            "status": "exception",
            "task_status": "exception",
            "temperature_index": int(task["temperature_index"]),
            "composition_index": int(task["composition_index"]),
            "a_0plus_target": float(task["a_0plus_target"]),
            "T_0minus": float(task["T_0minus"]),
            "nB_0minus": float(task["nB_0minus"]),
            "u_0minus": np.nan,
            "jB": np.nan,
            "velocity_m_s": np.nan,
            "timeout_s": float(timeout_s),
            "child_terminated": False,
            "message": f"{type(exc).__name__}: {exc}",
        }


def _method_payload(domain_payload: dict[str, Any], kind: str) -> dict[str, Any]:
    temperatures = np.asarray(domain_payload["temperature_axis_MeV"], dtype=float)
    targets = np.asarray(domain_payload["a_0plus_target_axis"], dtype=float)
    payload = domain.base_payload(
        kind=kind,
        temperature_axis=temperatures,
        a_0plus_target_axis=targets,
        B_one_forth=float(domain_payload["B_one_forth_MeV"]),
        xi=float(domain_payload["xi"]),
        ms=float(domain_payload["ms_MeV"]),
        NM_type=str(domain_payload["NM_type"]),
        upB=int(domain_payload["upB"]),
    )
    shape = (temperatures.size, targets.size)
    payload.update(
        {
            "domain_git_commit": domain_payload.get("git_commit", "unknown"),
            "domain_fingerprint": domain_payload.get("domain_fingerprint"),
            "nB_0minus_grid": np.asarray(domain_payload["nB_0minus_grid"], dtype=float),
            "a_0plus_max_grid": np.asarray(domain_payload["a_0plus_max_grid"], dtype=float),
            "task_status": np.full(shape, "pending", dtype=object),
            "u_0minus_grid": np.full(shape, np.nan),
            "velocity_m_s_grid": np.full(shape, np.nan),
            "jB_grid": np.full(shape, np.nan),
            "momentum_flux_ratio_grid": np.full(shape, np.nan),
            "gamma_minus_1_grid": np.full(shape, np.nan),
            "slow_front_consistent_grid": np.full(shape, None, dtype=object),
            "a_0plus_max_status_grid": np.full(shape, None, dtype=object),
            "a_0plus_max_residual_grid": np.full(shape, np.nan),
            "delta_muB_grid": np.full(shape, np.nan),
            "analytic_velocity_residual_grid": np.full(shape, np.nan),
            "momentum_flux_inf_residual_grid": np.full(shape, np.nan),
            "momentum_flux_0plus_residual_grid": np.full(shape, np.nan),
            "composition_residual_grid": np.full(shape, np.nan),
            "records": np.full(shape, None, dtype=object),
        }
    )
    return payload


def _analytic_task_list(domain_payload: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    temperatures = np.asarray(domain_payload["temperature_axis_MeV"], dtype=float)
    targets = np.asarray(domain_payload["a_0plus_target_axis"], dtype=float)
    domain_status = np.asarray(domain_payload["cell_status"], dtype=object)
    densities = np.asarray(domain_payload["nB_0minus_grid"], dtype=float)
    for i, T in enumerate(temperatures):
        for j, target in enumerate(targets):
            if domain_status[i, j] != "success":
                payload["task_status"][i, j] = "domain_masked"
                continue
            if payload["task_status"][i, j] != "pending":
                continue
            tasks.append(
                {
                    "temperature_index": i,
                    "composition_index": j,
                    "T_0minus": float(T),
                    "nB_0minus": float(densities[i, j]),
                    "B_one_forth": float(domain_payload["B_one_forth_MeV"]),
                    "a_0plus_target": float(target),
                    "xi": float(domain_payload["xi"]),
                    "ms": float(domain_payload["ms_MeV"]),
                    "NM_type": str(domain_payload["NM_type"]),
                    "upB": int(domain_payload["upB"]),
                }
            )
    return tasks


def _store_record(payload: dict[str, Any], record: dict[str, Any]) -> None:
    i = int(record["temperature_index"])
    j = int(record["composition_index"])
    payload["records"][i, j] = record
    payload["task_status"][i, j] = str(record["task_status"])
    object_diagnostics = {
        "slow_front_consistent": "slow_front_consistent_grid",
        "a_0plus_max_status": "a_0plus_max_status_grid",
    }
    scalar_diagnostics = {
        "momentum_flux_ratio": "momentum_flux_ratio_grid",
        "gamma_minus_1": "gamma_minus_1_grid",
        "a_0plus_max_residual": "a_0plus_max_residual_grid",
        "delta_muB": "delta_muB_grid",
        "analytic_velocity_residual": "analytic_velocity_residual_grid",
        "momentum_flux_inf_residual": "momentum_flux_inf_residual_grid",
        "momentum_flux_0plus_residual": "momentum_flux_0plus_residual_grid",
        "composition_residual": "composition_residual_grid",
    }
    for record_key, grid_key in object_diagnostics.items():
        if record_key in record:
            payload[grid_key][i, j] = record[record_key]
    for record_key, grid_key in scalar_diagnostics.items():
        if record_key in record:
            try:
                payload[grid_key][i, j] = float(record[record_key])
            except (TypeError, ValueError):
                payload[grid_key][i, j] = np.nan
    if record["task_status"] == "success":
        payload["u_0minus_grid"][i, j] = float(record["u_0minus"])
        payload["velocity_m_s_grid"][i, j] = float(record["velocity_m_s"])
        payload["jB_grid"][i, j] = float(record["jB"])


def run_analytic_stage(
    *,
    domain_payload: dict[str, Any],
    output_path: str | Path,
    workers: int | None = None,
    resume: bool = False,
    timeout_s: float = DEFAULT_ANALYTIC_TIMEOUT_S,
    progress_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if domain_payload.get("domain_fingerprint") is not None:
        domain.assert_domain_fingerprint(domain_payload)
    if not np.isfinite(float(timeout_s)) or float(timeout_s) <= 0.0:
        raise ValueError("Analytical timeout must be positive and finite")
    output = Path(output_path)
    reference = _method_payload(domain_payload, "isothermal_analytic")
    requested_controls = {"timeout_s": float(timeout_s)}
    if resume and output.exists():
        payload = domain.load_payload(output, expected_kind="isothermal_analytic")
        domain.assert_compatible_payload(reference, payload)
        if payload.get("analytical_controls") != requested_controls:
            raise RuntimeError("Resume payload analytical controls mismatch")
    else:
        payload = reference
    payload["analytical_controls"] = requested_controls
    tasks = _analytic_task_list(domain_payload, payload)
    progress = domain._open_progress(
        progress_factory,
        desc="Analytical contour scan",
        total=payload["task_status"].size,
        initial=int(np.count_nonzero(payload["task_status"] != "pending")),
    )

    def store_checkpoint(record: dict[str, Any]) -> None:
        _store_record(payload, record)
        payload["run_complete"] = not np.any(payload["task_status"] == "pending")
        domain.atomic_save(output, payload)
        domain._update_progress(progress)

    worker_count = max(1, min(discover_workers(workers), max(len(tasks), 1)))
    try:
        if worker_count == 1:
            for task in tasks:
                store_checkpoint(
                    run_analytic_task_with_timeout(task, timeout_s=float(timeout_s))
                )
        elif tasks:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        run_analytic_task_with_timeout,
                        task,
                        timeout_s=float(timeout_s),
                    )
                    for task in tasks
                ]
                for future in as_completed(futures):
                    store_checkpoint(future.result())
    finally:
        domain._close_progress(progress)
    if not tasks:
        payload["run_complete"] = not np.any(payload["task_status"] == "pending")
        domain.atomic_save(output, payload)
    return payload


def _numerical_task(
    domain_payload: dict[str, Any],
    i: int,
    j: int,
    *,
    tail_eps: float,
    n_mesh: int,
    tol_bvp: float,
    max_nodes: int,
    kappa_factor: float,
) -> dict[str, Any]:
    nB_0minus = float(np.asarray(domain_payload["nB_0minus_grid"])[i, j])
    return {
        "temperature_index": int(i),
        "composition_index": int(j),
        "T": float(np.asarray(domain_payload["temperature_axis_MeV"])[i]),
        "nB_0minus": nB_0minus,
        "B_one_forth": float(domain_payload["B_one_forth_MeV"]),
        "a_0plus_target": float(
            np.asarray(domain_payload["a_0plus_target_axis"])[j]
        ),
        "ms": float(domain_payload["ms_MeV"]),
        "NM_type": str(domain_payload["NM_type"]),
        "tail_eps": float(tail_eps),
        "n_mesh": int(n_mesh),
        "tol_bvp": float(tol_bvp),
        "max_nodes": int(max_nodes),
        "kappa_factor": float(kappa_factor),
        "jB_lower_bound": DEFAULT_JB_LOWER_BOUND,
        "jB_upper_bound": DEFAULT_JB_UPPER_FACTOR * nB_0minus,
    }


def _finite_grid_value(grid: Any, i: int, j: int) -> float | None:
    if j < 0:
        return None
    value = float(np.asarray(grid, dtype=float)[i, j])
    return value if np.isfinite(value) and value > 0.0 else None


def _initial_numerical_candidates(
    payload: dict[str, Any],
    analytic_payload: dict[str, Any],
    task: dict[str, Any],
) -> list[tuple[float, str]]:
    i = int(task["temperature_index"])
    j = int(task["composition_index"])
    return build_jB_candidates(
        nB_0minus=float(task["nB_0minus"]),
        previous_jB=_finite_grid_value(payload["jB_grid"], i, j - 1),
        earlier_jB=_finite_grid_value(payload["jB_grid"], i, j - 2),
        analytic_jB=_finite_grid_value(analytic_payload["jB_grid"], i, j),
        nearby_shell_jB=(),
        lower_bound=float(task["jB_lower_bound"]),
        upper_bound=float(task["jB_upper_bound"]),
    )


def _nearby_shell_values(
    payload: dict[str, Any], i: int, j: int
) -> tuple[float, ...]:
    successful = np.asarray(payload["task_status"], dtype=object)[:, j] == "success"
    successful[i] = False
    indexes = np.flatnonzero(successful)
    if indexes.size == 0:
        return ()
    nearest = indexes[np.argsort(np.abs(indexes - i), kind="stable")[:2]]
    return tuple(float(payload["jB_grid"][index, j]) for index in nearest)


def _merge_cell_attempts(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(second)
    merged["attempts"] = list(first.get("attempts", ())) + list(
        second.get("attempts", ())
    )
    merged["cell_runtime_s"] = float(first.get("cell_runtime_s", 0.0)) + float(
        second.get("cell_runtime_s", 0.0)
    )
    return merged


def run_numerical_stage(
    *,
    domain_payload: dict[str, Any],
    analytic_payload: dict[str, Any],
    output_path: str | Path,
    workers: int | None = None,
    resume: bool = False,
    trial_timeout_s: float = DEFAULT_TRIAL_TIMEOUT_S,
    cell_timeout_s: float = DEFAULT_CELL_TIMEOUT_S,
    tail_eps: float = DEFAULT_TAIL_EPS,
    n_mesh: int = DEFAULT_N_MESH,
    tol_bvp: float = DEFAULT_BVP_TOL,
    max_nodes: int = DEFAULT_MAX_NODES,
    kappa_factor: float = DEFAULT_KAPPA_FACTOR,
    progress_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run exact isothermal BVPs in increasing-composition shells."""
    if domain_payload.get("domain_fingerprint") is not None:
        domain.assert_domain_fingerprint(domain_payload)
    positive_controls = {
        "trial_timeout_s": trial_timeout_s,
        "cell_timeout_s": cell_timeout_s,
        "tail_eps": tail_eps,
        "n_mesh": n_mesh,
        "tol_bvp": tol_bvp,
        "max_nodes": max_nodes,
        "kappa_factor": kappa_factor,
    }
    for name, value in positive_controls.items():
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    output = Path(output_path)
    reference = _method_payload(domain_payload, "isothermal_numerical")
    analytic_reference = _method_payload(domain_payload, "isothermal_analytic")
    if analytic_payload.get("payload_kind") != "isothermal_analytic":
        raise RuntimeError("Analytical seed payload has the wrong payload kind")
    domain.assert_compatible_payload(analytic_reference, analytic_payload)
    if not np.array_equal(
        np.asarray(domain_payload["nB_0minus_grid"], dtype=float),
        np.asarray(analytic_payload["nB_0minus_grid"], dtype=float),
        equal_nan=True,
    ):
        raise RuntimeError("Analytical seed density grid differs from the domain")
    requested_controls = {
        "trial_timeout_s": float(trial_timeout_s),
        "cell_timeout_s": float(cell_timeout_s),
        "tail_eps": float(tail_eps),
        "n_mesh": int(n_mesh),
        "tol_bvp": float(tol_bvp),
        "max_nodes": int(max_nodes),
        "kappa_factor": float(kappa_factor),
        "jB_lower_bound": DEFAULT_JB_LOWER_BOUND,
        "jB_upper_factor": DEFAULT_JB_UPPER_FACTOR,
    }
    if resume and output.exists():
        payload = domain.load_payload(output, expected_kind="isothermal_numerical")
        domain.assert_compatible_payload(reference, payload)
        if payload.get("numerical_controls") != requested_controls:
            raise RuntimeError("Resume payload numerical controls mismatch")
    else:
        payload = reference

    payload["numerical_controls"] = requested_controls
    domain_status = np.asarray(domain_payload["cell_status"], dtype=object)
    pending = np.asarray(payload["task_status"], dtype=object) == "pending"
    payload["task_status"][pending & (domain_status != "success")] = "domain_masked"
    domain.atomic_save(output, payload)

    n_temperature, n_composition = domain_status.shape
    progress = domain._open_progress(
        progress_factory,
        desc="Numerical contour scan",
        total=payload["task_status"].size,
        initial=int(np.count_nonzero(payload["task_status"] != "pending")),
    )
    worker_count = max(1, min(discover_workers(workers), n_temperature))
    try:
        for j in range(n_composition):
            tasks = [
                _numerical_task(
                    domain_payload,
                    i,
                    j,
                    tail_eps=tail_eps,
                    n_mesh=n_mesh,
                    tol_bvp=tol_bvp,
                    max_nodes=max_nodes,
                    kappa_factor=kappa_factor,
                )
                for i in range(n_temperature)
                if domain_status[i, j] == "success"
                and payload["task_status"][i, j] == "pending"
            ]
            if not tasks:
                continue

            def solve_first(task: dict[str, Any]) -> dict[str, Any]:
                return solve_numerical_cell(
                    task,
                    _initial_numerical_candidates(payload, analytic_payload, task),
                    trial_timeout_s=float(trial_timeout_s),
                    cell_timeout_s=float(cell_timeout_s),
                )

            first_records: list[dict[str, Any]] = []

            def store_first(record: dict[str, Any]) -> None:
                first_records.append(record)
                _store_record(payload, record)
                payload["run_complete"] = not np.any(
                    payload["task_status"] == "pending"
                )
                domain.atomic_save(output, payload)
                domain._update_progress(progress)

            if worker_count == 1:
                for task in tasks:
                    store_first(solve_first(task))
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [executor.submit(solve_first, task) for task in tasks]
                    for future in as_completed(futures):
                        store_first(future.result())

            # A second pass lets failed cells use successful temperatures from the
            # just-completed shell without coupling the disposable BVP children.
            for first in first_records:
                if first.get("task_status") == "success":
                    continue
                i = int(first["temperature_index"])
                nearby = _nearby_shell_values(payload, i, j)
                remaining = float(cell_timeout_s) - float(
                    first.get("cell_runtime_s", 0.0)
                )
                if not nearby or remaining <= 0.0:
                    continue
                task = next(
                    item for item in tasks if int(item["temperature_index"]) == i
                )
                retry_candidates = build_jB_candidates(
                    nB_0minus=float(task["nB_0minus"]),
                    previous_jB=None,
                    earlier_jB=None,
                    analytic_jB=None,
                    nearby_shell_jB=nearby,
                    lower_bound=float(task["jB_lower_bound"]),
                    upper_bound=float(task["jB_upper_bound"]),
                )
                second = solve_numerical_cell(
                    task,
                    retry_candidates,
                    trial_timeout_s=float(trial_timeout_s),
                    cell_timeout_s=remaining,
                )
                _store_record(payload, _merge_cell_attempts(first, second))
                domain.atomic_save(output, payload)

            payload["completed_composition_shells"] = int(j + 1)
            domain.atomic_save(output, payload)
    finally:
        domain._close_progress(progress)

    payload["run_complete"] = not np.any(payload["task_status"] == "pending")
    domain.atomic_save(output, payload)
    return payload
