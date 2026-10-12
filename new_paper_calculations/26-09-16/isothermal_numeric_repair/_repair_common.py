"""Portable I/O, validation and isolated numerical trials for contour repair."""
from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time

import numpy as np

LEVELS = (50., 100., 200., 500., 1000., 2000., 5000.)
SCHEMA = 1
C = 299792458.0
DEFAULT_CONTROLS = dict(
    trial_timeout=180., target_timeout=900., n_mesh=300, tol_bvp=1.e-4,
    max_nodes=50000, tail_eps=1.e-3, kappa_factor=1., speed_rtol=.005,
    anchor_rtol=.01, density_step=.05, temperature_step=2., cold_ratio=1.5,
    continuation_depth=8, angular_depth=8, max_ray_evaluations=24,
    # Line connection is decided by angular adjacency on the scheduled theta
    # axis; these are only vetoes against a gross jump between neighbouring
    # rays, so they are far looser than the continuation step sizes above.
    connect_density_jump=.8, connect_temperature_jump=25., connect_radial_jump=.25,
)
INPUT_NAMES = ("isothermal-domain.npy", "isothermal-analytic.npy", "isothermal-numerical.npy")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, payload, allow_pickle=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path):
    return np.load(path, allow_pickle=True).item()


def domain_fingerprint(payload):
    """Exact schema-4 fingerprint; never rewrite the inherited fingerprint."""
    h = hashlib.sha256()
    for key in ("schema_version", "run_tag", "B_one_forth_MeV", "xi", "ms_MeV", "NM_type", "upB"):
        h.update(key.encode()); h.update(json.dumps(payload[key]).encode())
    for key in (
        "theta_axis_rad", "radial_fraction_axis", "boundary_theta_axis_rad",
        "rho_inner_boundary", "rho_outer_boundary", "lower_phase_nB_0minus",
        "lower_phase_T_0minus_MeV", "upper_phase_nB_0minus", "upper_phase_T_0minus_MeV",
        "cell_status", "nB_0minus_grid", "T_0minus_grid_MeV",
    ):
        values = np.asarray(payload[key])
        h.update(key.encode()); h.update(str(values.shape).encode())
        if values.dtype == object:
            h.update(json.dumps(values.tolist(), sort_keys=True).encode())
        else:
            values = np.ascontiguousarray(values)
            h.update(values.dtype.str.encode()); h.update(values.tobytes())
    h.update(json.dumps(payload["phase_velocity_signatures"], sort_keys=True).encode())
    return h.hexdigest()


def load_baseline(directory, *, check_live=True):
    directory = Path(directory)
    inputs = [load(directory / name) for name in INPUT_NAMES]
    domain, analytic, numerical = inputs
    expected = domain_fingerprint(domain)
    if expected != domain.get("domain_fingerprint"):
        raise ValueError("Baseline domain fingerprint is corrupt")
    for payload, kind in zip(inputs, ("isothermal_domain", "isothermal_analytic", "isothermal_numerical")):
        if payload.get("payload_kind") != kind or payload.get("domain_fingerprint") != expected:
            raise ValueError("Baseline payload kind or domain fingerprint mismatch")
        for key in ("theta_axis_rad", "radial_fraction_axis", "nB_0minus_grid", "T_0minus_grid_MeV"):
            if not np.array_equal(payload[key], domain[key]):
                raise ValueError(f"Baseline axes or coordinates mismatch: {key}")
        for key in ("B_one_forth_MeV", "ms_MeV", "NM_type", "upB", "xi"):
            if payload[key] != domain[key]:
                raise ValueError(f"Baseline physics mismatch: {key}")
    if not np.all(domain["cell_status"] == "success"):
        raise ValueError("The saved domain is incomplete")
    if domain["ms_MeV"] != 0 or domain["NM_type"] != "PNM":
        raise ValueError("This repair workflow requires the massless PNM baseline")
    if check_live:
        from RMFsolver import phase_velocity as pv
        saved = domain["phase_velocity_signatures"]
        live = inspect.signature(pv.solve_front_isothermal)
        for name, default in (("continuation_guess", None), ("return_continuation", False)):
            if name not in live.parameters or live.parameters[name].default is not default:
                raise ValueError("Install phase_velocity.py with lightweight isothermal continuation")
        legacy = live.replace(parameters=[p for name, p in live.parameters.items()
                                          if name not in ("continuation_guess", "return_continuation")])
        if str(legacy) != saved["solve_front_isothermal"]:
            raise ValueError("Unsupported numerical API change beyond the optional continuation arguments")
        if str(inspect.signature(pv.analytic_velocity_isothermal)) != saved["analytic_velocity_isothermal"]:
            raise ValueError("Baseline analytical API mismatch")
    return inputs, {name: sha256(directory / name) for name in INPUT_NAMES}


def workers(requested=None):
    if requested is not None:
        if requested < 1:
            raise ValueError("workers must be positive")
        return requested
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        try:
            if int(os.environ.get(name, "0")) > 0:
                return int(os.environ[name])
        except ValueError:
            pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def current_bracket(*estimates):
    usable = [float(j) for j in estimates if j is not None and np.isfinite(j) and j > 0]
    if not usable:
        raise ValueError("No finite positive numerical or analytical current seed")
    return min(usable) / 4., max(usable) * 4.


def near_current_bound(jB, bounds):
    return min(np.log(jB / bounds[0]), np.log(bounds[1] / jB)) < .05 * np.log(bounds[1] / bounds[0])


def valid_result(result, controls):
    try:
        residual = np.asarray(result["boundary_residuals"], float)
        return bool(
            result["success"] and result["bvp_status"] == 0
            and result["a_0plus_source"] == "maximum"
            and result["a_0plus_max_status"] == "interior"
            and 0 < result["a_0plus"] < 1
            and abs(result["a_0plus_derived"] - result["a_0plus"]) <= 1.e-7
            and result["rate_model"] == "exact_nonleptonic"
            and result["diffusion_model"] == "local_muB_fixed_T"
            and result["composition_definition"] == "nK_over_local_nB"
            and result["current_definition"] == "u_nK_minus_D_K_dnK_dx"
            and np.isfinite(result["jB"]) and result["jB"] > 0
            and np.isfinite(result["u_0minus"]) and result["u_0minus"] > 0
            and residual.size == 3 and np.all(np.isfinite(residual))
            and np.max(np.abs(residual)) <= controls["tol_bvp"]
            and np.isfinite(result["compact_scale"]) and result["compact_scale"] > 0
            and np.isfinite(result["tail_residual_norm"])
            and np.isfinite(result["momentum_flux_ratio"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def numerical_trial(point, physics, controls, jB_guess, bounds, continuation):
    """Runs in a disposable child. No chemical-potential scan or frozen transport."""
    from RMFsolver import phase_velocity as pv
    from RMFsolver import RMFparameter as para
    start = time.monotonic()
    upstream = pv._validated_pnm_state_from_nB(
        point["nB_0minus"], point["T_0minus"], param=para.paraQMCRMF3
    )
    muB_inf = pv._solve_muB_inf_at_muK0_for_given_Pi(
        upstream["P_0minus"], 0., physics["B_one_forth_MeV"], point["T_0minus"],
        ms=physics["ms_MeV"], upB=physics["upB"],
    )
    delta_muB = float(muB_inf - upstream["muB_0minus"])
    if delta_muB >= -1.e-6:
        return dict(status="outside_moving_domain", success=False, delta_muB=delta_muB,
                    message="Stable or coexistence state; no moving BVP requested")
    result = pv.solve_front_isothermal(
        point["T_0minus"], point["nB_0minus"], physics["B_one_forth_MeV"],
        ms=physics["ms_MeV"], NM_type=physics["NM_type"],
        **{key: controls[key] for key in ("tail_eps", "n_mesh", "tol_bvp", "max_nodes", "kappa_factor")},
        jB_guess=jB_guess, jB_bounds=bounds, continuation_guess=continuation,
        return_continuation=True, return_profile=False,
    )
    result["delta_muB"] = delta_muB
    result["status"] = "success" if valid_result(result, controls) else "solver_failure"
    result["runtime_s"] = time.monotonic() - start
    if result["status"] == "success":
        # Endpoint EOS and flux validation uses only two additional state evaluations.
        errors = []
        for suffix in ("0plus", "inf"):
            state = pv._quark_thermo_state(
                result[f"muB_{suffix}"], result[f"muK_{suffix}"], physics["B_one_forth_MeV"],
                point["T_0minus"], result["jB"], ms=physics["ms_MeV"], upB=physics["upB"],
            )
            errors.extend((abs(state["Pi"] / result["Pi"] - 1.),
                           abs(state["nB"] * state["u"] / result["jB"] - 1.)))
        result["endpoint_flux_residual_max"] = max(errors)
        if not np.all(np.isfinite(errors)) or max(errors) > 1.e-7:
            result["status"] = "endpoint_validation_failed"
    result["success"] = result["status"] == "success"
    u = result.get("u_0minus", np.nan)
    result["velocity_m_s"] = C * u / np.hypot(1., u) if result["success"] else np.nan
    return result


def _child(connection, kwargs):
    try:
        connection.send(numerical_trial(**kwargs))
    except Exception as exc:
        connection.send(dict(status="exception", success=False, message=f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def hard_trial(*, timeout, **kwargs):
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(send, kwargs))
    start = time.monotonic()
    process.start(); send.close()
    try:
        if receive.poll(max(.001, timeout)):
            try:
                result = receive.recv()
            except EOFError:
                result = dict(status="child_error", success=False, message="Worker exited without a result")
        else:
            result = dict(status="trial_timeout", success=False, message=f"Hard trial limit: {timeout:g} s")
    finally:
        receive.close()
        process.join(timeout=.1)
        if process.is_alive():
            process.terminate(); process.join(timeout=2.)
        if process.is_alive():
            process.kill(); process.join(timeout=2.)
    result["runtime_s"] = time.monotonic() - start
    return result
