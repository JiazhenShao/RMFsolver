"""Bounded, single-worker probes of the unmodified full isothermal BVP."""
import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(ROOT))
import numpy as np

BAG = 189.1565957288247


def atomic_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def worker(connection, density, temperature, controls, seed_file, core_seed=False):
    from RMFsolver import phase_velocity as pv, constants as const
    import scipy
    started = time.monotonic()
    warning_counts = Counter()

    def event(kind, **values):
        connection.send(dict(kind=kind, elapsed_s=time.monotonic() - started, **values))

    def warning(message, category, filename, lineno, file=None, line=None):
        key = f"{Path(filename).name}:{lineno}: {category.__name__}: {message}"
        warning_counts[key] += 1
        if warning_counts[key] == 1:
            event("warning", message=key)

    warnings.showwarning = warning
    warnings.simplefilter("always")
    try:
        event("imported", phase_velocity_file=pv.__file__, scipy_version=scipy.__version__,
              scipy_file=scipy.__file__, numpy_version=np.__version__)
        nB = density * const.NuclearDensity_nucleons_MeV3
        analytic = pv.analytic_velocity_isothermal(temperature, nB, BAG, xi=-0.5)
        event("analytic_seed", result=analytic)
        if not analytic.get("front_exists"):
            event("result", result=dict(success=False, status="no_analytic_seed", analytic_status=analytic["status"]))
            return
        current = float(analytic["jB"])
        continuation = None
        if seed_file:
            saved = np.load(seed_file, allow_pickle=True).item()
            continuation = saved["result"]["continuation_state"]
            current = float(continuation["jB"])
        original_bvp = pv.solve_bvp
        bvp_diagnostics = {}

        def diagnostic_bvp(fun, bc, x, y, **kwargs):
            closure = dict(zip(fun.__code__.co_freevars, (cell.cell_contents for cell in fun.__closure__)))
            state = closure["build_global_state"](kwargs["p"][0]) if "build_global_state" in closure else closure["state_or_none"](kwargs["p"][0])
            if state is not None:
                bvp_diagnostics["initial_state"] = state
            if core_seed:
                # Initial-guess experiment only. The actual ODE, exact local EOS,
                # reaction, diffusion, endpoints and acceptance checks are unchanged.
                lam = state["lambda_inf"]
                slope = (state["jK_0plus"] / state["nK_0plus"] - state["u_0plus"]) / state["D_K_0plus"]
                x_core = 0.5 / (slope - lam)
                if not np.isfinite(x_core) or x_core <= 0:
                    raise ValueError("Core seed requires a positive interface decay scale")
                physical_x = np.r_[0., np.geomspace(x_core*1.e-3, state["x_end"], controls["n_mesh"]-1)]
                x = -np.expm1(-lam*physical_x/controls["kappa_factor"])
                fraction = np.exp(-lam*physical_x)/np.sqrt(1.0+physical_x/x_core)
                nK = state["nK_0plus"] * fraction
                nB_seed = state["nB_inf"]+(state["nB_0plus"]-state["nB_inf"])*fraction**2
                diffusion_seed = state["D_K_inf"]+(state["D_K_0plus"]-state["D_K_inf"])*fraction**2
                jK = nK*(state["jB"]/nB_seed + diffusion_seed*(lam+0.5/(x_core+physical_x)))
                jK[0] = state["jK_0plus"]
                jK[-1] = state["tail_coefficient"]*nK[-1]
                y = np.array([nK/closure["bvp_nK_scale"], jK/closure["bvp_jK_scale"]])
                bvp_diagnostics["core_seed_x_core"] = x_core
            event("bvp_start", mesh_nodes=len(x), first_mesh_step=float(x[1]-x[0]), diagnostics=bvp_diagnostics)
            call_count = 0
            largest_mesh = len(x)

            def monitored_fun(s, fields, parameters):
                nonlocal call_count, largest_mesh
                call_count += 1
                if len(s) > largest_mesh or call_count % 10 == 0:
                    largest_mesh = max(largest_mesh, len(s))
                    event("bvp_progress", calls=call_count, nodes=len(s),
                          normalized_nK_range=[float(np.min(fields[0])), float(np.max(fields[0]))],
                          stats=closure.get("stats", {}), last_failure=closure.get("last_failure", {}))
                return fun(s, fields, parameters)

            solution = original_bvp(monitored_fun, bc, x, y, **kwargs)
            bvp_diagnostics.update(max_rms_residual=float(np.max(solution.rms_residuals)),
                                   bvp_nodes=len(solution.x), bvp_niter=solution.niter)
            event("bvp_finish", diagnostics=bvp_diagnostics)
            return solution

        pv.solve_bvp = diagnostic_bvp
        result = pv.solve_front_isothermal(
            temperature, nB, BAG, ms=0.0, NM_type="PNM", jB_guess=current,
            jB_bounds=(current/4.0, current*4.0),
            continuation_guess=continuation, return_continuation=True, **controls,
        )
        result["bvp_diagnostics"] = bvp_diagnostics
        if result.get("success"):
            endpoint_errors = []
            for suffix in ("0plus", "inf"):
                state = pv._quark_thermo_state(result[f"muB_{suffix}"], result[f"muK_{suffix}"],
                                             BAG, temperature, result["jB"], ms=0.0, upB=5000)
                endpoint_errors += [abs(state["Pi"]/result["Pi"]-1),
                                    abs(state["nB"]*state["u"]/result["jB"]-1)]
            result["endpoint_flux_residual_max"] = max(endpoint_errors)
            result["velocity_m_s"] = 299792458.0*result["u_0minus"]/np.hypot(1.0, result["u_0minus"])
            profile = result["continuation_state"]
            result["nK_min_fraction"] = float(np.min(profile["nK"])/profile["nK_0plus"])
            result["nK_max_fraction"] = float(np.max(profile["nK"])/profile["nK_0plus"])
            result["positive_nK_step_max_fraction"] = float(np.max(np.diff(profile["nK"]))/profile["nK_0plus"])
        result["warnings"] = dict(warning_counts)
        event("result", result=result)
    except Exception as exc:
        event("result", result=dict(success=False, status="exception", message=f"{type(exc).__name__}: {exc}",
                                    traceback=traceback.format_exc(), warnings=dict(warning_counts)))
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--densities", nargs="+", type=float, default=[3.001, 3.01, 3.1, 3.5, 4.5, 5.2])
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--seed")
    parser.add_argument("--core-seed", action="store_true")
    parser.add_argument("--tail-eps", type=float, default=1.e-3)
    parser.add_argument("--tol", type=float, default=1.e-4)
    parser.add_argument("--mesh", type=int, default=300)
    parser.add_argument("--kappa", type=float, default=1.0)
    args = parser.parse_args()
    # Fail once, before spawning a sweep, if the environment cannot import the solver.
    from RMFsolver import phase_velocity
    controls = dict(tail_eps=args.tail_eps, tol_bvp=args.tol, n_mesh=args.mesh,
                    max_nodes=50000, kappa_factor=args.kappa)
    solver = ROOT / "RMFsolver/phase_velocity.py"
    identity = dict(sha256=hashlib.sha256(solver.read_bytes()).hexdigest(), file=str(solver))
    for density in args.densities:
        label = f"{args.tag}_T{args.temperature:g}_n{density:g}"
        target = OUT / (label + ".npy")
        if target.exists():
            raise FileExistsError(f"Refusing to replace existing diagnostic {target}")
        row = dict(T_0minus=args.temperature, nB_0minus_over_n0=density, B_one_forth=BAG,
                   solver_identity=identity, controls=controls, timeout_s=args.timeout, seed_file=args.seed,
                   core_seed=args.core_seed, events=[])
        context = mp.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=worker, args=(sender, density, args.temperature, controls, args.seed, args.core_seed))
        started = time.monotonic()
        process.start()
        sender.close()
        print(f"START {label}: timeout {args.timeout:g}s, one worker", flush=True)
        try:
            while time.monotonic()-started < args.timeout:
                if receiver.poll(min(0.25, max(0, args.timeout-(time.monotonic()-started)))):
                    try:
                        item = receiver.recv()
                    except EOFError:
                        break
                    row["events"].append(item)
                    if item["kind"] == "result":
                        row["result"] = item["result"]
                    atomic_save(target, row)
                    print(f"  {item['elapsed_s']:.2f}s {item['kind']}" +
                          (f": {item['message']}" if item['kind'] == 'warning' else ""), flush=True)
                    if item["kind"] == "result":
                        break
                elif not process.is_alive():
                    break
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            receiver.close()
        row["runtime_s"] = time.monotonic()-started
        if "result" not in row:
            row["result"] = dict(success=False, status="timeout" if row["runtime_s"] >= args.timeout else "child_exit")
        atomic_save(target, row)
        result = row["result"]
        print(json.dumps(dict(label=label, runtime_s=row["runtime_s"], **{
            key: result.get(key) for key in ("success", "status", "message", "velocity_m_s", "a_0plus", "bvp_nodes", "bvp_niter")
        })), flush=True)


if __name__ == "__main__":
    main()
