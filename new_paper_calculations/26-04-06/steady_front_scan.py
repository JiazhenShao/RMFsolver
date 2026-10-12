#!/usr/bin/env python3
"""
Cluster-ready compact-BVP steady-front scan over fixed aQstar values, using the
latest rich-branch-only 1D BVP steady-front solver.

This script uses the latest RMFsolver.phase_velocity BVP path:

    solve_steady_front_1d_aQstar_rescale_bvp(...)

The old numerically inconsistent muK-poor branch has been removed. This scan
therefore follows only the muK-rich steady-front branch.

For each fixed aQstar, solve_bvp adjusts the quark-side profile and the
unknown jB parameter together. The aQstar values are split into contiguous
chunks and run in parallel; within each chunk, the previous successful jB is
used as the next jB_guess for continuation.

Outputs are saved under this 6Apr folder:
- one timestamped output directory
- run_config.json
- trial_summary.jsonl with one line per aQstar point
- per-chunk .log files containing solver diagnostics
- per-point .npz files containing the returned result fields and profile arrays

Recommended cluster usage:
    python steady_front_scan.py --workers 1 --chunk-size 20 --verb-mode simple

Useful variants:
    python steady_front_scan.py --workers 1 --chunk-size 20 --tail-eps 1e-8
    python steady_front_scan.py --workers 1 --chunk-size 20 --aQstar-list 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6
    python steady_front_scan.py --workers 1 --chunk-size 20 --jB-aQstar-slope 1.2 --jB-bounds 1e-5,5.0
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Reduce thread oversubscription on clusters when using many worker processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("XDG_CACHE_HOME", str(Path(os.environ.get("TMPDIR", "/tmp")) / "xdg-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import RMFsolver.constants as const
from RMFsolver.phase_velocity import solve_steady_front_1d_aQstar_rescale_bvp


BRANCH_LABEL = "muK-rich"

DENSITY_CONVENTION = {
    "name": "Convention B",
    "base_helpers": "P_f, E_f, n_B include spin only and exclude quark color",
    "quark_species": "n_u, n_d, n_s are physical quark densities including color",
    "nB_QM": "(n_u + n_d + n_s) / 3",
    "nK_QM": "(n_d - n_s) / 2",
}


class PrefixedStream:
    """Write to a stream while prefixing each new line."""

    def __init__(self, stream, prefix: str):
        self.stream = stream
        self.prefix = prefix
        self.at_line_start = True

    def write(self, text: str):
        if not text:
            return 0
        for chunk in text.splitlines(keepends=True):
            if self.at_line_start:
                self.stream.write(self.prefix)
            self.stream.write(chunk)
            self.at_line_start = chunk.endswith("\n")
        self.stream.flush()
        return len(text)

    def flush(self):
        self.stream.flush()


class MultiWriter:
    """Broadcast writes to multiple file-like objects."""

    def __init__(self, *writers):
        self.writers = writers

    def write(self, text: str):
        for writer in self.writers:
            writer.write(text)
        return len(text)

    def flush(self):
        for writer in self.writers:
            writer.flush()


def _parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _sanitize_float_for_name(x: float) -> str:
    s = f"{x:.6g}"
    return s.replace("+", "").replace("-", "m").replace(".", "p")


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return default
        return int(value)
    except Exception:
        return default


def _jB_guess_from_aQstar(aQstar: float, task: dict) -> float:
    """Build a pointwise jB guess from the observed BVP branch.

    The default slope is tuned to the last clean continuation branch:
    the successful 6Apr points up to aQstar=0.2 give jB/aQstar ~= 1.0--1.15,
    so jB ~= 1.2 * aQstar is a conservative first guess.
    """
    if str(task.get("jB_guess_mode", "linear-aQstar")) == "constant":
        guess = float(task["jB_guess"])
    else:
        guess = float(task["jB_aQstar_slope"]) * max(0.0, float(aQstar))
        guess = max(float(task["jB_guess_floor"]), guess)

    bounds = task.get("jB_bounds")
    if bounds is not None:
        lo, hi = map(float, bounds)
        # Keep the initial guess safely inside the bounded BVP parameter map.
        margin = 1.0e-6 * max(1.0, hi - lo)
        guess = min(max(guess, lo + margin), hi - margin)
    return float(guess)


def _save_result_npz(npz_path: Path, result: dict, point_meta: dict):
    payload = {"message": np.array(str(result.get("message", "")))}
    for key, value in point_meta.items():
        payload[f"meta_{key}"] = np.array(value if value is not None else "")
    for key, value in result.items():
        if key.startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            payload[key] = value
        elif np.isscalar(value) or isinstance(value, str):
            payload[key] = np.array(value)
    np.savez(npz_path, **payload)


def _point_npz_path(output_dir: Path, chunk_id: int, point_index: int, aQstar: float):
    stem = (
        f"point_{point_index:03d}"
        f"__chunk_{chunk_id:02d}"
        f"__aQstar_{_sanitize_float_for_name(aQstar)}"
    )
    return output_dir / f"{stem}.npz"


def _chunk_log_path(output_dir: Path, chunk_id: int, a_values: list[float]):
    if a_values:
        label = f"aQ_{_sanitize_float_for_name(a_values[0])}_to_{_sanitize_float_for_name(a_values[-1])}"
    else:
        label = "empty"
    return output_dir / f"chunk_{chunk_id:02d}__{label}.log"


def _chunk_values(values: list[float], workers: int, chunk_size: int | None) -> list[list[float]]:
    if not values:
        return []
    if chunk_size is not None and chunk_size > 0:
        return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]
    n_chunks = min(max(1, workers), len(values))
    chunks = []
    for idx in range(n_chunks):
        start = round(idx * len(values) / n_chunks)
        stop = round((idx + 1) * len(values) / n_chunks)
        if start < stop:
            chunks.append(values[start:stop])
    return chunks


def _point_summary_from_result(result: dict, point_meta: dict, npz_path: Path, profile_saved: bool) -> dict:
    a_N = _safe_float(result.get("a_N", np.nan))
    aQ = _safe_float(result.get("aQstar", point_meta["aQstar_input"]))
    summary = dict(point_meta)
    summary.update(
        {
            "success": bool(result.get("success", False)),
            "message": str(result.get("message", "")),
            "profile_saved": bool(profile_saved),
            "npz_path": str(npz_path),
            "jB": _safe_float(result.get("jB", np.nan)),
            "u_N": _safe_float(result.get("u_N", np.nan)),
            "u_Q": _safe_float(result.get("u_Q", np.nan)),
            "a_N": a_N,
            "aQstar": aQ,
            "aQstar_gt_aN": bool(np.isfinite(a_N) and np.isfinite(aQ) and aQ > a_N),
            "Pi": _safe_float(result.get("Pi", np.nan)),
            "muB_Qstar": _safe_float(result.get("muB_Qstar", np.nan)),
            "muK_Qstar": _safe_float(result.get("muK_Qstar", np.nan)),
            "nB_Qstar": _safe_float(result.get("nB_Qstar", np.nan)),
            "muB_Q": _safe_float(result.get("muB_Q", np.nan)),
            "nB_Q": _safe_float(result.get("nB_Q", np.nan)),
            "nK_Q": _safe_float(result.get("nK_Q", np.nan)),
            "DQ": _safe_float(result.get("DQ", np.nan)),
            "eta": _safe_float(result.get("eta", np.nan)),
            "gamma": _safe_float(result.get("gamma", np.nan)),
            "tau": _safe_float(result.get("tau", np.nan)),
            "lambda": _safe_float(result.get("lambda", np.nan)),
            "kappa": _safe_float(result.get("kappa", np.nan)),
            "tail_eps": _safe_float(result.get("tail_eps", np.nan)),
            "s_end": _safe_float(result.get("s_end", np.nan)),
            "x_end": _safe_float(result.get("x_end", np.nan)),
            "tail_residual": _safe_float(result.get("tail_residual", np.nan)),
            "tail_residual_norm": _safe_float(result.get("tail_residual_norm", np.nan)),
            "tail_scale": _safe_float(result.get("tail_scale", np.nan)),
            "a_end": _safe_float(result.get("a_end", np.nan)),
            "q_end": _safe_float(result.get("q_end", np.nan)),
            "closure_error_max": _safe_float(result.get("closure_error_max", np.nan)),
            "root_method": str(result.get("_root_method", "")),
            "branch_label": str(result.get("branch_label", point_meta["branch_label"])),
            "coordinate": str(result.get("coordinate", "")),
            "bvp_status": _safe_int(result.get("bvp_status", -999), -999),
            "bvp_message": str(result.get("bvp_message", "")),
            "bvp_niter": _safe_int(result.get("bvp_niter", -1), -1),
            "bvp_nodes": _safe_int(result.get("bvp_nodes", 0)),
            "bvp_ode_calls": _safe_int(result.get("bvp_ode_calls", 0)),
            "bvp_bc_calls": _safe_int(result.get("bvp_bc_calls", 0)),
            "q_root_calls": _safe_int(result.get("q_root_calls", 0)),
            "qstar_root_calls": _safe_int(result.get("qstar_root_calls", 0)),
            "local_state_calls": _safe_int(result.get("local_state_calls", 0)),
            "local_root_calls": _safe_int(result.get("local_root_calls", 0)),
            "local_fast_failures": _safe_int(result.get("local_fast_failures", 0)),
            "local_state_failures": _safe_int(result.get("local_state_failures", 0)),
            "profile_state_calls": _safe_int(result.get("profile_state_calls", 0)),
            "global_state_builds": _safe_int(result.get("global_state_builds", 0)),
            "global_state_failures": _safe_int(result.get("global_state_failures", 0)),
            "density_convention": DENSITY_CONVENTION["name"],
        }
    )
    return {key: _json_safe(value) for key, value in summary.items()}


def _run_one_chunk(task: dict) -> list[dict]:
    chunk_id = int(task["chunk_id"])
    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    a_values = [float(x) for x in task["aQstar_values"]]
    log_path = _chunk_log_path(output_dir, chunk_id, a_values)
    prefixed_stdout = PrefixedStream(sys.stdout, prefix=f"[chunk {chunk_id:02d}] ")

    point_summaries: list[dict] = []
    t0 = time.perf_counter()
    last_success_jB = None

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = MultiWriter(prefixed_stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(
                json.dumps(
                    {
                        "event": "chunk_start",
                        "chunk_id": chunk_id,
                        "aQstar_values": a_values,
                        "T": task["T"],
                        "nB_N": task["nB_N"],
                        "B_one_forth": task["B_one_forth"],
                        "tail_eps": task["tail_eps"],
                        "n_mesh": task["n_mesh"],
                        "tol_bvp": task["tol_bvp"],
                        "max_nodes": task["max_nodes"],
                        "jB_guess": task["jB_guess"],
                        "jB_guess_mode": task["jB_guess_mode"],
                        "jB_aQstar_slope": task["jB_aQstar_slope"],
                        "jB_guess_floor": task["jB_guess_floor"],
                        "jB_bounds": task["jB_bounds"],
                        "branch_label": task["branch_label"],
                        "return_profile": task["return_profile"],
                        "density_convention": DENSITY_CONVENTION["name"],
                    }
                ),
                flush=True,
            )

            for local_idx, aQstar in enumerate(a_values):
                global_idx = int(task["point_indices"][local_idx])
                branch_guess = _jB_guess_from_aQstar(aQstar, task)
                current_jB_guess = (
                    float(last_success_jB)
                    if bool(task.get("use_continuation", True)) and last_success_jB is not None
                    else branch_guess
                )
                point_meta = {
                    "chunk_id": chunk_id,
                    "point_index": global_idx,
                    "aQstar_input": float(aQstar),
                    "log_path": str(log_path),
                    "branch_label": task["branch_label"],
                    "T": float(task["T"]),
                    "nB_N": float(task["nB_N"]),
                    "nB_factor": float(task["nB_factor"]),
                    "B_one_forth": float(task["B_one_forth"]),
                    "ms": float(task["ms"]),
                    "tail_eps": float(task["tail_eps"]),
                    "n_mesh": int(task["n_mesh"]),
                    "tol_bvp": float(task["tol_bvp"]),
                    "max_nodes": int(task["max_nodes"]),
                    "jB_guess_initial": float(current_jB_guess),
                    "jB_guess_branch_estimate": float(branch_guess),
                    "jB_guess_mode": str(task["jB_guess_mode"]),
                    "jB_aQstar_slope": float(task["jB_aQstar_slope"]),
                }
                npz_path = _point_npz_path(output_dir, chunk_id, global_idx, aQstar)
                print(
                    json.dumps(
                        {
                            "event": "point_start",
                            "chunk_id": chunk_id,
                            "point_index": global_idx,
                            "aQstar": aQstar,
                            "jB_guess": current_jB_guess,
                            "jB_guess_branch_estimate": branch_guess,
                            "jB_guess_mode": task["jB_guess_mode"],
                        }
                    ),
                    flush=True,
                )
                try:
                    # Solve the scalar BVP first.  If profile reconstruction fails later,
                    # keep the scalar jB/tail result instead of losing the whole point.
                    result = solve_steady_front_1d_aQstar_rescale_bvp(
                        T=task["T"],
                        nB_N=task["nB_N"],
                        B_one_forth=task["B_one_forth"],
                        aQstar=aQstar,
                        ms=task["ms"],
                        tail_eps=task["tail_eps"],
                        n_mesh=task["n_mesh"],
                        tol_bvp=task["tol_bvp"],
                        max_nodes=task["max_nodes"],
                        jB_guess=current_jB_guess,
                        jB_bounds=tuple(task["jB_bounds"]) if task["jB_bounds"] is not None else None,
                        kappa_factor=task["kappa_factor"],
                        return_profile=False,
                        verb=task["verb_mode"],
                    )
                    has_profile = False
                    if bool(result.get("success", False)) and np.isfinite(_safe_float(result.get("jB", np.nan))):
                        last_success_jB = float(result["jB"])
                        if bool(task["return_profile"]):
                            print(
                                json.dumps(
                                    {
                                        "event": "profile_start",
                                        "chunk_id": chunk_id,
                                        "point_index": global_idx,
                                        "aQstar": aQstar,
                                        "jB_guess": last_success_jB,
                                    }
                                ),
                                flush=True,
                            )
                            try:
                                profile_result = solve_steady_front_1d_aQstar_rescale_bvp(
                                    T=task["T"],
                                    nB_N=task["nB_N"],
                                    B_one_forth=task["B_one_forth"],
                                    aQstar=aQstar,
                                    ms=task["ms"],
                                                tail_eps=task["tail_eps"],
                                    n_mesh=task["n_mesh"],
                                    tol_bvp=task["tol_bvp"],
                                    max_nodes=task["max_nodes"],
                                    jB_guess=last_success_jB,
                                    jB_bounds=tuple(task["jB_bounds"]) if task["jB_bounds"] is not None else None,
                                    kappa_factor=task["kappa_factor"],
                                    return_profile=True,
                                    verb=task["verb_mode"],
                                )
                                profile_has_arrays = all(k in profile_result for k in ["x", "a", "q", "u", "nB", "muB", "muK"])
                                if bool(profile_result.get("success", False)) and profile_has_arrays:
                                    result = profile_result
                                    has_profile = True
                                else:
                                    result["profile_message"] = str(profile_result.get("message", "profile solve did not return arrays"))
                                print(
                                    json.dumps(
                                        {
                                            "event": "profile_done",
                                            "chunk_id": chunk_id,
                                            "point_index": global_idx,
                                            "profile_saved": has_profile,
                                            "profile_message": str(result.get("profile_message", "")),
                                        }
                                    ),
                                    flush=True,
                                )
                            except Exception as profile_exc:
                                result["profile_message"] = f"profile reconstruction failed: {profile_exc}"
                                print(
                                    json.dumps(
                                        {
                                            "event": "profile_exception",
                                            "chunk_id": chunk_id,
                                            "point_index": global_idx,
                                            "message": result["profile_message"],
                                        }
                                    ),
                                    flush=True,
                                )
                    _save_result_npz(npz_path, result, point_meta)
                    summary = _point_summary_from_result(result, point_meta, npz_path, has_profile)
                    point_summaries.append(summary)
                    print(json.dumps({"event": "point_done", **summary}), flush=True)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    traceback.print_exc()
                    result = {
                        "success": False,
                        "message": f"point exception: {exc}",
                        "aQstar": float(aQstar),
                        "jB": np.nan,
                        "branch_label": task["branch_label"],
                    }
                    _save_result_npz(npz_path, result, point_meta)
                    summary = _point_summary_from_result(result, point_meta, npz_path, False)
                    point_summaries.append(summary)
                    print(json.dumps({"event": "point_exception", **summary}), flush=True)

    elapsed = time.perf_counter() - t0
    for summary in point_summaries:
        summary["chunk_elapsed_s"] = float(elapsed)
    return point_summaries


def _build_aQstar_values(args) -> list[float]:
    if args.aQstar_list:
        values = _parse_float_list(args.aQstar_list)
    else:
        if int(args.aQstar_num) > 0:
            base = np.linspace(float(args.aQstar_min), float(args.aQstar_max), int(args.aQstar_num))
        else:
            step = float(args.aQstar_step)
            if step <= 0.0:
                raise RuntimeError("--aQstar-step must be positive when --aQstar-num <= 0")
            base = list(np.arange(float(args.aQstar_min), float(args.aQstar_max) + 0.25 * step, step))
            if not base or abs(float(base[-1]) - float(args.aQstar_max)) > 1.0e-12:
                base.append(float(args.aQstar_max))
        high = _parse_float_list(args.aQstar_high_list) if args.aQstar_high_list else []
        values = list(base) + high
    seen = set()
    unique = []
    for val in values:
        key = round(float(val), 14)
        if key not in seen:
            seen.add(key)
            unique.append(float(val))
    return unique


def _build_tasks(args):
    n0 = 0.16 * const.MeV_fm**3
    nB_N = float(args.nB_factor * n0)
    a_values = _build_aQstar_values(args)
    chunks = _chunk_values(a_values, args.workers, args.chunk_size)

    jB_bounds = None
    if args.jB_bounds:
        vals = _parse_float_list(args.jB_bounds)
        if len(vals) != 2:
            raise RuntimeError("--jB-bounds must be a comma-separated pair: min,max")
        jB_bounds = (float(vals[0]), float(vals[1]))

    tasks = []
    next_point_idx = 1
    for chunk_id, chunk in enumerate(chunks, start=1):
        point_indices = list(range(next_point_idx, next_point_idx + len(chunk)))
        next_point_idx += len(chunk)
        tasks.append(
            {
                "chunk_id": chunk_id,
                "point_indices": point_indices,
                "aQstar_values": chunk,
                "T": float(args.T),
                "nB_N": nB_N,
                "nB_factor": float(args.nB_factor),
                "B_one_forth": float(args.B_one_forth),
                "ms": float(args.ms),
                "tail_eps": float(args.tail_eps),
                "n_mesh": int(args.n_mesh),
                "tol_bvp": float(args.tol_bvp),
                "max_nodes": int(args.max_nodes),
                "jB_guess": float(args.jB_guess),
                "jB_guess_mode": str(args.jB_guess_mode),
                "jB_aQstar_slope": float(args.jB_aQstar_slope),
                "jB_guess_floor": float(args.jB_guess_floor),
                "use_continuation": bool(args.use_continuation),
                "jB_bounds": jB_bounds,
                "kappa_factor": float(args.kappa_factor),
                "branch_label": BRANCH_LABEL,
                "return_profile": bool(args.return_profile),
                "output_dir": str(args.output_dir),
                "verb_mode": str(args.verb_mode),
            }
        )
    return tasks, a_values, jB_bounds


def main():
    parser = argparse.ArgumentParser(description="Parallel compact-BVP steady-front scan over fixed aQstar values")
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--nB-factor", type=float, default=2.0, dest="nB_factor", help="Use nB_N = nB_factor * n0")
    parser.add_argument("--B-one-forth", type=float, default=180.0, dest="B_one_forth")
    parser.add_argument("--ms", type=float, default=0.0)
    parser.add_argument("--tail-eps", type=float, default=1.0e-8, dest="tail_eps", help="Compact endpoint uses s_end = 1 - tail_eps")
    parser.add_argument("--n-mesh", type=int, default=300, dest="n_mesh")
    parser.add_argument("--tol-bvp", type=float, default=1.0e-4, dest="tol_bvp")
    parser.add_argument("--max-nodes", type=int, default=50000, dest="max_nodes")
    parser.add_argument("--kappa-factor", type=float, default=1.0, dest="kappa_factor")
    parser.add_argument("--jB-guess", type=float, default=0.5, dest="jB_guess", help="Fallback constant jB guess; used when --jB-guess-mode=constant")
    parser.add_argument("--jB-guess-mode", type=str, choices=["linear-aQstar", "constant"], default="linear-aQstar", dest="jB_guess_mode", help="Default linear-aQstar uses jB ~= slope*aQstar from the previous BVP branch")
    parser.add_argument("--jB-aQstar-slope", type=float, default=1.2, dest="jB_aQstar_slope", help="Code-unit slope for jB ~= slope*aQstar; tuned from the successful low/mid-aQstar branch")
    parser.add_argument("--jB-guess-floor", type=float, default=0.02, dest="jB_guess_floor", help="Minimum jB guess used by the linear-aQstar strategy")
    parser.add_argument("--no-continuation", action="store_false", default=True, dest="use_continuation", help="Disable within-chunk continuation from the previous successful jB")
    parser.add_argument("--jB-bounds", type=str, default="1e-5,5.0", dest="jB_bounds", help="Comma-separated positive jB bounds, e.g. 1e-5,5.0. Use empty string for unbounded exp map.")
    parser.add_argument("--aQstar-list", type=str, default="", help="Comma-separated fixed aQstar values. If set, overrides the default range/high list.")
    parser.add_argument("--aQstar-min", type=float, default=0.05, dest="aQstar_min")
    parser.add_argument("--aQstar-max", type=float, default=0.6, dest="aQstar_max")
    parser.add_argument("--aQstar-num", type=int, default=0, dest="aQstar_num", help="If >0, use linspace; default 0 uses --aQstar-step")
    parser.add_argument("--aQstar-step", type=float, default=0.05, dest="aQstar_step", help="Default step samples 0.05 to 0.6 densely and includes aQstar=0.6")
    parser.add_argument("--aQstar-high-list", type=str, default="", dest="aQstar_high_list", help="Optional extra high aQstar values, comma-separated")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=20, dest="chunk_size", help="Number of aQstar points per worker chunk. Default 20 keeps one continuous branch continuation.")
    parser.add_argument("--verb-mode", type=str, choices=["off", "simple", "full"], default="simple", dest="verb_mode")
    parser.add_argument("--profile", action="store_true", default=True, dest="return_profile", help="Reconstruct/save profile arrays for every point")
    parser.add_argument("--no-profile", action="store_false", dest="return_profile", help="Save scalar results only")
    parser.add_argument("--output-dir", type=Path, default=None, dest="output_dir")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / f"steady_front_bvp_run_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = int(args.chunk_size) if int(args.chunk_size) > 0 else None
    args.chunk_size = chunk_size
    tasks, a_values, jB_bounds = _build_tasks(args)
    if not tasks:
        raise RuntimeError("No aQstar tasks were built; check --aQstar-list or --aQstar-num")

    summary_path = args.output_dir / "trial_summary.jsonl"
    config_path = args.output_dir / "run_config.json"

    config = {
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "solver": "solve_steady_front_1d_aQstar_rescale_bvp",
        "branch_policy": "muK-rich only",
        "T": args.T,
        "nB_factor": args.nB_factor,
        "B_one_forth": args.B_one_forth,
        "ms": args.ms,
        "tail_eps": args.tail_eps,
        "n_mesh": args.n_mesh,
        "tol_bvp": args.tol_bvp,
        "max_nodes": args.max_nodes,
        "kappa_factor": args.kappa_factor,
        "jB_guess": args.jB_guess,
        "jB_guess_mode": args.jB_guess_mode,
        "jB_aQstar_slope": args.jB_aQstar_slope,
        "jB_guess_floor": args.jB_guess_floor,
        "use_continuation": args.use_continuation,
        "jB_bounds": jB_bounds,
        "aQstar_values": a_values,
        "aQstar_min": args.aQstar_min,
        "aQstar_max": args.aQstar_max,
        "aQstar_num": args.aQstar_num,
        "aQstar_step": args.aQstar_step,
        "aQstar_high_list": args.aQstar_high_list,
        "branch_label": BRANCH_LABEL,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "verb_mode": args.verb_mode,
        "return_profile": args.return_profile,
        "n_tasks": sum(len(task["aQstar_values"]) for task in tasks),
        "n_chunks": len(tasks),
        "output_dir": str(args.output_dir),
        "density_convention": DENSITY_CONVENTION,
    }
    config_path.write_text(json.dumps(config, indent=2))

    print(json.dumps({"event": "run_start", **config}), flush=True)
    t0 = time.perf_counter()
    summaries: list[dict] = []

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            futures = {pool.submit(_run_one_chunk, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    chunk_summaries = future.result()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    chunk_summaries = []
                    for local_idx, aQstar in enumerate(task["aQstar_values"]):
                        chunk_summaries.append(
                            {
                                "chunk_id": task["chunk_id"],
                                "point_index": task["point_indices"][local_idx],
                                "aQstar_input": float(aQstar),
                                "aQstar": float(aQstar),
                                "success": False,
                                "message": f"future exception: {exc}",
                                "profile_saved": False,
                                "npz_path": "",
                                "branch_label": task["branch_label"],
                                "density_convention": DENSITY_CONVENTION["name"],
                            }
                        )
                for summary in chunk_summaries:
                    summaries.append(summary)
                    summary_file.write(json.dumps(summary) + "\n")
                    summary_file.flush()
                    print(json.dumps({"event": "summary", **summary}), flush=True)

    elapsed = time.perf_counter() - t0
    n_success = sum(1 for s in summaries if s.get("success"))
    n_profile = sum(1 for s in summaries if s.get("profile_saved"))
    best = None
    if summaries:
        best = min(
            summaries,
            key=lambda s: abs(float(s.get("tail_residual_norm", np.inf) or np.inf)),
        )

    final_report = {
        "event": "run_done",
        "elapsed_s": elapsed,
        "n_points": len(summaries),
        "n_success": n_success,
        "n_profile_saved": n_profile,
        "summary_path": str(summary_path),
        "config_path": str(config_path),
    }
    if best is not None:
        final_report["best_point_index"] = best.get("point_index")
        final_report["best_aQstar"] = best.get("aQstar")
        final_report["best_jB"] = best.get("jB")
        final_report["best_tail_residual_norm"] = best.get("tail_residual_norm")
        final_report["best_message"] = best.get("message", "")
    print(json.dumps(final_report), flush=True)


if __name__ == "__main__":
    main()
