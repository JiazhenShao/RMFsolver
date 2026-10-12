#!/usr/bin/env python3
"""
Cluster-ready driver for steady-front shooting trials.

This script mirrors the interactive test trial from 26-07-17-1.ipynb, but runs a
small grid of initial guesses in parallel so multiple shooting basins can be
explored at once. Each worker:
- runs solve_steady_front_2d(...) with a selectable diagnostic mode
- prints diagnostics to stdout with a trial prefix
- writes the same diagnostics to a per-trial log file
- saves a successful profile to .npz

Diagnostic modes:
- simple: one concise line per shooting evaluation
- full: detailed stage-by-stage diagnostics from the solver
- off: suppress solver diagnostics inside each worker
now
Usage example:
    python steady_front_scan.py --workers 8

Saved steady-front outputs follow the current phase_velocity density convention:
- P_f, E_f, n_B are single-species with spin included and color excluded
- quark species densities include color
- nB_QM = (n_u + n_d + n_s) / 3
- nK_QM = (n_d - n_s) / 2
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import RMFsolver.constants as const
from RMFsolver.phase_velocity import solve_steady_front_2d


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


def _sanitize_float_for_name(x: float) -> str:
    s = f"{x:.6g}"
    return s.replace("+", "").replace("-", "m").replace(".", "p")


def _save_result_npz(npz_path: Path, result: dict, trial_meta: dict):
    payload = {"message": np.array(str(result.get("message", "")))}
    for key, value in trial_meta.items():
        payload[f"meta_{key}"] = np.array(value)
    for key, value in result.items():
        if key.startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            payload[key] = value
        elif np.isscalar(value) or isinstance(value, str):
            payload[key] = np.array(value)
    np.savez(npz_path, **payload)


def _trial_output_paths(output_dir: Path, trial_id: int, jB_guess: float, aQstar_guess: float):
    stem = (
        f"trial_{trial_id:02d}"
        f"__jBguess_{_sanitize_float_for_name(jB_guess)}"
        f"__aQstar_{_sanitize_float_for_name(aQstar_guess)}"
    )
    return output_dir / f"{stem}.log", output_dir / f"{stem}.npz"


def _run_one_trial(task: dict):
    trial_id = int(task["trial_id"])
    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path, npz_path = _trial_output_paths(output_dir, trial_id, task["jB_guess"], task["aQstar_guess"])
    prefixed_stdout = PrefixedStream(sys.stdout, prefix=f"[trial {trial_id:02d}] ")

    t0 = time.perf_counter()
    summary = {
        "trial_id": trial_id,
        "jB_guess": float(task["jB_guess"]),
        "aQstar_guess": float(task["aQstar_guess"]),
        "muK_rich": bool(task["muK_rich"]),
        "density_convention": DENSITY_CONVENTION["name"],
        "branch_label": "muK-rich" if task["muK_rich"] else "muK-poor",
        "log_path": str(log_path),
        "npz_path": str(npz_path),
        "success": False,
        "elapsed_s": np.nan,
        "message": "",
        "profile_saved": False,
        "jB": np.nan,
        "u_N": np.nan,
        "a_N": np.nan,
        "aQstar": np.nan,
        "Pi": np.nan,
        "muB_Qstar": np.nan,
        "muK_Qstar": np.nan,
        "muB_Q": np.nan,
        "nB_Q": np.nan,
        "nK_Q": np.nan,
        "a_end": np.nan,
        "q_end": np.nan,
        "root_success": False,
        "root_message": "",
    }

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = MultiWriter(prefixed_stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(
                json.dumps(
                    {
                        "event": "trial_start",
                        "trial_id": trial_id,
                        "T": task["T"],
                        "nB_N": task["nB_N"],
                        "B_one_forth": task["B_one_forth"],
                        "x_max": task["x_max"],
                        "n_eval": task["n_eval"],
                        "max_nfev": task["max_nfev"],
                        "jB_guess": task["jB_guess"],
                        "aQstar_guess": task["aQstar_guess"],
                        "muK_rich": task["muK_rich"],
                        "density_convention": DENSITY_CONVENTION["name"],
                    }
                ),
                flush=True,
            )
            try:
                result = solve_steady_front_2d(
                    T=task["T"],
                    nB_N=task["nB_N"],
                    B_one_forth=task["B_one_forth"],
                    muK_rich=task["muK_rich"],
                    ms=task["ms"],
                    x_max=task["x_max"],
                    n_eval=task["n_eval"],
                    rtol_ode=task["rtol_ode"],
                    atol_ode=task["atol_ode"],
                    root_tol=task["root_tol"],
                    max_nfev=task["max_nfev"],
                    jB_guess=task["jB_guess"],
                    aQstar_guess=task["aQstar_guess"],
                    return_profile=True,
                    verb=task["verb_mode"],
                )
                elapsed = time.perf_counter() - t0
                summary.update(
                    {
                        "success": bool(result.get("success", False)),
                        "elapsed_s": float(elapsed),
                        "message": str(result.get("message", "")),
                        "jB": float(result.get("jB", np.nan)),
                        "u_N": float(result.get("u_N", np.nan)),
                        "a_N": float(result.get("a_N", np.nan)),
                        "aQstar": float(result.get("aQstar", np.nan)),
                        "Pi": float(result.get("Pi", np.nan)),
                        "muB_Qstar": float(result.get("muB_Qstar", np.nan)),
                        "muK_Qstar": float(result.get("muK_Qstar", np.nan)),
                        "muB_Q": float(result.get("muB_Q", np.nan)),
                        "nB_Q": float(result.get("nB_Q", np.nan)),
                        "nK_Q": float(result.get("nK_Q", np.nan)),
                        "branch_label": str(result.get("branch_label", "muK-rich" if task["muK_rich"] else "muK-poor")),
                        "a_end": float(result.get("a_end", np.nan)),
                        "q_end": float(result.get("q_end", np.nan)),
                        "root_success": bool(result.get("_root_success", False)),
                        "root_message": str(result.get("_root_message", "")),
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "trial_done",
                            "trial_id": trial_id,
                            "success": summary["success"],
                            "elapsed_s": summary["elapsed_s"],
                            "jB": summary["jB"],
                            "aQstar": summary["aQstar"],
                            "nB_Q": summary["nB_Q"],
                            "nK_Q": summary["nK_Q"],
                            "a_end": summary["a_end"],
                            "q_end": summary["q_end"],
                            "message": summary["message"],
                        }
                    ),
                    flush=True,
                )
                has_profile = all(k in result for k in ["x", "a", "q", "u", "nB", "muB", "muK"])
                summary["profile_saved"] = bool(has_profile)
                if has_profile:
                    _save_result_npz(npz_path, result, summary)
                    print(
                        json.dumps({
                            "event": "npz_saved",
                            "trial_id": trial_id,
                            "path": str(npz_path),
                            "success": summary["success"],
                        }),
                        flush=True,
                    )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                summary.update(
                    {
                        "success": False,
                        "elapsed_s": float(elapsed),
                        "message": f"worker exception: {exc}",
                    }
                )
                print(json.dumps({"event": "trial_exception", "trial_id": trial_id, "error": str(exc)}), flush=True)
                traceback.print_exc()

    return summary


def _build_tasks(args):
    n0 = 0.16 * const.MeV_fm**3
    nB_N = float(args.nB_factor * n0)
    base_jB = float(args.jB_factor * nB_N)

    if args.jB_scales:
        jB_scales = [float(x) for x in args.jB_scales.split(",") if x.strip()]
    else:
        # Focus the long-domain scan around the nontrivial shooting basins found
        # near jB ~ 0.49, 0.54, and 0.74 for nB_N = 2 n0.
        jB_scales = [0.197, 0.220, 0.300]

    if args.aQstar_list:
        aQstar_list = [float(x) for x in args.aQstar_list.split(",") if x.strip()]
    else:
        aQstar_list = [0.178, 0.196, 0.257]

    tasks = []
    trial_id = 1
    for scale, aQstar_guess in itertools.product(jB_scales, aQstar_list):
        tasks.append(
            {
                "trial_id": trial_id,
                "T": float(args.T),
                "nB_N": nB_N,
                "B_one_forth": float(args.B_one_forth),
                "ms": float(args.ms),
                "x_max": float(args.x_max),
                "n_eval": int(args.n_eval),
                "rtol_ode": float(args.rtol_ode),
                "atol_ode": float(args.atol_ode),
                "root_tol": float(args.root_tol),
                "max_nfev": int(args.max_nfev),
                "jB_guess": float(scale * base_jB),
                "aQstar_guess": float(aQstar_guess),
                "muK_rich": bool(args.muK_rich),
                "output_dir": str(args.output_dir),
                "verb_mode": str(args.verb_mode),
            }
        )
        trial_id += 1
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Parallel steady-front shooting driver for cluster runs")
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--nB-factor", type=float, default=2.0, dest="nB_factor", help="Use nB_N = nB_factor * n0")
    parser.add_argument("--B-one-forth", type=float, default=180.0, dest="B_one_forth")
    parser.add_argument("--ms", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=5.0e6, dest="x_max")
    parser.add_argument("--n-eval", type=int, default=300, dest="n_eval")
    parser.add_argument("--rtol-ode", type=float, default=1.0e-8, dest="rtol_ode")
    parser.add_argument("--atol-ode", type=float, default=1.0e-10, dest="atol_ode")
    parser.add_argument("--root-tol", type=float, default=1.0e-5, dest="root_tol")
    parser.add_argument("--max-nfev", type=int, default=60, dest="max_nfev")
    parser.add_argument("--jB-factor", type=float, default=1.0e-6, dest="jB_factor", help="Base jB guess is jB_factor * nB_N")
    parser.add_argument("--jB-scales", type=str, default="", help="Comma-separated multipliers applied to the base jB guess")
    parser.add_argument("--aQstar-list", type=str, default="", help="Comma-separated aQstar guesses")
    parser.add_argument("--muK-rich", action="store_true", default=True, dest="muK_rich", help="Follow the muK-rich steady-front branch")
    parser.add_argument("--muK-poor", action="store_false", dest="muK_rich", help="Follow the muK-poor steady-front branch")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--verb-mode", type=str, choices=["off", "simple", "full"], default="simple", dest="verb_mode")
    parser.add_argument("--output-dir", type=Path, default=None, dest="output_dir")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / f"steady_front_run_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _build_tasks(args)
    summary_path = args.output_dir / "trial_summary.jsonl"
    config_path = args.output_dir / "run_config.json"

    config = {
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "T": args.T,
        "nB_factor": args.nB_factor,
        "B_one_forth": args.B_one_forth,
        "ms": args.ms,
        "x_max": args.x_max,
        "n_eval": args.n_eval,
        "rtol_ode": args.rtol_ode,
        "atol_ode": args.atol_ode,
        "root_tol": args.root_tol,
        "max_nfev": args.max_nfev,
        "jB_factor": args.jB_factor,
        "jB_scales": args.jB_scales,
        "aQstar_list": args.aQstar_list,
        "muK_rich": args.muK_rich,
        "workers": args.workers,
        "verb_mode": args.verb_mode,
        "n_tasks": len(tasks),
        "output_dir": str(args.output_dir),
        "density_convention": DENSITY_CONVENTION,
    }
    config_path.write_text(json.dumps(config, indent=2))

    print(json.dumps({"event": "run_start", **config}), flush=True)
    t0 = time.perf_counter()
    summaries = []

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            futures = {pool.submit(_run_one_trial, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    summary = future.result()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    summary = {
                        "trial_id": task["trial_id"],
                        "jB_guess": task["jB_guess"],
                        "aQstar_guess": task["aQstar_guess"],
                        "muK_rich": bool(task["muK_rich"]),
                        "density_convention": DENSITY_CONVENTION["name"],
                        "branch_label": "muK-rich" if task["muK_rich"] else "muK-poor",
                        "success": False,
                        "elapsed_s": np.nan,
                        "message": f"future exception: {exc}",
                        "profile_saved": False,
                        "jB": np.nan,
                        "u_N": np.nan,
                        "a_N": np.nan,
                        "aQstar": np.nan,
                        "Pi": np.nan,
                        "muB_Qstar": np.nan,
                        "muK_Qstar": np.nan,
                        "muB_Q": np.nan,
                        "nB_Q": np.nan,
                        "nK_Q": np.nan,
                        "a_end": np.nan,
                        "q_end": np.nan,
                        "root_success": False,
                        "root_message": "",
                    }
                summaries.append(summary)
                summary_file.write(json.dumps(summary) + "\n")
                summary_file.flush()
                print(json.dumps({"event": "summary", **summary}), flush=True)

    elapsed = time.perf_counter() - t0
    n_success = sum(1 for s in summaries if s.get("success"))
    best = None
    if summaries:
        best = min(
            summaries,
            key=lambda s: abs(s.get("a_end", np.inf)) + abs(s.get("q_end", np.inf)),
        )

    final_report = {
        "event": "run_done",
        "elapsed_s": elapsed,
        "n_trials": len(summaries),
        "n_success": n_success,
        "summary_path": str(summary_path),
    }
    if best is not None:
        final_report["best_trial_id"] = best.get("trial_id")
        final_report["best_message"] = best.get("message", "")
        final_report["best_a_end"] = best.get("a_end", np.nan)
        final_report["best_q_end"] = best.get("q_end", np.nan)
    print(json.dumps(final_report), flush=True)


if __name__ == "__main__":
    main()
