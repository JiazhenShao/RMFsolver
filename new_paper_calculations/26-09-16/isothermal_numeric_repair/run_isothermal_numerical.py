#!/usr/bin/env python3
"""Resume targeted, numerical-only isothermal contour repair."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-isothermal-repair")

from _repair_common import DEFAULT_CONTROLS, load, load_baseline, sha256, workers
from _repair_progress import ProgressRepair


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--trial-timeout", type=float, default=DEFAULT_CONTROLS["trial_timeout"])
    parser.add_argument("--target-timeout", type=float, default=DEFAULT_CONTROLS["target_timeout"])
    parser.add_argument("--output", default=None, help="Checkpoint filename in this directory")
    parser.add_argument("--smoke", action="store_true", help="Four cluster targets plus tighter mesh/tail checks; separate checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Verify inputs and API, report saved counts, and perform no BVP work")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)
    if args.trial_timeout <= 0 or args.target_timeout <= 0:
        parser.error("Timeouts must be positive")
    output_name = args.output or ("isothermal-contour-smoke.npy" if args.smoke else "isothermal-contour-repair.npy")
    if Path(output_name).name != output_name or not output_name.endswith(".npy"):
        parser.error("--output must be a .npy filename, not a path")
    baseline, hashes = load_baseline(SCRIPT_DIR, check_live=not args.plot_only)
    manifest = SCRIPT_DIR / "baseline-manifest.json"
    if not manifest.is_file():
        raise RuntimeError("Copy baseline-manifest.json and all three baseline payloads with the cluster source")
    import json
    if json.loads(manifest.read_text())["sha256"] != hashes:
        raise RuntimeError("Immutable baseline hashes differ from baseline-manifest.json")
    count = int((baseline[2]["task_status"] == "success").sum())
    worker_count = workers(args.workers)
    print(f"Preserving {count}/600 accepted baseline points. Using up to {worker_count} CPU workers.", flush=True)
    if not args.plot_only:
        from RMFsolver import phase_velocity
        print(f"Loaded solver: {Path(phase_velocity.__file__).resolve()}", flush=True)
        print(f"Solver SHA256: {sha256(phase_velocity.__file__)}", flush=True)
    if args.dry_run:
        print("Stages: warm 500/1000 gaps; cold continuation; remaining levels; angular refinement; validation; plot.")
        print(f"Checkpoint: {SCRIPT_DIR / output_name}")
        return 0
    if args.plot_only:
        payload = load(SCRIPT_DIR / output_name) if (SCRIPT_DIR / output_name).exists() else None
        if payload is not None and payload["baseline_hashes"] != hashes:
            raise RuntimeError("Repair checkpoint uses a different baseline")
    else:
        from tqdm.auto import tqdm
        controls = dict(DEFAULT_CONTROLS, trial_timeout=args.trial_timeout, target_timeout=args.target_timeout)
        def bar_factory(**kwargs):
            return tqdm(**kwargs, dynamic_ncols=True,
                        bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]")
        print("Target total starts at 4 for smoke, or 210 for production; refinement/retries add targets.\n"
              "evals counts saved BVP evaluations in this invocation; finished targets may remain unresolved.", flush=True)
        repair = ProgressRepair(SCRIPT_DIR, baseline, hashes, controls=controls,
                                bar_factory=bar_factory, output_name=output_name)
        payload = repair.run(worker_count, smoke=args.smoke)
        repair.validate_representatives()
        # A run never rewrites any of the three baseline inputs.
        _, after = load_baseline(SCRIPT_DIR, check_live=False)
        if hashes != after:
            raise RuntimeError("Baseline input changed during the run")
        print(f"Saved {len(payload['points'])} additional solutions/anchors; contours complete: {payload['contours_complete']}")
        print("Convergence checks:", {key: row["status"] for key, row in payload["validation"].items()})
    from plot_isothermal_contours import plot_contours
    plot_contours(*baseline[::2], repair=payload, output_dir=SCRIPT_DIR,
                  stem="isothermal-contour-smoke" if args.smoke else "isothermal-numerical-repaired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
