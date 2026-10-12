#!/usr/bin/env python3
"""Solve the cold rays that keep 1000/2000/5000 m/s off the T(0-) = 0 axis.

The 26-09-16 sweep left those three levels starting at 1.6, 3.5 and 7.3 MeV,
because every ray below their first accepted root ran out of target budget.
The 50--500 m/s contours reach 0.4 MeV and all of them approach the cold corner
(nB(0-) = 3 n0, T(0-) = 0) as the radial fraction goes to zero, so the missing
piece is short but it is exactly what makes the paper figure look unfinished.

This runs only the two scheduled angular nodes immediately below each of those
three levels: six targets, nothing else.  Results go to their own checkpoint,
so the 26-09-16 payload stays byte-identical and is never resumed -- it cannot
be, since its continuation profiles were never copied back from the cluster.

    ./bin/python3 new_paper_calculations/26-09-23/isothermal_cold_endpoints/run_cold_endpoints.py
    ... --dry-run          list the targets and verify inputs, solve nothing
    ... --target-timeout   per-contour-target budget in seconds (default 900)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
REPAIR_DIR = SCRIPT_DIR.parents[1] / "26-09-16/isothermal_numeric_repair"
REPO_ROOT = SCRIPT_DIR.parents[2]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-isothermal-cold")
for path in (str(REPO_ROOT), str(REPAIR_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

from _repair_common import DEFAULT_CONTROLS, atomic_save, load, load_baseline, sha256
from _repair_search import Repair

OUTPUT_NAME = "isothermal-cold-endpoints.npy"
COLD_LEVELS = (1000., 2000., 5000.)
NODES_PER_LEVEL = 2


def cold_targets(domain, repair_payload, levels=COLD_LEVELS, nodes=NODES_PER_LEVEL):
    """The `nodes` scheduled angles just below each level's coldest accepted root."""
    axis = np.asarray(domain["theta_axis_rad"], float)
    targets = []
    for level in levels:
        solved = [ray["theta_rad"] for ray in repair_payload["rays"].values()
                  if ray["level"] == level and ray["status"] == "success" and ray.get("roots")]
        if not solved:
            continue
        first = int(np.argmin(np.abs(axis - min(solved))))
        for node in range(max(0, first - nodes), first):
            targets.append((level, float(axis[node]), node))
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial-timeout", type=float, default=DEFAULT_CONTROLS["trial_timeout"])
    parser.add_argument("--target-timeout", type=float, default=DEFAULT_CONTROLS["target_timeout"])
    parser.add_argument("--nodes", type=int, default=NODES_PER_LEVEL,
                        help="Angular nodes to attempt below each level's coldest root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    baseline, hashes = load_baseline(SCRIPT_DIR)
    if json.loads((SCRIPT_DIR / "baseline-manifest.json").read_text())["sha256"] != hashes:
        raise RuntimeError("Immutable baseline hashes differ from baseline-manifest.json")
    domain = baseline[0]
    repair_payload = load(REPAIR_DIR / "isothermal-contour-repair.npy")
    if repair_payload["domain_fingerprint"] != domain["domain_fingerprint"]:
        raise RuntimeError("The 26-09-16 repair checkpoint uses a different domain")
    targets = cold_targets(domain, repair_payload, nodes=args.nodes)

    from RMFsolver import phase_velocity
    print(f"Loaded solver: {Path(phase_velocity.__file__).resolve()}", flush=True)
    print(f"Solver SHA256: {sha256(phase_velocity.__file__)}", flush=True)
    print(f"{len(targets)} cold targets, {args.target_timeout:g} s each at most:", flush=True)
    for level, theta, node in targets:
        print(f"  {level:>6g} m/s  node {node:>2}  theta {theta:.6f} rad", flush=True)
    if args.dry_run:
        return 0

    controls = dict(DEFAULT_CONTROLS, trial_timeout=args.trial_timeout,
                    target_timeout=args.target_timeout)
    repair = Repair(SCRIPT_DIR, baseline, hashes, controls=controls, output_name=OUTPUT_NAME)
    started = time.monotonic()
    for level, theta, node in targets:
        elapsed = time.monotonic()
        row = repair.solve_ray(level, theta)
        detail = ""
        if row["roots"]:
            root = row["roots"][0]
            detail = (f"  nB/n0={root['nB_0minus_over_n0']:.4f} T={root['T_0minus']:.4f} MeV"
                      f" f={root['radial_fraction']:.4f} u={root['velocity_m_s']:.2f} m/s")
        print(f"{level:>6g} m/s node {node:>2}: {row['status']}"
              f" in {time.monotonic() - elapsed:.1f} s{detail}", flush=True)

    payload = load(SCRIPT_DIR / OUTPUT_NAME)
    payload["payload_kind"] = "isothermal_cold_endpoints"
    payload["cold_targets"] = [(level, theta, node) for level, theta, node in targets]
    payload["source_checkpoint_sha256"] = sha256(REPAIR_DIR / "isothermal-contour-repair.npy")
    atomic_save(SCRIPT_DIR / OUTPUT_NAME, payload)
    accepted = sum(len(row.get("roots") or []) for row in payload["rays"].values())
    print(f"Accepted {accepted} new cold roots in {time.monotonic() - started:.1f} s", flush=True)
    print(f"Checkpoint: {SCRIPT_DIR / OUTPUT_NAME}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
