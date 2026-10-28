"""High-temperature isothermal contours (polar mesh, T(0-) >= 8 MeV).

From the repository root:

    PYTHONPATH=local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:. \
    ./bin/python3 -B new_paper_calculations/26-10-27/isothermal/highT/run_highT.py --method both --workers 4

Outputs: highT/isothermal-highT-{analytic,numerical}.npy (+ numerical -profiles/).
Rerunning resumes; --restart writes a new suffixed payload after a code change.
"""
import argparse
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np  # noqa: E402

import _iso_engine as engine  # noqa: E402
import polar_mesh  # noqa: E402

TIMEOUT_S = dict(analytic=90.0, numerical=120.0)
SMOKE_THETA_DEG = (15.0, 45.0, 75.0)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=("analytic", "numerical", "both"), default="both")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--smoke", action="store_true", help="three rays (15, 45, 75 deg) into highT/smoke/")
    p.add_argument("--dry-run", action="store_true", help="build the mesh and report counts only")
    p.add_argument("--restart", action="store_true", help="new suffixed payload if the identity changed")
    p.add_argument("--repair", action="store_true",
                   help="re-evaluate only unresolved/timeout points in place with the current code")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    out = HERE / "smoke" if args.smoke else HERE
    theta = np.radians(SMOKE_THETA_DEG) if args.smoke else None
    mesh = polar_mesh.build_mesh(workers=args.workers, theta=theta)
    pending = int(np.sum(mesh["point_status"] == "pending"))
    print(f"mesh {mesh['x'].shape}: {pending} pending points, fingerprint {mesh['fingerprint'][:12]}",
          flush=True)
    if args.dry_run:
        return 0
    methods = ("analytic", "numerical") if args.method == "both" else (args.method,)
    for method in methods:
        started = time.perf_counter()
        path = out / f"isothermal-highT-{method}.npy"
        if args.repair:
            payload = engine.repair(path, args.workers, timeout_s=TIMEOUT_S[method], progress=True)
        else:
            payload = engine.run(mesh, path, method, args.workers, timeout_s=TIMEOUT_S[method],
                                 restart=args.restart, progress=True)
        statuses = payload["status"].astype(str)
        counts = {s: int(np.sum(statuses == s)) for s in np.unique(statuses)}
        print(f"{method}: {counts}; {payload['new_evaluations']} new evaluations, "
              f"{time.perf_counter() - started:.0f} s wall -> {payload['path']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
