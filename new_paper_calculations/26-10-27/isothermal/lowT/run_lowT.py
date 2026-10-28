"""Low-temperature isothermal contours (row mesh, 0.01 <= T(0-) <= 10 MeV).

From the repository root:

    PYTHONPATH=local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:. \
    ./bin/python3 -B new_paper_calculations/26-10-27/isothermal/lowT/run_lowT.py --method both --workers 4

Outputs: lowT/isothermal-lowT-{analytic,numerical}.npy (+ numerical -profiles/).
Same resume/--restart/--repair semantics as highT/run_highT.py.
"""
import argparse
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np  # noqa: E402

import _iso_engine as engine  # noqa: E402
import row_mesh  # noqa: E402

TIMEOUT_S = dict(analytic=90.0, numerical=120.0)
SMOKE_ROWS = (0, 13, 26)   # T = 0.01, 0.1, 1 MeV


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=("analytic", "numerical", "both"), default="both")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--smoke", action="store_true", help="rows at 0.01, 0.1 and 1 MeV into lowT/smoke/")
    p.add_argument("--dry-run", action="store_true", help="build the mesh and report counts only")
    p.add_argument("--restart", action="store_true", help="new suffixed payload if the identity changed")
    p.add_argument("--repair", action="store_true",
                   help="re-evaluate unresolved/timeout points in place (standard attempts + documented variants)")
    p.add_argument("--points", nargs="+", metavar="I,J",
                   help="with --repair: only these mesh points, e.g. --points 8,28 10,29")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    out = HERE / "smoke" if args.smoke else HERE
    rows = row_mesh.T_ROWS[list(SMOKE_ROWS)] if args.smoke else row_mesh.T_ROWS
    mesh = row_mesh.build_mesh(T_rows=rows)
    pending = int(np.sum(mesh["point_status"] == "pending"))
    print(f"mesh {mesh['x'].shape}: {pending} pending points, fingerprint {mesh['fingerprint'][:12]}",
          flush=True)
    if args.dry_run:
        return 0
    methods = ("analytic", "numerical") if args.method == "both" else (args.method,)
    for method in methods:
        started = time.perf_counter()
        path = out / f"isothermal-lowT-{method}.npy"
        if args.repair:
            points = [tuple(int(k) for k in point.split(",")) for point in args.points] if args.points else None
            payload = engine.repair(path, args.workers, evaluator="_iso_evaluate:evaluate_repair",
                                   timeout_s=TIMEOUT_S[method], progress=True, points=points)
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
