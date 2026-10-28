"""Cross-check the static boundaries against every ray of the saved 26-09-16 domain.

The Task 1 test checks 6 of the 24 saved rays; this checks all rays with T > 0
(theta = 0 is the T = 0 axis, not a valid state). Criterion as in the test:
|rho - rho_saved| <= 1e-3 for both boundaries. Writes verify_saved_domain.json.
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np  # noqa: E402

import _iso_boundaries as boundaries  # noqa: E402
import _iso_common as common  # noqa: E402

SAVED_DOMAIN = common.ROOT / "new_paper_calculations/26-09-16/isothermal_numeric_repair/isothermal-domain.npy"
CRITERION = 1e-3


def main(workers=8):
    saved = np.load(SAVED_DOMAIN, allow_pickle=True).item()
    saved_rho = {}
    for kind, prefix in (("rho_inner", "lower_phase"), ("rho_outer", "upper_phase")):
        x = np.asarray(saved[prefix + "_nB_0minus"], dtype=float)
        x = x / common.N0_MEV3 if np.max(x) > 100 else x
        saved_rho[kind], theta = common.to_polar(x, np.asarray(saved[prefix + "_T_0minus_MeV"], dtype=float))
    rays = [float(t) for t in theta if t > 0]
    solved = common.map_parallel(boundaries.ray_boundaries, rays, workers)
    rows = []
    for t, ray in zip(rays, solved):
        index = int(np.flatnonzero(theta == t)[0])
        rows.append(dict(index=index, theta_deg=float(np.degrees(t)),
                         **{f"d{kind}": float(abs(ray[kind] - saved_rho[kind][index]))
                            for kind in ("rho_inner", "rho_outer")}))
    worst = {key: max(row[key] for row in rows) for key in ("drho_inner", "drho_outer")}
    out = dict(rays=rows, worst=worst, criterion=CRITERION,
               passed=bool(all(np.isfinite(v) and v <= CRITERION for v in worst.values())))
    (HERE / "verify_saved_domain.json").write_text(json.dumps(out, indent=1))
    print(f"{len(rows)} rays, worst {worst}, passed={out['passed']}")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
