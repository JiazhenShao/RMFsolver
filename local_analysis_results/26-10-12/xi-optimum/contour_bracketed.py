"""Along-ray contour displacement using only rays where all three crossings are bracketed."""
import numpy as np
import analyze as A

d = A.table()
widths = A.ray_widths()
grid = np.round(np.arange(-0.99, -0.295, 0.005), 3)


def displacement(xi, levels):
    per = {m: {L: [] for L in levels} for m in A.METHODS}
    for th in np.unique(d["theta"]):
        r = d["theta"] == th
        for L in levels:
            fn, kn = A.crossing(d["f"][r], np.log(d["u_num"][r]), np.log(L))
            if kn != "bracketed":
                continue
            res = {}
            for m in A.METHODS:
                u = A.rescale(d[f"{m}_u0"][r], d[f"{m}_a"][r], xi)
                ok = np.isfinite(u)
                res[m] = A.crossing(d["f"][r][ok], np.log(u[ok]), np.log(L))
            if all(kind == "bracketed" for _, kind in res.values()):
                for m in A.METHODS:
                    per[m][L].append(abs(res[m][0] - fn) * widths[int(th)])
    return {m: np.mean([np.mean(per[m][L]) for L in levels if per[m][L]]) for m in A.METHODS}


for levels in (A.LEVELS[:5], A.LEVELS):
    vals = [displacement(xi, levels) for xi in grid]
    full = np.array([v["analytic"] for v in vals])
    semi = np.array([v["semi_analytic"] for v in vals])
    joint = 0.5 * (full + semi)
    print(f"levels {levels}: full best {grid[full.argmin()]:+.3f}, semi best {grid[semi.argmin()]:+.3f}, "
          f"joint best {grid[joint.argmin()]:+.3f}; joint at -0.85 {joint[np.argmin(abs(grid + 0.85))]:.4f}, "
          f"min {joint.min():.4f}")
