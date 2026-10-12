"""Per-speed-band optimum xi and band-balanced joint RMS of ln(u_analytic/u_numerical)."""
import numpy as np
import analyze as A

d = A.table()
grid = np.round(np.arange(-0.99, -0.295, 0.005), 3)
BANDS = [(50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000)]


def err(m, xi):
    return np.log(A.rescale(d[f"{m}_u0"], d[f"{m}_a"], xi) / d["u_num"])


at = lambda arr, x: arr[int(np.argmin(abs(grid - x)))]
print("band [m/s] | n | mean numerical a(0+) | per method: best xi, RMS at best, RMS at -0.5, RMS at -0.85")
for lo, hi in BANDS:
    sel = (d["u_num"] >= lo) & (d["u_num"] < hi) & d["analytic_ok"]
    line = f"[{lo},{hi}) | {sel.sum()} | {np.mean(d['a_num'][sel]):.2f} |"
    for m in A.METHODS:
        rms = np.array([np.sqrt(np.mean(err(m, x)[sel] ** 2)) for x in grid])
        line += f" {m}: {grid[rms.argmin()]:+.3f} {rms.min():.3f} {at(rms, -0.5):.3f} {at(rms, -0.85):.3f} |"
    print(line)
for label, bands in (("50-1000", BANDS[:4]), ("50-2000", BANDS[:5]), ("50-5000", BANDS)):
    rms = {m: np.array([np.sqrt(np.mean([np.mean(err(m, x)[(d["u_num"] >= lo) & (d["u_num"] < hi) & np.isfinite(err(m, x))] ** 2)
                                         for lo, hi in bands])) for x in grid]) for m in A.METHODS}
    joint = 0.5 * (rms["analytic"] + rms["semi_analytic"])
    basin = grid[joint <= 1.01 * joint.min()]
    print(f"{label}: full {grid[rms['analytic'].argmin()]:+.3f}, semi {grid[rms['semi_analytic'].argmin()]:+.3f}, "
          f"joint {grid[joint.argmin()]:+.3f} [1% basin {basin.min():+.3f}, {basin.max():+.3f}], "
          f"joint RMS min {joint.min():.4f}, at -0.85 {at(joint, -0.85):.4f}, at -0.5 {at(joint, -0.5):.4f}")
