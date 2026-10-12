import json
import numpy as np
import analyze as A

d = A.table()
widths = A.ray_widths()
print(f"benchmark points: {len(d['key'])}; analytic ok {d['analytic_ok'].sum()}, semi ok {d['semi_analytic_ok'].sum()}")

# 1. Rescaling check against live solves at other xi.
for xi in (-0.5, -0.95):
    live = A.load_rows(xi)
    worst = {m: 0.0 for m in A.METHODS}
    for k, r in live.items():
        i = int(np.flatnonzero(d["key"] == k)[0])
        for m in A.METHODS:
            if r[m]["success"] and np.isfinite(d[f"{m}_u0"][i]):
                pred = A.rescale(d[f"{m}_u0"][i], d[f"{m}_a"][i], xi)
                worst[m] = max(worst[m], abs(pred / (r[m]["u_0minus"] * A.common.C_M_S) - 1))
    print(f"rescaling vs live solve at xi={xi}: max rel diff {worst}")

# 2. Scan.
grid = np.round(np.arange(-0.99, -0.295, 0.005), 3)
rows = []
for xi in grid:
    s = A.speed_metrics(d, xi)
    s_all = A.speed_metrics(d, xi, lo=0.0, hi=np.inf)
    c, flags = A.contour_metrics(d, xi, widths)
    row = dict(xi=xi)
    for m in A.METHODS:
        row[f"{m}_rms"] = s[m]["rms"]; row[f"{m}_mae"] = s[m]["mae"]; row[f"{m}_bias"] = s[m]["bias"]
        row[f"{m}_rms_all"] = s_all[m]["rms"]
        per_level = [np.mean(c[m][L]) for L in A.LEVELS]
        row[f"{m}_contour"] = np.mean(per_level)
        row[f"{m}_contour_levels"] = per_level
        row[f"{m}_unbracketed"] = sum(flags[m].values())
    for metric in ("rms", "mae", "rms_all", "contour"):
        row[f"joint_{metric}"] = 0.5 * (row[f"analytic_{metric}"] + row[f"semi_analytic_{metric}"])
    rows.append(row)
print("n in 50-2000 window:", A.speed_metrics(d, -0.85)["analytic"]["n"], A.speed_metrics(d, -0.85)["semi_analytic"]["n"])

def best(key):
    r = min(rows, key=lambda r: r[key]); return r["xi"], r[key]
print("\nOptimum xi by metric (full analytic / semi / joint):")
for metric in ("rms", "mae", "rms_all", "contour"):
    print(f"  {metric:8s}", best(f"analytic_{metric}"), best(f"semi_analytic_{metric}"), best(f"joint_{metric}"))

print("\nxi    | RMS ln(u/u_num) 50-2000: full  semi  joint | mean|drho| contours: full  semi  joint | bias full semi")
for r in rows:
    if np.isclose(r["xi"], [-0.5, -0.6, -0.65, -0.7, -0.75, -0.8, -0.85, -0.9, -0.95, -0.98]).any() or r["xi"] in [best(f"joint_{m}")[0] for m in ("rms","contour","mae")]:
        print(f"{r['xi']:+.3f} | {r['analytic_rms']:.4f} {r['semi_analytic_rms']:.4f} {r['joint_rms']:.4f} | "
              f"{r['analytic_contour']:.4f} {r['semi_analytic_contour']:.4f} {r['joint_contour']:.4f} | "
              f"{r['analytic_bias']:+.3f} {r['semi_analytic_bias']:+.3f} | unbracketed {r['analytic_unbracketed']},{r['semi_analytic_unbracketed']}")

print("\nPer-level mean |delta rho| (full; semi):", A.LEVELS)
for xi in (-0.5, -0.7, -0.8, -0.85, -0.9, -0.95):
    r = rows[int(np.argmin(np.abs(grid - xi)))]
    print(f"{xi:+.2f}", np.round(r["analytic_contour_levels"], 4), np.round(r["semi_analytic_contour_levels"], 4))
np.save("scan.npy", dict(rows=rows), allow_pickle=True)
