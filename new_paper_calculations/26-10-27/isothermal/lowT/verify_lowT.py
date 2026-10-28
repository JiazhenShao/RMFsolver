"""Low-T verification: seam with the high-T polar mesh at 10 MeV, plotted-crossing
placement against direct roots along rows, and resolution independence.
Writes verify_lowT.json."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE.parent / "highT")]

import numpy as np  # noqa: E402

import _iso_common as common  # noqa: E402
import _iso_plot as plot  # noqa: E402
import _iso_evaluate as evaluation  # noqa: E402

LOW = HERE / "isothermal-lowT-numerical.npy"
HIGH = HERE.parent / "highT/isothermal-highT-numerical.npy"
UPSAMPLE = 8
SEAM_LEVELS = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0)
CROSSING_LEVELS = (100.0, 1e4, 1e6)
ROW_TEMPERATURES = (0.01, 0.1, 1.0)
RESOLUTION_POINTS = ((0.01, 0.001), (0.01, 0.9), (0.1, 0.01), (0.1, 0.95), (1.0, 0.1), (1.0, 0.99))
TIGHT = dict(n_mesh=600, tol_bvp=2e-5, path_rtol=2e-7, tail_eps=1e-9)


def _row_point(mesh, i, f):
    return float(mesh["x_inner"][i] + f * (mesh["x_outer"][i] - mesh["x_inner"][i])), float(mesh["T_rows"][i])


def direct_root(task):
    """Illinois iteration in the row fraction on log v - log level; |v/level - 1| <= 1e-3."""
    mesh, i, level, f_low, f_high, v_low, v_high = task
    g_low, g_high, side, evaluations = np.log(v_low / level), np.log(v_high / level), 0, 0
    for _ in range(14):
        f = (f_low * g_high - f_high * g_low) / (g_high - g_low)
        record, _ = evaluation.evaluate("numerical", *_row_point(mesh, i, f))
        evaluations += 1
        if not record["accepted"]:
            return dict(status="unresolved", message=record["message"])
        g = np.log(record["speed_m_s"] / level)
        if abs(record["speed_m_s"] / level - 1) <= 1e-3:
            x, T = _row_point(mesh, i, f)
            return dict(status="resolved", f=f, x=x, T=T, evaluations=evaluations)
        if g * g_high > 0:
            f_high, g_high = f, g
            g_low = g_low / 2 if side == -1 else g_low
            side = -1
        else:
            f_low, g_low = f, g
            g_high = g_high / 2 if side == 1 else g_high
            side = 1
    return dict(status="unconverged")


def tight_resolve(task):
    x, T, speed = task
    from RMFsolver import phase_velocity as pv
    result = pv.solve_front_isothermal_slow(T, x * common.N0_MEV3, common.PHYSICS["B_one_forth"],
                                            **dict(common.NUMERICAL, **TIGHT))
    tight = result["u_0minus"] * common.C_M_S if result["success"] else np.nan
    return dict(x=x, T=T, speed=speed, tight=tight, relative_change=abs(tight / speed - 1))


def _crossing_on_row(x, logv_row, level):
    row = logv_row.filled(np.nan)
    for k in range(len(row) - 1):
        if np.isfinite(row[k]) and np.isfinite(row[k + 1]) and row[k] <= np.log10(level) < row[k + 1]:
            w = (np.log10(level) - row[k]) / (row[k + 1] - row[k])
            return float(x[k] + w * (x[k + 1] - x[k]))
    return np.nan


def seam(low, high):
    """x where each level crosses T = 10 MeV: top low-T row vs the high-T polar contours."""
    x_low, _, logv_low = plot.contour_arrays(low, UPSAMPLE, "logit")
    top = int(np.argmin(abs(low["mesh"]["T_rows"] - 10.0)))
    x_h, T_h, logv_h = plot.contour_arrays(high, UPSAMPLE)
    lines = plot.contour_lines(x_h, T_h, logv_h, SEAM_LEVELS)
    rows = []
    for level in SEAM_LEVELS:
        crossings = []
        for segment in lines[level]:
            for (xa, Ta), (xb, Tb) in zip(segment[:-1], segment[1:]):
                if (Ta - 10.0) * (Tb - 10.0) <= 0 and Ta != Tb:
                    crossings.append(xa + (10.0 - Ta) / (Tb - Ta) * (xb - xa))
        x_row = _crossing_on_row(x_low[top], logv_low[top], level)
        x_poly = min(crossings, key=lambda c: abs(c - x_row)) if crossings and np.isfinite(x_row) else np.nan
        rows.append(dict(level=level, x_low_row=x_row, x_high_polar=float(x_poly), dx_n0=abs(x_row - x_poly)))
    return rows


def main(workers=4):
    low, high = common.load(LOW), common.load(HIGH)
    mesh = low["mesh"]
    seam_rows = seam(low, high)
    x_fine, _, logv_fine = plot.contour_arrays(low, UPSAMPLE, "logit")
    tasks, drawn = [], []
    for T in ROW_TEMPERATURES:
        i = int(np.argmin(abs(mesh["T_rows"] - T)))
        speeds = np.where(low["status"][i].astype(str) == "accepted", low["speed_m_s"][i], np.nan)
        for level in CROSSING_LEVELS:
            j = next((k for k in range(len(speeds) - 1)
                      if np.isfinite(speeds[k]) and np.isfinite(speeds[k + 1]) and speeds[k] <= level < speeds[k + 1]), None)
            if j is None:
                continue
            drawn.append(dict(T=float(mesh["T_rows"][i]), level=level,
                              x=_crossing_on_row(x_fine[i], logv_fine[i], level)))
            tasks.append((mesh, i, level, mesh["fractions"][j], mesh["fractions"][j + 1], speeds[j], speeds[j + 1]))
    roots = common.map_parallel(direct_root, tasks, workers)
    placement = [dict(d, direct=r, dx_n0=abs(r["x"] - d["x"]) if r["status"] == "resolved" else np.nan)
                 for d, r in zip(drawn, roots)]
    points = []
    for T, fraction in RESOLUTION_POINTS:
        i = int(np.argmin(abs(mesh["T_rows"] - T)))
        j = int(np.argmin(abs(mesh["fractions"] - fraction)))
        if low["status"][i, j] == "accepted":
            points.append((float(mesh["x"][i, j]), float(mesh["T"][i, j]), float(low["speed_m_s"][i, j])))
    resolution = common.map_parallel(tight_resolve, points, workers)
    worst = dict(seam_dx_n0=float(np.nanmax([r["dx_n0"] for r in seam_rows])),
                 placement_dx_n0=float(np.nanmax([p["dx_n0"] for p in placement])),
                 tight_relative_change=float(max(r["relative_change"] for r in resolution)))
    (HERE / "verify_lowT.json").write_text(json.dumps(dict(seam=seam_rows, placement=placement,
                                                           resolution=resolution, worst=worst), indent=1,
                                                      default=float))
    print(json.dumps(worst))
    for r in seam_rows:
        print(f"seam {r['level']:6g}: row {r['x_low_row']:.4f} polar {r['x_high_polar']:.4f} dx={r['dx_n0']:.1e}")
    for p in placement:
        print(f"T={p['T']:g} level={p['level']:g}: drawn x={p['x']:.5f} direct {p['direct']['status']} dx={p['dx_n0']:.1e}")
    for r in resolution:
        print(f"x={r['x']:.5f} T={r['T']:g}: v={r['speed']:.6g} tight={r['tight']:.6g} rel={r['relative_change']:.1e}")


if __name__ == "__main__":
    main()
