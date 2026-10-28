"""High-T verification: plotted-contour placement against direct roots, and
resolution independence of accepted numerical points. Writes verify_highT.json."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np  # noqa: E402

import _iso_common as common  # noqa: E402
import _iso_evaluate as evaluation  # noqa: E402
import _iso_plot as plot  # noqa: E402

PAYLOAD = HERE / "isothermal-highT-numerical.npy"
UPSAMPLE = 8
LEVELS = (100.0, 1000.0, 5000.0)
RAY_DEGREES = (20.0, 40.0, 60.0, 80.0)
RESOLUTION_POINTS = ((15.0, 0.5), (15.0, 0.9999), (45.0, 0.5), (45.0, 0.9999), (75.0, 0.5), (75.0, 0.9999))
TIGHT = dict(n_mesh=600, tol_bvp=2e-5, path_rtol=2e-7, tail_eps=1e-9)


def _point(task):
    """Physical point at fraction f on ray i of the saved mesh."""
    mesh, i, f = task
    b = mesh["boundaries"][i]
    rho = b["rho_inner"] + f * (b["rho_outer"] - b["rho_inner"])
    x, T = common.to_physical(rho, mesh["theta"][i])
    return float(x), float(T)


def direct_root(task):
    """Illinois iteration in the radial fraction on log v - log level; |v/level - 1| <= 1e-3."""
    mesh, i, level, f_low, f_high, v_low, v_high = task
    g_low, g_high, side, evaluations = np.log(v_low / level), np.log(v_high / level), 0, 0
    for _ in range(12):
        f = (f_low * g_high - f_high * g_low) / (g_high - g_low)
        record, _ = evaluation.evaluate("numerical", *_point((mesh, i, f)))
        evaluations += 1
        if not record["accepted"]:
            return dict(status="unresolved", message=record["message"])
        g = np.log(record["speed_m_s"] / level)
        if abs(record["speed_m_s"] / level - 1) <= 1e-3:
            x, T = _point((mesh, i, f))
            return dict(status="resolved", f=f, x=x, T=T, speed=record["speed_m_s"], evaluations=evaluations)
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
    settings = dict(common.NUMERICAL, **TIGHT)
    result = pv.solve_front_isothermal_slow(T, x * common.N0_MEV3, common.PHYSICS["B_one_forth"], **settings)
    tight = result["u_0minus"] * common.C_M_S if result["success"] else np.nan
    return dict(x=x, T=T, speed=speed, tight=tight, relative_change=abs(tight / speed - 1),
                solution_method=result.get("solution_method"))


def main(workers=4):
    payload = common.load(PAYLOAD)
    mesh = payload["mesh"]
    x_fine, T_fine, logv_fine = plot.contour_arrays(payload, UPSAMPLE)
    rays = [int(np.argmin(abs(np.degrees(mesh["theta"]) - d))) for d in RAY_DEGREES]
    tasks, plotted = [], []
    for i in rays:
        speeds = np.where(payload["status"][i].astype(str) == "accepted", payload["speed_m_s"][i], np.nan)
        for level in LEVELS:
            j = next((k for k in range(len(speeds) - 1) if speeds[k] <= level < speeds[k + 1]), None)
            if j is None:
                continue
            row = logv_fine[i].filled(np.nan)
            m = next(k for k in range(len(row) - 1) if row[k] <= np.log10(level) < row[k + 1])
            w = (np.log10(level) - row[m]) / (row[m + 1] - row[m])
            plotted.append(dict(ray=i, theta_deg=float(np.degrees(mesh["theta"][i])), level=level,
                                x=float(x_fine[i, m] + w * (x_fine[i, m + 1] - x_fine[i, m])),
                                T=float(T_fine[i, m] + w * (T_fine[i, m + 1] - T_fine[i, m]))))
            tasks.append((mesh, i, level, mesh["fractions"][j], mesh["fractions"][j + 1],
                          speeds[j], speeds[j + 1]))
    roots = common.map_parallel(direct_root, tasks, workers)
    placement = []
    for drawn, root in zip(plotted, roots):
        entry = dict(drawn, direct=root)
        if root["status"] == "resolved":
            entry.update(dx_n0=abs(root["x"] - drawn["x"]), dT_MeV=abs(root["T"] - drawn["T"]))
        placement.append(entry)
    points = []
    for degrees, fraction in RESOLUTION_POINTS:
        i = int(np.argmin(abs(np.degrees(mesh["theta"]) - degrees)))
        j = int(np.argmin(abs(mesh["fractions"] - fraction)))
        if payload["status"][i, j] == "accepted":
            points.append((float(mesh["x"][i, j]), float(mesh["T"][i, j]), float(payload["speed_m_s"][i, j])))
    resolution = common.map_parallel(tight_resolve, points, workers)
    worst = dict(dx_n0=max((p.get("dx_n0", np.nan) for p in placement), default=np.nan),
                 dT_MeV=max((p.get("dT_MeV", np.nan) for p in placement), default=np.nan),
                 tight_relative_change=max(r["relative_change"] for r in resolution))
    report = dict(placement=placement, resolution=resolution, worst=worst,
                  criteria=dict(dx_n0=0.01, dT_MeV=0.5, tight_relative_change=1e-3))
    (HERE / "verify_highT.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(worst, default=float))
    for p in placement:
        print(f"theta={p['theta_deg']:5.1f} level={p['level']:6g}: drawn ({p['x']:.4f}, {p['T']:.2f}) "
              f"direct {p['direct']['status']} dx={p.get('dx_n0', np.nan):.1e} dT={p.get('dT_MeV', np.nan):.1e}")
    for r in resolution:
        print(f"x={r['x']:.4f} T={r['T']:.2f}: v={r['speed']:.6g} tight={r['tight']:.6g} rel={r['relative_change']:.1e}")


if __name__ == "__main__":
    main()
