"""Scan xi by exact rescaling of xi=-0.85 analytical speeds at numerical benchmark states."""
import json
import numpy as np
import common

XI0 = -0.85
METHODS = ("analytic", "semi_analytic")
LEVELS = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)


def load_rows(xi):
    with open(f"analytical_xi{xi:+.4f}.jsonl") as handle:
        return {r["key"]: r for r in map(json.loads, handle)}


def rescale(u0, a, xi):
    return u0 * np.sqrt((1.0 + XI0 * a) / (1.0 + xi * a))


def table():
    pts = common.benchmark_points()
    rows = load_rows(XI0)
    keys = sorted(pts, key=lambda k: (pts[k]["theta_index"], pts[k]["radial_fraction"]))
    data = dict(key=np.array(keys),
                theta=np.array([pts[k]["theta_index"] for k in keys]),
                f=np.array([pts[k]["radial_fraction"] for k in keys]),
                nB=np.array([pts[k]["nB_0minus_over_n0"] for k in keys]),
                T=np.array([pts[k]["T_0minus_MeV"] for k in keys]),
                u_num=np.array([pts[k]["u_num_m_s"] for k in keys]),
                a_num=np.array([pts[k]["a_0plus_num"] for k in keys]))
    for m in METHODS:
        ok = np.array([bool(rows.get(k, {}).get(m, {}).get("success")) for k in keys])
        get = lambda name: np.array([rows[k][m].get(name) if (k in rows and rows[k][m].get(name) is not None) else np.nan
                                     for k in keys], dtype=float)
        data[f"{m}_ok"] = ok
        data[f"{m}_u0"] = np.where(ok, get("u_0minus") * common.C_M_S, np.nan)
        data[f"{m}_a"] = np.where(ok, get("a_0plus"), np.nan)
    return data


def ray_widths():
    domain = common.load(common.DOMAIN_PATH)
    widths = {}
    for p in domain["points"]:
        s = p["states"][0]
        widths[int(p["theta_index"])] = float(s["rho_outer_boundary"] - s["rho_inner_boundary"])
    return widths


def crossing(f, log_u, log_level):
    """Radial fraction where log u crosses log_level; linear in (f, log u), end-extrapolated."""
    order = np.argsort(f)
    f, log_u = f[order], log_u[order]
    if len(f) < 2:
        return np.nan, "none"
    above = log_u >= log_level
    if above.all():
        i, kind = 0, "extrapolated_low"
    elif not above.any():
        i, kind = len(f) - 2, "extrapolated_high"
    else:
        i, kind = int(np.argmax(above)) - 1, "bracketed"
        if i < 0:
            i, kind = 0, "extrapolated_low"
    df, dl = f[i + 1] - f[i], log_u[i + 1] - log_u[i]
    if dl == 0:
        return np.nan, kind
    return f[i] + (log_level - log_u[i]) * df / dl, kind


def speed_metrics(d, xi, lo=50.0, hi=2000.0):
    out = {}
    window = (d["u_num"] >= lo) & (d["u_num"] <= hi)
    for m in METHODS:
        u = rescale(d[f"{m}_u0"], d[f"{m}_a"], xi)
        e = np.log(u / d["u_num"])
        sel = window & np.isfinite(e)
        out[m] = dict(rms=np.sqrt(np.mean(e[sel] ** 2)), mae=np.mean(np.abs(e[sel])),
                      bias=np.mean(e[sel]), n=int(sel.sum()))
    return out


def contour_metrics(d, xi, widths, levels=LEVELS):
    """Along-ray contour displacement |delta rho| per level; bracketed numerical crossings only."""
    out = {m: {L: [] for L in levels} for m in METHODS}
    flags = {m: {L: 0 for L in levels} for m in METHODS}
    for th in np.unique(d["theta"]):
        r = d["theta"] == th
        for L in levels:
            fn, kn = crossing(d["f"][r], np.log(d["u_num"][r]), np.log(L))
            if kn != "bracketed":
                continue
            for m in METHODS:
                u = rescale(d[f"{m}_u0"][r], d[f"{m}_a"][r], xi)
                ok = np.isfinite(u)
                if ok.sum() < 2:
                    continue
                fm, km = crossing(d["f"][r][ok], np.log(u[ok]), np.log(L))
                if km != "bracketed":
                    flags[m][L] += 1
                out[m][L].append(abs(fm - fn) * widths[int(th)])
    return ({m: {L: np.array(v) for L, v in lv.items()} for m, lv in out.items()}, flags)
