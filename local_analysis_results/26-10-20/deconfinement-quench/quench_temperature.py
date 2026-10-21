"""Temperature of freshly deconfined udd quark matter (pure deconfinement quench).

Question: inside 0^- < x < 0^+, neutrons deconfine by the strong interaction
(flavor frozen: n -> udd) and the new quark state thermalizes (entanglement ->
thermal entropy).  Energy is conserved, so the thermalized state obeys the slow
front jump conditions P(0+) = P(0-), h/nB(0+) = h/nB(0-).  For massless quarks
h = 4(P+U), so nB(0+) is fixed and T_q follows from one equation.

Upstream: QMC-RMF3 pure neutron matter.  Quark side: massless u,d,s + gluons +
bag U, charge neutral (2 n_u = n_d + n_s), udd means n_u:n_d:n_s = 1:2:0, mu_s=0.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64"))
sys.path.insert(0, ROOT)

import numpy as np
import RMFsolver.phase_velocity as pv
import RMFsolver.constants as const

HC3 = const.MeV_fm**3          # MeV^3 fm^3
N0 = 0.16                      # fm^-3
B14 = 189.1566                 # MeV, cold PNM/SQM coexistence at 3 n0
U = B14**4                     # MeV^4 (solver works in MeV units)
PI2 = np.pi**2


def mu_from_n(n, T):
    """Massless g=6 flavor: n = mu^3/pi^2 + mu T^2 (MeV^3). Exact cubic root."""
    if n == 0.0:
        return 0.0
    lo, hi = 0.0, (PI2 * n) ** (1 / 3) + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid**3 / PI2 + mid * T**2 < n:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def P_flavor(mu, T):
    return mu**4 / (4 * PI2) + mu**2 * T**2 / 2 + 7 * PI2 * T**4 / 60


def s_flavor(mu, T):
    return mu**2 * T + 7 * PI2 * T**3 / 15


def udd_thermal(nB_MeV3, T):
    """Thermal pressure (MeV^4) and entropy density (MeV^3) of udd at fixed nB."""
    mu_u, mu_d = mu_from_n(nB_MeV3, T), mu_from_n(2 * nB_MeV3, T)
    Pth = P_flavor(mu_u, T) + P_flavor(mu_d, T) + P_flavor(0.0, T) + 8 * PI2 * T**4 / 45
    s = s_flavor(mu_u, T) + s_flavor(mu_d, T) + s_flavor(0.0, T) + 32 * PI2 * T**3 / 45
    return Pth, s, mu_u, mu_d


def mu_udd_cold(P):
    """Cold udd baryon chemical potential at pressure P (MeV^4) = h/nB at T=0."""
    return (1 + 2 ** (4 / 3)) ** 0.75 * (4 * PI2 * (P + U)) ** 0.25


def upstream(nB_ratio, T0):
    st = pv._validated_pnm_state_from_nB(nB_ratio * N0 * HC3, T0)  # MeV^3 in, MeV^4 out
    st["s_over_nB_0minus"] = (
        (st["h_over_nB_0minus"] - st["muB_0minus"]) / T0 if T0 > 0 else 0.0
    )
    return st


def quench(st):
    P, hn = st["P_0minus"], st["h_over_nB_0minus"]
    nB3 = 4 * (P + U) / hn                  # MeV^3, exact for massless quarks
    nB_plus = nB3 / HC3                     # fm^-3
    target = P + U                          # MeV^4
    gap = hn - mu_udd_cold(P)               # >0 <=> udd quench state exists
    out = {"nB_0plus_fm3": nB_plus, "enthalpy_gap_MeV": gap}
    if gap <= 0:
        out.update(T_q=np.nan, s_over_nB_0plus=np.nan)
        return out
    lo, hi = 0.0, 400.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if udd_thermal(nB3, mid)[0] < target:
            lo = mid
        else:
            hi = mid
    Tq = 0.5 * (lo + hi)
    Pth, s, mu_u, mu_d = udd_thermal(nB3, Tq)
    cQ = (mu_u**2 + mu_d**2) / nB3          # low-T s/(nB T), MeV^-1
    out.update(
        T_q=Tq,
        s_over_nB_0plus=s / nB3,
        mu_u=mu_u,
        mu_d=mu_d,
        cQ=cQ,
        T_q_lowT=np.sqrt(2 * gap / cQ),     # Sommerfeld estimate
        pressure_residual=Pth / target - 1,
    )
    return out


if __name__ == "__main__":
    rows = []
    # (1) reference point, (2) density scan at T(0-)=10 MeV, (3) temperature scan at 3 n0
    grid = [(3.0, 10.0)] + [(r, 10.0) for r in (1.5, 2, 3, 4, 5, 6, 7, 8, 10)] \
        + [(3.0, T) for T in (0.5, 5, 20, 40, 60, 80)]
    for r, T0 in grid:
        st = upstream(r, T0)
        q = quench(st)
        row = {"nB_0minus_over_n0": r, "T_0minus": T0,
               "P_0minus_MeV_fm3": st["P_0minus"] / HC3, "muB_0minus": st["muB_0minus"],
               "h_over_nB_0minus": st["h_over_nB_0minus"],
               "s_over_nB_0minus": st["s_over_nB_0minus"],
               "mu_udd_cold": mu_udd_cold(st["P_0minus"]), **q}
        rows.append({k: float(v) for k, v in row.items()})
        print(json.dumps({k: round(v, 5) for k, v in rows[-1].items()}))
    with open(os.path.join(HERE, "quench_temperature.json"), "w") as f:
        json.dump({"B_one_forth_MeV": B14, "param": "QMC-RMF3 PNM", "rows": rows}, f, indent=1)
