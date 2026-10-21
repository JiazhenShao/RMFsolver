"""Checks for quench_temperature.py (run after it).

1. Production EOS: SQMsolver.PQM / nB_QM / nK_QM at the udd point reproduce the
   closed-form pressure, density and a = 1 used in quench_temperature.py.
2. Wolfram series at fixed P: h/nB = muB0 + c2 T^2 + c4 T^4,
   c2 = pi^2 (2+2^(1/3))(1+2^(4/3)) / (2^(2/3)(4+2^(2/3)) mu_u0) = (mu_u0^2+mu_d0^2)/(2 nB0),
   c4 = pi^4 (265 + 48 2^(1/3) + 6 2^(2/3)) / (408 mu_u0^3).
   Near threshold the exact T_q must approach the series root.
3. Threshold density at T(0-) = 10 MeV, and T(inf) (fully converted, mu_K = 0)
   on the same (P, h/nB) for the reference state.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from quench_temperature import *            # noqa: F401,F403  (same EOS, constants)
import RMFsolver.SQMsolver as sqm

K = 2 ** (1 / 3)
out = {}

# 1. production EOS at the 6 n0 udd quench state
st = upstream(6.0, 10.0); q = quench(st)
mu_u, mu_d, Tq = q["mu_u"], q["mu_d"], q["T_q"]
muB, muK = mu_u + mu_d, mu_d               # mu_s = 0, mu_B = mu_u+mu_d+mu_s, mu_K = mu_d-mu_s
P_prod = float(sqm.PQM(muB, muK, B14, Tq)); nB_prod = float(sqm.nB_QM(muB, muK, B14, Tq))
nK_prod = float(sqm.nK_QM(muB, muK, B14, Tq))
out["prod_EOS_P_rel_err"] = P_prod / st["P_0minus"] - 1
out["prod_EOS_nB_rel_err"] = nB_prod / (q["nB_0plus_fm3"] * HC3) - 1
out["prod_EOS_a"] = nK_prod / nB_prod


# 2. series root vs exact near threshold
def series_root(gap, muB0):
    mu0 = muB0 / (1 + 2 * K)
    c2 = PI2 * (2 + K) * (1 + 2 * K) / (K**2 * (4 + K**2) * mu0)
    c4 = PI2**2 * (265 + 48 * K + 6 * K**2) / (408 * mu0**3)
    T2_only = np.sqrt(gap / c2)
    T2T4 = np.sqrt((-c2 + np.sqrt(c2**2 + 4 * c4 * gap)) / (2 * c4))
    return T2_only, T2T4


def gap_at(r, T0=10.0):
    s = upstream(r, T0); return s["h_over_nB_0minus"] - mu_udd_cold(s["P_0minus"])

lo, hi = 5.0, 6.0                           # gap(5)<0<gap(6)
for _ in range(30):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if gap_at(mid) < 0 else (lo, mid)
out["threshold_nB_over_n0_at_T10"] = 0.5 * (lo + hi)
series = []
for r in (hi + 0.002, hi + 0.02, 5.6, 6.0, 8.0):
    s = upstream(r, 10.0); qq = quench(s)
    t2, t24 = series_root(qq["enthalpy_gap_MeV"], s["h_over_nB_0minus"] - qq["enthalpy_gap_MeV"])
    series.append({"nB/n0": r, "gap": qq["enthalpy_gap_MeV"], "T_q_exact": qq["T_q"],
                   "T_q_series_T2": t2, "T_q_series_T4": t24, "ratio_T4": qq["T_q"] / t24})
out["series_vs_exact"] = series


# 3. fully converted endpoint T(inf) on the reference (P, h/nB): mu_u=mu_d=mu_s=muB/3
def T_inf(st):
    P, hn = st["P_0minus"], st["h_over_nB_0minus"]
    nB3 = 4 * (P + U) / hn
    def Pth(T):
        m = mu_from_n(nB3, T); return 3 * P_flavor(m, T) + 8 * PI2 * T**4 / 45
    lo, hi = 0.0, 400.0
    if Pth(0.0) > P + U:
        return np.nan                       # below SQM threshold: no moving front
    for _ in range(200):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if Pth(mid) < P + U else (lo, mid)
    return 0.5 * (lo + hi)

out["T_inf_ref_3n0_T10"] = T_inf(upstream(3.0, 10.0))
out["T_inf_6n0_T10"] = T_inf(st)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "checks.json"), "w"), indent=1)
