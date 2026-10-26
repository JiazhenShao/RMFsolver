"""Pair corrected full-analytic records with semi-analytic records at identical states."""
import glob, numpy as np, sys
base = "new_paper_calculations/26-10-26/local/"
def load(folder, method):
    out = {}
    for f in glob.glob(base + folder + "/*.npy"):
        r = np.load(f, allow_pickle=True).item()
        if r.get("method") != method or r.get("status") != "accepted" and not r.get("result", {}).get("success"):
            continue
        p = r["point"]
        key = (round(p["T_0minus_MeV"], 9), round(p["nB_0minus_over_n0"], 9))
        out[key] = r
    return out
A = load("adiabatic-T0plus0-xi-minus0p85-second-order-records", "analytic")
S = load("adiabatic-T0plus0-xi-minus0p85-records", "semi_analytic")
r0 = next(iter(A.values())); print("record keys:", list(r0.keys()), "status", r0["status"])
rows = []
for k in A.keys() & S.keys():
    a, s = A[k]["result"], S[k]["result"]
    if not (a.get("success") and s.get("success")): continue
    rows.append((k[0], k[1], a["u_0minus"], s["u_0minus"], a["a_0plus"], s["a_0plus"], a["T_inf"], s["T_inf"],
                 a["velocity_m_s"], s["velocity_m_s"]))
R = np.array(rows); print("paired", len(R))
ratio = R[:,2]/R[:,3]
print("u_analytic/u_semi: min %.3f median %.3f max %.3f" % (ratio.min(), np.median(ratio), ratio.max()))
print("a ratio: min %.4f med %.4f max %.4f" % tuple(np.percentile(R[:,4]/R[:,5], [0,50,100])))
print("T_inf ratio: min %.4f med %.4f max %.4f" % tuple(np.percentile(R[:,6]/R[:,7], [0,50,100])))
bins = [0, .2, .3, .4, .5, .6, .7, .8, .9, 1]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (R[:,5] >= lo) & (R[:,5] < hi)
    if m.any():
        print(f"semi a0+ in [{lo:.1f},{hi:.1f}): n={m.sum():4d}  u ratio median {np.median(ratio[m]):.3f} range {ratio[m].min():.3f}-{ratio[m].max():.3f}  a ratio med {np.median(R[m,4]/R[m,5]):.4f}")
for lo, hi in [(40,60),(90,110),(180,220),(450,550),(900,1100),(1800,2200),(4500,5500)]:
    m = (R[:,9] >= lo) & (R[:,9] < hi)
    if m.any(): print(f"semi speed {lo}-{hi} m/s: n={m.sum():3d} u ratio median {np.median(ratio[m]):.3f}")
np.save("/private/tmp/claude-501/-Users-janshao-NSM-related/40f37acf-942e-4401-9f68-7847c0b03aa5/scratchpad/paired_second_order.npy", R)
