"""Second-law bound on the temperature drop across the deconfinement interface.

The interface conserves baryon, K (= (n_d - n_s)/2; neutrons carry K = 1 per
baryon) and energy fluxes.  The first-order entropy current gives, to leading
order in u, T J_s = E - sum_a mu_a J_a, so per baryon flux j:

    J_s(0-)/j = (h/nB - mu_n) / T(0-)                         = s/nB(0-)
    J_s(0+)/j = (h/nB - mu_B - mu_K)|0+ / T(0+) = s/nB(0+) - mu_K (1-a)/T(0+)

mu_B + mu_K = mu_u + 2 mu_d.  The second law across the layer, J_s(0+) >= J_s(0-),
is the affinity of n -> u + 2d.  Energy conservation (q = 0) puts 0+ on the
conserved (P, h/nB) trajectory, so the inequality caps a(0+) and floors T(0+).
Isothermal model (heat bath, T(0+) = T(0-)): mu_n(0-) >= mu_B(0+) + mu_K(0+).
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from quench_temperature import *            # noqa: F401,F403
import RMFsolver.SQMsolver as sqm


def inf_endpoint(st):
    """Fully converted (mu_K = 0) massless uds on the same (P, h/nB)."""
    P, hn = st["P_0minus"], st["h_over_nB_0minus"]
    nB3 = 4 * (P + U) / hn
    def Pth(T):
        m = mu_from_n(nB3, T); return 3 * P_flavor(m, T) + 8 * PI2 * T**4 / 45
    lo, hi = 0.0, 400.0
    for _ in range(200):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if Pth(mid) < P + U else (lo, mid)
    T = 0.5 * (lo + hi)
    return 3 * mu_from_n(nB3, T), T


def crossing(x, y, level):
    """x where y (monotone decreasing along x) first falls to level; linear interp."""
    i = int(np.argmax(y < level))
    return x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])


out = {}
st = upstream(3.0, 10.0)
hn, mu_n, T_m, P = st["h_over_nB_0minus"], st["muB_0minus"], 10.0, st["P_0minus"]
s_m = (hn - mu_n) / T_m
muB_inf, T_inf = inf_endpoint(st)
tr = pv._analytic_layer_trajectory(P, hn, muB_inf, T_inf, B14, n_steps=4000)
a, muB, muK, T = tr["a"], tr["muB"], tr["muK"], tr["T"]
G = (hn - muB - muK) / T                    # entropy flux per baryon flux at 0+
s_plus = (hn - muB - a * muK) / T           # thermal s/nB at 0+ (Euler)

# Euler consistency against the production entropy, at 5 trajectory points
idx = np.linspace(0, len(a) - 2, 5).astype(int)
s_prod = np.array([float(sqm.entropyQM(muB[i], muK[i], B14, T[i])) /
                   float(sqm.nB_QM(muB[i], muK[i], B14, T[i])) for i in idx])
out["euler_vs_entropyQM_max_rel_err"] = float(np.max(np.abs(s_prod / s_plus[idx] - 1)))

A = tr["A_boundary"]; s_inf = G[0]; r = s_m / s_inf
a2 = crossing(a, G, s_m); T2 = float(np.interp(a2, a, T))
a_naive = crossing(a, s_plus, s_m); T_naive = float(np.interp(a_naive, a, T))
# quadratic closed form (Wolfram): 1 - a/A^2 = r sqrt(1 - a^2/A^2)
lo, hi = 0.0, A
for _ in range(200):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if 1 - mid / A**2 > r * np.sqrt(1 - (mid / A) ** 2) else (lo, mid)
out.update(T_0minus=T_m, s_over_nB_0minus=s_m, mu_n_0minus=mu_n, h_over_nB=hn,
           T_inf=T_inf, muB_inf=muB_inf, s_over_nB_inf=float(s_inf), A=A,
           entropy_budget_whole_front_per_baryon=float(s_inf - s_m),
           a_0plus_max_second_law=float(a2), T_0plus_min_second_law=T2,
           max_drop_T0minus_minus_T0plus=T_m - T2,
           a_0plus_max_quadratic=0.5 * (lo + hi),
           a_naive_s_only=float(a_naive), T_naive_s_only=T_naive,
           ln3=float(np.log(3)))

# isothermal model at the same state: code ceiling vs second-law ceiling
code = pv._solve_a_0plus_max(mu_n, P, T_m, B14)
def iso_affinity(ap):
    mB, mK = pv._solve_interface_0plus_from_local_a_and_Pi(ap, P, 0.0, B14, T_m)
    return mu_n - mB - mK, mK
aff_code, muK_code = iso_affinity(code["a_0plus_max"])
lo, hi = 1e-8, code["a_0plus_max"]
for _ in range(60):
    mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if iso_affinity(mid)[0] > 0 else (lo, mid)
out.update(iso_a_max_code=code["a_0plus_max"], iso_affinity_at_code_max_MeV=aff_code,
           iso_minus_1_minus_a_muK_MeV=-(1 - code["a_0plus_max"]) * muK_code,
           iso_a_max_second_law=0.5 * (lo + hi),
           iso_delta_MeV=mu_n - iso_affinity(1e-10)[0] * 0 - pv._solve_interface_0plus_from_local_a_and_Pi(1e-10, P, 0.0, B14, T_m)[0])
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "second_law_bound.json"), "w"), indent=1)
