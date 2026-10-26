"""Closed-form adiabatic speed with a thermodynamically consistent quadratic endpoint.

Same manuscript formulas as analytic_velocity_adiabatic_slow (ellipse, A, closed I2,
leading Landau D_K), but mu_B(inf), T(inf) solve P_quad = P(0-) and h/n_B = h(0-)/n_B(0-)
with n_B = dP_quad/dmu_B = mu_B^3/(27 pi^2) + mu_B T^2/3 instead of the leading term only.
"""
import sys, warnings
import numpy as np
from scipy.optimize import fsolve
warnings.simplefilter("ignore")
import RMFsolver.constants as const
import RMFsolver.phase_velocity as pv

N0, B, XI = 0.16 * const.MeV_fm**3, 180.0, -0.5
for T0, nfac, T0p in [(10.0, 3.0, 5.0), (10.0, 3.0, 1.0), (20.0, 3.5, 5.0), (5.0, 3.0, 1.0)]:
    ref = pv.analytic_velocity_adiabatic_slow(T0, nfac * N0, B, T_0plus=T0p, xi=XI)
    P0, H, Bag = ref["P_0minus"], ref["h_0minus"] / (nfac * N0), B**4
    def eqs(v):
        muB, T = v
        P = -Bag + muB**4 / (108 * np.pi**2) + muB**2 * T**2 / 6
        nB = muB**3 / (27 * np.pi**2) + muB * T**2 / 3
        return [(P - P0) / Bag, 4 * (P + Bag) / nB / H - 1]
    muB, Tinf = fsolve(eqs, [ref["muB_inf"], 30.0], xtol=1e-14)
    nB_inf = muB**3 / (27 * np.pi**2) + muB * Tinf**2 / 3
    A = 9 * np.pi * Tinf / (np.sqrt(2) * muB)
    a = 9 * np.pi / (np.sqrt(2) * muB) * np.sqrt(Tinf**2 - T0p**2)
    z = (T0p / Tinf)**2
    bracket = 24 / 7 - 3 * z**(1 / 6) - 3 / 7 * z**(7 / 6)
    gam = pv._weak_rate_from_mu_q(muB / 3)["gamma"]
    I2 = (9 * np.pi**(7 / 3) * gam / (8 * np.sqrt(2) * pv._TRANSPORT_H_CONST * pv._TRANSPORT_ALPHA_S**(5 / 3) * muB)
          * A**(7 / 3) * bracket)
    lam = nfac * N0 / nB_inf
    u = np.sqrt(2 * I2 / (lam**2 * (1 - a) * (1 + XI * a)))
    exact_nB = 4 * (P0 + Bag) / H
    print(f"T0={T0:g} nB={nfac:g}n0 T0+={T0p:g}: consistent-quadratic muB={muB:.3f} Tinf={Tinf:.4f} "
          f"A={A:.4f} a0+={a:.5f} u={u:.6e}  (nB_inf/exact-4(P+B)/H - 1 = {nB_inf/exact_nB-1:.1e}); "
          f"leading-density code: Tinf={ref['T_inf']:.4f} A={ref['A_adb']:.4f} a0+={ref['a_0plus']:.5f} u={ref['u_0minus']:.6e}")
