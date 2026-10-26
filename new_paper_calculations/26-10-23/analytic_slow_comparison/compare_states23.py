"""Slow (non-relativistic) analytical branches vs previous relativistic methods and BVPs.

Run with the isolated SciPy runtime on PYTHONPATH. Prints one line per method/state
and pickles everything to the path given as argv[1].
"""
import pickle, sys, time, warnings
import numpy as np
from scipy.optimize import fsolve
warnings.simplefilter("ignore")
import RMFsolver.constants as const
import RMFsolver.phase_velocity as pv

N0 = 0.16 * const.MeV_fm**3
B = 180.0
XI = -0.5
STATES = [(20.0, 3.5), (5.0, 3.0)]  # state (10, 3n0) is in run_state1.log
T0PLUS = {10.0: [5.0, 1.0], 20.0: [5.0], 5.0: [1.0]}
ISO_A = {10.0: [0.6, np.nan], 20.0: [np.nan], 5.0: [np.nan]}
out = []


def run(tag, fn, **kw):
    t0 = time.time()
    try:
        r = fn(**kw)
    except Exception as exc:  # report, keep going
        r = {"success": False, "message": f"{type(exc).__name__}: {exc}"}
    dt = time.time() - t0
    u = r.get("u_0minus", np.nan)
    print(f"{tag:58s} ok={r.get('success')} u={u:.6e} a0+={r.get('a_0plus', np.nan):.5f} "
          f"T0+={r.get('T_0plus', np.nan):.4g} Tinf={r.get('T_inf', np.nan):.5g} "
          f"({dt:.0f}s) {'' if r.get('success') else str(r.get('message'))[:90]}", flush=True)
    out.append((tag, kw, {k: v for k, v in r.items() if np.isscalar(v) or isinstance(v, dict)}))
    return r


def consistent_quadratic_adiabatic(nB0, T0, T0p, P0, H):
    """Quadratic pressure with its own derivative n_B = dP/dmuB (thermodynamically consistent)."""
    Bag = B**4
    def eqs(v):
        muB, T = v
        P = -Bag + muB**4 / (108 * np.pi**2) + muB**2 * T**2 / 6
        nB = muB**3 / (27 * np.pi**2) + muB * T**2 / 3
        h = 4 * (P + Bag)
        return [(P - P0) / Bag, h / nB / H - 1]
    muB, Tinf = fsolve(eqs, [1200.0, 30.0], xtol=1e-14)
    A = 9 * np.pi * Tinf / (np.sqrt(2) * muB)  # T=0 intercept of the same quadratic isobar
    return muB, Tinf, A


for T0, nfac in STATES:
    nB0 = nfac * N0
    head = f"T0={T0:g} nB={nfac:g}n0"
    # ---------------- isothermal ----------------
    for a in ISO_A[T0]:
        lab = f"{head} iso a0+={'auto' if np.isnan(a) else a}"
        new = run(lab + " | NEW analytic slow (Landau D)", pv.analytic_velocity_isothermal_slow,
                  T=T0, nB_0minus=nB0, B_one_forth=B, a_0plus=a, xi=XI)
        if new.get("success"):
            invD_full = pv._microphysics_from_quark_state_energy(new["muB_0plus"], T0)["invD"]
            u_full = new["u_0minus"] * np.sqrt(new["invD_K"] / invD_full)
            print(f"{lab + ' | NEW analytic slow, full D (diag)':58s} u={u_full:.6e}  "
                  f"[Landau/full invD = {new['invD_K']/invD_full:.4f}]", flush=True)
            out.append((lab + " | NEW full-D diag", {}, {"u_0minus": u_full}))
        run(lab + " | OLD analytic relativistic", pv.analytic_velocity_isothermal,
            T_0minus=T0, nB_0minus=nB0, B_one_forth=B, a_0plus=a, xi=XI)
        a_bvp = new.get("a_0plus", a) if np.isnan(a) else a
        run(lab + " | NEW slow BVP", pv.solve_front_isothermal_slow,
            T=T0, nB_0minus=nB0, B_one_forth=B, a_0plus=a)
        run(lab + " | OLD relativistic BVP", pv.solve_front_isothermal,
            T=T0, nB_0minus=nB0, B_one_forth=B, a_0plus=a_bvp)
    # ---------------- adiabatic ----------------
    for T0p in T0PLUS[T0]:
        lab = f"{head} adb T0+={T0p:g}"
        newq = run(lab + " | NEW analytic slow (quadratic)", pv.analytic_velocity_adiabatic_slow,
                   T_0minus=T0, nB_0minus=nB0, B_one_forth=B, T_0plus=T0p, xi=XI)
        if newq.get("success"):
            muB_c, Tinf_c, A_c = consistent_quadratic_adiabatic(nB0, T0, T0p, newq["P_0minus"],
                                                               newq["h_0minus"] / nB0)
            print(f"{lab + ' | diag: consistent-quadratic endpoint':58s} muB={muB_c:.3f} "
                  f"Tinf={Tinf_c:.4f} A={A_c:.4f}  vs leading-density muB={newq['muB_inf']:.3f} "
                  f"Tinf={newq['T_inf']:.4f} A={newq['A_adb']:.4f}", flush=True)
            out.append((lab + " | consistent quadratic", {}, {"muB": muB_c, "T_inf": Tinf_c, "A": A_c}))
        semi = run(lab + " | NEW semi-analytic slow (full EOS)", pv.semi_analytic_velocity_adiabatic_slow,
                   T_0minus=T0, nB_0minus=nB0, B_one_forth=B, T_0plus=T0p, xi=XI)
        muB0 = semi.get("muB_0minus", newq.get("muB_0minus"))
        run(lab + " | OLD analytic relativistic (hybrid)", pv.analytic_velocity_bound,
            muB_0minus=muB0, T_0minus=T0, B_one_forth=B, T_0plus=T0p, xi=XI)
        run(lab + " | OLD semi-analytic relativistic", pv.semi_analytic_velocity_bound,
            muB_0minus=muB0, T_0minus=T0, B_one_forth=B, T_0plus=T0p, xi=XI)
        bvp = run(lab + " | NEW slow BVP", pv.solve_front_adiabatic_slow,
                  T_0minus=T0, nB_0minus=nB0, B_one_forth=B, T_0plus=T0p)
        if False and bvp.get("success"):  # old relativistic adiabatic BVP: ~15 min each, matched to 7 digits in run_state1.log
            run(lab + " | OLD relativistic BVP (same a0+)", pv.solve_front_energy_conserving_nK,
                T_0minus=T0, nB_0minus=nB0, B_one_forth=B, a_0plus=bvp["a_0plus"])

pickle.dump(out, open(sys.argv[1], "wb"))
print("saved")
