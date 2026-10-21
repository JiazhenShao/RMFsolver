"""Independent check of the slow-front solutions against the manuscript's system.

Uses only the manuscript formulas (Appendix A EOS, Madsen rate with (muB/3)^5,
HP93 D_K with alpha_s=0.4) and constants; no RMFsolver solver internals.
"""
import pickle, sys, warnings
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import gamma as G, zeta as Z
import RMFsolver.constants as const
warnings.simplefilter("ignore")

PI = np.pi
ALPHA_S = 0.4
H_D = G(8 / 3) * Z(5 / 3) * (2 * PI) ** (2 / 3) / 6
C_K = 16 / (5 * PI**5) * const.G_Fermi**2 * np.sin(const.Cabibbo_angle_rad)**2 * np.cos(const.Cabibbo_angle_rad)**2


def mu_of_n(n, T):
    """Invert n = mu^3/pi^2 + mu T^2 (three colours, one flavour)."""
    p, q = PI**2 * T**2, PI**2 * n
    d = np.sqrt((q / 2)**2 + (p / 3)**3)
    mu = np.cbrt(q / 2 + d) + np.cbrt(q / 2 - d)
    for _ in range(2):
        mu -= (mu**3 / PI**2 + mu * T**2 - n) / (3 * mu**2 / PI**2 + T**2)
    return mu


def eos(nB, a, T, B):
    mus = np.array([mu_of_n(nB, T), mu_of_n(nB * (1 + a), T), mu_of_n(nB * (1 - a), T)])
    ns = np.array([nB, nB * (1 + a), nB * (1 - a)])
    P = np.sum(mus**4 / (4 * PI**2) + mus**2 * T**2 / 2 + 7 * PI**2 * T**4 / 60) + 8 * PI**2 * T**4 / 45 - B
    s = np.sum(mus**2 * T + 7 * PI**2 * T**3 / 15) + 32 * PI**2 * T**3 / 45
    h = np.sum(mus * ns) + T * s
    return dict(P=P, h=h, muB=mus.sum(), muK=mus[1] - mus[2], nB=nB, nK=a * nB, a=a, T=T)


def invD(muB, T):
    qD = np.sqrt(2 * ALPHA_S / (3 * PI)) * muB
    return 8 * 3 * ALPHA_S**2 / PI * (H_D * T**(5 / 3) / qD**(2 / 3) + PI**3 * T**2 / (12 * qD))


def rate(muB, muK, T):
    return C_K * (muB / 3)**5 * muK * (muK**2 + 4 * PI**2 * T**2)


def state_from_nK(nK, res, B):
    """Local state on the manuscript's conserved path at given K density."""
    P0 = res["P_0minus"]
    if res["_closure"] == "isothermal":
        T = res["_kwargs"]["T"]
        f = lambda nB: eos(nB, nK / nB, T, B)["P"] - P0
        nB = brentq(f, nK * (1 + 1e-14), 50 * res["nB_0minus"], xtol=1e-14 * res["nB_0minus"], rtol=1e-15)
        return eos(nB, nK / nB, T, B)
    # Adiabatic: P = P0 and h/nB = h0/nB0. For the massless bag EOS h = 4(P+B)
    # exactly (checked in check_profile), so nB is fixed and T solves P = P0.
    nB = 4 * (P0 + B) / (res["h_0minus"] / res["nB_0minus"])
    a = nK / nB
    f = lambda T: eos(nB, a, T, B)["P"] - P0
    return eos(nB, a, brentq(f, 0.0, 300.0, xtol=1e-13, rtol=1e-15), B)


def check_profile(res):
    B = res["_kwargs"]["B_one_forth"]**4
    jB = res["jB"]
    T_arr = res["T"]
    # (A) EOS and conservation, recomputed from (nB, a, T) of the profile.
    mine = [eos(nB, a, T, B) for nB, a, T in zip(res["nB"], res["a"], T_arr)]
    rel = lambda x, y: np.max(np.abs(np.asarray(x) - np.asarray(y)) / np.maximum(np.abs(np.asarray(y)), 1.0))
    out = {
        "eos_P_vs_code": rel([m["P"] for m in mine], res["P"]),
        "eos_muB_vs_code": rel([m["muB"] for m in mine], res["muB"]),
        "eos_muK_vs_code": np.max(np.abs(np.array([m["muK"] for m in mine]) - res["muK"])),
        "eos_h_vs_code": rel([m["h"] for m in mine], res["h"]),
        "P_const": np.max(np.abs(np.array([m["P"] for m in mine]) - res["P_0minus"])) / abs(res["P_0minus"]),
        "baryon_flux": np.max(np.abs(res["u"] * res["nB"] / jB - 1)),
    }
    if res["_closure"] == "isothermal":
        out["T_const"] = np.max(np.abs(T_arr / res["_kwargs"]["T"] - 1))
    else:
        out["identity_h_eq_4(P+B)"] = max(abs(m["h"] / (4 * (m["P"] + B)) - 1) for m in mine)
        out["hu_const"] = np.max(np.abs(np.array([m["h"] for m in mine]) * res["u"] /
                                        (res["h_0minus"] * res["u_0minus"]) - 1))
    # (B) ODE residuals with independent splines and independent Gamma_K, D_K.
    s, x = res["s_coord"], res["x"]
    keep = np.r_[True, np.diff(x) > 0]
    s, x = s[keep], x[keep]
    nK, JK = res["nK"][keep], res["jK"][keep]
    muB = np.array([m["muB"] for m in mine])[keep]
    muK = np.array([m["muK"] for m in mine])[keep]
    T = T_arr[keep]
    a = res["a"][keep]
    dxds = np.gradient(x, s)  # only used as a cross-check below
    lam_scale = x[1] / -np.log1p(-s[1])  # kappa/lambda from the compact map
    dxds = lam_scale / (1 - s)
    dnK_dx = CubicSpline(s, nK)(s, 1) / dxds
    dJK_dx = CubicSpline(s, JK)(s, 1) / dxds
    Gam = rate(muB, muK, T)
    with np.errstate(divide="ignore"):
        D = 1 / invD(muB, T)
    flux_res = JK - (jB * a - D * dnK_dx)
    reac_res = dJK_dx + Gam
    # Skip the last few tail nodes where |Gamma| and gradients underflow toward roundoff.
    inner = s < 1 - 1e-5
    out["flux_eq_resid_rel"] = np.max(np.abs(flux_res[inner & np.isfinite(D)])) / (jB * (res["a_0minus"] - res["a_inf"]))
    out["reaction_eq_resid_rel"] = np.max(np.abs(reac_res[inner])) / np.max(np.abs(Gam))
    # Integral form, robust to differentiation noise: JK(x) - JK(0) + int_0^x Gamma dx = 0.
    integral = CubicSpline(s, Gam * dxds).antiderivative()(s)
    out["reaction_integral_resid_rel"] = np.max(np.abs(JK - JK[0] + integral)) / abs(JK[0] - JK[-1])
    # (C) Boundary conditions.
    out["JK0_minus_jB_a0minus"] = JK[0] / (jB * res["a_0minus"]) - 1
    out["muK_tail_MeV"] = muK[-1]
    out["a_tail"] = a[-1]
    if np.isnan(res["_kwargs"].get("a_0plus", np.nan)) and res["_closure"] == "isothermal":
        m0 = mine[0]
        out["ceiling_resid_MeV"] = m0["muB"] + m0["a"] * m0["muK"] - res["muB_0minus"]
    return out


def shoot(res, jB, x_max_factor=200.0):
    """Integrate the manuscript ODEs from 0+ with trial jB; classify the failure mode."""
    B = res["_kwargs"]["B_one_forth"]**4
    a0m = res["a_0minus"]
    nK0 = res["nK_0plus"]
    def rhs(x, y):
        nK, JK = y
        st = state_from_nK(max(nK, 1e-30), res, B)
        return [(jB * st["a"] - JK) * invD(st["muB"], st["T"]), -rate(st["muB"], st["muK"], st["T"])]
    def overshoot(x, y):
        return y[0] - 1e-9 * nK0
    overshoot.terminal, overshoot.direction = True, -1
    def turn(x, y):
        st = state_from_nK(max(y[0], 1e-30), res, B)
        return jB * st["a"] - y[1]
    turn.terminal, turn.direction = True, 1
    L = 1 / res["lambda_inf"]
    sol = solve_ivp(rhs, (0, x_max_factor * L), [nK0, jB * a0m], method="LSODA",
                    events=(overshoot, turn), rtol=1e-11, atol=[1e-14 * nK0, 1e-14 * jB])
    if sol.t_events[0].size:
        return "overshoot", sol.t[-1] / L
    if sol.t_events[1].size:
        return "turn", sol.t[-1] / L
    return "none", sol.t[-1] / L


def shoot_eigenvalue(res, rel=1e-3, iters=40):
    jB = res["jB"]
    lo, hi = jB * (1 - rel), jB * (1 + rel)
    mlo, mhi = shoot(res, lo)[0], shoot(res, hi)[0]
    if mlo == mhi:
        return dict(error=f"no bracket: {mlo} at both ends")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        mode, reach = shoot(res, mid)
        if mode == mlo:
            lo = mid
        elif mode == mhi:
            hi = mid
        else:
            break
        if (hi - lo) / mid < 1e-11:
            break
    jB_shoot = 0.5 * (lo + hi)
    return dict(jB_shoot=jB_shoot, rel_diff=res["jB"] / jB_shoot - 1, low_mode=mlo, high_mode=mhi,
                bracket_rel=(hi - lo) / jB_shoot, reach_in_decay_lengths=reach)


if __name__ == "__main__":
    cases = pickle.load(open(sys.argv[1], "rb"))
    for name, res in cases.items():
        if not res["success"]:
            print(name, "NOT CONVERGED:", res["status"], res["message"], flush=True)
            continue
        prof = check_profile(res)
        print(name, "profile:", {k: float(f"{v:.3g}") for k, v in prof.items()}, flush=True)
        print(name, "shooting:", shoot_eigenvalue(res), flush=True)
