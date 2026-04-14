import numpy as np
import math
import time
import warnings
import matplotlib.pyplot as plt
from fractions import Fraction
from scipy.interpolate import interp1d
from scipy.integrate import quad, solve_bvp, solve_ivp
from scipy.optimize import fsolve, approx_fprime, root_scalar, root, least_squares, minimize_scalar
from numdifftools import Gradient, Derivative
import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver.Solver import RMFsolve, RMFsolve_mu, RMFpressureSYM, RMFpressurePNM, pressure_RMF
from RMFsolver.Solver import _set_couplings, RMFsolvePNM, RMFsolvePNM_mu, RMFedensPNM, RMFbaryon_densityPNM

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# Public functions
__all__ = ["P_f", "E_f", "n_B", "nB_QM", "nK_QM", "nQM_em", "PQM", "PQM_em", "edensQM", "entropyQM", "uN", "solve_steady_front_2d", "solve_steady_front_2d_simple", "solve_steady_front_1d_aQstar", "solve_steady_front_1d_aQstar_rescaled", "solve_steady_front_1d_aQstar_rescale_bvp", "solve_steady_front_entropy", "extract_jB_curve_vs_aQstar", "vNtoQ_Pc", "vNtoQ_B", "vNtoQ_nc", "Get_Delta_n_max",
           "Get_Delta_P_max", "extract_contour_coords_num", "extract_contour_coords_ana",
           "Get_Two_as_pres", "Get_Two_as_dens", "z_time_evolution", "z_time_evolution1",
           "vNtoQ_fixB", "vNtoQ_Pc_fixB", "z_time_evolution2", "Get_aQstar_eta", "Get_aQstar_eta0",
           "Get_Two_as_dens"]


# helper functions
def _Ek(k, m):
    return np.sqrt(k*k + m*m)

def _kF(mu, m):
    dm = mu*mu - m*m
    return np.sqrt(dm) if dm > 0.0 else 0.0

def _log1p_exp_neg(a):
    # stable log(1 + exp(-a))
    if a > 50.0:
        return np.exp(-a)
    if a < -50.0:
        return -a
    return np.log1p(np.exp(-a))

def _smoothstep01(x):
    # C^1 smooth blend on [0,1]
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    return x*x*(3.0 - 2.0*x)

def _massless_fermion_pressure(mu, Tem):
    """
    Exact finite-T pressure for one massless fermion species with spin
    degeneracy included and color excluded.
    """
    return float(mu**4 / (12.0 * np.pi**2) + mu * mu * Tem * Tem / 6.0 + 7.0 * np.pi**2 * Tem**4 / 180.0)

def _massless_fermion_density(mu, Tem):
    """
    Exact finite-T net density dP/dmu for one massless fermion species with spin
    degeneracy included and color excluded.
    """
    return float(mu**3 / (3.0 * np.pi**2) + mu * Tem * Tem / 3.0)

def _massless_fermion_entropy(mu, Tem):
    """
    Exact finite-T entropy density for one massless fermion species with spin
    degeneracy included and color excluded.
    """
    if Tem <= 0.0:
        return 0.0
    return float(mu * mu * Tem / 3.0 + 7.0 * np.pi**2 * Tem**3 / 45.0)

def _binary_entropy_density_term(f_occ):
    """
    Return -[f log f + (1-f) log(1-f)] for a scalar occupation number.
    """
    f_occ = float(f_occ)
    if f_occ <= 0.0 or f_occ >= 1.0:
        return 0.0
    f_occ = min(max(f_occ, np.finfo(float).tiny), 1.0 - np.finfo(float).eps)
    return float(-(f_occ * np.log(f_occ) + (1.0 - f_occ) * np.log1p(-f_occ)))

def _gauge_pressure(T):
    """
    Exact thermal gauge-boson pressure used by PQM_em.
    """
    return float(16.0 * np.pi**2 * T**4 / 90.0)

def _gauge_energy(T):
    """
    Exact thermal gauge-boson energy density used by edensQM(..., include_em=True).
    """
    return float(16.0 * np.pi**2 * T**4 / 30.0)

def _gauge_entropy(T):
    """
    Exact thermal gauge-boson entropy density.
    """
    if T <= 0.0:
        return 0.0
    return float(32.0 * np.pi**2 * T**3 / 45.0)

def PNM(mu_B, Temp, param = para.paraQMCRMF3, NM_type = "PNM"):
    """
    Universal nuclear-matter pressure helper as a function of mu_B.
    """
    if NM_type == "Beta_eq":
        rmf_sol = RMFsolve_mu(mub=mu_B, Trmf=Temp, para=param, sigma_init=30, w0_init=20, r03_init=-3, mu_e_init=50, verb=False,)
        return float(np.asarray(pressure_RMF(rmf_sol)).item())

    if NM_type == "PNM":
        pre = RMFpressurePNM(input_num=mu_B, input_type="muB", Trmf=Temp, para=param, sigma_init=30, w0_init=20, r03_init=-3, mub_init=990, verb=False,)
        return float(np.asarray(pre).item())

    if NM_type == "SYM":
        pre = RMFpressureSYM(input_num=mu_B,input_type="muB", Trmf=Temp, para=param, sigma_init=30, w0_init=20, mub_init=990, verb=False,)
        return float(np.asarray(pre).item())

    raise ValueError("Nuclear matter type not defined.")

def PNM_n(nB, Temp, param = para.paraQMCRMF3, NM_type = "PNM"):
    """
    Universal nuclear-matter pressure helper as a function of n_B.
    """
    if NM_type == "Beta_eq":
        rmf_sol = RMFsolve(nbext=nB, Trmf=Temp, para=param, sigma_init=30, w0_init=20, r03_init=-3, mu_e_init=50, verb=False,)
        return float(np.asarray(pressure_RMF(rmf_sol)).item())

    if NM_type == "PNM":
        pre = RMFpressurePNM(input_num=nB, input_type="nB", Trmf=Temp, para=param, sigma_init=30, w0_init=20, r03_init=-3, mub_init=990, verb=False,)
        return float(np.asarray(pre).item())

    if NM_type == "SYM":
        pre = RMFpressureSYM(input_num=nB, input_type="nB", Trmf=Temp, para=param, sigma_init=30, w0_init=20, mub_init=990, verb=False,)
        return float(np.asarray(pre).item())

    raise ValueError("Nuclear matter type not defined.")

def edensNM(mu_B, Temp, param = para.paraQMCRMF3, ):
    edens = RMFedensPNM(input_num = mu_B, input_type = "muB", Trmf = Temp, para = param, 
        sigma_init = 30, w0_init = 20, r03_init = -3, mub_init = 990, verb = False
        )
    return float(edens.item())

def edensNM_n(nB, Temp, param = para.paraQMCRMF3, ):
    edens = RMFedensPNM(input_num = nB, input_type = "nB", Trmf = Temp, para = param, 
        sigma_init = 30, w0_init = 20, r03_init = -3, mub_init = 990, verb = False
        )
    return float(edens.item())

def hNM(mu_B, Temp):
    return PNM(mu_B, Temp) + edensNM(mu_B, Temp)

def hNM_n(nB, Temp):
    return PNM_n(nB, Temp) + edensNM_n(nB, Temp)

def nB_NM(mu_B, Temp, param = para.paraQMCRMF3, ):
    nB = RMFbaryon_densityPNM(
        input_num=mu_B,
        input_type="muB",
        Trmf=Temp,
        para=param,
        sigma_init=30,
        w0_init=20,
        r03_init=-3,
        mub_init=990,
        verb=False,
    )
    return float(np.asarray(nB).item())

def Pi_NM(mu_B, Temp, j_B):
    nB = nB_NM(mu_B, Temp)
    if nB <= 0:
        return np.nan
    uN = j_B / nB
    return hNM(mu_B, Temp) * uN * uN + PNM(mu_B, Temp)

# adaptive T-switch for free-fermion 
def Tswitch_free_pressure(mu, m, rtol = 1e-6, band = 1.25):
    """
    Return (T_lo, T_hi) to blend T=0 pressure with the finite-T integral.

    Logic:
    - If mu <= m (non-degenerate): use integral only.
    - If mu > m (degenerate): set switch where O(T^2) term changes P by ≲ rtol*P0.
    """
    mu = float(mu); m = float(m)

    if mu <= m:
        return (0.0, 0.0)

    # exact T=0 pressure (degenerate Fermi gas)
    if m == 0.0:
        P0 = mu**4 / (12.0*np.pi**2)
        f  = (mu*mu) / (np.pi**2)
    else:
        kF = _kF(mu, m)
        P0 = ((2.0/3.0 * kF**3 - m*m * kF) * mu
              + m**4 * np.log((kF + mu) / max(m, 1e-300))) / (8.0*np.pi**2)
        f  = (mu * kF) / (np.pi**2)

    if P0 <= 0.0 or f <= 0.0:
        return (0.0, 0.0)

    T_star = np.sqrt((6.0 * rtol * P0) / (np.pi**2 * f))
    return (T_star / band, T_star * band)

# free-fermion pressure
def P_f(mu, m, Tem, upB=np.inf, rtol=1e-6, Sommerfeld=False):
    """
    Pressure of a single free fermion flavor (particles + antiparticles).
    The normalization already includes spin degeneracy g=2, but not quark color.

    - If Sommerfeld=True: use Sommerfeld O(T^2) approximation:
        P(T) = P0(mu,m) + (pi^2/6) T^2 f(mu,m)
      with P0 the exact T=0 pressure and f(mu,m)=mu*kF/pi^2 (massive) or mu^2/pi^2 (massless).

    - If Sommerfeld=False: use the original adaptive T-switch + finite-T integral
      and smoothstep blend (code kept exactly as before).

    Parameters:
    - mu         : chemical potential
    - m          : mass
    - Tem        : temperature
    - upB        : numerical integral upper bound, default = np.inf
    - rtol       : controls adaptive switch width, default = 1e-6
    - Sommerfeld : whether to use Sommerfeld approximation, default = True

    """
    mu = float(mu); m = float(m); Tem = float(Tem)
    if Tem < 0.0:
        raise RuntimeError("Negative Temperature")
    if m == 0.0:
        return _massless_fermion_pressure(mu, Tem)

    # ---- Sommerfeld mode (fast) ----
    if Sommerfeld:
        # P0(mu,m)
        if mu <= m:
            P0 = 0.0
        elif m == 0.0:
            P0 = mu**4 / (12.0 * np.pi**2)
        else:
            kF = np.sqrt(max(mu*mu - m*m, 0.0))
            P0 = ((2.0*kF**3 - 3.0*m*m*kF)*mu
                  + 3.0*m**4*np.log((kF + mu)/max(m, 1e-300))) / (24.0*np.pi**2)

        # fP(mu,m)
        if mu <= m:
            fP = 0.0
        elif m == 0.0:
            fP = (mu*mu) / (np.pi**2)
        else:
            kF = np.sqrt(max(mu*mu - m*m, 0.0))
            fP = (mu * kF) / (np.pi**2)

        return P0 + (np.pi**2 / 6.0) * (Tem*Tem) * fP

    # ---- exact T=0 pressure ----
    if Tem == 0.0:
        if mu <= m:
            return 0.0
        if m == 0.0:
            return mu**4 / (12.0*np.pi**2)
        kF = _kF(mu, m)
        return ((2.0*kF**3 - 3.0*m*m*kF)*mu + 3.0*m**4*np.log((kF + mu)/max(m, 1e-300))) / (24.0*np.pi**2)

    # ---- adaptive switch band ----
    T_lo, T_hi = Tswitch_free_pressure(mu, m, rtol=rtol, band=1.25)

    # ---- check regime ----
    if T_hi == 0.0:
        use_blend = False
    else:
        use_blend = (T_lo < Tem < T_hi)

    if not use_blend and Tem <= T_lo:
        if mu <= m:
            return 0.0
        if m == 0.0:
            return mu**4 / (12.0*np.pi**2)
        kF = _kF(mu, m)
        return ((2.0*kF**3 - 3.0*m*m*kF)*mu + 3.0*m**4*np.log((kF + mu)/max(m, 1e-300))) / (24.0*np.pi**2)

    # ---- finite-T integral ----
    def _P_int(Tloc):
        def integrand(k):
            Ek = _Ek(k, m)
            a  = (Ek - mu)/Tloc
            b  = (Ek + mu)/Tloc
            return Tloc * (_log1p_exp_neg(a) + _log1p_exp_neg(b)) * (k*k)
        ub = np.inf if (upB is None or not np.isfinite(float(upB))) else float(upB)
        Pint, _ = quad(integrand, 0.0, ub, epsabs=1e-10, epsrel=1e-8, limit=200)
        return Pint / (np.pi**2)

    if not use_blend and Tem >= T_hi:
        return _P_int(Tem)

    # ---- blend inside the band on log(Tem) ----
    if mu <= m:
        P0 = 0.0
    elif m == 0.0:
        P0 = mu**4 / (12.0*np.pi**2)
    else:
        kF = _kF(mu, m)
        P0 = ((2.0*kF**3 - 3.0*m*m*kF)*mu + 3.0*m**4*np.log((kF + mu)/max(m,1e-300))) / (24.0*np.pi**2)

    P_int = _P_int(Tem)
    x = (np.log(Tem) - np.log(T_lo)) / (np.log(T_hi) - np.log(T_lo))
    w = _smoothstep01(x)
    return (1.0 - w)*P0 + w*P_int

def E_f(mu, m, Tem, upB=np.inf, rtol=1e-6, Sommerfeld=False):
    """
    Energy density of a single free fermion flavor (particles + antiparticles).
    The normalization already includes spin degeneracy g=2, but not quark color.
    Uses the same numerical strategy as P_f: T=0 exact branch + adaptive blend.
    """
    mu = float(mu)
    m = float(m)
    Tem = float(Tem)
    if Tem < 0.0:
        raise RuntimeError("Negative Temperature")
    if m == 0.0:
        return 3.0 * _massless_fermion_pressure(mu, Tem)

    def _E0():
        if mu <= m:
            return 0.0
        if m == 0.0:
            return mu**4 / (4.0 * np.pi**2)
        kF = _kF(mu, m)
        return ((2.0 * kF**3 + m*m*kF) * mu - m**4 * np.log((kF + mu) / max(m, 1e-300))) / (8.0 * np.pi**2)

    if Sommerfeld:
        if mu <= m:
            return 0.0
        if m == 0.0:
            corr = 0.5 * mu * mu * Tem * Tem
        else:
            corr = 0.5 * mu * _kF(mu, m) * Tem * Tem
        return _E0() + corr

    if Tem == 0.0:
        return _E0()

    T_lo, T_hi = Tswitch_free_pressure(mu, m, rtol=rtol, band=1.25)
    use_blend = (T_lo < Tem < T_hi) if (T_hi > 0.0) else False

    if not use_blend and Tem <= T_lo:
        return _E0()

    def _E_int(Tloc):
        def integrand(k):
            Ek = _Ek(k, m)
            z = np.clip((Ek - mu) / Tloc, -700, 700)
            zbar = np.clip((Ek + mu) / Tloc, -700, 700)
            f = 1.0 / (1.0 + np.exp(z))
            fbar = 1.0 / (1.0 + np.exp(zbar))
            return k * k * Ek * (f + fbar)
        ub = np.inf if (upB is None or not np.isfinite(float(upB))) else float(upB)
        Eint, _ = quad(integrand, 0.0, ub, epsabs=1e-10, epsrel=1e-8, limit=200)
        return Eint / (np.pi**2)

    if not use_blend and Tem >= T_hi:
        return _E_int(Tem)

    E0 = _E0()
    Eint = _E_int(Tem)
    x = (np.log(Tem) - np.log(T_lo)) / (np.log(T_hi) - np.log(T_lo))
    w = _smoothstep01(x)
    return (1.0 - w) * E0 + w * Eint

# number density for a single free fermion flavor (spin g=2 only; no color factor)
def n_B(mu, m, Tem, upB=5000):
    '''
    Returns number density for a single fermion species.
    The normalization already includes spin degeneracy g=2, but not quark color.
    Uses thermodynamic relation: dP/dmu = n
    '''
    mu = float(mu)
    m = float(m)
    Tem = float(Tem)
    if Tem < 0.0:
        raise RuntimeError("Negative Temperature")
    if m == 0.0:
        return _massless_fermion_density(mu, Tem)

    if Tem > 1e-3:
        def integrand(k):
            Ek = np.sqrt(k**2 + m**2)
            f = 1 / (1 + np.exp(np.clip((Ek - mu)/Tem, -700, 700)))
            f_bar = 1 / (1 + np.exp(np.clip((Ek + mu)/Tem, -700, 700)))
            return (f - f_bar) * k**2

        integral, _ = quad(integrand, 0, upB, epsabs=1e-10, epsrel=1e-8)

        return float( integral / (np.pi**2) )

    else:
        if mu > m:
            k_F = np.sqrt(np.maximum(mu**2 - m**2, 0.0))
            return float( k_F**3 / (3 * np.pi**2) )
        else:
            return float( 0.0 )

def _entropy_f_direct(mu, m, Tem, upB=np.inf):
    """
    Direct phase-space entropy density for a single free fermion flavor
    (particles + antiparticles), including spin degeneracy but excluding color.
    """
    mu = float(mu)
    m = float(m)
    Tem = float(Tem)
    if Tem < 0.0:
        raise RuntimeError("Negative Temperature")
    if Tem == 0.0:
        return 0.0
    if m == 0.0:
        return _massless_fermion_entropy(mu, Tem)

    def integrand(k):
        Ek = _Ek(k, m)
        z = np.clip((Ek - mu) / Tem, -700, 700)
        zbar = np.clip((Ek + mu) / Tem, -700, 700)
        f = 1.0 / (1.0 + np.exp(z))
        fbar = 1.0 / (1.0 + np.exp(zbar))
        return k * k * (
            _binary_entropy_density_term(f) + _binary_entropy_density_term(fbar)
        )

    ub = np.inf if (upB is None or not np.isfinite(float(upB))) else float(upB)
    Sint, _ = quad(integrand, 0.0, ub, epsabs=1e-10, epsrel=1e-8, limit=200)
    return float(Sint / (np.pi**2))

def _quark_mu_triplet(muB, muK):
    """
    Return (mu_u, mu_d, mu_s) from (muB, muK).
    """
    return (
        float(muB / 3.0),
        float(muB / 3.0 + muK / 2.0),
        float(muB / 3.0 - muK / 2.0),
    )

def _quark_pressure_uds(mu_u, mu_d, mu_s, T, ms, upB=5000):
    """
    Physical u,d,s quark pressure.
    P_f is spin-only per flavor, so this wrapper adds the quark color factor Nc=3.
    """
    return float(
        3.0 * (
            P_f(mu_u, m=0.0, Tem=T, upB=upB)
            + P_f(mu_d, m=0.0, Tem=T, upB=upB)
            + P_f(mu_s, m=ms, Tem=T, upB=upB)
        )
    )

def _quark_edens_uds(mu_u, mu_d, mu_s, T, ms, upB=5000):
    """
    Physical u,d,s quark energy density.
    E_f is spin-only per flavor, so this wrapper adds the quark color factor Nc=3.
    """
    return float(
        3.0 * (
            E_f(mu_u, m=0.0, Tem=T, upB=upB)
            + E_f(mu_d, m=0.0, Tem=T, upB=upB)
            + E_f(mu_s, m=ms, Tem=T, upB=upB)
        )
    )

def _quark_density_uds(mu_u, mu_d, mu_s, T, ms, upB=5000):
    """
    Return physical QM baryon density plus physical per-flavor quark densities.

    The base n_B helper is spin-only per flavor, so this wrapper multiplies each
    quark flavor by the color factor Nc=3 before returning the species
    dictionary {'n_u', 'n_d', 'n_s'}.
    With that Convention B bookkeeping, the physical QM baryon density is
        nB_QM = (n_u + n_d + n_s) / 3.
    """
    n_u = 3.0 * n_B(mu_u, 0.0, T, upB=upB)
    n_d = 3.0 * n_B(mu_d, 0.0, T, upB=upB)
    n_s = 3.0 * n_B(mu_s, ms, T, upB=upB)
    nB_qm = float((n_u + n_d + n_s) / 3.0)
    return nB_qm, {"n_u": float(n_u), "n_d": float(n_d), "n_s": float(n_s)}

def _quark_entropy_uds_direct(mu_u, mu_d, mu_s, T, ms, upB=5000):
    """
    Physical u,d,s quark entropy density from the direct phase-space integral.
    """
    return float(
        3.0 * (
            _entropy_f_direct(mu_u, m=0.0, Tem=T, upB=upB)
            + _entropy_f_direct(mu_d, m=0.0, Tem=T, upB=upB)
            + _entropy_f_direct(mu_s, m=ms, Tem=T, upB=upB)
        )
    )


# pressure for SQM under bag model
def PQM(muB, muK, B_one_forth, T, ms=0, upB=5000):
    '''
    Calculates the pressure of strange quark matter (SQM) under bag model

    Parameters:
    - muB         : baryon chemical potential, one third of average quark chemical potential
    - muK         : kaon-ness chemical potential, equals mu_d - mu_s
    - B_one_forth : bag constant for SQM bag model, input is B^(1/4)
    - T           : temperature
    - ms          : strange quark mass, default 0.0
    - upB         : integral upper bound

    Returns:
    - pressure for SQM matter

    Notes:
    - P_f is spin-only per flavor; quark color Nc=3 is added in _quark_pressure_uds.
    '''

    B = B_one_forth**4
    mu_u, mu_d, mu_s = _quark_mu_triplet(muB, muK)
    return _quark_pressure_uds(mu_u, mu_d, mu_s, T, ms, upB=upB) - B

def nB_QM(muB, muK, B_one_forth, T, ms=0, upB=5000, return_species=False):
    """
    Baryon number density of quark matter (u,d,s) in the bag-model setup.
    Note: B_one_forth is accepted for API parity with PQM but does not affect density.

    Returns:
    - physical nB_QM = (n_u + n_d + n_s) / 3
    - if return_species=True, also return physical per-flavor quark densities
      {'n_u', 'n_d', 'n_s'}, each including both spin and color
    """
    mu_u, mu_d, mu_s = _quark_mu_triplet(muB, muK)
    nB_qm, species = _quark_density_uds(mu_u, mu_d, mu_s, T, ms, upB=upB)

    if return_species:
        return nB_qm, species
    return nB_qm

def nK_QM(muB, muK, B_one_forth, T, ms=0, upB=5000, dmu=None):
    """
    Kaon-density-like variable nK = dPQM / dmuK.

    _quark_density_uds returns physical per-flavor quark densities including
    color, so under Convention B
        nK = dPQM/dmuK = (n_d - n_s) / 2.
    """
    mu_u, mu_d, mu_s = _quark_mu_triplet(muB, muK)
    _, species = _quark_density_uds(mu_u, mu_d, mu_s, T, ms, upB=upB)
    n_d = float(species["n_d"])
    n_s = float(species["n_s"])
    return float(0.5 * (n_d - n_s))

def edensQM(muB, muK, B_one_forth, T, ms=0, include_em=False, muQ_init=300, upB=5000):
    """
    Energy density of strange quark matter under the bag model.

    Parameters:
    - include_em=False : match PQM composition (u,d,s quarks + bag constant)
    - include_em=True  : match PQM_em composition (charge-neutral u,d,s,e + thermal gauge term + bag constant)

    Notes:
    - E_f is spin-only per flavor; quark color Nc=3 is added in _quark_edens_uds.
    - The electron contribution remains colorless.
    """
    B = B_one_forth**4

    if include_em:
        mu_u, mu_d, mu_s, mu_e = _solve_quark_mu_em(muB, muK, T, ms, muQ_init=muQ_init, upB=upB)
        return float(
            16*np.pi**2*T**4 / 30
            + E_f(mu_e, m=0.511, Tem=T, upB=upB)
            + _quark_edens_uds(mu_u, mu_d, mu_s, T, ms, upB=upB)
            + B
        )

    mu_u, mu_d, mu_s = _quark_mu_triplet(muB, muK)
    return _quark_edens_uds(mu_u, mu_d, mu_s, T, ms, upB=upB) + B

def entropyQM(muB, muK, B_one_forth, T, ms=0, include_em=False, muQ_init=300, upB=5000, use_thermal=True):
    """
    Entropy density of strange quark matter under the bag model.

    Parameters:
    - include_em=False : match PQM composition (u,d,s quarks + bag constant)
    - include_em=True  : match PQM_em composition (charge-neutral u,d,s,e + thermal gauge term + bag constant)
    - use_thermal=True : use s = (e + P - sum(mu_i n_i)) / T
    - use_thermal=False: use the direct phase-space entropy integral
    """
    T = float(T)
    if T < 0.0:
        raise RuntimeError("Negative Temperature")
    if T == 0.0:
        return 0.0

    B = B_one_forth**4

    if include_em:
        mu_u, mu_d, mu_s, mu_e = _solve_quark_mu_em(muB, muK, T, ms, muQ_init=muQ_init, upB=upB)
        quark_entropy = _quark_entropy_uds_direct(mu_u, mu_d, mu_s, T, ms, upB=upB)
        electron_entropy = _entropy_f_direct(mu_e, m=0.511, Tem=T, upB=upB)

        if not use_thermal:
            return float(quark_entropy + electron_entropy + _gauge_entropy(T))

        quark_pressure = _quark_pressure_uds(mu_u, mu_d, mu_s, T, ms, upB=upB)
        quark_edens = _quark_edens_uds(mu_u, mu_d, mu_s, T, ms, upB=upB)
        _, species = _quark_density_uds(mu_u, mu_d, mu_s, T, ms, upB=upB)
        n_e = float(n_B(mu_e, 0.511, T, upB=upB))

        pressure_total = _gauge_pressure(T) + P_f(mu_e, m=0.511, Tem=T, upB=upB) + quark_pressure - B
        edens_total = _gauge_energy(T) + E_f(mu_e, m=0.511, Tem=T, upB=upB) + quark_edens + B
        chemical_term = (
            mu_u * species["n_u"]
            + mu_d * species["n_d"]
            + mu_s * species["n_s"]
            + mu_e * n_e
        )
        return float((edens_total + pressure_total - chemical_term) / T)

    mu_u, mu_d, mu_s = _quark_mu_triplet(muB, muK)
    quark_entropy = _quark_entropy_uds_direct(mu_u, mu_d, mu_s, T, ms, upB=upB)
    if not use_thermal:
        return quark_entropy

    quark_pressure = _quark_pressure_uds(mu_u, mu_d, mu_s, T, ms, upB=upB)
    quark_edens = _quark_edens_uds(mu_u, mu_d, mu_s, T, ms, upB=upB)
    _, species = _quark_density_uds(mu_u, mu_d, mu_s, T, ms, upB=upB)
    chemical_term = (
        mu_u * species["n_u"]
        + mu_d * species["n_d"]
        + mu_s * species["n_s"]
    )
    pressure_total = quark_pressure - B
    edens_total = quark_edens + B
    return float((edens_total + pressure_total - chemical_term) / T)

def hQM(muB, muK, B_one_forth, Temp):
    return PQM(muB, muK, B_one_forth, T=Temp, ms=0, upB=5000) + edensQM(muB, muK, B_one_forth, T=Temp, ms=0, include_em=False, muQ_init=300, upB=5000)

def Pi_QM(mu_B, mu_K, B_one_forth, Temp, j_B):
    nB = nB_QM(mu_B, mu_K, B_one_forth, Temp)
    if nB <= 0:
        return np.nan
    uQ = j_B / nB
    return hQM(mu_B, mu_K, B_one_forth, Temp) * uQ * uQ + PQM(mu_B, mu_K, B_one_forth, Temp)


def _solve_quark_mu_em(muB, muK, T, ms, muQ_init=300, upB=5000):
    """
    Solve charge-neutral quark chemical potentials for PQM_em-like compositions.
    Returns (mu_u, mu_d, mu_s, mu_e).
    """
    def equation(mu_u_in):
        mu_u_val = float(np.atleast_1d(mu_u_in)[0])
        mu_d = (muB + muK - mu_u_val) / 2.0
        mu_s = (muB - muK - mu_u_val) / 2.0
        mu_e = (muB - 3.0 * mu_u_val) / 2.0
        n_u = 3.0 * n_B(mu_u_val, 0, T, upB=upB)
        n_d = 3.0 * n_B(mu_d, 0, T, upB=upB)
        n_s = 3.0 * n_B(mu_s, ms, T, upB=upB)
        n_e = n_B(mu_e, 0, T, upB=upB)
        return float(
            # n_u, n_d, n_s are full physical quark densities including color,
            # while n_e is the physical electron density.
            3.0 * n_e + n_d + n_s - 2.0 * n_u
        )

    sol = root(equation, muQ_init, method='hybr', options={'maxfev': 60000})
    if not sol.success:
        print("Root finding failed:", sol.message)
        raise RuntimeError("PQM failed to converge")

    mu_u = float(sol.x[0])
    mu_d = (muB + muK - mu_u) / 2.0
    mu_s = (muB - muK - mu_u) / 2.0
    mu_e = (muB - 3.0 * mu_u) / 2.0
    return mu_u, mu_d, mu_s, mu_e

# PQM including electromagnetism 
def PQM_em(muB, muK, B_one_forth, T, ms, muQ_init=300, upB=5000):
    '''
    Calculates the pressure of strange quark matter (SQM) under bag model including electrons 

    Parameters:
    - muB         : baryon chemical potential, one third of average quark chemical potential
    - muK         : kaon-ness chemical potential, equals mu_d - mu_s
    - B_one_forth : bag constant for SQM bag model, input is B^(1/4)
    - T           : temperature
    - ms          : strange quark mass
    - upB         : integral upper bound

    Returns:
    - pressure for SQM matter

    Notes:
    - P_f is spin-only per flavor; quark color Nc=3 is added in _quark_pressure_uds.
    - The electron contribution remains colorless.
    '''
    B = B_one_forth**4
    mu_u, mu_d, mu_s, mu_e = _solve_quark_mu_em(muB, muK, T, ms, muQ_init=muQ_init, upB=upB)
    return float(
        16*np.pi**2*T**4 / 90
        + P_f(mu_e, m=0.511, Tem=T, upB=upB)
        + _quark_pressure_uds(mu_u, mu_d, mu_s, T, ms, upB=upB)
        - B
    )

def nQM_em(muB, muK, B_one_forth, T, ms, muQ_init=300, upB=5000, return_species=False):
    """
    Baryon number density of charge-neutral quark matter (u,d,s,e composition).
    Note: B_one_forth is accepted for API parity with PQM_em but does not affect density.

    Returns:
    - physical nB_QM = (n_u + n_d + n_s) / 3
    - if return_species=True, return physical quark species densities including
      color together with the physical electron density n_e
    """
    mu_u, mu_d, mu_s, mu_e = _solve_quark_mu_em(muB, muK, T, ms, muQ_init=muQ_init, upB=upB)
    nB_qm, species = _quark_density_uds(mu_u, mu_d, mu_s, T, ms, upB=upB)
    species["n_e"] = float(n_B(mu_e, 0.0, T, upB=upB))

    if return_species:
        return nB_qm, species
    return nB_qm


# extract coordinates of a contour
def extract_contour_coords_num(X, Y, Z, level):
    '''
    extract the coordinates of a contour plot at certain level

    Parameters:
    - X: meshgrid coordinates for x axis
    - Y: meshgrid coordinates for y axis
    - Z: value of the function for the meshgrid Z(X,Y)
    - level: at which level you wish to extract the contour line

    Returns:
    - X_coor: x coordinates of the targeted contour line
    - Y_coor: y coordinates of the targeted contour line
    '''

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    contour_obj = ax.contour(X, Y, Z, levels=[level])

    paths = []
    for coll in contour_obj.collections:
        for c in coll.get_paths():
            paths.append(c.vertices)
    plt.close(fig)

    if not paths:
        raise RuntimeError(f"No contours found at level {level}")

    X_coor = paths[0][:, 0]
    Y_coor = paths[0][:, 1]
    return X_coor, Y_coor

def extract_contour_coords_ana(func, x_range, y_list, level, bracket=None, method='brentq'):
    '''
    Extract coordinates (x, y) satisfying func(x, y) = level using root finding.

    Parameters:
    - func   : function of (x, y) returning a scalar (e.g., func = lambda x, y: f(x, y))
    - x_range: tuple (x_min, x_max), search interval for x at each y
    - y_list : 1D array of y values to scan over
    - level  : contour level (constant value)
    - bracket: optional custom bracket (x_min, x_max), overrides x_range
    - method : root-finding method, default is 'brentq'

    Returns:
    - x_coords: array of x values where func(x, y) = level
    - y_coords: array of y values (corresponding to input y_list)
    '''

    x_coords = []
    y_coords = []

    x_min, x_max = bracket if bracket else x_range

    for y in y_list:
        f_root = lambda x: func(x, y) - level
        try:
            sol = root_scalar(f_root, bracket=[x_min, x_max], method=method)
            if sol.converged:
                x_coords.append(sol.root)
                y_coords.append(y)
        except ValueError:
            continue  # skip if no root in bracket

    return np.array(x_coords), np.array(y_coords)


def _pi_target_state(T, B_one_forth, jB, Pi_over_crit):
    """
    Convert the dimensionless Pi_over_crit input into the absolute Pi target and
    solve the corresponding mu_B roots on the mu_K = 0 reference surface.
    """
    muB_crit = float(
        fsolve(
            lambda x: Pi_NM(x, T, jB) - Pi_QM(x, 0.0, B_one_forth, T, jB),
            1050,
        )[0]
    )
    Pi_crit = Pi_QM(muB_crit, 0.0, B_one_forth, T, jB)
    Pi_target = Pi_over_crit * Pi_crit
    muB_N = float(fsolve(lambda x: Pi_NM(x, T, jB) - Pi_target, 1050)[0])
    muB_Q = float(fsolve(lambda x: Pi_QM(x, 0.0, B_one_forth, T, jB) - Pi_target, 1050)[0])
    return muB_crit, Pi_crit, Pi_target, muB_N, muB_Q


def _find_Qstar_on_target(T, B_one_forth, lambda_val, jB, Pi_target, muB_N, muB_Q, return_aux=False):
    """
    Solve the Q* system once the absolute Pi target and the associated mu_B
    roots are already known.
    """
    alpha_s = 0.3
    g_s = np.sqrt(4.0 * np.pi * alpha_s)
    h_const = 1.81317
    qd_coeff = np.sqrt(3.0 * g_s**2 / (2.0 * np.pi**2))
    D_prefactor = 24.0 * alpha_s**2 / np.pi
    T53 = T**(5.0 / 3.0)
    T2 = T * T

    epsilon = np.sqrt(np.finfo(float).eps)
    PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_one_forth, T, 0)
    grad_Q = approx_fprime(np.array([muB_Q, 0.0]), PQM_wrap, epsilon)
    nB_Q = grad_Q[0]
    nK_Q = grad_Q[1]

    PNM_wrap = lambda x: PNM(x, T)
    dPNM_dmuB = Derivative(PNM_wrap, n=1)
    nB_N_diff = dPNM_dmuB(muB_N)
    aN = (nB_N_diff - nK_Q) / nB_Q
    nB_N_surface = nB_NM(muB_N, T)
    uN = jB / nB_N_surface

    def system(vec):
        muBstar, muKstar = map(float, vec)
        muQ = muBstar / 3.0

        if muBstar <= 0.0 or muKstar < 0.0 or muQ <= 0.0:
            return np.array([1e30, 1e30], dtype=float)

        grad_Qstar = approx_fprime(np.array([muBstar, muKstar]), PQM_wrap, epsilon)
        nK_Qstar = grad_Qstar[1]
        aQstar = (nK_Qstar - nK_Q) / nB_Q

        qD = qd_coeff * muQ
        eta = (9.0 * np.pi**2 * T2) / (muQ * muQ)
        gamma = 1.0 / (1.98e12 * ((300.0 / muQ) ** 5))
        part1 = h_const * T53 / (qD ** (2.0 / 3.0))
        part2 = np.pi**3 * T2 / (12.0 * qD)
        D = 1.0 / (D_prefactor * (part1 + part2))

        eq1 = Pi_QM(muBstar, muKstar, B_one_forth, T, jB) - Pi_target
        uQ = jB / nB_QM(muBstar, muKstar, B_one_forth, T)
        delta_u = uQ * aQstar - uN * aN

        eq2 = (
            0.5 * delta_u**2
            - lambda_val * uQ * aQstar * delta_u
            - 0.25 * D * gamma * (aQstar**4 + 2.0 * eta * aQstar**2)
        )

        return np.array([eq1, eq2], dtype=float)

    initial_guess = [1020, 40]
    res = least_squares(system, initial_guess, bounds=([0, 0], [np.inf, np.inf]))

    if not res.success:
        raise RuntimeError(f"Root finding failed: {res.message}")

    muB_star = float(res.x[0])
    muK_star = float(res.x[1])

    if muK_star < 0:
        raise RuntimeError("muK_star < 0")

    if return_aux:
        return muB_star, muK_star, nB_N_surface

    return muB_star, muK_star


def _solve_muB_inf_at_muK0_from_nB(nB_inf, B_one_forth, T, ms=0, upB=5000):
    """
    Solve for the equilibrated QM endpoint chemical potential muB_inf at muK=0
    from the target baryon density nB_inf.
    """
    if nB_inf <= 0.0:
        raise RuntimeError("nB_inf must be positive when solving for muB_inf")

    def density_residual(muB):
        return float(nB_QM(muB, 0.0, B_one_forth, T, ms=ms, upB=upB) - nB_inf)

    mu_lo = 1.0e-8
    mu_hi = 1500.0
    f_lo = density_residual(mu_lo)
    f_hi = density_residual(mu_hi)

    for _ in range(40):
        if abs(f_lo) == 0.0:
            return float(mu_lo)
        if abs(f_hi) == 0.0:
            return float(mu_hi)
        if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi < 0.0:
            break
        if not np.isfinite(f_hi) or f_hi < 0.0:
            mu_hi *= 1.5
            f_hi = density_residual(mu_hi)
        else:
            mu_lo *= 0.5
            if mu_lo <= 0.0:
                mu_lo = np.nextafter(0.0, 1.0)
            f_lo = density_residual(mu_lo)
    else:
        raise RuntimeError(
            f"Failed to bracket muB_inf for nB_inf={nB_inf:.6g} at T={T:.6g}"
        )

    sol = root_scalar(density_residual, bracket=[mu_lo, mu_hi], method="brentq")
    if not sol.converged:
        raise RuntimeError("Root solve for muB_inf failed to converge")

    return float(sol.root)


def uN(T, nB_N, Delta_n, B_one_forth, param=para.paraQMCRMF3, ms=0, upB=5000, return_more=False):
    """
    Compute the phase-boundary flux data from endpoint states.

    Parameters
    ----------
    T : float
        Temperature.
    nB_N : float
        Upstream nuclear baryon density n_B(0^-).
    Delta_n : float
        Density jump defined by n_B(infty) = nB_N + Delta_n.
    B_one_forth : float
        Bag constant parameter B^(1/4) for the quark EOS.
    param : sequence, optional
        RMF parameter set for the nuclear endpoint.
    ms : float, optional
        Strange-quark mass used on the QM side.
    upB : float, optional
        Upper integration bound passed to the quark EOS helpers.
    return_more : bool, optional
        If True, return a dict with endpoint thermodynamic details.

    Notes
    -----
    - The far-right quark endpoint is assumed fully equilibrated: muK = 0.
    - The returned values are Pi, jB, uN, where uN is in natural units.
    """
    if nB_N <= 0.0:
        raise RuntimeError("nB_N must be positive")

    nB_inf = nB_N + Delta_n
    if nB_inf <= 0.0:
        raise RuntimeError("nB_inf = nB_N + Delta_n must be positive")

    # Upstream nuclear endpoint from (T, nB_N).
    P_N = float(PNM_n(nB_N, T, param=param, NM_type="PNM"))
    h_N = float(P_N + edensNM_n(nB_N, T, param=param))

    # Fully equilibrated QM endpoint at muK = 0 and nB_inf.
    muB_inf = _solve_muB_inf_at_muK0_from_nB(nB_inf, B_one_forth, T, ms=ms, upB=upB)
    P_inf = float(PQM(muB_inf, 0.0, B_one_forth, T, ms=ms, upB=upB))
    h_inf = float(
        P_inf
        + edensQM(muB_inf, 0.0, B_one_forth, T, ms=ms, include_em=False, upB=upB)
    )

    # Momentum-flux matching fixes jB from the two endpoint states.
    term_N = h_N / (nB_N * nB_N)
    term_inf = h_inf / (nB_inf * nB_inf)
    denom = term_N - term_inf
    denom_scale = max(abs(term_N), abs(term_inf), 1.0)
    if abs(denom) <= 1.0e-12 * denom_scale:
        raise RuntimeError("Endpoint momentum-flux denominator is too close to zero")

    jB_sq = (P_inf - P_N) / denom
    if (not np.isfinite(jB_sq)) or jB_sq <= 0.0:
        raise RuntimeError(f"Endpoint matching gives non-physical jB^2={jB_sq}")

    jB = float(np.sqrt(jB_sq))
    uN = float(jB / nB_N)
    Pi = float(h_N * uN * uN + P_N)

    # Numerical consistency check on the equilibrated QM endpoint.
    uQ_inf = float(jB / nB_inf)
    Pi_inf = float(h_inf * uQ_inf * uQ_inf + P_inf)
    Pi_scale = max(abs(Pi), abs(Pi_inf), 1.0)
    if not np.isclose(Pi, Pi_inf, rtol=1.0e-8, atol=1.0e-10 * Pi_scale):
        raise RuntimeError(
            f"Endpoint momentum-flux mismatch: Pi_N={Pi:.12g}, Pi_inf={Pi_inf:.12g}"
        )

    if return_more:
        return {
            "Pi": Pi,
            "jB": jB,
            "uN": uN,
            "nB_inf": float(nB_inf),
            "P_N": P_N,
            "h_N": h_N,
            "P_inf": P_inf,
            "h_inf": h_inf,
            "muB_inf": muB_inf,
            "uQ_inf": uQ_inf,
        }

    return Pi, jB, uN


def _Pi_QM_state(muB, muK, B_one_forth, T, jB, ms=0.0, upB=5000):
    """
    Momentum flux Pi = h*u^2 + P for a quark state at fixed (muB, muK).
    """
    nB = nB_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB)
    if nB <= 0.0:
        return np.nan
    P = PQM(muB, muK, B_one_forth, T, ms=ms, upB=upB)
    h = P + edensQM(muB, muK, B_one_forth, T, ms=ms, include_em=False, upB=upB)
    u = jB / nB
    return float(h * u * u + P)


def _quark_thermo_state(muB, muK, B_one_forth, T, jB, ms=0.0, upB=5000):
    """
    Build a fully ms-consistent local quark thermodynamic state.

    The entropy density is reconstructed from the thermal relation
        s = (e + P - muB * nB - muK * nK) / T
    using the existing EOS helpers.
    """
    T = float(T)
    if (not np.isfinite(T)) or T <= 0.0:
        raise RuntimeError("Quark thermodynamic state requires T > 0")

    muB = float(muB)
    muK = float(muK)
    nB = float(nB_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
    if (not np.isfinite(nB)) or nB <= 0.0:
        raise RuntimeError("Quark thermodynamic state has non-positive baryon density")

    nK = float(nK_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
    P = float(PQM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
    e = float(edensQM(muB, muK, B_one_forth, T, ms=ms, include_em=False, upB=upB))
    if (not np.isfinite(nK)) or (not np.isfinite(P)) or (not np.isfinite(e)):
        raise RuntimeError("Quark thermodynamic state returned non-finite EOS quantities")

    u = float(jB / nB)
    h = float(P + e)
    s_density = float((e + P - muB * nB - muK * nK) / T)
    if not np.isfinite(s_density):
        raise RuntimeError("Quark thermodynamic state returned a non-finite entropy density")

    Pi = float(h * u * u + P)
    entropy_flux = float(s_density * u)
    if (not np.isfinite(Pi)) or (not np.isfinite(entropy_flux)):
        raise RuntimeError("Quark thermodynamic state returned non-finite fluxes")

    return {
        "muB": muB,
        "muK": muK,
        "T": T,
        "nB": nB,
        "nK": nK,
        "P": P,
        "e": e,
        "h": h,
        "u": u,
        "s": s_density,
        "w": entropy_flux,
        "Pi": Pi,
    }


def _solve_muB_Q_at_muK0_for_given_Pi(Pi, jB, B_one_forth, T, ms=0.0, upB=5000, stats=None):
    """
    Solve for the equilibrated QM endpoint muB_Q at muK=0 for a given Pi.
    """
    if stats is not None:
        stats["q_root_calls"] = stats.get("q_root_calls", 0) + 1
    muB_Q = float(fsolve(lambda x: Pi_QM(x, 0.0, B_one_forth, T, jB) - Pi, 1100.0)[0])

    Pi_residual = float(_Pi_QM_state(muB_Q, 0.0, B_one_forth, T, jB, ms=ms, upB=upB) - Pi)
    nB = nB_QM(muB_Q, 0.0, B_one_forth, T, ms=ms, upB=upB)
    if (not np.isfinite(muB_Q)) or (not np.isfinite(Pi_residual)) or abs(Pi_residual) > 1.0e-8 * max(abs(Pi), 1.0) or nB <= 0.0:
        raise RuntimeError("Solved muB_Q lies on a non-physical density branch")

    return muB_Q


def _solve_muB_Q_at_muK0_for_given_Pi_ms(
    Pi,
    jB,
    B_one_forth,
    T,
    ms=0.0,
    upB=5000,
    stats=None,
    stats_key="q_root_calls",
    initial_guess=None,
):
    """
    Solve for the equilibrated QM endpoint muB_Q at muK=0 using the ms-aware
    quark momentum-flux helper throughout.
    """
    if stats is not None:
        stats[stats_key] = stats.get(stats_key, 0) + 1

    def equation(muB_in):
        muB = float(np.atleast_1d(muB_in)[0])
        return float(_Pi_QM_state(muB, 0.0, B_one_forth, T, jB, ms=ms, upB=upB) - Pi)

    guesses = []
    if initial_guess is not None:
        try:
            guesses.append(float(initial_guess))
        except Exception:
            pass
    guesses.extend([1100.0, 900.0, 1300.0, 700.0, 1500.0])

    tol = 1.0e-8 * max(abs(Pi), 1.0)
    best_muB = None
    best_metric = np.inf
    last_error = "fsolve did not produce a physical root"

    for muB_guess in guesses:
        if (not np.isfinite(muB_guess)) or muB_guess <= 0.0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                muB_arr, info, ier, mesg = fsolve(equation, muB_guess, full_output=True)
            muB_Q = float(np.atleast_1d(muB_arr)[0])
            Pi_residual = float(_Pi_QM_state(muB_Q, 0.0, B_one_forth, T, jB, ms=ms, upB=upB) - Pi)
            nB = float(nB_QM(muB_Q, 0.0, B_one_forth, T, ms=ms, upB=upB))
            if (not np.isfinite(muB_Q)) or (not np.isfinite(Pi_residual)) or (not np.isfinite(nB)) or nB <= 0.0:
                last_error = "Solved muB_Q lies on a non-physical density branch"
                continue
            metric = abs(Pi_residual)
            if metric < best_metric:
                best_metric = metric
                best_muB = muB_Q
            if ier == 1 and metric <= tol:
                return muB_Q
            last_error = str(mesg)
        except Exception as exc:
            last_error = str(exc)
            continue

    if best_muB is not None and np.isfinite(best_metric) and best_metric <= tol:
        return float(best_muB)
    raise RuntimeError(last_error)


def _branch_muK_seed(a_like):
    """
    Positive muK seed used by the steady-front quark-state solves.
    """
    return float(max(1.0, 20.0, 250.0 * abs(float(a_like))))


def _quark_state_residual(muB, muK, a_target, Pi, jB, nB_Q, nK_Q, B_one_forth, T, ms=0.0, upB=5000):
    return np.array(
        [
            _Pi_QM_state(muB, muK, B_one_forth, T, jB, ms=ms, upB=upB) - Pi,
            (nK_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB) - nK_Q) / nB_Q - a_target,
        ],
        dtype=float,
    )


def _quark_state_residual_ok(residual, Pi, a_target):
    if not np.all(np.isfinite(residual)):
        return False
    pi_tol = 1.0e-8 * max(abs(Pi), 1.0)
    a_tol = 1.0e-8 * max(abs(a_target), 1.0)
    return bool(abs(float(residual[0])) <= pi_tol and abs(float(residual[1])) <= a_tol)


def _solve_quark_state_once_from_guess(a_target, Pi, jB, nB_Q, nK_Q, B_one_forth, T, ms=0.0, upB=5000, initial_guess=None, stats=None, stats_key="quark_state_root_calls"):
    """
    Try one local quark-state root solve from a single continuation guess.
    This is the fast path used during IVP integration.
    """
    if initial_guess is None:
        raise RuntimeError("initial_guess is required for single-guess quark-state solve")
    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving for a quark state")

    guess = np.asarray(initial_guess, dtype=float)
    if guess.shape[0] != 2 or not np.all(np.isfinite(guess)):
        raise RuntimeError("initial_guess must contain finite (muB, muK)")

    def equations(vec):
        muB, muK = map(float, vec)
        return _quark_state_residual(
            muB,
            muK,
            a_target,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            T,
            ms=ms,
            upB=upB,
        )

    if stats is not None:
        stats[stats_key] = stats.get(stats_key, 0) + 1
    sol = root(equations, guess, method="hybr", options={"maxfev": 120, "xtol": 1.0e-10})
    if not (sol.success and np.all(np.isfinite(sol.x))):
        raise RuntimeError(f"single-guess quark-state solve failed: {sol.message}")

    muB = float(sol.x[0])
    muK = float(sol.x[1])
    if muK < -1.0e-8:
        raise RuntimeError("single-guess quark-state solve returned negative muK")
    if muK < 0.0:
        muK = 0.0

    residual = _quark_state_residual(
        muB,
        muK,
        a_target,
        Pi,
        jB,
        nB_Q,
        nK_Q,
        B_one_forth,
        T,
        ms=ms,
        upB=upB,
    )
    if not _quark_state_residual_ok(residual, Pi, a_target):
        raise RuntimeError(
            "single-guess quark-state solve returned an unacceptable residual "
            f"({residual[0]:.3e}, {residual[1]:.3e})"
        )

    nB = float(nB_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
    if nB <= 0.0:
        raise RuntimeError("single-guess quark-state solve returned non-positive density")
    return muB, muK


def _solve_interface_Qstar_from_aQstar_and_Pi(aQstar, Pi, jB, nB_Q, nK_Q, B_one_forth, T, ms=0.0, upB=5000, initial_guess=None, stats=None, stats_key="qstar_root_calls"):
    """
    Solve for the interface state Qstar from Pi and aQstar.
    """
    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving for Qstar")

    guesses = []
    muK_seed = _branch_muK_seed(aQstar)
    muK_seed_strong = float(max(muK_seed, 400.0 * abs(float(aQstar))))
    if initial_guess is not None:
        guess0 = np.asarray(initial_guess, dtype=float)
        guesses.append(guess0)
        muB_seed = float(guess0[0])
    else:
        muB_seed = 1200.0

    guesses.append(np.array([muB_seed, muK_seed], dtype=float))
    guesses.append(np.array([1200.0, muK_seed], dtype=float))
    guesses.append(np.array([1500.0, max(muK_seed, 100.0 * abs(float(aQstar)))], dtype=float))
    guesses.append(np.array([muB_seed, muK_seed_strong], dtype=float))
    guesses.append(np.array([1500.0, muK_seed_strong], dtype=float))

    def equations(vec):
        muB, muK = map(float, vec)
        return _quark_state_residual(
            muB,
            muK,
            aQstar,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            T,
            ms=ms,
            upB=upB,
        )

    best_message = "Qstar solve did not converge"
    candidates = []
    candidate_tol = 1.0e-8
    nonneg_tol = 1.0e-8
    for guess in guesses:
        if stats is not None:
            stats[stats_key] = stats.get(stats_key, 0) + 1
        sol = root(equations, guess, method="hybr", options={"maxfev": 6000, "xtol": 1.0e-10})
        if sol.success and np.all(np.isfinite(sol.x)):
            muB_Qstar = float(sol.x[0])
            muK_Qstar = float(sol.x[1])
            residual = _quark_state_residual(
                muB_Qstar,
                max(muK_Qstar, 0.0),
                aQstar,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                T,
                ms=ms,
                upB=upB,
            )
            nB_Qstar = nB_QM(muB_Qstar, muK_Qstar, B_one_forth, T, ms=ms, upB=upB)
            if nB_Qstar > 0.0 and muK_Qstar >= -nonneg_tol and _quark_state_residual_ok(residual, Pi, aQstar):
                if muK_Qstar < 0.0:
                    muK_Qstar = 0.0
                is_new = True
                for cand in candidates:
                    if (
                        abs(muB_Qstar - cand["muB"]) <= candidate_tol * max(1.0, abs(cand["muB"]))
                        and abs(muK_Qstar - cand["muK"]) <= candidate_tol * max(1.0, abs(cand["muK"]), 1.0)
                    ):
                        is_new = False
                        break
                if is_new:
                    candidates.append({"muB": muB_Qstar, "muK": muK_Qstar})
        best_message = sol.message

    if candidates:
        if initial_guess is not None:
            muB_ref = float(initial_guess[0])
            muK_ref = max(0.0, float(initial_guess[1]))
        else:
            muB_ref = muB_seed
            muK_ref = 0.0

        muK_pref = max(muK_seed, muK_ref)
        candidates.sort(
            key=lambda cand: (
                abs(cand["muK"] - muK_pref),
                abs(cand["muB"] - muB_ref),
                -cand["muK"],
            )
        )
        return candidates[0]["muB"], candidates[0]["muK"]

    raise RuntimeError(f"Qstar state solve failed: {best_message}")


def _solve_local_quark_state_from_a_and_Pi(a, Pi, jB, nB_Q, nK_Q, B_one_forth, T, ms=0.0, upB=5000, initial_guess=None, stats=None):
    """
    Solve the local quark state (muB, muK, nB, u) at fixed a, Pi, jB.
    """
    if stats is not None:
        stats["local_state_calls"] = stats.get("local_state_calls", 0) + 1

    if initial_guess is not None:
        try:
            muB, muK = _solve_quark_state_once_from_guess(
                a,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                T,
                ms=ms,
                upB=upB,
                initial_guess=initial_guess,
                stats=stats,
                stats_key="local_root_calls",
            )
            nB = float(nB_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
            if nB <= 0.0:
                raise RuntimeError("Local quark state has non-positive density")
            u = float(jB / nB)
            return muB, muK, nB, u
        except Exception:
            # Fall back to the broader branch-aware candidate search only when
            # nearest-neighbor continuation fails.
            if stats is not None:
                stats["local_fast_failures"] = stats.get("local_fast_failures", 0) + 1

    muB, muK = _solve_interface_Qstar_from_aQstar_and_Pi(
        a,
        Pi,
        jB,
        nB_Q,
        nK_Q,
        B_one_forth,
        T,
        ms=ms,
        upB=upB,
        initial_guess=initial_guess,
        stats=stats,
        stats_key="local_root_calls",
    )
    nB = float(nB_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB))
    if nB <= 0.0:
        raise RuntimeError("Local quark state has non-positive density")
    u = float(jB / nB)
    return muB, muK, nB, u


def _microphysics_at_Qstar(muB_Qstar, T):
    """
    Frozen diffusion/reaction coefficients evaluated at Qstar.
    """
    return _microphysics_from_quark_state(muB_Qstar, T)


def _microphysics_from_quark_state(muB, T):
    """
    Compute diffusion/reaction coefficients from a local quark thermodynamic
    state. In the current convention the coefficients depend on muQ = muB / 3
    and the local temperature.
    """
    T = float(T)
    if (not np.isfinite(T)) or T <= 0.0:
        raise RuntimeError("Local microphysics requires T > 0")

    muQ = float(muB) / 3.0
    if (not np.isfinite(muQ)) or muQ <= 0.0:
        raise RuntimeError("Local microphysics requires muQ > 0")

    alpha_s = 0.3
    g_s = np.sqrt(4.0 * np.pi * alpha_s)
    qD = np.sqrt(3.0 * g_s**2 * muQ**2 / (2.0 * np.pi**2))
    h_const = 1.81317
    part1 = h_const * T**(5.0 / 3.0) / qD**(2.0 / 3.0)
    part2 = np.pi**3 * T**2 / (12.0 * qD)
    D = 1.0 / (24.0 * alpha_s**2 / np.pi * (part1 + part2))
    eta = 9.0 * np.pi**2 * T**2 / muQ**2
    tau = 1.98e12 * (300.0 / muQ) ** 5
    gamma = 1.0 / tau

    if (
        (not np.isfinite(qD))
        or (not np.isfinite(D))
        or (not np.isfinite(eta))
        or (not np.isfinite(tau))
        or (not np.isfinite(gamma))
        or D <= 0.0
        or eta < 0.0
        or tau <= 0.0
        or gamma <= 0.0
    ):
        raise RuntimeError("Local microphysics returned non-physical coefficients")

    return {
        "alpha_s": float(alpha_s),
        "muQ": float(muQ),
        "qD": float(qD),
        "D": float(D),
        "eta": float(eta),
        "gamma": float(gamma),
        "tau": float(tau),
    }


def _quark_state_entropy_residual(muB, muK, logT, a_target, w_target, Pi, jB, nB_Q, nK_Q, B_one_forth, ms=0.0, upB=5000):
    """
    Unscaled residual for the entropy-enabled local quark-state closure.
    """
    if w_target <= 0.0:
        raise RuntimeError("Entropy-enabled closure requires w > 0")
    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving an entropy-enabled quark state")

    logT = float(logT)
    if (not np.isfinite(logT)) or abs(logT) > 700.0:
        return np.array([1.0e12, 1.0e12, 1.0e12], dtype=float)

    T = float(np.exp(logT))
    try:
        thermo = _quark_thermo_state(muB, muK, B_one_forth, T, jB, ms=ms, upB=upB)
    except Exception:
        return np.array([1.0e12, 1.0e12, 1.0e12], dtype=float)

    return np.array(
        [
            thermo["Pi"] - Pi,
            (thermo["nK"] - nK_Q) / nB_Q - a_target,
            thermo["w"] - w_target,
        ],
        dtype=float,
    )


def _quark_state_entropy_residual_ok(residual, Pi, a_target, w_target):
    """
    Accept/reject an entropy-enabled local quark-state solve using relative
    tolerances matched to the three closure equations.
    """
    if not np.all(np.isfinite(residual)):
        return False
    pi_tol = 1.0e-8 * max(abs(Pi), 1.0)
    a_tol = 1.0e-8 * max(abs(a_target), 1.0)
    w_tol = 1.0e-8 * max(abs(w_target), 1.0)
    return bool(
        abs(float(residual[0])) <= pi_tol
        and abs(float(residual[1])) <= a_tol
        and abs(float(residual[2])) <= w_tol
    )


def _solve_quark_entropy_state_once_from_guess(a_target, w_target, Pi, jB, nB_Q, nK_Q, B_one_forth, ms=0.0, upB=5000, initial_guess=None, stats=None, stats_key="quark_state_entropy_root_calls"):
    """
    Try one entropy-enabled local quark-state root solve from a single
    continuation guess. The nonlinear solve is carried out in (muB, muK, logT).
    """
    if initial_guess is None:
        raise RuntimeError("initial_guess is required for single-guess entropy-enabled quark-state solve")
    if w_target <= 0.0:
        raise RuntimeError("Entropy-enabled closure requires w > 0")
    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving an entropy-enabled quark state")

    guess = np.asarray(initial_guess, dtype=float)
    if guess.shape[0] != 3 or not np.all(np.isfinite(guess)):
        raise RuntimeError("initial_guess must contain finite (muB, muK, T)")
    if guess[2] <= 0.0:
        raise RuntimeError("initial_guess temperature must be positive")

    pi_scale = max(abs(Pi), 1.0)
    w_scale = max(abs(w_target), 1.0)

    def equations(vec):
        residual = _quark_state_entropy_residual(
            float(vec[0]),
            float(vec[1]),
            float(vec[2]),
            a_target,
            w_target,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            ms=ms,
            upB=upB,
        )
        return np.array([residual[0] / pi_scale, residual[1], residual[2] / w_scale], dtype=float)

    if stats is not None:
        stats[stats_key] = stats.get(stats_key, 0) + 1
    sol = root(
        equations,
        np.array([float(guess[0]), float(guess[1]), float(np.log(float(guess[2])))], dtype=float),
        method="hybr",
        options={"maxfev": 180, "xtol": 1.0e-10},
    )
    if not (sol.success and np.all(np.isfinite(sol.x))):
        raise RuntimeError(f"single-guess entropy-enabled quark-state solve failed: {sol.message}")

    muB = float(sol.x[0])
    muK = float(sol.x[1])
    logT = float(sol.x[2])
    if muK < -1.0e-8:
        raise RuntimeError("single-guess entropy-enabled quark-state solve returned negative muK")
    if (not np.isfinite(logT)) or abs(logT) > 700.0:
        raise RuntimeError("single-guess entropy-enabled quark-state solve returned an invalid logT")

    if muK < 0.0:
        muK = 0.0
    T_loc = float(np.exp(logT))
    residual = _quark_state_entropy_residual(
        muB,
        muK,
        np.log(T_loc),
        a_target,
        w_target,
        Pi,
        jB,
        nB_Q,
        nK_Q,
        B_one_forth,
        ms=ms,
        upB=upB,
    )
    if not _quark_state_entropy_residual_ok(residual, Pi, a_target, w_target):
        raise RuntimeError(
            "single-guess entropy-enabled quark-state solve returned an unacceptable residual "
            f"({residual[0]:.3e}, {residual[1]:.3e}, {residual[2]:.3e})"
        )

    thermo = _quark_thermo_state(muB, muK, B_one_forth, T_loc, jB, ms=ms, upB=upB)
    if thermo["s"] <= 0.0:
        raise RuntimeError("single-guess entropy-enabled quark-state solve returned non-positive entropy density")
    if thermo["w"] <= 0.0:
        raise RuntimeError("single-guess entropy-enabled quark-state solve returned non-positive entropy flux")
    return thermo


def _solve_local_quark_state_from_a_w_and_Pi(a, w, Pi, jB, nB_Q, nK_Q, B_one_forth, ms=0.0, upB=5000, initial_guess=None, T_ref=None, stats=None):
    """
    Solve the local quark state (muB, muK, T, nB, u, s) at fixed (a, w, Pi, jB)
    using the entropy-enabled closure equations.
    """
    if stats is not None:
        stats["local_state_calls"] = stats.get("local_state_calls", 0) + 1
    if w <= 0.0:
        raise RuntimeError("Local entropy-enabled closure requires w > 0")

    if initial_guess is not None:
        try:
            return _solve_quark_entropy_state_once_from_guess(
                a,
                w,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                ms=ms,
                upB=upB,
                initial_guess=initial_guess,
                stats=stats,
                stats_key="local_root_calls",
            )
        except Exception:
            if stats is not None:
                stats["local_fast_failures"] = stats.get("local_fast_failures", 0) + 1

    if nB_Q <= 0.0:
        raise RuntimeError("nB_Q must be positive when solving an entropy-enabled quark state")

    guesses = []
    muK_seed = _branch_muK_seed(a)
    muK_seed_strong = float(max(muK_seed, 400.0 * abs(float(a))))
    if initial_guess is not None:
        guess0 = np.asarray(initial_guess, dtype=float)
        if guess0.shape[0] != 3 or not np.all(np.isfinite(guess0)):
            raise RuntimeError("initial_guess must contain finite (muB, muK, T)")
        if guess0[2] <= 0.0:
            raise RuntimeError("initial_guess temperature must be positive")
        guesses.append(guess0)
        muB_seed = float(guess0[0])
        muK_ref = max(0.0, float(guess0[1]))
        T_seed = float(guess0[2])
    else:
        if T_ref is None or (not np.isfinite(T_ref)) or T_ref <= 0.0:
            raise RuntimeError("T_ref must be positive when no entropy-closure initial_guess is provided")
        muB_seed = 1200.0
        muK_ref = muK_seed
        T_seed = float(T_ref)

    guesses.append(np.array([muB_seed, muK_seed, T_seed], dtype=float))
    guesses.append(np.array([1200.0, muK_seed, T_seed], dtype=float))
    guesses.append(np.array([1500.0, max(muK_seed, 100.0 * abs(float(a))), T_seed], dtype=float))
    guesses.append(np.array([muB_seed, muK_seed_strong, T_seed], dtype=float))
    guesses.append(np.array([1500.0, muK_seed_strong, T_seed], dtype=float))
    if T_ref is not None and np.isfinite(T_ref) and T_ref > 0.0:
        guesses.append(np.array([muB_seed, max(muK_seed, muK_ref), float(T_ref)], dtype=float))

    best_message = "Entropy-enabled local quark-state solve did not converge"
    candidates = []
    candidate_tol = 1.0e-8
    nonneg_tol = 1.0e-8
    for guess in guesses:
        try:
            thermo = _solve_quark_entropy_state_once_from_guess(
                a,
                w,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                ms=ms,
                upB=upB,
                initial_guess=guess,
                stats=stats,
                stats_key="local_root_calls",
            )
            if thermo["muK"] < -nonneg_tol or thermo["s"] <= 0.0 or thermo["w"] <= 0.0:
                continue
            is_new = True
            for cand in candidates:
                if (
                    abs(thermo["muB"] - cand["muB"]) <= candidate_tol * max(1.0, abs(cand["muB"]))
                    and abs(thermo["muK"] - cand["muK"]) <= candidate_tol * max(1.0, abs(cand["muK"]), 1.0)
                    and abs(thermo["T"] - cand["T"]) <= candidate_tol * max(1.0, abs(cand["T"]))
                ):
                    is_new = False
                    break
            if is_new:
                candidates.append(thermo)
        except Exception as exc:
            best_message = str(exc)

    if candidates:
        if initial_guess is not None:
            muB_ref = float(initial_guess[0])
            muK_ref = max(0.0, float(initial_guess[1]))
            T_pref = float(initial_guess[2])
        else:
            muB_ref = muB_seed
            muK_ref = muK_seed
            T_pref = T_seed

        muK_pref = max(muK_seed, muK_ref)
        candidates.sort(
            key=lambda cand: (
                abs(cand["muK"] - muK_pref),
                abs(np.log(cand["T"] / T_pref)),
                abs(cand["muB"] - muB_ref),
                -cand["muK"],
            )
        )
        return candidates[0]

    raise RuntimeError(f"Entropy-enabled local quark-state solve failed: {best_message}")


def solve_steady_front_2d(
    T,
    nB_N,
    B_one_forth,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    x_max=5e7,
    n_eval=1200,
    rtol_ode=1e-8,
    atol_ode=1e-10,
    root_tol=1e-8,
    max_nfev=80,
    jB_guess=None,
    aQstar_guess=None,
    return_profile=False,
    verb=False,
):
    """
    Solve the full steady hydro + diffusion + reaction front with a 2D shooting method.

    Diagnostics:
    - verb=False: silent
    - verb=True: concise per-shoot summary lines only
    - verb="full": detailed stage-by-stage timing plus tqdm progress bars

    The solver explicitly rejects the near-trivial equilibrium branch where
    Qstar collapses onto the far-right equilibrated state Q and both a(0+) and
    q(0+) are numerically negligible.
    """
    if nB_N <= 0.0:
        raise RuntimeError("nB_N must be positive")

    upB = 5000
    t_start = time.perf_counter()
    diag_state = {"trial_count": 0}
    a_trivial_tol = max(1.0e-6, 10.0 * root_tol)
    q_trivial_tol = max(1.0e-10, root_tol * 1.0e-4)
    muK_trivial_tol = 1.0e-6
    nB_rel_trivial_tol = 1.0e-6
    if isinstance(verb, str):
        verb_mode = "full" if verb.lower() == "full" else ("simple" if verb else "off")
    else:
        verb_mode = "simple" if verb else "off"
    full_diag = verb_mode == "full"
    simple_diag = verb_mode in ("simple", "full")

    def _diag(msg):
        if full_diag:
            dt = time.perf_counter() - t_start
            print(f"[steady_front_2d +{dt:8.2f}s] {msg}", flush=True)

    if jB_guess is None:
        jB_guess = 1.0e-6 * nB_N
    if jB_guess <= 0.0:
        raise RuntimeError("jB_guess must be positive")

    if aQstar_guess is None:
        aQstar_guess = 0.5

    jB_lower_bound = max(1.0e-12, 1.0e-3 * float(jB_guess))
    jB_upper_bound = max(10.0 * float(jB_guess), 10.0)
    aQstar_lower_bound = max(1.0e-8, 1.0e-3 * max(abs(float(aQstar_guess)), 1.0e-2), 10.0 * root_tol)
    aQstar_upper_bound = np.inf
    best_trial = {"metric": np.inf, "z": None}

    if full_diag and tqdm is None:
        _diag("tqdm is not available; using timed prints only")

    def _build_trial_state(jB, aQstar, with_profile=False, trial_label="trial"):
        result = {
            "success": False,
            "message": "",
            "jB": float(jB),
            "aQstar": float(aQstar),
            "branch_label": "muK-rich",
        }

        try:
            _diag(f"{trial_label}: building N, Q, and Qstar states")

            # Upstream nuclear state N at x = 0^-.
            P_N = float(PNM_n(nB_N, T, param=param, NM_type=NM_type))
            e_N = float(edensNM_n(nB_N, T, param=param))
            h_N = float(P_N + e_N)
            u_N = float(jB / nB_N)
            Pi = float(h_N * u_N * u_N + P_N)

            # Far-right equilibrated quark state Q with muK = 0.
            _diag(f"{trial_label}: solving equilibrated Q at muK=0")
            muB_Q = _solve_muB_Q_at_muK0_for_given_Pi(Pi, jB, B_one_forth, T, ms=ms, upB=upB)
            nB_Q = float(nB_QM(muB_Q, 0.0, B_one_forth, T, ms=ms, upB=upB))
            if nB_Q <= 0.0:
                raise RuntimeError("Equilibrated Q state has non-positive density")
            if abs(ms) <= 1.0e-12:
                nK_Q = 0.0
            else:
                nK_Q = float(nK_QM(muB_Q, 0.0, B_one_forth, T, ms=ms, upB=upB))

            # Pure neutron matter implies nK_N = nB_N.
            a_N = float((nB_N - nK_Q) / nB_Q)

            # Interface state Qstar at x = 0+.
            _diag(f"{trial_label}: solving interface Qstar")
            muK_Qstar_seed = _branch_muK_seed(aQstar)
            muB_Qstar, muK_Qstar = _solve_interface_Qstar_from_aQstar_and_Pi(
                aQstar,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                T,
                ms=ms,
                upB=upB,
                initial_guess=(muB_Q, muK_Qstar_seed),
            )
            nB_Qstar = float(nB_QM(muB_Qstar, muK_Qstar, B_one_forth, T, ms=ms, upB=upB))
            if nB_Qstar <= 0.0:
                raise RuntimeError("Qstar state has non-positive density")

            micro = _microphysics_at_Qstar(muB_Qstar, T)
            D = float(micro["D"])
            eta = float(micro["eta"])
            gamma = float(micro["gamma"])
            tau = float(micro["tau"])

            # Jump condition gives the initial flux q(0+).
            q0 = float(-a_N * u_N)
            y0 = np.array([float(aQstar), q0], dtype=float)
            qstar_is_equilibrated = (
                abs(muK_Qstar) <= muK_trivial_tol
                and abs(nB_Qstar - nB_Q) <= nB_rel_trivial_tol * max(1.0, abs(nB_Q))
            )
            if abs(aQstar) <= a_trivial_tol and abs(q0) <= q_trivial_tol and qstar_is_equilibrated:
                raise RuntimeError("Rejected trivial equilibrium branch with a(0+)≈0 and q(0+)≈0")
            a_limit = 10.0 * max(1.0, abs(aQstar))
            cache = {"guess": (muB_Qstar, muK_Qstar)}
            ivp_state = {"rhs_calls": 0, "last_report": time.perf_counter()}

            _diag(f"{trial_label}: integrating IVP on [0, {x_max:.3e}]")

            def rhs(x, y):
                ivp_state["rhs_calls"] += 1
                a_val = float(y[0])
                q_val = float(y[1])
                if (not np.isfinite(a_val)) or (not np.isfinite(q_val)):
                    raise RuntimeError("Non-finite IVP state encountered")
                if abs(a_val) > a_limit:
                    raise RuntimeError("Composition variable exceeded stability guard")

                muB_loc, muK_loc, nB_loc, u_loc = _solve_local_quark_state_from_a_and_Pi(
                    a_val,
                    Pi,
                    jB,
                    nB_Q,
                    nK_Q,
                    B_one_forth,
                    T,
                    ms=ms,
                    upB=upB,
                    initial_guess=cache["guess"],
                    )
                cache["guess"] = (muB_loc, muK_loc)

                da_dx = (q_val + u_loc * a_val) / D
                dq_dx = gamma * (a_val**3 + eta * a_val)
                if (not np.isfinite(da_dx)) or (not np.isfinite(dq_dx)):
                    raise RuntimeError("Non-finite IVP derivative encountered")
                if full_diag:
                    now = time.perf_counter()
                    if now - ivp_state["last_report"] >= 5.0:
                        frac = (x / x_max) if x_max > 0.0 else 1.0
                        _diag(
                            f"{trial_label}: IVP x={x:.3e} ({100.0*frac:5.1f}%), "
                            f"a={a_val:.6g}, q={q_val:.6g}, rhs_calls={ivp_state['rhs_calls']}"
                        )
                        ivp_state["last_report"] = now
                return np.array([da_dx, dq_dx], dtype=float)

            t_eval = np.linspace(0.0, x_max, n_eval) if with_profile else None
            sol_ivp = solve_ivp(
                rhs,
                (0.0, x_max),
                y0,
                method="RK45",
                rtol=rtol_ode,
                atol=atol_ode,
                t_eval=t_eval,
            )
            if not sol_ivp.success:
                raise RuntimeError(f"IVP integration failed: {sol_ivp.message}")

            a_end = float(sol_ivp.y[0, -1])
            q_end = float(sol_ivp.y[1, -1])
            x_end = float(sol_ivp.t[-1])
            _diag(
                f"{trial_label}: IVP finished at x={x_end:.3e} with "
                f"rhs_calls={ivp_state['rhs_calls']}, a_end={a_end:.6g}, q_end={q_end:.6g}"
            )

            result.update(
                {
                    "success": True,
                    "message": "Steady front shooting trial completed",
                    "u_N": u_N,
                    "a_N": a_N,
                    "aQstar": float(aQstar),
                    "Pi": Pi,
                    "muB_Qstar": float(muB_Qstar),
                    "muK_Qstar": float(muK_Qstar),
                    "muB_Q": float(muB_Q),
                    "nB_Q": float(nB_Q),
                    "nK_Q": float(nK_Q),
                    "D": D,
                    "eta": eta,
                    "gamma": gamma,
                    "tau": tau,
                    "x_end": x_end,
                    "a_end": a_end,
                    "q_end": q_end,
                    "_residual": np.array([a_end, q_end], dtype=float),
                }
            )

            if with_profile:
                _diag(f"{trial_label}: reconstructing profile arrays at {len(sol_ivp.t)} points")
                x_prof = np.asarray(sol_ivp.t, dtype=float)
                a_prof = np.asarray(sol_ivp.y[0], dtype=float)
                q_prof = np.asarray(sol_ivp.y[1], dtype=float)
                muB_prof = np.empty_like(x_prof)
                muK_prof = np.empty_like(x_prof)
                nB_prof = np.empty_like(x_prof)
                u_prof = np.empty_like(x_prof)
                profile_guess = (muB_Qstar, muK_Qstar)
                profile_iter = range(len(a_prof))
                if full_diag and tqdm is not None:
                    profile_iter = tqdm(profile_iter, total=len(a_prof), desc=f"{trial_label} profile", unit="pt", leave=False)

                for i in profile_iter:
                    a_val = float(a_prof[i])
                    muB_loc, muK_loc, nB_loc, u_loc = _solve_local_quark_state_from_a_and_Pi(
                        a_val,
                        Pi,
                        jB,
                        nB_Q,
                        nK_Q,
                        B_one_forth,
                        T,
                        ms=ms,
                        upB=upB,
                        initial_guess=profile_guess,
                            )
                    profile_guess = (muB_loc, muK_loc)
                    muB_prof[i] = muB_loc
                    muK_prof[i] = muK_loc
                    nB_prof[i] = nB_loc
                    u_prof[i] = u_loc

                result.update(
                    {
                        "x": x_prof,
                        "a": a_prof,
                        "q": q_prof,
                        "u": u_prof,
                        "nB": nB_prof,
                        "muB": muB_prof,
                        "muK": muK_prof,
                    }
                )

            return result

        except Exception as exc:
            _diag(f"{trial_label}: failed with {exc}")
            result.update(
                {
                    "success": False,
                    "message": str(exc),
                    "u_N": np.nan,
                    "a_N": np.nan,
                    "Pi": np.nan,
                    "muB_Qstar": np.nan,
                    "muK_Qstar": np.nan,
                    "muB_Q": np.nan,
                    "nB_Q": np.nan,
                    "nK_Q": np.nan,
                    "D": np.nan,
                    "eta": np.nan,
                    "gamma": np.nan,
                    "tau": np.nan,
                    "x_end": np.nan,
                    "a_end": np.nan,
                    "q_end": np.nan,
                    "_residual": np.array([1.0e20, 1.0e20], dtype=float),
                }
            )
            return result

    def shooting_residual(z):
        diag_state["trial_count"] += 1
        trial_idx = diag_state["trial_count"]
        log_jB = float(z[0])
        aQstar = float(z[1])
        jB = float(np.exp(np.clip(log_jB, -700.0, 700.0)))

        if (jB < jB_lower_bound) or (aQstar < aQstar_lower_bound):
            trial_residual = np.array([1.0e20, 1.0e20], dtype=float)
            if simple_diag:
                print(
                    f"shoot jB={jB:.6g}, aQstar={aQstar:.6g}, "
                    f"res=({trial_residual[0]:.6g}, {trial_residual[1]:.6g}), ok=False"
                )
            if shooting_bar is not None:
                shooting_bar.update(1)
                shooting_bar.set_postfix_str(
                    f"jB={jB:.3e}, aQ={aQstar:.3g}, res={np.max(np.abs(trial_residual)):.2e}, ok=False"
                )
            return trial_residual

        _diag(f"shoot#{trial_idx}: start jB={jB:.6g}, aQstar={aQstar:.6g}")
        trial = _build_trial_state(jB, aQstar, with_profile=False, trial_label=f"shoot#{trial_idx}")
        trial_metric = float(np.max(np.abs(trial["_residual"])))
        if trial["success"] and np.isfinite(trial_metric) and trial_metric < best_trial["metric"]:
            best_trial["metric"] = trial_metric
            best_trial["z"] = np.array([log_jB, aQstar], dtype=float)
        if simple_diag:
            print(
                f"shoot jB={jB:.6g}, aQstar={aQstar:.6g}, "
                f"res=({trial['_residual'][0]:.6g}, {trial['_residual'][1]:.6g}), "
                f"ok={trial['success']}"
            )
        if shooting_bar is not None:
            shooting_bar.update(1)
            shooting_bar.set_postfix_str(
                f"jB={jB:.3e}, aQ={aQstar:.3g}, res={np.max(np.abs(trial['_residual'])):.2e}, ok={trial['success']}"
            )
        return trial["_residual"]

    z0 = np.array([np.log(float(jB_guess)), float(aQstar_guess)], dtype=float)
    lower_bounds = np.array([np.log(jB_lower_bound), aQstar_lower_bound], dtype=float)
    upper_bounds = np.array([np.log(jB_upper_bound), aQstar_upper_bound], dtype=float)
    z0 = np.minimum(np.maximum(z0, lower_bounds), upper_bounds)
    shooting_bar = None
    if full_diag and tqdm is not None:
        shooting_bar = tqdm(total=max_nfev, desc="shooting", unit="eval", leave=False)

    _diag(
        f"starting bounded least-squares shooting with jB_guess={jB_guess:.6g}, "
        f"aQstar_guess={aQstar_guess:.6g}, max_nfev={max_nfev}, "
        "branch=muK-rich"
    )
    try:
        sol_root = least_squares(
            shooting_residual,
            z0,
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            xtol=root_tol,
            ftol=root_tol,
            gtol=root_tol,
            max_nfev=max_nfev,
        )
    finally:
        if shooting_bar is not None:
            shooting_bar.close()

    z_best = sol_root.x if np.all(np.isfinite(sol_root.x)) else z0
    if best_trial["z"] is not None:
        best_trial_metric = float(best_trial["metric"])
        solver_metric = float(np.max(np.abs(sol_root.fun))) if np.all(np.isfinite(sol_root.fun)) else np.inf
        if best_trial_metric < solver_metric:
            z_best = best_trial["z"]
    jB_best = float(np.exp(np.clip(z_best[0], -700.0, 700.0)))
    aQstar_best = float(z_best[1])
    _diag(f"least-squares finished; rebuilding best state at jB={jB_best:.6g}, aQstar={aQstar_best:.6g}")

    result = _build_trial_state(jB_best, aQstar_best, with_profile=return_profile, trial_label="best")
    resid = np.asarray(result["_residual"], dtype=float)
    resid_norm = float(np.max(np.abs(resid)))
    success = bool(sol_root.success and result["success"] and resid_norm <= max(root_tol, 1.0e-6))

    result["success"] = success
    if success:
        result["message"] = "2D steady-front shooting converged"
    else:
        root_msg = sol_root.message if hasattr(sol_root, "message") else "root solve failed"
        if result["message"]:
            result["message"] = f"{root_msg}; last trial: {result['message']}"
        else:
            result["message"] = str(root_msg)

    result["_root_success"] = bool(sol_root.success)
    result["_root_message"] = str(sol_root.message) if hasattr(sol_root, "message") else ""
    result["_root_residual"] = resid
    _diag(
        f"finished solve_steady_front_2d: success={success}, "
        f"root_success={result['_root_success']}, resid_norm={resid_norm:.6g}"
    )
    return result


def solve_steady_front_2d_simple(
    T,
    nB_N,
    B_one_forth,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    x_max=5e7,
    n_eval=1200,
    rtol_ode=1e-8,
    atol_ode=1e-10,
    root_tol=1e-8,
    max_nfev=80,
    jB_guess=None,
    aQstar_guess=None,
    verb=False,
):
    """
    Convenience wrapper that returns the steady-front shooting summary without
    profile arrays.
    """
    return solve_steady_front_2d(
        T=T,
        nB_N=nB_N,
        B_one_forth=B_one_forth,
        ms=ms,
        param=param,
        NM_type=NM_type,
        x_max=x_max,
        n_eval=n_eval,
        rtol_ode=rtol_ode,
        atol_ode=atol_ode,
        root_tol=root_tol,
        max_nfev=max_nfev,
        jB_guess=jB_guess,
        aQstar_guess=aQstar_guess,
        return_profile=False,
        verb=verb,
    )


def solve_steady_front_1d_aQstar(
    T,
    nB_N,
    B_one_forth,
    aQstar,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    x_max=1e6,
    n_eval=1200,
    rtol_ode=1e-8,
    atol_ode=1e-10,
    root_tol=1e-8,
    max_nfev=60,
    jB_guess=None,
    jB_bounds=None,
    return_profile=False,
    verb=False,
):
    """
    Solve the steady-front problem with fixed aQstar and scalar shooting in jB.

    The scalar shooting residual is the asymptotic stable-tail condition at x = x_max:

        F(jB, aQstar) = q(L) + (D * lambda + u_Q) * a(L)

    where
        u_Q = jB / nB_Q
        lambda = (-u_Q + sqrt(u_Q^2 + 4 D gamma eta)) / (2 D)

    using D, eta, gamma frozen at Qstar and u_Q evaluated at the far-right
    equilibrated quark state Q.
    """
    if nB_N <= 0.0:
        raise RuntimeError("nB_N must be positive")

    upB = 5000
    t_start = time.perf_counter()
    diag_state = {"trial_count": 0}
    a_trivial_tol = max(1.0e-6, 10.0 * root_tol)
    q_trivial_tol = max(1.0e-10, root_tol * 1.0e-4)
    muK_trivial_tol = 1.0e-6
    nB_rel_trivial_tol = 1.0e-6
    if isinstance(verb, str):
        verb_mode = "full" if verb.lower() == "full" else ("simple" if verb else "off")
    else:
        verb_mode = "simple" if verb else "off"
    full_diag = verb_mode == "full"
    simple_diag = verb_mode in ("simple", "full")

    def _diag(msg):
        if full_diag:
            dt = time.perf_counter() - t_start
            print(f"[steady_front_1d +{dt:8.2f}s] {msg}", flush=True)

    if jB_guess is None:
        jB_guess = 1.0e-6 * nB_N
    if jB_guess <= 0.0:
        raise RuntimeError("jB_guess must be positive")

    if jB_bounds is None:
        jB_lower_bound = max(1.0e-12, 1.0e-3 * float(jB_guess))
        jB_upper_bound = max(10.0 * float(jB_guess), 10.0)
    else:
        if len(jB_bounds) != 2:
            raise RuntimeError("jB_bounds must be a 2-tuple (jB_min, jB_max)")
        jB_lower_bound = float(jB_bounds[0])
        jB_upper_bound = float(jB_bounds[1])
        if jB_lower_bound <= 0.0 or jB_upper_bound <= 0.0 or jB_upper_bound <= jB_lower_bound:
            raise RuntimeError("jB_bounds must satisfy 0 < jB_min < jB_max")

    best_trial = {"metric": np.inf, "log_jB": None}
    trial_cache = {}

    def _build_trial_state_1d(jB, with_profile=False, trial_label="trial"):
        result = {
            "success": False,
            "message": "",
            "jB": float(jB),
            "aQstar": float(aQstar),
            "branch_label": "muK-rich",
        }

        try:
            _diag(f"{trial_label}: building N, Q, and Qstar states")
            stats = {
                "q_root_calls": 0,
                "qstar_root_calls": 0,
                "local_state_calls": 0,
                "local_root_calls": 0,
                "local_fast_failures": 0,
                "profile_state_calls": 0,
            }

            # Upstream nuclear state N at x = 0^-.
            P_N = float(PNM_n(nB_N, T, param=param, NM_type=NM_type))
            e_N = float(edensNM_n(nB_N, T, param=param))
            h_N = float(P_N + e_N)
            u_N = float(jB / nB_N)
            Pi = float(h_N * u_N * u_N + P_N)

            # Far-right equilibrated quark state Q with muK = 0.
            _diag(f"{trial_label}: solving equilibrated Q at muK=0")
            muB_Q = _solve_muB_Q_at_muK0_for_given_Pi(Pi, jB, B_one_forth, T, ms=ms, upB=upB, stats=stats)
            nB_Q = float(nB_QM(muB_Q, 0.0, B_one_forth, T, ms=ms, upB=upB))
            if nB_Q <= 0.0:
                raise RuntimeError("Equilibrated Q state has non-positive density")
            if abs(ms) <= 1.0e-12:
                nK_Q = 0.0
            else:
                nK_Q = float(nK_QM(muB_Q, 0.0, B_one_forth, T, ms=ms, upB=upB))

            # Pure neutron matter implies nK_N = nB_N.
            a_N = float((nB_N - nK_Q) / nB_Q)

            # Interface state Qstar at x = 0+.
            _diag(f"{trial_label}: solving interface Qstar")
            muK_Qstar_seed = _branch_muK_seed(aQstar)
            muB_Qstar, muK_Qstar = _solve_interface_Qstar_from_aQstar_and_Pi(
                aQstar,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                T,
                ms=ms,
                upB=upB,
                initial_guess=(muB_Q, muK_Qstar_seed),
                stats=stats,
                stats_key="qstar_root_calls",
            )
            nB_Qstar = float(nB_QM(muB_Qstar, muK_Qstar, B_one_forth, T, ms=ms, upB=upB))
            if nB_Qstar <= 0.0:
                raise RuntimeError("Qstar state has non-positive density")

            micro = _microphysics_at_Qstar(muB_Qstar, T)
            D = float(micro["D"])
            eta = float(micro["eta"])
            gamma = float(micro["gamma"])
            tau = float(micro["tau"])

            # Jump condition gives the initial flux q(0+).
            q0 = float(-a_N * u_N)
            y0 = np.array([float(aQstar), q0], dtype=float)
            qstar_is_equilibrated = (
                abs(muK_Qstar) <= muK_trivial_tol
                and abs(nB_Qstar - nB_Q) <= nB_rel_trivial_tol * max(1.0, abs(nB_Q))
            )
            if abs(aQstar) <= a_trivial_tol and abs(q0) <= q_trivial_tol and qstar_is_equilibrated:
                raise RuntimeError("Rejected trivial equilibrium branch with a(0+)≈0 and q(0+)≈0")
            a_limit = 10.0 * max(1.0, abs(aQstar))
            cache = {"guess": (muB_Qstar, muK_Qstar)}
            ivp_state = {"rhs_calls": 0, "last_report": time.perf_counter()}

            _diag(f"{trial_label}: integrating IVP on [0, {x_max:.3e}]")

            def rhs(x, y):
                ivp_state["rhs_calls"] += 1
                a_val = float(y[0])
                q_val = float(y[1])
                if (not np.isfinite(a_val)) or (not np.isfinite(q_val)):
                    raise RuntimeError("Non-finite IVP state encountered")
                if abs(a_val) > a_limit:
                    raise RuntimeError("Composition variable exceeded stability guard")

                muB_loc, muK_loc, nB_loc, u_loc = _solve_local_quark_state_from_a_and_Pi(
                    a_val,
                    Pi,
                    jB,
                    nB_Q,
                    nK_Q,
                    B_one_forth,
                    T,
                    ms=ms,
                    upB=upB,
                    initial_guess=cache["guess"],
                        stats=stats,
                )
                cache["guess"] = (muB_loc, muK_loc)

                da_dx = (q_val + u_loc * a_val) / D
                dq_dx = gamma * (a_val**3 + eta * a_val)
                if (not np.isfinite(da_dx)) or (not np.isfinite(dq_dx)):
                    raise RuntimeError("Non-finite IVP derivative encountered")
                if full_diag:
                    now = time.perf_counter()
                    if now - ivp_state["last_report"] >= 5.0:
                        frac = (x / x_max) if x_max > 0.0 else 1.0
                        _diag(
                            f"{trial_label}: IVP x={x:.3e} ({100.0*frac:5.1f}%), "
                            f"a={a_val:.6g}, q={q_val:.6g}, rhs_calls={ivp_state['rhs_calls']}"
                        )
                        ivp_state["last_report"] = now
                return np.array([da_dx, dq_dx], dtype=float)

            t_eval = np.linspace(0.0, x_max, n_eval) if with_profile else None
            sol_ivp = solve_ivp(
                rhs,
                (0.0, x_max),
                y0,
                method="RK45",
                rtol=rtol_ode,
                atol=atol_ode,
                t_eval=t_eval,
            )
            if not sol_ivp.success:
                raise RuntimeError(f"IVP integration failed: {sol_ivp.message}")

            a_end = float(sol_ivp.y[0, -1])
            q_end = float(sol_ivp.y[1, -1])
            x_end = float(sol_ivp.t[-1])
            _diag(
                f"{trial_label}: IVP finished at x={x_end:.3e} with "
                f"rhs_calls={ivp_state['rhs_calls']}, a_end={a_end:.6g}, q_end={q_end:.6g}"
            )

            u_Q = float(jB / nB_Q)
            disc = float(u_Q * u_Q + 4.0 * D * gamma * eta)
            if (not np.isfinite(disc)) or disc <= 0.0:
                raise RuntimeError("Tail discriminant is non-positive")
            lam = float((-u_Q + np.sqrt(disc)) / (2.0 * D))
            tail_coeff = float(D * lam + u_Q)
            tail_drive = float(tail_coeff * a_end)
            tail_residual = float(q_end + tail_drive)
            tail_scale = float(max(abs(q_end), abs(tail_drive), np.finfo(float).tiny))
            tail_residual_norm = float(tail_residual / tail_scale)

            result.update(
                {
                    "success": True,
                    "message": "1D steady-front shooting trial completed",
                    "u_N": u_N,
                    "u_Q": u_Q,
                    "a_N": a_N,
                    "aQstar": float(aQstar),
                    "Pi": Pi,
                    "muB_Qstar": float(muB_Qstar),
                    "muK_Qstar": float(muK_Qstar),
                    "nB_Qstar": float(nB_Qstar),
                    "muB_Q": float(muB_Q),
                    "nB_Q": float(nB_Q),
                    "nK_Q": float(nK_Q),
                    "D": D,
                    "eta": eta,
                    "gamma": gamma,
                    "tau": tau,
                    "lambda": lam,
                    "tail_residual": tail_residual,
                    "tail_residual_norm": tail_residual_norm,
                    "tail_scale": tail_scale,
                    "x_end": x_end,
                    "a_end": a_end,
                    "q_end": q_end,
                    "_residual": np.array([tail_residual_norm], dtype=float),
                    "shooting_evals": int(diag_state["trial_count"]),
                    "ivp_rhs_calls": int(ivp_state["rhs_calls"]),
                    "q_root_calls": int(stats["q_root_calls"]),
                    "qstar_root_calls": int(stats["qstar_root_calls"]),
                    "local_state_calls": int(stats["local_state_calls"]),
                    "local_root_calls": int(stats["local_root_calls"]),
                    "local_fast_failures": int(stats["local_fast_failures"]),
                    "profile_state_calls": int(stats["profile_state_calls"]),
                }
            )

            if with_profile:
                _diag(f"{trial_label}: reconstructing profile arrays at {len(sol_ivp.t)} points")
                x_prof = np.asarray(sol_ivp.t, dtype=float)
                a_prof = np.asarray(sol_ivp.y[0], dtype=float)
                q_prof = np.asarray(sol_ivp.y[1], dtype=float)
                muB_prof = np.empty_like(x_prof)
                muK_prof = np.empty_like(x_prof)
                nB_prof = np.empty_like(x_prof)
                u_prof = np.empty_like(x_prof)
                profile_guess = (muB_Qstar, muK_Qstar)
                profile_iter = range(len(a_prof))
                if full_diag and tqdm is not None:
                    profile_iter = tqdm(profile_iter, total=len(a_prof), desc=f"{trial_label} profile", unit="pt", leave=False)

                for i in profile_iter:
                    a_val = float(a_prof[i])
                    stats["profile_state_calls"] += 1
                    muB_loc, muK_loc, nB_loc, u_loc = _solve_local_quark_state_from_a_and_Pi(
                        a_val,
                        Pi,
                        jB,
                        nB_Q,
                        nK_Q,
                        B_one_forth,
                        T,
                        ms=ms,
                        upB=upB,
                        initial_guess=profile_guess,
                                stats=stats,
                    )
                    profile_guess = (muB_loc, muK_loc)
                    muB_prof[i] = muB_loc
                    muK_prof[i] = muK_loc
                    nB_prof[i] = nB_loc
                    u_prof[i] = u_loc

                result.update(
                    {
                        "x": x_prof,
                        "a": a_prof,
                        "q": q_prof,
                        "u": u_prof,
                        "nB": nB_prof,
                        "muB": muB_prof,
                        "muK": muK_prof,
                        "profile_state_calls": int(stats["profile_state_calls"]),
                        "local_state_calls": int(stats["local_state_calls"]),
                        "local_root_calls": int(stats["local_root_calls"]),
                        "local_fast_failures": int(stats["local_fast_failures"]),
                    }
                )

            return result

        except Exception as exc:
            _diag(f"{trial_label}: failed with {exc}")
            result.update(
                {
                    "success": False,
                    "message": str(exc),
                    "u_N": np.nan,
                    "u_Q": np.nan,
                    "a_N": np.nan,
                    "Pi": np.nan,
                    "muB_Qstar": np.nan,
                    "muK_Qstar": np.nan,
                    "nB_Qstar": np.nan,
                    "muB_Q": np.nan,
                    "nB_Q": np.nan,
                    "nK_Q": np.nan,
                    "D": np.nan,
                    "eta": np.nan,
                    "gamma": np.nan,
                    "tau": np.nan,
                    "lambda": np.nan,
                    "tail_residual": np.nan,
                    "tail_residual_norm": np.nan,
                    "tail_scale": np.nan,
                    "x_end": np.nan,
                    "a_end": np.nan,
                    "q_end": np.nan,
                    "_residual": np.array([1.0e20], dtype=float),
                    "shooting_evals": int(diag_state["trial_count"]),
                    "ivp_rhs_calls": 0,
                    "q_root_calls": 0,
                    "qstar_root_calls": 0,
                    "local_state_calls": 0,
                    "local_root_calls": 0,
                    "local_fast_failures": 0,
                    "profile_state_calls": 0,
                }
            )
            return result

    def _trial_from_log_jB(log_jB, with_profile=False, trial_label="trial"):
        cache_key = (round(float(log_jB), 14), bool(with_profile))
        if cache_key in trial_cache:
            return trial_cache[cache_key]
        jB = float(np.exp(np.clip(log_jB, -700.0, 700.0)))
        result = _build_trial_state_1d(jB, with_profile=with_profile, trial_label=trial_label)
        trial_cache[cache_key] = result
        return result

    def _tail_residual_scalar(log_jB):
        diag_state["trial_count"] += 1
        trial_idx = diag_state["trial_count"]
        jB = float(np.exp(np.clip(log_jB, -700.0, 700.0)))
        if jB < jB_lower_bound or jB > jB_upper_bound:
            tail_residual = 1.0e20
            if simple_diag:
                print(
                    f"shoot1d jB={jB:.6g}, aQstar={aQstar:.6g}, "
                    f"tail_norm=({tail_residual:.6g}), ok=False"
                )
            return tail_residual

        trial = _trial_from_log_jB(log_jB, with_profile=False, trial_label=f"shoot1d#{trial_idx}")
        trial_resid = float(trial.get("tail_residual_norm", np.nan))
        trial_metric = float(abs(trial_resid)) if np.isfinite(trial_resid) else np.inf
        if trial["success"] and trial_metric < best_trial["metric"]:
            best_trial["metric"] = trial_metric
            best_trial["log_jB"] = float(log_jB)
        if simple_diag:
            print(
                f"shoot1d jB={jB:.6g}, aQstar={aQstar:.6g}, "
                f"tail_norm=({trial.get('tail_residual_norm', np.nan):.6g}), "
                f"tail_raw=({trial.get('tail_residual', np.nan):.6g}), ok={trial['success']}"
            )
        return trial_resid if np.isfinite(trial_resid) else 1.0e20

    log_lower = float(np.log(jB_lower_bound))
    log_upper = float(np.log(jB_upper_bound))
    log_guess = float(np.log(jB_guess))
    log_guess = min(max(log_guess, log_lower), log_upper)

    _diag(
        f"starting 1D shooting with jB_guess={jB_guess:.6g}, aQstar={aQstar:.6g}, "
        f"bounds=[{jB_lower_bound:.6g}, {jB_upper_bound:.6g}], "
        "branch=muK-rich"
    )

    bracket = None
    bracket_found = False
    bracket_evals = 0
    root_method = "least_squares"
    root_success = False
    root_message = ""
    z_best = log_guess

    f_guess = _tail_residual_scalar(log_guess)
    bracket_evals += 1
    if np.isfinite(f_guess) and abs(f_guess) <= max(root_tol, 1.0e-6):
        root_method = "initial_guess"
        root_success = True
        root_message = "initial guess satisfies normalized tail residual tolerance"
        z_best = log_guess
    else:
        expand_factor = 1.5
        step = float(np.log(expand_factor))
        max_expand_steps = max(1, min(8, max_nfev))
        for k in range(1, max_expand_steps + 1):
            left_z = max(log_lower, log_guess - k * step)
            right_z = min(log_upper, log_guess + k * step)

            if left_z < log_guess:
                f_left = _tail_residual_scalar(left_z)
                bracket_evals += 1
                if np.isfinite(f_left) and np.isfinite(f_guess) and f_left * f_guess <= 0.0:
                    bracket = (left_z, log_guess)
                    bracket_found = True
                    break

            if right_z > log_guess:
                f_right = _tail_residual_scalar(right_z)
                bracket_evals += 1
                if np.isfinite(f_right) and np.isfinite(f_guess) and f_guess * f_right <= 0.0:
                    bracket = (log_guess, right_z)
                    bracket_found = True
                    break

            if left_z <= log_lower and right_z >= log_upper:
                break

    if bracket is not None:
        _diag(f"found scalar bracket in log_jB: [{bracket[0]:.6g}, {bracket[1]:.6g}]")
        sol_root = root_scalar(lambda z: _tail_residual_scalar(float(z)), bracket=bracket, method="brentq", xtol=root_tol)
        root_method = "brentq"
        root_success = bool(sol_root.converged)
        root_message = str(sol_root.flag)
        if root_success:
            z_best = float(sol_root.root)
        elif best_trial["log_jB"] is not None:
            z_best = float(best_trial["log_jB"])
    elif not root_success:
        _diag("no sign-changing bracket found; falling back to bounded least-squares in log_jB")
        sol_root = least_squares(
            lambda z: np.array([_tail_residual_scalar(float(z[0]))], dtype=float),
            np.array([log_guess], dtype=float),
            bounds=(np.array([log_lower]), np.array([log_upper])),
            method="trf",
            xtol=root_tol,
            ftol=root_tol,
            gtol=root_tol,
            max_nfev=max_nfev,
        )
        root_success = bool(sol_root.success)
        root_message = str(sol_root.message)
        if np.all(np.isfinite(sol_root.x)):
            z_best = float(sol_root.x[0])
        if best_trial["log_jB"] is not None:
            best_metric = float(best_trial["metric"])
            solver_metric = abs(_tail_residual_scalar(float(z_best)))
            if best_metric < solver_metric:
                z_best = float(best_trial["log_jB"])

    jB_best = float(np.exp(np.clip(z_best, -700.0, 700.0)))
    _diag(f"1D shooting finished; rebuilding best state at jB={jB_best:.6g}, aQstar={aQstar:.6g}")
    result = _trial_from_log_jB(z_best, with_profile=return_profile, trial_label="best1d")
    tail_resid = float(result["tail_residual_norm"]) if np.isfinite(result["tail_residual_norm"]) else np.inf
    tail_resid_raw = float(result["tail_residual"]) if np.isfinite(result["tail_residual"]) else np.inf
    success = bool(root_success and result["success"] and abs(tail_resid) <= root_tol)

    result["success"] = success
    if success:
        result["message"] = "1D steady-front shooting converged"
    else:
        if result["message"]:
            result["message"] = f"{root_message}; last trial: {result['message']}"
        else:
            result["message"] = root_message

    result["_root_success"] = root_success
    result["_root_message"] = root_message
    result["_root_method"] = root_method
    result["_root_residual"] = np.array([tail_resid], dtype=float)
    result["_root_raw_residual"] = np.array([tail_resid_raw], dtype=float)
    result["_bracket_found"] = bool(bracket_found)
    result["_bracket_evals"] = int(bracket_evals)
    result["shooting_evals"] = int(diag_state["trial_count"])
    return result



def solve_steady_front_1d_aQstar_rescaled(
    T,
    nB_N,
    B_one_forth,
    aQstar,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    x_max=None,
    n_eval=1200,
    rtol_ode=1e-8,
    atol_ode=1e-10,
    root_tol=1e-8,
    max_nfev=60,
    jB_guess=None,
    jB_bounds=None,
    tail_eps=1e-8,
    kappa_factor=1.0,
    compact_tail_lengths=0.35,
    return_profile=False,
    verb=False,
):
    """
    Solve the fixed-aQstar steady-front problem with scalar shooting in jB on a
    compactified coordinate.

    This is the same hydro + diffusion + reaction problem solved by
    solve_steady_front_1d_aQstar, but the quark-side IVP is integrated in

        s = 1 - exp(-lambda*x/kappa_factor),     s in [0, 1)

    instead of directly in x. The physical equations are unchanged; only the
    independent variable is transformed with
        dx/ds = kappa_factor / (lambda*(1-s)).
    The scale is
    chosen from the same frozen-Qstar linear tail used by the finite-domain 1D
    solver,

        lambda = (-u_Q + sqrt(u_Q^2 + 4 D gamma eta)) / (2 D),

    with u_Q = jB/nB_Q at the far-right equilibrated quark state Q.
    """
    if nB_N <= 0.0:
        raise RuntimeError("nB_N must be positive")
    if tail_eps <= 0.0 or tail_eps >= 1.0:
        raise RuntimeError("tail_eps must satisfy 0 < tail_eps < 1")
    if kappa_factor <= 0.0:
        raise RuntimeError("kappa_factor must be positive")
    if compact_tail_lengths <= 0.0:
        raise RuntimeError("compact_tail_lengths must be positive")
    if x_max is not None and x_max <= 0.0:
        raise RuntimeError("x_max must be positive when provided")

    upB = 5000
    t_start = time.perf_counter()
    diag_state = {"trial_count": 0}
    a_trivial_tol = max(1.0e-6, 10.0 * root_tol)
    q_trivial_tol = max(1.0e-10, root_tol * 1.0e-4)
    muK_trivial_tol = 1.0e-6
    nB_rel_trivial_tol = 1.0e-6
    if isinstance(verb, str):
        verb_mode = "full" if verb.lower() == "full" else ("simple" if verb else "off")
    else:
        verb_mode = "simple" if verb else "off"
    full_diag = verb_mode == "full"
    simple_diag = verb_mode in ("simple", "full")

    def _diag(msg):
        if full_diag:
            dt = time.perf_counter() - t_start
            print(f"[steady_front_1d_rescaled +{dt:8.2f}s] {msg}", flush=True)

    if jB_guess is None:
        jB_guess = 1.0e-6 * nB_N
    if jB_guess <= 0.0:
        raise RuntimeError("jB_guess must be positive")

    if jB_bounds is None:
        jB_lower_bound = max(1.0e-12, 1.0e-3 * float(jB_guess))
        jB_upper_bound = max(10.0 * float(jB_guess), 10.0)
    else:
        if len(jB_bounds) != 2:
            raise RuntimeError("jB_bounds must be a 2-tuple (jB_min, jB_max)")
        jB_lower_bound = float(jB_bounds[0])
        jB_upper_bound = float(jB_bounds[1])
        if jB_lower_bound <= 0.0 or jB_upper_bound <= 0.0 or jB_upper_bound <= jB_lower_bound:
            raise RuntimeError("jB_bounds must satisfy 0 < jB_min < jB_max")

    best_trial = {"metric": np.inf, "log_jB": None}
    trial_cache = {}

    def _build_trial_state_1d_rescaled(jB, with_profile=False, trial_label="trial"):
        result = {
            "success": False,
            "message": "",
            "jB": float(jB),
            "aQstar": float(aQstar),
            "branch_label": "muK-rich",
            "coordinate": "s=1-exp(-lambda*x/kappa_factor)",
            "tail_eps": float(tail_eps),
            "kappa_factor": float(kappa_factor),
            "compact_tail_lengths": float(compact_tail_lengths),
        }

        try:
            _diag(f"{trial_label}: building N, Q, and Qstar states")
            stats = {
                "q_root_calls": 0,
                "qstar_root_calls": 0,
                "local_state_calls": 0,
                "local_root_calls": 0,
                "local_fast_failures": 0,
                "profile_state_calls": 0,
            }

            # Upstream nuclear state N at x = 0^-.
            P_N = float(PNM_n(nB_N, T, param=param, NM_type=NM_type))
            e_N = float(edensNM_n(nB_N, T, param=param))
            h_N = float(P_N + e_N)
            u_N = float(jB / nB_N)
            Pi = float(h_N * u_N * u_N + P_N)

            # Far-right equilibrated quark state Q with muK = 0.
            _diag(f"{trial_label}: solving equilibrated Q at muK=0")
            muB_Q = _solve_muB_Q_at_muK0_for_given_Pi(Pi, jB, B_one_forth, T, ms=ms, upB=upB, stats=stats)
            nB_Q = float(nB_QM(muB_Q, 0.0, B_one_forth, T, ms=ms, upB=upB))
            if nB_Q <= 0.0:
                raise RuntimeError("Equilibrated Q state has non-positive density")
            if abs(ms) <= 1.0e-12:
                nK_Q = 0.0
            else:
                nK_Q = float(nK_QM(muB_Q, 0.0, B_one_forth, T, ms=ms, upB=upB))

            # Pure neutron matter implies nK_N = nB_N.
            a_N = float((nB_N - nK_Q) / nB_Q)

            # Interface state Qstar at x = 0+.
            _diag(f"{trial_label}: solving interface Qstar")
            muK_Qstar_seed = _branch_muK_seed(aQstar)
            muB_Qstar, muK_Qstar = _solve_interface_Qstar_from_aQstar_and_Pi(
                aQstar,
                Pi,
                jB,
                nB_Q,
                nK_Q,
                B_one_forth,
                T,
                ms=ms,
                upB=upB,
                initial_guess=(muB_Q, muK_Qstar_seed),
                stats=stats,
                stats_key="qstar_root_calls",
            )
            nB_Qstar = float(nB_QM(muB_Qstar, muK_Qstar, B_one_forth, T, ms=ms, upB=upB))
            if nB_Qstar <= 0.0:
                raise RuntimeError("Qstar state has non-positive density")

            micro = _microphysics_at_Qstar(muB_Qstar, T)
            D = float(micro["D"])
            eta = float(micro["eta"])
            gamma = float(micro["gamma"])
            tau = float(micro["tau"])

            u_Q = float(jB / nB_Q)
            disc = float(u_Q * u_Q + 4.0 * D * gamma * eta)
            if (not np.isfinite(disc)) or disc <= 0.0:
                raise RuntimeError("Tail discriminant is non-positive")
            lam = float((-u_Q + np.sqrt(disc)) / (2.0 * D))
            if (not np.isfinite(lam)) or lam <= 0.0:
                raise RuntimeError("Tail decay lambda must be positive")
            if x_max is None:
                # Do not integrate directly to 1-tail_eps by default: the compact
                # ODE has dx/ds = kappa_factor/(lambda*(1-s)), so the endpoint is singular. Use a
                # tunable number of tail lengths and keep tail_eps as a hard cap.
                s_from_auto_tail = float(-np.expm1(-float(compact_tail_lengths)))
                s_end = float(min(1.0 - tail_eps, s_from_auto_tail))
            else:
                # 1 - exp(-lambda*x/kappa_factor), evaluated via expm1 for small lambda*x/kappa_factor.
                s_from_xmax = float(-np.expm1(-float(x_max) * lam / float(kappa_factor)))
                s_end = float(min(1.0 - tail_eps, max(0.0, s_from_xmax)))
            if (not np.isfinite(s_end)) or s_end <= 0.0 or s_end >= 1.0:
                raise RuntimeError("Invalid compact-coordinate endpoint")
            x_end_physical = float(-float(kappa_factor) * np.log1p(-s_end) / lam)

            # Jump condition gives the initial flux q(0+).
            q0 = float(-a_N * u_N)
            y0 = np.array([float(aQstar), q0], dtype=float)
            qstar_is_equilibrated = (
                abs(muK_Qstar) <= muK_trivial_tol
                and abs(nB_Qstar - nB_Q) <= nB_rel_trivial_tol * max(1.0, abs(nB_Q))
            )
            if abs(aQstar) <= a_trivial_tol and abs(q0) <= q_trivial_tol and qstar_is_equilibrated:
                raise RuntimeError("Rejected trivial equilibrium branch with a(0+)≈0 and q(0+)≈0")
            a_limit = 10.0 * max(1.0, abs(aQstar))
            cache = {"guess": (muB_Qstar, muK_Qstar)}
            ivp_state = {"rhs_calls": 0, "last_report": time.perf_counter()}

            _diag(
                f"{trial_label}: integrating compact IVP on s=[0, {s_end:.8g}] "
                f"(x_end={x_end_physical:.3e}, lambda={lam:.3e})"
            )

            def rhs_s(s, y):
                ivp_state["rhs_calls"] += 1
                a_val = float(y[0])
                q_val = float(y[1])
                if (not np.isfinite(a_val)) or (not np.isfinite(q_val)):
                    raise RuntimeError("Non-finite IVP state encountered")
                if abs(a_val) > a_limit:
                    raise RuntimeError("Composition variable exceeded stability guard")

                muB_loc, muK_loc, nB_loc, u_loc = _solve_local_quark_state_from_a_and_Pi(
                    a_val,
                    Pi,
                    jB,
                    nB_Q,
                    nK_Q,
                    B_one_forth,
                    T,
                    ms=ms,
                    upB=upB,
                    initial_guess=cache["guess"],
                        stats=stats,
                )
                cache["guess"] = (muB_loc, muK_loc)

                one_minus_s = max(1.0 - float(s), np.finfo(float).tiny)
                dx_ds = float(kappa_factor) / (lam * one_minus_s)
                da_ds = ((q_val + u_loc * a_val) / D) * dx_ds
                dq_ds = (gamma * (a_val**3 + eta * a_val)) * dx_ds
                if (not np.isfinite(da_ds)) or (not np.isfinite(dq_ds)):
                    raise RuntimeError("Non-finite IVP derivative encountered")
                if full_diag:
                    now = time.perf_counter()
                    if now - ivp_state["last_report"] >= 5.0:
                        x_now = -float(kappa_factor) * np.log1p(-min(float(s), 1.0 - np.finfo(float).eps)) / lam
                        frac = float(s) / s_end if s_end > 0.0 else 1.0
                        _diag(
                            f"{trial_label}: IVP s={s:.6g} ({100.0*frac:5.1f}%), "
                            f"x={x_now:.3e}, a={a_val:.6g}, q={q_val:.6g}, "
                            f"rhs_calls={ivp_state['rhs_calls']}"
                        )
                        ivp_state["last_report"] = now
                return np.array([da_ds, dq_ds], dtype=float)

            s_eval = np.linspace(0.0, s_end, n_eval) if with_profile else None
            sol_ivp = solve_ivp(
                rhs_s,
                (0.0, s_end),
                y0,
                method="RK45",
                rtol=rtol_ode,
                atol=atol_ode,
                t_eval=s_eval,
            )
            if not sol_ivp.success:
                raise RuntimeError(f"Compact IVP integration failed: {sol_ivp.message}")

            a_end = float(sol_ivp.y[0, -1])
            q_end = float(sol_ivp.y[1, -1])
            s_reached = float(sol_ivp.t[-1])
            x_reached = float(-float(kappa_factor) * np.log1p(-s_reached) / lam)
            _diag(
                f"{trial_label}: compact IVP finished at s={s_reached:.6g}, "
                f"x={x_reached:.3e} with rhs_calls={ivp_state['rhs_calls']}, "
                f"a_end={a_end:.6g}, q_end={q_end:.6g}"
            )

            tail_coeff = float(D * lam + u_Q)
            tail_drive = float(tail_coeff * a_end)
            tail_residual = float(q_end + tail_drive)
            tail_scale = float(max(abs(q_end), abs(tail_drive), np.finfo(float).tiny))
            tail_residual_norm = float(tail_residual / tail_scale)

            result.update(
                {
                    "success": True,
                    "message": "Rescaled 1D steady-front shooting trial completed",
                    "u_N": u_N,
                    "u_Q": u_Q,
                    "a_N": a_N,
                    "aQstar": float(aQstar),
                    "Pi": Pi,
                    "muB_Qstar": float(muB_Qstar),
                    "muK_Qstar": float(muK_Qstar),
                    "nB_Qstar": float(nB_Qstar),
                    "muB_Q": float(muB_Q),
                    "nB_Q": float(nB_Q),
                    "nK_Q": float(nK_Q),
                    "D": D,
                    "eta": eta,
                    "gamma": gamma,
                    "tau": tau,
                    "lambda": lam,
                    "kappa": float(kappa_factor / lam),
                    "compact_tail_lengths": float(compact_tail_lengths),
                    "s_end": s_reached,
                    "x_end": x_reached,
                    "x_end_target": x_end_physical,
                    "tail_residual": tail_residual,
                    "tail_residual_norm": tail_residual_norm,
                    "tail_scale": tail_scale,
                    "a_end": a_end,
                    "q_end": q_end,
                    "_residual": np.array([tail_residual_norm], dtype=float),
                    "shooting_evals": int(diag_state["trial_count"]),
                    "ivp_rhs_calls": int(ivp_state["rhs_calls"]),
                    "q_root_calls": int(stats["q_root_calls"]),
                    "qstar_root_calls": int(stats["qstar_root_calls"]),
                    "local_state_calls": int(stats["local_state_calls"]),
                    "local_root_calls": int(stats["local_root_calls"]),
                    "local_fast_failures": int(stats["local_fast_failures"]),
                    "profile_state_calls": int(stats["profile_state_calls"]),
                }
            )

            if with_profile:
                _diag(f"{trial_label}: reconstructing profile arrays at {len(sol_ivp.t)} points")
                s_prof = np.asarray(sol_ivp.t, dtype=float)
                x_prof = -float(kappa_factor) * np.log1p(-s_prof) / lam
                a_prof = np.asarray(sol_ivp.y[0], dtype=float)
                q_prof = np.asarray(sol_ivp.y[1], dtype=float)
                muB_prof = np.empty_like(s_prof)
                muK_prof = np.empty_like(s_prof)
                nB_prof = np.empty_like(s_prof)
                u_prof = np.empty_like(s_prof)
                profile_guess = (muB_Qstar, muK_Qstar)
                profile_iter = range(len(a_prof))
                if full_diag and tqdm is not None:
                    profile_iter = tqdm(profile_iter, total=len(a_prof), desc=f"{trial_label} profile", unit="pt", leave=False)

                for i in profile_iter:
                    a_val = float(a_prof[i])
                    stats["profile_state_calls"] += 1
                    muB_loc, muK_loc, nB_loc, u_loc = _solve_local_quark_state_from_a_and_Pi(
                        a_val,
                        Pi,
                        jB,
                        nB_Q,
                        nK_Q,
                        B_one_forth,
                        T,
                        ms=ms,
                        upB=upB,
                        initial_guess=profile_guess,
                                stats=stats,
                    )
                    profile_guess = (muB_loc, muK_loc)
                    muB_prof[i] = muB_loc
                    muK_prof[i] = muK_loc
                    nB_prof[i] = nB_loc
                    u_prof[i] = u_loc

                result.update(
                    {
                        "s": s_prof,
                        "x": x_prof,
                        "a": a_prof,
                        "q": q_prof,
                        "u": u_prof,
                        "nB": nB_prof,
                        "muB": muB_prof,
                        "muK": muK_prof,
                        "profile_state_calls": int(stats["profile_state_calls"]),
                        "local_state_calls": int(stats["local_state_calls"]),
                        "local_root_calls": int(stats["local_root_calls"]),
                        "local_fast_failures": int(stats["local_fast_failures"]),
                    }
                )

            return result

        except Exception as exc:
            _diag(f"{trial_label}: failed with {exc}")
            result.update(
                {
                    "success": False,
                    "message": str(exc),
                    "u_N": np.nan,
                    "u_Q": np.nan,
                    "a_N": np.nan,
                    "Pi": np.nan,
                    "muB_Qstar": np.nan,
                    "muK_Qstar": np.nan,
                    "nB_Qstar": np.nan,
                    "muB_Q": np.nan,
                    "nB_Q": np.nan,
                    "nK_Q": np.nan,
                    "D": np.nan,
                    "eta": np.nan,
                    "gamma": np.nan,
                    "tau": np.nan,
                    "lambda": np.nan,
                    "kappa": np.nan,
                    "compact_tail_lengths": float(compact_tail_lengths),
                    "s_end": np.nan,
                    "x_end": np.nan,
                    "x_end_target": np.nan,
                    "tail_residual": np.nan,
                    "tail_residual_norm": np.nan,
                    "tail_scale": np.nan,
                    "a_end": np.nan,
                    "q_end": np.nan,
                    "_residual": np.array([1.0e20], dtype=float),
                    "shooting_evals": int(diag_state["trial_count"]),
                    "ivp_rhs_calls": 0,
                    "q_root_calls": 0,
                    "qstar_root_calls": 0,
                    "local_state_calls": 0,
                    "local_root_calls": 0,
                    "local_fast_failures": 0,
                    "profile_state_calls": 0,
                }
            )
            return result

    def _trial_from_log_jB(log_jB, with_profile=False, trial_label="trial"):
        cache_key = (round(float(log_jB), 14), bool(with_profile))
        if cache_key in trial_cache:
            return trial_cache[cache_key]
        jB = float(np.exp(np.clip(log_jB, -700.0, 700.0)))
        result = _build_trial_state_1d_rescaled(jB, with_profile=with_profile, trial_label=trial_label)
        trial_cache[cache_key] = result
        return result

    def _tail_residual_scalar(log_jB):
        diag_state["trial_count"] += 1
        trial_idx = diag_state["trial_count"]
        jB = float(np.exp(np.clip(log_jB, -700.0, 700.0)))
        if jB < jB_lower_bound or jB > jB_upper_bound:
            tail_residual = 1.0e20
            if simple_diag:
                print(
                    f"shoot1d-rescaled jB={jB:.6g}, aQstar={aQstar:.6g}, "
                    f"tail_norm=({tail_residual:.6g}), ok=False"
                )
            return tail_residual

        trial = _trial_from_log_jB(log_jB, with_profile=False, trial_label=f"shoot1d-rescaled#{trial_idx}")
        trial_resid = float(trial.get("tail_residual_norm", np.nan))
        trial_metric = float(abs(trial_resid)) if np.isfinite(trial_resid) else np.inf
        if trial["success"] and trial_metric < best_trial["metric"]:
            best_trial["metric"] = trial_metric
            best_trial["log_jB"] = float(log_jB)
        if simple_diag:
            print(
                f"shoot1d-rescaled jB={jB:.6g}, aQstar={aQstar:.6g}, "
                f"tail_norm=({trial.get('tail_residual_norm', np.nan):.6g}), "
                f"tail_raw=({trial.get('tail_residual', np.nan):.6g}), "
                f"s_end={trial.get('s_end', np.nan):.6g}, x_end={trial.get('x_end', np.nan):.6g}, "
                f"ok={trial['success']}"
            )
        return trial_resid if np.isfinite(trial_resid) else 1.0e20

    log_lower = float(np.log(jB_lower_bound))
    log_upper = float(np.log(jB_upper_bound))
    log_guess = float(np.log(jB_guess))
    log_guess = min(max(log_guess, log_lower), log_upper)

    _diag(
        f"starting rescaled 1D shooting with jB_guess={jB_guess:.6g}, "
        f"aQstar={aQstar:.6g}, bounds=[{jB_lower_bound:.6g}, {jB_upper_bound:.6g}], "
        f"branch=muK-rich, tail_eps={tail_eps:.3g}, "
        f"kappa_factor={kappa_factor:.3g}, compact_tail_lengths={compact_tail_lengths:.3g}"
    )

    bracket = None
    bracket_found = False
    bracket_evals = 0
    root_method = "least_squares"
    root_success = False
    root_message = ""
    z_best = log_guess

    f_guess = _tail_residual_scalar(log_guess)
    bracket_evals += 1
    if np.isfinite(f_guess) and abs(f_guess) <= root_tol:
        root_method = "initial_guess"
        root_success = True
        root_message = "initial guess satisfies normalized tail residual tolerance"
        z_best = log_guess
    else:
        expand_factor = 1.5
        step = float(np.log(expand_factor))
        max_expand_steps = max(1, min(8, max_nfev))
        for k in range(1, max_expand_steps + 1):
            left_z = max(log_lower, log_guess - k * step)
            right_z = min(log_upper, log_guess + k * step)

            if left_z < log_guess:
                f_left = _tail_residual_scalar(left_z)
                bracket_evals += 1
                if np.isfinite(f_left) and np.isfinite(f_guess) and f_left * f_guess <= 0.0:
                    bracket = (left_z, log_guess)
                    bracket_found = True
                    break

            if right_z > log_guess:
                f_right = _tail_residual_scalar(right_z)
                bracket_evals += 1
                if np.isfinite(f_right) and np.isfinite(f_guess) and f_guess * f_right <= 0.0:
                    bracket = (log_guess, right_z)
                    bracket_found = True
                    break

            if left_z <= log_lower and right_z >= log_upper:
                break

    if bracket is not None:
        _diag(f"found scalar bracket in log_jB: [{bracket[0]:.6g}, {bracket[1]:.6g}]")
        sol_root = root_scalar(lambda z: _tail_residual_scalar(float(z)), bracket=bracket, method="brentq", xtol=root_tol)
        root_method = "brentq"
        root_success = bool(sol_root.converged)
        root_message = str(sol_root.flag)
        if root_success:
            z_best = float(sol_root.root)
        elif best_trial["log_jB"] is not None:
            z_best = float(best_trial["log_jB"])
    elif not root_success:
        _diag("no sign-changing bracket found; falling back to bounded least-squares in log_jB")
        sol_root = least_squares(
            lambda z: np.array([_tail_residual_scalar(float(z[0]))], dtype=float),
            np.array([log_guess], dtype=float),
            bounds=(np.array([log_lower]), np.array([log_upper])),
            method="trf",
            xtol=root_tol,
            ftol=root_tol,
            gtol=root_tol,
            max_nfev=max_nfev,
        )
        root_success = bool(sol_root.success)
        root_message = str(sol_root.message)
        if np.all(np.isfinite(sol_root.x)):
            z_best = float(sol_root.x[0])
        if best_trial["log_jB"] is not None:
            best_metric = float(best_trial["metric"])
            solver_metric = abs(_tail_residual_scalar(float(z_best)))
            if best_metric < solver_metric:
                z_best = float(best_trial["log_jB"])

    jB_best = float(np.exp(np.clip(z_best, -700.0, 700.0)))
    _diag(f"rescaled 1D shooting finished; rebuilding best state at jB={jB_best:.6g}, aQstar={aQstar:.6g}")
    result = _trial_from_log_jB(z_best, with_profile=return_profile, trial_label="best1d-rescaled")
    tail_resid = float(result["tail_residual_norm"]) if np.isfinite(result["tail_residual_norm"]) else np.inf
    tail_resid_raw = float(result["tail_residual"]) if np.isfinite(result["tail_residual"]) else np.inf
    success = bool(root_success and result["success"] and abs(tail_resid) <= root_tol)

    result["success"] = success
    if success:
        result["message"] = "Rescaled 1D steady-front shooting converged"
    else:
        if result["message"]:
            result["message"] = f"{root_message}; last trial: {result['message']}"
        else:
            result["message"] = root_message

    result["_root_success"] = root_success
    result["_root_message"] = root_message
    result["_root_method"] = root_method
    result["_root_residual"] = np.array([tail_resid], dtype=float)
    result["_root_raw_residual"] = np.array([tail_resid_raw], dtype=float)
    result["_bracket_found"] = bool(bracket_found)
    result["_bracket_evals"] = int(bracket_evals)
    result["shooting_evals"] = int(diag_state["trial_count"])
    return result

def solve_steady_front_1d_aQstar_rescale_bvp(
    T,
    nB_N,
    B_one_forth,
    aQstar,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1.0e-3,
    n_mesh=300,
    tol_bvp=1.0e-4,
    max_nodes=50000,
    jB_guess=None,
    jB_bounds=None,
    kappa_factor=1.0,
    return_profile=False,
    verb=False,
):
    """
    Solve the fixed-aQstar steady-front problem as a compact-coordinate BVP.

    This solver keeps the same hydro + diffusion + reaction equations as the
    1D IVP shooting solvers, but uses solve_bvp with jB as an unknown BVP
    parameter. The compact coordinate is s = 1 - exp(-lambda*x/kappa_factor), integrated on
    s in [0, 1 - tail_eps]. The endpoint truncation is controlled only by
    tail_eps; no compact_tail_lengths cutoff is used in this BVP path.
    """
    if nB_N <= 0.0:
        raise RuntimeError("nB_N must be positive")
    if tail_eps <= 0.0 or tail_eps >= 1.0:
        raise RuntimeError("tail_eps must satisfy 0 < tail_eps < 1")
    if int(n_mesh) < 5:
        raise RuntimeError("n_mesh must be at least 5")
    if tol_bvp <= 0.0:
        raise RuntimeError("tol_bvp must be positive")
    if max_nodes <= int(n_mesh):
        raise RuntimeError("max_nodes must be larger than n_mesh")
    if kappa_factor <= 0.0:
        raise RuntimeError("kappa_factor must be positive")

    upB = 5000
    t_start = time.perf_counter()
    if isinstance(verb, str):
        verb_mode = "full" if verb.lower() == "full" else ("simple" if verb else "off")
    else:
        verb_mode = "simple" if verb else "off"
    full_diag = verb_mode == "full"
    simple_diag = verb_mode in ("simple", "full")

    def _diag(msg):
        if full_diag:
            dt = time.perf_counter() - t_start
            print(f"[steady_front_bvp +{dt:8.2f}s] {msg}", flush=True)

    if jB_guess is None:
        jB_guess = 1.0e-6 * nB_N
    jB_guess = float(jB_guess)
    if jB_guess <= 0.0:
        raise RuntimeError("jB_guess must be positive")

    bounded_jB = jB_bounds is not None
    if bounded_jB:
        if len(jB_bounds) != 2:
            raise RuntimeError("jB_bounds must be a 2-tuple (jB_min, jB_max)")
        jB_lower_bound = float(jB_bounds[0])
        jB_upper_bound = float(jB_bounds[1])
        if jB_lower_bound <= 0.0 or jB_upper_bound <= 0.0 or jB_upper_bound <= jB_lower_bound:
            raise RuntimeError("jB_bounds must satisfy 0 < jB_min < jB_max")
        if not (jB_lower_bound < jB_guess < jB_upper_bound):
            jB_guess = min(max(jB_guess, jB_lower_bound * (1.0 + 1.0e-6)), jB_upper_bound * (1.0 - 1.0e-6))
    else:
        jB_lower_bound = 0.0
        jB_upper_bound = np.inf

    def _param_from_jB(jB):
        jB = float(jB)
        if bounded_jB:
            frac = (jB - jB_lower_bound) / (jB_upper_bound - jB_lower_bound)
            frac = float(np.clip(frac, 1.0e-12, 1.0 - 1.0e-12))
            return float(np.log(frac / (1.0 - frac)))
        return float(np.log(jB))

    def _jB_from_param(theta):
        theta = float(theta)
        if bounded_jB:
            theta_clip = float(np.clip(theta, -60.0, 60.0))
            sig = 1.0 / (1.0 + np.exp(-theta_clip))
            return float(jB_lower_bound + (jB_upper_bound - jB_lower_bound) * sig)
        return float(np.exp(np.clip(theta, -700.0, 700.0)))

    stats = {
        "bvp_ode_calls": 0,
        "bvp_bc_calls": 0,
        "q_root_calls": 0,
        "qstar_root_calls": 0,
        "local_state_calls": 0,
        "local_root_calls": 0,
        "local_fast_failures": 0,
        "profile_state_calls": 0,
        "global_state_builds": 0,
        "global_state_failures": 0,
        "local_state_failures": 0,
    }
    state_cache = {}
    last_failure = {"message": ""}
    s_end = float(1.0 - tail_eps)

    def _build_global_state(theta):
        key = round(float(theta), 12)
        if key in state_cache:
            return state_cache[key]

        jB = _jB_from_param(theta)
        stats["global_state_builds"] += 1

        # Upstream nuclear state N at x = 0^-.
        P_N = float(PNM_n(nB_N, T, param=param, NM_type=NM_type))
        e_N = float(edensNM_n(nB_N, T, param=param))
        h_N = float(P_N + e_N)
        u_N = float(jB / nB_N)
        Pi = float(h_N * u_N * u_N + P_N)

        # Far-right equilibrated quark state Q with muK = 0.
        muB_Q = _solve_muB_Q_at_muK0_for_given_Pi(Pi, jB, B_one_forth, T, ms=ms, upB=upB, stats=stats)
        nB_Q = float(nB_QM(muB_Q, 0.0, B_one_forth, T, ms=ms, upB=upB))
        if nB_Q <= 0.0:
            raise RuntimeError("Equilibrated Q state has non-positive density")
        if abs(ms) <= 1.0e-12:
            nK_Q = 0.0
        else:
            nK_Q = float(nK_QM(muB_Q, 0.0, B_one_forth, T, ms=ms, upB=upB))

        # Pure neutron matter implies nK_N = nB_N.
        a_N = float((nB_N - nK_Q) / nB_Q)

        # Interface state Qstar at x = 0+.
        muK_Qstar_seed = _branch_muK_seed(aQstar)
        muB_Qstar, muK_Qstar = _solve_interface_Qstar_from_aQstar_and_Pi(
            aQstar,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            T,
            ms=ms,
            upB=upB,
            initial_guess=(muB_Q, muK_Qstar_seed),
            stats=stats,
            stats_key="qstar_root_calls",
        )
        nB_Qstar = float(nB_QM(muB_Qstar, muK_Qstar, B_one_forth, T, ms=ms, upB=upB))
        if nB_Qstar <= 0.0:
            raise RuntimeError("Qstar state has non-positive density")

        micro = _microphysics_at_Qstar(muB_Qstar, T)
        D = float(micro["D"])
        eta = float(micro["eta"])
        gamma = float(micro["gamma"])
        tau = float(micro["tau"])

        u_Q = float(jB / nB_Q)
        disc = float(u_Q * u_Q + 4.0 * D * gamma * eta)
        if (not np.isfinite(disc)) or disc <= 0.0:
            raise RuntimeError("Tail discriminant is non-positive")
        lam = float((-u_Q + np.sqrt(disc)) / (2.0 * D))
        if (not np.isfinite(lam)) or lam <= 0.0:
            raise RuntimeError("Tail decay lambda must be positive")
        q0 = float(-a_N * u_N)
        x_end = float(-float(kappa_factor) * np.log1p(-s_end) / lam)
        tail_coeff = float(D * lam + u_Q)
        state = {
            "jB": jB,
            "P_N": P_N,
            "e_N": e_N,
            "h_N": h_N,
            "u_N": u_N,
            "Pi": Pi,
            "muB_Q": float(muB_Q),
            "nB_Q": nB_Q,
            "nK_Q": float(nK_Q),
            "a_N": a_N,
            "muB_Qstar": float(muB_Qstar),
            "muK_Qstar": float(muK_Qstar),
            "nB_Qstar": nB_Qstar,
            "D": D,
            "eta": eta,
            "gamma": gamma,
            "tau": tau,
            "lambda": lam,
            "u_Q": u_Q,
            "q0": q0,
            "tail_coeff": tail_coeff,
            "x_end": x_end,
        }
        state_cache[key] = state
        return state

    def _state_or_none(theta):
        try:
            return _build_global_state(theta)
        except Exception as exc:
            stats["global_state_failures"] += 1
            last_failure["message"] = str(exc)
            return None

    def _ode(s, y, p):
        stats["bvp_ode_calls"] += 1
        state = _state_or_none(float(p[0]))
        if state is None:
            return np.zeros_like(y) + 1.0e12

        dyds = np.empty_like(y)
        guess = (state["muB_Qstar"], state["muK_Qstar"])
        for i in range(y.shape[1]):
            a_val = float(y[0, i])
            q_val = float(y[1, i])
            if (not np.isfinite(a_val)) or (not np.isfinite(q_val)):
                stats["local_state_failures"] += 1
                dyds[:, i] = 1.0e12
                continue
            try:
                muB_loc, muK_loc, nB_loc, u_loc = _solve_local_quark_state_from_a_and_Pi(
                    a_val,
                    state["Pi"],
                    state["jB"],
                    state["nB_Q"],
                    state["nK_Q"],
                    B_one_forth,
                    T,
                    ms=ms,
                    upB=upB,
                    initial_guess=guess,
                        stats=stats,
                )
                guess = (muB_loc, muK_loc)
                one_minus_s = max(1.0 - float(s[i]), np.finfo(float).tiny)
                dx_ds = float(kappa_factor) / (state["lambda"] * one_minus_s)
                dyds[0, i] = ((q_val + u_loc * a_val) / state["D"]) * dx_ds
                dyds[1, i] = (state["gamma"] * (a_val**3 + state["eta"] * a_val)) * dx_ds
            except Exception as exc:
                stats["local_state_failures"] += 1
                last_failure["message"] = str(exc)
                dyds[:, i] = 1.0e12
        return dyds

    def _bc(ya, yb, p):
        stats["bvp_bc_calls"] += 1
        state = _state_or_none(float(p[0]))
        if state is None:
            return np.array([1.0e12, 1.0e12, 1.0e12], dtype=float)
        return np.array(
            [
                ya[0] - float(aQstar),
                ya[1] - state["q0"],
                yb[1] + state["tail_coeff"] * yb[0],
            ],
            dtype=float,
        )

    theta_guess = _param_from_jB(jB_guess)
    state0 = _build_global_state(theta_guess)
    s_mesh = np.linspace(0.0, s_end, int(n_mesh))
    tail_shape = np.maximum(1.0 - s_mesh, tail_eps) ** max(float(kappa_factor), 1.0e-12)
    a_guess = float(aQstar) * tail_shape
    q_tail_guess = -state0["tail_coeff"] * a_guess
    blend = s_mesh / max(s_end, np.finfo(float).tiny)
    q_guess = (1.0 - blend) * state0["q0"] + blend * q_tail_guess
    y_guess = np.vstack((a_guess, q_guess))

    _diag(
        f"starting compact BVP with jB_guess={jB_guess:.6g}, aQstar={aQstar:.6g}, "
        f"tail_eps={tail_eps:.3g}, branch=muK-rich"
    )

    try:
        sol = solve_bvp(
            _ode,
            _bc,
            s_mesh,
            y_guess,
            p=np.array([theta_guess], dtype=float),
            tol=tol_bvp,
            max_nodes=max_nodes,
            verbose=2 if full_diag else 0,
        )
    except Exception as exc:
        return {
            "success": False,
            "message": f"solve_bvp raised: {exc}; last failure: {last_failure['message']}",
            "aQstar": float(aQstar),
            "jB": np.nan,
            "branch_label": "muK-rich",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "bvp_status": -999,
            **{k: int(v) for k, v in stats.items()},
        }

    theta_sol = float(sol.p[0])
    state = _state_or_none(theta_sol)
    if state is None:
        return {
            "success": False,
            "message": f"BVP final state construction failed: {last_failure['message']}",
            "aQstar": float(aQstar),
            "jB": np.nan,
            "branch_label": "muK-rich",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "bvp_status": int(sol.status),
            **{k: int(v) for k, v in stats.items()},
        }

    a_end = float(sol.y[0, -1])
    q_end = float(sol.y[1, -1])
    tail_drive = float(state["tail_coeff"] * a_end)
    tail_residual = float(q_end + tail_drive)
    tail_scale = float(max(abs(q_end), abs(tail_drive), np.finfo(float).tiny))
    tail_residual_norm = float(tail_residual / tail_scale)

    success = bool(sol.success and np.isfinite(tail_residual_norm) and abs(tail_residual_norm) <= max(tol_bvp, 10.0 * np.finfo(float).eps))
    result = {
        "success": success,
        "message": "Compact BVP steady-front solve converged" if success else f"{sol.message}; last failure: {last_failure['message']}",
        "jB": float(state["jB"]),
        "aQstar": float(aQstar),
        "branch_label": "muK-rich",
        "coordinate": "BVP: s in [0, 1-tail_eps], s=1-exp(-lambda*x/kappa_factor)",
        "tail_eps": float(tail_eps),
        "kappa_factor": float(kappa_factor),
        "u_N": float(state["u_N"]),
        "u_Q": float(state["u_Q"]),
        "a_N": float(state["a_N"]),
        "Pi": float(state["Pi"]),
        "muB_Qstar": float(state["muB_Qstar"]),
        "muK_Qstar": float(state["muK_Qstar"]),
        "nB_Qstar": float(state["nB_Qstar"]),
        "muB_Q": float(state["muB_Q"]),
        "nB_Q": float(state["nB_Q"]),
        "nK_Q": float(state["nK_Q"]),
        "D": float(state["D"]),
        "eta": float(state["eta"]),
        "gamma": float(state["gamma"]),
        "tau": float(state["tau"]),
        "lambda": float(state["lambda"]),
        "kappa": float(kappa_factor / state["lambda"]),
        "s_end": float(s_end),
        "x_end": float(state["x_end"]),
        "a_end": a_end,
        "q_end": q_end,
        "tail_residual": tail_residual,
        "tail_residual_norm": tail_residual_norm,
        "tail_scale": tail_scale,
        "_residual": np.array([tail_residual_norm], dtype=float),
        "_root_method": "solve_bvp_parameter",
        "bvp_status": int(sol.status),
        "bvp_message": str(sol.message),
        "bvp_niter": int(getattr(sol, "niter", -1)),
        "bvp_nodes": int(sol.x.size),
        **{k: int(v) for k, v in stats.items()},
    }

    if return_profile:
        s_prof = np.asarray(sol.x, dtype=float)
        a_prof = np.asarray(sol.y[0], dtype=float)
        q_prof = np.asarray(sol.y[1], dtype=float)
        x_prof = -float(kappa_factor) * np.log1p(-s_prof) / float(state["lambda"])
        muB_prof = np.empty_like(s_prof)
        muK_prof = np.empty_like(s_prof)
        nB_prof = np.empty_like(s_prof)
        u_prof = np.empty_like(s_prof)
        guess = (state["muB_Qstar"], state["muK_Qstar"])
        for i, a_val in enumerate(a_prof):
            stats["profile_state_calls"] += 1
            muB_loc, muK_loc, nB_loc, u_loc = _solve_local_quark_state_from_a_and_Pi(
                float(a_val),
                state["Pi"],
                state["jB"],
                state["nB_Q"],
                state["nK_Q"],
                B_one_forth,
                T,
                ms=ms,
                upB=upB,
                initial_guess=guess,
                stats=stats,
            )
            guess = (muB_loc, muK_loc)
            muB_prof[i] = muB_loc
            muK_prof[i] = muK_loc
            nB_prof[i] = nB_loc
            u_prof[i] = u_loc
        closure_prof = np.abs(
            np.array(
                [
                    (nK_QM(float(muB_prof[i]), float(muK_prof[i]), B_one_forth, T, ms=ms, upB=upB) - state["nK_Q"])
                    / state["nB_Q"]
                    - a_prof[i]
                    for i in range(len(a_prof))
                ],
                dtype=float,
            )
        )
        closure_error_max = float(np.max(closure_prof)) if len(closure_prof) else 0.0
        result.update(
            {
                "s": s_prof,
                "x": x_prof,
                "a": a_prof,
                "q": q_prof,
                "u": u_prof,
                "nB": nB_prof,
                "muB": muB_prof,
                "muK": muK_prof,
                "closure_error": closure_prof,
                "closure_error_max": closure_error_max,
                "profile_state_calls": int(stats["profile_state_calls"]),
                "local_state_calls": int(stats["local_state_calls"]),
                "local_root_calls": int(stats["local_root_calls"]),
                "local_fast_failures": int(stats["local_fast_failures"]),
            }
        )

    if simple_diag:
        print(
            f"bvp jB={result['jB']:.6g}, aQstar={aQstar:.6g}, "
            f"tail_norm={tail_residual_norm:.6g}, status={sol.status}, success={success}"
        )
    return result


def _solve_steady_front_entropy_once(
    T,
    nB_N,
    B_one_forth,
    aQstar,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    jB_bounds=None,
    return_profile=False,
    verb=False,
    profile_guess=None,
    seed_profile=False,
):
    """
    Solve the entropy-enabled steady-front problem as a one-shot compact-coordinate BVP.

    The ODE unknowns are (a, q, w) with
        w = s * u
    and the local thermodynamic closure is solved in (muB, muK, logT).
    """
    T = float(T)
    if (not np.isfinite(T)) or T <= 0.0:
        raise RuntimeError("solve_steady_front_entropy requires T > 0")
    if nB_N <= 0.0:
        raise RuntimeError("nB_N must be positive")
    if tail_eps <= 0.0 or tail_eps >= 1.0:
        raise RuntimeError("tail_eps must satisfy 0 < tail_eps < 1")
    if int(n_mesh) < 5:
        raise RuntimeError("n_mesh must be at least 5")
    if tol_bvp <= 0.0:
        raise RuntimeError("tol_bvp must be positive")
    if max_nodes <= int(n_mesh):
        raise RuntimeError("max_nodes must be larger than n_mesh")

    upB = 5000
    t_start = time.perf_counter()
    if isinstance(verb, str):
        verb_mode = "full" if verb.lower() == "full" else ("simple" if verb else "off")
    else:
        verb_mode = "simple" if verb else "off"
    full_diag = verb_mode == "full"
    simple_diag = verb_mode in ("simple", "full")

    def _diag(msg):
        if full_diag:
            dt = time.perf_counter() - t_start
            print(f"[steady_front_entropy +{dt:8.2f}s] {msg}", flush=True)

    if jB_guess is None:
        # The entropy-enabled BVP is much more robust when started near the
        # physically relevant low-flux branch rather than the much larger
        # historical shooting guess used by older solvers.
        jB_guess = 1.0e-8 * nB_N
    jB_guess = float(jB_guess)
    if jB_guess <= 0.0:
        raise RuntimeError("jB_guess must be positive")

    bounded_jB = jB_bounds is not None
    if bounded_jB:
        if len(jB_bounds) != 2:
            raise RuntimeError("jB_bounds must be a 2-tuple (jB_min, jB_max)")
        jB_lower_bound = float(jB_bounds[0])
        jB_upper_bound = float(jB_bounds[1])
        if jB_lower_bound <= 0.0 or jB_upper_bound <= 0.0 or jB_upper_bound <= jB_lower_bound:
            raise RuntimeError("jB_bounds must satisfy 0 < jB_min < jB_max")
        if not (jB_lower_bound < jB_guess < jB_upper_bound):
            jB_guess = min(max(jB_guess, jB_lower_bound * (1.0 + 1.0e-6)), jB_upper_bound * (1.0 - 1.0e-6))
    else:
        jB_lower_bound = 0.0
        jB_upper_bound = np.inf

    def _param_from_jB(jB):
        jB = float(jB)
        if bounded_jB:
            frac = (jB - jB_lower_bound) / (jB_upper_bound - jB_lower_bound)
            frac = float(np.clip(frac, 1.0e-12, 1.0 - 1.0e-12))
            return float(np.log(frac / (1.0 - frac)))
        return float(np.log(jB))

    def _jB_from_param(theta):
        theta = float(theta)
        if bounded_jB:
            theta_clip = float(np.clip(theta, -60.0, 60.0))
            sig = 1.0 / (1.0 + np.exp(-theta_clip))
            return float(jB_lower_bound + (jB_upper_bound - jB_lower_bound) * sig)
        return float(np.exp(np.clip(theta, -700.0, 700.0)))

    stats = {
        "bvp_ode_calls": 0,
        "bvp_bc_calls": 0,
        "q_root_calls": 0,
        "qstar_root_calls": 0,
        "local_state_calls": 0,
        "local_root_calls": 0,
        "local_fast_failures": 0,
        "profile_state_calls": 0,
        "global_state_builds": 0,
        "global_state_failures": 0,
        "local_state_failures": 0,
    }
    state_cache = {}
    muB_Q_last_guess = {"value": None}
    qstar_last_guess = {"value": None}
    last_failure = {"message": ""}
    s_coord_end = float(1.0 - tail_eps)

    def _build_global_state(theta):
        key = round(float(theta), 12)
        if key in state_cache:
            return state_cache[key]

        jB = _jB_from_param(theta)
        stats["global_state_builds"] += 1

        # Upstream nuclear state N at x = 0^-.
        P_N = float(PNM_n(nB_N, T, param=param, NM_type=NM_type))
        e_N = float(edensNM_n(nB_N, T, param=param))
        h_N = float(P_N + e_N)
        u_N = float(jB / nB_N)
        Pi = float(h_N * u_N * u_N + P_N)

        # Far-right equilibrated quark state Q with muK = 0.
        muB_Q = _solve_muB_Q_at_muK0_for_given_Pi_ms(Pi, jB, B_one_forth, T, ms=ms, upB=upB, stats=stats)
        thermo_Q = _quark_thermo_state(muB_Q, 0.0, B_one_forth, T, jB, ms=ms, upB=upB)
        nB_Q = float(thermo_Q["nB"])
        if abs(ms) <= 1.0e-12:
            nK_Q = 0.0
        else:
            nK_Q = float(thermo_Q["nK"])

        # Pure neutron matter implies nK_N = nB_N.
        a_N = float((nB_N - nK_Q) / nB_Q)

        # Interface state Qstar at x = 0+ with fixed interface temperature T.
        muK_Qstar_seed = _branch_muK_seed(aQstar)
        muB_Qstar, muK_Qstar = _solve_interface_Qstar_from_aQstar_and_Pi(
            aQstar,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            T,
            ms=ms,
            upB=upB,
            initial_guess=(muB_Q, muK_Qstar_seed),
            stats=stats,
            stats_key="qstar_root_calls",
        )
        thermo_Qstar = _quark_thermo_state(muB_Qstar, muK_Qstar, B_one_forth, T, jB, ms=ms, upB=upB)
        if thermo_Qstar["s"] <= 0.0:
            raise RuntimeError("Interface Qstar state has non-positive entropy density")
        if thermo_Qstar["w"] <= 0.0:
            raise RuntimeError("Interface Qstar state has non-positive entropy flux")

        micro_Q = _microphysics_from_quark_state(muB_Q, T)
        D_Q = float(micro_Q["D"])
        eta_Q = float(micro_Q["eta"])
        gamma_Q = float(micro_Q["gamma"])
        tau_Q = float(micro_Q["tau"])

        u_Q = float(thermo_Q["u"])
        disc = float(u_Q * u_Q + 4.0 * D_Q * gamma_Q * eta_Q)
        if (not np.isfinite(disc)) or disc <= 0.0:
            raise RuntimeError("Entropy-solver tail discriminant is non-positive")
        lam = float((-u_Q + np.sqrt(disc)) / (2.0 * D_Q))
        if (not np.isfinite(lam)) or lam <= 0.0:
            raise RuntimeError("Entropy-solver tail decay lambda must be positive")
        q0 = float(-a_N * u_N)
        w0 = float(thermo_Qstar["w"])
        x_end = float(-np.log1p(-s_coord_end) / lam)
        tail_coeff_Q = float(D_Q * lam + u_Q)
        state = {
            "jB": float(jB),
            "P_N": P_N,
            "e_N": e_N,
            "h_N": h_N,
            "u_N": u_N,
            "Pi": Pi,
            "muB_Q": float(muB_Q),
            "nB_Q": float(nB_Q),
            "nK_Q": float(nK_Q),
            "a_N": float(a_N),
            "muB_Qstar": float(muB_Qstar),
            "muK_Qstar": float(muK_Qstar),
            "nB_Qstar": float(thermo_Qstar["nB"]),
            "s_Q": float(thermo_Q["s"]),
            "w_Q": float(thermo_Q["w"]),
            "s_Qstar": float(thermo_Qstar["s"]),
            "u_Qstar": float(thermo_Qstar["u"]),
            "T_Qstar": float(T),
            "D_Q": D_Q,
            "eta_Q": eta_Q,
            "gamma_Q": gamma_Q,
            "tau_Q": tau_Q,
            "lambda": float(lam),
            "u_Q": float(u_Q),
            "q0": float(q0),
            "w0": float(w0),
            "tail_coeff_Q": float(tail_coeff_Q),
            "x_end": float(x_end),
        }
        state_cache[key] = state
        return state

    def _state_or_none(theta):
        try:
            return _build_global_state(theta)
        except Exception as exc:
            stats["global_state_failures"] += 1
            last_failure["message"] = str(exc)
            return None

    def _ode(s_coord, y, p):
        stats["bvp_ode_calls"] += 1
        state = _state_or_none(float(p[0]))
        if state is None:
            return np.zeros_like(y) + 1.0e12

        dyds = np.empty_like(y)
        guess = (state["muB_Qstar"], state["muK_Qstar"], state["T_Qstar"])
        for i in range(y.shape[1]):
            a_val = float(y[0, i])
            q_val = float(y[1, i])
            w_val = float(y[2, i])
            if (not np.isfinite(a_val)) or (not np.isfinite(q_val)) or (not np.isfinite(w_val)):
                stats["local_state_failures"] += 1
                dyds[:, i] = 1.0e12
                continue
            try:
                thermo_loc = _solve_local_quark_state_from_a_w_and_Pi(
                    a_val,
                    w_val,
                    state["Pi"],
                    state["jB"],
                    state["nB_Q"],
                    state["nK_Q"],
                    B_one_forth,
                    ms=ms,
                    upB=upB,
                    initial_guess=guess,
                    T_ref=state["T_Qstar"],
                    stats=stats,
                )
                guess = (thermo_loc["muB"], thermo_loc["muK"], thermo_loc["T"])
                micro_loc = _microphysics_from_quark_state(thermo_loc["muB"], thermo_loc["T"])
                reaction = float(micro_loc["gamma"] * (a_val**3 + micro_loc["eta"] * a_val))
                one_minus_s = max(1.0 - float(s_coord[i]), np.finfo(float).tiny)
                dx_ds = 1.0 / (state["lambda"] * one_minus_s)
                dyds[0, i] = ((q_val + thermo_loc["u"] * a_val) / micro_loc["D"]) * dx_ds
                dyds[1, i] = reaction * dx_ds
                dyds[2, i] = ((state["nB_Q"] / thermo_loc["T"]) * thermo_loc["muK"] * reaction) * dx_ds
            except Exception as exc:
                stats["local_state_failures"] += 1
                last_failure["message"] = str(exc)
                dyds[:, i] = 1.0e12
        return dyds

    def _bc(ya, yb, p):
        stats["bvp_bc_calls"] += 1
        state = _state_or_none(float(p[0]))
        if state is None:
            return np.array([1.0e12, 1.0e12, 1.0e12, 1.0e12], dtype=float)
        return np.array(
            [
                ya[0] - float(aQstar),
                ya[1] - state["q0"],
                ya[2] - state["w0"],
                yb[1] + state["tail_coeff_Q"] * yb[0],
            ],
            dtype=float,
        )

    theta_guess = _param_from_jB(jB_guess)
    state0 = _state_or_none(theta_guess)
    if state0 is None:
        return {
            "success": False,
            "message": f"Initial entropy-solver state construction failed: {last_failure['message']}",
            "aQstar": float(aQstar),
            "jB": np.nan,
            "branch_label": "muK-rich",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "w_end": np.nan,
            "bvp_status": -999,
            "bvp_message": "initial_state_failure",
            "bvp_niter": -1,
            "bvp_nodes": 0,
            **{k: int(v) for k, v in stats.items()},
        }

    s_coord_mesh = np.linspace(0.0, s_coord_end, int(n_mesh))
    blend = s_coord_mesh / max(s_coord_end, np.finfo(float).tiny)
    tail_shape = np.maximum(1.0 - s_coord_mesh, tail_eps)

    def _default_initial_guess():
        a_guess_loc = float(aQstar) * tail_shape
        q_tail_guess_loc = -state0["tail_coeff_Q"] * a_guess_loc
        q_guess_loc = (1.0 - blend) * state0["q0"] + blend * q_tail_guess_loc
        w_guess_loc = (1.0 - blend) * state0["w0"] + blend * state0["w_Q"]
        return a_guess_loc, q_guess_loc, w_guess_loc

    def _profile_initial_guess(profile):
        try:
            s_prev = np.asarray(profile["s_coord"], dtype=float)
            a_prev = np.asarray(profile["a"], dtype=float)
            q_prev = np.asarray(profile["q"], dtype=float)
            w_prev = np.asarray(profile["w"], dtype=float)
        except Exception:
            return _default_initial_guess()

        if (
            s_prev.ndim != 1
            or a_prev.shape != s_prev.shape
            or q_prev.shape != s_prev.shape
            or w_prev.shape != s_prev.shape
            or s_prev.size < 5
            or not np.all(np.isfinite(s_prev))
            or not np.all(np.isfinite(a_prev))
            or not np.all(np.isfinite(q_prev))
            or not np.all(np.isfinite(w_prev))
        ):
            return _default_initial_guess()

        a_prev0 = float(a_prev[0])
        if abs(a_prev0) > 1.0e-12:
            a_scale = float(aQstar) / a_prev0
        else:
            a_scale = 1.0

        a_guess_loc = np.interp(s_coord_mesh, s_prev, a_prev) * a_scale
        a_guess_loc[0] = float(aQstar)

        q_base = np.interp(s_coord_mesh, s_prev, q_prev) * a_scale
        q_base0 = float(q_base[0])
        q_base_end = float(q_base[-1])
        q_target0 = float(state0["q0"])
        q_target_end = float(-state0["tail_coeff_Q"] * a_guess_loc[-1])
        q_span = q_base_end - q_base0
        if np.isfinite(q_span) and abs(q_span) > 1.0e-12 * max(1.0, abs(q_base0), abs(q_base_end)):
            q_frac = (q_base - q_base0) / q_span
            q_guess_loc = q_target0 + q_frac * (q_target_end - q_target0)
        else:
            q_guess_loc = (1.0 - blend) * q_target0 + blend * q_target_end
        q_guess_loc[0] = q_target0

        w_base = np.interp(s_coord_mesh, s_prev, w_prev)
        w_base0 = float(w_base[0])
        w_base_end = float(w_base[-1])
        w_span = w_base_end - w_base0
        if (
            np.all(np.isfinite(w_base))
            and w_base0 > 0.0
            and w_base_end > 0.0
            and abs(w_span) > 1.0e-12 * max(1.0, abs(w_base0), abs(w_base_end))
        ):
            w_frac = (w_base - w_base0) / w_span
            w_guess_loc = state0["w0"] + w_frac * (state0["w_Q"] - state0["w0"])
        else:
            w_guess_loc = (1.0 - blend) * state0["w0"] + blend * state0["w_Q"]

        if (not np.all(np.isfinite(a_guess_loc))) or (not np.all(np.isfinite(q_guess_loc))) or (not np.all(np.isfinite(w_guess_loc))) or np.any(w_guess_loc <= 0.0):
            return _default_initial_guess()

        w_guess_loc[0] = float(state0["w0"])
        return a_guess_loc, q_guess_loc, w_guess_loc

    if profile_guess is None:
        a_guess, q_guess, w_guess = _default_initial_guess()
    else:
        a_guess, q_guess, w_guess = _profile_initial_guess(profile_guess)
    y_guess = np.vstack((a_guess, q_guess, w_guess))

    _diag(
        f"starting entropy-enabled compact BVP with jB_guess={jB_guess:.6g}, "
        f"aQstar={aQstar:.6g}, tail_eps={tail_eps:.3g}, branch=muK-rich"
    )

    try:
        sol = solve_bvp(
            _ode,
            _bc,
            s_coord_mesh,
            y_guess,
            p=np.array([theta_guess], dtype=float),
            tol=tol_bvp,
            max_nodes=max_nodes,
            verbose=2 if full_diag else 0,
        )
    except Exception as exc:
        return {
            "success": False,
            "message": f"solve_bvp raised: {exc}; last failure: {last_failure['message']}",
            "aQstar": float(aQstar),
            "jB": np.nan,
            "branch_label": "muK-rich",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "w_end": np.nan,
            "bvp_status": -999,
            "bvp_message": f"solve_bvp raised: {exc}",
            "bvp_niter": -1,
            "bvp_nodes": 0,
            **{k: int(v) for k, v in stats.items()},
        }

    theta_sol = float(sol.p[0])
    state = _state_or_none(theta_sol)
    if state is None:
        return {
            "success": False,
            "message": f"BVP final state construction failed: {last_failure['message']}",
            "aQstar": float(aQstar),
            "jB": np.nan,
            "branch_label": "muK-rich",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "w_end": np.nan,
            "bvp_status": int(sol.status),
            "bvp_message": str(sol.message),
            "bvp_niter": int(getattr(sol, "niter", -1)),
            "bvp_nodes": int(sol.x.size),
            **{k: int(v) for k, v in stats.items()},
        }

    a_end = float(sol.y[0, -1])
    q_end = float(sol.y[1, -1])
    w_end = float(sol.y[2, -1])
    tail_drive = float(state["tail_coeff_Q"] * a_end)
    tail_residual = float(q_end + tail_drive)
    tail_scale = float(max(abs(q_end), abs(tail_drive), np.finfo(float).tiny))
    tail_residual_norm = float(tail_residual / tail_scale)

    success = bool(
        sol.success
        and np.isfinite(tail_residual_norm)
        and abs(tail_residual_norm) <= max(tol_bvp, 10.0 * np.finfo(float).eps)
    )
    result = {
        "success": success,
        "message": "Entropy-enabled compact BVP steady-front solve converged" if success else f"{sol.message}; last failure: {last_failure['message']}",
        "jB": float(state["jB"]),
        "aQstar": float(aQstar),
        "branch_label": "muK-rich",
        "coordinate": "BVP: s_coord in [0, 1-tail_eps], s_coord=1-exp(-lambda*x)",
        "tail_eps": float(tail_eps),
        "u_N": float(state["u_N"]),
        "u_Q": float(state["u_Q"]),
        "a_N": float(state["a_N"]),
        "Pi": float(state["Pi"]),
        "muB_Q": float(state["muB_Q"]),
        "nB_Q": float(state["nB_Q"]),
        "nK_Q": float(state["nK_Q"]),
        "muB_Qstar": float(state["muB_Qstar"]),
        "muK_Qstar": float(state["muK_Qstar"]),
        "nB_Qstar": float(state["nB_Qstar"]),
        "s_Qstar": float(state["s_Qstar"]),
        "D_Q": float(state["D_Q"]),
        "eta_Q": float(state["eta_Q"]),
        "gamma_Q": float(state["gamma_Q"]),
        "tau_Q": float(state["tau_Q"]),
        "lambda": float(state["lambda"]),
        "kappa": float(1.0 / state["lambda"]),
        "s_end": float(s_coord_end),
        "x_end": float(state["x_end"]),
        "a_end": a_end,
        "q_end": q_end,
        "w_end": w_end,
        "tail_residual": tail_residual,
        "tail_residual_norm": tail_residual_norm,
        "tail_scale": tail_scale,
        "_residual": np.array([tail_residual_norm], dtype=float),
        "_root_method": "solve_bvp_parameter",
        "bvp_status": int(sol.status),
        "bvp_message": str(sol.message),
        "bvp_niter": int(getattr(sol, "niter", -1)),
        "bvp_nodes": int(sol.x.size),
        **{k: int(v) for k, v in stats.items()},
    }

    if return_profile or seed_profile:
        s_coord_prof = np.asarray(sol.x, dtype=float)
        a_prof = np.asarray(sol.y[0], dtype=float)
        q_prof = np.asarray(sol.y[1], dtype=float)
        w_prof = np.asarray(sol.y[2], dtype=float)
        x_prof = -np.log1p(-s_coord_prof) / float(state["lambda"])
        result.update(
            {
                "s_coord": s_coord_prof,
                "x": x_prof,
                "a": a_prof,
                "q": q_prof,
                "w": w_prof,
            }
        )
    if return_profile:
        muB_prof = np.empty_like(s_coord_prof)
        muK_prof = np.empty_like(s_coord_prof)
        T_prof = np.empty_like(s_coord_prof)
        nB_prof = np.empty_like(s_coord_prof)
        u_prof = np.empty_like(s_coord_prof)
        entropy_prof = np.empty_like(s_coord_prof)
        D_prof = np.empty_like(s_coord_prof)
        eta_prof = np.empty_like(s_coord_prof)
        gamma_prof = np.empty_like(s_coord_prof)
        guess = (state["muB_Qstar"], state["muK_Qstar"], state["T_Qstar"])
        for i, a_val in enumerate(a_prof):
            stats["profile_state_calls"] += 1
            thermo_loc = _solve_local_quark_state_from_a_w_and_Pi(
                float(a_val),
                float(w_prof[i]),
                state["Pi"],
                state["jB"],
                state["nB_Q"],
                state["nK_Q"],
                B_one_forth,
                ms=ms,
                upB=upB,
                initial_guess=guess,
                T_ref=state["T_Qstar"],
                stats=stats,
            )
            guess = (thermo_loc["muB"], thermo_loc["muK"], thermo_loc["T"])
            micro_loc = _microphysics_from_quark_state(thermo_loc["muB"], thermo_loc["T"])
            muB_prof[i] = thermo_loc["muB"]
            muK_prof[i] = thermo_loc["muK"]
            T_prof[i] = thermo_loc["T"]
            nB_prof[i] = thermo_loc["nB"]
            u_prof[i] = thermo_loc["u"]
            entropy_prof[i] = thermo_loc["s"]
            D_prof[i] = micro_loc["D"]
            eta_prof[i] = micro_loc["eta"]
            gamma_prof[i] = micro_loc["gamma"]

        result.update(
            {
                "u": u_prof,
                "nB": nB_prof,
                "muB": muB_prof,
                "muK": muK_prof,
                "T_profile": T_prof,
                "s_profile": entropy_prof,
                "D_profile": D_prof,
                "eta_profile": eta_prof,
                "gamma_profile": gamma_prof,
                "profile_state_calls": int(stats["profile_state_calls"]),
                "local_state_calls": int(stats["local_state_calls"]),
                "local_root_calls": int(stats["local_root_calls"]),
                "local_fast_failures": int(stats["local_fast_failures"]),
            }
        )

    if simple_diag:
        print(
            f"entropy-bvp jB={result['jB']:.6g}, aQstar={aQstar:.6g}, "
            f"tail_norm={tail_residual_norm:.6g}, status={sol.status}, success={success}"
        )
    return result


def _strip_entropy_profile_fields(result):
    """
    Return a shallow copy of an entropy-solver result without profile arrays.
    """
    result_out = dict(result)
    for key in (
        "s_coord",
        "x",
        "a",
        "q",
        "w",
        "u",
        "nB",
        "muB",
        "muK",
        "T_profile",
        "s_profile",
        "D_profile",
        "eta_profile",
        "gamma_profile",
    ):
        result_out.pop(key, None)
    return result_out


def _seed_entropy_jB_guess(
    T,
    nB_N,
    B_one_forth,
    aQstar_seed,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    verb=False,
):
    """
    Build a jB seed for the entropy-enabled solver by reusing the
    current production compact-coordinate BVP on a small-aQstar seed problem.
    """
    heuristic_guess = float(max(1.0e-12, 1.0e-8 * float(nB_N)))
    seed_tol = float(max(1.0e-3, tol_bvp))
    seed_mesh = int(min(max(int(n_mesh), 60), 120))
    seed_nodes = int(min(max(int(max_nodes), 500), 2000))
    guess_ladder = []
    for factor in (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 5.0e-2):
        guess_ladder.append(float(max(1.0e-12, factor * float(nB_N))))
    guess_ladder.append(float(max(heuristic_guess, 1.0e-3)))

    tried = set()
    for guess in guess_ladder:
        guess_key = round(float(guess), 16)
        if guess_key in tried:
            continue
        tried.add(guess_key)
        try:
            res_seed = solve_steady_front_1d_aQstar_rescale_bvp(
                T,
                nB_N,
                B_one_forth,
                aQstar_seed,
                ms=ms,
                param=param,
                NM_type=NM_type,
                tail_eps=max(float(tail_eps), 1.0e-6),
                n_mesh=seed_mesh,
                tol_bvp=seed_tol,
                max_nodes=seed_nodes,
                jB_guess=guess,
                return_profile=False,
                verb=verb,
            )
            jB_seed = float(res_seed.get("jB", np.nan))
            if bool(res_seed.get("success")) and np.isfinite(jB_seed) and jB_seed > 0.0:
                return jB_seed
        except Exception:
            continue
    return heuristic_guess


def _seed_entropy_jB_guess_basic(
    T,
    nB_N,
    B_one_forth,
    aQstar_seed,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    verb=False,
):
    """
    Legacy-style single-try jB seed for the plain entropy-TQguess solver.

    This preserves the older behavior: try one compact-BVP seed solve with a
    tiny heuristic initial flux, and if that fails, fall straight back to the
    heuristic itself.
    """
    heuristic_guess = float(max(1.0e-12, 1.0e-8 * float(nB_N)))
    seed_tol = float(max(1.0e-3, tol_bvp))
    seed_mesh = int(min(max(int(n_mesh), 60), 120))
    seed_nodes = int(min(max(int(max_nodes), 500), 2000))
    try:
        res_seed = solve_steady_front_1d_aQstar_rescale_bvp(
            T,
            nB_N,
            B_one_forth,
            aQstar_seed,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=max(float(tail_eps), 1.0e-6),
            n_mesh=seed_mesh,
            tol_bvp=seed_tol,
            max_nodes=seed_nodes,
            jB_guess=heuristic_guess,
            return_profile=False,
            verb=verb,
        )
        jB_seed = float(res_seed.get("jB", np.nan))
        if bool(res_seed.get("success")) and np.isfinite(jB_seed) and jB_seed > 0.0:
            return jB_seed
    except Exception:
        pass
    return heuristic_guess


def solve_steady_front_entropy(
    T,
    nB_N,
    B_one_forth,
    aQstar,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    jB_bounds=None,
    return_profile=False,
    verb=False,
):
    """
    Solve the entropy-enabled steady-front problem as a compact-coordinate BVP.

    The solver first attempts a direct one-shot solve. If that fails, it
    automatically falls back to continuation in aQstar, reusing the last
    converged entropy profile as the next BVP initial guess.
    """
    aQstar = float(aQstar)
    abs_target = abs(aQstar)
    sign = -1.0 if aQstar < 0.0 else 1.0

    if isinstance(verb, str):
        verb_mode = "full" if verb.lower() == "full" else ("simple" if verb else "off")
    else:
        verb_mode = "simple" if verb else "off"
    full_diag = verb_mode == "full"
    simple_diag = verb_mode in ("simple", "full")

    direct_jB_guess = jB_guess
    if direct_jB_guess is None:
        seed_abs = min(abs_target, 1.0e-2) if abs_target > 0.0 else 1.0e-2
        direct_jB_guess = _seed_entropy_jB_guess(
            T,
            nB_N,
            B_one_forth,
            sign * seed_abs,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=n_mesh,
            tol_bvp=tol_bvp,
            max_nodes=max_nodes,
            verb=False,
        )

    direct_result = _solve_steady_front_entropy_once(
        T,
        nB_N,
        B_one_forth,
        aQstar,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=n_mesh,
        tol_bvp=tol_bvp,
        max_nodes=max_nodes,
        jB_guess=direct_jB_guess,
        jB_bounds=jB_bounds,
        return_profile=return_profile,
        verb=verb,
        profile_guess=None,
    )
    if bool(direct_result.get("success")) or abs_target <= 1.0e-2:
        result_out = dict(direct_result)
        result_out["continuation_used"] = False
        result_out["continuation_steps"] = 0
        if return_profile:
            return result_out
        return _strip_entropy_profile_fields(result_out)

    if simple_diag:
        print(
            f"entropy-bvp continuation: direct solve failed at aQstar={aQstar:.6g}; "
            f"retrying with staged continuation"
        )

    stage_n_mesh = int(min(max(max(int(n_mesh) // 2, 25), 25), 40))
    stage_tol = float(max(float(tol_bvp), 2.0e-2))
    stage_max_nodes = int(max(stage_n_mesh + 5, min(int(max_nodes), 250)))
    base_step_abs = 1.0e-3
    min_step_abs = 1.0e-4
    step_abs = float(min(base_step_abs, max(abs_target - min(abs_target, 1.0e-2), 0.0)))
    current_abs = min(abs_target, 1.0e-2)
    current = _solve_steady_front_entropy_once(
        T,
        nB_N,
        B_one_forth,
        sign * current_abs,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=stage_n_mesh,
        tol_bvp=stage_tol,
        max_nodes=stage_max_nodes,
        jB_guess=direct_jB_guess,
        jB_bounds=jB_bounds,
        return_profile=False,
        verb=False,
        profile_guess=None,
        seed_profile=True,
    )
    if not bool(current.get("success")):
        failed = dict(current)
        failed["message"] = (
            f"{direct_result['message']}; continuation seed at aQstar={sign * current_abs:.6g} failed: "
            f"{current['message']}"
        )
        failed["continuation_used"] = True
        failed["continuation_steps"] = 0
        failed["continuation_seed_aQstar"] = float(sign * current_abs)
        if return_profile:
            return failed
        return _strip_entropy_profile_fields(failed)

    steps_taken = 0
    while current_abs < abs_target - 1.0e-12:
        next_abs = min(abs_target, current_abs + step_abs)
        next_a = sign * next_abs
        trial = _solve_steady_front_entropy_once(
            T,
            nB_N,
            B_one_forth,
            next_a,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=stage_n_mesh,
            tol_bvp=stage_tol,
            max_nodes=stage_max_nodes,
            jB_guess=current["jB"],
            jB_bounds=jB_bounds,
            return_profile=False,
            verb=False,
            profile_guess=current,
            seed_profile=True,
        )
        if bool(trial.get("success")):
            current = trial
            current_abs = next_abs
            steps_taken += 1
            if simple_diag and (steps_taken % 25 == 0 or current_abs >= abs_target - 1.0e-12):
                print(
                    f"entropy-bvp continuation: reached aQstar={next_a:.6g} "
                    f"with jB={current['jB']:.6g}"
                )
            step_abs = min(base_step_abs, max(abs_target - current_abs, 0.0))
            continue

        step_abs *= 0.5
        if step_abs < min_step_abs:
            failed = dict(trial)
            failed["message"] = (
                f"Entropy continuation failed after reaching aQstar={sign * current_abs:.6g}; "
                f"last attempted aQstar={next_a:.6g}. {trial['message']}"
            )
            failed["continuation_used"] = True
            failed["continuation_steps"] = steps_taken
            failed["continuation_seed_aQstar"] = float(sign * min(abs_target, 1.0e-2))
            if return_profile:
                return failed
            return _strip_entropy_profile_fields(failed)

    refined = _solve_steady_front_entropy_once(
        T,
        nB_N,
        B_one_forth,
        aQstar,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=n_mesh,
        tol_bvp=tol_bvp,
        max_nodes=max_nodes,
        jB_guess=current["jB"],
        jB_bounds=jB_bounds,
        return_profile=return_profile,
        verb=False,
        profile_guess=current,
    )
    if bool(refined.get("success")):
        result_out = dict(refined)
        result_out["continuation_refined"] = True
    else:
        if return_profile:
            coarse_profile = _solve_steady_front_entropy_once(
                T,
                nB_N,
                B_one_forth,
                aQstar,
                ms=ms,
                param=param,
                NM_type=NM_type,
                tail_eps=tail_eps,
                n_mesh=stage_n_mesh,
                tol_bvp=stage_tol,
                max_nodes=stage_max_nodes,
                jB_guess=current["jB"],
                jB_bounds=jB_bounds,
                return_profile=True,
                verb=False,
                profile_guess=current,
            )
            result_out = dict(coarse_profile if bool(coarse_profile.get("success")) else current)
        else:
            result_out = dict(current)
        result_out["message"] = (
            f"{result_out['message']}; final refinement failed: {refined['message']}"
        )
        result_out["continuation_refined"] = False

    result_out["continuation_used"] = True
    result_out["continuation_steps"] = steps_taken
    result_out["continuation_seed_aQstar"] = float(sign * min(abs_target, 1.0e-2))
    if return_profile:
        return result_out
    return _strip_entropy_profile_fields(result_out)


def _solve_steady_front_entropy_TQguess_once(
    T,
    nB_N,
    B_one_forth,
    aQstar,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    TQ_guess=None,
    jB_bounds=None,
    return_profile=False,
    verb=False,
    profile_guess=None,
    seed_profile=False,
):
    """
    Solve the entropy-enabled steady-front problem as a one-shot compact-coordinate BVP
    with downstream temperature T_Q promoted to a global unknown.

    The ODE unknowns are (a, q, w) with
        w = s * u
    and the BVP parameter vector is (jB, T_Q).
    """
    T = float(T)
    if (not np.isfinite(T)) or T <= 0.0:
        raise RuntimeError("solve_steady_front_entropy_TQguess requires T > 0")
    if nB_N <= 0.0:
        raise RuntimeError("nB_N must be positive")
    if tail_eps <= 0.0 or tail_eps >= 1.0:
        raise RuntimeError("tail_eps must satisfy 0 < tail_eps < 1")
    if int(n_mesh) < 5:
        raise RuntimeError("n_mesh must be at least 5")
    if tol_bvp <= 0.0:
        raise RuntimeError("tol_bvp must be positive")
    if max_nodes <= int(n_mesh):
        raise RuntimeError("max_nodes must be larger than n_mesh")

    upB = 5000
    t_start = time.perf_counter()
    if isinstance(verb, str):
        verb_mode = "full" if verb.lower() == "full" else ("simple" if verb else "off")
    else:
        verb_mode = "simple" if verb else "off"
    full_diag = verb_mode == "full"
    simple_diag = verb_mode in ("simple", "full")

    def _diag(msg):
        if full_diag:
            dt = time.perf_counter() - t_start
            print(f"[steady_front_entropy_TQguess +{dt:8.2f}s] {msg}", flush=True)

    if jB_guess is None:
        jB_guess = 1.0e-8 * nB_N
    jB_guess = float(jB_guess)
    if jB_guess <= 0.0:
        raise RuntimeError("jB_guess must be positive")

    if TQ_guess is None:
        TQ_guess = T
    TQ_guess = float(TQ_guess)
    if (not np.isfinite(TQ_guess)) or TQ_guess <= 0.0:
        raise RuntimeError("TQ_guess must be positive")

    bounded_jB = jB_bounds is not None
    if bounded_jB:
        if len(jB_bounds) != 2:
            raise RuntimeError("jB_bounds must be a 2-tuple (jB_min, jB_max)")
        jB_lower_bound = float(jB_bounds[0])
        jB_upper_bound = float(jB_bounds[1])
        if jB_lower_bound <= 0.0 or jB_upper_bound <= 0.0 or jB_upper_bound <= jB_lower_bound:
            raise RuntimeError("jB_bounds must satisfy 0 < jB_min < jB_max")
        if not (jB_lower_bound < jB_guess < jB_upper_bound):
            jB_guess = min(max(jB_guess, jB_lower_bound * (1.0 + 1.0e-6)), jB_upper_bound * (1.0 - 1.0e-6))
    else:
        jB_lower_bound = 0.0
        jB_upper_bound = np.inf

    def _param_from_jB(jB):
        jB = float(jB)
        if bounded_jB:
            frac = (jB - jB_lower_bound) / (jB_upper_bound - jB_lower_bound)
            frac = float(np.clip(frac, 1.0e-12, 1.0 - 1.0e-12))
            return float(np.log(frac / (1.0 - frac)))
        return float(np.log(jB))

    def _jB_from_param(theta):
        theta = float(theta)
        if bounded_jB:
            theta_clip = float(np.clip(theta, -60.0, 60.0))
            sig = 1.0 / (1.0 + np.exp(-theta_clip))
            return float(jB_lower_bound + (jB_upper_bound - jB_lower_bound) * sig)
        return float(np.exp(np.clip(theta, -700.0, 700.0)))

    def _param_from_TQ(TQ):
        TQ = float(TQ)
        if (not np.isfinite(TQ)) or TQ <= 0.0:
            raise RuntimeError("T_Q must be positive")
        return float(np.log(TQ))

    def _TQ_from_param(phi):
        return float(np.exp(np.clip(float(phi), -700.0, 700.0)))

    stats = {
        "bvp_ode_calls": 0,
        "bvp_bc_calls": 0,
        "q_root_calls": 0,
        "qstar_root_calls": 0,
        "local_state_calls": 0,
        "local_root_calls": 0,
        "local_fast_failures": 0,
        "profile_state_calls": 0,
        "global_state_builds": 0,
        "global_state_failures": 0,
        "local_state_failures": 0,
    }
    state_cache = {}
    muB_Q_last_guess = {"value": None}
    qstar_last_guess = {"value": None}
    last_failure = {"message": ""}
    s_coord_end = float(1.0 - tail_eps)

    def _build_global_state(theta, phi):
        key = (round(float(theta), 12), round(float(phi), 12))
        if key in state_cache:
            return state_cache[key]

        jB = _jB_from_param(theta)
        T_Q = _TQ_from_param(phi)
        if (not np.isfinite(T_Q)) or T_Q <= 0.0:
            raise RuntimeError("Entropy-TQguess solver requires T_Q > 0")

        stats["global_state_builds"] += 1

        # Upstream nuclear state N at x = 0^-.
        P_N = float(PNM_n(nB_N, T, param=param, NM_type=NM_type))
        e_N = float(edensNM_n(nB_N, T, param=param))
        h_N = float(P_N + e_N)
        u_N = float(jB / nB_N)
        Pi = float(h_N * u_N * u_N + P_N)

        # Far-right equilibrated quark state Q with muK = 0 and solved T_Q.
        muB_Q_guess = muB_Q_last_guess["value"]
        if muB_Q_guess is None and isinstance(profile_guess, dict):
            muB_Q_guess = profile_guess.get("muB_Q", None)
        muB_Q = _solve_muB_Q_at_muK0_for_given_Pi_ms(
            Pi,
            jB,
            B_one_forth,
            T_Q,
            ms=ms,
            upB=upB,
            stats=stats,
            initial_guess=muB_Q_guess,
        )
        muB_Q_last_guess["value"] = float(muB_Q)
        thermo_Q = _quark_thermo_state(muB_Q, 0.0, B_one_forth, T_Q, jB, ms=ms, upB=upB)
        nB_Q = float(thermo_Q["nB"])
        if abs(ms) <= 1.0e-12:
            nK_Q = 0.0
        else:
            nK_Q = float(thermo_Q["nK"])

        a_N = float((nB_N - nK_Q) / nB_Q)

        # Interface state Qstar at x = 0+ with fixed interface temperature T.
        muK_Qstar_seed = _branch_muK_seed(aQstar)
        qstar_initial_guess = (muB_Q, muK_Qstar_seed)
        if qstar_last_guess["value"] is not None:
            qstar_initial_guess = qstar_last_guess["value"]
        elif isinstance(profile_guess, dict):
            muB_Qstar_guess = profile_guess.get("muB_Qstar", None)
            muK_Qstar_guess = profile_guess.get("muK_Qstar", None)
            if muB_Qstar_guess is not None and muK_Qstar_guess is not None:
                try:
                    muB_Qstar_guess = float(muB_Qstar_guess)
                    muK_Qstar_guess = float(muK_Qstar_guess)
                except Exception:
                    muB_Qstar_guess = None
                    muK_Qstar_guess = None
                if (
                    muB_Qstar_guess is not None
                    and muK_Qstar_guess is not None
                    and np.isfinite(muB_Qstar_guess)
                    and np.isfinite(muK_Qstar_guess)
                    and muB_Qstar_guess > 0.0
                    and muK_Qstar_guess >= 0.0
                ):
                    qstar_initial_guess = (muB_Qstar_guess, muK_Qstar_guess)
        muB_Qstar, muK_Qstar = _solve_interface_Qstar_from_aQstar_and_Pi(
            aQstar,
            Pi,
            jB,
            nB_Q,
            nK_Q,
            B_one_forth,
            T,
            ms=ms,
            upB=upB,
            initial_guess=qstar_initial_guess,
            stats=stats,
            stats_key="qstar_root_calls",
        )
        qstar_last_guess["value"] = (float(muB_Qstar), float(muK_Qstar))
        thermo_Qstar = _quark_thermo_state(muB_Qstar, muK_Qstar, B_one_forth, T, jB, ms=ms, upB=upB)
        if thermo_Qstar["s"] <= 0.0:
            raise RuntimeError("Interface Qstar state has non-positive entropy density")
        if thermo_Qstar["w"] <= 0.0:
            raise RuntimeError("Interface Qstar state has non-positive entropy flux")

        micro_Q = _microphysics_from_quark_state(muB_Q, T_Q)
        D_Q = float(micro_Q["D"])
        eta_Q = float(micro_Q["eta"])
        gamma_Q = float(micro_Q["gamma"])
        tau_Q = float(micro_Q["tau"])

        u_Q = float(thermo_Q["u"])
        disc = float(u_Q * u_Q + 4.0 * D_Q * gamma_Q * eta_Q)
        if (not np.isfinite(disc)) or disc <= 0.0:
            raise RuntimeError("Entropy-TQguess solver tail discriminant is non-positive")
        lam = float((-u_Q + np.sqrt(disc)) / (2.0 * D_Q))
        if (not np.isfinite(lam)) or lam <= 0.0:
            raise RuntimeError("Entropy-TQguess solver tail decay lambda must be positive")
        q0 = float(-a_N * u_N)
        w0 = float(thermo_Qstar["w"])
        w_Q = float(thermo_Q["w"])
        x_end = float(-np.log1p(-s_coord_end) / lam)
        tail_coeff_Q = float(D_Q * lam + u_Q)
        state = {
            "jB": float(jB),
            "T_Q": float(T_Q),
            "P_N": P_N,
            "e_N": e_N,
            "h_N": h_N,
            "u_N": u_N,
            "Pi": Pi,
            "muB_Q": float(muB_Q),
            "nB_Q": float(nB_Q),
            "nK_Q": float(nK_Q),
            "a_N": float(a_N),
            "muB_Qstar": float(muB_Qstar),
            "muK_Qstar": float(muK_Qstar),
            "nB_Qstar": float(thermo_Qstar["nB"]),
            "s_Q": float(thermo_Q["s"]),
            "w_Q": float(w_Q),
            "s_Qstar": float(thermo_Qstar["s"]),
            "u_Qstar": float(thermo_Qstar["u"]),
            "T_Qstar": float(T),
            "D_Q": D_Q,
            "eta_Q": eta_Q,
            "gamma_Q": gamma_Q,
            "tau_Q": tau_Q,
            "lambda": float(lam),
            "u_Q": float(u_Q),
            "q0": float(q0),
            "w0": float(w0),
            "tail_coeff_Q": float(tail_coeff_Q),
            "x_end": float(x_end),
        }
        state_cache[key] = state
        return state

    def _state_or_none(theta, phi):
        try:
            return _build_global_state(theta, phi)
        except Exception as exc:
            stats["global_state_failures"] += 1
            last_failure["message"] = str(exc)
            return None

    def _ode(s_coord, y, p):
        stats["bvp_ode_calls"] += 1
        state = _state_or_none(float(p[0]), float(p[1]))
        if state is None:
            return np.zeros_like(y) + 1.0e12

        dyds = np.empty_like(y)
        guess = (state["muB_Qstar"], state["muK_Qstar"], state["T_Qstar"])
        for i in range(y.shape[1]):
            a_val = float(y[0, i])
            q_val = float(y[1, i])
            w_val = float(y[2, i])
            if (not np.isfinite(a_val)) or (not np.isfinite(q_val)) or (not np.isfinite(w_val)):
                stats["local_state_failures"] += 1
                dyds[:, i] = 1.0e12
                continue
            try:
                thermo_loc = _solve_local_quark_state_from_a_w_and_Pi(
                    a_val,
                    w_val,
                    state["Pi"],
                    state["jB"],
                    state["nB_Q"],
                    state["nK_Q"],
                    B_one_forth,
                    ms=ms,
                    upB=upB,
                    initial_guess=guess,
                    T_ref=state["T_Q"],
                    stats=stats,
                )
                guess = (thermo_loc["muB"], thermo_loc["muK"], thermo_loc["T"])
                micro_loc = _microphysics_from_quark_state(thermo_loc["muB"], thermo_loc["T"])
                reaction = float(micro_loc["gamma"] * (a_val**3 + micro_loc["eta"] * a_val))
                one_minus_s = max(1.0 - float(s_coord[i]), np.finfo(float).tiny)
                dx_ds = 1.0 / (state["lambda"] * one_minus_s)
                dyds[0, i] = ((q_val + thermo_loc["u"] * a_val) / micro_loc["D"]) * dx_ds
                dyds[1, i] = reaction * dx_ds
                dyds[2, i] = ((state["nB_Q"] / thermo_loc["T"]) * thermo_loc["muK"] * reaction) * dx_ds
            except Exception as exc:
                stats["local_state_failures"] += 1
                last_failure["message"] = str(exc)
                dyds[:, i] = 1.0e12
        return dyds

    def _bc(ya, yb, p):
        stats["bvp_bc_calls"] += 1
        state = _state_or_none(float(p[0]), float(p[1]))
        if state is None:
            return np.array([1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12], dtype=float)
        return np.array(
            [
                ya[0] - float(aQstar),
                ya[1] - state["q0"],
                ya[2] - state["w0"],
                yb[1] + state["tail_coeff_Q"] * yb[0],
                yb[2] - state["w_Q"],
            ],
            dtype=float,
        )

    theta_guess = _param_from_jB(jB_guess)
    phi_guess = _param_from_TQ(TQ_guess)
    state0 = _state_or_none(theta_guess, phi_guess)
    if state0 is None:
        return {
            "success": False,
            "message": f"Initial entropy-TQguess state construction failed: {last_failure['message']}",
            "aQstar": float(aQstar),
            "jB": np.nan,
            "T_Q": np.nan,
            "w_Q": np.nan,
            "branch_label": "muK-rich",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "entropy_tail_residual": np.nan,
            "entropy_tail_residual_norm": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "w_end": np.nan,
            "bvp_status": -999,
            "bvp_message": "initial_state_failure",
            "bvp_niter": -1,
            "bvp_nodes": 0,
            **{k: int(v) for k, v in stats.items()},
        }

    s_coord_mesh = np.linspace(0.0, s_coord_end, int(n_mesh))
    blend = s_coord_mesh / max(s_coord_end, np.finfo(float).tiny)
    tail_shape = np.maximum(1.0 - s_coord_mesh, tail_eps)

    def _default_initial_guess():
        a_guess_loc = float(aQstar) * tail_shape
        q_tail_guess_loc = -state0["tail_coeff_Q"] * a_guess_loc
        q_guess_loc = (1.0 - blend) * state0["q0"] + blend * q_tail_guess_loc
        w_guess_loc = (1.0 - blend) * state0["w0"] + blend * state0["w_Q"]
        return a_guess_loc, q_guess_loc, w_guess_loc

    def _profile_initial_guess(profile):
        seed_input = profile
        try:
            s_prev = np.asarray(seed_input["s_coord"], dtype=float)
            a_prev = np.asarray(seed_input["a"], dtype=float)
            q_prev = np.asarray(seed_input["q"], dtype=float)
            w_prev = np.asarray(seed_input["w"], dtype=float)
        except Exception:
            return _default_initial_guess()

        if (
            s_prev.ndim != 1
            or a_prev.shape != s_prev.shape
            or q_prev.shape != s_prev.shape
            or w_prev.shape != s_prev.shape
            or s_prev.size < 5
            or not np.all(np.isfinite(s_prev))
            or not np.all(np.isfinite(a_prev))
            or not np.all(np.isfinite(q_prev))
            or not np.all(np.isfinite(w_prev))
        ):
            return _default_initial_guess()

        a_prev0 = float(a_prev[0])
        if abs(a_prev0) > 1.0e-12:
            a_scale = float(aQstar) / a_prev0
        else:
            a_scale = 1.0

        a_guess_loc = np.interp(s_coord_mesh, s_prev, a_prev) * a_scale
        a_guess_loc[0] = float(aQstar)

        q_base = np.interp(s_coord_mesh, s_prev, q_prev) * a_scale
        q_base0 = float(q_base[0])
        q_base_end = float(q_base[-1])
        q_target0 = float(state0["q0"])
        q_target_end = float(-state0["tail_coeff_Q"] * a_guess_loc[-1])
        q_span = q_base_end - q_base0
        if np.isfinite(q_span) and abs(q_span) > 1.0e-12 * max(1.0, abs(q_base0), abs(q_base_end)):
            q_frac = (q_base - q_base0) / q_span
            q_guess_loc = q_target0 + q_frac * (q_target_end - q_target0)
        else:
            q_guess_loc = (1.0 - blend) * q_target0 + blend * q_target_end
        q_guess_loc[0] = q_target0

        w_base = np.interp(s_coord_mesh, s_prev, w_prev)
        w_base0 = float(w_base[0])
        w_base_end = float(w_base[-1])
        w_span = w_base_end - w_base0
        if (
            np.all(np.isfinite(w_base))
            and w_base0 > 0.0
            and w_base_end > 0.0
            and abs(w_span) > 1.0e-12 * max(1.0, abs(w_base0), abs(w_base_end))
        ):
            w_frac = (w_base - w_base0) / w_span
            w_guess_loc = state0["w0"] + w_frac * (state0["w_Q"] - state0["w0"])
        else:
            w_guess_loc = (1.0 - blend) * state0["w0"] + blend * state0["w_Q"]

        if (not np.all(np.isfinite(a_guess_loc))) or (not np.all(np.isfinite(q_guess_loc))) or (not np.all(np.isfinite(w_guess_loc))) or np.any(w_guess_loc <= 0.0):
            return _default_initial_guess()

        w_guess_loc[0] = float(state0["w0"])
        return a_guess_loc, q_guess_loc, w_guess_loc

    if profile_guess is None:
        a_guess, q_guess, w_guess = _default_initial_guess()
    else:
        a_guess, q_guess, w_guess = _profile_initial_guess(profile_guess)
    y_guess = np.vstack((a_guess, q_guess, w_guess))

    _diag(
        f"starting entropy-TQguess compact BVP with jB_guess={jB_guess:.6g}, "
        f"TQ_guess={TQ_guess:.6g}, aQstar={aQstar:.6g}, tail_eps={tail_eps:.3g}, branch=muK-rich"
    )

    try:
        sol = solve_bvp(
            _ode,
            _bc,
            s_coord_mesh,
            y_guess,
            p=np.array([theta_guess, phi_guess], dtype=float),
            tol=tol_bvp,
            max_nodes=max_nodes,
            verbose=2 if full_diag else 0,
        )
    except Exception as exc:
        return {
            "success": False,
            "message": f"solve_bvp raised: {exc}; last failure: {last_failure['message']}",
            "aQstar": float(aQstar),
            "jB": np.nan,
            "T_Q": np.nan,
            "w_Q": np.nan,
            "branch_label": "muK-rich",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "entropy_tail_residual": np.nan,
            "entropy_tail_residual_norm": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "w_end": np.nan,
            "bvp_status": -999,
            "bvp_message": f"solve_bvp raised: {exc}",
            "bvp_niter": -1,
            "bvp_nodes": 0,
            **{k: int(v) for k, v in stats.items()},
        }

    theta_sol = float(sol.p[0])
    phi_sol = float(sol.p[1])
    state = _state_or_none(theta_sol, phi_sol)
    if state is None:
        return {
            "success": False,
            "message": f"BVP final state construction failed: {last_failure['message']}",
            "aQstar": float(aQstar),
            "jB": np.nan,
            "T_Q": np.nan,
            "w_Q": np.nan,
            "branch_label": "muK-rich",
            "tail_residual": np.nan,
            "tail_residual_norm": np.nan,
            "entropy_tail_residual": np.nan,
            "entropy_tail_residual_norm": np.nan,
            "a_end": np.nan,
            "q_end": np.nan,
            "w_end": np.nan,
            "bvp_status": int(sol.status),
            "bvp_message": str(sol.message),
            "bvp_niter": int(getattr(sol, "niter", -1)),
            "bvp_nodes": int(sol.x.size),
            **{k: int(v) for k, v in stats.items()},
        }

    a_end = float(sol.y[0, -1])
    q_end = float(sol.y[1, -1])
    w_end = float(sol.y[2, -1])
    tail_drive = float(state["tail_coeff_Q"] * a_end)
    tail_residual = float(q_end + tail_drive)
    tail_scale = float(max(abs(q_end), abs(tail_drive), np.finfo(float).tiny))
    tail_residual_norm = float(tail_residual / tail_scale)
    entropy_tail_residual = float(w_end - state["w_Q"])
    entropy_tail_scale = float(max(abs(w_end), abs(state["w_Q"]), np.finfo(float).tiny))
    entropy_tail_residual_norm = float(entropy_tail_residual / entropy_tail_scale)

    success = bool(
        sol.success
        and np.isfinite(tail_residual_norm)
        and np.isfinite(entropy_tail_residual_norm)
        and abs(tail_residual_norm) <= max(tol_bvp, 10.0 * np.finfo(float).eps)
        and abs(entropy_tail_residual_norm) <= max(tol_bvp, 10.0 * np.finfo(float).eps)
    )
    result = {
        "success": success,
        "message": "Entropy-TQguess compact BVP steady-front solve converged" if success else f"{sol.message}; last failure: {last_failure['message']}",
        "jB": float(state["jB"]),
        "T_Q": float(state["T_Q"]),
        "w_Q": float(state["w_Q"]),
        "aQstar": float(aQstar),
        "branch_label": "muK-rich",
        "coordinate": "BVP: s_coord in [0, 1-tail_eps], s_coord=1-exp(-lambda*x)",
        "tail_eps": float(tail_eps),
        "u_N": float(state["u_N"]),
        "u_Q": float(state["u_Q"]),
        "a_N": float(state["a_N"]),
        "Pi": float(state["Pi"]),
        "muB_Q": float(state["muB_Q"]),
        "nB_Q": float(state["nB_Q"]),
        "nK_Q": float(state["nK_Q"]),
        "s_Q": float(state["s_Q"]),
        "muB_Qstar": float(state["muB_Qstar"]),
        "muK_Qstar": float(state["muK_Qstar"]),
        "nB_Qstar": float(state["nB_Qstar"]),
        "s_Qstar": float(state["s_Qstar"]),
        "D_Q": float(state["D_Q"]),
        "eta_Q": float(state["eta_Q"]),
        "gamma_Q": float(state["gamma_Q"]),
        "tau_Q": float(state["tau_Q"]),
        "lambda": float(state["lambda"]),
        "kappa": float(1.0 / state["lambda"]),
        "s_end": float(s_coord_end),
        "x_end": float(state["x_end"]),
        "a_end": a_end,
        "q_end": q_end,
        "w_end": w_end,
        "tail_residual": tail_residual,
        "tail_residual_norm": tail_residual_norm,
        "tail_scale": tail_scale,
        "entropy_tail_residual": entropy_tail_residual,
        "entropy_tail_residual_norm": entropy_tail_residual_norm,
        "entropy_tail_scale": entropy_tail_scale,
        "_residual": np.array([tail_residual_norm, entropy_tail_residual_norm], dtype=float),
        "_root_method": "solve_bvp_parameter_2d",
        "bvp_status": int(sol.status),
        "bvp_message": str(sol.message),
        "bvp_niter": int(getattr(sol, "niter", -1)),
        "bvp_nodes": int(sol.x.size),
        **{k: int(v) for k, v in stats.items()},
    }

    if return_profile or seed_profile:
        s_coord_prof = np.asarray(sol.x, dtype=float)
        a_prof = np.asarray(sol.y[0], dtype=float)
        q_prof = np.asarray(sol.y[1], dtype=float)
        w_prof = np.asarray(sol.y[2], dtype=float)
        x_prof = -np.log1p(-s_coord_prof) / float(state["lambda"])
        result.update(
            {
                "s_coord": s_coord_prof,
                "x": x_prof,
                "a": a_prof,
                "q": q_prof,
                "w": w_prof,
            }
        )
    if return_profile and success:
        muB_prof = np.empty_like(s_coord_prof)
        muK_prof = np.empty_like(s_coord_prof)
        T_prof = np.empty_like(s_coord_prof)
        nB_prof = np.empty_like(s_coord_prof)
        u_prof = np.empty_like(s_coord_prof)
        entropy_prof = np.empty_like(s_coord_prof)
        D_prof = np.empty_like(s_coord_prof)
        eta_prof = np.empty_like(s_coord_prof)
        gamma_prof = np.empty_like(s_coord_prof)
        guess = (state["muB_Qstar"], state["muK_Qstar"], state["T_Qstar"])
        try:
            for i, a_val in enumerate(a_prof):
                stats["profile_state_calls"] += 1
                thermo_loc = _solve_local_quark_state_from_a_w_and_Pi(
                    float(a_val),
                    float(w_prof[i]),
                    state["Pi"],
                    state["jB"],
                    state["nB_Q"],
                    state["nK_Q"],
                    B_one_forth,
                    ms=ms,
                    upB=upB,
                    initial_guess=guess,
                    T_ref=state["T_Q"],
                    stats=stats,
                )
                guess = (thermo_loc["muB"], thermo_loc["muK"], thermo_loc["T"])
                micro_loc = _microphysics_from_quark_state(thermo_loc["muB"], thermo_loc["T"])
                muB_prof[i] = thermo_loc["muB"]
                muK_prof[i] = thermo_loc["muK"]
                T_prof[i] = thermo_loc["T"]
                nB_prof[i] = thermo_loc["nB"]
                u_prof[i] = thermo_loc["u"]
                entropy_prof[i] = thermo_loc["s"]
                D_prof[i] = micro_loc["D"]
                eta_prof[i] = micro_loc["eta"]
                gamma_prof[i] = micro_loc["gamma"]
        except Exception as exc:
            result["success"] = False
            result["message"] = f"{result['message']}; profile reconstruction failed: {exc}"
            result["profile_reconstruction_failed"] = True
            result["profile_reconstruction_message"] = str(exc)
            result["profile_state_calls"] = int(stats["profile_state_calls"])
            result["local_state_calls"] = int(stats["local_state_calls"])
            result["local_root_calls"] = int(stats["local_root_calls"])
            result["local_fast_failures"] = int(stats["local_fast_failures"])
            if simple_diag:
                print(
                    f"entropy-TQguess-bvp profile reconstruction failed for aQstar={aQstar:.6g}: {exc}"
                )
            return result

        result.update(
            {
                "u": u_prof,
                "nB": nB_prof,
                "muB": muB_prof,
                "muK": muK_prof,
                "T_profile": T_prof,
                "s_profile": entropy_prof,
                "D_profile": D_prof,
                "eta_profile": eta_prof,
                "gamma_profile": gamma_prof,
                "profile_state_calls": int(stats["profile_state_calls"]),
                "local_state_calls": int(stats["local_state_calls"]),
                "local_root_calls": int(stats["local_root_calls"]),
                "local_fast_failures": int(stats["local_fast_failures"]),
            }
        )

    if simple_diag:
        print(
            f"entropy-TQguess-bvp jB={result['jB']:.6g}, T_Q={result['T_Q']:.6g}, "
            f"aQstar={aQstar:.6g}, tail_norm={tail_residual_norm:.6g}, "
            f"w_tail_norm={entropy_tail_residual_norm:.6g}, status={sol.status}, success={success}"
        )
    return result


def solve_steady_front_entropy_TQguess(
    T,
    nB_N,
    B_one_forth,
    aQstar,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=1e-8,
    n_mesh=200,
    tol_bvp=1e-4,
    max_nodes=10000,
    jB_guess=None,
    TQ_guess=None,
    jB_bounds=None,
    return_profile=False,
    verb=False,
):
    """
    Solve the entropy-enabled steady-front problem as a compact-coordinate BVP
    with downstream asymptotic temperature T_Q treated as a global unknown.

    The solver first attempts a direct one-shot solve. If that fails, it
    automatically falls back to continuation in aQstar, reusing the last
    converged entropy profile and (jB, T_Q) pair as the next BVP initial guess.
    """
    aQstar = float(aQstar)
    abs_target = abs(aQstar)
    sign = -1.0 if aQstar < 0.0 else 1.0

    if isinstance(verb, str):
        verb_mode = "full" if verb.lower() == "full" else ("simple" if verb else "off")
    else:
        verb_mode = "simple" if verb else "off"
    simple_diag = verb_mode in ("simple", "full")

    direct_jB_guess = jB_guess
    if direct_jB_guess is None:
        # Keep the first direct attempt cheap, but seed it on the rough flux
        # scale implied by both the upstream density and the interface aQstar.
        direct_jB_guess = float(max(1.0e-12, 1.0e-8 * float(nB_N), 0.8 * abs_target))

    direct_TQ_guess = float(T if TQ_guess is None else TQ_guess)
    if (not np.isfinite(direct_TQ_guess)) or direct_TQ_guess <= 0.0:
        raise RuntimeError("TQ_guess must be positive")

    direct_result = _solve_steady_front_entropy_TQguess_once(
        T,
        nB_N,
        B_one_forth,
        aQstar,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=n_mesh,
        tol_bvp=tol_bvp,
        max_nodes=max_nodes,
        jB_guess=direct_jB_guess,
        TQ_guess=direct_TQ_guess,
        jB_bounds=jB_bounds,
        return_profile=return_profile,
        verb=verb,
        profile_guess=None,
    )
    if bool(direct_result.get("success")) or abs_target <= 1.0e-2:
        result_out = dict(direct_result)
        result_out["continuation_used"] = False
        result_out["continuation_steps"] = 0
        if return_profile:
            return result_out
        return _strip_entropy_profile_fields(result_out)

    if simple_diag:
        print(
            f"entropy-TQguess-bvp continuation: direct solve failed at aQstar={aQstar:.6g}; "
            f"retrying with staged continuation"
        )

    stage_n_mesh = int(min(max(max(int(n_mesh) // 2, 25), 25), 40))
    stage_tol = float(max(float(tol_bvp), 2.0e-2))
    stage_max_nodes = int(max(stage_n_mesh + 5, min(int(max_nodes), 250)))
    base_step_abs = 1.0e-3
    min_step_abs = 1.0e-4
    step_abs = float(min(base_step_abs, max(abs_target - min(abs_target, 1.0e-2), 0.0)))
    current_abs = min(abs_target, 1.0e-2)
    stage_jB_guess = direct_jB_guess
    if not (np.isfinite(stage_jB_guess) and stage_jB_guess > 0.0):
        stage_jB_guess = None
    direct_jB_out = direct_result.get("jB", np.nan)
    if np.isfinite(direct_jB_out) and direct_jB_out > 0.0:
        stage_jB_guess = float(direct_jB_out)
    if stage_jB_guess is None:
        seed_abs = current_abs if current_abs > 0.0 else 1.0e-2
        stage_jB_guess = _seed_entropy_jB_guess_basic(
            T,
            nB_N,
            B_one_forth,
            sign * seed_abs,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=n_mesh,
            tol_bvp=tol_bvp,
            max_nodes=max_nodes,
            verb=False,
        )
    current = _solve_steady_front_entropy_TQguess_once(
        T,
        nB_N,
        B_one_forth,
        sign * current_abs,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=stage_n_mesh,
        tol_bvp=stage_tol,
        max_nodes=stage_max_nodes,
        jB_guess=stage_jB_guess,
        TQ_guess=direct_TQ_guess,
        jB_bounds=jB_bounds,
        return_profile=False,
        verb=False,
        profile_guess=None,
        seed_profile=True,
    )
    if not bool(current.get("success")):
        failed = dict(current)
        failed["message"] = (
            f"{direct_result['message']}; continuation seed at aQstar={sign * current_abs:.6g} failed: "
            f"{current['message']}"
        )
        failed["continuation_used"] = True
        failed["continuation_steps"] = 0
        failed["continuation_seed_aQstar"] = float(sign * current_abs)
        if return_profile:
            return failed
        return _strip_entropy_profile_fields(failed)

    steps_taken = 0
    while current_abs < abs_target - 1.0e-12:
        next_abs = min(abs_target, current_abs + step_abs)
        next_a = sign * next_abs
        trial = _solve_steady_front_entropy_TQguess_once(
            T,
            nB_N,
            B_one_forth,
            next_a,
            ms=ms,
            param=param,
            NM_type=NM_type,
            tail_eps=tail_eps,
            n_mesh=stage_n_mesh,
            tol_bvp=stage_tol,
            max_nodes=stage_max_nodes,
            jB_guess=current["jB"],
            TQ_guess=current["T_Q"],
            jB_bounds=jB_bounds,
            return_profile=False,
            verb=False,
            profile_guess=current,
            seed_profile=True,
        )
        if bool(trial.get("success")):
            current = trial
            current_abs = next_abs
            steps_taken += 1
            if simple_diag and (steps_taken % 25 == 0 or current_abs >= abs_target - 1.0e-12):
                print(
                    f"entropy-TQguess-bvp continuation: reached aQstar={next_a:.6g} "
                    f"with jB={current['jB']:.6g}, T_Q={current['T_Q']:.6g}"
                )
            step_abs = min(base_step_abs, max(abs_target - current_abs, 0.0))
            continue

        step_abs *= 0.5
        if step_abs < min_step_abs:
            failed = dict(trial)
            failed["message"] = (
                f"Entropy-TQguess continuation failed after reaching aQstar={sign * current_abs:.6g}; "
                f"last attempted aQstar={next_a:.6g}. {trial['message']}"
            )
            failed["continuation_used"] = True
            failed["continuation_steps"] = steps_taken
            failed["continuation_seed_aQstar"] = float(sign * min(abs_target, 1.0e-2))
            if return_profile:
                return failed
            return _strip_entropy_profile_fields(failed)

    refined = _solve_steady_front_entropy_TQguess_once(
        T,
        nB_N,
        B_one_forth,
        aQstar,
        ms=ms,
        param=param,
        NM_type=NM_type,
        tail_eps=tail_eps,
        n_mesh=n_mesh,
        tol_bvp=tol_bvp,
        max_nodes=max_nodes,
        jB_guess=current["jB"],
        TQ_guess=current["T_Q"],
        jB_bounds=jB_bounds,
        return_profile=return_profile,
        verb=False,
        profile_guess=current,
    )
    if bool(refined.get("success")):
        result_out = dict(refined)
        result_out["continuation_refined"] = True
    else:
        if return_profile:
            coarse_profile = _solve_steady_front_entropy_TQguess_once(
                T,
                nB_N,
                B_one_forth,
                aQstar,
                ms=ms,
                param=param,
                NM_type=NM_type,
                tail_eps=tail_eps,
                n_mesh=stage_n_mesh,
                tol_bvp=stage_tol,
                max_nodes=stage_max_nodes,
                jB_guess=current["jB"],
                TQ_guess=current["T_Q"],
                jB_bounds=jB_bounds,
                return_profile=True,
                verb=False,
                profile_guess=current,
            )
            result_out = dict(coarse_profile if bool(coarse_profile.get("success")) else current)
        else:
            result_out = dict(current)
        result_out["message"] = (
            f"{result_out['message']}; final refinement failed: {refined['message']}"
        )
        result_out["continuation_refined"] = False

    result_out["continuation_used"] = True
    result_out["continuation_steps"] = steps_taken
    result_out["continuation_seed_aQstar"] = float(sign * min(abs_target, 1.0e-2))
    if return_profile:
        return result_out
    return _strip_entropy_profile_fields(result_out)


def extract_jB_curve_vs_aQstar(
    T,
    nB_N,
    B_one_forth,
    aQstar_values,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    x_max=1e6,
    n_eval=1200,
    rtol_ode=1e-8,
    atol_ode=1e-10,
    root_tol=1e-8,
    max_nfev=60,
    jB_guess=None,
    jB_bounds=None,
    return_profile=False,
    verb=False,
):
    """
    Sweep over fixed aQstar values and extract the numerical curve jB(aQstar)
    by scalar shooting in jB.
    """
    results = []
    next_jB_guess = jB_guess

    for aQ in aQstar_values:
        result = solve_steady_front_1d_aQstar(
            T=T,
            nB_N=nB_N,
            B_one_forth=B_one_forth,
            aQstar=float(aQ),
            ms=ms,
            param=param,
            NM_type=NM_type,
            x_max=x_max,
            n_eval=n_eval,
            rtol_ode=rtol_ode,
            atol_ode=atol_ode,
            root_tol=root_tol,
            max_nfev=max_nfev,
            jB_guess=next_jB_guess,
            jB_bounds=jB_bounds,
            return_profile=return_profile,
            verb=verb,
        )
        results.append(result)
        if result.get("success"):
            next_jB_guess = float(result["jB"])

    success_flags = np.array([bool(res.get("success")) for res in results], dtype=bool)
    aQ_all = np.array([float(res.get("aQstar", np.nan)) for res in results], dtype=float)
    jB_all = np.array([float(res.get("jB", np.nan)) if res.get("success") else np.nan for res in results], dtype=float)
    tail_all = np.array([float(res.get("tail_residual", np.nan)) for res in results], dtype=float)

    return {
        "results": results,
        "branch_label": "muK-rich",
        "aQstar_all": aQ_all,
        "jB_all": jB_all,
        "tail_residual_all": tail_all,
        "success_flags": success_flags,
        "aQstar_success": aQ_all[success_flags],
        "jB_success": jB_all[success_flags],
        "tail_residual_success": tail_all[success_flags],
    }


"""
Deprecated jB/Pi-based interface retained for reference only.

def _find_Qstar(T, B_one_forth, lambda_val, jB, Pi_over_crit):
    '''
    solves system of equations to get Qstar

    Eq1: Pi_NM = Pi_QM
    Eq2: piecewise constant approximation equation of the Olinto type ODE

    returns muBstar, muKstar
    '''

    _, _, Pi_target, muB_N, muB_Q = _pi_target_state(T, B_one_forth, jB, Pi_over_crit)
    return _find_Qstar_on_target(T, B_one_forth, lambda_val, jB, Pi_target, muB_N, muB_Q)


def uQ_uN(T, B_one_forth, lambda_val, jB, Pi_over_crit):
    '''
    returns uQ and uN in [m/s]
    '''
    _, _, Pi_target, muB_N, muB_Q = _pi_target_state(T, B_one_forth, jB, Pi_over_crit)
    muB_star, muK_star, nB_N = _find_Qstar_on_target(
        T,
        B_one_forth,
        lambda_val,
        jB,
        Pi_target,
        muB_N,
        muB_Q,
        return_aux=True,
    )
    nB_Q = nB_QM(muB_star, muK_star, B_one_forth, T)

    if nB_N <= 0 or nB_Q <= 0:
        raise RuntimeError("Non-positive density encountered while computing uQ_uN")

    uN = (jB / nB_N) * 3e8
    uQ = (jB / nB_Q) * 3e8
    return float(uQ), float(uN)
"""




######################## Old useless code ##########################

# find Q star
def _find_muB_muK_star(PQM, P_target, C, B_one_forth, T, ms, upB=5000, initial_guess=(1020.0, 700.0)):
    '''
    Solves:
      1. PQM(muB, muK) = P_target
      2. muB + muK * (∂PQM/∂muK)/(∂PQM/∂muB) = C

    Returns:
      muB_star, muK_star
    '''

    def f(muB, muK):
        return PQM(muB, muK, B_one_forth, T, ms, upB)

    def dPQM_dmuB(muB, muK):
        return nB_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB)

    def dPQM_dmuK(muB, muK):
        return nK_QM(muB, muK, B_one_forth, T, ms=ms, upB=upB)

    def system(vec):
        muB, muK = vec
        P_val = f(muB, muK)
        df_dmuB = dPQM_dmuB(muB, muK)
        df_dmuK = dPQM_dmuK(muB, muK)

        eq1 = P_val - P_target
        eq2 = df_dmuB * muB + df_dmuK * muK - df_dmuB * C

        return np.array([eq1, eq2], dtype=float)

    res = least_squares(system, initial_guess, bounds=([0, 0], [np.inf, np.inf]))

    if not res.success:
        raise RuntimeError(f"Root finding failed: {res.message}")

    if float(res.x[1]) < 0:
        raise RuntimeError(f"muK_star < 0")

    return float(res.x[0]), float(res.x[1])


# Directly uses Eq.(29) to calculate velocity
def _vNtoQ_formula(T, aQstar, aN=1.0, muQ=360):
    '''
    Computes the phase boundary velocity for burning nuclear matter to quark matter

    Parameters:
    - T      : temperature
    - aQstar : normalized kaon isospin density at the boundary, range from 0 ~ 1
    - aN     : normalized kaon isospin density in nuclear matter, default = 1.0
    - muQ    : quark chemical potential, default = 360 MeV

    Returns:
    - velocity of phase boundary in meters per second

    '''
    alpha_s = 0.3
    g_s = np.sqrt(4 * np.pi * alpha_s)

    # Compute intermediate quantities
    eta = (9 * np.pi**2 * T**2) / muQ**2
    tau = 1.98e12 * ((300 / muQ)**5)
    qD = np.sqrt(3 * g_s**2 * muQ**2 / (2 * np.pi**2))
    h = 1.81317

    part1 = h*T**(5/3)/qD**(2/3)
    part2 = np.pi**3 * T**2 / (12*qD)

    D = 1/( 24*alpha_s**2/np.pi * ( part1 + part2 ) )
    # D = (np.pi / (24 * (0.3)**2 * 1.81 * T**(5/3))) * ((6 * 0.3 / np.pi * muQ**2)**(1/3))

    # Compute v_N->Q
    if aN < aQstar:
        print("hit aN < aQstar, returning velocity = 0")
        return 0.0
    else:
        return np.sqrt((D / tau) * ((aQstar**4 + 2 * eta * aQstar**2) / (2 * aN * (aN - aQstar)))) * 3e8

# Use numerical rescaling method to calculate velocity
def _vNtoQ_num_rescale(T, aQstar, aN, muQ):
    '''
    Computes the phase boundary velocity for burning nuclear matter to quark matter

    Parameters:
    - T      : temperature
    - aQstar : normalized kaon isospin density at the boundary, range from 0 ~ 1
    - aN     : normalized kaon isospin density in nuclear matter, default = 1.0
    - muQ    : quark chemical potential, default = 360 MeV

    Returns:
    - velocity of phase boundary in meters per second

    '''

    if aN < aQstar:
        print("hit aN < aQstar, returning velocity = 0.0")
        return 0.0

    alpha_s = 0.3
    c1 = aQstar

    # --- microphysics ---
    g_s   = np.sqrt(4*np.pi*alpha_s)
    eta   = (9*np.pi**2 * T**2) / muQ**2
    tau   = 1.98e12 * ((300.0 / muQ)**5)

    qD    = np.sqrt(3 * g_s**2 * muQ**2 / (2 * np.pi**2))
    h     = 1.81317
    part1 = h * T**(5/3) / qD**(2/3)
    part2 = (np.pi**3 * T**2) / (12.0 * qD)
    D     = 1.0 / ((24.0 * alpha_s**2 / np.pi) * (part1 + part2))

    # lambda(v) from the linearized tail
    def get_lambda(v):
        return (-v + np.sqrt(v * v + 4 * D * eta / tau)) / (2 * D)

    # Nonlinearity
    def R(a):
        a_safe = np.clip(a, -7e2, 7e2)
        return (a_safe**3 + eta * a_safe) / tau

    # ODE in t = \tilde{x}: y[0]=a(t), y[1]=p(t)=da/dx
    def ode_system(t, y, v, lam):
        a = y[0]
        p = y[1]
        dxdt = 1.0 / (lam * (c1 - t))          # dx/dt
        a_t  = p * dxdt
        p_t  = ((v * p + R(a)) / D) * dxdt
        return np.vstack((a_t, p_t))

    # Boundary conditions: a(0)=aQ*, a(c1)=0
    def bc(ya, yb):
        return np.array([ya[0] - aQstar, yb[0]])

    # Solve BVP for a given v and return a'(x=0)=p(0)
    def get_derivatives_at_zero(v):
        v = float(abs(v))
        lam = get_lambda(v)
        t = np.linspace(0, c1 - 1e-9, 1000)

        # Smooth monotone initial guess
        a_guess = aQstar * np.exp(-5 * t / c1)
        # p = a' = a_t / (dt/dx), with dt/dx = lam * (c1 - t)
        p_guess = -(5 * aQstar / c1) * np.exp(-5 * t / c1) * lam * (c1 - t)
        y_init  = np.vstack((a_guess, p_guess))

        try:
            sol = solve_bvp(lambda tt, yy: ode_system(tt, yy, v, lam),
                            bc, t, y_init, max_nodes=100000)
            if sol.status != 0:
                return None
            return sol.y[1, 0]   # p(0) = a'(x=0)
        except Exception:
            return None

    # Residual for remaining BC at x=0: a'(0+) = - v (aN - aQ*) / D
    def slope_residual(v_arr):
        v = float(abs(v_arr[0]))          # root() passes arrays
        a_prime_x0 = get_derivatives_at_zero(v)
        if a_prime_x0 is None or not np.isfinite(a_prime_x0):
            return np.array([1e20], dtype=float)   # 1-element array
        rhs = -v * (aN - aQstar) / D
        return np.array([a_prime_x0 - rhs], dtype=float)

    # Solve for v (natural units), then convert to m/s for backward compatibility
    sol = root(slope_residual, [1e-7], method='hybr')
    if sol.success and sol.x[0] > 0:
        v_nat = float(abs(sol.x[0]))      # c = 1 units
        return v_nat * 3e8                # return m/s as your original code did
    else:
        raise RuntimeError(f" Root finding failed: {sol.message}")

# Use numerical truncation method to calculate velocity
def _vNtoQ_num(
    T, aQstar, aN, muQ,
    Ntail=50.0,
    C_lin=1e3,            # demand a(L)^2 <= eta/C_lin (larger = stricter)
    refine_once=True,     # do at most one L-update using solved v
    v_init=1e-7,          # initial guess for v in natural units (c=1)
    tol=1e-6,
    max_nodes=300000,
    n_mesh=900,
    L_change_trigger=0.2  # redo if |L2-L|/L > this
):
    """
    Fast truncated Robin BVP solver for v.

    Solves on x ∈ [0, L]:
      y0 = a, y1 = ∂_x a
      ∂_x a = y1
      ∂_x y1 = (v y1 + R(a)) / D

    BCs:
      a(0) = aQstar
      ∂_x a(0) = - v (aN-aQstar)/D
      ∂_x a(L) = λ_-(v) a(L)

    L is chosen as max(L_tail, L_lin) computed from λ_-(v_guess).
    Returns (v_si, L_km).
    """

    if aN < aQstar:
        return 0.0, 0.0

    T = float(T)
    if T <= 0.0:
        raise RuntimeError("T must be positive")

    alpha_s = 0.3

    # --- microphysics ---
    g_s   = np.sqrt(4*np.pi*alpha_s)
    eta   = (9*np.pi**2 * T**2) / (muQ**2)
    tau   = 1.98e12 * ((300.0 / muQ)**5)

    qD    = np.sqrt(3 * g_s**2 * muQ**2 / (2 * np.pi**2))
    h     = 1.81317
    part1 = h * T**(5/3) / qD**(2/3)
    part2 = (np.pi**3 * T**2) / (12.0 * qD)
    D     = 1.0 / ((24.0 * alpha_s**2 / np.pi) * (part1 + part2))

    def R(a):
        a_safe = np.clip(a, -7e2, 7e2)
        return (a_safe**3 + eta * a_safe) / tau

    def lambda_minus(v):
        v = float(abs(v))
        disc = v*v + 4.0 * D * eta / tau
        return (v - np.sqrt(disc)) / (2.0 * D)  # < 0

    def pick_L(v_guess):
        lam = lambda_minus(v_guess)
        if not np.isfinite(lam) or lam >= 0.0:
            raise RuntimeError("Bad lambda_- from v_guess; check parameters.")

        L_tail = float(Ntail / abs(lam))

        # amplitude target to ensure eta*a >> a^3 at x=L
        a_lin = np.sqrt(max(eta / C_lin, 1e-300))

        if aQstar <= a_lin:
            L_lin = 0.0
        else:
            L_lin = float((1.0 / abs(lam)) * np.log(aQstar / a_lin))

        return float(max(L_tail, L_lin)), float(lam)

    # --- BVP with unknown parameter p[0]=v ---
    def ode(x, y, p):
        v = float(abs(p[0]))
        a = y[0]
        px = y[1]
        return np.vstack((px, (v * px + R(a)) / D))

    def bc(ya, yb, p):
        v = float(abs(p[0]))
        lam = lambda_minus(v)
        return np.array([
            ya[0] - aQstar,
            ya[1] + v * (aN - aQstar) / D,
            yb[1] - lam * yb[0],
        ], dtype=float)

    # -----------------------
    # Pass #1: pick L from v_init and solve
    # -----------------------
    L, lam0 = pick_L(v_init)

    x = np.linspace(0.0, L, int(n_mesh))
    a_guess = aQstar * np.exp(lam0 * x)
    px_guess = lam0 * a_guess
    y_init = np.vstack((a_guess, px_guess))
    p_init = np.array([float(v_init)], dtype=float)

    sol = solve_bvp(
        ode, bc, x, y_init, p=p_init,
        tol=tol, max_nodes=max_nodes, verbose=0
    )
    if (not sol.success) or (not np.isfinite(sol.p[0])) or (sol.p[0] <= 0.0):
        raise RuntimeError(f"solve_bvp failed (pass #1): {sol.message}")

    v_sol = float(abs(sol.p[0]))

    # -----------------------
    # Optional pass #2: recompute L from v_sol and re-solve if L changes a lot
    # -----------------------
    if refine_once:
        L2, _ = pick_L(v_sol)
        if abs(L2 - L) / max(L, 1e-300) > float(L_change_trigger):
            x2 = np.linspace(0.0, L2, max(int(n_mesh), len(x)))
            y2 = sol.sol(x2)
            sol2 = solve_bvp(
                ode, bc, x2, y2, p=np.array([v_sol], dtype=float),
                tol=tol, max_nodes=max_nodes, verbose=0
            )
            if sol2.success and np.isfinite(sol2.p[0]) and sol2.p[0] > 0.0:
                sol = sol2
                v_sol = float(abs(sol.p[0]))
                L = float(L2)
            # else keep pass #1 result silently

    v_si = v_sol * 3e8
    #L_km = float(L) * 1.97327e-16
    return v_si#, L_km

# Taking transition pressure, strange quark mass, NM model name as input
def vNtoQ_Pc(T, P_crit, DelP, m_s, param, NM_type, method="analytical", aQmax=True):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - Trmf     : temperature
    - P_crit   : critical pressure for 1st order phase transition
    - DelP     : Messures how far away from equilibrium
    - m_s      : strange quark mass
    - param    : mean field theory model settings for nuclear matter
    - NM_type  : assumptions for nuclear matter, choose from:
                 - "Beta_eq" for beta equilibriated nuclear matter
                 - "PNM" for pure neutron matter
                 - "SYM" for symmetric nuclear matter
    - method   : ways to calculate velocity
                 - "numerical" in principle more precise, but have numerical noise
                 - "analytical" analytical approximation, direct, faster, and robust, ~ 10% difference with numerical method
    - aQmax    : whether not to use analytical approximation for calculating aQmax
                 - Default: True

    Returns:
    - velocity of phase boundary in meters per second
    '''


    # solve for bag constant
    P_diff = lambda x: PNM(x, T, param, NM_type) - P_crit
    muB_crit = fsolve(P_diff, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit, 0, x, T, m_s) - P_crit
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]

    # solve for points Q and N at beta eq.
    PNM_minus_PShift = lambda x: PNM(x, T, param, NM_type) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if aQmax:
        epsilon = np.sqrt(np.finfo(float).eps)
        muB_star, muK_star = _find_muB_muK_star(PQM, P_crit + DelP, muB_N, B_SQM, T, m_s)
        PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_SQM, T, m_s)
        grad_Qstar = approx_fprime(np.array([muB_star, muK_star]), PQM_wrap, epsilon)
        nB_Qstar = grad_Qstar[0]
        nK_Qstar = grad_Qstar[1]
        grad_Q = approx_fprime(np.array([muB_Q, 0]), PQM_wrap, epsilon)
        nB_Q = grad_Q[0]
        nK_Q = grad_Q[1]
        aQstar = (nK_Qstar - nK_Q)/nB_Q  
        
    else:
        PQM_muB = lambda Mu: PQM(Mu, 0, B_SQM, T, m_s)
        PQM_muK = lambda Mu: PQM(muB_Q, Mu, B_SQM, T, m_s)
        ddPQM_ddmuK = Derivative(PQM_muK, n=2)
        dPQM_dmuK = Derivative(PQM_muK, n=1)
        dPQM_dmuB = Derivative(PQM_muB, n=1)
        chiK_Q = ddPQM_ddmuK(0)
        nB_Q = dPQM_dmuB(muB_Q)
        nK_Q = dPQM_dmuK(0)
        aQstar = np.sqrt(2*(muB_N - muB_Q)*chiK_Q / nB_Q)
        muK_star = np.sqrt(2 * nB_Q * (muB_N - muB_Q) / chiK_Q)
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit - DelP
        muB_star = fsolve(PQM_solve_for_muBstar, muB_Q)[0]

    # calculating aN
    PNM_wrap = lambda x: PNM(x, T, param, NM_type)
    dPNM_dmuB = Derivative(PNM_wrap, n=1)
    nB_N = dPNM_dmuB(muB_N)
    aN = (nB_N - nK_Q) / nB_Q   

    # calculating velocity
    if method == "analytical":
        vel = _vNtoQ_formula(T, aQstar, aN, muB_star/3)
    elif method == "numerical":
        vel = _vNtoQ_num(T, aQstar, aN, muB_star/3)
    else:
        raise RuntimeError(f"Input method unknown")
        return None

    return vel, B_SQM


# Taking transition pressure, strange quark mass, NM model name as input
def vNtoQ_Pc_fixB(T, P_crit0, DelP, m_s, param, NM_type, method="analytical", aQmax=True):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - Trmf     : temperature
    - P_crit0  : critical pressure for 1st order phase transition at T=0
    - DelP     : Messures how far away from equilibrium
    - m_s      : strange quark mass
    - param    : mean field theory model settings for nuclear matter
    - NM_type  : assumptions for nuclear matter, choose from:
                 - "Beta_eq" for beta equilibriated nuclear matter
                 - "PNM" for pure neutron matter
                 - "SYM" for symmetric nuclear matter
    - method   : ways to calculate velocity
                 - "numerical" in principle more precise, but have numerical noise
                 - "analytical" analytical approximation, direct, faster, and robust, ~ 10% difference with numerical method
    - aQmax    : whether not to use analytical approximation for calculating aQmax
                 - Default: True

    Returns:
    - velocity of phase boundary in meters per second
    '''


    # solve for bag constant
    muB_crit0 = fsolve(lambda x: PNM(x, Temp=0, param=param, NM_type=NM_type) - P_crit0, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit0, 0, x, 0, m_s) - P_crit0
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]
    muB_crit = fsolve(lambda x: PNM(x, T, param, NM_type) - PQM(x, 0, B_SQM, T, m_s), 1050)[0]
    P_crit = PQM(muB_crit, 0, B_SQM, T, m_s)

    # solve for points Q and N at beta eq.
    PNM_minus_PShift = lambda x: PNM(x, T, param, NM_type) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if aQmax:
        epsilon = np.sqrt(np.finfo(float).eps)
        muB_star, muK_star = _find_muB_muK_star(PQM, P_crit + DelP, muB_N, B_SQM, T, m_s)
        PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_SQM, T, m_s)
        grad_Qstar = approx_fprime(np.array([muB_star, muK_star]), PQM_wrap, epsilon)
        nB_Qstar = grad_Qstar[0]
        nK_Qstar = grad_Qstar[1]
        grad_Q = approx_fprime(np.array([muB_Q, 0]), PQM_wrap, epsilon)
        nB_Q = grad_Q[0]
        nK_Q = grad_Q[1]
        aQstar = (nK_Qstar - nK_Q)/nB_Q  
        
    else:
        PQM_muB = lambda Mu: PQM(Mu, 0, B_SQM, T, m_s)
        PQM_muK = lambda Mu: PQM(muB_Q, Mu, B_SQM, T, m_s)
        ddPQM_ddmuK = Derivative(PQM_muK, n=2)
        dPQM_dmuK = Derivative(PQM_muK, n=1)
        dPQM_dmuB = Derivative(PQM_muB, n=1)
        chiK_Q = ddPQM_ddmuK(0)
        nB_Q = dPQM_dmuB(muB_Q)
        nK_Q = dPQM_dmuK(0)
        aQstar = np.sqrt(2*(muB_N - muB_Q)*chiK_Q / nB_Q)
        muK_star = np.sqrt(2 * nB_Q * (muB_N - muB_Q) / chiK_Q)
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit - DelP
        muB_star = fsolve(PQM_solve_for_muBstar, muB_Q)[0]

    # calculating aN
    PNM_wrap = lambda x: PNM(x, T, param, NM_type)
    dPNM_dmuB = Derivative(PNM_wrap, n=1)
    nB_N = dPNM_dmuB(muB_N)
    aN = (nB_N - nK_Q) / nB_Q   

    # calculating velocity
    if method == "analytical":
        vel = _vNtoQ_formula(T, aQstar, aN, muB_star/3)
    elif method == "numerical":
        vel = _vNtoQ_num(T, aQstar, aN, muB_star/3)
    else:
        raise RuntimeError(f"Input method unknown")
        return None

    return vel, B_SQM


# Taking SQM bag constant, strange quark mass, NM model name as input
def vNtoQ_B(T, B_SQM, DelP, m_s, param, NM_type, method="analytical", aQmax=True):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - Trmf     : temperature
    - B_SQM    : strange quark matter bag constant, in (1/4) power
                 e.g. take input 165, not 165**4. 
    - DelP     : Messures how far away from equilibrium
    - m_s      : strange quark mass
    - param    : mean field theory model settings for nuclear matter
    - NM_type  : assumptions for nuclear matter, choose from:
                 - "Beta_eq" for beta equilibriated nuclear matter
                 - "PNM" for pure neutron matter
                 - "SYM" for symmetric nuclear matter
    - method   : ways to calculate velocity
                 - "numerical" in principle more precise, but have numerical noise
                 - "analytical" analytical approximation, direct, faster, and robust, ~ 10% difference with numerical method
    - aQmax    : whether not to use analytical approximation for calculating aQmax
                 - Default: True

    Returns:
    - velocity of phase boundary in meters per second
    '''


    # solve for critical point
    P_diff = lambda x: PQM(x, 0, B_SQM, T, m_s) - PNM(x, T, param, NM_type)
    muB_crit = fsolve(P_diff, 1050)[0]
    P_crit = PQM(muB_crit, 0, B_SQM, T, m_s)

    # solve for points Q and N at beta eq.
    PNM_minus_PShift = lambda x: PNM(x, T, param, NM_type) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if aQmax:
        epsilon = np.sqrt(np.finfo(float).eps)
        muB_star, muK_star = _find_muB_muK_star(PQM, P_crit + DelP, muB_N, B_SQM, T, m_s)
        PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_SQM, T, m_s)
        grad_Qstar = approx_fprime(np.array([muB_star, muK_star]), PQM_wrap, epsilon)
        nB_Qstar = grad_Qstar[0]
        nK_Qstar = grad_Qstar[1]
        grad_Q = approx_fprime(np.array([muB_Q, 0]), PQM_wrap, epsilon)
        nB_Q = grad_Q[0]
        nK_Q = grad_Q[1]
        aQstar = (nK_Qstar - nK_Q)/nB_Q  
 
    else:
        PQM_muB = lambda Mu: PQM(Mu, 0, B_SQM, T, m_s)
        PQM_muK = lambda Mu: PQM(muB_Q, Mu, B_SQM, T, m_s)
        ddPQM_ddmuK = Derivative(PQM_muK, n=2)
        dPQM_dmuK = Derivative(PQM_muK, n=1)
        dPQM_dmuB = Derivative(PQM_muB, n=1)
        chiK_Q = ddPQM_ddmuK(0)
        nB_Q = dPQM_dmuB(muB_Q)
        nK_Q = dPQM_dmuK(0)
        aQstar = np.sqrt(2*(muB_N - muB_Q)*chiK_Q / nB_Q)
        muK_star = np.sqrt(2 * nB_Q * (muB_N - muB_Q) / chiK_Q)
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit - DelP
        muB_star = fsolve(PQM_solve_for_muBstar, muB_Q)[0]


    # calculating aN
    PNM_wrap = lambda x: PNM(x, T, param, NM_type)
    dPNM_dmuB = Derivative(PNM_wrap, n=1)
    nB_N = dPNM_dmuB(muB_N)
    aN = (nB_N - nK_Q) / nB_Q   

    # calculating velocity
    if method == "analytical":
        vel = _vNtoQ_formula(T, aQstar, aN, muB_star/3)
    elif method == "numerical":
        vel = _vNtoQ_num(T, aQstar, aN, muB_star/3)
    else:
        raise RuntimeError(f"Input method unknown")
        return None

    return vel, P_crit


# Taking transition density, strange quark mass, NM model name as input
def vNtoQ_nc(T, n_crit, Deln, m_s, param, NM_type, method="analytical", aQmax=True):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - T        : temperature
    - n_crit   : critical density for 1st order phase transition
    - Deln     : Messures how far away from equilibrium
    - m_s      : strange quark mass
    - param    : mean field theory model settings for nuclear matter
    - NM_type  : assumptions for nuclear matter, choose from:
                 - "Beta_eq" for beta equilibriated nuclear matter
                 - "PNM" for pure neutron matter
                 - "SYM" for symmetric nuclear matter
    - method   : ways to calculate velocity
                 - "numerical" in principle more precise, but have numerical noise
                 - "analytical" analytical approximation, direct, faster, and robust, ~ 10% difference with numerical method
    - aQmax    : whether not to use analytical approximation for calculating aQmax
                 - Default: True

    Returns:
    - velocity of phase boundary in meters per second
    '''

    # solve for bag constant
    P_crit = PNM_n(n_crit, T, param, NM_type)
    P_diff = lambda x: PNM(x, T, param, NM_type) - P_crit
    muB_crit = fsolve(P_diff, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit, 0, x, T, m_s) - P_crit
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]

    # solve for points Q and N at beta eq.
    DelP = PNM_n(n_crit + Deln, T, param, NM_type) - P_crit
    PNM_minus_PShift = lambda x: PNM(x, T, param, NM_type) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if aQmax:
        epsilon = np.sqrt(np.finfo(float).eps)
        muB_star, muK_star = _find_muB_muK_star(PQM, P_crit + DelP, muB_N, B_SQM, T, m_s)
        PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_SQM, T, m_s)
        grad_Qstar = approx_fprime(np.array([muB_star, muK_star]), PQM_wrap, epsilon)
        nB_Qstar = grad_Qstar[0]
        nK_Qstar = grad_Qstar[1]
        grad_Q = approx_fprime(np.array([muB_Q, 0]), PQM_wrap, epsilon)
        nB_Q = grad_Q[0]
        nK_Q = grad_Q[1]
        aQstar = (nK_Qstar - nK_Q)/nB_Q  
 
    else:
        PQM_muB = lambda Mu: PQM(Mu, 0, B_SQM, T, m_s)
        PQM_muK = lambda Mu: PQM(muB_Q, Mu, B_SQM, T, m_s)
        ddPQM_ddmuK = Derivative(PQM_muK, n=2)
        dPQM_dmuK = Derivative(PQM_muK, n=1)
        dPQM_dmuB = Derivative(PQM_muB, n=1)
        chiK_Q = ddPQM_ddmuK(0)
        nB_Q = dPQM_dmuB(muB_Q)
        nK_Q = dPQM_dmuK(0)
        aQstar = np.sqrt(2*(muB_N - muB_Q)*chiK_Q / nB_Q)
        muK_star = np.sqrt(2 * nB_Q * (muB_N - muB_Q) / chiK_Q)
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit - DelP
        muB_star = fsolve(PQM_solve_for_muBstar, muB_Q)[0]


    # calculating aN
    PNM_wrap = lambda x: PNM(x, T, param, NM_type)
    dPNM_dmuB = Derivative(PNM_wrap, n=1)
    nB_N = dPNM_dmuB(muB_N)
    aN = (nB_N - nK_Q) / nB_Q   

    # calculating velocity
    if method == "analytical":
        vel = _vNtoQ_formula(T, aQstar, aN, muB_star/3)
    elif method == "numerical":
        vel = _vNtoQ_num(T, aQstar, aN, muB_star/3)
    else:
        raise RuntimeError(f"Input method unknown")
        return None

    return vel, B_SQM


# Taking transition density, strange quark mass, NM model name as input
def vNtoQ_fixB(T, n_crit, Deln, m_s, param, NM_type, method="analytical", aQmax=True):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - T        : temperature
    - n_crit   : critical density for 1st order phase transition at T=0
    - Deln     : Messures how far away from equilibrium
    - m_s      : strange quark mass
    - param    : mean field theory model settings for nuclear matter
    - NM_type  : assumptions for nuclear matter, choose from:
                 - "Beta_eq" for beta equilibriated nuclear matter
                 - "PNM" for pure neutron matter
                 - "SYM" for symmetric nuclear matter
    - method   : ways to calculate velocity
                 - "numerical" in principle more precise, but have numerical noise
                 - "analytical" analytical approximation, direct, faster, and robust, ~ 10% difference with numerical method
    - aQmax    : whether not to use analytical approximation for calculating aQmax
                 - Default: True

    Returns:
    - velocity of phase boundary in meters per second
    '''


    n0 = 0.16 * const.MeV_fm**3

    # solve for bag constant
    P_crit0 = PNM_n(n_crit, Temp=0, param=param, NM_type=NM_type)
    muB_crit0 = fsolve(lambda x: PNM(x, Temp=0, param=param, NM_type=NM_type) - P_crit0, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit0, muK=0, B_one_forth=x, T=0, ms=m_s) - P_crit0
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]

    # solve for critical point
    muB_crit = fsolve(lambda x: PQM(x, 0, B_SQM, T, m_s) - PNM(x, T, param, NM_type), 1050)[0]
    P_crit = PQM(muB_crit, 0, B_SQM, T, m_s)
    n_crit_T = fsolve(lambda x: PNM_n(x, T, param, NM_type) - P_crit, 2*n0)[0]
    DelP = PNM_n(n_crit_T + Deln, T, param, NM_type) - P_crit

    # solve for points Q and N at beta eq.
    PNM_minus_PShift = lambda x: PNM(x, T, param, NM_type) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if aQmax:
        epsilon = np.sqrt(np.finfo(float).eps)
        muB_star, muK_star = _find_muB_muK_star(PQM, P_crit + DelP, muB_N, B_SQM, T, m_s)
        PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_SQM, T, m_s)
        grad_Qstar = approx_fprime(np.array([muB_star, muK_star]), PQM_wrap, epsilon)
        nB_Qstar = grad_Qstar[0]
        nK_Qstar = grad_Qstar[1]
        grad_Q = approx_fprime(np.array([muB_Q, 0]), PQM_wrap, epsilon)
        nB_Q = grad_Q[0]
        nK_Q = grad_Q[1]
        aQstar = (nK_Qstar - nK_Q)/nB_Q  
 
    else:
        PQM_muB = lambda Mu: PQM(Mu, 0, B_SQM, T, m_s)
        PQM_muK = lambda Mu: PQM(muB_Q, Mu, B_SQM, T, m_s)
        ddPQM_ddmuK = Derivative(PQM_muK, n=2)
        dPQM_dmuK = Derivative(PQM_muK, n=1)
        dPQM_dmuB = Derivative(PQM_muB, n=1)
        chiK_Q = ddPQM_ddmuK(0)
        nB_Q = dPQM_dmuB(muB_Q)
        nK_Q = dPQM_dmuK(0)
        aQstar = np.sqrt(2*(muB_N - muB_Q)*chiK_Q / nB_Q)
        muK_star = np.sqrt(2 * nB_Q * (muB_N - muB_Q) / chiK_Q)
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit - DelP
        muB_star = fsolve(PQM_solve_for_muBstar, muB_Q)[0]


    # calculating aN
    PNM_wrap = lambda x: PNM(x, T, param, NM_type)
    dPNM_dmuB = Derivative(PNM_wrap, n=1)
    nB_N = dPNM_dmuB(muB_N)
    aN = (nB_N - nK_Q) / nB_Q   

    # calculating velocity
    if method == "analytical":
        vel = _vNtoQ_formula(T, aQstar, aN, muB_star/3)
    elif method == "numerical":
        vel = _vNtoQ_num(T, aQstar, aN, muB_star/3)
    else:
        raise RuntimeError(f"Input method unknown")
        return None

    return vel, B_SQM


# Calculates \Delta n max for given ncrit and T
def Get_Delta_n_max(T, n_crit, m_s, param, NM_type, aQmax=True, tol=5e-3, coarse_pts=5, bounds=None):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - T        : temperature
    - n_crit   : critical density for 1st order phase transition
    - m_s      : strange quark mass
    - param    : mean field theory model settings for nuclear matter
    - NM_type  : assumptions for nuclear matter, choose from:
                 - "Beta_eq" for beta equilibriated nuclear matter
                 - "PNM" for pure neutron matter
                 - "SYM" for symmetric nuclear matter
    - aQmax    : whether not to use analytical approximation for calculating aQmax
                 - Default: True

    Returns:
    - Maximum pussible Delta_n before aQstar > aN
    '''


    # --- compute the expensive, Δn‑independent pieces ONCE ---
    P_crit = PNM_n(n_crit, T, param, NM_type)

    # small wrapper to keep fsolve snappy
    def fsolve_fast(fun, x0):
        return fsolve(fun, x0, xtol=1e-6, maxfev=400)[0]

    muB_crit = fsolve_fast(lambda x: PNM(x, T, param, NM_type) - P_crit, 1050.0)
    B_SQM    = fsolve_fast(lambda B: PQM(muB_crit, 0, B, T, m_s) - P_crit, 180.0)

    def Get_aN_aQstar_diff(Delta_n):
        # solve for points Q and N at beta eq. (these DO depend on Δn)
        DelP = PNM_n(n_crit + Delta_n, T, param, NM_type) - P_crit

        muB_N = fsolve_fast(lambda x: PNM(x, T, param, NM_type) - P_crit - DelP, 1050.0)
        muB_Q = fsolve_fast(lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP, 1050.0)

        # calculating aQstar
        if aQmax:
            epsilon = np.sqrt(np.finfo(float).eps)
            muB_star, muK_star = _find_muB_muK_star(PQM, P_crit + DelP, muB_N, B_SQM, T, m_s)
            PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_SQM, T, m_s)
            grad_Qstar = approx_fprime(np.array([muB_star, muK_star]), PQM_wrap, epsilon)
            nB_Qstar = grad_Qstar[0]
            nK_Qstar = grad_Qstar[1]
            grad_Q = approx_fprime(np.array([muB_Q, 0]), PQM_wrap, epsilon)
            nB_Q = grad_Q[0]
            nK_Q = grad_Q[1]
            aQstar = (nK_Qstar - nK_Q)/nB_Q  
     
        else:
            PQM_muB = lambda Mu: PQM(Mu, 0, B_SQM, T, m_s)
            PQM_muK = lambda Mu: PQM(muB_Q, Mu, B_SQM, T, m_s)
            ddPQM_ddmuK = Derivative(PQM_muK, n=2)
            dPQM_dmuK = Derivative(PQM_muK, n=1)
            dPQM_dmuB = Derivative(PQM_muB, n=1)
            chiK_Q = ddPQM_ddmuK(0)
            nB_Q = dPQM_dmuB(muB_Q)
            nK_Q = dPQM_dmuK(0)
            aQstar = np.sqrt(2*(muB_N - muB_Q)*chiK_Q / nB_Q)
            muK_star = np.sqrt(2 * nB_Q * (muB_N - muB_Q) / chiK_Q)
            muB_star = fsolve_fast(lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit - DelP, muB_Q)


        # calculating aN
        PNM_wrap = lambda x: PNM(x, T, param, NM_type)
        dPNM_dmuB = Derivative(PNM_wrap, n=1)
        nB_N = dPNM_dmuB(muB_N)
        aN = (nB_N - nK_Q) / nB_Q   

        if aN >= aQstar:
            diff = aN - aQstar
        else:
            diff = 1e4*(aQstar - aN)

        if diff < tol:
            # fast early-exit INSIDE this function only
            raise RuntimeError("__EARLY_OK__:" + str(Delta_n))
        return diff

    # ---- single bounded refine; catch early-exit; never leak exceptions ----
    if bounds is None:
        left, right = 0.0, 1.3*0.16*const.MeV_fm**3
    else:
        left, right = float(bounds[0]), float(bounds[1])

    try:
        res = minimize_scalar(Get_aN_aQstar_diff, bounds=(left, right), method='bounded',
                              options={'maxiter': 20, 'xatol': (right-left)*1e-3})
        # good enough?
        if hasattr(res, 'fun') and np.isfinite(res.fun) and res.fun <= tol:
            return float(res.x)
        return float(res.x) if hasattr(res, 'x') and np.isfinite(res.x) else np.nan
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("__EARLY_OK__:"):
            return float(msg.split(":")[1])
        # any other error: fall back gracefully
        return np.nan
    except Exception:
        return np.nan

    # sanity check & clamp (belt-and-suspenders)
    if np.isfinite(x_best):
        x_best = max(left, min(x_best, right))
    return x_best


# Calculates \Delta P max for given Pcrit and T
def Get_Delta_P_max(T, P_crit, m_s, param, NM_type, aQmax=True, tol=5e-3, coarse_pts=5, bounds=None):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - T        : temperature
    - P_crit   : critical pressure for 1st order phase transition (given)
    - m_s      : strange quark mass
    - param    : mean field theory model settings for nuclear matter
    - NM_type  : assumptions for nuclear matter, choose from:
                 - "Beta_eq" for beta equilibriated nuclear matter
                 - "PNM" for pure neutron matter
                 - "SYM" for symmetric nuclear matter
    - aQmax    : whether not to use analytical approximation for calculating aQmax
                 - Default: True

    Returns:
    - Maximum possible Delta_P before aQstar > aN
    '''

    # --- compute the expensive, ΔP-independent pieces ONCE ---

    # small wrapper to keep fsolve snappy
    def fsolve_fast(fun, x0):
        return fsolve(fun, x0, xtol=1e-6, maxfev=400)[0]

    muB_crit = fsolve_fast(lambda x: PNM(x, T, param, NM_type) - P_crit, 1050.0)
    B_SQM    = fsolve_fast(lambda B: PQM(muB_crit, 0, B, T, m_s) - P_crit, 180.0)

    def Get_aN_aQstar_diff(Delta_P):
        # solve for points Q and N on the shifted isobar (these DO depend on ΔP)
        DelP = float(Delta_P)

        muB_N = fsolve_fast(lambda x: PNM(x, T, param, NM_type) - P_crit - DelP, 1050.0)
        muB_Q = fsolve_fast(lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP, 1050.0)

        # calculating aQstar
        if aQmax:
            epsilon = np.sqrt(np.finfo(float).eps)
            muB_star, muK_star = _find_muB_muK_star(PQM, P_crit + DelP, muB_N, B_SQM, T, m_s)
            PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_SQM, T, m_s)
            grad_Qstar = approx_fprime(np.array([muB_star, muK_star]), PQM_wrap, epsilon)
            nB_Qstar = grad_Qstar[0]
            nK_Qstar = grad_Qstar[1]
            grad_Q = approx_fprime(np.array([muB_Q, 0]), PQM_wrap, epsilon)
            nB_Q = grad_Q[0]
            nK_Q = grad_Q[1]
            aQstar = (nK_Qstar - nK_Q)/nB_Q  
     
        else:
            PQM_muB = lambda Mu: PQM(Mu, 0, B_SQM, T, m_s)
            PQM_muK = lambda Mu: PQM(muB_Q, Mu, B_SQM, T, m_s)
            ddPQM_ddmuK = Derivative(PQM_muK, n=2)
            dPQM_dmuK = Derivative(PQM_muK, n=1)
            dPQM_dmuB = Derivative(PQM_muB, n=1)
            chiK_Q = ddPQM_ddmuK(0)
            nB_Q = dPQM_dmuB(muB_Q)
            nK_Q = dPQM_dmuK(0)
            aQstar = np.sqrt(2*(muB_N - muB_Q)*chiK_Q / nB_Q)
            muK_star = np.sqrt(2 * nB_Q * (muB_N - muB_Q) / chiK_Q)
            # NOTE: target is P_crit + DelP (NOT P_crit)
            muB_star = fsolve_fast(lambda x: PQM(x, muK_star, B_SQM, T, m_s) - (P_crit + DelP), muB_Q)


        # calculating aN
        PNM_wrap = lambda x: PNM(x, T, param, NM_type)
        dPNM_dmuB = Derivative(PNM_wrap, n=1)
        nB_N = dPNM_dmuB(muB_N)
        aN = (nB_N - nK_Q) / nB_Q   

        if aN >= aQstar:
            diff = aN - aQstar
        else:
            diff = 1e4*(aQstar - aN)

        if diff < tol:
            # fast early-exit INSIDE this function only
            raise RuntimeError("__EARLY_OK__:" + str(DelP))
        return diff

    # ---- single bounded refine; catch early-exit; never leak exceptions ----
    if bounds is None:
        left, right = 0.0, 8.0*float(P_crit)   # start reasonably wide (root can be > P_crit)
    else:
        left, right = float(bounds[0]), float(bounds[1])

    try:
        res = minimize_scalar(Get_aN_aQstar_diff, bounds=(left, right), method='bounded',
                              options={'maxiter': 20, 'xatol': (right-left)*1e-3})
        # good enough?
        if hasattr(res, 'fun') and np.isfinite(res.fun) and res.fun <= tol:
            return float(res.x)
        return float(res.x) if hasattr(res, 'x') and np.isfinite(res.x) else np.nan
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("__EARLY_OK__:"):
            return float(msg.split(":")[1])
        # any other error: fall back gracefully
        return np.nan
    except Exception:
        return np.nan


# Calculate aN and aQstar with input density
def Get_Two_as_dens(T, n_crit, Deln, m_s, param, NM_type, aQmax=True):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - T        : temperature
    - n_crit   : critical density for 1st order phase transition
    - Deln     : Messures how far away from equilibrium
    - m_s      : strange quark mass
    - param    : mean field theory model settings for nuclear matter
    - NM_type  : assumptions for nuclear matter, choose from:
                 - "Beta_eq" for beta equilibriated nuclear matter
                 - "PNM" for pure neutron matter
                 - "SYM" for symmetric nuclear matter
    - aQmax    : whether not to use analytical approximation for calculating aQmax
                 - Default: True

    Returns:
    - aQstar, aN
    '''

    # solve for bag constant
    P_crit = PNM_n(n_crit, T, param, NM_type)
    P_diff = lambda x: PNM(x, T, param, NM_type) - P_crit
    muB_crit = fsolve(P_diff, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit, 0, x, T, m_s) - P_crit
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]

    # solve for points Q and N at beta eq.
    DelP = PNM_n(n_crit + Deln, T, param, NM_type) - P_crit
    PNM_minus_PShift = lambda x: PNM(x, T, param, NM_type) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if aQmax:
        epsilon = np.sqrt(np.finfo(float).eps)
        muB_star, muK_star = _find_muB_muK_star(PQM, P_crit + DelP, muB_N, B_SQM, T, m_s)
        PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_SQM, T, m_s)
        grad_Qstar = approx_fprime(np.array([muB_star, muK_star]), PQM_wrap, epsilon)
        nB_Qstar = grad_Qstar[0]
        nK_Qstar = grad_Qstar[1]
        grad_Q = approx_fprime(np.array([muB_Q, 0]), PQM_wrap, epsilon)
        nB_Q = grad_Q[0]
        nK_Q = grad_Q[1]
        aQstar = (nK_Qstar - nK_Q)/nB_Q  
 
    else:
        PQM_muB = lambda Mu: PQM(Mu, 0, B_SQM, T, m_s)
        PQM_muK = lambda Mu: PQM(muB_Q, Mu, B_SQM, T, m_s)
        ddPQM_ddmuK = Derivative(PQM_muK, n=2)
        dPQM_dmuK = Derivative(PQM_muK, n=1)
        dPQM_dmuB = Derivative(PQM_muB, n=1)
        chiK_Q = ddPQM_ddmuK(0)
        nB_Q = dPQM_dmuB(muB_Q)
        nK_Q = dPQM_dmuK(0)
        aQstar = np.sqrt(2*(muB_N - muB_Q)*chiK_Q / nB_Q)
        muK_star = np.sqrt(2 * nB_Q * (muB_N - muB_Q) / chiK_Q)
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit
        muB_star = fsolve(PQM_solve_for_muBstar, muB_Q)[0]


    # calculating aN
    PNM_wrap = lambda x: PNM(x, T, param, NM_type)
    dPNM_dmuB = Derivative(PNM_wrap, n=1)
    nB_N = dPNM_dmuB(muB_N)
    aN = (nB_N - nK_Q) / nB_Q   

    return aQstar, aN


# Calculate aN and aQstar with input density
def Get_Two_as_dens0(T, n_crit, Deln, m_s, param, NM_type, aQmax=True):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - T        : temperature
    - n_crit   : critical density at T=0 for 1st order phase transition
    - Deln     : Messures how far away from equilibrium
    - m_s      : strange quark mass
    - param    : mean field theory model settings for nuclear matter
    - NM_type  : assumptions for nuclear matter, choose from:
                 - "Beta_eq" for beta equilibriated nuclear matter
                 - "PNM" for pure neutron matter
                 - "SYM" for symmetric nuclear matter
    - aQmax    : whether not to use analytical approximation for calculating aQmax
                 - Default: True

    Returns:
    - aQstar, aN
    '''


    n0 = 0.16 * const.MeV_fm**3

    # solve for bag constant
    P_crit0 = PNM_n(n_crit, Temp=0, param=param, NM_type=NM_type)
    muB_crit0 = fsolve(lambda x: PNM(x, Temp=0, param=param, NM_type=NM_type) - P_crit0, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit0, muK=0, B_one_forth=x, T=0, ms=m_s) - P_crit0
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]

    # solve for critical point
    muB_crit = fsolve(lambda x: PQM(x, 0, B_SQM, T, m_s) - PNM(x, T, param, NM_type), 1050)[0]
    P_crit = PQM(muB_crit, 0, B_SQM, T, m_s)
    n_crit_T = fsolve(lambda x: PNM_n(x, T, param, NM_type) - P_crit, 2*n0)[0]
    DelP = PNM_n(n_crit_T + Deln, T, param, NM_type) - P_crit

    # solve for points Q and N at beta eq.
    PNM_minus_PShift = lambda x: PNM(x, T, param, NM_type) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if aQmax:
        epsilon = np.sqrt(np.finfo(float).eps)
        muB_star, muK_star = _find_muB_muK_star(PQM, P_crit + DelP, muB_N, B_SQM, T, m_s)
        PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_SQM, T, m_s)
        grad_Qstar = approx_fprime(np.array([muB_star, muK_star]), PQM_wrap, epsilon)
        nB_Qstar = grad_Qstar[0]
        nK_Qstar = grad_Qstar[1]
        grad_Q = approx_fprime(np.array([muB_Q, 0]), PQM_wrap, epsilon)
        nB_Q = grad_Q[0]
        nK_Q = grad_Q[1]
        aQstar = (nK_Qstar - nK_Q)/nB_Q  
 
    else:
        PQM_muB = lambda Mu: PQM(Mu, 0, B_SQM, T, m_s)
        PQM_muK = lambda Mu: PQM(muB_Q, Mu, B_SQM, T, m_s)
        ddPQM_ddmuK = Derivative(PQM_muK, n=2)
        dPQM_dmuK = Derivative(PQM_muK, n=1)
        dPQM_dmuB = Derivative(PQM_muB, n=1)
        chiK_Q = ddPQM_ddmuK(0)
        nB_Q = dPQM_dmuB(muB_Q)
        nK_Q = dPQM_dmuK(0)
        aQstar = np.sqrt(2*(muB_N - muB_Q)*chiK_Q / nB_Q)
        muK_star = np.sqrt(2 * nB_Q * (muB_N - muB_Q) / chiK_Q)
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit
        muB_star = fsolve(PQM_solve_for_muBstar, muB_Q)[0]


    # calculating aN
    PNM_wrap = lambda x: PNM(x, T, param, NM_type)
    dPNM_dmuB = Derivative(PNM_wrap, n=1)
    nB_N = dPNM_dmuB(muB_N)
    aN = (nB_N - nK_Q) / nB_Q   

    return aQstar, aN

# Calculate aN and aQstar with input density
def Get_aQstar_eta(T, n_crit, Deln, m_s, param, NM_type, aQmax=True):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - T        : temperature
    - n_crit   : critical density for 1st order phase transition
    - Deln     : Messures how far away from equilibrium
    - m_s      : strange quark mass
    - param    : mean field theory model settings for nuclear matter
    - NM_type  : assumptions for nuclear matter, choose from:
                 - "Beta_eq" for beta equilibriated nuclear matter
                 - "PNM" for pure neutron matter
                 - "SYM" for symmetric nuclear matter
    - aQmax    : whether not to use analytical approximation for calculating aQmax
                 - Default: True

    Returns:
    - aQstar, eta
    '''

    # solve for bag constant
    P_crit = PNM_n(n_crit, T, param, NM_type)
    P_diff = lambda x: PNM(x, T, param, NM_type) - P_crit
    muB_crit = fsolve(P_diff, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit, 0, x, T, m_s) - P_crit
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]

    # solve for points Q and N at beta eq.
    DelP = PNM_n(n_crit + Deln, T, param, NM_type) - P_crit
    PNM_minus_PShift = lambda x: PNM(x, T, param, NM_type) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if aQmax:
        epsilon = np.sqrt(np.finfo(float).eps)
        muB_star, muK_star = _find_muB_muK_star(PQM, P_crit + DelP, muB_N, B_SQM, T, m_s)
        PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_SQM, T, m_s)
        grad_Qstar = approx_fprime(np.array([muB_star, muK_star]), PQM_wrap, epsilon)
        nB_Qstar = grad_Qstar[0]
        nK_Qstar = grad_Qstar[1]
        grad_Q = approx_fprime(np.array([muB_Q, 0]), PQM_wrap, epsilon)
        nB_Q = grad_Q[0]
        nK_Q = grad_Q[1]
        aQstar = (nK_Qstar - nK_Q)/nB_Q  
 
    else:
        PQM_muB = lambda Mu: PQM(Mu, 0, B_SQM, T, m_s)
        PQM_muK = lambda Mu: PQM(muB_Q, Mu, B_SQM, T, m_s)
        ddPQM_ddmuK = Derivative(PQM_muK, n=2)
        dPQM_dmuK = Derivative(PQM_muK, n=1)
        dPQM_dmuB = Derivative(PQM_muB, n=1)
        chiK_Q = ddPQM_ddmuK(0)
        nB_Q = dPQM_dmuB(muB_Q)
        nK_Q = dPQM_dmuK(0)
        aQstar = np.sqrt(2*(muB_N - muB_Q)*chiK_Q / nB_Q)
        muK_star = np.sqrt(2 * nB_Q * (muB_N - muB_Q) / chiK_Q)
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit
        muB_star = fsolve(PQM_solve_for_muBstar, muB_Q)[0]

    eta = (9 * np.pi**2 * T**2) / (muB_star/3)**2

    return aQstar, eta


# Calculate aN and aQstar with input density
def Get_aQstar_eta0(T, n_crit, Deln, m_s, param, NM_type, aQmax=True):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - T        : temperature
    - n_crit   : critical density at T=0 for 1st order phase transition
    - Deln     : Messures how far away from equilibrium
    - m_s      : strange quark mass
    - param    : mean field theory model settings for nuclear matter
    - NM_type  : assumptions for nuclear matter, choose from:
                 - "Beta_eq" for beta equilibriated nuclear matter
                 - "PNM" for pure neutron matter
                 - "SYM" for symmetric nuclear matter
    - aQmax    : whether not to use analytical approximation for calculating aQmax
                 - Default: True

    Returns:
    - aQstar, eta
    '''


    n0 = 0.16 * const.MeV_fm**3

    # solve for bag constant
    P_crit0 = PNM_n(n_crit, Temp=0, param=param, NM_type=NM_type)
    muB_crit0 = fsolve(lambda x: PNM(x, Temp=0, param=param, NM_type=NM_type) - P_crit0, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit0, muK=0, B_one_forth=x, T=0, ms=m_s) - P_crit0
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]

    # solve for critical point
    muB_crit = fsolve(lambda x: PQM(x, 0, B_SQM, T, m_s) - PNM(x, T, param, NM_type), 1050)[0]
    P_crit = PQM(muB_crit, 0, B_SQM, T, m_s)
    n_crit_T = fsolve(lambda x: PNM_n(x, T, param, NM_type) - P_crit, 2*n0)[0]
    DelP = PNM_n(n_crit_T + Deln, T, param, NM_type) - P_crit

    # solve for points Q and N at beta eq.
    PNM_minus_PShift = lambda x: PNM(x, T, param, NM_type) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if aQmax:
        epsilon = np.sqrt(np.finfo(float).eps)
        muB_star, muK_star = _find_muB_muK_star(PQM, P_crit + DelP, muB_N, B_SQM, T, m_s)
        PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_SQM, T, m_s)
        grad_Qstar = approx_fprime(np.array([muB_star, muK_star]), PQM_wrap, epsilon)
        nB_Qstar = grad_Qstar[0]
        nK_Qstar = grad_Qstar[1]
        grad_Q = approx_fprime(np.array([muB_Q, 0]), PQM_wrap, epsilon)
        nB_Q = grad_Q[0]
        nK_Q = grad_Q[1]
        aQstar = (nK_Qstar - nK_Q)/nB_Q  
 
    else:
        PQM_muB = lambda Mu: PQM(Mu, 0, B_SQM, T, m_s)
        PQM_muK = lambda Mu: PQM(muB_Q, Mu, B_SQM, T, m_s)
        ddPQM_ddmuK = Derivative(PQM_muK, n=2)
        dPQM_dmuK = Derivative(PQM_muK, n=1)
        dPQM_dmuB = Derivative(PQM_muB, n=1)
        chiK_Q = ddPQM_ddmuK(0)
        nB_Q = dPQM_dmuB(muB_Q)
        nK_Q = dPQM_dmuK(0)
        aQstar = np.sqrt(2*(muB_N - muB_Q)*chiK_Q / nB_Q)
        muK_star = np.sqrt(2 * nB_Q * (muB_N - muB_Q) / chiK_Q)
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit
        muB_star = fsolve(PQM_solve_for_muBstar, muB_Q)[0] 

    eta = (9 * np.pi**2 * T**2) / (muB_star/3)**2

    return aQstar, eta


# Calculate aN and aQstar with input pressure
def Get_Two_as_pres(T, P_crit, DelP, m_s, param, NM_type, aQmax=True):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - T        : temperature
    - P_crit   : critical pressure for 1st order phase transition
    - DelP     : Messures how far away from equilibrium
    - m_s      : strange quark mass
    - param    : mean field theory model settings for nuclear matter
    - NM_type  : assumptions for nuclear matter, choose from:
                 - "Beta_eq" for beta equilibriated nuclear matter
                 - "PNM" for pure neutron matter
                 - "SYM" for symmetric nuclear matter
    - aQmax    : whether not to use analytical approximation for calculating aQmax
                 - Default: True

    Returns:
    - aQstar, aN
    '''


    # solve for bag constant
    P_diff = lambda x: PNM(x, T, param, NM_type) - P_crit
    muB_crit = fsolve(P_diff, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit, 0, x, T, m_s) - P_crit
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]

    # solve for points Q and N at beta eq.
    PNM_minus_PShift = lambda x: PNM(x, T, param, NM_type) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if aQmax:
        epsilon = np.sqrt(np.finfo(float).eps)
        muB_star, muK_star = _find_muB_muK_star(PQM, P_crit + DelP, muB_N, B_SQM, T, m_s)
        PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_SQM, T, m_s)
        grad_Qstar = approx_fprime(np.array([muB_star, muK_star]), PQM_wrap, epsilon)
        nB_Qstar = grad_Qstar[0]
        nK_Qstar = grad_Qstar[1]
        grad_Q = approx_fprime(np.array([muB_Q, 0]), PQM_wrap, epsilon)
        nB_Q = grad_Q[0]
        nK_Q = grad_Q[1]
        aQstar = (nK_Qstar - nK_Q)/nB_Q  
 
    else:
        PQM_muB = lambda Mu: PQM(Mu, 0, B_SQM, T, m_s)
        PQM_muK = lambda Mu: PQM(muB_Q, Mu, B_SQM, T, m_s)
        ddPQM_ddmuK = Derivative(PQM_muK, n=2)
        dPQM_dmuK = Derivative(PQM_muK, n=1)
        dPQM_dmuB = Derivative(PQM_muB, n=1)
        chiK_Q = ddPQM_ddmuK(0)
        nB_Q = dPQM_dmuB(muB_Q)
        nK_Q = dPQM_dmuK(0)
        aQstar = np.sqrt(2*(muB_N - muB_Q)*chiK_Q / nB_Q)
        muK_star = np.sqrt(2 * nB_Q * (muB_N - muB_Q) / chiK_Q)
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit
        muB_star = fsolve(PQM_solve_for_muBstar, muB_Q)[0]


    # calculating aN
    PNM_wrap = lambda x: PNM(x, T, param, NM_type)
    dPNM_dmuB = Derivative(PNM_wrap, n=1)
    nB_N = dPNM_dmuB(muB_N)
    aN = (nB_N - nK_Q) / nB_Q   

    return aQstar, aN


# Calculate aN and aQstar with input pressure
def Get_Two_as_pres0(T, P_crit0, DelP, m_s, param, NM_type, aQmax=True):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - T        : temperature
    - P_crit   : critical pressure for 1st order phase transition at T=0
    - DelP     : Messures how far away from equilibrium
    - m_s      : strange quark mass
    - param    : mean field theory model settings for nuclear matter
    - NM_type  : assumptions for nuclear matter, choose from:
                 - "Beta_eq" for beta equilibriated nuclear matter
                 - "PNM" for pure neutron matter
                 - "SYM" for symmetric nuclear matter
    - aQmax    : whether not to use analytical approximation for calculating aQmax
                 - Default: True

    Returns:
    - aQstar, aN
    '''


    # solve for bag constant
    muB_crit0 = fsolve(lambda x: PNM(x, Temp=0, param=param, NM_type=NM_type) - P_crit0, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit0, 0, x, 0, m_s) - P_crit0
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]
    muB_crit = fsolve(lambda x: PNM(x, T, param, NM_type) - PQM(x, 0, B_SQM, T, m_s), 1050)[0]
    P_crit = PQM(muB_crit, 0, B_SQM, T, m_s)

    # solve for points Q and N at beta eq.
    PNM_minus_PShift = lambda x: PNM(x, T, param, NM_type) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if aQmax:
        epsilon = np.sqrt(np.finfo(float).eps)
        muB_star, muK_star = _find_muB_muK_star(PQM, P_crit + DelP, muB_N, B_SQM, T, m_s)
        PQM_wrap = lambda Mu: PQM(Mu[0], Mu[1], B_SQM, T, m_s)
        grad_Qstar = approx_fprime(np.array([muB_star, muK_star]), PQM_wrap, epsilon)
        nB_Qstar = grad_Qstar[0]
        nK_Qstar = grad_Qstar[1]
        grad_Q = approx_fprime(np.array([muB_Q, 0]), PQM_wrap, epsilon)
        nB_Q = grad_Q[0]
        nK_Q = grad_Q[1]
        aQstar = (nK_Qstar - nK_Q)/nB_Q  
 
    else:
        PQM_muB = lambda Mu: PQM(Mu, 0, B_SQM, T, m_s)
        PQM_muK = lambda Mu: PQM(muB_Q, Mu, B_SQM, T, m_s)
        ddPQM_ddmuK = Derivative(PQM_muK, n=2)
        dPQM_dmuK = Derivative(PQM_muK, n=1)
        dPQM_dmuB = Derivative(PQM_muB, n=1)
        chiK_Q = ddPQM_ddmuK(0)
        nB_Q = dPQM_dmuB(muB_Q)
        nK_Q = dPQM_dmuK(0)
        aQstar = np.sqrt(2*(muB_N - muB_Q)*chiK_Q / nB_Q)
        muK_star = np.sqrt(2 * nB_Q * (muB_N - muB_Q) / chiK_Q)
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit
        muB_star = fsolve(PQM_solve_for_muBstar, muB_Q)[0]


    # calculating aN
    PNM_wrap = lambda x: PNM(x, T, param, NM_type)
    dPNM_dmuB = Derivative(PNM_wrap, n=1)
    nB_N = dPNM_dmuB(muB_N)
    aN = (nB_N - nK_Q) / nB_Q   

    return aQstar, aN


# Get the distance $z$ between phase boundary and isobar as function of time
def z_time_evolution(Pressure_arr, radius_arr, P_c, T, t_up=20, Interp1d_Num=10, ms=0, para = para.paraQMCRMF3, NM_ty = "PNM", aQmax=True):
    '''
    Get the distance $z$ between phase boundary and isobar as function of time
    ---------------------------
    Parameters:
    - Pressure_arr : an array of pressure profile
    - radius_arr   : an array of radius
    - P_c          : critical pressure for nuclear matter to quark matter phase transition
    - T            : temperature in [MeV]
    - t_up         : upper bound for time window, default = 20 [s]
    - Interp1d_Num : type = int, number of interpolation points, the larger the more precise
                     default = 10
    - ms           : strange quark mass
    - para         : relativistic mean field theory model parameter settings for nuclear matter
    - NM_ty        : nuclear matter type
                     "Beta_eq"  beta equilibriated nuclear matter
                     "PNM"      pure neutron matter
                     "SYM"      symmetrical nuclear matter
    
    --------------------------- 
    Returns:
    - t, z, z0, r_crit
        t      : an array of time
        z      : an array of z, function of t
        z0     : starting point of z when vNtoQ is maximum value
        r_crit : radius at which transition happens
    '''

    # 0) Safety: interpolation needs increasing x
    order = np.argsort(radius_arr)
    radius_arr = np.asarray(radius_arr, float)[order]
    Pressure_arr = np.asarray(Pressure_arr, float)[order]

    # 1) P(r) and locate r_crit from P(r)=P_c
    P_of_r = interp1d(radius_arr, Pressure_arr, kind="linear", fill_value="extrapolate")
    r_crit = float(fsolve(lambda r: P_of_r(r) - P_c, x0=3.0)[0])

    # 2) initial gap z0 from Delta P_max
    DelP_max = Get_Delta_P_max(T, P_c, ms, para, NM_ty, aQmax)
    if P_c+DelP_max < Pressure_arr[0]:
        r_at_Pmax = float(fsolve(lambda r: P_of_r(r) - (P_c + DelP_max), x0=3.0)[0])
        z0 = r_crit - r_at_Pmax
    else:
        z0 = r_crit - radius_arr[0]

    if z0 <= 0:
        raise ValueError(f"Computed z0 <= 0 (z0={z0}); check inputs/units.")

    # 3) Pre-tabulate v(z) on [0, z0] and build cheap interpolant
    Interp1d_Num = max(int(Interp1d_Num), 2)  # need at least 2 points
    z_list = np.linspace(0.0, z0, Interp1d_Num)

    def vz_accurate(z):
        # z can be scalar or array; ensure scalar use here
        z = float(np.asarray(z).reshape(()))
        DelP_P_c = P_of_r(r_crit - z)
        vel, _ = vNtoQ_Pc(T=T, P_crit=P_c, DelP=float(DelP_P_c) - P_c,
                          m_s=ms, param=para, NM_type=NM_ty, aQmax=aQmax)
        return float(vel) / 1e3  # km/s

    v_list = np.array([vz_accurate(z) for z in z_list], dtype=float)

    # Use linear (more stable with few points). Keep cubic only if you have enough points & smoothness
    vz_interp1d = interp1d(z_list, v_list, kind="linear", fill_value="extrapolate")

    # 4) ODE: dz/dt = -v(z). Must accept y as 1-D array and return 1-D array
    def dzdt(t, y):
        z = float(y[0])
        v = float(vz_interp1d(z))
        return np.array([-v], dtype=float)

    # Stop when z hits 0
    def hit_z_target(t, y, z_target=0.0):
        return y[0] - z_target

    hit_z_target.terminal = True
    hit_z_target.direction = 0

    t0, t1 = 0.0, float(t_up)
    t_eval = np.linspace(t0, t1, 500)

    # y0 must be 1-D:
    y0 = np.array([z0], dtype=float)

    try:
        sol = solve_ivp(
            dzdt, (t0, t1), y0,
            method="RK45",
            t_eval=t_eval,
            events=lambda t, y: hit_z_target(t, y, z_target=0.0),
            rtol=1e-7, atol=1e-9
        )
    except Exception as e:
        raise RuntimeError(f"solve_ivp failed with error: {e}") from e

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Return arrays directly
    t = np.array(sol.t, dtype=float)                   # shape (M,)
    z = np.array(sol.y[0], dtype=float)                # shape (M,)
    return t, z, z0, r_crit


def z_time_evolution1(Pressure_arr, radius_arr, P_kink, T, z0 = 1, t_up=20, Interp1d_Num=10, ms=0, para = para.paraQMCRMF3, NM_ty = "PNM", aQmax=True):
    '''
    Get the distance $z$ between phase boundary and isobar as function of time
    ---------------------------
    Parameters:
    - Pressure_arr : an array of pressure profile
    - radius_arr   : an array of radius
    - P_kink       : pressure at the phase boundary
    - T            : temperature in [MeV]
    - z0           : starting distance between phase boundary and isobar, default 1, unit km
    - t_up         : upper bound for time window, default = 20 [s]
    - Interp1d_Num : type = int, number of interpolation points, the larger the more precise
                     default = 10
    - ms           : strange quark mass
    - para         : relativistic mean field theory model parameter settings for nuclear matter
    - NM_ty        : nuclear matter type
                     "Beta_eq"  beta equilibriated nuclear matter
                     "PNM"      pure neutron matter
                     "SYM"      symmetrical nuclear matter
    
    --------------------------- 
    Returns:
    - t, z, z0, r_crit
        t      : an array of time [s]
        z      : an array of z [km], function of t 
        P_c    : critical pressure [MeV^4]
    '''

    # 0) Safety: interpolation needs increasing x
    order = np.argsort(radius_arr)
    radius_arr = np.asarray(radius_arr, float)[order]
    Pressure_arr = np.asarray(Pressure_arr, float)[order]

    # 1) P(r) and locate r_crit from P(r)=P_c
    P_of_r = interp1d(radius_arr, Pressure_arr, kind="linear", fill_value="extrapolate")
    r_kink = float(fsolve(lambda r: P_of_r(r) - P_kink, x0=3.0)[0])

    # 2) calculate P_critical. We define it to be whatever pressure 1km away from the kink
    P_c = P_of_r(r_kink + z0)
    DelP_max = P_kink - P_c

    # 3) Pre-tabulate v(z) on [0, z0] and build cheap interpolant
    Interp1d_Num = max(int(Interp1d_Num), 2)  # need at least 2 points
    z_list = np.logspace(-8, 0, Interp1d_Num)

    def vz_accurate(z):
        # z can be scalar or array; ensure scalar use here
        z = float(np.asarray(z).reshape(()))
        DelP_P_c = P_of_r(r_kink + z0 - z)
        vel, _ = vNtoQ_Pc(T=T, P_crit=P_c, DelP=float(DelP_P_c) - P_c,
                          m_s=ms, param=para, NM_type=NM_ty, aQmax=aQmax)
        return float(vel) / 1e3  # km/s

    v_list = np.array([vz_accurate(z) for z in z_list], dtype=float)

    # Use linear (more stable with few points). Keep cubic only if you have enough points & smoothness
    vz_interp1d = interp1d(z_list, v_list, kind="linear", fill_value="extrapolate")

    # 4) ODE: dz/dt = -v(z). Must accept y as 1-D array and return 1-D array
    def dzdt(t, y):
        z = float(y[0])
        v = float(vz_interp1d(z))
        return np.array([-v], dtype=float)

    # Stop when z hits 0
    def hit_z_target(t, y, z_target=0.0):
        return y[0] - z_target

    hit_z_target.terminal = True
    hit_z_target.direction = 0

    t0, t1 = 0.0, float(t_up)
    t_eval = np.linspace(t0, t1, 500)

    # y0 must be 1-D:
    y0 = np.array([z0], dtype=float)

    try:
        sol = solve_ivp(
            dzdt, (t0, t1), y0,
            method="RK45",
            t_eval=t_eval,
            events=lambda t, y: hit_z_target(t, y, z_target=0.0),
            rtol=1e-7, atol=1e-9
        )
    except Exception as e:
        raise RuntimeError(f"solve_ivp failed with error: {e}") from e

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Return arrays directly
    t = np.array(sol.t, dtype=float)                   # shape (M,)
    z = np.array(sol.y[0], dtype=float)                # shape (M,)
    return t, z, P_c


def z_time_evolution2(dP_over_dr, n_crit, T, z0 = 1, t_up=20, Interp1d_Num=10, m_s=0, param = para.paraQMCRMF3, NM_type = "PNM", aQmax=True):
    '''
    Get the distance $z$ between phase boundary and isobar as function of time
    ---------------------------
    Parameters:
    - dP_over_dr   : absolute value of typical dP/dr value in units MeV^4/km
    - n_crit       : critical density at T=0
    - T            : temperature in [MeV]
    - z0           : starting distance between phase boundary and isobar, default 1, unit km
    - t_up         : upper bound for time window, default = 20 [s]
    - Interp1d_Num : type = int, number of interpolation points, the larger the more precise
                     default = 10
    - ms           : strange quark mass
    - param        : relativistic mean field theory model parameter settings for nuclear matter
    - NM_type      : nuclear matter type
                     "Beta_eq"  beta equilibriated nuclear matter
                     "PNM"      pure neutron matter
                     "SYM"      symmetrical nuclear matter
    
    --------------------------- 
    Returns:
    - t, z, z0, r_crit
        t      : an array of time [s]
        z      : an array of z [km], function of t 
        P_c    : critical pressure [MeV^4]
    '''

    n0 = 0.16 * const.MeV_fm**3

    # 1) calculate P_critical. We define it to be whatever pressure 1km away from the kink
    P_crit0 = PNM_n(n_crit, Temp=0, param=param, NM_type=NM_type)
    muB_crit0 = fsolve(lambda x: PNM(x, Temp=0, param=param, NM_type=NM_type) - P_crit0, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit0, muK=0, B_one_forth=x, T=0, ms=m_s) - P_crit0
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]

    muB_crit = fsolve(lambda x: PQM(x, 0, B_SQM, T, m_s) - PNM(x, T, param, NM_type), 1050)[0]
    P_c = PQM(muB_crit, 0, B_SQM, T, m_s)

    # 2) P(r) and locate r_crit from P(r)=P_c
    P_of_r = lambda r: P_c + dP_over_dr * (z0 - r)
    DelP_max = P_of_r(0)

    # 3) Pre-tabulate v(z) on [0, z0] and build cheap interpolant
    Interp1d_Num = max(int(Interp1d_Num), 2)  # need at least 2 points
    z_list = np.logspace(-8, 0, Interp1d_Num)

    def vz_accurate(z):
        # z can be scalar or array; ensure scalar use here
        z = float(np.asarray(z).reshape(()))
        DelP_P_c = P_of_r(z0 - z)
        vel, _ = vNtoQ_Pc(T=T, P_crit=P_c, DelP=float(DelP_P_c) - P_c,
                          m_s=m_s, param=param, NM_type=NM_type, aQmax=aQmax)
        return float(vel) / 1e3  # km/s

    v_list = np.array([vz_accurate(z) for z in z_list], dtype=float)

    # Use linear (more stable with few points). Keep cubic only if you have enough points & smoothness
    vz_interp1d = interp1d(z_list, v_list, kind="linear", fill_value="extrapolate")

    # 4) ODE: dz/dt = -v(z). Must accept y as 1-D array and return 1-D array
    def dzdt(t, y):
        z = float(y[0])
        v = float(vz_interp1d(z))
        return np.array([-v], dtype=float)

    # Stop when z hits 0
    def hit_z_target(t, y, z_target=0.0):
        return y[0] - z_target

    hit_z_target.terminal = True
    hit_z_target.direction = 0

    t0, t1 = 0.0, float(t_up)
    t_eval = np.linspace(t0, t1, 5000)

    # y0 must be 1-D:
    y0 = np.array([z0], dtype=float)

    try:
        sol = solve_ivp(
            dzdt, (t0, t1), y0,
            method="RK45",
            t_eval=t_eval,
            events=lambda t, y: hit_z_target(t, y, z_target=0.0),
            rtol=1e-7, atol=1e-9
        )
    except Exception as e:
        raise RuntimeError(f"solve_ivp failed with error: {e}") from e

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Return arrays directly
    t = np.array(sol.t, dtype=float)                   # shape (M,)
    z = np.array(sol.y[0], dtype=float)                # shape (M,)
    return t, z


def z_time_evolution3(dn_over_dr, n_crit0, T, z0 = 1, t_up=20, Interp1d_Num=10, m_s=0, param = para.paraQMCRMF3, NM_type = "PNM", aQmax=True):
    '''
    Get the distance $z$ between phase boundary and isobar as function of time
    ---------------------------
    Parameters:
    - dn_over_dr   : absolute value of typical dn/dr value in units of MeV^3/km
    - n_crit0      : critical density at T=0
    - T            : temperature in [MeV]
    - z0           : starting distance between phase boundary and isobar, default 1, unit km
    - t_up         : upper bound for time window, default = 20 [s]
    - Interp1d_Num : type = int, number of interpolation points, the larger the more precise
                     default = 10
    - ms           : strange quark mass
    - param        : relativistic mean field theory model parameter settings for nuclear matter
    - NM_type      : nuclear matter type
                     "Beta_eq"  beta equilibriated nuclear matter
                     "PNM"      pure neutron matter
                     "SYM"      symmetrical nuclear matter
    
    --------------------------- 
    Returns:
    - t, z, z0, r_crit
        t      : an array of time [s]
        z      : an array of z [km], function of t 
        P_c    : critical pressure [MeV^4]
    '''

    n0 = 0.16 * const.MeV_fm**3

    # 1) calculate P_critical. We define it to be whatever pressure 1km away from the kink
    P_crit0 = PNM_n(n_crit0, Temp=0, param=param, NM_type=NM_type)
    muB_crit0 = fsolve(lambda x: PNM(x, Temp=0, param=param, NM_type=NM_type) - P_crit0, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit0, muK=0, B_one_forth=x, T=0, ms=m_s) - P_crit0
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]

    muB_crit = fsolve(lambda x: PQM(x, 0, B_SQM, T, m_s) - PNM(x, T, param, NM_type), 1050)[0]
    P_crit = PQM(muB_crit, 0, B_SQM, T, m_s)
    n_crit = fsolve(lambda x: PNM_n(x, T, param, NM_type) - P_crit, 3*n0)[0]

    # 2) P(r) and locate r_crit from P(r)=P_c
    n_of_r = lambda r: n_crit + dn_over_dr * (z0 - r)
    Deln_max = n_of_r(0)

    # 3) Pre-tabulate v(z) on [0, z0] and build cheap interpolant
    Interp1d_Num = max(int(Interp1d_Num), 2)  # need at least 2 points
    z_list = np.logspace(-8, 0, Interp1d_Num)

    def vz_accurate(z):
        # z can be scalar or array; ensure scalar use here
        z = float(np.asarray(z).reshape(()))
        Deln_n_c = n_of_r(z0 - z)
        vel, _ = vNtoQ_nc(T=T, n_crit=n_crit, Deln=float(Deln_n_c) - n_crit,
                          m_s=m_s, param=param, NM_type=NM_type, aQmax=aQmax)
        return float(vel) / 1e3  # km/s

    v_list = np.array([vz_accurate(z) for z in z_list], dtype=float)

    # Use linear (more stable with few points). Keep cubic only if you have enough points & smoothness
    vz_interp1d = interp1d(z_list, v_list, kind="linear", fill_value="extrapolate")

    # 4) ODE: dz/dt = -v(z). Must accept y as 1-D array and return 1-D array
    def dzdt(t, y):
        z = float(y[0])
        v = float(vz_interp1d(z))
        return np.array([-v], dtype=float)

    # Stop when z hits 0
    def hit_z_target(t, y, z_target=0.0):
        return y[0] - z_target

    hit_z_target.terminal = True
    hit_z_target.direction = 0

    t0, t1 = 0.0, float(t_up)
    t_eval = np.linspace(t0, t1, 5000)

    # y0 must be 1-D:
    y0 = np.array([z0], dtype=float)

    try:
        sol = solve_ivp(
            dzdt, (t0, t1), y0,
            method="RK45",
            t_eval=t_eval,
            events=lambda t, y: hit_z_target(t, y, z_target=0.0),
            rtol=1e-7, atol=1e-9
        )
    except Exception as e:
        raise RuntimeError(f"solve_ivp failed with error: {e}") from e

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Return arrays directly
    t = np.array(sol.t, dtype=float)                   # shape (M,)
    z = np.array(sol.y[0], dtype=float)                # shape (M,)
    return t, z

#
