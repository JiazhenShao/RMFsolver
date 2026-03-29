"""Phase-transition and combustion-front helpers built on top of the RMF solvers."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad, solve_bvp, solve_ivp
from scipy.optimize import fsolve, approx_fprime, root_scalar, root, least_squares, minimize_scalar
from numdifftools import Derivative

from . import RMFparameter as para
from . import constants as const
from .Solver import RMFbaryon_densityPNM, RMFedensPNM, RMFpressurePNM, RMFpressureSYM, RMFsolve, RMFsolve_mu, pressure_RMF


# Public functions
__all__ = ["P_f", "E_f", "n_B", "nB_QM", "nQM_em", "PQM", "PQM_em", "edensQM", "uQ_uN", "vNtoQ_Pc", "vNtoQ_B", "vNtoQ_nc", "Get_Delta_n_max",
           "Get_Delta_P_max", "extract_contour_coords_num", "extract_contour_coords_ana",
           "Get_Two_as_pres", "Get_Two_as_dens", "z_time_evolution", "z_time_evolution1",
           "vNtoQ_fixB", "vNtoQ_Pc_fixB", "z_time_evolution2", "Get_aQstar_etaQ", "Get_aQstar_etaQ0",
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
    """Return nuclear-matter energy density as a function of ``mu_B``.

    Inputs:
    - mu_B: baryon chemical potential in MeV
    - Temp: temperature in MeV
    - param: RMF parameter set

    Output:
    - energy density in MeV^4
    """
    edens = RMFedensPNM(input_num = mu_B, input_type = "muB", Trmf = Temp, para = param, 
        sigma_init = 30, w0_init = 20, r03_init = -3, mub_init = 990, verb = False
        )
    return float(edens.item())

def edensNM_n(nB, Temp, param = para.paraQMCRMF3, ):
    """Return nuclear-matter energy density as a function of baryon density.

    Inputs:
    - nB: baryon density in MeV^3
    - Temp: temperature in MeV
    - param: RMF parameter set

    Output:
    - energy density in MeV^4
    """
    edens = RMFedensPNM(input_num = nB, input_type = "nB", Trmf = Temp, para = param, 
        sigma_init = 30, w0_init = 20, r03_init = -3, mub_init = 990, verb = False
        )
    return float(edens.item())

def hNM(mu_B, Temp):
    """Return nuclear-matter enthalpy density ``epsilon + P`` in MeV^4."""
    return PNM(mu_B, Temp) + edensNM(mu_B, Temp)

def hNM_n(nB, Temp):
    """Return nuclear-matter enthalpy density from baryon density input."""
    return PNM_n(nB, Temp) + edensNM_n(nB, Temp)

def nB_NM(mu_B, Temp, param = para.paraQMCRMF3, ):
    """Return nuclear-matter baryon density as a function of ``mu_B``.

    Inputs:
    - mu_B: baryon chemical potential in MeV
    - Temp: temperature in MeV
    - param: RMF parameter set

    Output:
    - baryon density in MeV^3
    """
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
    """Return the nuclear-matter momentum flux ``Pi`` for a baryon current.

    Inputs:
    - mu_B: baryon chemical potential in MeV
    - Temp: temperature in MeV
    - j_B: baryon current in MeV^3

    Output:
    - momentum-flux invariant in MeV^4
    """
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
    Pressure of a single free fermion (particles + antiparticles).

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

    # ---- Sommerfeld mode (fast) ----
    if Sommerfeld:
        if Tem < 0.0:
            raise RuntimeError("Negative Temperature")

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
    if Tem < 0:
        raise RuntimeError("Negative Temperature")
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
    Energy density of a single free fermion (particles + antiparticles).
    Uses the same numerical strategy as P_f: T=0 exact branch + adaptive blend.
    """
    mu = float(mu)
    m = float(m)
    Tem = float(Tem)

    def _E0():
        if mu <= m:
            return 0.0
        if m == 0.0:
            return mu**4 / (4.0 * np.pi**2)
        kF = _kF(mu, m)
        return ((2.0 * kF**3 + m*m*kF) * mu - m**4 * np.log((kF + mu) / max(m, 1e-300))) / (8.0 * np.pi**2)

    if Sommerfeld:
        if Tem < 0.0:
            raise RuntimeError("Negative Temperature")
        if mu <= m:
            return 0.0
        if m == 0.0:
            corr = 0.5 * mu * mu * Tem * Tem
        else:
            corr = 0.5 * mu * _kF(mu, m) * Tem * Tem
        return _E0() + corr

    if Tem < 0.0:
        raise RuntimeError("Negative Temperature")
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

# number density for a single free fermion
def n_B(mu, m, Tem, upB=5000):
    '''
    Returns number density for a single fermion species.
    Uses thermodynamic relation: dP/dmu = n
    '''

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
    return float(
        3.0 * (
            P_f(mu_u, m=0.0, Tem=T, upB=upB)
            + P_f(mu_d, m=0.0, Tem=T, upB=upB)
            + P_f(mu_s, m=ms, Tem=T, upB=upB)
        )
    )

def _quark_edens_uds(mu_u, mu_d, mu_s, T, ms, upB=5000):
    return float(
        3.0 * (
            E_f(mu_u, m=0.0, Tem=T, upB=upB)
            + E_f(mu_d, m=0.0, Tem=T, upB=upB)
            + E_f(mu_s, m=ms, Tem=T, upB=upB)
        )
    )

def _quark_density_uds(mu_u, mu_d, mu_s, T, ms, upB=5000):
    n_u = n_B(mu_u, 0.0, T, upB=upB)
    n_d = n_B(mu_d, 0.0, T, upB=upB)
    n_s = n_B(mu_s, ms, T, upB=upB)
    n_tot = float((n_u + n_d + n_s) / 3.0)
    return n_tot, {"n_u": float(n_u), "n_d": float(n_d), "n_s": float(n_s)}


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
    '''

    B = B_one_forth**4
    mu_u, mu_d, mu_s = _quark_mu_triplet(muB, muK)
    return _quark_pressure_uds(mu_u, mu_d, mu_s, T, ms, upB=upB) - B

def nB_QM(muB, muK, B_one_forth, T, ms=0, upB=5000, return_species=False):
    """
    Baryon number density of quark matter (u,d,s) in the bag-model setup.
    Note: B_one_forth is accepted for API parity with PQM but does not affect density.

    Returns:
    - nB_QM = (n_u + n_d + n_s)/3
    - if return_species=True, also return species dictionary
    """
    mu_u, mu_d, mu_s = _quark_mu_triplet(muB, muK)
    n_tot, species = _quark_density_uds(mu_u, mu_d, mu_s, T, ms, upB=upB)

    if return_species:
        return n_tot, species
    return n_tot

def edensQM(muB, muK, B_one_forth, T, ms=0, include_em=False, muQ_init=300, upB=5000):
    """
    Energy density of strange quark matter under the bag model.

    Parameters:
    - include_em=False : match PQM composition (u,d,s quarks + bag constant)
    - include_em=True  : match PQM_em composition (charge-neutral u,d,s,e + thermal gauge term + bag constant)
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

def hQM(muB, muK, B_one_forth, Temp):
    """Return quark-matter enthalpy density ``epsilon + P`` in MeV^4."""
    return PQM(muB, muK, B_one_forth, T=Temp, ms=0, upB=5000) + edensQM(muB, muK, B_one_forth, T=Temp, ms=0, include_em=False, muQ_init=300, upB=5000)

def Pi_QM(mu_B, mu_K, B_one_forth, Temp, j_B):
    """Return the quark-matter momentum flux ``Pi`` for a baryon current.

    Inputs:
    - mu_B: baryon chemical potential in MeV
    - mu_K: charge/isospin chemical potential in MeV
    - B_one_forth: bag-constant fourth root in MeV
    - Temp: temperature in MeV
    - j_B: baryon current in MeV^3

    Output:
    - momentum-flux invariant in MeV^4
    """
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
        return float(
            n_B(mu_e, 0, T, upB=upB)
            + n_B(mu_s, ms, T, upB=upB) / 3.0
            + n_B(mu_d, 0, T, upB=upB) / 3.0
            - 2.0 * n_B(mu_u_val, 0, T, upB=upB) / 3.0
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
    - nB_QM = (n_u + n_d + n_s)/3
    - if return_species=True, also return species dictionary including n_e
    """
    mu_u, mu_d, mu_s, mu_e = _solve_quark_mu_em(muB, muK, T, ms, muQ_init=muQ_init, upB=upB)
    n_tot, species = _quark_density_uds(mu_u, mu_d, mu_s, T, ms, upB=upB)
    species["n_e"] = float(n_B(mu_e, 0.0, T, upB=upB))

    if return_species:
        return n_tot, species
    return n_tot


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
        _, mu_d, mu_s = _quark_mu_triplet(muB, muK)

        n_d = n_B(mu_d, 0, T, upB)
        n_s = n_B(mu_s, ms, T, upB)

        return 0.5 * (n_d - n_s)

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
    etaQ = (9 * np.pi**2 * T**2) / muQ**2
    tauQ = 1.98e12 * ((300 / muQ)**5)
    qD = np.sqrt(3 * g_s**2 * muQ**2 / (2 * np.pi**2))
    h = 1.81317

    part1 = h*T**(5/3)/qD**(2/3)
    part2 = np.pi**3 * T**2 / (12*qD)

    DQ = 1/( 24*alpha_s**2/np.pi * ( part1 + part2 ) )
    # DQ = (np.pi / (24 * (0.3)**2 * 1.81 * T**(5/3))) * ((6 * 0.3 / np.pi * muQ**2)**(1/3))

    # Compute v_N->Q
    if aN < aQstar:
        print("hit aN < aQstar, returning velocity = 0")
        return 0.0
    else:
        return np.sqrt((DQ / tauQ) * ((aQstar**4 + 2 * etaQ * aQstar**2) / (2 * aN * (aN - aQstar)))) * 3e8

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
    etaQ  = (9*np.pi**2 * T**2) / muQ**2
    tauQ  = 1.98e12 * ((300.0 / muQ)**5)

    qD    = np.sqrt(3 * g_s**2 * muQ**2 / (2 * np.pi**2))
    h     = 1.81317
    part1 = h * T**(5/3) / qD**(2/3)
    part2 = (np.pi**3 * T**2) / (12.0 * qD)
    DQ    = 1.0 / ((24.0 * alpha_s**2 / np.pi) * (part1 + part2))

    # kappa(v) from the linearized tail (η_Q must be present!)
    def get_kappa(v):
        return 2 * DQ / (-v + np.sqrt(v*v + 4 * DQ * etaQ / tauQ))

    # Nonlinearity
    def R(a):
        a_safe = np.clip(a, -7e2, 7e2)
        return (a_safe**3 + etaQ * a_safe) / tauQ

    # ODE in t = \tilde{x}: y[0]=a(t), y[1]=p(t)=da/dx
    def ode_system(t, y, v, kappa):
        a = y[0]
        p = y[1]
        dxdt = kappa / (c1 - t)                # dx/dt
        a_t  = p * dxdt
        p_t  = ((v * p + R(a)) / DQ) * dxdt
        return np.vstack((a_t, p_t))

    # Boundary conditions: a(0)=aQ*, a(c1)=0
    def bc(ya, yb):
        return np.array([ya[0] - aQstar, yb[0]])

    # Solve BVP for a given v and return a'(x=0)=p(0)
    def get_derivatives_at_zero(v):
        v = float(abs(v))
        kappa = get_kappa(v)
        t = np.linspace(0, c1 - 1e-9, 1000)

        # Smooth monotone initial guess
        a_guess = aQstar * np.exp(-5 * t / c1)
        # p = a' = a_t / (dt/dx), with dt/dx = (c1 - t)/kappa
        p_guess = -(5 * aQstar / c1) * np.exp(-5 * t / c1) * (c1 - t) / kappa
        y_init  = np.vstack((a_guess, p_guess))

        try:
            sol = solve_bvp(lambda tt, yy: ode_system(tt, yy, v, kappa),
                            bc, t, y_init, max_nodes=100000)
            if sol.status != 0:
                return None
            return sol.y[1, 0]   # p(0) = a'(x=0)
        except Exception:
            return None

    # Residual for remaining BC at x=0: a'(0+) = - v (aN - aQ*) / DQ
    def slope_residual(v_arr):
        v = float(abs(v_arr[0]))          # root() passes arrays
        a_prime_x0 = get_derivatives_at_zero(v)
        if a_prime_x0 is None or not np.isfinite(a_prime_x0):
            return np.array([1e20], dtype=float)   # 1-element array
        rhs = -v * (aN - aQstar) / DQ
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
    C_lin=1e3,            # demand a(L)^2 <= etaQ/C_lin (larger = stricter)
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
      ∂_x y1 = (v y1 + R(a)) / DQ

    BCs:
      a(0) = aQstar
      ∂_x a(0) = - v (aN-aQstar)/DQ
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
    etaQ  = (9*np.pi**2 * T**2) / (muQ**2)
    tauQ  = 1.98e12 * ((300.0 / muQ)**5)

    qD    = np.sqrt(3 * g_s**2 * muQ**2 / (2 * np.pi**2))
    h     = 1.81317
    part1 = h * T**(5/3) / qD**(2/3)
    part2 = (np.pi**3 * T**2) / (12.0 * qD)
    DQ    = 1.0 / ((24.0 * alpha_s**2 / np.pi) * (part1 + part2))

    def R(a):
        a_safe = np.clip(a, -7e2, 7e2)
        return (a_safe**3 + etaQ * a_safe) / tauQ

    def lambda_minus(v):
        v = float(abs(v))
        disc = v*v + 4.0 * DQ * etaQ / tauQ
        return (v - np.sqrt(disc)) / (2.0 * DQ)  # < 0

    def pick_L(v_guess):
        lam = lambda_minus(v_guess)
        if not np.isfinite(lam) or lam >= 0.0:
            raise RuntimeError("Bad lambda_- from v_guess; check parameters.")

        L_tail = float(Ntail / abs(lam))

        # amplitude target to ensure etaQ*a >> a^3 at x=L
        a_lin = np.sqrt(max(etaQ / C_lin, 1e-300))

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
        return np.vstack((px, (v * px + R(a)) / DQ))

    def bc(ya, yb, p):
        v = float(abs(p[0]))
        lam = lambda_minus(v)
        return np.array([
            ya[0] - aQstar,
            ya[1] + v * (aN - aQstar) / DQ,
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
    - aQstar, aN, etaQ
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
def Get_aQstar_etaQ(T, n_crit, Deln, m_s, param, NM_type, aQmax=True):
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

    etaQ = (9 * np.pi**2 * T**2) / (muB_star/3)**2

    return aQstar, etaQ


# Calculate aN and aQstar with input density
def Get_aQstar_etaQ0(T, n_crit, Deln, m_s, param, NM_type, aQmax=True):
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
    - aQstar, aN, etaQ
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

    etaQ = (9 * np.pi**2 * T**2) / (muB_star/3)**2

    return aQstar, etaQ


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
