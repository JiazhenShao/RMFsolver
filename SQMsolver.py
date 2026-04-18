"""Quark-matter thermodynamics helpers for the RMFsolver package."""

import numpy as np
from functools import lru_cache
from scipy.integrate import quad, quad_vec
from scipy.optimize import fsolve

__all__ = [
    "PQM_bag",
    "P_f",
    "E_f",
    "n_B",
    "nB_QM",
    "nK_QM",
    "nQM_em",
    "PQM",
    "PQM_em",
    "edensQM",
    "entropyQM",
]


def _P_f(mu, m):
    """Return the zero-temperature free-fermion pressure in MeV^4."""
    if mu <= m:
        return 0.0
    k_F = np.sqrt(mu**2 - m**2)
    return ((2 * k_F**3 - 3 * m**2 * k_F) * mu + 3 * m**4 * np.log((k_F + mu) / m)) / (24 * np.pi**2)


def PQM_bag(muB, muK, bag_constant, m_s=0.0):
    """Return bag-model quark pressure at T=0."""
    mu_u = muB / 3 + muK / 2
    mu_d = muB / 3
    mu_s = muB / 3 - muK / 2
    return (mu_u**4 + mu_d**4) * 3 / (12 * np.pi**2) + 3 * _P_f(mu_s, m_s) - bag_constant


def _Ek(k, m):
    return np.sqrt(k * k + m * m)


def _kF(mu, m):
    dm = mu * mu - m * m
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
    return x * x * (3.0 - 2.0 * x)


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


def Tswitch_free_pressure(mu, m, rtol=1e-6, band=1.25):
    """
    Return (T_lo, T_hi) to blend T=0 pressure with the finite-T integral.

    Logic:
    - If mu <= m (non-degenerate): use integral only.
    - If mu > m (degenerate): set switch where O(T^2) term changes P by ≲ rtol*P0.
    """
    mu = float(mu)
    m = float(m)

    if mu <= m:
        return (0.0, 0.0)

    if m == 0.0:
        P0 = mu**4 / (12.0 * np.pi**2)
        f = (mu * mu) / (np.pi**2)
    else:
        kF = _kF(mu, m)
        P0 = ((2.0 / 3.0 * kF**3 - m * m * kF) * mu + m**4 * np.log((kF + mu) / max(m, 1e-300))) / (8.0 * np.pi**2)
        f = (mu * kF) / (np.pi**2)

    if P0 <= 0.0 or f <= 0.0:
        return (0.0, 0.0)

    T_star = np.sqrt((6.0 * rtol * P0) / (np.pi**2 * f))
    return (T_star / band, T_star * band)


def P_f(mu, m, Tem, upB=np.inf, rtol=1e-6, Sommerfeld=False):
    """
    Pressure of a single free fermion flavor (particles + antiparticles).
    The normalization already includes spin degeneracy g=2, but not quark color.
    """
    mu = float(mu)
    m = float(m)
    Tem = float(Tem)
    if Tem < 0.0:
        raise RuntimeError("Negative Temperature")
    if m == 0.0:
        return _massless_fermion_pressure(mu, Tem)

    if Sommerfeld:
        if mu <= m:
            P0 = 0.0
        elif m == 0.0:
            P0 = mu**4 / (12.0 * np.pi**2)
        else:
            kF = np.sqrt(max(mu * mu - m * m, 0.0))
            P0 = ((2.0 * kF**3 - 3.0 * m * m * kF) * mu + 3.0 * m**4 * np.log((kF + mu) / max(m, 1e-300))) / (24.0 * np.pi**2)

        if mu <= m:
            fP = 0.0
        elif m == 0.0:
            fP = (mu * mu) / (np.pi**2)
        else:
            kF = np.sqrt(max(mu * mu - m * m, 0.0))
            fP = (mu * kF) / (np.pi**2)

        return P0 + (np.pi**2 / 6.0) * (Tem * Tem) * fP

    if Tem == 0.0:
        if mu <= m:
            return 0.0
        if m == 0.0:
            return mu**4 / (12.0 * np.pi**2)
        kF = _kF(mu, m)
        return ((2.0 * kF**3 - 3.0 * m * m * kF) * mu + 3.0 * m**4 * np.log((kF + mu) / max(m, 1e-300))) / (24.0 * np.pi**2)

    T_lo, T_hi = Tswitch_free_pressure(mu, m, rtol=rtol, band=1.25)
    use_blend = (T_lo < Tem < T_hi) if T_hi != 0.0 else False

    if not use_blend and Tem <= T_lo:
        if mu <= m:
            return 0.0
        if m == 0.0:
            return mu**4 / (12.0 * np.pi**2)
        kF = _kF(mu, m)
        return ((2.0 * kF**3 - 3.0 * m * m * kF) * mu + 3.0 * m**4 * np.log((kF + mu) / max(m, 1e-300))) / (24.0 * np.pi**2)

    def _P_int(Tloc):
        def integrand(k):
            Ek = _Ek(k, m)
            a = (Ek - mu) / Tloc
            b = (Ek + mu) / Tloc
            return Tloc * (_log1p_exp_neg(a) + _log1p_exp_neg(b)) * (k * k)

        ub = np.inf if (upB is None or not np.isfinite(float(upB))) else float(upB)
        Pint, _ = quad(integrand, 0.0, ub, epsabs=1e-10, epsrel=1e-8, limit=200)
        return Pint / (np.pi**2)

    if not use_blend and Tem >= T_hi:
        return _P_int(Tem)

    if mu <= m:
        P0 = 0.0
    elif m == 0.0:
        P0 = mu**4 / (12.0 * np.pi**2)
    else:
        kF = _kF(mu, m)
        P0 = ((2.0 * kF**3 - 3.0 * m * m * kF) * mu + 3.0 * m**4 * np.log((kF + mu) / max(m, 1e-300))) / (24.0 * np.pi**2)

    P_int = _P_int(Tem)
    x = (np.log(Tem) - np.log(T_lo)) / (np.log(T_hi) - np.log(T_lo))
    w = _smoothstep01(x)
    return (1.0 - w) * P0 + w * P_int


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
        return ((2.0 * kF**3 + m * m * kF) * mu - m**4 * np.log((kF + mu) / max(m, 1e-300))) / (8.0 * np.pi**2)

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
    use_blend = (T_lo < Tem < T_hi) if T_hi > 0.0 else False

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


def n_B(mu, m, Tem, upB=5000):
    """
    Returns number density for a single fermion species.
    The normalization already includes spin degeneracy g=2, but not quark color.
    Uses thermodynamic relation: dP/dmu = n
    """
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
            f = 1 / (1 + np.exp(np.clip((Ek - mu) / Tem, -700, 700)))
            f_bar = 1 / (1 + np.exp(np.clip((Ek + mu) / Tem, -700, 700)))
            return (f - f_bar) * k**2

        integral, _ = quad(integrand, 0, upB, epsabs=1e-10, epsrel=1e-8)
        return float(integral / (np.pi**2))

    if mu > m:
        k_F = np.sqrt(np.maximum(mu**2 - m**2, 0.0))
        return float(k_F**3 / (3 * np.pi**2))
    return float(0.0)


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
        return k * k * (_binary_entropy_density_term(f) + _binary_entropy_density_term(fbar))

    ub = np.inf if (upB is None or not np.isfinite(float(upB))) else float(upB)
    Sint, _ = quad(integrand, 0.0, ub, epsabs=1e-10, epsrel=1e-8, limit=200)
    return float(Sint / (np.pi**2))


def _normalize_upB(upB):
    if upB is None:
        return float(np.inf)
    upB = float(upB)
    return float(np.inf) if not np.isfinite(upB) else upB


@lru_cache(maxsize=16384)
def _fermion_thermo_state_cached(mu, m, Tem, upB):
    """
    Return (P, E, n, s, chi) for a single fermion species with spin
    degeneracy included and color excluded.
    """
    mu = float(mu)
    m = float(m)
    Tem = float(Tem)
    upB = _normalize_upB(upB)

    if Tem < 0.0:
        raise RuntimeError("Negative Temperature")

    if m == 0.0:
        pressure = _massless_fermion_pressure(mu, Tem)
        energy = 3.0 * pressure
        density = _massless_fermion_density(mu, Tem)
        entropy = _massless_fermion_entropy(mu, Tem)
        susceptibility = float(mu * mu / (np.pi**2) + Tem * Tem / 3.0)
        return (pressure, energy, density, entropy, susceptibility)

    def _massive_zeroT_state():
        if mu <= m:
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        kF = _kF(mu, m)
        pressure = ((2.0 * kF**3 - 3.0 * m * m * kF) * mu + 3.0 * m**4 * np.log((kF + mu) / max(m, 1e-300))) / (24.0 * np.pi**2)
        energy = ((2.0 * kF**3 + m * m * kF) * mu - m**4 * np.log((kF + mu) / max(m, 1e-300))) / (8.0 * np.pi**2)
        density = kF**3 / (3.0 * np.pi**2)
        entropy = 0.0
        susceptibility = mu * kF / (np.pi**2)
        return (float(pressure), float(energy), float(density), float(entropy), float(susceptibility))

    if Tem == 0.0:
        return _massive_zeroT_state()

    T_lo, T_hi = Tswitch_free_pressure(mu, m, rtol=1e-6, band=1.25)
    use_blend = (T_lo < Tem < T_hi) if T_hi > 0.0 else False

    if (not use_blend) and Tem <= T_lo:
        return _massive_zeroT_state()

    def integrand(k):
        Ek = _Ek(k, m)
        a = (Ek - mu) / Tem
        b = (Ek + mu) / Tem
        z = np.clip(a, -700.0, 700.0)
        zbar = np.clip(b, -700.0, 700.0)

        f = 1.0 / (1.0 + np.exp(z))
        fbar = 1.0 / (1.0 + np.exp(zbar))
        pressure_density = Tem * (_log1p_exp_neg(a) + _log1p_exp_neg(b))
        susceptibility_density = (f * (1.0 - f) + fbar * (1.0 - fbar)) / Tem

        kk = k * k
        return np.array(
            [
                kk * pressure_density,
                kk * Ek * (f + fbar),
                kk * (f - fbar),
                kk * susceptibility_density,
            ],
            dtype=float,
        )

    integral, _ = quad_vec(integrand, 0.0, upB, epsabs=1e-10, epsrel=1e-8, limit=200)
    pressure_int, energy_int, density_int, susceptibility_int = integral / (np.pi**2)

    if use_blend:
        pressure_0, energy_0, density_0, _, susceptibility_0 = _massive_zeroT_state()
        x = (np.log(Tem) - np.log(T_lo)) / (np.log(T_hi) - np.log(T_lo))
        w = _smoothstep01(x)
        pressure = (1.0 - w) * pressure_0 + w * pressure_int
        energy = (1.0 - w) * energy_0 + w * energy_int
        density = (1.0 - w) * density_0 + w * density_int
        susceptibility = (1.0 - w) * susceptibility_0 + w * susceptibility_int
    else:
        pressure = pressure_int
        energy = energy_int
        density = density_int
        susceptibility = susceptibility_int

    entropy = (energy + pressure - mu * density) / Tem
    if entropy < 0.0 and abs(entropy) <= 1.0e-10 * max(1.0, abs(energy) / max(Tem, 1.0)):
        entropy = 0.0
    return tuple(float(x) for x in (pressure, energy, density, entropy, susceptibility))


def _fermion_thermo_state(mu, m, Tem, upB=np.inf):
    return _fermion_thermo_state_cached(float(mu), float(m), float(Tem), _normalize_upB(upB))


def _quark_mu_triplet(muB, muK):
    """
    Return (mu_u, mu_d, mu_s) from (muB, muK).
    """
    return (
        float(muB / 3.0),
        float(muB / 3.0 + muK / 2.0),
        float(muB / 3.0 - muK / 2.0),
    )


@lru_cache(maxsize=8192)
def _quark_uds_state_from_triplet_cached(mu_u, mu_d, mu_s, T, ms, upB):
    """
    Return the physical u,d,s quark thermodynamic state, including color.
    """
    mu_u = float(mu_u)
    mu_d = float(mu_d)
    mu_s = float(mu_s)
    T = float(T)
    ms = float(ms)
    upB = _normalize_upB(upB)

    Pu, Eu, nu, Su, chiu = _fermion_thermo_state(mu_u, 0.0, T, upB=upB)
    Pd, Ed, nd, Sd, chid = _fermion_thermo_state(mu_d, 0.0, T, upB=upB)
    Ps, Es, ns, Ss, chis = _fermion_thermo_state(mu_s, ms, T, upB=upB)

    quark_pressure = 3.0 * (Pu + Pd + Ps)
    quark_energy = 3.0 * (Eu + Ed + Es)
    quark_entropy = 3.0 * (Su + Sd + Ss)

    n_u = 3.0 * nu
    n_d = 3.0 * nd
    n_s = 3.0 * ns
    chi_u = 3.0 * chiu
    chi_d = 3.0 * chid
    chi_s = 3.0 * chis

    nB_qm = (n_u + n_d + n_s) / 3.0
    nK_qm = 0.5 * (n_d - n_s)
    chiK_qm = 0.25 * (chi_d + chi_s)

    return (
        float(quark_pressure),
        float(quark_energy),
        float(quark_entropy),
        float(nB_qm),
        float(nK_qm),
        float(n_u),
        float(n_d),
        float(n_s),
        float(chi_u),
        float(chi_d),
        float(chi_s),
        float(chiK_qm),
    )


def _quark_uds_state_from_triplet(mu_u, mu_d, mu_s, T, ms, upB=5000):
    (
        quark_pressure,
        quark_energy,
        quark_entropy,
        nB_qm,
        nK_qm,
        n_u,
        n_d,
        n_s,
        chi_u,
        chi_d,
        chi_s,
        chiK_qm,
    ) = _quark_uds_state_from_triplet_cached(
        float(mu_u),
        float(mu_d),
        float(mu_s),
        float(T),
        float(ms),
        _normalize_upB(upB),
    )
    return {
        "pressure": quark_pressure,
        "energy": quark_energy,
        "entropy": quark_entropy,
        "nB": nB_qm,
        "nK": nK_qm,
        "species": {
            "n_u": n_u,
            "n_d": n_d,
            "n_s": n_s,
        },
        "susceptibilities": {
            "chi_u": chi_u,
            "chi_d": chi_d,
            "chi_s": chi_s,
            "chiK": chiK_qm,
        },
    }


def _quark_uds_state(muB, muK, T, ms=0, upB=5000):
    mu_u, mu_d, mu_s = _quark_mu_triplet(muB, muK)
    return _quark_uds_state_from_triplet(mu_u, mu_d, mu_s, T, ms, upB=upB)


def _quark_pressure_uds(mu_u, mu_d, mu_s, T, ms, upB=5000):
    """
    Physical u,d,s quark pressure.
    """
    return float(_quark_uds_state_from_triplet(mu_u, mu_d, mu_s, T, ms, upB=upB)["pressure"])


def _quark_edens_uds(mu_u, mu_d, mu_s, T, ms, upB=5000):
    """
    Physical u,d,s quark energy density.
    """
    return float(_quark_uds_state_from_triplet(mu_u, mu_d, mu_s, T, ms, upB=upB)["energy"])


def _quark_density_uds(mu_u, mu_d, mu_s, T, ms, upB=5000):
    """
    Return physical QM baryon density plus physical per-flavor quark densities.
    """
    state = _quark_uds_state_from_triplet(mu_u, mu_d, mu_s, T, ms, upB=upB)
    species = state["species"].copy()
    return float(state["nB"]), species


def _quark_entropy_uds_direct(mu_u, mu_d, mu_s, T, ms, upB=5000):
    """
    Physical u,d,s quark entropy density from the direct phase-space integral.
    """
    return float(_quark_uds_state_from_triplet(mu_u, mu_d, mu_s, T, ms, upB=upB)["entropy"])


def PQM(muB, muK, B_one_forth, T, ms=0, upB=5000):
    """
    Calculates the pressure of strange quark matter (SQM) under bag model.
    """
    B = B_one_forth**4
    quark_state = _quark_uds_state(muB, muK, T, ms=ms, upB=upB)
    return float(quark_state["pressure"] - B)


def nB_QM(muB, muK, B_one_forth, T, ms=0, upB=5000, return_species=False):
    """
    Baryon number density of quark matter (u,d,s) in the bag-model setup.
    """
    quark_state = _quark_uds_state(muB, muK, T, ms=ms, upB=upB)
    nB_qm = float(quark_state["nB"])
    species = quark_state["species"].copy()

    if return_species:
        return nB_qm, species
    return nB_qm


def nK_QM(muB, muK, B_one_forth, T, ms=0, upB=5000, dmu=None):
    """
    Kaon-density-like variable nK = dPQM / dmuK.
    """
    quark_state = _quark_uds_state(muB, muK, T, ms=ms, upB=upB)
    return float(quark_state["nK"])


def _chiK_QM(muB, muK, B_one_forth, T, ms=0, upB=5000):
    """
    Susceptibility d nK_QM / d muK = d^2 P / d muK^2.
    """
    quark_state = _quark_uds_state(muB, muK, T, ms=ms, upB=upB)
    return float(quark_state["susceptibilities"]["chiK"])


def edensQM(muB, muK, B_one_forth, T, ms=0, include_em=False, muQ_init=300, upB=5000):
    """
    Energy density of strange quark matter under the bag model.
    """
    B = B_one_forth**4

    if include_em:
        mu_u, mu_d, mu_s, mu_e = _solve_quark_mu_em(muB, muK, T, ms, muQ_init=muQ_init, upB=upB)
        electron_state = _fermion_thermo_state(mu_e, 0.511, T, upB=upB)
        quark_state = _quark_uds_state_from_triplet(mu_u, mu_d, mu_s, T, ms, upB=upB)
        return float(_gauge_energy(T) + electron_state[1] + quark_state["energy"] + B)

    quark_state = _quark_uds_state(muB, muK, T, ms=ms, upB=upB)
    return float(quark_state["energy"] + B)


def entropyQM(muB, muK, B_one_forth, T, ms=0, include_em=False, muQ_init=300, upB=5000, use_thermal=True):
    """
    Entropy density of strange quark matter under the bag model.
    """
    T = float(T)
    if T < 0.0:
        raise RuntimeError("Negative Temperature")
    if T == 0.0:
        return 0.0

    B = B_one_forth**4

    if include_em:
        mu_u, mu_d, mu_s, mu_e = _solve_quark_mu_em(muB, muK, T, ms, muQ_init=muQ_init, upB=upB)
        quark_state = _quark_uds_state_from_triplet(mu_u, mu_d, mu_s, T, ms, upB=upB)
        electron_state = _fermion_thermo_state(mu_e, 0.511, T, upB=upB)
        quark_entropy = quark_state["entropy"]
        electron_entropy = electron_state[3]

        if not use_thermal:
            return float(quark_entropy + electron_entropy + _gauge_entropy(T))

        quark_pressure = quark_state["pressure"]
        quark_edens = quark_state["energy"]
        species = quark_state["species"]
        n_e = float(electron_state[2])

        pressure_total = _gauge_pressure(T) + electron_state[0] + quark_pressure - B
        edens_total = _gauge_energy(T) + electron_state[1] + quark_edens + B
        chemical_term = mu_u * species["n_u"] + mu_d * species["n_d"] + mu_s * species["n_s"] + mu_e * n_e
        return float((edens_total + pressure_total - chemical_term) / T)

    mu_u, mu_d, mu_s = _quark_mu_triplet(muB, muK)
    quark_state = _quark_uds_state_from_triplet(mu_u, mu_d, mu_s, T, ms, upB=upB)
    quark_entropy = quark_state["entropy"]
    if not use_thermal:
        return quark_entropy

    quark_pressure = quark_state["pressure"]
    quark_edens = quark_state["energy"]
    species = quark_state["species"]
    chemical_term = mu_u * species["n_u"] + mu_d * species["n_d"] + mu_s * species["n_s"]
    pressure_total = quark_pressure - B
    edens_total = quark_edens + B
    return float((edens_total + pressure_total - chemical_term) / T)


def _solve_scalar_root(fun, x0, xtol=1e-8, maxfev=600):
    return float(fsolve(lambda x: fun(float(np.atleast_1d(x)[0])), float(x0), xtol=xtol, maxfev=maxfev)[0])


@lru_cache(maxsize=4096)
def _solve_quark_mu_em_cached(muB, muK, T, ms, muQ_init, upB):
    muB = float(muB)
    muK = float(muK)
    T = float(T)
    ms = float(ms)
    muQ_init = float(muQ_init)
    upB = _normalize_upB(upB)

    def equation(mu_u_val):
        mu_d = (muB + muK - mu_u_val) / 2.0
        mu_s = (muB - muK - mu_u_val) / 2.0
        mu_e = (muB - 3.0 * mu_u_val) / 2.0

        n_u = 3.0 * _fermion_thermo_state(mu_u_val, 0.0, T, upB=upB)[2]
        n_d = 3.0 * _fermion_thermo_state(mu_d, 0.0, T, upB=upB)[2]
        n_s = 3.0 * _fermion_thermo_state(mu_s, ms, T, upB=upB)[2]
        n_e = _fermion_thermo_state(mu_e, 0.511, T, upB=upB)[2]
        return float(3.0 * n_e + n_d + n_s - 2.0 * n_u)

    mu_u = _solve_scalar_root(equation, muQ_init, xtol=1e-8, maxfev=60000)
    mu_d = (muB + muK - mu_u) / 2.0
    mu_s = (muB - muK - mu_u) / 2.0
    mu_e = (muB - 3.0 * mu_u) / 2.0
    return float(mu_u), float(mu_d), float(mu_s), float(mu_e)


def _solve_quark_mu_em(muB, muK, T, ms, muQ_init=300, upB=5000):
    """
    Solve charge-neutral quark chemical potentials for PQM_em-like compositions.
    Returns (mu_u, mu_d, mu_s, mu_e).
    """
    return _solve_quark_mu_em_cached(
        float(muB),
        float(muK),
        float(T),
        float(ms),
        float(muQ_init),
        _normalize_upB(upB),
    )


def PQM_em(muB, muK, B_one_forth, T, ms, muQ_init=300, upB=5000):
    """
    Calculates the pressure of strange quark matter (SQM) under bag model including electrons.
    """
    B = B_one_forth**4
    mu_u, mu_d, mu_s, mu_e = _solve_quark_mu_em(muB, muK, T, ms, muQ_init=muQ_init, upB=upB)
    electron_state = _fermion_thermo_state(mu_e, 0.511, T, upB=upB)
    quark_state = _quark_uds_state_from_triplet(mu_u, mu_d, mu_s, T, ms, upB=upB)
    return float(_gauge_pressure(T) + electron_state[0] + quark_state["pressure"] - B)


def nQM_em(muB, muK, B_one_forth, T, ms, muQ_init=300, upB=5000, return_species=False):
    """
    Baryon number density of charge-neutral quark matter (u,d,s,e composition).
    """
    mu_u, mu_d, mu_s, mu_e = _solve_quark_mu_em(muB, muK, T, ms, muQ_init=muQ_init, upB=upB)
    quark_state = _quark_uds_state_from_triplet(mu_u, mu_d, mu_s, T, ms, upB=upB)
    nB_qm = float(quark_state["nB"])
    species = quark_state["species"].copy()
    species["n_e"] = float(_fermion_thermo_state(mu_e, 0.511, T, upB=upB)[2])

    if return_species:
        return nB_qm, species
    return nB_qm
