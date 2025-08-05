import numpy as np
from numpy.linalg import eigvals
from numdifftools import Derivative
from scipy.integrate import quad
from scipy.optimize import approx_fprime
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.optimize import brentq



# Public functions
__all__ = ["RMFsolve", "RMFsolve_nu", "proton_fraction_RMF", 
           "RMFpressure", "electron_pressure", "n_electron", 
           "n_neutrino", "nucleon_pressure", "RMFsolveSYM", 
           "RMFpressureSYM", "edens_RMF", "pressure_RMF", 
           "entropy_RMF", "binding_energy_RMF", "pressure_RMF_PNM", 
           "RMFpressurePNM", "RMFsolvePNM", "RMFbindingPNM", 
           "RMFsolve_nb", "baryon_density", "RMFbindingSYM", 
           "RMFsolve_nb_PNM", "RMFsolve_nb_SYM", "create_EOS", 
           "DeltaMU", "dU_threshold", "freeener_RMF", "RMFsolve_mu", 
           "RMFsolve_nonequil", "RMFsolve_nonequil_xpconst", 
           "RMFsolve_xp", "neutrino_test", "neutrino_pressure", 
           "neutrino_entropy", "neutrino_edens", "RMFsolve_nu_brent", 
           "pressure_hesse", "RMFsolveSYM_renorm", "ds_dT", "d2s_dT2", 
           "RMFsolve_muB_muQ", "analyze_dPds", "RMFsolvePNM_mu", 
           "RMFsolveSYM_mu"];



###################### Define the coupling setter ########################
# para={{mn,mp},{m$sigma,m$omega,m$rho},{I3n,I3p},{gsn,gsp},{gwn,gwp},
# {grn,grp},{b,c,Mn$scale},{omega4$coupling},{rho4$coupling},
# {b1,b2,b3,a1,a2,a3,a4,a5,a6},nsat,bar};
def _set_couplings(para):
    """
    Set RMF model couplings and constants from a parameter list.

    Parameters:
    para : list of lists/values
        Format:
        [
            [mn, mp],
            [ms, mw, mr],
            [I3n, I3p],
            [gsn, gsp],
            [gwn, gwp],
            [grn, grp],
            [b, c, Mn],
            [zet],
            [xi],
            [b1, b2, b3, a1, a2, a3, a4, a5, a6],
            nsat,
            bar,
            BagConst (optional)
        ]
    """

    global ms, mw, mr, b, c, Mn, mB
    global gsn, gs, gw, gr, I3
    global zet, xi, bar
    global b1, b2, b3, a1, a2, a3, a4, a5, a6
    global me, upB, P_offset

    # Assignments
    ms = para[1][0]
    mw = para[1][1]
    mr = para[1][2]

    b = para[6][0]
    c = para[6][1]
    Mn = para[6][2]

    mB = para[0]
    gsn = para[3][0]
    gs = para[3]
    gw = para[4]
    gr = para[5]
    I3 = para[2]

    zet = para[7][0]
    xi = para[8][0]
    bar = -1

    b1, b2, b3, a1, a2, a3, a4, a5, a6 = para[9]

    me = 0.511  # Electron mass in MeV
    upB = 5000  # Upper integration bound

    P_offset = para[12] if len(para) > 12 and isinstance(para[12], (int, float)) else 0.0



########################### private functions ############################
# Lagrangian of SFH0 RMF (see compose for couplings) 
# https://compose.obspm.fr/eos/34, Lagrangian in nucl-th/0410066
def _fswr(sigma, w0):
    """
    Cross-coupling terms between rho/omega and rho/sigma, _set_couplings() should be called beforehand

    Parameters:
    sigma : scalar field (float)
    w0    : vector field (float)
    a1-a6 : coefficients for sigma terms
    b1-b3 : coefficients for w0 terms

    Returns:
    float : value of the cross-coupling potential
    """
    return (
        b1 * w0**2 + b2 * w0**4 + b3 * w0**6 +
        a1 * sigma + a2 * sigma**2 + a3 * sigma**3 +
        a4 * sigma**4 + a5 * sigma**5 + a6 * sigma**6
    )

def _U_sigma(s):
    """
    Sigma meson potential, _set_couplings() should be called beforehand

    Parameters:
    s    : sigma field value (float)
    ms   : sigma meson mass (float)
    gsn  : sigma-nucleon coupling constant (float)
    b, c : nonlinear self-interaction coefficients (float)
    Mn   : nucleon mass (float)

    Returns:
    float : value of the sigma potential
    """
    return (
        0.5 * ms**2 * s**2 +
        (b * Mn / 3) * (gsn * s)**3 +
        (c / 4) * (gsn * s)**4
    )

# meson lagrangian
def _Lmes(sigma, w0, r03, U_sigma_func=_U_sigma, fswr_func=_fswr):
    """
    Meson part of the Lagrangian density, _set_couplings() should be called beforehand.

    Parameters:
    sigma        : scalar mean field
    w0           : vector mean field
    r03          : isovector mean field
    mw, mr       : omega and rho meson masses
    zet, xi      : self-coupling coefficients
    gw, gr       : omega and rho coupling constants (e.g., gw[0], gr[0])
    U_sigma_func : function U_sigma(sigma)
    fswr_func    : function fswr(sigma, w0)

    Returns:
    float : meson Lagrangian value
    """
    return (
        -U_sigma_func(sigma) +
        0.5 * mw**2 * w0**2 +
        (zet / 24) * gw[1]**4 * w0**4 +
        0.5 * mr**2 * r03**2 +
        (xi / 24) * gr[1]**4 * r03**4 +
        gr[1]**2 * fswr_func(sigma, w0) * r03**2
    )

# renormalization terms
def _Lmes_renorm(sigma, w0, r03, Lmes_func=_Lmes):
    """
    Renormalized meson Lagrangian, _set_couplings() should be called beforehand.

    Parameters:
    sigma     : scalar mean field
    w0, r03   : vector and isovector mean fields
    Mn        : nucleon mass (assumed = mB[0])
    mB        : baryon mass of species 1 (mB[0])
    gs        : scalar coupling of species 1 (gs[0])
    Lmes_func : meson Lagrangian function, default is _Lmes

    Returns:
    float : renormalized meson Lagrangian value
    """
    phi = gs[1] * sigma

    renorm_term = (
        Mn**3 * phi
        - (7/2) * Mn**2 * phi**2
        + (13/3) * Mn * phi**3
        - (25/12) * phi**4
        + (mB[1] - phi)**4 * np.log(np.clip((mB[1] - phi) / mB[1], -700, 700))
    )

    return Lmes_func(sigma, w0, r03) + (1 / (4 * np.pi**2)) * renorm_term

# baryon density for interacting baryons
def _nB(T, muB, BI, Sigma, w0, r03, upB=5000):
    """
    Calculate baryon density for species BI, _set_couplings() should be called beforehand.

    Parameters:
    T     : temperature (MeV)
    muB   : baryon chemical potential (MeV)
    BI    : baryon index
    Sigma : scalar mean field
    w0    : vector mean field
    r03   : isovector mean field
    upB   : upper momentum cutoff for integration
    mB    : list or array of baryon masses
    gs    : list or array of scalar couplings
    gw    : list or array of omega couplings
    gr    : list or array of rho couplings
    I3    : list or array of isospin values
    """

    # Effective mass and chemical potential
    m_eff = mB[BI] - gs[BI] * Sigma
    mu_eff = muB - gw[BI] * w0 - gr[BI] * I3[BI] * r03

    if T >= 0.1:
        def integrand(k):
            E_star = np.sqrt(k**2 + m_eff**2)
            term1 = k**2 / (1.0 + np.exp(np.clip((E_star - mu_eff) / T, -700, 700)))
            term2 = k**2 / (1.0 + np.exp(np.clip((E_star + mu_eff) / T, -700, 700)))
            return term1 - term2

        integral, _ = quad(integrand, 0, upB, limit=100, epsabs=1e-8, epsrel=1e-6)
        return integral / (np.pi**2)

    else:
        if mu_eff > m_eff:
            kF = np.sqrt(mu_eff**2 - m_eff**2)
            return kF**3 / (3 * np.pi**2)
        else:
            return 0.0

# public baryon density function
def baryon_density(T, muB, BI, sigma, w0, r03, para):
    """
    Public function that sets model parameters and computes baryon density.

    Parameters:
    T, muB, BI, sigma, w0, r03 : physical inputs
    para : model identifier or parameter set (e.g., 'QMC1')

    Returns:
    float : baryon number density
    """
    _set_couplings(para)  # must be defined elsewhere
    return _nB(T, muB, BI, sigma, w0, r03, mB, gs, gw, gr, I3)

# scalar density 
def _ns(T, muB, BI, sigma, w0, r03, upB=5000):
    """
    Calculate baryon scalar density (source term for sigma meson).

    Parameters:
    T     : temperature (MeV)
    muB   : baryon chemical potential (MeV)
    BI    : baryon index (int)
    sigma : scalar field
    w0    : vector field
    r03   : isovector field
    mB, gs, gw, gr, I3 : model parameter arrays
    upB   : upper bound for momentum integration (default 5000 MeV)

    Returns:
    float : baryon scalar density
    """

    m_eff = mB[BI] - gs[BI] * sigma
    mu_eff = muB - gw[BI] * w0 - gr[BI] * I3[BI] * r03

    if T >= 0.1:
        def integrand(k):
            Ek = np.sqrt(k**2 + m_eff**2)
            f  = 1 / (1 + np.exp(np.clip((Ek - mu_eff) / T, -700, 700)))
            fbar = 1 / (1 + np.exp(np.clip((Ek + mu_eff) / T, -700, 700)))
            return (m_eff / Ek) * k**2 * (f + fbar)

        integral, _ = quad(integrand, 0, upB, epsabs=1e-10, epsrel=1e-8, limit=200)
        return integral / np.pi**2

    else:
        if mu_eff <= m_eff:
            return 0.0
        kF = np.sqrt(mu_eff**2 - m_eff**2)
        term1 = m_eff * kF * mu_eff
        term2 = m_eff**3 * np.log((mu_eff + kF) / m_eff)
        return (term1 - term2) / (2 * np.pi**2)

# Fermi momentum for baryons in RMF, needs _set_couplings
def _kf_bar(sigma, w0, r03, muB, BI):
    """
    Compute Fermi momentum for baryon species BI in RMF.

    Parameters:
    sigma : scalar field
    w0    : vector field
    r03   : isovector field
    muB   : baryon chemical potential
    BI    : baryon index
    mB, gs, gw, gr, I3 : global model arrays (must be set via _set_couplings)

    Returns:
    float : Fermi momentum (MeV), or 0 if below threshold
    """

    m_eff = mB[BI] - gs[BI] * sigma
    mu_eff = muB - gw[BI] * w0 - gr[BI] * I3[BI] * r03

    if mu_eff > m_eff and m_eff > 0:
        return np.sqrt(mu_eff**2 - m_eff**2)
    else:
        return 0.0

# E_F^* for baryons = Sqrt[kf^2+m*^2] (no vector mean fields), 
# needs _set_couplings
def _Ef_bar(sigma, w0, r03, muB, BI):
    """
    Compute effective Fermi energy for baryon BI (scalar part only).

    Parameters:
    sigma : scalar field
    w0    : vector field
    r03   : isovector field
    muB   : baryon chemical potential
    BI    : baryon index
    mB, gs, gw, gr, I3 : global model arrays

    Returns:
    float : Effective Fermi energy (MeV)
    """

    kf = _kf_bar(sigma, w0, r03, muB, BI)
    m_eff = mB[BI] - gs[BI] * sigma
    return np.sqrt(kf**2 + m_eff**2)

# fermi dirac distribution
def _fd(E, mu, T):
    """
    Smooth Fermi-Dirac distribution, numerically stable even near T = 0.
    """
    if T <= 1e-6:
        return np.clip(float(E < mu), 1e-15, 1.0)
    z = np.clip((E - mu) / T, -700, 700)
    return np.clip(1.0 / (1.0 + np.exp(z)), 1e-15, 1.0)

# Fermi Dirac distribution for Baryons, 
def _fd_bar(k, sigma, w0, r03, muB, BI, T):
    """
    Fermi-Dirac distribution for baryons in RMF.

    Parameters:
    k     : momentum
    sigma : scalar field
    w0    : vector field
    r03   : isovector field
    muB   : baryon chemical potential
    BI    : baryon index
    T     : temperature
    mB, gs, gw, gr, I3 : model parameter arrays

    Returns:
    float : Fermi-Dirac occupation number
    """
    m_eff = mB[BI] - gs[BI] * sigma
    mu_eff = muB - gw[BI] * w0 - gr[BI] * I3[BI] * r03
    E = np.sqrt(k**2 + m_eff**2)

    if T < 1e-3:
        return float(E < mu_eff)

    z = np.clip((E - mu_eff) / T, -700, 700)
    return 1.0 / (1.0 + np.exp(z))



########################## Different species ############################
# electron contributions
# electrons=free fermion density, needs setcouplings
def n_electron(T, muQ, me=0.511, upB=5000):
    """
    Electron number density assuming a free Fermi gas.

    Parameters:
    T    : temperature (MeV)
    muQ  : charge chemical potential (MeV)
    me   : electron mass (MeV), default 0.511
    upB  : upper bound for integration (MeV)

    Returns:
    float : electron number density (1/fm^3)
    """
    if T >= 0.1:
        def integrand_f(k):
            Ek = np.sqrt(k**2 + me**2)
            return k**2 / (1 + np.exp(np.clip((Ek - muQ) / T, -700, 700)))

        def integrand_fbar(k):
            Ek = np.sqrt(k**2 + me**2)
            return k**2 / (1 + np.exp(np.clip((Ek + muQ) / T, -700, 700)))

        f_int, _ = quad(integrand_f, 0, upB, epsabs=1e-10, epsrel=1e-8)
        fbar_int, _ = quad(integrand_fbar, 0, upB, epsabs=1e-10, epsrel=1e-8)
        return (f_int - fbar_int) / np.pi**2

    else:
        if muQ > me:
            return ((muQ**2 - me**2)**1.5) / (3 * np.pi**2)
        else:
            return 0.0

def _kF_electron(muQ, me=0.511):
    """
    Electron Fermi momentum at zero temperature.

    Parameters:
    muQ : chemical potential for electrons (MeV)

    Returns:
    float : Fermi momentum (MeV)
    """
    ne = n_electron(0, muQ, me)
    return (3 * np.pi**2 * ne)**(1/3)

def electron_pressure(T, muQ, me=0.511, upB=5000):
    """
    Free electron pressure at finite or zero temperature.

    Parameters:
    T    : temperature (MeV)
    muQ  : charge chemical potential
    me   : electron mass (MeV)
    upB  : upper bound for integration (MeV)

    Returns:
    float : pressure (MeV^4)
    """
    if T <= 1e-3:
        return (2 / 3) * muQ**4 / (8 * np.pi**2)

    def integrand(k):
        Ek = np.sqrt(k**2 + me**2)
        arg1 = np.clip(-(Ek - muQ) / T, -700, 700)
        arg2 = np.clip(-(Ek + muQ) / T, -700, 700)
        return 2 * T * k**2 * (np.log(1 + np.exp(arg1)) + np.log(1 + np.exp(arg2))) / (2 * np.pi**2)

    result, _ = quad(integrand, 0, upB, epsabs=1e-10, epsrel=1e-8)
    return result

# neutrino contributions
def n_neutrino(T, mu_nu, up=4000):
    """
    Neutrino number density.

    Parameters:
    T      : temperature (MeV)
    mu_nu  : neutrino chemical potential (MeV)
    up     : integration upper limit (MeV)

    Returns:
    float : neutrino number density
    """
    if T >= 0.1:
        f = lambda k: k**2 * _fd(k, mu_nu, T)
        fbar = lambda k: k**2 * _fd(k, -mu_nu, T)
        I1, _ = quad(f, 0, up, epsabs=1e-10, epsrel=1e-8)
        I2, _ = quad(fbar, 0, up, epsabs=1e-10, epsrel=1e-8)
        return (I1 - I2) / (2 * np.pi**2)
    else:
        return (np.sqrt(mu_nu**2))**3 / (6 * np.pi**2)

def neutrino_pressure(T, mu_nu, up=4000):
    """
    Neutrino pressure.

    Parameters:
    T      : temperature (MeV)
    mu_nu  : neutrino chemical potential
    up     : upper bound (MeV)

    Returns:
    float : pressure
    """
    if T <= 1e-3:
        return mu_nu**4 / (24 * np.pi**2)

    def integrand(k):
        arg1 = np.clip(-(k - mu_nu) / T, -700, 700)
        arg2 = np.clip(-(k + mu_nu) / T, -700, 700)
        return 4 * np.pi * T * k**2 * (np.log(1 + np.exp(arg1)) + np.log(1 + np.exp(arg2)))

    result, _ = quad(integrand, 0, up, epsabs=1e-10, epsrel=1e-8)
    return result / (2 * np.pi)**3

def neutrino_edens(T, mu_nu, up=4000):
    """
    Neutrino energy density.

    Parameters:
    T      : temperature (MeV)
    mu_nu  : neutrino chemical potential
    up     : upper bound (MeV)

    Returns:
    float : energy density
    """
    if T <= 0.1:
        return mu_nu**4 / (8 * np.pi**2)
    else:
        total = 0
        for sgn in [-1, 1]:
            integrand = lambda k: k**3 / (1 + np.exp(np.clip((k - sgn * mu_nu) / T, -700, 700)))
            val, _ = quad(integrand, 0, up, epsabs=1e-10, epsrel=1e-8)
            total += val
        return total * 4 * np.pi / (2 * np.pi)**3

def neutrino_entropy(T, mu_nu, up=4000):
    """
    Neutrino entropy density.

    Parameters:
    T      : temperature
    mu_nu  : neutrino chemical potential
    up     : integration upper bound

    Returns:
    float : entropy density
    """
    if T > 0:
        def integrand(k):
            tmp = 0
            for sgn in [-1,1]:
                fd_pos = _fd(k, sgn * mu_nu, -T)
                fd_neg = _fd(k, sgn * mu_nu, T)
                tmp += k**2 * (
                    fd_pos * np.log(np.clip(fd_pos, 1e-300, 1.0)) +
                    fd_neg * np.log(np.clip(fd_neg, 1e-300, 1.0))
                )
            return tmp
        val, _ = quad(
            integrand, 0, up,
            epsabs=1e-10, epsrel=1e-8,
            limit=500,  # <-- allow much deeper subdivision
            points=[abs(mu_nu)]  # <-- inform quadrature near peak
        )
        return -4 * np.pi * val / (2 * np.pi)**3
    else:
        return 0.0

def neutrino_test(T, mu_nu, up=4000):
    """
    Thermodynamic consistency test: P = -ε + μn + Ts

    Returns deviation from thermodynamic identity.
    """
    kF_nu = (6 * np.pi**2 * n_neutrino(T, mu_nu, up))**(1/3)
    eps = neutrino_edens(T, mu_nu, up)
    P = neutrino_pressure(T, mu_nu, up)
    s = neutrino_entropy(T, mu_nu, up)
    n = n_neutrino(T, mu_nu, up)
    return (P - (-eps + mu_nu * n + T * s))/P

# photon contributions
def _photon_pressure(T):
    """
    Photon pressure (blackbody radiation): P = (π² / 45) T⁴
    """
    return (np.pi**2 / 45) * T**4

def _photon_entropy(T):
    """
    Photon entropy density: s = (4π² / 45) T³
    """
    return (4 * np.pi**2 / 45) * T**3

def _photon_energy(T):
    """
    Photon energy density: ε = (π² / 15) T⁴
    """
    return (np.pi**2 / 15) * T**4

def _photon_freeener(T):
    """
    Photon free energy density: F = −(π² / 45) T⁴
    """
    return -(np.pi**2 / 45) * T**4

# nucleon contributions
# Module to set couplings, me and upper bound for NIntegrate FIXED!
def nucleon_pressure(T, mu, sigma, w0, r03, BI):
    """
    Pressure of a baryon species in RMF.

    Parameters:
    T      : temperature (MeV)
    mu     : baryon chemical potential (MeV)
    sigma  : scalar mean field
    w0     : vector mean field
    r03    : isovector mean field
    BI     : baryon index

    Returns:
    float : pressure (MeV^4)
    """
    m_eff = mB[BI] - gs[BI] * sigma
    mu_eff = mu - gw[BI] * w0 - gr[BI] * I3[BI] * r03

    if T <= 0.1:
        kf = _kf_bar(sigma, w0, r03, mu, BI)
        Ef = _Ef_bar(sigma, w0, r03, mu, BI)
        if m_eff <= 0 or (kf + Ef) <= 0:
            return 0.0
        term1 = (2/3 * kf**3 - m_eff**2 * kf) * Ef
        term2 = m_eff**4 * np.log((kf + Ef) / m_eff)
        return (term1 + term2) / (8 * np.pi**2)
    else:
        def integrand(k):
            E = np.sqrt(k**2 + m_eff**2)
            pos = np.log(1 + np.exp(np.clip(-(E - mu_eff) / T, -700, 700)))
            neg = np.log(1 + np.exp(np.clip(-(E + mu_eff) / T, -700, 700)))
            return k**2 * (pos + neg)

        result, _ = quad(integrand, 0, upB, epsabs=1e-10, epsrel=1e-8)
        return 2 * T * 4 * np.pi / (2 * np.pi)**3 * result

# Hessian matrix
def pressure_hesse(T, mu, sigma, w0, r03, para, epsilon=1e-5):
    """
    Compute the Hessian matrix of the pressure with respect to sigma and omega0.

    Parameters:
    T     : temperature (MeV)
    mu    : baryon chemical potential (MeV)
    sigma : scalar mean field
    w0    : vector mean field
    r03   : isovector mean field
    para  : RMF parameter set
    epsilon : small step for numerical derivative (optional)

    Returns:
    hesse : 2x2 Hessian matrix
    eigs  : Eigenvalues of the Hessian
    """
    set_parameters(para)

    def total_pressure(vec):
        s, w = vec
        p_total = 0.0
        for BI in range(2):  # Assume neutron and proton = BI = 0,1
            p_total += nucleon_pressure(T, mu, s, w, r03, BI)
        p_total += _Lmes(s, w, r03)
        return p_total

    # Numerical second derivatives (finite difference)
    def hessian(f, x0, dx):
        grad = lambda x: approx_fprime(x, f, dx)
        return np.array([
            approx_fprime(x0, lambda xi: grad(xi)[i], dx)
            for i in range(len(x0))
        ])

    x0 = np.array([sigma, w0])
    hesse = hessian(total_pressure, x0, epsilon)
    eigs = eigvals(hesse)
    return hesse, eigs



############################# RMF solvers #################################
# RMFsolver at given baryon chemical potential, automatically enforces 
# beta equilibrium and charge neutrality
def RMFsolve_mu(mub, Trmf, para, sigma_init, w0_init, r03_init, mu_e_init, verb=False):
    """
    Solve RMF equations at given baryon chemical potential enforcing beta equilibrium and charge neutrality.

    Parameters:
    - mub       : baryon chemical potential (float)
    - Trmf      : temperature (float)
    - para      : parameter set (dict or object used by _set_parameter)
    - sigma_init, w0_init, r03_init, mu_e_init : initial guesses for mean fields and electron chemical potential
    - verb      : verbosity flag

    Returns:
    - list containing solution, residuals, temperature, baryon density, parameter
    """
    _set_couplings(para)

    if verb and Trmf <= 0.1:
        print("Temperature is either 0 or T < 1 MeV, T=0 RMF solver is used")

    def Eq1(sigma, w0, r03, mub, mu_e):
        return float(
            -gs[0] * _ns(Trmf, mub, 0, sigma, w0, r03)
            -gs[1] * _ns(Trmf, mub - mu_e, 1, sigma, w0, r03)
            -Derivative(lambda s: _Lmes(s, w0, r03))(sigma)
        )

    def Eq2(sigma, w0, r03, mub, mu_e):
        return float(
            -gw[0] * _nB(Trmf, mub, 0, sigma, w0, r03)
            -gw[1] * _nB(Trmf, mub - mu_e, 1, sigma, w0, r03)
            +Derivative(lambda w: _Lmes(sigma, w, r03))(w0)
        )

    def Eq3(sigma, w0, r03, mub, mu_e):
        return float(
            -gr[0] * I3[0] * _nB(Trmf, mub, 0, sigma, w0, r03)
            -gr[1] * I3[1] * _nB(Trmf, mub - mu_e, 1, sigma, w0, r03)
            +Derivative(lambda r: _Lmes(sigma, w0, r))(r03)
        )

    def Eq5(sigma, w0, r03, mub, mu_e):
        return float(n_electron(Trmf, mu_e) - _nB(Trmf, mub - mu_e, 1, sigma, w0, r03))

    def system(vars):
        sigma, w0, r03, mu_e = vars
        return np.array([
            Eq1(sigma, w0, r03, mub, mu_e),
            Eq2(sigma, w0, r03, mub, mu_e),
            Eq3(sigma, w0, r03, mub, mu_e),
            Eq5(sigma, w0, r03, mub, mu_e)
        ], dtype=float)

    init = [sigma_init, w0_init, r03_init, mu_e_init]
    sol = root(system, init, method="hybr", options={"xtol": 1e-10, "maxfev": 60000})

    sigma, w0, r03, mu_e = sol.x
    chk = system([sigma, w0, r03, mu_e])
    nb = _nB(Trmf, mub, 0, sigma, w0, r03) + _nB(Trmf, mub - mu_e, 1, sigma, w0, r03)

    if verb:
        if max(np.abs(chk)) > 1e-6:
            print("Accuracy not achieved, check residuals > 10⁻⁶")

    return [[sigma, w0, r03, mub, mub - mu_e, mu_e, 0], chk, Trmf, nb, para]

def analyze_dPds(mub, Trmf, para, sigma, w0s, r03s, mu_es, verb):
    """
    Evaluate dP/dsigma with sigma fixed, by solving the rest of the RMF equations.

    Parameters:
    - mub     : baryon chemical potential
    - Trmf    : temperature
    - para    : parameter set (dict)
    - sigma   : fixed sigma value
    - w0s     : initial guess for w0
    - r03s    : initial guess for r03
    - mu_es   : initial guess for mu_e
    - verb    : verbosity flag

    Returns:
    - float : dP/dsigma evaluated at fixed sigma with equilibrium (w0, r03, mu_e)
    """
    _set_couplings(para)

    if verb and Trmf <= 0.1:
        print("Temperature is either 0 or T < 1 MeV, T=0 RMF solver is used")

    def Eq1(sigma_, w0, r03, mub_, mu_e):
        return (
            -gs[0] * _ns(Trmf, mub_, 0, sigma_, w0, r03)
            -gs[1] * _ns(Trmf, mub_ - mu_e, 1, sigma_, w0, r03)
            -Derivative(lambda s: _Lmes(s, w0, r03))(sigma_)
        )

    def Eq2(w0, r03, mub_, mu_e):
        return (
            -gw[0] * _nB(Trmf, mub_, 0, sigma, w0, r03)
            -gw[1] * _nB(Trmf, mub_ - mu_e, 1, sigma, w0, r03)
            +Derivative(lambda w: _Lmes(sigma, w, r03))(w0)
        )

    def Eq3(w0, r03, mub_, mu_e):
        return (
            -gr[0] * I3[0] * _nB(Trmf, mub_, 0, sigma, w0, r03)
            -gr[1] * I3[1] * _nB(Trmf, mub_ - mu_e, 1, sigma, w0, r03)
            +Derivative(lambda r: _Lmes(sigma, w0, r))(r03)
        )

    def Eq5(w0, r03, mub_, mu_e):
        return n_electron(Trmf, mu_e) - _nB(Trmf, mub_ - mu_e, 1, sigma, w0, r03)

    def system(x):
        w0, r03, mu_e = x
        return [
            Eq2(w0, r03, mub, mu_e),
            Eq3(w0, r03, mub, mu_e),
            Eq5(w0, r03, mub, mu_e)
        ]

    guess = [w0s, r03s, mu_es]
    sol = root(system, guess, method="hybr", options={"xtol": 1e-10, "maxfev": 60000})
    w0, r03, mu_e = sol.x

    return Eq1(sigma, w0, r03, mub, mu_e)

# RMFsolver at given density and for difference between 
# dmu=mun-mup-mue neutron, electron and proton chemical potential 
# (how far out of cold beta equil), automatically enforces charge neutrality
def RMFsolve_nonequil(nbext, dmu, Trmf, para, sigma_init, w0_init, r03_init, mu_e_init, mub_init, verb=False):
    """
    RMF solver for nonequilibrium beta matter at given baryon density and dmu = mu_n - mu_p - mu_e.

    Parameters:
    - nbext       : external baryon density (fm⁻³)
    - dmu         : deviation from beta equilibrium (MeV)
    - Trmf        : temperature (MeV)
    - para        : model parameter set
    - sigma_init  : initial guess for sigma field
    - w0_init     : initial guess for omega field
    - r03_init    : initial guess for rho03 field
    - mu_e_init   : initial guess for electron chemical potential
    - mub_init    : initial guess for neutron chemical potential
    - verb        : verbosity flag

    Returns:
    - list of: [solution], [residuals], Trmf, nbext, para
    """
    _set_couplings(para)

    if verb and Trmf <= 0.1:
        print("Temperature is either 0 or T < 1 MeV, T=0 RMF solver is used")

    def Eq1(sigma, w0, r03, mu_n, mu_p):
        return float(
            -gs[0] * _ns(Trmf, mu_n, 0, sigma, w0, r03)
            -gs[1] * _ns(Trmf, mu_p, 1, sigma, w0, r03)
            -Derivative(lambda s: _Lmes(s, w0, r03), step=1e-4)(sigma)
        )

    def Eq2(sigma, w0, r03, mu_n, mu_p):
        return float(
            -gw[0] * _nB(Trmf, mu_n, 0, sigma, w0, r03)
            -gw[1] * _nB(Trmf, mu_p, 1, sigma, w0, r03)
            +Derivative(lambda w: _Lmes(sigma, w, r03), step=1e-4)(w0)
        )

    def Eq3(sigma, w0, r03, mu_n, mu_p):
        return float(
            -gr[0] * I3[0] * _nB(Trmf, mu_n, 0, sigma, w0, r03)
            -gr[1] * I3[1] * _nB(Trmf, mu_p, 1, sigma, w0, r03)
            +Derivative(lambda r: _Lmes(sigma, w0, r), step=1e-4)(r03)
        )

    def Eq4(sigma, w0, r03, mu_n, mu_p):
        return float(
            _nB(Trmf, mu_n, 0, sigma, w0, r03)
            +_nB(Trmf, mu_p, 1, sigma, w0, r03)
            - nbext
        )

    def Eq5(sigma, w0, r03, mu_p, mu_e):
        return float(n_electron(Trmf, mu_e) - _nB(Trmf, mu_p, 1, sigma, w0, r03))

    def system(vars):
        sigma, w0, r03, mu_e, mu_n = vars
        mu_p = mu_n - mu_e - dmu
        return np.array([
            Eq1(sigma, w0, r03, mu_n, mu_p),
            Eq2(sigma, w0, r03, mu_n, mu_p),
            Eq3(sigma, w0, r03, mu_n, mu_p),
            Eq4(sigma, w0, r03, mu_n, mu_p),
            Eq5(sigma, w0, r03, mu_p, mu_e)
        ], dtype=float)

    init = [sigma_init, w0_init, r03_init, mu_e_init, mub_init]  # mub_init is mu_n guess
    sol = root(system, init, method="hybr", options={"xtol": 1e-10, "maxfev": 60000})

    sigma, w0, r03, mu_e, mu_n = sol.x
    mu_p = mu_n - mu_e - dmu
    chk = system([sigma, w0, r03, mu_e, mu_n])

    out = [[sigma, w0, r03, mu_n, mu_p, mu_e, 0], chk, Trmf, nbext, para]

    if verb:
        if max(np.abs(chk)) > 1e-6:
            print("Accuracy not achieved, check functions > 10⁻⁶")
        else:
            print("Solution converged.")

    return out

# RMFsolver out of equilibrium at given T,nb and xp, enforces charge neutrality
def RMFsolve_xp(nbext, xp, Trmf, para, sigma_init, w0_init, r03_init, mub_init, mup_init, mu_e_init, verb=False, electrons=True):
    """
    Solve RMF equations at given T, nb and proton fraction xp, enforcing charge neutrality.
    
    Parameters:
    - nbext       : baryon density
    - xp          : proton fraction
    - Trmf        : temperature
    - para        : parameter set (used by _set_couplings)
    - *_init      : initial guesses for sigma, w0, r03, mu_b, mu_p, mu_e
    - verb        : verbose output flag
    - electrons   : whether to include electrons
    
    Returns:
    - [solution, residuals, temperature, nbext, para]
    """
    _set_couplings(para)

    if verb and Trmf <= 0.1:
        print("Temperature is either 0 or T < 1 MeV, T=0 RMF solver is used")

    def Eq1(sigma, w0, r03, mu_b, mu_p):
        return float(
            -gs[0] * _ns(Trmf, mu_b, 0, sigma, w0, r03)
            -gs[1] * _ns(Trmf, mu_p, 1, sigma, w0, r03)
            -Derivative(lambda s: _Lmes(s, w0, r03))(sigma)
        )

    def Eq2(sigma, w0, r03, mu_b, mu_p):
        return float(
            -gw[0] * _nB(Trmf, mu_b, 0, sigma, w0, r03)
            -gw[1] * _nB(Trmf, mu_p, 1, sigma, w0, r03)
            +Derivative(lambda w: _Lmes(sigma, w, r03))(w0)
        )

    def Eq3(sigma, w0, r03, mu_b, mu_p):
        return float(
            -gr[0] * I3[0] * _nB(Trmf, mu_b, 0, sigma, w0, r03)
            -gr[1] * I3[1] * _nB(Trmf, mu_p, 1, sigma, w0, r03)
            +Derivative(lambda r: _Lmes(sigma, w0, r))(r03)
        )

    def Eq4(sigma, w0, r03, mu_b, mu_p):
        return float(_nB(Trmf, mu_p, 1, sigma, w0, r03) / nbext - xp)

    def Eq5(sigma, w0, r03, mu_p, mu_e):
        return float(n_electron(Trmf, mu_e) - _nB(Trmf, mu_p, 1, sigma, w0, r03))

    def Eq6(sigma, w0, r03, mu_b, mu_p, mu_e):
        return float(
            _nB(Trmf, mu_b, 0, sigma, w0, r03)
            + _nB(Trmf, mu_p, 1, sigma, w0, r03)
            - nbext
        )

    if electrons:
        def system(vars):
            sigma, w0, r03, mu_e, mu_b, mu_p = vars
            return np.array([
                Eq1(sigma, w0, r03, mu_b, mu_p),
                Eq2(sigma, w0, r03, mu_b, mu_p),
                Eq3(sigma, w0, r03, mu_b, mu_p),
                Eq4(sigma, w0, r03, mu_b, mu_p),
                Eq5(sigma, w0, r03, mu_p, mu_e),
                Eq6(sigma, w0, r03, mu_b, mu_p, mu_e)
            ], dtype=float)
        init = [sigma_init, w0_init, r03_init, mu_e_init, mub_init, mup_init]
    else:
        def system(vars):
            sigma, w0, r03, mu_b, mu_p = vars
            mu_e = 0  # implicitly set to 0
            return np.array([
                Eq1(sigma, w0, r03, mu_b, mu_p),
                Eq2(sigma, w0, r03, mu_b, mu_p),
                Eq3(sigma, w0, r03, mu_b, mu_p),
                Eq4(sigma, w0, r03, mu_b, mu_p),
                Eq6(sigma, w0, r03, mu_b, mu_p, mu_e)
            ], dtype=float)
        init = [sigma_init, w0_init, r03_init, mub_init, mup_init]

    sol = root(system, init, method="hybr", options={"xtol": 1e-10, "maxfev": 60000})

    if electrons:
        sigma, w0, r03, mu_e, mu_b, mu_p = sol.x
    else:
        sigma, w0, r03, mu_b, mu_p = sol.x
        mu_e = 0

    chk = system(sol.x)
    if verb and max(np.abs(chk)) > 1e-6:
        print("Accuracy not achieved, check residuals > 10⁻⁶")

    return [[sigma, w0, r03, mu_b, mu_p, mu_e, 0], chk, Trmf, nbext, para]

# (CHECK AGAIN)
# RMFsolver out of equilibrium at given T,mu_B and xp, enforces charge neutrality
def RMFsolve_muB_muQ(mub, mu_e, Trmf, para, sigma_init, w0_init, r03_init, mu_p_init, verb=False):
    _set_couplings(para)

    if verb and Trmf <= 0.1:
        print("Temperature is either 0 or T < 1 MeV, T=0 RMF solver is used")

    def Eq1(sigma, w0, r03, mu_n, mu_p):
        return float(
            -gs[0] * _ns(Trmf, mu_n, 0, sigma, w0, r03)
            -gs[1] * _ns(Trmf, mu_p, 1, sigma, w0, r03)
            -_Lmes(sigma + 1e-5, w0, r03) + _Lmes(sigma - 1e-5, w0, r03)
        ) / (2e-5)

    def Eq2(sigma, w0, r03, mu_n, mu_p):
        return float(
            -gw[0] * _nB(Trmf, mu_n, 0, sigma, w0, r03)
            -gw[1] * _nB(Trmf, mu_p, 1, sigma, w0, r03)
            + (_Lmes(sigma, w0 + 1e-5, r03) - _Lmes(sigma, w0 - 1e-5, r03)) / (2e-5)
        )

    def Eq3(sigma, w0, r03, mu_n, mu_p):
        return float(
            -gr[0] * I3[0] * _nB(Trmf, mu_n, 0, sigma, w0, r03)
            -gr[1] * I3[1] * _nB(Trmf, mu_p, 1, sigma, w0, r03)
            + (_Lmes(sigma, w0, r03 + 1e-5) - _Lmes(sigma, w0, r03 - 1e-5)) / (2e-5)
        )

    def Eq5(sigma, w0, r03, mu_p):
        return float(n_electron(Trmf, mu_e) - _nB(Trmf, mu_p, 1, sigma, w0, r03))

    def system(vars):
        sigma, w0, r03, mu_p = vars
        return np.array([
            Eq1(sigma, w0, r03, mub, mu_p),
            Eq2(sigma, w0, r03, mub, mu_p),
            Eq3(sigma, w0, r03, mub, mu_p),
            Eq5(sigma, w0, r03, mu_p)
        ], dtype=float)

    init = [sigma_init, w0_init, r03_init, mu_p_init]
    sol = root(system, init, method="hybr", options={"xtol": 1e-10, "maxfev": 60000})

    sigma, w0, r03, mu_p = sol.x
    nb_total = _nB(Trmf, mub, 0, sigma, w0, r03) + _nB(Trmf, mu_p, 1, sigma, w0, r03)

    residuals = system(sol.x)

    output = [
        [sigma, w0, r03, mub, mu_p, mu_e, 0],
        residuals,
        Trmf,
        nb_total,
        para,
    ]

    if verb:
        if max(abs(r) for r in residuals) > 1e-6:
            print("Accuracy not achieved, check functions > 10^(-6)")
        else:
            return output
    return output

# neutrino trapped matter
# charge neutral, beta equilibrated, fixed lepton number Yl, at given nB
def RMFsolve_nu(nB, T, Yl, para, sigma_init, w0_init, r03_init, mu_n_init, mu_e_init, mu_nu_init, verb=False):
    """
    Solve RMF equations out of equilibrium for fixed lepton number Yl and charge neutrality.
    Returns solution, residuals, temperature, baryon density, and parameter set.
    """
    _set_couplings(para)

    def equations(vars):
        sigma, w0, r03, mu_n, mu_e, mu_nu = vars
        mu_p = mu_n - mu_e + mu_nu

        eq1 = float(-(gs[0] * _ns(T, mu_n, 0, sigma, w0, r03)
                + gs[1] * _ns(T, mu_p, 1, sigma, w0, r03)) \
              - Derivative(lambda s: _Lmes(s, w0, r03))(sigma))

        eq2 = float(-(gw[0] * _nB(T, mu_n, 0, sigma, w0, r03)
                + gw[1] * _nB(T, mu_p, 1, sigma, w0, r03)) \
              + Derivative(lambda w: _Lmes(sigma, w, r03))(w0))

        eq3 = float(-(gr[0] * I3[0] * _nB(T, mu_n, 0, sigma, w0, r03)
                + gr[1] * I3[1] * _nB(T, mu_p, 1, sigma, w0, r03)) \
              + Derivative(lambda r: _Lmes(sigma, w0, r))(r03))

        eq4 = float((_nB(T, mu_n, 0, sigma, w0, r03)
               + _nB(T, mu_p, 1, sigma, w0, r03)) - nB)

        eq5 = float(n_electron(T, mu_e) - _nB(T, mu_p, 1, sigma, w0, r03))

        eq6 = float(n_electron(T, mu_e) + n_neutrino(T, mu_nu) - Yl * nB)

        return np.array([eq1, eq2, eq3, eq4, eq5, eq6], dtype=float)

    initial_guess = [sigma_init, w0_init, r03_init, mu_n_init, mu_e_init, mu_nu_init]
    sol = root(equations, initial_guess, method='hybr', options={'xtol': 1e-10, 'maxfev': 60000})

    if not sol.success:
        if verb:
            print("Root finding did not converge:", sol.message)
        raise RuntimeError("RMFsolver_nu did not converge")

    sigma, w0, r03, mu_n, mu_e, mu_nu = sol.x
    mu_p = mu_n - mu_e + mu_nu
    residuals = equations(sol.x)

    if verb:
        max_res = np.max(np.abs(residuals))
        if max_res > 1e-6:
            print(f"Warning: max residual = {max_res:.2e}")

    return ([sigma, w0, r03, mu_n, mu_p, mu_e, mu_nu],
            residuals, T, nB, para)

# charge neutral, beta equilibrated, fixed lepton number Yl, at given range of nB
def RMFsolve_nu_nb(nb_start, nb_end, nb_points, T, Yl, para, sigma_init, w0_init, r03_init, muB_init, muE_init, muNu_init, verb):
    """
    RMF out-of-equilibrium solver at fixed lepton number Yl over a range of baryon densities.
    Enforces charge neutrality and beta equilibrium.

    Parameters
    ----------
    nb_start, nb_end : float
        Range of baryon number densities.
    nb_points : int
        Number of intervals (nb_points + 1 total steps).
    T : float
        Temperature in MeV.
    Yl : float
        Lepton fraction (n_e + n_nu)/n_B.
    para : str
        RMF model name.
    sigma_init, w0_init, r03_init : float
        Initial guesses for mean fields.
    muB_init, muE_init, muNu_init : float
        Initial guesses for chemical potentials.
    verb : bool
        Verbose mode for debugging.

    Returns
    -------
    list of tuples : [(nB1, sol1), (nB2, sol2), ...]
    """
    _set_couplings(para)

    nbrestab = []
    dnb = (nb_end - nb_start) / nb_points
    nb_vals = [nb_start + i * dnb for i in range(nb_points + 1)]

    for i, nb in enumerate(nb_vals):
        sol = RMFsolve_nu(nb, T, Yl, para,
                          sigma_init, w0_init, r03_init,
                          muB_init, muE_init, muNu_init, verb)

        nbrestab.append((nb, sol))

        # update guesses
        sigma_init = float(np.real(sol[0][0]))
        w0_init    = float(np.real(sol[0][1]))
        r03_init   = float(np.real(sol[0][2]))
        muB_init   = float(np.real(sol[0][3]))
        muE_init   = float(np.real(sol[0][5]))
        muNu_init  = float(np.real(sol[0][6]))

    return nbrestab

# RMF solving module for beta equilibrated 
# nuclear matter + electrons (mun=mup+mue), inputs are baryon density, 
# temperature, coupling constants for the RMF model in para and initial 
# guesses. For T<1MeV a T=0 solver is used, for higher T the full T 
# dependence is taken into account. Output is in form of replacement 
# rules + checks + T+ RMFmodel couplings 
def RMFsolve(nbext, Trmf, para, sigma_init, w0_init, r03_init, mub_init, mu_e_init, verb=False):
    """
    Solve RMF equations at given baryon density and temperature in beta equilibrium.

    Parameters:
    - nbext     : target baryon density (1/fm^3)
    - Trmf      : temperature (MeV)
    - para      : parameter set
    - sigma_init, w0_init, r03_init : initial guesses for mean fields
    - mub_init, mu_e_init           : initial guesses for baryon and electron chemical potentials
    - verb      : verbosity flag

    Returns:
    - List with solution, residuals, temperature, density, and parameter set
    """
    _set_couplings(para)

    if verb and Trmf <= 0.1:
        print("Temperature is either 0 or T < .1 MeV, T too small for finite T numerical integration, T=0 RMF solver is used")

    def dLmes_dsigma(sigma, w0, r03):
        return Derivative(lambda s: _Lmes(s, w0, r03))(sigma)

    def dLmes_dw0(sigma, w0, r03):
        return Derivative(lambda w: _Lmes(sigma, w, r03))(w0)

    def dLmes_dr03(sigma, w0, r03):
        return Derivative(lambda r: _Lmes(sigma, w0, r))(r03)

    def Eq1(sigma, w0, r03, mub, mu_e):
        return float(
            -gs[0] * _ns(Trmf, mub, 0, sigma, w0, r03)
            -gs[1] * _ns(Trmf, mub - mu_e, 1, sigma, w0, r03)
            -dLmes_dsigma(sigma, w0, r03)
        )

    def Eq2(sigma, w0, r03, mub, mu_e):
        return float(
            -gw[0] * _nB(Trmf, mub, 0, sigma, w0, r03)
            -gw[1] * _nB(Trmf, mub - mu_e, 1, sigma, w0, r03)
            +dLmes_dw0(sigma, w0, r03)
        )

    def Eq3(sigma, w0, r03, mub, mu_e):
        return float(
            -gr[0] * I3[0] * _nB(Trmf, mub, 0, sigma, w0, r03)
            -gr[1] * I3[1] * _nB(Trmf, mub - mu_e, 1, sigma, w0, r03)
            +dLmes_dr03(sigma, w0, r03)
        )

    def Eq4(sigma, w0, r03, mub, mu_e):
        return float(
            _nB(Trmf, mub, 0, sigma, w0, r03)
            + _nB(Trmf, mub - mu_e, 1, sigma, w0, r03)
            - nbext
        )

    def Eq5(sigma, w0, r03, mub, mu_e):
        return float(
            n_electron(Trmf, mu_e)
            - _nB(Trmf, mub - mu_e, 1, sigma, w0, r03)
        )

    def system(vars):
        sigma, w0, r03, mu_e, mub = vars
        return np.array([
            Eq1(sigma, w0, r03, mub, mu_e),
            Eq2(sigma, w0, r03, mub, mu_e),
            Eq3(sigma, w0, r03, mub, mu_e),
            Eq4(sigma, w0, r03, mub, mu_e),
            Eq5(sigma, w0, r03, mub, mu_e)
        ], dtype=float)

    init = [sigma_init, w0_init, r03_init, mu_e_init, mub_init]
    sol = root(system, init, method="hybr", options={"xtol": 1e-10, "maxfev": 60000})

    sigma, w0, r03, mu_e, mub = sol.x
    chk = system([sigma, w0, r03, mu_e, mub])
    mu_p = mub - mu_e

    if verb:
        if max(np.abs(chk)) > 1e-6:
            print("Accuracy not achieved, check functions > 10⁻⁶")

    return [[sigma, w0, r03, mub, mu_p, mu_e, 0], chk, Trmf, nbext, para]

# module to solve RMF for various densities and automatically updating the initial guesses
def RMFsolve_nb(nb_start, nb_end, nb_points, T, para, sigma_init, w0_init, r03_init, muB_init, muE_init, verb):
    """
    Solve RMF equations for a range of baryon densities and automatically update initial guesses.

    Parameters
    ----------
    nb_start, nb_end : float
        Start and end values of baryon number density.
    nb_points : int
        Number of density intervals (nb_points + 1 steps total).
    T : float
        Temperature in MeV.
    para : str
        RMF model name.
    sigma_init, w0_init, r03_init : float
        Initial guesses for mean fields.
    muB_init, muE_init : float
        Initial guesses for chemical potentials.
    verb : bool
        Verbose flag.

    Returns
    -------
    list of tuples : [(nB1, sol1), (nB2, sol2), ...]
    """
    _set_couplings(para)

    nbrestab = []
    dnb = (nb_end - nb_start) / nb_points
    nb_vals = [nb_start + i * dnb for i in range(nb_points + 1)]

    for i, nb in enumerate(nb_vals):
        sol = RMFsolve(nb, T, para,
                       sigma_init, w0_init, r03_init,
                       muB_init, muE_init, verb)

        nbrestab.append((nb, sol))

        # Update initial guesses for next step
        sigma_init = float(np.real(sol[0][0]))
        w0_init    = float(np.real(sol[0][1]))
        r03_init   = float(np.real(sol[0][2]))
        muB_init   = float(np.real(sol[0][3]))
        muE_init   = abs(float(np.real(sol[0][3]) - np.real(sol[0][4])))

    return nbrestab

# module to solve RMF AND compute pressure for beta-equilibrated charge neutral npe matter
def RMFpressure(nbext, Trmf, para, sigma_init, w0_init, r03_init, mub_init, mue_init, verb=False, add_photons=False):
    """
    Solve RMF for charge-neutral, beta-equilibrated npe matter and compute total pressure.

    Parameters:
    - nbext: baryon density
    - Trmf: temperature in MeV
    - para: model parameter set
    - sigma_init, w0_init, r03_init, mub_init, mue_init: initial guesses
    - verb: verbose output
    - add_photons: whether to include photon pressure contribution

    Returns:
    - total pressure (meson + nucleon + electron [+ photon if enabled] + offset)
    """
    _set_couplings(para)

    vevs = RMFsolve(nbext, Trmf, para, sigma_init, w0_init, r03_init, mub_init, mue_init, verb)
    chk = vevs[1]  # list of equation residuals

    sigma = vevs[0][0]
    w0 = vevs[0][1]
    r03 = vevs[0][2]
    mub = vevs[0][3]
    mue = vevs[0][3] - vevs[0][4]  # mu_e = mu_n - mu_p

    muv = [mub, mub - mue]  # neutron and proton effective chemical potentials

    meson_pressure = _Lmes(sigma, w0, r03)
    nucpress = sum(nucleon_pressure(Trmf, muv[BI], sigma, w0, r03, BI) for BI in [0, 1])
    
    P_elec = _electron_pressure(Trmf, mue)
    P_photon = _photon_pressure(Trmf) if add_photons else 0

    if max(abs(r) for r in chk) > 1e-6:
        print("Accuracy not achieved, check functions > 10^(-6)")
    
    return meson_pressure + nucpress + P_elec + P_photon + _P_offset

# SYMMETRIC NUCLEAR MATTER, NO ELECTRONS
# RMF solver for symmetric nuclear matter, no electrons; THIS ONLY IS CORRECT IF MN=MP!!!! FIX THAT
def RMFsolveSYM(nbext, Trmf, para, sigma_init, w0_init, mub_init, verb):
    """
    Solve RMF equations for isospin symmetric nuclear matter (SNM) at given baryon density and temperature.

    Assumes:
    - n_p = n_n (equal proton and neutron densities)
    - μ_p = μ_n = μ_B (equal chemical potentials)
    - ρ_03 = 0 (vanishes due to isospin symmetry)
    - No electrons present

    The function supports both cases:
    - If m_p == m_n: μ_p = μ_n is enforced, and 3 equations are solved: σ, ω₀, and n_B = nbext.
    - If m_p ≠ m_n: μ_p ≠ μ_n allowed, and 4 equations are solved including a symmetry constraint n_p = n_n.

    Parameters
    ----------
    nbext : float
        Total baryon density (1/fm³)
    Trmf : float
        Temperature in MeV
    para : object
        Model parameter set (used in _set_couplings)
    sigma_init, w0_init, mub_init : float
        Initial guesses for mean fields and baryon chemical potential
    verb : bool
        Verbosity flag

    Returns
    -------
    list
        [
            [σ, w₀, ρ₀₃=0, μ_n, μ_p, μ_e=0, μ_ν=0],
            [residuals of equations],
            Trmf,
            nbext,
            para
        ]
    """

    _set_couplings(para)
    
    def Eq1(sigma, w0, muB):
        return float( -(
            gs[0] * _ns(Trmf, muB, 0, sigma, w0, 0) +
            gs[1] * _ns(Trmf, muB, 1, sigma, w0, 0)
        ) - Derivative(lambda s: _Lmes(s, w0, 0), n=1)(sigma) )

    def Eq2(sigma, w0, muB):
        return float( -(
            gw[0] * _nB(Trmf, muB, 0, sigma, w0, 0) +
            gw[1] * _nB(Trmf, muB, 1, sigma, w0, 0)
        ) + Derivative(lambda w: _Lmes(sigma, w, 0), n=1)(w0) )

    def Eq4(sigma, w0, muB):
        return float(
            _nB(Trmf, muB, 0, sigma, w0, 0) +
            _nB(Trmf, muB, 1, sigma, w0, 0) - nbext
        )

    def EqSYM(sigma, w0, muN, muP):
        return float(
            _nB(Trmf, muN, 0, sigma, w0, 0) -
            _nB(Trmf, muP, 1, sigma, w0, 0)
        )

    if verb and Trmf <= 0.1:
        print("Temperature is either 0 or T < 1 MeV; T too small for finite-T integration, using T=0 RMF solver.")

    if mB[0] == mB[1]:
        def equations(x):
            sigma, w0, muB = x
            return np.array([
                Eq1(sigma, w0, muB),
                Eq2(sigma, w0, muB),
                Eq4(sigma, w0, muB)
            ], dtype=float)

        sol = root(equations, [sigma_init, w0_init, mub_init], method='hybr', options={'xtol': 1e-10})

        sigma_sol, w0_sol, muB_sol = sol.x
        out = [
            [sigma_sol, w0_sol, 0.0, muB_sol, muB_sol, 0.0, 0.0],
            equations([sigma_sol, w0_sol, muB_sol]),
            Trmf,
            nbext,
            para
        ]

    else:
        def equations(x):
            sigma, w0, muN, muP = x
            return np.array([
                Eq1(sigma, w0, muN),
                Eq2(sigma, w0, muN),
                Eq4(sigma, w0, muN),
                EqSYM(sigma, w0, muN, muP)
            ], dtype=float)

        sol = root(equations, [sigma_init, w0_init, mub_init, mub_init + 1.0], method='hybr', options={'xtol': 1e-10})

        sigma_sol, w0_sol, muN_sol, muP_sol = sol.x
        out = [
            [sigma_sol, w0_sol, 0.0, muN_sol, muP_sol, 0.0, 0.0],
            equations([sigma_sol, w0_sol, muN_sol, muP_sol]),
            Trmf,
            nbext,
            para
        ]

    chk = out[1]
    if verb:
        if max(np.abs(chk)) > 1e-6:
            print("Accuracy not achieved, check functions > 10^(-6)")
    return out

# module to solve isospin symmetric nuclear matter for a renormalized model of the bogota bodmer type
def RMFsolveSYM_renorm(nbext, Trmf, para, sigma_init, w0_init, muB_init, verb):
    _set_couplings(para)

    def Eq1(sigma, w0, muN, muP):
        return float( -(
            gs[0] * _ns(Trmf, muN, 0, sigma, w0, 0) +
            gs[1] * _ns(Trmf, muP, 1, sigma, w0, 0)
        ) - Derivative(lambda s: _Lmes_renorm(s, w0, 0))(sigma) )

    def Eq2(sigma, w0, muN, muP):
        return float( -(
            gw[0] * _nB(Trmf, muN, 0, sigma, w0, 0) +
            gw[1] * _nB(Trmf, muP, 1, sigma, w0, 0)
        ) + Derivative(lambda w: _Lmes_renorm(sigma, w, 0))(w0) )

    def Eq4(sigma, w0, muN, muP):
        return float(
            _nB(Trmf, muN, 0, sigma, w0, 0) +
            _nB(Trmf, muP, 1, sigma, w0, 0) - nbext
        )

    def EqSYM(sigma, w0, muN, muP):
        return float(
            _nB(Trmf, muN, 0, sigma, w0, 0) -
            _nB(Trmf, muP, 1, sigma, w0, 0)
        )

    if verb and Trmf <= 0.5:
        print("Temperature is either 0 or T < 0.5 MeV; T too small for finite-T integration, using T=0 RMF solver.")

    if mB[0] == mB[1]:
        def equations(x):
            sigma, w0, muB = x
            return np.array([
                Eq1(sigma, w0, muB, muB),
                Eq2(sigma, w0, muB, muB),
                Eq4(sigma, w0, muB, muB)
            ], dtype=float)

        sol = root(equations, [sigma_init, w0_init, muB_init], method='hybr', options={'xtol': 1e-10})
        sigma_sol, w0_sol, muB_sol = sol.x
        out = [
            [sigma_sol, w0_sol, 0.0, muB_sol, muB_sol, 0.0, 0.0],
            equations([sigma_sol, w0_sol, muB_sol]),
            Trmf,
            nbext,
            para
        ]

    else:
        def equations(x):
            sigma, w0, muN, muP = x
            return np.array([
                Eq1(sigma, w0, muN, muP),
                Eq2(sigma, w0, muN, muP),
                Eq4(sigma, w0, muN, muP),
                EqSYM(sigma, w0, muN, muP)
            ], dtype=float)

        sol = root(equations, [sigma_init, w0_init, muB_init, muB_init + 1.0], method='hybr', options={'xtol': 1e-10})
        sigma_sol, w0_sol, muN_sol, muP_sol = sol.x
        out = [
            [sigma_sol, w0_sol, 0.0, muN_sol, muP_sol, 0.0, 0.0],
            equations([sigma_sol, w0_sol, muN_sol, muP_sol]),
            Trmf,
            nbext,
            para
        ]

    chk = out[1]
    if verb:
        if max(np.abs(chk)) > 1e-6:
            print("Accuracy not achieved, check functions > 10^(-6)")
    return out

# module to solve isospin symmetric nuclear matter given baryon chemical potential
def RMFsolveSYM_mu(mub, Trmf, para, sigma_init, w0_init, verb):
    """
    Solve RMF for isospin symmetric nuclear matter at given baryon chemical potential.

    Parameters
    ----------
    mub  : float
        Baryon chemical potential (mu_B), assumed same for neutron and proton.
    Trmf : float
        Temperature (MeV).
    para : RMF parameter set
    sigma_init, w0_init : float
        Initial guesses for sigma and w0 mean fields.
    verb : bool
        Verbosity flag.

    Returns
    -------
    list
        Solution structure:
        [ [sigma, w0, 0.0, mub, mub, 0.0, 0.0], residuals, T, n_B, para ]
    """
    _set_couplings(para)

    if verb and Trmf <= 0.1:
        print("Temperature is either 0 or T < 1 MeV; T too small for finite-T integration, using T=0 RMF solver.")

    def Eq1(sigma, w0):
        dLmes_dsigma = Derivative(lambda s: _Lmes(s, w0, 0))(sigma)
        return float(
            -gs[0] * _ns(Trmf, mub, 0, sigma, w0, 0)
            -gs[1] * _ns(Trmf, mub, 1, sigma, w0, 0)
            -dLmes_dsigma
        )

    def Eq2(sigma, w0):
        dLmes_dw0 = Derivative(lambda w: _Lmes(sigma, w, 0))(w0)
        return float(
            -gw[0] * _nB(Trmf, mub, 0, sigma, w0, 0)
            -gw[1] * _nB(Trmf, mub, 1, sigma, w0, 0)
            +dLmes_dw0
        )

    def equations(x):
        sigma, w0 = x
        return np.array([Eq1(sigma, w0), Eq2(sigma, w0)], dtype=float)

    x0 = [sigma_init, w0_init]
    sol = root(equations, x0, method='hybr', options={'xtol': 1e-10, 'maxfev': 60000})

    if not sol.success:
        if verb:
            print("Root finding failed:", sol.message)
        raise RuntimeError("RMFsolveSYM_mu failed to converge")

    sigma_sol, w0_sol = sol.x
    chk = equations(sol.x)
    nb = _nB(Trmf, mub, 0, sigma_sol, w0_sol, 0) + _nB(Trmf, mub, 1, sigma_sol, w0_sol, 0)

    out = [
        [sigma_sol, w0_sol, 0.0, mub, mub, 0.0, 0.0],
        chk,
        Trmf,
        nb,
        para
    ]

    if verb:
        if max(np.abs(chk)) > 1e-6:
            print("Accuracy not achieved, check functions > 10^(-6)")
    return out

# module to solve SNM RMFs as density table
def RMFsolve_nb_SYM(nb_start, nb_end, nb_points, T, para, sigma_init, w0_init, mub_init, verb):
    """
    Solve symmetric nuclear matter RMF equations over a density table.

    Parameters:
    nb_start : float
        Starting baryon density (MeV^3)
    nb_end   : float
        Ending baryon density (MeV^3)
    nb_points: int
        Number of density points
    T        : float
        Temperature (MeV)
    para     : list
        Model parameters
    sigma_init  : float
        Initial guess for sigma field
    w0_init     : float
        Initial guess for omega field
    mub_init   : float
        Initial guess for baryon chemical potential
    verb     : bool
        Verbosity flag

    Returns:
    nbrestab : list
        Table of results for each density point: [nB, RMF solution]
    """
    ss = sigma_init
    ws = w0_init
    mus = mub_init
    nbrestab = []

    nbs = nb_start
    nbe = nb_end
    nbp = nb_points
    dnb = (nbe - nbs) / nbp

    for i in range(nbp + 1):
        nb_val = nbs + i * dnb
        try:
            result = RMFsolveSYM(nb_val, T, para, ss, ws, mus, verb)
        except Exception:
            result = None
        nbrestab.append([nb_val, result])

        if result is not None:
            ss = np.real(result[0][0])   # sigma
            ws = np.real(result[0][1])   # omega
            mus = np.real(result[0][3])  # mu_b

    return nbrestab

# module to solve RMF AND compute pressure for symmetric  np matter, no electrons no photons no bag constant?
def RMFpressureSYM(input_num, input_type, Trmf, para, sigma_init, w0_init, mub_init, verb):
    """
    Compute pressure for symmetric nuclear matter (np) with no electrons, photons, or bag constant.

    Parameters
    ----------
    input_num : float
        Either baryon density or baryon chemical potential.
    input_type : string
        "nB"  : for baryon density input
        "muB" : for baryon chemical potential input 
    Trmf : float
        Temperature in MeV.
    para : list
        Model parameter list.
    sigma_init : float
        Initial guess for sigma field.
    w0_init : float
        Initial guess for omega_0 field.
    mub_init : float
        Initial guess for baryon chemical potential.
    verb : bool
        Verbose flag.

    Returns
    -------
    float
        Total pressure in MeV^4.
    """

    if input_type == "nB":
        vevs = RMFsolveSYM(input_num, Trmf, para, sigma_init, w0_init, mub_init, verb)

    elif input_type == "muB":
        vevs = RMFsolveSYM_mu(input_num, Trmf, para, sigma_init, w0_init, verb)

    chk = vevs[1]
    sigma = vevs[0][0]
    w0 = vevs[0][1]
    r03 = 0.0
    mub = vevs[0][3]
    muv = [mub, mub]

    meson_pressure = _Lmes(sigma, w0, r03)
    nucpress = sum(
        nucleon_pressure(Trmf, muv[BI], sigma, w0, r03, BI)
        for BI in range(2)
    )

    if verb:
        if max(chk) > 1e-6:
            print("Accuracy not achieved, check functions > 10^(-6)")
        else:
            return meson_pressure + nucpress
    return meson_pressure + nucpress

# module to solve RMF AND compute binding energy SNM
def RMFbindingSYM(nbext, Trmf, para, sigma_init, w0_init, r03_init=0.0, mub_init=0.0, verb=True):
    """
    Solve RMF and compute binding energy for symmetric nuclear matter (SNM).

    Parameters
    ----------
    nbext : float
        Baryon density.
    Trmf : float
        Temperature in MeV.
    para : list
        RMF parameter list.
    sigma_init : float
        Initial guess for sigma field.
    w0_init : float
        Initial guess for omega_0 field.
    r03_init : float, optional
        Initial guess for rho_03 field (default is 0).
    mub_init : float
        Initial guess for baryon chemical potential.
    verb : bool
        Verbose flag.

    Returns
    -------
    float
        Binding energy per baryon in MeV.
    """
    _set_couplings(para)
    vevs = RMFsolveSYM(nbext, Trmf, para, sigma_init, w0_init, mub_init, verb)
    chk = vevs[1]
    bind = binding_energy_RMF(vevs)

    if verb:
        if max(chk) > 1e-6:
            print("Accuracy not achieved, check functions > 10^(-6)")
        else:
            return bind
    return bind

# PURE NEUTRON MATTER 
# RMF solving module, inputs are baryon density, temperature, 
# coupling constants for the RMF model in para and initial guesses. 
# For T<1MeV a T=0 solver is used, for higher T the full T dependence is taken into account. 
# Output is in form of replacement rules + checks + T+ RMFmodel couplings 
def RMFsolvePNM(nbext, Trmf, para, sigma_init, w0_init, r03_init, mub_init, verb=False):
    """
    Solve RMF for pure neutron matter (PNM).

    Parameters
    ----------
    nbext : float
        Baryon density.
    Trmf : float
        Temperature in MeV.
    para : list
        RMF model parameter set.
    sigma_init, w0_init, r03_init, mub_init : float
        Initial guesses for sigma, omega_0, rho_03, and mu_B.
    verb : bool
        Verbosity flag.

    Returns
    -------
    list
        Solution structure: [[sigma, w0, r03, mu_n, 0, 0], [Eq1, Eq2, Eq3, Eq4, 0], T, n_B, para]
    """
    _set_couplings(para)

    if verb and Trmf <= 0.1:
        print("Temperature is either 0 or T < 1 MeV, T too small for finite T numerical integration, T=0 RMF solver is used")

    def Eq1(sigma, w0, r03, mub):
        dLmes_dsigma = Derivative(lambda s: _Lmes(s, w0, r03))(sigma)
        return float( -gs[0] * _ns(Trmf, mub, 0, sigma, w0, r03) - dLmes_dsigma )

    def Eq2(sigma, w0, r03, mub):
        dLmes_dw0 = Derivative(lambda w: _Lmes(sigma, w, r03))(w0)
        return float( -gw[0] * _nB(Trmf, mub, 0, sigma, w0, r03) + dLmes_dw0 )

    def Eq3(sigma, w0, r03, mub):
        dLmes_dr03 = Derivative(lambda r: _Lmes(sigma, w0, r))(r03)
        return float( -gr[0] * I3[0] * _nB(Trmf, mub, 0, sigma, w0, r03) + dLmes_dr03 )

    def Eq4(sigma, w0, r03, mub):
        return float( _nB(Trmf, mub, 0, sigma, w0, r03) - nbext )

    def equations(x):
        sigma, w0, r03, mub = x
        return np.array([
            Eq1(sigma, w0, r03, mub),
            Eq2(sigma, w0, r03, mub),
            Eq3(sigma, w0, r03, mub),
            Eq4(sigma, w0, r03, mub)
        ], dtype=float)

    x0 = [float(sigma_init), float(w0_init), float(r03_init), float(mub_init)]
    sol = root(equations, x0, method='hybr', options={'maxfev': 60000})

    #if not sol.success:
    #    if verb:
    #        print("Root finding failed:", sol.message)
    #    raise RuntimeError("RMFsolvePNM failed to converge")

    sigma, w0, r03, mub = sol.x
    chk = equations(sol.x)

    output = [[sigma, w0, r03, mub, 0.0, 0.0], chk, Trmf, nbext, para]

    if verb:
        if max(abs(c) for c in chk) > 1e-6:
            print("Accuracy not achieved, check functions > 10^(-6)")
        else:
            return output
    return output

# module to solve PNM RMFs given baryon chemical potential
def RMFsolvePNM_mu(mub, Trmf, para, sigma_init, w0_init, r03_init, verb=False):
    """
    Solve RMF for pure neutron matter (PNM) given baryon chemical potential.

    Parameters
    ----------
    mub  : float
        Baryon chemical potential.
    Trmf : float
        Temperature in MeV.
    para : list
        RMF model parameter set.
    sigma_init, w0_init, r03_init : float
        Initial guesses for sigma, omega_0, rho_03.
    verb : bool
        Verbosity flag.

    Returns
    -------
    list
        Solution structure: [[sigma, w0, r03, mu_n, 0, 0], [Eq1, Eq2, Eq3, 0, 0], T, n_B, para]
    """
    _set_couplings(para)

    if verb and Trmf <= 0.1:
        print("Temperature is either 0 or T < 1 MeV, T too small for finite T numerical integration, T=0 RMF solver is used")

    def Eq1(sigma, w0, r03):
        dLmes_dsigma = Derivative(lambda s: _Lmes(s, w0, r03))(sigma)
        return float(-gs[0] * _ns(Trmf, mub, 0, sigma, w0, r03) - dLmes_dsigma)

    def Eq2(sigma, w0, r03):
        dLmes_dw0 = Derivative(lambda w: _Lmes(sigma, w, r03))(w0)
        return float(-gw[0] * _nB(Trmf, mub, 0, sigma, w0, r03) + dLmes_dw0)

    def Eq3(sigma, w0, r03):
        dLmes_dr03 = Derivative(lambda r: _Lmes(sigma, w0, r))(r03)
        return float(-gr[0] * I3[0] * _nB(Trmf, mub, 0, sigma, w0, r03) + dLmes_dr03)

    def equations(x):
        sigma, w0, r03 = x
        return np.array([
            Eq1(sigma, w0, r03),
            Eq2(sigma, w0, r03),
            Eq3(sigma, w0, r03)
        ], dtype=float)

    x0 = [float(sigma_init), float(w0_init), float(r03_init)]
    sol = root(equations, x0, method='hybr', options={'maxfev': 60000})

    #if not sol.success:
        #if verb:
    #    print("Root finding failed:", sol.message)
    #    raise RuntimeError("RMFsolvePNM_mu failed to converge")

    sigma, w0, r03 = sol.x
    chk = equations(sol.x)
    nb = _nB(Trmf, mub, 0, sigma, w0, r03)

    output = [[sigma, w0, r03, mub, 0.0, 0.0], list(chk) + [0, 0], Trmf, nb, para]

    if verb:
        if max(abs(c) for c in chk) > 1e-6:
            print("Accuracy not achieved, check functions > 10^(-6)")
        else:
            return output
    return output

# module to solve PNM RMFs for density range 
def RMFsolve_nb_PNM(nbstart, nbeend, nbpoints, T, para, sigma_init, w0_init, r03_init, mub_init, verb=True):
    """
    Solve RMF for pure neutron matter (PNM) over a range of baryon densities.

    Parameters
    ----------
    nbstart : float
        Starting baryon density.
    nbeend : float
        Ending baryon density.
    nbpoints : int
        Number of intervals between nbstart and nbeend.
    T : float
        Temperature in MeV.
    para : list
        RMF model parameter set.
    sigma_init, w0_init, r03_init, mub_init : float
        Initial guesses for mean fields and baryon chemical potential.
    verb : bool
        Verbosity flag.

    Returns
    -------
    list
        Table of RMF solutions: [[n_B, solution], ...]
    """
    ss = sigma_init
    ws = w0_init
    rs = r03_init
    mus = mub_init
    nbrestab = []

    dnb = (nbeend - nbstart) / nbpoints

    for i in range(nbpoints + 1):
        nb_i = nbstart + i * dnb
        try:
            sol = RMFsolvePNM(nb_i, T, para, ss, ws, rs, mus, verb)
        except RuntimeError:
            if verb:
                print(f"RMFsolvePNM failed at n_B = {nb_i}")
            sol = None

        nbrestab.append([nb_i, sol])

        # update initial guesses for next iteration only if sol is valid
        if sol is not None:
            ss = float(sol[0][0])  # sigma
            ws = float(sol[0][1])  # w0
            rs = float(sol[0][2])  # r03
            mus = float(sol[0][3])  # muB

    return nbrestab

# module to solve RMF AND compute pressure PNM
def RMFpressurePNM(input_num, input_type, Trmf, para, sigma_init, w0_init, r03_init, mub_init, verb=True):
    """
    Compute pressure in pure neutron matter (PNM) using RMF solution.

    Parameters
    ----------
    input_num  : float
        Either baryon number density or baryon chemical potential
    inpyt_type : string
        "muB"  : for baryon chemical potential input
        "nB"   : for baryon number density input
    Trmf : float
        Temperature in MeV.
    para : list
        RMF model parameter set.
    sigma_init, w0_init, r03_init, mub_init : float
        Initial guesses for mean fields and chemical potential.
    verb : bool
        Verbosity flag.

    Returns
    -------
    float
        Total pressure (meson + neutron).
    """
    _set_couplings(para)

    if input_type == "nB":
        vevs = RMFsolvePNM(input_num, Trmf, para, sigma_init, w0_init, r03_init, mub_init, verb)

    elif input_type == "muB":
        vevs = RMFsolvePNM_mu(input_num, Trmf, para, sigma_init, w0_init, r03_init, verb)

    else:
        print("Unknown input type")
        return None

    chk = vevs[1]
    sigma = vevs[0][0]
    w0 = vevs[0][1]
    r03 = vevs[0][2]
    mub = vevs[0][3]
    muv = [mub, 0.0]  # only neutron contributes

    meson_pressure = _Lmes(sigma, w0, r03)
    nuc_pressure = nucleon_pressure(Trmf, muv[0], sigma, w0, r03, BI=0)

    if verb:
        if max(abs(c) for c in chk) > 1e-6:
            print("Accuracy not achieved, check functions > 10^(-6)")
        else:
            return meson_pressure + nuc_pressure

    return meson_pressure + nuc_pressure

# module to solve RMF AND compute binding energy PNM
def RMFbindingPNM(nbext, Trmf, para, sigma_init, w0_init, r03_init, mub_init, verb=True):
    """
    Compute binding energy for pure neutron matter (PNM) using RMF.

    Parameters
    ----------
    nbext : float
        Baryon density.
    Trmf : float
        Temperature in MeV.
    para : list
        RMF model parameter set.
    sigma_init, w0_init, r03_init, mub_init : float
        Initial guesses for mean fields and chemical potential.
    verb : bool
        Verbosity flag.

    Returns
    -------
    float
        Binding energy.
    """
    _set_couplings(para)

    vevs = RMFsolvePNM(nbext, Trmf, para, sigma_init, w0_init, r03_init, mub_init, verb)
    chk = vevs[1]

    # Ensure the solution vector has length at least 7 to match what binding_energy_RMF expects
    if len(vevs[0]) < 7:
        vevs[0] += [0.0] * (7 - len(vevs[0]))

    bind = binding_energy_RMF(vevs)

    if verb:
        if max(abs(c) for c in chk) > 1e-6:
            print("Accuracy not achieved, check functions > 10^(-6)")
        else:
            return bind

    return bind

# module to compute pressure for given solution of RMF obtained by RMFsolver- UPDATE: this module should be obsolete and not be used in new codes
def pressure_RMF_PNM(soltab):
    """
    Compute pressure for a solution of RMF obtained by RMFsolvePNM.

    Parameters
    ----------
    soltab : list
        Solution table directly from RMFsolvePNM

    Returns
    -------
    float
        Total pressure (meson + nucleon)
    """
    para = soltab[4]
    _set_couplings(para)

    T = soltab[2]
    sigma = soltab[0][0]
    w0 = soltab[0][1]
    r03 = soltab[0][2]
    mun = soltab[0][3]
    mup = 0.0  # pure neutron matter

    muv = [mun, mup]

    meson_pressure = _Lmes(sigma, w0, r03)
    nucpress = nucleon_pressure(T, muv[0], sigma, w0, r03, 0)  # BI=1 in MMA → BI=0 in Python

    return meson_pressure + nucpress

# This module computes the out of equilibrium chemical potential if initially beta equilibrated matter is pushed out of equilibrium by a density oscillation of magnitude deltan
def DeltaMU(nbext, deltan, Trmf, para, sigma_init, w0_init, r03_init, mub_init, mue_init, verb=True):
    """
    Compute the out-of-equilibrium chemical potential deviation DeltaMU 
    when initially beta-equilibrated matter is pushed out of equilibrium 
    by a density oscillation of magnitude deltan.

    Parameters
    ----------
    nbext : float
        Initial baryon density.
    deltan : float
        Density perturbation.
    Trmf : float
        Temperature (MeV).
    para : list
        RMF model parameter set.
    sigma_init, w0_init, r03_init : float
        Initial guesses for meson fields.
    mub_init, mue_init : float
        Initial guesses for chemical potentials.
    verb : bool
        Verbosity flag.

    Returns
    -------
    list
        Solution structure including DeltaMU.
    """
    _set_couplings(para)

    # First solve beta-equilibrated state
    equres = RMFsolve(nbext, Trmf, para, sigma_init, w0_init, r03_init, mub_init, mue_init, verb)
    xp = _proton_fraction_RMF(equres)

    def Eq1(sigma, w0, r03, mun, mup):
        dLmes_dsigma = Derivative(lambda s: _Lmes(s, w0, r03))(sigma)
        return -gs[0] * _ns(Trmf, mun, 0, sigma, w0, r03) - gs[1] * _ns(Trmf, mup, 1, sigma, w0, r03) - dLmes_dsigma

    def Eq2(sigma, w0, r03, mun, mup):
        dLmes_dw0 = Derivative(lambda w: _Lmes(sigma, w, r03))(w0)
        return -gw[0] * _nB(Trmf, mun, 0, sigma, w0, r03) - gw[1] * _nB(Trmf, mup, 1, sigma, w0, r03) + dLmes_dw0

    def Eq3(sigma, w0, r03, mun, mup):
        dLmes_dr03 = Derivative(lambda r: _Lmes(sigma, w0, r))(r03)
        return -gr[0] * I3[0] * _nB(Trmf, mun, 0, sigma, w0, r03) - gr[1] * I3[1] * _nB(Trmf, mup, 1, sigma, w0, r03) + dLmes_dr03

    def Eq4(sigma, w0, r03, mun, mup):
        total_nB = _nB(Trmf, mun, 0, sigma, w0, r03) + _nB(Trmf, mup, 1, sigma, w0, r03)
        return total_nB - (nbext + deltan)

    def Eq5(sigma, w0, r03, mup, mue):
        return n_electron(Trmf, mue) - _nB(Trmf, mup, 1, sigma, w0, r03)

    def Eq6(sigma, w0, r03, mup):
        return _nB(Trmf, mup, 1, sigma, w0, r03) - xp * (nbext + deltan)

    # System of equations for root finding
    def equations(x):
        sigma, w0, r03, mue, mun, mup = x
        return [
            Eq1(sigma, w0, r03, mun, mup),
            Eq2(sigma, w0, r03, mun, mup),
            Eq3(sigma, w0, r03, mun, mup),
            Eq4(sigma, w0, r03, mun, mup),
            Eq5(sigma, w0, r03, mup, mue),
            Eq6(sigma, w0, r03, mup)
        ]

    # Initial guess (respecting Mathematica order)
    x0 = [
        float(sigma_init),
        float(w0_init),
        float(r03_init),
        float(mue_init),
        float(mub_init),
        float(mub_init - mue_init)
    ]

    sol = root(equations, x0, method='hybr', options={'maxfev': 60000})

    if not sol.success:
        if verb:
            print("Root finding failed:", sol.message)
        raise RuntimeError("DeltaMU solver failed to converge")

    sigma, w0, r03, mue, mun, mup = sol.x

    # Compute residuals for check
    chk = equations(sol.x)

    # Compute DeltaMU
    delta_mu = mun - mue - mup

    output = [[sigma, w0, r03, mun, mue, mup, delta_mu], chk, Trmf, nbext, para]

    if verb:
        if max(abs(c) for c in chk) > 1e-6:
            print("Accuracy not achieved, check functions > 10^(-6)")
        else:
            return output
    return output



########################## Thermal dynamical functions #######################
# module to compute proton fraction for given solution of RMF obtained by RMFsolver
# soltab is input created directly from RMVsolve
def _proton_fraction_RMF(soltab):
    """
    Compute the proton fraction from an RMF solution.

    Parameters:
    soltab : list
        A solution table returned by RMFsolve.

    Returns:
    float
        Proton fraction x_p.
    """
    para = soltab[4]
    _set_couplings(para)

    T = soltab[2]
    dens = soltab[3]  # total baryon density n_B

    sigma = soltab[0][0]
    w0 = soltab[0][1]
    r03 = soltab[0][2]
    mup = soltab[0][4]

    xp = _nB(T, mup, 1, sigma, w0, r03) / dens
    return xp

# Calculating pressure from given solution table from RMF solvers
def pressure_RMF(soltab, add_photons=False, renorm=False, electrons=True, neutrinos=False):
    """
    Compute pressure from RMF solution.

    Parameters:
    - soltab      : output from RMFsolve (Python version)
    - add_photons : include photon pressure if True
    - renorm      : if True, use renormalized meson Lagrangian
    - electrons   : if True, include electron pressure
    - neutrinos   : if True, include neutrino pressure

    Returns:
    - total pressure (MeV^4)
    """
    T = soltab[2]                 # temperature
    para = soltab[4]              # RMF parameter
    _set_couplings(para)

    sigma = soltab[0][0]
    w0 = soltab[0][1]
    r03 = soltab[0][2]
    mu_n = soltab[0][3]
    mu_p = soltab[0][4]

    # Handle sketchy mu_e inference if not available
    if isinstance(soltab[0][5], (float, np.floating)):
        mu_e = soltab[0][5]
    else:
        mu_e = mu_n - mu_p if mu_p > 0 else 0.0

    if isinstance(soltab[0][6], (float, np.floating)):
        mu_nu = soltab[0][6]
    else:
        mu_nu = 0.0

    muv = [mu_n, mu_p]  # chemical potentials

    # Meson pressure
    meson_pressure = _Lmes_renorm(sigma, w0, r03) if renorm else _Lmes(sigma, w0, r03)

    # Electron pressure
    if electrons:
        P_elec = electron_pressure(T, mu_e)
    else:
        P_elec = 0.0
        if mu_e != 0:
            print("WARNING: electrons have finite chemical potential but electron flag = False")

    # Neutrino pressure
    if neutrinos:
        P_nu = neutrino_pressure(T, mu_nu)
    else:
        P_nu = 0.0
        if mu_nu != 0:
            print("WARNING: neutrinos have finite chemical potential but neutrino flag = False")

    # Photon pressure
    P_photon = photon_pressure(T) if add_photons else 0.0

    # Nucleon pressure
    if mu_p > 0:
        nucpress = sum(
            nucleon_pressure(T, muv[BI], sigma, w0, r03, BI)
            for BI in range(2)
        )
    else:
        nucpress = nucleon_pressure(T, mu_n, sigma, w0, r03, 0)

    return meson_pressure + nucpress + P_elec + P_photon + P_nu + P_offset

# (CHECK AGAIN!!! not so accurate, %2.5 error)
# module to compute entropy density at finite T
def entropy_RMF(soltab, boundmult=40, add_photons=False, integration_method="LocalAdaptive", electrons=True, neutrinos=False):
    """
    Compute entropy density from RMF solution.

    Parameters:
    - soltab: solution list returned by RMFsolve
    - boundmult: integration bound multiplier for entropy integrals
    - add_photons: include photon entropy contribution
    - integration_method: method name placeholder for future compatibility
    - electrons: whether to include electron contribution
    - neutrinos: whether to include neutrino contribution

    Returns:
    - total entropy density
    """
    def adaptive_entropy_integral(integrand, kmin, kmax, tol=1e-10, maxiter=10):
        """
        Adaptive entropy integral for Fermi integrals with better handling of sharp features.
        """
        result = 0.0
        intervals = [(kmin, kmax)]
        iter_count = 0

        while intervals and iter_count < maxiter:
            a, b = intervals.pop()
            mid = 0.5 * (a + b)
            I_full, _ = quad(integrand, a, b, epsabs=tol)
            I_left, _ = quad(integrand, a, mid, epsabs=tol)
            I_right, _ = quad(integrand, mid, b, epsabs=tol)

            if abs((I_left + I_right) - I_full) < tol:
                result += I_full
            else:
                intervals.append((a, mid))
                intervals.append((mid, b))
            iter_count += 1

        if intervals:
            # If still unfinished after maxiter subdivisions, fallback to final quad
            for a, b in intervals:
                I, _ = quad(integrand, a, b, epsabs=tol)
                result += I

        return result

    para = soltab[4]
    _set_couplings(para)

    T = soltab[2]
    nb = soltab[3]
    sigma = soltab[0][0]
    w0 = soltab[0][1]
    r03 = soltab[0][2]
    mun = soltab[0][3]
    mup = soltab[0][4]

    mue = soltab[0][5] if isinstance(soltab[0][5], (int, float)) else (mun - mup if mup > 0 else 0)
    munu = soltab[0][6] if isinstance(soltab[0][6], (int, float)) else 0

    muv = [mun, mup]
    xp = _proton_fraction_RMF(soltab)

    arg_p = 3 * np.pi**2 * xp * nb
    arg_n = 3 * np.pi**2 * (1 - xp) * nb

    kfp = arg_p**(1 / 3) if arg_p >= 0 else 0.0
    kfn = arg_n**(1 / 3) if arg_n >= 0 else 0.0

    kfv = [kfn, kfp]

    arg_e = 3 * np.pi**2 * n_electron(T, mue)
    arg_nu = 3 * np.pi**2 * n_neutrino(T, munu)

    kfe = arg_e**(1 / 3) if arg_e >=0 else 0.0
    kfnu = arg_nu**(1 / 3) if arg_nu >= 0 else 0.0

    if T > 0:
        entr_nucl = 0
        for sgn in [-1, 1]:
            for BI in range(2):
                def integrand(k):
                    fd_val = _fd_bar(k, sigma, sgn * w0, sgn * r03, sgn * muv[BI], BI, T)
                    return k**2 * (
                        (1 - fd_val) * np.clip(np.log(1 - fd_val), -700, 700) +
                        fd_val * np.clip(np.log(fd_val), -700, 700)
                    )
                kmin = max(0, kfv[BI] - boundmult * T)
                kmax = kfv[BI] + boundmult * T
                integral = adaptive_entropy_integral(integrand, kmin, kmax, tol=1e-10) # quad(integrand, kmin, kmax, epsabs=1e-10, limit=1000)[0]
                entr_nucl += -2 / (2 * np.pi)**3 * 4 * np.pi * integral
    else:
        entr_nucl = 0

    if electrons:
        if T > 0:
            entr_electron = 0
            for sgn in [-1, 1]:
                def integrand(k):
                    E = np.sqrt(k**2 + me**2)
                    fd_pos = _fd(E, sgn * mue, -T)
                    fd_neg = _fd(E, sgn * mue, T)
                    return k**2 * (fd_pos * np.clip(np.log(np.clip(fd_pos, 1e-300, 1.0)), -700, 700) +
                                   fd_neg * np.clip(np.log(np.clip(fd_neg, 1e-300, 1.0)), -700, 700))
                kmin = max(0, kfe - boundmult * T)
                kmax = kfe + boundmult * T
                integral = adaptive_entropy_integral(integrand, kmin, kmax, tol=1e-10) # quad(integrand, kmin, kmax, epsabs=1e-10, limit=100000)[0]
                entr_electron += -2 / (2 * np.pi)**3 * 4 * np.pi * integral
        else:
            entr_electron = 0
    else:
        entr_electron = 0

    entr_neutrino = neutrino_entropy(T, munu, 4000) if neutrinos else 0
    entr_photon = _photon_entropy(T) if add_photons else 0

    if not electrons and mue != 0:
        print("WARNING: electrons have finite chemical potential but electron flag = False")
    if not neutrinos and munu != 0:
        print("WARNING: neutrinos have finite chemical potential but neutrino flag = False")

    entropy_total = entr_photon + entr_nucl + entr_electron + entr_neutrino
    return entropy_total

# (CHECK AGAIN!!!) uses entropy function which has errors
# module to compute energy density for given solution of RMF obtained by RMFsolver
# soltab is input created directly from RMVsolve,RMVsolvePNM or RMVsolveSYM
def edens_RMF(soltab, up=5000, boundmult=20, add_photons=False, renorm=False, electrons=True, neutrinos=False):
    """
    Compute energy density from RMF solution `soltab`.

    Parameters:
    - soltab: output from RMF solver (e.g., RMFsolve)
    - up: upper momentum cutoff for numerical integration
    - boundmult: bounds multiplier for entropy integral
    - add_photons: whether to include photon contribution
    - renorm: whether to use renormalized meson pressure
    - electrons: whether to include electrons
    - neutrinos: whether to include neutrinos

    Returns:
    - Thermodynamic energy density
    - Total energy density
    - Relative error between thermo and total
    """
    # Unpack solution and set couplings
    para = soltab[4]
    _set_couplings(para)

    T = soltab[2]
    sig, w0, r03 = soltab[0][:3]
    mun, mup = soltab[0][3:5]
    mue = soltab[0][5] if isinstance(soltab[0][5], (int, float)) else (mun - mup if mup > 0 else 0)
    munu = soltab[0][6] if isinstance(soltab[0][6], (int, float)) else 0
    muv = [mun, mup]

    # Total pressure and entropy
    totpress = pressure_RMF(
        soltab,
        add_photons=add_photons,
        renorm=renorm,
        electrons=electrons,
        neutrinos=neutrinos
    )
    entr = entropy_RMF(
        soltab,
        boundmult=boundmult,
        add_photons=add_photons,
        electrons=electrons,
        neutrinos=neutrinos
    )

    if not electrons and mue != 0:
        print("WARNING: electrons have finite chemical potential but electron flag = False")
    if not neutrinos and munu != 0:
        print("WARNING: neutrinos have finite chemical potential but neutrino flag = False")

    # Thermodynamic energy density
    edens_thermo = -(
        totpress
        - mun * _nB(T, mun, 0, sig, w0, r03)
        - mup * _nB(T, mup, 1, sig, w0, r03)
        - (mue * n_electron(T, mue) if electrons else 0)
        - munu * n_neutrino(T, munu)
    )
    if T > 0.1:
        edens_thermo += T * entr

    # Microscopic meson part
    mespart = (
        _Lmes(sig, w0, r03)
        + 2 * _U_sigma(sig)
        + gr[0] ** 2 * r03 ** 2 * (2 * b1 * w0 ** 2 + 4 * b2 * w0 ** 4 + 6 * b3 * w0 ** 6)
        + 2 * zet / 24 * gw[0] ** 4 * w0 ** 4
        + 2 * xi / 24 * gr[0] ** 4 * r03 ** 4
    )

    # Nucleon energy density
    if T < 0.1:
        edens_nucleons = sum(
            (
                (1 / (8 * np.pi ** 2))
                * (
                    (2 * _kf_bar(sig, w0, r03, muv[BI], BI) ** 3
                     + (mB[BI] - gs[BI] * sig) ** 2 * _kf_bar(sig, w0, r03, muv[BI], BI))
                    * _Ef_bar(sig, w0, r03, muv[BI], BI)
                    - (mB[BI] - gs[BI] * sig) ** 4
                    * np.log(
                        (_Ef_bar(sig, w0, r03, muv[BI], BI) + _kf_bar(sig, w0, r03, muv[BI], BI))
                        / (mB[BI] - gs[BI] * sig)
                    )
                )
            )
            for BI in range(2)
        )
    else:
        def integrand(k, BI, sgn):
            m_eff = mB[BI] - gs[BI] * sig
            Ek = np.sqrt(k ** 2 + m_eff ** 2)
            mu_eff = muv[BI] - gw[BI] * w0 - gr[BI] * I3[BI] * r03
            return k ** 2 * Ek / (1 + np.exp(np.minimum((Ek - sgn * mu_eff) / T, 700)))

        edens_nucleons = sum(
            (8 * np.pi / (2 * np.pi) ** 3)
            * sum(
                quad(lambda k: integrand(k, BI, sgn), 0, up, limit=500)[0]
                for sgn in [-1, 1]
            )
            for BI in range(2)
        )

    # Electron energy density
    if electrons:
        if T <= 0.1:
            edens_electron = mue ** 4 / (4 * np.pi ** 2)
        else:
            edens_electron = sum(
                (8 * np.pi / (2 * np.pi) ** 3)
                * quad(lambda k: k ** 2 * np.sqrt(k ** 2 + me ** 2) / (1 + np.exp(np.minimum((np.sqrt(k ** 2 + me ** 2) - sgn * mue) / T, 700))), 0, up, limit=500)[0]
                for sgn in [-1, 1]
            )
    else:
        edens_electron = 0

    # Photon and neutrino energy densities
    E_photon = photon_energy(T) if add_photons else 0
    edens_neutrino = neutrino_edens(T, munu, up) if neutrinos else 0

    edens_total = edens_nucleons + edens_electron + mespart + E_photon + edens_neutrino

    return edens_thermo, edens_total - P_offset, (edens_thermo - edens_total + P_offset) / (edens_total - P_offset)

# (CHECK AGAIN!!!) uses entropy function which has errors
# module to compute free energy density at finite T
def free_energy_density_RMF(soltab, add_photons=False, electrons=True, neutrinos=False):
    """
    Compute free energy density from a given RMF solution.

    Parameters:
    - soltab: list containing RMF solution
    - add_photons: include photon contribution
    - electrons: include electron contribution
    - neutrinos: include neutrino contribution

    Returns:
    - free energy density (MeV/fm^3)
    """
    # Unpack RMF solution and parameters
    para = soltab[4]
    _set_couplings(para)

    T = soltab[2]
    sig = soltab[0][0]
    w0 = soltab[0][1]
    r03 = soltab[0][2]
    mun = soltab[0][3]
    mup = soltab[0][4]
    mue = soltab[0][5] if isinstance(soltab[0][5], (int, float)) else (mun - mup if mup > 0 else 0)
    munu = soltab[0][6] if isinstance(soltab[0][6], (int, float)) else 0

    muv = [mun, mup]

    # Total pressure
    totpress = pressure_RMF(
        soltab,
        add_photons=add_photons,
        electrons=electrons,
        neutrinos=neutrinos
    )

    # Free energy density
    freeener = -totpress
    for BI in range(2):
        freeener += muv[BI] * _nB(T, muv[BI], BI, sig, w0, r03)

    if electrons:
        freeener += mue * n_electron(T, mue)
    elif mue != 0:
        print("WARNING: electrons have finite chemical potential but electron flag = False")

    if add_photons:
        freeener += photon_freeener(T)

    if neutrinos:
        freeener += munu * n_neutrino(T, munu)
    elif munu != 0:
        print("WARNING: neutrinos have finite chemical potential but neutrino flag = False")

    return freeener

# (CHECK AGAIN!!!) uses entropy function which has errors
# module to compute EOS P(eps) for given solution of RMF obtained by RMFsolver
# soltab is input created directly from RMFsolve,RMFsolvePNM or RMFsolveSYM
def create_EOS(soltab, add_photons=False, electrons=True, neutrinos=False):
    """
    Compute EOS (P vs energy density) for given RMF solution.

    Parameters:
    soltab       : RMF solver output
    add_photons  : include photon contribution (default: False)
    electrons    : include electrons (default: True)
    neutrinos    : include neutrinos (default: False)

    Returns:
    [pressure, energy_density, mu_n] : list of thermodynamic values
    """
    para = soltab[4]  # fifth element is parameter set
    _set_couplings(para)

    T = soltab[2]
    sig = soltab[0][0]
    w0 = soltab[0][1]
    r03 = soltab[0][2]
    mun = soltab[0][3]
    mup = soltab[0][4]

    mue = soltab[0][5] if isinstance(soltab[0][5], (int, float)) else (mun - mup if mup > 0 else 0)
    munu = soltab[0][6] if len(soltab[0]) > 6 and isinstance(soltab[0][6], (int, float)) else 0

    if not electrons and mue != 0:
        print("WARNING: electrons have finite chemical potential but electron flag = False")
    if not neutrinos and munu != 0:
        print("WARNING: neutrinos have finite chemical potential but neutrino flag = False")

    totpress = pressure_RMF(
        soltab,
        add_photons=add_photons,
        electrons=electrons,
        neutrinos=neutrinos
    )

    edens = edens_RMF(
        soltab,
        add_photons=add_photons,
        electrons=electrons,
        neutrinos=neutrinos
    )[0]

    return [totpress, edens, mun]

# (CHECK AGAIN!!!) uses entropy function which has errors
# module to compute binding energy for given solution of RMF obtained by RMFsolver,RMVsolvePNM or RMVsolveSYM
# soltab is input created directly from RMVsolveSYM
def binding_energy_RMF(soltab, add_photons=False, renorm=False, electrons=True, neutrinos=False):
    """
    Compute binding energy for given RMF solution.
    soltab: output from RMFsolve, RMFsolvePNM, or RMFsolveSYM
    """

    para = soltab[4]
    _set_couplings(para)

    T = soltab[2]
    sig = soltab[0][0]
    w0 = soltab[0][1]
    r03 = soltab[0][2]
    mun = soltab[0][3]
    mup = soltab[0][4]
    dens = soltab[3]

    if isinstance(soltab[0][5], (float, int)):
        mue = soltab[0][5]
    else:
        mue = mun - mup if mup > 0 else 0

    if isinstance(soltab[0][6], (float, int)):
        munu = soltab[0][6]
    else:
        munu = 0

    if not electrons and mue != 0:
        print("WARNING: electrons have finite chemical potential but electron flag = False")
    if not neutrinos and munu != 0:
        print("WARNING: neutrinos have finite chemical potential but neutrino flag = False")

    edens = edens_RMF(
        soltab,
        add_photons=add_photons,
        renorm=renorm,
        electrons=electrons,
        neutrinos=neutrinos
    )[0]

    bind = (edens + P_offset) / dens - mB[0]  # mB[0] corresponds to mB[[1]] in Mathematica

    return bind

# compute dU threshold in beta equilibrated npe matter, input is para list
def dU_threshold(para):
    """
    Compute the dU threshold in beta-equilibrated npe matter.
    Input:
        para: list of model parameters for RMF
    Output:
        moms_surplus: list of (nB/n0, momentum surplus)
        dU_threshold_out: either the value of dU threshold or "No dU threshold found between 0.5 and 10 n0"
    """
    _set_couplings(para)

    nucdens = 1.2293605098323745e6  # corresponds to 0.16 fm^-3
    ss = -30 if para[3][0] < 0 else 30

    soltab_tab = RMFsolve_nb(0.5 * nucdens, 10 * nucdens, 100, 0, para, ss, 20, -3, 990, 100, verb=False)

    kfntab = [(row[0], _kf_bar(row[1][0][0], row[1][0][1], row[1][0][2], row[1][0][3], 0)) for row in soltab_tab]
    kfptab = [(row[0], _kf_bar(row[1][0][0], row[1][0][1], row[1][0][2], row[1][0][4], 1)) for row in soltab_tab]
    kfetab = [(row[0], _kF_electron(row[1][0][3] - row[1][0][4])) for row in soltab_tab]

    moms_surplus = [(kfntab[i][0] / nucdens, -kfntab[i][1] + kfptab[i][1] + kfetab[i][1]) for i in range(len(soltab_tab))]

    # Interpolation and root finding
    nB_arr, surplus_arr = zip(*moms_surplus)
    interp_func = interp1d(nB_arr, surplus_arr, kind='linear', fill_value='extrapolate')

    try:
        dUthr = brentq(interp_func, 3.0, 9.999)
        dUthrout = dUthr if 0.5 <= dUthr <= 10 else "No dU threshold found between 0.5 and 10 n0"
    except ValueError:
        dUthrout = "No dU threshold found between 0.5 and 10 n0"

    return moms_surplus, dUthrout

#



