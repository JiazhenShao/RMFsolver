import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve, approx_fprime, root_scalar, root, least_squares
from numdifftools import Gradient, Derivative
import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver.Solver import RMFsolve_mu, RMFpressureSYM, RMFpressurePNM, pressure_RMF




# Public functions
__all__ = ["P_f", "n_B", "PQM", "PQM_em", "vNtoQ_Pc", "vNtoQ_B", "vNtoQ_nc"]


# pressure for a single free fermion
def P_f(mu, m, Tem, upB=5000):
    '''
    Computes pressure for a single fermion species

    Parameters:
    - mu   : fermion chemical potential
    - m    : fermion mass
    - Tem  : temperature
    - upB  : upper limit for infinite integral, default=5000

    Returns:
    - pressure of this fermion species
    '''

    if Tem > 0.2:
        def integrand(k):
            Ek = np.sqrt(k**2 + m**2)
            arg = Tem * np.log(1 + np.exp(-np.clip((Ek - mu)/Tem, -700, 700)))
            arg_bar = Tem * np.log(1 + np.exp(-np.clip((Ek + mu)/Tem, -700, 700)))
            return (arg + arg_bar) * k**2

        integral, _ = quad(integrand, 0, upB, epsabs=1e-10, epsrel=1e-8)

        return integral / (np.pi**2)

    else:
        if m < 1e-3:
            return mu**4 / (12 * np.pi**2)
        else:
            k_F = np.sqrt(mu**2 - m**2)
            return ((2*k_F**3 - 3*m**2*k_F)*mu + 3*m**4*np.log((k_F+mu)/m)) / (24*np.pi**2)

# number density for a single free fermion
def n_B(mu, m, Tem, upB=5000):
    '''
    Returns number density for a single fermion species.
    Uses thermodynamic relation: dP/dmu = n
    '''

    if Tem > 0.2:
        def integrand(k):
            Ek = np.sqrt(k**2 + m**2)
            f = 1 / (1 + np.exp(np.clip((Ek - mu)/Tem, -700, 700)))
            f_bar = 1 / (1 + np.exp(np.clip((Ek + mu)/Tem, -700, 700)))
            return (f - f_bar) * k**2

        integral, _ = quad(integrand, 0, upB, epsabs=1e-10, epsrel=1e-8)

        return integral / (np.pi**2)

    else:
        if mu > m:
            k_F = np.sqrt(mu**2 - m**2)
            return k_F**3 / (3 * np.pi**2)
        else:
            return 0.0

# pressure for SQM under bag model
def PQM(muB, muK, B_one_forth, T, ms, upB=5000):
    '''
    Calculates the pressure of strange quark matter (SQM) under bag model

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
    mu_u = float(muB/3)
    mu_d = float(muB/3 + muK/2)
    mu_s = float(muB/3 - muK/2)

    return float( 16*np.pi**2*T**4 / 90 + 3*P_f(mu_u, m=0, Tem=T) + 3*P_f(mu_d, m=0, Tem=T) + 3*P_f(mu_s, ms, Tem=T) - B )

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

    def equation(mu_u):
        mu_d = (muB + muK - mu_u)/2
        mu_s = (muB - muK - mu_u)/2
        mu_e = (muB - 3*mu_u)/2
        return float(n_B(mu_e,0,T) + n_B(mu_s,ms,T)/3 + n_B(mu_d,0,T)/3 - 2*n_B(mu_u,0,T)/3)

    sol = root(equation, muQ_init, method='hybr', options={'maxfev': 60000})

    if not sol.success:
        print("Root finding failed:", sol.message)
        raise RuntimeError("PQM failed to converge")

    mu_u = sol.x
    mu_d = (muB + muK - mu_u)/2
    mu_s = (muB - muK - mu_u)/2
    mu_e = (muB - 3*mu_u)/2

    return float( 16*np.pi**2*T**4 / 90 + P_f(mu_e, m=0.511, Tem=T) + 3*P_f(mu_u, m=0, Tem=T) + 3*P_f(mu_d, m=0, Tem=T) + 3*P_f(mu_s, ms, Tem=T) - B )




# extract coordinates of a contour
def _extract_contour_coords_num(X, Y, Z, level):
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

    fig, ax = plt.subplots()
    contour_obj = ax.contour(X, Y, Z, levels=[level])

    # Directly access the paths from the contour object
    paths = []
    for c in contour_obj.get_paths():
        paths.append(c.vertices)
    plt.close(fig)  # Close the plot if you only want data
    if not paths:
        raise RuntimeError(f"No contours found at level {level}")
    X_coor = paths[0][:, 0]
    Y_coor = paths[0][:, 1]

    return X_coor, Y_coor 

def _extract_contour_coords_ana(func, x_range, y_list, level, bracket=None, method='brentq'):
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
        mu_u = muB / 3
        mu_d = muB / 3 + muK / 2
        mu_s = muB / 3 - muK / 2

        n_u = n_B(mu_u, 0, T, upB)
        n_d = n_B(mu_d, 0, T, upB)
        n_s = n_B(mu_s, ms, T, upB)

        return (1/3) * (n_u + n_d + n_s)

    def dPQM_dmuK(muB, muK):
        mu_d = muB / 3 + muK / 2
        mu_s = muB / 3 - muK / 2

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

        return [eq1, eq2]

    res = least_squares(system, initial_guess, bounds=([0, 0], [np.inf, np.inf]))

    if not res.success:
        raise RuntimeError(f"Root finding failed: {res.message}")

    if res.x[1] < 0:
        raise RuntimeError(f"muK_star < 0")

    return res.x[0], res.x[1]



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



# Taking transition pressure, strange quark mass, NM model name as input
def vNtoQ_Pc(T, P_crit, DelP, m_s, param, NM_type, method):
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

    Returns:
    - velocity of phase boundary in meters per second
    '''

    def PNM(mu_B, Temp):
        if NM_type == "Beta_eq":
            rmf_sol = RMFsolve_mu(
                mub = mu_B,
                Trmf = Temp,
                para = param,
                sigma_init = 30,
                w0_init = 20,
                r03_init = -3,
                mu_e_init = 50,
                verb = False
                )
            return (pressure_RMF(rmf_sol)).item()

        elif NM_type == "PNM":
            pre = RMFpressurePNM(
                input_num = mu_B,
                input_type = "muB", 
                Trmf = Temp, 
                para = param, 
                sigma_init = 30, 
                w0_init = 20, 
                r03_init = -3, 
                mub_init = 990,
                verb = False
                )
            return float(pre.item())

        elif NM_type == "SYM":
            pre = RMFpressureSYM(
                input_num = mu_B,
                input_type = "muB", 
                Trmf = Temp, 
                para = param, 
                sigma_init = 30, 
                w0_init = 20, 
                mub_init = 990,
                verb = False
                )
            return float(pre.item())

        else:
            return ValueError("Nuclear matter type not defined.")


    # solve for bag constant
    P_diff = lambda x: PNM(x, T) - P_crit
    muB_crit = fsolve(P_diff, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit, 0, x, T, m_s) - P_crit
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]

    # solve for points Q and N at beta eq.
    PNM_minus_PShift = lambda x: PNM(x,T) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if method == "numerical":
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
 
    elif method == "analytical":
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

    else:
        raise RuntimeError(f"Input method unknown")
        return None

    # calculating aN
    PNM_wrap = lambda x: PNM(x, T)
    dPNM_dmuB = Derivative(PNM_wrap, n=1)
    nB_N = dPNM_dmuB(muB_N)
    aN = (nB_N - nK_Q) / nB_Q   

    # calculating velocity
    vel = _vNtoQ_formula(T, aQstar, aN, muB_star/3)

    return vel, B_SQM


# Taking SQM bag constant, strange quark mass, NM model name as input
def vNtoQ_B(T, B_SQM, DelP, m_s, param, NM_type, method):
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

    Returns:
    - velocity of phase boundary in meters per second
    '''

    def PNM(mu_B, Temp):
        if NM_type == "Beta_eq":
            rmf_sol = RMFsolve_mu(
                mub = mu_B,
                Trmf = Temp,
                para = param,
                sigma_init = 30,
                w0_init = 20,
                r03_init = -3,
                mu_e_init = 50,
                verb = False
                )
            return (pressure_RMF(rmf_sol)).item()

        elif NM_type == "PNM":
            pre = RMFpressurePNM(
                input_num = mu_B,
                input_type = "muB", 
                Trmf = Temp, 
                para = param, 
                sigma_init = 30, 
                w0_init = 20, 
                r03_init = -3, 
                mub_init = 990,
                verb = False
                )
            return float(pre.item())

        elif NM_type == "SYM":
            pre = RMFpressureSYM(
                input_num = mu_B,
                input_type = "muB", 
                Trmf = Temp, 
                para = param, 
                sigma_init = 30, 
                w0_init = 20, 
                mub_init = 990,
                verb = False
                )
            return float(pre.item())

        else:
            return ValueError("Nuclear matter type not defined.")


    # solve for critical point
    P_diff = lambda x: PQM(x, 0, B_SQM, T, m_s) - PNM(x, T)
    muB_crit = fsolve(P_diff, 1050)[0]
    P_crit = PQM(muB_crit, 0, B_SQM, T, m_s)

    # solve for points Q and N at beta eq.
    PNM_minus_PShift = lambda x: PNM(x,T) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if method == "numerical":
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
 
    elif method == "analytical":
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

    else:
        raise RuntimeError(f"Input method unknown")
        return None

    # calculating aN
    PNM_wrap = lambda x: PNM(x, T)
    dPNM_dmuB = Derivative(PNM_wrap, n=1)
    nB_N = dPNM_dmuB(muB_N)
    aN = (nB_N - nK_Q) / nB_Q   

    # calculating velocity
    vel = _vNtoQ_formula(T, aQstar, aN, muB_star/3)

    return vel, P_crit


# Taking transition density, strange quark mass, NM model name as input
def vNtoQ_nc(T, n_crit, Deln, m_s, param, NM_type, method):
    '''
    Computes NM to SQM phase boundary velocity 

    Parameters:
    - Trmf     : temperature
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

    Returns:
    - velocity of phase boundary in meters per second
    '''

    def PNM(mu_B, Temp):
        if NM_type == "Beta_eq":
            rmf_sol = RMFsolve_mu(
                mub = mu_B,
                Trmf = Temp,
                para = param,
                sigma_init = 30,
                w0_init = 20,
                r03_init = -3,
                mu_e_init = 50,
                verb = False
                )
            return (pressure_RMF(rmf_sol)).item()

        elif NM_type == "PNM":
            pre = RMFpressurePNM(
                input_num = mu_B,
                input_type = "muB", 
                Trmf = Temp, 
                para = param, 
                sigma_init = 30, 
                w0_init = 20, 
                r03_init = -3, 
                mub_init = 990,
                verb = False
                )
            return float(pre.item())

        elif NM_type == "SYM":
            pre = RMFpressureSYM(
                input_num = mu_B,
                input_type = "muB", 
                Trmf = Temp, 
                para = param, 
                sigma_init = 30, 
                w0_init = 20, 
                mub_init = 990,
                verb = False
                )
            return float(pre.item())

        else:
            return ValueError("Nuclear matter type not defined.")


    def PNM_n(nB, Temp):
        if NM_type == "Beta_eq":
            rmf_sol = RMFsolve(
                nbext = nB,
                Trmf = Temp,
                para = param,
                sigma_init = 30,
                w0_init = 20,
                r03_init = -3,
                mu_e_init = 50,
                verb = False
                )
            return (pressure_RMF(rmf_sol)).item()

        elif NM_type == "PNM":
            pre = RMFpressurePNM(
                input_num = nB,
                input_type = "nB", 
                Trmf = Temp, 
                para = param, 
                sigma_init = 30, 
                w0_init = 20, 
                r03_init = -3, 
                mub_init = 990,
                verb = False
                )
            return float(pre.item())

        elif NM_type == "SYM":
            pre = RMFpressureSYM(
                input_num = nB,
                input_type = "nB", 
                Trmf = Temp, 
                para = param, 
                sigma_init = 30, 
                w0_init = 20, 
                mub_init = 990,
                verb = False
                )
            return float(pre.item())

        else:
            return ValueError("Nuclear matter type not defined.")


    # solve for bag constant
    P_crit = PNM_n(n_crit, T)
    P_diff = lambda x: PNM(x, T) - P_crit
    muB_crit = fsolve(P_diff, 1050)[0]
    PQM_solve_for_B = lambda x: PQM(muB_crit, 0, x, T, m_s) - P_crit
    B_SQM = fsolve(PQM_solve_for_B, 180)[0]

    # solve for points Q and N at beta eq.
    DelP = PNM_n(n_crit + Deln, T) - P_crit
    PNM_minus_PShift = lambda x: PNM(x,T) - P_crit - DelP
    muB_N = fsolve(PNM_minus_PShift, 1050)[0]

    PQM_minus_Pshift = lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP
    muB_Q = fsolve(PQM_minus_Pshift, 1050)[0]

    # calculating aQstar
    if method == "numerical":
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
 
    elif method == "analytical":
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

    else:
        raise RuntimeError(f"Input method unknown")
        return None

    # calculating aN
    PNM_wrap = lambda x: PNM(x, T)
    dPNM_dmuB = Derivative(PNM_wrap, n=1)
    nB_N = dPNM_dmuB(muB_N)
    aN = (nB_N - nK_Q) / nB_Q   

    # calculating velocity
    vel = _vNtoQ_formula(T, aQstar, aN, muB_star/3)

    return vel, B_SQM



#









