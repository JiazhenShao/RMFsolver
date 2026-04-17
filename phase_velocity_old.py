import numpy as np
import math
import matplotlib.pyplot as plt
from fractions import Fraction
from scipy.interpolate import interp1d
from scipy.integrate import quad, solve_bvp, solve_ivp
from scipy.optimize import fsolve, approx_fprime, root_scalar, root, least_squares, minimize_scalar
from numdifftools import Gradient, Derivative
import RMFsolver.constants as const
import RMFsolver.RMFparameter as para
from RMFsolver.Solver import RMFsolve_mu, RMFpressureSYM, RMFpressurePNM, pressure_RMF



# Public functions
__all__ = ["P_f", "n_B", "PQM", "PQM_em", "vNtoQ_Pc", "vNtoQ_B", "vNtoQ_nc", "Get_Delta_n_max",
           "Get_Delta_P_max", "extract_contour_coords_num", "extract_contour_coords_ana",
           "z_time_evolution"]


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

        return float( integral / (np.pi**2) )

    else:
        if m < 1e-3:
            return float( mu**4 / (12 * np.pi**2) )
        else:
            k_F = np.sqrt(mu**2 - m**2)
            return float( ((2*k_F**3 - 3*m**2*k_F)*mu + 3*m**4*np.log((k_F+mu)/m)) / (24*np.pi**2) )

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

        return float( integral / (np.pi**2) )

    else:
        if mu > m:
            k_F = np.sqrt(mu**2 - m**2)
            return float( k_F**3 / (3 * np.pi**2) )
        else:
            return float( 0.0 )


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

# Use numerical method to calculate velocity
def _vNtoQ_num(T, aQstar, aN, muQ):
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

    # kappa(v) from the linearized tail (η_Q must be present!)
    def get_kappa(v):
        return 2 * D / (-v + np.sqrt(v*v + 4 * D * eta / tau))

    # Nonlinearity
    def R(a):
        a_safe = np.clip(a, -7e2, 7e2)
        return (a_safe**3 + eta * a_safe) / tau

    # ODE in t = \tilde{x}: y[0]=a(t), y[1]=p(t)=da/dx
    def ode_system(t, y, v, kappa):
        a = y[0]
        p = y[1]
        dxdt = kappa / (c1 - t)                # dx/dt
        a_t  = p * dxdt
        p_t  = ((v * p + R(a)) / D) * dxdt
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
            rmf_sol = RMFsolve_mu(mub = mu_B, Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mu_e_init = 50, verb = False
                )
            return (pressure_RMF(rmf_sol)).item()

        elif NM_type == "PNM":
            pre = RMFpressurePNM(input_num = mu_B, input_type = "muB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mub_init = 990, verb = False
                )
            return float(pre.item())

        elif NM_type == "SYM":
            pre = RMFpressureSYM(input_num = mu_B, input_type = "muB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, mub_init = 990, verb = False
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
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit - DelP
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
            rmf_sol = RMFsolve_mu(mub = mu_B, Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mu_e_init = 50, verb = False
                )
            return (pressure_RMF(rmf_sol)).item()

        elif NM_type == "PNM":
            pre = RMFpressurePNM(input_num = mu_B, input_type = "muB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mub_init = 990, verb = False
                )
            return float(pre.item())

        elif NM_type == "SYM":
            pre = RMFpressureSYM(input_num = mu_B, input_type = "muB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, mub_init = 990, verb = False
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
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit - DelP
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

    Returns:
    - velocity of phase boundary in meters per second
    '''

    def PNM(mu_B, Temp):
        if NM_type == "Beta_eq":
            rmf_sol = RMFsolve_mu(mub = mu_B, Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mu_e_init = 50, verb = False
                )
            return (pressure_RMF(rmf_sol)).item()

        elif NM_type == "PNM":
            pre = RMFpressurePNM(input_num = mu_B, input_type = "muB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mub_init = 990, verb = False
                )
            return float(pre.item())

        elif NM_type == "SYM":
            pre = RMFpressureSYM(input_num = mu_B, input_type = "muB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, mub_init = 990, verb = False
                )
            return float(pre.item())

        else:
            return ValueError("Nuclear matter type not defined.")


    def PNM_n(nB, Temp):
        if NM_type == "Beta_eq":
            rmf_sol = RMFsolve(nbext = nB, Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mu_e_init = 50, verb = False
                )
            return (pressure_RMF(rmf_sol)).item()

        elif NM_type == "PNM":
            pre = RMFpressurePNM(input_num = nB, input_type = "nB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mub_init = 990, verb = False
                )
            return float(pre.item())

        elif NM_type == "SYM":
            pre = RMFpressureSYM(input_num = nB, input_type = "nB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, mub_init = 990, verb = False
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
        PQM_solve_for_muBstar = lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit - DelP
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



# Calculates \Delta n max for given ncrit and T
def Get_Delta_n_max(T, n_crit, m_s, param, NM_type, method, tol=1e-2, coarse_pts=5, bounds=None):
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
    - method   : ways to calculate velocity
                 - "numerical" in principle more precise, but have numerical noise
                 - "analytical" analytical approximation, direct, faster, and robust, ~ 10% difference with numerical method

    Returns:
    - Maximum pussible Delta_n before aQstar > aN
    '''

    def PNM(mu_B, Temp):
        if NM_type == "Beta_eq":
            rmf_sol = RMFsolve_mu(mub = mu_B, Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mu_e_init = 50, verb = False
                )
            return (pressure_RMF(rmf_sol)).item()

        elif NM_type == "PNM":
            pre = RMFpressurePNM(input_num = mu_B, input_type = "muB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mub_init = 990, verb = False
                )
            return float(pre.item())

        elif NM_type == "SYM":
            pre = RMFpressureSYM(input_num = mu_B, input_type = "muB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, mub_init = 990, verb = False
                )
            return float(pre.item())

        else:
            return ValueError("Nuclear matter type not defined.")


    def PNM_n(nB, Temp):
        if NM_type == "Beta_eq":
            rmf_sol = RMFsolve(nbext = nB, Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mu_e_init = 50, verb = False
                )
            return (pressure_RMF(rmf_sol)).item()

        elif NM_type == "PNM":
            pre = RMFpressurePNM(input_num = nB, input_type = "nB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mub_init = 990, verb = False
                )
            return float(pre.item())

        elif NM_type == "SYM":
            pre = RMFpressureSYM(input_num = nB, input_type = "nB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, mub_init = 990, verb = False
                )
            return float(pre.item())

        else:
            return ValueError("Nuclear matter type not defined.")


    # --- compute the expensive, Δn‑independent pieces ONCE ---
    P_crit = PNM_n(n_crit, T)

    # small wrapper to keep fsolve snappy
    def fsolve_fast(fun, x0):
        return fsolve(fun, x0, xtol=1e-6, maxfev=400)[0]

    muB_crit = fsolve_fast(lambda x: PNM(x, T) - P_crit, 1050.0)
    B_SQM    = fsolve_fast(lambda B: PQM(muB_crit, 0, B, T, m_s) - P_crit, 180.0)

    def Get_aN_aQstar_diff(Delta_n):
        # solve for points Q and N at beta eq. (these DO depend on Δn)
        DelP = PNM_n(n_crit + Delta_n, T) - P_crit

        muB_N = fsolve_fast(lambda x: PNM(x, T) - P_crit - DelP, 1050.0)
        muB_Q = fsolve_fast(lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP, 1050.0)

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
            muB_star = fsolve_fast(lambda x: PQM(x, muK_star, B_SQM, T, m_s) - P_crit - DelP, muB_Q)

        else:
            raise RuntimeError(f"Input method unknown")
            return None

        # calculating aN
        PNM_wrap = lambda x: PNM(x, T)
        dPNM_dmuB = Derivative(PNM_wrap, n=1)
        nB_N = dPNM_dmuB(muB_N)
        aN = (nB_N - nK_Q) / nB_Q   

        diff = float(np.abs(aN - aQstar))
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



# Calculates ΔP max for given Pcrit and T
def Get_Delta_P_max(T, P_crit, m_s, param, NM_type, method, tol=1e-2, coarse_pts=5, bounds=None):
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
    - method   : ways to calculate velocity
                 - "numerical" in principle more precise, but have numerical noise
                 - "analytical" analytical approximation, direct, faster, and robust, ~ 10% difference with numerical method

    Returns:
    - Maximum possible Delta_P before aQstar > aN
    '''

    def PNM(mu_B, Temp):
        if NM_type == "Beta_eq":
            rmf_sol = RMFsolve_mu(mub = mu_B, Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mu_e_init = 50, verb = False
                )
            return (pressure_RMF(rmf_sol)).item()

        elif NM_type == "PNM":
            pre = RMFpressurePNM(input_num = mu_B, input_type = "muB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, r03_init = -3, mub_init = 990, verb = False
                )
            return float(pre.item())

        elif NM_type == "SYM":
            pre = RMFpressureSYM(input_num = mu_B, input_type = "muB", Trmf = Temp, para = param, 
                sigma_init = 30, w0_init = 20, mub_init = 990, verb = False
                )
            return float(pre.item())

        else:
            return ValueError("Nuclear matter type not defined.")

    # --- compute the expensive, ΔP-independent pieces ONCE ---

    # small wrapper to keep fsolve snappy
    def fsolve_fast(fun, x0):
        return fsolve(fun, x0, xtol=1e-6, maxfev=400)[0]

    muB_crit = fsolve_fast(lambda x: PNM(x, T) - P_crit, 1050.0)
    B_SQM    = fsolve_fast(lambda B: PQM(muB_crit, 0, B, T, m_s) - P_crit, 180.0)

    def Get_aN_aQstar_diff(Delta_P):
        # solve for points Q and N on the shifted isobar (these DO depend on ΔP)
        DelP = float(Delta_P)

        muB_N = fsolve_fast(lambda x: PNM(x, T) - P_crit - DelP, 1050.0)
        muB_Q = fsolve_fast(lambda x: PQM(x, 0, B_SQM, T, m_s) - P_crit - DelP, 1050.0)

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
            # NOTE: target is P_crit + DelP (NOT P_crit)
            muB_star = fsolve_fast(lambda x: PQM(x, muK_star, B_SQM, T, m_s) - (P_crit + DelP), muB_Q)

        else:
            raise RuntimeError(f"Input method unknown")
            return None

        # calculating aN
        PNM_wrap = lambda x: PNM(x, T)
        dPNM_dmuB = Derivative(PNM_wrap, n=1)
        nB_N = dPNM_dmuB(muB_N)
        aN = (nB_N - nK_Q) / nB_Q   

        diff = float(np.abs(aN - aQstar))
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




# Get the distance $z$ between phase boundary and isobar as function of time
def z_time_evolution(Pressure_arr, radius_arr, P_c, T, t_up=20, Interp1d_Num=10, ms=0, para = para.paraQMCRMF3, NM_ty = "PNM", meth = "numerical"):
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
    - meth         : method, "analytical" or "numerical" (default, as robust and more accurate)

    --------------------------- 
    Returns:
    - t, z
        t: an array of time
        z: an array of z, function of t
    '''

    # 0) Safety: interpolation needs increasing x
    order = np.argsort(radius_arr)
    radius_arr = np.asarray(radius_arr, float)[order]
    Pressure_arr = np.asarray(Pressure_arr, float)[order]

    # 1) P(r) and locate z_crit from P(r)=P_c
    P_of_r = interp1d(radius_arr, Pressure_arr, kind="linear", fill_value="extrapolate")
    z_crit = float(fsolve(lambda zz: P_of_r(zz) - P_c, x0=3.0)[0])

    # 2) initial gap z0 from Delta P_max
    DelP_max = Get_Delta_P_max(T, P_c, ms, para, NM_ty, meth)
    z_at_Pmax = float(fsolve(lambda zz: P_of_r(zz) - (P_c + DelP_max), x0=3.0)[0])
    z0 = z_crit - z_at_Pmax
    if z0 <= 0:
        raise ValueError(f"Computed z0 <= 0 (z0={z0}); check inputs/units.")

    # 3) Pre-tabulate v(z) on [0, z0] and build cheap interpolant
    Interp1d_Num = max(int(Interp1d_Num), 2)  # need at least 2 points
    z_list = np.linspace(0.0, z0, Interp1d_Num)

    def vz_accurate(z):
        # z can be scalar or array; ensure scalar use here
        z = float(np.asarray(z).reshape(()))
        DelP_P_c = P_of_r(z_crit - z)
        vel, _ = vNtoQ_Pc(T=T, P_crit=P_c, DelP=float(DelP_P_c) - P_c,
                          m_s=ms, param=para, NM_type=NM_ty, method=meth)
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
    return t, z




#

