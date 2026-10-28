"""Static gray-region boundaries of the isothermal (n_B(0-), T(0-)) plane.

Lower gray region: pure neutron matter is stable against equilibrated uds quark
matter at the same pressure and temperature, so the front speed is zero.
Upper gray region: neutron matter converts directly to udd quark matter
(a(0+) = 1), with no weak reaction or strangeness diffusion, so the speed
diverges. Both boundaries are thermodynamic: independent of transport, xi and
the front solver, and evaluated with the same equations the solvers use.
"""
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from RMFsolver import phase_velocity as pv
import _iso_common as common

BAG = common.PHYSICS["B_one_forth"]


@lru_cache(maxsize=8192)
def _upstream(x, T):
    return pv._validated_pnm_state_from_nB(x * common.N0_MEV3, T)


def coexistence_residual(x, T):
    """muB of equilibrated uds quark matter at P(0-), T minus muB(0-); > 0 where PNM is stable."""
    up = _upstream(float(x), float(T))
    return float(pv._slow_isobar_state(up["P_0minus"], 0.0, float(T), BAG, 0.0)["muB"]
                 - up["muB_0minus"])


def udd_residual(x, T):
    """Minus the a(0+) = 1 Gibbs balance of _solve_a_0plus_max; > 0 where udd matter is favored."""
    up = _upstream(float(x), float(T))
    muB, muK = pv._solve_interface_0plus_from_local_a_and_Pi(
        common.A_CEILING, up["P_0minus"], 0.0, BAG, float(T))
    return float(up["muB_0minus"] - muB - common.A_CEILING * muK)


def _safe(function, x, T):
    try:
        return function(x, T)
    except (RuntimeError, ValueError, FloatingPointError):
        return np.nan


def _sign_changes(values, falling):
    """Indices k with a sign change between finite values k and k+1."""
    finite = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    if falling:
        return np.flatnonzero(finite & (values[:-1] > 0) & (values[1:] <= 0))
    return np.flatnonzero(finite & (values[:-1] < 0) & (values[1:] >= 0))


def _root(function, point, grid, values, falling, label):
    changes = _sign_changes(values, falling)
    if len(changes) != 1:
        raise RuntimeError(f"{label}: expected one boundary root, found {len(changes)}")
    k = changes[0]
    return float(brentq(lambda s: function(*point(s)), grid[k], grid[k + 1], xtol=1e-12))


def ray_boundaries(theta, rho_step=0.05):
    """Inner (coexistence) and outer (udd) radii on one polar ray, 0 < theta <= pi/2."""
    theta = float(theta)
    if not 0 < theta <= np.pi / 2:
        raise ValueError("theta must lie in (0, pi/2]")
    point = lambda rho: tuple(float(v) for v in common.to_physical(rho, theta))
    limits = [125.0 / (common.POLAR["T_scale"] * np.sin(theta))]
    if np.cos(theta) > 1e-12:
        limits.append((6.2 - common.POLAR["x_origin"]) / (common.POLAR["x_scale"] * np.cos(theta)))
    grid = np.arange(0.01, min(limits) + rho_step, rho_step)
    coexistence = np.array([_safe(coexistence_residual, *point(r)) for r in grid])
    udd = np.array([_safe(udd_residual, *point(r)) for r in grid])
    rho_inner = _root(coexistence_residual, point, grid, coexistence, True, f"theta={theta} inner")
    rho_outer = _root(udd_residual, point, grid, udd, False, f"theta={theta} outer")
    if not rho_outer > rho_inner:
        raise RuntimeError(f"theta={theta}: udd boundary does not lie outside coexistence")
    (x_inner, T_inner), (x_outer, T_outer) = point(rho_inner), point(rho_outer)
    return dict(theta=theta, rho_inner=rho_inner, rho_outer=rho_outer, x_inner=x_inner,
                T_inner=T_inner, x_outer=x_outer, T_outer=T_outer)


def fixed_T_window(T, x_window=common.DENSITY_WINDOW, x_step=0.05):
    """Moving-front density interval at fixed T, clipped to the plotting window."""
    T = float(T)
    point = lambda x: (float(x), T)
    grid = np.linspace(x_window[0], x_window[1], int(round((x_window[1] - x_window[0]) / x_step)) + 1)
    coexistence = np.array([_safe(coexistence_residual, x, T) for x in grid])
    udd = np.array([_safe(udd_residual, x, T) for x in grid])
    window = dict(T=T, status="open", inner_clipped=False, outer_clipped=False,
                  inner_root=np.nan, outer_root=np.nan)
    if np.all(coexistence[np.isfinite(coexistence)] > 0) or np.all(udd[np.isfinite(udd)] >= 0):
        return dict(window, status="closed", x_inner=np.nan, x_outer=np.nan)
    if coexistence[0] <= 0:
        window.update(x_inner=float(x_window[0]), inner_clipped=True)
    else:
        window["inner_root"] = window["x_inner"] = _root(
            coexistence_residual, point, grid, coexistence, True, f"T={T} inner")
    if udd[-1] < 0:
        window.update(x_outer=float(x_window[1]), outer_clipped=True)
    else:
        window["outer_root"] = window["x_outer"] = _root(
            udd_residual, point, grid, udd, False, f"T={T} outer")
    if not window["x_outer"] > window["x_inner"]:
        window.update(status="closed")
    return window


def _row_points(T):
    window = fixed_T_window(T, x_window=(1.0, 6.0))
    return T, window["inner_root"], window["outer_root"]


def _ray_points(theta):
    ray = ray_boundaries(theta)
    return ray["x_inner"], ray["T_inner"], ray["x_outer"], ray["T_outer"]


def boundary_curves(n_rows=40, n_rays=81, T_rows_max=10.0, workers=1):
    """Smooth inner/outer boundary curves for the gray regions of both figures.

    Fixed-T rows on a logarithmic grid cover 0.01 MeV to T_rows_max, where the
    boundaries are nearly vertical; Chebyshev-clustered rays cover the upper
    part, where they bend toward the n_0 edge. Each curve is sorted by T.
    """
    rows = common.map_parallel(_row_points, np.geomspace(0.01, T_rows_max, n_rows), workers)
    t = (1 - np.cos(np.pi * np.arange(1, n_rays) / (n_rays - 1))) / 2
    rays = common.map_parallel(_ray_points, np.pi / 2 * t, workers)
    curves = {}
    for name, columns in (("inner", (0, 1)), ("outer", (2, 3))):
        points = [(x, T) for T, *roots in rows
                  for x in [roots[columns[0] // 2]] if np.isfinite(x)]
        points += [(ray[columns[0]], ray[columns[1]]) for ray in rays if ray[columns[1]] > T_rows_max]
        points = np.array(sorted(points, key=lambda p: p[1]))
        curves[name] = (points[:, 0], points[:, 1])
    return curves
