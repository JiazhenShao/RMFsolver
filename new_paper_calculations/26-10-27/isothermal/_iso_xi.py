"""Reusable xi cache for the isothermal analytical contours (no EOS or solver imports).

At every accepted analytic mesh point the manuscript speed is

    v(xi) = c*sqrt(2*I2/(lambda_n^2*(1-a)*(1+xi*a))),   a = a(0+),

with I2, lambda_n and a independent of xi. Retuning xi therefore rescales the
saved grid exactly, and contours are redrawn from the rescaled grid.
"""
import numpy as np
from scipy.optimize import minimize_scalar

import _iso_common as common

PHASE_VELOCITY = common.ROOT / "RMFsolver/phase_velocity.py"


def build_cache(analytic_payload):
    records = analytic_payload["records"]
    mesh = analytic_payload["mesh"]
    accepted = np.vectorize(lambda r: bool(r and r.get("accepted")), otypes=[bool])(records)
    def grid(key):
        return np.vectorize(lambda r: float(r[key]) if r and r.get("accepted") else np.nan,
                            otypes=[float])(records)
    sources = analytic_payload["identity"]["sources"]
    return dict(I2=grid("I2"), lambda_n=grid("lambda_n"), a_0plus=grid("a_0plus"), accepted=accepted,
                x=mesh["x"], T=mesh["T"], rho=mesh.get("rho"), fractions=mesh.get("fractions"),
                provenance=dict(phase_velocity_sha256=sources["RMFsolver/phase_velocity.py"],
                                mesh_fingerprint=mesh["fingerprint"], xi_solved=common.XI))


def _check(cache, check_source):
    if check_source and cache["provenance"]["phase_velocity_sha256"] != common.sha256(PHASE_VELOCITY):
        raise RuntimeError("phase_velocity.py changed since this xi cache was built; rebuild it "
                           "or pass check_source=False knowingly")


def retune(cache, xi, check_source=True):
    """Analytic speed grid [m/s] at a new xi; NaN where the analytic point was not accepted."""
    _check(cache, check_source)
    a = cache["a_0plus"]
    with np.errstate(invalid="ignore"):
        return common.C_M_S * np.sqrt(2 * cache["I2"] / (cache["lambda_n"]**2 * (1 - a) * (1 + xi * a)))


def effective_xi(cache, numerical_payload):
    """xi at which the analytic speed equals the numerical one, point by point."""
    a, v = cache["a_0plus"], numerical_payload["speed_m_s"] / common.C_M_S
    with np.errstate(invalid="ignore", divide="ignore"):
        return (2 * cache["I2"] / (cache["lambda_n"]**2 * (1 - a) * v**2) - 1) / a


def _ray_crossings(speeds, rho, fractions, level):
    """Radius where each ray's monotone accepted speeds cross the level (NaN otherwise)."""
    out = np.full(speeds.shape[0], np.nan)
    for i, row in enumerate(speeds):
        ok = np.isfinite(row)
        if ok.sum() < 2 or not np.all(np.diff(row[ok]) > 0) or not row[ok][0] <= level <= row[ok][-1]:
            continue
        fraction = np.interp(np.log(level), np.log(row[ok]), fractions[ok])
        out[i] = np.interp(fraction, fractions, rho[i])
    return out


def contour_offsets(cache, numerical_payload, xi, levels, check_source=True):
    """Analytic minus numerical contour radius, per ray and level (plot-normalized rho)."""
    analytic = retune(cache, xi, check_source)
    return np.array([_ray_crossings(analytic, cache["rho"], cache["fractions"], L)
                     - _ray_crossings(numerical_payload["speed_m_s"], cache["rho"], cache["fractions"], L)
                     for L in levels])


def fit_xi(cache, numerical_payload, levels, objective="log_speed", check_source=True,
           bounds=(-0.99, 0.99)):
    """Single xi minimizing the RMS log-speed misfit or the RMS contour radial offset."""
    _check(cache, check_source)
    numerical = numerical_payload["speed_m_s"]
    def misfit(xi):
        if objective == "log_speed":
            residual = np.log(retune(cache, xi, False)) - np.log(numerical)
        elif objective == "contour":
            residual = contour_offsets(cache, numerical_payload, xi, levels, False)
        else:
            raise ValueError("objective must be 'log_speed' or 'contour'")
        return float(np.sqrt(np.nanmean(residual**2)))
    return float(minimize_scalar(misfit, bounds=bounds, method="bounded",
                                 options=dict(xatol=1e-8)).x)
