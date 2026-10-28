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


def _same_mesh(cache, numerical_payload):
    """Refuse numerical data from another mesh when both sides carry a fingerprint."""
    ours = cache["provenance"].get("mesh_fingerprint")
    theirs = numerical_payload.get("mesh", {}).get("fingerprint")
    if ours is not None and theirs is not None and ours != theirs:
        raise ValueError("the xi cache and the numerical payload were made on different meshes")


def effective_xi(cache, numerical_payload):
    """xi at which the analytic speed equals the numerical one, point by point."""
    _same_mesh(cache, numerical_payload)
    a, v = cache["a_0plus"], numerical_payload["speed_m_s"] / common.C_M_S
    with np.errstate(invalid="ignore", divide="ignore"):
        return (2 * cache["I2"] / (cache["lambda_n"]**2 * (1 - a) * v**2) - 1) / a


def _ray_crossings(speeds, coordinate, fractions, level):
    """Coordinate where each ray or row crosses the level between two adjacent accepted
    points (log speed linear in f); NaN if no such pair, or several, bracket it, so a
    crossing is never interpolated across a failed point."""
    out = np.full(speeds.shape[0], np.nan)
    for i, row in enumerate(speeds):
        pairs = [j for j in range(row.size - 1) if np.isfinite(row[j]) and np.isfinite(row[j + 1])
                 and row[j] > 0 and row[j + 1] > 0 and min(row[j], row[j + 1]) <= level <= max(row[j], row[j + 1])]
        if len(pairs) != 1 or row[pairs[0]] == row[pairs[0] + 1]:
            continue
        j = pairs[0]
        weight = (np.log(level) - np.log(row[j])) / (np.log(row[j + 1]) - np.log(row[j]))
        out[i] = np.interp(fractions[j] + weight * (fractions[j + 1] - fractions[j]), fractions, coordinate[i])
    return out


def _coordinate(cache):
    """Plot-normalized position along each ray (polar rho) or row ((x - 1)/4.5 at fixed T)."""
    if cache.get("rho") is not None:
        return np.asarray(cache["rho"], dtype=float)
    return (np.asarray(cache["x"], dtype=float) - common.POLAR["x_origin"]) / common.POLAR["x_scale"]


def contour_offsets(cache, numerical_payload, xi, levels, check_source=True):
    """Analytic minus numerical contour position, per ray/row and level (plot-normalized)."""
    _same_mesh(cache, numerical_payload)
    analytic, coordinate = retune(cache, xi, check_source), _coordinate(cache)
    return np.array([_ray_crossings(analytic, coordinate, cache["fractions"], L)
                     - _ray_crossings(numerical_payload["speed_m_s"], coordinate, cache["fractions"], L)
                     for L in levels])


def fit_xi(cache, numerical_payload, levels, objective="log_speed", check_source=True,
           bounds=(-0.99, 0.99)):
    """Single xi minimizing the RMS log-speed misfit or the RMS contour radial offset."""
    _check(cache, check_source)
    _same_mesh(cache, numerical_payload)
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
