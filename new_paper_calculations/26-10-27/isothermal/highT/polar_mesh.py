"""High-temperature polar mesh: rays between the two static boundaries.

Each ray theta runs from the polar origin (n_B = n_0, T = 0). Points sit at
radial fractions f strictly between the coexistence radius (f = 0) and the udd
radius (f = 1), clustered at both ends: the inner shell resolves the slow
contours, the outer columns follow v ~ (1 - A)^(-1/2) toward the udd boundary.
"""
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _iso_common as common  # noqa: E402
import _iso_boundaries as boundaries  # noqa: E402

FRACTIONS = (0.005, 0.02, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
             0.55, 0.60, 0.66, 0.72, 0.78, 0.84, 0.90, 0.95, 0.99, 0.995, 0.998, 0.999,
             0.9995, 0.9998, 0.9999, 0.99995, 0.99999)
CACHE = Path(__file__).with_name("ray-boundaries.npy")


def _boundary_key():
    """Boundary radii depend only on RMFsolver, the boundary code, the libraries and these constants."""
    sources = {k: v for k, v in common.source_hashes().items()
               if k.startswith("RMFsolver/") or k.endswith("_iso_boundaries.py")}
    return repr((sorted(sources.items()), common.library_versions(), common.PHYSICS, common.A_CEILING,
                 common.POLAR, common.N0_MEV3))


def ray_boundaries_cached(theta, workers):
    """Solve (or reuse) both boundary radii per ray angle; only new angles are solved."""
    key = _boundary_key()
    cache = common.load(CACHE) if CACHE.is_file() else {}
    rays = cache.get(key, {})
    missing = [float(t) for t in theta if round(float(t), 14) not in rays]
    if missing:
        for t, ray in zip(missing, common.map_parallel(boundaries.ray_boundaries, missing, workers)):
            rays[round(t, 14)] = ray
        common.atomic_save(CACHE, {key: rays})   # entries for other sources are stale
    return [rays[round(float(t), 14)] for t in theta]


def lowest_angle(T_min):
    """Angle of the ray whose udd-boundary point sits at T = T_min."""
    window = boundaries.fixed_T_window(T_min, x_window=(1.0, 6.0))
    return float(common.to_polar(window["outer_root"], T_min)[1])


def build_mesh(n_rays=81, fractions=FRACTIONS, T_min=8.0, T_max=100.0, workers=4, theta=None):
    if theta is None:
        theta = np.linspace(lowest_angle(T_min), np.pi / 2, n_rays)
        theta[-1] = np.pi / 2
    theta = np.asarray(theta, dtype=float)
    rays = ray_boundaries_cached(theta, workers)
    fractions = np.asarray(fractions, dtype=float)
    inner = np.array([r["rho_inner"] for r in rays])[:, None]
    outer = np.array([r["rho_outer"] for r in rays])[:, None]
    rho = inner + fractions[None, :] * (outer - inner)
    x, T = common.to_physical(rho, np.broadcast_to(theta[:, None], rho.shape))
    status = np.where((T >= T_min) & (T <= T_max), "pending", "outside_panel").astype(object)
    fingerprint = hashlib.sha256(np.concatenate([theta, fractions, inner[:, 0], outer[:, 0]])
                                 .tobytes()).hexdigest()
    return dict(theta=theta, fractions=fractions, rho=rho, x=np.asarray(x), T=np.asarray(T),
                point_status=status, boundaries=rays, T_min=T_min, T_max=T_max,
                fingerprint=fingerprint)
