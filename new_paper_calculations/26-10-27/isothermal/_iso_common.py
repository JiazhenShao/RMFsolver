"""Shared constants, polar transform, checkpoint I/O and provenance for the
2026-10-27 isothermal contour workflow (high-T polar mesh, low-T log mesh)."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCIPY_RUNTIME = ROOT / "local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64"

N0_MEV3 = 1229360.5098323745          # 0.16 fm^-3
PHYSICS = dict(B_one_forth=189.1565957288247, ms=0.0, NM_type="PNM")
XI = -0.85
C_M_S = 299792458.0
A_CEILING = 1.0 - 1.0e-9               # udd boundary convention, as in the solvers
DENSITY_WINDOW = (1.0, 5.5)            # n_B(0-)/n_0
# Hidden polar coordinates of the 26-08-20 isothermal domain:
# X = (x - 1)/4.5 = rho cos(theta), Y = T/120 MeV = rho sin(theta).
POLAR = dict(x_origin=1.0, x_scale=4.5, T_scale=120.0)
NUMERICAL = dict(tail_eps=1e-8, n_mesh=300, tol_bvp=1e-4, max_nodes=50000,
                 kappa_factor=1.0, path_rtol=1e-6, initial_guess_mode="auto")
LEVELS_HIGH = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0)
LEVELS_LOW = LEVELS_HIGH + (1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6)


def to_physical(rho, theta):
    """Polar (rho, theta) -> (x = n_B(0-)/n_0, T(0-) in MeV)."""
    rho, theta = np.asarray(rho, dtype=float), np.asarray(theta, dtype=float)
    x = POLAR["x_origin"] + POLAR["x_scale"] * rho * np.cos(theta)
    x = np.where(theta == np.pi / 2, POLAR["x_origin"], x)   # exact n_0 edge
    return x, POLAR["T_scale"] * rho * np.sin(theta)


def to_polar(x, T):
    X = (np.asarray(x, dtype=float) - POLAR["x_origin"]) / POLAR["x_scale"]
    Y = np.asarray(T, dtype=float) / POLAR["T_scale"]
    return np.hypot(X, Y), np.arctan2(Y, X)


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, payload, allow_pickle=True)
    os.replace(temporary, path)


def load(path):
    return np.load(path, allow_pickle=True).item()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


RESULT_SOURCES = ("_iso_common.py", "_iso_boundaries.py", "_iso_evaluate.py")   # engine/plot/xi excluded


def source_hashes():
    """Hashes of every file that can change a computed point (solver + physics modules)."""
    files = [ROOT / "RMFsolver/phase_velocity.py"] + [HERE / name for name in RESULT_SOURCES]
    return {str(Path(f).relative_to(ROOT)): sha256(f) for f in files}


def identity(**extra):
    """Everything that must match for a checkpoint to be resumed."""
    return dict(sources=source_hashes(), physics=PHYSICS, xi=XI, numerical=NUMERICAL,
                a_ceiling=A_CEILING, polar=POLAR, **extra)


def _single_thread():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def map_parallel(function, items, workers):
    """Process-parallel map (RMF solves are GIL-bound, threads do not help)."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    _single_thread()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_single_thread) as pool:
        return list(pool.map(function, items))
