"""Pure grid/search rules and portable checkpoint I/O (no EOS imports)."""
from pathlib import Path
import hashlib
import os
import tempfile

import numpy as np

LEVELS = (50., 100., 200., 500., 1000., 2000., 5000.)
TEMPERATURES = np.geomspace(.01, 20., 31)
C = 299792458.
PHYSICS = dict(B_one_forth=189.1565957288247, ms=0., NM_type="PNM", upB=5000, xi=-.5)
CONTROLS = dict(trial_timeout=180., target_timeout=600., max_attempts=900,
                attempts_per_target=4, attempts_per_point=2, speed_rtol=.005,
                n_mesh=300, tol_bvp=1.e-4, tail_eps=1.e-3, max_nodes=50000,
                kappa_factor=1., initial_guess_mode="layer_resolved")
CHECK_CONTROLS = dict(n_mesh=450, tol_bvp=1.e-5, tail_eps=1.e-4)


def load(path):
    return np.load(path, allow_pickle=True).item()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, payload, allow_pickle=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def temperature_id(T):
    return float(T).hex()


def target(level, T, kind="base"):
    return dict(id=f"{kind}:{float(level):g}:{temperature_id(T)}", level=float(level),
                T_0minus=float(T), kind=kind, status="pending", root=None)


def base_targets(smoke=False):
    if smoke:
        return [target(50, .01), target(5000, .01), target(500, TEMPERATURES[15]), target(1000, 20.)]
    return [target(v, T) for v in LEVELS for T in TEMPERATURES]


def anchor_ids():
    return {target(v, T)["id"] for v in LEVELS for T in TEMPERATURES[::5]}


def block_chains():
    return [[target(v, TEMPERATURES[i])["id"] for i in range(start+4, start, -1)]
            for v in LEVELS for start in range(0, 30, 5)]


def point(T, density, n0):
    if not (.01 <= T <= 20. and 1. <= density <= 5.5):
        raise ValueError("Numerical point outside the cold physical window")
    return dict(id=f"{temperature_id(T)}:{float(density).hex()}", T_0minus=float(T),
                nB_0minus_over_n0=float(density), nB_0minus=float(density*n0))


def current_bounds(estimates):
    finite = [float(x) for x in estimates if x is not None and np.isfinite(x) and x > 0]
    if not finite:
        raise ValueError("No positive current estimate")
    return min(finite)/4., max(finite)*4.


def near_bound(value, bounds):
    return not bounds[0] < value < bounds[1] or min(
        np.log(value/bounds[0]), np.log(bounds[1]/value)) < .05*np.log(bounds[1]/bounds[0])


def propose_density(samples, failures, predicted, level, limits):
    """Only numerical velocities determine the secant; no fixed density steps."""
    lo, hi = limits
    margin = max(1.e-10, (hi-lo)*1.e-10)
    points = sorted({p["nB_0minus_over_n0"]: p for p in samples}.values(),
                    key=lambda p: p["nB_0minus_over_n0"])
    bracket = next(((a,b) for a,b in zip(points[:-1], points[1:])
                    if (a["velocity_m_s"]-level)*(b["velocity_m_s"]-level) < 0), None)
    if bracket or len(points) >= 2:
        a,b = bracket or sorted(points, key=lambda p: abs(p["velocity_m_s"]-level))[:2]
        dv = b["velocity_m_s"]-a["velocity_m_s"]
        guess = (a["nB_0minus_over_n0"] + (level-a["velocity_m_s"])/dv *
                 (b["nB_0minus_over_n0"]-a["nB_0minus_over_n0"])) if abs(dv) > 1.e-12 else predicted
        if bracket:
            left,right = sorted([a["nB_0minus_over_n0"], b["nB_0minus_over_n0"]])
            guess = np.clip(guess, left+.05*(right-left), right-.05*(right-left))
    elif points:
        last = points[0]
        distance = max(last["nB_0minus_over_n0"]-lo, margin)
        guess = lo + distance*np.clip(level/last["velocity_m_s"], .5, 2.)
    else:
        guess = predicted
    guess = float(np.clip(guess, lo+margin, hi-margin))
    # After failure, approach from a smaller positive distance to coexistence.
    # Never silently re-evaluate a failed density with an identical seed.
    if any(abs(guess-x) < 1.e-10 for x in failures):
        guess = lo + .5*(guess-lo)
    tried = failures + [p["nB_0minus_over_n0"] for p in points]
    return None if any(abs(guess-x) < 1.e-10 for x in tried) else guess


def workers(requested=None):
    if requested is not None:
        if requested < 1:
            raise ValueError("workers must be positive")
        return requested
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        try:
            if int(os.environ.get(name, "0")) > 0:
                return int(os.environ[name])
        except ValueError:
            pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
