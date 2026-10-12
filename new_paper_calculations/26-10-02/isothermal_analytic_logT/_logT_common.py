"""Analytical axes, density targeting and atomic checkpoint I/O."""
from pathlib import Path
import hashlib
import os
import tempfile
import numpy as np

LEVELS = (50., 100., 200., 500., 1000., 2000., 5000.)
TEMPERATURES = np.r_[np.geomspace(.01, 20., 31), np.geomspace(20., 120., 16)[1:]]
C = 299792458.
PHYSICS = dict(B_one_forth=189.1565957288247, ms=0., NM_type='PNM', upB=5000, xi=-.85)
CONTROLS = dict(trial_timeout=180., target_timeout=600., evaluations_per_target=24,
                speed_rtol=.005, boundary_timeout=180.)


def load(path):
    return np.load(path, allow_pickle=True).item()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.save(stream, payload, allow_pickle=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def temperature_id(T):
    return float(T).hex()


def target(level, T):
    return dict(id=f'base:{float(level):g}:{temperature_id(T)}', level=float(level),
                T_0minus=float(T), kind='base', status='pending', root=None)


def base_targets(smoke=False):
    if smoke:
        return [target(50., .01), target(5000., .01), target(500., TEMPERATURES[15]), target(1000., 80.)]
    return [target(v, T) for T in TEMPERATURES for v in LEVELS]


def point(T, density, n0):
    if not (.01 <= T <= 120. and 1. <= density <= 5.5):
        raise ValueError('Point outside the physical window')
    return dict(id=f'{temperature_id(T)}:{float(density).hex()}', T_0minus=float(T),
                nB_0minus_over_n0=float(density), nB_0minus=float(density*n0))


def next_density(samples, failures, predicted, level, limits):
    """Safeguarded secant using only this run's analytical values.

    An exact coexistence zero is a bracket anchor, not a positive-speed root.
    Failed evaluations never acquire fabricated velocities.
    """
    lo, hi = limits
    margin = max(1.e-11, (hi-lo)*1.e-11)
    rows = sorted({p['nB_0minus_over_n0']:p for p in samples}.values(),
                  key=lambda p:p['nB_0minus_over_n0'])
    bracket = next(((a,b) for a,b in zip(rows[:-1],rows[1:])
                    if (a['velocity_m_s']-level)*(b['velocity_m_s']-level)<0), None)
    moving = [p for p in rows if p['velocity_m_s']>0]
    if bracket:
        a,b = bracket
        width = b['nB_0minus_over_n0']-a['nB_0minus_over_n0']
        guess = a['nB_0minus_over_n0'] + (level-a['velocity_m_s'])*width/(b['velocity_m_s']-a['velocity_m_s'])
        guess = np.clip(guess, a['nB_0minus_over_n0']+.02*width, b['nB_0minus_over_n0']-.02*width)
    elif moving:
        p = min(moving, key=lambda p:abs(np.log(p['velocity_m_s']/level)))
        distance = max(p['nB_0minus_over_n0']-lo, margin)
        guess = lo + distance*np.clip(level/p['velocity_m_s'], .25, 4.)
        if guess >= hi-margin:
            guess = (p['nB_0minus_over_n0']+hi)/2.
    else:
        guess = predicted
    guess = float(np.clip(guess, lo+margin, hi-margin))
    tried = failures + [p['nB_0minus_over_n0'] for p in rows]
    # Repeated halving fixes the copied workflow's two-failure dead end.
    for _ in range(50):
        if not any(abs(guess-d)<1.e-12 for d in tried):
            return guess
        guess = .5*(guess+(bracket[0]['nB_0minus_over_n0'] if bracket else lo))
        if guess <= lo+margin:
            break
    return None


def workers(requested=None):
    if requested is not None:
        if requested<1:
            raise ValueError('workers must be positive')
        return requested
    for name in ('SLURM_CPUS_PER_TASK','PBS_NP','NSLOTS'):
        try:
            if int(os.environ.get(name,'0'))>0:
                return int(os.environ[name])
        except ValueError:
            pass
    return len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
