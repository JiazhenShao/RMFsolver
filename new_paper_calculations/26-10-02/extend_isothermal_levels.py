"""Bounded local extension of saved analytical contours; no BVP or boundary scan.

Run from the repository with its Python environment. Defaults to two workers.
Original cluster and recovery files are read-only; every evaluation is saved to
local_analysis_results/26-10-02/isothermal-high-speed-extension.npy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = HERE / 'isothermal_analytic_logT/isothermal-analytic-logT.npy'
REPAIR = ROOT / 'local_analysis_results/26-10-02/analytical-local-repair.npy'
OUTPUT = ROOT / 'local_analysis_results/26-10-02/isothermal-high-speed-extension.npy'
LEVELS = (10000., 20000., 50000., 100000.)


def accepted_root(samples, level):
    return next((r for r in samples if r.get('result', {}).get('validated')
                 and r['result'].get('success') and np.isfinite(r['velocity_m_s'])
                 and abs(r['velocity_m_s'] / level - 1.) <= .005), None)


def next_density(samples, level, limits):
    """Log-speed secant prediction inside actual evaluated brackets, else bisect."""
    rows = sorted((r for r in samples if np.isfinite(r['velocity_m_s']) and r['velocity_m_s'] > 0),
                  key=lambda r: r['nB_0minus_over_n0'])
    brackets = [(a, b) for a, b in zip(rows[:-1], rows[1:])
                if (a['velocity_m_s']-level)*(b['velocity_m_s']-level) < 0]
    if brackets:
        a, b = min(brackets, key=lambda pair: pair[1]['nB_0minus_over_n0']-pair[0]['nB_0minus_over_n0'])
        fraction = np.log(level/a['velocity_m_s']) / np.log(b['velocity_m_s']/a['velocity_m_s'])
        return float(a['nB_0minus_over_n0'] + np.clip(fraction, .05, .95)
                     * (b['nB_0minus_over_n0']-a['nB_0minus_over_n0']))
    below = [r['nB_0minus_over_n0'] for r in rows if r['velocity_m_s'] < level]
    above = [r['nB_0minus_over_n0'] for r in rows if r['velocity_m_s'] > level]
    return .5 * ((max(below) if below else limits[0]) + (min(above) if above else limits[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-temperatures', type=int, default=None, help='Bounded initial check; resume the rest later')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path[:0] = [str(ROOT), str(HERE / 'isothermal_analytic_logT')]
    from _logT_common import atomic_save
    from _logT_trial import bounded_call
    from tqdm.auto import tqdm
    # Fail before reserving any physical point if the local binary runtime is broken.
    from RMFsolver import phase_velocity  # noqa: F401

    baseline = np.load(BASELINE, allow_pickle=True).item()
    repair = np.load(REPAIR, allow_pickle=True).item()
    inputs = [BASELINE, REPAIR, HERE / 'isothermal_analytic_logT/inputs/static-boundaries.npy']
    sources = [Path(__file__), ROOT / 'RMFsolver/phase_velocity.py',
               HERE / 'isothermal_analytic_logT/_logT_trial.py']
    hashes = lambda paths: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    identity = dict(input_hashes=hashes(inputs), source_hashes=hashes(sources),
                    levels=list(LEVELS), temperatures=list(map(float, baseline['temperatures'])),
                    physics=baseline['identity']['physics'],
                    controls=dict(trial_timeout=90., max_evaluations=24, speed_rtol=.005))
    if OUTPUT.exists():
        payload = np.load(OUTPUT, allow_pickle=True).item()
        if payload['identity'] != identity:
            raise ValueError('Extension sources/inputs/controls changed; preserve the existing checkpoint')
    else:
        payload = dict(schema=1, method='analytic_isothermal', identity=identity,
                       temperatures=baseline['temperatures'], levels=list(LEVELS),
                       domain_fingerprint=baseline['domain_fingerprint'], physics=identity['physics'],
                       targets={}, evaluations=[])
        for temperature in payload['temperatures']:
            for level in LEVELS:
                key = f'base:{level:g}:{float(temperature).hex()}'
                payload['targets'][key] = dict(id=key, kind='base', T_0minus=float(temperature),
                                               level=level, status='pending', attempts=0)
    n0 = np.load(inputs[2], allow_pickle=True).item()['n0']
    lock = threading.RLock()

    def save():
        payload['updated_unix_s'] = time.time()
        atomic_save(OUTPUT, payload)

    save()
    total = len(payload['targets'])
    initial = sum(r['status'] != 'pending' for r in payload['targets'].values())
    with tqdm(total=total, initial=initial, desc='Higher-speed analytical contours', unit='target', ncols=100) as progress:
        def temperature_chain(T):
            boundary = min(baseline['boundaries'].values(), key=lambda b: abs(b['T_0minus']-T))
            if not np.isclose(boundary['T_0minus'], T, rtol=1.e-12, atol=0.):
                raise ValueError('Missing directly solved boundary temperature')
            samples = [r for r in baseline['points'].values() if r['T_0minus']==T and r['result'].get('validated')]
            samples += [r['root'] for r in repair['isothermal'].values() if r['T_0minus']==T and r['status']=='success']
            with lock:
                samples += [e['root'] for e in payload['evaluations'] if e['T_0minus']==T and e.get('root')]
            for level in LEVELS:
                key = f'base:{level:g}:{float(T).hex()}'
                row = payload['targets'][key]
                if row['status'] != 'pending':
                    continue
                if boundary['status'] == 'outside_window':
                    row.update(status='outside_window', message=boundary.get('reason', 'No allowed density interval'))
                else:
                    limits = boundary['lower'], boundary['upper']
                    root = accepted_root(samples, level)
                    while root is None and row['attempts'] < identity['controls']['max_evaluations']:
                        density = next_density(samples, level, limits)
                        if not limits[0] < density < limits[1] or any(abs(density-r['nB_0minus_over_n0']) < 1.e-13 for r in samples):
                            row['message'] = 'No new interior density proposal'
                            break
                        point = dict(T_0minus=T, nB_0minus_over_n0=density, nB_0minus=density*n0)
                        evaluation = dict(target_id=key, T_0minus=T, point=point, status='reserved')
                        with lock:
                            row['attempts'] += 1
                            payload['evaluations'].append(evaluation)
                            save()
                        result = bounded_call('analytical', timeout=90., point=point)
                        new_root = dict(point, velocity_m_s=result['velocity_m_s'], result=result) if result.get('validated') else None
                        with lock:
                            evaluation.update(status='success' if new_root else 'failed', result=result, root=new_root)
                            save()
                        if new_root is None:
                            row['message'] = result.get('message', result.get('status', 'Invalid analytical evaluation'))
                            break
                        samples.append(new_root)
                        root = accepted_root(samples, level)
                    row.update(status='success' if root else 'unresolved', root=root,
                               source='local_high_speed_analytical_extension')
                with lock:
                    save()
                    progress.update()
        temperatures = list(map(float, payload['temperatures']))[:args.max_temperatures]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(temperature_chain, T) for T in temperatures]):
                future.result()
    from collections import Counter
    print(dict(Counter(row['status'] for row in payload['targets'].values())))
    print('Saved', OUTPUT)


if __name__ == '__main__':
    main()
