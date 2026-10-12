"""Targeted local analytical recovery; immutable cluster inputs, separate overlay.

Default: two workers, 90-second hard limit per solve. No numerical BVP calls.
Only failed samples adjacent to the displayed contours are selected.
"""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ADIABATIC = HERE / 'adiabatic_xi_minus0p85/T0plus0p1-analytical-xi-minus0p85.npy'
ISOTHERMAL = HERE / 'isothermal_analytic_logT/isothermal-analytic-logT.npy'
OUTPUT = ROOT / 'local_analysis_results/26-10-02/analytical-local-repair.npy'
C = 299792458.
LEVELS = (50., 100., 200., 500., 1000., 2000.)


def merge_adiabatic(baseline, overlay):
    merged = copy.deepcopy(baseline)
    for key, methods in overlay.get('adiabatic', {}).items():
        row = merged['records'].setdefault(key, {})
        for method, value in methods.items():
            if value.get('success') and not row.get(method, {}).get('success'):
                row[method] = copy.deepcopy(value)
    for grid in merged.values():
        if not isinstance(grid, dict) or 'point_keys' not in grid:
            continue
        for ij in np.ndindex(grid['point_keys'].shape):
            key = str(grid['point_keys'][ij])
            for method in ('analytic', 'semi_analytic'):
                row = merged['records'].get(key, {}).get(method, {})
                if not grid[method + '_success'][ij] and row.get('success'):
                    grid[method + '_success'][ij] = True
                    grid[method + '_u_0minus_m_s'][ij] = C * row['u_0minus']
    return merged


def merge_isothermal(baseline, overlay):
    merged = copy.deepcopy(baseline)
    for key, row in overlay.get('isothermal', {}).items():
        if row.get('status') == 'success' and merged['targets'].get(key, {}).get('status') != 'success':
            merged['targets'][key] = copy.deepcopy(row)
    return merged


def propose_density(samples, level, limits):
    rows = sorted(samples, key=lambda row: row['nB_0minus_over_n0'])
    brackets = [(a, b) for a, b in zip(rows[:-1], rows[1:])
                if (a['velocity_m_s'] - level) * (b['velocity_m_s'] - level) < 0]
    if brackets:
        a, b = min(brackets, key=lambda pair: pair[1]['nB_0minus_over_n0'] - pair[0]['nB_0minus_over_n0'])
        left, right = a['nB_0minus_over_n0'], b['nB_0minus_over_n0']
        fraction = (level - a['velocity_m_s']) / (b['velocity_m_s'] - a['velocity_m_s'])
        return float(left + np.clip(fraction, .05, .95) * (right - left))
    lo, hi = limits
    # At a clipped edge the distance to the window boundary is NOT the distance
    # to coexistence. Move into the saved allowed interval instead of multiplying
    # a roundoff-sized offset from nB/n0=1 thousands of times.
    below = [row for row in rows if row['velocity_m_s'] < level]
    above = [row for row in rows if row['velocity_m_s'] > level]
    if below and not above:
        return .5 * (max(row['nB_0minus_over_n0'] for row in below) + hi)
    if above and not below:
        return .5 * (lo + min(row['nB_0minus_over_n0'] for row in above))
    return .5 * (lo + hi)


def adiabatic_tasks(payload):
    grid = payload['plot_grid']
    selected = set()
    for method in ('analytic', 'semi_analytic'):
        valid = grid[method + '_success']
        velocities = grid[method + '_u_0minus_m_s']
        estimate = np.log(velocities.copy())
        # Predictions select missing quad corners only; they never become
        # plotted values. Angular neighbors can cover an entirely timed-out ray.
        for j in range(valid.shape[1]):
            good = valid[:, j] & np.isfinite(velocities[:, j]) & (velocities[:, j] > 0)
            if np.any(good):
                estimate[:, j] = np.interp(np.arange(valid.shape[0]), np.flatnonzero(good), estimate[good, j])
        for i in range(valid.shape[0]-1):
            for j in range(valid.shape[1]-1):
                block = estimate[i:i+2, j:j+2]
                if any(np.min(block) <= np.log(level) <= np.max(block) for level in LEVELS):
                    for a, b in ((i,j), (i+1,j), (i,j+1), (i+1,j+1)):
                        if not valid[a,b]:
                            selected.add((str(grid['point_keys'][a,b]), method))
        # In particular, recover the single failed full-analytical cold corner.
        if method == 'analytic':
            selected.update((key, method) for key, row in payload['records'].items() if not row[method]['success'])
    return sorted(selected, key=lambda item: (item[1] != 'analytic', tuple(map(float, item[0].split(':')))))


def evaluate_adiabatic(point, calibration, method):
    from RMFsolver import phase_velocity as pv
    import warnings
    started = time.monotonic()
    density = point['nB_0minus_over_n0'] * calibration['n0_MeV3']
    temperature = point['T_0minus_MeV']
    fallback = None
    with warnings.catch_warnings(record=True) as caught:
        try:
            upstream = pv._validated_pnm_state_from_nB(density, temperature)
            muB = upstream['muB_0minus']
            density_error = abs(upstream['upstream_density_relative_residual'])
        except RuntimeError as exc:
            if temperature != 0. or not np.isclose(point['nB_0minus_over_n0'], 4.6545676259103885, rtol=0, atol=1.e-12):
                raise
            # Diagnosed single cold state: two nearby direct-density seeds
            # independently converge. Keep the same residual/density checks;
            # do not alter solver globals or relax branch acceptance.
            solutions, errors = [], []
            for seed in ((80., 100., -3., 1500.), (81., 105., -5., 1506.)):
                solution = pv.RMFsolvePNM(density, temperature, pv.para.paraQMCRMF3,
                    sigma_init=seed[0], w0_init=seed[1], r03_init=seed[2], mub_init=seed[3], verb=False)
                state = np.asarray(solution[0][:4], float)
                residual = np.max(np.abs(solution[1][:4])) / density
                error = abs(float(np.asarray(pv.baryon_density_RMF(solution)).item()) / density - 1.)
                pressure = float(np.asarray(pv.pressure_RMF(solution, add_photons=False,
                    renorm=False, electrons=False, neutrinos=False)).item())
                if not np.all(np.isfinite(state)) or state[0] <= 0 or state[3] <= 0 or residual > 1.e-6 or error > 2.e-6 or pressure <= 0:
                    raise RuntimeError('Cold direct-density seed failed validation')
                solutions.append(state)
                errors.append(dict(scaled_residual=float(residual), density_error=error))
            if not np.allclose(np.array(solutions[0])[[0, 3]], np.array(solutions[1])[[0, 3]], rtol=1.e-6, atol=0):
                raise RuntimeError('Cold branch seeds disagree')
            muB = float(solutions[0][3])
            density_error = max(row['density_error'] for row in errors)
            fallback = dict(original_error=str(exc), states=solutions, validation=errors)
        function = pv.analytic_velocity_bound if method == 'analytic' else pv.semi_analytic_velocity_bound
        result = function(muB, temperature, calibration['B14_MeV'], T_0plus=.1, xi=-.85,
                          ms=0., NM_type='PNM', upB=5000)
    u = float(result['u_0minus'])
    density_error = max(density_error, abs(result['nB_0minus'] / density - 1.))
    closure_error = abs(result['analytic_velocity_residual']) / max(u*u, 1.e-30)
    if (not result['success'] or not np.isfinite(u) or u <= 0 or density_error > 2.e-6
            or not 0 < result['a_0plus'] < 1 or closure_error > 1.e-3
            or result['endpoint_scaled_residual'] > 1.e-6):
        raise RuntimeError('Recovered adiabatic sample failed independent checks')
    # Drop the public legacy alias from this new saved artifact.
    result.pop('muB_bar', None)
    return dict(result, success=True, validated=True, source='local_targeted_recovery',
                runtime_s=time.monotonic()-started, density_relative_error=density_error,
                closure_relative_error=closure_error, upstream_fallback=fallback,
                warning_count=len(caught))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=90.)
    parser.add_argument('--stage', choices=('all', 'adiabatic', 'isothermal'), default='all')
    parser.add_argument('--limit', type=int, default=80)
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path[:0] = [str(HERE), str(HERE / 'adiabatic_xi_minus0p85'), str(HERE / 'isothermal_analytic_logT'), str(ROOT)]
    from _bounded_call import run_module_call_with_hard_timeout
    from _logT_common import atomic_save
    from tqdm.auto import tqdm
    baseline = np.load(ADIABATIC, allow_pickle=True).item()
    iso = np.load(ISOTHERMAL, allow_pickle=True).item()
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (ADIABATIC, ISOTHERMAL)}
    payload = np.load(OUTPUT, allow_pickle=True).item() if OUTPUT.exists() else dict(
        schema=1, source_hashes=hashes, xi=-.85, adiabatic={}, isothermal={}, evaluations=[],
        solver_sha256=hashlib.sha256((ROOT/'RMFsolver/phase_velocity.py').read_bytes()).hexdigest())
    if payload['source_hashes'] != hashes:
        raise ValueError('Baseline files changed; do not merge incompatible repairs')
    lock = threading.RLock()

    def save():
        payload['updated_unix_s'] = time.time()
        atomic_save(OUTPUT, payload)

    def call(module, function, *positional, **kwargs):
        try:
            return run_module_call_with_hard_timeout(module, function, args=positional, kwargs=kwargs, timeout_s=args.timeout)
        except Exception as exc:
            return dict(success=False, message=f'{type(exc).__name__}: {exc}')

    if args.stage in ('all', 'isothermal'):
        todo = [(key, row) for key, row in iso['targets'].items() if row['status']=='unresolved'
                and payload['isothermal'].get(key, {}).get('status') != 'success']
        def repair_target(item):
            key, row = item
            T, level = row['T_0minus'], row['level']
            samples = [p for p in iso['points'].values() if p['T_0minus']==T and p['result'].get('validated')]
            with lock:
                samples += [e['root'] for e in payload['evaluations'] if e.get('target_id') == key and e.get('root')]
            boundary = min(iso['boundaries'].values(), key=lambda b: abs(b['T_0minus']-T))
            limits = (boundary['lower'], boundary['upper'])
            for attempt in range(10):
                found = next((p for p in samples if abs(p['velocity_m_s']/level-1.) <= .005), None)
                if found:
                    with lock:
                        payload['isothermal'][key] = dict(row, status='success', root=found,
                            message='Recovered by validated local analytical evaluation', source='local_targeted_recovery')
                        save()
                    return key, 'success'
                density = propose_density(samples, level, limits)
                point = dict(T_0minus=T, nB_0minus_over_n0=density,
                             nB_0minus=density*baseline['calibration']['n0_MeV3'])
                result = call('_logT_trial', 'analytical', point)
                root = dict(point, velocity_m_s=result['velocity_m_s'], result=result) if result.get('validated') else None
                with lock:
                    payload['evaluations'].append(dict(target_id=key, point=point, result=result, root=root))
                    save()
                if root:
                    samples.append(root)
                else:
                    return key, 'unresolved'
            return key, 'unresolved'
        with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(total=len(todo), desc='Isothermal missing targets', unit='target') as progress:
            for future in as_completed([pool.submit(repair_target, item) for item in todo]):
                print(future.result(), flush=True)
                progress.update()
    if args.stage in ('all', 'adiabatic'):
        merged = merge_adiabatic(baseline, payload)
        todo = adiabatic_tasks(merged)[:args.limit]
        def repair_sample(item):
            key, method = item
            result = call('repair_analytical_local', 'evaluate_adiabatic', baseline['points'][key], baseline['calibration'], method)
            with lock:
                payload['adiabatic'].setdefault(key, {})[method] = result
                payload['evaluations'].append(dict(point_key=key, method=method, success=result.get('success',False), runtime_s=result.get('runtime_s')))
                save()
            return key, method, result.get('success',False), result.get('runtime_s'), result.get('message','')[:200]
        with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(total=len(todo), desc='Adiabatic contour samples', unit='point') as progress:
            for future in as_completed([pool.submit(repair_sample, item) for item in todo]):
                print(future.result(), flush=True)
                progress.update()
    print('Saved', OUTPUT)


if __name__ == '__main__':
    main()
