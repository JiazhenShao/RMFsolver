"""Static downstream mean-free-path / thermal-tail comparison; never runs a BVP.

Natural units: D_K and lengths are MeV^-1, gamma_K is MeV. The EOS cache
contains both diffusion choices and is independent of the requested fraction.
Importing this module imports neither SciPy nor RMFsolver.
"""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'static-tail-mfp-v1'
MEV_INVERSE_TO_M = 197.326960e-15


class SearchLimit(Exception):
    """A root-search budget, not an EOS failure or a missing dictionary key."""


def lengths(muB_inf, T, D_K, gamma_K):
    if not all(math.isfinite(v) and v > 0 for v in (muB_inf, T, D_K, gamma_K)):
        raise ValueError('Positive finite muB_inf, T, D_K, gamma_K required')
    mfp = 3*D_K
    tail = muB_inf/(9*math.pi*T)*math.sqrt(D_K/gamma_K)
    return dict(D_K=D_K, gamma_K=gamma_K, lambda_mfp_MeV_inv=mfp,
                L_conv_MeV_inv=tail, lambda_mfp_m=mfp*MEV_INVERSE_TO_M,
                L_conv_m=tail*MEV_INVERSE_TO_M, ratio=mfp/tail)


def transport_lengths(muB_inf, T, diffusion='full'):
    """Manuscript coefficients, matching phase_velocity.py, not rounded length."""
    if diffusion not in ('full', 'landau'):
        raise ValueError('diffusion must be full or landau')
    if not all(math.isfinite(v) and v > 0 for v in (muB_inf, T)):
        raise ValueError('Positive finite temperature and muB_inf required')
    alpha_s, h_D, GF = .3, 1.81317, 1.16637e-11
    theta_C = math.radians(13.02)
    qD = math.sqrt(3*(4*math.pi*alpha_s)*muB_inf**2/(18*math.pi**2))
    inverse_D = 24*alpha_s**2/math.pi * (
        h_D*T**(5/3)/qD**(2/3) + (math.pi**3*T*T/(12*qD) if diffusion == 'full' else 0))
    gamma = 128/(3**8*5*math.pi**3)*GF**2*math.sin(theta_C)**2*math.cos(theta_C)**2*muB_inf**5
    return lengths(muB_inf, T, 1/inverse_D, gamma)


def atomic_save(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.save(stream, payload, allow_pickle=True)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _load(path):
    return np.load(path, allow_pickle=True).item()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def identity(inputs):
    files = sorted((ROOT/'RMFsolver').rglob('*.py')) + [Path(__file__), Path(__file__).with_name('_mfp_worker.py')]
    return dict(version=VERSION, domain_fingerprint=inputs['fingerprint'],
                physics=inputs['physics'], n0=float(inputs['n0']), input_hashes=inputs['source_hashes'],
                boundaries=[(k, r['status'], r.get('lower'), r.get('upper'))
                            for k,r in sorted(inputs['boundary_cache'].items())],
                source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def _bounded_evaluator(inputs, timeout, import_paths):
    helper = ROOT/'new_paper_calculations/26-10-02'
    for path in (helper, Path(__file__).parent):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    import notebook_isothermal
    paths = notebook_isothermal.worker_import_paths(import_paths)
    def call(point=None):
        return notebook_isothermal.run_module_call_with_hard_timeout(
            '_mfp_worker', 'preflight' if point is None else 'evaluate',
            kwargs=dict(import_paths=paths) if point is None else dict(point=point, physics=inputs['physics'], import_paths=paths),
            timeout_s=timeout)
    return call


def calculate(inputs, directory, *, fraction=1., diffusion='full', temperatures=None,
              density_samples=2, ratio_rtol=1.e-6, max_evaluations=40,
              trial_timeout=60., temperature_timeout=180., import_paths=None,
              evaluator=None, progress=None, show_progress=True):
    """Bounded density roots at saved temperatures; single local worker.

    Terminal failures are retained on resume, not repeatedly retried. Use a new
    output directory for an explicit retry. Source/physics/domain changes reject
    an incompatible cache. Fraction/model/control changes get separate results.
    `progress(kind, record)` is called only after atomic checkpointing.
    """
    from tqdm.auto import tqdm
    temperatures = tuple(sorted(float(r['T_0minus']) for r in inputs['boundary_cache'].values())) if temperatures is None else tuple(map(float, temperatures))
    if (not math.isfinite(fraction) or fraction <= 0 or diffusion not in ('full','landau')
        or density_samples < 2 or int(density_samples) != density_samples or max_evaluations < density_samples
        or int(max_evaluations) != max_evaluations or not 0 < ratio_rtol <= 1.e-6
        or not all(math.isfinite(v) and v > 0 for v in (trial_timeout, temperature_timeout))
        or not temperatures or any(not .01 <= t <= 120 for t in temperatures)
        or any(b <= a for a,b in zip(temperatures[:-1],temperatures[1:]))
        or any(float(t).hex() not in inputs['boundary_cache'] for t in temperatures)):
        raise ValueError('Invalid fraction, diffusion model, temperatures, or bounded controls')
    if inputs['physics']['NM_type'] != 'PNM' or inputs['physics']['ms'] != 0:
        raise ValueError('This manuscript estimate requires the established PNM, ms=0 physics')
    directory = Path(directory)
    cache_path = directory/'thermodynamics.npy'
    signature = identity(inputs)
    cache = _load(cache_path) if cache_path.exists() else dict(identity=signature, evaluations={})
    if cache['identity'] != signature:
        raise ValueError('Incompatible thermodynamic cache: physics, sources, or saved domain changed')
    controls = dict(fraction=float(fraction), diffusion=diffusion, temperatures=temperatures,
                    density_samples=int(density_samples), ratio_rtol=float(ratio_rtol),
                    max_evaluations=int(max_evaluations), trial_timeout=trial_timeout,
                    temperature_timeout=temperature_timeout)
    result_path = directory/f'overlay-{diffusion}-f{fraction:.12g}-{_digest(controls)[:12]}.npy'
    result = _load(result_path) if result_path.exists() else dict(identity=signature, controls=controls,
               rows={}, work_complete=False, result_path=str(result_path))
    if result['identity'] != signature or result['controls'] != controls:
        raise ValueError('Incompatible overlay result')
    # A killed kernel leaves a reservation, not a fictitious evaluated state.
    for row in cache['evaluations'].values():
        if row['status'] == 'running':
            row.update(status='interrupted', message='Prior evaluation interrupted; use a fresh directory to retry')
    atomic_save(cache_path, cache)
    call = evaluator or _bounded_evaluator(inputs, trial_timeout, import_paths)
    if evaluator is None and any(float(t).hex() not in result['rows'] or
                                 result['rows'][float(t).hex()]['status'] == 'running' for t in temperatures):
        # Fail once before reserving physical states if the binary runtime is broken.
        cache['runtime'] = call()
        atomic_save(cache_path, cache)
    completed = sum(r['status'] != 'running' for r in result['rows'].values())
    with tqdm(total=len(temperatures), initial=completed, desc='MFP temperature slices', unit='slice',
              dynamic_ncols=True, disable=not show_progress) as target_bar, \
         tqdm(total=None, initial=len(cache['evaluations']), desc='Saved EOS evaluations', unit='state',
              dynamic_ncols=True, disable=not show_progress) as evaluation_bar:
        for T in temperatures:
            key = float(T).hex()
            if key in result['rows'] and result['rows'][key]['status'] != 'running': continue
            started = time.monotonic(); used = 0; sample_rows = []
            boundary = inputs['boundary_cache'][key]
            row = dict(key=key, T_0minus=T, status='running', samples=sample_rows)
            result['rows'][key] = row
            atomic_save(result_path, result)

            def evaluate(density):
                nonlocal used
                if used >= max_evaluations: raise SearchLimit('evaluation_budget')
                if time.monotonic()-started >= temperature_timeout: raise SearchLimit('time_budget')
                used += 1
                point_key = f'{key}:{float(density).hex()}'
                point = dict(key=point_key, T_0minus=T, nB_0minus_over_n0=float(density),
                             nB_0minus=float(density*inputs['n0']))
                if point_key not in cache['evaluations']:
                    record = dict(point, status='running')
                    cache['evaluations'][point_key] = record
                    atomic_save(cache_path, cache)
                    tick = time.monotonic()
                    try:
                        # The last live evaluation cannot overrun the slice budget.
                        remaining = temperature_timeout-(time.monotonic()-started)
                        evaluate_call = call if evaluator else _bounded_evaluator(inputs, min(trial_timeout, remaining), import_paths)
                        state = evaluate_call(point)
                        if not state.get('success') or not state.get('validated'):
                            raise ValueError('EOS state failed validation')
                        for model in ('full','landau'):
                            lengths(state['muB_inf'], T, state[f'D_{model}'], state['gamma_K'])
                        record.update(status='success', state=state)
                    except Exception as exc:
                        record.update(status='failed_evaluation', message=f'{type(exc).__name__}: {exc}')
                    record['runtime_s'] = time.monotonic()-tick
                    atomic_save(cache_path, cache)
                    if progress: progress('evaluation', record)
                    evaluation_bar.update(1)
                record = cache['evaluations'][point_key]
                if record['status'] != 'success':
                    raise RuntimeError(record.get('message', record['status']))
                state = record['state']
                value = lengths(state['muB_inf'], T, state[f'D_{diffusion}'], state['gamma_K'])
                point.update(value, evaluation_key=point_key)
                sample_rows.append(point)
                return point

            try:
                if boundary['status'] == 'outside_window':
                    row.update(status='outside_domain', message='Saved boundaries have no allowed interval in this window')
                elif boundary['status'] != 'success':
                    row.update(status='failed_boundary', message='Saved boundary is not validated')
                else:
                    lo, hi = max(1., float(boundary['lower'])), min(5.5, float(boundary['upper']))
                    row['density_interval'] = (lo,hi)
                    if not math.isfinite(lo+hi) or lo >= hi:
                        row.update(status='outside_domain', message='Empty allowed density interval')
                    else:
                        points = [evaluate(d) for d in np.linspace(lo, hi, density_samples)]
                        roots = [p for p in points if abs(p['ratio']/fraction-1) <= ratio_rtol]
                        bracket = next(((a,b) for a,b in zip(points[:-1], points[1:])
                                        if (a['ratio']-fraction)*(b['ratio']-fraction) < 0), None)
                        if not roots and bracket is None:
                            row.update(status='no_crossing', message='Requested fraction is not bracketed inside the saved allowed interval')
                        else:
                            if not roots:
                                a,b = bracket
                                row['initial_bracket'] = (a['nB_0minus_over_n0'],b['nB_0minus_over_n0'])
                                while not roots:
                                    weight = (fraction-a['ratio'])/(b['ratio']-a['ratio'])
                                    weight = float(np.clip(weight, .1, .9))
                                    p = evaluate(a['nB_0minus_over_n0'] + weight*(b['nB_0minus_over_n0']-a['nB_0minus_over_n0']))
                                    if abs(p['ratio']/fraction-1) <= ratio_rtol: roots.append(p)
                                    elif (a['ratio']-fraction)*(p['ratio']-fraction) < 0: b=p
                                    else: a=p
                            row.update(status='success', **{k:v for k,v in roots[0].items() if k != 'key'})
                            row['relative_ratio_residual'] = abs(row['ratio']/fraction-1)
            except SearchLimit as exc:
                row.update(status=str(exc), message='Bounded root search exhausted its allowance')
            except Exception as exc:
                row.update(status='failed_evaluation', message=f'{type(exc).__name__}: {exc}')
            row['runtime_s'] = time.monotonic()-started
            row['ratio_range'] = (min(p['ratio'] for p in sample_rows), max(p['ratio'] for p in sample_rows)) if sample_rows else None
            result['work_complete'] = len(result['rows']) == len(temperatures) and all(
                r['status'] in ('success','no_crossing','outside_domain') for r in result['rows'].values())
            atomic_save(result_path, result)
            atomic_save(directory/'latest-overlay.npy', result)
            if progress: progress('temperature', row)
            target_bar.update(1)
        # Also refresh selection after a cache-only resume / fraction switch.
        atomic_save(directory/'latest-overlay.npy', result)
    return result


def report(result):
    counts = Counter(r['status'] for r in result['rows'].values())
    ranges = [r['ratio_range'] for r in result['rows'].values() if r.get('ratio_range')]
    print(f"Fraction = {result['controls']['fraction']:g}; diffusion = {result['controls']['diffusion']}")
    print('Temperature slices:', dict(counts))
    if ranges: print(f'Sampled mfp / tail-length ratio: {min(r[0] for r in ranges):.6g} .. {max(r[1] for r in ranges):.6g}')
    print(f"Validated roots: {counts['success']}; unresolved slices: {sum(v for k,v in counts.items() if k not in ('success','no_crossing','outside_domain'))}")
    if not counts['success']: print('No validated crossing: no overlay or overlay legend will be drawn.')
    if not result['work_complete']: print('Incomplete assessment: inspect failed/unresolved rows; these are not no-crossing results.')


def root_segments(result):
    """Never connect through a failed, missing, or no-crossing temperature slice."""
    if not result: return []
    rows = result['rows']; segments=[]; current=[]
    temperatures = result['controls'].get('temperatures', sorted(r['T_0minus'] for r in rows.values()))
    by_temperature = {r['T_0minus']:r for r in rows.values()}
    fraction = result['controls']['fraction']
    for T in temperatures:
        row = by_temperature.get(T, {})
        if (row.get('status') == 'success' and math.isfinite(row.get('ratio', math.nan))
                and abs(row['ratio']/fraction-1) <= result['controls'].get('ratio_rtol',1.e-6)):
            current.append((row['nB_0minus_over_n0'],T))
        else:
            if current: segments.append(np.asarray(current)); current=[]
    if current: segments.append(np.asarray(current))
    return segments
