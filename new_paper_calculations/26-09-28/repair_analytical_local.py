#!/usr/bin/env python3
"""Recover failed analytical samples in place, keeping the cluster backup."""
from __future__ import annotations

import argparse
import copy
import hashlib
import platform
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import run_T0plus0p1_analytical as scan
import numpy as np
import scipy
from tqdm.auto import tqdm
from _bounded_call import run_module_call_with_hard_timeout


def now():
    return datetime.now(timezone.utc).isoformat()


def backup_once(output):
    backup = output.with_name(output.stem + '.pre-local-repair.npy')
    if not backup.exists():
        shutil.copy2(output, backup)
    return backup


def evaluate_method(point, domain, method):
    """Direct validated PNM recovery; otherwise unchanged analytical physics."""
    started = time.monotonic()
    density = point['nB_0minus_over_n0'] * domain['calibration']['n0_MeV3']
    temperature = point['T_0minus_MeV']
    state = scan.engine.phase_velocity._validated_pnm_state_from_nB(
        density, temperature, param=scan.engine.para.paraQMCRMF3)
    density_error = abs(state['upstream_density_relative_residual'])
    if density_error > domain['density_match_rtol']:
        raise RuntimeError(f'Direct upstream density residual {density_error:g}')
    function = dict(analytic=scan.engine.phase_velocity.analytic_velocity_bound,
                    semi_analytic=scan.engine.phase_velocity.semi_analytic_velocity_bound)[method]
    result = function(state['muB_0minus'], temperature, domain['calibration']['B14_MeV'],
                      T_0plus=float(domain['T_0plus_values_MeV'][0]), xi=domain['xi'],
                      ms=domain['ms_MeV'], param=scan.engine.para.paraQMCRMF3,
                      NM_type=domain['NM_type'], upB=domain['upB'])
    row = {key: scan.engine._as_finite_float(result.get(key))
           for key in ('u_0minus', 'jB', 'a_0plus', 'a_inf', 'lambda_n', 'A_boundary')}
    row.update(success=True, message='', T_inf_MeV=scan.engine._as_finite_float(result.get('T_inf')),
               runtime_s=time.monotonic() - started, muB_0minus_MeV=float(state['muB_0minus']),
               density_relative_error=float(density_error), source='local_direct_density_repair',
               evaluated_at_utc=now())
    if not scan.good(row):
        raise RuntimeError('Non-positive or non-finite analytical result')
    return row


def evaluate_point(point, domain, methods, timeout_s, cancel=None):
    results = {}
    for method in methods:
        try:
            results[method] = run_module_call_with_hard_timeout(
                'repair_analytical_local', 'evaluate_method', args=(point, domain, method),
                timeout_s=timeout_s, cancel=cancel)
        except Exception as exc:
            results[method] = dict(success=False, u_0minus=np.nan, jB=np.nan,
                                   message=f'{type(exc).__name__}: {exc}',
                                   source='local_direct_density_repair', evaluated_at_utc=now())
    return results


def merge_results(saved, results):
    # Successful cluster values remain bit-for-bit intact, even on repeated runs.
    updates = {method: value for method, value in results.items()
               if not scan.good(saved.get(method, {}))}
    if updates:
        saved.setdefault('local_repair_history', []).append(dict(
            at_utc=now(), previous={method: copy.deepcopy(saved.get(method)) for method in updates}))
        saved.update(updates)


def checkpoint(payload, output):
    points, records = payload['points'], payload['records']
    ntheta = len(payload['theta_rad_axis'])
    report = scan.coverage(records, ntheta, payload['target_velocity_m_s'], payload['contract']['width'])
    bracket_keys = np.full((ntheta, 2), '', dtype='U40')
    for theta, ray in enumerate(report['rays']):
        if ray['complete']:
            bracket_keys[theta] = ray['keys']
    for name in ('matched_numerical_grid', 'original_domain_grid'):
        payload[name] = scan.grid_payload(payload[name]['point_keys'], points, records)
    payload['semi_analytic_bracket_grid'] = scan.grid_payload(bracket_keys, points, records)
    if 'local_inner_point_keys' in payload:
        keys = np.column_stack((payload['local_inner_point_keys'],
                                payload['original_domain_grid']['point_keys']))
        payload['repaired_plot_grid'] = scan.grid_payload(keys, points, records)
    payload.update(coverage=report, complete_target_contour=report['complete'],
                   complete_5000_contour=report['complete'] and payload['target_velocity_m_s'] == 5000.,
                   completed_point_count=len(records), planned_point_count=len(points), checkpoint_at_utc=now())
    scan.atomic_save(output, payload)


def run_points(points, payload, domain, args):
    tasks = []
    for point in points:
        key = scan.point_key(point)
        payload['points'][key] = point
        saved = payload['records'].get(key, {})
        methods = [m for m in scan.METHODS if not scan.good(saved.get(m, {}))]
        if methods:
            tasks.append((point, methods))
    cancel = threading.Event()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        pending = {pool.submit(evaluate_point, p, domain, methods, args.timeout, cancel): p
                   for p, methods in tasks}
        with tqdm(total=len(tasks), desc='Local analytical repair', unit='point') as progress:
            for future in as_completed(pending):
                point = pending[future]
                row = payload['records'].setdefault(scan.point_key(point), dict(point=point, T_0plus_MeV=.1))
                merge_results(row, future.result())
                checkpoint(payload, args.output)
                progress.update(1)
                progress.set_postfix(theta=point['theta_index'], rho=point['radial_fraction'],
                                     full=scan.good(row['analytic']), semi=scan.good(row['semi_analytic']))
    finally:
        cancel.set()
        pool.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=scan.DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=120.)
    parser.add_argument('--inner-fraction', type=float, default=.005)
    parser.add_argument('--rounds', type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1 or not 0 < args.timeout < np.inf or not 0 < args.inner_fraction < .02 or args.rounds < 0:
        raise ValueError('Invalid repair parameters')
    domain, numerical = scan.load_inputs()
    scan.build_points(domain, numerical)
    payload = np.load(args.output, allow_pickle=True).item()
    if not payload['production_run'] or payload['T_0plus_MeV'] != .1 or payload['xi'] != domain['xi']:
        raise ValueError('Expected the production 0.1 MeV analytical comparison')
    for key in ('B14_MeV', 'n0_MeV3', 'ncrit_over_n0'):
        if payload['calibration'][key] != domain['calibration'][key]:
            raise ValueError(f'Calibration mismatch: {key}')
    backup = backup_once(args.output)
    source_files = [Path(__file__), scan.SCRIPT_DIR / '_bounded_call.py']
    source_files += sorted((scan.SCRIPT_DIR.parents[1] / 'RMFsolver').glob('*.py'))
    repair = dict(started_at_utc=now(), backup=str(backup),
                  backup_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                  runtime_versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  method_timeout_s=args.timeout, workers=args.workers, inner_fraction=args.inner_fraction,
                  completed=False)
    payload.setdefault('local_repairs', []).append(repair)
    failed = [r['point'] for r in payload['records'].values()
              if any(not scan.good(r.get(m, {})) for m in scan.METHODS)]
    print(f'Backup: {backup}\nRetrying {len(failed)} failed points', flush=True)
    run_points(failed, payload, domain, args)

    inner = [dict(scan.point_on_ray(domain, i, args.inner_fraction), source='local_inner_extension')
             for i in range(len(payload['theta_rad_axis']))]
    payload['local_inner_point_keys'] = np.array([scan.point_key(p) for p in inner]).reshape(-1, 1)
    run_points(inner, payload, domain, args)

    for round_index in range(args.rounds):
        additions = []
        for theta, ray in enumerate(payload['coverage']['rays']):
            if not ray['complete'] and ray['keys']:
                fraction = np.mean([payload['points'][k]['radial_fraction'] for k in ray['keys']])
                point = dict(scan.point_on_ray(domain, theta, fraction), source='local_bracket_refinement')
                if scan.point_key(point) not in payload['records']:
                    additions.append(point)
        if not additions:
            break
        print(f'Local bracket refinement {round_index + 1}: {len(additions)} points', flush=True)
        run_points(additions, payload, domain, args)
    repair.update(completed=True, finished_at_utc=now(),
                  failed_remaining={m: sum(not scan.good(r[m]) for r in payload['records'].values())
                                    for m in scan.METHODS})
    checkpoint(payload, args.output)
    print('Remaining failures:', repair['failed_remaining'])
    print('5000 m/s bracket rays:', payload['coverage']['complete_ray_count'])


if __name__ == '__main__':
    main()
