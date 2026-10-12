#!/usr/bin/env python3
"""Full/semi analytical xi=-0.85 scan at T(0+)=0.1 MeV on saved geometry.

Direct fixed-density upstream recovery and each unchanged analytical method
run in separate bounded calls. No numerical BVP or boundary solve is called.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
import platform
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import scipy
from tqdm.auto import tqdm

import _adiabatic_physics as engine
from _bounded_call import run_module_call_with_hard_timeout


# Tunable run parameters
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = engine.REPO_ROOT
RUN_XI = -0.85
INNER_FRACTION = 0.005
DEFAULT_DOMAIN = SCRIPT_DIR / "inputs/T0plus0p1-domain.npy"
DEFAULT_NUMERICAL = SCRIPT_DIR / "inputs/T0plus0p1-uNmax-state-of-art-complete-5km.npy"
DEFAULT_OUTPUT = SCRIPT_DIR / "T0plus0p1-analytical-xi-minus0p85.npy"
DEFAULT_WORKERS = engine.default_workers()
DEFAULT_METHOD_TIMEOUT_S = 120.0
DEFAULT_TARGET_M_S = 5000.0
DEFAULT_BRACKET_WIDTH = 0.01
DEFAULT_MAX_RHO = 0.999
DEFAULT_ADAPTIVE_ROUNDS = 12
SPEED_OF_LIGHT = 299792458.0
METHODS = ("analytic", "semi_analytic")
atomic_save = engine.atomic_save


def load_inputs(domain_path=DEFAULT_DOMAIN, numerical_path=DEFAULT_NUMERICAL):
    return (np.load(domain_path, allow_pickle=True).item(),
            np.load(numerical_path, allow_pickle=True).item())


def point_key(point):
    return f"{int(point['theta_index'])}:{float(point['radial_fraction']):.12f}"


def point_on_ray(domain, theta_index, fraction):
    if not 0 < fraction < 1:
        raise ValueError("Radial fraction must be strictly inside (0, 1)")
    reference = next(p for p in domain['points'] if p['theta_index'] == theta_index)
    state = reference['states'][0]
    rho = state['rho_inner_boundary'] + fraction * (state['rho_outer_boundary'] - state['rho_inner_boundary'])
    theta = float(reference['theta_rad'])
    transform = domain['transform']
    return dict(theta_index=int(theta_index), theta_rad=theta,
                radial_fraction=float(fraction),
                nB_0minus_over_n0=float(transform['density_origin_over_n0'] +
                    transform['density_scale_over_n0'] * rho * np.cos(theta)),
                T_0minus_MeV=0.0 if theta_index == 0 else float(
                    transform['temperature_scale_MeV'] * rho * np.sin(theta)))


def build_points(domain, numerical):
    engine.validate_domain_manifest(domain)
    if not np.array_equal(np.asarray(domain['T_0plus_values_MeV']), [0.1]):
        raise ValueError("Expected fixed T(0+)=0.1 MeV")
    if numerical['T_0plus_MeV'] != 0.1:
        raise ValueError("Numerical temperature mismatch")
    for key in ('B14_MeV', 'n0_MeV3', 'ncrit_over_n0'):
        if not np.isclose(domain['calibration'][key], numerical['calibration'][key], rtol=1e-12):
            raise ValueError(f"Numerical/domain calibration mismatch: {key}")
    if (domain['calibration']['ncrit_over_n0'] != 3 or domain['xi'] != -0.5
            or domain['ms_MeV'] != 0 or domain['NM_type'] != 'PNM'):
        raise ValueError("Expected source geometry with 3n0 calibration, xi=-0.5, ms=0 and PNM")
    if not np.all(numerical['plot_success_grid']):
        raise ValueError("The reference numerical plotting grid contains failures")
    if not numerical['complete_5000_contour_bracket']:
        raise ValueError("The numerical reference must bracket 5000 m/s")
    points = {}
    ntheta, nradial = (1 + max(p[k] for p in domain['points'])
                       for k in ('theta_index', 'radial_index'))
    original_keys = np.empty((ntheta, nradial), dtype='U40')
    for p in domain['points']:
        state = p['states'][0]
        point = point_on_ray(domain, p['theta_index'], p['radial_fraction_used'])
        # Keep the stored coordinates, not regenerated round-off variants.
        point.update({k: float(state[k]) for k in ('nB_0minus_over_n0', 'T_0minus_MeV')})
        point['source'] = 'original_domain'
        key = point_key(point)
        points[key] = point
        original_keys[p['theta_index'], p['radial_index']] = key
    theta_axis = np.array([points[k]['theta_rad'] for k in original_keys[:, 0]])
    if not (theta_axis[0] == 0 and theta_axis[-1] == np.pi / 2 and np.all(np.diff(theta_axis) > 0)):
        raise ValueError("Both exact angular endpoints must be present")
    if not np.allclose(theta_axis, numerical['plot_theta_rad_axis'], rtol=0, atol=1e-14):
        raise ValueError("Numerical/domain angular grids differ")
    matched_keys = np.empty(numerical['plot_rho_grid'].shape, dtype='U40')
    for ij in np.ndindex(matched_keys.shape):
        point = point_on_ray(domain, ij[0], float(numerical['plot_rho_grid'][ij]))
        for key, field in (('nB_0minus_over_n0', 'plot_nB_0minus_over_n0_grid'),
                           ('T_0minus_MeV', 'plot_T_0minus_MeV_grid')):
            if not np.isclose(point[key], numerical[field][ij], rtol=1e-10, atol=1e-10):
                raise ValueError("Numerical/domain ray geometry mismatch")
            point[key] = float(numerical[field][ij])
        point['source'] = 'numerical_plot_grid'
        key = point_key(point)
        points[key] = point
        matched_keys[ij] = key
    for theta in range(ntheta):
        point = dict(point_on_ray(domain, theta, INNER_FRACTION), source='inner_extension')
        points[point_key(point)] = point
    return points, matched_keys, original_keys


def good(method):
    return bool(method.get('success') and np.isfinite(method.get('u_0minus', np.nan))
                and method['u_0minus'] > 0 and np.isfinite(method.get('jB', np.nan))
                and method['jB'] > 0)


def ray_rows(records, theta):
    return sorted(((key, row) for key, row in records.items()
                   if row['point']['theta_index'] == theta),
                  key=lambda pair: pair[1]['point']['radial_fraction'])


def ray_bracket(records, theta, target, width):
    rows = ray_rows(records, theta)
    for (left_key, left), (right_key, right) in zip(rows, rows[1:]):
        if not (good(left['semi_analytic']) and good(right['semi_analytic'])):
            continue
        lo, hi = (row['semi_analytic']['u_0minus'] * SPEED_OF_LIGHT for row in (left, right))
        span = right['point']['radial_fraction'] - left['point']['radial_fraction']
        if lo <= target <= hi:
            return dict(keys=(left_key, right_key), width=span, complete=span <= width + 1e-12)
    return dict(keys=None, width=np.nan, complete=False)


def next_probe(records, theta, target, width, max_rho):
    report = ray_bracket(records, theta, target, width)
    if report['complete']:
        return None
    if report['keys']:
        return float(np.mean([records[k]['point']['radial_fraction'] for k in report['keys']]))
    rows = ray_rows(records, theta)
    below = [row['point']['radial_fraction'] for _, row in rows
             if good(row['semi_analytic']) and row['semi_analytic']['u_0minus'] * SPEED_OF_LIGHT < target]
    if not below:
        # Seek the lower side if even the innermost converged value exceeds target.
        return rows[0][1]['point']['radial_fraction'] / 2 if rows else None
    lo = max(below)
    upper = [row['point']['radial_fraction'] for _, row in rows
             if good(row['semi_analytic']) and row['point']['radial_fraction'] > lo]
    hi = min(upper) if upper else max_rho
    # Failed samples break contour adjacency, but must not be search boundaries.
    # Avoid repeatedly requesting an existing failed midpoint.
    existing = {round(row['point']['radial_fraction'], 12) for _, row in rows}
    for depth in range(1, 21):
        for fraction in (1 - 2. ** -depth, 2. ** -depth):
            candidate = lo + fraction * (hi - lo)
            if hi - lo > 1e-7 and round(candidate, 12) not in existing:
                return candidate
    return None


def coverage(records, ntheta, target, width):
    rays = [ray_bracket(records, theta, target, width) for theta in range(ntheta)]
    return dict(complete=all(r['complete'] for r in rays), rays=rays,
                complete_ray_count=sum(r['complete'] for r in rays),
                unresolved_theta_indices=[i for i, r in enumerate(rays) if not r['complete']])


def evaluate_point(point, domain, timeout_s, cancel=None, methods=METHODS):
    started = time.monotonic()
    row = {}
    for method in methods:
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Calculation cancelled')
        try:
            row[method] = run_module_call_with_hard_timeout(
                '_adiabatic_physics', 'evaluate_method', args=(point, domain, method),
                timeout_s=timeout_s, cancel=cancel)
        except Exception as exc:
            row[method] = dict(success=False, u_0minus=np.nan, jB=np.nan,
                               message=f'{type(exc).__name__}: {exc}')
    row['point'] = point
    row['total_runtime_s'] = time.monotonic() - started
    for method in methods:
        if not good(row[method]):
            row[method]['success'] = False
            row[method]['message'] = row[method].get('message') or 'Non-positive or non-finite solver result'
    return row


def accept_result(records, row, checkpoint, increment):
    key = point_key(row['point'])
    if key in records:
        saved = records[key]
        updates = {m: row[m] for m in METHODS if m in row and not good(saved.get(m, {}))}
        if not updates:
            return
        saved.setdefault('retry_history', []).append({
            'at_utc': datetime.now(timezone.utc).isoformat(),
            'previous': {m: copy.deepcopy(saved.get(m)) for m in updates}})
        saved.update(updates)
        saved['last_retry_runtime_s'] = row.get('total_runtime_s', np.nan)
    else:
        records[key] = row
    checkpoint()
    increment()


def run_batch(points, records, domain, args, checkpoint, progress, evaluator=evaluate_point):
    cancel = threading.Event()
    iterator = iter(points)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    pending = {}
    def submit():
        point = next(iterator, None)
        if point is not None:
            saved = records.get(point_key(point), {})
            methods = tuple(m for m in METHODS if not good(saved.get(m, {})))
            future = pool.submit(evaluator, point, domain, args.method_timeout, cancel, methods=methods)
            pending[future] = point
    try:
        for _ in range(args.workers):
            submit()
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                point = pending.pop(future)
                row = future.result()
                accept_result(records, row, checkpoint, lambda: progress.update(1))
                row = records[point_key(point)]
                progress.set_postfix(theta=point['theta_index'], rho=f"{point['radial_fraction']:.5f}",
                                     full=good(row['analytic']), semi=good(row['semi_analytic']))
                submit()
    finally:
        cancel.set()
        pool.shutdown(wait=True, cancel_futures=True)


def validate_resume(saved, contract):
    if saved.get('contract') != contract:
        raise ValueError("Resume contract differs: choose a new --output for changed inputs/settings/code")


def grid_payload(keys, points, records):
    grid = dict(point_keys=keys.copy())
    for field in ('nB_0minus_over_n0', 'T_0minus_MeV', 'radial_fraction'):
        grid[field] = np.array([points[k][field] if k else np.nan for k in keys.flat]).reshape(keys.shape)
    for method in METHODS:
        valid = np.array([good(records.get(k, {}).get(method, {})) for k in keys.flat]).reshape(keys.shape)
        velocity = np.array([records.get(k, {}).get(method, {}).get('u_0minus', np.nan)
                             for k in keys.flat]).reshape(keys.shape) * SPEED_OF_LIGHT
        grid[f'{method}_success'] = valid
        grid[f'{method}_u_0minus_m_s'] = np.where(valid, velocity, np.nan)
    return grid


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain', type=Path, default=DEFAULT_DOMAIN)
    p.add_argument('--numerical', type=Path, default=DEFAULT_NUMERICAL)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    p.add_argument('--method-timeout', type=float, default=DEFAULT_METHOD_TIMEOUT_S,
                   help='Hard seconds per method, including direct-density recovery')
    p.add_argument('--target-velocity', type=float, default=DEFAULT_TARGET_M_S)
    p.add_argument('--bracket-width', type=float, default=DEFAULT_BRACKET_WIDTH)
    p.add_argument('--max-rho', type=float, default=DEFAULT_MAX_RHO)
    p.add_argument('--adaptive-rounds', type=int, default=DEFAULT_ADAPTIVE_ROUNDS)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Evaluate only three representative points, to a separate output')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.workers < 1 or not np.isfinite(args.method_timeout) or args.method_timeout <= 0
            or not 0.99 < args.max_rho < 1 or not 0 < args.bracket_width < 1
            or not np.isfinite(args.target_velocity) or args.target_velocity <= 0
            or args.adaptive_rounds < 0):
        raise ValueError('Invalid worker, timeout, target or refinement settings')
    domain, numerical = load_inputs(args.domain, args.numerical)
    points, matched_keys, domain_keys = build_points(domain, numerical)
    ntheta = domain_keys.shape[0]
    geometry_source_xi = domain['xi']
    domain = dict(domain, xi=RUN_XI)
    inner_keys = np.array([point_key(point_on_ray(domain, theta, INNER_FRACTION))
                           for theta in range(ntheta)]).reshape(-1, 1)
    plot_keys = np.column_stack((inner_keys, domain_keys))
    if args.smoke:
        keep = [domain_keys[0, 0], domain_keys[ntheta // 2, 16], domain_keys[-1, -1]]
        points = {k: points[k] for k in keep}
        matched_keys = domain_keys = np.array(keep, dtype='U40').reshape(-1, 1)
        plot_keys = domain_keys
        args.output = args.output.with_name(args.output.stem + '-smoke.npy')
    print(f'T(0+)=0.1 MeV; B^(1/4)={domain["calibration"]["B14_MeV"]:.10f} MeV; xi={domain["xi"]}')
    print(f'{len(points)} paired points; {ntheta} rays including 0 and 90 degrees; {args.workers} workers')
    print(f'Geometry and numerical reference source xi={geometry_source_xi}; numerical speeds are not xi={RUN_XI} results.')
    print(f'Hard time cap: {args.method_timeout:g} s/method; output: {args.output}')
    if args.dry_run:
        print('Inputs validated. No calculations launched or output modified.')
        return 0
    source_files = [args.domain, args.numerical, Path(__file__), SCRIPT_DIR / '_adiabatic_physics.py',
                    SCRIPT_DIR / '_bounded_call.py']
    source_files += sorted((REPO_ROOT / 'RMFsolver').glob('*.py'))
    contract = dict(schema='fixed_T0plus_adiabatic_xi_v2', xi=RUN_XI, inner_fraction=INNER_FRACTION,
                    source_sha256={str(p.resolve().relative_to(REPO_ROOT)) if p.resolve().is_relative_to(REPO_ROOT)
                                   else str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in source_files},
                    target=args.target_velocity, width=args.bracket_width, max_rho=args.max_rho,
                    method_timeout=args.method_timeout, adaptive_rounds=args.adaptive_rounds, smoke=args.smoke)
    records, adaptive_round = {}, 0
    if args.output.exists():
        saved = np.load(args.output, allow_pickle=True).item()
        validate_resume(saved, contract)
        records = saved['records']
        points.update(saved['points'])
        adaptive_round = saved['adaptive_round']
        print(f'Resuming {len(records)} terminal points; completed methods kept, failed methods retried once.')

    def checkpoint(run_complete=False):
        report = coverage(records, ntheta, args.target_velocity, args.bracket_width)
        bracket_keys = np.full((ntheta, 2), '', dtype='U40')
        for theta, ray in enumerate(report['rays']):
            if ray['complete']:
                bracket_keys[theta] = ray['keys']
        payload = dict(contract=contract, checkpoint_at_utc=datetime.now(timezone.utc).isoformat(),
                       run_complete=run_complete, production_run=not args.smoke,
                       T_0plus_MeV=0.1, xi=domain['xi'], geometry_source_xi=geometry_source_xi,
                       calibration=domain['calibration'],
                       boundary_diagnostics=domain['boundary_diagnostics'], transform=domain['transform'],
                       points=points, records=records, completed_point_count=len(records),
                       planned_point_count=len(points), adaptive_round=adaptive_round,
                       theta_rad_axis=numerical['plot_theta_rad_axis'], coverage=report,
                       complete_5000_contour=report['complete'] and not args.smoke and args.target_velocity == 5000,
                       complete_target_contour=report['complete'] and not args.smoke,
                       target_velocity_m_s=args.target_velocity,
                       runtime_versions=dict(python=platform.python_version(), numpy=np.__version__,
                                             scipy=scipy.__version__),
                       matched_numerical_grid=grid_payload(matched_keys, points, records),
                       original_domain_grid=grid_payload(domain_keys, points, records),
                       plot_grid=grid_payload(plot_keys, points, records),
                       semi_analytic_bracket_grid=grid_payload(bracket_keys, points, records),
                       numerical_reference_grid=dict(xi=geometry_source_xi, role='source_reference_only',
                           nB_0minus_over_n0=numerical['plot_nB_0minus_over_n0_grid'],
                           T_0minus_MeV=numerical['plot_T_0minus_MeV_grid'],
                           u_0minus_m_s=numerical['plot_u_0minus_m_s_grid']))
        atomic_save(args.output, payload)

    checkpoint()
    pending = [p for k, p in points.items() if any(not good(records.get(k, {}).get(m, {})) for m in METHODS)]
    with tqdm(total=len(points), initial=len(points) - len(pending),
              desc='Full + semi analytical', unit='point') as progress:
        run_batch(pending, records, domain, args, checkpoint, progress)
        while not args.smoke and adaptive_round < args.adaptive_rounds:
            if coverage(records, ntheta, args.target_velocity, args.bracket_width)['complete']:
                break
            additions = []
            for theta in range(ntheta):
                rho = next_probe(records, theta, args.target_velocity, args.bracket_width, args.max_rho)
                if rho is not None:
                    point = point_on_ray(domain, theta, rho)
                    point['source'] = 'semi_analytic_bracket_refinement'
                    if point_key(point) not in points:
                        additions.append(point)
                        points[point_key(point)] = point
            if not additions:
                break
            adaptive_round += 1
            progress.total = len(points)
            progress.set_description(f'5000 bracket round {adaptive_round}')
            progress.refresh()
            checkpoint()
            run_batch(additions, records, domain, args, checkpoint, progress)
    checkpoint(run_complete=True)
    report = coverage(records, ntheta, args.target_velocity, args.bracket_width)
    print('Successful points:', {m: sum(good(r[m]) for r in records.values()) for m in METHODS})
    print(f'Semi-analytical {args.target_velocity:g} m/s brackets: {report["complete_ray_count"]}/{ntheta}')
    print('Unresolved theta indices:', report['unresolved_theta_indices'])
    if args.smoke:
        return 0 if all(good(r[m]) for r in records.values() for m in METHODS) else 2
    return 0 if report['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
