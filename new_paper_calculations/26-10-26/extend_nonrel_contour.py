#!/usr/bin/env python3
"""Resolve the 50 and 5000 m/s contours without changing the 26-10-24 dataset."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / 'new_paper_calculations/26-10-24'
sys.path.insert(0, str(PREVIOUS))
import _radial_shell_continuation as radial
import _slow_front_adapter as adapter
import _method_engine as engine

BASE = PREVIOUS / 'T0plus0p1-adiabatic-slow.npy'
DOMAIN = PREVIOUS / 'T0plus0p1-domain.npy'
OUTPUT = Path(__file__).with_name('nonrel-contour-extension.npy')
TARGET_M_S = 5000.
OUTER_MARGIN_M_S = 150.
OUTER_STEP = .02
MAX_FRACTION = .985
BRACKET_WIDTH = .003
INNER_FRACTION = .005
TRIAL_TIMEOUT_S = 120.


def solve_ray(index):
    base = np.load(BASE, allow_pickle=True).item()
    domain = np.load(DOMAIN, allow_pickle=True).item()
    transform = domain['transform']
    points = sorted((p for p in base['points'] if p['theta_index'] == index), key=lambda p: p['radial_index'])
    records = {r['point_index']: r for r in base['radial_shell_records']}
    endpoints = points[-1]['states'][0]
    theta = points[-1]['theta_rad']
    attempts, accepted = [], []

    def endpoint(p):
        r = records[p['point_index']]
        return dict(fraction=p['radial_fraction_used'], speed=r['u_0minus'] * 299792458.,
                    nB_0minus_over_n0=p['nB_0minus_over_n0'], T_0minus_MeV=p['T_0minus_MeV'],
                    continuation_state=base['radial_shell_profiles'][p['point_index']]['continuation_state'],
                    source_point_index=p['point_index'])

    def evaluate(fraction, previous):
        rho = endpoints['rho_inner_boundary'] + fraction * (endpoints['rho_outer_boundary'] - endpoints['rho_inner_boundary'])
        density = transform['density_origin_over_n0'] + transform['density_scale_over_n0'] * rho * np.cos(theta)
        temperature = transform['temperature_scale_MeV'] * rho * np.sin(theta)
        point = dict(theta_index=index, theta_rad=theta, T_0minus_MeV=float(temperature),
                     nB_0minus_over_n0=float(density), nB_0minus_MeV3=float(density * base['calibration']['n0_MeV3']))
        task = dict(point=point, B14_MeV=base['calibration']['B14_MeV'], T_0plus_MeV=.1,
                    ms_MeV=0., nm_type='PNM', tail_eps=1e-8, n_mesh=300,
                    tol_bvp=1e-4, max_nodes=50000, path_rtol=1e-6, kappa_factor=1.)
        state = previous['continuation_state']
        for multiplier, use_profile in ((1., True), (.75, True), (1.25, True), (1., False)):
            candidate = dict(jB_guess=state['jB'] * multiplier, use_profile=use_profile)
            kwargs = radial._solver_kwargs_for_candidate(task, candidate, dict(continuation_state=state))
            try:
                result = radial._solve_slow_with_hard_timeout(kwargs, TRIAL_TIMEOUT_S)
                error = adapter.validation_error(result, task)
            except Exception as exc:
                result, error = {}, str(exc)
            attempts.append(dict(fraction=fraction, jB_guess=candidate['jB_guess'],
                                 use_profile=use_profile, message=error,
                                 success=error is None, path_rtol=result.get('path_rtol')))
            if error is None:
                row = dict(point, fraction=float(fraction), speed=result['u_0minus'] * 299792458.,
                           continuation_state=result['continuation_state'], result=result)
                accepted.append(row)
                return row
        raise RuntimeError(f'Ray {index}, fraction {fraction}: {attempts[-1]["message"]}')

    try:
        inner = evaluate(INNER_FRACTION, endpoint(points[0]))
        if not inner['speed'] < 50.:
            raise RuntimeError('Inner extension does not bracket 50 m/s')
        lower = endpoint(points[-1])
        previous = lower
        while previous['speed'] <= TARGET_M_S + OUTER_MARGIN_M_S:
            fraction = min(previous['fraction'] + OUTER_STEP, MAX_FRACTION)
            if fraction <= previous['fraction']:
                raise RuntimeError('No 5000 m/s bracket below the composition ceiling')
            current = evaluate(fraction, previous)
            if current['speed'] < TARGET_M_S:
                lower = current
            previous = current
        upper = previous
        while upper['fraction'] - lower['fraction'] > BRACKET_WIDTH:
            middle = evaluate((upper['fraction'] + lower['fraction']) / 2, lower)
            if middle['speed'] < TARGET_M_S:
                lower = middle
            else:
                upper = middle
        return dict(theta_index=index, success=True, inner=inner, lower=lower, upper=upper,
                    accepted=accepted, attempts=attempts)
    except Exception as exc:
        return dict(theta_index=index, success=False, message=str(exc), accepted=accepted, attempts=attempts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    base = np.load(BASE, allow_pickle=True).item()
    if not base.get('all_targets_successful'):
        raise ValueError('Base contour is incomplete')
    # Supplementary results must use precisely the same solver/EOS as the base.
    for name, expected in base['run_identity']['code_sha256'].items():
        if adapter.sha256(ROOT / name) != expected:
            raise ValueError('Source changed since the base run: ' + name)
    identity = dict(base_sha256=adapter.sha256(BASE), domain_sha256=adapter.sha256(DOMAIN),
                    script_sha256=adapter.sha256(__file__))
    payload = dict(identity=identity, rays={}, complete=False, target_m_s=TARGET_M_S,
                   hydrodynamic_model='nonrelativistic_isobaric', T_0plus_MeV=.1)
    if OUTPUT.exists():
        payload = np.load(OUTPUT, allow_pickle=True).item()
        if payload['identity'] != identity:
            raise ValueError('Existing extension has different source hashes')
    pending = [i for i in range(30) if not payload['rays'].get(i, {}).get('success')]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(solve_ray, i): i for i in pending}
        for future in as_completed(futures):
            row = future.result()
            payload['rays'][row['theta_index']] = row
            engine.atomic_save(OUTPUT, payload)
            print(f'Ray {row["theta_index"]:02d}: ' + (f'{len(row["accepted"])} accepted, 5000 bracket resolved'
                  if row['success'] else row['message']), flush=True)
    payload['complete'] = all(payload['rays'].get(i, {}).get('success') for i in range(30))
    engine.atomic_save(OUTPUT, payload)
    print('Complete:', payload['complete'], flush=True)
    return 0 if payload['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
