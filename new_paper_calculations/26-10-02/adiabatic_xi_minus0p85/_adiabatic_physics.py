"""Small analytical-only support extracted from the proven September workflow.

Physics imports are deferred to bounded child calls, so dry-run needs no EOS.
The direct-density path is the one used by 26-09-28/repair_analytical_local.py.
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def default_workers():
    for name in ('SLURM_CPUS_PER_TASK', 'PBS_NP', 'NSLOTS'):
        try:
            count = int(os.environ[name])
            if count > 0:
                return count
        except (KeyError, ValueError):
            pass
    if hasattr(os, 'sched_getaffinity'):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npy', delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, payload, allow_pickle=True)
    temporary.replace(path)


def validate_domain_manifest(manifest):
    if manifest.get('manifest_type') != 'T0plus_semi_analytic_contour_domain_v1':
        raise ValueError('Unsupported domain manifest type')
    if manifest.get('selection_rule') != 'semi_analytic_safe_domain':
        raise ValueError('Expected saved semi-analytical geometry')
    if not isinstance(manifest.get('points'), list) or not manifest['points']:
        raise ValueError('Domain manifest contains no points')
    temperatures = np.asarray(manifest.get('T_0plus_values_MeV', []), dtype=float)
    if temperatures.ndim != 1 or not temperatures.size or not np.all(np.isfinite(temperatures)):
        raise ValueError('Invalid domain temperatures')
    for key in ('n0_MeV3', 'B14_MeV'):
        if not np.isfinite(manifest.get('calibration', {}).get(key, np.nan)):
            raise ValueError(f'Invalid calibration: {key}')
    for point in manifest['points']:
        states = point.get('states', [])
        if len(states) != len(temperatures) or {s['T_0plus_MeV'] for s in states} != set(temperatures):
            raise ValueError('Domain trajectory temperatures differ')
        for state in states:
            values = [state.get(k, np.nan) for k in ('T_0minus_MeV', 'nB_0minus_over_n0',
                      'rho', 'rho_inner_boundary', 'rho_outer_boundary')]
            if not np.all(np.isfinite(values)) or not values[3] < values[2] < values[4]:
                raise ValueError('Invalid saved domain geometry')


def _as_finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def evaluate_method(point, domain, method):
    """Validated fixed-density recovery and unchanged repository methods."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from RMFsolver import phase_velocity, RMFparameter as para

    started = time.monotonic()
    density = point['nB_0minus_over_n0'] * domain['calibration']['n0_MeV3']
    temperature = point['T_0minus_MeV']
    state = phase_velocity._validated_pnm_state_from_nB(
        density, temperature, param=para.paraQMCRMF3)
    density_error = abs(state['upstream_density_relative_residual'])
    if not np.isfinite(density_error) or density_error > domain['density_match_rtol']:
        raise RuntimeError(f'Direct upstream density residual {density_error:g}')
    function = dict(analytic=phase_velocity.analytic_velocity_bound,
                    semi_analytic=phase_velocity.semi_analytic_velocity_bound)[method]
    result = function(state['muB_0minus'], temperature, domain['calibration']['B14_MeV'],
                      T_0plus=float(domain['T_0plus_values_MeV'][0]), xi=domain['xi'],
                      ms=domain['ms_MeV'], param=para.paraQMCRMF3,
                      NM_type=domain['NM_type'], upB=domain['upB'])
    if 'success' in result and not result['success']:
        raise RuntimeError(result.get('message') or 'Analytical solver explicitly rejected the result')
    row = {key: _as_finite_float(result.get(key))
           for key in ('u_0minus', 'jB', 'a_0plus', 'a_inf', 'lambda_n', 'A_boundary')}
    row.update(success=True, message='', T_inf_MeV=_as_finite_float(result.get('T_inf')),
               runtime_s=time.monotonic() - started, muB_0minus_MeV=float(state['muB_0minus']),
               density_relative_error=float(density_error), source='direct_density',
               evaluated_at_utc=datetime.now(timezone.utc).isoformat())
    if not all(np.isfinite(row[key]) and row[key] > 0 for key in ('u_0minus', 'jB')):
        raise RuntimeError('Non-positive or non-finite analytical result')
    return row
