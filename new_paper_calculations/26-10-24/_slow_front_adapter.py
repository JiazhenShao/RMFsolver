"""Slow-front API, numerical-reference provenance, and checkpoint validation."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MODEL = 'nonrelativistic_isobaric'
RESULT_METADATA = (
    'hydrodynamic_model', 'closure', 'rate_model', 'rate_approximation',
    'alpha_s', 'h_D', 'ms', 'quark_branch', 'conservation_errors',
    'boundary_residuals', 'a_monotone_nonincreasing', 'path_rtol',
    'path_refinement_attempts', 'T_0plus', 'T_inf', 'a_0plus', 'a_inf',
    'muB_0plus', 'muK_0plus', 'nB_0plus', 'nB_inf', 'jK_inf',
    'upstream_seed_extension', 'upstream_rmf_scaled_residual',
    'upstream_density_relative_residual', 'upstream_seed_count',
)
PROFILE_FIELDS = ('s_coord', 'x', 'r', 'muB', 'muK', 'T', 'P', 'e', 'h',
                  'nB', 'nK', 'a', 'u', 'jK', 'Gamma_K', 'D_K', 'invD_K', 'dnK_dx')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solve_with_path_refinement(**kwargs):
    """Retry only table refinement exhaustion, with strictly tighter accuracy.

    The EOS table's fixed iteration budget can be exhausted as a spline-error
    interval moves across the grid. Tighter integral refinement adds neighboring
    knots sooner; this keeps all direct-EOS and physical acceptance gates intact.
    All retries run inside the caller's single hard wall-clock deadline.
    """
    from RMFsolver import phase_velocity
    requested = float(kwargs['path_rtol'])
    attempts = []
    original_seeds = phase_velocity._PNM_BRANCH_SEEDS
    extension = ()
    try:
        for factor in (1., .5, .2, .1):
            result = phase_velocity.solve_front_adiabatic_slow(**dict(kwargs, path_rtol=requested * factor))
            attempts.append(dict(path_rtol=requested * factor, status=result['status'],
                                 message=result.get('message', ''), upstream_seed_extension=extension))
            if (not extension and kwargs.get('NM_type', 'PNM') == 'PNM' and
                    'no two physical seeds agreed' in result.get('message', '')):
                # The established low-field seed can be the only converged
                # start at a cold density. Add two nearby starts, retaining
                # the EOS's strict two-start agreement and residual checks.
                # This runs in a disposable child; restore the module setting
                # even for direct calls or exceptions. No EOS code is changed.
                extension = ((35., 25., -3.), (40., 30., -3.))
                phase_velocity._PNM_BRANCH_SEEDS = original_seeds + extension
                result = phase_velocity.solve_front_adiabatic_slow(**dict(kwargs, path_rtol=requested * factor))
                attempts.append(dict(path_rtol=requested * factor, status=result['status'],
                                     message=result.get('message', ''), upstream_seed_extension=extension))
            if result.get('message') != 'Slow-front EOS/diffusion table did not converge':
                break
    finally:
        phase_velocity._PNM_BRANCH_SEEDS = original_seeds
    result['path_refinement_attempts'] = attempts
    result['upstream_seed_extension'] = extension
    return result


def solver_kwargs(task, candidate, profile, param):
    p = task['point']
    kwargs = dict(T_0minus=float(p['T_0minus_MeV']), nB_0minus=float(p['nB_0minus_MeV3']),
                  B_one_forth=float(task['B14_MeV']), T_0plus=float(task['T_0plus_MeV']),
                  ms=float(task['ms_MeV']), param=param, NM_type=task['nm_type'],
                  tail_eps=float(task['tail_eps']), n_mesh=int(task['n_mesh']),
                  tol_bvp=float(task['tol_bvp']), max_nodes=int(task['max_nodes']),
                  path_rtol=float(task['path_rtol']), kappa_factor=float(task['kappa_factor']),
                  jB_guess=float(candidate['jB_guess']), return_profile=True,
                  return_continuation=True, verb=False)
    if candidate.get('use_profile') and profile is not None:
        state = profile.get('continuation_state')
        if not isinstance(state, dict) or state.get('model') != 'slow_front_v1':
            raise ValueError('Only a new slow-front continuation state can seed a profile')
        # The solver gives state['jB'] precedence over jB_guess. Preserve the
        # normalized shapes while honoring the extrapolated/multiplier flux.
        kwargs['continuation_guess'] = dict(state, jB=kwargs['jB_guess'])
    return kwargs


def extract_profile(result):
    profile = {k: np.asarray(result[k], dtype=float).copy()
               for k in PROFILE_FIELDS if k in result}
    profile['continuation_state'] = copy.deepcopy(result['continuation_state'])
    return profile


def validation_error(result, task):
    if not result.get('success', False) or result.get('status') != 'converged':
        return str(result.get('message') or 'solver did not converge')
    if result.get('hydrodynamic_model') != MODEL or result.get('closure') != 'adiabatic':
        return 'wrong hydrodynamic model or closure'
    for key in ('jB', 'u_0minus'):
        if not np.isfinite(result.get(key, np.nan)) or result[key] <= 0:
            return 'invalid ' + key
    if result.get('bvp_status') != 0:
        return 'BVP collocation did not converge'
    state = result.get('continuation_state')
    if not isinstance(state, dict) or state.get('model') != 'slow_front_v1':
        return 'missing slow-front continuation state'
    if state.get('closure') != 'adiabatic' or state.get('ms') != task['ms_MeV']:
        return 'incompatible slow-front continuation state'
    limits = dict(closure_error_max=1e-8, bvp_max_rms_residual=float(task['tol_bvp']))
    limits.update({key: max(5 * float(task['tol_bvp']), 1e-7) for key in
                   ('constitutive_residual_norm', 'reaction_residual_norm',
                    'diffusion_integral_residual_norm')})
    for key, limit in limits.items():
        value = result.get(key, np.nan)
        if not np.isfinite(value) or abs(value) > limit:
            return f'{key}={value!r} exceeds {limit:g}'
    conservation = result.get('conservation_errors', {})
    for key in ('baryon_flux', 'pressure', 'enthalpy_flux'):
        value = conservation.get(key, np.nan)
        if not np.isfinite(value) or abs(value) > 1e-8:
            return 'invalid conservation error: ' + key
    residual = np.asarray(result.get('boundary_residuals', []))
    if residual.shape != (3,) or not np.all(np.isfinite(residual)) or np.max(abs(residual)) > 1e-7:
        return 'invalid boundary residuals'
    if not np.isclose(result.get('T_0plus', np.nan), task['T_0plus_MeV'], rtol=0., atol=1e-10):
        return 'wrong interface temperature'
    return None


def smoke_manifest(manifest):
    """A rectangular six-point scan; retains exact source indices and fluxes."""
    selected = copy.deepcopy(manifest)
    theta = (0, 15, 29)
    radial = (0, 8)
    selected['points'] = []
    for p in manifest['points']:
        if p['theta_index'] in theta and p['radial_index'] in radial:
            q = copy.deepcopy(p)
            q.update(point_index=len(selected['points']), trajectory_index=len(selected['points']),
                     theta_index=theta.index(p['theta_index']), radial_index=radial.index(p['radial_index']))
            selected['points'].append(q)
    return selected


def run_identity(manifest, args):
    """Hash physics/code and inputs before allowing any checkpoint to resume."""
    files = [REPO_ROOT / 'RMFsolver' / name for name in
             ('phase_velocity.py', 'SQMsolver.py', 'Solver.py', 'RMFparameter.py', 'constants.py')]
    files += [SCRIPT_DIR / name for name in ('_slow_front_adapter.py', '_method_engine.py',
              '_radial_shell_continuation.py', '_T0plus0p1_runner.py',
              'run_T0plus0p1_numerical.py', 'prepare_T0plus0p1_domain.py')]
    return dict(schema='adiabatic_slow_cluster_v1', model=MODEL, closure='adiabatic',
                prepared_domain_sha256=sha256(SCRIPT_DIR / args.domain_file),
                source_sha256=manifest['reference_source_sha256'],
                legacy_domain_sha256=manifest['legacy_domain_sha256'],
                code_sha256={str(p.relative_to(REPO_ROOT)): sha256(p) for p in files},
                calibration=manifest['calibration'], ms=manifest['ms_MeV'], NM_type=manifest['NM_type'],
                T_0plus=.1, settings={k: getattr(args, k) for k in
                    ('tail_eps', 'n_mesh', 'tol_bvp', 'max_nodes', 'path_rtol', 'kappa_factor',
                     'reference_jB_scale', 'max_bridge_depth', 'min_bridge_fraction_step')},
                points=[(p['point_index'], p['theta_index'], p['radial_index'],
                         p['T_0minus_MeV'], p['nB_0minus_over_n0'], p['relativistic_jB_guess'],
                         p['states'], p['radial_fraction_used'])
                        for p in manifest['points']])


def validate_resume(payload, identity):
    if payload.get('run_identity') != identity:
        raise ValueError('Checkpoint model, inputs, solver code, or settings differ; use --restart for a new output')


def payload_metadata(manifest, records, identity, run_complete):
    successes = [r for r in records.values() if r.get('success')]
    all_success = len(successes) == len(manifest['points'])
    theta_indices = {p['theta_index'] for p in manifest['points']}
    bracketed = []
    for theta in sorted(theta_indices):
        velocities = [r['u_0minus'] * 299792458. for r in successes if r['theta_index'] == theta]
        if velocities and min(velocities) <= 5000. <= max(velocities):
            bracketed.append(theta)
    return dict(run_identity=identity, hydrodynamic_model=MODEL, closure='adiabatic',
                solver='solve_front_adiabatic_slow', alpha_s=.4,
                rate_model='exact_nonleptonic', rate_approximation='madsen_muB_over_3',
                reference_source=manifest['reference_source'],
                reference_source_sha256=manifest['reference_source_sha256'],
                reference_grid_is_physical_boundary=False,
                numerical_seed_records=[dict(point_index=p['point_index'],
                    source_grid_index=p['source_grid_index'], jB_guess=p['relativistic_jB_guess'])
                    for p in manifest['points']],
                scheduling_finished=bool(run_complete), all_targets_successful=all_success,
                bracketed_5000_theta_indices=bracketed,
                complete_5000_contour_bracket=all_success and len(bracketed) == len(theta_indices))
