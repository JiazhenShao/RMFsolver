"""Disposable, deadline-bounded workers. No chemical-potential scans."""
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import warnings

import numpy as np
from _logT_common import C, PHYSICS, current_bounds


def density_root(function, seed, limits, *, step=.05, xtol=2.e-10):
    """Expand a density bracket around a saved boundary, caching EOS probes."""
    from scipy.optimize import brentq
    lower, upper = limits
    seed = float(np.clip(seed, lower, upper))
    values = {}
    def evaluate(density):
        density = float(density)
        if density not in values:
            try:
                values[density] = float(function(density))
            except (RuntimeError, ValueError, FloatingPointError):
                values[density] = np.nan
        return values[density]
    for index in range(9):
        radius = step*2**index
        for density in (seed, max(lower, seed-radius), min(upper, seed+radius)):
            evaluate(density)
        ordered = sorted((d,v) for d,v in values.items() if np.isfinite(v))
        for d,v in ordered:
            if v == 0:
                return d
        for (a,fa),(b,fb) in zip(ordered[:-1], ordered[1:]):
            if fa*fb < 0:
                try:
                    return float(brentq(evaluate, a, b, xtol=xtol, rtol=1.e-12))
                except ValueError:
                    continue
        if seed-radius <= lower and seed+radius >= upper:
            break
    raise RuntimeError("No valid density bracket in the physical window")


def boundary(T, seeds, n0):
    from RMFsolver import phase_velocity as pv
    from RMFsolver import RMFparameter as para
    cache = {}
    def states(density):
        if density not in cache:
            upstream = pv._validated_pnm_state_from_nB(density*n0, T, param=para.paraQMCRMF3)
            cache[density] = dict(upstream=upstream)
        return cache[density]
    def residual(density, which):
        row = states(density)
        if which not in row:
            upstream = row['upstream']
            if which == 'lower':
                muB = pv._solve_muB_inf_at_muK0_for_given_Pi(
                    upstream['P_0minus'], 0., PHYSICS['B_one_forth'], T,
                    ms=PHYSICS['ms'], upB=PHYSICS['upB'])
                row[which] = muB-upstream['muB_0minus']
            else:
                a = 1.-1.e-9  # Established static upper-boundary convention.
                interface = pv._solve_interface_0plus_from_local_a_and_Pi(
                    a, upstream['P_0minus'], 0., PHYSICS['B_one_forth'], T,
                    ms=PHYSICS['ms'], upB=PHYSICS['upB'])
                muB_0plus, muK_0plus = interface
                row[which] = muB_0plus+a*muK_0plus-upstream['muB_0minus']
        return row[which]
    roots = {name: density_root(lambda d:residual(d,name), seeds[name], (1.,5.5))
             for name in ('lower','upper')}
    errors = {name: float(residual(d,name)) for name,d in roots.items()}
    if roots['lower'] >= roots['upper'] or max(abs(v) for v in errors.values()) > 1.e-5:
        raise RuntimeError(f"Boundary validation failed: {roots}, {errors}")
    return dict(success=True, status='success', T_0minus=T, **roots, residuals_MeV=errors,
                density_evaluations=len(cache), upper_composition=1.-1.e-9)


def proposal(T, level, limits, n0):
    from RMFsolver import phase_velocity as pv
    def residual(density):
        result = pv.analytic_velocity_isothermal(T, density*n0, PHYSICS['B_one_forth'],
            xi=PHYSICS['xi'], ms=PHYSICS['ms'], NM_type=PHYSICS['NM_type'], upB=PHYSICS['upB'])
        if not result['success'] or not result['front_exists']:
            raise RuntimeError('Analytical proposal is not an admissible moving state')
        u = result['u_0minus']
        return C*u/np.hypot(1.,u)-level
    lower,upper = limits
    # Probe distances logarithmically from coexistence; never enter either boundary.
    distances = np.geomspace(max(1.e-8,(upper-lower)*1.e-8), (upper-lower)*.999, 18)
    previous = None
    from scipy.optimize import brentq
    for distance in distances:
        density = lower+distance
        try:
            value = residual(density)
        except (ValueError,RuntimeError):
            continue
        if previous and previous[1]*value <= 0:
            root = brentq(residual, previous[0], density, xtol=2.e-9, rtol=1.e-10)
            return dict(success=True, status='success', density=float(root), source='analytical_proposal_only')
        previous = (density,value)
    raise RuntimeError('Analytical speed did not bracket this target')


def valid_result(result, controls):
    try:
        residual = np.asarray(result['boundary_residuals'], float)
        return bool(result['success'] and result['bvp_status'] == 0
            and result['a_0plus_source'] == 'maximum' and result['a_0plus_max_status'] == 'interior'
            and 0 < result['a_0plus'] < 1
            and abs(result['a_0plus_derived']-result['a_0plus']) <= 1.e-7
            and result['rate_model'] == 'exact_nonleptonic'
            and result['diffusion_model'] == 'local_muB_fixed_T'
            and result['composition_definition'] == 'nK_over_local_nB'
            and result['current_definition'] == 'u_nK_minus_D_K_dnK_dx'
            and np.isfinite(result['jB']) and result['jB'] > 0
            and np.isfinite(result['u_0minus']) and result['u_0minus'] > 0
            and residual.size == 3 and np.all(np.isfinite(residual))
            and max(abs(residual)) <= controls['tol_bvp']
            and np.isfinite(result['bvp_max_rms_residual'])
            and result['bvp_max_rms_residual'] <= controls['tol_bvp']
            and np.isfinite(result['compact_scale']) and result['compact_scale'] > 0
            and np.isfinite(result['tail_residual_norm']) and np.isfinite(result['momentum_flux_ratio']))
    except (KeyError,TypeError,ValueError):
        return False


def numerical(point, controls, continuation, jB_guess, bounds, use_analytic=False, expand=False):
    from RMFsolver import phase_velocity as pv
    T, nB = point['T_0minus'], point['nB_0minus']
    if not (.01 <= T <= 20 and 1 <= point['nB_0minus_over_n0'] <= 5.5):
        raise ValueError('Numerical trial outside the requested cold physical window')
    if use_analytic:
        analytic = pv.analytic_velocity_isothermal(T, nB, PHYSICS['B_one_forth'],
            xi=PHYSICS['xi'], ms=PHYSICS['ms'], NM_type=PHYSICS['NM_type'], upB=PHYSICS['upB'])
        if not analytic['success'] or not analytic['front_exists']:
            return dict(success=False, status='outside_moving_domain', analytic_status=analytic.get('status'))
        analytic_jB = analytic.get('jB', nB*analytic['u_0minus'])
        # Existing bounds already encode all available continued estimates.
        factor = 16. if expand else 4.
        bounds = (min(bounds[0],analytic_jB/factor), max(bounds[1],analytic_jB*factor))
        jB_guess = analytic_jB
    result = pv.solve_front_isothermal(T, nB, PHYSICS['B_one_forth'],
        ms=PHYSICS['ms'], NM_type=PHYSICS['NM_type'],
        **{k:controls[k] for k in ('tail_eps','n_mesh','tol_bvp','max_nodes','kappa_factor','initial_guess_mode')},
        jB_guess=jB_guess, jB_bounds=bounds, continuation_guess=continuation,
        return_continuation=True, return_profile=False)
    valid = valid_result(result, controls)
    if valid:
        errors = []
        for suffix in ('0plus','inf'):
            state = pv._quark_thermo_state(result[f'muB_{suffix}'], result[f'muK_{suffix}'],
                PHYSICS['B_one_forth'], T, result['jB'], ms=PHYSICS['ms'], upB=PHYSICS['upB'])
            errors.extend((abs(state['Pi']/result['Pi']-1), abs(state['nB']*state['u']/result['jB']-1)))
        result['endpoint_flux_residual_max'] = max(errors)
        valid = bool(np.all(np.isfinite(errors)) and max(errors) <= 1.e-7)
    result.update(success=valid, validated=valid, status='success' if valid else 'validation_failed',
                  current_bounds=bounds)
    u = result.get('u_0minus', np.nan)
    result['velocity_m_s'] = C*u/np.hypot(1.,u) if valid else np.nan
    return result


def _child(connection, deadline, kind, kwargs):
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    for parent in Path(__file__).resolve().parents:
        if (parent/'RMFsolver').is_dir():
            sys.path.insert(0,str(parent))
            break
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result = {'boundary':boundary,'proposal':proposal,'numerical':numerical}[kind](**kwargs)
            except Exception as exc:
                result = dict(success=False,status='exception',message=f'{type(exc).__name__}: {exc}')
            result['warning_count'] = len(caught)
            result['warnings'] = list(dict.fromkeys(str(w.message) for w in caught))[:8]
        connection.send(result)
    finally:
        connection.close()


def bounded_call(kind, *, timeout, **kwargs):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('A positive finite trial timeout is required')
    start = time.monotonic()
    deadline = start+timeout
    context = mp.get_context('spawn')
    receive,send = context.Pipe(duplex=False)
    process = context.Process(target=_child,args=(send,deadline,kind,kwargs))
    started = False
    result = dict(success=False,status='child_error',message='Worker exited without a complete result')
    try:
        process.start()
        started = True
        send.close()
        if receive.poll(max(0.,deadline-time.monotonic())):
            try:
                result = receive.recv()
            except (EOFError,OSError):
                pass
        else:
            result = dict(success=False,status='trial_timeout',message=f'Hard trial limit: {timeout:g} s')
    finally:
        receive.close()
        send.close()
        if started:
            process.join(timeout=.05)
            if process.is_alive():
                process.terminate()
                process.join(timeout=.1)
            if process.is_alive():
                process.kill()
                process.join(timeout=.1)
            if process.exitcode == -signal.SIGALRM:
                result.update(success=False,status='trial_timeout',message=f'Child trial limit: {timeout:g} s')
            if not process.is_alive():
                process.close()
    result['runtime_s'] = time.monotonic()-start
    return result
