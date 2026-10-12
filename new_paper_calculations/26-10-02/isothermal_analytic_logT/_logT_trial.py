"""Bounded analytical workers and direct-density static boundary recovery."""
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import warnings
import numpy as np
from _logT_common import C, PHYSICS


def density_root(function, seed, limits, step=.05):
    from scipy.optimize import brentq
    lo,hi = limits
    seed = float(np.clip(seed,lo,hi))
    cache = {}
    def evaluate(density):
        if density not in cache:
            cache[density] = float(function(density))
        return cache[density]
    for index in range(9):
        radius=step*2**index
        for d in (seed,max(lo,seed-radius),min(hi,seed+radius)):
            evaluate(d)
        rows=sorted(cache.items())
        for d,v in rows:
            if abs(v)<1.e-10:
                return float(d)
        for (a,fa),(b,fb) in zip(rows[:-1],rows[1:]):
            if fa*fb<0:
                return float(brentq(evaluate,a,b,xtol=2.e-10,rtol=1.e-12))
        if seed-radius<=lo and seed+radius>=hi:
            break
    raise RuntimeError('No physical density bracket')


def boundary(T, seeds, n0):
    from RMFsolver import phase_velocity as pv
    cache={}
    def residual(density,which):
        row=cache.setdefault(float(density),{})
        if 'upstream' not in row:
            row['upstream']=pv._validated_pnm_state_from_nB(density*n0,T)
        if which not in row:
            upstream=row['upstream']
            if which=='lower':
                muB=pv._solve_muB_inf_at_muK0_for_given_Pi(
                    upstream['P_0minus'],0.,PHYSICS['B_one_forth'],T,ms=PHYSICS['ms'],upB=PHYSICS['upB'])
                row[which]=muB-upstream['muB_0minus']
            else:
                a=1.-1.e-9
                muB,muK=pv._solve_interface_0plus_from_local_a_and_Pi(
                    a,upstream['P_0minus'],0.,PHYSICS['B_one_forth'],T,ms=PHYSICS['ms'],upB=PHYSICS['upB'])
                row[which]=muB+a*muK-upstream['muB_0minus']
        return row[which]
    # The physical window may cut off one or both static boundaries at warm T.
    if residual(1.,'upper')<=0:
        return dict(success=True,status='outside_window',T_0minus=T,reason='composition_ceiling_over_entire_window')
    if residual(5.5,'lower')>=0:
        return dict(success=True,status='outside_window',T_0minus=T,reason='stable_neutron_matter_over_entire_window')
    lower_clipped=residual(1.,'lower')<=0
    upper_clipped=residual(5.5,'upper')>=0
    lo=1. if lower_clipped else density_root(lambda d:residual(d,'lower'),seeds['lower'],(1.,5.5))
    hi=5.5 if upper_clipped else density_root(lambda d:residual(d,'upper'),seeds['upper'],(1.,5.5))
    errors={k:residual(d,k) for k,d,clipped in (('lower',lo,lower_clipped),('upper',hi,upper_clipped)) if not clipped}
    if lo>=hi or any(abs(v)>1.e-5 for v in errors.values()):
        raise RuntimeError('Static boundary validation failed')
    return dict(success=True,status='success',T_0minus=T,lower=lo,upper=hi,
                lower_clipped=lower_clipped,upper_clipped=upper_clipped,
                residuals_MeV=errors,density_evaluations=len(cache),upper_composition=1.-1.e-9)


def valid_analytical_result(r):
    try:
        u=float(r['u_0minus'])
        scale=max(u*u,abs(r['u_0minus_formula_squared']),1.e-30)
        flux_scale=max(abs(r['Pi']),1.)
        return bool(r['success'] and r['front_exists'] and r['status']=='moving_front'
            and r['a_0plus_source']=='maximum' and r['a_0plus_max_status']=='interior'
            and 0<r['a_0plus']<1 and np.isfinite(u) and u>0
            and np.isfinite(r['jB']) and r['jB']>0 and r['slow_front_consistent']
            and np.isfinite(r['u_0minus_formula_squared']) and np.isfinite(r['Pi'])
            and np.isfinite(r['momentum_flux_ratio']) and 0<=r['momentum_flux_ratio']<=1.e-3
            and abs(r['analytic_velocity_residual'])<=1.e-8*scale
            and abs(r['composition_residual'])<=1.e-8
            and abs(r['momentum_flux_inf_residual'])<=1.e-8*flux_scale
            and abs(r['momentum_flux_0plus_residual'])<=1.e-8*flux_scale)
    except (KeyError,ValueError,TypeError):
        return False


def analytical(point):
    from RMFsolver import phase_velocity as pv
    T,nB=point['T_0minus'],point['nB_0minus']
    if not (.01<=T<=120. and 1<=point['nB_0minus_over_n0']<=5.5):
        raise ValueError('Analytical trial outside physical window')
    result=pv.analytic_velocity_isothermal(T,nB,PHYSICS['B_one_forth'],a_0plus=np.nan,
        xi=PHYSICS['xi'],ms=PHYSICS['ms'],NM_type=PHYSICS['NM_type'],upB=PHYSICS['upB'])
    valid=valid_analytical_result(result)
    result['validated']=valid
    result['velocity_m_s']=float(C*result['u_0minus']/np.hypot(1.,result['u_0minus'])) if valid else np.nan
    if result.get('status') in ('stable_neutron_matter','isothermal_coexistence') and result.get('success'):
        result['velocity_m_s']=0.  # Classified zero used only to bracket, never a positive target.
    if result.get('status')=='moving_front' and not valid:
        result.update(success=False,status='validation_failed')
    return result


def _child(connection, deadline, kind, kwargs):
    signal.signal(signal.SIGALRM,signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    for parent in Path(__file__).resolve().parents:
        if (parent/'RMFsolver').is_dir():
            sys.path.insert(0,str(parent))
            break
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result={'boundary':boundary,'analytical':analytical}[kind](**kwargs)
            except Exception as exc:
                result=dict(success=False,status='exception',message=f'{type(exc).__name__}: {exc}')
            result['warning_count']=len(caught)
            result['warnings']=list(dict.fromkeys(str(w.message) for w in caught))[:8]
        connection.send(result)
    finally:
        connection.close()


def bounded_call(kind, *, timeout, **kwargs):
    if not math.isfinite(timeout) or timeout<=0:
        raise ValueError('A positive finite timeout is required')
    start=time.monotonic()
    deadline=start+timeout
    context=mp.get_context('spawn')
    receive,send=context.Pipe(duplex=False)
    process=context.Process(target=_child,args=(send,deadline,kind,kwargs))
    started=False
    result=dict(success=False,status='child_error',message='Worker exited without a complete result')
    try:
        process.start()
        started=True
        send.close()
        if receive.poll(max(0.,deadline-time.monotonic())):
            try:
                result=receive.recv()
            except (EOFError,OSError):
                pass
        else:
            result=dict(success=False,status='trial_timeout',message=f'Hard limit: {timeout:g} s')
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
            if process.exitcode==-signal.SIGALRM:
                result.update(success=False,status='trial_timeout',message=f'Child limit: {timeout:g} s')
            if not process.is_alive():
                process.close()
    result['runtime_s']=time.monotonic()-start
    return result
