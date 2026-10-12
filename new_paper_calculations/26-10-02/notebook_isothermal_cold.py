"""Small full-numerical cold comparison, independent of analytical checkpoints.

Analytical roots and old profiles are seeds only. Every added plotted dot must
be a new converged numerical contour root. Importing this module runs no solver.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import threading
import time
import warnings
import numpy as np
import notebook_isothermal as analytic
from _logT_common import atomic_save, sha256, C

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TEMPERATURES=(.01,.027550997842357025,.09779327685429287)
LEVELS=(500.,2000.)


def load_inputs(payload,*,temperatures=TEMPERATURES,levels=LEVELS,root=ROOT):
    root=Path(root)
    static=analytic.load_boundary_inputs(root)
    if payload['domain_fingerprint']!=static['fingerprint']:
        raise ValueError('Analytical seed and boundary domains differ')
    physics={k:static['physics'][k] for k in ('B_one_forth','ms','NM_type','upB')}
    if any(payload['identity']['physics'][k]!=v for k,v in physics.items()):
        raise ValueError('Analytical seed physics mismatch')
    boundaries={};seeds={}
    for T in map(float,temperatures):
        matches=[b for b in static['boundary_cache'].values() if np.isclose(b['T_0minus'],T,rtol=1.e-12,atol=0)]
        if len(matches)!=1 or matches[0]['status']!='success':
            raise ValueError(f'Choose a temperature with directly solved saved boundaries: {T:g}')
        boundaries[T]=copy.deepcopy(matches[0])
        for level in levels:
            matches=[r for r in payload['targets'].values() if r['level']==level and
                     np.isclose(r['T_0minus'],T,rtol=1.e-12,atol=0) and r['status']=='success'
                     and r.get('root',{}).get('result',{}).get('validated')]
            if matches:
                seeds[float(level),T]=float(matches[0]['root']['nB_0minus_over_n0'])
    folder=root/'new_paper_calculations/26-10-01/isothermal_numeric_logT'
    manifest_path=folder/'inputs/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['fingerprint']!=static['fingerprint'] or any(manifest['physics'][k]!=v for k,v in physics.items()):
        raise ValueError('Historical seed physics/domain mismatch')
    hashes=dict(static['source_hashes']);hashes[str(manifest_path.relative_to(root))]=sha256(manifest_path)
    profiles=[]
    for name in ('core_mesh_T0.01_n3.001.npy','core_mesh_T0.01_n3.01.npy','core_mesh_T0.01_n3.05.npy'):
        relative='inputs/profiles/'+name;path=folder/relative
        if sha256(path)!=manifest['files'][relative]['sha256']:
            raise ValueError(f'Seed profile hash mismatch: {name}')
        state=np.load(path,allow_pickle=True).item()
        profiles.append(dict(state,**state['point'],lower=static['boundary_cache'][float(.01).hex()]['lower']))
        hashes[str(path.relative_to(root))]=sha256(path)
    return dict(n0=static['n0'],physics=physics,fingerprint=static['fingerprint'],
                boundaries=boundaries,seeds=seeds,profiles=profiles,source_hashes=hashes)


def nearby_profile(profiles,physical,lower):
    """Never let a distant historical profile override the cold initializer."""
    distance=physical['nB_0minus_over_n0']-lower
    choices=[]
    for row in profiles:
        delta=row['nB_0minus_over_n0']-row['lower']
        if delta<=0 or distance<=0: continue
        t_ratio=physical['T_0minus']/row['T_0minus']
        d_ratio=distance/delta
        if 1/1.5<=t_ratio<=1.5 and .5<=d_ratio<=2:
            choices.append((abs(np.log(t_ratio))+abs(np.log(d_ratio)),row))
    return min(choices,key=lambda item:item[0])[1] if choices else None


def correct_density(samples,level,limits):
    """Small offset-scaled correction; actual numerical brackets take precedence."""
    lo,hi=limits
    rows=sorted((p for p in samples if p['velocity_m_s']>0),key=lambda p:p['nB_0minus_over_n0'])
    if any((a['velocity_m_s']-level)*(b['velocity_m_s']-level)<0 for a,b in zip(rows[:-1],rows[1:])):
        return analytic.propose_density(rows,level,limits)
    nearest=min(rows,key=lambda p:abs(np.log(p['velocity_m_s']/level)))
    density=lo+(nearest['nB_0minus_over_n0']-lo)*np.clip(level/nearest['velocity_m_s'],.5,2.)
    return float(min(density,.5*(hi+nearest['nB_0minus_over_n0'])))


def valid_result(result,tolerance):
    try:
        bc=np.asarray(result['boundary_residuals'],float)
        return bool(result['success'] and result['bvp_status']==0 and
            result['a_0plus_source']=='maximum' and result['a_0plus_max_status']=='interior' and
            0<result['a_0plus']<1 and abs(result['a_0plus_derived']-result['a_0plus'])<=1.e-7 and
            result['rate_model']=='exact_nonleptonic' and result['diffusion_model']=='local_muB_fixed_T' and
            result['composition_definition']=='nK_over_local_nB' and
            result['current_definition']=='u_nK_minus_D_K_dnK_dx' and
            np.isfinite(result['jB']) and result['jB']>0 and np.isfinite(result['u_0minus']) and result['u_0minus']>0 and
            bc.shape==(3,) and np.all(np.isfinite(bc)) and max(abs(bc))<=tolerance and
            np.isfinite(result['bvp_max_rms_residual']) and result['bvp_max_rms_residual']<=tolerance and
            np.isfinite(result['tail_residual_norm']) and abs(result['tail_residual_norm'])<=tolerance and
            0<=result['momentum_flux_ratio']<=1.e-3 and result['profile_physical'] and
            0<=result['endpoint_flux_residual_max']<=1.e-7 and 0<=result['profile_eos_residual_max']<=1.e-7)
    except (KeyError,TypeError,ValueError):
        return False


def evaluate_point(physical,physics,controls,continuation,jB_guess,bounds):
    from RMFsolver import phase_velocity as pv
    T=physical['T_0minus']
    if not .01<=T<20 or not 1<=physical['nB_0minus_over_n0']<=5.5:
        raise ValueError('Cold numerical trial outside the physical window')
    with warnings.catch_warnings(record=True) as caught:
        result=pv.solve_front_isothermal(T,physical['nB_0minus'],physics['B_one_forth'],
            ms=physics['ms'],NM_type=physics['NM_type'],a_0plus=np.nan,
            jB_guess=jB_guess,jB_bounds=bounds,continuation_guess=continuation,
            return_continuation=True,return_profile=False,initial_guess_mode='layer_resolved',
            **{k:controls[k] for k in ('n_mesh','tol_bvp','max_nodes','tail_eps')})
    result['warnings']=list(dict.fromkeys(str(w.message) for w in caught))[:8]
    result.update(validated=False,endpoint_flux_residual_max=np.inf,profile_eos_residual_max=np.inf,profile_physical=False)
    if result.get('success') and result.get('bvp_status')==0 and result.get('continuation_state'):
        state=result['continuation_state'];nK=np.asarray(state['nK']);jK=np.asarray(state['jK'])
        scale=max(abs(state['nK_0plus']),1.);flux=max(abs(state['jB']),1.e-30)
        result['profile_physical']=bool(np.all(np.isfinite(nK)) and np.all(np.isfinite(jK)) and
            np.min(nK)>=-1.e-7*scale and np.max(np.diff(nK))<=1.e-5*scale and np.min(jK)>=-1.e-7*flux)
        endpoint_errors=[]
        for suffix in ('0plus','inf'):
            thermo=pv._quark_thermo_state(result['muB_'+suffix],result['muK_'+suffix],physics['B_one_forth'],T,
                                          result['jB'],ms=physics['ms'],upB=physics['upB'])
            endpoint_errors.extend((abs(thermo['Pi']/result['Pi']-1),abs(thermo['nB']*thermo['u']/result['jB']-1)))
        result['endpoint_flux_residual_max']=max(endpoint_errors)
        errors=[]
        for index in np.unique(np.linspace(0,len(nK)-1,7,dtype=int)):
            thermo=pv._solve_local_quark_state_from_nK_T_and_Pi(float(nK[index]),T,result['Pi'],result['jB'],
                physics['B_one_forth'],ms=physics['ms'],upB=physics['upB'],initial_guess=(result['muB_0plus'],result['muK_0plus']))
            errors.extend((abs(thermo['Pi']/result['Pi']-1),abs(thermo['nK']-nK[index])/scale))
        result['profile_eos_residual_max']=max(errors)
    result['validated']=valid_result(result,controls['tol_bvp'])
    u=result.get('u_0minus',np.nan)
    result['velocity_m_s']=float(C*u/np.hypot(1.,u)) if result['validated'] else np.nan
    result['status']='success' if result['validated'] else 'validation_failed'
    return result


def new_points(payload,level):
    """Selection only: unvalidated, off-target, and failed attempts are not dots."""
    rows=[]
    for task in payload.get('targets',{}).values():
        root=task.get('root') or {}
        if (task['level']==level and task['status']=='success' and root.get('result',{}).get('validated') and
                abs(root['velocity_m_s']/level-1)<=.005):
            rows.append((root['nB_0minus_over_n0'],root['T_0minus']))
    return np.asarray(rows,float).reshape(-1,2)


def calculate(inputs,path,*,temperatures=TEMPERATURES,levels=LEVELS,workers=2,trial_timeout=90.,
              target_timeout=300.,max_evaluations=4,n_mesh=300,tol_bvp=1.e-4,tail_eps=1.e-3,max_nodes=12000,
              import_paths=None,evaluator=None,evaluation_saved=None,show_progress=True):
    from tqdm.auto import tqdm
    path=Path(path);temperatures=tuple(map(float,temperatures));levels=tuple(map(float,levels))
    if (not temperatures or not levels or len(set(temperatures))!=len(temperatures) or len(set(levels))!=len(levels)
        or not all(np.isfinite(t) and .01<=t<20 and t in inputs['boundaries'] for t in temperatures)
        or not all(np.isfinite(v) and 0<v<C for v in levels) or workers<1 or max_evaluations<1
        or min(trial_timeout,target_timeout,tol_bvp,tail_eps)<=0 or tol_bvp>1.e-4 or tail_eps>=1
        or n_mesh<5 or max_nodes<=n_mesh):
        raise ValueError('Invalid cold numerical schedule or controls')
    controls=dict(trial_timeout=float(trial_timeout),target_timeout=float(target_timeout),max_evaluations=int(max_evaluations),
                  n_mesh=int(n_mesh),tol_bvp=float(tol_bvp),tail_eps=float(tail_eps),max_nodes=int(max_nodes))
    files=[Path(__file__),HERE/'_cold_notebook_worker.py',ROOT/'RMFsolver/phase_velocity.py']
    identity=dict(physics=inputs['physics'],temperatures=temperatures,levels=levels,controls=controls,
        fingerprint=inputs['fingerprint'],input_hashes=inputs['source_hashes'],
        analytical_density_seeds={f'{v.hex()}:{t.hex()}':d for (v,t),d in inputs['seeds'].items()},
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in files})
    cancel=threading.Event();lock=threading.RLock()
    if evaluator is None:
        paths=analytic.worker_import_paths(import_paths)
        runtime=analytic.run_module_call_with_hard_timeout('_isothermal_notebook_worker','call',
            args=('preflight',),kwargs=dict(import_paths=paths),timeout_s=30.)
        identity['runtime_environment']=runtime
        def evaluator(physical,*,continuation,jB_guess,bounds,timeout):
            return analytic.run_module_call_with_hard_timeout('_cold_notebook_worker','call',
                args=(physical,inputs['physics'],controls,continuation,jB_guess,bounds,paths),timeout_s=timeout,cancel=cancel)
    payload=dict(schema=1,method='full_numerical_cold_notebook',identity=identity,domain_fingerprint=inputs['fingerprint'],
                 targets={},evaluations=[],points={},work_complete=False,contours_complete=False)
    if path.exists():
        payload=np.load(path,allow_pickle=True).item()
        if payload['identity']!=identity:
            raise ValueError('Cold checkpoint incompatible; keep it and choose a new output filename')
        for event in payload['evaluations']:
            if event['status']=='running': event.update(status='interrupted',runtime_s=event['reserved_seconds'])
    for T in temperatures:
        for level in levels:
            key=f'{level.hex()}:{T.hex()}'
            payload['targets'].setdefault(key,dict(id=key,level=level,T_0minus=T,status='pending',root=None))
    def save():
        with lock: atomic_save(path,payload)
    save()
    with tqdm(total=len(temperatures)*len(levels),desc='Cold numerical targets',unit='target',ncols=110,
              mininterval=0.,miniters=1,position=0,disable=not show_progress) as targets, \
         tqdm(desc='Saved numerical evaluations',unit='point',ncols=110,mininterval=0.,miniters=1,position=1,
              disable=not show_progress) as progress:
        targets.update(sum(r['status'] not in ('pending','running') for r in payload['targets'].values()))
        progress.update(sum(e['status'] not in ('running','interrupted') for e in payload['evaluations']))
        def chain(T):
            lo,hi=(inputs['boundaries'][T][k] for k in ('lower','upper'))
            for level in sorted(levels):
                if cancel.is_set(): return
                task=payload['targets'][f'{level.hex()}:{T.hex()}']
                if task['status'] not in ('pending','running'): continue
                task['status']='running'
                while not cancel.is_set():
                    with lock:
                        prior=[e for e in payload['evaluations'] if e['target_id']==task['id']]
                        samples=[p for p in payload['points'].values() if p['T_0minus']==T]
                    accepted=next((p for p in samples if abs(p['velocity_m_s']/level-1)<=.005),None)
                    if accepted:
                        task.update(status='success',root=copy.deepcopy(accepted));break
                    remaining=target_timeout-sum(e.get('runtime_s',e['reserved_seconds']) for e in prior)
                    if len(prior)>=max_evaluations or remaining<=0: break
                    if not prior and (level,T) in inputs['seeds']:
                        density=inputs['seeds'][level,T]
                    elif samples:
                        density=correct_density(samples,level,(lo,hi))
                    else:
                        density=inputs['seeds'].get((level,T),lo+1.e-4*(hi-lo))
                    # A spent interrupted/failed reservation cannot silently be repeated.
                    failed=[e['point']['nB_0minus_over_n0'] for e in prior if e['status']!='success']
                    for _ in range(20):
                        if not any(abs(density-d)<1.e-12 for d in failed): break
                        density=lo+.9*(density-lo)
                    if not lo<density<hi: break
                    physical=dict(T_0minus=T,nB_0minus_over_n0=float(density),nB_0minus=float(density*inputs['n0']))
                    with lock:
                        profiles=list(inputs['profiles'])+[dict(p['result']['continuation_state'],**p,lower=lo)
                            for p in payload['points'].values() if p['T_0minus']==T and 'continuation_state' in p['result']]
                    profile=nearby_profile(profiles,physical,lo)
                    # One failed continuation at this target switches to a fresh layer seed.
                    if any(e['status']!='success' and e.get('continuation_used') for e in prior): profile=None
                    u=level/np.sqrt(C*C-level*level);current=physical['nB_0minus']*u
                    estimates=[current]+([float(profile['jB'])] if profile is not None else [])
                    bounds=(min(estimates)/4,max(estimates)*4)
                    with lock:
                        event=dict(id=len(payload['evaluations']),target_id=task['id'],point=physical,status='running',
                            reserved_seconds=min(trial_timeout,remaining),continuation_used=profile is not None,
                            jB_guess=current,current_bounds=bounds)
                        payload['evaluations'].append(event);save()
                    started=time.monotonic()
                    try:
                        answer=evaluator(physical,continuation=profile,jB_guess=current,bounds=bounds,timeout=event['reserved_seconds'])
                    except Exception as exc:
                        answer=dict(success=False,validated=False,status='trial_timeout' if isinstance(exc,TimeoutError) else 'exception',
                                    message=f'Full numerical BVP exceeded {event["reserved_seconds"]:g} s' if isinstance(exc,TimeoutError)
                                    else f'{type(exc).__name__}: {exc}')
                    with lock:
                        valid=valid_result(answer,tol_bvp) and np.isfinite(answer.get('velocity_m_s',np.nan))
                        answer['validated']=bool(valid)
                        event.update(status='success' if valid else answer.get('status','validation_failed'),
                                     result=answer,runtime_s=time.monotonic()-started)
                        if valid:
                            p=dict(physical,velocity_m_s=float(answer['velocity_m_s']),result=answer,
                                   source='new_full_numerical_notebook')
                            payload['points'][str(event['id'])]=p
                        save()
                        if evaluation_saved: evaluation_saved(copy.deepcopy(event))
                        progress.update(1)
                    if not valid and profile is None: break
                if cancel.is_set(): return
                if task['status']=='running':
                    task.update(status='unresolved',message='No converged contour root within the bounded allowance')
                with lock:
                    save();targets.update(1)
        pool=ThreadPoolExecutor(max_workers=workers)
        try:
            for future in as_completed([pool.submit(chain,T) for T in temperatures]): future.result()
        except BaseException:
            cancel.set();raise
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
    payload['work_complete']=all(t['status'] not in ('pending','running') for t in payload['targets'].values())
    payload['contours_complete']=all(t['status']=='success' for t in payload['targets'].values())
    save()
    return payload
