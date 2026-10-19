"""Standalone core criterion, distinct from the downstream-tail comparison.

Solve 3*a(0+)*sqrt(D_K(0+)*gamma_K(0+)) = fraction with the static
maximum-composition interface closure. No moving velocity or BVP is solved.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import math
import os
from pathlib import Path
import sys
import time
import warnings
import numpy as np
from isothermal_mfp import atomic_save, transport_lengths, MEV_INVERSE_TO_M

ROOT = Path(__file__).resolve().parents[2]


def core_ratio(a_0plus, D_K, gamma_K):
    if not 0 < a_0plus < 1 or not all(math.isfinite(x) and x > 0 for x in (D_K,gamma_K)):
        raise ValueError('Core criterion requires 0 < a(0+) < 1 and positive finite transport')
    return 3*a_0plus*math.sqrt(D_K*gamma_K)


def temperature_root(evaluate, fraction, *, temperature_limits=(1.e-8,1.),
                     reference_temperature=.01, ratio_rtol=1.e-6, max_evaluations=16):
    """Bracket and correct in log T; cold power law only proposes a guess."""
    low, high = map(float,temperature_limits)
    if not (0 < low < high and low <= reference_temperature <= high and
            math.isfinite(fraction) and fraction > 0 and 0 < ratio_rtol <= 1.e-6 and max_evaluations >= 3):
        raise ValueError('Invalid core root controls')
    records=[]; cache={}; result=dict(status='running',evaluations=records)

    def sample(T):
        T=float(np.clip(T,low,high))
        if T in cache: return cache[T]
        if len(records) >= max_evaluations: raise RuntimeError('Evaluation budget exhausted')
        state=dict(T_0minus=T,status='running');records.append(state)
        try:
            state.update(evaluate(T))
            if state['status'] != 'interior': raise ValueError(f"Outside admissible interface domain: {state['status']}")
            if not math.isfinite(state['ratio']) or state['ratio'] <= 0: raise RuntimeError('Invalid core ratio')
            state['log_residual']=math.log(state['ratio']/fraction)
        except Exception as exc:
            state['message']=f'{type(exc).__name__}: {exc}'
            if state['status']=='running': state['status']='failed'
            raise
        cache[T]=state
        return state

    def finish(state):
        result.update(status='success', root=state.copy(), relative_ratio_residual=abs(state['ratio']/fraction-1))
        return result

    try:
        initial=sample(reference_temperature)
        if abs(initial['ratio']/fraction-1) <= ratio_rtol: return finish(initial)
        # Exact for the leading cold scaling at fixed a and muB, but never
        # accepted without reevaluating the nuclear + quark endpoint EOS.
        seed=float(np.clip(reference_temperature*(initial['ratio']/fraction)**(6/5),low,high))
        proposed=sample(seed)
        if abs(proposed['ratio']/fraction-1) <= ratio_rtol: return finish(proposed)
        sample(seed/2); sample(seed*2)
        while True:
            ordered=sorted(cache.values(), key=lambda r:r['T_0minus'])
            accepted=next((r for r in ordered if abs(r['ratio']/fraction-1) <= ratio_rtol),None)
            if accepted: return finish(accepted)
            bracket=next(((a,b) for a,b in zip(ordered[:-1],ordered[1:]) if a['log_residual']*b['log_residual'] < 0),None)
            if bracket:
                a,b=bracket
                weight=float(np.clip(-a['log_residual']/(b['log_residual']-a['log_residual']),.05,.95))
                T=math.exp(math.log(a['T_0minus'])+weight*math.log(b['T_0minus']/a['T_0minus']))
                sample(T)
                result['bracket']=(a['T_0minus'],b['T_0minus'])
            else:
                before=len(cache)
                sample(ordered[0]['T_0minus']/4)
                sample(ordered[-1]['T_0minus']*4)
                if len(cache)==before:
                    result.update(status='no_crossing_in_window',message='No sign-changing ratio bracket in the temperature window')
                    return result
    except ValueError as exc:
        result.update(status='outside_domain',message=str(exc))
    except Exception as exc:
        result.update(status='failed',message=f'{type(exc).__name__}: {exc}')
    return result


def solve_density(density, physics, fraction, controls, import_paths=()):
    """Runs inside one killable subprocess, retaining all root-search states."""
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
    sys.path[:0]=list(import_paths)
    from RMFsolver import phase_velocity as pv
    started=time.monotonic()

    def evaluate(T):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            upstream=pv._validated_pnm_state_from_nB(density*physics['n0'],T)
            ceiling=pv._solve_a_0plus_max(upstream['muB_0minus'],upstream['P_0minus'],T,
                                        physics['B_one_forth'],ms=physics['ms'],upB=physics['upB'])
            if ceiling['status'] != 'interior':
                return dict(status=ceiling['status'],ceiling=ceiling,upstream=upstream)
            a=ceiling['a_0plus_max'];mu=ceiling['muB_0plus'];muK=ceiling['muK_0plus']
            endpoint=pv._quark_thermo_state(mu,muK,physics['B_one_forth'],T,0.,ms=physics['ms'],upB=physics['upB'])
            pressure_residual=abs(endpoint['P']/upstream['P_0minus']-1)
            composition_residual=abs(endpoint['nK']/endpoint['nB']-a)
            if pressure_residual > 1.e-8 or composition_residual > 1.e-8 or abs(ceiling['gibbs_residual']) > 1.e-6:
                raise RuntimeError('Maximum-interface EOS validation failed')
            micro=pv._microphysics_from_quark_state_isothermal_baseline(mu,T)
            coefficients=transport_lengths(mu,T,controls['diffusion'])
            if not np.isclose(coefficients['gamma_K'],micro['gamma'],rtol=1.e-12,atol=0):
                raise RuntimeError('Weak-rate coefficient mismatch')
            if controls['diffusion']=='full' and not np.isclose(coefficients['D_K'],micro['D'],rtol=1.e-12,atol=0):
                raise RuntimeError('Diffusion coefficient mismatch')
            D=coefficients['D_K'];gamma=coefficients['gamma_K']
            ratio=core_ratio(a,D,gamma)
            return dict(status='interior',T_0minus=T,a_0plus=a,muB_0plus=mu,muK_0plus=muK,
                        D_K=D,gamma_K=gamma,ratio=ratio,lambda_mfp_m=3*D*MEV_INVERSE_TO_M,
                        L_core_m=math.sqrt(D/gamma)/a*MEV_INVERSE_TO_M,
                        eta_over_a_squared=micro['eta']/a**2,
                        pressure_relative_residual=pressure_residual,composition_residual=composition_residual,
                        gibbs_residual_MeV=ceiling['gibbs_residual'],upstream=upstream,
                        warnings=list(dict.fromkeys(str(w.message) for w in caught)))

    result=temperature_root(evaluate,fraction,temperature_limits=controls['temperature_limits'],
                            reference_temperature=controls['reference_temperature'],
                            ratio_rtol=controls['ratio_rtol'],max_evaluations=controls['max_evaluations'])
    result.update(nB_0minus_over_n0=float(density),runtime_s=time.monotonic()-started)
    return result


def calculate(densities, physics, path, *, fraction=1.e-4, diffusion='full',
              temperature_limits=(1.e-8,1.), reference_temperature=.01,
              ratio_rtol=1.e-6, max_evaluations=16, timeout=60., workers=2,
              import_paths=None, solver=None, progress=None, show_progress=True):
    """Checkpoint each completed density root, then advance its tqdm bar."""
    from tqdm.auto import tqdm
    path=Path(path);densities=tuple(map(float,densities))
    if (not densities or any(not 1 <= d <= 5.5 for d in densities) or any(b<=a for a,b in zip(densities[:-1],densities[1:]))
        or diffusion not in ('full','landau') or not math.isfinite(fraction) or fraction<=0 or workers<1 or timeout<=0
        or physics.get('NM_type')!='PNM' or physics.get('ms')!=0):
        raise ValueError('Invalid core-curve schedule or physics')
    controls=dict(temperature_limits=tuple(map(float,temperature_limits)),reference_temperature=reference_temperature,
                  ratio_rtol=ratio_rtol,max_evaluations=max_evaluations,diffusion=diffusion,timeout=timeout)
    sources=[Path(__file__),Path(__file__).with_name('isothermal_mfp.py')]+sorted((ROOT/'RMFsolver').rglob('*.py'))
    identity=dict(physics=physics,densities=densities,fraction=fraction,controls=controls,
                  sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    result=np.load(path,allow_pickle=True).item() if path.exists() else dict(identity=identity,rows={})
    if result['identity']!=identity: raise ValueError('Incompatible core-curve checkpoint; choose a new output file')
    if solver is None:
        helper=ROOT/'new_paper_calculations/26-10-02'
        if str(helper) not in sys.path: sys.path.insert(0,str(helper))
        import notebook_isothermal as runtime
        paths=runtime.worker_import_paths(import_paths)
        def solver(density):
            try:
                return runtime.run_module_call_with_hard_timeout('isothermal_core_mfp','solve_density',
                    args=(density,physics,fraction,controls,paths),timeout_s=timeout)
            except Exception as exc:
                return dict(nB_0minus_over_n0=density,status='failed',message=f'{type(exc).__name__}: {exc}',evaluations=[])
    pending=[d for d in densities if d.hex() not in result['rows']]
    with tqdm(total=len(densities),initial=len(result['rows']),desc='Core MFP boundary',unit='point',
              dynamic_ncols=True,disable=not show_progress) as bar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(solver,d):d for d in pending}
            for future in as_completed(futures):
                d=futures[future]
                try: row=future.result()
                except Exception as exc: row=dict(nB_0minus_over_n0=d,status='failed',message=str(exc),evaluations=[])
                result['rows'][d.hex()]=row
                atomic_save(path,result)
                if progress: progress(row)
                bar.update(1)
    return result


def segments(rows):
    """Break the stand-alone curve at every unsuccessful scheduled density."""
    out=[];current=[]
    for row in sorted(rows,key=lambda r:r['nB_0minus_over_n0']):
        if row['status']=='success':
            current.append((row['nB_0minus_over_n0'],row['root']['T_0minus']))
        else:
            if current: out.append(np.asarray(current));current=[]
    if current: out.append(np.asarray(current))
    return out
