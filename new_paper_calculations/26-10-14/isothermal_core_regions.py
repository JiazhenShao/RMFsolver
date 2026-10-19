"""Static region boundaries for the cold core figure; never solves a front."""
import hashlib
from pathlib import Path
import sys
import warnings
import numpy as np
from isothermal_mfp import atomic_save

ROOT=Path(__file__).resolve().parents[2]


def solve_temperature(T,seeds,physics,n0,import_paths=()):
    import os
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
    sys.path[:0]=list(import_paths)
    from scipy.optimize import brentq
    from RMFsolver import phase_velocity as pv
    cache={}
    def residual(density,kind):
        row=cache.setdefault(float(density),{})
        if 'upstream' not in row:
            row['upstream']=pv._validated_pnm_state_from_nB(density*n0,T)
        upstream=row['upstream']
        if kind not in row:
            if kind=='lower':
                mu=pv._solve_muB_inf_at_muK0_for_given_Pi(upstream['P_0minus'],0.,
                    physics['B_one_forth'],T,ms=physics['ms'],upB=physics['upB'])
                row[kind]=mu-upstream['muB_0minus']
            else:
                a=1.-1.e-9
                mu,muK=pv._solve_interface_0plus_from_local_a_and_Pi(a,upstream['P_0minus'],0.,
                    physics['B_one_forth'],T,ms=physics['ms'],upB=physics['upB'])
                row[kind]=mu+a*muK-upstream['muB_0minus']
        return row[kind]
    roots={};errors={}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for kind in ('lower','upper'):
            seed=seeds[kind]
            for radius in (.0001,.001,.01,.1):
                lo,hi=max(1.,seed-radius),min(5.5,seed+radius)
                if residual(lo,kind)*residual(hi,kind)<=0:
                    roots[kind]=brentq(lambda density:residual(density,kind),lo,hi,xtol=2.e-10,rtol=1.e-12)
                    errors[kind]=residual(roots[kind],kind)
                    break
            else:
                raise RuntimeError(f'No local density bracket for {kind} boundary')
    if roots['lower']>=roots['upper'] or max(map(abs,errors.values()))>1.e-5:
        raise RuntimeError('Static boundary residual validation failed')
    return dict(status='success',T_0minus=T,**roots,residuals_MeV=errors,
                upper_composition=1.-1.e-9,density_evaluations=len(cache),
                warnings=list(dict.fromkeys(str(w.message) for w in caught)),source='direct_fixed_density')


def calculate(inputs,temperatures,path,*,timeout=90.,solver=None,progress=None,show_progress=True):
    from tqdm.auto import tqdm
    temperatures=tuple(map(float,temperatures));path=Path(path)
    if not temperatures or min(temperatures)<=0 or max(temperatures)>1. or np.any(np.diff(temperatures)<=0):
        raise ValueError('Core-region schedule must increase within 0 < T <= 1 MeV')
    sources=[Path(__file__)]+sorted((ROOT/'RMFsolver').rglob('*.py'))
    identity=dict(temperatures=temperatures,physics=inputs['physics'],input_hashes=inputs['source_hashes'],
                  timeout=timeout,sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    result=np.load(path,allow_pickle=True).item() if path.exists() else dict(identity=identity,rows={})
    if result['identity']!=identity: raise ValueError('Incompatible core-region checkpoint; select a new file')
    if solver is None:
        import notebook_isothermal as runtime
        paths=runtime.worker_import_paths()
        def solver(T,seeds):
            return runtime.run_module_call_with_hard_timeout('isothermal_core_regions','solve_temperature',
                args=(T,seeds,inputs['physics'],inputs['n0'],paths),timeout_s=timeout)
    saved=list(inputs['boundary_cache'].values())
    with tqdm(total=len(temperatures),initial=len(result['rows']),desc='Static region boundaries',
              unit='temperature',dynamic_ncols=True,disable=not show_progress) as bar:
        for T in temperatures:
            key=T.hex()
            if key in result['rows']: continue
            nearest=min(saved,key=lambda row:abs(np.log(row['T_0minus']/T)))
            try:
                if np.isclose(nearest['T_0minus'],T,rtol=1.e-12,atol=0):
                    row=dict(nearest,T_0minus=T)
                else:
                    row=solver(T,{k:nearest[k] for k in ('lower','upper')})
            except Exception as exc:
                row=dict(status='failed',T_0minus=T,message=f'{type(exc).__name__}: {exc}')
            result['rows'][key]=row
            atomic_save(path,result)
            if progress: progress(row)
            bar.update(1)
    return result


def segments(rows):
    """Return T/lower/upper vertices, breaking at every failed boundary slice."""
    output=[];part=[]
    for row in sorted(rows,key=lambda row:row['T_0minus']):
        if row['status']=='success':
            part.append((row['T_0minus'],row['lower'],row['upper']))
        else:
            if part: output.append(np.asarray(part));part=[]
    if part: output.append(np.asarray(part))
    return output
