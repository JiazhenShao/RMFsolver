"""Reproducible notebook calculation, independent of saved analytical velocities.

Reuse static boundary solutions, not earlier contour roots. Each moving speed is
evaluated by the public analytical solver. Importing this module runs no physics.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import platform
import shlex
import sys
import threading
import warnings
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLUSTER = HERE / 'isothermal_analytic_logT'
for directory in (HERE, CLUSTER, HERE/'adiabatic_xi_minus0p85', ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from _logT_common import TEMPERATURES, PHYSICS, C, load, sha256, target, point, temperature_id
from _logT_engine import AnalyticRun
from _logT_trial import valid_analytical_result
from _bounded_call import run_module_call_with_hard_timeout

LEVELS = (50.,100.,200.,500.,1000.,2000.,5000.,10000.,20000.,50000.,100000.,200000.,500000.,1000000.)


def bundled_runtime_path(root=ROOT):
    """Keep compiled wheels separate from the kernel and other Python/platform ABIs."""
    name=f'scipy-1.18.1-{sys.implementation.cache_tag}-{sys.platform}-{platform.machine()}'
    return Path(root).resolve()/'local_analysis_results/26-10-02/runtime'/name


def worker_import_paths(import_paths=None,*,root=ROOT):
    """None selects the installed sidecar; () explicitly requests kernel packages."""
    if import_paths is not None:
        return tuple(str(Path(p).expanduser().resolve()) for p in import_paths)
    runtime=bundled_runtime_path(root)
    return (str(runtime),) if (runtime/'scipy/__init__.py').is_file() else ()


def load_boundary_inputs(root=ROOT):
    root = Path(root)
    folder = root/'new_paper_calculations/26-10-02/isothermal_analytic_logT'
    source = folder/'isothermal-analytic-logT.npy'
    static_path = folder/'inputs/static-boundaries.npy'
    original, static = load(source), load(static_path)
    if original['domain_fingerprint'] != static['domain_fingerprint']:
        raise ValueError('Static boundary provenance mismatch')
    cache = {}
    for T in TEMPERATURES:
        row = min(original['boundaries'].values(), key=lambda b:abs(b['T_0minus']-T))
        if not np.isclose(row['T_0minus'],T,rtol=1.e-12,atol=0) or row['status'] not in ('success','outside_window'):
            raise ValueError(f'No validated static boundary at T={T:g}')
        cache[temperature_id(T)] = dict(copy.deepcopy(row), T_0minus=float(T))
    # Intentionally discard all old points, velocities, targets and evaluations.
    return dict(n0=static['n0'],fingerprint=original['domain_fingerprint'],
                boundary_cache=cache,physics=copy.deepcopy(original['identity']['physics']),
                source_hashes={str(p.relative_to(root)):sha256(p) for p in (source,static_path)})


def load_numerical(root=ROOT):
    root=Path(root)
    folder=root/'local_analysis_results/All_Plots_new/isothermal_contour_data'
    manifest=json.loads((folder/'manifest.json').read_text())
    warm_path=folder/'isothermal-numerical-contour.npz'
    cold_path=folder/'isothermal-cold-endpoints.npy'
    numeric_path=root/'new_paper_calculations/26-10-01/isothermal_numeric_logT/isothermal-logT.npy'
    if sha256(warm_path)!=manifest['plot_archive_sha256'] or sha256(cold_path)!=manifest['source_files'][cold_path.name]['sha256']:
        raise ValueError('Saved numerical archive hash mismatch')
    with np.load(warm_path,allow_pickle=False) as archive:
        warm={k:archive[k].copy() for k in archive.files}
    numeric,cold=load(numeric_path),load(cold_path)
    if len({manifest['domain_fingerprint'],numeric['domain_fingerprint'],cold['domain_fingerprint']})!=1:
        raise ValueError('Numerical domain fingerprints differ')
    return warm,numeric,cold,manifest


def load_reference(root=ROOT):
    """Explicit plot-only mode; not an input to calculate()."""
    from repair_analytical_local import merge_isothermal
    from _saved_contour_data import merge_analytical_extension
    root=Path(root);folder=root/'local_analysis_results/26-10-02'
    baseline=load(root/'new_paper_calculations/26-10-02/isothermal_analytic_logT/isothermal-analytic-logT.npy')
    repair=load(folder/'analytical-local-repair.npy')
    for relative,digest in repair['source_hashes'].items():
        if sha256(root/relative)!=digest:
            raise ValueError('Saved reference recovery input hash mismatch')
    result=merge_isothermal(baseline,repair)
    for name in ('isothermal-high-speed-extension.npy','isothermal-outer-speed-extension.npy'):
        extension=load(folder/name)
        for relative,digest in extension['identity']['input_hashes'].items():
            if sha256(root/relative)!=digest:
                raise ValueError('Saved reference extension input hash mismatch')
        result=merge_analytical_extension(result,extension)
    return result


def runtime_info(import_paths=()):
    import os
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    sys.path[:0]=[str(p) for p in import_paths]
    import scipy
    from RMFsolver import phase_velocity  # Check the full import before starting work.
    return dict(python=sys.version,scipy=scipy.__version__,numpy=np.__version__,scipy_path=scipy.__file__)


def evaluate_point(physical,physics,import_paths=()):
    runtime=runtime_info(import_paths)
    from RMFsolver import phase_velocity as pv
    with warnings.catch_warnings(record=True) as caught:
        result=pv.analytic_velocity_isothermal(
            physical['T_0minus'],physical['nB_0minus'],physics['B_one_forth'],
            a_0plus=np.nan,xi=physics['xi'],ms=physics['ms'],NM_type=physics['NM_type'],upB=physics['upB'])
    result['validated']=valid_analytical_result(result)
    result['velocity_m_s']=float(C*result['u_0minus']/np.hypot(1.,result['u_0minus'])) if result['validated'] else np.nan
    if result.get('success') and result.get('status') in ('stable_neutron_matter','isothermal_coexistence'):
        result['velocity_m_s']=0.
    result['warnings']=list(dict.fromkeys(str(w.message) for w in caught))
    result['runtime_environment']=runtime
    return result


def propose_density(samples,level,limits):
    lo,hi=limits
    rows=sorted({r['nB_0minus_over_n0']:r for r in samples if np.isfinite(r['velocity_m_s'])}.values(),
                key=lambda r:r['nB_0minus_over_n0'])
    brackets=[(a,b) for a,b in zip(rows[:-1],rows[1:]) if (a['velocity_m_s']-level)*(b['velocity_m_s']-level)<0]
    if brackets:
        a,b=min(brackets,key=lambda pair:pair[1]['nB_0minus_over_n0']-pair[0]['nB_0minus_over_n0'])
        va,vb=a['velocity_m_s'],b['velocity_m_s']
        fraction=np.log(level/va)/np.log(vb/va) if va>0 and vb>0 else (level-va)/(vb-va)
        return float(a['nB_0minus_over_n0']+np.clip(fraction,.02,.98)*(b['nB_0minus_over_n0']-a['nB_0minus_over_n0']))
    below=[r['nB_0minus_over_n0'] for r in rows if r['velocity_m_s']<level]
    above=[r['nB_0minus_over_n0'] for r in rows if r['velocity_m_s']>level]
    if above:
        return .5*(lo+min(above))
    if below and max(below)>lo:
        last=max(below)
        return min(lo+2*(last-lo),.5*(last+hi))
    if any(r['velocity_m_s']>0 for r in rows):
        return .5*(lo+hi)  # Clipped density edge, not coexistence.
    return lo+1.e-4*(hi-lo)


class NotebookRun(AnalyticRun):
    """Keep established checkpoint/timeout accounting; use improved proposals."""
    def solve_temperature(self,T,levels):
        boundary=self.payload['boundaries'][temperature_id(T)]
        for level in levels:
            if self.cancel.is_set():
                return
            task=self.payload['targets'][target(level,T)['id']]
            if task['status'] not in ('pending','running'):
                continue
            task['status']='running'
            if boundary['status']=='outside_window':
                self.finish_target(task,'outside_window',message=boundary.get('reason','No allowed interval'))
                continue
            lo,hi=boundary['lower'],boundary['upper']
            with self.lock:
                samples=[p for p in self.payload['points'].values() if p['T_0minus']==T]
            if not boundary.get('lower_clipped'):
                samples.append(dict(nB_0minus_over_n0=lo,velocity_m_s=0.))
            classified=False
            for density,clipped,lower in ((lo,boundary.get('lower_clipped'),True),(hi,boundary.get('upper_clipped'),False)):
                if not clipped:
                    continue
                edge=self.shoot(task,point(T,density,self.inputs['n0']))
                if edge is None:
                    self.finish_target(task,'unresolved',message='Failed to evaluate clipped window edge')
                    classified=True;break
                samples.append(edge)
                v=edge['velocity_m_s'];tol=self.controls['speed_rtol']
                if (lower and v>level*(1+tol)) or (not lower and v<level*(1-tol)):
                    self.finish_target(task,'outside_window',message='Requested speed outside density window')
                    classified=True;break
            if classified:
                continue
            found=None
            for _ in range(self.controls['evaluations_per_target']+1):
                if self.cancel.is_set():
                    return
                found=next((p for p in samples if p.get('result',{}).get('validated') and
                            abs(p['velocity_m_s']/level-1)<=self.controls['speed_rtol']),None)
                if found or self.spent(task)>=self.controls['target_timeout']:
                    break
                density=propose_density(samples,level,(lo,hi))
                # Interrupted reservations remain spent. Avoid re-proposing their
                # exact coordinates, which the shared engine correctly refuses.
                with self.lock:
                    failed=[e['point']['nB_0minus_over_n0'] for e in self.payload['evaluations']
                            if e['point']['T_0minus']==T and e['point']['id'] not in self.payload['points']]
                for _ in range(32):
                    if not any(abs(density-d)<1.e-12 for d in failed):
                        break
                    below=[p['nB_0minus_over_n0'] for p in samples if p['nB_0minus_over_n0']<density]
                    density=.5*(density+(max(below) if below else lo))
                if not lo<density<hi or any(abs(density-p['nB_0minus_over_n0'])<1.e-13 for p in samples):
                    break
                answer=self.shoot(task,point(T,density,self.inputs['n0']))
                if answer is None:
                    break
                samples.append(answer)
            if self.cancel.is_set():
                return
            self.finish_target(task,'success' if found else 'unresolved',root=found,
                               message=None if found else 'No validated root within the evaluation allowance; inspect evaluations')


def calculate(inputs,directory,*,temperatures=TEMPERATURES,levels=LEVELS,workers=2,
              xi=-.85,trial_timeout=90.,target_timeout=600.,max_evaluations=32,speed_rtol=.005,
              import_paths=None,evaluator=None,evaluation_saved=None,show_progress=True):
    temperatures=list(map(float,temperatures));levels=list(map(float,levels))
    if (workers<1 or not temperatures or not levels or not np.all(np.isfinite(temperatures+levels))
            or np.any(np.diff(temperatures)<=0) or np.any(np.diff(levels)<=0)
            or min(temperatures)<.01 or max(temperatures)>120 or min(levels)<=0
            or not np.all(np.isfinite([xi,trial_timeout,target_timeout,max_evaluations,speed_rtol]))
            or not -1<=xi or trial_timeout<=0 or target_timeout<=0 or max_evaluations<1
            or not 0<speed_rtol<=.005):
        raise ValueError('Invalid axes or numerical controls')
    if any(temperature_id(T) not in inputs['boundary_cache'] for T in temperatures):
        raise ValueError('Temperature axis needs directly solved cached boundaries; no silent boundary interpolation')
    physics=dict(inputs['physics'],xi=float(xi))
    controls=dict(trial_timeout=float(trial_timeout),target_timeout=float(target_timeout),
                  evaluations_per_target=int(max_evaluations),speed_rtol=float(speed_rtol))
    files=[Path(__file__),HERE/'_isothermal_notebook_worker.py',CLUSTER/'_logT_engine.py',CLUSTER/'_logT_trial.py',CLUSTER/'_logT_common.py',
           HERE/'adiabatic_xi_minus0p85/_bounded_call.py',ROOT/'RMFsolver/phase_velocity.py']
    identity=dict(method='notebook_fresh_analytical_v1',physics=physics,temperatures=temperatures,levels=levels,
                  domain_fingerprint=inputs['fingerprint'],
                  boundary_input_hashes=inputs['source_hashes'],source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in files})
    cancel=threading.Event()
    from tqdm.auto import tqdm
    if evaluator is None:
        import_paths=worker_import_paths(import_paths)
        try:
            with tqdm(total=1,desc='Checking analytical runtime',unit='check',ncols=110,
                      disable=not show_progress) as preflight:
                runtime=run_module_call_with_hard_timeout('_isothermal_notebook_worker','call',
                    args=('preflight',),kwargs=dict(import_paths=import_paths),timeout_s=30.)
                preflight.update(1)
        except Exception as exc:
            command=shlex.join([sys.executable,'-m','pip','install','--no-deps','--target',
                               str(bundled_runtime_path()),'scipy==1.18.1'])
            raise RuntimeError('Analytical runtime import failed before any solves or reservations. '
                               'To install an isolated runtime without changing your environment, run in a terminal:\n'
                               +command+'\nThen use WORKER_IMPORT_PATHS = None and rerun the calculation cell. '
                               'Alternatively select a working kernel and WORKER_IMPORT_PATHS = ().\n'+str(exc)) from exc
        identity['runtime_environment']=runtime
        if show_progress:
            print(f"Analytical workers: SciPy {runtime['scipy']} — {runtime['scipy_path']}")
        def evaluator(physical):
            return run_module_call_with_hard_timeout('_isothermal_notebook_worker','call',
                args=('analytical',physical,physics,import_paths),timeout_s=trial_timeout,cancel=cancel)
    with tqdm(total=len(temperatures)*len(levels),desc='Analytical contour targets',unit='target',
              ncols=110,position=0,mininterval=0.,miniters=1,disable=not show_progress) as progress, \
         tqdm(total=None,desc='Saved analytical evaluations',unit='point',ncols=110,position=1,
              mininterval=0.,miniters=1,disable=not show_progress) as evaluations:
        def on_evaluation(event):
            if evaluation_saved:
                evaluation_saved(event)
            evaluations.update(1)  # Every completed solve is already atomically saved.
        def on_target(row):
            statuses=[r['status'] for r in run.payload['targets'].values()]
            progress.set_postfix(accepted=statuses.count('success'),unresolved=statuses.count('unresolved'),refresh=False)
            progress.update(1)  # AnalyticRun saves before invoking this callback.
        run=NotebookRun(directory,inputs,identity,controls=controls,evaluator=evaluator,
                        evaluation_saved=on_evaluation,target_saved=on_target)
        run.cancel=cancel
        run.payload.update(temperatures=np.array(temperatures),levels=np.array(levels))
        for T in temperatures:
            for level in levels:
                row=target(level,T)
                run.payload['targets'].setdefault(row['id'],row)
        run.save()
        progress.update(sum(r['status'] not in ('pending','running') for r in run.payload['targets'].values()))
        evaluations.update(sum(e['status'] not in ('running','interrupted') for e in run.payload['evaluations']))
        pool=ThreadPoolExecutor(max_workers=workers)
        try:
            for future in as_completed([pool.submit(run.solve_temperature,T,levels) for T in temperatures]):
                future.result()
        except BaseException:
            cancel.set()
            raise
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
        run.finalize()
    print('Checkpoint:',run.output)
    return run.payload
