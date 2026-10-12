#!/usr/bin/env python3
"""Cold numerical isothermal contour targets; automatic bounded resume."""
import argparse
import ast
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

# Spawned children inherit these before importing NumPy/SciPy.
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MPLBACKEND','Agg')

import numpy as np
from _logT_common import (PHYSICS, CONTROLS, CHECK_CONTROLS, TEMPERATURES, LEVELS,
                         base_targets, sha256, workers, temperature_id, load)

HERE = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int)
    p.add_argument('--dry-run',action='store_true',help='Validate inputs and print schedule; no EOS/BVP calls')
    p.add_argument('--smoke',action='store_true',help='Four bounded cluster targets in a separate smoke directory')
    p.add_argument('--plot-only',action='store_true',help='Render saved data without importing the solver')
    p.add_argument('--output-dir',type=Path)
    return p


def output_directory(args):
    return args.output_dir.resolve() if args.output_dir else HERE/('smoke' if args.smoke else '')


def project_root():
    return next(p for p in HERE.parents if (p/'RMFsolver/phase_velocity.py').is_file())


def identity(inputs, smoke):
    root = project_root()
    source = (root/'RMFsolver/phase_velocity.py').read_text()
    signature = next(n.args for n in ast.parse(source).body
                     if isinstance(n,ast.FunctionDef) and n.name == 'solve_front_isothermal')
    names = {a.arg for a in signature.args+signature.kwonlyargs}
    required = {'continuation_guess','return_continuation','initial_guess_mode'}
    if not required <= names:
        raise ValueError('Upload current RMFsolver/phase_velocity.py: missing optional continuation/layer initializer API')
    return dict(schema=1, physics=PHYSICS, temperatures=TEMPERATURES.tolist(), levels=list(LEVELS),
        controls=CONTROLS, accuracy_controls=CHECK_CONTROLS, smoke=bool(smoke),
        inputs=inputs['manifest'], domain_fingerprint=inputs['fingerprint'],
        solver_files={str(p.relative_to(root)):sha256(p) for p in sorted((root/'RMFsolver').glob('*.py'))},
        workflow_files={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        optional_api=sorted(required))


def boundary_pending(row):
    return not row or (row['status']=='interrupted' and row.get('attempts',1)<2)


def prepare_boundaries(run, temperatures, worker_count):
    from _logT_trial import bounded_call
    def solve(T):
        key = temperature_id(T)
        with run.lock:
            previous = run.payload['boundaries'].get(key)
            if not boundary_pending(previous):
                return
            seeds = {}
            for name,archive in (('lower','inner_boundary'),('upper','outer_boundary')):
                curve = np.asarray(run.inputs['warm'][archive],float)
                curve = curve[np.isfinite(curve).all(axis=1)]
                curve = curve[np.argsort(curve[:,1])]
                seeds[name] = float(np.interp(T,curve[:,1],curve[:,0]))
            attempt = (previous.get('attempts',1) if previous else 0)+1
            run.payload['boundaries'][key] = dict(T_0minus=T,status='running',seeds=seeds,
                                                 attempts=attempt,previous_attempt=previous)
            run.save()
        answer = bounded_call('boundary',timeout=180.,T=T,seeds=seeds,n0=run.inputs['n0'])
        with run.lock:
            run.payload['boundaries'][key] = dict(answer,T_0minus=T,seeds=seeds,attempts=attempt,
                                                 previous_attempt=previous)
            run.save()
            run.boundary_saved(run.payload['boundaries'][key])
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for f in as_completed([pool.submit(solve,float(T)) for T in sorted(set(temperatures))]):
            f.result()


def refresh(inputs, payload, directory):
    from _logT_plot import plot_contours, export_plot_archive
    import matplotlib.pyplot as plt
    export_plot_archive(inputs['warm'],payload,directory/'isothermal-logT-plot.npz')
    fig = plot_contours(inputs['warm'],payload,output_dir=directory)
    diagnostics = fig.logT_diagnostics
    plt.close(fig)
    return diagnostics


@contextmanager
def execution_lock(directory):
    """One writer owns reservations; read-only plotting remains independent."""
    import fcntl
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.run.lock').open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another runner already owns this checkpoint') from exc
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def _main(args):
    from _logT_inputs import load_inputs
    inputs = load_inputs(HERE)
    directory = output_directory(args)
    checkpoint = directory/'isothermal-logT.npy'
    if args.plot_only:
        if not checkpoint.is_file():
            raise FileNotFoundError(f'No saved cold checkpoint: {checkpoint}')
        saved = load(checkpoint)
        if saved['identity']['inputs'] != inputs['manifest']:
            raise ValueError('Warm snapshot differs from this checkpoint')
        refresh(inputs,saved,directory)
        print(f'Refreshed {directory}/isothermal-logT.png from saved data only')
        return 0
    run_identity = identity(inputs,args.smoke)
    worker_count = workers(args.workers)
    tasks = base_targets(smoke=args.smoke)
    if args.dry_run:
        if checkpoint.exists():
            saved = load(checkpoint)
            if saved['identity'] != run_identity or saved['controls'] != CONTROLS:
                raise ValueError('Checkpoint identity/controls changed')
        print(json.dumps(dict(base_targets=len(tasks),temperatures=sorted(set(t['T_0minus'] for t in tasks)),
            levels=LEVELS,workers=worker_count,max_attempts=900,accuracy_reserve=14,
            cached_cold_seed_points=len(inputs['points']),output=str(directory),smoke=args.smoke,
            stage_order=['boundaries','anchors','block continuation','one midpoint refinement','accuracy checks'],
            expensive_solves_started=0),indent=2))
        return 0
    from _logT_engine import ContourRun
    from tqdm import tqdm
    run = ContourRun(directory,inputs,run_identity)
    # Input profiles reside beside the bundled snapshots, including when --smoke
    # or --output-dir puts the new checkpoint in a separate directory.
    run.input_directory = HERE
    run.payload.update(temperatures=TEMPERATURES.tolist(),levels=list(LEVELS))
    for task in tasks:
        run.payload['targets'].setdefault(task['id'],task)
    run.save()
    refresh(inputs,run.payload,directory)
    bars = {}
    def bar(description,total,initial,position):
        return tqdm(total=total,initial=initial,desc=description,unit='target',position=position,
                    ncols=110,dynamic_ncols=False,miniters=1,mininterval=0.,
                    bar_format='{desc}: {percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    temperatures = sorted(set(t['T_0minus'] for t in tasks))
    boundary_temperatures = set(temperatures) | {float.fromhex(k) for k in run.payload['boundaries']}
    bars['boundary'] = bar('Cold boundaries',len(boundary_temperatures),
        sum(not boundary_pending(run.payload['boundaries'].get(temperature_id(T))) for T in boundary_temperatures),0)
    bars['base'] = bar('Base contours',len(tasks),sum(run.payload['targets'][t['id']]['status']
        not in ('pending','running') for t in tasks),1)
    refinement_rows = [t for t in run.payload['targets'].values() if t['kind']=='refinement']
    bars['refinement'] = bar('Scheduled refinements',len(refinement_rows),
        sum(t['status'] not in ('pending','running') for t in refinement_rows),2)
    check_bases = [t for t in tasks if t['T_0minus'] in (.01,20.)]
    bars['validation'] = bar('Accuracy checks',len(check_bases),len(run.payload['validation']),3)
    def report_attempts(event=None):
        used = len(run.payload['evaluations'])
        accepted = sum(t['status']=='success' for t in run.payload['targets'].values())
        bars['base'].set_postfix_str(f'attempts {used}/900; accepted {accepted}',refresh=True)
    def target_saved(task):
        bars[task['kind']].update(1)
        if task['status']=='success' and task['kind']!='validation':
            refresh(inputs,run.payload,directory)
    run.evaluation_saved = report_attempts
    run.target_saved = target_saved
    run.boundary_saved = lambda row: bars['boundary'].update(1)
    print(f'Using up to {worker_count} workers; warm inputs immutable; new solves only 0.01–20 MeV.',flush=True)
    report_attempts()
    try:
        prepare_boundaries(run,temperatures,worker_count)
        run.run_base(worker_count)
        if not args.smoke:
            run.schedule_refinements()
            rows = [t for t in run.payload['targets'].values() if t['kind']=='refinement']
            bars['refinement'].total = len(rows)
            bars['refinement'].refresh()
            refinement_temperatures = {t['T_0minus'] for t in rows}
            new_temperatures = refinement_temperatures-boundary_temperatures
            bars['boundary'].total += len(new_temperatures)
            bars['boundary'].refresh()
            prepare_boundaries(run,refinement_temperatures,worker_count)
            # One chain per level retains profile continuation between refinements.
            def refine(level):
                for task in sorted([t for t in rows if t['level']==level],key=lambda t:-t['T_0minus']):
                    run.solve_target(task['id'])
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                list(pool.map(refine,LEVELS))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            list(pool.map(run.validate_target,[run.payload['targets'][t['id']] for t in check_bases]))
        run.payload['plot_diagnostics'] = refresh(inputs,run.payload,directory)
        run.finalize()
    finally:
        for b in bars.values():
            b.close()
    accepted = sum(t['status']=='success' for t in run.payload['targets'].values())
    print(f'Work terminal: {run.payload["work_complete"]}; validated targets: {accepted}/{len(run.payload["targets"])}; '
          f'contours complete: {run.payload["contours_complete"]}; attempts: {len(run.payload["evaluations"])}/900')
    print(f'Saved {checkpoint}; unresolved intervals remain gaps.')
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    if args.dry_run or args.plot_only:
        return _main(args)
    with execution_lock(output_directory(args)):
        return _main(args)


if __name__ == '__main__':
    raise SystemExit(main())
