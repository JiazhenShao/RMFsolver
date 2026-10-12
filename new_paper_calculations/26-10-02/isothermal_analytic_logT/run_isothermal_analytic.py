#!/usr/bin/env python3
"""Analytical isothermal log-temperature contours at xi=-0.85."""
import argparse
import ast
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('MPLBACKEND','Agg')

from _logT_common import PHYSICS, CONTROLS, LEVELS, TEMPERATURES, base_targets, load, sha256, workers, temperature_id
from _logT_inputs import load_inputs

HERE=Path(__file__).resolve().parent


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int)
    p.add_argument('--dry-run',action='store_true',help='Check input hashes and schedule; no solver calls')
    p.add_argument('--smoke',action='store_true',help='Four actual bounded targets in a separate smoke directory')
    p.add_argument('--plot-only',action='store_true',help='Plot the saved analytical checkpoint only')
    p.add_argument('--output-dir',type=Path)
    return p


def identity(inputs,smoke):
    root=next(p for p in HERE.parents if (p/'RMFsolver/phase_velocity.py').is_file())
    source=(root/'RMFsolver/phase_velocity.py').read_text()
    functions={node.name:node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
    if 'analytic_velocity_isothermal' not in functions:
        raise ValueError('Upload current RMFsolver/phase_velocity.py with analytic_velocity_isothermal')
    return dict(schema=2,method='analytic_isothermal',physics=PHYSICS,temperatures=TEMPERATURES.tolist(),
        levels=list(LEVELS),controls=CONTROLS,smoke=bool(smoke),inputs=inputs['manifest'],
        solver_files={str(p.relative_to(root)):sha256(p) for p in sorted((root/'RMFsolver').glob('*.py'))},
        workflow_files={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))})


def refresh(inputs,payload,directory):
    from _logT_plot import plot_contours,export_plot_archive
    import matplotlib.pyplot as plt
    directory=Path(directory)
    export_plot_archive(inputs['boundary_geometry'],payload,directory/'isothermal-analytic-logT-plot.npz')
    fig=plot_contours(inputs['boundary_geometry'],payload,output_dir=directory)
    plt.close(fig)


@contextmanager
def execution_lock(directory):
    import fcntl
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.run.lock').open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another writer already owns this analytical checkpoint') from exc
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def _main(args,directory):
    inputs=load_inputs(HERE)
    checkpoint=directory/'isothermal-analytic-logT.npy'
    if args.plot_only:
        saved=load(checkpoint)
        if saved.get('method')!='analytic_isothermal' or saved['identity']['physics']!=PHYSICS:
            raise ValueError('Not an analytical xi=-0.85 checkpoint')
        if saved['identity']['inputs']!=inputs['manifest']:
            raise ValueError('Input snapshot differs from the checkpoint')
        refresh(inputs,saved,directory)
        print(f'Refreshed {directory}/isothermal-analytic-logT.png from saved analytical data')
        return 0
    run_identity=identity(inputs,args.smoke)
    tasks=base_targets(args.smoke)
    temperatures=sorted({t['T_0minus'] for t in tasks})
    worker_count=workers(args.workers)
    if args.dry_run:
        if checkpoint.exists():
            saved=load(checkpoint)
            if saved['identity']!=run_identity or saved['controls']!=CONTROLS:
                raise ValueError('Checkpoint identity/controls changed')
        print(json.dumps(dict(method='analytical_isothermal',xi=PHYSICS['xi'],
            temperatures=temperatures,contour_targets=len(tasks),levels=LEVELS,workers=worker_count,
            cached_boundary_temperatures=sum(temperature_id(T) in inputs['boundary_cache'] for T in temperatures),
            stage_order=['static boundaries (reuse validated records)','analytical contour targets','analytical plot'],
            output=str(directory),expensive_solves_started=0),indent=2))
        return 0
    from _logT_engine import AnalyticRun
    from tqdm import tqdm
    run=AnalyticRun(directory,inputs,run_identity)
    run.payload.update(temperatures=temperatures,levels=sorted({t['level'] for t in tasks}))
    for task in tasks:
        run.payload['targets'].setdefault(task['id'],task)
    run.save()
    def bar(description,total,initial):
        return tqdm(total=total,initial=initial,desc=description,unit='point',ncols=110,
            dynamic_ncols=False,miniters=1,mininterval=0.,
            bar_format='{desc}: {percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    print(f'Using up to {worker_count} workers; analytical xi=-0.85, T=0.01–120 MeV. No numerical BVPs.',flush=True)
    done=lambda row: row and row['status'] not in ('running','interrupted')
    boundary_bar=bar('Static boundary temperatures',len(temperatures),sum(bool(done(
        run.payload['boundaries'].get(temperature_id(T)))) for T in temperatures))
    run.boundary_saved=lambda row:boundary_bar.update(1)
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            for future in as_completed([pool.submit(run.prepare_boundary,T) for T in temperatures]):
                future.result()
    finally:
        boundary_bar.close()
    initial=sum(run.payload['targets'][t['id']]['status'] not in ('pending','running') for t in tasks)
    contour_bar=bar('Analytical contour targets',len(tasks),initial)
    figure_dirty=threading.Event()
    def report(event=None):
        accepted=sum(t['status']=='success' for t in run.payload['targets'].values())
        contour_bar.set_postfix_str(f"evaluations {len(run.payload['evaluations'])}; accepted {accepted}",refresh=True)
    def target_saved(task):
        contour_bar.update(1)
        report()
        if task['status']=='success':
            figure_dirty.set()
    run.evaluation_saved=report  # Evaluations are already atomically saved here.
    run.target_saved=target_saved
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures=[pool.submit(run.solve_temperature,T,[t['level'] for t in tasks if t['T_0minus']==T])
                     for T in temperatures]
            pending=set(futures)
            while pending:
                ready,pending=wait(pending,timeout=.2,return_when=FIRST_COMPLETED)
                for future in ready:
                    future.result()
                if figure_dirty.is_set():
                    figure_dirty.clear()
                    with run.lock:
                        snapshot=copy.deepcopy(run.payload)
                    # Coalesce simultaneous successes and render on the main
                    # thread without blocking evaluation saves behind plotting.
                    refresh(inputs,snapshot,directory)
        run.finalize()
        refresh(inputs,run.payload,directory)
    finally:
        contour_bar.close()
    statuses=[t['status'] for t in run.payload['targets'].values()]
    print(f"Saved {checkpoint}; accepted {statuses.count('success')}/{len(statuses)}, "
          f"outside physical window {statuses.count('outside_window')}, unresolved {statuses.count('unresolved')}.")
    print(f"Work terminal: {run.payload['work_complete']}; contours complete: {run.payload['contours_complete']}.")
    return 0 if run.payload['contours_complete'] else 2


def main(argv=None):
    args=parser().parse_args(argv)
    directory=args.output_dir.resolve() if args.output_dir else HERE/('smoke' if args.smoke else '')
    if args.dry_run or args.plot_only:
        return _main(args,directory)
    with execution_lock(directory):
        return _main(args,directory)


if __name__=='__main__':
    raise SystemExit(main())
