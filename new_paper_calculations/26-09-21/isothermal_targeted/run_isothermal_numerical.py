#!/usr/bin/env python3
"""Twenty designated targets, at most 120 new BVP calls including validation."""
import argparse
import contextlib
import fcntl
import os
from pathlib import Path
import sys

DIRECTORY=Path(__file__).resolve().parent
ROOT=DIRECTORY.parents[2]
sys.path.insert(0,str(ROOT))
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[variable]="1"
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib-isothermal-targeted")

import numpy as np
from _targeted import OLD, TargetRun, PHASES, PHASE_BUDGETS, initial_targets
from _targeted_io import read_inputs, snapshot_inputs, identity, profile_count, verify_snapshot
from _repair_common import load, workers


@contextlib.contextmanager
def single_writer(directory):
    with (directory/".targeted-run.lock").open("a") as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another targeted runner owns this output directory") from None
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=OLD,help="Completed/partial 26-09-16 repair folder")
    parser.add_argument("--profile-cache",type=Path,help="Existing repair continuation-profile directory")
    parser.add_argument("--workers",type=int)
    parser.add_argument("--dry-run",action="store_true",help="List exact targets and cache availability; no writes/BVP")
    parser.add_argument("--plot-only",action="store_true",help="Refresh saved results without any BVP")
    args=parser.parse_args(argv)
    count=workers(args.workers)
    source=DIRECTORY/"inputs" if (DIRECTORY/"inputs").exists() else args.source
    baseline,repair,roots,hashes=read_inputs(source,check_live=not args.plot_only)
    if source==DIRECTORY/"inputs":
        verify_snapshot(source)
    domain,_,numerical=baseline
    tasks=initial_targets(roots)
    cache=args.profile_cache or args.source/"isothermal-contour-repair-profiles"
    available=profile_count(repair,cache)
    print(f"Preserving {int(np.sum(numerical['task_status']=='success'))} baseline successes and {len(repair['points'])} repair solutions.",flush=True)
    print(f"20 initial targets; persistent BVP budget: warm 40, bend 44, cold 32, accuracy checks 4 (120 total).",flush=True)
    print(f"Continuation profiles available: {available}/{len(repair['points'])} in {cache}",flush=True)
    if args.dry_run:
        print("Read-only preflight: no solves or output writes.")
        for task in tasks:
            print(f"{task['phase']:5s}  {task['level']:4g} m/s  T={task['T_0minus']:.6f} MeV  density guess={task['predicted_density']:.6f} n0")
        if not available:
            print("Calculation requires the existing cluster profile cache; no automatic anchor rebuilding.")
        return 0
    from plot_targeted_contours import plot_targeted
    import matplotlib.pyplot as plt
    def draw(payload):
        fig,_=plot_targeted(domain,numerical,repair,payload,output_dir=DIRECTORY)
        plt.close(fig)
    if args.plot_only:
        output=DIRECTORY/"isothermal-targeted.npy"
        payload=load(output) if output.exists() else None
        if payload and payload["identity"]["input_hashes"]!=hashes:
            raise ValueError("Plot input hashes differ from the targeted checkpoint")
        draw(payload)
        return 0
    if not available:
        raise RuntimeError("No saved continuation profiles found. Supply --profile-cache pointing to the cluster's existing isothermal-contour-repair-profiles; no BVP was launched.")
    from tqdm.auto import tqdm
    with single_writer(DIRECTORY):
        source=snapshot_inputs(DIRECTORY,args.source)
        baseline,repair,roots,hashes=read_inputs(source)
        domain,_,numerical=baseline
        config_identity=identity(DIRECTORY,domain,repair,hashes)
        n0=float(np.median(domain["nB_0minus_grid"]/domain["nB_0minus_over_n0_grid"]))
        physics={k:domain[k] for k in ("B_one_forth_MeV","ms_MeV","NM_type","upB","xi")}
        runner=TargetRun(DIRECTORY,domain=domain,numerical=numerical,repair=repair,roots=roots,
                         identity=config_identity,n0=n0,physics=physics,profile_directory=cache)
        runner.payload["domain_fingerprint"]=domain["domain_fingerprint"]
        for task in initial_targets(roots):
            runner.payload["targets"].setdefault(task["id"],task)
        runner.save()
        bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
        terminal=lambda t:t["depth"]==0 and t["status"]!="pending"
        initial_done=sum(terminal(t) for t in runner.payload["targets"].values())
        with tqdm(total=20,initial=initial_done,desc="Initial targets",unit="target",bar_format=bar_format) as target_bar, \
             tqdm(total=120,initial=len(runner.payload["evaluations"]),desc="BVP calls / hard cap",unit="call",bar_format=bar_format) as call_bar:
            def evaluated(event,total):
                call_bar.update(max(0,total-call_bar.n))
                call_bar.set_postfix(status=event["status"],refresh=True)
            runner.evaluation_saved=evaluated
            def completed(task,payload):
                done=sum(terminal(t) for t in payload["targets"].values())
                target_bar.update(max(0,done-target_bar.n))
                refined=sum(t["depth"]>0 and t["status"]!="pending" for t in payload["targets"].values())
                target_bar.set_postfix(phase=task["phase"],refinements=refined,refresh=True)
                draw(payload)
            draw(runner.payload)
            for phase in PHASES:
                if phase not in runner.payload["finished_phases"]:
                    chains=2 if phase=="warm" else 6
                    tqdm.write(f"Starting {phase}: at most {PHASE_BUDGETS[phase]} calls; up to {min(count,chains)} independent contour chains.")
                    runner.run_phase(phase,count,completed)
                runner.validate(phase)
                completed(dict(phase=phase),runner.payload)
                verify_snapshot(source)
            runner.payload["run_complete"]=all(p in runner.payload["finished_phases"] for p in PHASES)
            runner.payload["validation_complete"]=len(runner.payload["validation"])==4 and all(v["status"]=="success" for v in runner.payload["validation"].values())
            runner.save()
        draw(runner.payload)
        statuses={}
        for task in runner.payload["targets"].values():
            statuses[task["status"]]=statuses.get(task["status"],0)+1
        print(f"Saved {runner.output}; consumed {len(runner.payload['evaluations'])}/120 BVP calls.")
        print("Target states:",statuses)
        print("Accuracy checks:",{k:v['status'] for k,v in runner.payload['validation'].items()})
        print("Completed work does not imply all requested intervals were certified; unresolved intervals remain gaps.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
