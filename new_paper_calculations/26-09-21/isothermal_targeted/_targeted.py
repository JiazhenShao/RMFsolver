"""Bounded, fixed-temperature numerical shots; no global scan or recursive BVPs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "new_paper_calculations/26-09-16/isothermal_numeric_repair"
sys.path.insert(0, str(OLD))
from _repair_common import (C, DEFAULT_CONTROLS, atomic_save, current_bracket,
                            load, near_current_bound)
from _targeted_trial import hard_trial

PHASES = ("warm", "bend", "cold")
PHASE_BUDGETS = dict(warm=40, bend=44, cold=32)
MAX_CALLS = 120
MAX_TARGET_CALLS = 4
TARGET_SECONDS = 600.
SPEED_RTOL = .005
DENSITY_TOL = .005
TEMPERATURE_FLOOR = 1.
MAX_REFINEMENT_DEPTH = 3
CONTROLS = {key: DEFAULT_CONTROLS[key] for key in
            ("trial_timeout", "n_mesh", "tol_bvp", "max_nodes", "tail_eps", "kappa_factor")}
CONTROLS["target_timeout"] = TARGET_SECONDS


def physical_id(temperature, density):
    return f"{temperature:.12g}:{density:.12g}"


def source_roots(repair):
    roots = []
    for ray in repair["rays"].values():
        if ray["status"] == "success":
            for point in ray["roots"]:
                if abs(point["velocity_m_s"] / ray["level"] - 1.) <= SPEED_RTOL:
                    roots.append(dict(point, level=float(ray["level"]), profile_origin="source"))
    return sorted(roots, key=lambda p: (p["level"], p["T_0minus"]))


def make_target(phase, level, temperature, roots, *, left=None, right=None, depth=0):
    if temperature < TEMPERATURE_FLOOR:
        raise ValueError("New shots must respect the 1 MeV floor")
    same = sorted([p for p in roots if p["level"] == level], key=lambda p: p["T_0minus"])
    if len(same) < 2:
        raise ValueError(f"Need two numerical roots to seed level {level}")
    if left is None and right is None:
        below = [p for p in same if p["T_0minus"] < temperature]
        above = [p for p in same if p["T_0minus"] > temperature]
        left, right = (below[-1] if below else None), (above[0] if above else None)
    if left is not None and right is not None:
        a, b = left, right
    elif left is None:
        a, b = same[:2]
    else:
        a, b = same[-2:]
    weight = (temperature - a["T_0minus"]) / (b["T_0minus"] - a["T_0minus"])
    prediction = a["nB_0minus_over_n0"] + weight * (b["nB_0minus_over_n0"] - a["nB_0minus_over_n0"])
    return dict(id=f"{phase}:{level:g}:{temperature:.10g}", phase=phase, level=float(level),
                T_0minus=float(temperature), predicted_density=float(np.clip(prediction, 1., 5.5)),
                left=left, right=right, depth=depth, status="pending", root=None,
                certified=False, children=[], density_error=None)


def initial_targets(roots):
    tasks = []
    for level, angles in ((500., (70,75,80,85,90)), (1000., (40,45,50,55,60))):
        selected=[]
        for angle in angles:
            matches=[p for p in roots if p["level"]==level and abs(p.get("theta_rad",-1)-np.radians(angle))<1.e-10]
            if len(matches)!=1:
                raise ValueError(f"Need one unambiguous saved {level:g} root on the {angle:g}-degree ray")
            selected.append(matches[0])
        for a, b in zip(selected[:-1], selected[1:]):
            tasks.append(make_target("warm", level, .5*(a["T_0minus"]+b["T_0minus"]), roots, left=a, right=b))
    # Midpoints of the actual intervals straddling these temperatures.
    for level, temperature in ((50.,15.48), (100.,17.), (200.,19.60), (500.,24.11),
                               (1000.,27.08), (2000.,16.64), (2000.,28.93)):
        task = make_target("bend", level, temperature, roots)
        if task["left"] is None or task["right"] is None:
            raise ValueError("The bend targets are not bracketed by saved numerical roots")
        tasks.append(make_target("bend", level, .5*(task["left"]["T_0minus"] + task["right"]["T_0minus"]), roots))
    for level, temperatures in ((1000., (1.25,1.)), (2000., (2.3,1.55,1.))):
        tasks.extend(make_target("cold", level, temperature, roots) for temperature in temperatures)
    if len(tasks) != 20 or len({t["id"] for t in tasks}) != 20:
        raise ValueError("Source checkpoint does not support the approved 20-target manifest")
    return tasks


def certify_interval(left, middle, right, level):
    if any(p is None for p in (left,middle,right)):
        return False, None
    if not left["T_0minus"] < middle["T_0minus"] < right["T_0minus"]:
        return False, None
    w = (middle["T_0minus"]-left["T_0minus"])/(right["T_0minus"]-left["T_0minus"])
    predicted = (1-w)*left["nB_0minus_over_n0"] + w*right["nB_0minus_over_n0"]
    error = abs(middle["nB_0minus_over_n0"]-predicted)
    speed_ok = all(np.isfinite(p["velocity_m_s"]) and abs(p["velocity_m_s"]/level-1) <= SPEED_RTOL
                   for p in (left,middle,right))
    return bool(speed_ok and error <= DENSITY_TOL), float(error)


class TargetRun:
    def __init__(self, directory, *, domain, numerical, repair, roots, identity, n0, physics,
                 profile_directory=None, trial=hard_trial, controls=None, evaluation_saved=None):
        self.directory = Path(directory)
        self.output = self.directory / "isothermal-targeted.npy"
        self.profile_directory = Path(profile_directory or OLD / "isothermal-contour-repair-profiles")
        self.domain, self.numerical, self.repair = domain, numerical, repair
        self.n0, self.physics, self.trial = n0, physics, trial
        self.controls = dict(CONTROLS, **(controls or {}))
        self.lock = threading.RLock()
        self.evaluation_saved = evaluation_saved
        self.seed_points = [dict(p, profile_origin="source") for p in repair.get("points", {}).values()]
        expected = dict(identity=identity, controls=self.controls, phase_budgets=PHASE_BUDGETS,
                        max_calls=MAX_CALLS, max_target_calls=MAX_TARGET_CALLS,
                        temperature_floor=TEMPERATURE_FLOOR, density_tolerance=DENSITY_TOL,
                        speed_tolerance=SPEED_RTOL, max_refinement_depth=MAX_REFINEMENT_DEPTH)
        if self.output.exists():
            self.payload = load(self.output)
            for k,v in expected.items():
                if self.payload.get(k) != v:
                    raise ValueError(f"Targeted resume identity/config mismatch: {k}")
            for event in self.payload["evaluations"]:
                if event["status"] == "running":
                    event.update(status="interrupted", runtime_s=self.controls["trial_timeout"])
        else:
            self.payload = dict(expected, payload_kind="isothermal_targeted", schema_version=1,
                                source_roots=copy.deepcopy(roots), points={}, targets={}, evaluations=[],
                                validation={}, finished_phases=[], run_complete=False)
        self.save()

    def save(self):
        with self.lock:
            self.payload["updated_unix_s"] = time.time()
            atomic_save(self.output, self.payload)

    def roots(self):
        with self.lock:
            roots = list(self.payload["source_roots"])
            roots += [dict(t["root"],level=t["level"]) for t in self.payload["targets"].values() if t.get("root")]
        unique = {(p["level"],p["T_0minus"]):p for p in roots}
        return sorted(unique.values(), key=lambda p:(p["level"],p["T_0minus"]))

    def profile_path(self, point):
        name = point.get("profile_file")
        if not name:
            return None
        return (self.directory / name if point.get("profile_origin") == "targeted"
                else self.profile_directory / Path(name).name)

    def nearest_seed(self, point):
        with self.lock:
            candidates = self.seed_points + list(self.payload["points"].values())
        candidates = [p for p in candidates if self.profile_path(p) is not None and self.profile_path(p).is_file()]
        if not candidates:
            return None
        return min(candidates, key=lambda p: (np.log(p["T_0minus"]/point["T_0minus"])/.2)**2
                   + ((p["nB_0minus_over_n0"]-point["nB_0minus_over_n0"])/.1)**2)

    def reserve(self, target_id, phase, point, purpose):
        """Persist reservations before spawn; interruption cannot reset the allowance."""
        with self.lock:
            events = self.payload["evaluations"]
            limit = 4 if phase == "validation" else PHASE_BUDGETS[phase]
            used = [e for e in events if e["target_id"] == target_id]
            if (len(events) >= MAX_CALLS or sum(e["phase"] == phase for e in events) >= limit
                    or len(used) >= MAX_TARGET_CALLS
                    or sum(e.get("runtime_s",0.) for e in used) >= TARGET_SECONDS):
                return None
            events.append(dict(target_id=target_id, phase=phase, point=dict(point), purpose=purpose,
                               status="running", result=None, runtime_s=0.))
            self.save()
            return len(events)-1

    def point(self, temperature, density):
        if not (TEMPERATURE_FLOOR <= temperature <= 120. and 1. <= density <= 5.5):
            raise ValueError("Shot outside the physical window")
        return dict(id=physical_id(temperature,density), T_0minus=float(temperature),
                    nB_0minus_over_n0=float(density), nB_0minus=float(density*self.n0))

    def shoot(self, task, point, *, controls=None, purpose="solve", confirm=None):
        with self.lock:
            old = self.payload["points"].get(point["id"])
            if old is not None and controls is None and confirm is None:
                return old, "cached"
        seed = confirm or self.nearest_seed(point)
        if seed is None:
            return None, "missing_profile"
        profile = self.profile_path(seed)
        try:
            continuation = load(profile)
        except (OSError,ValueError) as exc:
            return None, f"profile_error: {exc}"
        phase = "validation" if controls is not None else task["phase"]
        token = self.reserve(task["id"], phase, point, purpose)
        if token is None:
            return None, "budget_exhausted"
        current_controls = controls or self.controls
        guess = point["nB_0minus"]*task["level"]/np.sqrt(C*C-task["level"]**2)
        bounds = current_bracket(guess, seed["jB"]*point["nB_0minus"]/seed["nB_0minus"])
        if confirm is not None:
            bounds = (bounds[0]/4., bounds[1]*4.)
        with self.lock:
            spent = sum(e.get("runtime_s",0) for e in self.payload["evaluations"] if e["target_id"] == task["id"])
        timeout = min(self.controls["trial_timeout"], max(.001,TARGET_SECONDS-spent))
        start = time.monotonic()
        try:
            result = self.trial(timeout=timeout, point=point, physics=self.physics, controls=current_controls,
                                jB_guess=guess, bounds=bounds, continuation=continuation)
        except Exception as exc:
            result = dict(success=False,status="exception",message=f"{type(exc).__name__}: {exc}")
        result = dict(result)
        state = result.pop("continuation_state",None)
        if result.get("success") and (state is None or not np.isfinite(result.get("velocity_m_s",np.nan))):
            result.update(success=False,status="invalid_continuation_result")
        answer = None
        with self.lock:
            event = self.payload["evaluations"][token]
            event.update(status=result.get("status","unknown"), result=result,
                         runtime_s=time.monotonic()-start, seed_point_id=seed["id"], jB_bounds=bounds)
            if result.get("success"):
                name = "profiles/" + hashlib.sha256(point["id"].encode()).hexdigest()[:24] + ".npy"
                # Tighter validation profiles must not overwrite an accepted seed.
                if controls is not None:
                    answer = dict(point,velocity_m_s=result["velocity_m_s"],jB=result["jB"],result=result)
                else:
                    atomic_save(self.directory / name,state)
                    answer = dict(point,velocity_m_s=result["velocity_m_s"],jB=result["jB"],result=result,
                                  profile_file=name,profile_origin="targeted")
                    answer["near_current_bound"] = near_current_bound(answer["jB"],bounds)
                    if not answer["near_current_bound"]:
                        self.payload["points"][point["id"]] = answer
            self.save()
            if self.evaluation_saved is not None:
                self.evaluation_saved(copy.deepcopy(event), len(self.payload["evaluations"]))
        return answer, result.get("status","unknown")

    def solve_target(self, task):
        if task["status"] != "pending":
            return task
        temperature, level = task["T_0minus"], task["level"]
        # Cold extension targets use the newly converged warmer roots when available.
        if task["phase"] == "cold" and task["depth"] == 0 and task["left"] is None:
            prediction = make_target("cold",level,temperature,self.roots())
            task.update({k:prediction[k] for k in ("left","right","predicted_density")})
        predicted = task["predicted_density"]
        density = predicted
        samples = [p for p in self.roots() if p["level"]==level and abs(p["T_0minus"]-temperature)<1.e-8]
        failures = []
        with self.lock:
            previous = [e for e in self.payload["evaluations"] if e["target_id"] == task["id"]]
            for e in previous:
                if e.get("result",{} ) and e["result"].get("success"):
                    p = self.payload["points"].get(e["point"]["id"])
                    if p is not None:
                        samples.append(p)
                else:
                    failures.append(e["point"]["nB_0minus_over_n0"])
        root = next((p for p in samples if abs(p["velocity_m_s"]/level-1) <= SPEED_RTOL),None)
        reason = "target_call_limit"
        for attempt in range(MAX_TARGET_CALLS):
            if root is not None:
                break
            if attempt or samples or failures:
                density = self.propose_density(samples,failures,predicted,level)
                if density is None:
                    reason = "local_density_bracket_unresolved"
                    break
            point = self.point(temperature,density)
            answer, reason = self.shoot(task,point)
            if answer is not None and answer.get("near_current_bound"):
                answer, reason = self.shoot(task,point,confirm=answer,purpose="current_bound_confirmation")
                if answer is not None and answer.get("near_current_bound"):
                    answer, reason = None, "current_bound_unconfirmed"
            if answer is None:
                failures.append(density)
                if reason == "budget_exhausted" or reason.startswith("missing_profile") or reason.startswith("profile_error"):
                    break
                # One alternative density after a failed trial; never recurse or retry it.
                continue
            samples.append(answer)
            if abs(answer["velocity_m_s"]/level-1) <= SPEED_RTOL:
                root = dict(answer,level=level)
        with self.lock:
            task["root"] = root
            if root is not None:
                task["certified"], task["density_error"] = certify_interval(task["left"],root,task["right"],level)
                task["status"] = "success" if task["certified"] or task["left"] is None or task["right"] is None else "needs_refinement"
            else:
                task.update(status="failed",message=reason)
            self.payload["targets"][task["id"]] = task
            self.save()
        return task

    @staticmethod
    def propose_density(samples, failures, predicted, level):
        ordered = sorted({p["nB_0minus_over_n0"]:p for p in samples}.values(),key=lambda p:p["nB_0minus_over_n0"])
        tried = [p["nB_0minus_over_n0"] for p in ordered]+failures
        bracket = next(((a,b) for a,b in zip(ordered[:-1],ordered[1:])
                        if (a["velocity_m_s"]-level)*(b["velocity_m_s"]-level)<0),None)
        if bracket or len(ordered)>=2:
            a,b = bracket or tuple(ordered[-2:])
            delta = b["velocity_m_s"]-a["velocity_m_s"]
            guess = a["nB_0minus_over_n0"] + (level-a["velocity_m_s"])/delta*(b["nB_0minus_over_n0"]-a["nB_0minus_over_n0"]) if abs(delta)>1.e-12 else predicted
            if bracket:
                lo,hi=a["nB_0minus_over_n0"],b["nB_0minus_over_n0"]
                guess=float(np.clip(guess,lo+.05*(hi-lo),hi-.05*(hi-lo)))
        elif ordered:
            last=ordered[-1]
            guess=last["nB_0minus_over_n0"]+(.02 if last["velocity_m_s"]<level else -.02)
        else:
            guess=predicted + (.01 if len(failures)%2 else -.01)
        guess=float(np.clip(guess,max(1.,predicted-.15),min(5.5,predicted+.15)))
        return None if any(abs(guess-d)<1.e-10 for d in tried) else guess

    def add_children(self, task):
        if not task.get("root") or task["certified"] or task["children"] or task["depth"]>=MAX_REFINEMENT_DEPTH:
            return
        root=task["root"]
        pairs = [(task["left"],root),(root,task["right"])]
        for a,b in pairs:
            if a is None or b is None:
                continue
            temperature=.5*(a["T_0minus"]+b["T_0minus"])
            if temperature < TEMPERATURE_FLOOR:
                continue
            child=make_target(task["phase"],task["level"],temperature,self.roots(),left=a,right=b,depth=task["depth"]+1)
            if child["id"] not in self.payload["targets"]:
                self.payload["targets"][child["id"]]=child
            task["children"].append(child["id"])

    def seed_cold_connections(self):
        if self.payload.get("cold_connections_seeded"):
            return
        roots=self.roots()
        for level in (50.,100.,200.,500.,1000.,2000.):
            selected=[p for p in roots if p["level"]==level and TEMPERATURE_FLOOR<=p["T_0minus"]<=10.]
            below=[p for p in roots if p["level"]==level and p["T_0minus"]<TEMPERATURE_FLOOR]
            if below and selected and selected[0]["T_0minus"]>TEMPERATURE_FLOOR:
                task=make_target("cold",level,TEMPERATURE_FLOOR,roots,left=below[-1],right=selected[0],depth=1)
                self.payload["targets"].setdefault(task["id"],task)
            for a,b in zip(selected[:-1],selected[1:]):
                task=make_target("cold",level,.5*(a["T_0minus"]+b["T_0minus"]),roots,left=a,right=b,depth=1)
                self.payload["targets"].setdefault(task["id"],task)
        self.payload["cold_connections_seeded"]=True
        self.save()

    def run_phase(self, phase, workers, completed=lambda task,payload:None):
        # A process may stop after saving a completed root but before enqueuing
        # its local midpoint checks. Rebuild that deterministic queue on resume.
        with self.lock:
            for task in list(self.payload["targets"].values()):
                if task["phase"]==phase and task["status"]!="pending":
                    self.add_children(task)
            self.save()
        while True:
            if phase == "cold" and all(t["status"]!="pending" for t in self.payload["targets"].values() if t["phase"]==phase and t["depth"]==0):
                self.seed_cold_connections()
            pending=[t for t in self.payload["targets"].values() if t["phase"]==phase and t["status"]=="pending"]
            if not pending:
                break
            # One target per level at a time: later shots reuse the new profile.
            chosen={}
            for task in sorted(pending,key=lambda t:(t["depth"],-t["T_0minus"])):
                chosen.setdefault(task["level"],task)
            with ThreadPoolExecutor(max_workers=max(1,min(workers,len(chosen)))) as pool:
                futures={pool.submit(self.solve_target,t):t for t in chosen.values()}
                for future in as_completed(futures):
                    task=futures[future]
                    try:
                        task=future.result()
                    except Exception as exc:
                        task.update(status="failed",message=f"{type(exc).__name__}: {exc}")
                    with self.lock:
                        self.add_children(task)
                        self.save()
                        snapshot=copy.deepcopy(self.payload)
                    completed(task,snapshot)
            with self.lock:
                used=sum(e["phase"]==phase for e in self.payload["evaluations"])
                if used>=PHASE_BUDGETS[phase]:
                    break
        with self.lock:
            for task in self.payload["targets"].values():
                if task["phase"]==phase and task["status"]=="pending":
                    task.update(status="deferred",message="Phase BVP-call allowance exhausted")
            if phase not in self.payload["finished_phases"]:
                self.payload["finished_phases"].append(phase)
            self.save()

    def validate(self, phase):
        categories={"warm":(("warm_500",500.),("warm_1000",1000.)),
                    "bend":(("bend_2000",2000.),),"cold":(("cold_1000",1000.),)}
        for name,level in categories[phase]:
            if name in self.payload["validation"]:
                continue
            saved = [e for e in self.payload["evaluations"] if e["target_id"] == "check:"+name]
            if saved:
                # The check itself may have finished just before interruption of
                # its summary write. Never spend another reserved check on it.
                event=saved[-1]
                result=event.get("result") or {}
                reference=event["point"].get("velocity_m_s",np.nan)
                difference=abs(result.get("velocity_m_s",np.nan)/reference-1.) if result.get("success") else np.nan
                self.payload["validation"][name]=dict(status="success" if np.isfinite(difference) and difference<=.01 else "failed",
                    relative_difference=difference,point_id=event["point"]["id"],message=event["status"])
                self.save()
                continue
            tasks=[t for t in self.payload["targets"].values() if t["phase"]==phase and t["level"]==level and t.get("root")]
            if not tasks:
                self.payload["validation"][name]=dict(status="unresolved",message="No recovered root in this group")
                self.save();continue
            task=min(tasks,key=lambda t:t["T_0minus"]) if phase=="cold" else max(tasks,key=lambda t:t["T_0minus"])
            root=task["root"]
            check=dict(task,id="check:"+name)
            controls=dict(self.controls,n_mesh=2*self.controls["n_mesh"],tail_eps=self.controls["tail_eps"]/10.,tol_bvp=self.controls["tol_bvp"]/5.)
            answer,reason=self.shoot(check,root,controls=controls,purpose="accuracy_check")
            difference=abs(answer["velocity_m_s"]/root["velocity_m_s"]-1) if answer else np.nan
            self.payload["validation"][name]=dict(status="success" if np.isfinite(difference) and difference<=.01 else "failed",
                relative_difference=difference, target_id=task["id"],message=reason)
            self.save()
