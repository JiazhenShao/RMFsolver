"""Bounded contour targeting, continuation scheduling and single-writer ledger."""
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import time

import numpy as np

from _logT_common import (CONTROLS, CHECK_CONTROLS, LEVELS, TEMPERATURES, C, atomic_save,
                         load, point, target, temperature_id, anchor_ids, block_chains,
                         propose_density, current_bounds, near_bound)


class ContourRun:
    def __init__(self, directory, inputs, identity, *, controls=None, trial=None, proposal=None,
                 evaluation_saved=None, target_saved=None, boundary_saved=None, filename="isothermal-logT.npy"):
        self.directory = Path(directory)
        self.input_directory = self.directory
        self.profile_cache = {}
        self.output = self.directory/filename
        self.inputs, self.identity = inputs, identity
        self.controls = dict(CONTROLS, **(controls or {}))
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.trial = trial
        self.proposal = proposal
        self.evaluation_saved = evaluation_saved or (lambda e: None)
        self.target_saved = target_saved or (lambda t: None)
        self.boundary_saved = boundary_saved or (lambda row: None)
        self.payload = dict(schema=1, identity=identity, controls=self.controls,
                            domain_fingerprint=inputs.get("fingerprint"), targets={}, boundaries={},
                            evaluations=[], auxiliary=[], points={}, validation={}, work_complete=False,
                            contours_complete=False, updated_unix_s=time.time())
        if self.output.exists():
            self.payload = load(self.output)
            if self.payload["identity"] != identity or self.payload["controls"] != self.controls:
                raise ValueError("Checkpoint identity/controls changed; use a separate output directory")
            for e in self.payload["evaluations"]:
                if e["status"] == "running":
                    e.update(status="interrupted", runtime_s=e["reserved_seconds"])
            for t in self.payload["targets"].values():
                if t["status"] == "running":
                    t["status"] = "pending"
            for key,v in list(self.payload['validation'].items()):
                if v['status'] != 'running':
                    continue
                events = [e for e in self.payload['evaluations'] if e['target_id']==key]
                if not events:
                    # The controller died before reserving/launching this check.
                    del self.payload['validation'][key]
                elif events[-1]['status']=='success':
                    result = events[-1]['result']
                    base = self.payload['targets'][v['base_target']]
                    difference = abs(result['velocity_m_s']/base['root']['velocity_m_s']-1.)
                    valid = difference < .01 and not near_bound(result['jB'],result['current_bounds'])
                    v.update(status='success' if valid else 'failed',relative_speed_change=difference,
                             result=result,recovered_from_evaluation=events[-1]['id'])
                else:
                    v['status'] = 'interrupted'
            for row in list(self.payload['boundaries'].values())+self.payload['auxiliary']:
                if row.get('status') == 'running':
                    row['status'] = 'interrupted'
        self.save()

    def save(self):
        with self.lock:
            self.payload["updated_unix_s"] = time.time()
            atomic_save(self.output, self.payload)

    def points(self):
        with self.lock:
            return list(self.inputs["points"]) + list(self.payload["points"].values())

    def spent(self, task):
        return task.get("aux_runtime_s", 0.) + sum(e.get("runtime_s", e["reserved_seconds"])
            for e in self.payload["evaluations"] if e["target_id"] == task["id"])

    def can_attempt(self, task):
        with self.lock:
            events = self.payload["evaluations"]
            is_check = task["kind"] == "validation"
            checked = sum(e["kind"] == "validation" for e in events)
            limit = self.controls["max_attempts"] - (0 if is_check else max(0, 14-checked))
            prior = [e for e in events if e["target_id"] == task["id"]]
            remaining = self.controls["target_timeout"] - self.spent(task)
            return (len(events) < limit and len(prior) < (1 if is_check else self.controls['attempts_per_target'])
                    and remaining > 0)

    def reserve(self, task, physical):
        with self.lock:
            events = self.payload['evaluations']
            if (not self.can_attempt(task) or sum(e['point']['id']==physical['id'] for e in events)
                    >= self.controls['attempts_per_point']):
                return None
            remaining = self.controls['target_timeout']-self.spent(task)
            event = dict(id=len(events), target_id=task["id"], kind=task["kind"], point=physical,
                         status="running", reserved_seconds=min(self.controls["trial_timeout"], remaining),
                         started_unix_s=time.time())
            events.append(event)
            self.save()  # A crashed controller cannot erase a launched attempt.
            return event["id"]

    def profile(self, p):
        if "continuation_state" in p:
            return p["continuation_state"]
        name = p.get("profile_file")
        if name:
            path = (self.input_directory if name.startswith('inputs/') else self.directory)/name
            if path.is_file():
                if path not in self.profile_cache:
                    self.profile_cache[path] = load(path)
                return self.profile_cache[path]
        return None

    def seed(self, physical):
        candidates = [p for p in self.points() if self.profile(p) is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda p: (np.log(p["T_0minus"]/physical["T_0minus"])/.3)**2
                   + ((p["nB_0minus_over_n0"]-physical["nB_0minus_over_n0"])/.1)**2)

    def shoot(self, task, physical, *, expand=False, validation=False):
        with self.condition:
            while any(e["point"]["id"] == physical["id"] and e["status"] == "running"
                      for e in self.payload["evaluations"]):
                self.condition.wait(timeout=self.controls["trial_timeout"]+1.)
            if not expand and not validation and physical["id"] in self.payload["points"]:
                cached = self.payload["points"][physical["id"]]
                if not cached.get("near_current_bound", False):
                    return cached
            token = self.reserve(task, physical)
        if token is None:
            return None
        started = time.monotonic()
        try:
            source = self.seed(physical)
            continuation = self.profile(source) if source else None
            guesses = [physical["nB_0minus"]*task["level"]/C]
            if source:
                guesses.append(source["jB"]*physical["nB_0minus"]/source["nB_0minus"])
            bounds = current_bounds(guesses)
            if expand:
                bounds = (bounds[0]/4., bounds[1]*4.)
            controls = dict(self.controls, **(CHECK_CONTROLS if validation else {}))
            if self.trial is None:
                from _logT_trial import bounded_call
                result = bounded_call("numerical", timeout=self.payload["evaluations"][token]["reserved_seconds"],
                    point=physical, controls=controls, continuation=continuation,
                    jB_guess=guesses[-1], bounds=bounds, use_analytic=source is None, expand=expand)
            else:
                result = self.trial(point=physical, controls=controls, continuation=continuation,
                                    jB_guess=guesses[-1], bounds=bounds)
        except Exception as exc:
            result = dict(success=False, status="exception", message=f"{type(exc).__name__}: {exc}")
        with self.condition:
            event = self.payload["evaluations"][token]
            result = dict(result)
            profile = result.pop("continuation_state", None)
            valid = bool(result.get("success") and result.get("validated") and
                         np.isfinite(result.get("velocity_m_s", np.nan)) and result["velocity_m_s"] > 0)
            accepted = None
            if valid:
                # The child may augment bounds with the analytical current.
                actual_bounds = result.get("current_bounds", bounds)
                result['current_bounds'] = actual_bounds
                accepted = dict(physical, velocity_m_s=float(result["velocity_m_s"]),
                                jB=float(result["jB"]), result=result,
                                near_current_bound=near_bound(result["jB"], actual_bounds), source="new_numerical")
                if profile is not None:
                    name = f"profiles/evaluation-{token:05d}.npy"
                    atomic_save(self.directory/name, profile)
                    accepted["profile_file"] = name
                # Accuracy checks never overwrite their original accepted point.
                if not validation:
                    self.payload["points"][physical["id"]] = accepted
            event.update(status="success" if valid else result.get("status", "failed"),
                         runtime_s=time.monotonic()-started, result=result,
                         profile_file=accepted.get("profile_file") if accepted else None)
            self.save()
            self.evaluation_saved(copy.deepcopy(event))
            self.condition.notify_all()
            return accepted

    def prediction(self, task, limits):
        if self.proposal:
            return self.proposal(task, self.points(), limits)
        T, level = task["T_0minus"], task["level"]
        points = self.points()
        same_level = sorted([p for p in points if abs(p["velocity_m_s"]/level-1) <= .005],
                            key=lambda p: p["T_0minus"])
        if same_level and same_level[0]["T_0minus"] <= T <= same_level[-1]["T_0minus"]:
            return float(np.interp(np.log(T), np.log([p["T_0minus"] for p in same_level]),
                                   [p["nB_0minus_over_n0"] for p in same_level]))
        nearby = [p for p in points if abs(np.log(p["T_0minus"]/T)) < np.log(1.5)]
        if nearby:
            best_T = min(nearby, key=lambda p: abs(np.log(p["T_0minus"]/T)))["T_0minus"]
            row = sorted([p for p in nearby if p["T_0minus"] == best_T], key=lambda p:p["velocity_m_s"])
            if len(row) >= 2 and row[0]["velocity_m_s"] <= level <= row[-1]["velocity_m_s"]:
                return float(np.interp(level, [p["velocity_m_s"] for p in row], [p["nB_0minus_over_n0"] for p in row]))
        from _logT_trial import bounded_call
        with self.lock:
            allowance = min(60., self.controls['target_timeout']-self.spent(task))
            if allowance <= 0:
                raise RuntimeError('Target time budget exhausted before analytical proposal')
            record = dict(task_id=task['id'],purpose='analytical_density_proposal',
                          status='running',reserved_seconds=allowance)
            self.payload['auxiliary'].append(record)
            task['aux_runtime_s'] = task.get('aux_runtime_s',0.)+allowance
            self.save()
        answer = bounded_call("proposal", timeout=allowance,T=T, level=level, limits=limits, n0=self.inputs["n0"])
        with self.lock:
            task['aux_runtime_s'] += answer['runtime_s']-allowance
            record.update(status=answer['status'],result=answer)
            self.save()
        if not answer.get("success"):
            raise RuntimeError("Analytical proposal failed: " + answer.get("message", answer.get("status", "")))
        return answer["density"]

    def solve_target(self, task_id):
        task = self.payload["targets"][task_id]
        if task["status"] != "pending":
            return task
        try:
            boundary = self.payload["boundaries"].get(temperature_id(task["T_0minus"]), {})
            if boundary.get("status") != "success":
                raise RuntimeError("No validated cold boundary for this temperature")
            limits = (boundary["lower"], boundary["upper"])
            with self.lock:
                task["status"] = "running"
                self.save()
            samples = [p for p in self.points() if p["T_0minus"] == task["T_0minus"]
                       and limits[0] < p['nB_0minus_over_n0'] < limits[1]
                       and not p.get("near_current_bound", False)]
            failures = [e["point"]["nB_0minus_over_n0"] for e in self.payload["evaluations"]
                        if e["target_id"] == task_id and e["status"] != "success"]
            found = next((p for p in samples if abs(p["velocity_m_s"]/task["level"]-1) <= self.controls["speed_rtol"]), None)
            if found is None and not self.can_attempt(task):
                raise RuntimeError('Persistent numerical budget exhausted; no proposal or BVP launched')
            predicted = self.prediction(task, limits) if found is None else found["nB_0minus_over_n0"]
            for _ in range(self.controls["attempts_per_target"]):
                if found is not None:
                    break
                density = propose_density(samples, failures, predicted, task["level"], limits)
                if density is None:
                    break
                physical = point(task["T_0minus"], density, self.inputs["n0"])
                answer = self.shoot(task, physical)
                if answer and answer["near_current_bound"]:
                    answer = self.shoot(task, physical, expand=True)
                    if answer and answer["near_current_bound"]:
                        answer = None
                if answer is None:
                    failures.append(density)
                else:
                    samples.append(answer)
                    if abs(answer["velocity_m_s"]/task["level"]-1) <= self.controls["speed_rtol"]:
                        found = answer
            task.update(root=found, status="success" if found else "unresolved")
            if found is None:
                task["message"] = "No accepted root within the persistent point/target/run budgets"
        except Exception as exc:
            task.update(status="unresolved", message=f"{type(exc).__name__}: {exc}")
        with self.lock:
            self.save()
            self.target_saved(copy.deepcopy(task))
        return task

    def run_base(self, workers):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            anchors = [i for i in self.payload["targets"] if i in anchor_ids()]
            for future in as_completed([pool.submit(self.solve_target, i) for i in anchors]):
                future.result()
            def chain(ids):
                for i in ids:
                    if i in self.payload["targets"]:
                        self.solve_target(i)
            for future in as_completed([pool.submit(chain, ids) for ids in block_chains()]):
                future.result()

    def schedule_refinements(self):
        with self.lock:
            for level in LEVELS:
                rows = [self.payload["targets"].get(target(level,T)["id"]) for T in TEMPERATURES]
                flagged = set()
                for i,(a,b) in enumerate(zip(rows[:-1], rows[1:])):
                    if a is None or b is None:
                        continue
                    if a["status"] != "success" or b["status"] != "success" or abs(
                            a["root"]["nB_0minus_over_n0"]-b["root"]["nB_0minus_over_n0"]) > .05:
                        flagged.add(i)
                for i in range(1,30):
                    triple = rows[i-1:i+2]
                    if all(t and t["status"] == "success" for t in triple):
                        a,b,c = [t["root"]["nB_0minus_over_n0"] for t in triple]
                        if abs(b-(a+c)/2.) > .01:
                            flagged.update([i-1,i])
                for i in sorted(flagged):
                    t = target(level, np.sqrt(TEMPERATURES[i]*TEMPERATURES[i+1]), "refinement")
                    t["original_interval"] = i
                    self.payload["targets"].setdefault(t["id"], t)
            self.save()

    def validate_target(self, base):
        key = "check:"+base["id"]
        if key in self.payload["validation"]:
            return
        record = dict(status="running", base_target=base["id"])
        with self.lock:
            self.payload["validation"][key] = record
            self.save()
        if base["status"] != "success":
            record.update(status="blocked", message="No converged base target")
        else:
            check = dict(base, id=key, kind="validation")
            answer = self.shoot(check, base["root"], validation=True)
            difference = abs(answer["velocity_m_s"]/base["root"]["velocity_m_s"]-1) if answer else np.inf
            record.update(status="success" if answer and not answer["near_current_bound"] and difference < .01 else "failed",
                          relative_speed_change=float(difference), result=answer)
        with self.lock:
            self.save()
            self.target_saved(dict(record, kind="validation", id=key))

    def finalize(self):
        with self.lock:
            tasks = list(self.payload["targets"].values())
            self.payload["work_complete"] = all(t["status"] not in ("pending", "running") for t in tasks)
            self.payload["contours_complete"] = bool(tasks and all(t["status"] == "success" for t in tasks)
                and len(self.payload["validation"]) == 14
                and all(v["status"] == "success" for v in self.payload["validation"].values())
                and all(j['status']=='joined' for j in self.payload.get('plot_diagnostics',{}).get('joins',{}).values()))
            self.save()
