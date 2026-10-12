"""Numerical contour tracing with immutable baseline points and profile continuation."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import inspect
from pathlib import Path
import threading
import time

import numpy as np

from _repair_common import (
    DEFAULT_CONTROLS, LEVELS, SCHEMA, atomic_save, current_bracket,
    hard_trial, load, near_current_bound, sha256,
)


def point_id(theta, fraction):
    return f"{float(theta):.14g}:{float(fraction):.14g}"


def ray_id(level, theta):
    return f"{float(level):g}:{float(theta):.14g}"


def logit(value):
    return float(np.log(value) - np.log1p(-value))


def logistic(value):
    return float(1. / (1. + np.exp(-np.clip(value, -700., 700.))))


def predictor_extrapolate(value, axis, values):
    """Extrapolation is only a seed; accepted velocities always come from BVPs."""
    index = int(np.clip(np.searchsorted(axis, value) - 1, 0, len(axis) - 2))
    if len(axis) == 1:
        return float(values[0])
    return float(values[index] + (values[index + 1] - values[index])
                 * (value - axis[index]) / (axis[index + 1] - axis[index]))


def seed_distance(first, second):
    angular = abs(np.log((first["theta_rad"] + 1.e-4) / (second["theta_rad"] + 1.e-4)))
    radial = abs(logit(first["radial_fraction"]) - logit(second["radial_fraction"]))
    return angular + radial


def angular_midpoint(left, right):
    if 0 < left < right <= np.radians(20.):
        return float(np.sqrt(left * right))
    return .5 * (left + right)


def midpoint(first, second):
    lo, hi = sorted((first["theta_rad"], second["theta_rad"]))
    return angular_midpoint(lo, hi), logistic(.5 * (logit(first["radial_fraction"]) + logit(second["radial_fraction"])))


def within_spacing(first, second, controls):
    low, high = sorted((first["T_0minus"], second["T_0minus"]))
    return bool(
        abs(first["nB_0minus_over_n0"] - second["nB_0minus_over_n0"]) <= controls["density_step"]
        and high - low <= controls["temperature_step"]
        and (low >= 1. or high / low <= controls["cold_ratio"])
    )


def connectable(first, second, controls):
    """Veto a gross discontinuity between two angularly adjacent roots.

    This is not a sampling-density test: `within_spacing` above answers that
    question for the search, in units of its own continuation steps.  Here the
    rays are already neighbours on the scheduled angular axis, so the only
    thing left to reject is a jump too large to be the same branch.
    """
    radial = (abs(second["radial_fraction"] - first["radial_fraction"])
              if "radial_fraction" in first and "radial_fraction" in second else 0.)
    return bool(
        abs(second["nB_0minus_over_n0"] - first["nB_0minus_over_n0"])
        <= controls.get("connect_density_jump", DEFAULT_CONTROLS["connect_density_jump"])
        and abs(second["T_0minus"] - first["T_0minus"])
        <= controls.get("connect_temperature_jump", DEFAULT_CONTROLS["connect_temperature_jump"])
        and radial <= controls.get("connect_radial_jump", DEFAULT_CONTROLS["connect_radial_jump"])
    )


def brackets(samples, level):
    """Keep every observed crossing; no assumed radial monotonicity."""
    ordered = sorted(samples, key=lambda row: row["radial_fraction"])
    return [(a, b) for a, b in zip(ordered[:-1], ordered[1:])
            if (a["velocity_m_s"] - level) * (b["velocity_m_s"] - level) < 0.]


def observed_roots(samples, level, tolerance):
    groups, active = [], []
    for point in sorted(samples, key=lambda p: p["radial_fraction"]):
        if abs(point["velocity_m_s"] / level - 1.) <= tolerance:
            active.append(point)
        elif active:
            groups.append(active); active = []
    if active:
        groups.append(active)
    return [min(group, key=lambda p: abs(p["velocity_m_s"] / level - 1.)) for group in groups]


def root_proposal(left, right, level):
    low, high = logit(left["radial_fraction"]), logit(right["radial_fraction"])
    a, b = np.log(left["velocity_m_s"] / level), np.log(right["velocity_m_s"] / level)
    weight = float(np.clip(-a / (b - a), .1, .9))
    return logistic(low + weight * (high - low))


class Repair:
    def __init__(self, directory, baseline, hashes, *, controls=None, progress=None,
                 trial=hard_trial, solver_identity=None, output_name="isothermal-contour-repair.npy"):
        self.directory = Path(directory)
        self.domain, self.analytic, self.numerical = baseline
        self.hashes = hashes
        self.controls = dict(DEFAULT_CONTROLS, **(controls or {}))
        self.progress = progress
        self.trial = trial
        self.lock = threading.RLock()
        self.inflight = {}
        self.attempted = set()
        self.output = self.directory / output_name
        self.profiles = self.directory / (self.output.stem + "-profiles")
        self.physics = {key: self.domain[key] for key in ("B_one_forth_MeV", "ms_MeV", "NM_type", "upB", "xi")}
        self.theta = np.asarray(self.domain["theta_axis_rad"], float)
        self.fractions = np.asarray(self.domain["radial_fraction_axis"], float)
        self.n0 = float(np.median(self.domain["nB_0minus_grid"] / self.domain["nB_0minus_over_n0_grid"]))
        if solver_identity is None:
            from RMFsolver import phase_velocity as pv
            solver_identity = {"sha256": sha256(pv.__file__), "signature": str(inspect.signature(pv.solve_front_isothermal))}
        expected = dict(schema_version=SCHEMA, baseline_hashes=hashes,
                        domain_fingerprint=self.domain["domain_fingerprint"],
                        solver_identity=solver_identity, controls=self.controls,
                        workflow_identity={name: sha256(Path(__file__).with_name(name))
                                           for name in ("_repair_search.py", "_repair_common.py")})
        if self.output.exists():
            self.payload = load(self.output)
            for key, value in expected.items():
                if self.payload.get(key) != value:
                    raise ValueError(f"Repair resume mismatch: {key}; use a different output name")
        else:
            self.payload = dict(expected, payload_kind="isothermal_contour_repair", points={},
                                evaluations=[], rays={}, run_complete=False, contours_complete=False,
                                validation={}, baseline_success_count=int(np.sum(self.numerical["task_status"] == "success")))
        self.baseline_points = {}
        for point in self.payload["points"].values():
            if not (self.directory / point["profile_file"]).is_file():
                raise ValueError(f"Missing continuation profile: {point['profile_file']}")
        for i, j in zip(*np.where(self.numerical["task_status"] == "success")):
            point = self.coordinate(float(self.theta[i]), float(self.fractions[j]))
            result = self.numerical["records"][i, j]
            point.update(velocity_m_s=float(self.numerical["velocity_m_s_grid"][i, j]),
                         jB=float(result["jB"]), result=result, source="baseline", baseline_index=(int(i), int(j)))
            self.baseline_points[point["id"]] = point
        self.save()

    def save(self):
        with self.lock:
            self.payload["updated_unix_s"] = time.time()
            atomic_save(self.output, self.payload)

    def coordinate(self, theta, fraction):
        if not (0 <= theta <= np.pi / 2. and 0 < fraction < 1):
            raise ValueError("Coordinates must lie inside the saved polar domain")
        # Interpolate the same contour-ray boundary radii used for the original grid.
        inner = np.interp(theta, self.theta, self.domain["rho_inner_boundary"])
        outer = np.interp(theta, self.theta, self.domain["rho_outer_boundary"])
        rho = inner + fraction * (outer - inner)
        transform = self.domain["ray_transform"]
        density = transform["density_origin_over_n0"] + transform["density_scale_over_n0"] * rho * np.cos(theta)
        temperature = max(transform["temperature_min_MeV"], transform["temperature_scale_MeV"] * rho * np.sin(theta))
        if not (1. - 1.e-12 <= density <= 5.5 + 1.e-12 and .01 <= temperature <= 120. + 1.e-12):
            raise ValueError("Point exceeds the physical density/temperature window")
        return dict(id=point_id(theta, fraction), theta_rad=theta, radial_fraction=fraction,
                    nB_0minus_over_n0=float(density), T_0minus=float(temperature),
                    nB_0minus=float(density * self.n0))

    def analytic_current(self, point):
        currents = np.asarray(self.analytic["jB_grid"], float)
        rows = []
        for row in currents:
            good = np.isfinite(row) & (row > 0)
            rows.append(predictor_extrapolate(logit(point["radial_fraction"]),
                        np.array([logit(r) for r in self.fractions[good]]), np.log(row[good])) if good.any() else np.nan)
        rows = np.asarray(rows)
        good = np.isfinite(rows)
        return float(np.exp(np.clip(np.interp(point["theta_rad"], self.theta[good], rows[good]), -700., 700.))) if good.any() else np.nan

    def all_points(self):
        with self.lock:
            points = dict(self.payload["points"])
        # Accepted baseline scalars always take precedence over reconstructed anchors.
        points.update(self.baseline_points)
        return list(points.values())

    def samples(self, theta):
        return [p for p in self.all_points() if abs(p["theta_rad"] - theta) < 1.e-12]

    def nearest(self, point, *, profiles=False):
        with self.lock:
            candidates = list(self.payload["points"].values()) if profiles else self.all_points()
        if profiles:
            candidates = [p for p in candidates if p.get("profile_file")]
        if not candidates:
            return None
        return min(candidates, key=lambda p: seed_distance(p, point))

    def _record(self, point, result, *, purpose, seed, bounds, guess):
        result = dict(result)
        state = result.pop("continuation_state", None)
        if result.get("success") and state is None:
            result.update(success=False, status="missing_continuation", message="Accepted trial did not return a continuation state")
        base = self.baseline_points.get(point["id"])
        discrepancy = None
        if base is not None and result.get("success"):
            discrepancy = abs(result["velocity_m_s"] / base["velocity_m_s"] - 1.)
            if discrepancy > self.controls["anchor_rtol"]:
                result.update(success=False, status="anchor_discrepancy", message="Reconstructed anchor differs by more than 1%; original record preserved")
        with self.lock:
            event = dict(point=point, purpose=purpose, result=result, jB_guess=guess,
                         jB_bounds=tuple(bounds), seed_point_id=None if seed is None else seed["id"],
                         anchor_relative_difference=discrepancy)
            self.payload["evaluations"].append(event)
            accepted = None
            if result.get("success"):
                profile_name = hashlib.sha256(point["id"].encode()).hexdigest()[:24] + ".npy"
                atomic_save(self.profiles / profile_name, state)
                accepted = dict(point, velocity_m_s=float(result["velocity_m_s"]), jB=float(result["jB"]),
                                source=purpose, result=result,
                                profile_file=str(Path(self.profiles.name) / profile_name))
                accepted["current_bracket_near_bound"] = near_current_bound(accepted["jB"], bounds)
                # Publish only bracket-confirmed solutions to other continuation chains.
                if not accepted["current_bracket_near_bound"]:
                    self.payload["points"][point["id"]] = accepted
            self.save()
            if self.progress is not None:
                self.progress.update(1)
        return accepted

    def _trial(self, point, seed, estimate, bounds, deadline, purpose):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        continuation = load(self.directory / seed["profile_file"]) if seed and seed.get("profile_file") else None
        try:
            result = self.trial(timeout=min(self.controls["trial_timeout"], remaining), point=point,
                                physics=self.physics, controls=self.controls, jB_guess=estimate,
                                bounds=bounds, continuation=continuation)
        except Exception as exc:
            result = dict(success=False, status="exception", message=f"{type(exc).__name__}: {exc}")
        return self._record(point, result, purpose=purpose, seed=seed, bounds=bounds, guess=estimate)

    def evaluate(self, theta, fraction, deadline, *, seed=None, depth=0, anchor=False):
        """One owner per point; shared targets wait only within their own budget."""
        point = self.coordinate(theta, fraction)
        key = point["id"]
        with self.lock:
            if key in self.payload["points"]:
                return self.payload["points"][key]
            if key in self.attempted:
                future = self.inflight.get(key)
                owner = False
            else:
                self.attempted.add(key)
                future = Future()
                self.inflight[key] = future
                owner = True
        if not owner:
            if future is None or future.done():
                return future.result() if future else None
            try:
                return future.result(timeout=max(.001, deadline - time.monotonic()))
            except TimeoutError:
                return None
        answer = None
        try:
            if not anchor and seed is None:
                seed = self.nearest(point, profiles=True)
                baseline_seed = self.nearest(point)
                if baseline_seed is not None and baseline_seed["id"] != key and (
                    seed is None or seed_distance(baseline_seed, point) < .5 * seed_distance(seed, point)
                ):
                    closer = self.evaluate(baseline_seed["theta_rad"], baseline_seed["radial_fraction"],
                                           deadline, anchor=True)
                    if closer is not None:
                        seed = closer
            base = self.baseline_points.get(key)
            analytic = self.analytic_current(point)
            if anchor or base is not None:
                estimate = base["jB"] if base is not None else analytic
            elif seed is not None:
                old_analytic = self.analytic_current(seed)
                estimate = seed["jB"] * analytic / old_analytic if np.isfinite(analytic) and old_analytic > 0 else seed["jB"]
            else:
                estimate = analytic
            bounds = current_bracket(estimate, analytic)
            purpose = "anchor" if base is not None else ("continuation_bridge" if depth else "contour_evaluation")
            answer = self._trial(point, seed, estimate, bounds, deadline, purpose)
            if answer is not None:
                if near_current_bound(answer["jB"], bounds):
                    expanded = (bounds[0] / 4., bounds[1] * 4.)
                    confirmed = self._trial(point, answer, answer["jB"], expanded, deadline, "expanded_current_bracket")
                    if confirmed is None or near_current_bound(confirmed["jB"], expanded):
                        # Keep the finite solution, but never use it to certify a contour.
                        with self.lock:
                            self.payload["points"].pop(key, None)
                            self.save()
                        answer = None
                        return None
                    answer = confirmed
                return answer
            if base is not None or deadline <= time.monotonic():
                return None
            # Failure gets a smaller physical step before the second current guess.
            if seed is not None and depth < self.controls["continuation_depth"]:
                theta_mid, fraction_mid = midpoint(seed, point)
                mid_key = point_id(theta_mid, fraction_mid)
                if mid_key not in (key, seed["id"]):
                    middle = self.evaluate(theta_mid, fraction_mid, deadline, seed=seed, depth=depth + 1)
                    if middle is not None:
                        seed = middle
                        estimate = middle["jB"]
            elif np.isfinite(analytic):
                estimate = analytic
            bounds = current_bracket(estimate, analytic)
            answer = self._trial(point, seed, estimate, bounds, deadline, purpose)
            if answer is not None and near_current_bound(answer["jB"], bounds):
                with self.lock:
                    self.payload["points"].pop(key, None)
                    self.save()
                answer = None
            return answer
        finally:
            with self.lock:
                future.set_result(answer)
                self.inflight.pop(key, None)

    def _predict_fraction(self, theta, level):
        with self.lock:
            roots = [root for ray in self.payload["rays"].values() if ray["level"] == level for root in ray["roots"]]
        if roots:
            roots.sort(key=lambda p: abs(p["theta_rad"] - theta))
            return roots[0]["radial_fraction"]
        velocities = np.asarray(self.analytic["velocity_m_s_grid"], float)
        predicted = np.array([np.interp(theta, self.theta[np.isfinite(column)], column[np.isfinite(column)])
                              if np.isfinite(column).any() else np.nan for column in velocities.T])
        good = np.isfinite(predicted) & (predicted > 0)
        if not good.any():
            return .5
        fractions, speeds = self.fractions[good], predicted[good]
        crossing = np.flatnonzero((speeds[:-1] - level) * (speeds[1:] - level) <= 0.)
        if crossing.size:
            index = int(crossing[0])
        elif level < speeds[0]:
            index = 0
        else:
            index = max(0, len(speeds) - 2)
        if len(speeds) < 2 or speeds[index + 1] == speeds[index]:
            return float(fractions[index])
        weight = np.log(level / speeds[index]) / np.log(speeds[index + 1] / speeds[index])
        predicted_logit = logit(fractions[index]) + weight * (logit(fractions[index + 1]) - logit(fractions[index]))
        return float(np.clip(logistic(predicted_logit), 2.e-8, 1. - 2.e-8))

    def solve_ray(self, level, theta, *, force=False):
        key = ray_id(level, theta)
        with self.lock:
            existing = self.payload["rays"].get(key)
            if existing and existing["status"] == "success" and not force:
                return existing
        deadline = time.monotonic() + self.controls["target_timeout"]
        tried, invalid = set(), []
        roots = []
        reason = "target_budget_exhausted"
        for _ in range(self.controls["max_ray_evaluations"]):
            if time.monotonic() >= deadline:
                break
            samples = self.samples(theta)
            roots = observed_roots(samples, level, self.controls["speed_rtol"])
            crossing = brackets(samples, level)
            # A solved root accounts only for its local bracket, not another branch.
            pending = [(a, b) for a, b in crossing if not any(
                a["radial_fraction"] <= p["radial_fraction"] <= b["radial_fraction"] for p in roots)]
            if roots and not pending:
                reason = "success"
                break
            if pending:
                left, right = pending[0]
                fraction = root_proposal(left, right, level)
                if point_id(theta, fraction) in tried:
                    fraction = logistic(.5 * (logit(left["radial_fraction"]) + logit(right["radial_fraction"])))
            elif not samples:
                fraction = self._predict_fraction(theta, level)
            elif min(p["velocity_m_s"] for p in samples) > level:
                inner = min(samples, key=lambda p: p["radial_fraction"])["radial_fraction"]
                excluded = max([r for r in invalid if r < inner], default=0.)
                fraction = .5 * (excluded + inner)
            else:
                outer = max(samples, key=lambda p: p["radial_fraction"])["radial_fraction"]
                excluded = min([r for r in invalid if r > outer], default=1.)
                fraction = .5 * (outer + excluded)
            if not 1.e-8 < fraction < 1. - 1.e-8 or point_id(theta, fraction) in tried:
                reason = "unresolved_bracket"
                break
            tried.add(point_id(theta, fraction))
            result = self.evaluate(theta, fraction, deadline)
            if result is None:
                invalid.append(fraction)
                if pending:
                    reason = "bvp_failure_inside_bracket"
                    break
        roots = observed_roots(self.samples(theta), level, self.controls["speed_rtol"])
        pending = [(a, b) for a, b in brackets(self.samples(theta), level) if not any(
            a["radial_fraction"] <= p["radial_fraction"] <= b["radial_fraction"] for p in roots)]
        if roots and not pending:
            reason = "success"
        row = dict(level=float(level), theta_rad=float(theta), roots=roots, status=reason,
                   attempted_points=sorted(tried), runtime_s=self.controls["target_timeout"] - max(0., deadline - time.monotonic()))
        with self.lock:
            self.payload["rays"][key] = row
            self.save()
        return row

    def segments(self, level):
        return connected_segments(list(self.payload["rays"].values()), level, self.controls)

    def validate_representatives(self):
        """Independent tighter-tail/mesh checks; never replace the accepted speeds."""
        categories = {
            "hot_500": lambda r: r["level"] == 500. and r["theta_rad"] >= np.radians(75.),
            "bridge_1000": lambda r: r["level"] == 1000. and np.radians(40.) <= r["theta_rad"] <= np.radians(60.),
            "cold_200": lambda r: r["level"] == 200. and 0. < r["theta_rad"] <= float(self.theta[7]),
            "floor_50": lambda r: r["level"] == 50. and r["theta_rad"] == 0.,
        }
        for name, select in categories.items():
            if self.payload["validation"].get(name, {}).get("status") == "success":
                continue
            candidates = [p for r in self.payload["rays"].values() if r["status"] == "success" and select(r) for p in r["roots"]]
            if not candidates:
                self.payload["validation"][name] = dict(status="unresolved", message="No recovered contour point available")
                self.save()
                continue
            original = candidates[0]
            point = self.evaluate(original["theta_rad"], original["radial_fraction"], time.monotonic() + self.controls["target_timeout"])
            if point is None:
                self.payload["validation"][name] = dict(status="unresolved", message="Profile anchor unavailable")
                self.save()
                continue
            controls = dict(self.controls, n_mesh=2 * self.controls["n_mesh"],
                            tail_eps=self.controls["tail_eps"] / 10., tol_bvp=self.controls["tol_bvp"] / 5.)
            try:
                result = self.trial(timeout=self.controls["trial_timeout"], point=point, physics=self.physics,
                                    controls=controls, jB_guess=point["jB"], bounds=current_bracket(point["jB"]),
                                    continuation=load(self.directory / point["profile_file"]))
            except Exception as exc:
                result = dict(success=False, status="exception", message=f"{type(exc).__name__}: {exc}")
            result.pop("continuation_state", None)
            difference = abs(result.get("velocity_m_s", np.nan) / original["velocity_m_s"] - 1.)
            passed = bool(result.get("success") and difference <= .01)
            self.payload["validation"][name] = dict(status="success" if passed else "failed",
                point_id=point["id"], relative_difference=difference, result=result, controls=controls)
            self.payload["evaluations"].append(dict(point=original, purpose="convergence_check", result=result))
            self.save()
            if self.progress:
                self.progress.update(1)
        self.payload["validation_complete"] = all(r["status"] == "success" for r in self.payload["validation"].values())
        self.save()

    def _batch(self, tasks, worker_count):
        if not tasks:
            return
        with ThreadPoolExecutor(max_workers=min(worker_count, len(tasks))) as pool:
            futures = {pool.submit(self.solve_ray, level, theta): (level, theta) for level, theta in tasks}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    level, theta = futures[future]
                    with self.lock:
                        self.payload["rays"][ray_id(level, theta)] = dict(level=level, theta_rad=theta,
                            roots=[], status="exception", message=f"{type(exc).__name__}: {exc}")
                        self.save()

    def run(self, worker_count, *, smoke=False):
        self.payload["run_complete"] = False
        self.save()
        if smoke:
            tasks = [(500., np.pi / 2), (1000., np.radians(50.)), (200., float(self.theta[7])), (50., 0.)]
            self._batch(tasks, worker_count)
        else:
            warm = [float(t) for t in self.theta if t >= np.radians(20.)]
            self._batch([(level, theta) for level in (500., 1000.) for theta in warm], worker_count)
            # Temperature continuation moves from the known solutions toward the floor.
            for theta in reversed([float(t) for t in self.theta if t < np.radians(20.)]):
                self._batch([(level, theta) for level in LEVELS], worker_count)
            self._batch([(level, theta) for level in LEVELS if level not in (500., 1000.) for theta in warm], worker_count)
            for _ in range(self.controls["angular_depth"]):
                tasks = []
                for level in LEVELS:
                    rays = sorted([r for r in self.payload["rays"].values() if r["level"] == level], key=lambda r: r["theta_rad"])
                    for left, right in zip(rays[:-1], rays[1:]):
                        if not left["roots"] and not right["roots"]:
                            continue
                        pairs = list(zip(left["roots"], right["roots"]))
                        needs = len(left["roots"]) != len(right["roots"]) or any(
                            not within_spacing(a, b, self.controls) for a, b in pairs)
                        if needs and right["theta_rad"] - left["theta_rad"] > 1.e-7:
                            tasks.append((level, angular_midpoint(left["theta_rad"], right["theta_rad"])))
                if not tasks:
                    break
                self._batch(tasks, worker_count)
            # Retry unresolved original rays once, now with the refined neighboring profiles.
            with self.lock:
                self.attempted.clear()
            unresolved = [(r["level"], r["theta_rad"]) for r in self.payload["rays"].values() if r["status"] != "success"]
            self._batch(unresolved, worker_count)
        with self.lock:
            self.payload["segments"] = {level: self.segments(level) for level in LEVELS}
            rays = list(self.payload["rays"].values())
            self.payload["run_complete"] = True
            complete = {}
            for level in LEVELS:
                selected = sorted([r for r in rays if r["level"] == level], key=lambda r: r["theta_rad"])
                complete[level] = bool(selected and selected[0]["theta_rad"] == 0.
                    and selected[-1]["theta_rad"] == np.pi / 2.
                    and all(r["status"] == "success" and r["roots"] for r in selected)
                    and all(len(a["roots"]) == len(b["roots"]) and all(within_spacing(x, y, self.controls)
                        for x, y in zip(a["roots"], b["roots"])) for a, b in zip(selected[:-1], selected[1:])))
            self.payload["level_complete"] = complete
            self.payload["contours_complete"] = all(complete.values())
            self.save()
        return self.payload


def connected_segments(rays, level, controls, *, theta_axis=None):
    """A failed ray, a skipped angular node or a branch change splits the line.

    Connectivity follows the angular sampling that was actually scheduled, not
    an absolute step in (density, temperature): a ray sweep is a valid polyline
    at whatever spacing it was run, and only a *missing* node leaves a gap.
    `theta_axis` is the scheduled axis; without it the union of every level's
    rays reconstructs it, since no level is refined beyond the common axis.
    """
    ordered = sorted([r for r in rays if r["level"] == level], key=lambda r: r["theta_rad"])
    axis = np.asarray(sorted(float(t) for t in (
        {r["theta_rad"] for r in rays} if theta_axis is None else np.ravel(theta_axis))), float)
    segments, active = [], []
    previous = previous_node = None
    for ray in ordered:
        roots = ray["roots"] if ray["status"] == "success" else []
        node = int(np.argmin(np.abs(axis - ray["theta_rad"]))) if axis.size else 0
        if not roots or previous is None or len(previous) != len(roots) or node - previous_node != 1:
            segments.extend(line for line in active if len(line) >= 2)
            active = [[p] for p in roots]
        else:
            for index, (left, right) in enumerate(zip(previous, roots)):
                if connectable(left, right, controls):
                    active[index].append(right)
                else:
                    if len(active[index]) >= 2:
                        segments.append(active[index])
                    active[index] = [right]
        previous = roots or None
        previous_node = node
    segments.extend(line for line in active if len(line) >= 2)
    return segments
