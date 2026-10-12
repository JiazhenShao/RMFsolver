"""Discover cold-interface states using read-only, branch-validated QMC-RMF3.

This module performs no front BVP or velocity calculation.  Every EOS query is
checkpointed, including failed queries; physical nonexistence is distinguished
from numerical failure.  All thermodynamic units are powers of MeV.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import warnings


ROOT = Path(__file__).resolve().parents[2]
SCIPY_SIDECAR = (ROOT / "local_analysis_results/26-10-02/runtime/"
                 "scipy-1.18.1-cpython-312-darwin-arm64")
# The workspace's original SciPy installation cannot load on this machine.
# This must precede imports of NumPy, SciPy, and the RMF package.
if SCIPY_SIDECAR.is_dir():
    sys.path.insert(0, str(SCIPY_SIDECAR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/massless-mpl")

import numpy as np
import scipy
from scipy.optimize import brentq, minimize_scalar


B14 = 189.1566
N0 = 0.16 * 197.326960**3
TEMPERATURES = (0.0, 0.1, 5.0, 20.0, 50.0, 100.0)
COMPOSITION_TARGETS = (
    0.005, 0.02, 0.05, 0.075, 0.1, 0.146, 0.2, 0.25, 0.3, 0.35,
    0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
    0.9, 0.95, 0.98, 0.99, 0.995,
)
HOT_SUPPLEMENTAL_TARGETS = (
    0.88, 0.885, 0.89, 0.895, 0.905, 0.91, 0.915, 0.92, 0.925,
    0.93, 0.935, 0.94, 0.945, 0.955, 0.96, 0.97, 0.975, 0.985,
)
DENSITY_ROOT_XTOL = 2e-12
COMPOSITION_RESIDUAL_TOLERANCE = 2e-8
RESULT_DIR = Path(__file__).resolve().parent / "results"
EOS_SOURCE_FILES = (
    "RMFsolver/phase_velocity.py", "RMFsolver/Solver.py",
    "RMFsolver/RMFparameter.py", "RMFsolver/constants.py",
)
# The validator initializes muB at 990 MeV.  Large vector-field seeds can then
# have zero initial Fermi occupation; low-vector-field seeds recover the same
# branch at high density while retaining every existing acceptance criterion.
SUPPLEMENTAL_BRANCH_SEEDS = (
    (20.0, 10.0, -3.0), (25.0, 15.0, -3.0), (35.0, 25.0, -3.0),
    (40.0, 30.0, -3.0), (50.0, 20.0, -3.0), (60.0, 20.0, -3.0),
    (70.0, 20.0, -3.0), (90.0, 20.0, -3.0),
)


def cold_kinetic_pressure(nB: float, a: float) -> float:
    """Neutral massless-quark pressure at zero temperature and fixed density."""
    if not math.isfinite(nB) or nB <= 0 or not 0 <= a <= 1:
        raise ValueError("Require finite nB > 0 and 0 <= a <= 1")
    return ((math.pi**2 * nB)**(4 / 3) / (4 * math.pi**2)
            * (1 + (1 + a)**(4 / 3) + (1 - a)**(4 / 3)))


def cold_interface(kinetic_pressure: float, h_over_nB_0minus: float) -> dict:
    """Classify and, where possible, solve the conserved cold interface.

    The supplied kinetic pressure is P_0minus + B14**4.  The massless identity
    h = 4 * kinetic_pressure fixes nB on the entire quark trajectory.
    """
    if (not math.isfinite(kinetic_pressure) or kinetic_pressure <= 0
            or not math.isfinite(h_over_nB_0minus) or h_over_nB_0minus <= 0):
        raise ValueError("Pressure and enthalpy per baryon must be positive and finite")
    nB = 4 * kinetic_pressure / h_over_nB_0minus
    cold_0 = cold_kinetic_pressure(nB, 0)
    cold_1 = cold_kinetic_pressure(nB, 1)
    gap_0 = kinetic_pressure / cold_0 - 1
    gap_1 = kinetic_pressure / cold_1 - 1
    edge_tolerance = 4e-15
    if gap_0 < -edge_tolerance:
        status, composition = "no_front", None
    elif gap_1 > edge_tolerance:
        status, composition = "no_cold_interface", None
    elif abs(gap_0) <= edge_tolerance:
        status, composition = "equilibrium_boundary", 0.0
    elif abs(gap_1) <= edge_tolerance:
        status, composition = "flavor_boundary", 1.0
    else:
        status = "physical"
        composition = brentq(
            lambda a: cold_kinetic_pressure(nB, a) / kinetic_pressure - 1,
            0, 1, xtol=2e-14, rtol=1e-14,
        )
    return {
        "status": status,
        "nB": nB,
        "a_0plus": composition,
        "P_kinetic": kinetic_pressure,
        "cold_equilibrium_pressure": cold_0,
        "cold_flavor_boundary_pressure": cold_1,
        "cold_equilibrium_relative_gap": gap_0,
        "cold_flavor_boundary_relative_gap": gap_1,
        "pressure_relative_residual": (
            None if composition is None else
            cold_kinetic_pressure(nB, composition) / kinetic_pressure - 1
        ),
        "enthalpy_relative_residual": (
            4 * kinetic_pressure / nB / h_over_nB_0minus - 1
        ),
    }


def source_hashes() -> dict:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in EOS_SOURCE_FILES if (ROOT / name).is_file()}


def validated_state_with_recovery(nB_0minus, temperature, *, original_error=None):
    """Use the existing validator, retrying once with additional runtime seeds.

    This serial diagnostic restores module globals even if the retry fails.
    Neither the RMF files nor the validator's branch/residual tolerances change.
    """
    from RMFsolver import phase_velocity

    provenance = {"recovery_used": False}
    if original_error is None:
        try:
            return phase_velocity._validated_pnm_state_from_nB(nB_0minus, temperature), provenance
        except RuntimeError as exc:
            original_error = str(exc)
    provenance.update(recovery_used=True, original_error=original_error,
                      supplemental_seeds=[list(seed) for seed in SUPPLEMENTAL_BRANCH_SEEDS])
    original_seeds = phase_velocity._PNM_BRANCH_SEEDS
    try:
        # The original set has already failed, so do not repeat those solves.
        phase_velocity._PNM_BRANCH_SEEDS = SUPPLEMENTAL_BRANCH_SEEDS
        try:
            result = phase_velocity._validated_pnm_state_from_nB(nB_0minus, temperature)
        except RuntimeError as exc:
            raise RuntimeError(f"Initial validator failure: {original_error}\n"
                               f"Supplemental-seed failure: {exc}") from exc
        return result, provenance
    finally:
        phase_velocity._PNM_BRANCH_SEEDS = original_seeds


class CachedUpstreamEOS:
    """Persist the actual EOS states, warnings and failures after each query."""

    def __init__(self, cache_path: Path, *, b14: float = B14):
        self.path = Path(cache_path)
        self.b14 = b14
        from RMFsolver import constants
        self.n0 = 0.16 * constants.MeV_fm**3
        if self.n0 != N0:
            raise RuntimeError("N0 disagrees with the live RMF constants")
        self.signature = {
            "EOS": "QMC-RMF3 pure neutron matter",
            "source_sha256": source_hashes(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "n0_MeV3": self.n0,
        }
        self.evaluations = {}
        self.new_evaluations = 0
        if self.path.is_file():
            payload = json.loads(self.path.read_text())
            if payload["metadata"] != self.signature:
                raise RuntimeError("EOS cache signature mismatch; use a fresh cache path")
            self.evaluations = payload["evaluations"]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "metadata": self.signature, "evaluations": self.evaluations,
        }, indent=2, allow_nan=False) + "\n")

    def __call__(self, temperature: float, density_ratio: float) -> dict:
        key = f"{temperature:.15g}|{density_ratio:.15g}"
        previous = self.evaluations.get(key)
        if previous is None or (previous["status"] == "eos_failure"
                                and not previous.get("recovery_attempted")):
            started = time.monotonic()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    raw, provenance = validated_state_with_recovery(
                        density_ratio * self.n0, temperature,
                        original_error=None if previous is None else previous["error"],
                    )
                    raw = {name: float(value) for name, value in raw.items()}
                    entry = {"status": "accepted", "state": raw, "provenance": provenance}
                except Exception as exc:
                    entry = {"status": "eos_failure", "error": str(exc),
                             "recovery_attempted": True}
            if previous is not None:
                entry["previous_attempt"] = previous
            entry.update({
                "T_0minus": float(temperature),
                "nB_0minus_over_n0": float(density_ratio),
                "wall_seconds": time.monotonic() - started,
                "warning_count": len(captured),
                "warnings": sorted({str(item.message) for item in captured}),
            })
            self.evaluations[key] = entry
            self.new_evaluations += 1
            self.save()
            if self.new_evaluations % 25 == 0:
                print(f"EOS {self.new_evaluations}: T={temperature:g}, "
                      f"nB_0minus/n0={density_ratio:.8f}", flush=True)
        entry = self.evaluations[key]
        if entry["status"] == "eos_failure":
            return {name: value for name, value in entry.items() if name != "state"}
        result = dict(entry["state"])
        result.update(cold_interface(result["P_0minus"] + self.b14**4,
                                     result["h_over_nB_0minus"]))
        result.update({
            "T_0minus": float(temperature),
            "T_0plus": 0.0,
            "nB_0minus_over_n0": float(density_ratio),
            "nB_0minus_recovered_over_n0": result["nB_0minus"] / self.n0,
            "upstream_warning_count": entry["warning_count"],
            "upstream_seed_recovery_used": entry.get("provenance", {}).get("recovery_used", False),
        })
        return result


def _gap(state: dict, composition: float) -> float:
    if state["status"] == "eos_failure":
        raise RuntimeError(state["error"])
    return state["P_kinetic"] / cold_kinetic_pressure(state["nB"], composition) - 1


def discover_slice(temperature, evaluate, *, density_min=1.0, density_max=10.0,
                   scan_points=37, scan_refinements=1,
                   composition_targets=COMPOSITION_TARGETS):
    """Find all bracketed physical intervals and invert a common target grid.

    A uniform midpoint refinement guards the coarse density classification;
    each composition/boundary sign change is then adaptively solved with Brent's
    method.  This is a finite numerical scan, not a proof excluding intervals
    narrower than the recorded scan spacing.
    """
    if density_max <= density_min or scan_points < 3 or scan_refinements < 0:
        raise ValueError("Invalid density scan")
    if any(not 0 < target < 1 for target in composition_targets):
        raise ValueError("Composition targets must be strictly between zero and one")
    density_grid = np.linspace(density_min, density_max,
                               (scan_points - 1) * 2**scan_refinements + 1)
    scan = [evaluate(temperature, float(density)) for density in density_grid]
    counts = dict(Counter(state["status"] for state in scan))
    boundaries = []
    issues = []
    for composition in (0.0, 1.0):
        for left, right in zip(scan[:-1], scan[1:]):
            if "eos_failure" in (left["status"], right["status"]):
                continue
            lo, hi = left["nB_0minus_over_n0"], right["nB_0minus_over_n0"]
            flo, fhi = _gap(left, composition), _gap(right, composition)
            root = None
            try:
                if abs(flo) < 4e-15:
                    root = lo
                elif flo * fhi < 0:
                    root = brentq(lambda density: _gap(evaluate(temperature, density), composition),
                                  lo, hi, xtol=DENSITY_ROOT_XTOL, rtol=1e-13)
                elif abs(fhi) < 4e-15:
                    root = hi
            except RuntimeError as exc:
                issues.append({"stage": "boundary", "density_bracket": [lo, hi],
                               "a_0plus": composition, "error": str(exc)})
            if root is not None and not any(abs(item["density"] - root) < 1e-8
                                           for item in boundaries):
                state = evaluate(temperature, root)
                boundaries.append({"density": root, "a_0plus": composition,
                                   "relative_pressure_residual": _gap(state, composition)})
    boundaries.sort(key=lambda item: item["density"])
    cuts = sorted({float(density_min), float(density_max),
                   *(item["density"] for item in boundaries),
                   *(state["nB_0minus_over_n0"] for state in scan
                     if state["status"] == "eos_failure")})
    intervals, selected = [], []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi - lo < 1e-9:
            continue
        midpoint = evaluate(temperature, (lo + hi) / 2)
        if midpoint["status"] != "physical":
            continue
        edge_states = [evaluate(temperature, edge) for edge in (lo, hi)]
        if any(state["status"] == "eos_failure" for state in edge_states):
            issues.append({"stage": "interval", "density_bracket": [lo, hi],
                           "error": "Interval touches an unresolved EOS state"})
            continue
        interval = {
            "interval_id": len(intervals),
            "nB_0minus_over_n0_bounds": [lo, hi],
            "a_0plus_at_density_bounds": [next(
                (boundary["a_0plus"] for boundary in boundaries
                 if abs(boundary["density"] - edge) < 1e-8), state["a_0plus"])
                for edge, state in zip((lo, hi), edge_states)],
            "computed_a_0plus_at_density_bounds": [state["a_0plus"] for state in edge_states],
            "selected_state_count": 0,
            "inaccessible_targets": [],
            "failed_targets": [],
        }
        interval["density_bound_inclusive"] = [
            composition is not None and 0 < composition < 1
            for composition in interval["a_0plus_at_density_bounds"]]
        # Scan nodes inside this interval allow nonmonotone a(density).
        nodes = [edge_states[0]] + [state for state in scan
                                  if lo < state["nB_0minus_over_n0"] < hi] + [edge_states[1]]
        extrema = []
        for left, center, right in zip(nodes[:-2], nodes[1:-1], nodes[2:]):
            values = [state["a_0plus"] for state in (left, center, right)]
            if any(value is None for value in values):
                continue
            direction = (1 if values[1] < min(values[0], values[2]) else
                         -1 if values[1] > max(values[0], values[2]) else 0)
            if not direction:
                continue
            bounds = (left["nB_0minus_over_n0"], right["nB_0minus_over_n0"])
            try:
                optimum = minimize_scalar(
                    lambda density: direction * evaluate(temperature, density)["a_0plus"],
                    bounds=bounds, method="bounded", options={"xatol": 1e-10})
                if not optimum.success:
                    raise RuntimeError(optimum.message)
                state = dict(evaluate(temperature, float(optimum.x)))
                state.update(extremum_kind="minimum" if direction == 1 else "maximum")
                extrema.append(state)
            except (RuntimeError, TypeError) as exc:
                issues.append({"stage": "extremum", "density_bracket": list(bounds),
                               "error": str(exc)})
        interval["composition_extrema"] = [
            {key: state[key] for key in ("nB_0minus_over_n0", "a_0plus", "extremum_kind")}
            for state in extrema]
        accessible_values = [value for value in interval["a_0plus_at_density_bounds"]
                             if value is not None] + [state["a_0plus"] for state in extrema]
        interval["reachable_a_0plus_bounds"] = [min(accessible_values), max(accessible_values)]
        interval["reachable_a_0plus_bound_inclusive"] = [
            0 < value < 1 for value in interval["reachable_a_0plus_bounds"]]
        # Including the refined turning point brackets targets lying between
        # the true minimum and the nearest uniform-scan sample.
        nodes = sorted(nodes + extrema, key=lambda state: state["nB_0minus_over_n0"])
        for target in composition_targets:
            target_roots = []
            had_bracket = False
            for left, right in zip(nodes[:-1], nodes[1:]):
                flo, fhi = _gap(left, target), _gap(right, target)
                if flo * fhi > 0:
                    continue
                had_bracket = True
                try:
                    root = brentq(lambda density: _gap(evaluate(temperature, density), target),
                                  left["nB_0minus_over_n0"], right["nB_0minus_over_n0"],
                                  xtol=DENSITY_ROOT_XTOL, rtol=1e-13)
                    if any(abs(root - previous) < 1e-8 for previous in target_roots):
                        continue
                    state = dict(evaluate(temperature, root))
                    residual = state["a_0plus"] - target
                    if state["status"] != "physical" or abs(residual) > COMPOSITION_RESIDUAL_TOLERANCE:
                        raise RuntimeError(f"Composition residual {residual:.3g}")
                    state.update(target_a_0plus=float(target), target_a_0plus_residual=residual,
                                 interval_id=interval["interval_id"], sample_role="composition_target")
                    selected.append(state)
                    interval["selected_state_count"] += 1
                    target_roots.append(root)
                except (RuntimeError, ValueError, TypeError) as exc:
                    issues.append({"stage": "target", "target_a_0plus": target,
                                   "density_bracket": [left["nB_0minus_over_n0"],
                                                       right["nB_0minus_over_n0"]],
                                   "error": str(exc)})
            if not target_roots:
                interval["failed_targets" if had_bracket else "inaccessible_targets"].append(float(target))
        for state, role in ([(state, "composition_extremum") for state in extrema]
                            + [(state, "density_domain_edge")
                               for state, include in zip(edge_states, interval["density_bound_inclusive"])
                               if include]):
            if any(item["interval_id"] == interval["interval_id"]
                   and abs(item["nB_0minus_over_n0"] - state["nB_0minus_over_n0"]) < 1e-8
                   for item in selected):
                continue
            state = dict(state)
            state.update(target_a_0plus=state["a_0plus"], target_a_0plus_residual=0.0,
                         interval_id=interval["interval_id"], sample_role=role)
            selected.append(state)
            interval["selected_state_count"] += 1
        intervals.append(interval)
    unresolved_targets = sorted({target for interval in intervals for target in interval["failed_targets"]})
    summary = {
        "T_0minus": float(temperature),
        "status": ("incomplete_eos_scan" if issues or counts.get("eos_failure") else
                   "physical_intervals" if intervals else "no_physical_interval"),
        "intervals": intervals,
        "boundaries": boundaries,
        "classification_counts": counts,
        "scan_points": len(scan),
        "maximum_scan_spacing_over_n0": float(density_grid[1] - density_grid[0]),
        "selected_state_count": sum(item["selected_state_count"] for item in intervals),
        "composition_targets": [float(target) for target in composition_targets],
        "unreachable_targets": [float(target) for target in composition_targets
                                if target not in unresolved_targets
                                and not any(state["target_a_0plus"] == target for state in selected)],
        "unresolved_targets": unresolved_targets,
        "issues": issues,
        "scan": [{key: state[key] for key in (
            "nB_0minus_over_n0", "status", "a_0plus", "cold_equilibrium_relative_gap",
            "cold_flavor_boundary_relative_gap", "error") if key in state} for state in scan],
    }
    return summary, sorted(selected, key=lambda state: state["nB_0minus_over_n0"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "adiabatic-upstream.json")
    parser.add_argument("--cache", type=Path, default=RESULT_DIR / "adiabatic-upstream-cache.json")
    parser.add_argument("--temperatures", type=float, nargs="+", default=TEMPERATURES)
    parser.add_argument("--scan-points", type=int, default=37)
    parser.add_argument("--scan-refinements", type=int, default=1)
    args = parser.parse_args(argv)
    started = time.monotonic()
    evaluator = CachedUpstreamEOS(args.cache)
    # Old checkpoints may predate the supplemental seed policy.  Preserve each
    # original failure and resolve it, including off-grid diagnostic queries.
    pending = [entry for entry in evaluator.evaluations.values()
               if entry["status"] == "eos_failure" and not entry.get("recovery_attempted")]
    if pending:
        print(f"Recovering {len(pending)} previous EOS failures", flush=True)
        for entry in pending:
            evaluator(entry["T_0minus"], entry["nB_0minus_over_n0"])
    payload = {"metadata": {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_EOS": "QMC-RMF3 pure neutron matter",
        "B14_MeV": B14,
        "B_MeV4": B14**4,
        "n0_MeV3": N0,
        "density_bounds_over_n0": [1.0, 10.0],
        "T_0plus_MeV": 0.0,
        "temperature_targets_MeV": list(args.temperatures),
        "composition_targets": list(COMPOSITION_TARGETS),
        "T100_supplemental_composition_targets": list(HOT_SUPPLEMENTAL_TARGETS),
        "density_root_xtol_over_n0": DENSITY_ROOT_XTOL,
        "scan_points_before_refinement": args.scan_points,
        "scan_refinements": args.scan_refinements,
        "composition_residual_tolerance": COMPOSITION_RESIDUAL_TOLERANCE,
        "classification": {
            "no_front": "P_0minus+B < cold equilibrium pressure at conserved nB",
            "no_cold_interface": "P_0minus+B > cold a=1 pressure at conserved nB",
            "physical": "A unique cold composition lies strictly between 0 and 1",
            "eos_failure": "The branch-validated upstream solver did not return a state",
        },
        "scope": "Cold interface and upstream states only; no BVP or velocity solve",
        "scan_limitation": "Finite density scan; narrower unbracketed intervals are not excluded",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scipy_path": scipy.__file__,
        "source_sha256": {**source_hashes(), str(Path(__file__).relative_to(ROOT)):
                          hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "supplemental_branch_seeds": [list(seed) for seed in SUPPLEMENTAL_BRANCH_SEEDS],
        "seed_recovery_policy": "Retry original validator once with supplemental seeds; "
                                "unchanged agreement/residual tolerances; original failures cached",
        "cache_path": str(args.cache.relative_to(ROOT) if args.cache.is_relative_to(ROOT) else args.cache),
    }, "slices": [], "states": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for temperature in args.temperatures:
        print(f"Scanning T_0minus={temperature:g} MeV", flush=True)
        targets = sorted(set(COMPOSITION_TARGETS + (HOT_SUPPLEMENTAL_TARGETS
                                                   if temperature == 100 else ())))
        summary, states = discover_slice(temperature, evaluator, scan_points=args.scan_points,
                                        scan_refinements=args.scan_refinements,
                                        composition_targets=targets)
        payload["slices"].append(summary)
        payload["states"].extend(states)
        payload["metadata"].update(wall_seconds=time.monotonic() - started,
                                   cached_evaluation_count=len(evaluator.evaluations),
                                   new_evaluation_count=evaluator.new_evaluations,
                                   recovered_evaluation_count=sum(
                                       entry.get("provenance", {}).get("recovery_used", False)
                                       for entry in evaluator.evaluations.values()),
                                   unresolved_evaluation_count=sum(
                                       entry["status"] == "eos_failure"
                                       for entry in evaluator.evaluations.values()))
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(f"T={temperature:g}: {summary['status']}, "
              f"{len(summary['intervals'])} intervals, {len(states)} selected states", flush=True)
    return payload


if __name__ == "__main__":
    main()
