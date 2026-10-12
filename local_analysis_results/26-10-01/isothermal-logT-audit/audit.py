"""Read-only checkpoint diagnosis; no EOS, BVP, or analytical root solves.

Writes audit JSON and frozen-endpoint comparisons beside this script.
Uses only the pure microphysics helper from phase_velocity.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from RMFsolver import phase_velocity as pv

HERE = Path(__file__).resolve().parent
RUN = ROOT / "new_paper_calculations/26-10-01/isothermal_numeric_logT"
CHECKPOINT = RUN / "isothermal-logT.npy"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(result, source, density):
    a = result["a_0plus"]
    lam = result["nB_0minus"] / result["nB_inf"]
    micro = pv._microphysics_from_quark_state_isothermal_baseline(
        result["muB_0plus"], result["T_0minus"])
    factor = (micro["D"] * micro["gamma"] * a**2 * (a**2 + 2 * micro["eta"])
              / (2 * lam**2 * (1 - a)))
    return dict(source=source, T_0minus=result["T_0minus"],
                nB_0minus_over_n0=density, a_0plus=a,
                numerical_velocity_m_s=float(result["velocity_m_s"]),
                frozen_ratio_xi_minus_half=float(np.sqrt(factor / (1 - .5*a)) / result["u_0minus"]),
                frozen_ratio_xi_minus_one_limit=float(np.sqrt(factor / (1 - a)) / result["u_0minus"]),
                speed_fitted_xi=float((factor / result["u_0minus"]**2 - 1) / a))


def main():
    p = np.load(CHECKPOINT, allow_pickle=True).item()
    successful = [e for e in p["evaluations"] if e["status"] == "success"]
    counts = {kind: dict(Counter(t["status"] for t in p["targets"].values()
                                if t["kind"] == kind)) for kind in ("base", "refinement")}
    source_matches = {}
    for key, directory in (("solver_files", ROOT), ("workflow_files", RUN)):
        source_matches[key] = {name: digest(directory/name) == saved
                               for name, saved in p["identity"][key].items()}
    rows = [comparison(e["result"], f"evaluation-{e['id']:05d}",
                       e["point"]["nB_0minus_over_n0"]) for e in successful]
    for density in ("3.001", "3.01", "3.05"):
        path = ROOT / ("local_analysis_results/isothermal_low_temperature_probe/"
                       f"core_mesh_T0.01_n{density}.npy")
        probe = np.load(path, allow_pickle=True).item()
        rows.append(comparison(probe["result"], str(path.relative_to(ROOT)), float(density)))
    report = dict(checkpoint_sha256=digest(CHECKPOINT), targets=counts,
        evaluations=dict(Counter(e["status"] for e in p["evaluations"])),
        proposals=dict(Counter(e["status"] for e in p["auxiliary"])),
        boundaries=dict(Counter(e["status"] for e in p["boundaries"].values())),
        accuracy_checks=dict(Counter(e["status"] for e in p["validation"].values())),
        work_complete=p["work_complete"], contours_complete=p["contours_complete"],
        initializers=dict(Counter(e["result"]["initial_guess_mode_used"] for e in successful)),
        new_profiles_missing=sum(not (RUN/e["profile_file"]).is_file() for e in successful),
        source_hash_matches=source_matches,
        worker_hours=dict(numerical=sum(e["runtime_s"] for e in p["evaluations"])/3600,
            proposals=sum(e["result"]["runtime_s"] for e in p["auxiliary"])/3600,
            boundaries=sum(e["runtime_s"] for e in p["boundaries"].values())/3600))
    for key in ("runtime_s", "bvp_niter", "bvp_nodes", "bvp_max_rms_residual",
                "thermal_local_state_calls", "T_0minus", "momentum_flux_ratio"):
        values = [e["result"][key] for e in successful]
        report["successful_"+key] = dict(min=float(np.min(values)),
                                       median=float(np.median(values)), max=float(np.max(values)))
    report["comparison_scope"] = ("Frozen formula evaluated at saved numerical endpoints, "
        "not an independent hydro-coupled analytic root; fitted xi is not a measured profile area.")
    for key in ("frozen_ratio_xi_minus_half", "frozen_ratio_xi_minus_one_limit", "speed_fitted_xi"):
        values = [r[key] for r in rows[:len(successful)]]
        report[key] = dict(min=min(values), median=float(np.median(values)), max=max(values))
    with (HERE / "frozen-endpoint-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
