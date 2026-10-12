"""Audit saved full-BVP probes; no BVP reruns or interpolation."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from RMFsolver import phase_velocity as pv


def main():
    digest = hashlib.sha256(Path(pv.__file__).read_bytes()).hexdigest()
    rows = []
    paths = sorted(p for pattern in ("baseline_*.npy", "core_mesh_*.npy", "validation_*.npy")
                   for p in HERE.glob(pattern))
    for path in paths:
        saved = np.load(path, allow_pickle=True).item()
        assert saved["solver_identity"]["sha256"] == digest
        result = saved["result"]
        success = bool(result.get("success"))
        row = dict(file=path.name, nB_0minus_over_n0=saved["nB_0minus_over_n0"],
                   T_0minus=saved["T_0minus"], runtime_s=saved["runtime_s"],
                   status="converged" if success else result.get("status", "failed"),
                   velocity_m_s=None, max_rms_residual=None, boundary_residual_max=None,
                   endpoint_flux_residual_max=None, profile_eos_residual_max=None,
                   muK_end_over_T=None, relative_speed_change=None)
        if success:
            assert result["bvp_status"] == 0
            assert result["rate_model"] == "exact_nonleptonic"
            assert result["diffusion_model"] == "local_muB_fixed_T"
            assert result["a_0plus_source"] == "maximum"
            assert result["a_0plus_max_status"] == "interior"
            assert 0 < result["a_0plus"] < 1
            assert abs(result["a_0plus"]-result["a_0plus_derived"]) < 1.e-7
            row.update(velocity_m_s=result["velocity_m_s"],
                       max_rms_residual=result["bvp_diagnostics"]["max_rms_residual"],
                       boundary_residual_max=float(np.max(np.abs(result["boundary_residuals"]))),
                       endpoint_flux_residual_max=result["endpoint_flux_residual_max"])
            assert row["max_rms_residual"] <= saved["controls"]["tol_bvp"]
            assert row["boundary_residual_max"] <= saved["controls"]["tol_bvp"]
            assert row["endpoint_flux_residual_max"] < 1.e-7
            state = result["continuation_state"]
            assert np.all(np.isfinite(state["nK"])) and np.all(state["nK"] > 0)
            assert np.all(np.diff(state["nK"]) <= 1.e-8*state["nK_0plus"])
            assert np.all(np.isfinite(state["jK"])) and np.all(state["jK"] > 0)
            errors = []
            for i in np.unique(np.linspace(0, len(state["s"])-1, 17).astype(int)):
                thermo = pv._solve_local_quark_state_from_nK_T_and_Pi(
                    state["nK"][i], saved["T_0minus"], result["Pi"], result["jB"],
                    saved["B_one_forth"], initial_guess=(result["muB_inf"], result["muK_0plus"]),
                )
                errors.append(float(thermo["closure_residual"]))
            row["profile_eos_residual_max"] = max(errors)
            row["muK_end_over_T"] = thermo["muK"]/saved["T_0minus"]
            assert max(errors) <= 1.e-8
            if path.name.startswith("validation_"):
                prior = np.load(HERE / path.name.replace("validation_", "core_mesh_"), allow_pickle=True).item()
                row["relative_speed_change"] = abs(result["velocity_m_s"]/prior["result"]["velocity_m_s"]-1)
                assert row["relative_speed_change"] < 0.01
        rows.append(row)
    summary = dict(solver_sha256=digest, production_solver_unchanged=True,
                   physical_probes=len(rows), converged=sum(r["status"] == "converged" for r in rows),
                   rows=rows)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (HERE / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
