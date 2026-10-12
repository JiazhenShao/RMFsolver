"""Check that recomputed xi=-0.85 speeds reproduce the 26-10-02 cluster checkpoint."""
import json
import numpy as np

CHECKPOINT = ("/Users/janshao/NSM_related/new_paper_calculations/26-10-02/adiabatic_xi_minus0p85/"
              "T0plus0p1-analytical-xi-minus0p85.npy")
saved = np.load(CHECKPOINT, allow_pickle=True).item()["records"]
with open("analytical_xi-0.8500.jsonl") as handle:
    rows = {r["key"]: r for r in map(json.loads, handle)}
for m in ("analytic", "semi_analytic"):
    rel = [rows[k][m]["u_0minus"] / saved[k][m]["u_0minus"] - 1 for k in rows
           if k in saved and saved[k][m].get("success") and rows[k][m].get("success")]
    print(f"{m}: {len(rel)} shared states, max |relative difference| {np.max(np.abs(rel)):.3e}")
