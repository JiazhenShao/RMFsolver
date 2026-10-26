# Exact-cold adiabatic analytical contours: cluster runner

This entry point uses the same engine, frozen inputs, physical acceptance
criteria and checkpoint format as the [local runner](../local/README.md).
Both slow analytical APIs use exactly `T_0plus=0`, `xi=-0.85`, massless quarks,
QMC-RMF3 PNM, and `B_one_forth=189.1565957288247` MeV. Neither solves a spatial
BVP. The full analytical and semi-analytical fields and physical masks are
computed independently.

## Copy and run

Preserve this directory layout on the cluster:

```text
repository-root/
  RMFsolver/
  new_paper_calculations/
    26-10-26/
      local/
      cluster/
        run_adiabatic_contours.py
      shared/
        cold_scan.py
        cold_physics.py
        bounded_call.py
      inputs/
        cold-scan-seeds.json
```

Copy the four sibling calculation folders and the current `RMFsolver/`
directory. Keep their current depth: both the shared engine and physics worker
derive the repository root from it. No other dated calculation is required by
the engine. The seed JSON already contains coordinates and calibration; its
historical source paths are provenance, not runtime reads.

Use a cluster-native Python environment containing NumPy, SciPy, numdifftools
and matplotlib, which are required by the existing solver package. Do not copy
the Mac-specific SciPy installation to a Linux cluster. From the repository
root, within the allocated job:

```bash
python3 -B new_paper_calculations/26-10-26/cluster/run_adiabatic_contours.py --dry-run
python3 -B new_paper_calculations/26-10-26/cluster/run_adiabatic_contours.py --smoke
python3 -B new_paper_calculations/26-10-26/cluster/run_adiabatic_contours.py
```

Worker count comes from the first valid positive value among
`SLURM_CPUS_PER_TASK`, `PBS_NP` and `NSLOTS`, then process CPU affinity, then the
machine CPU count. `--workers N` overrides it. Each bounded call uses a spawned
process, and BLAS/OpenMP are limited to one thread per call. On a shared node,
use an explicit worker count if the scheduler does not supply those variables
or CPU affinity.

The default manifest is
`cluster/adiabatic-T0plus0-xi-minus0p85.npy`; the sibling
`adiabatic-T0plus0-xi-minus0p85-records/` directory contains atomic pointwise
checkpoints. `--smoke` appends `-smoke.npy` and uses its own records.
`--dry-run` prints settings without solves or writes.

The supported selectors are `--method both` (default), `--method analytic`,
and `--method semi_analytic`. Other controls are `--output PATH.npy`,
`--workers N`, `--xi VALUE`, `--path-rtol VALUE` (default `1e-6`),
`--angular-rounds N` (default 6), and `--method-timeout SECONDS` (default 120).
Relative output paths are relative to the current working directory.

## Resume and inspect

Rerun the identical command after interruption. Accepted and physically
excluded records are reused; each previously failed point is retried once per
invocation with its prior failure history retained. Internal recovery remains
within the hard per-method deadline. A timed-out or cancelled child is
terminated, and each finished method-point result survives an interrupted ray.
Do not run simultaneous writers against the same output stem.

Boundary bisection uses direct cold EOS classification instead of a speed
integral. Its `boundary-` records are separate from speed records and store
`interior` or `excluded` status and thermodynamic margins. They use the same
hard deadline; failures enter the retry and completion checks. Near-boundary
speed evaluations are requested only as needed to bracket the plotted
50–5000 m/s levels.

Preserve the manifest and records together. Resume checks the exact physics,
input bytes, numerical controls and source hashes, including every
`RMFsolver/*.py` file plus `cold_scan.py`, `cold_physics.py` and
`bounded_call.py`. Cold/warm, source and settings mismatches require a new
output path. Worker count can change without invalidating the physics cache.
Verification and plot scripts are not included in this physics identity.

The local production workflow was corrected after five preliminary
120-second timeouts in the diffusion-table construction extremely close to
the stable-matter boundary. Direct EOS classification removes those unnecessary
integral tables without changing the physics. A one-time audited migration
reused 2,424 accepted/excluded records after checking that the original physics
module was an identical AST prefix and the EOS/input hashes matched. The rows
retain `calculation_identity` and `reuse_verification`; see
`local_analysis_results/26-10-26/cold-boundary-workflow-recovery.json`.
Ordinary source-mismatch rejection remains in force. The resumed local run
used eight workers and completed both methods; see the local result summary.

Inspect `complete`, `methods`, `coverage` and `unresolved_angular_intervals`
in the manifest. Require both method keys in `coverage` for the two-method deliverable.
The process returns 0 only when the selected methods have no failed samples,
unresolved contours/boundaries or pending angular refinement; otherwise it
returns 2. Supported levels and levels outside the window are reported
separately. A supported level need not cross every ray.

The window is `nB_0minus/n0=1..5.5`, `T_0minus=0..115` MeV, with requested
levels 50–5000 m/s. Root acceptance requires relative speed error below
`1e-3` and radial bracket width at most `0.001`; physical boundary width is
at most `1e-4`. Angular midpoint geometry uses a `0.0025` threshold in
window-normalized coordinates. Stable-matter and composition-ceiling masks
are recomputed separately for each method, and failed samples do not bridge
contours. Exact zero interface temperature requires infinite local diffusion,
zero inverse diffusion and finite positive integrated quantities.

The [local README](../local/README.md) records the 25 orchestration and 11
physics tests, independent three-state verification, completion inspection
command and notebook workflow. The orchestration run enables
`COLD_CONTOUR_LIVE=1`, including ten live classifier cases and the regression
against unnecessary boundary speed integrals. Those tests passed; the completed local manifest certifies 101 analytical and 61 semi-analytical rays. For notebook use, transfer
the complete two-method manifest and its records to the local default path
while preserving the matching source bundle. The notebook adds two individual
figures and one overlay against the **0.1 MeV** numerical reference; its
numerical and analytical curves intentionally have different interface
temperatures. Plotting also needs the existing notebook/reference data, which
are not dependencies of the standalone contour engine.
