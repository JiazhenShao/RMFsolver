# Adiabatic analytical scan: xi = -0.85

Compute both existing full analytical and semi-analytical methods at
T(0+) = 0.1 MeV, xi = -0.85, ms = 0, QMC-RMF3 PNM, and the unchanged
3 n0 calibration, B^(1/4) = 189.1565957288247 MeV.
No numerical BVP or boundary solve is launched.

## Upload and run

Keep this folder at this exact depth in the cluster repository:
`new_paper_calculations/26-10-02/adiabatic_xi_minus0p85/`.
Upload its three Python files, this README, and both `inputs/*.npy` files.
It uses the repository's existing `RMFsolver/`; no other dated calculation
folder is a runtime dependency.
Use the same solver source on the cluster; source hashes are recorded.
Activate the existing cluster `.NSM` environment with working NumPy, SciPy,
and tqdm, then run from the repository root:

```bash
cd new_paper_calculations/26-10-02/adiabatic_xi_minus0p85
python3 run_T0plus0p1_analytical.py --dry-run --workers 64
python3 run_T0plus0p1_analytical.py --workers 64
```

Match `--workers` to the CPU allocation; each worker runs at most one child
at a time and BLAS thread counts are one.
Omit `--workers` to use the first positive SLURM/PBS/NSLOTS allocation, then
CPU affinity or the machine CPU count, with no fixed worker cap.
The production command is the second Python command; dry-run validates inputs
without EOS imports, calculations, or output writes.
Default output is **`T0plus0p1-analytical-xi-minus0p85.npy`** beside the script.
The runner works from any current directory when invoked by its full path.

An optional real three-point check is available on the cluster:

```bash
python3 run_T0plus0p1_analytical.py --smoke --workers 3
```

This does run physics and writes a separate `*-smoke.npy`, not production data.

## Reused inputs and provenance

Both input NPYs are byte-identical to their `26-09-28/inputs/` counterparts.
Their xi = -0.5 metadata is deliberately unchanged: they supply stored
coordinates, boundaries and calibration, not new xi = -0.85 solver answers.
The legacy numerical filename `T0plus0p1-uNmax-state-of-art-complete-5km.npy`
is preserved solely for artifact compatibility.

Every new analytical call receives xi = -0.85 explicitly.
Output `xi` is -0.85, `geometry_source_xi` is -0.5, and
`numerical_reference_grid` is marked xi = -0.5 / `source_reference_only`.
Those numerical velocities must not be presented as numerical xi = -0.85 results.
Saved boundaries are reused geometry, not newly certified xi = -0.85 boundaries.

Initial sampling has **660 unique points**: the original 630-point union of
the 30 by 20 domain and exact 30 by 18 numerical plotting coordinates, plus
one inward radial-fraction 0.005 point on each of 30 angular rays.
Both angular endpoints remain present; the first ray has exactly T(0-) = 0.
The original outer fractions 0.90, 0.95 and 0.99 remain available.
Radial fraction refers to the saved inner-to-outer interval, not absolute rho.

## Evaluation, checkpoints and resume

`_adiabatic_physics.py` extracts the proven direct fixed-density upstream
recovery from `26-09-28/repair_analytical_local.py` and calls the unchanged
`analytic_velocity_bound` and `semi_analytic_velocity_bound` repository APIs.
It does not scan upstream chemical potential.
Each method has its own killable **120 s** cap, including recovery and import
overhead; one method timing out does not discard the other method's result.

Every terminal physical-point attempt is atomically checkpointed before one
tqdm progress increment, including failures.
Restart the identical command to resume: successful method records remain
untouched, unfinished points run, and previously failed methods are retried
once per invocation without rerunning successful methods.
Ctrl-C stops active children; restart from the last atomic checkpoint.
Changing source/input hashes, xi, timeout, or refinement settings requires
a new `--output`; worker count may change on resume.
Old xi = -0.5 outputs are rejected by the resume contract.

Semi-analytical acceptance requires adjacent successful samples bracketing
5000 m/s on every ray with radial-fraction width at most 0.01.
Wide brackets are bisected; otherwise search uses successful outer anchors
or extends toward fraction 0.999, for at most 12 adaptive rounds.
Failed samples break adjacency but do not become artificial search boundaries.
No failed velocity is interpolated or filled in.
The full analytical method has no 5000 m/s acceptance requirement.

Useful switches: `--workers`, `--method-timeout`, `--adaptive-rounds`,
`--bracket-width`, `--target-velocity`, `--max-rho`, and `--output`.
For a different cap use a new output, for example:

```bash
python3 run_T0plus0p1_analytical.py --workers 64 --method-timeout 300 --output T0plus0p1-analytical-xi-minus0p85-300s.npy
```

Output contains raw `records`, failure/retry history, exact `points`, source
hashes, `matched_numerical_grid` (30 by 18), `original_domain_grid` (30 by 20),
`plot_grid` (30 by 21, including 0.005), and the accepted
`semi_analytic_bracket_grid` (30 by 2).
`run_complete` only means bounded scheduling finished; inspect method masks
and `coverage` before claiming complete physical curves.
Exit 0 requires target bracket coverage, while exit 2 reports unresolved rays.
The criterion certifies sampled coverage, not continuum accuracy or all
full-analytical contour levels.

## Cleanup and local verification

Historical copies were moved recoverably to the sibling folder
`../legacy_backup_adiabatic_xi_minus0p85/`: both old analytical NPY outputs,
the old README, `_method_engine.py`, `_analytic_ray_grid.py`,
`repair_analytical_local.py`, its two old test files, `__pycache__/`, and
`.DS_Store`.
The substantive files and cache were confirmed byte-identical to the untouched
`26-09-28/` originals before moving; `.DS_Store` was preserved as metadata.
**Do not upload the backup folder as part of the active run.**

Run lightweight regression checks from the repository root:

```bash
./bin/python3 -B -m unittest discover -s tests -p 'test_adiabatic_xi_26_10_02.py'
```

Local verification uses mocked physics plus a real dry-run only.
No live EOS evaluation, BVP, boundary solve, smoke calculation or production
sweep has been run for xi = -0.85.
The local SciPy 1.15.2 integration import still has the previously observed
macOS loader failure; no package/environment changes were made.
Dry-run success is not a certification that the cluster's SciPy can execute physics.
