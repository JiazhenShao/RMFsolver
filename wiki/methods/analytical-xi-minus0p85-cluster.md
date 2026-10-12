---
summary: Local recovery connects all displayed adiabatic analytical curves, adds the supported 2000 m/s band, and raises isothermal accepted targets to 297/322; immutable cluster inputs are overlaid in 26-10-02.ipynb.
status: current
updated: 2026-10-04
tags: [method, cluster, analytical, isothermal, adiabatic, checkpoint]
---

# Analytical cluster experiments at xi = -0.85

The 2026-10-04 extension of the notebook's logarithmic-temperature panel through 1000000 m/s is documented in [[isothermal-high-speed-contours]]; counts below refer to the original seven-level dataset and its local recovery.

The user selected $\xi=-0.85$ for both new analytical experiments.
These are fresh calculations, not reinterpretations of saved $\xi=-0.5$ speeds.
No `RMFsolver/` file or previous result was changed when preparing the workflows.
The source and required portable inputs are in the two folders below; upload each active folder including `inputs/` into the existing repository/cluster environment.

## Adiabatic: fixed interface temperature

Folder: `new_paper_calculations/26-10-02/adiabatic_xi_minus0p85/`.

```bash
python3 run_T0plus0p1_analytical.py
```

Both unchanged repository methods, `analytic_velocity_bound` and `semi_analytic_velocity_bound`, receive $T(0^+)=0.1$ MeV and $\xi=-0.85$.
PNM, zero strange mass, QMC-RMF3, the $3n_0$ cold calibration, and $B^{1/4}=189.1565957288247$ MeV are preserved.
The wrapper uses validated direct fixed-density upstream recovery, replacing the old broad upstream chemical-potential inversion.

The original domain and numerical-coordinate input files remain byte-identical to the September snapshots; their old xi metadata is retained as source provenance, not passed as the new closure parameter.
Initial sampling contains 660 unique paired points, including the recovered inner radial fraction 0.005 on all 30 rays.
The established semi-analytical 5000 m/s adjacent-bracket refinement remains bounded to 12 rounds.
Failed coordinates break plotted adjacency but no longer act as impenetrable search boundaries.

Each method has its own 120-second hard cap; a returned success survives failure of the other method.
Every physical-point attempt is saved atomically before progress advances.
Resume keeps successful methods and retries failed methods once per invocation; source/input/physics/control changes require a new output.
Default output: `T0plus0p1-analytical-xi-minus0p85.npy`.
`--dry-run` launches no physics, while `--smoke` is a separate three-point real cluster check.
See [[fixed-interface-analytical-contours]] for the historical $\xi=-0.5$ dataset and the recovered September method.

## Isothermal: full logarithmic-temperature window

Folder: `new_paper_calculations/26-10-02/isothermal_analytic_logT/`.

```bash
python3 run_isothermal_analytic.py
```

The public `analytic_velocity_isothermal` supplies every new velocity with automatic maximum $a(0^+)$ and $\xi=-0.85$.
No numerical BVP, continuation profile, nested analytical density-proposal scan, or numerical warm-curve splice remains.
The user explicitly confirmed the full $0.01$–$120$ MeV temperature range.
Sampling preserves the previous 31 logarithmic temperatures from 0.01 to 20 MeV and adds 15 logarithmic temperatures above 20 through 120 MeV, with a unique 20-MeV seam.
The seven established speeds yield **322 contour targets**, not a rectangular density mesh.
Several analytical point evaluations may be required per target; evaluations and terminal contour targets are counted separately.

The 31 KiB static-only input bundle retains 61 validated cold boundary-temperature records and neutral-boundary plotting geometry, with source hashes and the original domain fingerprint.
All 31 scheduled cold boundaries are reused; last-bit NumPy temperature differences are treated as roundoff aliases, never interpolated roots.
Xi does not enter these static boundary equations.
New warm boundaries use cached direct fixed-density states and bracketed roots in density.
The allowed density interval remains $1$–$5.5$ in units of $n_0$; boundaries outside that interval are clipped or classified as an excluded window.
The upper static boundary retains $a(0^+)=1-10^{-9}$; no moving front is evaluated at exactly unit composition.

Analytical values are shared across the seven increasing-speed targets at each temperature.
Safeguarded secants and coexistence-distance-scaled steps replace the former nested proposal stage.
An exact coexistence zero may anchor a bracket; only a newly evaluated positive, validated speed can be accepted as a contour root.
At clipped density edges, a validated edge speed can identify an out-of-window target without spending the entire search budget approaching the edge.
This follows the established increasing-speed branch and does not certify additional disconnected branches.

Target acceptance requires 0.5% speed agreement and analytical, composition, momentum-flux and public-solver validity checks.
Controls are 180 seconds per point, 600 accumulated evaluation seconds and at most 24 evaluations per target; no automatic midpoint campaign is enabled.
Reservations precede launch; completed evaluations precede progress; off-target analytical successes are cached.
Interrupted reservations remain spent, successful/terminal targets are retained on resume, and incompatible source/physics/input/axis/control metadata are rejected.
Independent temperature chains use allocated CPUs with one BLAS thread per child.

Outputs use the `isothermal-analytic-logT` stem: NPY checkpoint, portable NPZ plotting archive, PNG and PDF.
Figures refresh on the main thread from consistent snapshots, coalescing simultaneous accepted roots without holding the checkpoint lock during rendering.
Only consecutive accepted targets connect; unresolved intervals remain visible gaps.
Plotting spans the full logarithmic temperature window with gray regions and no old numerical velocity curves.
Use `--plot-only` for partial results, `--dry-run` for solver-free preflight, or `--smoke --workers 4` for four bounded real cluster targets in a separate directory.
`work_complete` and `contours_complete` remain distinct; unresolved targets cause exit code 2.

## Cleanup and verification limits

Copied old outputs, caches and superseded helpers were moved to sibling `legacy_backup_adiabatic_xi_minus0p85/` and `legacy_backup_isothermal_analytic_logT/` folders.
Those copies are recoverable and are not cluster inputs; original dated directories remain untouched.
Source/README files are unignored in Git, while input binaries, generated figures, checkpoints and backup folders remain ignored.

Verification used lightweight API-routing, mocked orchestration, checkpoint/resume, source/input preservation, edge classification, failure-gap plotting and subprocess-timeout tests, plus both real dry-run commands.
No production contour or live EOS/BVP/analytical smoke calculation was run locally.
That preparation-stage verification did not measure cluster convergence; downloaded outcomes are recorded below.
See [[isothermal-log-temperature-contours]] for why the preceding numerical campaign is not being reused as the new velocity dataset, and [[isothermal-analytic-front-speed]] for approximation limits.

## Downloaded results and comparison notebook

`26-10-02.ipynb` was initially a saved-data comparison notebook. As of the later 2026-10-04 update, its isothermal section also has an explicit fresh local analytical calculation cell; visualization cells remain solver-free. See [[isothermal-high-speed-contours]] for execution and resume controls.
It exports PNG/PDF figures to `local_analysis_results/26-10-02/` and does not change either previous notebook.

The adiabatic checkpoint contains 749 paired evaluations: 748 full-analytical and 619 semi-analytical successes, with all 30 semi-analytical 5000 m/s brackets completed.
Use its **`plot_grid`** (30 by 21, including the inner 0.005 fraction), not the older `original_domain_grid` fallback.
The initial figure followed the existing banded comparison through 1000 m/s; local recovery below extends the display to 2000 m/s: numerical solid, semi-analytical dashed, full analytical dotted.
Bands join numerical and full-analytical paths only over shared angular coverage; failed samples remain masked.
The unchanged full-numerical reference keeps its historical source metadata, rather than being presented as new analytical data at the new xi.

The isothermal checkpoint has **292 successful, 25 outside-window, and 5 unresolved targets** out of 322; work is terminal but contours are not all complete.
The unresolved pairs (m/s, MeV) are (500, 58.6031), (50, 66.0385), (200, 74.4174), (500, 83.8593), and (5000, 94.4991).
The initial linear-temperature and logarithmic-temperature comparison views left those analytical gaps open; the local recovery below fills all five with accepted calculations.
The logarithmic window remains 0.01–120 MeV; neither plot adds a zero-temperature velocity anchor.

Numerical display rules in this notebook:

- At and above 20 MeV, retain the full saved `curve_vertices_*` warm polylines from the hash-checked `local_analysis_results/All_Plots_new/isothermal_contour_data/` archive. These retain historical interpolation/smoothing, not extra numerical solves.
- Below 20 MeV, draw **58 distinct accepted contour-root dots** from that archive's `roots_*`, the six accepted cold-endpoint roots, and the 13 accepted targets in the downloaded 26-10-01 log-temperature checkpoint. Never use the resampled `curve_vertices_*` as numerical dots.
- Match colors by velocity, not by data source. The numerical BVP has no xi closure; its old xi metadata describes analytical proposals, not the numerical equation.

The loader checks shared domain fingerprints, matching physics, archived input hashes and the new analytical xi, and prints source SHA256 hashes.
`new_paper_calculations/26-10-02/_saved_contour_data.py` implements gap-preserving analytical selection, accepted-root selection/deduplication, warm clipping at 20 MeV, and saved-boundary shading without EOS imports.
Eight lightweight tests cover these rules and the actual notebook's labeled temperature split; all notebook cells were executed and all three figures visually inspected without importing `RMFsolver`.

## Targeted local recovery and 2000 m/s band

`new_paper_calculations/26-10-02/repair_analytical_local.py` writes only `local_analysis_results/26-10-02/analytical-local-repair.npy`, leaving both downloaded checkpoints immutable.
The notebook checks baseline hashes and merges this overlay in memory, never replacing an original successful record.
The recovery used two local workers, one BLAS thread per worker, a 90-second hard cap per evaluation, and atomic pointwise saving.
No numerical BVP or solver-source modification was needed.

Recovered **one full-analytical and 85 semi-analytical samples**, taking overall adiabatic successes to 749/749 and 704/749 respectively.
The 45 remaining semi-analytical failures lie outside the quads needed by the six displayed contours; they were not blindly retried.
All displayed levels (50, 100, 200, 500, 1000, 2000 m/s) now have exactly one connected path for each analytical method.
Selection uses neighboring rays to identify missing contour-quad corners, including an entirely timed-out ray; predictions select work only and are never plotted as calculated speeds.
The maximum recovered relative velocity-formula residual is $1.18\times10^{-11}$.

The single cold failure was upstream branch corroboration at $n_B(0^-)/n_0=4.6545676259103885$, not absence of the state.
Two direct-density RMF seeds near the known solution independently reached the same positive state, with scaled residuals below $2.5\times10^{-9}$ and forward density errors below $3\times10^{-10}$.
The recovery wrapper keeps the original acceptance checks and uses this explicit two-seed fallback only at that diagnosed point; it does not patch solver globals.

All five unresolved isothermal targets were recovered from saved same-temperature brackets and new analytical evaluations, with maximum target-speed error **0.3421%**, below the existing 0.5% criterion.
There are now **297 accepted and 25 outside-window targets, no unresolved scheduled targets**.
At clipped density edges the original multiplicative predictor had advanced from roundoff-sized offsets from $n_B(0^-)/n_0=1$; the repair uses available speed brackets or halves the remaining allowed density interval.

⚠ The **full-analytical 2000 m/s curve is bracketed on only the hottest three saved rays inside the shared semi-analytical domain**.
Direct checks at radial fraction 0.9999 on rays 0, 10 and 20 give only 1963.96, 1806.53 and 1539.56 m/s, respectively.
The red band therefore covers only the common computed warm angular interval; a cold/intermediate full-analytical band would require extrapolation beyond this domain and is not drawn.
These checks are recorded in `local_analysis_results/26-10-02/adiabatic-outer-boundary-checks.json`.
The new figure stem is `adiabatic-xi-minus0p85-bands-to-2000`; previous exports remain available.
