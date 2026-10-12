---
summary: Local recovery yields 734/734 successes per analytical method at xi=-0.5, with endpoint-connected curves and all 30 semi-analytical 5000 m/s brackets complete.
status: current
updated: 2026-10-02
tags: [method, cluster, analytical, fixed-interface-temperature]
---

# Fixed-interface analytical comparison

The separate $\xi=-0.85$ cluster experiment is now prepared in `26-10-02/adiabatic_xi_minus0p85/`, using direct fixed-density recovery and separate method timeouts; see [[analytical-xi-minus0p85-cluster]].
The results and numerical comparison documented below remain the established $\xi=-0.5$ dataset and have not been overwritten.

`new_paper_calculations/26-09-28/run_T0plus0p1_analytical.py` computes both
fixed-interface analytical methods at $T(0^+)=0.1$ MeV. It reuses the saved
numerical coordinates and boundaries; it does not run a BVP or a boundary
solver. Physics remains QMC-RMF3 PNM, $m_s=0$, $\xi=-0.5$, and the $3n_0$
cold calibration, $B^{1/4}=189.1565957288247$ MeV.

## Inputs and grid

The portable folder bundles exact copies of:

- `26-08-21/T0plus0p1-domain.npy`: 30 angular rays by 20 radial fractions.
- `26-08-24/T0plus0p1-uNmax-state-of-art-complete-5km.npy`: the consolidated
  numerical result, whose displayed grid has 30 rays by 18 points. Its legacy
  observable filename is preserved for compatibility.

The union has **630 unique initial points** and preserves every displayed
numerical coordinate. Both $0^\circ$ and $90^\circ$ rays are present, with
exactly $T(0^-)=0$ on the first ray. The numerical plotting grid alone does
not guarantee a semi-analytical 5000 m/s contour: the methods reach that speed
at different upstream states. The union retains the original domain's outer
fractions 0.90, 0.95 and 0.99 for this reason.

`radial_fraction` is the fraction of the saved inner-to-outer interval, not
the absolute elliptical radius called `rho` in the domain manifest.

## Reused physics and new orchestration

`_method_engine.py` and `_analytic_ray_grid.py` are byte-identical copies from
`26-08-21/`. The unchanged `evaluate_analytical_point` recovers the upstream
chemical potential once and calls `analytic_velocity_bound` and
`semi_analytic_velocity_bound` with the same input state. Those implementations
remain in the repository's `RMFsolver/phase_velocity.py`; no solver files were
modified for this workflow. See [[phase-velocity-overview]].

A killable spawned process evaluates each paired point, bounded to 600 seconds
by default. Thread coordinators manage the independent processes without
performing the physics themselves. BLAS thread counts are fixed to one.
Every terminal physical point increments tqdm once and triggers an atomic NPY
checkpoint. Resume retains completed records and checks source/input hashes.

## Semi-analytical acceptance

Every angular ray must contain an adjacent pair of successful solver values
bracketing 5000 m/s. Wide brackets are bisected to radial-fraction width at
most 0.01; absent upper brackets trigger outward probes below fraction 0.999.
A failed point breaks adjacency and is never assigned an interpolated speed.
Default recovery is bounded to 12 rounds, with at most one new point per ray
per round. No 5000 m/s condition is imposed on the full analytical method.

The output includes the matched numerical grid, original domain grid, raw
records, masks and a 30 by 2 semi-analytical bracket grid. A plotting routine
can contour this last strip across all neighboring rays using only converged
samples. This certifies sampled connectivity, not continuum accuracy.

`run_complete` means scheduling finished. `complete_5000_contour` means all
30 target brackets met acceptance. Exit code 2 and unresolved theta indices
explicitly identify incomplete production runs.

Run instructions and tunables are in the dated folder's `README.md`.

## Downloaded result and visualization

Before local recovery, the downloaded `26-09-28/T0plus0p1-analytical-comparison.npy` was terminal with
702 scheduled points after 12 adaptive rounds. Both methods accept 634 points;
67 paired evaluations timed out at 600 seconds and one failed upstream density
recovery. The original 600-point grid contains 533 accepted points per method.
The semi-analytical 5000 m/s bracket is complete on 29/30 rays: theta index 1
remains unresolved, despite a converged outer value above the target. Failed
intervening samples prevent that isolated outer value from closing the gap.

`All_Plots_new.ipynb` contains an executed three-method comparison immediately
after the existing fixed-0.1-MeV numerical contour. Its size, fonts, colors,
legend and gray boundaries follow the 5 MeV comparison in `26-08-18-1.ipynb`.
It draws the existing numerical result unchanged, the analytical body from the
masked original grid, and the semi-analytical 5000 contour from its refined
bracket strip. Neither missing cells nor the unresolved angular row are filled.
PNG and PDF exports are at
`local_analysis_results/All_Plots_new/T0plus0p1-three-method-comparison.*`.

## Endpoint-gap diagnosis

The angles are present, not omitted. Original-grid successes for **each**
method are 20/20 at theta index 0 (0 degrees), 5/20 at index 1 (3.103 degrees),
and 9/20 at index 29 (90 degrees). The notebook uses `corner_mask=False`:
failure in an adjacent angular row suppresses whole contour quadrilaterals,
even when the endpoint row itself succeeded. Do not bridge those failures
with invented solver values.

The copied `_method_engine.upstream_muB` still calls
`muB_from_nB_physical`, which scans chemical potential and repeatedly solves
the RMF state. In successful cluster records, median total paired time is
362 s, versus 86 s for semi-analytical and 5.4 s for full analytical. Median
time outside those two method calls is 269 s (upstream recovery plus process
overhead). The shared 600 s watchdog discards **both** outputs on timeout,
including a full-analytical answer that might already have finished. Saved
timeouts do not identify which stage was running when killed.

The existing `_validated_pnm_state_from_nB` accepts the known upstream density
directly; see [[pnm-density-state-recovery]]. Four failed cluster points were
retested locally using that helper and the unchanged analytical methods:

| theta index | radial fraction | Full analytical [m/s] | Semi-analytical [m/s] |
|---|---|---|---|
| 1 | 0.02 | 44.240 | 39.385 |
| 29 | 0.02 | 80.958 | 72.089 |
| 27 | 0.35 | 508.473 | 513.162 |
| 2 | 0.97 | 1242.905 | 6423.642 |

At index 29, fraction 0.02, direct upstream recovery took 0.13 s versus 66 s
for the old inversion **on the same local machine**. These are diagnostic
timings, not a cluster speedup guarantee. All four direct density residuals
were below 3e-10. The local probes used temporary SciPy 1.18.1 because the
installed 1.15.2 failed the macOS loader; the cluster used 1.15.2. Production
source hashes matched local code.

The lone density-check rejection is a branch inconsistency: at index 2,
fraction 0.97, the old inverse returns muB=1620.162757 MeV. Its validated
forward density matches the target to 1e-14, while the wrapper's subsequent
unvalidated `nB_NM` check selects another branch and reports 88.9% error.

Two independent grid/orchestration problems remain:

- `next_probe` chooses the next radial coordinate even if its record failed.
  At index 1 it spent 12 adaptive rounds approaching failed fraction 0.90
  from below (semi-analytical speed only about 2977 m/s), despite a successful
  fraction 0.99 above 12000 m/s. A fresh fraction 0.97 probe succeeds at
  6460 m/s. Failed coordinates must be retried or valid outer anchors used to
  schedule recovery; failures must not become impenetrable search limits.
  Resume currently retains terminal failures, and adaptation only targets
  5000 m/s, not holes in the other contour levels.
- The minimum original fraction 0.02 is too far from the stable boundary
  for the 50 m/s contour at 90 degrees: even the recovered values are above
  50. A new fraction 0.005 probe gives 33.013 m/s (full) and 29.773 m/s (semi).
  Additional inner samples are needed on high-angle rays.

Evidence is saved in `local_analysis_results/26-09-28/endpoint-diagnosis.json`
and `endpoint-additional-probes.json`. No production solver, runner, notebook
or NPY was modified by this diagnosis, and these local samples were not merged
into contour data. A repair should reuse direct-density upstream recovery,
retain per-method successes, retry failed points, correct adaptive scheduling,
and extend inner sampling without changing xi or the numerical reference.

## Local recovery completed (2026-09-29)

The user authorized local retries and merging into the cluster NPY.
`repair_analytical_local.py` now provides this workflow without modifying the
cluster runner or any RMFsolver file. It uses the validated direct-density
upstream helper, unchanged analytical methods, four local workers, and separate
120 s hard timeouts per method. Each terminal physical point is checkpointed
atomically and increments progress once; one method's failure cannot discard
the other method's returned answer.

All **68 failed cluster points** succeeded for both methods. A new inner row
at radial fraction 0.005 adds 30 points, and fractions 0.97 and 0.96 on angular
index 1 add the two missing bracket-refinement points. Final acceptance is
**734/734 for each method**, with **30/30** semi-analytical 5000 m/s brackets
of radial-fraction width at most 0.01. All 634 original successful method
records per method are preserved exactly. The numerical reference and gray
boundaries are unchanged. Maximum local upstream density residual is 1.26e-9.

The original downloaded bytes are retained alongside the updated file as
`T0plus0p1-analytical-comparison.pre-local-repair.npy`. The updated NPY includes
`local_repairs` (versions, source hashes, backup hash, completion status),
per-point failure history, and a fully successful 30 by 21 `repaired_plot_grid`.
The original cluster runtime metadata remains historical; local work again
used temporary SciPy 1.18.1, with no project-environment changes.

`All_Plots_new.ipynb` comparison cell 13 now uses the extended grid, and was
re-executed to regenerate the existing PNG/PDF exports. Programmatic contour
checks confirm that every drawn analytical level is a single connected curve
meeting both angular endpoints, including the semi-analytical 5000 m/s curve.
Existing colors, fonts, limits, line styles and numerical data were preserved.

Evidence is in `local_analysis_results/26-09-28/local-repair-verification.json`
and `local-repair-contour-connectivity.json`. Resume local work with the repair
script rather than the original cluster runner; its copied legacy upstream
recovery and adaptive scheduling have not been changed by this repair.

### Current two-figure presentation

The earlier seven-band preview, using semi-analytical fallback at 2000/5000 m/s,
was superseded at the user's request because overlapping bands looked crowded.
Cell 13 now renders two figures via `COMPARISON_VARIANTS`:

- Unshaded curves through 5000 m/s: `Contour-diff-meth-0.1.pdf`.
- Curves and numerical-to-full-analytical bands only through 1000 m/s:
  `Contour-diff-meth-0.1-bands-to-1000.pdf`.

Both PDFs are saved in `Latex_writting/Hydrodynamical combustion/Plots/`, with
matching local PNG/PDF exports. Both manuscript copies still include the first,
unshaded figure; the second is available as an alternative. Axes and line styles
match. The banded figure uses enlarged 14-point contour labels with small white
backgrounds at alpha=0.85, fading crossing lines locally; unshaded labels remain
11-point without backgrounds. These settings are exposed through the cell's
`BANDED_CONTOUR_LABEL_*` constants. Bands use `BAND_ALPHA=0.16`, follow actual polylines over
shared angular coverage, and show method differences rather than confidence
intervals. Semi-analytical fallback is disabled; missing full-analytical levels
are not extrapolated. Eight tests cover band geometry, label styling, selection and the two
requested variants; execution checks confirm zero bands in the first figure
and no contour above 1000 m/s in the second.
