---
summary: The 540-point isobaric adiabatic scan is supplemented by 209 numerical solves to complete all 30 rays through 5000 m/s, with styled and historical-relativistic overlay figures in 26-10-26.ipynb.
status: current
updated: 2026-10-26
tags: [adiabatic, contour, cluster, slow-front, continuation]
---

# Nonrelativistic adiabatic contour at fixed interface temperature

`new_paper_calculations/26-10-24/` copies the numerical radial-shell/angular-wavefront workflow from `26-08-21`, replaces the moving-flux solve with `solve_front_adiabatic_slow`, and fixes $T(0^+)=0.1$ MeV.
It uses the full neutral EOS, constant pressure and enthalpy per baryon, and conserves baryon and enthalpy flux; see [[slow-front-solvers]].
The nuclear EOS is massive QMC-RMF3 PNM, the quark branch is massless, and the bag parameter is `B_one_forth=189.1565957288247` MeV from the original cold $3n_0$ calibration.
The module supplies the revised weak rate, strong coupling 0.4, and exact diffusion coefficient constant.
No analytical velocity calculation or analytical initial guess is used.

## Reference and grid

The immutable reference is `26-08-24/T0plus0p1-uNmax-state-of-art-complete-5km.npy`, copied into `inputs/relativistic-state-of-art.npy`.
Its complete 30-angle by 18-column display grid supplies exact upstream coordinates and a saved numerical baryon-flux guess at every target.
The original 600-cell August 21 grid is different; its unresolved outer points were replaced by an adaptive outer column in the consolidated reference.
September's later scans use a different $4n_0$ calibration and are not interchangeable.

The bundled preparation script performs no EOS solves: it copies coordinates and fluxes and uses `inputs/legacy-domain.npy` only for historical geometry metadata.
Source hashes and source-grid indices remain in the prepared manifest and result payload.
The old boundary is a sampling choice, not a newly established admissibility boundary.
Its outer radial fraction varies with angle; the physical density/temperature grids and per-point fractions are authoritative, while the inherited one-dimensional fraction axis represents the first angle only.

⚠ Old successful velocities never count as new solutions, and old profiles cannot seed the new integrated-density BVP.
Historical and current transport prescriptions can differ, so their speed differences cannot be attributed solely to relativity.
The inherited 5000 m/s coverage must also be checked afresh.

## Continuation and recovered initialization failures

First-shell targets use saved same-state relativistic numerical fluxes.
Later shells extrapolate new slow-front profiles along the same angle; direct/perturbed angular waves, scalar historical fallbacks, and midpoint bridges provide recovery.
When a radial predecessor is unavailable, a newly accepted immediate-angle profile supplies the shape.
The old 1 MeV bootstrap is removed.

Each saved physical profile contains a separate `continuation_state` with the `slow_front_v1` normalized fields.
The wrapper copies this state and replaces its internal `jB` for each proposed flux, because that internal value overrides the standalone solver guess.
Hidden midpoint bridges use new numerical continuation without requiring a nonexistent same-point historical seed.

The public defaults remain mesh 300, maximum nodes 50000, BVP tolerance `1e-4`, tail truncation `1e-8`, and table tolerance `1e-6`.
Every disposable solver child has a 300-second deadline; the target budget is 1200 seconds, with at most two midpoint bridge levels.

Two initialization issues appeared in the first full scan and were resolved in the wrapper without changing solver source or physical acceptance criteria:

- The thermodynamic table can exhaust its 14 refinement rounds while a spline error moves through successive intervals. Only that exact failure triggers stricter table tolerances, by factors 0.5, 0.2, and 0.1 within the same trial deadline. Source-grid point `(15,8)` agrees across tighter settings to better than `2e-10` relative flux.
- At cold upstream source-grid point `(0,14)`, only one standard RMF starting guess reaches the physical state. Conditional additional starts `(35,25,-3)` and `(40,30,-3)` independently corroborate it using the unchanged two-start agreement, RMF residual, and forward-density tests. The original module seed tuple is restored in `finally`. Neighboring-angle slow profiles also recover `(0,15)` through `(0,17)` after the missing radial predecessor.

Effective table tolerances, added RMF starts, attempts, conservation errors, and BVP residuals are saved explicitly.
No physical input density or temperature is perturbed.

## Running and interpreting output

Run `run_T0plus0p1_numerical.py --workers N` from the repository layout, followed by `plot_T0plus0p1_numerical.py`.
The README contains portable cluster commands and the isolated local SciPy command.
The standard output is `T0plus0p1-adiabatic-slow.npy`; exports are PNG and PDF.
`--smoke` runs six points into a separate output; `--restart` creates a new suffixed output.

Atomic per-cell checkpoints retain all attempts and successful profiles.
Resume verifies the model, exact prepared-domain bytes, authoritative nested states, physical parameters, solver/helper source hashes, and numerical controls before accepting even a completed checkpoint.
Completed matching checkpoints remain byte-identical.
`run_complete` means scheduling finished; `all_targets_successful` distinguishes a fully accepted grid.
Nonzero exit status and explicit masks preserve failures; plotting requires an explicit `--allow-partial` for incomplete data.
The independent `complete_5000_contour_bracket` flag requires new speeds to straddle 5000 m/s at every angle.

## Verified 2026-10-24 result

The clean delivered run has 540/540 accepted cells, 540 outer BVP trial records, and no timeouts.
Table-tolerance refinement was used at 145 cells; extra nuclear starts were used at one cell.
Ten migration tests and six end-to-end smoke points pass; completed-checkpoint resume leaves the file unchanged.
Maximum pressure, enthalpy-flux, and EOS closure errors are respectively `5.805e-9`, `1.806e-10`, and `1.306e-9`.
The largest BVP RMS residual is `1.533e-5`, below the requested `1e-4`.
All saved rates reproduce the revised formula within `4.45e-16` relative; sampled diffusion agrees with direct evaluations within `1.12e-16`.
Six reruns at source points `(0,0)`, `(15,8)`, and `(29,0)` combine mesh 600, BVP tolerance `5e-5`, tail `1e-9`, table tolerance `2e-7`, and two flux guesses; the largest speed change is `4.134e-7` relative.
The payload, PNG/PDF exports, and `verification.json` are in the calculation folder.

⚠ Speeds on the inherited grid span 30.20–4023.32 m/s: all targets are solved, but **no angle brackets the new 5000 m/s level**.
The original plot therefore does not claim complete 5000 m/s coverage; the separate October 26 extension below supplies it without overwriting this historical result.

## Executed October 26 notebook and extended coverage

`26-10-26.ipynb` contains two executed figures styled after `All_Plots_new.ipynb` cell 14: colored labeled contour lines, a white physical region, gray stable-neutron-matter and composition-ceiling regions, and the same axes and typography.
The first shows the new nonrelativistic result; the second overlays solid nonrelativistic and dashed historical relativistic curves with matching colors for the seven levels 50, 100, 200, 500, 1000, 2000, and 5000 m/s.
Both have PNG/PDF exports in `local_analysis_results/26-10-26/`, and every plot cell has its own tuning block.
The notebook only loads saved calculations.

`new_paper_calculations/26-10-26/extend_nonrel_contour.py` added **209 accepted BVP solutions** using the same solver/EOS hashes and numerical continuation states as the base run.
A new inner shell at radial fraction 0.005 connects the 50 m/s curve to the axis; adaptive outer continuation and bisection produce 5000 m/s brackets on **30/30 rays**, at maximum radial-fraction width 0.0025.
The separate extension payload retains all successful profiles and trial provenance; the original 540-point payload and historical relativistic input are unchanged.
The plotting code draws the high-speed contour from its resolved two-column bracket strip, without extrapolating missing velocities.
The largest additional pressure and enthalpy-flux errors are `6.238e-9` and `1.665e-10`; rates agree with the prescribed formula within `4.45e-16` relatively.

The shared gray regions retain the reference figure's saved boundaries, rather than acting as a new numerical acceptance mask.
Direct full-EOS checks at their 12 stored nodes give maximum relative specific-enthalpy mismatches `3.331e-10` on the fixed-interface inner boundary and `9.225e-6` on the cold unit-composition outer boundary; interpolation between nodes retains the original plotting prescription.
The notebook explicitly identifies the overlay as historical versus current: transport/rate changes are included, so visible differences cannot be attributed to relativity alone.

See [[fixed-interface-analytical-contours]] for the older analytical comparison grid and [[steady-front-bvp]] for the moving-flux formulation.
