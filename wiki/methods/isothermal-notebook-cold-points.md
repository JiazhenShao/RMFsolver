---
summary: Seven added cold numerical roots include one at 10000 m/s; 50000 and 500000 m/s attempts timed out, and the notebook preserves diagnostics rather than plotting failed trials.
status: current
updated: 2026-10-07
tags: [method, numerical, isothermal, notebook, low-temperature]
---

# Additional cold numerical dots in the comparison notebook

`26-10-02.ipynb` now has a full numerical computation cell immediately before the logarithmic-temperature visualization (zero-based cells 7 and 8).
It initially added six accepted dots, preserved all 58 existing numerical cold dots, and left the analytical data, warm curves, solver equations and plotting style unchanged.
The subsequent high-speed batch adds one more accepted dot, giving **65 numerical cold dots**, not successful coverage of all requested high speeds.

| Temperature (MeV) | Requested speed (m/s) | Density ratio | Numerical speed (m/s) |
|---|---:|---:|---:|
| 0.01 | 500 | 3.0065446226 | 500.1674673 |
| 0.01 | 2000 | 3.0251196937 | 2002.3679899 |
| 0.027550997842357025 | 500 | 3.0150292269 | 500.3598580 |
| 0.027550997842357025 | 2000 | 3.0563985172 | 2006.2907129 |
| 0.09779327685429287 | 500 | 3.0419166567 | 500.9914121 |
| 0.09779327685429287 | 2000 | 3.1487340130 | 2000.3278612 |

## Reproduce and inspect

Run the existing isothermal loader/calculation cell first (`CALCULATE_ANALYTICAL=False` avoids recomputing analytical data), then the new cold numerical cell, then the log visualization.
`CALCULATE_COLD_NUMERICAL=True` calculates or resumes the six-target checkpoint; `False` loads saved results only.
The separate `CALCULATE_HIGH_COLD_NUMERICAL` switch controls a second batch at 10000, 50000, and 500000 m/s, initially at 0.01 and 0.027550997842357025 MeV.
Its `HIGH_COLD_*` controls and separate checkpoint preserve the original six completed roots.
Set both calculation switches to `False` for a strictly saved-data-only numerical cell.
Levels with no accepted high-speed root receive one fallback target at 0.09779327685429287 MeV, saved in a third checkpoint; this is bounded recovery, not an unlimited retry loop.
Same filename and compatible controls preserve completed work; changed inputs, physics, axes, solver/helper hashes or runtime reject incompatible resume and require a separate filename.
Changing the displayed contour appearance never calls a solver.

The new cell exposes temperatures, levels, two workers, a 90-second trial limit, four attempts/300 seconds per target, 300 initial nodes, 12000 maximum nodes, BVP tolerance `1e-4`, and tail cutoff `1e-3`.
Use temperatures from the saved analytical schedule so the boundary states are directly solved, not interpolated.
Default physics are unchanged: $B^{1/4}=189.1565957288247$ MeV, PNM, $m_s=0$, and automatic maximum $a(0^+)$.
The numerical equations do **not** take an analytical $\xi$ parameter.

New source helpers live in `new_paper_calculations/26-10-02/notebook_isothermal_cold.py` and `_cold_notebook_worker.py`.
They reuse the persistent isolated SciPy installation and bounded spawn-process wrapper described in [[isothermal-high-speed-contours]].
No RMFsolver source changed; its phase-velocity hash remains `1a9baab4a77aa15cddd2437fdb4eb5a85f86d6d4a5396ecd6699b7ec39186fe4`.

Artifacts under `local_analysis_results/26-10-02/`:

- `isothermal-cold-numerical-points.npy`: six roots, all 13 final-batch evaluations, residuals, warnings, and adapted physical continuation profiles.
- `isothermal-cold-numerical-high-speed.npy`: separate bounded high-speed targets, including failed and off-target evaluations; only accepted roots are eligible for plotting.
- `isothermal-cold-numerical-high-speed-fallback.npy`: one warmer cold target for each still-missing high-speed level; exhausted attempts remain spent on resume.
- `isothermal-cold-numerical-points-pilot.npy`: preserved interrupted development pilot, including the two expensive over-large correction steps.
- `isothermal-cold-numerical-accuracy.npy`: the unresolved tighter-check diagnostic.
- `isothermal-analytic-numerical-logT.png` / `.pdf`: refreshed comparison exports.

## Small-step correction and initialization

The initial density for each speed comes from the saved analytical contour, but this is a guess only.
New full numerical evaluations determine all subsequent corrections and accepted velocities.
When a numerical bracket exists, use its safeguarded log-speed secant.
Otherwise scale the density's distance from coexistence by the target/computed speed ratio, limited to a factor between one half and two.
This avoids doubling the entire offset when a computed speed is already only 2% below target.

The first pilot exposed that needless doubling and two 90-second timeouts; it was stopped and preserved separately.
After fixing the correction, all six targets converged in **13 evaluations**, about three minutes wall time on two workers, with no final-batch failures.
This total excludes the eight preserved pilot records (four successes, two timeouts, two cancelled calls) and the separate tighter check.

A continuation profile is eligible only when its temperature ratio is within 1.5 and its coexistence-distance ratio within two.
Otherwise the public `initial_guess_mode='layer_resolved'` is used without a supplied profile, avoiding the distant-profile override diagnosed in [[isothermal-log-temperature-contours]].
Current bounds span one quarter to four times the target-speed and eligible continuation-current estimates.
Every attempt reservation and completed evaluation is saved atomically; separate `tqdm` displays count terminal targets and completed evaluations.
An interrupted reservation remains spent, and a failed trial is never turned into a plotted dot.

## Validation and limits

Maximum contour-speed error is **0.3146%**, below the 0.5% criterion.
Accepted roots have maximum BVP RMS residual $7.59\times10^{-5}$, endpoint flux error $1.37\times10^{-12}$, seven-node profile EOS error $4.87\times10^{-15}$, and momentum-flux ratio $3.19\times10^{-10}$.
The checks also require interior maximum composition, finite positive currents, nonnegative and monotone density profiles, and the existing local-diffusion/reaction model labels.

⚠ The 2000 m/s root at 0.01 MeV retains one quadrature roundoff warning; its numerical/EOS/flux acceptance checks pass.
The 500 m/s root at 0.01 MeV was retried with 450 initial nodes, `tol_bvp=1e-5`, and `tail_eps=1e-4`, but hit its 90-second limit.
No speed from that unfinished check exists, so these new points are residual-validated at the stated standard settings, **not demonstrated mesh/tail-independent**.
No zero-temperature point, analytic replacement velocity, or artificial cold line was added.

78 focused tests pass, including numerical-only acceptance, small-step targeting, profile proximity, persistent limits, checkpoint compatibility, save-before-progress and preservation of the old plotting data.
The actual new cell and log figure ran in a Jupyter kernel, resumed with zero additional BVP evaluations, and the rendered 64-dot figure was visually inspected.

## High-speed numerical additions

The second batch targeted 10000, 50000 and 500000 m/s at 0.01 and 0.027550997842357025 MeV without changing the solver or the original six-root checkpoint.
Only the 10000 m/s target at 0.027550997842357025 MeV converged: density ratio **3.2359548050553277**, speed **10003.19563277806 m/s**, target error **0.03196%**.
It required three full numerical evaluations; the first two computed 9345.6827 and 10132.7732 m/s and bracketed the target.
Maximum BVP RMS residual is `9.23e-5`, endpoint flux error `1.34e-15`, seven-node profile EOS error `9.00e-16`, momentum-flux ratio `7.16e-9`, and no warnings were captured on the accepted root.
No additional tighter mesh/tail test was run for this point.

All five other primary targets timed out at 90 seconds per solve.
The 50000 and 500000 m/s levels were then each tried at 0.09779327685429287 MeV, and both timed out at the same limit.
Overall: **eight targets, ten new numerical evaluations, one accepted root and seven unresolved targets**.
A timeout is unresolved numerics, not evidence that a physical solution is absent.
No uncomputed 50000/500000 m/s dots or analytical substitute velocities were added.
The cold-dot plotting branch is now independent of the seven archived warm numerical levels, so validated high-speed dots can appear in their existing contour colors without creating nonexistent warm curves.
All 78 focused tests passed again; the actual calculation/visualization cells resumed in a Jupyter kernel with zero new BVP calls and rendered 65 dots.
The original six-root checkpoint, downloaded archives and solver hash remain unchanged.
