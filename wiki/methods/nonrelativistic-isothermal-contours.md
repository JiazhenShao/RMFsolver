---
summary: Superseded contour-target plan; the scan was implemented on 2026-10-27 as a polar mesh (10-100 MeV) and a log-T row mesh (0.01-10 MeV) per the user's corrections — see isothermal-contours-polar-and-row-meshes.
status: superseded
updated: 2026-10-28
tags: [method, plan, isothermal, numerical, analytic, cluster, contour]
---

# Plan: nonrelativistic isothermal contours

⚠ Superseded 2026-10-28 by [[isothermal-contours-polar-and-row-meshes]].
The user replaced per-temperature contour targets with a polar quarter-ring mesh at high T and a Cartesian (fraction, $\log T$) mesh at low T, with no evaluation inside the gray regions.
The method-aware acceptance rules below were kept.

**Status: revised plan only; the new contour workflow and production scan have not been implemented or run.**
This replaces the earlier proposed general cold-solver repair phase.
The independently reviewed fix is already implemented in [[isothermal-slow-shooting-initializer]]; no new initializer, general rate-table rewrite or change of physical equations is planned.
The review did not change any solver source.

## Physics and sampling

Use `solve_front_isothermal_slow(initial_guess_mode="auto")` and `analytic_velocity_isothermal_slow`.
Keep massless quarks, QMC-RMF3 PNM, `B_one_forth=189.1565957288247` MeV, `n0=0.16 fm^-3`, automatic exact average-chemical-potential ceiling and analytical `xi=-0.85`.
Retain the revised Madsen rate, `alpha_s=0.4`, exact transport constant, full numerical diffusion and leading analytical diffusion.
All temperatures are strictly positive; no floor, zero-temperature anchor or adiabatic fitted xi is introduced.
Use `v=c*u_0minus`, without the legacy relativistic conversion.

| Panel | Temperature schedule [MeV] | Levels [m/s] |
|---|---|---|
| Cold, logarithmic axis | `np.geomspace(0.01,10.0,31)` | 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000 |
| Warm, linear axis | `np.linspace(10.0,100.0,19)` | 50, 100, 200, 500, 1000, 2000, 5000 |

Solve upstream density for each speed at each temperature, within `1 <= nB_0minus/n0 <= 5.5`.
Deduplicate the seven common targets at 10 MeV: 49 temperatures, 560 initial targets per method, 1120 total, excluding off-target probes and refinements.
The extra seven high-speed seam targets belong only to the cold panel.
Both methods share freshly validated static boundaries, with source/physics hashes; no legacy warm mask is inherited.
Boundary roots outside the display window produce explicit clipped/excluded rows, not a global domain-construction failure.

## Reuse and execution

Create `new_paper_calculations/26-10-27/isothermal/{shared,local,cluster}/` with one engine and thin local/cluster launchers.
Reuse the numerical October 1 log-temperature workflow and the October 2 analytical/notebook refinements described in [[isothermal-log-temperature-contours]] and [[isothermal-high-speed-contours]].
Retain their bounded subprocesses, single-writer locks, atomic reservations/checkpoints, profile sidecars, runtime/source/input fingerprints, separate evaluation/target progress and resume semantics.
Old relativistic points may provide density/current guesses only; old velocities and incompatible physical-profile payloads cannot become accepted new results.

Calculate and persist analytical roots first; reuse their density/current estimates in the numerical search instead of launching nested analytical proposal scans.
Share successful off-target evaluations across levels and save compatible new continuation states.
Retain adapted meshes through the implemented auto mode. Only use nearby continuation: temperature and positive distance from coexistence must each differ by at most a factor of two; choose the closest eligible state in those logarithmic coordinates.
If no eligible state exists, use the auto initializer afresh.
Persist analytical `a_0plus`, `I2`, `lambda_n` and EOS endpoints so later xi changes can reuse the thermodynamics.

Use four local workers and the allocated cluster CPU count, with one BLAS thread per worker.
Support method selection, smoke/dry runs, resume, output selection and solver-free plot-only operation.
Keep numerical defaults `n_mesh=300`, `tol_bvp=1e-4`, `tail_eps=1e-8`, `max_nodes=50000`, `kappa_factor=1`, `path_rtol=1e-6`.
Use 90-second analytical and 120-second numerical evaluation limits, up to 24 analytical or 12 numerical evaluations per target, and a 600-second accumulated target budget.
Allow at most two attempts per identical physical state, including the diagnostic retry below; continue searching other densities after both fail.
All reservations remain spent across restarts; no recursive unbounded recovery or automatic budget enlargement.

## Method-aware acceptance and failures

Both numerical solution methods must have solver success, positive finite speed/current, interior composition, nonrelativistic isothermal metadata, matching microphysics, conservation and EOS closure at most `1e-8`, boundary residuals at most `1e-7`, and physical ODE residuals at most `max(5*tol_bvp,1e-7)`.

- **Collocation:** require `bvp_status=0`, finite `bvp_max_rms_residual`, and RMS at most `tol_bvp`. A NaN/Inf RMS is rejected even when the solver says success. Permit one same-state retry with `n_mesh=600`, retaining all acceptance tolerances.
- **Shooting fallback:** require `solution_method="shooting"`, finite `shooting_flux_change<=1e-6`, finite absolute `shooting_mismatch<=1e-8` and the same direct physical checks. NaN collocation RMS is expected and stays NaN; do not invent a successful collocation residual.
- **IVP/event failures:** preserve the exception and state as unresolved. A failed tighter shot cannot be replaced by its unverified coarse flux. Use nearby accepted states and other density proposals within the ordinary budgets; never interpret failure as physical exclusion.

The independent review reproduced the NaN collocation case at 0.1 MeV, `4.8n0`; the 600-node retry and independently validated shooting recover it.
The 0.01 MeV, `5.2n0` tighter-LSODA event failure remains unresolved, including after nearby continuation and tighter path tolerance.
Accepted 0.01 MeV states at `5.0n0` and `5.1n0` bracket one million m/s, but do not prove all high-speed temperature targets are accessible.
Avoid requiring a successful numerical evaluation arbitrarily close to the composition ceiling when an interior accepted bracket already locates the requested level.

## Contour accuracy, pilot and output

Require target-speed errors below `1e-3` using actual accepted method evaluations.
Check connectivity at arithmetic temperature midpoints in the warm panel and geometric midpoints in the cold panel.
Solve the midpoint target independently; subdivide up to three levels if its density differs from the corresponding endpoint chord by more than `0.005*n0`.
Failed, pending and unresolved intervals stay disconnected. Budget exhaustion means work finished, not contours complete.

Pilot both methods at temperatures 0.01, 0.1, 1 and 10 MeV for levels 50, 5000 and 1000000 m/s, plus 50 and 90 MeV for levels 50, 500 and 5000 m/s: 18 targets per method.
Also classify the 100 MeV boundary row and exercise the seven shared seam targets.
Use measured target completion times and failure/retry rates to project cost; the 0.60–16.76 s point timings from the review are not per-contour timings.
Run local production if the projection is at most 30 minutes; otherwise prepare cluster commands and the measured estimate without submitting jobs.
Any unresolved pilot regions remain explicitly reported in the production plan and final coverage, rather than triggering an unapproved solver redesign or relaxed acceptance.

Create and execute `26-10-27-isothermal.ipynb` with warm and cold numerical/analytical comparisons, distinct line styles, EOS-based gray exclusions, per-cell tuning blocks and PNG/PDF exports.
Preserve earlier figures, notebooks, solver sources, manuscript and calculation artifacts.
Write coverage, physical exclusions, unresolved numerical failures, method counts and timings alongside the data.

## Verification

- Regression-check the auto initializer, its validated fallback and default-mode compatibility. Retain the original cold map and new NaN/outer-event cases as distinct scenarios.
- Test both acceptance branches, rejection of nonfinite collocation RMS, the bounded 600-node retry, and rejection of unverified coarse shooting results.
- Check independent mesh, EOS table, tail and BVP tolerance changes at low/high-speed cold roots and representative warm roots, requiring speed stability within `1e-3` and conservation within `1e-8`.
- Directly verify profile rates/diffusion and fixed pressure, temperature and baryon flux at representative accepted roots; record slow-flow diagnostics.
- Test seam deduplication, 560-target counts, physical clipping, no connections across failure gaps, timeout/resume accounting, stale checkpoint rejection and local/cluster parity.
- Use the isolated SciPy runtime and direct-function/unittest execution; report historical failures separately.
- Visually inspect executed notebook outputs, then update this page, the index and one session log entry with actual production outcomes.
