---
summary: Targeted numerical contour following preserves 362 baseline successes, adds full-profile continuation, and refines actual BVP points around the missing speed curves.
status: current
updated: 2026-09-21
tags: [method, cluster, contour, isothermal, continuation]
---

# Targeted numerical isothermal contour repair

The broad adaptive repair code is `new_paper_calculations/26-09-16/isothermal_numeric_repair/`, run with `python3 run_isothermal_numerical.py`.
For the bounded follow-up targeting only visible gaps, bends and a practical cold endpoint, use [[isothermal-targeted-shots]] (26-09-21); its inputs reuse this repair's successful points and profiles.
It follows selected speed contours instead of repeating every unsuccessful cell in the original 600-point grid.
See [[isothermal-contour-cluster]] for the inherited domain and previous production runs.

## Verified baseline and gaps

The completed 26-08-27 payload contains 362 successes, 194 cell timeouts, and 44 failures.
All 358 pre-scaling successes were preserved, and field normalization recovered only four additional cells.
Thus normalization alone did not solve the missing region described in [[known-issues]].

The saved numerical 500 m/s curve ends near 84.8 MeV because higher-temperature outer samples timed out.
The 1000 m/s curve has a warm gap between approximately 64.9 and 77.4 MeV.
The cold region also has a sampling limitation: even the nearly complete analytical grid stops tracing the 50 m/s contour below approximately 1.9 MeV because its smallest radial fraction is 0.02.
Repairing only the original failed cells cannot resolve that sampling limit.

## Method and preservation contract

The new folder contains byte-identical copies of the domain, analytical seeds and numerical checkpoint, with SHA-256 values in `baseline-manifest.json`.
Original successful records are immutable; added numerical solutions and reconstructed seed profiles are stored separately.
The runner verifies the inherited domain fingerprint, input hashes, physics and explicit compatibility of the added optional isothermal API arguments.

At each hidden polar angle, actual accepted numerical speeds bracket each target level (50, 100, 200, 500, 1000, 2000 and 5000 m/s).
Logarithmic interpolation proposes radial coordinates; each proposal is evaluated by the exact numerical BVP.
The requested speed must be matched within 0.5%, and existing BVP, composition and EOS/conservation validation must pass.
Analytical predictions select seeds and coordinates only; they never fill missing numerical velocities.
Boundary sampling can extend inside the original 0.02--0.99 radial-fraction range by repeatedly halving distance to a boundary.
Static phase admissibility and the automatic local composition ceiling are checked before solving a moving front.

Warm 500/1000 gaps are prioritized, followed by progressively colder rays and other warm contours.
Angular refinement limits adjacent accepted positions to 0.05 in density divided by saturation density and 2 MeV in temperature; below 1 MeV, the temperature ratio is limited to 1.5.
Those limits schedule *more* angular bisection; they no longer decide plotted connectivity — see below.

## Connectivity follows the angular schedule, not a step in $(n_B,T)$

⚠ Until 2026-09-21, `connected_segments` joined two consecutive rays only if `within_spacing` passed, i.e. if they lay within `density_step` = 0.05 and `temperature_step` = 2 MeV of each other.
Those are *continuation step sizes*, and the production sweep runs on a 30-node angular axis whose neighbouring roots are 0.11--0.24 apart in $n_B(0^-)/n_0$ and 3--8 MeV apart in $T(0^-)$.
Every adjacent pair was therefore rejected: 109 of 109 links were cut, the repair layer contributed no line geometry above 100 m/s, and the figure showed a perfect row of markers beside a broken curve.
The markers came from `accepted_root_groups`, which scatters each accepted root individually; the lines came from the merge, so the two paths disagreed by construction.

A ray sweep is a valid polyline at whatever angular spacing it was run, so connectivity is now decided by adjacency on `domain["theta_axis_rad"]`: a line breaks only at a failed ray, an angular node that was never scheduled, or a change in branch count.
The geometric test that survives, `connectable`, is a veto against a jump too large to be one branch (`connect_density_jump` = 0.8, `connect_temperature_jump` = 25 MeV, `connect_radial_jump` = 0.25), not a sampling-density test.
`connected_segments` takes the axis as a keyword argument and otherwise reconstructs it from the union of every level's rays.

With the 26-09-16 checkpoint this yields single continuous 50, 100, 200 and 500 m/s curves from $T(0^-)\approx0.4$ MeV to the $n_B(0^-)=n_0$ axis, and one 1000 m/s curve to $\theta = 1.309$ ($T(0^-)\approx86$ MeV).
The repair arcs then cover and remove every baseline grid arc, so no grid information is lost.
Residual gaps are missing data, not a plotting artifact: 1000 m/s lacks the three rays above $\theta=1.309$, 2000 m/s was never scheduled at eight interior nodes, and 5000 m/s has only three accepted roots.
Only real solves, i.e. [[isothermal-targeted-shots]], can remove them.

## Interpolated bridges are drawn dashed

`interpolated_bridges` fills an *interior* gap by interpolating the radial fraction in angle between the two accepted roots that bracket it, then mapping back through `ray_coordinates`.
Interpolating the ray coordinate rather than $(n_B,T)$ matters near the outer boundary, where the contours crowd together: a straight chord across the 2000 m/s gap cuts about 0.17 in $n_B(0^-)/n_0$ inside the true arc, while the ray interpolation follows the boundary.
Bridge endpoints reproduce their two end roots to machine precision, because the map is the exact inverse of the projection used to place the roots.

⚠ Bridges are the only fabricated geometry in the figure, so they are returned separately from the solved segments and every caller draws them dashed — never as part of the `ContourSet`.
A certified targeted interval always wins: `targeted_bridges` trims whatever a real result covers.
An open end of a sweep is never extrapolated and a multi-branch ray is skipped, so the 1000 m/s curve still stops at $\theta=1.309$; 2000 m/s becomes continuous to the $n_B(0^-)=n_0$ axis and 5000 m/s becomes a single short arc.

## Lightweight continuation and bounded work

`solve_front_isothermal` now accepts optional `continuation_guess` and `return_continuation` arguments.
Its `continuation_state` returns the adapted compact mesh and physical $n_K,J_K$ profiles directly from the BVP, without the full plotting-profile EOS reconstruction.
The next solve keeps the adapted mesh and rescales amplitudes to the new endpoints.
The exact reaction rate, local diffusion, physical outputs and separate internal field normalization remain in place.

Baseline files do not contain profiles, so the runner recomputes only needed anchors at their saved currents and requires speed agreement within 1%.
Anchor discrepancies remain explicit and cannot replace the preserved result.
Current brackets depend on the continued and analytical estimates rather than the old global current ceiling; a solution close to a bracket edge requires one expanded-bracket confirmation.
At most two current guesses are used at one physical point in a pass; failed continuation steps are subdivided.
The limits remain 180 seconds per trial and 900 seconds per contour target, with bounded root evaluations and subdivision depth.

Each evaluation is atomically saved before the progress counter advances.
The display-only progress adapter shows a full target bar starting at 210 (4 for smoke), with totals increasing for adaptive/retry batches; `evals` separately counts saved BVP evaluations in the current invocation.
Completed targets include unresolved failures, not just accepted roots; four convergence checks have their own full bar.
This display change leaves checkpoint-bound workflow files unchanged, permitting existing checkpoints to resume.
Startup prints the loaded solver path and SHA-256; the API preflight confirms continuation support but alone does not establish byte-identical solver versions.
The checkpoint records solver/workflow hashes, input hashes, controls, numerical attempts, profiles and contour connectivity.
`run_complete` describes the pass; `level_complete` and `contours_complete` describe actual angular coverage, while `validation_complete` reports independent tighter-mesh/tail checks.
These states must not be conflated.

## The paper figure is one solid curve per level

`26SU-14.ipynb` is the diagnostic view: markers show every accepted root and interpolated stretches are dashed.
`All_Plots_new.ipynb` is the paper view and shows neither, so `continuous_level_curves` returns a single solid polyline per level with interior gaps and open ends already filled.
The interpolated fraction stays auditable through `curve_solved_*` in the archive, printed by the cell as e.g. "1000 m/s curve: 17/20 nodes are accepted BVP roots".

An open end is continued only where the level is bracketed by a lower *and* a higher level that both have a value at that angle, interpolating linearly in $\log(1-f)$ against $\log u$.
The logarithm of the gap to the boundary is the right variable because the contours crowd against $a(0^+)=1$: on every angle where 500, 1000 and 2000 m/s are all solved, this rule reproduces the solved 1000 m/s root to within **0.34 MeV**, and to within 0.05 MeV at the warm end where it is actually used.
It gives 1000 m/s an endpoint at $(1.000, 94.54)$, between the solved 500 and 2000 endpoints at 94.10 and 94.64 MeV — the bracket is only 0.5 MeV wide, which is why the continuation is safe here.
2000 m/s likewise reaches the axis. 5000 m/s is not continued at all: nothing above it has data, so it stays a four-node arc from 7.3 to 37.5 MeV and reads as a fragment; dropping it from the plotted levels is the cleaner option if that matters.
The low-temperature ends are handled separately, by the cold anchor below.

Curves are resampled onto 361 angles with a monotone cubic (`PchipInterpolator`), which removes the polygonal look of the 5-degree angular axis.
Shape preservation rules out overshoot, and the interpolant passes through every scheduled node to machine precision, so no accepted root moves.

⚠ Two of the seven contour colours were both `#1f77b4`, making 50 and 2000 m/s indistinguishable; 2000 m/s is now `#d62728`.
Contour labels are placed manually, because 1000 and 2000 m/s converge near the outer boundary and `clabel` stacked their labels on top of each other: the low levels are labelled at high temperature and the high levels at low temperature, where each is well separated.

## Every contour ends at the cold corner (3 n0, 0)

The delta-muB = 0 boundary meets $T(0^-)=0$ at $n_B(0^-)/n_0 = 2.99999$, the front speed vanishes on that boundary, and the crossing fraction of every level goes to zero with the angle: the solved 50--500 m/s roots follow $f \propto \theta^{0.78}$ and the analytic grid shows the same for 1000--5000 m/s.
So the cold corner is the common endpoint of all seven contours, and `continuous_level_curves(..., cold_anchor=True)` adds it as an exact data point at angle zero.
Between the anchor and the coldest accepted root the curve is a single monotone cubic segment; no intermediate node is invented there.

That segment is only trustworthy if it is short, which is what `new_paper_calculations/26-09-23/isothermal_cold_endpoints/` is for.
`run_cold_endpoints.py` solves the two scheduled angles immediately below the coldest accepted root of 1000, 2000 and 5000 m/s -- six rays, nothing else -- into its own checkpoint, which `continuous_level_curves` merges through `extra=`.
All six converged, matching their target speeds to better than 0.25 m/s against a 0.5% tolerance:

| level | $\theta$ | $n_B(0^-)/n_0$ | $T(0^-)$ | $f$ | accepted $u(0^-)$ |
|---|---|---|---|---|---|
| 1000 | 0.007805 | 3.2574 | 0.470 | 0.1121 | 1000.03 |
| 1000 | 0.013433 | 3.3874 | 0.855 | 0.1687 | 1000.02 |
| 2000 | 0.013433 | 3.6804 | 0.960 | 0.2963 | 2000.06 |
| 2000 | 0.023119 | 3.9462 | 1.817 | 0.4123 | 2000.07 |
| 5000 | 0.023119 | 4.5376 | 2.181 | 0.6698 | 5000.02 |
| 5000 | 0.039788 | 4.8030 | 4.037 | 0.7868 | 5000.22 |

⚠ These are the same rays the cluster reported as `target_budget_exhausted`.
They converge locally in 140--410 s each because the neighbouring accepted roots now give the continuation a usable anchor, so a ray failing in a broad sweep does not mean it is unsolvable -- retry it once its neighbours exist.
The 26-09-16 payload is never resumed, and could not be: its continuation profiles were never copied back from the cluster, so `Repair` would fail its own missing-profile check.
The new folder therefore keeps hash-verified copies of the three baseline payloads and writes a separate checkpoint.

With those six roots the interpolated tail shrinks to $T(0^-) \le 0.42$, 0.72 and 1.23 MeV for 1000, 2000 and 5000 m/s, where it differs from the independent analytic contour by at most 0.018, 0.051 and 0.260 in $n_B(0^-)/n_0$.
Before the solves the 5000 m/s tail was out by 0.68 and visibly crawled along the temperature axis.

## Visualization and verification status

`26SU-14.ipynb` numerical cells 4--5 combine the original data with `isothermal-contour-repair.npy` when available, retaining the established square window, labels and gray shading.
The supplementary grid diagnostics still describe the original 600 cells.
The standalone runner saves a PNG/PDF and supports `--plot-only` and read-only `--dry-run`.
`All_Plots_new.ipynb` cells 15--16 do *not* recompute the merge: they read pre-merged polylines from the hash-checked snapshot `local_analysis_results/All_Plots_new/isothermal_contour_data/isothermal-numerical-contour.npz`, so the notebook stays free of cluster folders and EOS/BVP imports.
⚠ Any change to the merge logic therefore requires rerunning `export_isothermal_contour_archive.py` in that folder; otherwise the paper figure keeps the old geometry while `26SU-14.ipynb` shows the new one.
The archive carries bridges in separate `bridge_vertices_*`/`bridge_offsets_*` arrays for the same reason.
⚠ A live Jupyter kernel keeps its first import of these workflow helpers, so editing them is invisible to Run All; `26SU-14.ipynb` cell 4 now drops them from `sys.modules` to make a plain re-run pick up the current source.

The cluster `--smoke` mode targets the hot 500 extension, warm 1000 gap, a cold 200 point, and the 50 m/s contour at 0.01 MeV.
Representative recovered points are checked with twice the initial mesh, a tenfold smaller tail gap, and tighter BVP tolerance; speeds must agree within 1%.
Local implementation verification covers synthetic orchestration, immutable baselines and bounded warm-state/worker checks.
No production cold recovery or completed cluster smoke result is available yet; the new code must not be described as having already connected the missing curves.
