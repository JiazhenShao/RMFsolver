---
summary: The downloaded 26-10-01 run accepted only 13/427 targets; proposal timeouts and continuation/retry defects make an unchanged production rerun unjustified.
status: current
updated: 2026-10-02
tags: [method, isothermal, numerical, cluster, low-temperature, checkpoint]
---

# Cold numerical contours on a logarithmic temperature axis

The user selected a separate analytical $\xi=-0.85$ experiment over 0.01–120 MeV: [[analytical-xi-minus0p85-cluster]].
It reuses only static boundaries from this run, not numerical velocities or profiles; the numerical checkpoint discussed below is preserved unchanged.

## Downloaded-run audit: production not validated

The completed checkpoint contains 3/217 accepted base roots and 10/210 accepted refinements, from 27 converged numerical evaluations and 358 numerical timeouts.
All 61 boundary-temperature tasks succeeded, but all 225 analytical density-proposal tasks hit their 60-second limit before launching a numerical trial.
The 14 tighter endpoint checks remain blocked; `work_complete=True` does not mean `contours_complete=True`.
Saved hashes match all nine local RMFsolver files and six workflow files, so this is not an outdated upload.

⚠ The implementation's `layer_resolved` setting was bypassed whenever a profile was supplied: the wrapper always selected some available profile, even without a closeness threshold, and the solver correctly gave continuation precedence.
Every converged record reports `initial_guess_mode_used='continuation'`.
Transfer preserves the shape in compact coordinate and rescales field amplitudes but does not retune the physical layer width.
Separately, `propose_density` can return no next point after the original guess and one half-distance retry fail; 176 targets had exactly two attempts, and only 385/900 numerical attempts were spent.
These are workflow limitations, not evidence that finite-temperature fronts do not exist.

Successful evaluations took about 72–177 seconds for only 1–2 BVP iterations, with 17,799–45,845 local EOS-state calls.
Timeout records do not locate the internal stall, so sensitive seeds cannot be identified as the sole cause without controlled measurements.
The three earlier 0.01-MeV solutions in [[isothermal-cold-numerical-probes]] remain evidence that the cold BVP is solvable at some points.
Do not repeat the unchanged sweep; prefer an explicitly approximate analytical figure or first demonstrate affordable seed transfer with a tiny controlled comparison.
No retry or solver change was performed during this audit.

The analytical $\xi\to-1^+$ limit maximizes the frozen formula, not a proven bound on the exact local-EOS numerical solution; see [[isothermal-analytic-front-speed]].
At saved numerical endpoints the limit exceeds the three previous 0.01-MeV probe speeds by 0.76%, 2.40%, and 5.34%, but exceeds the 27 new converged speeds by 11.4–76.3%.
These checks reuse numerical endpoints; they are not independent finite-flux analytical solves or a domain-wide error bound.

Evidence, reproducible read-only audit, source hashes and CSV comparisons: `local_analysis_results/26-10-01/isothermal-logT-audit/`.
The download lacks the 27 new `profiles/evaluation-*.npy` files, preventing independent profile-area checks locally; bundled input profiles are present.
The implementation description below records intended behavior, not demonstrated cold-contour completion.

## Workflow

The source and portable input bundle are in `new_paper_calculations/26-10-01/isothermal_numeric_logT/`.
Run `python3 run_isothermal_numerical.py` there on the cluster.
The calculation targets the seven established velocity levels at 31 geometrically spaced temperatures from 0.01 to 20 MeV: **217 base contour targets, not a density-temperature grid**.
No new BVP is submitted above 20 MeV.
The warm archived plot is retained; smoothed archive vertices are display geometry, never numerical solver records.

## Inputs and physics

The `inputs/` directory snapshots and hashes the domain, baseline numerical results, repair results, cold-endpoint results and plotting archive from `local_analysis_results/All_Plots_new/isothermal_contour_data/`.
The original domain fingerprint is preserved.
Three optional continuation profiles come from the validated 0.01-MeV probes in [[isothermal-cold-numerical-probes]].
The prepared bundle also reuses 25 compatible saved cold-endpoint profiles, retaining per-point source provenance (28 profiles total).
Missing historical repair-profile caches do not block this workflow.
Source-only Git checkout is insufficient for cluster transfer: upload the bundled `inputs/` directory as well as the Python scripts and current `RMFsolver/phase_velocity.py`.

Defaults remain $B^{1/4}=189.1565957288247$ MeV, pure neutron matter, $m_s=0$, automatic maximum $a(0^+)$, and density ratios between 1 and 5.5.
Analytical proposals use `xi=-0.5`; accepted speeds always come from the full numerical solver.
Static boundary roots use direct fixed-density nuclear recovery and cached EOS states, with saved boundary seeds and bracket expansion in density, not broad nuclear chemical-potential scans.
The upper-boundary convention remains `a_0plus=1-1e-9`; no moving BVP is evaluated at composition one.

## Numerical changes

`solve_front_isothermal` adds a final optional argument `initial_guess_mode="default"`.
The default retains the previous initialization.
`"layer_resolved"` ports the diagnostic logarithmic physical-coordinate mesh and algebraic-times-exponential core-to-tail seed into the supported API.
A supplied physical continuation profile takes precedence; unsupported layer scales explicitly fall back to the default initializer.
Diagnostics record requested/used modes, fallback reason and `bvp_max_rms_residual`.
The exact rate, local diffusion, normalized fields and finite-flux closures are unchanged.

Seven shared temperature anchors per level are solved before six independent five-interval continuation blocks per level.
Density corrections use actual converged numerical speed brackets and safeguarded secants, with steps relative to coexistence distance rather than a fixed density increment.
Successful off-target evaluations and their lightweight adapted profiles are shared between levels.
Targets require relative speed error at most 0.5%, passing BVP/boundary/composition diagnostics and endpoint EOS/flux checks.
Initial current bounds span one quarter to four times available estimates; one near-bound expansion consumes the ordinary target allowance.

## Persistent limits and visibility

- Numerical trial: 180 seconds including startup.
- Target: four attempts and 600 accumulated seconds, including analytical proposal time.
- Identical physical state: at most two attempts.
- Whole checkpoint: 900 numerical attempts, including refinement and 14 reserved accuracy checks.
- Boundary temperature: a separate bounded 180-second worker.
- Refinement: at most one logarithmic midpoint per original temperature interval and level, triggered by failure, density-ratio jump above 0.05 or log-temperature chord deviation above 0.01.

Reservations are atomically saved before launch; each completed evaluation is saved before its progress callback.
Interrupted reservations remain spent across restart; exhausted/terminal work is not silently repeated.
An interrupted boundary receives at most one additional recorded attempt; a saved completed accuracy evaluation is recovered without rerunning it, while a never-launched placeholder is released.
One writer owns the checkpoint lock, while plotting may read atomic snapshots independently.
Full-width bars distinguish boundaries, base targets, refinements and checks; numerical-attempt consumption is reported separately.
Resume rejects incompatible physics, axes, controls, input hashes or solver/workflow source hashes.

Outputs are `isothermal-logT.npy`, `profiles/`, the portable `isothermal-logT-plot.npz`, and PNG/PDF figures.
The figure refreshes after every accepted target.
`--plot-only` rebuilds it without loading the solver; `--dry-run` checks inputs/schedule without EOS calls.
`--smoke --workers 4` performs four real bounded targets in a separate directory: low/high speeds at 0.01 MeV, an intermediate temperature and the 20-MeV join.

## Honest plotting and validation

`All_Plots_new.ipynb` has a separate saved-data loading/visualization section for the log-temperature result; the preceding linear-temperature figure is unchanged.
New cold lines connect consecutive accepted scheduled targets only.
The new 20-MeV root is used once when the archived warm seam agrees within 0.01 in density ratio; inconsistent joins are flagged and disconnected.
There is no artificial zero-temperature anchor, analytical substitution or cosmetic connection across failed cold targets.

Fourteen accuracy checks cover all seven levels at 0.01 and 20 MeV with 450 initial nodes, BVP tolerance `1e-5` and tail cutoff `1e-4`; speed changes must stay below 1%.
Work completion and contour completion are separate checkpoint fields; exhausting the budget cannot mark unresolved contours complete.

Local verification uses mocked orchestration, initializer/continuation contracts, immutable-input/hash tests, plotting and read-only preflight.
The broad legacy numerical regression was stopped when it entered live EOS work; expensive cold recovery and accuracy checks remain cluster work, not an established result of implementation.
The existing alternate SciPy verification runtime was used because the project SciPy binary fails to import locally; no environment package was changed.

⚠ Cold numerical timeouts do not establish absence of a physical front, and finite-temperature calculations do not validate an infinite physical speed at exactly zero temperature.

See [[isothermal-targeted-shots]], [[isothermal-contour-repair]] and [[isothermal-contour-cluster]] for predecessor workflows.
