---
summary: 26-10-02.ipynb now includes a fresh local analytical calculation for all 14 log-temperature contour levels through 1000000 m/s, plus separate saved-numerical comparison plots.
status: current
updated: 2026-10-07
tags: [method, analytical, isothermal, visualization, checkpoint]
---

# High-speed isothermal analytical contours

The logarithmic-temperature visualization in `26-10-02.ipynb` adds seven analytical levels: 10000, 20000, 50000, 100000, 200000, 500000, and 1000000 m/s.
The linear-temperature panel and adiabatic comparison retain their previous displayed levels.
The window stays $1\leq n_B(0^-)/n_0\leq5.5$ and $0.01\leq T(0^-)\leq120$ MeV.
All additional velocities use the existing public `analytic_velocity_isothermal`, $\xi=-0.85$, automatic maximum $a(0^+)$, PNM, zero strange mass, and $B^{1/4}=189.1565957288247$ MeV.
No solver source was modified.

## Reproduce from the notebook

The isothermal calculation cell (zero-based cell 4) is now standalone: it imports saved numerical data, obtains the validated static boundaries, and calls `notebook_isothermal.calculate`.
The helper and its stdlib-only worker entry point live under `new_paper_calculations/26-10-02/` as `notebook_isothermal.py` and `_isothermal_notebook_worker.py`.
The worker sets thread limits before importing numerical libraries and runs `analytic_velocity_isothermal` in a killable process.
No adiabatic cell or numerical BVP run is required.

- `CALCULATE_ANALYTICAL = True` is the default: calculate/resume all 14 levels at 46 temperatures, or **644 contour targets**. Cached boundaries are reused; old analytical velocities/targets/repairs are not calculation inputs.
- `RUN_SUBDIRECTORY = 'isothermal-notebook-run'` saves a separate `isothermal-analytic-logT.npy` under the existing local results directory. Use the same name to resume and a different name for a completely fresh run. Previous inputs and generated reference datasets are never overwritten.
- `CALCULATE_ANALYTICAL = False` loads that run's checkpoint when present; otherwise it explicitly loads the previously completed cluster/local comparison. This mode makes no solver calls.
- Rerun the separate logarithmic-temperature visualization cell (now zero-based cell 8) to obtain the figure. Cell 5 retains the linear-temperature view. Neither plotting cell calculates velocities. A new numerical calculation cell immediately before the log figure adds six bounded cold comparison roots; see [[isothermal-notebook-cold-points]].

The fresh notebook runner uses saved static coexistence/composition boundaries, validated numerical archive hashes, and a safeguarded density-root search.
It starts with no analytical-speed samples, sharing newly calculated off-target samples between speed levels at the same temperature.
At a clipped density edge it evaluates the actual speed before classifying targets outside the window, avoiding the old near-edge stepping problem.
Controls are exposed in the calculation cell: two workers, 90 seconds per trial, 600 seconds per target, at most 32 evaluations per target, and 0.5% speed tolerance.
Completed evaluations are atomically saved before progress updates; successful targets are retained on resume, and input/source/runtime/control changes reject an incompatible checkpoint.
The notebook now shows a runtime-check bar, a fixed-total contour-target bar, and a separate saved-evaluation counter that refreshes after every completed solve (including failures).
Several evaluations may locate one target; accepted/unresolved counts distinguish completed work from converged contours.
Resume restores both counts, excluding interrupted reservations from the completed-evaluation counter.
Interrupted reservations remain charged to the budget; recovery proposes a new interior density instead of becoming stuck on a previously reserved point.
Unresolved targets remain explicit and break plotted paths.

⚠ The project's SciPy 1.15.2 PROPACK binary is rejected by the macOS loader (`__thread_bss` zero-fill section offset); this fails before any EOS calculation and is not a physical convergence failure.
The earlier notebook default accidentally used that broken installation despite the temporary runtime used in development.
This is now fixed with a separately installed SciPy 1.18.1 at `local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64/`; existing environment packages and solver sources are unchanged.
`WORKER_IMPORT_PATHS = None` automatically selects a sidecar matching the current Python/platform; `()` explicitly uses kernel packages, and a tuple of paths provides an explicit override.
The runtime directory is persistent, not under `/private/tmp`, and its actual loaded version/path is printed and stored in checkpoint provenance.
On a computer without this runtime, the kernel is tried; if its full solver import fails, preflight prints a one-time isolated installation command before any point reservations. It never installs packages automatically.
Rerunning the updated calculation cell reloads the helper so a previous failed import does not require a kernel restart; reopen/reload the notebook from disk first if the editor still displays the old source.
The existing embedded figures are the completed reference; the full 644-target fresh workflow is not pre-run as part of this code edit.
Verification of the fresh runner used 66 focused tests and a nine-target live check at 0.01, 20, and 83.8593 MeV for 50, 500, and 1000000 m/s.
Eight targets converged and the warm 50 m/s target was correctly outside the density window; all accepted target errors were below 0.457%.
The live check is saved separately in `local_analysis_results/26-10-02/notebook-smoke-final/` and used 90 analytical evaluations.
Executing the standalone notebook calculation cell against that checkpoint resumed without adding any evaluations; no full 644-target rerun was performed.
After the runtime/progress fix, 70 focused tests passed, including runtime selection, failure-before-reservation, per-evaluation save-before-progress, and resume counters.
The actual cell also ran in a real Jupyter kernel with default runtime selection on two 500 m/s targets at 0.01 and 20 MeV: 499.2880 and 500.7463 m/s, respectively, in 24 evaluations.
That separate `notebook-runtime-fix-check/` checkpoint resumed with zero new evaluations; the production run directory was not created by verification.

## Targeted calculations and saved provenance

The local helper scripts are under `new_paper_calculations/26-10-02/`:

- `extend_isothermal_levels.py`: four levels from 10000 through 100000 m/s, at the original 46 temperatures; 184 scheduled targets.
- `extend_isothermal_outer_levels.py`: three further levels through 1000000 m/s; 138 scheduled targets. It reuses accepted off-target evaluations from the first extension via a separately hashed seed cache.

Each temperature chain uses the already solved fixed-temperature physical boundaries and validated analytical samples; it performs no new boundary scan, chemical-potential scan, or BVP solve.
Log-speed secants propose densities inside actual calculated speed brackets; when no bracket exists, halve the remaining physical density interval.
Predictions are never accepted or plotted as calculated velocities.
The default uses two local workers, one BLAS thread per worker, a 90-second hard evaluation limit, and at most 24 new evaluations per target.
Attempt reservations and completed evaluations are saved atomically; target progress advances only after saving.
Terminal targets are retained, while changed input/source/control metadata reject incompatible resume.

Separate artifacts in `local_analysis_results/26-10-02/`:

- `isothermal-high-speed-extension.npy`
- `isothermal-outer-speed-extension.npy`
- `isothermal-high-speed-seed-cache.npy`
- `isothermal-high-speed-extent-probe.npy` (the preliminary allowed-state check at $T(0^-)=0.01$ MeV and $n_B(0^-)/n_0=4.8$)

The first extension contains the original input hashes; the second hashes the seed cache, which itself records its original and first-extension provenance.
The loader verifies input hashes, temperature axes, physics, domain fingerprint, and non-overlapping target identifiers, then adds the two extensions in memory.
The downloaded checkpoints, earlier local recovery, and original numerical archives remain unchanged.

## Acceptance and visualization

Each accepted target agrees with its requested speed within 0.5% and passes the established analytical identity, composition, endpoint momentum, maximum-interface, and slow-front checks.
In particular, the wrapper retains `momentum_flux_ratio <= 1e-3`; moving states at exactly $a(0^+)=1$ are not accepted.
Only consecutive accepted temperature targets connect; missing, pending, failed, or outside-window rows break paths.
No zero-temperature anchor is added.

The seven original numerical levels retain their saved warm curves at $T(0^-)\geq20$ MeV and 58 distinct accepted cold dots below 20 MeV.
On 2026-10-07, six additional full numerical roots at 500 and 2000 m/s were added below 0.1 MeV, increasing the log-panel cold count to 64; their calculation and validation limits are in [[isothermal-notebook-cold-points]].
A subsequent bounded batch added one numerical 10000 m/s dot at 0.02755 MeV (65 cold dots total); 50000 and 500000 m/s attempts remain unresolved after 90-second timeouts, including a warmer fallback. See [[isothermal-notebook-cold-points]].
There are no warm numerical curves at the seven added speeds.
The speed legend uses `k` for thousands and `M` for millions of m/s, and is placed inside the gray stable-matter region to leave the new curves unobscured.
All colors, levels, legend positions, sizes, and export settings remain tunable in the notebook visualization cell; the cell calls no solver.
Exports retain `isothermal-analytic-numerical-logT.png` and `.pdf` in the same local results directory.

## Completed local extension

Both campaigns completed: **308 accepted contour targets, 14 outside-window targets, and no unresolved targets**.
Each of the seven new contours has one continuous sequence of 44 validated points; the two hottest scheduled temperatures have no allowed density interval.
The merged notebook contains 605 accepted and 39 outside-window targets across all 14 levels.
The two root searches used 660 and 515 analytical evaluations, respectively; these include off-target bracket samples, not 1175 separate plotted contour points.
Maximum target-speed error is 0.4971%; maximum relative velocity-formula residual is $4.38\times10^{-12}$, and maximum momentum-flux ratio is $1.183\times10^{-4}$, below the unchanged $10^{-3}$ limit.
At the 0.01-MeV floor the 100000 and 1000000 m/s contours lie at approximately $3.6531n_0$ and $4.8786n_0$, respectively, filling the previously empty cold-region wedge.

`isothermal-high-speed-validation.json` records final hashes and residual statistics.
One off-target bracket evaluation at 0.01 MeV reported an integration roundoff warning; its diagnostic is retained, and it is not a plotted contour root. No accepted new contour root carries that warning.
All 60 focused comparison/analytical-workflow tests pass, including preservation of baseline points, extension compatibility, numerical-only level selection, and the 58-dot cold numerical display.
The updated loading and plotting cells were executed and their outputs saved; rendering imported no `RMFsolver` modules.

The original extension campaigns used temporary SciPy 1.18.1 through `PYTHONPATH` because even a fresh SciPy 1.15.2 wheel failed the macOS loader.
The notebook now uses the persistent project-local runtime described above; the project Python environment remains unchanged.
See [[analytical-xi-minus0p85-cluster]] for the original data and comparison rules, and [[isothermal-analytic-front-speed]] for physical approximation limits.
