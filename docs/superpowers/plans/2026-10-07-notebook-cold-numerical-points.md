# Notebook cold numerical comparison points

**Goal:** Add six freshly solved numerical contour dots below 0.1 MeV and a reproducible computation cell immediately before the existing log-temperature figure.

**Architecture:** A separate notebook helper runs bounded public `solve_front_isothermal` calls, searches density for the requested speed, and checkpoints every evaluation. Existing analytical and numerical archives, solver equations and plot styling remain untouched. Analytical roots and historical profiles are guesses only, never accepted new numerical data.

**Defaults:** Temperatures are existing saved-boundary values 0.01, 0.027550997842357025 and 0.09779327685429287 MeV; speeds 500 and 2000 m/s. Two workers; 90 seconds per call, four calls and 300 seconds per target. Accept speed error <= 0.5% only with all numerical diagnostics passing. Reuse continuation only for nearby temperature and coexistence-distance ratios; otherwise use the supported layer-resolved initializer.

- [x] Add failing tests for bounded/resumable targeting, validation, nearby-profile selection, save-before-progress, interrupted reservations and numerical-only plotting.
- [x] Add `new_paper_calculations/26-10-02/notebook_isothermal_cold.py` and `_cold_notebook_worker.py`. Use the existing isolated runtime and bounded-process helper without editing them or invalidating analytical checkpoints.
- [x] Insert a tunable calculation cell before the log figure; merge new accepted targets in memory and retain the same colors. Leave the linear figure and all prior data unchanged.
- [x] Run mocked tests, then this small six-target live batch. Retain failures honestly; no full contour sweep. All six standard targets pass; the tighter 0.01-MeV check timed out and is explicitly unresolved.
- [x] Execute the plotting cell with saved results, inspect the figure, and update the relevant wiki method, index and session log.

Verification command: `MPLCONFIGDIR=/private/tmp/nsm-matplotlib ./bin/python3 -m unittest discover -s tests -p 'test_*26_10_02.py' -q`.
Live verification uses only the new notebook cell and writes a separate `local_analysis_results/26-10-02/isothermal-cold-numerical-points.npy` checkpoint; no numerical data may be synthesized from dashed analytical curves.
