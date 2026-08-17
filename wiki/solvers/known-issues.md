---
summary: Live bugs and performance traps — the isothermal strict-retry failure, GIL-bound "parallel" notebook cells, and asymmetric bisection depth.
status: current
updated: 2026-08-17
tags: [solver, bugs, performance]
---

# Known issues and traps

## Isothermal strict-retry — 2 tests deliberately failing

`solve_front_isothermal` had a pre-existing crash caused by two undefined legacy compatibility identifiers for the fixed $T(0^+)$ mode and target. The references were copy-paste debris from the energy-conserving solver and made every call raise `NameError`. Removing that dead line in the 2026-07-23 cleanup restored 4/6 isothermal tests.

That fix **exposed** (did not cause) a deeper pre-existing failure: the coarse-recovery strict-retry path fails with a **singular Jacobian**. Two regression tests remain failing by decision:

- `test_solve_front_isothermal_strict_retry_regression`
- `test_solve_front_isothermal_strict_profile_retry_regression`

Both expect `jB == 0.20069404233893348`.

**Do not treat these as a regression from the cleanup.** Restoring an unconditional `jB_upper_bound = min(jB_upper_bound, max(3.0, 1.5*jB_guess))` causes an infinite runaway, so that is *not* the fix. The correct original jB-bound logic is in no backup — cluster and `backup/` copies predate the strict-retry feature. Before attempting a fix, ask where the 0.20069 regression value came from and what the intended strict-retry bound behaviour is.

## ⚠ `ThreadPoolExecutor` gives zero parallelism

`_pnm_state_at_seed`, `analytic_velocity_bound` and `semi_analytic_velocity_bound` are pure-Python/scipy and hold the GIL. A `ThreadPoolExecutor` gives **no speedup at all**.

Measured 2026-08-07 on `26SU-10` cell 24: `ThreadPoolExecutor(max_workers=8)` driving `solve_boundary_curves` ran at **101% CPU** with CPU time equal to elapsed time — fully serial across ~356 evaluations, nearly two hours.

The cluster scripts (`run_hydro_analytic_contour.py`, `steady_front_scan.py`) use `ProcessPoolExecutor` for exactly this reason. Notebook cells reach for `ThreadPoolExecutor` because it is easier to set up and shares module state, and the `max_workers=8` argument makes it *look* parallel.

**When estimating a notebook scan, budget one core.** Roughly 1 s per PNM state (≈2 s since the seed-agreement validator) and **10–20 s per velocity bound**. To actually parallelize, switch to `ProcessPoolExecutor` with plain picklable arguments — the ray-grid entry points already take an `executor` parameter, so the swap is local to the caller. Prefer reducing the evaluation count first.

## Gap and A endpoints need very different bisection depth

In `new_paper_calculations/26-08-07/_analytic_ray_grid.py`, a single `bisection_steps` for both boundary families is wrong in one direction or the other:

- **$\Delta h$ edges** close at a hardcoded $10^{-4}$ MeV. Near the crossing at $T = 0.01$ MeV the gap runs about $-70$ MeV per $n_0$ (measured $+13.75$ MeV at $2.8n_0$, $-17.69$ at $3.25$, root near $2.997$), so from the $0.9n_0$ coarse bracket this takes **19 halvings**. With the old 8 it failed as `gap_bottom_endpoint: Delta h residual did not close`.
- **$A$ edges** close at $1 - A \le 5\times10^{-3}$ and are already inside it at the coarse probe ($A = 0.99995$ at $9.1n_0$), so they need essentially **no** refinement.

Coupling them made every $A$ evaluation — the expensive one — pay the gap's depth, which is what turned cell 24 into a two-hour run.

**Use `bisection_steps=8, gap_bisection_steps=20`.** The parameter exists only in the 26-08-07 copy; older dated copies still share one count.

## Also

`max_nodes=800` starves the $T(0^+)=0$ uNmax branch (needs ~1900). `_u0minus_max_collocation_status_is_acceptable` (renamed from `_uNmax_collocation_status_is_acceptable`) ignores its `exact_zero_left` argument, making `accepted_max_nodes` unreachable. See [[unmax-low-temperature]].
