---
summary: Live solver and performance traps, including numerical-isothermal low-temperature recovery and current bounds, cold quadrature, finite-ms cost, GIL-bound threads, and asymmetric bisection depth.
status: current
updated: 2026-08-26
tags: [solver, bugs, performance]
---

# Known issues and traps

## ⚠ PNM root success is unchecked and cold positive temperatures still use adaptive quadrature

Both `RMFsolvePNM` and `RMFsolvePNM_mu` currently return `sol.x` after the nonlinear root call even when `sol.success` is false; their failure checks are commented out.
Downstream callers must therefore validate scaled equation residuals rather than treating a returned state as convergence.

The PNM comments and verbose messages say that temperatures below $1$ MeV use the zero-temperature treatment, but `_nB` and `_ns` actually switch only below $10^{-3}$ MeV.
Consequently the contour value $T=0.01$ MeV uses adaptive finite-temperature momentum quadrature on a nearly discontinuous Fermi surface.
The cluster has emitted SciPy `IntegrationWarning` messages from the baryon-density integral in this regime; the integral is finite, so the warning indicates numerical difficulty during RMF root iterates rather than a physical divergence.

The analytical isothermal path no longer multiplies this problem through a broad chemical-potential scan; it uses the direct fixed-density validator in [[pnm-density-state-recovery]].
The underlying `Solver.py` behavior remains open and must be fixed separately, with an explicit decision between a controlled low-temperature approximation and a stabilized exact finite-$T$ quadrature.

## Numerical isothermal auto-ceiling scan — fixed; BVP edge failures remain

Before 2026-08-26, every physical-$n_K,J_K$ `solve_front_isothermal` trial separately solved fixed-density PNM pressure and energy, then called `muB_from_nB_physical(..., scan_points=48, auto_expand=True)` to recover $\mu_B(0^-)$ for the automatic composition ceiling.
Every chemical-potential sample invoked as many as five RMF seeds, even though the upstream state was independent of the trial current.
The numerical solver now uses one `_validated_pnm_state_from_nB` result and reuses its $\mu_B(0^-)$, $P(0^-)$, and $e(0^-)$, matching the analytical path.

A 2026-08-26 first-trial smoke test used the four corner cells of the 30-by-20 production domain.
The high-angle inner cell at $T(0^-)=66.055$ MeV and $n_B(0^-)=n_0$ converged on its analytical current seed in 71.9 s; the corresponding analytical cell, which uses direct fixed-density recovery, had taken 2.49 s.
The low-temperature outer cell failed after 154.9 s during nonlinear state recovery, while the low-temperature inner and high-angle outer cells exceeded the production 180 s trial limit.
This established that the original initial zero-progress interval and warning flood occurred inside the first trial, before fallback-seed count could be the primary cause.
After the direct-state fix, the clean high-angle inner corner fell from 71.9 s to 4.18 s with the same velocity and no warnings.
The low-temperature outer failure fell from 154.9 s to 54.9 s, but the low-temperature inner cell still failed after 111.5 s and the high-angle outer cell still exceeded 180 s.
Those remaining failures occur after upstream recovery and must be diagnosed in the dynamic quark-state/BVP path.

The cluster candidate list can still amplify the cost after that first failure: a first-shell cell has the analytical current, four multiplicative variants, and one density-scaled fallback, under a 900 s cell budget.
There is also an independent low-temperature outer-corner bound problem: the analytical seed is $j_B=6.38\times10^4$, but the global `1e-4*nB_0minus` upper bound clips it to $6.48\times10^2$ before the BVP starts.
This does not by itself prove the exact BVP eigenvalue, but it proves that the current cluster wrapper discards its intended best seed near $a(0^+)\to1$.

The upstream repair is complete.
The next work is to diagnose the remaining dynamic-state failures and retune the current bounds from measured BVP behavior before reducing fallback count.

## Isothermal strict-retry issue — superseded

The old normalized-composition `solve_front_isothermal` had a singular-Jacobian failure in its coarse-recovery retry and two historical tests expected `jB=0.20069404233893348`.
The 2026-08-18 rewrite replaced that system with a direct BVP in physical $n_K$ and $J_K$, the exact nonleptonic rate, and local $D_K$.
The previously failing strict case now converges directly, including profile reconstruction, at approximately `jB=0.1978159459` with a vanishing tail residual.

The old reference value and retry-specific assertions describe a different set of differential variables and approximations, so they are not valid regressions for the new model.
Current tests instead verify the physical constitutive and reaction residuals, endpoint momentum closure, local-fraction identity, and finite-flux boundary conditions.

## ⚠ Finite strange-quark mass is expensive in the isothermal BVP

Finite `ms` is supported, including nonzero equilibrated $n_K(\infty)$ and $J_K(\infty)$.
It is much slower than the massless branch because the existing strange-quark EOS performs numerical quadrature inside every local nonlinear closure, and `solve_bvp` repeats those closures for collocation and Jacobian perturbations.

A deliberately coarse live check with `ms=100`, 12 initial nodes, `tail_eps=1e-2`, and `tol_bvp=5e-2` took about two minutes and required roughly one thousand local EOS roots on the current workstation.
This is a performance limitation, not a frozen-$D_K$ shortcut license: $D_K$ must still be recomputed from the local $\mu_B$.
Use the massless default for broad scans unless finite-$m_s$ physics is specifically required, and budget finite-$m_s$ validation points individually.

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
