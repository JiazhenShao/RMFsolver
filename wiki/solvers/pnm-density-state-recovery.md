---
summary: The isothermal analytical solver now recovers the upstream PNM state directly at fixed density, using scaled-residual rejection and two-seed branch agreement instead of a broad chemical-potential scan.
status: current
updated: 2026-08-21
tags: [solver, pnm, rmf, performance, isothermal]
---

# Direct upstream PNM state recovery

`analytic_velocity_isothermal` receives $n_B(0^-)$, so its upstream state is now obtained directly with the density-input `RMFsolvePNM` system.
It no longer calls `muB_from_nB_physical` or scans a broad interval in $\mu_B$.

The old scan was a defensive response to isolated nonphysical roots of the chemical-potential-input RMF system.
It was safe but extremely expensive: each scan coordinate invoked a multi-seed validator, and automatic expansion at hot low-density points could probe unrelated states down to $\mu_B=0$.
The scan also amplified the cold finite-temperature quadrature problem in `Solver.py`, where $T=0.01$ MeV still follows the adaptive finite-$T$ integration branch.

## Acceptance contract

The direct fixed-density validator:

1. tries the established PNM mean-field seeds in order;
2. rejects non-finite states and equation residuals;
3. scales the four RMF equation residuals by the requested density and rejects an infinity norm above $10^{-6}$;
4. requires positive $\sigma(0^-)$ and $\mu_B(0^-)$;
5. requires two accepted seeds to agree on both $\sigma(0^-)$ and $\mu_B(0^-)$ to relative tolerance $10^{-6}$;
6. stops as soon as that agreement is found;
7. evaluates the forward density, pressure, and energy density from the accepted RMF solution table without another mean-field solve;
8. requires the reconstructed density to match the requested $n_B(0^-)$ within $2\times10^{-6}$ relatively.

The public analytical result records `sigma_0minus`, `upstream_rmf_scaled_residual`, `upstream_density_relative_residual`, and `upstream_seed_count`.

## Measured effect

For $B^{1/4}=189.1566$ MeV and $\xi=-0.5$:

- $T(0^-)=0.01$ MeV, $n_B(0^-)/n_0=3.046$: runtime fell from $102.8$ s to $0.78$ s; four seeds were needed and the forward-density error was $1.7\times10^{-10}$.
- $T(0^-)=66.055$ MeV, $n_B(0^-)/n_0=1$: runtime fell from $65.9$ s to $0.14$ s; the first two seeds agreed and the forward-density error was $-3.7\times10^{-11}$.

The corresponding velocities agree with the former scan-based values to better than $4\times10^{-8}$ relatively.
The analytical contour holes previously labeled `timeout` were therefore execution artifacts, not failures of the isothermal velocity formula.

⚠ This change does not repair the low-temperature quadrature or disabled root-success checks inside `Solver.py`; it removes the broad scan that repeatedly exposed them from the analytical isothermal path.
The older `muB_from_nB_physical` helper remains available for callers that explicitly need chemical-potential-axis inversion.

See [[isothermal-analytic-front-speed]] for the physical formula and [[known-issues]] for remaining solver-level traps.
