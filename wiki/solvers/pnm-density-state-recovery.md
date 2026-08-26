---
summary: Both isothermal solvers recover the upstream PNM state directly at fixed density, using scaled-residual rejection and branch agreement instead of a broad chemical-potential scan.
status: current
updated: 2026-08-26
tags: [solver, pnm, rmf, performance, isothermal]
---

# Direct upstream PNM state recovery

Both `analytic_velocity_isothermal` and `solve_front_isothermal` receive $n_B(0^-)$, so their upstream PNM state is obtained directly with the density-input `RMFsolvePNM` system.
Neither path now calls `muB_from_nB_physical` or scans a broad interval in $\mu_B$.
The numerical helper also reuses the accepted RMF solution for $\mu_B(0^-)$, $P(0^-)$, and $e(0^-)$ instead of running separate pressure and energy solves.

The old scan was a defensive response to isolated nonphysical roots of the chemical-potential-input RMF system.
It was safe but extremely expensive: each scan coordinate invoked a multi-seed validator, and automatic expansion at hot low-density points could probe unrelated states down to $\mu_B=0$.
The scan also amplified the cold finite-temperature quadrature problem in `Solver.py`, where $T=0.01$ MeV still follows the adaptive finite-$T$ integration branch.

## Acceptance contract

The direct fixed-density validator:

1. tries the established PNM mean-field seeds in order;
2. rejects non-finite states and equation residuals;
3. scales the four RMF equation residuals by the requested density and rejects an infinity norm above $10^{-6}$;
4. requires positive $\sigma(0^-)$ and $\mu_B(0^-)$;
5. requires two candidate seeds to agree on both $\sigma(0^-)$ and $\mu_B(0^-)$ to relative tolerance $10^{-6}$;
6. evaluates the agreeing branch's forward density, pressure, and energy density from its RMF solution table without another mean-field solve;
7. rejects that agreement and continues through the remaining seeds unless the reconstructed density matches within $2\times10^{-6}$ relatively and both pressure and enthalpy are positive;
8. stops at the first agreeing branch that passes those thermodynamic checks.

The direct state carries `sigma_0minus`, `upstream_rmf_scaled_residual`, `upstream_density_relative_residual`, and `upstream_seed_count`.

## Measured effect

For $B^{1/4}=189.1566$ MeV and $\xi=-0.5$:

- $T(0^-)=0.01$ MeV, $n_B(0^-)/n_0=3.046$: runtime fell from $102.8$ s to $0.78$ s; four seeds were needed and the forward-density error was $1.7\times10^{-10}$.
- $T(0^-)=66.055$ MeV, $n_B(0^-)/n_0=1$: runtime fell from $65.9$ s to $0.14$ s; the first two seeds agreed and the forward-density error was $-3.7\times10^{-11}$.

The corresponding velocities agree with the former scan-based values to better than $4\times10^{-8}$ relatively.
The analytical contour holes previously labeled `timeout` were therefore execution artifacts, not failures of the isothermal velocity formula.

The 2026-08-26 numerical four-corner smoke test measured the high-angle inner cell before and after adopting the same direct state.
Its first BVP trial fell from 71.9 s to 4.18 s and returned the same $23.7242\ \mathrm{m\,s^{-1}}$ velocity with zero integration warnings.
The low-temperature outer failure shortened from 154.9 s to 54.9 s, while the remaining low-temperature and near-$a(0^+)=1$ failures show that dynamic quark-state recovery and the current bound still need separate work.

An additional seed-ordering regression occurred at $T(0^-)=62.20097$ MeV and $n_B(0^-)=1.38240n_0$.
Seeds two and three agreed on a collapsed $\sigma(0^-)=178.26$ MeV branch with negative pressure before seed four could corroborate the physical $\sigma(0^-)=36.42$ MeV branch found by seed one.
Thermodynamic validation is now part of branch acceptance rather than a fatal check after early termination; the state returns $\mu_B(0^-)=962.0164$ MeV and positive $P(0^-)=1.4226\times10^8\ \mathrm{MeV}^4$ after four seeds.

⚠ This change does not repair the low-temperature quadrature or disabled root-success checks inside `Solver.py`; it removes the broad scan that repeatedly exposed them from both isothermal entry points.
The older `muB_from_nB_physical` helper remains available for callers that explicitly need chemical-potential-axis inversion.

See [[isothermal-analytic-front-speed]] for the physical formula and [[known-issues]] for remaining solver-level traps.
