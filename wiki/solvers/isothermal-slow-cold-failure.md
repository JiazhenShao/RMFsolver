---
summary: Cold solve_front_isothermal_slow fails because its tail-scaled compact coordinate squeezes the conversion layer inside the first mesh interval, so the tail-shaped default guess diverges and exhausts max_nodes in ~2 s; the 55-99% tail-derivative error is real but changes no speed.
status: current
updated: 2026-10-27
tags: [solver, isothermal, numerical, low-temperature, diagnosis, slow-front]
---

# Cold failure of the slow isothermal BVP

At low temperature `solve_front_isothermal_slow` fails because of how it represents the profile, not because of its equations or microphysics.
Its compact coordinate is scaled by the downstream tail length $1/\lambda_\infty$, while nearly all K conversion happens in a layer $10^3$–$10^7$ times thinner.
The layer then sits inside the first interval of the default mesh, the default guess has the shape of the tail, and `solve_bvp` diverges.
Diagnosed 2026-10-27 for massless PNM, $B^{1/4}=189.1566$ MeV, automatic $A_{\rm iso}$, solver defaults; no solver source was changed.
Scripts and `failure_map.json`: `new_paper_calculations/26-10-27/isothermal_cold_diagnosis/`.

## Two length scales behind the interface

- **Conversion layer:** $\mu_K$ falls from $\mu_K(0^+)=43$–$357$ MeV to about $2\pi T$, where the cubic rate $\propto\mu_K^3$ dominates. This layer sets the speed.
- **Linear tail:** below $\mu_K\approx2\pi T$ the rate is $\propto4\pi^2T^2\mu_K$, so the remaining imbalance relaxes slowly. Advection stretches the tail further at high speed. It carries only $r/r(0^+)\lesssim2\pi T/\mu_K(0^+)\sim10^{-3}$.

| $T$ [MeV] | $n_B(0^-)/n_0$ | layer half-width | tail $1/\lambda_\infty$ | ratio | default solve |
|---:|---:|---:|---:|---:|---|
| 10 | 3.5 | 151 nm | 787 nm | 5 | converges |
| 1 | 3.5 | 1.6 μm | 0.37 mm | 238 | converges |
| 0.1 | 3.5 | 11 μm | 0.26 m | $2.3\times10^4$ | fails |
| 0.01 | 3.05 | 0.26 mm | 12 m | $4.5\times10^4$ | fails |
| 0.01 | 4.8 | 36 μm | 1.5 km | $4.1\times10^7$ | fails |

The half-widths come from the advection-free first integral on the solver's own EOS path, $r_x=-\sqrt{2\,r_{\rm scale}\int_0^z\Gamma_K\,dz'}$; $\lambda_\infty$ uses the solver's tail formula.

## How the solver sees it

The coordinate is $x=-(\kappa/\lambda_\infty)\ln(1-s)$, with mesh `_thermal_compact_mesh(300, 1e-8)`, first interval $s_1=3.39\times10^{-4}$.
The default guess $r\propto(1-s)^\kappa$ is a pure tail exponential.
For 0.01 MeV it puts the half-point near $x\approx0.7/\lambda_\infty$ (metres) instead of tenths of a millimetre.

The layer's half-point position $s_{1/2}=1-e^{-\lambda_\infty x_{1/2}}$ predicts the outcome.
Over the 35-state map ($T=10,3,1,0.3,0.1,0.03,0.01$ MeV × $n_B/n_0=3.05,3.2,3.5,4.0,4.8$), all 19 successes have $s_{1/2}/s_1\ge0.59$.
Fifteen of the 16 failures have $s_{1/2}/s_1\le0.36$.
⚠ The exception is $4.8n_0$ at 3 MeV ($a(0^+)=0.91$, $s_{1/2}/s_1=3.2$). It is unexplained; a poor flux guess is plausible (the analytic speed is 60% high at that density and 10 MeV) but unverified.
The densest states fail first: $4.8n_0$ already fails at 3 MeV, while $3.05n_0$ holds down to 0.03 MeV.

## What `solve_bvp` does

At 0.01 MeV and $3.05n_0$ the node count roughly triples each iteration: 300, 898, 2692, 8074, 24220.
The maximum relative residual grows from $9\times10^3$ to $4\times10^8$.
The downstream boundary residual jumps to $10^{22}$ at iteration 3 ($10^{75}$ at $3.5n_0$), so the Newton iterates diverge rather than merely lack resolution.
The next step would exceed `max_nodes=50000`, so every failure returns "maximum number of mesh nodes is exceeded" in about 2 s, not a timeout.

## Evidence it is the representation, not the equations

- At $3.05n_0$, 0.01 MeV, the same equations converge once the layer is resolved: the layer-shaped guess with `n_mesh=3000` gives 3249.94 m/s in 8.9 s and 31,521 nodes. The default guess with `n_mesh=3000` gives the same speed in 49 s and 201,775 nodes.
- At $3.5n_0$ and $4.8n_0$ the layer half-point is at $s=6.6\times10^{-7}$ and $2.4\times10^{-8}$. ⚠ `continuation_guess` is interpolated onto the fixed `_thermal_compact_mesh(n_mesh)`, so even a perfect guess loses the layer. The public API cannot rescue these states without a source change.
- The table's equilibrium rate slope `drate_dr` is too large by $+0.5\%$, $+55\%$, $+99\%$ at $3.05$, $3.5$, $4.8n_0$, because its $\mu_K$ nodes never reach $\mu_K<2\pi T$. `da_dr` is exact to $10^{-9}$. Substituting the exact slope rescues none of the failures, and in the converged $3.05n_0$ solve the exact or doubled slope leaves 3249.94 m/s unchanged to six digits. The rate spline stays positive and monotone; its value is up to 55% high only below $\mu_K\approx0.1$ MeV.

## Implication

A fix needs a change in `_solve_front_slow`: an initial mesh built in physical $x$ and clustered at the layer scale, plus the advection-free first-integral guess above.
An exact one-line equilibrium slope, $\mathcal C_K(\mu_B/3)^5\,4\pi^2T^2/(D_K\,\partial n_K/\partial\mu_K)$, is cosmetic for the speed.
Even resolved cold solves need $\sim3\times10^4$–$2\times10^5$ nodes, so 50,000 is too small as a cold `max_nodes`.
The old physical-$n_K$ solver failed the same way and was partly rescued by a layer-resolved logarithmic mesh; see [[isothermal-cold-numerical-probes]].
See [[slow-front-solvers]] for the coordinate and boundary conditions and [[known-issues]] for other solver traps.
