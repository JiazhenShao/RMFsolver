---
summary: Full numerical fronts at 0.01 MeV were recovered at density ratios 3.001, 3.01 and 3.05 using a layer-resolved initial guess; higher-density robustness remains unresolved.
status: current
updated: 2026-09-30
tags: [method, isothermal, numerical, low-temperature, diagnosis]
---

# Numerical isothermal feasibility at 0.01 MeV

A logarithmic-temperature contour reaching $T(0^-)=0.01$ MeV is plausible near the stable-matter boundary, but the default BVP initialization is not reliable there and a full-domain run is not yet established.
This conclusion comes from bounded local tests, not extrapolation of the analytical formula or a new cluster sweep.

## Scope and measurements

Saved diagnostics and reproducible scripts are in `local_analysis_results/isothermal_low_temperature_probe/`.
Defaults are $B^{1/4}=189.1565957288247$ MeV, PNM, $m_s=0$, and the automatic maximum $a(0^+)$.
The full `solve_front_isothermal` equations retain exact local reaction, local diffusion, normalized density/current fields, and baryon/momentum closures.
Analytical speeds provide initial currents only; bounds span one quarter to four times the guess.

The production solver hash stayed `d3d88937d272d62683eb711333d80c405416f1a4dc0fee5c354bd3fc33c49473` throughout; no production solver or cluster files were modified.
Fourteen physical trials were made serially on one worker, with 40- or 60-second hard limits.

The default initialization at density ratios 3.001, 3.01, 3.1, 3.5 and 4.5 timed out at 40 seconds.
At 5.2 it failed with a local-EOS recovery exception after 38 seconds.
Upstream recovery and analytical seeding took about one second each; the long delay was inside the BVP, not a broad chemical-potential scan.

An initialization-only experiment recovered:

| $n_B(0^-)/n_0$ | Velocity (m/s) | Time (s) | Nodes | Maximum BVP RMS residual |
| --- | ---: | ---: | ---: | ---: |
| 3.001 | 74.5351043 | 17.5 | 456 | $4.40\times10^{-6}$ |
| 3.01 | 771.9789334 | 22.1 | 488 | $3.28\times10^{-5}$ |
| 3.05 | 4139.9647979 | 53.8 | 717 | $8.01\times10^{-5}$ |

These use tolerance `1e-4`, `tail_eps=1e-3`, and 300 initial nodes.
All have positive, monotone $n_K$ and positive $J_K$; endpoint flux errors are below $1.8\times10^{-15}$ and EOS closure checks at 17 profile nodes stay below $4.3\times10^{-10}$.
The cutoff values satisfy $\mu_K/T<0.011$, consistent with being in the linear downstream tail.

A same-state continuation check at 3.001 used 450 base nodes, tolerance `1e-5`, and `tail_eps=1e-4`.
It converged in 47 seconds; the speed changed by only $1.08\times10^{-10}$ relatively.
The analogous tighter check at 3.05 hit its 60-second limit, so mesh/tail independence of the 4140 m/s result remains unverified.
Improved-initialization trials at 3.1, 3.5 and 5.2 also timed out; 4.5 was not retried.

## Evidence for the initialization problem

At 3.01, the seed's reaction-layer scale occupies about $1.97\times10^{-4}$ of the compact coordinate, while the default first mesh interval is $3.34\times10^{-3}$.
At 3.5, the estimated layer occupies only $3.54\times10^{-7}$.
At 5.2, the default Newton iterates reached normalized density extrema near $-2.7\times10^7$ and $2.4\times10^7$, causing thousands of failed local closures.
Thus increasing the trial budget alone does not address the observed runaway iterates.

The diagnostic initialization changes only the arrays passed into SciPy: a logarithmic physical-coordinate mesh resolves the reaction layer, and an approximate algebraic-times-exponential density profile and flux estimate approach the downstream tail.
Thermodynamic interpolation is confined to the initial guess; every accepted solution is recomputed using the full EOS and exact transport ODE.
Both mesh and profile change together, so this establishes initialization sensitivity without isolating their individual contributions.
The prototype is not installed in the public API and is not validated near the upper composition boundary.

## Implication for a future plot

At 0.01 MeV, speeds from roughly 75 to 4140 m/s already lie between density ratios 3.001 and 3.05.
For the established 50–5000 m/s levels, useful cold sampling therefore needs fine radial resolution close to coexistence, in addition to logarithmic temperature spacing.
Use converged physical profiles for continuation between nearby temperatures and densities; validate mesh/tail stability before connecting newly recovered contour points.
This is a proposed next calculation, not a completed log-temperature contour.

⚠ A timeout means unresolved numerics, not absence of a physical front; no failed trial was converted into a velocity or connected across a gap.
Exact zero temperature remains outside the finite-temperature transport evaluation used here; nothing in these tests validates a physical infinite speed.

## Environment note

The project SciPy installation failed to import its PROPACK binary with a Mach-O `__thread_bss` error.
Tests used the existing separate SciPy 1.18.1 installation in `/private/tmp/nsm-scipy-verification` through `PYTHONPATH`; no environment packages were changed.
Setup-only failures are retained separately and excluded from the 14 physical trials.

Related: [[isothermal-targeted-shots]], [[isothermal-contour-repair]], [[known-issues]], [[isothermal-maximum-speed]].
