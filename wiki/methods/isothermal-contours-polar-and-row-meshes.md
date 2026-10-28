---
summary: Nonrelativistic isothermal contours were recomputed on 2026-10-27/28 on a polar mesh (8-100 MeV) and a log-T row mesh (0.01-10 MeV); numerical coverage is 99.6% and 94%, all verifications pass, and analytic xi=-0.85 overestimates numerical speeds by 1.2-6.7x at high T but only 1.0-1.6x at low T.
status: current
updated: 2026-10-28
tags: [method, isothermal, numerical, analytic, contour, xi]
---

# Isothermal contours on polar and row meshes

Both panels of the isothermal contour figure were recomputed with `RMFsolver` `[v4.9.6]`, replacing the target-search plan in [[nonrelativistic-isothermal-contours]].
The numerical speed is `solve_front_isothermal_slow(initial_guess_mode="auto")`; the analytical speed is `analytic_velocity_isothermal_slow(xi=-0.85)`.
The physics is massless quarks, QMC-RMF3 PNM, $B^{1/4}=189.1565957288247$ MeV, $\alpha_s=0.4$ and the revised Madsen rate, with $v=c\,u(0^-)$.
Code, commands and payloads are in `new_paper_calculations/26-10-27/isothermal/` (`highT/README.md`, `lowT/README.md`).

## Meshes

No point is evaluated inside a gray region.
The inner gray boundary is PNM–uds coexistence, $\mu_B(\infty)\big|_{\mu_K=0}=\mu_B(0^-)$, below which quark matter is not favored and the speed is zero.
The outer one is PNM–udd balance, $\mu_B(0^+)+a\,\mu_K(0^+)=\mu_B(0^-)$ at $a=1-10^{-9}$, beyond which the speed diverges.

| Panel | Mesh | Points per method |
|---|---|---|
| High $T$, linear axis | Polar $X=(n_B/n_0-1)/4.5=\rho\cos\theta$, $Y=T/120\,\mathrm{MeV}=\rho\sin\theta$ (as in [[isothermal-contour-cluster]]); 81 rays from the angle whose udd point sits at 8 MeV up to $90^\circ$; 29 radial fractions 0.005–0.99999 between the two boundaries | 2291 (58 below 8 MeV are `outside_panel`) |
| Low $T$, log axis | 40 rows `np.geomspace(0.01, 10, 40)`; 35 fractions $10^{-5}$–0.9999 per row, $x=x_{\rm in}+f\,(x_{\rm out}-x_{\rm in})$ | 1400 |

⚠ Contour kinks came from radial, not angular, resolution.
The plots add display-only PCHIP refinement of $\log_{10}v$ along each ray (in $-\ln(1-f)$) or row (in ${\rm logit}\,f$); payloads are never interpolated, and failed points stay holes.

## Coverage

| Panel | Analytical | Numerical |
|---|---|---|
| High $T$ | 2291/2291 | 2282/2291; 9 unresolved, all at $f\ge0.9999$ where the analytic speed exceeds $1.1\times10^5$ m/s, above the top 5000 m/s level |
| Low $T$ | 1400/1400 | 1310/1400 after one repair pass (17 recovered); 54 unresolved and 36 timeouts |

All but 2 low-$T$ failures sit at $f\ge0.99$, where the analytic speed is at least $2.3\times10^5$ m/s.
The exceptions are $f=0.9$ at $T=0.024$ and 0.029 MeV, $5.07\,n_0$ (analytic $5$–$6\times10^5$ m/s): the shooting fallback reaches flux change $\sim10^{-7}$ but fails its monotone-path check.
Holes can therefore interrupt the $5\times10^5$ and $10^6$ m/s numerical contours on up to 18 rows.
At high $T$ all accepted points are collocation; at low $T$, 156 used the validated shooting fallback.

The repair pass tries, in order, `initial_guess_mode="shooting"` with `path_rtol=2e-7`, then `tol_bvp=2e-5`, then `n_mesh=1200`, with unchanged acceptance; each recovered record names its variant.
Before that, every evaluation walks a `path_rtol` ladder ×0.5/0.2/0.1 on table non-convergence, or takes one 600-node retry otherwise.

## Verification

| Check | High $T$ | Low $T$ |
|---|---|---|
| Drawn contour vs direct root | $\Delta x\le2.2\times10^{-4}\,n_0$, $\Delta T\le0.014$ MeV (12 crossings) | $\Delta x\le3.8\times10^{-4}\,n_0$ (8 crossings, up to $10^6$ m/s) |
| Tighter tolerances at accepted points | relative speed change $\le2.0\times10^{-5}$ | $\le2.4\times10^{-6}$ |
| Seam at 10 MeV, row vs polar contour | — | $\Delta x\le1.9\times10^{-3}\,n_0$ for 50–5000 m/s |
| Static boundaries vs the saved 26-09-16 domain | all 24 rays, $|\Delta\rho|\le2.5\times10^{-5}$ (criterion $10^{-3}$) | same boundary code |

Evidence: `highT/verify_highT.json`, `lowT/verify_lowT.json` and `highT/verify_saved_domain.json`.

The final code review (2026-10-28) led to four workflow fixes; none changes a shipped point.
- **Non-finite metrics:** acceptance now rejects any non-finite validation metric. Python's `max()` and `>` had let NaN through, but no shipped accepted point has a non-finite stored metric.
- **Run identity:** it now records every `RMFsolver/*.py` hash and the NumPy/SciPy versions. Repair and seeding refuse a payload made with another solver, library or physics.
- **Boundary files:** they are refused under a stale key. The rebuilt files are bit-identical, and the mesh fingerprints match the shipped payloads.
- **Worker crashes:** a worker that dies leaves its point unresolved instead of aborting the run.

⚠ The shipped payloads predate the wider identity. A rerun therefore needs `--restart`, and `--seed-from` can reuse their points.

## Analytic vs numerical

| Region | Analytic/numerical speed |
|---|---|
| High $T$, near coexistence | 1.23–1.52 |
| High $T$, mid ring | 1.55–2.75 |
| High $T$, near udd | up to 6.7 |
| Low $T$, all | 1.02–1.60, median 1.22 |

The profile $\xi$ implied by each numerical solution, $\xi=2I_1/(\lambda u(1-a)a)-1$ with $a=a(0^+)$, ranges from $-0.48$ to $+1.59$ at high $T$ and from $-0.51$ to $0$ at low $T$.
So a single $\xi$ cannot fit both panels, and the high-$T$ mismatch is not only a $\xi$ effect.
Relative to the pre-2026-10 contours, the new ones sit at higher density, consistent with $\alpha_s$ 0.3→0.4.

## Retuning $\xi$ without new EOS calls

Every analytical record stores $I_2$, $\lambda_n$, $a(0^+)$, $D_K$, $\gamma_K$ and the endpoints.
Then, exactly,

$$
v(\xi)=c\sqrt{\frac{2I_2}{\lambda_n^2\,(1-a)(1+\xi a)}} .
$$

`_iso_xi.build_cache` / `retune(cache, xi)` rescales a whole grid in milliseconds, and `fit_xi` (objective `log_speed` or `contour`) compares against the numerical grid.
Numerical records keep $I_1$, $I_2$ (`I1_num`, `I2_num`) and `xi_profile`, `xi_profile_inf`, plus a 256-node profile per point.

## Figures

`local_analysis_results/26-10-27/`, built by `build_isothermal_notebook.py` into `26-10-27-isothermal.ipynb`:
- `isothermal-numerical-linear` (combined linear panel, reference style of `All_Plots_new` cell 25);
- `isothermal-comparison-linear` (analytic vs numerical);
- `isothermal-lowT-logT` (reference style of `26-10-02` cell 8, ylim extended to 10 MeV; per the user, half-decade levels 100–$10^6$ m/s labelled on the numerical contours instead of a speed legend, the gray regions named as in the linear figure, and $x$ widened to $5.8\,n_0$ for the $udd$ strip label);
- `isothermal-highT-old-vs-new`.
All text is at least 16 pt.

## Open

- **Analytic recipe vs manuscript (user decision pending).** The code evaluates the transport at $\mu_B(0^+)$, the density at $n_B(\infty)$ and the exact $A_{\rm iso}$.
The most obvious reader recipe takes everything at $\infty$ with the Taylor $A_{\rm iso}$, which differs by $+0.6\%$ to $+73\%$ and gives $A>1$ at some states.
Option A: keep the code and state its choices in Appendix C. Option B: switch the code to the reader's recipe.
- The two interior low-$T$ holes need a solver-side change to the shooting fallback's monotone check, in `phase_velocity.py` only.
