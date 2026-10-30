---
summary: Nonrelativistic isothermal contours were recomputed on 2026-10-27/28 on a polar mesh (8-100 MeV) and a log-T row mesh (0.01-10 MeV); numerical coverage is 99.6% and 94% with every plotted contour connected, all verifications pass, and analytic xi=-0.85 overestimates numerical speeds by 1.2-6.7x at high T but only 1.0-1.6x at low T.
status: current
updated: 2026-10-30
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
| Low $T$ | 1400/1400 | 1314/1400 after repairs (17 in the first pass, 4 targeted on 2026-10-28); 50 unresolved and 36 timeouts |

All remaining low-$T$ failures sit at $f\ge0.99$, where the analytic speed is at least $2.3\times10^5$ m/s, and every plotted level (100 to $10^6$ m/s) now has a crossing on every row it reaches.
At high $T$ all accepted points are collocation; at low $T$, 156 used the validated shooting fallback.

⚠ Next to the udd boundary the backward-shot flux is reproducible only to about $10^{-6}$ with the default tables (`path_rtol=1e-6`): integration tolerances $10^{-10}$ to $10^{-13}$ scatter it by $2\times10^{-7}$ to $1.4\times10^{-6}$, also at accepted points, with no trend.
That is the size of the fallback's own self-check ($10^{-6}$), and the interface boundary residual is the shot mismatch times $jB\,[a(0^-)-a(\infty)]/J_{\rm scale}\approx28$ against an absolute $10^{-7}$, so points fail by a hair.
Tighter tables (`path_rtol=1e-7`) lower the floor: the 1M-gap points (8,28) and (10,29) and the interior points (5,26) and (6,26) then pass unchanged acceptance, with speeds identical to 7 digits for `path_rtol` $10^{-7}$ to $10^{-8}$.
Evidence: `local_analysis_results/26-10-27/isothermal-1M-gap-repair/repair.json`.

The repair pass tries, in order, `initial_guess_mode="shooting"` with `path_rtol=2e-7`, then `tol_bvp=2e-5`, then `n_mesh=1200`, then `path_rtol=1e-7`, with unchanged acceptance; each recovered record names its variant, and `run_lowT.py --repair --points I,J ...` targets single points.
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

The manuscript figures come from the `All_Plots_new.ipynb` cells tagged `isothermal-contours-26-10-27` (rebuilt by `update_all_plots_new.py`); titles say "speed", not $u(0^-)$.
They write `Latex_writting/Hydrodynamical combustion/Plots/isothermal-numerical-contour.pdf` (linear numerical, Fig. `speed-isothermal-linear`), `isothermal-numerical-contour-logT.pdf` (log-$T$ numerical only, not yet included by `Main.tex`), `isothermal-analytic-numerical-logT.pdf` (log-$T$ analytic versus numerical at `ISO_CMP_XI`, default $-0.85$; Fig. `speed-isothermal-log`) and `isothermal-analytical-numerical-comparison.pdf` (the same at the fitted `ISO_FIT_XI` $=-0.57$, not yet included by `Main.tex`).
Rerunning a cell and recompiling updates the manuscript.
Fonts are sized for the manuscript, which includes every figure at `\linewidth` $=246$ pt (revtex4-1, 10 pt two-column): a font prints at size $\times\,246/(72\times$ width in inches$)$, so at the 7.2 in width the 22/24 pt settings (`MANUSCRIPT_FONTS` in `update_all_plots_new.py`) print at 10.4/11.4 pt, above the 10 pt body text; only super- and subscripts print smaller (8 pt).
Per the user (2026-10-29) the contour speed labels are the exception: they follow the adiabatic contour cell, placed by matplotlib `clabel` at the label temperatures with a white square box; hand-placed `axis.text` labels are a last resort.
Superseded 2026-10-30: the label size went from 11 to 13 pt at 7.2 in (6.2 pt printed) in every contour figure, the axes now read "Nuclear baryon number density [$n_0$]" and "Nuclear phase temperature [MeV]", and the linear panel's inner region name moved to $(1.08, 3)$ to stay on gray; see [[manuscript-figure-standard]].
Superseded 2026-10-30 (later the same day): `Main.tex` includes `isothermal-numerical-contour-logT.pdf` as Fig. `speed-isothermal-log` and `isothermal-analytic-numerical-logT.pdf` as Fig. `isothermal-contour-compare`. Per the user, the latter now draws the analytical contours at `ISO_CMP_XI` $=-0.57$, the 0.01–10 MeV contour fit of [[isothermal-xi-fit]], in the numerical log panel's shape: the same $5.91\times4.85$ in axes box, $x$ from 2.2 to $6\,n_0$, region names vertical on gray, and the method legend in one row between title and axes on a $7.2\times6.85$ in canvas. The fitted-$\xi$ cell (`ISO_FIT_`) now draws the same figure.
⚠ The `speed-isothermal-log` caption still mentions numerical dots; the figures have none.

## Open

- **Analytic recipe vs manuscript (user decision pending).** The code evaluates the transport at $\mu_B(0^+)$, the density at $n_B(\infty)$ and the exact $A_{\rm iso}$.
The most obvious reader recipe takes everything at $\infty$ with the Taylor $A_{\rm iso}$, which differs by $+0.6\%$ to $+73\%$ and gives $A>1$ at some states.
Option A: keep the code and state its choices in Appendix C. Option B: switch the code to the reader's recipe.
- The two interior low-$T$ holes need a solver-side change to the shooting fallback's monotone check, in `phase_velocity.py` only.
