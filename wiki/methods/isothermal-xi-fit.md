---
summary: Best single isothermal xi is -0.57 below 10 MeV (contours nearly coincide, RMS offset 0.03 n0) and +0.74 at 8-100 MeV (RMS 0.08 n0, still visibly off at 100-1000 m/s); the xi each contour level needs falls with speed and is the same in both panels above ~1000 m/s.
status: current
updated: 2026-10-30
tags: [method, isothermal, analytic, xi, contour]
---

# Isothermal $\xi$ fit to the numerical contours

The analytical isothermal speed was fitted to the numerical one using only the saved intermediates of [[isothermal-contours-polar-and-row-meshes]]; no solver was called and the fit takes 3 s.
At every point $u(0^-;\xi)=\sqrt{2I_2}/\big(\lambda_n\sqrt{[1-a(0^+)][1+\xi a(0^+)]}\big)$ with $I_2$, $\lambda_n$, $a(0^+)$ independent of $\xi$.

## Result

| Fit | High $T$ (8–100 MeV, 50–5000 m/s) | Low $T$ (0.01–10 MeV, 100–$10^6$ m/s) |
|---|---|---|
| Contour displacement (plotted levels) | $\xi=+0.74$ | $\xi=-0.57$ |
| Log speed (all points) | $\xi=+0.94$ | $\xi=-0.70$ |
| RMS / max contour offset at the contour fit | $0.078$ / $0.34\,n_0$ | $0.031$ / $0.13\,n_0$ |
| RMS / max contour offset at $\xi=-0.85$ | $0.29$ / $0.64\,n_0$ | $0.092$ / $0.27\,n_0$ |

The contour fit minimizes the RMS displacement of the plotted contours along rays or rows in $(n_B/n_0-1)/4.5$, as in [[reusable-adiabatic-xi-cache]]; it is the visual criterion and the recommended one.
These values include the four low-$T$ points recovered on 2026-10-28 (the low-$T$ fit moved from $-0.564$ to $-0.566$).
One $\xi$ for both panels ($+0.20$) fits neither: RMS $0.10\,n_0$ at high $T$ and $0.12\,n_0$ at low $T$.

Since 2026-10-30 (user) the manuscript comparison, Fig. `isothermal-contour-compare`, draws the analytical contours at the low-$T$ contour fit $-0.57$ instead of the adiabatic appendix value $-0.85$.

## The required $\xi$ depends on the speed level, not the panel

Fitting each contour level alone gives

| Level [m/s] | 50 | 100 | 200 | 300 | 500 | 1000 | 2000 | 3000 | 5000 | $10^4$ | $10^5$ | $10^6$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| High $T$ | +2.01 | +1.33 | +0.67 | — | +0.02 | −0.27 | −0.45 | — | −0.60 | — | — | — |
| Low $T$ | — | +0.51 | — | −0.03 | — | −0.35 | — | −0.50 | — | −0.57 | −0.63 | −0.74 |

Above about 1000 m/s the two panels need the same $\xi$ at the same speed, falling slowly toward $-0.74$ at $10^6$ m/s.
This matches the adiabatic finding that the effective shape factor becomes more negative at high speed ([[adiabatic-xi-diagnostics]]).
A single constant therefore works only where the plotted levels share one $\xi$: below 10 MeV most levels sit at $-0.35$ to $-0.74$, so the contours nearly coincide; at high $T$ the levels span $+2.0$ to $-0.6$.
Slow contours lie near coexistence, where $a(0^+)\to0$ and $\xi$ has little leverage, so their anchors are large and weakly constrained.

## Files

`new_paper_calculations/26-10-27/isothermal/xi_fit/fit_xi.py` writes `xi_fit.json` (fits, per-level anchors, offsets and speed ratios at each candidate).
`26-10-27-isothermal.ipynb` ("Tuning $\xi$" section) draws `local_analysis_results/26-10-27/isothermal-comparison-linear-xi-fit`, `isothermal-lowT-logT-xi-fit` and `isothermal-xi-anchors`.
The linear figure uses $+0.74$ above and $-0.57$ below 10 MeV, so dashed contours jump at the seam.
`_iso_xi` contour offsets now work on the row mesh, never interpolate across failed points, and refuse numerical data from another mesh.
