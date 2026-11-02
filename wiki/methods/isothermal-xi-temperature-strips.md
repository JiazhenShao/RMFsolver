---
summary: Fitted per temperature strip, the isothermal xi rises monotonically from -0.63 at 0.01 MeV to -0.43 at 3-10 MeV, -0.18 at 10-20 MeV and above +1 beyond 60 MeV; below 10 MeV the front speed sets xi, above 10 MeV xi rises with T at every speed and leaves the profile-shape range.
status: current
updated: 2026-11-01
tags: [method, isothermal, analytic, xi, contour, temperature]
---

# Isothermal $\xi$ in temperature strips

The contour fit of [[isothermal-xi-fit]] was restricted to the plotted-contour crossings in each temperature strip:
- half-decade strips on the low-$T$ row mesh (0.01–10 MeV), with levels 100–$10^6$ m/s;
- 10 MeV strips on the high-$T$ polar mesh (10–95 MeV, with 80–95 merged), with levels 50–5000 m/s.

Error bars come from 400 bootstrap resamples of the rays or rows. No solver was called.

| $T$ [MeV] | 0.01–0.03 | 0.1–0.3 | 1–3 | 3–10 | 10–20 | 20–30 | 30–40 | 40–50 | 50–60 | 60–70 | 70–80 | 80–95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $\xi$ | $-0.63$ | $-0.60$ | $-0.52$ | $-0.43$ | $-0.18$ | $+0.13$ | $+0.44$ | $+0.73$ | $+0.96$ | $+1.21$ | $+1.66$ | $+2.79$ |

The two panel fits, $-0.57$ and $+0.74$, are averages over different parts of this one trend. The trend crosses the 10 MeV mesh seam without a jump, and per level the two meshes agree there to within $0.16$.

## Speed versus temperature

**Below 10 MeV, speed sets $\xi$.** At fixed $T$, $\xi$ falls with speed: $+0.96$ at 100 m/s to $-0.74$ at $10^6$ m/s in the coldest strip.
- At fixed speed, $\xi$ decreases slightly as $T$ rises.
- The strip value still drifts upward because the fastest contours, which want the most negative $\xi$, stop crossing the rows: $10^6$ m/s above 0.24 MeV and $3\times10^5$ m/s above 2.9 MeV.

**Above 10 MeV, temperature takes over.** $\xi$ rises with $T$ at every level; for example, 1000 m/s goes from $-0.38$ at 10–20 MeV to $+1.57$ at 70–80 MeV.
- Above about 60 MeV, $\xi>1$, which no profile shape allows ($-1<\xi<1$, `Main.tex` appendix). There the fit absorbs the error of the rest of the analytical approximation rather than describing $g(a)$.
- ⚠ The 80–95 MeV value is weakly constrained. The contours crowd against the upper boundary, so offsets are $0.004\,n_0$ at its own fit and still only $0.033\,n_0$ at $+0.74$.

This refines the whole-panel per-level anchors of [[isothermal-xi-fit]]: there, "depends on the speed level, not the panel" holds only below about 10 MeV.

## Files

- `new_paper_calculations/26-10-27/isothermal/xi_fit/fit_xi_strips.py` writes `xi_strips.json` in 25 s. It tabulates the contour offsets once per panel on a $\xi$ grid (step 0.01) and refines the grid minimum with a parabola.
- `report_figures.py` draws the report figures.
- `tests/test_isothermal_xi_strips.py` checks that a whole-panel strip reproduces `fit_xi.py` and that a strip fit matches direct minimization.
- The report is `local_analysis_results/26-11-01/isothermal-xi-report/REPORT.md`.
