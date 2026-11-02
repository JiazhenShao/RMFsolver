---
summary: Fitted per temperature strip, the isothermal xi rises monotonically from -0.63 at 0.01 MeV to -0.43 at 3-10 MeV, -0.18 at 10-20 MeV and above +1 beyond 60 MeV; below 10 MeV the front speed sets xi, above 10 MeV xi rises with T at every speed; fitted xi > 1 mostly corrects the analytical I2, and only near a(0+) -> 1 is the profile's own xi > 1 (non-monotone flux).
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
- Above about 60 MeV, $\xi>1$. ~~No profile shape allows this~~ (superseded 2026-11-01, see the next section): $\xi>1$ needs the flux $g=-D_Ka'$ to exceed its interface value $G$ somewhere, which some profiles do.
- ⚠ The 80–95 MeV value is weakly constrained. The contours crowd against the upper boundary, so offsets are $0.004\,n_0$ at its own fit and still only $0.033\,n_0$ at $+0.74$.

This refines the whole-panel per-level anchors of [[isothermal-xi-fit]]: there, "depends on the speed level, not the panel" holds only below about 10 MeV.

## Why the fitted $\xi$ exceeds 1

Compare $\xi_{\rm speed}$, the value at which the analytical formula gives the numerical speed, with $\xi_{\rm profile}=2I_1/(G\,a(0^+))-1$ from each numerical profile (record field `xi_profile`, $G=\lambda u(0^-)[1-a(0^+)]$ with $\lambda=n_B(0^-)/n_B(0^+)$).
Of 2282 high-$T$ points, 1408 have $\xi_{\rm speed}>1$, but only 343 have $\xi_{\rm profile}>1$.

**$I_2$, not shape.** At 48–69 MeV, profiles with $\xi_{\rm speed}=1.6$–$2.1$ are straight ramps ($\xi_{\rm profile}\approx0$, $g\le G$).
- Where $\xi_{\rm speed}>1$, the analytical $I_2$ (cubic manuscript rate, transport frozen at $\mu_B(0^+)$) exceeds the numerical $I_2$ (full local rate along the profile) by a median 1.65.
- $\lambda_n$ taken at $n_B(\infty)$ adds a factor of 1.10 to $\lambda_n^2$.
- Because $u^2\propto I_2/(1+\xi a)$, the excess $I_2$ pushes $\xi$ above 1, most strongly where $a(0^+)$ is small (slow fronts near coexistence).

**Non-monotone flux.** For $T>72$ MeV and $a(0^+)\ge0.993$, $a(x)$ stays monotone, but $g$ peaks at 1.6–2.1 $G$ inside the layer, so $\xi_{\rm profile}$ reaches $+1.59$.
- The jump condition ties $G$ to $1-a(0^+)\to0$, while the diffusive flux inside the layer does not vanish with it.
- 40% of high-$T$ profiles have a non-monotone $g$, but only those near $a(0^+)\to1$ exceed $\xi_{\rm profile}=1$.
- ⚠ This contradicts the `Main.tex` appendix statement that every exact solution gives $-1<\xi<1$.

`xi_fit/xi_gt1_profiles.py` draws $\xi_{\rm speed}$ and $\xi_{\rm profile}$ versus $T$, and $g/G$ versus $a/a(0^+)$ for both cases.

## Files

- `new_paper_calculations/26-10-27/isothermal/xi_fit/fit_xi_strips.py` writes `xi_strips.json` in 25 s. It tabulates the contour offsets once per panel on a $\xi$ grid (step 0.01) and refines the grid minimum with a parabola.
- `report_figures.py` draws the report figures.
- `tests/test_isothermal_xi_strips.py` checks that a whole-panel strip reproduces `fit_xi.py` and that a strip fit matches direct minimization.
- The report is `26-11-01.md` (workspace root; figures in `local_analysis_results/26-11-01/isothermal-xi-report/`).
