---
summary: Fitted per temperature strip, the isothermal xi rises monotonically from -0.63 at 0.01 MeV to -0.43 at 3-10 MeV, -0.18 at 10-20 MeV and above +1 beyond 60 MeV; below 10 MeV speed sets xi, above it T does; fitted xi > 1 corrects the analytical I2 (1.6x too large), while the profiles' own xi stays <= 1.09 although -D_K a' is non-monotone in a third of them.
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

The bound $-1<\xi<1$ belongs to the $I_1$ parameterization and assumes $g=-D_Ka'\le G$. The fitted $\xi_{\rm speed}$, at which the analytical formula gives the numerical speed, also absorbs the formula's other approximations.

Compare it with $\xi_{\rm profile}=2I_1/(G\,a(0^+))-1$, computed literally on each saved profile: $g=-D_Ka'$ by finite differences, $I_1=\int D_Ka'^2dx$, $G=g(0^+)$.
- Of 2282 high-$T$ points, 1408 have $\xi_{\rm speed}>1$; only 59 have $\xi_{\rm profile}>1$, all above 90 MeV, with a maximum of $+1.09$.

**$I_2$, not shape.** At 48–71 MeV, profiles with $\xi_{\rm speed}=1.5$–$2.1$ are straight ramps ($\xi_{\rm profile}\approx0$).
- Where $\xi_{\rm speed}>1$, the analytical $I_2$ (cubic manuscript rate, transport frozen at $\mu_B(0^+)$) exceeds the numerical $\int D_K\Gamma_K\,da/n_B(0^+)$ by a median 1.6 (1.8 at 50–70 MeV).
- $\lambda_n$ taken at $n_B(\infty)$ adds about 1.1 to $\lambda_n^2$.
- Because $u^2\propto I_2/(1+\xi a)$, the excess pushes $\xi$ above 1, most strongly where $a(0^+)$ is small.

**Non-monotone flux.** $a(x)$ is monotone in every profile, but $g$ peaks inside the layer in 831 of them, by up to 1.47 $G$ near $a(0^+)\to1$. The area stays close to the rectangle, so the appendix bound is only marginally violated, but its premise (monotone $g$) fails.
- $|a'|$ alone peaks inside in 595 profiles, so $a(x)$ has an inflection point there. In the three peaked profiles (77–92 MeV; `isothermal-xi-gt1-profile-fields.png` shows the 91 MeV one in nm), $D_K$ also rises by 27% across the layer, lifting the peak of $g$ from 1.11–1.25 to 1.24–1.44 $G$.
- Mechanism (exact, from $a=n_K/n_B$, $J_K=j_Ba-D_Kn_K'$, $J_K'=-\Gamma_K$): $(-D_Kn_K')'=j_B|a'|-\Gamma_K$. The current grows wherever advective supply beats conversion.
  - The peaked profiles at 77–92 MeV balance at $0^+$ (ratio 1.00–1.03); the current then peaks at about 2 times its $0^+$ value.
  - The slow ramps at 48–71 MeV have conversion winning at $0^+$ (ratio 0.40–0.84).
  - This was checked at six points.
- $\xi_{\rm profile}$ never falls below $-0.40$.

⚠ **Definition trap.** Above 72 MeV, $n_B$ varies by up to 13% across the layer.
- $-D_Ka'$ then differs from the solver's diffusive current per baryon, $-D_Kn_K'/n_B$ (the solver uses $J_K=u\,n_K-D_Kn_K'$).
- The jump-condition $G=\lambda u(0^-)[1-a(0^+)]$, with $\lambda=n_B(0^-)/n_B(0^+)$, fixes the solver's flux; $-D_Ka'(0^+)$ is 1.15–1.56 times larger when $a(0^+)\ge0.99$.
- The record field `xi_profile` uses the solver flux with the jump $G$ and gives $\xi>1$ at 343 points, up to $+1.59$.
- Superseded 2026-11-01: an earlier version of this page reported those record values as the profile's own $\xi$ and called the appendix bound contradicted.

`xi_fit/xi_gt1_profiles.py` draws $\xi_{\rm speed}$ and the literal $\xi_{\rm profile}$ versus $T$, and the parametric curves $(a(x), g(x))/G$ of the saved profiles.

## Files

- `new_paper_calculations/26-10-27/isothermal/xi_fit/fit_xi_strips.py` writes `xi_strips.json` in 25 s. It tabulates the contour offsets once per panel on a $\xi$ grid (step 0.01) and refines the grid minimum with a parabola.
- `report_figures.py` draws the report figures.
- `tests/test_isothermal_xi_strips.py` checks that a whole-panel strip reproduces `fit_xi.py` and that a strip fit matches direct minimization.
- The report is `26-11-01.md` (workspace root; figures in `local_analysis_results/26-11-01/isothermal-xi-report/`).
