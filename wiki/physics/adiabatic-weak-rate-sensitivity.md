---
summary: Cold-interface conserved QMC-RMF3 trajectories give universal composition-dependent speed ratios, with Eq.23/exact ratios 1.037, 1.167 and 1.561 at interface composition 0.146, 0.5 and 0.9.
status: current
updated: 2026-10-11
tags: [physics, weak-rate, adiabatic, diffusion, eos, verification]
---

# Adiabatic weak-rate speed sensitivity

The standalone weak-rate manuscript now separates fixed-background isothermal sensitivity (Section 8.5, four-panel Figure 4 at 0,20,50,100 MeV) from conserved adiabatic EOS-trajectory sensitivity (Section 8.6, Figures 5–7; Figure 5 is the upstream-plane map below).
The new calculation estimates the effect on the rate integral in the speed formula. It does not solve a spatial BVP or allow the profile factor to respond to the chosen rate.

## Conserved trajectory

Inputs are upstream QMC-RMF3 pure neutron matter at $T(0^-)=0,0.1,5,20,50,100$ MeV and $1\le n_B(0^-)/n_0\le10$, with $n_0=0.16\,\mathrm{fm}^{-3}$ and $U_{\rm bag}^{1/4}=189.1566$ MeV.
The imposed quark interface has $T(0^+)=0$; the equilibrium downstream temperature is solved, not supplied.

$$
P(a)=P(0^-),\qquad h(a)/n_B(a)=h(0^-)/n_B(0^-),\qquad
n_B=\frac{4[P(0^-)+U_{\rm bag}]}{h(0^-)/n_B(0^-)}.
$$

The last relation follows from the massless bag EOS identity $h=4(P+U_{\rm bag})$. Thus quark density is exactly constant within this slow-front closure. Thermal gluons remain in the pressure; see [[bag-model-thermal-gluons]].
Solving the neutral cold pressure equation fixes $a(0^+)$. At smaller composition, invert the density cubics and constant pressure to reconstruct temperature and all flavor potentials, ending at $a=\mu_K=0$.

153 upstream states and 45 distinct target compositions are saved. Density intervals in units of $n_0$ are:

| Upstream T (MeV) | Density interval |
|---:|---:|
| 0 | (3.000000262, 5.297929093) |
| 0.1 | (2.999997731, 5.297931673) |
| 5 | (2.993658465, 5.294344371) |
| 20 | (2.894983162, 5.240216422) |
| 50 | (2.160893074, 4.923040193) |
| 100 | [1, 3.380011013) |

The first five slices cover the full interior composition interval. At 100 MeV, $a(0^+)$ has an interior minimum 0.8793682302 at $n_B(0^-)/n_0=1.4441844509$, so only compositions in $[0.8793682302,1)$ are reachable. The 0.146 and 0.5 benchmarks are unavailable; 0.9 is available. Both density roots at 0.88,0.885,0.89 are retained.

⚠ Composition is not monotone in density in the 100 MeV slice. Boundary compositions alone do not describe its accessible range. Numerical EOS failure is also distinguished from physical nonexistence: supplemental low-vector-field seeds recovered 64 queries under unchanged branch-validation criteria, leaving zero unresolved among 1,190 cached queries.

## Weighted rate comparison

$$
I_{2,\mathrm{model}}=\frac1{n_B}\int_0^{a(0^+)}D_K(a)\Gamma_{K,\mathrm{model}}(a)\,da,
\qquad \frac{u_{\mathrm{model,est}}(0^-)}{u_{\mathrm{exact,est}}(0^-)}
=\sqrt{I_{2,\mathrm{model}}/I_{2,\mathrm{exact}}}.
$$

All models use the same local EOS states, interface factors and profile parameter. Diffusion retains both existing transport terms with fixed coupling 0.3 and local temperature and baryon chemical potential.
At fixed interface composition, all temperatures and chemical potentials scale as $n_B^{1/3}$. With the weak prefactor fixed, rates scale as $n_B^{8/3}$, diffusion as $n_B^{-1/3}$, and the numerical value of $I_2$ as $n_B^{4/3}$. The weak prefactor supplies the dimensions, so $I_2$ is dimensionless in natural units.
Consequently the ratios depend only on interface composition: upstream temperature changes accessibility and dimensional states, not the ratio at matched composition.
The reporting code explicitly reuses a dimensionless quadrature at each target composition and scales dimensional integrals; identical saved ratios are therefore not independent numerical evidence for this property.
The separate `audit_adiabatic_physical.py` checks it without that reuse: six dimensional EOS reconstructions and composition-coordinate integrations at upstream temperatures 0,20,100 MeV and interface compositions 0.9,0.995 recover the published ratios within 2.10e-6, with cross-temperature ratio spread below 5.16e-13.
At composition 0.995, the cold/100 MeV upstream cases have downstream temperatures 73.9174/64.4126 MeV and exact weighted integrals 1.45833e-13/8.40912e-14, despite the same Eq.23 speed ratio 2.03970523.
The upstream densities are different when comparing the same interface composition; this is not temperature independence at fixed upstream density or fixed quark baryon chemical potential.

| Interface composition | Eq.23 / exact estimate | Derived quadratic / exact estimate |
|---:|---:|---:|
| 0.146 | 1.03698503 | 1.00350287 |
| 0.5 | 1.16660735 | 1.00814924 |
| 0.9 | 1.56107230 | 1.00462560 |

A constant-diffusion diagnostic uses exactly the same thermodynamic paths. Local diffusion increases the relative contribution of the cold unequal-flavor region. Small integrated quadratic errors are observed results, not uniform local asymptotic validity; see [[massless-finite-temperature-weak-rate]].

## Upstream-plane map (Figure 5)

With $T(0^+)=0$, $a(0^+)$ is an output of the upstream state, so plotting ratios against $a(0^+)$ was misleading and the six-panel figure was retired from the manuscript (still generated by `adiabatic_report.py`).
Figure 5 now contours Eq.23/exact over $(n_B(0^-)/n_0,T(0^-))$ via
$1+(1+a)^{4/3}+(1-a)^{4/3}=[(h/n_B)^4/(4\pi^2(P+U_{\rm bag}))]^{1/3}_{0^-}$; by scaling, iso-ratio lines are iso-$a(0^+)$ lines.
Grid: 49 temperatures 0–120 MeV × 101 densities 1–6 $n_0$, no EOS failures (`adiabatic_map.py`).
The $a(0^+)=0$ boundary runs from $3.00\,n_0$ at $T=0$ to $n_0$ near 60 MeV; the $a(0^+)=1$ boundary runs from $5.30\,n_0$ to $n_0$ at $T(0^-)=112.28$ MeV, so both close inside the plot.
Contours 1.05, 1.1, 1.2, 1.3, 1.5, 1.75 are $a(0^+)=0.192, 0.344, 0.563, 0.707, 0.870, 0.959$.

## Cold endpoint and verification

At an interior cold composition, diffusion scales as $T^{-5/3}$ and $|da/dT|$ as $T$, leaving an integrable $T^{-2/3}$ endpoint. The coordinate $z=[T/T(\infty)]^{1/3}$ and analytic EOS Jacobian produce a finite integrand, without a temperature floor.
Wolfram confirms the cold pressure derivatives, conformal identity and endpoint coefficient. Independent high-precision EOS inversion and integration in $a=a(0^+)(1-y^6)$ verify the production Jacobian-based quadrature. Saved checks separately vary EOS iterations, outer quadrature, phase-space order, thermal tail and decreasing cold cutoff, requiring speed-ratio differences below $10^{-3}$.

⚠ Finiteness of this formal integral does not establish validity of a local diffusion description at an exactly cold interface. The microscopic reference remains the tree-level four-quark channel, without extra antiquark reactions, pairing, quark masses or QCD corrections to the weak rate.

## Reproduction and sources

Redraw all seven manuscript figures with `plot_manuscript.py`; `plot_style.py` and `rate_figures.py` supply publication styling without changing the hashed collision-integral source.
Figures are 6.5 inches wide at the manuscript text width, with ordinary plot text at least 12 points against the 11-point body.
Isothermal and adiabatic curves share black circles, blue squares and orange triangles; blue labels show the last computed Eq. (23) speed ratio.
The six-panel layouts use two columns, and `results/plot-style-verification.json` records the typography and unchanged numerical-data checks.

- [Standalone manuscript](../../ref-list/massless-weak-rate/Main.pdf), [Section 8.6 source](../../ref-list/massless-weak-rate/adiabatic-section.tex).
- [Driver](../../ref-list/massless-weak-rate/run_adiabatic.py), [README](../../ref-list/massless-weak-rate/README.md), [Wolfram transcript](../../ref-list/massless-weak-rate/wolfram-adiabatic-transcript.md).
- [Physical benchmark data](../../ref-list/massless-weak-rate/results/adiabatic-speed-sensitivity.csv), [trajectories](../../ref-list/massless-weak-rate/results/adiabatic-trajectories.csv), [independent verification](../../ref-list/massless-weak-rate/results/adiabatic-independent-verification.json).

The EOS module imports production code read-only and restores temporary seed changes. No production solver or combustion manuscript is modified. The standalone directory is ignored by Git; ordinary staging does not include these artifacts.
Related: [[analytic-front-speed]], [[isothermal-analytic-front-speed]], [[quark-matter-eos]], [[unequal-flavor-weak-rate]].
