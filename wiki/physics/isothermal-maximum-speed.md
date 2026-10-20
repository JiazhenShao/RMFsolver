---
summary: The Taylor composition ceiling is not uniformly accurate to 1%; 113 of 292 saved targets exceed it, with up to 8.48% error. The exact ceiling is still solved numerically by the current solver.
status: current
updated: 2026-10-16
tags: [physics, analytic, front-speed, isothermal, bound]
---

# Isothermal maximum-speed estimate

For fixed analytical coefficients, the positive branch of [[isothermal-analytic-front-speed]] increases monotonically with $a(0^+)$.  Its thermodynamic upper estimate is therefore obtained by substituting the static-isobar ceiling $a_{\max}(0^+)$.

⚠ 2026-10-20 (unsigned): the ceiling below, $\mu_B(0^-)=\mu_B(0^+)+a\mu_K(0^+)$, gives interface entropy production $-j(1-a)\mu_K/T<0$; the second law requires $\mu_n\ge\mu_B+\mu_K$, which caps $a(0^+)$ at 0.0107 instead of 0.146 at $3n_0$, 10 MeV. See [[interface-second-law]].

## Accuracy of the Taylor ceiling (2026-10-16)

The manuscript's claim that the approximation agrees with the numerical ceiling to within 1% throughout the explored range is false for the current neutral EOS.
This concerns the error in the composition ceiling, not the speed, pressure, or chemical-potential drift.
The expansion coefficients must be evaluated at the equilibrated endpoint $\mu_B(\infty)$; replacing that endpoint with the interface value is not the same approximation.

The exact static ceiling solves pressure matching and $\mu_B(0^-)=\mu_B(0^+)+a(0^+)\mu_K(0^+)$.
An independent NumPy calculation used the exact flavor density cubics and nested density/composition bisections, with no production solver imports or BVP solves.
It reproduced all 292 successful targets in the original seven-level `new_paper_calculations/26-10-02/isothermal_analytic_logT/isothermal-analytic-logT.npy` checkpoint to an absolute composition difference below $1.1\times10^{-11}$.
Of those targets, 113 have Taylor errors above 1%; the largest is 8.4767%, at $T=66.0385$ MeV, $n_B(0^-)=3.77804n_0$, where the exact ceiling is 0.998550 and the approximation is 1.083195.
These statistics cover this checkpoint's successful targets, not every calculation in the project.

At $T=20$ MeV and $U_{\rm bag}^{1/4}=189.1565957$ MeV:

| $n_B(0^-)/n_0$ | Exact ceiling | Taylor ceiling | Relative error |
|---|---|---|---|
| 3.16988 | 0.411355 | 0.414176 | +0.686% |
| 3.48308 | 0.565781 | 0.573591 | +1.380% |
| 4.47275 | 0.865520 | 0.900944 | +4.093% |
| 5.11769 | 0.989281 | 1.056334 | +6.778% |

The displayed isothermal trajectory at $T=10$ MeV and $n_B(0^-)=3n_0$ is a good small-composition example: its exact ceiling is 0.1461083 and the approximation gives 0.1462255, an error of only 0.0802%.
That example cannot establish a global 1% bound.

At zero temperature the exact neutral EOS eliminates the pressure scale and gives

$$
\frac{\mu_B(0^-)}{\mu_B(\infty)}
=\left[\frac{1+(1+a(0^+))^{4/3}+(1-a(0^+))^{4/3}}{3}\right]^{3/4}
$$

at the exact ceiling.
Substituting this into the Taylor estimate gives its square as $a^2+2a^4/27+61a^6/2187+\cdots$.
Wolfram verified the series and the 1% crossing at $a=0.4955441761$.
Cold relative errors are 1.020%, 2.235%, 4.532%, and 7.050% at exact ceilings 0.5, 0.7, 0.9, and 1, respectively.
This is a cold-limit threshold, not a temperature-independent accuracy guarantee.

Reproduction and full state-level results are in `local_analysis_results/26-10-16/aiso-accuracy/audit.py` and `results.json`; Wolfram inputs and outputs are in `verification.wl`.
Increasing both bisection counts from 50 to 60 changes the independently recovered ceilings by at most $1.3\times10^{-13}$; pressure and Gibbs residuals are below $5\times10^{-15}$ relative.
Wolfram independently solved two finite-temperature neutral pressure/Gibbs systems and obtained the same 1.380% and 4.093% errors, with constraint residuals below $3\times10^{-16}$.
The production helper `phase_velocity._solve_a_0plus_max` already computes the exact static ceiling; this audit does not imply its saved speed calculations used the inaccurate Taylor estimate.
The manuscript and solver were left unchanged; the narrowest correction is to remove the quantitative accuracy clause until a specific supported range is stated.

The audit is plotted in `26-10-14.ipynb`, cells 11–12, and embedded under "Additional Check: Accuracy of Eq. (35)" in `26-10-14.md`.
Three panels show composition values, signed differences, and percentage differences versus exact composition, with temperature-colored saved points and the verified cold-limit curve.
The exact baseline is the identity in the value panel and zero in both difference panels; the percentage denominator is the exact ceiling.
PNG/PDF exports are `local_analysis_results/26-10-14/aiso-accuracy.*`, generated from the saved audit without new EOS/BVP solves.

## Static-isobar ceiling

As of 2026-10-05, Eq. (35), `eq:a0plus_max`, in `Latex_writting/Main.tex` directly substitutes the neutral EOS into the ceiling below and uses the paper's constant quark-side notation $\mu_B^{(+)}$ in place of $\mu_B(\infty)$. The finite-temperature ratio is retained, and Wolfram verified that the explicit radicand equals the susceptibility-over-density expression. This is an algebraic substitution, not a new numerical validation of the Taylor ceiling.

Expand the quark pressure about the equilibrated endpoint, where $\mu_K(\infty)=0$ and $n_K(\infty)=0$.  To quadratic order,

$$
P_{\rm QM}(0^+)-P_{\rm QM}(\infty)
\simeq n_B(\infty)[\mu_B(0^+)-\mu_B(\infty)]
+\frac12\left.\frac{\partial^2P_{\rm QM}}{\partial\mu_K^2}\right|_{\infty}\mu_K^2(0^+)=0,
$$

and

$$
a(0^+)\simeq\frac{1}{n_B(\infty)}
\left.\frac{\partial^2P_{\rm QM}}{\partial\mu_K^2}\right|_{\infty}\mu_K(0^+).
$$

Combining these with $\mu_B(0^-)=\mu_B(0^+)+a(0^+)\mu_K(0^+)$ gives

$$
a_{\max}^2(0^+)\simeq
\frac{2[\mu_B(0^-)-\mu_B(\infty)]}{n_B(\infty)}
\left.\frac{\partial^2P_{\rm QM}}{\partial\mu_K^2}\right|_{\infty}.
$$

For the massless bag-model expansion in [[quark-matter-eos]], this becomes

$$
a_{\max}^2(0^+)\simeq
\frac{9[\mu_B(0^-)-\mu_B(\infty)]}{\mu_B(\infty)}
\frac{\mu_B^2(\infty)+3\pi^2T^2}
{\mu_B^2(\infty)+9\pi^2T^2},
$$

whose cold limit is $a_{\max}^2(0^+)\simeq9[\mu_B(0^-)-\mu_B(\infty)]/\mu_B(\infty)$.

## Maximum-speed formula

Using $\lambda_n=n_B(0^-)/n_B(\infty)$ and selecting the positive branch gives

$$
u_{\max}(0^-)=\frac{n_B(\infty)}{n_B(0^-)}
\sqrt{\frac{D_K\gamma_K}{2}}\,
\frac{a_{\max}(0^+)\sqrt{a_{\max}^2(0^+)+2\eta}}
{\sqrt{1-a_{\max}(0^+)}\sqrt{1+\xi a_{\max}(0^+)}}.
$$

This is the direct isothermal analogue of the maximum-speed expression in [[analytic-front-speed]].  The formula is an exact substitution into the analytical speed relation, but the Taylor estimate for $a_{\max}(0^+)$ remains approximate.

Substituting the full flavor-diffusion coefficient and $\eta=9\pi^2T^2/\mu_q^2$ makes the retained temperature and chemical-potential dependence explicit:

$$
u_{\max}(0^-)=\frac{n_B(\infty)}{n_B(0^-)}
\frac{a_{\max}(0^+)}{4\alpha_s}
\sqrt{
\frac{\pi\gamma_K[a_{\max}^2(0^+)+18\pi^2T^2/\mu_q^2]}
{N_f\left[h_DT^{5/3}/q_D^{2/3}+\pi^3T^2/(12q_D)\right]
[1-a_{\max}(0^+)][1+\xi a_{\max}(0^+)]}
}.
$$

Isothermality alone does not make $D_K$ constant: this expression uses the same frozen representative quark state as the analytical derivation, presently the $0^+$ state in the solver.  The susceptibility and $n_B(\infty)$ in the ceiling are evaluated at the equilibrated endpoint.

When the Landau-damped term dominates, the Eq.-(31)-style normalized form that retains both weak-rate terms is

$$
c\,u_{\max}(0^-)\simeq(98.3\ {\rm m\,s^{-1}})
\frac{n_B(\infty)}{n_B(0^-)}
\left(\frac{\mu_B(0^+)}{1200\ {\rm MeV}}\right)^{17/6}
\left(\frac{10\ {\rm MeV}}{T}\right)^{5/6}
\left(\frac{3}{N_f}\right)^{1/3}
\left(\frac{1.81317}{h_D}\right)^{1/2}
\frac{a_{\max}(0^+)\sqrt{a_{\max}^2(0^+)+162\pi^2T^2/\mu_B^2(0^+)}}
{\alpha_s^{5/6}\sqrt{1-a_{\max}(0^+)}\sqrt{1+\xi a_{\max}(0^+)}}.
$$

Here $\mu_q=\mu_B(0^+)/3$ is the frozen representative analytical state.  In the suprathermal limit $a_{\max}^2(0^+)\gg162\pi^2T^2/\mu_B^2(0^+)$, the composition numerator reduces to $a_{\max}^2(0^+)$.  Retaining the longitudinal $T^2$ term in $D_K^{-1}$ multiplies the displayed Landau-damped estimate by $[1+\pi^3(T/q_D)^{1/3}/(12h_D)]^{-1/2}$.

The divergence as $a_{\max}(0^+)\to1$ is the same diffusion-limited singularity already identified in the contour workflow, not a physical infinite speed.  The Taylor ceiling and moving-front formula require $0<a_{\max}(0^+)<1$; stable neutron matter and coexistence are classified separately.

## Wolfram checks

The exact substitution was checked with the ASCII-equivalent input

```text
s = 2*dmu*chi/ninf
u2[a_] = dk*gk*a^2*(a^2 + 2*eta)/(2*lam^2*(1-a)*(1+xi*a))
FullSimplify[Sqrt[u2[Sqrt[s]]] -
  Sqrt[(dk*gk/2)*s*(s+2*eta)/((1-Sqrt[s])*(1+xi*Sqrt[s]))]/lam]
```

under the physical sign assumptions; Wolfram returned `0`.  It also returned

```text
d Log[u(a)]/da =
  (1/(1-a) + 2*a^2/(a^2+2*eta) + 1/(1+xi*a))/(2*a)
```

which is positive for $0<a<1$, $\eta\geq0$, and $-1<\xi<1$.  Substitution of the full $D_K$ and $\eta$ expressions also simplified to zero.  With $\alpha_s^{-5/6}$ explicit as in Eq. (31), the normalized Landau-damped coefficient is `98.29738983094435` m/s; at $\alpha_s=0.3$ it is `268.08608563251715` m/s.
