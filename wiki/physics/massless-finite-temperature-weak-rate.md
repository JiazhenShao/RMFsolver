---
summary: Exact neutral-ordering cold and T-squared weak rates are derived and checked against a 192-state grid; equal up thresholds produce a genuine T-cubed remainder.
status: current
updated: 2026-10-15
tags: [physics, weak-rates, phase-space, sommerfeld]
---

# Massless finite-temperature weak rate

For a cheap accurate rate across the proposed temperature range, use a correctly branched cold expansion and a tabulated massless finite-temperature integral outside its validated domain.
Equation (23) is not an expansion solely in temperature: it also uses nearly equal flavor chemical potentials.
The earlier open question about the finite-imbalance T-squared coefficient was resolved on 2026-10-07 by the standalone derivation in `ref-list/massless-weak-rate/`.
That package contains the manuscript, local bibliography, Wolfram transcript, independent symbolic helpers, numerical reference, and saved comparison grid.
See [[unequal-flavor-weak-rate]] for the invariant rate and cold benchmarks.

The collaborator export in `local_analysis_results/26-10-15/massless-phase-space/` freshly evaluates 60 fixed-muB,a states: muB=900,1200 MeV, a=0.146,0.5,0.9 and T=0,0.01,0.1,1,5,10,20,25,50,100 MeV.
Combined and six panel CSVs distinguish forward/reverse/net unprefactored phase space (MeV^8) from weak rates (MeV^4), include all flavor potentials and the four plotted rate columns, and document the prefactor and detailed balance explicitly.
The green column replaces only mu_u^5 by (mu_B/3)^5 in Eq.23, not the Eq.24 correction. Separate convergence data include 12 independent radial 3D checks, six analytical cold checks and agreement with all 48 overlapping manuscript rows; all meet the 1e-6 comparison threshold.
The directory and sibling ZIP include a reproducible driver, unchanged source snapshots, metadata and export verification; no production solver or manuscript calculation was modified.

## Direct inspection of the original paper

The user supplied `ref-list/Rate-PhysRevD.47.325.pdf` on 2026-10-07.
Pages 326--329 were rendered and visually inspected; this supersedes the earlier indexed-text-only access limitation.
Both printed cold formulas (20) and (21) require mu_u < mu_d in Madsen's original s -> d labeling.
Their other conditions distinguish mu_s < mu_u+mu_d from mu_s > mu_u+mu_d.
For our d -> s process the down and strange labels must be exchanged, requiring mu_u < mu_s for those printed branches.
Our neutral massless matter with positive a instead has mu_s < mu_u < mu_d, at any common temperature, because the density is monotone in chemical potential.
Thus neither printed branch is directly applicable; the earlier recommendation to select between those two branches for our matter was too broad.

Equation (20) has no explicit temperature dependence.
Evaluating its chemical potentials using a finite-temperature EOS does not restore the thermal smearing of reaction phase space.
Madsen's improved approximation adds the T-squared part of Eq. (23), multiplied by Eq. (24), to his applicable cold formula.
The Eq. (24) multiplier, in his original z and v variables, reads

```text
z = mu_d/mu_u
v = (mu_s-mu_d)/mu_u
F24 = -v^5/128 - 5*z*v^4/64 + 5*(2-z^2)*v^3/64
      + 5*(1+3*z)*v^2/16 + 5*(4*z+3*z^2)*v/16
      + 5*z^2/4 - 1/4
```

⚠ The v^4 term carries a factor z: the 600-dpi render of p. 329 reads `-(5/64) z v^4`.
Before 2026-10-07 this page, the standalone manuscript's Eq. (8.2), and `eq24_factor` in `validate_rate.py` all dropped that z.
The original "Cold + Eq. (24)" diagnostic curves in its Figs. 1--2 therefore did not use Madsen's printed formula; the missing term matters more as a grows, because z = mu_s/mu_u after relabeling.
The manuscript and code were corrected later on 2026-10-07, and the full grid and figures were regenerated; only the two Eq. (24) diagnostic columns changed, with all other 192-state data identical.
At mu_B=900 MeV and a=1, the corrected diagnostic errors are +15.4916%, +32.9033%, and +11.2145% at T=25,50,100 MeV.
At mu_B=900 MeV and a=1, the printed formula gives errors of +15.5%, +32.9% and +11.2% at 25, 50 and 100 MeV, versus +1.7%, +0.4% and -26.8% for the dropped-z version.
No derived result, table, or conclusion of the manuscript depends on this curve.

This is the separate thermal multiplier, not the braces in Eq. (20).
Neither the low-temperature fit nor a Sommerfeld truncation is guaranteed accurate at T=100 MeV or when the strange Fermi energy approaches zero.

## Exact two-dimensional reduction

This reduction is independent work based on Madsen's tree-level matrix element, retaining the four-quark channel but no QCD or additional antiquark channels.
Use the common total energy E and total spatial momentum magnitude k of the incoming and outgoing pairs.
All single-quark energies lie in [(E-k)/2,(E+k)/2].
Changing integration order in the massless reduction from [[unequal-flavor-weak-rate]] yields

```text
R(d+u -> s+u) = 9*G_F^2*sin(theta_C)^2*cos(theta_C)^2/(8*pi^5)
  * integral_0^infinity dE integral_0^E dk (E^2-k^2)^2
  * [integral_(E-k)/2^((E+k)/2) dp f_u(p)*f_d(E-p)]
  * [integral_(E-k)/2^((E+k)/2) dp (1-f_u(p))*(1-f_s(E-p))]
Gamma_K = (1-exp(-mu_K/T))*R(d+u -> s+u)
f_i(p) = 1/(1+exp((p-mu_i)/T))
```

The two bracketed integrals have elementary logarithmic antiderivatives, leaving only E and k for numerical quadrature.
For any pair of chemical potentials ma and mb, an antiderivative of f_a(p)*f_b(E-p) is

```text
[-T*log(1+exp((ma-p)/T)) + T*log(1+exp((p-E+mb)/T)) - p]
    / [1-exp((E-ma-mb)/T)]
```

The hole-pair integral is exp((E-ma-mb)/T) times the occupied-pair integral.
At E=ma+mb the apparent singularity is removable: the occupied-pair definite integral is T*(f_a((E-k)/2)-f_a((E+k)/2)); the hole pair is identical there.
Use stable softplus/log1p and expm1 expressions, explicit removable limits, and integration splits at the Fermi boundaries.
At T=0 the pair integrals become lengths of interval intersections, so the zero-temperature limit does not require resolving narrow Fermi functions.

Wolfram verified both antiderivative branches with the following input, returning {0,0}:

```wolfram
Clear[x,en,mua,mub,t,f,h];
f[x_,m_]:=1/(1+Exp[(x-m)/t]);
h=(-t Log[1+Exp[(mua-x)/t]]+t Log[1+Exp[(x-en+mub)/t]]-x)/
  (1-Exp[(en-mua-mub)/t]);
{FullSimplify[D[h,x]-f[x,mua] f[en-x,mub],
  Assumptions->{t>0,Element[{x,en,mua,mub},Reals],en!=mua+mub}],
 FullSimplify[D[-t f[x,mua],x]-f[x,mua] f[en-x,mub]/.en->mua+mub,
  Assumptions->{t>0,Element[{x,mua,mub},Reals]}]}
```

It also verified that half of the angular kernel reproduces the polynomial integrand printed in Eq. (20), on that equation's branch.
These are algebraic checks, not a production implementation or finite-temperature front validation.

## Sommerfeld and near-equilibrium checks

On smooth cold branches at fixed chemical potentials, a Sommerfeld treatment retains the exact cold rate and adds temperature corrections derived from the full collision integral.
It must account for all four Fermi factors; the two up-quark factors cannot be collapsed into one differentiation variable before constructing a multivariate expansion.
Near branch boundaries, vanishing Fermi surfaces, or simultaneous small imbalance and temperature, uniform treatment of the Fermi factors may be needed instead of a fixed-imbalance expansion.
See the [standard Sommerfeld derivation](https://www.imsc.res.in/~ganesh/courses/CMP1_2016/Sommerfeld_expansion.pdf) for the smoothness and degeneracy conditions.

An independent leading near-equilibrium check exposes a limitation of treating Eq. (24) as an exact general thermal coefficient.
With mu_s approaching mu_d and z=mu_d/mu_u fixed, the Fermi-surface angular coefficient relative to Eq. (23) is

```text
F = (1+5*z+10*z^2)/16, z>=1
F = z^3*(10+5*z+z^2)/16, 0<=z<=1
```

These follow from integrating (E^2-k^2)^2 at E=mu_u+mu_d and k from abs(mu_u-mu_d) to mu_u+mu_d.
Wolfram input for the upper branch was `Factor[(15/256) Integrate[((1+z)^2-k^2)^2,{k,z-1,z+1}]]`.
At z=2 this gives 51/16, whereas the printed Eq. (24) at v=0 gives 19/4.
A prototype two-dimensional quadrature at mu_u=1, mu_d=2, mu_s=1.999 gives full/Eq. (23) ratios 3.10954105 at T=0.01 and 3.16189588 at T=0.003; 48 and 96 Gauss-Legendre nodes per subinterval agree to within 4e-7.
This is a non-neutral diagnostic approaching the angular limit, not a benchmark of the manuscript's front.
It reinforces that the printed thermal correction should not be assumed to be an exact systematic coefficient for arbitrary flavor potentials.

For neutral massless matter, scale invariance permits tabulation in a and T/mu_B, with the overall rate scaling as G_F^2*mu_B^8 times the Cabibbo factors.
The main combustion manuscript and production solver have not been changed by this standalone calculation.

## Completed cold and thermal derivation

The exact cold integral for `0 <= mu_s < mu_u < mu_d` has only two branches, separated by `mu_d-mu_s = mu_u`.
Writing its upper-branch degree-eight polynomial as `P8`, the lower-branch correction is

```text
I0 = P8 - max(mu_u+mu_s-mu_d,0)^6/315
     * [15*mu_d^2 + 10*mu_d*(mu_s+mu_u) + 3*(mu_s+mu_u)^2]
Gamma0 = (45/128)*C_K*I0
C_K = 16*G_F^2*sin(theta_C)^2*cos(theta_C)^2/(5*pi^5)
```

The full polynomial is in `cold-result.tex`, with its three elementary integration regions in `algebra-appendix.tex`.
The neutral cold branch switches at `a = 0.983149408702098`; the rate matches through five derivatives there.
The physical common up potential must not be differentiated as one threshold when deriving the thermal coefficient: four independent occupation thresholds give a Hessian trace `J2`.
The complete degree-six coefficient and all edge terms are in `thermal-result.tex` and Appendix C.

```text
GammaK = (45/128)*C_K * [I0 + (pi^2/6)*J2*T^2
          - (64/3)*zeta(3)*mu_u^2*(mu_d^3-mu_s^3)*T^3 + O(T^4)]
```

This is the fixed-positive-imbalance, strictly ordered asymptotic expansion.
The coincident incoming/outgoing up thresholds introduce a cubic positive-part cusp in the independently displaced cold integral.
Smoothing that cusp explains the T-cubed term: the quadratic truncation does **not** have a generic T-fourth remainder.
At `mu_s=0`, the angular kernel vanishes as the third power of strange momentum, so the strange endpoint first changes the expansion at fourth order; the displayed quadratic and cubic coefficients remain valid.
At exact equilibrium the net rate is zero, while the forward rate starts at T-cubed; do not use the fixed-imbalance cubic coefficient to compute that forward limit.

At fixed neutral `mu_B,mu_K`, the flavor shifts also contribute:

```text
mu_f(T) = mu_f(0) + c_f*T^2 + ...
c_d = c_s = pi^2*(2*mu_u-mu_d-mu_s)/(3*(4*mu_u^2+mu_d^2+mu_s^2))
c_u = -2*c_d
EOS contribution to the T^2 phase-space coefficient
  = c_d*(-2*d/dmu_u + d/dmu_d + d/dmu_s)*I0
```

All potentials in these coefficients are the cold neutral values.
The density EOS includes antiparticles through net densities, but this does not add antiquark reaction channels to the calculated collision integral.

## Numerical validation and scope

The saved grid has 192 states: `mu_B = 900,1200 MeV`, `a = .001,.146,.5,.9,.99,1`, eight temperatures from `.01` to `100 MeV`, and fixed-composition versus fixed-imbalance neutral EOS paths.
The maximum quadrature change from 24 to 40 nodes per split interval is `1.71e-13`; a 48T-to-64T tail extension changes results by at most `5.56e-15`.
Independent finite-temperature three-dimensional energy quadrature agrees within `3.11e-15` at two tested states; separate cold quadrature agrees with the polynomial within `1.72e-8`.
These are observed convergence differences, not rigorous interval error bounds.
The corrected constrained T-squared coefficient is numerically reproduced within `9.71e-8`.

The 5% range is composition dependent, not the full requested temperature span.
At fixed `a=.146`, the highest consecutively passing sampled temperatures are 5 MeV for `mu_B=900 MeV` and 10 MeV for 1200 MeV.
At `a=.001`, both cases pass only through the sampled 1 MeV point; several larger compositions pass through 25 or 50 MeV.
The apparently good 100 MeV result for `mu_B=1200 MeV,a=.146` is an error crossing after failure at 25 and 50 MeV, not renewed asymptotic validity.
The published Eq. (23) overestimates the cold zero-strangeness endpoint rate by about a factor of 9.12.
These are local-rate errors, not errors in a solved front speed or tests of omitted high-temperature channels.

## Independent re-verification (2026-10-07)

A separate session re-derived the manuscript without using its scripts, and every derivation held.
- **Cold result.** Rebuilt the cold integral with four independent thresholds directly from the triple integral. It reproduces `I0` on both branches, `F1`–`F3`, the sixth-power matching, `J2` on both branches, and each of the four Appendix C boundary sums.
- **Cubic cusp.** The exact correction for nu1 = nu2 + h is a polynomial whose h^3 coefficient is -(16/9) mu_u^2 (mu_d^3 - mu_s^3).
- **Normalization.** The spin trace was checked with explicit Dirac matrices. Madsen's Eq. (20) integrand is exactly half the angular kernel, with prefactor 72/(2 pi)^5, which reproduces 9/(8 pi^5).
- **Finite temperature.** A new (E,m) integrator with numerically integrated pair factors agrees with a brute-force three-dimensional energy-space integral to 2e-15. It reproduces the quoted cubic residuals -8.71812 … -7.64170, the a=1 quartic coefficient 595.4, all 192 saved grid rates to 5e-12, Table 2 cell for cell, and the neutral-EOS chain rule.
- **Normalization caveat.** Madsen's printed Eq. (4) reads 36(2 pi)^(-5) with bare d^3p. That is (2 pi)^3 larger than the standard (2 pi)^(-8) normalization the manuscript attributes to it. The manuscript's value is the one his own Eqs. (13) and (20)–(22) require, so the printed Eq. (4) is a misprint.

The standalone manuscript now states this normalization misprint in a footnote and attributes the matrix element separately from the invariant measure.

## Three-model figures and speed sensitivity (2026-10-07)

The two-temperature scope and test/build counts recorded below are historical, superseded on 2026-10-11 by Section 8.5 at T=0,20,50,100 MeV and the new conserved-trajectory Section 8.6; see [[adiabatic-weak-rate-sensitivity]] for the current calculation and results.

The standalone PDF now compares exactly three models in every figure: the exact phase-space integral, Madsen Eq. (23), and the derived T0+T2 prescription evaluated at the same finite-temperature neutral chemical potentials.
The T3 and Eq. (24) diagnostics remain in the CSV rather than the plot legends.
`ref-list/massless-weak-rate/speed_sensitivity.py` records reproducible rate-integral speed ratios at fixed muB=1200 MeV and T=0,20 MeV, with the diffusion, density normalization and interface/profile factors fixed.
These are sensitivity estimates, not BVP solutions or constant-pressure trajectories.
At a(0+)=0.146,0.5,0.9 the exact-rate estimate changes by -2.71,-11.39,-29.78 percent in the cold limit and -8.68,-13.19,-29.03 percent at 20 MeV; the derived T0+T2 estimate gives -3.12,-11.46,-28.67 percent at 20 MeV.
The exact and derived estimates coincide in the cold limit.
The maximum refined integral changes are 5.57e-9 (exact), 2.09e-8 (Eq.23), and 5.58e-9 (T0+T2); CSV/JSON convergence settings, the figure, and a generated LaTeX table are saved beside the derivation.
All 17 tests pass and the 20-page PDF builds without warnings; the production solver and combustion manuscript remain unchanged.

## Fourth local-rate model: Eq. (23) with (mu_B/3)^5 (2026-10-11)

Figures 1-4 and 6 of the standalone PDF add Madsen Eq. (23) with $\mu_u^5$ replaced by $(\mu_B/3)^5$; Figures 5 and 7 keep the three shared models.
Isothermal speed ratios at $a(0^+)=1$, $\mu_B=1200$ MeV: 1.29, 1.27, 1.22, 1.03 at T=0,20,50,100 MeV (Eq. 23: 1.77, 1.75, 1.68, 1.43). On the adiabatic $T(0^-)=20$ MeV, $a(0^+)=0.9$ trajectory: 1.252 (Eq. 23: 1.561).
Figure 6's lower-left panel now plots each weighted integrand relative to the exact one in percent; $D_K$ and the Jacobian cancel, so it is the local rate error along the path.
This is the prefactor choice of the combustion code's analytic approximations, which use $\mu_q=\mu_B(\infty)/3$ (`RMFsolver/phase_velocity.py:1698`); the numerical solver uses $\mu_u^5$ (line 976).
It equals Eq. (23) at $a=0$, and at $T=0$ it is Eq. (23) times $[1+(1+a)^{1/3}+(1-a)^{1/3}]^5/3^5$ (0.243 at $a=1$).
At $T=1$ MeV its errors against the exact rate are +6.5, +24.4, +65.1, +121 percent at $a=0.146,0.5,0.9,1$ (both $\mu_B$), versus +7.4, +38, +178, +812 percent for Eq. (23): smaller, but the composition dependence is still wrong.
The model is plotting-only, in `rate_figures.py`; `validate_rate.py` and the quadrature source hashes are unchanged.
