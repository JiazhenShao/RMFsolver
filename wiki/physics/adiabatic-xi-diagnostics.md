---
summary: Profile xi approaches -1 at high adiabatic speeds; replacing -0.5 fixes the semi-analytical discrepancy, but full-analytical trajectory and integral errors remain.
status: current
updated: 2026-09-29
tags: [physics, adiabatic, analytical, xi, validation]
---

# Why the high-speed analytical contours separate

For the repaired $T(0^+)=0.1$ MeV, $3n_0$-calibrated comparison, both
analytical methods use $\xi=-0.5$. The actual shape factor becomes progressively
more negative as $a(0^+)$ and the numerical speed increase. This explains
essentially all of the semi-analytical discrepancy, but not all of the
full-analytical discrepancy. See [[fixed-interface-analytical-contours]] for
grid provenance and [[analytic-front-speed]] for the structural formula.

## Independent profile check

With the existing diffusive flux $g=-D_Ka'=(j_K-u n_K)/n_B$ and constant $n_B$,

$$\xi_{\rm profile}=\frac{2\int_0^{a(0^+)}g(a)\,da}{g(a(0^+))a(0^+)}-1.$$

All 614 saved numerical profiles have monotonic $a$; their largest relative
density span is $1.58\times10^{-13}$. The normalized flux becomes increasingly
convex, so its area and hence $1+\xi$ decrease. The speed sensitivity is

$$\left.\frac{\partial\log u(0^-)}{\partial\xi}\right|_{I_2,\lambda_n,a(0^+)}
=-\frac{a(0^+)}{2[1+\xi a(0^+)]}.$$

Holding $\xi=-0.5$ is increasingly inaccurate as $a(0^+)\to1$. The analytical
speed is too small at a fixed upstream state, so the equal-speed analytical
contour moves outward. Spatial contour separation is not a relative speed error.

At 540 exactly matched points used in the plot, replacing the semi-analytical
shape factor with the directly measured profile value gives relative speed
errors of 0.0125% median, 0.0282% at the 95th percentile, and 0.0994% maximum.
The profile-derived and speed-fitted semi-analytical xi agree to median absolute
difference 0.000188 and maximum 0.00130 across available matched samples.

## Exact speed-matching value

At the endpoint evaluated at the numerical eigenvalue, invert the shared formula:

$$\xi_{\rm eff}=\frac{1}{a(0^+)}\left[
\frac{2I_2}{\lambda_n^2[1-a(0^+)]u_{\rm numerical}(0^-)^2}-1\right].$$

The stored-speed shortcut freezes the analytical endpoint. Eight live endpoint
checks show its difference from the exact-endpoint fitted xi is at most
$1.44\times10^{-7}$. Inverting a speed by construction is not validation by
itself; agreement with the independently measured profile xi is the useful check.
This does not establish uniqueness of an eigenvalue or provide a predictive
closure without an existing numerical profile.

Representative points on $\theta=46.55^\circ$:

| Numerical speed [m/s] | Profile xi | Semi / numerical speed | Full / numerical speed |
| ---: | ---: | ---: | ---: |
| 120.17 | -0.5353 | 0.9928 | 1.1035 |
| 216.18 | -0.6253 | 0.9598 | 1.0250 |
| 482.40 | -0.7560 | 0.8618 | 0.8152 |
| 1030.10 | -0.8508 | 0.7248 | 0.5662 |
| 2459.78 | -0.9187 | 0.5535 | 0.3195 |
| 5147.66 | -0.9524 | 0.4217 | 0.1778 |

At 1030 m/s the exact-endpoint fitted values are -0.85060 for semi-analytical
and -1.06732 for full analytical, compared with profile xi = -0.85075.
A fitted value below -1 implies negative flux area and cannot be a physical
shape factor. It compensates for other approximation errors.

## Full analytical is different

The semi-analytical method uses the conserved pressure/enthalpy trajectory and
numerical quadrature of the full reaction and diffusion coefficients. The full
analytical method uses a fixed-muB isobar for the interface composition and a
closed-form integral derived with a quadratic trajectory, approximate rate,
and leading Landau-damped diffusion. Constant density does not imply constant
muB: the saved profiles show up to 6.01% variation in muB.

At the same 1030 m/s point, full analytical gives $a(0^+)=0.74909$ instead of
0.80738, and its $I_2$ is 0.83372 times the semi-analytical integral. The error in
the small denominator $1-a(0^+)$ compounds the integral error. Even using the
physical profile xi leaves the full analytical speed at 0.74346 times the
numerical speed; semi-analytical gives 1.00020.

At 216 m/s, the full analytical integral is instead 17.36% larger than the
semi-analytical one. The close 200 m/s contours partly reflect cancellation.
At the smallest speeds full analytical is still about 10-13% high even though
the contour displacement looks small. This analysis isolates the combined
full-analytical errors, not each microphysical approximation individually.

## Evidence

- `local_analysis_results/26-09-29/xi-comparison/README.md`: complete definitions,
  reproducible scripts, Wolfram algebra verification input, environment caveat.
- `matched-state-xi.csv`, `matched-state-decomposition.csv`,
  `velocity-bin-summary.csv`, `endpoint-xi-check.csv` in that directory.
- `xi-speed-diagnostics.png` and `diffusive-flux-shapes.png`: measured trends.
- Live source: `RMFsolver/phase_velocity.py`, `_analytic_velocity_formula_from_endpoint`,
  `_analytic_layer_trajectory`, `_analytic_I2_numerical`.

No production solver, cluster data, manuscript, or notebook was changed by
this diagnosis. Any future predictive xi calibration needs held-out validation;
do not replace -0.5 by another universal constant based on this scan alone.
