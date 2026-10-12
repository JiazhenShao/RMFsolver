# Why the high-speed adiabatic contours separate

Scope: T(0+) = 0.1 MeV, the 3n0 bag calibration, massless quarks, and the
734-point analytical comparison saved on 2026-09-29. No solver or cluster
payload is modified by this analysis.

## Main findings

1. Both approximations use xi = -0.5. The actual numerical profile shape is
   state dependent, with xi progressing from about -0.42 to -0.95.
2. This accounts for essentially all of the semi-analytical discrepancy.
   Substitution of the measured profile xi into the saved semi-analytical
   closure gives speed errors of 0.0125% median, 0.0282% at the 95th percentile,
   and 0.0994% maximum on the 540 points used by the comparison plot.
3. The full analytical expression additionally approximates the thermodynamic
   trajectory and reaction-diffusion integral. Its required fitted xi drops
   below -1 at high speed. Such a fit cannot represent the positive diffusive
   flux area; it is compensating for errors beyond the shape approximation.
4. The apparent low-speed contour agreement is not uniform pointwise accuracy:
   full analytical speeds are about 10-13% too high at the smallest speeds.
   Especially good agreement around 200 m/s partly reflects error cancellation.

## Definitions and checks

The shared structural closure is

```text
u(0-)^2 = 2*I2 / (lambda_n^2 * (1-a(0+)) * (1+xi*a(0+))).
```

At fixed thermodynamic endpoint its algebraic inverse is

```text
xi_eff = (2*I2/(lambda_n^2*(1-a(0+))*u_numerical(0-)^2) - 1)/a(0+).
```

For the stored analytical solution, the equivalent frozen-endpoint shortcut is

```text
xi_eff = ((u_approx/u_numerical)^2*(1-0.5*a(0+)) - 1)/a(0+).
```

Because the endpoint formally depends on the trial speed, `check_endpoints.py`
independently solves the endpoint at the numerical eigenvalue and evaluates
each live closure there. On eight representative states, the shortcut differs
from that exact-endpoint fitted xi by at most 1.44e-7. Substitution into the
closure recovers the numerical speed to floating-point accuracy, as required
algebraically. No new BVP was solved, and no uniqueness claim follows from
this algebraic residual check.

The independent profile measurement uses the existing diffusive flux notation:

```text
g(a) = (jK - u*nK)/nB = -D_K*da/dx   (constant nB),
xi_profile = 2*integral(g(a), a=0..a(0+))/(g(a(0+))*a(0+)) - 1.
```

All 614 saved profiles are monotonic in a. The largest relative density span
is 1.58e-13. The saved tail is completed by the exact equilibrium point (a,g)=(0,0).
The profile-derived xi and speed-fitted semi-analytical xi agree to median
absolute difference 0.000188 and maximum 0.00130 across available matched
samples. This is the substantive check, unlike fitting a speed by construction.

The formula's sensitivity at fixed I2, lambda_n and a(0+) is

```text
d(log(u))/d(xi) = -a(0+) / (2*(1+xi*a(0+))).
```

It increases strongly when a approaches 1 and xi approaches -1. The normalized
diffusive-flux area shrinks as profiles become more convex. Holding xi=-0.5
then overestimates that area and underestimates the speed. The high-speed
analytical contours consequently sit farther outward in the upstream plane.

## Representative ray: theta index 15 (46.55 degrees)

These are evaluations at identical upstream points, not distances between
equal-speed contour lines. All speeds are in m/s.

| Numerical speed | a(0+) numerical | Profile xi | Semi / numerical | Full / numerical |
| ---: | ---: | ---: | ---: | ---: |
| 39.21 | 0.1531 | -0.4207 | 1.0066 | 1.1306 |
| 120.17 | 0.3408 | -0.5353 | 0.9928 | 1.1035 |
| 216.18 | 0.4793 | -0.6253 | 0.9598 | 1.0250 |
| 482.40 | 0.6692 | -0.7560 | 0.8618 | 0.8152 |
| 1030.10 | 0.8074 | -0.8508 | 0.7248 | 0.5662 |
| 2459.78 | 0.9061 | -0.9187 | 0.5535 | 0.3195 |
| 5147.66 | 0.9520 | -0.9524 | 0.4217 | 0.1778 |

At 1030 m/s, exact-endpoint fitted xi is -0.85060 for the semi-analytical
closure, compared with -0.85075 from the actual profile. For the full
analytical closure it is -1.06732, outside the physical profile range.

## Additional full-analytical errors

The semi-analytical method integrates the full reaction and diffusion
coefficients along the conserved pressure/enthalpy-per-baryon trajectory.
The full analytical method instead uses a fixed-muB isobar for its interface
composition, and the closed-form I2 derived with a quadratic trajectory,
approximate rate and leading Landau-damped diffusion coefficient.

At the representative 1030 m/s point:

- The numerical/semi-analytical interface fraction is 0.80738, whereas the full
  analytical interface fraction is 0.74909.
- I2(full)/I2(semi) = 0.83372.
- The composition-denominator contribution to the squared-speed ratio is
  (1-0.80738)/(1-0.74909) = 0.76768.
- The shape-denominator contribution is 0.95340; the density contribution is
  indistinguishable from 1 at the displayed precision.
- Their product gives the full/semi squared-speed ratio. The same factorization
  is checked to relative tolerance 1e-12 at all 540 plotted matched points.
- Even substituting the actual profile xi leaves the full analytical speed
  at 0.74346 times the numerical speed, versus 1.00020 for semi-analytical.

Across the available numerical profiles, muB changes by up to 6.01%, despite
nearly constant density. This matters for the inferred composition and the
small denominator 1-a(0+). Constancy of nB is not evidence that muB is constant.
The present comparison isolates the combined full-analytical approximations;
it does not separately attribute the I2 error to each microphysical term.

At the 216 m/s example, full I2 is instead 17.36% larger than semi-analytical
I2. This partly offsets its other errors, explaining the close 200 m/s curves.

## Interpretation and limits

There is no single exact xi for the whole contour. The directly measured xi
is a post-processing diagnostic, not a new predictive closure: it requires
the BVP solution already to exist. A future calibration should predict xi
from thermodynamic/compositional inputs and be validated on held-out states,
temperatures and bag calibrations. Do not fit full-analytical xi below -1
and interpret it as a physical flux shape.

## Files and reproduction

- `xi-speed-diagnostics.png` / `.pdf`: same-state speed ratios and fitted xi.
  Dots show individual points; solid lines show velocity-bin medians, not a
  continuous fixed-theta trajectory. Black dashes show profile xi medians.
- `diffusive-flux-shapes.png` / `.pdf`: directly measured normalized profiles.
- `matched-state-xi.csv`: 614 numerical states and matched analytical diagnostics.
- `matched-state-decomposition.csv`: 540 plotted states and multiplicative factors.
- `velocity-bin-summary.csv`: medians in explicitly recorded velocity bins.
- `endpoint-xi-check.csv`: eight states, both live endpoint closures.

```bash
PYTHONPATH=/private/tmp/nsm-scipy-verification ./bin/python3 local_analysis_results/26-09-29/xi-comparison/analyze_xi.py
PYTHONPATH=/private/tmp/nsm-scipy-verification MPLCONFIGDIR=/private/tmp/nsm-matplotlib ./bin/python3 local_analysis_results/26-09-29/xi-comparison/check_endpoints.py
PYTHONPATH=/private/tmp/nsm-scipy-verification MPLCONFIGDIR=/private/tmp/nsm-matplotlib ./bin/python3 local_analysis_results/26-09-29/xi-comparison/plot_diagnostics.py
```

The temporary PYTHONPATH is only an environment workaround for a broken local
SciPy 1.15.2 dynamic-library installation; the diagnostic used SciPy 1.18.1.
It can be omitted in a functioning project environment.

Wolfram verified the inverse and sensitivity with the following input
(a denotes the existing a(0+), u the numerical speed, v the approximation):

```text
{
 Solve[u^2 == 2 I2/(lambda^2 (1-a) (1+xi a)), xi],
 Solve[v^2/u^2 == (1+xi a)/(1+xi0 a), xi],
 FullSimplify[D[Log[Sqrt[2 I2/(lambda^2 (1-a) (1+xi a))]], xi],
  Assumptions -> {I2 > 0, lambda > 0, 0 < a < 1, -1 < xi < 1}]
}
```

The returned expressions are algebraically equivalent to those above.
Authoritative code: `RMFsolver/phase_velocity.py`, functions
`_analytic_velocity_formula_from_endpoint`, `_analytic_layer_trajectory`,
`_analytic_I2_numerical`, and `_solve_analytic_inf_endpoint_for_u_0minus`.
Input provenance is preserved in `analyze_xi.py`; saved profile source paths
come directly from the consolidated numerical payload.
