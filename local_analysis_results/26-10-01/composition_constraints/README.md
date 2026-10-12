# Composition-constraint plot audit

Assumptions: massless noninteracting u,d,s quarks, no electrons, fixed
muB = 1200 MeV, T = 0 in the main plots. The supplied email does not specify
temperature or masses; its neutral screenshot is consistent with this limit.
Bag and gluon terms do not enter the flavor densities or the fraction.

## Revised simple plot

The revised PI simple screenshot supplied on 2026-10-01 resolves the original
discrepancy: its curve agrees with the exact massless result. Ten independently
digitized samples differ by at most 0.00845 in a, consistent with screenshot
resolution and axis calibration.

The updated standalone comparison is `simple_revised_comparison.png` and
`simple_revised_comparison.pdf`. Its digitized points are saved separately as
`revised_simple_screenshot_samples.csv`; the original audit files are preserved.
Reproduce the updated figure from the saved samples with:

```bash
env MPLCONFIGDIR=/private/tmp/nsm-matplotlib ./bin/python3 -B local_analysis_results/26-10-01/composition_constraints/plot_revised_simple.py
```

## Original screenshot verdict (simple discrepancy now resolved)

- The neutral screenshot agrees with the exact massless calculation within
  screenshot digitization precision (a few thousandths).
- The simple, muQ = 0 screenshot is not reproduced with the project's definitions.
  At muK = 500 MeV it shows about 1.42, whereas the exact result is 1.18969.
  Its a = 1 crossing appears near 240 MeV; the exact crossing is 344.12777 MeV.
  The email's approximate 200 MeV crossing is also inconsistent with these definitions.
- The neutral a = 1 endpoint is finite: muK = 669.007999 MeV.
  The apparent saturation is a cubic flattening at T = 0, not an infinite-muK limit.
- The email writes ns + nd = nu. Quark-only electrical neutrality is instead
  ns + nd = 2 nu. Its plotted limiting fraction is consistent with the latter.
  This audit does not assume the typo appears in the PI's calculation.

## Common definitions

```text
n_i = mu_i^3/pi^2 + mu_i*T^2
nB = (nu + nd + ns)/3
nK = (nd - ns)/2
a = nK/nB = 3*(nd - ns)/(2*(nu + nd + ns))
muK = mud - mus
muB = muu + mud + mus
muq = muB/3

muu = muq + 2*muQ/3
mud = muq + muK/2 - muQ/3
mus = muq - muK/2 - muQ/3
```

The densities are net densities. At T = 0 and nonnegative chemical potentials
they are ordinary quark densities. At T > 0, zero net strange density does not
mean absence of thermally produced strange quark-antiquark pairs.

## Constraint 1: neutral quark matter

```text
nd + ns = 2*nu
nB = nu
nu = nB
nd = nB*(1+a)
ns = nB*(1-a)
```

At T = 0, invert the cubic densities:

```text
mud = muu*(1+a)^(1/3)
mus = muu*(1-a)^(1/3)
muu = muB/[1+(1+a)^(1/3)+(1-a)^(1/3)]

muK(a) = muB*[(1+a)^(1/3)-(1-a)^(1/3)]
         /[1+(1+a)^(1/3)+(1-a)^(1/3)]
```

This is an exact parameterization over 0 <= a <= 1. Alternatively solve the
neutrality equation in muQ at specified muK. The script does both independently.

```text
0 = 4*muQ^3 + 4*muB*muQ^2
    + (4*muB^2 + 3*muK^2 + 12*pi^2*T^2)*muQ - 3*muB*muK^2

muK(a=1,T=0) = 2^(1/3)*muB/[1+2^(1/3)] = 669.007999 MeV
(muu,mud,mus) = (530.992001,669.007999,0) MeV
muQ = 196.488001 MeV
nu:nd:ns = 1:2:0
```

Because 1-a = ns/nu = (mus/muu)^3 at zero temperature, the curve has a
horizontal tangent at a = 1. Continuing beyond this endpoint in the massless
net-density EOS requires negative net strange density. The extended plot
therefore stops the neutral curve there. At finite T the endpoint shifts and
the approach is ultimately linear in mus because ns includes mus*T^2.

| Target a | Neutral muK [MeV], T = 0 |
|---|---:|
| 0.5 | 143.3482 |
| 0.9 | 343.8328 |
| 0.99 | 505.7489 |
| 0.999 | 589.7558 |
| 1 | 669.0080 |

## Constraint 2: symmetric chemical potentials

```text
mud + mus = 2*muu  <=>  muQ = 0
(muu,mud,mus) = (muq,muq+muK/2,muq-muK/2)

nK = [3*muq^2*muK/2 + muK^3/8]/pi^2 + muK*T^2/2
nB = [muq^3 + muq*muK^2/2]/pi^2 + muq*T^2

a(muK,T=0) = [3*muq^2*muK/2 + muK^3/8]
             /[muq^3 + muq*muK^2/2]
```

For muq = 400 MeV, a = 1 yields:

```text
muK = 344.127767 MeV
(muu,mud,mus) = (400,572.063884,227.936116) MeV
nu:nd:ns = 1:2.925186876:0.185037375
(2*nu-nd-ns)/(3*nB) = -0.27011282
```

The a = 1 condition alone, for either EOS, is nd = 2*nu + 5*ns. Only when
neutrality is also imposed does this force ns = 0. In the simple model ns = 0
instead occurs at muK = 800 MeV with nu:nd:ns = 1:8:0 and a = 4/3.

At small muK both models have a = 9*muK/(2*muB) + O(muK^3).
Consequently their tangents at the origin must agree. Large differences in
initial slope cannot be attributed to the neutrality correction.

The finite-T simple-model expression was checked symbolically: a decreases
with T for muK < sqrt(8/9)*muB, which covers the entire screenshot range.
Finite temperature cannot raise the predicted 1.18969 to the displayed 1.42.
Without the plotting source, the cause (normalization, parameter values,
definition, or implementation) cannot be identified uniquely.

Both a = 1 states are outside the small-muK/muq regime: the ratios are
1.67252 for the neutral model and 0.86032 for the simple model. A large
muK here is an allowed composition endpoint of the model, not by itself
evidence that the EOS state is unphysical. It does invalidate a small-imbalance
approximation there. These are fixed-muB curves, not conserved-flux front trajectories.

## Reproduction and validation

Files:

- `comparison_with_screenshots.png` and `.pdf`: the two exact curves with
  approximate screenshot samples overlaid.
- `extended_endpoints.png` and `.pdf`: both endpoints on a wider horizontal range.
- `curves.csv`: exact zero-temperature curves.
- `temperature_checks.csv`: T = 0, 0.5, 10, 50 MeV samples.
- `verification.json`: endpoint states and independent consistency checks.
- `verify.wl`: Wolfram input verifying the formula and numerical endpoints.

Run from the repository root with a working SciPy environment:

```bash
python3 -B local_analysis_results/26-10-01/composition_constraints/reproduce.py
```

On this machine, the existing separate SciPy installation avoids the damaged
project SciPy binary without modifying packages:

```bash
PYTHONPATH=/private/tmp/nsm-scipy-verification MPLCONFIGDIR=/private/tmp/nsm-matplotlib ./bin/python3 -B local_analysis_results/26-10-01/composition_constraints/reproduce.py
```

The standalone neutrality calculation and live `SQMsolver._quark_uds_state`
agree to 1.23e-15 in a over 28 states. The explicit simple-model expression,
the exact inverse neutral expression, and normalized neutrality residuals
also pass the script's assertions. Wolfram independently returns the same
endpoint values. Both generated figures were visually inspected.

Screenshot samples are approximate measurements, not the PI's original data.
They are sampled from blue curve pixels using the following pixel calibrations
(coordinates measured from each image's upper-left corner):

```text
neutral screenshot: x(muK=0)=32, x(muK=500)=435, y(a=0)=269, y(a=1)=34
simple screenshot:  x(muK=0)=28, x(muK=500)=432, y(a=0)=305, y(a=1)=142
```

The CSV retains these samples so replotting does not depend on temporary
clipboard files. Pixel rounding/line thickness introduces uncertainty of
roughly 0.003-0.01 in a, far below the simple-curve discrepancy.
