# Independent read-only review of the second-order endpoint update

Reviewed at 2026-10-26T17:57:00.182167+00:00.
The review made no changes to solver, classifier, manuscript, tests, or saved numerical artifacts.
Source SHA-256: `3c22ccbb039bcc0b7812daeed4c2e233cfdf09c8d5f650a2160c415bc494140b`.
Classifier SHA-256: `8e0a06bea40856a0b14328618b31805dad0c81f4f1191e846182db2852f26ef0`.

## Result

The requested endpoint algebra, physical root branch, classifier signs, conserved density convention, and unchanged closed integral are correct.
The strict relative `A_adb` check has a demonstrated floating-point resolution limitation very close to coexistence.
It fails closed: affected states return a numerical failure instead of accepting a result outside the requested `1e-10` relative tolerance.
The user explicitly requires that relative agreement, so this review does not recommend replacing or loosening the gate.
The four validation states and the resolved contour points examined below are far from the demonstrated cancellation regime.

## Scope and preservation

Comparison with `phase_velocity.py.before` found 116 top-level function/class definitions before and after.
Only `analytic_velocity_adiabatic_slow` changed.
Its nested state function, `nB = h/H` assignment, and full source tail beginning at the weak-rate call are byte-identical; this tail includes the finite cold integral, warm `expm1` evaluation, diffusion calls, and speed evaluation.
The isothermal and semi-analytical APIs, spatial solvers, microphysics helpers, and constants are unchanged.
The `cold_physics.py` snapshot diff changes only the classifier documentation and its analytical second-order endpoint/composition margins; the semi-analytical classification branch is unchanged.

All 10,015 entries in `protected-artifacts-before.json` match their recorded SHA-256 hashes, with no missing or modified protected file.
The hash audit includes the protected notebook and saved contour payloads; no notebook content was read into the review.
`Latex_writting/Main.blg` is not in the manifest; its previously reported regeneration is an untracked build-log side effect, not a protected-file discrepancy.

## Algebra, root, and residual signs

Eliminating the retained heat relation from the pressure equation gives exactly:

```text
T_inf^2 = muB*(H-muB)/(9*pi^2)
muB^4 - 2*H*muB^3 + C = 0
C = 108*pi^2*(P_0minus + bag^4)
A_adb^2 = (9/2)*(H/muB - 1)
```

For positive `H` and `C`, the code requires `C^(1/4) < H`.
The quartic is positive at zero and negative at `C^(1/4)`; its derivative is `-2*(3*H-2*muB)*muB^2`, strictly negative throughout this bracket.
Therefore the bracket contains exactly one root, with `0 < muB < C^(1/4) < H`, guaranteeing positive equilibrium temperature.
The other positive quartic branch lies outside the physical interval and is excluded by the bracket.

The cold classifier's lower margin `H-C^(1/4)` has the correct coexistence sign.
Its upper margin `2/9-(H/muB-1)` is positive exactly when the cold composition ceiling is below one.
The nonpositive-lower-margin branch preserves stable-matter exclusion without solving a nonexistent positive-temperature endpoint.

Quartic, pressure, and heat residual signs match the displayed defining equations.
The `A_adb` residual is the requested relative `A/A_eq-1`.
The second-order density diagnostic is not used to replace the conserved density, and the unexpanded heat-ratio discrepancy is fourth order and negative:

```text
h/(nB_second_order*H) - 1 = -(H-muB)^2/H^2
```

Wolfram independently verified these identities.
Input:

```text
p = muB^4/(108 Pi^2)+muB^2 T^2/6; n = D[p,muB];
{Expand[(C-muB^4-18 Pi^2 muB^2 T^2)/.T^2->muB (H-muB)/(9 Pi^2)],
 Factor[D[muB^4-2 H muB^3+C,muB]],
 FullSimplify[(4 p/(n H)-1)/.T^2->muB (H-muB)/(9 Pi^2)],
 FullSimplify[((9/2)(H/muB-1)<1)==(2/9-(H/muB-1)>0),Assumptions->{H>muB>0}]}
```

Returned result:

```text
{C - 2*H*muB^3 + muB^4,
 -2*(3*H - 2*muB)*muB^2,
 -((H - muB)^2/H^2),
 True}
```

The tool issued undefined-symbol warnings for the deliberately symbolic `H` and `T`; all four expressions evaluated.

## Near-coexistence floating-point limitation

The equation-based `A_eq` subtracts `C-muB^4`.
This difference vanishes at coexistence, so roundoff in the large terms is amplified in a relative comparison of the small composition ceilings.
A synthetic upstream state supplied through a mocked upstream helper reproduces the limitation without involving nuclear-state recovery:

```text
H = 1200
muB = H*(1-1e-7)
C = 2*H*muB^3-muB^4
bag = 180
P_0minus = C/(108*pi^2)-bag^4
nB_0minus = 3*0.16*MeV_fm^3
h_0minus = H*nB_0minus
T_0plus = 0
xi = -0.85
```

The corresponding physical equilibrium temperature is `0.04026336765773861` MeV.
The solver returns `success=False`, message `Second-order quadratic endpoint failed residual checks`, and the thermodynamic classifier returns `interior`.
Measured residuals were:

| Residual | Value |
|---|---:|
| Quartic | 3.5321278042264866e-16 |
| Pressure | 2.8956057394319145e-16 |
| Heat | -1.1293772630057339e-23 |
| Relative composition ceiling | -7.239017163485073e-10 |

The first three checks pass; the relative composition-ceiling check exceeds its required tolerance.
Repeating the construction at `1-muB/H` values from `1e-8` through `1e-14` also produced failures in the sampled cases.
The boundary is not a sharp temperature cutoff: accidental cancellation can let individual smaller gaps pass, as seen at `3e-8`.
No accepted result with an out-of-tolerance relative ceiling residual was observed.

## Separation of the requested cases from this limitation

The four validation outputs were independently recomputed from their stored pressure, enthalpy, chemical potential, and temperature.
All satisfy the physical root ordering and all four required residual checks, while retaining `nB=4*(P+bag^4)/H`.

| Validation case | T_inf (MeV) | 1-muB/H | Absolute relative A residual |
|---|---:|---:|---:|
| 1 | 9.30658354517 | 0.00539630235662 | 5.32907e-15 |
| 2 | 17.3274044868 | 0.0187496880914 | 4.21885e-15 |
| 3 | 25.5555603322 | 0.0466156048776 | 1.11022e-15 |
| 4 | 54.7745396896 | 0.142703852576 | 0 |

The currently saved new contour snapshot has `scan['complete']=False`, 16 rays, and 112 resolved points at target levels from 50 through 5000 m/s.
Across those resolved points, the minimum equilibrium temperature is 10.5356130862 MeV, the minimum `1-muB/H` is 0.0072697038874, and the maximum absolute relative `A_adb` residual is 1.08801856413e-14.
These gaps exceed the demonstrated `1e-7` example by more than four orders of magnitude.
The point-record audit found {'accepted': 853, 'excluded': 311} and 0 residual-check failures.
This is a snapshot of the ongoing scan, not a claim that the full contour scan has finished.

## Review disposition

No algebraic, sign, scope-preservation, or protected-artifact blocker was found.
Keep the requested relative tolerance unchanged and retain the explicitly documented fail-closed limitation near coexistence.
A final full-scan audit should repeat the minimum-gap and residual-failure summary after the scan completes.
