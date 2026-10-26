# Second-order endpoint test changes

The revised `tests/test_phase_velocity_analytic_slow.py` passes all 14 tests.
The original 11-test module passed before the implementation change; its source is preserved in `test_phase_velocity_analytic_slow.py.before` and its baseline result in `baseline_analytic_slow.json`.
Only this test module and this new report were edited by the test task.

## Changed existing assertions

Define the report's reference state as `B_one_forth=180`, `nB_0minus=3*n0`, `T_0minus=10`, `T_0plus=5`, and `xi=-0.5`.
The old values below come from the pre-change snapshot or the independent targets saved before the implementation change, rather than from a retained old implementation path.

| Existing assertion or input | Old expected value or relation | New expected value or relation | Tolerance |
|---|---|---|---|
| Quadratic endpoint `muB_0plus` and `muB_inf` | `cbrt(27*pi**2*nB)`, giving `1140.4580278410635` | Unique root of `muB**4-2*H*muB**3+C=0` in `(0,C**0.25)`, giving `1117.1974821689246` | Existing `rtol=2e-13`, unchanged |
| Quadratic endpoint `A_adb` | The quadratic pressure ceiling evaluated at the leading-density chemical potential, giving `0.3228643376202176` | `3/sqrt(2)*sqrt((H-muB)/muB)` at the second-order root, giving `0.5536383787984904` | Existing `rtol=2e-13`, unchanged |
| Quadratic ellipse `a_0plus**2+81*pi**2*T_0plus**2/(2*muB**2)=A_adb**2` | Same ellipse, with leading-density endpoint; warm `a_0plus=0.3107382921355764` | Same ellipse, with second-order endpoint; warm `a_0plus=0.5463598619750096` | Existing `rtol=2e-13`, unchanged |
| Closed `I2` versus independent quadrature | `4.270829447913979e-15` at the reference state, with `T_0plus=5,0` coverage | `1.689040577225153e-14` at the reference state, with `T_0plus=5,1,0` coverage | Tightened from `rtol=2e-10` to `rtol=1e-12` |
| `eos_approximation` | `quadratic_pressure_leading_density` | `quadratic_pressure_second_order_density` | Exact string equality, unchanged |
| Warm-interface ceiling test input at `10*n0` | `T_0plus=0.95*T_inf=88.95553131177294`, computed with `T_inf=93.63740138081363` | `T_0plus=0.95*T_inf=130.8733193794406`, computed with `T_inf=137.7613888204638` | The same success, `A_adb>1`, `a_0plus<1`, and cold rejection assertions |

The warm-interface ceiling test's derived `A_adb` changes from `1.066899312612193` to `1.7590534611639168`, and its warm composition changes from `0.3331392035877887` to `0.5492642672022219`.
Those inequalities continue to test a warm valid interface despite a cold intercept above one.

The retained-density assertion remains `nB_0plus=nB_inf=4*(P_0minus+bag**4)/H`, with the same `rtol=2e-13`; its reference value remains `5566399.98570578`.
The endpoint temperature changes from `18.417131942270593` to `30.93703952716917`; a new direct assertion checks the independent second-order value at `rtol=2e-13`.
The reference baryon flux changes from `0.6742484791762723` to `1.7817232919345092`; there was no literal old-flux assertion to replace in this module.
All other existing tests, conservation thresholds, cold/hot limit tolerances, semi-analytical path checks, and rounded manuscript normalization checks are retained.
An AST comparison against the pre-change snapshot confirms that the only changed existing test methods are `test_quadratic_adiabatic_endpoints_and_independent_integral` and `test_quadratic_warm_interface_survives_ceiling_above_one`.

## Independent new coverage

The endpoint reference uses `numpy.roots` to find all roots of the scaled quartic, then selects its unique physical root.
This differs from the implementation's bracketed scalar root solve.
The fixed numerical validation references were saved before modifying the implementation in `endpoint-before-and-independent-targets.json`.
The four validation states check `muB_inf`, `T_inf`, and `A_adb` at `rtol=1e-6` and speed at `rtol=1e-4`.
They separately compare formatted output with every rounded value in the supplied table, at four decimal places for endpoints and two for speed.
For example, `0.15625335814259514` correctly rounds to `0.1563`, but substituting the rounded value into the `rtol=1e-6` assertion would introduce about `2.98e-4` relative error.
No tolerance was relaxed to accommodate the rounded table.

The four reported signed residuals are recomputed independently as follows, checked in absolute value against `1e-10`, and compared with the returned metadata within absolute `2e-15`:

```text
C = 108*pi**2*(P_0minus + bag**4)
H = h_0minus/nB_0minus
quartic = (muB**4 - 2*H*muB**3 + C)/C
pressure = (C - muB**4 - 18*pi**2*muB**2*T_inf**2)/C
heat = (H - muB - 9*pi**2*T_inf**2/muB)/H
A_eq = 1.5*sqrt(C - muB**4)/muB**2
A_adb_residual = A_adb/A_eq - 1
```

The tests also independently recompute `nB_inf_second_order`, its relative difference from the retained density, and the unexpanded enthalpy-per-baryon residual.
The latter is demonstrably nonzero while the state succeeds, confirming that it is a diagnostic rather than an acceptance gate.
Both metadata fields identify `quadratic_pressure_second_order_density`.

| Validation row | `A_adb` minus full-EOS composition, relative percent | Maximum absolute explicit residual | Unexpanded enthalpy residual |
|---|---:|---:|---:|
| 1 | 0.0242781301% | `6.21725e-15` | `-2.9120079124e-05` |
| 2 | 0.1001955992% | `4.21885e-15` | `-0.0003515508035` |
| 3 | 0.3369590375% | `5.55112e-16` | `-0.0021730146181` |
| 4 | 2.2875050541% | `2.85830e-16` | `-0.0203643895399` |

The first three composition comparisons pass the requested 1% test.
The supplied narrative's 0.3% characterization of row 3 is slightly optimistic; the actual difference is 0.337%.
At row 4, the unexpanded residual is about -2.04%, larger than a few parts in a thousand, and remains diagnostic only.

The independent quadrature uses `T=T_inf*t**3`, substitutes the quadratic ellipse, and cancels the cold power and equilibrium square-root factors algebraically before numerical integration.
Its integrand is finite at both endpoints and does not evaluate the closed-form fractional-power bracket.

| `T_0plus` | Closed `I2` | Independent quadrature | Relative difference |
|---:|---:|---:|---:|
| 0 | `3.23817891495854e-14` | `3.23817891495853e-14` | `3.10862e-15` |
| 1 | `2.3354705622196165e-14` | `2.3354705622196086e-14` | `3.33067e-15` |
| 5 | `1.689040577225153e-14` | `1.689040577225147e-14` | `3.55271e-15` |

The same test independently checks the unchanged weak-rate coefficient, leading diffusion, and speed closure.
Synthetic near-zero-temperature endpoints verify `A_new/A_old_leading_density` approaches `sqrt(3)` within the requested 2%, with successively smaller relative errors `3.31823e-4`, `3.33182e-5`, and `3.33318e-6`.
The old leading-density ceiling is evaluated inline only in this regression guard.
Additional synthetic endpoints test both equality and strict inequality in `C**0.25>=H`, requiring `success=False`, no front, and `status="no_adiabatic_equilibrium"`.

## Execution evidence

```sh
PYTHONPATH=local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:.:tests \
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/nsm-rate-mpl \
./bin/python3 -B -m unittest test_phase_velocity_analytic_slow -v
```

Before the core fix, the revised 14-test module ran in 19.333 seconds and produced nine expected assertion failures across five test methods, with no execution errors.
The old endpoint values, the old EOS label, the missing `sqrt(3)` factor, and the old warm-interface temperature range were each caught.
After the fix and completion of the requested signed-residual checks, all 14 tests passed in 23.116 seconds.
The verification command exited with status zero.
