---
summary: The quadratic adiabatic endpoint now matches pressure and heat consistently to second order; the old leading-density endpoint underestimated the ceiling by roughly sqrt(3).
status: current
updated: 2026-10-26
tags: [solver, analytic, isothermal, adiabatic, verification]
---

# Analytical slow-front solvers

The three new APIs in `RMFsolver/phase_velocity.py` determine thermodynamics independently of baryon flux and then evaluate the integrated speed closure.
They preserve the original analytical and numerical comparison methods.
See [[analytic-slow-front-audit]] for the historical audit, [[slow-front-solvers]] for spatial BVPs, and [[phase-velocity-overview]] for the full API map.

| API | Thermodynamic prescription | Diffusion and integral |
|---|---|---|
| `analytic_velocity_isothermal_slow` | Static full-EOS endpoints at fixed pressure and temperature; prescribed local fraction or exact automatic ceiling | Leading Landau term, frozen coefficients, closed cubic integral |
| `analytic_velocity_adiabatic_slow` | Quadratic pressure and second-order equilibrium heat matching; conserved density from exact massless enthalpy | Quadratic ceiling/ellipse, leading Landau term, closed fractional-power integral |
| `semi_analytic_velocity_adiabatic_slow` | Full EOS curve at fixed pressure and enthalpy per baryon | Full two-term diffusion, local modified Madsen rate, regular numerical integral |

All three accept upstream temperature, upstream baryon density and bag parameter as their first three arguments.
The two adiabatic APIs require keyword `T_0plus`, including exactly zero; the isothermal API accepts `a_0plus=np.nan` for automatic selection.
Defaults are `xi=0`, `ms=0`, `param=paraQMCRMF3`, and `NM_type="PNM"`; only the massless PNM reduction is implemented here.
Both mass branches and the existing nuclear-composition choices remain available in the numerical APIs.
The semi-analytical integral uses `path_rtol=1e-6`; isothermal phase classification uses `delta_muB_tol=1e-6` MeV.
Input density is in MeV cubed, temperature and bag fourth root in MeV; returned `u_0minus` is the slow velocity in units of light speed and `velocity_m_s` is its direct conversion to metres per second.

## Isothermal recipe

Use the exact neutral EOS for the two static endpoints and for the automatic composition ceiling:

$$
\mu_B(0^-)=\mu_B(0^+)+a(0^+)\mu_K(0^+).
$$

This is not the displayed Taylor approximation to the ceiling.
Prescribed composition retains the comparison API's semantics: it does not silently enforce the automatic ceiling.
Freeze transport and the analytical weak coefficient at the interface chemical potential, while the density ratio uses the actual equilibrium density, `lambda_n=nB_0minus/nB_inf`.
The two full-EOS endpoint densities need not coincide; both are returned.
This preserves the prior frozen-background convention, now stated explicitly in the manuscript recipe.

Only the leading Landau contribution to inverse diffusion enters this new analytical estimate, matching the diffusion approximation in manuscript Eq. (D12).
Neither shared full-diffusion helper changes.
The cubic-plus-linear weak-source reduction gives a closed integral; it is not a full-EOS integral of the local rate.
Positive temperature is required.
Stable nuclear matter and coexistence return zero speed with `front_exists=False`; a saturated automatic ceiling returns `composition_ceiling_saturated` without a finite speed.

## Two different adiabatic approximations

The full analytical method uses the manuscript quadratic pressure, `a=9*muK/(2*muB)`, and exact massless `h=4*(P+bag^4)`.
As corrected on 2026-10-26, pressure and equilibrium enthalpy per baryon are matched at the same second order; the leading density alone cannot determine the heat balance.
The constant conserved density remains `nB=4*(P+bag^4)/H`, where `H=h_0minus/nB_0minus`.
The physical endpoint is the unique root in `(0,C**0.25)` of `muB**4-2*H*muB**3+C=0`, where `C=108*pi**2*(P+bag**4)`, provided `C**0.25<H`.
It gives `T_inf**2=muB*(H-muB)/(9*pi**2)` and `A_adb=3/sqrt(2)*sqrt((H-muB)/muB)`.
The chemical potential is then frozen along the manuscript ellipse.
`A_adb` is its quadratic zero-temperature intercept, not a full-EOS ceiling.
Warm interfaces with `A_adb>1` remain possible provided their actual fraction lies below one; the corresponding cold interface is rejected.
The integral is evaluated with `expm1` near the hot endpoint and an explicit finite value at the exact cold endpoint.
Residuals are labelled `conservation_eos="quadratic_pressure_second_order_density"`.
Signed quartic, pressure, second-order heat, and Eq. (a-max) agreement residuals each must satisfy `abs(residual)<=1e-10`; both endpoint pressures also retain the `1e-8` check.
The derivative density `nB_inf_second_order` and its difference from the conserved density are reported.
The unexpanded enthalpy-ratio residual is fourth order and diagnostic only; it reaches 2.04% in the highest-composition validation state.

⚠ **Superseded endpoint:** the original leading-density recipe used `muB=cbrt(27*pi**2*nB)` and a denominator `18*pi**2` in the temperature relation.
Dropping `muB*T_inf**2/3` from the pressure derivative removes a term of the same order as the heat balance, raising the leading chemical potential by `3*pi**2*T_inf**2/muB`.
Together these errors suppress the temperature and ceiling by approximately `1/sqrt(3)` in the small-temperature limit.
The closed integral, flux density ratio, rate normalization, diffusion coefficient, and displayed 82 m/s formula were not the cause and remain unchanged.

The semi-analytical method instead reuses the full constrained EOS path of the numerical slow-front solver, without solving a spatial BVP.
It retains changing chemical potential and the full EOS temperature-composition curve.
For massless quarks the density is constant, and the same integrated closure uses

```text
I2 = integral(Gamma_K dr)/nB_inf^2
   = path.r_scale*path.rate.integrate(0,1)/path.inf["nB"]^2
```

The precomputed diffusion integral is regularized in temperature to the one-third power at the cold endpoint.
Exactly zero interface temperature is included without a floor; the reported local diffusion is infinite and inverse diffusion is zero there.
`thermodynamic_path` contains 65 samples ordered from interface to equilibrium, including direct EOS checks and local rate/diffusion diagnostics.
It is a state-space path, not a spatial profile.

## Metadata and domain

Success and returned numerical/physical failure dictionaries identify `hydrodynamic_model`, `closure`, `eos_approximation`, `diffusion_approximation` and `integral_approximation`.
They retain legacy `rate_model="exact_nonleptonic"` together with `rate_approximation="madsen_muB_over_3"`; neither denotes phase-space integration.
Shared constants remain `alpha_s=0.4` and exact HP93 `h_D`.
Unsupported mass/composition, nonfinite inputs and invalid basic domains raise `ValueError`; unavailable physical states return `success=False` and a message.
An invalid integrated-speed denominator returns `invalid_speed_denominator` separately from thermodynamic failure.
The manuscript interpretation of the shape factor assumes a finite physical diffusive-flux profile; finite input and a positive algebraic denominator alone do not establish that interpretation.

## Historical verification and comparison values — endpoint superseded

The full-analytical values in this section are the pre-correction results, retained as history rather than current benchmarks.

The new unittest module covers independent unrounded isothermal speed/integral evaluation, both rounded manuscript speed normalizations, the exact automatic ceiling, explicit-interface equivalence, direct quadratic endpoints, independent warm/cold quadrature, direct full-EOS path conservation, local rate/diffusion, cold-limit and path refinement, failure metadata, zero upstream temperature, near-hot cancellation, and warm interfaces with a quadratic ceiling above one.
Pressure and enthalpy-per-baryon checks retain a relative tolerance of `1e-8`; no existing conservation tolerance was relaxed.

All 11 new unittest methods pass through the isolated SciPy runtime.
The selected existing regression suite has 60 tests: 55 pass, with the same five failures reproduced before and after implementation.
Four synthetic analytical cases raise the existing `SlowFrontNoSolution` with the same 42.0272 MeV enthalpy gap: `test_analytic_velocity_recovers_synthetic_equilibrium_and_uses_eos_density`, `test_analytic_velocity_lte_formula_and_default_mode`, `test_analytic_lte_domain_is_checked_with_a_0plus_lte_not_A`, and `test_analytic_A_matches_exact_eos_nK_over_nB`.
The remaining pre-existing failure is `test_analytic_velocity_validates_xi_endpoint_and_density_domain`, whose old ceiling-domain expectation does not raise.
The passing selection includes both fixed-temperature API checks, numerical-I2 regression, 19 isothermal analytical tests, 13 ceiling tests, warm isothermal and both energy-conserving BVP regressions, rate/diffusion checks and compatibility metadata checks.
The existing export-list test was updated only to add the three approved API names; existing physical reference values were not changed.
An AST comparison confirms every pre-existing function/class definition in the module is unchanged.
The manuscript compiles to a 17-page PDF in a temporary output directory, and all 65 existing numbered equation blocks are byte-for-byte unchanged.
The unresolved `sec:XX` reference and 5.09543-point equation-width warning reproduce in the original manuscript; tracked build outputs were preserved.

At PNM, upstream temperature 10 MeV, upstream density three times saturation density, bag fourth root 180 MeV, and `xi=-0.5`, the isothermal comparison uses interface fraction 0.6 and the adiabatic comparison uses interface temperature 5 MeV:

| Method | Baryon flux in module natural units |
|---|---:|
| Retained isothermal analytical | 2.07871141868808 |
| New leading-diffusion isothermal analytical | 2.50209202700153 |
| Isothermal slow BVP | 2.01742521308527 |
| Retained hybrid full analytical adiabatic | 1.66746293359299 |
| New consistently quadratic full analytical adiabatic | 0.674248479176272 |
| Retained semi-analytical adiabatic | 1.46086188958050 |
| New semi-analytical adiabatic | 1.46077132581715 |
| Adiabatic slow BVP | 1.40708418628695 |

The full quadratic endpoint has equilibrium temperature 18.4171319423 MeV and quadratic ceiling 0.3228643376; the full EOS has equilibrium temperature 31.8240303932 MeV.
The original interpretation as a consistent thermodynamic approximation was superseded on 2026-10-26: the leading-density endpoint had inconsistent expansion order, rather than a relativistic correction.
The isothermal change is the intended leading-only diffusion choice; moving-pressure corrections at this state are negligible.
Tightening the warm semi-analytical path tolerance from `1e-6` to `2.5e-7` changes flux by about six parts in one hundred billion; exact-cold refinement also passes the `1e-6` relative stability test.
These are new comparison values, not replacements for historical artifacts or general accuracy bounds.

## ⚠ Historical independent comparison (2026-10-23; endpoint superseded)

Formulas match the manuscript: Eq. (speed_general), the cubic isothermal $I_2$, the ellipse, $A_{\rm adb}$ and the adiabatic bracket; the closed adiabatic $I_2$ equals 40-digit quadrature in $T$ to $2\times10^{-15}$, including $T(0^+)=0$.
Relativity is irrelevant: old relativistic and new slow BVPs agree to 7 digits (three isothermal states, one adiabatic), old and new semi-analytical agree to $\le7\times10^{-5}$, and the old relativistic isothermal analytic equals the new branch with full diffusion to 7 digits.
The claim that every old-versus-new analytical difference was an approximation choice was superseded by the second-order endpoint diagnosis.

Speed relative to the slow BVP, $B^{1/4}=180$ MeV, PNM, $\xi=-0.5$:

| $T(0^-)$, $n_B(0^-)$, case | New iso (Landau $D_K$) | Old iso (full $D_K$) | New quadratic adb | Old hybrid adb | Semi-analytic adb |
|---|---|---|---|---|---|
| 10 MeV, $3n_0$, $A_{\rm iso}$ / $T(0^+)=5$ | +25.0% | +3.9% | −52.1% | +18.5% | +3.8% |
| 10 MeV, $3n_0$, $a=0.6$ / $T(0^+)=1$ | +24.0% | +3.0% | −53.2% | +7.9% | −2.6% |
| 20 MeV, $3.5n_0$, $A_{\rm iso}$ / $T(0^+)=5$ | +32.9% | +6.3% | −60.6% | −0.1% | −3.8% |
| 5 MeV, $3n_0$, $A_{\rm iso}$ / $T(0^+)=1$ | +17.2% | +0.6% | −52.8% | +8.5% | −2.3% |

⚠ The quadratic adiabatic endpoint recipe uses the leading density $\mu_B^3/27\pi^2$, which drops $\mu_BT^2/3$, the term that carries the heat in $h/n_B$.
It gives $T(\infty)\approx18$ MeV against the full EOS's 31.8 MeV (25.8 against 44.4 at 20 MeV, $3.5n_0$), and $A_{\rm adb}\approx0.32$ against a full-EOS $a(0^+)\approx0.54$.
With the consistent density $\partial P_{\rm quad}/\partial\mu_B$ the endpoint is close (33.1 MeV, $A=0.596$), but the speed then overshoots by 34–95%, so the quadratic layer itself is crude at $a\gtrsim0.5$.
Full-EOS endpoints with the quadratic layer (the old hybrid) stay within −0.1% to +18.5%.
Scripts and logs: `new_paper_calculations/26-10-23/analytic_slow_comparison/`.

## Second-order validation, 2026-10-26

At exact `T_0plus=0`, `xi=-0.85`, massless QMC-RMF3 PNM, and `B_one_forth=189.1565957288247` MeV:

| Upstream temperature [MeV] | Density / saturation | Corrected chemical potential [MeV] | Equilibrium temperature [MeV] | Quadratic ceiling | Old / corrected / semi / BVP speed [m/s] |
|---:|---:|---:|---:|---:|---:|
| 8.86099275 | 3.02686243 | 1190.798571 | 9.30658355 | 0.156253358 | 19.5901 / 39.6547 / 35.9614 / 34.8871 |
| 27.83373842 | 2.96875000 | 1181.401509 | 17.32740449 | 0.293233360 | 44.5890 / 96.3749 / 85.9899 / 81.6343 |
| 70.67359438 | 1.43448804 | 1089.243289 | 25.55556033 | 0.469070261 | 126.1055 / 304.4747 / 267.9588 / 250.4187 |
| 10.15113868 | 4.50018170 | 1265.311488 | 54.77453969 | 0.865483255 | 289.9968 / 1372.3870 / 1054.4172 / unavailable |

The first three matched-temperature BVP values were rerun; the fourth was reported unavailable and was not retried.
The first three ceiling discrepancies from the full EOS are below 0.34%, satisfying the requested 1% regression bound.
The remaining 10–30% speed excess over the semi-analytical method reflects the retained quadratic/frozen-rate/leading-diffusion approximations; [[adiabatic-analytic-error-budget]] breaks it down factor by factor.
At the historical warm `B_one_forth=180`, `3*n0`, `T_0minus=10`, `T_0plus=5`, `xi=-0.5` state, the corrected equilibrium temperature, ceiling and flux are `30.9370395272`, `0.5536383788`, and `1.78172329193`.
The expanded 14-test analytical suite and 25-test contour suite pass; independent warm/cold integral quadrature agrees within `3.6e-15`.
The frozen 60-test historical selection retains exactly its five existing failures before and after correction.
The 16-test spatial BVP selection has 15 passing methods after bounded retries, with one cold seed-independence test timing out at 600 seconds in both the baseline and corrected runs; no new assertion failure appeared.
The manuscript now states the two endpoint matching equations in an unnumbered insertion; both isolated builds pass with unchanged numbering and warnings.
Detailed before/after regressions, changed assertions, protected-artifact hashes, and source comparisons are recorded under `local_analysis_results/26-10-26-second-order/`.
See [[cold-adiabatic-analytical-contours]] for the new, separately saved contour data.
