# Slow-front analytical branches implementation plan

> **For agentic workers:** Use executing-plans to implement this plan task-by-task. The user's implementation direction takes precedence over redundant workflow approval gates. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add three separate slow-front analytical/semi-analytical APIs, with approximation choices explicit enough to reproduce their predictions.

**Architecture:** Determine the thermodynamic background independently of baryon flux, then evaluate the existing integrated speed closure. Isothermal analytical and full analytical adiabatic use leading Landau-damped diffusion; semi-analytical adiabatic uses the full EOS path and full two-term diffusion. Preserve the existing solver APIs for comparison.

**Tech stack:** Python, NumPy, the existing isolated SciPy runtime, direct-function and unittest execution.

**Status:** Implemented and verified on 2026-10-23. Decision retained: the user selected leading-only diffusion for the new isothermal analytical branch, superseding the earlier proposed full-diffusion choice.

## Scope and common decisions

- Modify only `RMFsolver/phase_velocity.py` under `RMFsolver/`; supporting tests and documentation live outside that directory.
- Leave existing analytical, numerical, and thermal-conducting APIs, saved artifacts, historical reference values, and the nested `RMFsolver_Pub` repository unchanged.
- New analytical branches initially support massless quarks (`ms=0`) and PNM only. Reject unsupported strange masses and nuclear compositions explicitly; retain both massless and massive branches in the existing slow-front numerical APIs.
- Nuclear matter retains the full massive RMF EOS and the existing `paraQMCRMF3` default.
- Preserve `alpha_s=0.4`, the exact HP93 expression for `h_D`, and the revised Madsen weak-rate prescription. The analytical weak coefficient and timescale are unchanged at fixed chemical potential.
- Remove velocity-dependent momentum and energy matching from the new branches. Do not remove relativistic particle thermodynamics from the EOS.
- Keep `xi=0` as the public default and allow an explicit value; do not introduce a fitted default.
- Use location-based endpoint names throughout.
- Return endpoint quantities, speed, baryon flux, approximation metadata, and thermodynamic residuals. Keep `rate_model="exact_nonleptonic"` as an explicitly legacy label and `rate_approximation="madsen_muB_over_3"` on success and failure results.

## Public interfaces

```python
analytic_velocity_isothermal_slow(
    T, nB_0minus, B_one_forth, a_0plus=np.nan, *,
    xi=0.0, ms=0.0, param=para.paraQMCRMF3, NM_type="PNM",
    delta_muB_tol=1e-6,
)

analytic_velocity_adiabatic_slow(
    T_0minus, nB_0minus, B_one_forth, *, T_0plus,
    xi=0.0, ms=0.0, param=para.paraQMCRMF3, NM_type="PNM",
)

semi_analytic_velocity_adiabatic_slow(
    T_0minus, nB_0minus, B_one_forth, *, T_0plus,
    xi=0.0, ms=0.0, param=para.paraQMCRMF3, NM_type="PNM",
    path_rtol=1e-6,
)
```

Both adiabatic interfaces accept `T_0plus=0` exactly. The isothermal speed formula requires positive temperature; do not manufacture a finite zero-temperature isothermal speed using a temperature floor.

## Task 1: Isothermal analytical branch

**Files:** Modify `RMFsolver/phase_velocity.py`; create `tests/test_phase_velocity_analytic_slow.py`.

- [x] Add API/domain tests before implementation: default mass and nuclear EOS; explicit and automatic interface selection; rejection of `ms>0`, non-PNM input, invalid temperature, and nonpositive speed denominator.
- [x] Recover the upstream RMF state from prescribed temperature and density. Solve the neutral quark endpoints at fixed upstream pressure and temperature, with equilibrium `muK_inf=0`. Do not root-find over velocity to determine these states.
- [x] For automatic interface selection, solve the exact full-EOS condition below on the static isobar. Preserve the existing physical composition domain and distinguish an unavailable ceiling from numerical failure. For prescribed composition, retain the existing explicit-interface semantics.

```text
muB_0minus = muB_0plus + a_0plus*muK_0plus
a_0plus = nK_0plus/nB_0plus
```

- [x] Use only the leading Landau-damped term in the inverse diffusion coefficient. Do not modify either shared full-diffusion microphysics helper; isolate this approximation in the new analytical evaluation.

```text
qD = (muB_0plus/3)*sqrt(2*Nf*alpha_s/pi)
D_K = pi*qD^(2/3)/(8*Nf*alpha_s^2*h_D*T^(5/3))
eta = 81*pi^2*T^2/muB_0plus^2
I2 = D_K*gamma_K*a_0plus^2*(a_0plus^2 + 2*eta)/4
u_0minus^2 = 2*I2/(lambda_n^2*(1-a_0plus)*(1+xi*a_0plus))
jB = nB_0minus*u_0minus
```

- [x] Preserve the prior frozen-background convention: evaluate transport and the analytical weak coefficient at `muB_0plus`, and use `lambda_n=nB_0minus/nB_inf`. Report both physical endpoint densities and explicitly document the density used by the formula. This convention is a separate approximation from dropping the second diffusion term; do not silently identify the two full-EOS endpoint densities.
- [x] Add an independent direct-formula check of Eqs. (D11) and (D12), using the same explicitly declared representative chemical potential and density. Use unrounded constants in the primary assertion; allow the printed `98 m/s` coefficient its rounding error. Explain the representative-state convention in the manuscript recipe so it is reproducible.
- [x] Verify leading-only diffusion differs from the existing full helper by the known positive Debye contribution, and verify exact ceiling pressure/composition/average-chemical-potential residuals. The displayed Taylor ceiling is not used by this branch.

## Task 2: Full analytical adiabatic branch

**Files:** Modify `RMFsolver/phase_velocity.py`; extend `tests/test_phase_velocity_analytic_slow.py`.

- [x] Add independent manuscript-formula tests for warm and exactly cold interfaces. Pin the EOS order rather than reusing full-EOS helper outputs as expected values.
- [x] Use the manuscript's quadratic quark pressure and its retained leading density/composition expressions consistently:

```text
P = -B_one_forth^4 + muB^4/(108*pi^2)
    + muB^2*muK^2/(12*pi^2) + muB^2*T^2/6
h = 4*(P + B_one_forth^4)
nB = muB^3/(27*pi^2)
nK = muB^2*muK/(6*pi^2)
a = 9*muK/(2*muB)
```

The retained density is the leading expression used by the displayed manuscript derivation, not the full derivative of every term in the quadratic pressure. Do not mix those two approximation orders.

- [x] Match constant pressure and `h/nB` to the upstream RMF state and determine the equilibrium endpoint algebraically. With constant pressure, enthalpy density and retained baryon density are constant, and hence the retained `muB` is constant.
- [x] Compute the quadratic cold ceiling and temperature-composition ellipse, without substituting a full-EOS ceiling:

```text
A_adb = (3/2)*sqrt(108*pi^2*(P_0minus+B_one_forth^4)
                   - muB_inf^4)/muB_inf^2
a_0plus^2 = A_adb^2 - 81*pi^2*T_0plus^2/(2*muB_inf^2)
muK_0plus = 2*muB_inf*a_0plus/9
```

- [x] Use the leading Landau-damped diffusion term and the manuscript's closed-form fractional-power `I2` bracket. Evaluate `T_0plus=0` at the finite integrated limit; do not evaluate an infinite local diffusion coefficient as an ordinary finite number.
- [x] Reject unavailable equilibria, negative ellipse radicands outside numerical tolerance, interface fractions outside `0<a_0plus<1`, and nonpositive speed denominators. A raw quadratic ceiling above one does not itself exclude a warm interface below one; a cold interface at or above one is outside the finite-speed domain.
- [x] Report residuals against the retained quadratic model and label them accordingly. Do not present full-EOS residuals as conservation failures of this deliberately approximate branch.
- [x] Verify direct reproduction of the manuscript full analytical contour formula, including the warm-to-cold limit and numerical quadrature of the same approximate integrand.

## Task 3: Semi-analytical adiabatic branch

**Files:** Modify `RMFsolver/phase_velocity.py`; extend `tests/test_phase_velocity_analytic_slow.py`.

- [x] Add warm and exact-cold tests using the existing constrained-path infrastructure as the thermodynamic reference and an independent integral evaluation as the speed reference.
- [x] Follow the full neutral massless EOS at constant pressure and `h/nB`, finding equilibrium at `muK_inf=0`. Reuse the new slow-front path without solving the spatial BVP.
- [x] Select the interface by prescribed `T_0plus` along the continuous full-EOS temperature-composition curve. Do not substitute the quadratic ellipse or a fixed-chemical-potential isobar.
- [x] Use full two-term diffusion and the revised local weak rate throughout the integral. For constant massless baryon density, integrate equivalently using the regular density integral already supplied by the path:

```text
I2 = integral(D_K*Gamma_K, a=0..a_0plus)/nB_inf
   = integral(Gamma_K, r=0..r_0plus)/nB_inf^2
u_0minus^2 = 2*I2/(lambda_n^2*(1-a_0plus)*(1+xi*a_0plus))
lambda_n = nB_0minus/nB_inf
```

- [x] Include the exact cold endpoint using the existing regular quadrature in `T^(1/3)`, without a temperature floor. Test convergence from decreasing positive interface temperatures and under independent path refinement.
- [x] Verify constant density, pressure and `h/nB` along the returned thermodynamic path. Label any optional path output as a thermodynamic path, not a solved spatial profile.

## Task 4: Compatibility, documentation and verification

**Files:** `tests/test_phase_velocity_analytic_slow.py`; `wiki/solvers/analytic-slow-front-audit.md`; `wiki/physics/analytic-front-speed.md`; `wiki/physics/isothermal-analytic-front-speed.md`; `wiki/solvers/phase-velocity-overview.md`; `wiki/index.md`; `wiki/log.md`; relevant analytical recipe prose in `Latex_writting/Main.tex`.

- [x] Add distinct EOS, hydrodynamic, diffusion and integral-approximation metadata to success and failure results. Verify new APIs never enter the old velocity-dependent endpoint solvers.
- [x] Compare representative new warm solutions with existing analytical methods and the new numerical slow-front solvers. Separate the effects of endpoint matching, the quadratic EOS, leading-only diffusion, and the assumed `xi`; do not require exact agreement between different approximations.
- [x] Run the new tests through the isolated runtime:

```sh
PYTHONPATH=local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:.:tests PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/nsm-rate-mpl ./bin/python3 -B -m unittest test_phase_velocity_analytic_slow -v
```

- [x] Run existing analytical, numerical-I2, isothermal, energy-conserving and microphysics checks through direct functions/unittest. Record the exact selected tests and retain previously reproduced baseline failures separately; do not refresh historical references merely to make different physics agree.
- [x] Update the analytical recipe with `alpha_s=0.4`, exact `h_D`, leading-only isothermal diffusion, the explicit representative-state convention, exact isothermal ceiling selection, and quadratic full analytical adiabatic closure. Keep the manuscript speed formulas; the full-diffusion correction discussed previously is not part of the new isothermal analytical branch.
- [x] Document that the semi-analytical and numerical branches retain full diffusion. Preserve historical benchmarks as historical, update the wiki index, and append one implementation log entry after verification.

## Acceptance summary

The new isothermal branch uses static full-EOS endpoint/ceiling selection and the leading-diffusion closed-form speed formula. The full analytical adiabatic branch uses the manuscript quadratic EOS, quadratic ceiling and closed-form integral throughout. The semi-analytical adiabatic branch uses the full constrained EOS curve and numerical integral with full diffusion. All three remove moving-flux endpoint corrections while retaining the existing rate and coupling choices.


## Completion record (2026-10-23)

- Added and exported all three APIs; every pre-existing function/class definition is unchanged.
- New `tests/test_phase_velocity_analytic_slow.py`: 11/11 methods pass using the isolated runtime.
- Selected existing regressions: 55/60 pass, with identical baseline failures (four synthetic `SlowFrontNoSolution` cases and one old ceiling-domain expectation). Updated only the export-list assertion in `tests/test_phase_velocity.py` for the new names.
- Independent physics/spec and code-quality reviews found no outstanding substantive issues; the denominator failure now has a distinct status. Wolfram verified endpoint and integral algebra.
- Added the live contract and comparison/verification record in `wiki/solvers/analytic-slow-front-solvers.md`; updated the requested physics/API pages, audit, index and one implementation log entry.
- Updated manuscript recipe prose and added unnumbered background matching equations; existing numbered analytical equations are unchanged.
- Temporary manuscript build succeeds (17 pages). The unresolved `sec:XX` reference and 5.09543-point equation-width warning reproduce in the original source; tracked build outputs were preserved.
- Existing solver algorithms, EOS files, saved calculation artifacts, and nested repository were preserved.
