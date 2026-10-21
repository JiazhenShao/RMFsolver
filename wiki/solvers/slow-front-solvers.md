---
summary: New isobaric slow-front BVPs precompute a flux-independent neutral EOS path and evolve a diffusion-density integral; massless and massive branches share the existing microphysics.
status: current
updated: 2026-10-21
tags: [solver, bvp, isobaric, isothermal, adiabatic, cold-limit]
---

# Isobaric slow-front solvers

`solve_front_isothermal_slow` and `solve_front_adiabatic_slow` in `RMFsolver/phase_velocity.py` precompute the neutral isobaric EOS path independently of baryon flux, then use spatial `solve_bvp` collocation with two fields and one flux eigenvalue, without shooting.
The nuclear RMF model, analytical APIs, thermal-conducting solver, and older moving-flux BVPs remain unchanged; see [[phase-velocity-overview]] and [[steady-front-bvp]].

## Closures and branches

The velocity $u$ is the nonrelativistic velocity in natural units, $h=e+P$ is the unboosted enthalpy density, and $a=n_K/n_B$ is always the local composition fraction.

| API | Conserved quantities | Precomputed quark path |
|---|---|---|
| `solve_front_isothermal_slow` | $un_B$, $P$, $T$ | Fixed $P=P(0^-)$ and $T=T(0^-)$ |
| `solve_front_adiabatic_slow` | $un_B$, $P$, $hu$ | Fixed $P=P(0^-)$ and $h/n_B=h(0^-)/n_B(0^-)$ |

The paths omit the comparison solvers' $hu^2$ momentum correction and relativistic energy factor, while retaining the existing neutral bag EOS and varying chemical potentials.
All EOS roots precede collocation; the BVP uses adaptively refined interpolants, followed by direct EOS profile validation.
`path_rtol` defaults to `1e-6` for diffusion-integral refinement; thermodynamic interpolation uses the tighter `min(path_rtol, 1e-8)` threshold, and direct EOS conservation checks retain the `1e-8` acceptance gate.
The integral tolerance accommodates massive finite-temperature quadrature noise amplified by diffusion and remains 100 times smaller than default `tol_bvp`.
Cached paths contain no trial `jB`; inverse reconstruction uses a PCHIP guess and Newton inversion of the stored integral primitive, without EOS roots.

With `ms=0`, $h=4(P+B)$, where $B=\texttt{B\_one\_forth}^4$, makes adiabatic $n_B$ constant without assuming constant $\mu_B$; isothermal $n_B$ generally varies.
With `ms>0`, the full fixed-$P,h/n_B$ constraints give varying adiabatic $n_B(x)$.
Equilibrium always means $\mu_K(\infty)=0$; for finite strange mass, $n_K(\infty)$ and $J_K(\infty)=j_Ba(\infty)$ can be nonzero.

## Spatial fields and cold endpoint

The evolved density field is the accumulated diffusion integral along that path,

$$
r(n_K)=\int_{n_K(\infty)}^{n_K}D_K\,dn_K.
$$

It converts the constitutive equation into

$$
\frac{dr}{dx}=j_Ba-J_K,
\qquad
\frac{dJ_K}{dx}=-\Gamma_K.
$$

Collocation scales $r$ by its interface value, shifts/scales the current about $j_Ba(\infty)$, and represents the positive flux eigenvalue logarithmically.
The interface state fixes $r(0^+)$; nuclear composition fixes $J_K(0^+)=j_Ba(0^-)$.
Returned `jK` is the unshifted physical current.

Warm adiabatic paths with $T(0^+)\ge0.9T(\infty)$ use chemical imbalance as the table coordinate throughout.
Colder paths join a chemical-imbalance chart near equilibrium to a $T^{1/3}$ bulk/cold chart with continuous first and second derivatives, preserving the shallow finite-mass temperature turn near equilibrium.
Implicit thermodynamic derivatives in $T^2$ cancel the cold diffusion singularity inside the integral.
`T_0plus=0` is the exact implemented-EOS endpoint without a temperature floor: `invD_K=0` and `D_K=inf` are intentional, while $r$ remains finite.
The diffusion-integral residual checks the $r$ equation, while the physical constitutive residual separately checks

$$
\frac{dn_K}{dx}=D_K^{-1}(j_Ba-J_K).
$$

The density derivative uses the thermodynamic-table/integral chain rule and collocation-polynomial derivative, remaining defined at zero temperature.
This differs from the old physical-$n_K$ plateau formulation in [[unmax-degeneracy]] and [[unmax-low-temperature]].

⚠ The unchanged `SQMsolver` massive-flavor EOS freezes thermodynamics to its zero-temperature state below a state-dependent switch, blending into finite-temperature quadrature above it.
“Exact cold endpoint” therefore refers to the implemented EOS, not exact finite-temperature massive thermodynamics arbitrarily close to zero.
The metadata flags this as `massive_low_temperature_eos="existing_SQMsolver_cutoff"`.

## Tail and continuation

The compact spatial coordinate is

$$
x=-\frac{\kappa_{\rm factor}}{\lambda_\infty}\ln(1-s),
\qquad 0\le s\le1-\epsilon_{\rm tail}.
$$

Here $\lambda_\infty$ is the positive decay rate from the derivatives of the actual fixed-$P,T$ or fixed-$P,h/n_B$ path at equilibrium.
It satisfies

$$
\lambda_\infty^2+j_B\left.\frac{da}{dr}\right|_\infty\lambda_\infty-
\left.\frac{d\Gamma_K}{dr}\right|_\infty=0.
$$

The truncated downstream boundary imposes the shifted Robin condition

$$
J_K-j_Ba(\infty)=
\left(j_B\left.\frac{da}{dr}\right|_\infty+\lambda_\infty\right)r.
$$

`return_continuation=True` returns an accepted adapted mesh and normalized fields for `continuation_guess`.
Its `model="slow_front_v1"`, `closure`, `ms`, `s`, `r_fraction`, `current_excess_fraction`, and `jB` must be compatible and valid; older physical-density payloads are incompatible.

## Call and output contract

```python
solve_front_isothermal_slow(T, nB_0minus, B_one_forth, a_0plus=np.nan, *, ms=0.0, ...)
solve_front_adiabatic_slow(T_0minus, nB_0minus, B_one_forth,
                          *, a_0plus=None, T_0plus=None, ms=0.0, ...)
```

- Isothermal input requires `T>0`; `a_0plus=np.nan` selects the existing automatic interface ceiling only for `NM_type="PNM"`. An explicit fraction must be interior.
- Adiabatic input allows `T_0minus>=0` and requires exactly one of `a_0plus` or `T_0plus`. Both default to `None`, so omitting both or supplying both raises `ValueError`.
- Temperature control selects the larger-$\mu_K$ K-rich root when two roots exist, bracketing from the residual minimum. Finite mass permits $T(0^+)$ slightly above $T(\infty)$; the path must satisfy $a(\infty)<a(0^+)<1$ and $a(0^+)<a(0^-)$.
- Defaults shared by both APIs are `tail_eps=1e-8`, `n_mesh=300`, `tol_bvp=1e-4`, `max_nodes=50000`, `kappa_factor=1.0`, and `path_rtol=1e-6`. Upstream defaults remain `NM_type="PNM"` and `paraQMCRMF3`.
- Mesh controls require `n_mesh>=13` and `max_nodes>=n_mesh`. Both `tail_eps` and the floating-point endpoint `1-tail_eps` must lie strictly between zero and one.
- Neither new numerical API applies the analytical stable-phase mask. An automatic ceiling must be interior; a saturated ceiling returns a thermodynamic-path failure.

For the automatic isothermal ceiling, the new caller first finds the pressure-matched $\mu_K=0$ state and supplies `a_floor=max(1e-8, a_inf*(1+1e-6))` to the unchanged `_solve_a_0plus_max` helper.
This avoids a bracket below massive equilibrium while preserving the pressure match, local fraction, and ceiling condition $\mu_B(0^-)=\mu_B(0^+)+a(0^+)\mu_K(0^+)$.
Explicit controls support `NM_type="PNM"`, `"SYM"`, and `"Beta_eq"` wherever a conserved quark path exists.

All fields use the module's natural MeV units; multiply returned `u_0minus` by the speed of light for SI speed.
Core result metadata includes `hydrodynamic_model="nonrelativistic_isobaric"`, `closure`, `quark_branch`, and `composition_definition="nK_over_local_nB"`.
Endpoint keys use `_0minus`, `_0plus`, and `_inf`; accepted results include `jB`, `u_0minus`, `u_0plus`, `u_inf`, and `jK_inf`.
`return_profile=True` adds `s_coord`, `x`, `r`, `muB`, `muK`, `T`, `P`, `e`, `h`, `nB`, `nK`, `a`, `u`, `jK`, `Gamma_K`, `D_K`, `invD_K`, and `dnK_dx`.

Statuses distinguish `thermodynamic_path_failed`, `bvp_failed`, `profile_rejected`, and `converged`.
Adapted nodes and midpoints undergo direct EOS, monotone-path and physical ODE checks even without profile output.
Diagnostics include `path_nodes`, `path_error`, `bvp_status`, `bvp_nodes`, `bvp_max_rms_residual`, `boundary_residuals`, `closure_error_max`, `constitutive_residual_norm`, `diffusion_integral_residual_norm`, `reaction_residual_norm`, and `a_monotone_nonincreasing`.
`constitutive_residual_norm` measures the physical density equation above; `diffusion_integral_residual_norm` separately measures the $r$ equation.
The `conservation_errors` dictionary reports maximum relative `baryon_flux` and `pressure` errors, plus `temperature` for isothermal flow or `enthalpy_flux` for adiabatic flow.
Pressure differences use `max(abs(P_0minus), 1)` as the normalization in natural units, so the diagnostic remains defined near zero pressure.
Every conservation entry and the aggregate `closure_error_max` must be at most `1e-8` for acceptance.

Microphysics retains the approximate modified Madsen rate with `(muB/3)**5`, `alpha_s=0.4`, and closed-form HP93 $h_D$; see [[strangeness-reaction-diffusion]] and [[heiselberg-pethick-1993]].
The legacy `rate_model="exact_nonleptonic"` is paired with `rate_approximation="madsen_muB_over_3"` and does not claim exact phase-space integration.

## Validation status and sources

Verified 2026-10-21: all 16 methods in `tests/test_phase_velocity_slow.py` pass across targeted unittest/direct runs; this was not one uninterrupted 16-test batch.
Coverage includes both EOS masses and closures, exact cold endpoints, seed/temperature/mesh/tail/table checks, automatic ceilings, shallow temperature turns, continuation and failure handling, physical gradients, absence of EOS calls inside collocation, and warm agreement with older relativistic solvers.
These establish representative cases rather than broad parameter-domain robustness; the API export check also passes.

At $T(0^-)=10$ MeV, $n_B(0^-)=3n_0$, $B^{1/4}=180$ MeV, and $n_0=0.16\,\mathrm{fm}^{-3}$:

- Both $m_s=0$ and 100 MeV approach the exact-cold flux through $T(0^+)=0.01,10^{-5}$ MeV; the latter differs by less than 1%.
- For $m_s=100$ MeV and $T(0^+)=0$, doubling the flux seed changes $j_B$ by about $10^{-14}$ relatively. Mesh/tolerance and tail changes give relative flux differences $8.7\times10^{-8}$ and $4.8\times10^{-8}$, with cubic-resampled profile differences below $2.7\times10^{-6}$.
- Tightening `path_rtol` from `1e-6` to `5e-7` grows the table from 737 to 799 nodes and changes $j_B$ from 1.5418252832974357 to 1.5418252842361182 in natural units, about $6.1\times10^{-10}$ relatively. The tighter table's pressure, enthalpy-flux, and baryon-flux errors are $1.51\times10^{-10}$, $2.62\times10^{-11}$, and $2.22\times10^{-16}$; common-grid profile comparisons pass `3e-4`.

The stricter `path_rtol=2e-7` attempt was stopped for excessive runtime; its convergence remains unverified.
Explicit symmetric/beta-equilibrated cases pass where endpoints exist; symmetric adiabatic matter is available at $4n_0$ but not at the tested $3n_0$ endpoint.

An existing-API selection across `test_phase_velocity.py`, `test_phase_velocity_numerical_i2.py`, `test_phase_velocity_fixed_T_analytic.py`, and `test_phase_velocity_isothermal_analytic.py` passes 37 of 43 tests before and after, with identical six failure names and exceptions: four synthetic `SlowFrontNoSolution` fixtures (42.0272 MeV gap), the retired `q` profile key, and an `A_boundary` rejection expectation.

### Independent check against the manuscript system (2026-10-21)

Both APIs implement the manuscript's first-order-in-$u$ system (Secs. 3.1 and 4.1 of `Latex_writting/Main.tex`): $un_B$, $P$ and $T$ (or $hu$) constant, $J_K=un_K-D_K\,dn_K/dx$, $dJ_K/dx=-\Gamma_K$, $J_K(0^+)=u(0^-)n_K(0^-)$, $\mu_K(\infty)=0$.
A separate checker that uses only the Appendix A massless EOS, the Madsen $(\mu_B/3)^5$ rate and HP93 $D_K$ ($\alpha_s=0.4$) tested four massless solves at $B^{1/4}=180$ MeV: isothermal $(T,n_B(0^-))=(10,3n_0)$ with the automatic ceiling and $(20,3.5n_0)$ with $a(0^+)=0.3$; adiabatic $T(0^-)=10$ MeV, $3n_0$ with $a(0^+)=0.55$ and with $T(0^+)=1$ MeV.

- EOS recomputed from profile $(n_B,a,T)$ matches the code's $P$, $\mu_B$, $h$ to $3\times10^{-15}$; $h=4(P+B)$ holds to $7\times10^{-16}$.
- $P$, $T$ and $hu$ stay constant to $\le4.4\times10^{-10}$; both ODEs hold to $\le3.2\times10^{-7}$ with spline derivatives, and the ceiling condition $\mu_B(0^-)=\mu_B(0^+)+a\mu_K(0^+)$ holds to $2\times10^{-11}$ MeV.
- Bisection shooting of the physical $(n_K,J_K)$ ODEs from $0^+$ ($j_B$ low turns back, high overshoots $n_K=0$) reproduces every eigenvalue $j_B$ to $\le4.9\times10^{-9}$ relative.

Not covered: $m_s>0$ and the exact $T(0^+)=0$ endpoint. Scripts and logs are in `new_paper_calculations/26-10-21/slow_front_independent_check/`.

- `RMFsolver/phase_velocity.py`: `_SlowFrontPath`, `_solve_front_slow`, and the two public wrappers; regression sources are under `tests/`.
- `RMFsolver/SQMsolver.py`: `Tswitch_free_pressure` and the shared massive-flavor state evaluator, read without modification.
- Wolfram algebra check: `Expand[Det[{{jB aR, -1}, {-gammaR, 0}} + lambda IdentityMatrix[2]]]` returns `lambda^2 + jB*aR*lambda - gammaR`, confirming the displayed tail equation for the path derivatives.
