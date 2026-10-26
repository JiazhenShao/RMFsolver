---
summary: Historical audit of moving-flux analytical methods and the approximation choices now implemented by the separate massless slow analytical APIs.
status: current
updated: 2026-10-23
tags: [solver, analytic, isothermal, adiabatic, audit]
---

# Analytical methods after the slow-front update

The existing analytical methods remain useful comparison approximations, but they are not implementations of the new [[slow-front-solvers|isobaric slow-front closures]].
This audit checked `RMFsolver/phase_velocity.py` at commit `a5c4f3d`; no solver behavior was changed.
Historical benchmarks in [[analytic-front-speed]], [[isothermal-analytic-front-speed]] and [[adiabatic-xi-diagnostics]] remain historical.

## Agreed implementation direction (2026-10-23; now implemented separately)

The three separate massless, PNM slow-front APIs are now implemented, retaining the audited methods for comparison.
See [[analytic-slow-front-solvers]] for the current behavior and new comparison values; the plan is recorded in `docs/superpowers/plans/2026-10-23-analytic-slow-front.md`.

- **Isothermal analytical:** remove moving-momentum endpoint terms, keep the closed-form speed estimate, and use only the leading Landau-damped contribution to inverse diffusion, as in manuscript Eq. (D12). This supersedes the earlier proposed full-diffusion choice for this new branch. Select the automatic ceiling using the exact full-EOS equality $\mu_B(0^-)=\mu_B(0^+)+a(0^+)\mu_K(0^+)$, not its Taylor approximation. Preserve and explicitly document the existing frozen-background convention: interface chemical potential for transport and equilibrium density for the density ratio.
- **Full analytical adiabatic:** use the manuscript quadratic pressure, retained leading density/composition, quadratic $A_{\rm adb}$, temperature-composition ellipse, leading diffusion and closed-form $I_2$ consistently. Do not replace the ceiling with a full-EOS value.
- **Semi-analytical adiabatic:** follow the full EOS at fixed pressure and $h/n_B$, use full two-term diffusion and the local revised rate, and integrate $I_2$ including the exact cold endpoint without a temperature floor.

All three retain $\alpha_s=0.4$ and exact $h_D$; the new analytical restriction does not remove either mass branch from the numerical solvers.
No shared full-diffusion helper is to be changed for the leading-only analytical choice.
Manuscript reproducibility also requires an explicit representative-state recipe and the exact isothermal ceiling prescription; choosing leading diffusion alone does not settle those separate approximations.

## What already agrees

The analytical weak coefficient already uses $\mu_q=\mu_B/3$; changing the local rate's fifth-power prefactor therefore requires no further change to `_weak_rate_from_mu_q`.
The analytical formulas and transport helpers already share $\alpha_s=0.4$ and the exact HP93 coefficient $h_D=1.813174518048292$.
The semi-analytical integral calls the revised approximate Madsen rate directly.
The closed-form adiabatic bracket is still algebraically correct within its quadratic EOS and leading Landau-damped diffusion approximations; it is not an exact-EOS result.

## Updates needed for a matching analytical model

| Area | Live behavior | Recommended treatment |
|---|---|---|
| Endpoint hydrodynamics | `_solve_analytic_inf_endpoint_for_u_0minus` retains $P+hu^2$ and $h\sqrt{1+u^2}/n_B$; `analytic_velocity_isothermal` also solves moving-momentum endpoints inside a velocity root | For slow-front analytical variants, determine endpoints once from fixed $P,T$ or fixed $P,h/n_B$, independent of flux. Preserve old methods as explicitly identified comparison models. |
| Massless adiabatic layer | `semi_analytic_velocity_bound` already follows fixed $P,h/n_B$, with variable $\mu_B$ and constant $n_B$ | Reuse the new constrained path; the constant-density integrated speed identity survives. |
| Full analytical adiabatic layer | `analytic_velocity_bound` and the LTE variant use a fixed-$\mu_B$ isobar and a quadratic temperature bracket | Keep these as additional approximations. Exact constant $h$ at fixed pressure does not justify constant $\mu_B$ or make their EOS interface satisfy exact $h/n_B$ conservation. |
| Cold semi-analytical endpoint | `_analytic_layer_trajectory` stops at `T_cold=1e-3`; the interface helper rejects zero and the integral requires positive temperature | Use the new exact-cold path and regular quadrature in $T^{1/3}$, not a smaller temperature floor. |
| Isothermal density | The speed reduction freezes density although exact isobaric $n_B$ varies even for massless quarks | Retain an explicitly named frozen-background approximation, or include the actual density derivative in a generalized integral model. |
| Finite strange mass | Isothermal analytical API rejects `ms>0`; the adiabatic analytical APIs accept it but still use a constant-density, zero-equilibrium-composition speed formula | Guard the unextended formulas as massless, or derive a variable-density model including nonzero $n_K(\infty)$ and $J_K(\infty)$. Merely passing `ms` into the EOS is insufficient. |
| Upstream composition | The adiabatic APIs accept `NM_type`, but their denominator still hardcodes the PNM current jump $a(0^-)=1$ | Restrict that reduction to PNM until its actual upstream K fraction is incorporated. New numerical support for other nuclear compositions does not establish analytical support. |
| Profile shape | $\xi$ remains a caller-supplied approximation; public defaults are zero | Reassess it using new BVP profiles. Historical fitted values are not universal constants. |

The LTE variant also depends on velocity through its interface prescription, so eliminating flux dependence from the thermodynamic endpoint does **not** eliminate every scalar root in that variant.

## Why the massive extension needs more than a coefficient change

Using the new solver's already-defined $r$ and $J_K$ gives

$$
r''=j_B\frac{da}{dr}r'+\Gamma_K.
$$

Multiplication by $r'$ and integration, with $r(\infty)=r'(\infty)=0$ and $r'(0^+)=j_B[a(0^+)-a(0^-)]$, yields

$$
\frac{j_B^2}{2}[a(0^-)-a(0^+)]^2
+j_B\int_{0^+}^{\infty}\frac{da}{dr}(r')^2\,dx
=\int_0^{r(0^+)}\Gamma_K\,dr
=\int_{n_K(\infty)}^{n_K(0^+)}D_K\Gamma_K\,dn_K.
$$

For constant $n_B$ and $a(\infty)=0$, division by $n_B^2$ recovers the existing $I_1,I_2$ identity.
For varying density, $dn_K=n_B\,da+a\,dn_B$ and the path derivative $da/dr$ must be retained.
The right-hand integral is path-only, but the other integral still depends on the spatial profile: it does not by itself supply a new closed-form eigenvalue or justify reusing the old $\xi$ denominator.
The current `_analytic_I2_numerical` instead integrates over local fraction and divides by one endpoint density; its docstring's zero lower limit also fails at finite strange mass.

## Reproduced warm checks

Inputs: PNM, `paraQMCRMF3`, $n_B(0^-)=3n_0$, $n_0=0.16\,\mathrm{fm}^{-3}$, $T(0^-)=10$ MeV, $B_{\mathrm{one\_forth}}=180$ MeV, $m_s=0$.
Isothermal uses $a(0^+)=0.6$; adiabatic uses $T(0^+)=5$ MeV.
All analytical comparisons below explicitly use $\xi=-0.5$; BVPs use `n_mesh=120` and otherwise default controls.
The analytical adiabatic upstream chemical potential was recovered from the same RMF state: $\mu_B(0^-)=1190.6390196794925$ MeV.

| Method | $j_B$ in module natural units | Difference from matching slow BVP |
|---|---:|---:|
| Isothermal slow BVP | 2.01742521308527 | — |
| Isothermal analytical | 2.07871141868808 | +3.038% |
| Adiabatic slow BVP | 1.40708418628695 | — |
| Adiabatic semi-analytical | 1.46086188958050 | +3.822% |
| Adiabatic full analytical | 1.66746293359299 | +18.505% |

These are single-state comparisons, not domain-wide accuracy estimates or newly imposed reference tolerances.
The analytical adiabatic endpoint pressure corrections are below $5\times10^{-13}$ relative; the isothermal momentum correction is $2.29\times10^{-12}$.
They cannot account for the percent-level speed differences at this state.
The full analytical interface has a $-0.2057\%$ discrepancy in $h/n_B$ relative to upstream, while the semi-analytical interface discrepancy is $4.4\times10^{-14}$.
The exact isothermal density changes by $3.9431\%$ of downstream density; the massless adiabatic BVP density varies by only $2.1\times10^{-11}$ relatively.
A direct trapezoidal profile diagnostic estimates adiabatic $\xi\simeq-0.396$; this is not a fitted universal replacement.

A separate direct constrained-EOS endpoint solve with $m_s=100$ MeV and the same adiabatic inputs gives $T(\infty)=23.6337449258$ MeV, $a(\infty)=0.0502613138$, and $n_B(0^+)/n_B(\infty)-1=-0.00913729852$.
Its scaled EOS residual is below $9\times10^{-16}$.
Thus neither zero equilibrium composition nor constant density holds for this massive example.
The old semi-analytical interface helper reproducibly rejects $T(0^+)=0$ with `T_0plus lies below the resolved conserved-layer trajectory`; its last sample is $10^{-3}$ MeV.

## Verification and algebra record

Executed through the isolated SciPy 1.18.1 runtime using direct functions and `unittest`: four rate/microphysics checks, both fixed-temperature analytical API checks, and the numerical-$I_2$ dense-state regression all passed.
This was a targeted audit, not a full regression rerun; previously recorded baseline failures were not reclassified.

Wolfram verified the differential energy identity with the following input (`g` represents $\Gamma_K$), returning `0`:

```wl
FullSimplify[
 (D[r'[x]^2/2, x] - jB a'[r[x]] r'[x]^2 - g[r[x]] r'[x])
 /. r''[x] -> jB a'[r[x]] r'[x] + g[r[x]]
]
```

It also verified the existing closed-form bracket, returning `0` for the input below (`ap` represents the interface fraction and `A` the existing adiabatic ceiling):

```wl
FullSimplify[
 Integrate[a*(2*A^2-a^2)/(A^2-a^2)^(5/6), {a,0,ap},
   Assumptions -> 0<ap<A]
 - A^(7/3)*(24/7-3*(1-ap^2/A^2)^(1/6)
   -(3/7)*(1-ap^2/A^2)^(7/6)),
 Assumptions -> 0<ap<A
]
```

Source helpers: `_solve_analytic_inf_endpoint_for_u_0minus`, `_analytic_velocity_formula_from_endpoint`, `_analytic_layer_trajectory`, `_analytic_trajectory_interface_state`, `_analytic_I2_numerical`, `analytic_velocity_isothermal`, and `_SlowFrontPath` in `RMFsolver/phase_velocity.py`.
