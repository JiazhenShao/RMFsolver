---
summary: Map of the phase_velocity.py public API — three separate analytic velocity entry points, not one — and which solver answers which question.
status: current
updated: 2026-08-17
tags: [solver, api, map]
---

# `phase_velocity.py` — what exists and when to use it

`RMFsolver/phase_velocity.py`, 7353 lines as of 2026-08-17. **Editing rule: this is the only file under `RMFsolver/` that may be modified** without an explicit instruction naming another file (see `AGENTS.md` and the project `CLAUDE.md`).

For the physics formulation behind these APIs, start with [[diffusion-limited-front]] and [[steady-front-bvp]]. For the analytic reduction, see [[analytic-front-speed]].

## Public API

Front solvers: `solve_front_isothermal` · `solve_front_energy_conserving_nK` · `solve_front_energy_conserving_uNmax` · `solve_front_thermal_conducting`

Analytic velocity: `analytic_velocity_bound` · `semi_analytic_velocity_bound` · `analytic_velocity_bound_lte`

Helpers: `u_0minus` · `z_time_evolution` · plus the EOS wrappers `PNM` `PNM_n` `edensNM` `edensNM_n` `sNM_n` `muB_from_nB_physical` `hNM` `hNM_n` `nB_NM` `Pi_NM` `hQM` `Pi_QM`.

⚠ `__all__` lists only six names — `analytic_velocity_bound`, the four `solve_front_*`, and `z_time_evolution`. `semi_analytic_velocity_bound`, `analytic_velocity_bound_lte` and `u_0minus` are **public but not exported**, so `from ... import *` silently omits them. Import them by name.

⚠ `uN` was **renamed `u_0minus`** under the endpoint-notation rule (`_N`→`_0minus`, `_Qstar`→`_0plus`, `_Q`→`_inf`). Result-dict keys followed: `u_0minus_max`, `T_inf`, `a_0plus_LTE`, `muB_inf`. Notes citing `uN`, `T_Q`, `a_interface_LTE` or `u_N` are stale.

`solve_front_adiabatic` and `solve_front_energy_conserving` **no longer exist** — removed in a restructuring. Older notes and specs referencing them are stale.

## Three analytic velocity entry points

All three are thin wrappers over one shared eigenvalue solve, `_solve_analytic_velocity_bound`, selected by two switches: `velocity_closure` (how $I_2$ is obtained) and `interface_control` (what fixes $a(0^+)$).

| Function | $I_2$ closure | Interface | `velocity_method` tag |
|---|---|---|---|
| `analytic_velocity_bound` | `closed_form` | `fixed_T_0plus` | `full_analytic_closed_form_I2` |
| `semi_analytic_velocity_bound` | `numerical_I2` | `fixed_T_0plus` | `semi_analytic_numerical_I2` |
| `analytic_velocity_bound_lte` | `closed_form` | `LTE` | `full_analytic_LTE` |

The fourth combination (`numerical_I2` + `LTE`) has no public wrapper.

- **Full analytical** keeps the quadratic-temperature $I_2$ bracket from the closed-form derivation. That approximation does *not* enter $a(0^+)$, which comes from the exact EOS definition $a = n_K/n_B$ at $T(0^+)$.
- **Semi-analytical** follows the conserved $(P,\ h/n_B)$ layer trajectory and integrates $I_2$ numerically with the full diffusion coefficient and exact weak rate. This is the `numerical-I2` method behind the `26-08-02` ray grid.
- **LTE** is the preserved closed-form bound where $a(0^+)$ is set by local equilibrium rather than a prescribed temperature — see [[lte-composition-bound]].

`T_0plus` is a **required keyword-only** argument for the first two; `analytic_velocity_bound_lte` does not take it. `interface_fraction_mode` is no longer a parameter — it survives only as an *output* key echoing `interface_control`.

All three return the same key set (~70 keys); only the values and the `velocity_method` / `analytic_formula_variant` / `interface_fraction_mode` tags differ.

## Exact zero upstream temperature

`analytic_velocity_bound`, `semi_analytic_velocity_bound`, and
`solve_front_energy_conserving_uNmax` accept finite $T(0^-)\geq0$ as of
2026-08-12. The nuclear EOS already had a genuine zero-temperature branch;
strict-positive input guards were the only reason the $T(0^-)=0$ contour ray
previously contained no results. This is distinct from the fixed-interface
$T(0^+)=0$ degeneracy in [[unmax-degeneracy]]: the validated contour case may
have $T(0^-)=0$ while keeping $T(0^+)>0$.

The clean endpoint-inclusive cluster workflow is
`new_paper_calculations/26-08-12/`. Its boundary construction and contour mesh
both evaluate the $0^\circ$ ray at exactly $T(0^-)=0$; no small positive
temperature is substituted. `run_T0plus5_all.py` is the normal entry point: it
runs domain, full analytical, semi-analytical, and numerical stages
sequentially, with one process pool using the allocation at a time.

## Which one to reach for

| Question | Solver | Caveat |
|---|---|---|
| Fast analytic bound at prescribed $T(0^+)$ | `analytic_velocity_bound` | Closed-form $I_2$. Feeds [[two-block-contour-scans]]. |
| Same, with $I_2$ integrated numerically | `semi_analytic_velocity_bound` | Slower, no quadratic-$T$ bracket. Supplies the hidden ray mesh and analytic seeds. |
| Analytic bound with $a(0^+)$ from local equilibrium | `analytic_velocity_bound_lte` | No $T(0^+)$ input. See [[lte-composition-bound]] |
| Front speed at prescribed interface composition | `solve_front_energy_conserving_nK` | — |
| Maximised front speed at prescribed $T(0^+)$ | `solve_front_energy_conserving_uNmax` | **Untrustworthy below $T(0^+)\approx$ a few MeV** — see [[unmax-degeneracy]] |
| Front speed with conduction, $a(0^+)$ as output | `solve_front_thermal_conducting` | Current best. See [[thermal-conducting]] |
| Isothermal front | `solve_front_isothermal` | Strict-retry path broken — see [[known-issues]] |
| Catch-up / $z(t)$ evolution | `z_time_evolution` | Needs full continuation state, not scalar guesses |

## Units

The solver is in **pure natural MeV units end to end**. `const.MeV_fm` appears exactly once in 7353 lines (line 7072, converting $n_0 = 0.16\ \mathrm{fm^{-3}}$). Verified dimensionally:

```text
Gamma_K = G_F^2 * mu_u^5 * muK * (muK^2 + 4 pi^2 T^2) = MeV^4   OK
invD    = MeV = 1/[x]                                            OK
h*gamma*u = MeV^4                                                OK
kappa_th * dT/dx = MeV^2 * MeV^2 = MeV^4                         OK
```

So $\kappa_{\rm th}$ carries units MeV² and drops in with **no conversion factor**.

## Default parameter set

`paraQMCRMF3` throughout notebooks and scripts.

⚠ **The old 196/202/228 m/s reference triple is not reproducible under the current API** and should not be reused. Re-measured 2026-08-17 at $n_B = 3n_{\rm sat}$, $B^{1/4} = 180$ MeV, with $\mu_B$ derived from `muB_from_nB_physical` (giving 1190.86/1190.64/1189.74 MeV at $T(0^-)=5/10/20$ — not the 1181 MeV the old note quoted), reading `u_0minus_max`:

| $T(0^-)$ | full, $T(0^+)=5$ | full, $T(0^+)=1$ | semi, $T(0^+)=5$ | LTE |
|---|---|---|---|---|
| 5 MeV | 143.93 | 169.61 | 129.24 | 189.22 |
| 10 MeV | 147.82 | 173.93 | 133.25 | 193.84 |
| 20 MeV | 163.23 | 190.99 | 149.55 | 212.06 |

The closest path to the retired triple is LTE, still 3.5–7.5% below it; passing $\mu_B=1181$ directly gives 176.59/181.64/201.48, further away. The old numbers predate a change in the analytic formulation that has not been identified. **Anything derived from them needs recomputing** — including the $X_q$ estimate in [[gw-observables-section]].

## $z(t)$ continuation

`z_time_evolution` must propagate the **complete** previous solver result — previous downstream endpoint, interface state, BVP profile and baryon current — not just scalar guesses. Rebuilding the endpoint seed as $(1100\ \mathrm{MeV}, T_Q^{\rm guess})$ at every density leaves the local convergence basin near equilibrium.

Adaptive stepping: on failure, insert the logarithmic midpoint between the last successful distance and the target, solve that, retry. `adaptive_continuation=True`, `max_continuation_subdivisions=12`. Inserted points participate in the velocity interpolation and travel-time integral.

**Relaxing the residual tolerance is not a valid repair** — it does not preserve the physical solution branch and merely moves the failure later.
