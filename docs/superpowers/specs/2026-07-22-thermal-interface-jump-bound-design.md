# SUPERSEDED — DO NOT IMPLEMENT: Thermal-conductivity limit on T(0⁺)

> **Superseded on 2026-07-23.**
> The claim below that thermal and flavor relaxation share the same low-temperature `T^(-5/3)` transport time is not consistent with the Heiselberg–Pethick thermal-relaxation result.
> Read `docs/research/THERMAL_INTERFACE_TEMPERATURE_CLOSURE__CURRENT_CONCLUSION.md` before doing any related work.

**Date:** 2026-07-22
**Paper:** `Latex_writting/Hydrodynamical combustion/Main.tex`, `Notes.tex`
**Code:** `RMFsolver/phase_velocity.py`
**Status:** superseded; retained only for historical context

---

## 1. Problem

The nuclear→quark deflagration front is fixed by three flux-conservation laws
(baryon `J_B`, momentum `Π`, energy `E`) plus the K-current equation. Given the
upstream state `T(0⁻)`, `n_B(0⁻)` and a trial `u(0⁻)`, the just-downstream state
has four unknowns:

```
muB(0+), muK(0+), T(0+),  u(0+)
```

against three conservation equations. The system is one short. The current code
closes it by taking `a(0+)` (equivalently `n_K(0+)`) or `T(0+)` as an input
(`solve_front_energy_conserving_uNmax` already accepts `TQstar`), and picks the
value that maximises the boundary speed.

The existing "Local thermal equilibrium" appendix places one limit on that free
choice: it requires the K-diffusion coefficient `D_K` to vary slowly enough that
a diffusive description is valid,

```
lambda_mfp = 3 D_K  <  D_K / |dD_K/dx|,
```

which on the constant-pressure trajectory reduces to (published Eq. LTE_inequality)

```
5 u(0-) lambda_n a(0+) [1 - a(0+)]  <  A^2 - a^2(0+).            (EXISTING)
```

**Goal.** Add a second, physically distinct limit coming from *thermal
conductivity across the front* — the interface temperature jump — and use it to
bound `T(0+)`. This is what "study the thermal conductivity across the boundary
and place a limit on `T(0+)`" means.

## 2. Physical model

Ideal hydrodynamics (`T^{mu nu} = (P+eps)U^mu U^nu + P g^{mu nu}`) carries **no**
heat flux, which is exactly why the system is one equation short: the missing
fourth relation is the heat equation, deleted when `q^mu` was dropped from the
stress tensor. We do **not** re-add `q^mu` to the bulk solver (that would be the
"conduction as the real closure" option, rejected). Instead we treat the heat
flux as a perturbation localised at the front and ask a consistency question.

### 2a. Deliverable — the mean-free-path interface jump (Le bound)

Heat carried on the quark side has a mean free path `lambda_thermal`. Across the
front the temperature changes; kinetic theory (Péraud 2015; Meilakhs 2019) gives
a temperature jump of order

```
Delta T  ~  lambda_thermal * |dT/dx|      (Knudsen-order jump).
```

This is the moving-front Kapitza picture of Buchbinder (2017) and Palmieri
(2012): a planar phase front moving into a medium develops an interface
temperature discontinuity set by the carrier mean free path and the local
gradient. A local temperature `T(0+)` is only meaningful if this jump is small
compared with `T` itself:

```
lambda_thermal * |dT/dx| / T  <  1        at 0+.                 (LTE at interface)
```

Write the thermal mean free path as a multiple of the strangeness mean free path,

```
lambda_thermal = 3 Le D_K,   Le = lambda_thermal / lambda_K,
```

`Le` the Lewis-number-like ratio (thermal vs strangeness transport mean free
path). Evaluate on the constant-pressure trajectory `T^2(a) = (2 muB^2 / 81 pi^2)(A^2 - a^2)`
with the interface gradient `D_K(0+) a'(0+) = -lambda_n u(0-)[1 - a(0+)]`. The
`D_K` normalisation cancels (verified with Wolfram: the result is independent of
`kD`), leaving

```
3 Le u(0-) lambda_n a(0+) [1 - a(0+)]  <  A^2 - a^2(0+).         (NEW, thermal)
```

**Same functional form as EXISTING, coefficient 5 → 3 Le.** Both define the same
`A`-limited curve; saturating either gives an `a_LTE(0+)`, hence a `T_min(0+)` via
the isobar. Therefore:

```
thermal bound is the binding constraint  <=>  3 Le > 5  <=>  Le > 5/3 ~ 1.67.
```

The effective LTE coefficient used to cap `a(0+)` is `c_eff = max(5, 3 Le)`.

**The crux — value of Le.** Heiselberg & Pethick (1993) obtain both the quark
thermal conductivity `kappa_T` and the flavor diffusion `D_K` from the same
Landau-damped single-gluon-exchange transport time `tau ∝ T^{-5/3}` (the same 5/3
in the code's `D_K`). So `kappa_T = (1/3) c_V v^2 tau` and `D_K = (1/3) v^2 tau`
share `tau`, `Le` is a pure O(1) number, and its position relative to 5/3 is set
by the *ratio of transport integrals*: heat is carried by all quark flavors
weighted by energy, strangeness only by the `(n_d - n_s)/2` asymmetry weighted by
number. Computing that ratio is the physics task of this deliverable. If
`Le < 5/3` the thermal bound is a consistency cross-check proving the framework's
temperature is well-defined (the existing "5" already guarantees it). If
`Le > 5/3` it tightens `T_min(0+)`, carving the cold-star corner of parameter
space while leaving merger conditions (large `T(0-)`) untouched.

### 2b. Outlook — genuine Kapitza resistance R_K (derived, not solved)

The construction in 2a is the LTE/validity jump. There is a second, physically
distinct interface jump: the true thermal boundary resistance `R_K` from the
mismatch between confined nucleon heat carriers and deconfined quark carriers,

```
T(0-) - T(0+) = R_K q,
```

nonzero even when both sides are in perfect LTE, with `q` fixed by the conversion
energetics (energy conservation) and `R_K` from the diffuse-mismatch / radiation
limit conductance `h_K ~ (1/4) C v` (Reddy 2005; Shi 2018 radiation limit). Unlike
`D_K`, `R_K` for a nucleon↔quark interface has **no** dense-matter literature
value. We do **not** build an `R_K` solver. The outlook section estimates the size
of `R_K q` against `T`, argues whether it is sub-dominant to the 2a jump, and
frames it as the answer to the referee question "what about the real Kapitza
resistance?".

## 3. Scope

**In scope**
1. Derivation of the 2a thermal bound `3 Le u lambda_n a(1-a) < A^2 - a^2`,
   presented in the LTE appendix parallel to the existing "5" bound, with the
   `Le > 5/3` crossover stated explicitly. (Algebra already Wolfram-verified.)
2. Calculation / defensible estimate of `Le` from Heiselberg–Pethick quark
   transport (`kappa_T` vs `D_K` transport-integral ratio), with a stated value
   and its uncertainty band relative to 5/3.
3. Code: generalise the hard-wired LTE coefficient `5.0`
   (`phase_velocity.py:339`, `_analytic_aqstar_lte`) to `max(5.0, 3.0*Le)` via a
   new `Le`/`lewis` parameter threaded through `_analytic_aqstar_lte` and its
   callers (`analytic_velocity_bound`, `_solve_analytic_downstream_endpoint_for_uN`,
   any `_analytic_aqstar_lte` call sites). Default `Le` chosen so behaviour is
   **unchanged** unless the caller opts in (default reproduces the current `5`).
4. Outlook subsection (2b): derived estimate of the genuine `R_K` jump, argued
   sub-dominant or not. No new solver.
5. A unit test asserting: (a) the default (`Le=None`, coefficient exactly `5.0`)
   reproduces the existing `_analytic_aqstar_lte` output bit-for-bit; (b) for
   `Le > 5/3` the returned
   `a_LTE(0+)` is smaller (tighter) than the `5`-coefficient value; (c) the
   `D_K`-normalisation independence (result invariant under a rescaling of the
   `D_K` prefactor at fixed trajectory).

**Out of scope (YAGNI)**
- Re-adding `q^mu` to the bulk stress tensor / a 3-field `(n_K, j_K, T)` BVP.
- A standalone `R_K` closure solver.
- Upstream conductive preheat of nuclear matter (different mechanism; set aside).
- Neutrino-cooling coupling to the interface temperature.

## 4. Interfaces and data flow

```
Le (new input, O(1))
  │
  ▼
c_eff = max(5, 3 Le)                         # effective LTE coefficient
  │
  ▼
_analytic_aqstar_lte(A, u_N, lambda_n, c_eff)  # beta = c_eff * u_N * lambda_n
  │  returns a_interface_LTE  (smaller ⇒ tighter cap on a(0+))
  ▼
analytic_velocity_bound(...) / downstream endpoint solve
  │  a(0+) capped ⇒ T_min(0+) via isobar T^2(a)=(2 muB^2/81 pi^2)(A^2-a^2)
  ▼
solve_front_energy_conserving_uNmax(TQstar = T_min(0+), ...)   # existing path
```

The bound never touches the conservation laws or the BVP; it only tightens the
admissible `a(0+)` / `T(0+)` window that those solvers already accept as input.

## 5. Verification

- **Algebra:** reproduce EXISTING "5" and NEW "3 Le" with Wolfram from the same
  trajectory + interface gradient (done: `kD` cancels, coefficient split is
  `5` vs `3 Le`).
- **Le:** state value with method (transport-integral ratio) and error bar vs 5/3.
- **Code:** unit test items 5(a)–(c) above; existing `tests/test_phase_velocity`
  must still pass (`python3 -m unittest tests.test_phase_velocity`).
- **Numeric sanity:** for representative `n_B(0-)=3 n0`, `B^{1/4}=180` MeV, scan
  `T(0-)` and confirm the bound removes the cold, high-`a(0+)` corner and is
  vacuous at merger temperatures (large `T(0-)`), matching the qualitative
  expectation `T(0-)^2` scaling of the admissible `a(0+)`.

## 6. Key references

- Heiselberg & Pethick, transport in degenerate quark plasmas — source of both
  `kappa_T` and `D_K`; sets `Le`.
- Buchbinder et al. 2017 (Physica A); Palmieri et al. 2012 (PRE 86, 051605) —
  moving planar front with Kapitza interface temperature discontinuity, steady
  velocity–undercooling relation. Structural analogue of the combustion front.
- Péraud & Hadjiconstantinou 2015 (PRB 93, 045424); Meilakhs 2019 — kinetic
  temperature-jump boundary condition ~ Knudsen order (mean free path × gradient).
- Reddy et al. 2005 (diffuse-mismatch model); Shi et al. 2018 (radiation limit) —
  for the 2b `R_K` outlook.
- Rosero-Gil et al. 2020 (PRC 103, 055816); Niebergal et al. 2010 — flame interior
  temperature rise, for numeric context on `T(0+)` ranges.
```
