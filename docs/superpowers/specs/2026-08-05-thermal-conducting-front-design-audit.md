# Design audit: `solve_front_thermal_conducting`

**Date:** 2026-08-05
**Target:** `RMFsolver/phase_velocity.py` (only file to be modified under `RMFsolver/`)
**Tests:** `tests/test_phase_velocity.py`
**Status:** audit complete, NO CODE WRITTEN — awaiting review

---

## 0. Working-tree observations (read before editing)

- `/Users/janshao/NSM_related` is **not a git repository** (`git status` → fatal). No diffing,
  no stash safety net, no commit. Any edit must be additive and surgical.
- `RMFsolver/phase_velocity.py` is **5255 lines**, mtime 2026-08-02 23:43. It has been
  substantially restructured recently: `solve_front_adiabatic` and
  `solve_front_energy_conserving` no longer exist. Surviving public API:
  `solve_front_isothermal`, `solve_front_energy_conserving_nK`,
  `solve_front_energy_conserving_uNmax`, `analytic_velocity_bound`, `uN`,
  `z_time_evolution`.
- `AGENTS.md` confirms: only `RMFsolver/phase_velocity.py` may be modified.

## 1. Units — resolved, no conversion needed

`const.MeV_fm` appears **exactly once** in 5255 lines (line 4974, converting `n0 = 0.16 fm^-3`).
The solver is in **pure natural MeV units** end to end. Verified dimensionally:

```
Gamma_K = G_F^2 * mu_u^5 * muK * (muK^2 + 4 pi^2 T^2)
        = MeV^-4 * MeV^5 * MeV * MeV^2 = MeV^4 = [nK]/[x] = MeV^3 / MeV^-1   OK
invD    = MeV = 1/[x]                                                        OK
h*gamma*u = MeV^4                                                            OK
kappa_th * dT/dx = MeV^2 * MeV^2 = MeV^4                                     OK
```

So `kappa_th` carries units **MeV^2** and drops into the existing machinery with **no
conversion factor**. A dimensional regression test will lock this in.

## 2. Reduced ODE state vector

Three first-order ODEs on the quark side, `x >= 0`:

```
y = [nK, jK, T]

dnK/dx = (u*nK - jK) * invD(muB, T)
djK/dx = -Gamma_K(muB, muK, T)
dT/dx  = (h*gamma*u - F_E) / kappa_th(muB, T)
```

The three conservation laws are used as **first integrals**, not integrated:

```
jB  = nB*u                      (constant)
Pi  = P + h*u^2                 (constant)
F_E = h*gamma*u - kappa_th*T'   (constant)
```

## 3. Algebraic thermodynamic unknowns — a NEW 2x2 closure

At each collocation point, given `(nK, T)` from the ODE state and the constants `(Pi, jB)`,
recover `(muB, muK)` from **two** equations:

```
nK_QM(muB, muK, T)                        = nK
P_QM(muB, muK, T) + h(muB,muK,T)*u^2      = Pi,      u = jB / nB_QM(muB, muK, T)
```

then evaluate `nB, u, gamma, h, Gamma_K, invD, kappa_th`, and `mu_q = muB/3`.

**This is a 2x2 solve, not the existing 3x3.** The existing
`_solve_local_quark_state_from_nK_E_and_Pi` (line 2663) solves `(muB, muK, w=T^2)` using
`E = h*gamma*u = const` as its third equation. That constraint is **false** once conduction
is present, so it must not be reused. `_solve_interface_Qstar_from_TQstar_E_and_Pi`
(line 2529) is invalid for the same reason.

Dropping `T` from the algebraic unknowns is a strict conditioning improvement: the existing
3x3 needed the `w = T^2` trick (documented at lines 2678-2684) precisely because the
Jacobian goes singular in `T` as `T -> 0`. In the new formulation `T` is an ODE variable and
never appears as a root-finder unknown, so that failure mode is structurally absent.

## 4. Scalar BVP parameters

One: `jB`, carried as `theta = log(jB)` (or the logit map when `jB_bounds` is given), reusing
the existing `param_from_jB` / `jB_from_param` pattern at lines 4166-4176.

`p = 1`.

## 5. Boundary-condition count — audited, and it closes exactly

For `n` first-order ODEs, `p` parameters, and a downstream stable manifold of dimension `k`:

```
(conditions at x=0) + (n - k) = n + p     =>     conditions at x=0 = k + p
```

**Existing 2D system** (validation of the rule): `J2 = [[invD*u, -invD], [-dGamma/dnK, 0]]`,
`det = -b*c < 0`, `trace = a > 0`, eigenvalues `(a -/+ sqrt(a^2+4bc))/2` → one negative, one
positive, `k=1`. So conditions at 0 = 1+1 = 2 (`nK(0)`, `jK(0)`) and 1 at infinity
(`_nK_tail_residual`), total 3. This matches the code exactly (lines 4402-4419). Rule confirmed.

**New 3D system.** With
`a = invD*u > 0`, `b = invD > 0`, `c = dGamma_K/dnK`, `d = dGamma_K/dT`,
`e = (1/kappa)*dEflux/dnK`, `f = (1/kappa)*dEflux/dT`:

```
J = [[ a, -b,  0],
     [-c,  0, -d],
     [ e,  0,  f]]

det J   = b*(d*e - c*f)
trace J = a + f
```

We need `k = 1`, because only two conditions are available at `x=0`
(`T(0)=0`, `jK(0)=jB`) and `k + p = 2` with `p = 1`.

```
k = 1  <=  det J < 0  AND  trace J > 0
       <=> c*f > d*e   AND   a + f > 0
```

Equivalently: **the 2x2 Jacobian of `(Gamma_K, Eflux)` with respect to `(nK, T)` must have
positive determinant at the downstream equilibrium.** `f > 0` is expected (raising `T` at
fixed `nK, Pi, jB` raises the enthalpy flux), which also secures `trace > 0`.

**This is the single go/no-go condition for the whole formulation and it must be verified
numerically at the actual downstream state, not assumed.** If it fails (`k = 2`), three
conditions would be needed at `x = 0`, forcing `nK(0)` to be prescribed again — which would
reintroduce the free interface-composition parameter and defeat the purpose.

Final BC set (4 = n + p):

| # | Location | Condition |
|---|----------|-----------|
| 1 | `x = 0`  | `T(0) = 0` |
| 2 | `x = 0`  | `jK(0) = jB` |
| 3 | `x -> inf` | deviation has zero component along growing eigenvector 1 |
| 4 | `x -> inf` | deviation has zero component along growing eigenvector 2 |

`T'(0+)` is **not** imposed. It is `(h_Qstar*gamma_Qstar*u_Qstar - F_E)/kappa_th(0+)`,
selected globally. `nK(0)` is **not** imposed either.

### Consequence worth flagging to the physics

`nK(0)`, hence `a(0+) = nK(0)/nB(0+)`, is **selected by the BVP** rather than being a free
parameter. In the current ideal-fluid treatment `a(0+)` is unconstrained and the paper
maximises the speed over it. With conduction the interface composition is determined. This is
a stronger result than merely removing the non-Lipschitz plateau, and it should be stated
carefully and only after the `k=1` check and the seed-independence tests pass.

## 6. Downstream asymptotic conditions

Prefer the **eigenvector-projection** form over naively evaluating `muK = 0` and `q_th = 0` at
a truncated endpoint: build `J` numerically at the downstream equilibrium, eigendecompose,
and require the deviation `yb - y_eq` to have zero component along the two growing
eigenvectors. This is the rigorous generalisation of the existing `_nK_tail_residual`
(line 3537), which already encodes the single decaying mode analytically in 1D.

The compactification `s = 1 - exp(-lambda*x)` is retained, with `lambda = |Re(lambda_stable)|`
of the single decaying eigenvalue — replacing the current chemical-only
`lam = (-u + sqrt(u^2 + 4*D*rate_slope))/(2*D)` (line 4292), which is **not** valid for the
coupled system and must be rederived.

## 7. Treatment of exact `T = 0` — direct `T`, and why the alternatives are worse

Three candidate formulations were compared:

- **(a) Solve directly for `T`, reject invalid trial states.** RECOMMENDED.
- **(b) Thermal potential `Theta = integral_0^T kappa_th(theta) dtheta`.** In the physical
  domain `kappa_th` is essentially constant (see section 8), so `Theta ≈ kappa_th * T` — a
  linear rescaling. It buys nothing and costs an inversion.
- **(c) `T = sigma^2` to enforce `T >= 0`.** Then `dsigma/dx = (Eflux - F_E)/(2*sigma*kappa)`,
  which is **singular at `sigma = 0`**. This reintroduces exactly the non-Lipschitz pathology
  being eliminated. Rejected.

So (a) it is: the simplest option is also the only one that keeps the interface regular.

**Regularity at `x = 0` (Wolfram-verified).** `kappa_th(0+)` is finite and positive, and
`Eflux(0+) - F_E != 0`, so `dT/dx(0) = T'_0` is finite and nonzero. Then near `x = 0`:

```
T(x) ~ T'_0 * x
invD ∝ T^(5/3) ~ (T'_0 x)^(5/3)                    (Landau term dominates at small T)
dnK/dx = G_0 * invD,   G_0 = u(0)*nK(0) - jB != 0
=>  nK(x) = nK(0) + (3/8) * G_0 * KK * T'_0^(5/3) * x^(8/3)
```

Wolfram returns the exponent `8/3` exactly, and `d(T^(5/3))/dT -> 0` as `T -> 0+`, i.e. the
right-hand side is **Lipschitz** at the interface. Contrast the old system: there `T` was
slaved algebraically to `nK`, so `nK = nK(0)` implied `T = 0` implied `dnK/dx = 0` — the
interface state was a genuine **fixed point** of the `nK` flow, permitting an arbitrary
waiting time. Now `T` is independent and leaves zero at a globally fixed nonzero rate, so no
plateau freedom exists.

**No temperature floor** will be introduced into the residual. A small `T(0) = T_eps > 0` is
used only as a **continuation waypoint**, with `T_eps -> 0` demonstrated to converge.

## 8. Thermal conductivity and the `I_kappa` problem — HONEST STATUS

Target expression:

```
kappa_th(T, mu_q) = pi^3 * v_F^2 * T^2 / (24 * alpha_s^2 * I_kappa(T/q_D))
q_D^2             = 2 * N_q * alpha_s * mu_q^2 / pi
```

**Good news — `q_D` already exists.** With `N_q = 3`, `sqrt(6*alpha_s/pi) = 0.756939756606048`,
which equals the code's `_TRANSPORT_QD_COEFF = sqrt(3*g_s^2/(2*pi^2))` to machine precision.
Reuse it directly.

**Verified `T -> 0+` limit (Wolfram).** With `I_kappa(y) ~ 2*zeta(3)*y^2`:

```
limit(T->0+) kappa_th = pi^3 * v_F^2 * q_D^2 / (48 * zeta(3) * alpha_s^2)
```

confirmed symbolically to match the target exactly. Finite, positive, no `0/0` evaluated.

**The full `I_kappa` could not be recovered.** Heiselberg & Pethick, PRD **48**, 2916 (1993)
(`Heiselberg:1993cr` in `Latex_writting/reflist.bib`) has **no arXiv preprint** and is
paywalled at APS. Equation (62) is not available from any accessible source. Per instruction
item 7, `I_kappa` will therefore be an **explicit injectable callback**, not an invented
crossover.

**Mitigating fact that makes this acceptable — the physical domain is deep in the low-`y`
regime.** Computed with the code's own constants:

| `mu_q` (MeV) | `q_D` (MeV) | `y` at `T=1` | `y` at `T=10` | `y` at `T=50` | `kappa_th` (MeV^2) |
|---|---|---|---|---|---|
| 300 | 227.1 | 0.0044 | 0.0440 | 0.2202 | 3.08e5 |
| 370 | 280.1 | 0.0036 | 0.0357 | 0.1785 | 4.68e5 |
| 450 | 340.6 | 0.0029 | 0.0294 | 0.1468 | 6.93e5 |

`y = T/q_D < 0.25` across the entire parameter range, and for the solutions of interest
(`T` from 0 to a few MeV) `y < 0.05`. The `y >> 1` logarithmic asymptote is unreachable
(it needs `T >> 280 MeV`). The default `I_kappa` will therefore be the **validated low-`y`
asymptote with a hard domain guard** that raises outside `y <= y_max_validated` (default 0.3)
rather than silently extrapolating. This is documented as an approximation, not labelled
"full".

Note the consequence: with the pure asymptote, `kappa_th` is **independent of `T`**. In-domain
that is a <~2% effect, but it must be stated, and it is the reason option (b) in section 7
degenerates.

## 9. Variable scaling

Reuse the existing `bvp_nK_scale` / `bvp_jK_scale` pattern and add `bvp_T_scale`, set from the
downstream equilibrium temperature `T_Q` (not from `T(0) = 0`, which carries no scale).
Residuals are normalised by `pi_scale`, `nK_scale`, `jK_scale`, `T_scale` as the existing code
does at lines 4399-4401.

## 10. Invalid collocation states — no silent clipping

The existing solver **snaps** out-of-domain trial states to precomputed endpoints at lines
4339-4345 (`if exact_zero_left and nK >= nK_Qstar: return thermo_Qstar`). The new path will
**not** do this. Instead:

- trial states with `T < 0`, non-convergent 2x2 closure, `nB <= 0`, or non-positive
  `kappa_th` return a large finite residual and **increment a typed failure counter**;
- the counters are surfaced in the result so a "converged" answer built on many rejected
  nodes is visible rather than hidden;
- continuation from `T_eps > 0` keeps trial profiles positive in practice, so rejection
  should be rare on the successful path — and if it is not, that is reported.

## 11. Test plan (TDD — written before implementation)

1. `kappa_th` `T -> 0+` limit equals `pi^3*v_F^2*q_D^2/(48*zeta(3)*alpha_s^2)`; finite,
   positive, no NaN; and `kappa_th(T=0)` does **not** evaluate `T^2/I_kappa` directly.
2. `I_kappa` low-`y` asymptote reproduced; domain guard raises above `y_max_validated`;
   callback injection works and is honoured.
3. Dimensional test: `kappa_th * dT/dx` and `h*gamma*u` agree in magnitude scaling (locks
   the natural-MeV convention).
4. 2x2 local closure: recovered `(muB, muK)` satisfy both `nK` and `Pi` equations to tolerance
   on representative states.
5. Downstream Jacobian: assert numerically that `c*f > d*e` and `a + f > 0`, i.e. `k = 1`.
   **This test is the go/no-go gate.**
6. All four BCs satisfied on a converged solution.
7. `jB`, `Pi`, `F_E` constant across the reconstructed profile within stated tolerance.
8. Interface: `T(0) = 0`; `T'(0+) != 0`; `q_th(0+) = F_E - h_Qstar*gamma_Qstar*u_Qstar`.
9. Near-interface fit: `T ∝ x` and `nK(0) - nK(x) ∝ x^(8/3)`, convergence-aware exponent fit.
10. Seed independence: >= 2 very different initial guesses (including different apparent
    plateau lengths) converge to the same `jB`, `uN`, `nK(0)`, and near-interface coefficient.
11. Mesh/domain robustness: refine `n_mesh`, vary `tail_eps` and the compactification scale;
    physical outputs converge.
12. Continuation-schedule independence: `T_eps -> 0` along >= 2 schedules gives the same answer.
13. Failure behaviour: invalid states / failed closure / unresolved tail produce explicit
    diagnostics, never a plausible-looking profile.
14. Regression: full existing `tests/` suite; `solve_front_energy_conserving_uNmax` byte-identical
    behaviour on a fixed case.

## 12. Public API

```
solve_front_thermal_conducting(
    T, nB_N, B_one_forth,
    TQstar=0.0,
    ms=0.0, param=para.paraQMCRMF3, NM_type="PNM",
    tail_eps=1e-8, n_mesh=200, tol_bvp=1e-4, max_nodes=10000,
    jB_guess=None, jB_bounds=None,
    I_kappa=None,                  # callback; default = guarded low-y asymptote
    kappa_model="HP93_low_y",
    continuation_schedule=None,
    return_profile=False, verb=False,
)
```

Returns (at minimum): `success`, detailed `message`, `uN`, `jB`, `x`/`s`, profiles for
`nK, jK, T, muB, muK, nB, u, kappa_th, q_th`, the three flux profiles, `Tprime_Qstar`,
`q_th_Qstar`, BC residuals, closure residuals, max conservation errors, `solve_bvp` status /
iterations / mesh size, conductivity-model metadata, typed failure counters, and any
Knudsen/LTE warning derivable from the transport model.

Profiles are the **solved** ones; `T` is never reconstructed from the old ideal closure.

## 13. Known risks, ranked

1. **`k = 1` may fail** (`c*f <= d*e`). Kills the BC set; would force prescribing `nK(0)`.
   Mitigation: test 5 is a cheap standalone gate — run it first, before building the solver.
2. **`I_kappa` is not the published full integral.** Mitigated by the `y < 0.25` domain fact
   and the guarded callback, but must be reported as a limitation, not papered over.
3. **Existence:** requires `h_Qstar*gamma_Qstar*u_Qstar < F_E` so that `T'(0+) > 0`. Sign to be
   checked and reported, never imposed.
4. **Two growing downstream modes** make the BVP stiffer than the current one; expect heavier
   reliance on continuation.
5. **Fourier conduction at exactly `T = 0`** has an LTE / mean-free-path validity limit even
   though the formal `kappa_th` limit is finite — report the Knudsen diagnostic alongside.

## 14. Proposed execution order

1. Test 5 alone (downstream Jacobian gate) as a standalone script — **stop if it fails**.
2. Tests 1-3 + `kappa_th` / `I_kappa` helpers.
3. Test 4 + the new 2x2 local closure.
4. Downstream eigen-analysis helper + compactification `lambda`.
5. BVP assembly, continuation driver.
6. Tests 6-9.
7. Tests 10-13 (uniqueness / robustness — the ones that license the physics claim).
8. Test 14 regression; completion report.
