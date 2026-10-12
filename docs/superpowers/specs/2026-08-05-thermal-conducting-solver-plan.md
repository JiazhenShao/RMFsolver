# `solve_front_thermal_conducting`: what is solved, what is reconstructed

**Date:** 2026-08-05
**Companion:** `2026-08-05-thermal-conducting-front-design-audit.md` (BC counting, `k=1` gate, units, `I_kappa`)
**Style reference:** `Notes.tex` §"(Cleaner version) Energy-conserving front as a reduced boundary-value problem" (line 1660)

---

## 1. The change in one sentence

**`T` moves from the reconstructed-algebraically column into the propagated-by-ODE column.**

Everything else in the plan is a consequence. In the existing energy-conserving solver, `T(x)` is
never integrated: it is recovered pointwise from the algebraic constraint `E = h*gamma*u`. Once
conduction enters, that constraint becomes

```
F_E = h*gamma*u - kappa_th*T'   =  const
```

which contains `T'`. It is no longer an algebraic relation for `T` — it *is* the ODE for `T`.
So `T` is promoted to a propagated field, and the local algebraic closure loses one unknown and
one equation, dropping from 3x3 to 2x2.

## 2. Side-by-side

| | Existing `..._uNmax` | New `..._thermal_conducting` |
|---|---|---|
| First integrals | `jB`, `Pi`, `E = h*gamma*u` | `jB`, `Pi`, `F_E = h*gamma*u - kappa_th*T'` |
| **Propagated ODE fields** | `nK`, `jK` (2) | `nK`, `jK`, **`T`** (3) |
| **Scalar eigenvalue** | `jB` (1) | `jB` (1) |
| **Local algebraic closure** | `(muB, muK, T)` from 3 eqs: `Pi`, `E`, `nK` | `(muB, muK)` from 2 eqs: `Pi`, `nK` |
| Interface state `Q*` | separate 2x2 solve `_solve_interface_Qstar_from_TQstar_E_and_Pi` | **none** — same local closure evaluated at `(nK(0), T=0)` |
| Downstream endpoint `Q` | `_solve_analytic_downstream_endpoint_for_uN` | **same helper, reused unchanged** |
| Reconstructed | `nB, u, h, D_K, eta, Gamma` | `nB, u, gamma, h, Gamma_K, invD, kappa_th, T', q_th` |
| Boundary conditions | 3 | 4 |
| `a(0+)` | **input** | **output**, selected by the BVP |
| `T(0+)` | input (`TQstar`) | **fixed to 0 by BC**; `T'(0+)` is output |

### Why the downstream helper is reusable

At the downstream equilibrium `q_th(inf) = 0`, so `h*gamma*u|_Q = F_E`. And
`F_E = h_N*u_N*gamma_N` from the upstream side. That is exactly the `energy_target` that
`_solve_analytic_downstream_endpoint_for_uN` (line 1057) already solves against. The
`muK=0` endpoint `(muB_Q, T_Q)` is therefore determined by `jB` alone, unchanged. `T_Q` is
**not** a second eigenvalue.

### Why the 2x2 closure is strictly better conditioned

`_solve_local_quark_state_from_nK_E_and_Pi` needed the `w = T^2` reparameterisation (documented
at lines 2678-2684) precisely because its Jacobian goes singular in `T` as `T -> 0`, leaving the
residual insensitive to temperature. In the new closure **`T` is not an unknown at all** — it
arrives from the ODE state. That failure mode is structurally absent, and no `w = T^2` trick,
no `logT`, and no interface temperature seed are needed.

## 3. What the solver actually hands to `solve_bvp`

```
solve_bvp(fun, bc, s_mesh, y_guess, p=[theta])

y = [ nK/S_nK , jK/S_jK , T/S_T ]        3 fields
p = [ theta ],   jB = exp(theta)          1 parameter
bc returns 4 residuals                     (n + k = 3 + 1)
```

Compactified coordinate `s = 1 - exp(-lambda*x)`, `dx/ds = 1/(lambda*(1-s))`, with `lambda`
now taken from the **coupled** downstream Jacobian (section 6), not the chemical-only
`lam = (-u + sqrt(u^2 + 4*D*rate_slope))/(2*D)` of line 4292.

Scales: `S_nK`, `S_jK` as in the existing code; `S_T = T_Q` (the downstream temperature —
`T(0)=0` carries no scale of its own).

## 4. Per-`jB` global state (cached once per `theta`)

```
jB      = exp(theta)
u_N     = jB / nB_N
gamma_N = sqrt(1 + u_N^2)
Pi      = P_N + h_N*u_N^2
F_E     = h_N*u_N*gamma_N                      <- replaces E, same numerical value

Q  = _solve_analytic_downstream_endpoint_for_uN(u_N, ...)   ->  (muB_Q, T_Q)   [REUSED]
J  = downstream Jacobian at Q  (section 6)
     -> lambda_stable  (compactification rate)
     -> W = the two LEFT eigenvectors of the growing modes
```

Note what is **absent** compared with the existing `build_state` (lines 4221-4323): no
`thermo_Qstar`, no `aQstar`, no `_solve_interface_Qstar_from_TQstar_E_and_Pi`, no
`rate_slope` probe. The interface state is not precomputed because `nK(0)` is unknown until
the BVP converges.

## 5. Per-collocation-point evaluation — the core loop

At each mesh point `s_i` with state `y[:, i]` and parameter `theta`:

```
1. unscale        nK = y0*S_nK ,  jK = y1*S_jK ,  T = y2*S_T

2. LOCAL CLOSURE  (the only root-find inside the ODE)  -- 2 unknowns, 2 equations
       solve (muB, muK)  from
           nK_QM(muB, muK, T)                     = nK
           P_QM(muB,muK,T) + h(muB,muK,T)*u^2     = Pi,    u = jB/nB_QM(muB,muK,T)
       reject (typed counter, large residual) if: T < 0, no convergence,
       nB <= 0, or kappa_th <= 0.   NO snapping to endpoints.

3. RECONSTRUCT    nB      = nB_QM(muB, muK, T)
                  u       = jB / nB
                  gamma   = sqrt(1 + u^2)
                  h       = P + e
                  Gamma_K = _exact_kaon_transport_rate(muB, muK, T)
                  invD    = _microphysics_..._energy(muB, T)["invD"]
                  mu_q    = muB/3 ,  qD = _TRANSPORT_QD_COEFF*mu_q
                  kappa_th= pi^3*vF^2*qD^2/(48*zeta(3)*alpha_s^2)   [guard y=T/qD]
                  Eflux   = h*gamma*u
                  Tprime  = (Eflux - F_E)/kappa_th
                  q_th    = F_E - Eflux            (= -kappa_th*Tprime)

4. RHS in x       dnK/dx = (u*nK - jK)*invD
                  djK/dx = -Gamma_K
                  dT/dx  = Tprime

5. chain to s     multiply by dx/ds = 1/(lambda*(1-s_i)) , then divide by scales
```

`T'` is **reconstructed**, not propagated — it is the third RHS component. `q_th` is likewise
reconstructed. Neither is ever an unknown.

## 6. Downstream Jacobian, compactification, and the two tail conditions

With `a = invD*u`, `b = invD`, `c = dGamma_K/dnK`, `d = dGamma_K/dT`,
`e = (1/kappa)*dEflux/dnK`, `f = (1/kappa)*dEflux/dT`, evaluated at `Q` by finite differences
**through the same local closure** used in section 5 (so it can never drift from the ODE):

```
J = [[ a, -b,  0],
     [-c,  0, -d],
     [ e,  0,  f]]

det J = b*(d*e - c*f)        trace J = a + f
```

Required: `c*f > d*e` and `f > 0`  =>  one negative eigenvalue, `k = 1`.
This is the **go/no-go gate**; it is tested standalone before any solver is built.

- `lambda_stable = |Re(negative eigenvalue)|`  -> the compactification rate.
- `W = [w1, w2]`, the **left** eigenvectors of the two growing eigenvalues.
  Left, not right: if `J = sum_i lambda_i * v_i * w_i^T` with `w_i^T v_j = delta_ij`, the
  component of a deviation along `v_i` is `w_i^T*delta`. Killing the growing components is
  `w_i^T (y_b - y_eq) = 0`.

## 7. The four boundary conditions

```
bc(ya, yb, p):
  r1 = ya[2]                                     # T(0) = 0            (scaled)
  r2 = (ya[1]*S_jK - jB) / S_jK                  # jK(0) = jB
  r3 = w1 . (yb_physical - y_eq) / scale         # kill growing mode 1
  r4 = w2 . (yb_physical - y_eq) / scale         # kill growing mode 2
```

`nK(0)` appears in **no** boundary condition. It is fixed only by the global solution — which
is the whole point: `a(0+) = nK(0)/nB(0+)` becomes an output.

`jK(0) = jB` matches the existing convention exactly (line 4405) and follows from
`jK(0-) = u(0-)*nK(0-) = u(0-)*nB(0-) = jB` with `nK(0-) = nB(0-)` for pure neutron matter,
plus continuity of `jK` across the sharp interface.

## 8. Initial guess (uses the verified asymptotics)

In the compactified coordinate the leading behaviour is remarkably simple:

```
T(x) ~ T_Q*(1 - exp(-lambda*x))   =>   y2(s) = s          (exact in the linear tail,
                                                            and T(0)=0 automatically)
nK :  y0(s) = [nK_Q + (nK_0_guess - nK_Q)*(1-s)] / S_nK
jK :  y1(s) = [jK_Q + (jB - jK_Q)*(1-s)] / S_jK ,   jK_Q = u_Q*nK_Q
```

`nK_0_guess` is seeded either from an existing `solve_front_energy_conserving_uNmax` run at
small `TQstar`, or from the analytic `A` boundary. Because `nK(0)` is now an *output*, a poor
seed only costs iterations, not correctness — and seed independence is exactly what test 10
checks.

## 9. Continuation schedule

The exact `T(0)=0` problem is approached, not attacked directly:

```
for T_eps in [0.5, 0.2, 0.05, 0.01, 0.0] MeV:      # BC r1 becomes ya[2] - T_eps/S_T
    solve; use converged (y, p) as the next guess
```

Only BC `r1` changes along the schedule; the ODE, the closure, and the tail conditions are
untouched. **This is a continuation waypoint, not a temperature floor** — the final solve uses
the exact `T(0) = 0` boundary condition, and convergence as `T_eps -> 0` is asserted. At least
two different schedules must give the same answer (test 12).

## 10. Post-solve reconstruction and validation

Nothing is back-filled from the old ideal closure. On a dense `s` grid, re-run the section-5
pipeline to emit every profile: `nK, jK, T, muB, muK, nB, u, kappa_th, q_th`, plus the three
flux profiles `jB(x) = nB*u`, `Pi(x) = P + h*u^2`, `F_E(x) = h*gamma*u - kappa_th*T'`.

Acceptance requires **all** of:

- `max|jB(x)/jB - 1|`, `max|Pi(x)/Pi - 1|`, `max|F_E(x)/F_E - 1|` below tolerance;
- all four BC residuals below tolerance;
- max local-closure residual below tolerance;
- `solve_bvp` status `0` — reaching `max_nodes` is **not** success;
- typed rejection counters reported (a solution built on many rejected nodes is visible);
- reported signs: expect `T'(0+) > 0` and `q_th(0+) < 0`; these are **checked and reported,
  never imposed**. `T'(0+) > 0` requires `h_Q*(gamma*u)_Q* < F_E`; if it fails the solver says
  "no physical solution here" rather than returning a sign-flipped profile.

## 11. Why the plateau ambiguity is structurally gone

Old system: `T` was slaved to `nK`, so `nK = nK(0)` implied `T = 0` implied `invD = 0` implied
`dnK/dx = 0`. The interface state was a genuine **fixed point** of the `nK` flow — solutions
could sit there for an arbitrary distance, and the step size picked the plateau length.

New system: `T` is independent and leaves zero at a rate `T'(0+)` fixed by the *global* BVP, so
(Wolfram-verified)

```
T(x)             ~  Tprime_0 * x
invD             ~  T^(5/3)  ~  x^(5/3)
nK(0) - nK(x)    ~  x^(8/3)          exponent returned exactly as 8/3
d(T^(5/3))/dT    ->  0  as T->0+     => RHS is Lipschitz at the interface
```

There is no fixed point and no waiting-time freedom.

**Caveat that must survive into the report:** `nK` is still numerically *very* flat near `x=0`
(`x^(8/3)`), so visual flatness alone proves nothing either way. The claim "the plateau
ambiguity is resolved" may only be made after tests 10-12 (seed, mesh/domain, continuation
independence) pass — not before.

## 12. Build order

1. **Gate:** standalone check of `c*f > d*e` and `f > 0` at a representative `Q`. Stop if it fails.
2. `kappa_th` / `I_kappa` helpers + zero-limit, asymptote, domain-guard, dimensional tests.
3. New 2x2 local closure + its residual test.
4. Downstream Jacobian / eigen helper + `lambda_stable` + left-eigenvector tail conditions.
5. BVP assembly + continuation driver.
6. Interface and asymptotic tests (BCs, conservation, `T(0)=0`, `T'(0+)`, `x^(8/3)` fit).
7. Uniqueness/robustness tests (seed, mesh, domain, schedule) — these license the physics claim.
8. Full regression; completion report.
