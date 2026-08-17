---
summary: The composition ceiling is beta_LTE = 5*u(0-)*lambda_n, with an exact-local-isobar correction and a coefficient ambiguity that keeps biting.
status: current
updated: 2026-08-17
tags: [physics, closure, bound]
---

# LTE composition bound

An independent validity constraint on any [[interface-closure|interface closure]]: it caps the downstream strangeness fraction $a(0^+)$ and therefore implies a *lower* bound on $T(0^+)$.

## The bound

$$\beta_{\rm LTE} = 5\,u(0^-)\,\lambda_n,\qquad \lambda_n = \frac{n_B(0^-)}{n_B^{(+)}}.$$

The saturated composition solves

$$(1-\beta_{\rm LTE})\,a_{\rm LTE}^2 + \beta_{\rm LTE}\,a_{\rm LTE} - A^2 = 0,$$

and the numerically stable positive root used in `RMFsolver/phase_velocity.py` is

$$a_{\rm LTE} = \frac{2A^2}{\sqrt{\beta_{\rm LTE}^2 + 4(1-\beta_{\rm LTE})A^2} + \beta_{\rm LTE}}.$$

On the analytic constant-pressure trajectory this gives

$$T_{\rm LTE}(0^+) = T(\infty)\sqrt{1 - [a_{\rm LTE}(0^+)/A]^2}.$$

## ⚠ The coefficient trap

**`beta_LTE` is not 5.** The code sets the *coefficient multiplying* $u_N\lambda_n$ to 5. The user confirmed on 2026-07-23 that the intended definition is exactly

```text
beta_LTE = 5*lambda_n*u(0-)
```

obtained by rewriting `A^2 - a_LTE^2 = 5*lambda_n*u(0-)*a_LTE*(1 - a_LTE)`.

This has been misread more than once. Any new code or payload metadata must spell it out — the 26-07-23 scan carries `lte_beta_definition = "beta_LTE_equals_5_times_u_N_times_lambda_n"` for exactly this reason.

## Exact local isobar correction (2026-07-23)

The original scan used the analytic $A$ boundary as the composition ceiling. That was corrected: the ceiling should be the **exact local EOS fraction** on the downstream endpoint's zero-temperature, fixed-$\mu_{B,Q}$, constant-pressure isobar. Solve

```text
PQM(muB_Q, muK_A, B14, T=0) = P_Q
```

then take `A_boundary_exact_local = nK_QM/nB_QM` at that root, and recompute $a_{\rm LTE}$ from it. Method: bracket from $\mu_K = 0$, expand geometrically until $f$ changes sign, then `scipy.optimize.brentq`.

Both boundaries are retained in the payload — the analytic `A_boundary` still defines the velocity closure and the gray boundary; `A_boundary_exact_local` is used only for the LTE jump target.

## Where the coefficient 5 might change

The superseded 2026-07-22 spec proposed generalising the coefficient to $\max(5, 3\,\mathrm{Le})$ where Le is a Lewis-number-like ratio of thermal to strangeness transport mean free path, making the thermal bound binding when $\mathrm{Le} > 5/3$. **That derivation is not safe to use** — see [[quark-transport]] for why the shared-$\tau$ assumption it rests on is wrong.

## Code facts

- `phase_velocity._analytic_a_0plus_lte` (renamed from `_analytic_aqstar_lte`) hard-wires `5.0` — verified 2026-08-17: `beta = float(5.0 * u_0minus * lambda_n)`.
- ⚠ The LTE path is now its own function, **`analytic_velocity_bound_lte(muB_0minus, T_0minus, B_one_forth, *, xi=0.0, ...)`**. `interface_fraction_mode` is no longer a parameter — passing it raises `TypeError`. It survives only as an *output* key echoing `interface_control`.
- Verified keys (2026-08-17): `T_inf`, `A_boundary`, `a_0plus_LTE`, `beta_LTE`, `u_0minus_max`, `lambda_n`. The older note cited `T_Q`, `a_interface_LTE` and `u_N` — those names no longer exist; the result dict now uses location-based endpoint notation throughout. See [[phase-velocity-overview]].
- It does **not** return the LTE-limited interface temperature; callers must compute it from the downstream temperature and the two composition values.

## Related

- [[interface-temperature-jump]] — the observable built on this bound
- [[two-block-contour-scans]] — the mesh it is scanned over
