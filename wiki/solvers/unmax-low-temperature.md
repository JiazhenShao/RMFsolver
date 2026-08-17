---
summary: Two independent low-T problems in solve_front_energy_conserving_uNmax — the logT→T² closure fix (landed) and the nK-resolution limit (open).
status: current
updated: 2026-08-17
tags: [solver, numerics]
---

# uNmax low-temperature limit

Investigated 2026-07-29. Two independent problems near $T(0^+) = 0$; one fixed, one structural.

## Fixed — the `logT` → $w = T^2$ closure

`_solve_local_quark_state_from_nK_E_and_Pi` solved for $(\mu_B, \mu_K, \log T)$. Because $E$ and $\Pi$ are analytic in $T^2$, the residual's sensitivity to $\log T$ scales as $T^2$ while the acceptance gate sat fixed at $10^{-7}$. Below $T \sim 0.05$ MeV the solve therefore **returned its own seed** — five seeds gave $4\times10^{-191}$ to $2.7\times10^{-10}$ for identical inputs — and for $n_K$ above its $T=0$ limit it *fabricated* a state ($T \sim 10^{-103}$) rather than failing. Since the ODE carries the seed from the previous collocation node, the BVP right-hand side was not a function of the state.

**Fix:** the third unknown is now $w = T^2$, which keeps the Jacobian regular at $w = 0$ and makes $w < 0$ an explicit infeasibility. Tolerances were deliberately left alone — results are identical at $10^{-7}$, $10^{-10}$ and $10^{-12}$, so the reparameterisation alone carries the fix.

The closure is **shared**: four call sites in `_solve_front_energy_conserving_nK_once` and two in `_solve_front_energy_conserving_uNmax_once`, so one fix covers both entry points. `_solve_interface_state_from_local_a_E_and_Pi` had the identical bug and got the same treatment.

`_solve_analytic_inf_endpoint_for_u_0minus` (renamed from `_solve_analytic_downstream_endpoint_for_uN`) **still uses `logT` and was deliberately left alone**: its $T(\infty)$ never falls below about 9.7 MeV across the band, against a degeneracy threshold near 0.026 MeV, and it feeds all three analytic entry points and the published contour. Verified 2026-08-17: it is the only surviving `logT` parameterisation in the file.

## Not fixed — formulation, not a bug

The BVP uses absolute $n_K$ as its state variable. The interface layer has width $n_{K,\rm zero} - n_K(0^+) = [T(0^+)/c]^2$ with $c \approx 0.0265$, against $n_K \sim 3\times10^6$, while `scipy.solve_bvp` probes $\partial f/\partial n_K$ with a forward step of $\sqrt{\epsilon}\,(1+|y|)\,n_{K,\rm scale}$.

The layer is thinner than the probe once $T(0^+) < c\sqrt{\text{step}} \approx 0.008$ MeV, so $T(0^+) \approx 0.001$–0.005 now **fails outright**. Before the closure fix those same cases reported `success=True` with `jB ~ 3.02–3.03` — values that break the monotonic trend of the trusted sequence. Silent garbage became loud failure, which is the improvement.

The real remedy is to make $w = T^2$ (or $n_{K,\rm zero} - n_K$) the BVP unknown instead of $n_K$: a forward difference in $w$ always steps into the feasible region, whereas in $n_K$ it steps off the domain.

## ⚠ $j_B[T(0^+)]$ is not analytic at zero

It varies as $\sqrt{T(0^+)}$, i.e. $w^{1/4}$ — **not** as $w = T^2(0^+)$. At $T(0^-)=10$ MeV, $n_B(0^-)=3n_0$, and $B^{1/4}=180$ MeV, a fit linear in $\sqrt{T(0^+)}$ over the trusted points has 5× smaller residual than one linear in $w$ ($9.1\times10^{-3}$ vs $4.6\times10^{-2}$).

**Do not extrapolate this branch in $T^2(0^+)$.**

That fit extrapolates to $j_B(0) = 3.122$ against the direct exact-zero solve's 3.182 — 1.9%. This page originally concluded the exact-zero result therefore "looks sound". **That conclusion is superseded by [[unmax-degeneracy]]**, which shows the exact-zero point sits on a continuum and the agreement was a single lucky seed.

## Also

`max_nodes=800` starves this branch ($T(0^+)=0$ needs ~1900 nodes), and `_u0minus_max_collocation_status_is_acceptable` (renamed from `_uNmax_collocation_status_is_acceptable`) ignores its `exact_zero_left` argument — its body is just `return bool(solver_success)` — making `accepted_max_nodes` and its message unreachable. Still true 2026-08-17.
