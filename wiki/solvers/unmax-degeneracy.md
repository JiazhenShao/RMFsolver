---
summary: At T(0+)=0 the uNmax eigenvalue jB is a non-Lipschitz continuum, not a set of branches — success=True proves nothing there.
status: current
updated: 2026-08-17
tags: [solver, pathology, core-finding]
---

# The uNmax $T(0^+)=0$ plateau degeneracy

Diagnosed 2026-08-04. The most consequential numerical finding in the project — it invalidated a class of cluster results and motivated [[thermal-conducting]].

## The claim

The scattered `jB` values across the 26-08-03 uNmax ray grids are **not** multiple physical branches and **not** different thermodynamic or EOS roots. They are a **one-parameter continuum** produced by a genuinely ill-posed left boundary at $T(0^+) = 0$.

## Why the global state cannot be the culprit

These fronts have $u(0^-) \sim 10^{-7}$, so $\Pi = hu^2 + P$ equals $P(0^-)$ to $2.8\times10^{-13}$, and $E/j_B = h(0^-)\gamma_v(0^-)/n_B(0^-)$ is $j_B$-free. The whole closure is therefore $j_B$-independent: sweeping $j_B = 0.4 \to 3.2$ moves $n_K(0^+)$, $n_B(\infty)$ and $T(\infty)$ by under $10^{-11}$ relative. Two cluster runs at cell (4,2) agree on every thermodynamic quantity to 12 digits while their `jB` differ by 1.75×. **All the multiplicity is in the profile.**

## The mechanism

At $T(0^+) = 0$, `_microphysics_from_quark_state_energy` returns `invD = 0` exactly. With $\Delta = 1 - n_K/n_K(0^+)$, measured exponents are $T \sim \Delta^{0.49}$ and $\mathrm{invD} \sim \Delta^{0.83}$ (predicted $1/2$ and $5/6$), so near the interface

$$\frac{d\Delta}{dx} = K\,\Delta^{5/6},\qquad K > 0.$$

Exponent $< 1$ means the Lipschitz/Osgood condition fails at $\Delta = 0$ — exactly like $dy/dx = y^{2/3}$, $y(0)=0$. The IVP from the interface admits a one-parameter family: $\Delta = 0$ on $[0, x_0]$, then $\sim (x-x_0)^6$, for **any** $x_0 \ge 0$. `jK` keeps draining at $-\Gamma_K$ on that plateau, so any surplus `jB` is simply burned off before the real transit starts, and `jB` is not pinned.

The `local_state` clamp inside `_solve_front_energy_conserving_uNmax_once` turns this into an exact absorbing plateau. Measured $|u n_K - j_K|/j_B = 0.8004 = 1 - a(0^+)$ there — the constitutive law $j_K = u n_K - D\,dn_K/dx$ is violated, the indeterminate $\infty\cdot 0$ silently resolved as 0. **The clamp does not create the degeneracy; it makes the spurious family trivially reachable.**

## Evidence

A 12-point seed scan at cell (4,2) ($T = 3.362$ MeV, $n_B = 3.073\,n_0$, only `jB_guess` varied) gives a *continuum*: `jB` from 0.746 to 1.633 ($u(0^-)$ 59 → 130 m/s), every one with `bvp_status=0`, `bc_max ~ 1e-9`, `kaon_residual ~ 4e-17`. Clamped-plateau nodes grow 1 → 265 and the plateau-drained fraction of `jB` grows 0.00 → 0.45, monotonically with `jB`.

## ⚠ It is ill-posedness, not ill-conditioning

Do **not** try to fix it with precision. The exact residual $R(j_B) = j_K(\text{interface}) - j_B$ is well conditioned ($dR/dj_B = -0.836$). On the plateau `invD` is *exactly* 0.0 in IEEE double at 265 nodes, so those profiles satisfy the discretised system exactly, not merely to tolerance. Confirmed: `tol_bvp` $10^{-4} \to 10^{-7}$ (1000×) removed only 18% of the seed spread, answers stayed monotonically ordered by seed, and all four runs then hit `max_nodes`. Rescaling, extended precision, and working in $u/c$ will not help.

**Consequence: every residual gate is satisfied *on* the continuum, so `success=True` proves nothing here.** The old analytic-seeded run looked smoother only because its seeds were a smooth field; both runs are contaminated (median `jB`/analytic 1.17 old vs 1.28 new).

## A finite interface temperature cures it

Measured 2026-08-04, same cell, same two seeds (0.6288 / 2.4677) that spread 1.42× at $T(0^+)=0$: at $T(0^+) = 4$ MeV both converge to $u(0^-) = 33.7229$ m/s — six digits — in 5–13 s instead of 78–266 s. `invD(0)` rises $0 \to 2.0\times10^{-4} \to 2.1\times10^{-3} \to 3.4\times10^{-2} \to 0.374$ over $T(0^+) = 0, 0.05, 0.2, 1, 4$ MeV; that is the Lipschitz constant of the boundary.

**Caution:** at $T(0^+) = 0.05$–1.0 MeV the high seed *fails outright* (NaN) rather than agreeing. A sub-MeV floor only converts silent garbage into loud failure. Seed-independence needed a few MeV.

$u(0^-)$ falls monotonically with $T(0^+)$ (55.3, 51.8, 44.9, 33.7 m/s at 0.05, 0.2, 1, 4 MeV), roughly linear in $\sqrt{T(0^+)}$, extrapolating to ~58.7 m/s against the reference solver's 63.8 — the same ~8% systematic the BVP shows plateau-free, still unexplained.

## Supersedes

The "the exact-zero result looks sound" conclusion in [[unmax-low-temperature]], which was drawn at a single point where the seed happened to be consistent.

## Fix direction

Reformulate away from $n_K$-in-$x$: with $v = T^{1/3}$ both $dn_K/dv$ and `invD` scale as $v^5$, so $(dn_K/dv)/\mathrm{invD}$ is regular (verified constant to ~2% over three decades in $T$), and the phase-plane graph $j_K(n_K)$ has no translation freedom and no plateau. The route actually taken was [[thermal-conducting]].
