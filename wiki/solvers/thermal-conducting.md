---
summary: solve_front_thermal_conducting — T becomes a propagated ODE field, a(0+) becomes an output, and six facts that cost real debugging.
status: current
updated: 2026-08-17
tags: [solver, core-finding, conduction]
---

# `solve_front_thermal_conducting`

Landed 2026-08-05 in `RMFsolver/phase_velocity.py`. The structural answer to both [[interface-closure]] and [[unmax-degeneracy]].

## The change in one sentence

**$T$ moves from the reconstructed-algebraically column into the propagated-by-ODE column.**

In the existing energy-conserving solver $T(x)$ is never integrated — it is recovered pointwise from $E = h\gamma u$. Once conduction enters, that becomes

$$F_E = h\gamma u - \kappa_{\rm th}T' = \text{const},$$

which contains $T'$. It is no longer an algebraic relation for $T$; it **is** the ODE for $T$.

## Side by side

| | `..._uNmax` | `..._thermal_conducting` |
|---|---|---|
| First integrals | $j_B$, $\Pi$, $E = h\gamma u$ | $j_B$, $\Pi$, $F_E = h\gamma u - \kappa_{\rm th}T'$ |
| Propagated fields | $n_K$, $j_K$ (2) | $n_K$, $j_K$, **$T$** (3) |
| Local algebraic closure | $(\mu_B,\mu_K,T)$ from 3 eqs | $(\mu_B,\mu_K)$ from **2** eqs |
| Interface state at $0^+$ | separate 2×2 solve | **none** |
| Boundary conditions | 3 | 4 |
| $a(0^+)$ | **input** | **output**, selected by the BVP |
| $T(0^+)$ | input through a legacy compatibility argument | fixed to 0 by BC; $T'(0^+)$ is output |

The 2×2 closure is strictly better conditioned: the 3×3 needed the $w=T^2$ trick ([[unmax-low-temperature]]) precisely because its Jacobian goes singular in $T$ as $T\to0$. Here $T$ is not an unknown at all, so that failure mode is structurally absent.

## The go/no-go gate

With $a = \mathrm{invD}\cdot u$, $b = \mathrm{invD}$, $c = \partial\Gamma_K/\partial n_K$, $d = \partial\Gamma_K/\partial T$, $e = \kappa^{-1}\partial E_{\rm flux}/\partial n_K$, $f = \kappa^{-1}\partial E_{\rm flux}/\partial T$:

$$J = \begin{pmatrix} a & -b & 0\\ -c & 0 & -d\\ e & 0 & f\end{pmatrix},\quad \det J = b(de-cf),\quad \mathrm{tr}\,J = a+f.$$

The BC count closes only if the stable manifold has dimension $k=1$, requiring $cf > de$ and $f > 0$. **It passes with ~13 orders of margin** because $d\approx0$ and $e\approx0$ at the $\mu_K=0$ equilibrium.

## Six facts that cost real debugging

1. **The downstream thermal mode is anti-damped.** $\partial E_{\rm flux}/\partial T > 0$, so $f>0$ is a *growing* eigenvalue. The coupled Jacobian has 1 stable + 2 growing directions, which is exactly what makes the BC count close.

2. **A long tail is reachable via a linearized tail, not a truncated domain.** A purely nonlinear compactified solve at `tail_eps=1e-8` (~147 growing e-foldings) gives a singular collocation Jacobian. Capping the domain at ~10 e-foldings works but truncates — it shifts `uN` by ~0.5%. The current method keeps the full compact domain and replaces the RHS with $J\delta$ where $1-s \le 10^{-4}$ (C1 blend over $[10^{-4}, 2\times10^{-4}]$), continuing `tail_eps` $10^{-4}\to10^{-6}\to10^{-8}$. This reaches $x_{\rm end}\sim1.2\times10^7$ with $\mu_K(\infty)\sim3\times10^{-6}$ MeV, and $|\delta|/(1-s)$ is constant to ~1.39 over seven decades — demonstrably on the stable manifold.

3. **$T'(0^+)$ changes sign at $a(0^+)\approx0.6$.** Below it the interface cools into $T<0$ and no solve can start. Conduction requires the fresh quark matter to carry *more* enthalpy flux than the nuclear matter, selecting high $a(0^+)$ (low strangeness). **Seeds must start above the threshold.**

4. **`_default_energy_jB_guess` is unusable here** ($u(0^-) \sim 10^{-8}$ vs the physical $\sim7\times10^{-7}$). The uNmax solver tolerates it; this BVP diverges to overflow. Seed instead from `solve_front_energy_conserving_uNmax` at $T(0^+)=1$ MeV — initialisation only, its ideal energy closure is never used.

5. **The full $I_\kappa$ is implemented**, not the asymptote — see [[quark-transport]] for the log branch, the analytic azimuthal average, and the 16%/13% magnitude correction.

6. **Two traps in the tail scheme:** the deep tail is fragile (at `tail_eps=1e-8`, stepping `T_interface` 2→1 MeV hits a singular Jacobian and only survives via adaptive bisection), and `q_th` must be built from the spline derivative $-\kappa\,dT/dx$, **not** from $F_E - E_{\rm flux}$ — the latter makes the $F_E$ conservation check tautologically zero.

## Interface regularity (Wolfram-verified)

$\kappa_{\rm th}(0^+)$ is finite and positive and $E_{\rm flux}(0^+)\ne F_E$, so $T'(0)$ is finite and nonzero. Then near $x=0$: $T \sim T'_0 x$, $\mathrm{invD} \propto T^{5/3}$, and

$$n_K(0) - n_K(x) \propto x^{8/3}$$

with the exponent returned exactly as $8/3$. Since $d(T^{5/3})/dT \to 0$ as $T\to0^+$, the RHS is **Lipschitz** at the interface — no fixed point, no waiting-time freedom, which is precisely what kills the [[unmax-degeneracy]] plateau.

**Caveat that must survive into any report:** $n_K$ is still numerically very flat near $x=0$ ($x^{8/3}$), so visual flatness proves nothing. The claim "the plateau ambiguity is resolved" may only be made after the seed-, mesh- and schedule-independence tests pass.

## Continuation

$T(0)=0$ is approached, not attacked: `T_eps` over $[0.5, 0.2, 0.05, 0.01, 0.0]$ MeV, only BC $r_1$ changing. This is a **continuation waypoint, not a temperature floor** — the final solve uses exact $T(0)=0$, and at least two schedules must agree.

## No silent clipping

Unlike the uNmax path, invalid trial states ($T<0$, non-convergent closure, $n_B\le0$, $\kappa_{\rm th}\le0$) return a large finite residual and increment a **typed failure counter** surfaced in the result. A "converged" answer built on many rejected nodes is visible rather than hidden.
