---
summary: Heiselberg–Pethick transport for degenerate quark matter — thermal conductivity, flavor diffusion, the I_kappa integral, and why they do NOT share a relaxation time.
status: current
updated: 2026-08-11
tags: [physics, transport, literature]
---

# Quark transport (Heiselberg–Pethick)

Everything in [[interface-closure]] and [[thermal-conducting]] rests on these coefficients. Source: [[heiselberg-pethick-1993]].

The explicit flavor-diffusion coefficient and weak K-changing source used by the combustion model are cataloged together in [[strangeness-reaction-diffusion]].

## ⚠ Thermal diffusivity is NOT the flavor-diffusion coefficient

**Do not use $D_K$ as if it were the thermal diffusivity.** HP93 calculate *different* relaxation times for different transport processes. In the low-temperature degenerate regime the thermal-relaxation rate has a different temperature dependence from the momentum/flavor transport rate.

This correction on 2026-07-23 is what superseded `docs/superpowers/specs/2026-07-22-thermal-interface-jump-bound-design.md`. That spec assumed both channels shared a single Landau-damped $\tau \propto T^{-5/3}$, which made the Lewis number a pure $O(1)$ ratio of transport integrals. The assumption does not hold, so the $3\,\mathrm{Le}$ generalisation of the [[lte-composition-bound]] coefficient must be reconciled before any use.

## Thermal relaxation

For unpaired degenerate quark matter at $T \ll q_D$:

$$\frac{1}{\tau_\kappa} = \frac{4}{\pi^3}N_f\alpha_s^2\mu_q^2\cdot\frac{2\zeta(3)T}{q_D^2},\qquad q_D^2 = \frac{2N_f\alpha_s\mu_q^2}{\pi}.$$

Wolfram simplification gives

$$\frac{1}{\tau_\kappa} = \frac{4\zeta(3)\alpha_s T}{\pi^2},\qquad \chi_Q = \frac{\kappa_Q}{c_{V,Q}} = \frac{\tau_\kappa}{3} = \frac{\pi^2}{12\zeta(3)\alpha_s T}.$$

At $\alpha_s = 0.3$, $T = 10$ MeV: $\chi_Q = 0.228072\ \mathrm{MeV^{-1}} = 45.005$ fm, $\tau_\kappa = 0.684216\ \mathrm{MeV^{-1}} = 135.014$ fm.

## Why the transparent interface is defensible

Steady advection–diffusion $\chi_Q T'' - vT' = 0$ has thermal length $\ell_T = \chi_Q/v$ with $v = u/\sqrt{1+u^2}$. At $v = 10^{-6}$, $\alpha_s = 0.3$, $T = 10$ MeV this gives $\ell_T \approx 45$ nm. For a microscopic layer of thickness $\delta_{\rm int} = 1\text{–}10$ fm,

$$\frac{v\,\delta_{\rm int}}{\chi_Q} \approx 2\times10^{-8}\text{–}2\times10^{-7},$$

supporting negligible temperature variation across the layer — **conditional on the absence of a large interfacial thermal resistance**, which is the unresolved part ([[kapitza-resistance]]).

## Thermal conductivity for the solver

$$\kappa_{\rm th}(T,\mu_q) = \frac{\pi^3 v_F^2 T^2}{24\alpha_s^2 I_\kappa(T/q_D)},\qquad q_D^2 = \frac{2N_q\alpha_s\mu_q^2}{\pi}.$$

`_TRANSPORT_QD_COEFF` in the code already equals $\sqrt{6\alpha_s/\pi}$ to machine precision — reuse it, do not recompute.

Verified $T\to0^+$ limit (Wolfram), using $I_\kappa(y) \sim 2\zeta(3)y^2$:

$$\lim_{T\to0^+}\kappa_{\rm th} = \frac{\pi^3 v_F^2 q_D^2}{48\zeta(3)\alpha_s^2},$$

finite and positive — no $0/0$ is ever evaluated.

## The $I_\kappa$ integral — status

The **full** $I_\kappa$ **is implemented** (HP93 Eq. 59 plus the Eq. 7 susceptibilities), not merely the Eq. 60 asymptote. Two facts that cost debugging:

- The log branch is $\ln\!\big((1+x)/(1-x)\big) + i\pi$, fixed by Eq. (8) requiring $\chi_t \to i\pi x/4$.
- The azimuthal average is analytic: $\langle(1-\cos\phi)|A - B\cos\phi|^2\rangle = |A|^2 + |B|^2/2 + \mathrm{Re}(A\bar B)$, which correctly reduces to the Eq. (21) momentum-relaxation integrand when the $(1-\cos\phi)$ weight is dropped.

**In our domain the full integral runs up to 16% above the low-$y$ asymptote, so $\kappa_{\rm th}$ is ~13% smaller — not the ~2% originally guessed.**

HP93's quoted high-$y$ constant 0.30 appears to be an analytic estimate rather than the numerical curve; converged quadrature gives 0.388 (slope in $\ln y$ matches $1/3$ exactly). Appendix B documents the same kind of gap for $I_s$ (2.810 analytic vs 2.72 numerical). Irrelevant for the solver ($y \lesssim 0.19$) but relevant if the asymptote is quoted in the paper.

## Domain

$y = T/q_D < 0.25$ across the entire parameter range; for solutions of interest ($T$ up to a few MeV), $y < 0.05$. The $y \gg 1$ logarithmic asymptote is unreachable — it needs $T \gg 280$ MeV.

| $\mu_q$ (MeV) | $q_D$ (MeV) | $y$ at $T{=}1$ | $y$ at $T{=}10$ | $y$ at $T{=}50$ |
|---|---|---|---|---|
| 300 | 227.1 | 0.0044 | 0.0440 | 0.2202 |
| 370 | 280.1 | 0.0036 | 0.0357 | 0.1785 |
| 450 | 340.6 | 0.0029 | 0.0294 | 0.1468 |

The paper PDF is at `~/Documents/GRAD_STUDY/Research/Dense_Matter/ref[35]-D_Q.pdf`.
