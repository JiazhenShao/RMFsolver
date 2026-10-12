---
summary: Mathews/Rupp/Boccioli's ms-scale supernova core oscillations assume equilibrium (instantaneous) conversion in the coexistence region; our laminar speeds move an interface only ~m per ms, so the effect needs dense nucleation.
status: current
updated: 2026-10-05
tags: [paper, literature, supernova, context]
---

# Supernova core oscillations from a quark coexistence region

Forwarded by Liam Brodie on 2026-10-05: APS DNP talk (Oct 2026) "Dynamics Core Collapse Supernovae with a Transition to Quark Matter", G. Mathews (presenter), O. Rupp, L. Boccioli. No matching arXiv paper found as of that date.

## Their claim

1D GR supernova models with STIR turbulence, 14--40 $M_\odot$ solar-metallicity progenitors. Most explode by neutrino-driven convection before the phase-transition second shock or oscillations. At least two models oscillate. The driver is the coexistence region having adiabatic index $\Gamma<4/3$ relatively far out in the star; the mode is a first harmonic with its node where $\Gamma$ crosses $4/3$. Semi-analytic models for onset and frequency are promised.

## Relevance to this project

The $\Gamma<4/3$ coexistence region comes from an **equilibrium** hybrid EOS (Gibbs/Maxwell, e.g. DD2F-SF in this group's earlier GR1D work). That is the rapid-conversion limit: matter in the coexistence region converts within one oscillation period. Our front speed is the microscopic rate that decides whether this limit holds.

Over one oscillation period $P_{\rm osc}$ (ms-scale in comparable simulations), a laminar interface advances

$$\ell_{\rm conv}\simeq u_N P_{\rm osc}\approx 0.03\text{--}3\ {\rm m}\quad(u_N=25\text{--}3415\ {\rm m/s},\ P_{\rm osc}=1\ {\rm ms}).$$

So the equilibrium $\Gamma$ applies only if converting interfaces (nucleated droplets) are spaced by $\lesssim\ell_{\rm conv}$. Otherwise the oscillating matter responds with the frozen-phase stiffness and the slow-conversion junction conditions apply (Pereira et al. 2018), which can remove or relocate the $\Gamma=4/3$ node. This is the supernova analogue of the postmerger factor $300$--$10^4$ in [[front-speed-phenomenology]] and [[gw-observables-section]].

The same weak channel $d+u\leftrightarrow u+s$ that sets our speed also gives quark-matter bulk viscosity, which damps these oscillations.

## ⚠ Regime mismatch before quoting our numbers

Supernova cores have trapped neutrinos, proton fraction $\sim0.3$, entropy a few $k_B$/baryon and $T\sim10$--$50$ MeV. The baseline front assumes pure neutron matter and no neutrinos ([[diffusion-limited-front]]). Temperature is covered by the contours; composition and neutrino trapping are not. At high $T$ bulk thermal nucleation is plausible, so the relevant quantity is the effective volume-conversion rate (nucleation density times interface speed), not $u_N$ alone.

## Related literature

- Fischer et al. 2011, ApJS 194, 39 — mixed phase lowers $\Gamma$, PNS collapse, second neutrino burst.
- Zha, O'Connor et al. 2021, ApJ 911, 74 — second bounce, $\sim1$ ms oscillatory neutrino signal; Lin et al. 2022, PRD — its detectability.
- Jakobus et al. 2022, MNRAS 516, 2554 — 97 models; DD2F-SF explodes only low-compactness cases.
- Couch et al. 2020, ApJ 890, 127 (STIR); Boccioli et al. 2021, ApJ (GR STIR in GR1D; EOS dependence).
- Pereira, Flores & Lugones 2018, ApJ 860, 12 — rapid vs slow interface conversion in radial oscillations.
- Rupp, Mathews, Boccioli, APS DNP 2025 — explodability with DD2F-SF + STIR (abstract only).
