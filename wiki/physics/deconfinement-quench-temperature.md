---
summary: Thermalizing freshly deconfined matter adds entropy, not energy; pure n->udd deconfinement at 3n0, T(0^-)=10 MeV lacks 150 MeV/baryon, and even full conversion cools it 10->6.87 MeV.
status: current
updated: 2026-10-20
tags: [physics, interface, deconfinement, thermalization, closure]
---

# Temperature of freshly deconfined matter

Question (2026-10-20): inside $0^-<x<0^+$, does fast, entanglement-driven thermalization of the newly deconfined quarks raise the temperature, as in heavy-ion collisions?
Result not yet signed off by the user.

## Conclusion

Entanglement thermalization produces entropy, not energy.
In Kaufman et al. (Science 353, 794, 2016) a subsystem's entanglement entropy reaches the thermal entropy *at the energy of the quench*. The temperature is fixed by the conserved energy.
At the slow front the conserved pair is $P$ and $h/n_B$. For massless bag quarks $h=4(P+U)$, so fixing both fixes $\varepsilon$ and $n_B$. The deconfinement step is therefore an isolated quench at fixed energy and baryon density, the direct analogue of the Kaufman setup, and $T(0^+)$ follows from the EOS with no extra heating term.

Inputs: QMC-RMF3 pure neutron matter, $B^{1/4}=189.1566$ MeV, massless $u,d,s$ plus gluons, no electrons. "Pure deconfinement" means flavor frozen by the strong interaction, $n\to udd$ ($a=1$, $\mu_s=0$).

| $n_B(0^-)/n_0$ | $T(0^-)$ | $h/n_B(0^-)$ | cold $udd$ $\mu_B$ at same $P$ | gap | $T_q$ |
|---|---|---|---|---|---|
| 3 | 10 | 1193.29 | 1343.25 | −149.96 | none |
| 3 | 80 | 1320.00 | 1377.16 | −57.16 | none |
| 5.2836 | 10 | — | — | 0 | 0 (threshold) |
| 6 | 10 | 1789.39 | 1732.44 | +56.95 | 45.58 |
| 8 | 10 | 2216.32 | 1978.95 | +237.37 | 93.02 |
| 10 | 10 | 2642.40 | 2200.68 | +441.73 | 126.61 |

All energies are in MeV. "Gap" is $h/n_B(0^-)-\mu_B^{udd}(P)$, and a negative gap means no thermal $udd$ state exists at any $T\ge0$.

- **At the reference state, pure deconfinement is forbidden.** The neutrons lack 150 MeV per baryon of enthalpy needed to make $udd$ matter at their pressure. Deconfinement there needs strangeness already present at the interface ($a(0^+)\le A<1$), supplied by diffusion from $x>0$.
- **Where allowed, the jump is large.** Above $5.28\,n_0$ at $T(0^-)=10$ MeV, the latent heat drives $T_q$ well above $T(0^-)$.
- **Even complete conversion cools at the reference state.** On the same $(P,h/n_B)$, fully equilibrated $uds$ has $T(\infty)=6.868$ MeV $<T(0^-)=10$ MeV. Quark matter has about 2.8 times the heat capacity per baryon ($s/n_B=cT$: $c_N=0.0266$, $c_{uds}\approx0.075$ MeV$^{-1}$), so the same enthalpy spread over more degrees of freedom gives a lower $T$ and a higher $s/n_B$.

## Low-temperature law (Wolfram-verified)

Along $udd$ at fixed $P$,

$$
\frac{h}{n_B}=\mu_B^{udd}(P)+c_2T^2+c_4T^4+O(T^6),\qquad
\mu_B^{udd}(P)=(1+2^{4/3})^{3/4}\,[4\pi^2(P+U)]^{1/4},
$$

$$
c_2=\frac{\mu_{u0}^2+\mu_{d0}^2}{2n_{B0}}=\frac{\pi^2(2+2^{1/3})(1+2^{4/3})}{2^{2/3}(4+2^{2/3})\,\mu_{u0}},\qquad
c_4=\frac{\pi^4(265+48\cdot2^{1/3}+6\cdot2^{2/3})}{408\,\mu_{u0}^3}>0 .
$$

$c_4$ comes from gluons and thermal $s\bar s$ pairs. $T_q$ solves $h/n_B(0^-)-\mu_B^{udd}(P)=c_2T_q^2+c_4T_q^4$.
The exact root agrees with this to a relative error of $1.3\times10^{-8}$ at $T_q=2.36$ MeV, $1.4\times10^{-3}$ at 45.6 MeV and $1.3\times10^{-2}$ at 93 MeV.

The general leading-order rule, for fixed composition and any EOS, follows from $dg=dP/n_B-(s/n_B)\,dT$:

$$
T_q^2\simeq\frac{c_N T(0^-)^2-2\Delta g_0(P)}{c_Q},\qquad \Delta g_0=g_Q(P,0)-g_N(P,0).
$$

So deconfinement heats only if the released latent heat satisfies $-\Delta g_0>\tfrac12(c_Q-c_N)T(0^-)^2$.

## Consequences

- ⚠ **Closure conflict at the reference state.** Along the conserved trajectory $T(0^+)\le T(\infty)=6.87$ MeV, so the [[thermal-transparent-closure]] $T(0^+)=T(0^-)=10$ MeV has no energy-conserving solution there. This agrees with [[interface-temperature-jump]], where positive $\Delta T$ (quark side colder) dominates.
- Where heating happens, it comes from the weak conversion in $x>0$ ($T(0^+)\to T(\infty)$), not from the strong deconfinement step. Rosero-Gil et al. (2020) find 20–60 MeV heating along the flame.
- The $gap=0$ line is where the exact-trajectory $A$ reaches 1, the $u_{N,\max}\propto1/\sqrt{1-A}$ pole of [[analytic-front-speed]].

Follow-up: the entropy budget and second-law bound on the interface temperature drop are in [[interface-second-law]].

## Caveats

Massless $s$ overstates the thermal $s\bar s$ part of $c_4$. Slow-front jump conditions drop $O(u^2)$. The upstream matter is pure neutron matter without electrons. The thermalization time is not computed: the result assumes the layer reaches equilibrium (a strong-interaction time of about 1 fm/$c$).

## Files

`local_analysis_results/26-10-20/deconfinement-quench/`: `quench_temperature.py` and `.json` (table), `checks.py` and `.json` (production-EOS match at the $udd$ point to $10^{-15}$, series checks, threshold, $T(\infty)$), and `PROJECT.md`.
