---
summary: The numerical front currently uses an approximate low-temperature Madsen weak rate, not the full phase-space integral; alpha_s=0.3 is a phenomenological input.
status: current
updated: 2026-10-07
tags: [physics, weak-rates, diffusion, transport]
---

# Strangeness reaction and diffusion

The microscopic competition that sets the laminar front speed is:

- nonleptonic weak relaxation $d+u\leftrightarrow u+s$, which removes positive $\mu_K$;
- strong-interaction diffusion of the down--strange imbalance back toward the interface.

## Weak source

For positive $\mu_K$, the active manuscript uses the Madsen form

$$\Gamma_K=\mathcal A_K\mu_u^5\mu_K(\mu_K^2+4\pi^2T^2),$$

$$\mathcal A_K=\frac{16}{5\pi^5}G_F^2\sin^2\theta_C\cos^2\theta_C.$$

In the quadratic [[quark-matter-eos|EOS]] approximation, with `mu_u = mu_B/3`, this becomes

```text
Gamma_K = n_B*gamma_K*(a^3 + 81*pi^2*T^2*a/mu_B^2)
gamma_K = 128*G_F^2*sin(theta_C)^2*cos(theta_C)^2*mu_B^5/(3^8*5*pi^3)
```

The manuscript now writes the thermal factor explicitly and uses the baryon chemical potential throughout; this replaces the former common quark chemical potential and thermal-factor shorthand without changing the rate.
At constant temperature and baryon chemical potential, integrating from zero to the interface fraction gives `I2 = D_K*gamma_K*a(0+)^2*(a(0+)^2 + 162*pi^2*T^2/mu_B^2)/4`.
The cubic rate term dominates for `a(0+) >> 9*pi*T/mu_B`, not `3*pi*T/mu_B`; the latter was a factor-of-three error corrected on 2026-09-23.
Wolfram verified the rate substitution and the integral exactly.

The cubic term is suprathermal; the term linear in $a$ is the finite-temperature subthermal contribution. The earlier description of the numerical calculation as using a full rate is superseded by the 2026-10-07 source/code check below: it evaluates the same approximate polynomial with the neutral-EOS chemical potentials.

### Rate approximation audit (2026-10-07)

The `Madsen:1993xx` entry correctly identifies J. Madsen, Phys. Rev. D 47, 325--330 (1993), DOI `10.1103/PhysRevD.47.325`. Indexed text of the original paper, pp. 327--329, distinguishes the exact zero-temperature, zero-strange-mass expressions (20)--(21), their small-imbalance limit (22), and the low-temperature net-rate approximation (23). Section IV states the equal-up/down-chemical-potential and near-equilibrium assumptions and discusses corrections for unequal chemical potentials. The finite-temperature cubic-plus-linear structure used in B1 is the Eq. (23) approximation, not an exact arbitrary-composition rate. A small imbalance relative to the Fermi scale need not be small relative to temperature.

`RMFsolver/phase_velocity.py`, `_exact_kaon_transport_rate`, obtains the neutral chemical potentials, checks their down/strange difference, then directly evaluates the polynomial with the up-quark fifth-power prefactor. It does not integrate weak-interaction phase space. Solving the EOS exactly therefore does not remove the rate approximation, and the spectator-quark explanation alone does not justify the prefactor away from nearly equal flavor chemical potentials. No rate error across saved front profiles has been measured in this audit; that remains an open validation task, especially away from equilibrium.

Source-access limitation: the original PDF could not be downloaded or visually inspected; equation numbering and assumptions were checked against its indexed text. The same polynomial and normalization are independently reproduced in Eqs. (17)--(18) of [Xu et al., Chin. Phys. C 45, 015103 (2021)](https://cpc.ihep.ac.cn/fileZGWLC/journal/article/zgwlc/2021/1/PDF/CPC-2020-0270.pdf). The [original article record](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.47.325) confirms the citation metadata; manuscript and code were not changed.

The follow-up [[unequal-flavor-weak-rate]] derives the massless collision integral and quantifies the zero-temperature error on the neutral composition curve.
The full rate divided by Eq. (23) is 0.93249 at a=0.146, 0.72499 at a=0.5 and 0.35942 at a=0.9; Wolfram also verified the a=1 endpoint analytically.
These are cold composition benchmarks, not finite-temperature profile errors.
Direct inspection of the subsequently supplied Madsen PDF showed that neither printed cold branch (20)/(21) directly covers our neutral conversion ordering.
See [[massless-finite-temperature-weak-rate]] for the branch correction, the verified Eq. (24), and a reduction to two numerical dimensions at finite temperature.

## Flavor diffusion

For unpaired degenerate quark matter,

$$D_K^{-1}=\frac{8N_f\alpha_s^2}{\pi}
\left(h_D\frac{T^{5/3}}{q_D^{2/3}}+\frac{\pi^3T^2}{12q_D}\right),$$

with $N_f=3$, $\alpha_s=0.3$, $h_D\simeq1.81$ and

```text
q_D^2 = g^2*N_f*mu_B^2/(18*pi^2)
g^2 = 4*pi*alpha_s
```

The $T^{5/3}$ term comes from Landau-damped transverse gluons and dominates the analytic low-temperature treatment. The full numerical calculation keeps both displayed terms because they become comparable in the merger-temperature range.

The fixed value $\alpha_s=0.3$ is a conventional phenomenological choice in bag-model studies, represented by the parameter survey of Farhi and Jaffe and the review by Madsen. It should not be described as a controlled value obtained by running the vacuum QCD coupling directly to a scale of 400 MeV; modern perturbative calculations find reasonable convergence only at substantially higher quark chemical potentials.

This $D_K$ is **flavor diffusion**, not thermal diffusivity. The two relaxation channels do not share a common time; see [[quark-transport]].

## Modeling limits

- The nonleptonic channel is the fastest flavor process in the stated regime. Urca channels are omitted and generally add slower equilibration requirements, so the computed laminar speed is best interpreted as an upper bound.
- The single-$D_K$ K equation follows from neglecting cross-diffusion terms and taking the down and strange diagonal diffusion coefficients equal. The older `Notes.tex` derivation flags a missing independent flavor combination; this reduction is an assumption, not a general three-component transport theorem.
- Pairing changes both the weak rates and transport and is outside the model.
- `Notes.tex` contains an order-of-magnitude argument that neutrinos neither trap nor cool efficiently across the microscopic layer, but the active manuscript still lists neutrino transport as future work. Treat neutrino irrelevance as plausible, not settled.
- The [[lte-composition-bound]] is needed before taking $T(0^+)\to0$, where $D_K$ and the mean free path diverge.

## Former EOS mismatch and remaining rate limitation

The former mismatch between the manuscript EOS and the neutral numerical EOS was resolved in Appendix A on 2026-10-05; see [[quark-matter-eos]]. The use of $\mu_u\simeq\mu_B/3$ in the analytical approximation is explicitly leading order. A separate limitation remains: the numerical weak-rate polynomial is itself approximate, even when evaluated with exact neutral chemical potentials.

## Speed impact of the exact rate and the NLO correction (2026-10-07)

Speed scales as $\sqrt{I_2}$, so what matters is $I_2=\int_0^{a(0^+)}\Gamma_K\,da$, not the local rate.
The numbers below are isothermal estimates at fixed $\mu_B=900$ or 1200 MeV, $T=0$ to 80 MeV.
They come from the exact tree-level rate of [[massless-finite-temperature-weak-rate]], not from BVP re-solves.
The rows compare the exact rate with the two rates now in use.
"Numerical" is Madsen's Eq. (23) at exact neutral potentials, as in `_exact_kaon_transport_rate`.
"Analytic" is the linearized form $n_B\gamma_K(a^3+81\pi^2T^2a/\mu_B^2)$.

| $a(0^+)$ | 0.3 | 0.5 | 0.7 | 0.9 | 0.99 |
|---|---|---|---|---|---|
| speed change, numerical → exact | −14 to −3% | −16 to −6% | −20 to −10% | −30 to −19% | −40 to −27% |
| speed change, analytic → exact ($T\le0.035\mu_B$) | −9 to −3% | −8 to −2% | −4 to +3% | +5 to +14% | +17 to +28% |

- **Analytic rate.** Its error is small by cancellation: Madsen's overestimate offsets the $\mu_K$ linearization. It degrades at $T/\mu_B\approx0.09$, where the exact speed exceeds the analytic one by up to 45%.
- **Boundaries.** The zero-speed and $A=1$ boundaries are thermodynamic and do not move.
- **Analytic vs numerical comparisons.** These mix the two rate models. At $T=0$, the rate alone gives the numerical path 1.11, 1.62 and 2.15 times the analytic speed at $a(0^+)=0.5$, 0.9 and 0.99.

**NLO.** Heikinheimo et al., arXiv:2609.35165 (local PDF `ref-list/Rate-NLO-2609.35165v1.pdf`), compute the $O(\alpha_s)$ correction to the linear-response coefficient for $|\delta\mu|\ll T\ll m_E\ll\mu$ and $\mu_u=\mu_d$.
The correction is $\Gamma^{(1)}/\Gamma^{(0)}=\alpha_s(-0.7366\ln\alpha_s+0.4943)$, which is +41% at $\alpha_s=0.3$ and +47 to +52% at $\alpha_s=0.4$ to 0.6.
Their tree level, $64N_c^2/(45\pi^3)$, equals Madsen's $\mathcal C_K\cdot4\pi^2$.
The calculation does not cover suprathermal $\mu_K\gg T$, unequal Fermi momenta, or $\mu_q\approx300$--$400$ MeV, where $\alpha_s(2\mu_q)\approx0.5$.
Applied as a uniform K-factor, it gives speeds about +19% at $\alpha_s=0.3$.
That partly cancels the tree-level correction: exact tree level times K lies within about ±17% of the current numerical speeds for $a(0^+)\le0.9$, and 13 to 29% below them at 0.99.
The 2026-10-07 assessment was to keep both rates in the code and analytics, and to revise only the appendix text.

**Adiabatic universality (checked 2026-10-11).** The standalone note's Sec. 8.6 gives identical adiabatic speed ratios at every $T(0^-)$, for example Eq. (23) at +3.70%, +16.66% and +56.11% for $a(0^+)=0.146$, 0.5 and 0.9.
This is an exact scaling of the model, not a bug.
For massless quarks plus a bag constant, $h=4(P+U)$, so conserved $P$ and $h/n_B$ fix $n_B$.
With $T(0^+)=0$, the invariant $(P+U)/n_B^{4/3}=\tfrac{\pi^{2/3}}{4}\{1+(1+a)^{4/3}+(1-a)^{4/3}\}$ depends on $a(0^+)$ alone, which fixes the whole trajectory in units of $n_B^{1/3}$.
Rates scale as $n_B^{8/3}$ and $D_K$ as $n_B^{-1/3}$ at fixed $\alpha_s$, so $I_2$ ratios cancel the scale.
An independent solve reproduces their $T(\infty)/n_B^{1/3}$ to 10 digits.
A nonzero $T(0^+)$, $m_s$, a running $\alpha_s$, or pairing would break the scaling. Isothermal ratios differ because $T/\mu_B$ varies.

With $T(0^+)=0$, $a(0^+)$ is fixed by one upstream number through $[1+(1+a)^{4/3}+(1-a)^{4/3}]^3=(h/n_B)^4/[4\pi^2(P+U)]$.
That number is 27 at $a=0$ and $(1+2^{4/3})^3\approx43.6$ at $a=1$.
At $T(0^-)=100$ MeV the thermal enthalpy $Ts/n=220$--$370$ MeV per baryon keeps it at or above 38.58, its minimum, reached at $1.444n_0$.
Hence $a(0^+)\ge0.879$ there and there is no zero-speed boundary. The window is physical for that closure, not a solver failure.

## Fermi-surface literature form (2026-10-08)

The 2↔2 Fermi-surface ("phase-space decomposition") rate $\Gamma=T^3\mathcal I(\xi)\mathcal A/(4096\pi^8)$, with $\mathcal I=\prod k_{F,i}\,(4\pi^2\xi+\xi^3)/[6(1-e^{-\xi})]$ and $\xi=\delta\mu/T$, has three published forms:
- Eqs. (A8)–(A11) of Ghosh, Haber, Andersson and Counsell, arXiv:2608.07311, for nonleptonic hyperon rates.
- Eq. (14) of Alford and Haber, PRC 103, 045810 (2021).
- Madsen's Eq. (23), which is its net form, $\propto\delta\mu(\delta\mu^2+4\pi^2T^2)$.

Ghosh et al. and Alford–Haber take the decomposition from Shapiro and Teukolsky (1983), Yakovlev et al., Phys. Rep. 354, 1 (2001), and Kaminker, Yakovlev and Haensel, ApSS 361, 267 (2016).
Madsen credits Eq. (23) to Heiselberg, Baym and Pethick, and to Koch, both in the 1991 Aarhus proceedings; Heiselberg, Phys. Scr. 46, 485 (1992) is the analytic quark version.
Alford and Haber report that at merger temperatures the Fermi-surface approximation overestimates the one-meson-exchange hyperon rate by up to 10×.

For our quark rate, the Fermi-surface approximation with unequal Fermi momenta in the angular factor ($k_F=\mu$, two-triangle $k$ overlap) gives cold FSA/exact ratios of 1.012, 0.985, 0.683, 0.259 and 0 at a = 0.146, 0.5, 0.9, 0.99 and 1.
Madsen's Eq. (23) gives 1.072, 1.379, 2.782, 5.272 and 9.119 at the same points.
It is accurate to about 1.5% through a = 0.5, much better than Eq. (23) there.
It collapses as the strange Fermi sphere vanishes, because it ignores the $\mu_K$-wide energy shell.

## Sources

- `Latex_writting/Main.tex`, current manuscript: “Strangeness creation” and “Strangeness diffusion”; the hydrodynamical-combustion subdirectory also contains an older manuscript copy.
- `Latex_writting/Hydrodynamical combustion/Notes.tex`: “Basis Change from Flavors to Kaoness” and the neutrino discussion; these are working notes rather than final manuscript claims.
- Farhi and Jaffe, *Phys. Rev. D* **30**, 2379 (1984), for the classic bag-model survey including $\alpha_s=0.3$.
- Madsen, *Lect. Notes Phys.* **516**, 162 (1999), arXiv:astro-ph/9809032, for the standard strange-quark-matter review.
