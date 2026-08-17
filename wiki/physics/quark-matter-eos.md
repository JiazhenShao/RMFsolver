---
summary: The working quark phase is an unpaired massless three-flavor bag-model gas; its quadratic expansion makes the K fraction linear in mu_K and produces the analytic elliptical isobar.
status: current
updated: 2026-08-11
tags: [physics, equation-of-state, quark-matter, bag-model]
---

# Quark-matter EOS used by the front model

The downstream model is deliberately minimal: massless, noninteracting $u$, $d$ and $s$ quarks plus a bag energy $U_{\rm bag}$, with no electrons or electromagnetic sector. In production scans the bag constant can be tuned so zero-temperature neutron matter and beta-equilibrated quark matter meet at a chosen critical pressure. Reference calculations also hold $B^{1/4}$ fixed, commonly at 180 MeV; a draft figure discussion mentions about 189 MeV. These are distinct parameterizations, not interchangeable values or a universal prediction.

## Chemical-potential basis

With $\mu_q=\mu_B/3$,

$$\mu_u=\mu_q,\qquad \mu_d=\mu_q+\frac{\mu_K}{2},\qquad \mu_s=\mu_q-\frac{\mu_K}{2}.$$

The pressure is

$$P_{\rm QM}=P_q(\mu_q,T)+P_q(\mu_q+\mu_K/2,T)+P_q(\mu_q-\mu_K/2,T)-U_{\rm bag}.$$

The active manuscript retains the fourth-order expansion as a diagnostic of what the analytic approximation drops:

$$
\begin{aligned}
P_{\rm QM}={}&-U_{\rm bag}+\frac{\mu_B^4}{108\pi^2}
+\frac{\mu_B^2\mu_K^2}{12\pi^2}+\frac{\mu_K^4}{32\pi^2}\\
&+\frac{\mu_B^2T^2}{6}+\frac{\mu_K^2T^2}{4}+\frac{19\pi^2T^4}{36}+\cdots.
\end{aligned}
$$

## Quadratic analytic limit

For $\mu_K,T\ll\mu_q$,

$$P_{\rm QM}\simeq-U_{\rm bag}+\frac{\mu_B^4}{108\pi^2}+\frac{\mu_B^2\mu_K^2}{12\pi^2}+\frac{\mu_B^2T^2}{6},$$

$$n_K\simeq\frac{\mu_B^2\mu_K}{6\pi^2},\qquad n_B\simeq\frac{\mu_B^3}{27\pi^2},\qquad a\equiv\frac{n_K}{n_B}\simeq\frac{9\mu_K}{2\mu_B}.$$

This linear $a$--$\mu_K$ relation is why a fixed-$P$, fixed-$\mu_B$ conversion trajectory is a quarter ellipse in either $(\mu_K,T)$ or $(a,T)$. It underlies [[analytic-front-speed]] but is not substituted for the full EOS in the numerical closure.

## Domain and interpretation

- Equilibrated quark matter has $\mu_K=0$ and $n_K=0$ in this model.
- Positive $\mu_K$ represents an excess of down over strange quarks and relaxes through the nonleptonic process in [[strangeness-reaction-diffusion]].
- The upstream QMC-RMF3 phase has no strange chemical potential; assigning a nuclear-side $\mu_K$ is a convention and should be avoided.
- Masses, pairing, leptons, charge screening and interactions in the EOS are outside the present approximation.

## ⚠ Charge-neutrality mismatch

The active rate appendix says the exact numerical weak rate obtains $\mu_u$ using an electric chemical potential fixed by charge neutrality, while the EOS section explicitly ignores electromagnetism and leptons and sets $\mu_u=\mu_B/3$. The manuscript comments recognize this mismatch. Until the numerical and stated models are reconciled, distinguish the analytic choice $\mu_u\simeq\mu_B/3$ from any code path that solves for $\mu_Q$.

## Source

`Latex_writting/Hydrodynamical combustion/Main.tex`, first active document: “Bag model of quark matter” and the model assumptions in “Boundary propagation speed.”
