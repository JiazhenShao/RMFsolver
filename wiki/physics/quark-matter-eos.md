---
summary: The working quark phase is an unpaired massless three-flavor bag-model gas; its quadratic expansion makes the K fraction linear in mu_K and produces the analytic elliptical isobar.
status: current
updated: 2026-09-24
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

## Why the code is charge-neutral: $a=1$ must mean $udd$

The switch was made on 2026-05-22 in a Codex session, not in a notebook; [26SU-3](../../26SU-3.ipynb) cells 8–15 are a later old-vs-new comparison. The trigger was the analytic bound's $1/(1-a(0^+))$ divergence. MGA read it as "$a(0^+)=1$ means $udd$ quark matter, so neutrons convert without strangeness diffusion" and asked what stops $a(0^+)$ from reaching 1. The old basis above, which implicitly sets $\mu_Q=0$, cannot support that reading:

| $\mu_B=1200$ MeV, $T=0.5$ MeV, $m_s=0$ | old, $\mu_Q=0$ | neutral, $\mu_Q$ solved |
|---|---|---|
| $\mu_K$ at $a=1$ | 344 MeV | 669 MeV |
| $n_u:n_d:n_s$ at $a=1$ | 1 : 2.92 : 0.185 | 1 : 2 : 0 |
| charge per baryon at $a=1$ | $-0.27$ | 0 |
| $a$ where $n_s=0$ | 4/3 | 1 |

With $2n_u=n_d+n_s$, the fraction is exactly $a=(n_d-n_s)/(n_d+n_s)$, so $a=1$ is equivalent to $n_s=0$ (neutron flavor content) and $a>1$ requires net antistrangeness. In the old basis $a=1$ is an unremarkable, charged, strange state, and $a$ passes through 1 with nothing in the EOS marking it.

⚠ The table uses the local fraction $n_K/n_B$. At the time of the switch, $a$ was defined as $n_K(x)/n_B(\infty)$; the local definition landed in v4.4 (2026-08-18). The two agree wherever $n_B(x)=n_B(\infty)$. That holds exactly on the constant-$P$, constant-$h/n_B$ trajectory, because massless quarks have $h=4(P+U_{\rm bag})$. It fails on the fixed-$\mu_B$ isobar used by the analytic approximation, where $n_B$ drifts. Old basis, $\mu_B(\infty)=1100$ MeV, $T(\infty)=75$–90 MeV: $n_K/n_B(\infty)=1$ gives $n_s/n_u=0.19$–0.21 and charge $-0.27$ per baryon on either path, so the old definition does not rescue the $udd$ reading. Neutral basis, fixed-$\mu_B$ isobar, $T(\infty)=90$–110 MeV: $n_B$ drifts 7–8%, and $n_K/n_B(\infty)=1$ still leaves $n_s/n_u\approx0.06$–0.08. The identity $a=1\Leftrightarrow n_s=0$ is a statement about the local fraction only.

⚠ Do not mix orders at $a=1$. Using the linear $a\simeq9\mu_K/(2\mu_B)$ to locate $a=1$ ($\mu_K=2\mu_B/9$) and then exact $n_i\propto\mu_i^3$ gives $27:64:8$, which has exact $a=28/33$. For any EOS, $a=1$ is the one-parameter family $n_d=2n_u+5n_s$, and $udd$ is its $n_s=0$ member. The old basis forces $\mu_d+\mu_s=2\mu_u$, and its $T=0$ fraction is strictly increasing in $\mu_K/\mu_B$, so it meets that family exactly once, at $27:79:5$; no $\mu$'s reproduce $1:2:0$. This uses $n_s=0\Leftrightarrow\mu_s=0$, true for massless quarks at any $T$ and for massive ones at $T>0$. With $m_s>0$ at $T=0$, $n_s=0$ for all $|\mu_s|\le m_s$, and the old basis does contain $udd$ at $\mu_s=(2-2^{1/3})\mu_B/3$ once $m_s\ge0.247\,\mu_B$ (296 MeV at $\mu_B=1200$ MeV). Adding $2n_u=n_d+n_s$ to $n_d=2n_u+5n_s$ forces $n_s=0$. Neither estimate of $a=1$ is $udd$: the old basis gives strange quarks at $a=1$ whether $a$ is taken from the linear relation or computed exactly, and no state of it has $1:2:0$. The exact old-basis density corrections are $n_K\supset\mu_K^3/(8\pi^2)$ and $n_B\supset\mu_B\mu_K^2/(6\pi^2)$; the $n_B$ term is a 22% correction at $a=1$. The requirements set at the switch were no electrons, no Urca, no neutrinos, only $u+d\leftrightarrow u+s$, and a quark-only EOS that connects the neutral $udd$ endpoint to neutral equilibrated $uds$. Under $\mu_u+\mu_d+\mu_s=\mu_B$, the massless $T=0$ endpoint is $\mu_K^{udd}=2^{1/3}\mu_B/(1+2^{1/3})$.

The neutral and old pressures differ only at $O(\mu_K^4)$ (verified numerically: doubling $\mu_K$ multiplies the difference by 16). The quadratic analytic limit, including $a\simeq9\mu_K/(2\mu_B)$ and the elliptical isobar, is therefore unchanged; neutrality matters only far from $\mu_K=0$, near $a\to1$. The $\mu_K^4$ coefficient in the fourth-order expansion above is the old-basis value; the neutral value is $-1/(32\pi^2)$, derived with the rest of the neutral EOS in [[charge-neutral-quark-eos]].

## Source

`Latex_writting/Hydrodynamical combustion/Main.tex`, first active document: “Bag model of quark matter” and the model assumptions in “Boundary propagation speed.”
