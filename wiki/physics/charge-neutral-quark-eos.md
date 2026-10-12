---
summary: Neutrality leaves P(mu_B,mu_K,T) unchanged at quadratic order and flips the mu_K^4 sign; the mu_K series fails before a=1, but in density variables (n_u=n_B, n_d,s=n_B(1+-a)) the EOS is exact and explicit.
status: current
updated: 2026-10-05
tags: [physics, equation-of-state, quark-matter, charge-neutrality]
---

# Charge-neutral quark EOS in $(\mu_B,\mu_K)$

The code's quark phase (`SQMsolver._quark_mu_triplet_neutral`, since 2026-05-22) is still a two-input EOS in $\mu_B,\mu_K$ (plus $T$). The auxiliary $\mu_Q$ is fixed internally. Why this basis was adopted, and what goes wrong without it near $a=1$, is in [[quark-matter-eos]]. This page gives the resulting EOS for massless quarks plus gluons, $P_g=8\pi^2T^4/45$.

## Construction

$$\mu_u=\frac{\mu_B}{3}+\frac{2\mu_Q}{3},\qquad \mu_{d,s}=\frac{\mu_B}{3}-\frac{\mu_Q}{3}\pm\frac{\mu_K}{2}.$$

The charge density is $\partial P/\partial\mu_Q=(2n_u-n_d-n_s)/3$, so neutrality is stationarity of $P$ in $\mu_Q$. Since $P$ is convex, $P_{\rm neutral}(\mu_B,\mu_K,T)=\min_{\mu_Q}P\le P_{\rm old}$. By the envelope theorem, $n_B=\partial P/\partial\mu_B$ and $n_K=\partial P/\partial\mu_K$ keep their definitions, $(n_u+n_d+n_s)/3$ and $(n_d-n_s)/2$, with no $\partial\mu_Q$ terms. Neutrality also gives two exact identities, valid at any $T$ and $m_s$:

$$n_B=n_u,\qquad a=\frac{n_d-n_s}{n_d+n_s}.$$

## Expansion (Wolfram, 2026-09-24)

$$\mu_Q=\frac{3\mu_K^2}{4\mu_B}-\frac{9\mu_K^4}{8\mu_B^3}-\frac{9\pi^2\mu_K^2T^2}{4\mu_B^3}+\cdots,\qquad \mu_u=\frac{\mu_B}{3}+\frac{\mu_K^2}{2\mu_B}+\cdots$$

$$
\begin{aligned}
P_{\rm neutral}={}&-U_{\rm bag}+\frac{\mu_B^4}{108\pi^2}+\frac{\mu_B^2\mu_K^2}{12\pi^2}-\frac{\mu_K^4}{32\pi^2}+\frac{\mu_B^2T^2}{6}+\frac{\mu_K^2T^2}{4}+\frac{19\pi^2T^4}{36}\\
&+\frac{5\mu_K^6}{64\pi^2\mu_B^2}+\frac{3\mu_K^4T^2}{16\mu_B^2}+O(8).
\end{aligned}
$$

The old basis ($\mu_Q=0$) is the same through quadratic order, but its $\mu_K^4$ coefficient is $+1/(32\pi^2)$. For massless quarks it is an exact polynomial, so the manuscript's "$+\cdots$" after `eq:Qm-EoS-4th-order` is zero there. The neutral series does not terminate. The code agrees with the series: at $\mu_B=1200$, $T=2$ MeV, $\mu_K=40$ MeV it gives $\mu_Q=0.998255$ against 0.998251 from the series, and $\Delta P=-1.61876\times10^4$ against $-1.61875\times10^4$ MeV$^4$.

Densities and fraction, through third order:

$$n_B=\frac{\mu_B^3}{27\pi^2}+\frac{\mu_B\mu_K^2}{6\pi^2}+\frac{\mu_BT^2}{3}\ \text{(both bases)},\qquad n_K=\frac{\mu_B^2\mu_K}{6\pi^2}\mp\frac{\mu_K^3}{8\pi^2}+\frac{\mu_KT^2}{2}\ \text{(neutral / old)},$$

$$a=\frac{9\mu_K}{2\mu_B}\Bigl[1-c\,\frac{\mu_K^2}{\mu_B^2}-\frac{6\pi^2T^2}{\mu_B^2}\Bigr],\qquad c=\tfrac{21}{4}\ \text{neutral},\ \tfrac{15}{4}\ \text{old}.$$

At $T=0$ this inverts to $\mu_K/\mu_B=2a/9+14a^3/243$ (neutral) or $2a/9+10a^3/243$ (old). The rate `eq:exact_rate` in terms of $a$ at $T=0$ becomes the leading cubic form times $1+31a^2/27$ (neutral) or $1+5a^2/9$ (old).

## Manuscript status (2026-10-05)

Appendix A of `Latex_writting/Main.tex` now uses the exact neutral density parameterization for `eq:QM-bag-EoS`, defines each flavor pressure through its finite-temperature density cubic, and includes gluon pressure explicitly. The fourth-order pressure has the negative quartic term; `eq:nK-approx` and `eq:nB-approx` include the corresponding subleading derivatives. Wolfram independently checked the pressure expansion and both density series with zero symbolic residuals, and confirmed the exact density inversion at two finite-temperature points.

`eq:QM-EoS-2nd-order`, the leading relation in `eq:a-muK`, and the leading cubic rate are unchanged, so the quadratic-order analytical results survive. All A1--A6 labels and numbers are retained; the former purple non-neutral-EOS discussion is commented out verbatim. Appendix B and solver code were not changed; the rate appendix's collaborator discussion still refers to the former EOS presentation.

The revised Appendix A prose and equations are highlighted with `\JS{}`; the unchanged quadratic formula remains black and PI comments retain their original markup. Eq. (35) now cites A2 and directly substitutes its equilibrium susceptibility and baryon density, as recorded in [[isothermal-maximum-speed]].

## Exact form in density variables (Wolfram, 2026-09-25)

Neutrality makes the flavor densities linear in the hydro variables $(n_B,n_K)$:

$$n_u=n_B,\qquad n_d=n_B(1+a),\qquad n_s=n_B(1-a).$$

⚠ This is exact only for the neutral EOS. For the old $\mu_Q=0$ basis, $n_{d,s}\simeq n_u(1\pm a)$ is a first-order truncation that fails at $a\sim1$ (see [[quark-matter-eos]]).

The EOS is therefore explicit in densities and implicit in chemical potentials; the old basis is the reverse. For massless quarks at $T=0$, with $S(a)=1+(1+a)^{4/3}+(1-a)^{4/3}$ and $R(a)=1+(1+a)^{1/3}+(1-a)^{1/3}$:

$$\varepsilon=\tfrac34\pi^{2/3}n_B^{4/3}S(a)+U_{\rm bag},\qquad P=\tfrac14\pi^{2/3}n_B^{4/3}S(a)-U_{\rm bag},$$

$$\mu_B=\partial\varepsilon/\partial n_B|_{n_K}=(\pi^2n_B)^{1/3}R(a),\qquad \frac{\mu_K}{\mu_B}=\frac{(1+a)^{1/3}-(1-a)^{1/3}}{R(a)}=\frac{2a}{9}+\frac{14a^3}{243}+\frac{200a^5}{6561}+\cdots.$$

On an isobar, $\mu_B(P,a)=[4\pi^2(P+U_{\rm bag})]^{1/4}R(a)/S(a)^{1/4}$. The linear inverse $\mu_K=2a\mu_B/9$ is off by 1, 4, 10, 21 and 60% at $a=0.2$, 0.4, 0.6, 0.8 and 1. Exactly, $\mu_u=(\pi^2n_B)^{1/3}$, so the rate prefactor $\mu_u^5$ needs no $\mu_Q$. At finite $T$, each flavor inverts its own monotone cubic $\mu_f^3/\pi^2+\mu_fT^2=n_f$ independently. Checked against `SQMsolver` at $\mu_B=1200$, $\mu_K=344$, $T=5$ MeV, which reconstructs $\mu_B$ and $\mu_K$ to $10^{-15}$.

## ⚠ The series fails before $a=1$

Neutral $a(\mu_K)$ at $\mu_B=1200$, $T=0.5$ MeV, from the code, against the third-order series:

| $\mu_K/\mu_B$ | 0.10 | 0.20 | 0.28 | 0.35 | 0.45 | 0.5575 |
|---|---|---|---|---|---|---|
| exact $a$ | 0.428 | 0.743 | 0.892 | 0.960 | 0.995 | 1.000 |
| series | 0.426 | 0.711 | 0.741 | 0.562 | $-0.13$ | $-1.58$ |

At $T=0$ the $udd$ point is $\mu_K=2^{1/3}\mu_B/(1+2^{1/3})\approx0.5575\,\mu_B$, with $\mu_Q\approx0.164\,\mu_B$. Because $n_s\propto\mu_s^3$ there, $a\to1$ with zero slope ($1-a\propto\mu_s^3$). Finite $T$ softens this to $n_s\propto\mu_sT^2$. Anything near the $1/(1-A)$ pole therefore needs the exact EOS. In particular, flipping only the $\mu_K^3$ sign in the old $T=0$ densities gives $a=\tfrac{9r}{2}(1-\tfrac34r^2)/(1+\tfrac92r^2)$, with $r=\mu_K/\mu_B$. That truncated form never reaches $a=1$: $27r^3+36r^2-36r+8=0$ has roots $-2.053$ and $0.360\pm0.122i$. It peaks at $a=0.925$ at $r=0.372$, where the exact neutral EOS gives $a=0.972$ and $n_u:n_d:n_s=27:53.25:0.75$. At $T=0$ that is $\mu_u=\mu_B-2m$, $\mu_{d,s}=m\pm\mu_K/2$, with the cubic $\mu_u^3=m^3+\tfrac34m\mu_K^2$ fixing $m$.
