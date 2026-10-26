---
summary: The corrected full-analytical adiabatic speed follows the manuscript exactly and overshoots the semi-analytical speed by 10–47% at T(0+)=0; Landau-only diffusion dominates at small a, and the quadratic ceiling dominates as a approaches 1.
status: current
updated: 2026-10-26
tags: [analytic, adiabatic, verification, error-budget]
---

# Adiabatic full-analytical error budget

After the second-order endpoint fix described in [[analytic-slow-front-solvers]], `analytic_velocity_adiabatic_slow` implements every displayed step of the manuscript's adiabatic analytical appendix.
These are Eq. (a-max) with the inserted second-order endpoint relations, the ellipse Eq. (constant_pressure_path), Landau-only $D_K$, the cubic rate with $\gamma_K(\mu_B^{(+)})$, the closed $I_2$, and Eq. (speed_general).
On 2026-10-26 the code's $q_D$, $D_K$ prefactor, $h_D$ and $\gamma_K$ matched the manuscript expressions exactly.
The displayed $82\,\mathrm{m\,s^{-1}}$ formula runs $0.29\%$ above the code because its coefficient is rounded.
The code takes $n_B^{(+)}=4(P+U_{\rm bag})/(h/n_B)$, which differs from $\partial P_{\rm quad}/\partial\mu_B$ by $3\times10^{-5}$ to $2\times10^{-2}$ (a higher-order effect); the manuscript does not specify which of the two it uses.

The remaining excess is therefore a modelling error, not a bug.

## Ladder from semi-analytical to full analytical

Starting from the semi-analytical path (full EOS, full $D_K$, exact local rate), one approximation is swapped in at a time.
All cases use $T(0^+)=0$, $\xi=-0.85$, $B^{1/4}=189.157$ MeV and massless QMC-RMF3 PNM.
Speeds are in m/s, and BVP is `solve_front_adiabatic_slow` at the same $T(0^+)$.

| $T(0^-)$ [MeV] | $n_B(0^-)/n_0$ | $a(0^+)$ semi | BVP | semi | Landau $D_K$ | cubic rate | ellipse | quadratic ceiling | analytic | an/semi | an/BVP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8.9 | 3.03 | 0.156 | 34.9 | 36.0 | ×1.104 | ×0.995 | ×1.003 | ×1.000 | 39.7 | 1.103 | 1.137 |
| 27.8 | 2.97 | 0.293 | 81.6 | 86.0 | ×1.126 | ×0.983 | ×1.010 | ×1.002 | 96.4 | 1.121 | 1.181 |
| 70.7 | 1.43 | 0.467 | 250.4 | 268.0 | ×1.145 | ×0.957 | ×1.027 | ×1.009 | 304.5 | 1.136 | 1.216 |
| 45.6 | 3.25 | 0.599 | 291.2 | 310.3 | ×1.157 | ×0.928 | ×1.048 | ×1.025 | 357.8 | 1.153 | 1.229 |
| 81.3 | 2.03 | 0.700 | — | 522.6 | ×1.165 | ×0.901 | ×1.069 | ×1.051 | 616.1 | 1.179 | — |
| 91.7 | 1.76 | 0.799 | — | 880.1 | ×1.172 | ×0.869 | ×1.098 | ×1.108 | 1090.5 | 1.239 | — |
| 10.2 | 4.50 | 0.846 | — | 1054.4 | ×1.176 | ×0.852 | ×1.116 | ×1.164 | 1372.4 | 1.302 | — |
| 102.1 | 1.42 | 0.900 | — | 1842.8 | ×1.181 | ×0.829 | ×1.143 | ×1.317 | 2715.2 | 1.473 | — |

- Dropping the Debye term makes $D_K$ too large, which adds $+10\%$ to $+18\%$; this is the dominant excess at small $a$.
- The cubic rate at frozen $\mu_B$ partly cancels it ($-0.5\%$ to $-17\%$).
- The ellipse versus the true isobar adds $+0.3\%$ to $+14\%$.
- The quadratic ceiling sits $0$ to $3\%$ above the full-EOS $a(0^+)$, and $1/\sqrt{1-A_{\rm adb}}$ amplifies this to $+16\%$ at $a=0.85$ and $+32\%$ at $a=0.9$.
- Semi is itself $3$–$7\%$ above the matched BVP (the $\xi$ closure), so analytic/BVP is $1.14$–$1.23$ wherever the BVP converges ($a\lesssim0.6$).

Across the 1062 paired contour points, the median analytic/semi ratio is $1.10$ for $a<0.2$, $1.14$ at $0.4$–$0.5$, $1.30$ at $0.8$–$0.9$ and about $2$ above $0.9$.
The notebook overlay's numerical curves use $T(0^+)=0.1$ MeV, which lowers BVP speeds by $11$–$13\%$, so against that overlay the excess looks larger than it is at matched $T(0^+)$.

Scripts and logs: `new_paper_calculations/26-10-26/analytic_error_budget/`.
