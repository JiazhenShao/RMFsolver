---
summary: Slow adiabatic analytical speeds use a consistent quadratic EOS and closed integral, while the semi-analytical branch integrates the full EOS at fixed pressure and enthalpy per baryon.
status: current
updated: 2026-10-23
tags: [physics, analytic, front-speed, approximation]
---

# Piecewise-constant analytic front speed

**2026-10-23:** `analytic_velocity_adiabatic_slow` and `semi_analytic_velocity_adiabatic_slow` determine thermodynamic endpoints before evaluating the integrated speed; neither solves moving-flux endpoint matching or a spatial BVP. Both require massless quarks, PNM, and a keyword-only `T_0plus`, including exactly zero. See [[analytic-slow-front-solvers]] for the live API contract and validation record, and [[analytic-slow-front-audit]] for the preceding audit.

**Historical 2026-10-22 audit:** the retained `analytic_velocity_bound` and `semi_analytic_velocity_bound` comparison APIs use moving-flux endpoint matching, and the latter retains a positive temperature cutoff. Their massless/PNM qualifications and older numerical benchmarks remain applicable to those APIs.

`Latex_writting/Main.tex` denotes the adiabatic composition ceiling by $A_{\rm adb}$ (the $A$ used below), distinct from $A_{\rm iso}$.
The shared downstream layer-thickness analysis now appears at the end of the isothermal analytical subsection, with $T(\infty)=T$ there; its equations and label `eq:conversion-thickness` are unchanged.
Existing figure artwork still uses $A$.

**2026-09-29 qualification:** constant density does not establish constant
chemical potential. The current semi-analytical calculation follows the
conserved pressure/enthalpy trajectory with variable muB; the full analytical
calculation makes additional approximations. The saved adiabatic profiles have
up to 6.01% variation in muB. See [[adiabatic-xi-diagnostics]].

For massless quarks, fixed $P$ gives $h=4(P+U_{\rm bag})$, so fixed $h/n_B$ makes $n_B$ constant along the adiabatic path. Constant $\mu_B$ is an additional approximation obtained by retaining only $n_B=\mu_B^3/(27\pi^2)$; it is not a consequence of constant density in the full EOS. The manuscript's older representative numerical variations, no larger than $10^{-10}$ in $n_B$ and $10^{-12}$ in $P$, are historical profile measurements, not new validation of these APIs.

Define

$$\lambda_n\equiv\frac{n_B(0^-)}{n_B(\infty)},\qquad u(\infty)=\lambda_n u(0^-),\qquad a=\frac{n_K}{n_B(\infty)}.$$

## Interface gradient and profile identity

K-current matching gives

$$u(0^-)=-\frac{D_K(0^+)a'(0^+)}{\lambda_n[1-a(0^+)]}.$$

Multiplying the reaction--diffusion equation by $D_Ka'$ and integrating yields

$$\frac12[D_K(0^+)a'(0^+)]^2=-u(\infty)I_1+I_2,$$

with

$$I_1=\int_0^\infty D_K(a')^2dx,\qquad I_2=-\frac{1}{n_B(\infty)}\int_0^\infty D_K\Gamma_Ka'\,dx.$$

Both are positive for the monotone physical profile. The diffusive-flux shape is compressed into $-1<\xi<1$:

$$I_1=\frac{1+\xi}{2}\lambda_nu(0^-)[1-a(0^+)]a(0^+).$$

This gives the general relation

$$u^2(0^-)=\frac{2I_2}{\lambda_n^2[1-a(0^+)][1+\xi a(0^+)]}.$$

## Constant-pressure trajectory

`analytic_velocity_adiabatic_slow(T_0minus, nB_0minus, B_one_forth, *, T_0plus, xi=0, ...)` uses the quadratic pressure, leading density $n_B=\mu_B^3/(27\pi^2)$, leading fraction $a=9\mu_K/(2\mu_B)$, and $h=4(P+U_{\rm bag})$ throughout. The retained density is not the derivative of every term in the quadratic pressure. Starting from the upstream PNM state, its reproducible endpoint recipe is

$$
\begin{aligned}
n_B(\infty)&=\frac{4[P(0^-)+U_{\rm bag}]}{h(0^-)/n_B(0^-)},\\
\mu_B(\infty)&=[27\pi^2n_B(\infty)]^{1/3},\\
T^2(\infty)&=\frac{\mu_B(\infty)}{18\pi^2}\left[\frac{h(0^-)}{n_B(0^-)}-\mu_B(\infty)\right].
\end{aligned}
$$

Pressure and enthalpy per baryon match the upstream state in this same approximation. Both $n_B(0^+)=n_B(\infty)$ and $\mu_B(0^+)=\mu_B(\infty)$ follow within the retained model. It requires $0\leq T(0^+)<T(\infty)$ and $0<a(0^+)<1$; the interface fraction and $A$ come from the quadratic ellipse, without substituting a full-EOS ceiling.

The quadratic [[quark-matter-eos|bag-model EOS]] makes the quark-side isobar a quarter ellipse. If $A$ is where its extrapolation reaches $T=0$, then

$$T^2(a)=\frac{2(\mu_B(\infty))^2}{81\pi^2}(A^2-a^2),\qquad a(0^+)\le A.$$

Using the analytic weak rate and the leading Landau-damped part of $D_K$ gives

$$
\begin{aligned}
u^2(0^-)={}&\frac{9\pi^{7/3}}{4\sqrt2\,h_D\alpha_s^{5/3}}
\frac{\gamma_KA^{7/3}}{\mu_B(\infty)\lambda_n^2[1-a(0^+)][1+\xi a(0^+)]}\\
&\times\left[\frac{24}{7}-3\left(1-\frac{a^2(0^+)}{A^2}\right)^{1/6}
-\frac{3}{7}\left(1-\frac{a^2(0^+)}{A^2}\right)^{7/6}\right],
\end{aligned}
$$

Here $\gamma_K$ is the weak-rate coefficient, not a Lorentz factor.
The analytical branches share $\alpha_s=0.4$ and the exact expression $h_D=\Gamma(8/3)\zeta(5/3)(2\pi)^{2/3}/6$ with the numerical transport helpers. The full analytical branch keeps only the leading Landau term in diffusion and the cubic reduced weak rate.

The formal $a(0^+)=A$ endpoint scales as

$$u_{\max}(0^-)\propto\frac{A^{7/6}}{\alpha_s^{5/6}\sqrt{1-A}\sqrt{1+\xi A}}.$$

The divergence as $A\to1$ signals that no weak flavor change is needed when the newborn quark phase can inherit the neutron flavor content. It is a breakdown of the diffusion-limited bound, not a physical infinite speed.

## Full-EOS semi-analytical path

`semi_analytic_velocity_adiabatic_slow` has the same required `T_0plus` control and adds `path_rtol=1e-6`. It follows the full neutral massless EOS at fixed $P$ and $h/n_B$, retaining constant $n_B$ but allowing $\mu_B$ to vary. It uses full two-term diffusion and the local modified Madsen rate, rather than the quadratic ellipse or cubic endpoint integral.

The path coordinate is $r=\int_0^{n_K}D_K\,dn_K$, and the numerical integral is $I_2=n_B^{-2}(\infty)\int_0^{r(0^+)}\Gamma_K\,dr$. The returned `thermodynamic_path` runs from interface to equilibrium in state space; it is not a spatial profile. At exactly $T(0^+)=0$, both slow adiabatic branches retain finite $I_2$ and speed while reporting infinite local $D_K(0^+)$; the semi-analytical quadrature uses the regular variable $T^{1/3}$ without a positive-temperature cutoff.

Both return location-based endpoints, conservation residuals, `I2`, `u_0minus`, `jB`, and explicit EOS/diffusion/integral approximation metadata. The compatibility label `rate_model="exact_nonleptonic"` is accompanied by `rate_approximation="madsen_muB_over_3"`; it does not denote exact phase-space integration. Neither new analytical API supports finite strange-quark mass; numerical massless and massive branches remain separate ([[slow-front-solvers]]).

## Physical ceiling and thickness

The formal $a(0^+)=A$, $T(0^+)=0$ endpoint lies outside local thermal equilibrium because the mean free path diverges. Published contours must instead apply [[lte-composition-bound]] or a resolved [[interface-closure]]. Do not present the raw endpoint as the physical maximum.

The downstream linear tail gives a conversion-layer estimate

$$L_{\rm conv}\approx\frac{\mu_B(\infty)}{9\pi T(\infty)}\sqrt{\frac{D_K}{\gamma_K}}\sim0.2\ \mu\mathrm{m}$$

for the manuscript's reference scales. This is many orders of magnitude below kilometer curvature scales and is the quantitative planar-front justification.

## Shape-factor assessment updated on 2026-09-29

The earlier warning concerned inconsistent manuscript claims about $\xi=0$;
that specific warning is superseded. At that date both adiabatic comparison methods
and the manuscript used $\xi=-0.5$; the new slow analytical APIs default to `xi=0`. However, this constant is not accurate
throughout the domain: measured profile values approach -1 at high speeds.
[[adiabatic-xi-diagnostics]] quantifies the resulting errors and distinguishes
the physical profile shape from a full-analytical speed fit outside its allowed range.

## Sources

- `RMFsolver/phase_velocity.py`, `analytic_velocity_adiabatic_slow` and `semi_analytic_velocity_adiabatic_slow`; `Latex_writting/Main.tex`, the active analytical appendices.
- `Latex_writting/Hydrodynamical combustion/Main.tex`, first active document: “Piecewise-constant analytical approximation,” “Shape of the diffusive flux profile,” and “Local thermal equilibrium.”
- `Latex_writting/Hydrodynamical combustion/Notes.tex`, May 13 analytic derivation and flux-balance estimate; use only where consistent with the active manuscript and the corrected [[lte-composition-bound]].
