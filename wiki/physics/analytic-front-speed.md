---
summary: Slow-front scale separation reduces the quark layer to a constant hydrodynamic background and yields an analytic speed controlled by the weak rate, flavor diffusion, the interface fraction and a profile-shape factor.
status: current
updated: 2026-10-01
tags: [physics, analytic, front-speed, approximation]
---

# Piecewise-constant analytic front speed

As of 2026-10-01, `Latex_writting/Main.tex` denotes the adiabatic composition ceiling by $A_{\rm ad}$ (the $A$ used below), distinct from $A_{\rm iso}$.
The shared downstream layer-thickness analysis now appears at the end of the isothermal analytical subsection, with $T(\infty)=T$ there; its equations and label `eq:conversion-thickness` are unchanged.
Existing figure artwork still uses $A$.

**2026-09-29 qualification:** constant density does not establish constant
chemical potential. The current semi-analytical calculation follows the
conserved pressure/enthalpy trajectory with variable muB; the full analytical
calculation makes additional approximations. The saved adiabatic profiles have
up to 6.01% variation in muB. See [[adiabatic-xi-diagnostics]].

For $u_N\ll1$, the numerical solutions vary appreciably only in composition and temperature. Across the quark conversion layer, the manuscript reports relative variations no larger than $10^{-10}$ in $n_B$ and $10^{-12}$ in $P$. The analytic approximation therefore holds $n_B$, $u$, $\mu_B$, $P$ and $h$ constant for $x>0$ while retaining $a(x)$, $T(x)$ and $\mu_K(x)$.

Define

$$\lambda_n\equiv\frac{n_B(0^-)}{n_B^{(+)}},\qquad u^{(+)}=\lambda_n u_N,\qquad a=\frac{n_K}{n_B^{(+)}}.$$

## Interface gradient and profile identity

K-current matching gives

$$u_N=-\frac{D_K(0^+)a'(0^+)}{\lambda_n[1-a(0^+)]}.$$

Multiplying the reaction--diffusion equation by $D_Ka'$ and integrating yields

$$\frac12[D_K(0^+)a'(0^+)]^2=-u^{(+)}I_1+I_2,$$

with

$$I_1=\int_0^\infty D_K(a')^2dx,\qquad I_2=-\frac{1}{n_B^{(+)}}\int_0^\infty D_K\Gamma_Ka'\,dx.$$

Both are positive for the monotone physical profile. The diffusive-flux shape is compressed into $-1<\xi<1$:

$$I_1=\frac{1+\xi}{2}\lambda_nu_N[1-a(0^+)]a(0^+).$$

This gives the general relation

$$u_N^2=\frac{2I_2}{\lambda_n^2[1-a(0^+)][1+\xi a(0^+)]}.$$

## Constant-pressure trajectory

The quadratic [[quark-matter-eos|bag-model EOS]] makes the quark-side isobar a quarter ellipse. If $A$ is where its extrapolation reaches $T=0$, then

$$T^2(a)=\frac{2(\mu_B^{(+)})^2}{81\pi^2}(A^2-a^2),\qquad a(0^+)\le A.$$

Using the analytic weak rate and the leading Landau-damped part of $D_K$ gives

$$
\begin{aligned}
u_N^2={}&\frac{9\pi^{7/3}}{4\sqrt2\,h_D\alpha_s^{5/3}}
\frac{\gamma_KA^{7/3}}{\mu_B^{(+)}\lambda_n^2[1-a(0^+)][1+\xi a(0^+)]}\\
&\times\left[\frac{24}{7}-3\left(1-\frac{a^2(0^+)}{A^2}\right)^{1/6}
-\frac{3}{7}\left(1-\frac{a^2(0^+)}{A^2}\right)^{7/6}\right],
\end{aligned}
$$

Here $\gamma_K$ is the weak-rate coefficient, not a Lorentz factor.

The formal $a(0^+)=A$ endpoint scales as

$$u_{N,\max}\propto\frac{A^{7/6}}{\alpha_s^{5/6}\sqrt{1-A}\sqrt{1+\xi A}}.$$

The divergence as $A\to1$ signals that no weak flavor change is needed when the newborn quark phase can inherit the neutron flavor content. It is a breakdown of the diffusion-limited bound, not a physical infinite speed.

## Physical ceiling and thickness

The formal $a(0^+)=A$, $T(0^+)=0$ endpoint lies outside local thermal equilibrium because the mean free path diverges. Published contours must instead apply [[lte-composition-bound]] or a resolved [[interface-closure]]. Do not present the raw endpoint as the physical maximum.

The downstream linear tail gives a conversion-layer estimate

$$L_{\rm conv}\approx\frac{\mu_B^{(+)}}{9\pi T(\infty)}\sqrt{\frac{D_K}{\gamma_K}}\sim0.2\ \mu\mathrm{m}$$

for the manuscript's reference scales. This is many orders of magnitude below kilometer curvature scales and is the quantitative planar-front justification.

## Shape-factor assessment updated on 2026-09-29

The earlier warning concerned inconsistent manuscript claims about $\xi=0$;
that specific warning is superseded. Both current adiabatic comparison methods
and the manuscript now use $\xi=-0.5$. However, this constant is not accurate
throughout the domain: measured profile values approach -1 at high speeds.
[[adiabatic-xi-diagnostics]] quantifies the resulting errors and distinguishes
the physical profile shape from a full-analytical speed fit outside its allowed range.

## Sources

- `Latex_writting/Hydrodynamical combustion/Main.tex`, first active document: “Piecewise-constant analytical approximation,” “Shape of the diffusive flux profile,” and “Local thermal equilibrium.”
- `Latex_writting/Hydrodynamical combustion/Notes.tex`, May 13 analytic derivation and flux-balance estimate; use only where consistent with the active manuscript and the corrected [[lte-composition-bound]].
