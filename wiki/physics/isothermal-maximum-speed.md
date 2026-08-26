---
summary: Substituting the static-isobar composition ceiling gives the isothermal maximum-speed formula; at low temperature it scales as T^(-5/6) a_max(0+)^2 and retains the a_max(0+) -> 1 singularity.
status: current
updated: 2026-08-25
tags: [physics, analytic, front-speed, isothermal, bound]
---

# Isothermal maximum-speed estimate

For fixed analytical coefficients, the positive branch of [[isothermal-analytic-front-speed]] increases monotonically with $a(0^+)$.  Its thermodynamic upper estimate is therefore obtained by substituting the static-isobar ceiling $a_{\max}(0^+)$.

## Static-isobar ceiling

Expand the quark pressure about the equilibrated endpoint, where $\mu_K(\infty)=0$ and $n_K(\infty)=0$.  To quadratic order,

$$
P_{\rm QM}(0^+)-P_{\rm QM}(\infty)
\simeq n_B(\infty)[\mu_B(0^+)-\mu_B(\infty)]
+\frac12\left.\frac{\partial^2P_{\rm QM}}{\partial\mu_K^2}\right|_{\infty}\mu_K^2(0^+)=0,
$$

and

$$
a(0^+)\simeq\frac{1}{n_B(\infty)}
\left.\frac{\partial^2P_{\rm QM}}{\partial\mu_K^2}\right|_{\infty}\mu_K(0^+).
$$

Combining these with $\mu_B(0^-)=\mu_B(0^+)+a(0^+)\mu_K(0^+)$ gives

$$
a_{\max}^2(0^+)\simeq
\frac{2[\mu_B(0^-)-\mu_B(\infty)]}{n_B(\infty)}
\left.\frac{\partial^2P_{\rm QM}}{\partial\mu_K^2}\right|_{\infty}.
$$

For the massless bag-model expansion in [[quark-matter-eos]], this becomes

$$
a_{\max}^2(0^+)\simeq
\frac{9[\mu_B(0^-)-\mu_B(\infty)]}{\mu_B(\infty)}
\frac{\mu_B^2(\infty)+3\pi^2T^2}
{\mu_B^2(\infty)+9\pi^2T^2},
$$

whose cold limit is $a_{\max}^2(0^+)\simeq9[\mu_B(0^-)-\mu_B(\infty)]/\mu_B(\infty)$.

## Maximum-speed formula

Using $\lambda_n=n_B(0^-)/n_B(\infty)$ and selecting the positive branch gives

$$
u_{\max}(0^-)=\frac{n_B(\infty)}{n_B(0^-)}
\sqrt{\frac{D_K\gamma_K}{2}}\,
\frac{a_{\max}(0^+)\sqrt{a_{\max}^2(0^+)+2\eta}}
{\sqrt{1-a_{\max}(0^+)}\sqrt{1+\xi a_{\max}(0^+)}}.
$$

This is the direct isothermal analogue of the maximum-speed expression in [[analytic-front-speed]].  The formula is an exact substitution into the analytical speed relation, but the Taylor estimate for $a_{\max}(0^+)$ remains approximate.

Substituting the full flavor-diffusion coefficient and $\eta=9\pi^2T^2/\mu_q^2$ makes the retained temperature and chemical-potential dependence explicit:

$$
u_{\max}(0^-)=\frac{n_B(\infty)}{n_B(0^-)}
\frac{a_{\max}(0^+)}{4\alpha_s}
\sqrt{
\frac{\pi\gamma_K[a_{\max}^2(0^+)+18\pi^2T^2/\mu_q^2]}
{N_f\left[h_DT^{5/3}/q_D^{2/3}+\pi^3T^2/(12q_D)\right]
[1-a_{\max}(0^+)][1+\xi a_{\max}(0^+)]}
}.
$$

Isothermality alone does not make $D_K$ constant: this expression uses the same frozen representative quark state as the analytical derivation, presently the $0^+$ state in the solver.  The susceptibility and $n_B(\infty)$ in the ceiling are evaluated at the equilibrated endpoint.

When the Landau-damped term dominates, the Eq.-(31)-style normalized form that retains both weak-rate terms is

$$
c\,u_{\max}(0^-)\simeq(98.3\ {\rm m\,s^{-1}})
\frac{n_B(\infty)}{n_B(0^-)}
\left(\frac{\mu_B(0^+)}{1200\ {\rm MeV}}\right)^{17/6}
\left(\frac{10\ {\rm MeV}}{T}\right)^{5/6}
\left(\frac{3}{N_f}\right)^{1/3}
\left(\frac{1.81317}{h_D}\right)^{1/2}
\frac{a_{\max}(0^+)\sqrt{a_{\max}^2(0^+)+162\pi^2T^2/\mu_B^2(0^+)}}
{\alpha_s^{5/6}\sqrt{1-a_{\max}(0^+)}\sqrt{1+\xi a_{\max}(0^+)}}.
$$

Here $\mu_q=\mu_B(0^+)/3$ is the frozen representative analytical state.  In the suprathermal limit $a_{\max}^2(0^+)\gg162\pi^2T^2/\mu_B^2(0^+)$, the composition numerator reduces to $a_{\max}^2(0^+)$.  Retaining the longitudinal $T^2$ term in $D_K^{-1}$ multiplies the displayed Landau-damped estimate by $[1+\pi^3(T/q_D)^{1/3}/(12h_D)]^{-1/2}$.

The divergence as $a_{\max}(0^+)\to1$ is the same diffusion-limited singularity already identified in the contour workflow, not a physical infinite speed.  The Taylor ceiling and moving-front formula require $0<a_{\max}(0^+)<1$; stable neutron matter and coexistence are classified separately.

## Wolfram checks

The exact substitution was checked with the ASCII-equivalent input

```text
s = 2*dmu*chi/ninf
u2[a_] = dk*gk*a^2*(a^2 + 2*eta)/(2*lam^2*(1-a)*(1+xi*a))
FullSimplify[Sqrt[u2[Sqrt[s]]] -
  Sqrt[(dk*gk/2)*s*(s+2*eta)/((1-Sqrt[s])*(1+xi*Sqrt[s]))]/lam]
```

under the physical sign assumptions; Wolfram returned `0`.  It also returned

```text
d Log[u(a)]/da =
  (1/(1-a) + 2*a^2/(a^2+2*eta) + 1/(1+xi*a))/(2*a)
```

which is positive for $0<a<1$, $\eta\geq0$, and $-1<\xi<1$.  Substitution of the full $D_K$ and $\eta$ expressions also simplified to zero.  With $\alpha_s^{-5/6}$ explicit as in Eq. (31), the normalized Landau-damped coefficient is `98.29738983094435` m/s; at $\alpha_s=0.3$ it is `268.08608563251715` m/s.
