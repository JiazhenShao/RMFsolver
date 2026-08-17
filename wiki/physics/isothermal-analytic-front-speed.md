---
summary: The fixed-temperature speed has one profile-shape factor, vanishes intrinsically only when a(0+) does, and needs a separate isothermal coexistence mask for stable neutron matter.
status: current
updated: 2026-08-17
tags: [physics, analytic, front-speed, isothermal]
---

# Isothermal analytic front speed

Replacing energy-flux conservation by $T(x)=\mathrm{const}$ removes the temperature trajectory from the [[analytic-front-speed|piecewise-constant speed estimate]].
With the cubic-plus-thermal-linear weak source and constant quark-side coefficients, the positive incoming speed is

$$
u(0^-)=\frac{a(0^+)}{\lambda_n}
\sqrt{\frac{D_K\gamma_K\left(a^2(0^+)+2\eta\right)}
{2[1-a(0^+)][1+\xi a(0^+)]}},
$$

where $\lambda_n\equiv n_B(0^-)/n_B^{(+)}$, $-1<\xi<1$, and

$$
\eta=\frac{9\pi^2T^2}{\mu_q^2}.
$$

Thus the equivalent explicitly temperature-dependent numerator is $a^2(0^+)+18\pi^2T^2/\mu_q^2$.
The formula is exact in $I_2$ within the stated cubic-plus-linear weak-rate reduction; $\xi$ is the only conversion-profile input.

## Integrated K-ness equation

Under the piecewise-constant quark-side background,

$$
J_K=n_B^{(+)}\left(u^{(+)}a-D_Ka'\right),
\qquad
\frac{dJ_K}{dx}=-n_B^{(+)}\gamma_K(a^3+\eta a),
$$

with $u^{(+)}=\lambda_nu(0^-)$.  Therefore

$$
\frac{d}{dx}(D_Ka')=u^{(+)}a'+\gamma_K(a^3+\eta a).
$$

Multiplication by $D_Ka'$ and integration from $0^+$ to $\infty$ give

$$
\frac12[D_Ka'(0^+)]^2=-u^{(+)}I_1+I_2,
$$

where

$$
I_1=\int_{0^+}^{\infty}D_K(a')^2\,dx,
\qquad
I_2=D_K\gamma_K\int_0^{a(0^+)}(a^3+\eta a)\,da.
$$

Because $D_K$, $\gamma_K$, and $\eta$ are constant, the second integral is an endpoint expression:

$$
I_2=D_K\gamma_K\left[\frac{a^4(0^+)}{4}+\frac{\eta a^2(0^+)}{2}\right]
=\frac{D_K\gamma_Ka^2(0^+)[a^2(0^+)+2\eta]}{4}.
$$

For $g(a)\equiv-D_Ka'>0$, the interface condition fixes

$$
g[a(0^+)]=\lambda_nu(0^-)[1-a(0^+)],
$$

and $g(0)=0$ downstream.  The profile area is parameterized by

$$
I_1=\int_0^{a(0^+)}g(a)\,da
=\frac{1+\xi}{2}\lambda_nu(0^-)[1-a(0^+)]a(0^+).
$$

Substitution gives

$$
I_2=\frac12\lambda_n^2u^2(0^-)[1-a(0^+)][1+\xi a(0^+)],
$$

which yields the boxed speed above after selecting the positive root.

## Relation for the literal isothermal BVP

`solve_front_isothermal` reconstructs small variations of $n_B(x)$ and $u(x)$ rather than imposing them to be exactly constant.
For its massless-quark normalization $a(\infty)=0$, the integrated equation instead contains

$$
I_1^{\rm BVP}=\int_{0^+}^{\infty}D_Ka'(ua)'\,dx.
$$

Writing

$$
G\equiv-D_Ka'(0^+)
=u(0^-)\left[a(0^-)-\frac{u(0^+)}{u(0^-)}a(0^+)\right]
$$

and parameterizing the area in $ua$ by

$$
I_1^{\rm BVP}=\frac{1+\xi}{2}G\,u(0^+)a(0^+),
$$

gives the solver-faithful endpoint formula

$$
u^2(0^-)=\frac{2I_2}
{\left[a(0^-)-\dfrac{u(0^+)}{u(0^-)}a(0^+)\right]
 \left[a(0^-)+\xi\dfrac{u(0^+)}{u(0^-)}a(0^+)\right]}.
$$

When $n_B(0^+)=n_B(\infty)=n_B^{(+)}$, one has $a(0^-)=u(0^+)/u(0^-)=\lambda_n$, and this relation reduces to the piecewise-constant formula.

## Formal zero-temperature limit

Assume that $a(0^+)$, $\lambda_n$, $\mu_q$, $\gamma_K$, and $\xi$ approach finite constants with $0<a(0^+)<1$ as $T\to0^+$.  The full diffusion coefficient obeys

$$
D_K^{-1}=\frac{8N_f\alpha_s^2}{\pi}
\left(h_D\frac{T^{5/3}}{q_D^{2/3}}+\frac{\pi^3T^2}{12q_D}\right),
$$

so its leading behavior is

$$
D_K\sim\frac{\pi q_D^{2/3}}{8N_f\alpha_s^2h_D}\,T^{-5/3}.
$$

Meanwhile $\eta\propto T^2\to0$, but the suprathermal $a^3$ weak source remains nonzero.  Therefore

$$
u(0^-)\sim
\frac{a^2(0^+)}{4\alpha_s\lambda_n}
\sqrt{\frac{\pi q_D^{2/3}\gamma_K}
{N_fh_D[1-a(0^+)][1+\xi a(0^+)]}}
\,T^{-5/6}.
$$

Thus the fixed-composition mathematical limit diverges:

$$
\lim_{T\to0^+}u(0^-)=\infty.
$$

The exact point $T=0$ is not an admissible state of `solve_front_isothermal`, whose microphysics guard requires $T>0$; this result is only the one-sided asymptotic limit.  It is also distinct from the energy-conserving problem with only $T(0^+)=0$, because the isothermal construction sends the entire quark-side profile to zero temperature.

This is not a physical infinite front speed.  The same scaling gives a conversion length $L_{\rm conv}\propto\sqrt{D_K}\propto T^{-5/6}$, while the transport mean free path grows as $D_K\propto T^{-5/3}$.  Hence the Knudsen ratio grows as $T^{-5/6}$ and the local diffusion/LTE description fails before zero temperature; the assumed nonrelativistic slow-front regime also fails once $u(0^-)$ is no longer small.

The joint limit is path dependent if $a(0^+)$ also tends to zero.  For $a(0^+)\propto T^p$ with $0<p<1$, the cubic source dominates and

$$
u(0^-)\propto T^{2p-5/6}.
$$

It diverges for $p<5/12$, approaches a finite nonzero value for $p=5/12$, and vanishes for $p>5/12$.  Thus the isothermal equations do not define a unique zero-temperature velocity unless the interface closure specifies the limiting behavior of $a(0^+)$.

## Stable neutron-matter region and zero-speed boundary

For $T>0$, $D_K>0$, $\gamma_K>0$, $\eta>0$, $0\leq a(0^+)<1$, and $-1<\xi<1$, the analytical speed satisfies

$$
u(0^-)=0\quad\Longleftrightarrow\quad a(0^+)=0.
$$

Thus the reaction--diffusion formula alone does not locate a finite-density phase-stability boundary.  The [[interface-closure|energy-conserving]] condition $\Delta h=0$ must not be reused after energy conservation has been replaced by isothermality.

For a static isothermal interface, bulk coexistence instead requires a common temperature, baryon chemical potential, and pressure, with equilibrated quark matter:

$$
T_{\rm NM}=T_{\rm QM}=T,
\qquad
\mu_{B,\rm NM}=\mu_{B,\rm QM},
\qquad
P_{\rm NM}=P_{\rm QM},
\qquad
\mu_K=0.
$$

For a contour parameterized by the upstream density, first obtain $\mu_B(0^-)$ from the nuclear EOS and define

$$
\Delta P_{\rm iso}[T,n_B(0^-)]
=P_{\rm NM}[\mu_B(0^-),T]
-P_{\rm QM}[\mu_B(0^-),0,T].
$$

At common $T$ and $\mu_B$, the phase with larger pressure is thermodynamically favored.  Therefore

$$
\begin{array}{lll}
\Delta P_{\rm iso}>0 &: & \text{neutron matter stable; no forward conversion front},\\
\Delta P_{\rm iso}=0 &: & \text{isothermal coexistence and }u(0^-)=0,\\
\Delta P_{\rm iso}<0 &: & \text{quark matter favored; neutron matter can be metastable}.
\end{array}
$$

The stable region should be masked or hatched rather than filled by evaluating the positive-speed formula.  Setting its forward speed to zero is a plotting convention; inside that region the modeled neutron-to-quark front does not exist, and the reverse process is outside the present calculation.

A direct diagnostic scan of the live QMC-RMF3 and massless bag-model EOS gives the following neutron-side coexistence densities:

| $T$ (MeV) | $n_{B,\rm coexist}(T)/n_0$ for $B^{1/4}=180$ MeV | $n_{B,\rm coexist}(T)/n_0$ for $B^{1/4}=189.1566$ MeV |
|---:|---:|---:|
| 0 | 2.3300 | 3.0000 |
| 20 | 2.1028 | 2.8335 |
| 40 | 1.3768 | 2.2923 |
| 60 | 0.5246 | 1.3247 |

For these branches, neutron matter is stable on the lower-density side of the tabulated curve.  The second calibration is the one chosen so cold coexistence occurs at $3n_0$.

## ⚠ Why the coefficients are constant

Isothermality by itself does not make $D_K$ independent of position because [[strangeness-reaction-diffusion|the transport coefficient]] also depends on $q_D\propto\mu_q$.
The coefficients are constant here because the analytic treatment also freezes $\mu_q$, while the present `solve_front_isothermal` implementation explicitly evaluates $D_K$, $\gamma_K$, and $\eta$ once at $0^+$ and reuses them through the BVP.
If a future isothermal solver updates these coefficients using the local $\mu_B(x)$, $I_2$ is no longer the polynomial endpoint expression above.

## Symbolic check

Wolfram Language evaluated

```text
Integrate[dk*gammak*(a^3 + eta*a), {a, 0, a0}]
```

as `a0^2*dk*gammak*(a0^2 + 2*eta)/4`.
It also simplified the final balance residual to zero and returned

```text
u2 = a0^2*dk*gammak*(a0^2 + 2*eta) /
     (2*lambda^2*(1 - a0)*(1 + xi*a0))
```

for the piecewise-constant limit.
For the zero-temperature check, Wolfram also returned

```text
lim_(T->0+) T^(5/3)*u2 =
Pi*qd^(2/3)*gamma*a0^4 /
(16*nf*alpha^2*hd*lambda^2*(1-a0)*(1+xi*a0))
```

For the finite-temperature zero-speed question, Wolfram returned `a0 == 0` as the only root under the physical sign and domain assumptions.

## Sources

- `RMFsolver/phase_velocity.py`, `solve_front_isothermal` and `_microphysics_from_quark_state_isothermal_baseline`.
- `Latex_writting/Hydrodynamical combustion/Notes.tex`, “Isothermal Notes (Mar 16, 2026).”
- [[analytic-front-speed]] for the corresponding non-isothermal construction.
