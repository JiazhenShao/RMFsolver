---
summary: Eliminating the heat flux proves pressure corrections are O(h u^2) when q/h is O(u); the concise derivation is now commented out and the isothermal subsection starts from kappa_th -> infinity.
status: current
updated: 2026-09-24
tags: [physics, transport, hydrodynamics, manuscript]
---

# Conductivity limits followed by the slow-velocity limit

For the paper's slow subsonic fronts, both the adiabatic and isothermal branches have constant pressure at leading order.
The conductivity endpoint determines the thermal constraint; the slow-velocity expansion independently gives the isobar.
See [[thermal-limits]] for transport assumptions and [[thermal-conducting]] for the finite-conductivity implementation.

## Starting point before either limit

In stationary planar flat-spacetime flow, with $h=P+\epsilon$, $\gamma=\sqrt{1+u^2}$, and no neutrino four-force,

$$
J_B=n_Bu,\qquad
F_E=h\gamma u+\frac{1+2u^2}{\gamma}q_{\rm th}^x,\qquad
\Pi=P+hu^2+2u q_{\rm th}^x
$$

are constant.
The stated Eckart constitutive law is exactly

$$
q_{\rm th}^x=-\kappa_{\rm th}[\gamma^2T'+Tuu']
=-\kappa_{\rm th}\gamma(\gamma T)'.
$$

The slow-front ordering holds thermodynamic scales fixed and assumes heat transport driven by conversion, $q_{\rm th}^x/h=O(u)$, without a parametrically larger imposed heat current.
Both ram pressure and conductive stress are then $O(hu^2)$; they are small relative to the thermodynamic pressure in the regime studied.
Leading slow-velocity order means retaining the first nonzero baryon and energy fluxes, not setting $u=0$ in them.

## Eliminate heat before expanding

The two conserved fluxes give the exact identity

$$
\Pi=P+\frac{2\gamma uF_E-hu^2}{1+2u^2}.
$$

With $q_{\rm th}^x/h=O(u)$, hence $F_E/h=O(u)$, this implies

$$
\frac{P-\Pi}{h}=u^2-\frac{2uF_E}{h}+O(u^4)=O(u^2).
$$

The result is independent of conductivity and does not require the heat current to vanish.
Pressure departures scale as $hu^2$; the relative pressure error also includes the ratio $h/P$.
An externally imposed heat flux parametrically larger than $hu$ invalidates this ordering.
Wolfram verified the exact elimination and the series expansion.
The manuscript isolates pressure completely before expanding:

$$
P=\frac{(1+2u^2)\Pi+\epsilon u^2-2\gamma uF_E}{1+u^2}
=\Pi-2F_Eu+(\Pi+\epsilon)u^2+O(u^3).
$$

This Taylor remainder is at fixed energy flux; under $F_E/h=O(u)$, both displayed corrections are second order and the remainder improves to fourth order.

## Adiabatic first, then slow velocity

Set $\kappa_{\rm th}=0$ (or take its regular zero limit with bounded bulk thermal force), so $q_{\rm th}^x=0$.
The exact remaining constraints are $n_Bu=J_B$, $h\gamma u=F_E$, and $P+hu^2=\Pi$.
At leading slow velocity,

$$
P'=0,\qquad (hu)'=0,\qquad (n_Bu)'=0,
\qquad \frac{h}{n_B}=\frac{F_E}{J_B}=\text{constant}
$$

for a moving front with $J_B\ne0$.
Temperature can change as the composition evolves on this constant-pressure, constant-enthalpy-per-baryon trajectory.
This condition does not assert conservation of entropy in the reacting fluid.

## Infinite conductivity first, then slow velocity

Take $\kappa_{\rm th}\to\infty$ with finite $q_{\rm th}^x$ on a regular bulk branch.
The exact constitutive condition is

$$
(\gamma T)'=-\frac{q_{\rm th}^x}{\kappa_{\rm th}\gamma}\to0,
\qquad \gamma T=\text{constant}.
$$

Thus the finite-velocity result is not strictly constant rest-frame temperature.
The fractional temperature variation due to $\gamma$ is $O(u^2)$, since $1/\gamma=1-u^2/2+O(u^4)$.
After the leading slow-velocity reduction,

$$
T'=0,\qquad P'=0,\qquad (n_Bu)'=0,
\qquad (hu+q_{\rm th}^x)'=0.
$$

Finite heat flux survives: at this order $q_{\rm th}^x=F_E-hu=F_E-J_Bh/n_B$.
It balances changes in advected enthalpy while the leading thermodynamic profile is isothermal and isobaric.
Its momentum contribution stays $O(hu^2)$, so there is no new leading pressure gradient.
One must not evaluate the limiting product as infinity times an identically zero gradient and conclude that the heat flux is zero or divergent.

The acceleration term must be retained when taking infinite conductivity first: $\gamma^2T'$ and $Tuu'$ cancel at leading order in inverse conductivity.
Discarding acceleration first changes the finite-velocity condition to $T'=0$, although both orders of reduction give constant temperature at leading slow velocity under the stated regularity and flux assumptions.
Subleading temperature corrections and the heat flux cannot be inferred by interchanging the limits blindly.

## Boundary conditions and paper presentation

The infinite-conductivity argument is a bulk statement; common temperature across the sharp interface additionally requires thermal contact without a leading temperature jump.
A prescribed isothermal profile generally exchanges heat through its boundaries.
If conductive flux is instead required to vanish at both endpoints, energy conservation also requires matching endpoint advective enthalpy fluxes, which a generic fixed-temperature profile does not satisfy.
On the semi-infinite domain, convergence of the bulk thermal force alone does not establish uniform convergence of the entire profile or its far boundary conditions.

The earlier equation numbers (11) and (12) for the thermal and isothermal systems are superseded by subsequent manuscript edits; use the stable labels `eq:front_system_thermal` and `eq:front_system_isothermal`.
The user has replaced the main-text Fourier law with acceleration-neglected Eckart conduction, $q_{\rm th}^x=-\kappa_{\rm th}\gamma^2T'$, in `eq:heat_flux_main`.
Those user edits were left untouched in the isothermal-subsection revision.
The prior Appendix D proof and its main-text reference after the exact balances are now commented out, superseded by the concise pressure-isolation formula `eq:isobaric_pressure_expansion` at the start of Sec. III A.
As of 2026-09-24 that formula is itself commented out in both manuscript copies. The newer `Latex_writting/Main.tex` opens the subsection with $\kappa_{\rm th}\to\infty$ in `eq:front_system_thermal`, plus an MGA comment that eliminates $q$ between the energy and momentum integrals. That comment gives $P=C_2-2uC_1$ after dropping $u^2$ terms, which Wolfram confirms.
The subsection then takes infinite conductivity at finite heat current in the existing main heat law, obtains constant temperature, and states the leading slow-velocity system.
Its following momentum-flux definition is now $\Pi=P$ at this order, consistent with that system.
The former Fourier-coefficient mismatch was resolved on 2026-09-23 in the older copy: its reduced thermal equation now uses `5/2` (the newer $(+---)$ copy keeps exact fluxes instead), matching acceleration-neglected Eckart conduction at fixed thermal gradient, as explained in [[thermal-limits]].
The exact energy equation was also corrected from `q_th^x/gamma` to `(1+2*u^2)*q_th^x/gamma`; its previous form was not explained by a consistent metric-signature change.
The common K law becomes $J_K=un_K-D_Kn_K'$ and $J_K'=-\Gamma_K$ in both branches.
An isothermal system keeping $hu^2$ but dropping $2u q_{\rm th}^x$ is only a partial subleading correction, as detailed in [[thermal-limits]].
No solver changes accompany this manuscript derivation.

Wolfram verified the product-rule identity using the following ASCII input:

```wolfram
FullSimplify[
 (1+u[x]^2)*temp'[x] + temp[x]*u[x]*u'[x]
 - Sqrt[1+u[x]^2]*D[Sqrt[1+u[x]^2]*temp[x],x],
 Assumptions -> Element[u[x],Reals]]
```

The result was zero; it also returned $1/\sqrt{1+u^2}=1-u^2/2+3u^4/8+O(u^5)$ through fourth order.
