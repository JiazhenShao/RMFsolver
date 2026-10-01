---
summary: The baryon-Eckart thermal model neglects interdiffusive energy transport and cross-effects; isothermal flow permits reservoir heat exchange, while adiabatic flow neglects conduction and cooling.
status: current
updated: 2026-09-24
tags: [physics, transport, hydrodynamics, manuscript]
---

# Thermal limits of the steady front

## Current metric convention

As of 2026-09-24, the newer manuscript `Latex_writting/Main.tex` uses signature $(+---)$, matching Harutyunyan and Sedrakian ([arXiv:2302.09596](https://arxiv.org/abs/2302.09596)).
The older copy under `Latex_writting/Hydrodynamical combustion/` was not changed in this conversion.
The velocity normalization is $U_\mu U^\mu=1$, the rest-space projector is $g^{\mu\nu}-U^\mu U^\nu$, and the stress-tensor pressure term is $-P g^{\mu\nu}$.
The constitutive laws are

$$
D_K^\mu=D_K(g^{\mu\nu}-U^\mu U^\nu)\partial_\nu n_K,
\qquad
q_{\rm th}^\mu=\kappa_{\rm th}\bigl[(g^{\mu\nu}-U^\mu U^\nu)\nabla_\nu T-TU^\alpha\nabla_\alpha U^\mu\bigr].
$$

Wolfram verified that both currents remain orthogonal to $U^\mu$ and that their components and the entire contravariant stress tensor are unchanged by the consistent signature conversion.
The newer manuscript retains the exact energy and momentum fluxes in `eq:front_system_thermal`; its energy equation no longer substitutes the expanded heat law.

A second Wolfram pass on 2026-09-24, with $g={\rm diag}(1,-1,-1,-1)$, confirmed $U_\mu U^\mu=1$ and $q_{\rm th}^0=uq_{\rm th}^x/\gamma$. It also confirmed both flux components, $\Delta^{xx}=-\gamma^2$ (hence $J_K=un_K-\gamma^2D_Kn_K'$), and the Eckart components. The manuscript's Eckart law equals the standard $(-+++)$ form, and the commented-out $\Pi$-elimination identities hold. No sign errors remain.
The main-text `eq:heat_flux_main`, $q_{\rm th}^x=-\kappa_{\rm th}\gamma^2T'$, is Eckart conduction without the acceleration term. The Heat Transport appendix now justifies dropping $Tuu'$ right after `eq:eckart_heat_flux_steady`: $u'=-un_B'/n_B$ makes it $O(u^2)$, and $u$ and $n_B$ are nearly constant on each side of the boundary.
The end of the formalism section states that at $\gamma=1$ the three-velocity $v$ and $u=\gamma v$ are not distinguished.

## Verified factors and earlier manuscript form

As of 2026-09-23, `eq:heat_flux_main` uses acceleration-neglected Eckart conduction, `q_th^x = -kappa_th*(1+u^2)*dT/dx`.
The energy-flux factor in `eq:energy_balance_exact` is `(1+2*u^2)/gamma`, not `1/gamma`; the two heat-current terms in the stress tensor add.
Substitution into that exact flux gives the coefficient `1+5*u^2/2` in `eq:energy_conservation_thermal`, at fixed temperature gradient through absolute second order.
Both factors were corrected in the manuscript after Wolfram verification; the momentum equations were unchanged.
A consistent switch of metric signature does not change these physical fluxes.

## Historical initial reduction

The initial hydrodynamic opening of `Latex_writting/Hydrodynamical combustion/Main.tex` included heat transport and neutrino losses before taking the two thermal limits.
Its energy balance was

$$
\frac{d}{dx}\left[(P+\epsilon)u\gamma-\kappa_{\rm th}T'\right]=-Q_\nu.
$$

This combines the relativistic ideal-fluid advective flux with leading slow-flow Fourier conduction.
The heat current obeys $U_\mu q_{\rm th}^\mu=0$; its contribution to the stress-energy tensor is $U^\mu q_{\rm th}^\nu+q_{\rm th}^\mu U^\nu$.
The reduced system explicitly neglects the acceleration correction to the relativistic heat law, conductive momentum transport, and neutrino momentum loss.
It also states the $\gamma^2\simeq1$ reduction used in the K transport terms.
This opening system is superseded by the second-order substitution below; the earlier assumptions are recorded here for provenance.

## Superseded Fourier substitution

An earlier 2026-09-17 main-text heat law (`eq:heat_flux_main`) directly used Fourier conduction, $q_{\rm th}^x=-\kappa_{\rm th}T'$.
Expanding the **exact** stress-tensor fluxes at fixed $\kappa_{\rm th}T'$ gives, through second order,

$$
F_E=(P+\epsilon)u-\kappa_{\rm th}\left(1+\frac32u^2\right)T',
\qquad
\Pi=P+(P+\epsilon)u^2-2u\kappa_{\rm th}T'.
$$

These were the energy and momentum fluxes in `eq:front_system_thermal`, with zero neutrino source, before restoration of the acceleration-neglected Eckart closure.
Wolfram verified the coefficient using `Series[(1+2*u^2)/Sqrt[1+u^2], {u,0,2}]`, which gives `1 + 3*u^2/2 + O(u)^3`.
The coefficient `5/2` applies to acceleration-neglected Eckart conduction and was restored on 2026-09-23, but only in the older `Hydrodynamical combustion/` copy. The newer `Latex_writting/Main.tex` keeps the exact fluxes, so neither coefficient appears there. `3/2` remains the result for direct Fourier conduction only.
This is a second-order expansion of the stress-tensor fluxes with a prescribed Fourier law, not a second-order expansion of the full Eckart constitutive law.
If the earlier conversion-driven ordering $q_{\rm th}^x/(P+\epsilon)=O(u)$ is imposed, the entire $u^2\kappa_{\rm th}T'$ term is third order and the energy flux reduces to $(P+\epsilon)u-\kappa_{\rm th}T'$ at second order.

⚠ Steadiness sets $\partial_tu=0$, not $u'=0$.
Baryon conservation gives $u'=-u n_B'/n_B$; imposing $u'=0$ exactly on a nonzero-flow branch would also impose constant baryon density.
The Heat Transport appendix gives the condition for neglecting $Tuu'$; the main text refers there for the slow-velocity Fourier approximation.
It is not uniformly justified in the infinite-conductivity limit; see [[thermal-limit-ordering]].

## Eckart to Fourier

The Heat Transport appendix gives the first-order covariant Eckart heat law (`eq:eckart_heat_flux`) and derives its steady planar component (`eq:eckart_heat_flux_steady`).
For either signature, $c=1$, flat spacetime, and $U^\mu=(\sqrt{1+u^2},u,0,0)$, Wolfram verified

$$
q_{\rm th}^x=-\kappa_{\rm th}[(1+u^2)T'+Tuu'],\qquad q_{\rm th}^0=\frac{u}{\sqrt{1+u^2}}q_{\rm th}^x.
$$

Choosing first-order Eckart transport assumes heat-flux relaxation is short compared with changes experienced by the fluid; no quantitative relaxation-time check for the front has been made.
Stationarity does not establish this assumption because $U^\alpha\partial_\alpha=u\,d/dx$ remains nonzero through the profile.
Fourier conduction follows after $u^2\ll1$ and $|Tuu'|\ll|T'|$; slow speed alone does not establish the second inequality for arbitrarily small temperature gradients.
The manuscript currently takes the thermal limits after this reduction: strictly constant temperature does not remove the acceleration term of the unreduced Eckart law.
The 2026-09-17 analysis in [[thermal-limit-ordering]] also takes the conductivity endpoints first: infinite conductivity at finite heat flux gives $(\gamma T)'=0$, followed by $T'=0$ and $P'=0$ at leading slow velocity; the adiabatic branch is likewise isobaric at this order.
The derivation cites `Eckart:1940te`, `Lopez-Monsalvo:2010oeo` ([thermal dynamics](https://arxiv.org/abs/1006.2978)), and `Lander:2018cvq` ([stellar heat conduction](https://arxiv.org/abs/1804.01520)).

## Conductive momentum transport and approximation order

The heat current contributes to both fluxes. With $h=P+\epsilon$ and no external four-force, the exact first integrals of the stated stress tensor are

$$
F_E=h\gamma u+\frac{1+2u^2}{\gamma}q_{\rm th}^x,
\qquad
\Pi=P+hu^2+2u q_{\rm th}^x.
$$

Here $q_{\rm th}^x$ is the front-coordinate component, not the comoving scalar heat flux; orthogonality fixes $q_{\rm th}^0=u q_{\rm th}^x/\gamma$ even in a stationary profile.
Substituting the exact Eckart component above gives energy conduction $-\kappa_{\rm th}(1+2u^2)[\gamma^2T'+Tuu']/\gamma$ and momentum conduction $-2u\kappa_{\rm th}[\gamma^2T'+Tuu']$.
If acceleration alone is neglected, the respective coefficients of $-\kappa_{\rm th}T'$ are $\gamma(1+2u^2)$ and $2u\gamma^2$.

For a front with $q_{\rm th}^x/h=O(u)$, conductive energy transport enters at $O(hu)$, while conductive momentum transport is $O(hu^2)$, the same order as the ideal ram term.
A systematic leading slow-flow system may keep $d(hu-\kappa_{\rm th}T')/dx=0$ and $P'=0$.
Keeping $hu^2$ while dropping $2u q_{\rm th}^x$ remains a leading-accuracy approximation, but is neither an exact covariant reduction nor complete through order $u^2$.
Neglect relative to ram pressure needs the additional condition $2|q_{\rm th}^x|/(h|u|)\ll1$; small $u$ alone does not imply it.
For pressure accuracy compare $|2u q_{\rm th}^x|$ with the relevant thermodynamic stress scale and requested tolerance, not only with $hu^2$.
These estimates do not establish a bound on a sensitive BVP eigenvalue near a degeneracy.

Eliminating the heat current between the two exact fluxes leaves an algebraic momentum closure:

$$
\Pi=P+\frac{2\gamma u F_E-hu^2}{1+2u^2}.
$$

Including the conductive stress therefore does not itself introduce another independent boundary datum; retaining the acceleration term couples $T'$ and $u'$ in the constitutive equation.

Read-only diagnostic on `local_analysis_results/26SU-10/thermal-conducting-TN10-nB3n0-B180-tail1e-8.npy` (legacy filename), evaluated with its saved Fourier heat flux: $\max|u|=4.9785\times10^{-7}$, $\max|q_{\rm th}^x/(h\gamma u)|=0.0056516$, $\max|2u q_{\rm th}^x/P|=3.0465\times10^{-14}$, and $\max|2u q_{\rm th}^x/(hu^2)|=0.0113032$.
This measures the omitted term on one existing reduced solution, not a rerun of the covariant BVP or a validation of Fourier conduction: the saved run separately flags a large Knudsen number.
Wolfram verified orthogonality, both exact flux components, and the eliminated momentum closure on 2026-09-17; all five residual checks returned zero.
No solver or manuscript equations were changed in this diagnostic session.

## Frame and omitted transport

The coordinates remain in the phase-boundary rest frame.
The baryon (Eckart) hydrodynamic frame is a separate choice: $U^\mu$ is parallel to the baryon current, so only the baryon diffusion current vanishes.
K diffusion remains nonzero.

In the stress-energy decomposition, $q_{\rm th}^\mu$ denotes the nonadvective energy current.
The manuscript approximates it by Fourier conduction alone and explicitly neglects energy carried by K interdiffusion.
Independent density-gradient Fick and temperature-gradient Fourier laws also omit Soret/Dufour cross-effects.
These omissions are constitutive assumptions, not consequences of slow flow, and remain in the adiabatic calculation.
The closure is used for stationary profiles, not causal time-dependent evolution.

An additional $\mu_K D_K^\mu$ term is not universally compulsory in the stress tensor: the separation of heat and diffusion energy depends on the definition of the heat current.
Harutyunyan and Sedrakian, Eqs. (30), (31), and (66), distinguish the total energy diffusion current from the chemical-potential-weighted charge contribution to the entropy current.
The energy transport associated with composition gradients must nevertheless be specified or neglected explicitly.

The review's estimate $D_K|n_K'|\sim u n_K$ is not an interface identity.
Current matching for pure neutron matter gives $D_K^x(0^+)=J_K(0^+)-u(0^+)n_K(0^+)=u(0^+)n_B(0^+)[1-a(0^+)]$.
If $\mu_KD_K^x$ is used as an estimated chemical energy flux, its interface ratio to advection is $\mu_K n_B(1-a)/[(P+\epsilon)\gamma]$, evaluated at $0^+$, rather than $a\mu_K/\mu_B$.
No universal per-mille bound follows; the manuscript does not claim one.
This is a conditional scale estimate, not a calculation of the multicomponent energy transport coefficient.

## Limits and boundary assumptions

- After the slow-flow Fourier reduction, finite conductive flux and $\kappa_{\rm th}\to\infty$ force $T'\to0$ within each phase. Before that reduction the exact condition is $(\gamma T)'\to0$, as explained in [[thermal-limit-ordering]]. A single temperature across the sharp interface additionally assumes thermal contact without a temperature jump; bulk conductivity alone does not fix interface resistance.
- The product $-\kappa_{\rm th}T'$ can remain finite in this limit. Isothermal matter need not have constant advective energy flux; the idealization allows heat exchange through the boundaries to maintain the prescribed temperature.
- With $Q_\nu=0$, imposing zero conductive flux at both $0^-$ and $\infty$ would additionally require equal endpoint advective energy fluxes. This compatibility condition is not imposed by the fixed-temperature model, which treats matter as an open system exchanging heat with a reservoir; matter plus reservoir conserves energy.
- Setting $\kappa_{\rm th}=0$ recovers constant $(P+\epsilon)u\gamma$ only if $Q_\nu=0$ as well. Adiabatic here excludes conductive and radiative losses, not entropy production by weak reactions.
- The manuscript calculations omit neutrinos and use $Q_\nu=0$ in both limits. Weak conversion is already represented in the EOS and composition evolution, so it is not an extra source of total energy.

For finite conductivity, use the separate microscopic thermal coefficient in [[quark-transport]], not the K diffusion coefficient.
Interface thermal resistance remains a separate question in [[kapitza-resistance]].
The finite-conductivity BVP is recorded in [[thermal-conducting]].

## Verification and manuscript organization

With $q_{\rm th}^0=u q_{\rm th}^x/\sqrt{1+u^2}$, Wolfram verified zero $U_\mu q_{\rm th}^\mu$, conductive energy flux $(1+2u^2)q_{\rm th}^x/\sqrt{1+u^2}$, and conductive momentum flux $2u q_{\rm th}^x$.
The energy contribution is $q_{\rm th}^x[1+3u^2/2+O(u^4)]$.

The general system has label `eq:front_system_thermal`; the existing adiabatic system retains `eq:front_system`.
The isothermal system has distinct component labels and remains `eq:front_system_isothermal`.
Existing isothermal material was moved into the first isothermal section, before the adiabatic section.
All PI prose and author comments were preserved.

As of 2026-09-11, the isothermal section introduces the common conserved baryon/momentum fluxes, the two-field K transport system, upstream and interface conditions, and the shared analytical derivation through the definitions of $I_1$, $I_2$, $\xi$, and the general speed relation.
The common flux label `eq:conserved_fluxes` now contains only $J_B$ and $\Pi$; the additional adiabatic energy flux has label `eq:energy_flux_adiabatic`.
The common subsections are `sec:isothermal-equations`, `sec:boundary_conditions`, and `sec:isothermal_analytic`.
The later adiabatic subsections retain `sec:hydro-equations` and `sec:hydro_analytic`, referring back to the shared transport/boundary definitions and retaining the variable-temperature isobar and its integral evaluation.
No isothermal derivation depends on a definition in the later adiabatic section; appendix references for the shared cubic rate and diffusion approximation name both cases.
Existing formulas and all author-comment blocks were preserved, apart from splitting the flux-definition display into its shared and adiabatic parts.
The new layout builds without undefined references, citations, overfull boxes, or the previous stuck-float warning.

References added as verbatim INSPIRE-HEP exports: `Eckart:1940te` (Phys. Rev. 58, 919) and `Lander:2018cvq` (MNRAS 479, 4207).

On 2026-09-10, added the frame, cross-effect, interdiffusion-energy, and reservoir clarifications without changing equations or PI comments.
Added the INSPIRE export `Harutyunyan:2023nvt` (Symmetry 15, 494; https://arxiv.org/abs/2302.09596) for multicomponent current conventions.
Also restored missing bibliography entries `Alford:2025jtm`, `Alford:2026kwd`, and `Most:2022yhe`; the full manuscript now builds without undefined citations or references, though existing float-placement warnings remain.

Wolfram input for the conditional interface estimate (ASCII):

    FullSimplify[(muK (jB-u nK)/(h u gamma)) /. {jB->u nB,nK->a nB}, Assumptions -> u != 0]

Returned $\mu_K n_B(1-a)/(h\gamma)$, with $h=P+\epsilon$.
