---
summary: Infinite conductivity gives the isothermal limit only with finite heat flux and thermal contact; the adiabatic limit requires both zero conductivity and zero neutrino cooling.
status: current
updated: 2026-09-09
tags: [physics, transport, hydrodynamics, manuscript]
---

# Thermal limits of the steady front

The hydrodynamic opening of `Latex_writting/Hydrodynamical combustion/Main.tex` now includes heat transport and neutrino losses before taking the two thermal limits.
The retained energy balance is

$$
\frac{d}{dx}\left[(P+\epsilon)u\gamma-\kappa_{\rm th}T'\right]=-Q_\nu.
$$

This combines the relativistic ideal-fluid advective flux with leading slow-flow Fourier conduction.
The heat current obeys $U_\mu q_{\rm th}^\mu=0$; its contribution to the stress-energy tensor is $U^\mu q_{\rm th}^\nu+q_{\rm th}^\mu U^\nu$.
The reduced system explicitly neglects the acceleration correction to the relativistic heat law, conductive momentum transport, and neutrino momentum loss.
It also states the $\gamma^2\simeq1$ reduction used in the K transport terms.

## Limits and boundary assumptions

- For finite conductive flux, $\kappa_{\rm th}\to\infty$ forces $T'\to0$ within each phase. A single temperature across the sharp interface additionally assumes thermal contact without a temperature jump; bulk conductivity alone does not fix interface resistance.
- The product $-\kappa_{\rm th}T'$ can remain finite in this limit. Isothermal matter need not have constant advective energy flux; the idealization allows heat exchange through the boundaries to maintain the prescribed temperature.
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

References added as verbatim INSPIRE-HEP exports: `Eckart:1940te` (Phys. Rev. 58, 919) and `Lander:2018cvq` (MNRAS 479, 4207).
