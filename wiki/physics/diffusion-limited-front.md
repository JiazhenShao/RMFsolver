---
summary: The project models a slow planar deflagration whose speed is selected by quark-side strangeness diffusion and nonleptonic weak equilibration, not by shock hydrodynamics.
status: current
updated: 2026-08-11
tags: [physics, front, hydrodynamics, core]
---

# Diffusion-limited nuclear-to-quark front

The baseline problem is a **laminar, steady, planar deflagration** in the interface rest frame. Strong interactions deconfine incoming neutron matter essentially instantaneously at $x=0$; the slow step is flavor equilibration in quark matter at $x>0$, where weak reactions create strange quarks and diffusion carries strangeness back toward the interface. The resulting reaction--diffusion layer selects the front speed.

This is not a detonation calculation. Shock-driven, turbulent, nucleation-dominated and whole-star conversion require separate physics.

## Matter models and omissions

- Upstream: pure neutron matter using QMC-RMF3.
- Downstream: unpaired, massless $u,d,s$ quarks in a bag-model EOS; see [[quark-matter-eos]].
- Explicit flavor channel: $d+u\leftrightarrow u+s$.
- Electromagnetism, electrons, neutrinos and Urca processes are omitted.
- The interface is sharp; mixed phases and surface microphysics are not resolved.
- The nuclear side is homogeneous, with no precursor disturbance.
- The front curvature is macroscopic while the conversion layer is microscopic, supporting the planar approximation; see [[analytic-front-speed]].

These omissions make the result a microphysical upper-bound model, not a complete compact-star composition calculation. Additional slowly equilibrating fractions, pairing gaps and dissipative channels can lower the laminar speed.

## Variables and direction

$$n_K\equiv\frac{n_d-n_s}{2},\qquad \mu_K\equiv\mu_d-\mu_s,\qquad a\equiv\frac{n_K}{n_B}.$$

Pure neutron matter has $n_K(0^-)=n_B(0^-)$. Fully flavor-equilibrated quark matter has $\mu_K(\infty)=n_K(\infty)=0$. The freshly deconfined state at $0^+$ lies between them. As $x$ increases downstream, $a(x)$ decreases toward zero; the corresponding diffusive flux points back toward the interface.

The reported boundary speed is the incoming nuclear-side four-velocity component $u_N\equiv u(0^-)=\gamma_v v$. Since all calculated speeds satisfy $v\ll1$, $u_N$ and the ordinary speed are numerically indistinguishable after unit conversion.

## Conserved fluxes

With $h=\varepsilon+P$ and $\gamma_v=\sqrt{1+u^2}$, the steady ideal-fluid first integrals are

$$J_B=n_Bu,\qquad \Pi=hu^2+P,\qquad E=hu\gamma_v.$$

Only the K current is not conserved:

$$\frac{dJ_K}{dx}=-\Gamma_K.$$

In the nonrelativistic transport formulation used by the working solver,

$$J_K=un_K-D_K\frac{dn_K}{dx}.$$

The upstream state and trial $u_N$ fix $J_B$, $\Pi$ and $E$. At each downstream position the EOS and these fluxes reconstruct $n_B$, $u$, $\mu_B$, $\mu_K$ and $T$ from the propagated K variables. See [[steady-front-bvp]].

## Interface and downstream conditions

The sharp interface preserves baryon, momentum, energy and K current. On the K channel,

$$u(0^+)n_K(0^+)-D_K(0^+)n'_K(0^+)=u(0^-)n_K(0^-).$$

K density itself need not be continuous: diffusion is defined only in deconfined matter, and deconfinement can reprocess the local composition. The ideal-fluid jump laws do not determine $n_K(0^+)$, so the original formulation treats $a(0^+)$ as an input and finds the velocity that reaches $n_K(\infty)=0$.

That local underdetermination is the [[interface-closure]] problem. The formal $T(0^+)\to0$ maximization is additionally restricted by [[lte-composition-bound]], and the [[thermal-conducting]] formulation makes $a(0^+)$ a global BVP output.

## ⚠ Relativistic K-current notation is not reconciled

The active first document in `Latex_writting/Hydrodynamical combustion/Main.tex` derives a covariant diffusion current whose steady spatial component contains $\gamma_v^2D_K n'_K$, but its reduced BVP and jump condition immediately revert to $D_Kn'_K$. The difference is $O(u^2)$ and negligible for the reported speeds, but the manuscript must either keep the factor consistently or explicitly state the nonrelativistic reduction.

The same section first writes the energy equation as $d(hu)/dx=0$ but later defines the exact conserved flux as $E=hu\gamma_v$, consistent with `Notes.tex`. The former is missing $\gamma_v$ in a relativistic presentation. Again the numerical correction is $O(u^2)$, but one convention must be used throughout.

## Sources

- `Latex_writting/Hydrodynamical combustion/Main.tex`, first active document: “Boundary propagation speed,” “Hydrodynamic equations,” and “Boundary conditions.”
- `Latex_writting/Hydrodynamical combustion/Notes.tex`: “State of the art” and the May 13 reduced-BVP notes. Older entropy-flux variants in this file are superseded by energy-flux conservation.
