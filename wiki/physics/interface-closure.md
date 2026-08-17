---
summary: The nuclear→quark front is one equation short; this page holds the closure hierarchy and which closure is currently authoritative.
status: current
updated: 2026-08-11
tags: [physics, closure, core]
---

# The interface closure problem

This is the central open question of the [[combustion-paper|combustion paper]]. Everything in [[lte-composition-bound]], [[interface-temperature-jump]], [[kapitza-resistance]] and [[thermal-conducting]] exists to address it.

## The counting

The upstream state is known: $T(0^-)$, $n_B(0^-)$ (or $\mu_B(0^-)$), and a trial $u(0^-)$. The just-downstream state carries **four** unknowns,

$$\mu_B(0^+),\quad \mu_K(0^+),\quad T(0^+),\quad u(0^+).$$

Baryon-number, energy-flux and momentum-flux conservation supply **three** equations. A fourth sharp-interface relation is therefore required, and it cannot come from ideal hydrodynamics.

The reason is structural, not accidental. Ideal hydrodynamics uses $T^{\mu\nu} = (P+\varepsilon)U^\mu U^\nu + P g^{\mu\nu}$, which carries no heat flux. The missing fourth relation *is* the heat equation, deleted when $q^\mu$ was dropped from the stress tensor.

## The closure hierarchy

Ordered by how much they are trusted, as of 2026-07-23:

1. **Baseline — thermally transparent interface.** Impose $T(0^+) = T(0^-)$. This is the zero-interfacial-thermal-resistance limit. See [[thermal-transparent-closure]].
2. **Independent validity bound.** Cap $a(0^+)$ using the LTE composition bound and infer a *lower* bound on $T(0^+)$. See [[lte-composition-bound]]. This is not a closure — it is a constraint on any closure.
3. **Structural resolution.** Promote $T$ to a propagated ODE field with Fourier conduction, which makes $a(0^+)$ an **output** rather than an input. See [[thermal-conducting]]. This is the strongest available result.
4. **Future extension.** Introduce and scan a finite interfacial conductance $G_{\rm int}$ only when a defensible microscopic model exists. See [[kapitza-resistance]] — currently blocked.

## Terminology warning

Call option 1 a **locally isothermal** or **thermally transparent** interface. Do **not** call it adiabatic. The equality is a constitutive interface assumption supported by a short thermal-equilibration scale; it is not a consequence of the three ideal-hydrodynamic conservation equations.

## The fixed-temperature equilibrium-isobar state is only a candidate

The state obtained by solving

$$P_{\rm QM}(\mu_B,\mu_K=0,T(0^+))=P(0^-)$$

at an arbitrary upstream point is not generally the state at $0^+$ or at $\infty$.
It is an auxiliary equilibrium-isobar candidate used to test whether the conserved conversion-layer trajectory can reach the prescribed interface temperature.
Its chemical potential must therefore be named as a candidate, not as `muB_0plus` or `muB_inf`.

At the zero-speed boundary the composition and temperature profiles collapse: $\mu_K(0^+)=\mu_K(\infty)=0$ and $T(0^+)=T(\infty)$.
Only there does the candidate become the actual downstream endpoint.
The boundary can equivalently be found by solving the actual slow-limit downstream endpoint from

$$P(\infty)=P(0^-),\qquad \frac{h(\infty)}{n_B(\infty)}=\frac{h(0^-)}{n_B(0^-)},\qquad \mu_K(\infty)=0,$$

and then imposing $T(\infty)=T(0^+)$.
Pressure equality alone is one equation for the two endpoint unknowns $\mu_B(\infty)$ and $T(\infty)$ and is not a closed endpoint solve.
For finite velocity, use the full momentum- and energy-flux equations rather than these slow-limit equalities.

## Why $\Delta h=0$ gives the zero-speed boundary

For a prescribed $T(0^+)$, define the auxiliary equilibrium-isobar candidate by $\mu_K=0$ and

$$P_{\rm QM}(\mu_B,0,T(0^+))=P(0^-),$$

then evaluate

$$\Delta h\equiv\left.\frac{h}{n_B}\right|_{\rm candidate}-\frac{h(0^-)}{n_B(0^-)}.$$

For a nontrivial moving front, the downstream layer has finite extent in state space:

$$\mu_K(0^+)>\mu_K(\infty)=0,\qquad T(0^+)<T(\infty).$$

The condition $\Delta h=0$ makes the equilibrium-isobar candidate also satisfy the slow-limit energy-per-baryon matching condition.
At that root it is the actual downstream endpoint, so

$$T(0^+)=T(\infty),\qquad \mu_K(0^+)=\mu_K(\infty)=0.$$

The entire $x>0$ conversion profile is therefore trivial: the interface state is already the equilibrated endpoint, the composition gradient and integrated weak source vanish, and the reaction--diffusion speed functional has $I_2=0$.
Consequently the finite-speed branch terminates at $u(0^-)=0$.
The statement is stronger than the fact that $\Gamma_K(\infty)=0$, which holds for every moving front; zero speed occurs because the whole interval from $0^+$ to $\infty$ collapses.

At exactly zero baryon flux, energy-flux conservation alone is degenerate.
Thus $\Delta h=0$ should be understood as the continuous $u(0^-)\to0^+$ endpoint of the moving-front branch, not as a universal condition on arbitrary static interfaces.

## Why option 3 is the significant one

In the ideal-fluid treatment, $a(0^+)$ is unconstrained and the paper maximises the front speed over it. With conduction present, the boundary-condition count closes with $a(0^+)$ determined by the global solution. That is a stronger claim than merely removing the non-Lipschitz plateau documented in [[unmax-degeneracy]], and per the 2026-08-05 audit it may only be stated after the seed-, mesh- and schedule-independence tests pass.

## Related

- [[diffusion-limited-front]] — the ideal front model in which the missing relation appears
- [[steady-front-bvp]] — the reduced BVP and boundary-condition count
- [[phase-velocity-overview]] — which solver implements which closure
- [[quark-transport]] — the transport coefficients all of this rests on
