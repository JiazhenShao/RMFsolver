---
summary: The baseline closure T(0+) = T(0-) — what it assumes, what it is NOT, and the scope it deliberately leaves alone.
status: current
updated: 2026-08-11
tags: [physics, closure, baseline]
---

# Thermally transparent closure

The recommended baseline fourth relation for [[interface-closure]]. Established 2026-07-23, still authoritative.

## The closure

For a thermally transparent microscopic deconfinement layer, impose

$$T(0^+) = T(0^-).$$

This is the zero-interfacial-thermal-resistance limit.

## Naming

Describe it as **locally isothermal** or **thermally transparent**. Never as *adiabatic* — that word implies something about entropy that is not being claimed, and the distinction has caused confusion before.

It is a **constitutive interface assumption**, supported by the short thermal-equilibration scale computed in [[quark-transport]] ($v\delta_{\rm int}/\chi_Q \approx 10^{-8}$–$10^{-7}$ for a 1–10 fm layer). It is *not* a consequence of the three ideal-hydrodynamic conservation equations alone, and should never be presented as one.

## Conditionality

The supporting estimate holds **conditional on the absence of a large interfacial thermal resistance**. That condition is exactly what cannot currently be checked — see [[kapitza-resistance]]. State the conditional; do not drop it.

Shovkovy & Ellis (2002) assume the same continuity across a directly contacting nuclear–quark interface, which is precedent rather than independent confirmation.

## Deliberate scope limit

The current scope leaves the $x > 0$ bulk solver **unchanged**. The existing conservation equations determine $T(x)$ algebraically within the ideal-hydrodynamic approximation; they do not contain a conductive flux $-\kappa\,dT/dx$.

Re-adding $q^\mu$ to the bulk stress tensor was explicitly rejected as the closure route on 2026-07-22. What happened instead is [[thermal-conducting]], which adds conduction as a *propagated field* in a separate solver rather than perturbing the ideal one.

## Read before editing

`Latex_writting/Hydrodynamical combustion/Main.tex`, the same directory's `Notes.tex`, or `RMFsolver/phase_velocity.py`.
