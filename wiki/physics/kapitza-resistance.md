---
summary: The genuine interfacial thermal resistance G_int — why it is the honest gap in the framework and why it cannot currently be computed.
status: open-question
updated: 2026-08-11
tags: [physics, closure, open-question]
---

# Kapitza resistance at the nuclear–quark interface

The acknowledged gap in [[interface-closure]]. Flagged here because it is the natural referee question and there is no literature value to answer it with.

## What it is

A genuine interface temperature jump requires an interfacial conductance $G_{\rm int}$:

$$q_{\rm int} = G_{\rm int}\big(T(0^-) - T(0^+)\big).$$

This is physically distinct from the [[lte-composition-bound]] validity jump. It is nonzero even when both sides are in perfect local thermal equilibrium, arising from the mismatch between confined nucleon heat carriers and deconfined quark carriers.

## What is settled

The conductive entropy production is nonnegative (Wolfram-verified):

$$q_{\rm int}\left(\frac{1}{T(0^+)} - \frac{1}{T(0^-)}\right) = \frac{G_{\rm int}\big(T(0^-)-T(0^+)\big)^2}{T(0^-)\,T(0^+)} \ge 0.$$

## What is not

That inequality constrains the **sign** of the dissipation. It does not determine the **magnitude** of the temperature jump.

Critically: **bulk $\kappa_N$ and $\kappa_Q$ do not determine $G_{\rm int}$.** Knowing the transport coefficients on both sides ([[quark-transport]]) is not enough.

## Literature status

A literature search found **no dense-matter calculation of $G_{\rm int}$ for a sharp nuclear–quark interface**. Nearest analogues, none of which supply a number for this system:

- Reddy et al. 2005 — diffuse-mismatch model
- Shi et al. 2018 — radiation-limit conductance $h_K \sim \tfrac14 C v$
- Buchbinder et al. 2017 (Physica A); Palmieri et al. 2012 (PRE 86, 051605) — moving planar front with a Kapitza interface discontinuity; structural analogue of the combustion front
- Péraud & Hadjiconstantinou 2015 (PRB 93, 045424); Meilakhs 2019 — kinetic temperature-jump boundary condition at Knudsen order
- [[heiselberg-pethick-1993]] does **not** cover it

Shovkovy & Ellis 2002 simply *assume* temperature continuity across a directly contacting nuclear–quark interface, which is the same assumption as the [[thermal-transparent-closure]] baseline — supporting precedent, not independent evidence.

## Position to take

Treat $T(0^+) = T(0^-)$ as the baseline, state the $G_{\rm int}$ gap explicitly rather than burying it, and frame a finite-$G_{\rm int}$ scan as future work. Do not build an $R_K$ solver — it was ruled out of scope on 2026-07-22 and nothing since has changed that.
