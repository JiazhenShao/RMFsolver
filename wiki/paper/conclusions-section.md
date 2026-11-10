---
summary: Main.tex Conclusions rewritten 2026-11-09 in four paragraphs (mechanism; 100–1000 m/s upper estimates; instantaneous conversion in merger simulations unjustified; caveats (a)–(e) as next steps); verbatim LaTeX source, caveat selection and open author decisions kept here.
status: current
updated: 2026-11-10
tags: [paper, manuscript, conclusions, combustion]
---

# Conclusions section of Main.tex

The Conclusions of `Latex_writting/Main.tex` (`\section{Conclusions}`, `\label{sec:conclusion}`) were rewritten by JS on 2026-11-09 with the paper-writing skill's Alford conclusions recipe. The new text is wrapped in `\JS{...}`; the previous MGA/LB/AH draft, with its caveat bullets and comments, sits directly below it in an `\iffalse ... \fi` block. Context for the rest of the manuscript is in [[combustion-paper]].

⚠ 2026-11-10, 12:05: `Main.tex` on disk was replaced by a version without this rewrite (the old `\section{Conclusion}` with the MGA/LB/AH bullets is back, as are pre-2026-11-09 captions, since re-corrected). Until the authors re-insert it, the text below exists only here.

⚠ `Main.tex` itself is not committed together with this page; its working copy carries many other uncommitted edits. This page is the durable copy of the 2026-11-09 text. If the live `Main.tex` differs, the live file is the current text, and this page records the version and the selection decisions behind it.

## Structure

| Paragraph | Content |
|---|---|
| 1. Recap | Weak-interaction-limited conversion: deconfined quark matter is short of strangeness, so the boundary moves only as fast as $d\,u\leftrightarrow u\,s$ creates it and diffusion returns it; isothermal and adiabatic limits solved, analytic speeds Eqs. `isothermal_speed_scaling`, `adiabatic_speed_scaling` |
| 2. Numbers | For $T\gtrsim1$ MeV both limits give at most 100 to 1000 m/s, upper estimates by construction; zero speed at the transition pressure, divergence only near the edge where no weak interactions are needed; isothermal speed rises as $T$ falls (K diffusion grows with the mean free path); adiabatic speed independent of $T(0^-)$, set by $n_B(0^-)$ and quark-matter properties that change little when $T\ll\mu_B$ |
| 3. Take-home | Instantaneous conversion in merger simulations would need $\sim10^6$ m/s; at 1000 m/s the boundary moves ~1 m per ms, so on the ~0.4 ms post-merger timescale the phase content is closer to frozen, though a long-lived remnant still crosses simulation cells (Sec. `merger_impact`); interface-mode calculations need the finite boundary speed and its dissipation |
| 4. Caveats as next steps | (a)–(e) below, ending on the reverse conversion |

## Caveat selection

Kept, as an inline (a)–(e) list:

- (a) beta-equilibrated nuclear matter (protons, leptons, perhaps hyperons) instead of pure neutron matter, which adds slower Urca processes, so the speeds stay upper bounds (merges LB's paragraph and AH's first two bullets);
- (b) pairing at low $T$, where the isothermal speeds are largest;
- (c) breakdown of local thermal equilibrium once the mean free path reaches the profile thickness, which cuts off the $T\to0$ rise (taken from the Results text);
- (d) finite thermal conductivity between the two limits;
- (e) the reverse quark-to-nuclear conversion and the phase-conversion dissipation it allows, which also answers AH's comment.

Dropped: neutrinos (the old bullet itself argues trapping is unlikely), constant upstream profiles, validity of the steady state, the ideal-fluid versus dissipative stress tensor, "surface tension?", and the LB/AH collapse speculation.

## Open decisions for the authors

- MGA's old line "For the inspiral phase, we find that the speed may be slow enough to allow interface modes" was not carried over, because Sec. V D argues the speeds overlap the 1–15 m/s resonant range and damp the mode. The authors should settle which reading is right.
- Supernovas enter only through the $T\gtrsim1$ MeV conditions sentence (partly answering MGA's "or supernova?"); there is no supernova-specific timescale argument, because the body makes none. See [[supernova-quark-oscillations]].
- AH flags the Sec. V D merger numbers as needing an update ("use a hot plot"). Paragraph 3 borrows its 0.4 ms timescale and the cell-crossing statement, so it must be rechecked if Sec. V D changes.

## Verification

`style_stats.py check --as conclusions` reports no 8-word spans shared with the Alford corpus. Its sentence-length sd flag is an artifact: the splitter does not break before "(a)" items, and the real sentences run 13–38 words. The file compiled cleanly in a scratch build (two passes, all references resolved).

## Verbatim source (2026-11-09)

```latex
\section{Conclusions}
\label{sec:conclusion}

\JS{
We have calculated the maximum speed at which a planar boundary between metastable neutron matter and quark matter can advance when the conversion requires weak interactions.
This is the case when the ambient pressure is high enough for beta-equilibrated quark matter to be favored over neutron matter, but not high enough for quark matter with the flavor content of neutron matter to be favored.
Quark matter deconfined at the boundary is then short of strangeness, so the boundary can only advance as fast as the nonleptonic process $d\,u\leftrightarrow u\,s$ creates strangeness behind it and diffusion carries it back to the boundary.
We solved the steady-state equations for the boundary profile in the isothermal and adiabatic limits of heat transport, and derived approximate analytic expressions for the speed in each limit, Eqs.~\eqref{eq:isothermal_speed_scaling} and \eqref{eq:adiabatic_speed_scaling}.

Under the conditions expected in mergers and supernovas, $T\gtrsim 1\,\MeV$, both limits give speeds of order 100 to $1000\,\m\s^{-1}$ at most (Figs.~\ref{fig:speed-isothermal} and \ref{fig:speed-adiabatic}).
Since we chose our approximations and parameters to favor fast propagation, these are upper estimates.
The speed drops to zero as the ambient pressure approaches the transition pressure.
It grows without bound only near the opposite edge of the metastable region, where the freshly created quark matter can have the flavor content of neutron matter and weak interactions are no longer needed.
At low temperature the isothermal speed rises as $T$ falls, because the K diffusion coefficient grows with the mean free path.
The adiabatic speed instead becomes independent of the temperature of the incoming neutron matter, since it is set by $n_B(0^-)$ and by properties of the quark matter that change little when $T\ll\mu_B$.

We conclude that a pressure-driven nuclear/quark boundary is far too slow to justify the instantaneous conversion assumed in merger simulations, which would need speeds of order $10^6\,\m\s^{-1}$ to convert a kilometer-sized region in a millisecond.
At $1000\,\m\s^{-1}$ the boundary advances about a meter in a millisecond, so on the hydrodynamic timescale of the post-merger remnant, about $0.4$\,ms, the phase content is closer to frozen than to equilibrated.
In a long-lived remnant, however, the boundary would still cross several cells of a high-resolution simulation (Sec.~\ref{sec:merger_impact}).
In the inspiral the boundary can move a distance comparable to the amplitude of an interface mode within one oscillation period.
Calculations of these modes should therefore include the finite boundary speed and the dissipation it causes, rather than treating the phase content as frozen.

Our model is deliberately simple, and most of the ways in which it could be improved would make the boundary slower still.
(a) The nuclear phase should be beta-equilibrated nuclear matter containing protons, electrons and muons, and perhaps hyperons, rather than pure neutron matter.
Conversion would then also require Urca processes, which are slower than the nonleptonic process, so in this respect too our speeds are upper bounds.
(b) At the low temperatures of the inspiral, where our isothermal speeds are largest, nuclear and quark matter may be superfluid or color superconducting, which would suppress the flavor-changing rates and change the diffusion of K.
(c) Those low-temperature speeds also rely on local thermal equilibrium, which fails once the mean free path becomes comparable to the thickness of the boundary profile, so their rise as $T\to 0$ will be cut off.
(d) Heat transport was bracketed here by the isothermal and adiabatic limits, and a calculation with finite thermal conductivity would show where between them the real speed lies, and how the two phases reach a common temperature behind the boundary.
(e) Pressure oscillations in the inspiral and merger would also drive the reverse conversion, from quark matter to nuclear matter.
Calculating its speed would allow us to estimate the phase-conversion dissipation that arises when the boundary lags behind the oscillating pressure~\cite{Alford:2014jha}.
}
```
