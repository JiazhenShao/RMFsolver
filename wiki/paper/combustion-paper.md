---
summary: The front-speed paper — where the files live, house conventions, and what still blocks submission.
status: current
updated: 2026-08-17
tags: [paper, writing, hub]
---

# Hydrodynamical combustion paper

The project's primary output. Alford / Brodie / Haber / Shao. PRC two-column, natural units $c=1$.

## Files

| Path | Role |
|---|---|
| `Latex_writting/Hydrodynamical combustion/Main.tex` | the paper |
| `Latex_writting/Hydrodynamical combustion/Notes.tex` | working derivations |
| `Latex_writting/reflist.bib` | bibliography — INSPIRE-HEP exports **verbatim** |
| `Latex_drop_in_integrated/` | candidate sections not yet merged |
| `Latex_writting/Hydrodynamical combustion/Plots/` | figures |

Notation: $p_{\rm crit}$, $u = \gamma v$, $\gamma_K$, $\lambda_n$.

## Physics content

The nuclear→quark deflagration front, its speed, and what closes the interface:

- [[diffusion-limited-front]] — physical regime, assumptions, conserved fluxes and interface conditions.
- [[steady-front-bvp]] — reduced numerical formulation and boundary/eigenvalue counting.
- [[analytic-front-speed]] — piecewise-constant derivation, scaling, conversion-layer thickness and validity limits.
- [[quark-matter-eos]] and [[strangeness-reaction-diffusion]] — thermodynamic and microscopic inputs.
- [[interface-closure]], [[lte-composition-bound]] and [[thermal-conducting]] — how the formerly free interface composition is constrained or selected.

## Source precedence and manuscript state

`Main.tex` is newer than `Notes.tex` (2026-08-06 versus 2026-06-12). Use the **first document body only**, ending at its first `\end{document}` around line 1378. The file then contains comments followed by a stray `∫\documentclass` and a second full document ending around line 2880; TeX ignores that second body. Do not synthesize both copies as independent evidence.

`Notes.tex` is working derivation history. Its top “State of the art” section is useful for the June formulation, but its entropy-flux variants and coefficient-3 LTE derivation are superseded by the active manuscript and the corrected wiki pages. Where a newer wiki conclusion conflicts with either TeX file, follow the wiki and preserve the disagreement as history.

The active manuscript is still a collaborative draft: its abstract is `Place holder`, its conclusion is unfinished, and it contains author comments, hidden diagnostics and formal inconsistencies documented on [[diffusion-limited-front]], [[analytic-front-speed]], [[quark-matter-eos]] and [[steady-front-bvp]]. Author comments, `\hide{...}` / `\iffalse` bodies and the ignored duplicate are evidence of work in progress, not current manuscript claims.

In particular, the long postmerger-GW block in `Main.tex` is inside `\hide{...}` and does not compile. The newer standalone state recorded in [[gw-observables-section]] is authoritative. Likewise, the manuscript's $T(0^+)=0$ maximization/floor discussion is superseded by [[unmax-degeneracy]], [[lte-composition-bound]] and [[thermal-conducting]].

## Candidate sections

- [[gw-observables-section]] — framework complete, numbers in, citations corrected. Reviewed 2026-07-14 as internally consistent but needing compression; the $K_2/I_2$ mode formalism was flagged for cutting or moving.

## Open blockers

1. **[[kapitza-resistance]]** — no literature value for $G_{\rm int}$. The honest position is to state the gap, not close it.
2. **[[unmax-degeneracy]]** — any published $T(0^+)=0$ uNmax number is contaminated. Results must come from the analytic bound or from [[thermal-conducting]].
3. The ~8% systematic between the BVP and the reference solver, seen plateau-free, is **still unexplained**.
4. Clark:2015zxa resolution figure differs between the arXiv and Consensus-indexed abstracts — check the published CQG before submission.
5. Reconcile the active manuscript's relativistic $\gamma_v^2D_Kn'_K$ current with the factor-free reduced BVP and jump condition ([[diffusion-limited-front]]).
6. Reconcile the electron-free EOS with the exact-rate path that imposes electric charge neutrality ([[quark-matter-eos]]).
7. Resolve the manuscript's claim that $\xi=0$ matches smooth profiles versus its diagnostic showing otherwise ([[analytic-front-speed]]).
8. Fix the active manuscript's $d(hu)/dx=0$ line to match its exact conserved $E=hu\gamma_v$ flux ([[diffusion-limited-front]]).

## Conventions that have caused rework

- $\gamma$ was used for three different meanings in an early GW draft. Check before adding notation.
- Oversized displays break the two-column layout; equations that overflow need `aligned` splitting.
- One sentence per source line in `.tex` prose (project-wide rule).
- **This directory is not a git repository.** There is no diffing, no stash, no commit. Every edit is additive and surgical, and a stale `.py.old` may be the only backup of something.
