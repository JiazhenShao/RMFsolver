---
summary: The front-speed paper — where the files live, house conventions, and what still blocks submission.
status: current
updated: 2026-11-12
tags: [paper, writing, hub]
---

# Hydrodynamical combustion paper

The project's primary output. Alford / Brodie / Haber / Shao. PRC two-column, natural units $c=1$.

## Current build (2026-10-19)

The active manuscript is `Latex_writting/Main.tex`, built from `Latex_writting/` with `latexmk -pdf -interaction=nonstopmode -halt-on-error Main.tex`.
This supersedes the older nested source path and manuscript-state observations below; inspect the live source before relying on them.
The October 19 build repair restored `Hydrodynamical combustion/Plots/isothermal-analytic-numerical-logT.pdf` from the identical saved PDF under `local_analysis_results/26-10-02/`, separated the main-text and appendix isothermal speed labels, and repaired two stale cross-references.
Four missing bibliography entries were retrieved verbatim from INSPIRE-HEP: `Ghosh:2026ldg`, `Courant:1928lij`, `Kiuchi:2026pgb`, and `ParticleDataGroup:2026mpi`.
Regenerating the malformed `Main.bbl` with BibTeX allowed a successful 17-page build with no undefined citations or references; existing package/float warnings and a 5.1-point equation overfull box remain.

## Files

| Path | Role |
|---|---|
| `Latex_writting/Hydrodynamical combustion/Main.tex` | the paper |
| `Latex_writting/Hydrodynamical combustion/Notes.tex` | working derivations |
| `Latex_writting/reflist.bib` | bibliography — INSPIRE-HEP exports **verbatim** |
| `Latex_drop_in_integrated/` | candidate sections not yet merged |
| `Latex_writting/Hydrodynamical combustion/Plots/` | figures |

Notation: $p_{\rm crit}$, $u = \gamma v$, $\gamma_K$, $\lambda_n$.

## Appendix order

As of 2026-09-17, Heat Transport follows Strangeness diffusion; its previous position as Appendix A is superseded.
The compiled order is A: Bag model of quark matter; B: Strangeness creation; C: Strangeness diffusion; D: Heat Transport; E: Shape of the diffusive flux profile; F: Numerical method; G: Local thermal equilibrium.
Use the existing section and equation labels for references rather than literal appendix letters.
The move leaves appendix content unchanged apart from replacing two hard-coded references in author comments.

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

The active manuscript is still a collaborative draft: its abstract is `Place holder`, and it contains author comments, hidden diagnostics and formal inconsistencies documented on [[diffusion-limited-front]], [[analytic-front-speed]], [[quark-matter-eos]] and [[steady-front-bvp]]. Author comments, `\hide{...}` / `\iffalse` bodies and the ignored duplicate are evidence of work in progress, not current manuscript claims.

The Conclusions section was rewritten on 2026-11-09 (JS, Alford-style recipe, marked `\JS{}`): recap of the weak-interaction-limited mechanism, speeds of order 100 to 1000 m/s at most for $T\gtrsim1$ MeV in both limits, the isothermal low-$T$ rise and adiabatic $T$-independence, the take-home that instantaneous conversion in merger simulations is unjustified, and five caveats as next steps (beta-equilibrated nuclear matter with Urca, pairing at low $T$, breakdown of local thermal equilibrium, finite thermal conductivity, reverse conversion and phase-conversion dissipation). Dropped from the old bullet list: neutrinos, upstream disturbances, the steady-state assumption, ideal-fluid dissipation, surface tension and the collapse speculation; the old draft stays in an `\iffalse` block. MGA's earlier line that inspiral speeds "may be slow enough to allow interface modes" was not carried over, because Sec. V D argues the speeds overlap the resonant range instead. Full text and selection rationale: [[conclusions-section]].

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
- **The K fraction is typeset by the macro `\Kfr`** (currently $y_K$, adopted 2026-11-12 on Brodie's "change $a$ to $Y_K$ throughout?" comment). Write `\Kfr`, never a bare `a`, in new Main.tex math; relabel the whole paper by editing that one `\newcommand`. `\azp` is `\Kfr(0^+)`. The bounds formerly $A_{\rm iso}$, $A_{\rm adb}$ are `\Kfriso`, `\Kfradb` ($y_K^{\rm iso}$, $y_K^{\rm adb}$); write their powers as `(\Kfradb)^2`, since `\Kfradb^2` is a double superscript. The wiki and code keep $a$, $A_{\rm iso}$, $A_{\rm adb}$. Figure labels follow (All_Plots_new cells 3, 5, 11 re-exported 2026-11-12) except the xfig `boundary_schematic.pdf`, which still prints $a$ and has no source in the workspace.
- Oversized displays break the two-column layout; equations that overflow need `aligned` splitting.
- One sentence per source line in `.tex` prose (project-wide rule).
- **This directory is not a git repository.** There is no diffing, no stash, no commit. Every edit is additive and surgical, and a stale `.py.old` may be the only backup of something.
