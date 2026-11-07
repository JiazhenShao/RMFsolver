---
summary: Every Main.tex figure from All_Plots_new prints axis and tick labels, titles, gray-region names and contour-figure legends at the 10 pt body size in Computer Modern (cmr10, 21.07 pt at 7.2 in), other text at 10.4 pt (22 pt) and contour labels at 6.2 pt (13 pt); every analytical-numerical comparison uses xi = 0; two sourceless schematics are not resized.
status: current
updated: 2026-11-06
tags: [paper, figures, fonts, contour]
---

# Manuscript figure standard

`Latex_writting/Main.tex` (revtex4-1, 10 pt, two-column) includes every figure at `\linewidth` $=246$ pt, so a font prints at size $\times\,246/(72\,W)$ for an exported width $W$ in inches.
Per the user (2026-10-29/30, revised 2026-11-03), all figures share one standard, stated at the 7.2 in width of the contour figures:

| Text | Source size at 7.2 in | Printed |
|---|---|---|
| Axis and tick labels, titles, gray-region names, contour-figure legends | 21.0732 pt, cmr10 | 10.0 pt, the body size and font |
| Trajectory legends and annotations | 22 pt, DejaVu Sans | 10.4 pt |
| Contour speed labels | 17 pt, cmr10 | 8.1 pt |

The body text of `Main.tex` is Computer Modern Roman at 10 pt (revtex4-1 default), so the body-size text uses matplotlib's bundled `cmr10` with mathtext `cm` math (cmmi10); the user does not require the size to be exact to the last digit.
Superseded 2026-11-03: axis labels and titles were 24 pt DejaVu Sans (11.4 pt printed); region names, legends and ticks 22 pt DejaVu Sans.
⚠ `cmr10` has no Unicode minus: the boundary-profile cells, whose $x$ ticks are negative, set `axes.formatter.use_mathtext` so the minus comes from cmsy10.

Mathtext sub- and superscripts print at $0.7\times$ (7.3 and 8.0 pt).
At other widths the sizes scale in proportion: the $(a,T)$ and $(\mu_B,\mu_K)$ trajectories export at about 6.1 in (bbox tight) and use 18.6 pt text.
On the bbox-tight figures (the trajectories and the boundary profiles) the cell sizes the axis labels from the width of an in-memory PDF export; the Agg tight bbox differs from the PDF one by up to 0.4 %.
The contour-label size is the adiabatic contour cell's former 11 pt plus 2, as the user asked; labels are placed by matplotlib `clabel`, never by hand, with a white square box (pad 0.08).
`tests/test_manuscript_figure_fonts.py` checks the cell constants, $\xi=0$ in both comparisons, and the printed sizes and fonts in the exported PDFs.

## Contour figures

All six contour cells (Fig. `speed-adiabatic`, `adiabatic-contour-compare`, both isothermal linear/log panels, the log-$T$ comparison of Fig. `isothermal-contour-compare` and the linear comparison of Fig. `isothermal-contour-compare-linear`, added 2026-11-03) use the x label "Nuclear baryon number density [$n_0$]" and the y label "Nuclear phase temperature [MeV]".
On the linear panels both region names are drawn centred on the point where the text box has the largest clearance from everything that is not its gray region (frame, ticks, white region), found on a render without text (user, 2026-11-03: balanced, never touching white): isothermal $(1.89, 19.9)$ and $(4.54, 79.4)$, adiabatic $(1.97, 13.5)$ and $(4.55, 103.3)$ in $(n_B/n_0, T/\mathrm{MeV})$, with clearances 83/89 and 26/19 pt; each comparison reuses its numerical panel's places.
Since 2026-11-06 (user) the two linear isothermal panels stop at 100 MeV (the udd boundary peaks at 94.7 MeV), which moved their names from $(1.94, 12.8)$ and $(4.17, 93.3)$.
Their contour-label boxes are clipped to the white region, so a box next to the udd boundary no longer paints over gray, and the 2000/5000 m/s labels moved from 30/15 to 18/10.5 MeV, where their digits clear the gray by at least 0.9 pt.
⚠ Contour speed labels are 17 pt (8.1 pt printed; user, 2026-11-06): 21.07 pt (10 pt printed) was oversized, its boxes covering too much and reaching the shaded regions, 13 pt too small and 19 pt slightly too big.
All six panels clip their label boxes to the unshaded region; the closest digits clear the shading by 2.2 pt.
No label box may cover a line of another speed (user, 2026-11-06: the 2000 box broke the 5000 line); covering its own level's dashed analytical line is accepted. On the linear isothermal panels the labels sit at 500/1000/2000/5000 m/s $\to$ 50/25/8/4 MeV; 2000 and 5000 sit below the 10 MeV join, labelled by `clabel` on an invisible copy of the row-mesh line, because above 10 MeV those contours hug the udd boundary and each other. On both adiabatic panels the 2000 label sits at 100 MeV, since elsewhere its box covers the dashed analytical 1000 line of the comparison.
Superseded 2026-11-03: both names sat at the lower-left corner $(1.08, 3)$ and at $(3.62, 98)$. A pixel test renders the cells and checks the names stay on gray.
⚠ The isothermal inner gray region is narrow above $T\approx15$ MeV: at the old $(1.18, 20)$ the 22 pt "Quark matter not favored" ran into the white region.
The adiabatic comparison canvas grew to $7.2\times8.8$ in so the title and the outside legend fit; its unbanded labels now carry the white box, because the dashed analytical 50 and 200 m/s curves ran through them.
The three isothermal log-$T$ panels share one shape (`LOG_PANEL_SHAPE`, user 2026-10-30): a $5.91\times4.85$ in axes box, $x$ from 2.2 to $6\,n_0$, both region names vertical on gray.
The method legend of the comparison panels does not fit the narrow left gray region at 22 pt, so it sits in one row between the title and the axes on a $7.2\times6.85$ in canvas (`LOG_COMPARISON_LEGEND`).
⚠ That row must stay narrower than the axes: any overhang widens constrained layout's side margins and shrinks the box (a 6.02 in row cost 0.02 in of axes width); a render test checks the axes box against the numerical panel and the legend against the title.
Both comparisons draw the analytical contours at $\xi=0$ (user, 2026-11-03; the captions and the appendix of `Main.tex` say so).
The isothermal one rescales the saved intermediates exactly; the adiabatic one uses `26-10-26/xi_fit/cached-xi-plus0.npy` from [[reusable-adiabatic-xi-cache]], loaded against the cache's recorded source hashes because every `phase_velocity.py` change since its build (7a3df347) is in the numerical slow-front solver.
$\xi=0$ there is a cached estimate; direct checks at $+0.07$ and $+0.74$ differed by at most $1.15\times10^{-4}$.
The fitted-$\xi$ isothermal panel (`isothermal-analytical-numerical-comparison.pdf`) was dropped as a duplicate; the stale PDF stays on disk, unincluded.
The isothermal cells take these values from `MANUSCRIPT_FONTS` / `MANUSCRIPT_LAYOUT` in `new_paper_calculations/26-10-27/isothermal/update_all_plots_new.py`; see [[isothermal-contours-polar-and-row-meshes]].

## Other figures

The $(a,T)$ trajectory (`T_a_constant_pressure.pdf`) and its $(T,\mu_K)$ twin now run to 42 MeV, with the legend in the headroom at the upper right, because the larger legend text crossed the dashed isobar; see [[notebook-trajectory-figures]].

Not resized:
- `pressure_mu_pos.pdf` (an xfig drawing, 8.2–12.3 pt printed at $0.65$ column) and `boundary_schematic.pdf` (3.7–5.3 pt printed) have no source in the workspace.
- `boundary_schematic_real.pdf` and its isothermal twin: the axis labels print at 10 pt, but ticks, panel names and phase names still print at about 11 pt (the test marks the full standard as an expected failure).
  Their profile cell was rerun on 2026-11-03 to pass the provenance guard; the speeds reproduced exactly (104.140816 and 125.727416 m/s).
