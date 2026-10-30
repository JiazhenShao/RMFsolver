---
summary: Every Main.tex figure from All_Plots_new prints 10.4 pt text, 11.4 pt axis labels/titles and 6.2 pt contour labels (22/24/13 pt at 7.2 in); contour axes read "Nuclear baryon number density [n0]" and "Nuclear phase temperature [MeV]"; two sourceless schematics and the boundary-profile pair are not yet resized.
status: current
updated: 2026-10-30
tags: [paper, figures, fonts, contour]
---

# Manuscript figure standard

`Latex_writting/Main.tex` (revtex4-1, 10 pt, two-column) includes every figure at `\linewidth` $=246$ pt, so a font prints at size $\times\,246/(72\,W)$ for an exported width $W$ in inches.
Per the user (2026-10-29/30), all figures share one standard, stated at the 7.2 in width of the contour figures:

| Text | Source size at 7.2 in | Printed |
|---|---|---|
| Ticks, legends, region names, annotations | 22 pt | 10.4 pt |
| Axis labels, titles | 24 pt | 11.4 pt |
| Contour speed labels | 13 pt | 6.2 pt |

Mathtext sub- and superscripts print at $0.7\times$ (7.3 and 8.0 pt).
At other widths the sizes scale in proportion: the $(a,T)$ and $(\mu_B,\mu_K)$ trajectories export at about 6.1 in (bbox tight) and use 18.6 / 20.3 pt.
The contour-label size is the adiabatic contour cell's former 11 pt plus 2, as the user asked; labels are placed by matplotlib `clabel`, never by hand, with a white square box (pad 0.08).
`tests/test_manuscript_figure_fonts.py` checks the cell constants and the printed sizes in the exported PDFs.

## Contour figures

All six contour cells (Fig. `speed-adiabatic`, `adiabatic-contour-compare`, both isothermal linear/log panels, the log-$T$ comparison of Fig. `isothermal-contour-compare` and the fitted-$\xi$ comparison) use the x label "Nuclear baryon number density [$n_0$]" and the y label "Nuclear phase temperature [MeV]".
On the linear panels both region names sit wholly on gray, at $(1.08, 3)$ and $(3.62, 98)$ in $(n_B/n_0, T/\mathrm{MeV})$, the same places in all three; a pixel test renders the cells and checks this.
⚠ The isothermal inner gray region is narrow above $T\approx15$ MeV: at the old $(1.18, 20)$ the 22 pt "Quark matter not favored" ran into the white region.
The adiabatic comparison canvas grew to $7.2\times8.8$ in so the title and the outside legend fit; its unbanded labels now carry the white box, because the dashed analytical 50 and 200 m/s curves ran through them.
The three isothermal log-$T$ panels share one shape (`LOG_PANEL_SHAPE`, user 2026-10-30): a $5.91\times4.85$ in axes box, $x$ from 2.2 to $6\,n_0$, both region names vertical on gray.
The method legend of the comparison panels does not fit the narrow left gray region at 22 pt, so it sits in one row between the title and the axes on a $7.2\times6.85$ in canvas (`LOG_COMPARISON_LEGEND`).
⚠ That row must stay narrower than the axes: any overhang widens constrained layout's side margins and shrinks the box (a 6.02 in row cost 0.02 in of axes width); a render test checks the axes box against the numerical panel and the legend against the title.
The isothermal cells take these values from `MANUSCRIPT_FONTS` / `MANUSCRIPT_LAYOUT` in `new_paper_calculations/26-10-27/isothermal/update_all_plots_new.py`; see [[isothermal-contours-polar-and-row-meshes]].

## Other figures

The $(a,T)$ trajectory (`T_a_constant_pressure.pdf`) and its $(T,\mu_K)$ twin now run to 42 MeV, with the legend in the headroom at the upper right, because the larger legend text crossed the dashed isobar; see [[notebook-trajectory-figures]].

Not resized:
- `pressure_mu_pos.pdf` (an xfig drawing, 8.2–12.3 pt printed at $0.65$ column) and `boundary_schematic.pdf` (3.7–5.3 pt printed) have no source in the workspace.
- `boundary_schematic_real.pdf` and its isothermal twin print 11.1 pt throughout. Their cells refuse to run because `RMFsolver` changed after the matched profiles were solved (provenance guard), so they need the profile cell rerun first; the test marks this as an expected failure.
