---
summary: All_Plots_new exports matched trajectory figures and a four-panel numerical adiabatic boundary profile from saved data.
status: current
updated: 2026-10-26
tags: [methods, notebooks, figures, manuscript]
---

# Trajectory Figure Exports

`All_Plots_new.ipynb` uses solid red numerical profiles, dashed black reference curves, and matching fonts and endpoint markers for its two manuscript trajectories.
The adiabatic plot labels the reference curve "Analytic approximation", the red curve "Numerical solution", and the cold intercept `A_adb`.
Since 2026-10-30 (user) $A_{\rm adb}$ sits at the upper right of the open $T=0$ marker and the red star is labelled $a(0^+)$ instead of $x=0^+$: the old $A_{\rm adb}$ label stood beside the star's dotted projection and read as the star's $a$, although the star is at $a(0^+)=0.538$ and the intercept at $A_{\rm adb}=0.571$.
The isothermal reference is the full neutral constant-pressure isobar, not the quadratic approximation; its legend says "Quark-matter isobar".

The original isothermal plotting cell is retained.
The new code cell immediately below it has ID `isothermal-matching-trajectory-style`; it keeps the positive composition-imbalance branch and the upstream point, without the old flow arrows or curved crossing arrow.
Run this new cell last because both versions overwrite `Latex_writting/Hydrodynamical combustion/Plots/isothermal_muB_muK_plane.pdf`.
The adiabatic plot overwrites `T_a_constant_pressure.pdf` in the same directory.
Figure input labels use `U_bag^(1/4)` rather than `B^(1/4)`.

The isothermal calculation cell now saves `local_analysis_results/All_Plots_new/isothermal_mu_plane_plot_data.npz` as well as its original profile.
The new plotting cell loads this cache if its in-memory payload is absent, so it does not require a BVP rerun after a kernel restart.
For the 2026-10-15 export, the isobar was reconstructed from the saved neutral massless EOS profile without importing the production solver; relative pressure residuals were below 1e-15 and the interface chemical potential agreed to 2.3e-13 MeV.
No numerical profiles or solver files changed.

Three trajectory-style tests and eight existing contour-band tests pass, and all 23 unrelated notebook cells were preserved exactly.
The two figure PDFs and notebook outputs were refreshed and visually checked.
Only the two matching captions in `Latex_writting/Main.tex` were updated: they use the bag-energy notation, describe the new isothermal markers rather than the removed arrows, and quote the cached adiabatic interface fraction 0.538 rather than 0.55.
The completed adiabatic-label comment was removed; the manuscript body was not changed.
The manuscript rebuild succeeds with its pre-existing undefined-citation/reference and float-placement warnings.
See [[quark-matter-eos]] and [[steady-front-bvp]].

## Numerical Boundary Overview (2026-10-19)

The cell `adiabatic-boundary-profile`, immediately after the shared profile-generation cells in `All_Plots_new.ipynb`, replaces the schematic with four shallow panels for baryon density, fluid speed, K fraction, and temperature.
It reads the existing `local_analysis_results/All_Plots_new/energy_conserving_nK_TN10_TQstar3_nBN3n0_B180.csv` without importing or rerunning the solver.
The legacy filename is misleading: this saved profile has T(0-) = 10 MeV, T(0+) = 6.990575 MeV, nB(0-) = 3 n0, U_bag^(1/4) = 180 MeV, and a(0+) = 0.538.
All 200 quark samples are plotted directly, after unit conversion, through 548.219 nm; the final composition is 0.003318, not exactly zero, and the endpoint is not labeled infinity.
Density jumps from 3 to 4.527883 n0 and speed from 137.627 to 91.1864 m/s; both remain constant on the quark side in this massless example, unlike the former sketch.
The green/magenta/blue curves retain the original colors, with red added for temperature, open axes, and a dotted interface marker.
The cell overwrites `Latex_writting/Hydrodynamical combustion/Plots/boundary_schematic.pdf` and writes a PNG to `local_analysis_results/All_Plots_new/adiabatic_boundary_profile.png`.
As of October 20, the notebook exports instead to `boundary_schematic_real.pdf` in the same folder, preserving `boundary_schematic.pdf` for the separate original figure.
Axis labels, ticks, panel labels, and phase headings use 22-point source fonts, which render at 11.37 points at the manuscript's full column width, above its 10-point body text.
The source figure was initially 6.4 by 5.1 inches; the October 20 layout revision increases it to 6.4 by 6.6 inches, includes zero on the density/speed axes, and centers the interface between equal-width phase regions.
The headings fit wholly within their respective regions, with symmetric flow arrows; seven profile/layout tests now cover these constraints as well as font scaling and unchanged numerical data.
On October 26, the x limits changed to -400 through 600 nm for an exact 40:60 neutron/quark split, with ticks every 200 nm on both sides.
The phase headings use the narrower STIX serif at the unchanged source font size so they remain inside their regions; the exported full-column text size is about 11.0 points.
Eight profile/layout tests pass, including the split and negative-tick checks, and the notebook overwrites the same `boundary_schematic_real.pdf` referenced by Main.tex.
Only the corresponding figure width and caption changed in `Latex_writting/Main.tex`; its 17-page PDF builds with no unresolved references, and the figure was checked on page 2.
All 27 pre-existing notebook cells are preserved, and four numerical-profile checks plus three existing trajectory-style tests pass.

## Historical Momentum-Flux Profile

The original constant-momentum-flux contour figure was located in `26-04-09.ipynb`, cell 9 (zero based), with its matching vector PDF under `jargon/entropy_tqguess_uagrad_pi_qm_contour.pdf`.
The saved figure has interface temperature 0.10 MeV and downstream temperature 17.34 MeV; the old calculation cell has since been changed to different inputs, so a top-to-bottom rerun does not reproduce the stored output.
No matching raw profile was found.

The final two cells of `26-10-14.ipynb` now record the provenance and replot the original PDF's six vector contours, four sample points, and three endpoints using location labels and larger fonts.
Coordinates and source hash are cached in `local_analysis_results/26-10-14/constant-momentum-profile-data.json`; no raster digitization, new EOS/BVP run, or solver edit was used.
The endpoint change is approximately 0.073 MeV (0.0064%) at figure precision, supporting a nearly constant baryon chemical potential for this historical example, not uniformly across current-model solutions.
The PNG is embedded in `26-10-14.md`; PNG/PDF exports and the recovery/plotting scripts live under the corresponding October 14 results/calculation directories.
All nine pre-existing notebook cells are preserved, and four focused tests cover coordinate consistency, labels and tuning parameters, cached rendering, and Markdown embedding.
