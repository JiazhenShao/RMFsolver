---
summary: All_Plots_new has matched trajectory styles; run the new cached isothermal cell last to export the manuscript PDF.
status: current
updated: 2026-10-15
tags: [methods, notebooks, figures, manuscript]
---

# Trajectory Figure Exports

`All_Plots_new.ipynb` uses solid red numerical profiles, dashed black reference curves, and matching fonts and endpoint markers for its two manuscript trajectories.
The adiabatic plot labels the reference curve "Analytic approximation", the red curve "Numerical solution", and the cold intercept `A_adb`.
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

## Historical Momentum-Flux Profile

The original constant-momentum-flux contour figure was located in `26-04-09.ipynb`, cell 9 (zero based), with its matching vector PDF under `jargon/entropy_tqguess_uagrad_pi_qm_contour.pdf`.
The saved figure has interface temperature 0.10 MeV and downstream temperature 17.34 MeV; the old calculation cell has since been changed to different inputs, so a top-to-bottom rerun does not reproduce the stored output.
No matching raw profile was found.

The final two cells of `26-10-14.ipynb` now record the provenance and replot the original PDF's six vector contours, four sample points, and three endpoints using location labels and larger fonts.
Coordinates and source hash are cached in `local_analysis_results/26-10-14/constant-momentum-profile-data.json`; no raster digitization, new EOS/BVP run, or solver edit was used.
The endpoint change is approximately 0.073 MeV (0.0064%) at figure precision, supporting a nearly constant baryon chemical potential for this historical example, not uniformly across current-model solutions.
The PNG is embedded in `26-10-14.md`; PNG/PDF exports and the recovery/plotting scripts live under the corresponding October 14 results/calculation directories.
All nine pre-existing notebook cells are preserved, and four focused tests cover coordinate consistency, labels and tuning parameters, cached rendering, and Markdown embedding.
