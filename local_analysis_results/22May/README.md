# 22May Local Analysis Artifacts

This folder collects local contour outputs that were previously in the repository root.
It is intentionally outside `new_paper_calculations/`, which is kept for cluster-mirrored runs and cluster-facing scripts/results.

## `state_of_art/`

Most recent hydro-analytic/adaptive artifacts to keep using as references:

- `26SU-2-hydro-analytic-adaptive-contour__adaptive_v1.npy`
- `26SU-2-hydro-analytic-velocity-contour-linear_v1.npy`
- `26SU-2-adaptive-hydro-analytic-contour.png`
- `26SU-2-cluster-hydro-analytic-comparison.png`
- `new_EoS_contour.pdf`

## `legacy_contours/`

Older or superseded contour data products, including non-hydro analytic scans, log-mesh outputs, and duplicate cluster/full-grid files.

## `legacy_plots/`

Older exploratory PDFs for asymptotic behavior, isothermal/adiabatic comparisons, pressure plots, and pre-adaptive contour figures.

`26-07-17-3.ipynb` now looks for the current adaptive payload in `local_analysis_results/22May/state_of_art/`, while still falling back to old root-level paths if needed.
