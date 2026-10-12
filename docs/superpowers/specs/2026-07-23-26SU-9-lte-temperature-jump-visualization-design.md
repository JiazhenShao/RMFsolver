# 26SU-9 LTE Temperature-Jump Visualization Design

## Status

Approved in conversation on 2026-07-23.

## Goal

Create `26SU-9.ipynb` to inspect and visualize the downloaded cluster payload at `new_paper_calculations/26-07-23/lte-temperature-jump-two-block-linear.npy`.

The primary field is the signed interface-temperature bound

`delta_T_interface = T_N - T_Qstar_LTE`.

## Notebook structure

The notebook will contain four focused sections:

1. A short Markdown introduction defining the plotted quantity and the bound interpretation.
2. Imports, path resolution, and payload loading.
3. Diagnostics and assertions.
4. One publication-style contour figure.

## Diagnostics

The diagnostics section will display:

- payload identity and closure metadata;
- low- and high-block shapes;
- success and failure counts;
- finite value counts and ranges;
- maximum mismatch across the shared 80 MeV seam;
- maximum residual in `delta_T_interface = T_N - T_Qstar_LTE`;
- maximum residual in `beta_LTE = 5*lambda_n*u_N`.

Assertions will require:

- matching grid shapes within each block;
- all scanned statuses to indicate success;
- exact low/high seam agreement within floating-point tolerance;
- finite primary fields at every scanned point;
- the saved temperature and beta identities to hold within floating-point tolerance.

## Contour construction

The contour will be drawn directly on the two native adaptive blocks.
No rectangular interpolation or triangulation will be used.
The low and high blocks will share a common contour-level and color mapping, and the duplicated first high-temperature row will be omitted from the second contour call to avoid drawing the 80 MeV seam twice.

The plot will match the supplied reference style:

- nearly square aspect;
- pale lavender safe-domain background;
- individually colored, labeled contour lines;
- gray forbidden regions;
- inward ticks on all four sides;
- compact scientific title and axis labels;
- direct annotations inside the forbidden regions;
- no unnecessary legend or filled colorbar.

The zero contour will be emphasized because the negative range is much smaller than the positive range.
The default contour levels will be `-0.04, 0, 20, 40, 60, 80, 90 MeV`, with separate adjustable colors for every level.

## Boundary regions

The independently saved boundaries will define the gray masks:

- shade above the `A_boundary = 1` curve and label it `A_boundary > 1`;
- shade the low-metastability side of the `Delta h = 0` curve and label it `Delta h > 0`.

The boundary curves themselves will be drawn in neutral gray so the colored temperature contours remain the dominant marks.

## Plot controls

The visualization code cell will begin with a clearly labeled `# Tunable plotting parameters` block.
Every adjustable plotting value used by the cell will be defined there as an uppercase constant, including figure size, DPI, fonts, line widths, colors, contour levels, label formats, axis limits, ticks, background colors, annotation positions, and export settings.

## Output

The notebook will display the figure inline and save a publication-ready PDF and PNG under `local_analysis_results/26SU-9/`.
The output directory will be created by the notebook.

## Verification

The notebook will be executed noninteractively.
Verification requires:

- execution without errors;
- persisted diagnostic outputs;
- persisted inline figure output;
- nonempty PDF and PNG exports;
- visual inspection of the rendered notebook or exported PNG for clipping, incorrect masks, duplicated seams, and unreadable contour labels.
