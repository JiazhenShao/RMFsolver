"""Append reproducible cold contour cells while preserving all existing cells."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[3]
TAG = 'cold-adiabatic-analytical-contours'
LOAD = '''from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "RMFsolver/phase_velocity.py").is_file())
sys.path.insert(0, str(ROOT / "new_paper_calculations/26-10-26/shared"))
from cold_plot import segments, boundary_segments, excluded_polygons, label_point
COLD_PATH = ROOT / "new_paper_calculations/26-10-26/local/adiabatic-T0plus0-xi-minus0p85.npy"
cold = np.load(COLD_PATH, allow_pickle=True).item()
assert cold["complete"], "Resolve coverage/failures before plotting production contours"
assert set(cold["coverage"]) == {"analytic", "semi_analytic"}, "Both method scans are required"
assert all(not c["failures"] and not c["unresolved"] and not c["unresolved_boundaries"] for c in cold["coverage"].values())
assert cold["identity"]["physics"]["T_0plus_MeV"] == 0
assert cold["identity"]["physics"]["xi"] == -.85
assert cold["calibration"] == nonrel["calibration"]
assert cold["method_eos_approximations"]["analytic"] == "quadratic_pressure_second_order_density"
assert all(identity["physics"] == cold["identity"]["physics"] for identity in cold["method_identities"].values())
print("Full analytical: corrected second-order endpoint; semi-analytical: original full-EOS scan.")
for method, coverage in cold["coverage"].items():
    print(method, coverage)
print("Analytical interfaces: exactly 0 MeV; numerical reference: 0.1 MeV.")
print("Differences include the interface temperature as well as analytical approximations.")
'''

PARAMETERS = r'''# Tunable plotting parameters
FIGURE_SIZE = (7.2, 7.2)
FIGURE_DPI = 160
EXPORT_DPI = 220
EXPORT_FORMATS = ("png", "pdf")
BACKGROUND_COLOR = "white"
CONSTRAINED_LAYOUT = True
FONT_FAMILY = "DejaVu Sans"
MATH_FONTSET = "cm"
TITLE_FONT_SIZE = 18
TITLE_PAD = 9
AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 18
CONTOUR_LABEL_FONT_SIZE = 11
CONTOUR_LABEL_FACE_COLOR = "white"
CONTOUR_LABEL_EDGE_COLOR = "none"
CONTOUR_LABEL_BOX_PADDING = .09
CONTOUR_LABEL_ALPHA = .95
LABEL_HORIZONTAL_ALIGNMENT = "center"
LABEL_VERTICAL_ALIGNMENT = "center"
LABEL_ROTATION_MODE = "anchor"
CONTOUR_LABEL_FORMAT = "g"
LINE_WIDTH = 1.8
NUMERICAL_LINE_WIDTH = 1.5
LINE_ALPHA = 1.
CONTOUR_ZORDER = 3
LABEL_ZORDER = 5
LEVELS_M_S = (50., 100., 200., 500., 1000., 2000., 5000.)
LEVEL_COLORS = ("#1f77b4", "#17becf", "#ff7f0e", "#8c6d31", "#2ca02c", "#1f77b4", "#9467bd")
METHOD_STYLES = {"analytic": ":", "semi_analytic": "--"}
METHOD_LABELS = {"analytic": r"Full analytical, $T(0^+)=0$", "semi_analytic": r"Semi-analytical, $T(0^+)=0$"}
LABEL_TEMPERATURES = {"analytic": {50.: 33., 100.: 42., 200.: 51., 500.: 61., 1000.: 72., 2000.: 81., 5000.: 90.},
                      "semi_analytic": {50.: 38., 100.: 44., 200.: 53., 500.: 63., 1000.: 72., 2000.: 81., 5000.: 89.}}
NUMERICAL_LINE_STYLE = "-"
NUMERICAL_LABEL = r"Numerical, $T(0^+)=0.1\,\mathrm{MeV}$"
NUMERICAL_SPLIT_LEVEL_M_S = 5000.
EXCLUDED_REGION_COLOR = "0.72"
EXCLUDED_REGION_ALPHA = 1.
EXCLUDED_REGION_ZORDER = 0
EXCLUDED_REGION_EDGE_WIDTH = 0
BOUNDARY_COLOR = "0.5"
BOUNDARY_LINE_WIDTH = .8
BOUNDARY_ZORDER = 1
REGION_LABEL = "Stable Neutron\nMatter"
REGION_LABEL_POSITION = (1.18, 20.)
REGION_LABEL_FONT_SIZE = 18
REGION_LABEL_COLOR = "0.10"
CEILING_LABEL = r"$a(0^+)\geq 1$"
CEILING_LABEL_POSITION = (4.3, 103.)
X_LIMITS = (1., 5.5)
Y_LIMITS_MEV = (0., 115.)
X_TICKS = (1., 2., 3., 4., 5.)
Y_TICKS_MEV = (0., 20., 40., 60., 80., 100.)
X_SCALE = "linear"
Y_SCALE = "linear"
BOX_ASPECT = 1.
X_LABEL = r"$n_B(0^-)/n_0$"
Y_LABEL = r"$T(0^-)$ [MeV]"
TICK_DIRECTION = "in"
TICKS_TOP = True
TICKS_RIGHT = True
TICK_LENGTH = 4
TICK_WIDTH = .9
SPINE_LINE_WIDTH = .8
GRID_ENABLED = False
LEGEND_FONT_SIZE = 10
LEGEND_LOCATION = "lower left"
LEGEND_FRAME_ALPHA = .95
LEGEND_FACE_COLOR = "white"
LEGEND_EDGE_COLOR = "0.5"
LEGEND_COLOR = "0.15"
LEGEND_HANDLE_LENGTH = 3.
'''

BODY = r'''
with plt.rc_context({"font.family": FONT_FAMILY, "mathtext.fontset": MATH_FONTSET}):
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI,
        constrained_layout=CONSTRAINED_LAYOUT, facecolor=BACKGROUND_COLOR)
    axis.set_facecolor(BACKGROUND_COLOR)
    if SHADE_EXCLUSIONS:
        for polygon in excluded_polygons(cold, PLOT_METHODS[0], X_LIMITS, Y_LIMITS_MEV):
            axis.fill(polygon[:,0], polygon[:,1], color=EXCLUDED_REGION_COLOR,
                alpha=EXCLUDED_REGION_ALPHA, zorder=EXCLUDED_REGION_ZORDER, linewidth=EXCLUDED_REGION_EDGE_WIDTH)
        axis.text(*REGION_LABEL_POSITION, REGION_LABEL, fontsize=REGION_LABEL_FONT_SIZE,
                  color=REGION_LABEL_COLOR, zorder=LABEL_ZORDER)
        if boundary_segments(cold, PLOT_METHODS[0], "composition_ceiling"):
            axis.text(*CEILING_LABEL_POSITION, CEILING_LABEL, fontsize=REGION_LABEL_FONT_SIZE,
                      color=REGION_LABEL_COLOR, zorder=LABEL_ZORDER)
    if SHOW_NUMERICAL:
        for grid, selected in ((NR_GRID, [v for v in LEVELS_M_S if v < NUMERICAL_SPLIT_LEVEL_M_S]),
                               (NR_OUTER_STRIP, [v for v in LEVELS_M_S if v == NUMERICAL_SPLIT_LEVEL_M_S])):
            if selected:
                colors = [LEVEL_COLORS[LEVELS_M_S.index(v)] for v in selected]
                axis.contour(*grid, levels=selected, colors=colors,
                    linestyles=NUMERICAL_LINE_STYLE, linewidths=NUMERICAL_LINE_WIDTH,
                    alpha=LINE_ALPHA, zorder=CONTOUR_ZORDER)
    handles = []
    if SHOW_NUMERICAL:
        handles.append(Line2D([], [], color=LEGEND_COLOR, linestyle=NUMERICAL_LINE_STYLE,
                             linewidth=NUMERICAL_LINE_WIDTH, label=NUMERICAL_LABEL))
    for method in PLOT_METHODS:
        for level,color in zip(LEVELS_M_S, LEVEL_COLORS):
            curves = segments(cold, method, level)
            for curve in curves:
                axis.plot(curve[:,0], curve[:,1], color=color, linestyle=METHOD_STYLES[method],
                    linewidth=LINE_WIDTH, alpha=LINE_ALPHA, zorder=CONTOUR_ZORDER)
            if curves:
                curve = max(curves, key=len)
                position,angle = label_point(curve, LABEL_TEMPERATURES[method][level], X_LIMITS, Y_LIMITS_MEV)
                axis.text(*position, format(level, CONTOUR_LABEL_FORMAT), color=color,
                    fontsize=CONTOUR_LABEL_FONT_SIZE, rotation=angle, rotation_mode=LABEL_ROTATION_MODE,
                    ha=LABEL_HORIZONTAL_ALIGNMENT, va=LABEL_VERTICAL_ALIGNMENT, zorder=LABEL_ZORDER,
                    bbox=dict(facecolor=CONTOUR_LABEL_FACE_COLOR, edgecolor=CONTOUR_LABEL_EDGE_COLOR,
                        alpha=CONTOUR_LABEL_ALPHA, boxstyle=f"square,pad={CONTOUR_LABEL_BOX_PADDING}"))
        for name in ("stable_matter", "composition_ceiling"):
            for curve in boundary_segments(cold, method, name):
                axis.plot(curve[:,0], curve[:,1], color=BOUNDARY_COLOR,
                    linestyle=METHOD_STYLES[method], linewidth=BOUNDARY_LINE_WIDTH, zorder=BOUNDARY_ZORDER)
        handles.append(Line2D([], [], color=LEGEND_COLOR, linestyle=METHOD_STYLES[method],
                             linewidth=LINE_WIDTH, label=METHOD_LABELS[method]))
    axis.legend(handles=handles, loc=LEGEND_LOCATION, fontsize=LEGEND_FONT_SIZE,
        framealpha=LEGEND_FRAME_ALPHA, facecolor=LEGEND_FACE_COLOR, edgecolor=LEGEND_EDGE_COLOR,
        handlelength=LEGEND_HANDLE_LENGTH)
    axis.set(xlim=X_LIMITS, ylim=Y_LIMITS_MEV, xscale=X_SCALE, yscale=Y_SCALE,
             xticks=X_TICKS, yticks=Y_TICKS_MEV)
    axis.set_box_aspect(BOX_ASPECT)
    axis.set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    axis.set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    axis.set_title(TITLE, fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
    axis.tick_params(direction=TICK_DIRECTION, top=TICKS_TOP, right=TICKS_RIGHT,
                    length=TICK_LENGTH, width=TICK_WIDTH, labelsize=TICK_LABEL_FONT_SIZE)
    axis.grid(GRID_ENABLED)
    for spine in axis.spines.values():
        spine.set_linewidth(SPINE_LINE_WIDTH)
    output = ROOT / OUTPUT_STEM
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in EXPORT_FORMATS:
        figure.savefig(output.with_suffix("."+extension), dpi=EXPORT_DPI, facecolor=BACKGROUND_COLOR)
    print("Saved", output, "— levels outside window:",
          {m:cold["coverage"][m]["outside_window_levels"] for m in PLOT_METHODS})
    plt.show()
'''


def main():
    path = ROOT / '26-10-26.ipynb'
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = [c for c in notebook.cells if TAG not in c.metadata.get('tags', [])]
    cells = [nbformat.v4.new_markdown_cell(r'''## Exact-cold adiabatic analytical contours

Both analytical methods use $T(0^+)=0$ exactly, $\xi=-0.85$, massless quarks and QMC-RMF3 PNM with the same cold $3n_0$ calibration. The full analytical branch uses the quadratic EOS, quadratic ceiling, leading diffusion, and closed $I_2$; the semi-analytical branch follows the full EOS with full diffusion and numerical $I_2$.

The physical cold endpoint has $D_K=\infty$ and $D_K^{-1}=0$, while the integral and speed are finite. No temperature floor is used. Shading is calculated independently for each method. Curves consist of validated level crossings; excluded or failed samples are not interpolated through.

The numerical overlay retains the **$T(0^+)=0.1\,\mathrm{MeV}$** dataset. Its differences from these exact-cold curves include the interface temperature, so this is not a comparison at identical interface conditions. The existing numerical figures above are unchanged.'''),
        nbformat.v4.new_code_cell(LOAD)]
    for method, caption, title, stem in (
        ('analytic', 'Full analytical', r'Full analytical $u(0^-)$ [$\mathrm{m\,s^{-1}}$], $T(0^+)=0$', 'adiabatic-full-analytical-cold'),
        ('semi_analytic', 'Semi-analytical', r'Semi-analytical $u(0^-)$ [$\mathrm{m\,s^{-1}}$], $T(0^+)=0$', 'adiabatic-semi-analytical-cold'),
        ('both', 'Analytical methods and numerical reference', r'Adiabatic $u(0^-)$ [$\mathrm{m\,s^{-1}}$]', 'adiabatic-cold-analytical-vs-numerical')):
        methods = ('analytic','semi_analytic') if method=='both' else (method,)
        settings = (f'PLOT_METHODS = {methods!r}\nSHOW_NUMERICAL = {method=="both"}\n'
                    f'SHADE_EXCLUSIONS = {method!="both"}\nTITLE = {title!r}\n'
                    f'OUTPUT_STEM = "local_analysis_results/26-10-26/{stem}"\n')
        parameters = PARAMETERS
        if method == 'both':
            parameters = parameters.replace(
                '"analytic": {50.: 33., 100.: 42., 200.: 51., 500.: 61., 1000.: 72., 2000.: 81., 5000.: 90.}',
                '"analytic": {50.: 25., 100.: 31., 200.: 38., 500.: 47., 1000.: 57., 2000.: 45., 5000.: 60.}')
        cells += [nbformat.v4.new_markdown_cell('### '+caption), nbformat.v4.new_code_cell(parameters+settings+BODY)]
    cells.append(nbformat.v4.new_markdown_cell('''### Reproducing the cold scans

Use `new_paper_calculations/26-10-26/local/run_adiabatic_contours.py` with a working SciPy runtime; the cluster entry point is in the sibling `cluster/` folder. Both use the same engine and frozen coordinate/calibration input. Restart the identical command to resume. The plotting cells read saved results and perform no EOS or front solves.

The README records commands, numerical coverage, and the independent cold-limit verification. Pointwise records retain endpoint quantities, integrals, source hashes and retry history.'''))
    for cell in cells:
        cell.metadata['tags'] = [TAG]
    notebook.cells.extend(cells)
    nbformat.validate(notebook)
    nbformat.write(notebook,path)
    print('Appended',len(cells),'cells')


if __name__=='__main__':
    main()
