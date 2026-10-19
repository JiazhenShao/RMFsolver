# Tunable plotting parameters
DATA_FILE = "local_analysis_results/26-10-16/aiso-accuracy/results.json"
OUTPUT_PNG = "local_analysis_results/26-10-14/aiso-accuracy.png"
OUTPUT_PDF = "local_analysis_results/26-10-14/aiso-accuracy.pdf"
MEETING_PNG = "26-10-14.assets/aiso-accuracy.png"
FIGURE_SIZE = (10.4, 12.7)
FIGURE_DPI = 150
EXPORT_DPI = 200
TITLE_FONT_SIZE = 22
PANEL_TITLE_FONT_SIZE = 19
AXIS_LABEL_FONT_SIZE = 21
TICK_LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 17
FONT_FAMILY = "DejaVu Sans"
MATH_FONT_SET = "dejavusans"
AXIS_LINE_WIDTH = 1.1
AXIS_SCALE = "linear"
X_LIMITS = (0., 1.025)
X_TICKS = (0., 0.2, 0.4, 0.6, 0.8, 1.)
Y_LIMITS = ((-0.025, 1.12), (-0.004, 0.094), (-0.4, 9.4))
Y_TICKS = ((0., 0.2, 0.4, 0.6, 0.8, 1.), (0., 0.02, 0.04, 0.06, 0.08), (0., 1., 2., 4., 6., 8.))
X_AXIS_LABEL = r"Exact $A_{\rm iso}$"
Y_AXIS_LABELS = (r"$A_{\rm iso}$", r"$A_{\rm iso}^{\rm Eq.\,(35)}-A_{\rm iso}^{\rm exact}$", "Relative difference")
PANEL_TITLES = ("(a) Composition ceiling", "(b) Signed difference from exact", "(c) Percentage difference from exact")
TITLE_PAD = 10
TICK_DIRECTION = "in"
TOP_TICKS = True
RIGHT_TICKS = True
TICK_LENGTH = 5
GRID_ALPHA = 0.18
GRID_LINE_WIDTH = 0.8
BASELINE_COLOR = "#202020"
BASELINE_LINE_WIDTH = 2.1
BASELINE_LINE_STYLE = "-"
BASELINE_ZORDER = 2
COLD_COLOR = "#0072B2"
COLD_LINE_WIDTH = 2.3
COLD_LINE_STYLE = "--"
COLD_ZORDER = 4
COLD_A_MIN = 0.001
COLD_A_MAX = 1.
COLD_SAMPLE_COUNT = 601
POINT_COLORMAP = "plasma"
POINT_MARKER = "o"
POINT_SIZE = 28
POINT_ALPHA = 0.85
POINT_EDGE_COLOR = "none"
POINT_ZORDER = 3
TEMPERATURE_LIMITS = (0., 85.)
COLORBAR_TICKS = (0., 20., 40., 60., 80.)
COLORBAR_WIDTH_RATIO = 0.04
COLORBAR_LABEL_PAD = 14
PERCENT_TICK_DECIMALS = 0
GUIDE_PERCENT = 1.
GUIDE_COLOR = "#aa3535"
GUIDE_LINE_WIDTH = 1.4
GUIDE_LINE_STYLE = ":"
GUIDE_LABEL_POSITION = (0.025, 1.2)
ANNOTATION_ARROW_STYLE = "->"
ANNOTATION_LINE_WIDTH = 1.1
COLD_CROSSING_MARKER = "o"
COLD_CROSSING_MARKER_SIZE = 55
COLD_CROSSING_EDGE_WIDTH = 1.4
COLD_CROSSING_FACE_COLOR = "white"
COLD_CROSSING_LABEL_POSITION = (0.16, 3.1)
MAXIMUM_LABEL_POSITION = (0.48, 6.6)
COUNT_LABEL_POSITION = (0.035, 0.95)
LEGEND_LOCATION = "upper left"
LEGEND_FRAME_ALPHA = 0.95
LEGEND_HANDLE_LENGTH = 2.3
LEGEND_POINT_COLOR = "#a85d8e"
LEGEND_MARKER_SIZE = 6
LAYOUT_MODE = "constrained"
LAYOUT_VERTICAL_PAD = 0.12
EXPORT_BOUNDING_BOX = "tight"
EXPORT_TRANSPARENT = False
SHOW_FIGURE = True

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

audit = json.loads(Path(DATA_FILE).read_text())
summary, rows = audit["summary"], audit["rows"]
a_exact = np.array([row["A_exact"] for row in rows])
a_eq35 = np.array([row["A_taylor"] for row in rows])
temperature = np.array([row["T_MeV"] for row in rows])
difference = a_eq35 - a_exact
percent_difference = 100 * difference / a_exact
assert np.all(a_exact > 0) and np.all(a_exact <= 1)
assert len(rows) == summary["count"]
assert np.allclose(percent_difference, [row["relative_error_percent"] for row in rows], atol=1e-11, rtol=0)
assert np.count_nonzero(np.abs(percent_difference) > 1) == summary["count_above_one_percent"]
assert np.min(temperature) >= TEMPERATURE_LIMITS[0] and np.max(temperature) <= TEMPERATURE_LIMITS[1]

cold_a = np.linspace(COLD_A_MIN, COLD_A_MAX, COLD_SAMPLE_COUNT)
# Exact cold isobar/Gibbs relation, substituted into the Taylor ceiling.
cold_eq35 = 3 * np.sqrt(((1 + (1 + cold_a)**(4/3) + (1 - cold_a)**(4/3)) / 3)**(3/4) - 1)
cold_difference = cold_eq35 - cold_a
cold_percent = 100 * cold_difference / cold_a
cold_one_percent_a = 0.49554417610308704  # Verified in the audit's verification.wl.
maximum_index = int(np.argmax(percent_difference))

style = {"font.family": FONT_FAMILY, "mathtext.fontset": MATH_FONT_SET,
         "axes.linewidth": AXIS_LINE_WIDTH}
with plt.rc_context(style):
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, layout=LAYOUT_MODE)
    fig.get_layout_engine().set(h_pad=LAYOUT_VERTICAL_PAD)
    grid = fig.add_gridspec(3, 2, width_ratios=(1, COLORBAR_WIDTH_RATIO))
    axes = [fig.add_subplot(grid[0, 0])]
    axes.extend(fig.add_subplot(grid[i, 0], sharex=axes[0]) for i in (1, 2))
    color_axis = fig.add_subplot(grid[:, 1])
    normalization = Normalize(*TEMPERATURE_LIMITS)
    exact_values = (cold_a, np.zeros_like(cold_a), np.zeros_like(cold_a))
    cold_values = (cold_eq35, cold_difference, cold_percent)
    saved_values = (a_eq35, difference, percent_difference)
    for index, axis in enumerate(axes):
        axis.plot(cold_a, exact_values[index], color=BASELINE_COLOR,
                  linewidth=BASELINE_LINE_WIDTH, linestyle=BASELINE_LINE_STYLE,
                  zorder=BASELINE_ZORDER, label="Exact neutral EOS")
        points = axis.scatter(a_exact, saved_values[index], c=temperature,
                              cmap=POINT_COLORMAP, norm=normalization,
                              s=POINT_SIZE, marker=POINT_MARKER, alpha=POINT_ALPHA,
                              edgecolors=POINT_EDGE_COLOR, zorder=POINT_ZORDER)
        axis.plot(cold_a, cold_values[index], color=COLD_COLOR,
                  linewidth=COLD_LINE_WIDTH, linestyle=COLD_LINE_STYLE,
                  zorder=COLD_ZORDER, label=r"Eq. (35), $T=0$")
        axis.set_title(PANEL_TITLES[index], fontsize=PANEL_TITLE_FONT_SIZE, pad=TITLE_PAD)
        axis.set_ylabel(Y_AXIS_LABELS[index], fontsize=AXIS_LABEL_FONT_SIZE)
        axis.set_xlim(X_LIMITS)
        axis.set_ylim(Y_LIMITS[index])
        axis.set_xscale(AXIS_SCALE)
        axis.set_yscale(AXIS_SCALE)
        axis.set_xticks(X_TICKS)
        axis.set_yticks(Y_TICKS[index])
        axis.tick_params(which="both", direction=TICK_DIRECTION, top=TOP_TICKS,
                         right=RIGHT_TICKS, length=TICK_LENGTH,
                         labelsize=TICK_LABEL_FONT_SIZE, labelbottom=index == 2)
        axis.grid(alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH)
    axes[-1].set_xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    axes[-1].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=PERCENT_TICK_DECIMALS))
    axes[-1].axhline(GUIDE_PERCENT, color=GUIDE_COLOR, linewidth=GUIDE_LINE_WIDTH,
                    linestyle=GUIDE_LINE_STYLE, zorder=BASELINE_ZORDER)
    axes[-1].text(*GUIDE_LABEL_POSITION, f"{GUIDE_PERCENT:g}% guide", color=GUIDE_COLOR,
                 fontsize=ANNOTATION_FONT_SIZE)
    axes[-1].scatter(cold_one_percent_a, 1., marker=COLD_CROSSING_MARKER,
                    s=COLD_CROSSING_MARKER_SIZE, facecolor=COLD_CROSSING_FACE_COLOR, edgecolor=COLD_COLOR,
                    linewidths=COLD_CROSSING_EDGE_WIDTH, zorder=COLD_ZORDER + 1)
    axes[-1].annotate(r"$T=0$: 1% at $A=0.496$", (cold_one_percent_a, 1.),
                     xytext=COLD_CROSSING_LABEL_POSITION, fontsize=ANNOTATION_FONT_SIZE,
                     color=COLD_COLOR, arrowprops={"arrowstyle": ANNOTATION_ARROW_STYLE,
                                                  "color": COLD_COLOR, "lw": ANNOTATION_LINE_WIDTH})
    axes[-1].annotate(f"Largest saved error: {percent_difference[maximum_index]:.2f}%",
                     (a_exact[maximum_index], percent_difference[maximum_index]),
                     xytext=MAXIMUM_LABEL_POSITION, fontsize=ANNOTATION_FONT_SIZE,
                     ha="center", arrowprops={"arrowstyle": ANNOTATION_ARROW_STYLE,
                                               "color": BASELINE_COLOR, "lw": ANNOTATION_LINE_WIDTH})
    axes[-1].text(*COUNT_LABEL_POSITION,
                 f"{summary['count_above_one_percent']} / {len(rows)} points exceed 1%",
                 transform=axes[-1].transAxes, va="top", fontsize=ANNOTATION_FONT_SIZE)
    handles = [
        Line2D([], [], color=BASELINE_COLOR, linewidth=BASELINE_LINE_WIDTH,
               linestyle=BASELINE_LINE_STYLE, label="Exact neutral EOS"),
        Line2D([], [], color=COLD_COLOR, linewidth=COLD_LINE_WIDTH,
               linestyle=COLD_LINE_STYLE, label=r"Eq. (35), $T=0$"),
        Line2D([], [], color=LEGEND_POINT_COLOR, marker=POINT_MARKER, linestyle="none",
               markersize=LEGEND_MARKER_SIZE, label="Eq. (35), saved states"),
    ]
    axes[0].legend(handles=handles, loc=LEGEND_LOCATION, fontsize=LEGEND_FONT_SIZE,
                   framealpha=LEGEND_FRAME_ALPHA, handlelength=LEGEND_HANDLE_LENGTH)
    colorbar = fig.colorbar(points, cax=color_axis, ticks=COLORBAR_TICKS)
    colorbar.set_label(r"Saved-state temperature $T$ [MeV]", fontsize=AXIS_LABEL_FONT_SIZE,
                       labelpad=COLORBAR_LABEL_PAD)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE, length=TICK_LENGTH)
    fig.suptitle(r"Accuracy of $A_{\rm iso}$ from Eq. (35)", fontsize=TITLE_FONT_SIZE)
    for filename in (OUTPUT_PNG, OUTPUT_PDF):
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=EXPORT_DPI, bbox_inches=EXPORT_BOUNDING_BOX,
                    transparent=EXPORT_TRANSPARENT)
    Path(MEETING_PNG).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_PNG, MEETING_PNG)
    if SHOW_FIGURE:
        plt.show()

print(f"Exact-EOS baseline; {len(rows)} audited states, no new EOS/BVP solves.")
print(f"Relative error = 100*(A_Eq35/A_exact - 1); {summary['count_above_one_percent']} points exceed 1%.")
print(f"Largest saved error: {percent_difference[maximum_index]:.5f}%; "
      f"A_exact={a_exact[maximum_index]:.8f}, A_Eq35={a_eq35[maximum_index]:.8f}.")
print(f"Exports: {OUTPUT_PNG}, {OUTPUT_PDF}, {MEETING_PNG}")
