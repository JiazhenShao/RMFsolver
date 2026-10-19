# Tunable plotting parameters
DATA_FILE = "local_analysis_results/26-10-14/constant-momentum-profile-data.json"
OUTPUT_PNG = "local_analysis_results/26-10-14/constant-momentum-profile.png"
OUTPUT_PDF = "local_analysis_results/26-10-14/constant-momentum-profile.pdf"
MEETING_PNG = "26-10-14.assets/constant-momentum-profile.png"
FIGURE_SIZE = (9.4, 7.4)
FIGURE_DPI = 160
EXPORT_DPI = 220
TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 21
TICK_LABEL_FONT_SIZE = 17
LEGEND_FONT_SIZE = 15
ANNOTATION_FONT_SIZE = 17
FONT_FAMILY = "DejaVu Sans"
MATH_FONT_SET = "dejavusans"
X_LIMITS = (1106., 1226.)
Y_LIMITS = (-40., 120.)
X_TICKS = (1120., 1140., 1160., 1180., 1200., 1220.)
Y_TICKS = (-40., -20., 0., 20., 40., 60., 80., 100., 120.)
TICK_DIRECTION = "in"
TOP_TICKS = True
RIGHT_TICKS = True
TICK_LENGTH = 5
GRID_ALPHA = 0.2
GRID_LINE_WIDTH = 0.8
AXIS_LINE_WIDTH = 1.1
REFERENCE_COLOR = "#555555"
REFERENCE_LINE_WIDTH = 1.1
REFERENCE_LINE_STYLE = "-"
COLOR_0PLUS = "#1f77b4"
COLOR_INF = "#2ca02c"
COLOR_0MINUS = "#e6ad00"
ENDPOINT_LINE_WIDTH = 2.5
LINE_STYLE_0PLUS = "-"
LINE_STYLE_INF = "--"
SAMPLE_COLORS = ("#472a7a", "#2c728e", "#28ae80", "#b0dd2f")
SLICE_LINE_WIDTH = 1.4
SLICE_LINE_STYLE = "-"
SLICE_ALPHA = 0.65
PROFILE_COLOR = "#202020"
PROFILE_LINE_WIDTH = 1.8
PROFILE_LINE_STYLE = "--"
PROFILE_ALPHA = 0.85
POINT_EDGE_COLOR = "black"
POINT_EDGE_LINE_WIDTH = 0.8
SAMPLE_MARKER = "o"
SAMPLE_MARKER_SIZE = 58
ENDPOINT_MARKER = "o"
ENDPOINT_MARKER_SIZE = 86
UPSTREAM_MARKER = "*"
UPSTREAM_MARKER_SIZE = 150
LINE_ZORDER = 2
PROFILE_ZORDER = 3
SAMPLE_ZORDER = 4
ENDPOINT_ZORDER = 5
OFFSET_0PLUS = (-16, 18)
OFFSET_INF = (10, -22)
OFFSET_0MINUS = (10, -22)
SAMPLE_LABEL_OFFSETS = ((17, -5), (17, 0), (17, 0), (17, 0))
ANNOTATION_BOX_ALPHA = 0.9
ANNOTATION_BOX_COLOR = "white"
ANNOTATION_BOX_PAD = 0.08
LEGEND_LOCATION = "upper right"
LEGEND_FRAME_ALPHA = 0.94
LEGEND_HANDLE_LENGTH = 2.3
LEGEND_BORDER_PAD = 0.6
TITLE_PAD = 13
LAYOUT_PAD = 1.1
EXPORT_BOUNDING_BOX = "tight"
EXPORT_TRANSPARENT = False
X_AXIS_LABEL = r"$\mu_B\ [\mathrm{MeV}]$"
Y_AXIS_LABEL = r"$\mu_K\ [\mathrm{MeV}]$"
LABEL_0PLUS = r"$x=0^+$"
LABEL_INF = r"$x=\infty$"
LABEL_0MINUS = r"$x=0^-$"

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

data = json.loads(Path(DATA_FILE).read_text())
style = {"font.family": FONT_FAMILY, "mathtext.fontset": MATH_FONT_SET,
         "axes.linewidth": AXIS_LINE_WIDTH}
with plt.rc_context(style):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    ax.axhline(0., color=REFERENCE_COLOR, linewidth=REFERENCE_LINE_WIDTH,
               linestyle=REFERENCE_LINE_STYLE)
    for key, color, line_style in (("contour_0plus", COLOR_0PLUS, LINE_STYLE_0PLUS),
                                    ("contour_inf", COLOR_INF, LINE_STYLE_INF)):
        points = np.asarray(data[key])
        ax.plot(points[:, 0], points[:, 1], color=color,
                linewidth=ENDPOINT_LINE_WIDTH, linestyle=line_style, zorder=LINE_ZORDER)
    for contour, color in zip(data["contours_sample"], SAMPLE_COLORS):
        points = np.asarray(contour)
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=SLICE_LINE_WIDTH,
                linestyle=SLICE_LINE_STYLE, alpha=SLICE_ALPHA, zorder=LINE_ZORDER)
    path = np.asarray(data["profile_path"])
    ax.plot(path[:, 0], path[:, 1], color=PROFILE_COLOR, linewidth=PROFILE_LINE_WIDTH,
            linestyle=PROFILE_LINE_STYLE, alpha=PROFILE_ALPHA, zorder=PROFILE_ZORDER)
    for point, x_nm, color, offset in zip(data["sample_points"], data["sample_x_nm"],
                                          SAMPLE_COLORS, SAMPLE_LABEL_OFFSETS):
        ax.scatter(*point, s=SAMPLE_MARKER_SIZE, marker=SAMPLE_MARKER, color=color,
                   edgecolor=POINT_EDGE_COLOR, linewidths=POINT_EDGE_LINE_WIDTH,
                   zorder=SAMPLE_ZORDER)
        ax.annotate(rf"$x={x_nm:.0f}\,\mathrm{{nm}}$", point, xytext=offset,
                    textcoords="offset points", fontsize=ANNOTATION_FONT_SIZE,
                    va="center", bbox={"facecolor": ANNOTATION_BOX_COLOR, "edgecolor": "none",
                                       "alpha": ANNOTATION_BOX_ALPHA, "pad": ANNOTATION_BOX_PAD})
    for key, color, marker, size, label, offset in (
        ("point_0plus", COLOR_0PLUS, ENDPOINT_MARKER, ENDPOINT_MARKER_SIZE, LABEL_0PLUS, OFFSET_0PLUS),
        ("point_inf", COLOR_INF, ENDPOINT_MARKER, ENDPOINT_MARKER_SIZE, LABEL_INF, OFFSET_INF),
        ("point_0minus", COLOR_0MINUS, UPSTREAM_MARKER, UPSTREAM_MARKER_SIZE, LABEL_0MINUS, OFFSET_0MINUS),
    ):
        point = data[key]
        ax.scatter(*point, s=size, marker=marker, color=color, edgecolor=POINT_EDGE_COLOR,
                   linewidths=POINT_EDGE_LINE_WIDTH, zorder=ENDPOINT_ZORDER)
        ax.annotate(label, point, xytext=offset, textcoords="offset points",
                    fontsize=ANNOTATION_FONT_SIZE)

    handles = [
        Line2D([], [], color=COLOR_0PLUS, linewidth=ENDPOINT_LINE_WIDTH, linestyle=LINE_STYLE_0PLUS,
               label=rf"$T(0^+)={data['T_0plus_MeV']:.2f}\,\mathrm{{MeV}}$"),
        Line2D([], [], color=COLOR_INF, linewidth=ENDPOINT_LINE_WIDTH, linestyle=LINE_STYLE_INF,
               label=rf"$T(\infty)={data['T_inf_MeV']:.2f}\,\mathrm{{MeV}}$"),
        Line2D([], [], color=PROFILE_COLOR, linewidth=PROFILE_LINE_WIDTH,
               linestyle=PROFILE_LINE_STYLE, label="Sampled numerical profile"),
    ]
    ax.legend(handles=handles, loc=LEGEND_LOCATION, fontsize=LEGEND_FONT_SIZE,
              framealpha=LEGEND_FRAME_ALPHA, handlelength=LEGEND_HANDLE_LENGTH,
              borderpad=LEGEND_BORDER_PAD)
    ax.set_title("Constant momentum flux along the profile\n"
                 r"$\Pi_{\rm QM}=\Pi(0^-),\quad n_B(0^-)=3n_0$",
                 fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
    ax.set_xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlim(X_LIMITS)
    ax.set_ylim(Y_LIMITS)
    ax.set_xticks(X_TICKS)
    ax.set_yticks(Y_TICKS)
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.tick_params(which="both", direction=TICK_DIRECTION, top=TOP_TICKS,
                   right=RIGHT_TICKS, length=TICK_LENGTH, labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(alpha=GRID_ALPHA, linewidth=GRID_LINE_WIDTH)
    fig.tight_layout(pad=LAYOUT_PAD)
    for filename in (OUTPUT_PNG, OUTPUT_PDF):
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=EXPORT_DPI, bbox_inches=EXPORT_BOUNDING_BOX,
                    transparent=EXPORT_TRANSPARENT)
    Path(MEETING_PNG).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_PNG, MEETING_PNG)
    plt.show()

muB_0plus = data["point_0plus"][0]
muB_inf = data["point_inf"][0]
print("Original source: 26-04-09.ipynb, cell 9 (zero based).")
print("Replotted archived PDF vector coordinates; no new EOS/BVP calculation.")
print(f"Endpoint mu_B: {muB_0plus:.2f} -> {muB_inf:.2f} MeV")
print(f"Endpoint change: {muB_inf-muB_0plus:.3f} MeV ({100*(muB_inf/muB_0plus-1):.4f}%)")
print("This is one historical profile, not a uniform error bound on constant mu_B.")
print(f"Saved: {OUTPUT_PNG}, {OUTPUT_PDF}, {MEETING_PNG}")
