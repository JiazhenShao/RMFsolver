"""Figures of the isothermal xi report (local_analysis_results/26-11-01/isothermal-xi-report); no solver calls.

1. linear T: numerical versus analytical at the high-T contour fit (the low-T fit below 10 MeV);
2. log T, 0.01-10 MeV: numerical versus analytical at the low-T contour fit, as Fig. isothermal-contour-compare;
3. the strip fits of xi_strips.json versus T, all plotted levels together and each level alone.
Fonts, axis names and the legend above the axes follow the manuscript contour figures (update_all_plots_new.py).

    PYTHONPATH=local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:. \
    ./bin/python3 -B new_paper_calculations/26-10-27/isothermal/xi_fit/report_figures.py
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "highT")]

import update_all_plots_new as generator  # noqa: E402  (manuscript styles)
import matplotlib as mpl  # noqa: E402

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

import _iso_common as common  # noqa: E402
import _iso_plot as plot  # noqa: E402
import _iso_xi as iso_xi  # noqa: E402
import build_boundary_curves  # noqa: E402

OUT = generator.ROOT / "local_analysis_results/26-11-01/isothermal-xi-report"

# Tunable plotting parameters
SEAM_MEV = 10.0                  # polar mesh above, row mesh below
UPSAMPLE = 8                     # display-only PCHIP refinement along rays/rows
LINEAR_FIGURE_SIZE = (7.2, 8.35)  # the manuscript linear panel plus a two-line legend row above the axes
LINEAR_TITLE_PAD = 84
LEGEND_HANDLE_COLOR = "0.15"
STRIP_FIGURE_SIZE = (7.2, 10.4)
STRIP_X_LIMITS_MEV = (0.006, 120.0)   # tick labels clear of the corner
STRIP_X_TICKS = (0.01, 0.1, 1.0, 10.0, 100.0)
STRIP_X_TICK_LABELS = ("0.01", "0.1", "1", "10", "100")
STRIP_Y_LIMITS = ((-1.1, 3.2), (-1.1, 4.0))
STRIP_COLOR = "0.10"
STRIP_MARKER = "o"
STRIP_MARKER_SIZE = 9
STRIP_LINE_WIDTH = 1.6
PANEL_FIT_COLOR = "#c62828"
PANEL_FIT_LINE_STYLE = "--"
PANEL_FIT_LINE_WIDTH = 2.4
REFERENCE_XI_COLOR = "0.45"
REFERENCE_XI_LINE_STYLE = ":"
SHAPE_LIMIT_FILL_COLOR = "0.90"    # xi > 1: needs the flux g = -D_K a' to exceed its interface value G
SHAPE_LIMIT_EDGE_COLOR = "0.55"    # legend patch only, so it shows on the band
SEAM_LINE_COLOR = "0.70"
LEVEL_COLOR_MAP = LinearSegmentedColormap.from_list("level_blues", plt.get_cmap("Blues")(np.linspace(0.35, 1.0, 256)))
LEVEL_MARKER_SIZE = 6
LEVEL_LINE_WIDTH = 1.8
COLORBAR_BOUNDS = (0.06, 0.80, 0.44, 0.055)   # inset in the per-level panel, axes fraction
FONTS = dict(text=generator.MANUSCRIPT_FONTS["tick_label_font_size"],
             label=generator.MANUSCRIPT_FONTS["axis_label_font_size"])
EXPORT_DPI = 220


def legend_above(axis, entries, style):
    """One legend row between the title and the axes, as in Fig. isothermal-contour-compare."""
    legend = generator.LOG_COMPARISON_LEGEND
    handles = [Line2D([], [], color=LEGEND_HANDLE_COLOR, linestyle=line_style,
                      linewidth=style["velocity_contour_line_width"], label=label) for label, line_style in entries]
    axis.legend(handles=handles, loc=legend["legend_location"], bbox_to_anchor=legend["legend_anchor"],
                ncols=len(entries), frameon=legend["legend_frame"], fontsize=style["region_text_font_size"],
                handlelength=legend["legend_handle_length"], columnspacing=legend["legend_column_spacing"],
                handletextpad=legend["legend_handle_text_pad"], borderaxespad=legend["legend_border_axes_pad"])


def linear_figure(payloads, caches, boundaries, xi_high, xi_low):
    style = generator.manuscript_style(plot.LINEAR_REFERENCE_STYLE)
    style.update(figure_size=LINEAR_FIGURE_SIZE, title_pad=LINEAR_TITLE_PAD,
                 title=plot.LOG_REFERENCE_STYLE["title"])
    levels, colors = style["velocity_contour_levels_m_s"], style["velocity_contour_colors"]
    analytic_low = dict(payloads["analytic_lowT"], speed_m_s=iso_xi.retune(caches["lowT"], xi_low))
    figure, axis, rc = plot.new_axes(style)
    with mpl.rc_context(rc):
        plot.draw_background(axis, boundaries, style)
        plot.draw_contours(axis, payloads["numerical_highT"], T_range=(SEAM_MEV, np.inf), style=style,
                           radial_upsample=UPSAMPLE)
        plot.draw_contours(axis, dict(payloads["analytic_highT"], speed_m_s=iso_xi.retune(caches["highT"], xi_high)),
                           T_range=(SEAM_MEV, np.inf), style=style, line_style="--", label=False,
                           radial_upsample=UPSAMPLE)
        for payload, line_style in ((payloads["numerical_lowT"], "-"), (analytic_low, "--")):
            plot.draw_row_contours(axis, payload, levels, colors, line_style, style["velocity_contour_line_width"],
                                   style["contour_zorder"], T_range=(style["y_limits_mev"][0], SEAM_MEV),
                                   x_limits=style["x_limits"], radial_upsample=UPSAMPLE)
        plot.finish_axes(axis, style)
        legend_above(axis, [(f"Analytical, $\\xi={xi_high:+g}$\n(${xi_low:+g}$ below 10 MeV)", "--"),
                            ("Numerical", "-")], style)
    return figure


def log_figure(payloads, caches, boundaries, xi_low):
    style = {**generator.manuscript_style(plot.LOG_REFERENCE_STYLE), **generator.LOG_PANEL_SHAPE,
             **generator.LOG_COMPARISON_LEGEND}
    lines = {}
    for name, payload in (("numerical", payloads["numerical_lowT"]),
                          ("analytic", dict(payloads["analytic_lowT"], speed_m_s=iso_xi.retune(caches["lowT"], xi_low)))):
        lines[name] = plot.contour_lines(*plot.contour_arrays(payload, UPSAMPLE, "logit"), style["contour_levels"],
                                         log_T=True)
    with plt.rc_context({"font.family": style["font_family"], "mathtext.fontset": style["math_font_set"],
                         "text.usetex": style["use_tex"]}):
        figure, axis = plt.subplots(figsize=style["figure_size"], dpi=style["figure_dpi"],
                                    constrained_layout=style["constrained_layout"], facecolor=style["background_color"])
        plot.draw_log_reference(axis, boundaries, lines["numerical"], lines["analytic"], xi_low, style)
    return figure


def strip_figure(strips):
    panel_xi = {tag: round(value, 2) for tag, value in strips["panel_xi"].items()}
    panel_T = {tag: (fits[0]["T_range"][0], fits[-1]["T_range"][1]) for tag, fits in strips["strips"].items()}
    with plt.rc_context({"mathtext.fontset": "cm", "font.size": FONTS["text"]}):
        figure, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=STRIP_FIGURE_SIZE, dpi=160,
                                             constrained_layout=True)
        for axis, limits in zip((top, bottom), STRIP_Y_LIMITS):
            axis.axhspan(1.0, limits[1], color=SHAPE_LIMIT_FILL_COLOR, zorder=0, linewidth=0)
            axis.axvline(SEAM_MEV, color=SEAM_LINE_COLOR, linewidth=1.0, zorder=1)
            axis.set_ylim(*limits)
        fits = [fit for tag in ("lowT", "highT") for fit in strips["strips"][tag]]
        T = np.array([fit["T_median"] for fit in fits])
        xi = np.array([fit["xi"] for fit in fits])
        top.errorbar(T, xi, xerr=[T - [f["T_range"][0] for f in fits], [f["T_range"][1] for f in fits] - T],
                     yerr=[xi - [f["xi_16"] for f in fits], [f["xi_84"] for f in fits] - xi], fmt=STRIP_MARKER,
                     color=STRIP_COLOR, markersize=STRIP_MARKER_SIZE, linewidth=STRIP_LINE_WIDTH, capsize=0,
                     zorder=4, label="Strip fit")
        for tag in ("lowT", "highT"):
            top.plot(panel_T[tag], [panel_xi[tag]] * 2, color=PANEL_FIT_COLOR, linestyle=PANEL_FIT_LINE_STYLE,
                     linewidth=PANEL_FIT_LINE_WIDTH, zorder=3, label="Panel fit" if tag == "lowT" else None)
        top.axhline(common.XI, color=REFERENCE_XI_COLOR, linestyle=REFERENCE_XI_LINE_STYLE,
                    linewidth=PANEL_FIT_LINE_WIDTH, zorder=2, label=f"$\\xi={common.XI:g}$")
        handles, labels = top.get_legend_handles_labels()
        handles.append(Patch(facecolor=SHAPE_LIMIT_FILL_COLOR, edgecolor=SHAPE_LIMIT_EDGE_COLOR))
        labels.append(r"$\xi>1$: needs $g>G$")
        top.legend(handles, labels, loc="upper left", fontsize=FONTS["text"], frameon=False, handlelength=1.6,
                   borderaxespad=0.3)
        top.set_ylabel(r"Best-fit $\xi$, all levels", fontsize=FONTS["label"])

        levels = sorted({float(level) for tag in strips["levels"] for level in strips["levels"][tag]})
        norm = LogNorm(levels[0], levels[-1])
        for tag in ("lowT", "highT"):
            for level in strips["levels"][tag]:
                points = [(fit["per_level"][f"{level:g}"]["T_median"], fit["per_level"][f"{level:g}"]["xi"])
                          for fit in strips["strips"][tag] if f"{level:g}" in fit["per_level"]]
                if points:
                    bottom.plot(*np.array(points).T, marker="o", markersize=LEVEL_MARKER_SIZE,
                                linewidth=LEVEL_LINE_WIDTH, color=LEVEL_COLOR_MAP(norm(level)), zorder=3)
        bottom.set_ylabel(r"Best-fit $\xi$, per level", fontsize=FONTS["label"])
        bottom.set_xscale("log")
        bottom.set_xlim(*STRIP_X_LIMITS_MEV)
        bottom.set_xticks(STRIP_X_TICKS, labels=STRIP_X_TICK_LABELS)
        bottom.set_xlabel("Nuclear phase temperature [MeV]", fontsize=FONTS["label"])
        inset = bottom.inset_axes(COLORBAR_BOUNDS)
        colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=LEVEL_COLOR_MAP), cax=inset,
                                   orientation="horizontal")
        colorbar.set_ticks([1e2, 1e4, 1e6], labels=["$10^2$", "$10^4$", "$10^6$"])
        colorbar.ax.tick_params(labelsize=FONTS["text"])
        colorbar.set_label(r"Speed level [$\mathrm{m\,s^{-1}}$]", fontsize=FONTS["text"], labelpad=4)
        colorbar.ax.xaxis.set_label_position("top")
        for axis in (top, bottom):
            axis.tick_params(which="both", direction="in", top=True, right=True, labelsize=FONTS["text"])
            axis.minorticks_off()
    return figure


def main():
    payloads = {f"{method}_{panel}": common.load(HERE.parent / f"{panel}/isothermal-{panel}-{method}.npy")
                for method in ("numerical", "analytic") for panel in ("highT", "lowT")}
    caches = {panel: iso_xi.build_cache(payloads[f"analytic_{panel}"]) for panel in ("highT", "lowT")}
    boundaries = build_boundary_curves.load(payloads["numerical_highT"]["mesh"]["boundaries"])
    strips = json.loads((HERE / "xi_strips.json").read_text())
    xi_high, xi_low = (round(strips["panel_xi"][tag], 2) for tag in ("highT", "lowT"))
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, figure in (("isothermal-highT-linear-xi", linear_figure(payloads, caches, boundaries, xi_high, xi_low)),
                         ("isothermal-lowT-logT-xi", log_figure(payloads, caches, boundaries, xi_low)),
                         ("isothermal-xi-strips", strip_figure(strips))):
        for extension in ("png", "pdf"):
            figure.savefig(OUT / f"{stem}.{extension}", dpi=EXPORT_DPI, facecolor="white")
        print(f"Saved {OUT / stem}.png/.pdf")
        plt.close(figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
