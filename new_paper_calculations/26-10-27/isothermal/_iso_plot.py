"""Contours, gray regions, coverage and reference-style figures (no solver imports).

The linear figure reproduces All_Plots_new.ipynb cell 25 ("Isothermal numerical
u(0^-)"); only the contour-label size is raised to the project's 16-pt rule.
Contours are drawn from the curvilinear (n_B, T) mesh grid of log10 v, masked
wherever a point is not accepted, so no curve is drawn through a failure.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import PchipInterpolator

import _iso_common as common

LINEAR_REFERENCE_STYLE = dict(
    velocity_contour_levels_m_s=(50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0),
    velocity_contour_colors=("#1f77b4", "#17becf", "#ff7f0e", "#8c6d31", "#2ca02c", "#d62728", "#9467bd"),
    velocity_contour_line_style="-",
    velocity_contour_line_width=1.8,
    contour_label_font_size=16,
    contour_label_format="%g",
    contour_label_inline=True,
    contour_label_inline_spacing=4,
    contour_label_temperatures_mev={50.0: 50.0, 100.0: 55.0, 200.0: 60.0, 500.0: 55.0,
                                    1000.0: 45.0, 2000.0: 30.0, 5000.0: 15.0},
    contour_zorder=3,
    excluded_region_color="0.72",
    excluded_region_alpha=1.0,
    excluded_region_zorder=0,
    outer_region_text="$udd$ quark\nmatter favored",
    outer_region_text_position=(3.85, 94.0),
    inner_region_text="Quark matter\nnot favored",
    inner_region_text_position=(1.18, 20.0),
    region_text_color="0.10",
    region_text_font_size=18,
    region_text_zorder=5,
    figure_size=(7.2, 7.2),
    figure_dpi=160,
    export_dpi=220,
    background_color="white",
    constrained_layout=True,
    mathtext_fontset="cm",
    mathtext_roman="serif",
    mathtext_italic="serif:italic",
    mathtext_bold="serif:bold",
    x_limits=(1.0, 5.5),
    y_limits_mev=(0.0, 115.0),
    x_scale="linear",
    y_scale="linear",
    box_aspect=1.0,
    title="Isothermal numerical $u(0^-)$ [$\\mathrm{m\\,s^{-1}}$]",
    title_font_size=20,
    title_pad=8,
    x_label=r"$n_B(0^-)/n_0$",
    y_label=r"$T(0^-)$ [MeV]",
    axis_label_font_size=18,
    tick_label_font_size=18,
    tick_direction="in",
    ticks_top=True,
    ticks_right=True,
    tick_length=4,
    tick_width=0.9,
    spine_line_width=0.8,
)
FONT_RULE_KEYS = ("contour_label_font_size",)   # 11 pt in the reference, >= 16 pt here


def contour_arrays(payload, radial_upsample=1):
    """(x, T, log10 v) on the mesh; v masked wherever the point is not accepted.

    radial_upsample > 1 refines each ray for display: log10 v is interpolated
    monotonically (PCHIP) in u = -ln(1 - f), in which it is nearly linear next
    to the udd boundary (v ~ (1-A)^(-1/2)), only within runs of consecutive
    accepted points, so failures stay holes and nothing is extrapolated.
    """
    mesh = payload["mesh"]
    speed = np.asarray(payload["speed_m_s"], dtype=float)
    accepted = (np.asarray(payload["status"]).astype(str) == "accepted") & np.isfinite(speed) & (speed > 0)
    logv = np.where(accepted, np.log10(np.where(accepted, speed, 1.0)), np.nan)
    if radial_upsample <= 1:
        return (np.asarray(mesh["x"], dtype=float), np.asarray(mesh["T"], dtype=float),
                np.ma.masked_invalid(logv))
    u = -np.log1p(-np.asarray(mesh["fractions"], dtype=float))
    fine_u = np.concatenate([np.linspace(u[k], u[k + 1], radial_upsample + 1)[:-1]
                             for k in range(len(u) - 1)] + [u[-1:]])
    fine_f = -np.expm1(-fine_u)
    inner = np.array([b["rho_inner"] for b in mesh["boundaries"]])[:, None]
    outer = np.array([b["rho_outer"] for b in mesh["boundaries"]])[:, None]
    rho = inner + fine_f[None, :] * (outer - inner)
    x, T = common.to_physical(rho, np.broadcast_to(np.asarray(mesh["theta"])[:, None], rho.shape))
    fine = np.full(rho.shape, np.nan)
    for i, row in enumerate(logv):
        good = np.flatnonzero(np.isfinite(row))
        for run in np.split(good, np.flatnonzero(np.diff(good) > 1) + 1):
            if len(run) < 2:
                continue
            inside = (fine_u >= u[run[0]]) & (fine_u <= u[run[-1]])
            fine[i, inside] = PchipInterpolator(u[run], row[run])(fine_u[inside])
    return np.asarray(x), np.asarray(T), np.ma.masked_invalid(fine)


def coverage(payload, levels):
    """Per level: rays whose adjacent accepted columns bracket it, and why others do not."""
    speed = np.where(np.asarray(payload["status"]).astype(str) == "accepted", payload["speed_m_s"], np.nan)
    report = {}
    for level in levels:
        bracketed, missing = [], {}
        for i, row in enumerate(speed):
            ok = np.flatnonzero(np.isfinite(row))
            if len(ok) == 0:
                missing[i] = "masked"
            elif level < row[ok[0]]:
                missing[i] = "below_inner_column"
            elif level > row[ok[-1]]:
                missing[i] = "above_outer_column"
            elif any(j + 1 < row.size and np.isfinite(row[j + 1]) and
                     min(row[j], row[j + 1]) <= level <= max(row[j], row[j + 1]) for j in ok):
                bracketed.append(i)
            else:
                missing[i] = "masked"
        report[level] = dict(bracketed=bracketed, missing=missing)
    return report


def gray_polygons(curves, x_limits, T_limits):
    """Lower (PNM stable) and upper (udd favored) regions, closed as in the reference cell."""
    x_in, T_in = curves["inner"]
    x_out, T_out = curves["outer"]
    lower = np.column_stack([np.concatenate([x_in, (x_limits[0], x_in[0])]),
                             np.concatenate([T_in, (T_limits[0], T_limits[0])])])
    upper = np.column_stack([np.concatenate([x_out, (x_limits[0], x_limits[1], x_limits[1], x_out[0])]),
                             np.concatenate([T_out, (T_limits[1], T_limits[1], T_limits[0], T_limits[0])])])
    return dict(lower=lower, upper=upper)


def new_axes(style=LINEAR_REFERENCE_STYLE):
    rc = {"mathtext.fontset": style["mathtext_fontset"], "mathtext.rm": style["mathtext_roman"],
          "mathtext.it": style["mathtext_italic"], "mathtext.bf": style["mathtext_bold"]}
    with mpl.rc_context(rc):
        figure, axis = plt.subplots(figsize=style["figure_size"], dpi=style["figure_dpi"],
                                    constrained_layout=style["constrained_layout"],
                                    facecolor=style["background_color"])
    axis.set_facecolor(style["background_color"])
    return figure, axis, rc


def draw_background(axis, curves, style=LINEAR_REFERENCE_STYLE):
    for polygon in gray_polygons(curves, style["x_limits"], style["y_limits_mev"]).values():
        axis.fill(polygon[:, 0], polygon[:, 1], color=style["excluded_region_color"],
                  alpha=style["excluded_region_alpha"], zorder=style["excluded_region_zorder"])
    for text, position in ((style["outer_region_text"], style["outer_region_text_position"]),
                           (style["inner_region_text"], style["inner_region_text_position"])):
        axis.text(*position, text, color=style["region_text_color"],
                  fontsize=style["region_text_font_size"], zorder=style["region_text_zorder"])


def draw_contours(axis, payload, T_range=(-np.inf, np.inf), style=LINEAR_REFERENCE_STYLE,
                  line_style=None, label=True, zorder=None, radial_upsample=1):
    """Contours of one payload, clipped to T_range; inline labels at the style temperatures."""
    levels = np.asarray(style["velocity_contour_levels_m_s"], dtype=float)
    x, T, logv = contour_arrays(payload, radial_upsample)
    contours = axis.contour(x, T, logv, levels=np.log10(levels), colors=style["velocity_contour_colors"],
                            linestyles=line_style or style["velocity_contour_line_style"],
                            linewidths=style["velocity_contour_line_width"],
                            zorder=style["contour_zorder"] if zorder is None else zorder)
    # Finite clip box inside the axes: tight-bbox layout measures it.
    (x0, x1), (y0, y1) = style["x_limits"], style["y_limits_mev"]
    low, high = max(T_range[0], y0), min(T_range[1], y1)
    contours.set_clip_path(Rectangle((x0, low), x1 - x0, high - low, transform=axis.transData))
    if label:
        positions = []
        for level, segments in zip(levels, contours.allsegs):
            target = style["contour_label_temperatures_mev"][float(level)]
            inside = [s for s in segments if len(s) > 1 and s[:, 1].min() <= target <= s[:, 1].max()
                      and low <= target <= high]
            if inside:
                segment = max(inside, key=len)
                order = np.argsort(segment[:, 1])
                positions.append((float(np.interp(target, segment[order, 1], segment[order, 0])), target))
        formats = {np.log10(level): style["contour_label_format"] % level for level in levels}
        axis.clabel(contours, contours.levels, inline=style["contour_label_inline"], fmt=formats,
                    fontsize=style["contour_label_font_size"],
                    inline_spacing=style["contour_label_inline_spacing"], manual=positions)
    return contours


def finish_axes(axis, style=LINEAR_REFERENCE_STYLE, title=None):
    axis.set(xlim=style["x_limits"], ylim=style["y_limits_mev"], xscale=style["x_scale"],
             yscale=style["y_scale"])
    axis.set_box_aspect(style["box_aspect"])
    axis.set_title(title or style["title"], fontsize=style["title_font_size"], pad=style["title_pad"])
    axis.set_xlabel(style["x_label"], fontsize=style["axis_label_font_size"])
    axis.set_ylabel(style["y_label"], fontsize=style["axis_label_font_size"])
    axis.tick_params(direction=style["tick_direction"], top=style["ticks_top"], right=style["ticks_right"],
                     length=style["tick_length"], width=style["tick_width"],
                     labelsize=style["tick_label_font_size"])
    for spine in axis.spines.values():
        spine.set_linewidth(style["spine_line_width"])


def method_legend(axis, entries, font_size=16, location="lower right", color="0.15", line_width=1.8):
    handles = [Line2D([], [], color=color, linestyle=style, linewidth=line_width, label=label)
               for label, style in entries]
    return axis.legend(handles=handles, loc=location, fontsize=font_size, framealpha=0.94,
                       facecolor="white", edgecolor="0.8")


def export(figure, stem, style=LINEAR_REFERENCE_STYLE, formats=("png", "pdf")):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        figure.savefig(stem.with_suffix("." + extension), dpi=style["export_dpi"],
                       facecolor=style["background_color"])
