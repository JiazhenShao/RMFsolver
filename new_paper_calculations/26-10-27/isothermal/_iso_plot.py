"""Contours, gray regions, coverage and reference-style figures (no solver imports).

The linear figure reproduces All_Plots_new.ipynb cell 25 ("Isothermal numerical
u(0^-)"); only the contour-label size is raised to the project's 16-pt rule.
Contours are drawn from the curvilinear (n_B, T) mesh grid of log10 v, masked
wherever a point is not accepted, so no curve is drawn through a failure.
"""
from pathlib import Path

import contourpy
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
    title="Isothermal numerical speed [$\\mathrm{m\\,s^{-1}}$]",
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

# 26-10-02.ipynb cell 8 ("Isothermal u(0^-)", log T); overrides are listed below.
LOG_REFERENCE_STYLE = dict(
    contour_levels=(1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6),
    contour_colors=('#1f77b4', '#17becf', '#ff7f0e', '#8c6d31', '#2ca02c', '#e377c2', '#d62728',
                    '#008b8b', '#9467bd'),
    figure_size=(7.2, 7.2), figure_dpi=160, export_dpi=220, constrained_layout=True,
    background_color='white', font_family='DejaVu Sans', math_font_set='cm', use_tex=False,
    box_aspect=1., x_limits=(1., 5.8), y_limits=(.01, 10.), x_scale='linear', y_scale='log',
    x_ticks=(1., 2., 3., 4., 5.), y_ticks=(.01, .1, 1., 10.), y_tick_labels=('0.01', '0.1', '1', '10'),
    title=r'Isothermal speed [$\mathrm{m\,s^{-1}}$]', title_font_size=20, title_pad=8,
    x_label=r'$n_B(0^-)/n_0$', y_label=r'$T(0^-)$ [MeV]', axis_label_font_size=18, tick_label_font_size=18,
    tick_direction='in', ticks_top=True, ticks_right=True, tick_length=4, tick_width=.9,
    spine_line_width=.8, analytical_line_style='--', numerical_line_style='-', line_width=1.8,
    analytical_zorder=4, numerical_zorder=3, numerical_split_temperature=20., numerical_marker='o',
    marker_size=19., marker_edge_color='white', marker_edge_width=.35, marker_alpha=1., marker_zorder=5,
    isolated_analytical_marker='x', isolated_analytical_size=15., label_curves=True,
    contour_label_font_size=16, contour_label_format='%g', contour_label_inline=True,
    contour_label_inline_spacing=4, contour_label_background_color='white',
    contour_label_background_alpha=.92, contour_label_background_pad=.12,
    contour_label_background_style='round', contour_label_edge_color='none', contour_label_zorder=6,
    label_temperatures={1e2: 5., 3e2: 2., 1e3: 1., 3e3: .5, 1e4: .2, 3e4: .1, 1e5: .04, 3e5: .02,
                        1e6: .016},
    excluded_region_color='0.72', excluded_region_alpha=1., excluded_region_zorder=0,
    inner_region_text='Quark matter\nnot favored', inner_region_position=(1.12, 0.3),
    inner_region_rotation=0, inner_region_alignment='left', inner_region_vertical_alignment='baseline',
    outer_region_text='$udd$ quark matter favored', outer_region_position=(5.55, 0.316),
    outer_region_rotation=90, outer_region_alignment='center', outer_region_vertical_alignment='center',
    region_text_color='0.12', region_text_font_size=18, region_text_zorder=6,
    legend_location='upper left', legend_font_size=16, legend_handle_length=1.0,
    legend_border_axes_pad=0.25, legend_frame_alpha=.94, legend_face_color='white',
    legend_edge_color='0.8', legend_handle_color='0.15', legend_marker_size=5., show_method_legend=True,
    show_speed_legend=False, speed_legend_location='lower left', speed_legend_title=r'$u(0^-)$ [m s$^{-1}$]',
    speed_legend_font_size=16, speed_legend_columns=2, speed_legend_anchor=(.005, .005),
    speed_legend_handle_length=1.3, speed_legend_column_spacing=0.7, speed_legend_handle_text_pad=0.4,
    speed_legend_label_spacing=0.3,
    speed_labels=('100', '300', '1k', '3k', '10k', '30k', '100k', '300k', '1M'),
    show_grid=False, export_stem='isothermal-lowT-logT',
)
# Changed from the reference: the upper T limit is 10 MeV (user), so the y
# ticks/labels follow and the a(0+)>=1 label moves into the narrow udd strip;
# legend and label fonts follow the >= 16 pt rule, with compact legend spacing and
# the method legend moved to the upper left so both stay inside the gray region;
# cold numerics now converge,
# so numerical results are solid lines everywhere (no T < 20 MeV dots).
# 2026-10-28 (user): half-decade levels labelled on the contours instead of a speed
# legend, the gray regions named as in the linear figure, the x range widened
# to 5.8 n_0 so the udd strip holds its label, and 'speed' in the title.
LOG_STYLE_OVERRIDES = ("y_limits", "y_ticks", "y_tick_labels", "outer_region_position",
                       "inner_region_position", "legend_location", "speed_legend_anchor",
                       "contour_label_font_size", "legend_font_size", "speed_legend_font_size",
                       "export_stem", "contour_levels", "contour_colors", "label_curves",
                       "label_temperatures", "inner_region_text", "outer_region_text",
                       "show_speed_legend", "speed_labels", "x_limits", "title")


def draw_log_reference(axis, curves, numerical_lines, analytic_lines, xi, style=LOG_REFERENCE_STYLE):
    """Log-T contours in the 26-10-02 cell 8 style: numerical solid, analytic dashed (none if
    analytic_lines is empty), an optional method legend, and inline speed labels (label_curves)
    and/or a speed legend."""
    for polygon in gray_polygons(curves, style["x_limits"], style["y_limits"]).values():
        axis.fill(polygon[:, 0], polygon[:, 1], color=style["excluded_region_color"],
                  alpha=style["excluded_region_alpha"], zorder=style["excluded_region_zorder"])
    colors = dict(zip(style["contour_levels"], style["contour_colors"]))
    for lines, line_style, zorder in ((analytic_lines, style["analytical_line_style"], style["analytical_zorder"]),
                                      (numerical_lines, style["numerical_line_style"], style["numerical_zorder"])):
        draw_lines(axis, lines, [colors[level] for level in lines], line_style, style["line_width"], zorder)
    axis.set(xlim=style["x_limits"], ylim=style["y_limits"], xscale=style["x_scale"], yscale=style["y_scale"])
    axis.set_xticks(style["x_ticks"])
    axis.set_yticks(style["y_ticks"], labels=style["y_tick_labels"])
    axis.set_box_aspect(style["box_aspect"])
    axis.text(*style["inner_region_position"], style["inner_region_text"], color=style["region_text_color"],
              fontsize=style["region_text_font_size"], zorder=style["region_text_zorder"],
              rotation=style.get("inner_region_rotation", 0), ha=style.get("inner_region_alignment", "left"),
              va=style.get("inner_region_vertical_alignment", "baseline"))
    axis.text(*style["outer_region_position"], style["outer_region_text"], color=style["region_text_color"],
              fontsize=style["region_text_font_size"], zorder=style["region_text_zorder"],
              rotation=style["outer_region_rotation"], va=style.get("outer_region_vertical_alignment", "bottom"),
              ha=style.get("outer_region_alignment", "left"))
    axis.set_xlabel(style["x_label"], fontsize=style["axis_label_font_size"])
    axis.set_ylabel(style["y_label"], fontsize=style["axis_label_font_size"])
    axis.set_title(style["title"], fontsize=style["title_font_size"], pad=style["title_pad"])
    axis.tick_params(direction=style["tick_direction"], top=style["ticks_top"], right=style["ticks_right"],
                     length=style["tick_length"], width=style["tick_width"],
                     labelsize=style["tick_label_font_size"])
    for spine in axis.spines.values():
        spine.set_linewidth(style["spine_line_width"])
    axis.grid(style["show_grid"])
    if style["label_curves"]:
        _label_log_contours(axis, numerical_lines, analytic_lines, colors, style)
    method = None
    if style.get("show_method_legend", True):
        method = axis.legend(handles=[
            Line2D([], [], color=style["legend_handle_color"], linestyle=style["analytical_line_style"],
                   linewidth=style["line_width"], label=rf"Analytical, $\xi={xi:g}$"),
            Line2D([], [], color=style["legend_handle_color"], linestyle=style["numerical_line_style"],
                   linewidth=style["line_width"], label="Numerical")],
            loc=style["legend_location"], fontsize=style["legend_font_size"],
            framealpha=style["legend_frame_alpha"], handlelength=style["legend_handle_length"],
            borderaxespad=style["legend_border_axes_pad"], facecolor=style["legend_face_color"],
            edgecolor=style["legend_edge_color"])
    if style["show_speed_legend"]:
        if method is not None:
            axis.add_artist(method)
        axis.legend(handles=[Line2D([], [], color=color, linestyle=style["analytical_line_style"],
                                    linewidth=style["line_width"], label=label)
                             for label, color in zip(style["speed_labels"], style["contour_colors"])],
                    loc=style["speed_legend_location"], title=style["speed_legend_title"],
                    ncols=style["speed_legend_columns"], bbox_to_anchor=style["speed_legend_anchor"],
                    fontsize=style["speed_legend_font_size"], title_fontsize=style["speed_legend_font_size"],
                    handlelength=style["speed_legend_handle_length"],
                    columnspacing=style["speed_legend_column_spacing"],
                    handletextpad=style["speed_legend_handle_text_pad"],
                    labelspacing=style["speed_legend_label_spacing"],
                    framealpha=style["legend_frame_alpha"], facecolor=style["legend_face_color"],
                    edgecolor=style["legend_edge_color"])


def _label_log_contours(axis, numerical_lines, analytic_lines, colors, style):
    """One label per level at its label temperature, on the numerical line (the analytic one if
    no numerical segment reaches it), rotated along the line; the white box hides the line beneath."""
    (x0, x1), (y0, y1) = style["x_limits"], np.log10(style["y_limits"])
    for level, label in zip(style["contour_levels"], style["speed_labels"]):
        target = np.log10(style["label_temperatures"][level])
        for lines in (numerical_lines, analytic_lines):
            covering = [s for s in lines.get(float(level), [])
                        if np.log10(s[:, 1].min()) <= target - 0.02 and target + 0.02 <= np.log10(s[:, 1].max())]
            if covering:
                break
        else:
            continue
        segment = max(covering, key=len)
        order = np.argsort(segment[:, 1])
        x_at = lambda logT: np.interp(logT, np.log10(segment[order, 1]), segment[order, 0])
        slope = (x_at(target + 0.02) - x_at(target - 0.02)) / 0.04          # dx / dlog10 T
        angle = np.degrees(np.arctan2(style["box_aspect"] / (y1 - y0), slope / (x1 - x0)))   # display units
        axis.text(float(x_at(target)), 10 ** target, label, rotation=angle - 180 if angle > 90 else angle,
                  rotation_mode="anchor", ha="center", va="center", color=colors[level],
                  fontsize=style["contour_label_font_size"], zorder=style["contour_label_zorder"],
                  bbox=dict(boxstyle=f"{style['contour_label_background_style']},"
                                     f"pad={style['contour_label_background_pad']}",
                            facecolor=style["contour_label_background_color"],
                            alpha=style["contour_label_background_alpha"],
                            edgecolor=style["contour_label_edge_color"]))


RADIAL_VARIABLES = {   # u(f) and its inverse for display refinement along a ray or row
    "log1m": (lambda f: -np.log1p(-f), lambda u: -np.expm1(-u)),          # udd end: v ~ (1-f)^(-1/2)
    "logit": (lambda f: np.log(f / (1 - f)), lambda u: 1 / (1 + np.exp(-u))),   # both ends
}


def contour_arrays(payload, radial_upsample=1, radial_variable="log1m"):
    """(x, T, log10 v) on the mesh; v masked wherever the point is not accepted.

    radial_upsample > 1 refines each ray (polar mesh) or row (row mesh) for
    display: log10 v is interpolated monotonically (PCHIP) in u(f), in which it
    is nearly linear at the boundaries, only within runs of consecutive accepted
    points, so failures stay holes and nothing is extrapolated.
    """
    mesh = payload["mesh"]
    speed = np.asarray(payload["speed_m_s"], dtype=float)
    accepted = (np.asarray(payload["status"]).astype(str) == "accepted") & np.isfinite(speed) & (speed > 0)
    logv = np.where(accepted, np.log10(np.where(accepted, speed, 1.0)), np.nan)
    if radial_upsample <= 1:
        return (np.asarray(mesh["x"], dtype=float), np.asarray(mesh["T"], dtype=float),
                np.ma.masked_invalid(logv))
    forward, inverse = RADIAL_VARIABLES[radial_variable]
    u = forward(np.asarray(mesh["fractions"], dtype=float))
    fine_u = np.concatenate([np.linspace(u[k], u[k + 1], radial_upsample + 1)[:-1]
                             for k in range(len(u) - 1)] + [u[-1:]])
    fine_f = inverse(fine_u)
    if mesh.get("kind") == "rows":
        inner, outer = mesh["x_inner"][:, None], mesh["x_outer"][:, None]
        x = inner + fine_f[None, :] * (outer - inner)
        T = np.broadcast_to(mesh["T_rows"][:, None], x.shape)
    else:
        inner = np.array([b["rho_inner"] for b in mesh["boundaries"]])[:, None]
        outer = np.array([b["rho_outer"] for b in mesh["boundaries"]])[:, None]
        rho = inner + fine_f[None, :] * (outer - inner)
        x, T = common.to_physical(rho, np.broadcast_to(np.asarray(mesh["theta"])[:, None], rho.shape))
    fine = np.full(np.shape(x), np.nan)
    for i, row in enumerate(logv):
        good = np.flatnonzero(np.isfinite(row))
        for run in np.split(good, np.flatnonzero(np.diff(good) > 1) + 1):
            if len(run) < 2:
                continue
            inside = (fine_u >= u[run[0]]) & (fine_u <= u[run[-1]])
            fine[i, inside] = PchipInterpolator(u[run], row[run])(fine_u[inside])
    return np.asarray(x), np.asarray(T), np.ma.masked_invalid(fine)


def contour_lines(x, T, logv, levels, log_T=False):
    """{level: [N x 2 (x, T) arrays]} from the masked grid; log_T contours in log10 T."""
    y = np.log10(T) if log_T else np.asarray(T, dtype=float)
    generator = contourpy.contour_generator(np.asarray(x, dtype=float), y, np.ma.masked_invalid(logv),
                                            line_type=contourpy.LineType.Separate)
    lines = {}
    for level in levels:
        segments = [s for s in generator.lines(np.log10(level)) if len(s) > 1]
        lines[float(level)] = [np.column_stack([s[:, 0], 10 ** s[:, 1]]) if log_T else s
                               for s in segments]
    return lines


def draw_lines(axis, lines, colors, line_style, line_width, zorder, T_range=None, x_limits=None):
    """Plot contour_lines output, clipped to T_range inside the axes."""
    for (level, segments), color in zip(lines.items(), colors):
        for segment in segments:
            (artist,) = axis.plot(segment[:, 0], segment[:, 1], color=color, linestyle=line_style,
                                  linewidth=line_width, zorder=zorder)
            if T_range is not None:
                x0, x1 = x_limits
                artist.set_clip_path(Rectangle((x0, T_range[0]), x1 - x0, T_range[1] - T_range[0],
                                               transform=axis.transData))


def draw_row_contours(axis, payload, levels, colors, line_style, line_width, zorder, T_range, x_limits,
                      radial_upsample=8):
    """Row-mesh contours computed in log10 T (logit-f refinement), clipped to T_range."""
    x, T, logv = contour_arrays(payload, radial_upsample, "logit")
    draw_lines(axis, contour_lines(x, T, logv, levels, log_T=True), colors, line_style, line_width,
               zorder, T_range, x_limits)


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
