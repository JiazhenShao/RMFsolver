"""Portable analytical-isothermal plotting without solver or EOS imports.

Only validated scheduled analytical roots form velocity contours. Saved domain
boundaries provide exclusion-region geometry, never velocity samples.
"""
from pathlib import Path
import json
import os
import tempfile

import numpy as np


LEVELS = (50., 100., 200., 500., 1000., 2000., 5000.)
TEMPERATURES = np.concatenate((np.geomspace(.01, 20., 31), np.geomspace(20., 120., 16)[1:]))
XI = -.85
ARCHIVE_SCHEMA_VERSION = 2
MODEL = "analytical-isothermal"

DEFAULT_STYLE = dict(
    colors=("#1f77b4", "#17becf", "#ff7f0e", "#8c6d31", "#2ca02c", "#d62728", "#9467bd"),
    figure_size=(7.2, 7.2), figure_dpi=160, export_dpi=220,
    line_width=1.8, line_style="-", marker_size=9., contour_zorder=3,
    label_font_size=11, label_format="%g",
    label_temperatures={50.: 50., 100.: 55., 200.: 60., 500.: 55.,
                        1000.: 45., 2000.: 30., 5000.: 15.},
    excluded_color="0.72", excluded_alpha=1., excluded_zorder=0,
    inner_text="Stable Neutron\nMatter", inner_text_position=(1.18, 20.),
    outer_text=r"$a(0^+)\geq 1$", outer_text_position=(4.3, 102.),
    region_text_color="0.10", region_font_size=18, region_text_zorder=5,
    background_color="white", constrained_layout=True,
    mathtext_fontset="cm", mathtext_roman="serif", mathtext_italic="serif:italic",
    mathtext_bold="serif:bold", x_limits=(1., 5.5), y_limits=(.01, 120.),
    box_aspect=1., title=(r"Analytical $u(0^-)$ [$\mathrm{m\,s^{-1}}$]" "\n"
                         r"Isothermal, $\xi=-0.85$"),
    title_font_size=20, title_pad=8, x_label=r"$n_B(0^-)/n_0$",
    y_label=r"$T(0^-)$ [MeV]", axis_label_font_size=18,
    tick_label_font_size=18, tick_direction="in", ticks_top=True,
    ticks_right=True, tick_length=4, tick_width=.9, spine_line_width=.8,
)


def _xy(value):
    """Normalize saved boundary geometry without creating solver samples."""
    array = np.asarray(value, dtype=float).reshape(-1, 2)
    return array[np.all(np.isfinite(array), axis=1)]


def _boundary_snapshot(boundary_geometry):
    """Whitelist boundaries, discarding any velocity curves in legacy inputs."""
    return {key: _xy(boundary_geometry.get(key, [])).copy()
            for key in ("inner_boundary", "outer_boundary")}


def _accepted(row):
    if row is None or row.get("status") != "success":
        return None
    root = row.get("root") or {}
    validated = (root.get("result") or {}).get("validated", False)
    if not isinstance(validated, (bool, np.bool_)) or not validated:
        return None
    try:
        point = np.array([root["nB_0minus_over_n0"], root["T_0minus"]], dtype=float)
        velocity = float(root["velocity_m_s"])
        temperature = float(row["T_0minus"])
    except (KeyError, TypeError, ValueError):
        return None
    if (not np.all(np.isfinite(point)) or not np.isfinite(velocity) or velocity <= 0
            or point[0] <= 0 or not .01 <= temperature <= 120.
            or point[1] != temperature):
        return None
    return point


def _runs(points):
    runs, current = [], []
    for point in points:
        if point is None:
            if current:
                runs.append(np.asarray(current))
                current = []
        else:
            current.append(point)
    if current:
        runs.append(np.asarray(current))
    return runs


def _geometry(boundary_geometry, payload):
    boundaries = _boundary_snapshot(boundary_geometry)
    levels = list(map(float, payload.get("levels", LEVELS)))
    temperatures = sorted(set(map(float, payload.get("temperatures", TEMPERATURES))))
    temperatures = [temperature for temperature in temperatures if .01 <= temperature <= 120.]
    diagnostics = dict(
        schema_version=ARCHIVE_SCHEMA_VERSION, model=MODEL, xi=XI,
        boundary_provenance="Saved domain geometry plus successful fixed-temperature boundaries; shading only.",
        contour_provenance="Validated analytical roots at consecutive scheduled temperatures only.",
        missing_base_targets=0, connectivity={},
        source_inputs=payload.get("identity", {}).get("inputs", {}),
        domain_fingerprint=payload.get("domain_fingerprint"),
        checkpoint_updated_unix_s=payload.get("updated_unix_s"))
    curves, accepted = {}, {}
    for level in levels:
        records = {temperature: [] for temperature in temperatures}
        for row in payload.get("targets", {}).values():
            if row.get("kind") != "base" or float(row["level"]) != level:
                continue
            temperature = float(row["T_0minus"])
            if temperature in records:
                records[temperature].append(row)
        diagnostics["missing_base_targets"] += sum(not rows for rows in records.values())
        points = [_accepted(rows[0]) if len(rows) == 1 else None for rows in records.values()]
        accepted[level] = np.asarray([point for point in points if point is not None]).reshape(-1, 2)
        curves[level] = [run for run in _runs(points) if len(run) >= 2]
        diagnostics["connectivity"][f"{level:g}"] = dict(
            scheduled=len(temperatures), accepted=len(accepted[level]),
            unresolved=len(temperatures) - len(accepted[level]))
    return boundaries, curves, accepted, diagnostics


def _plot_payload(payload):
    """Keep scalar plotting evidence and identity, never profiles or EOS states."""
    result = dict(levels=list(map(float, payload.get("levels", LEVELS))),
                  temperatures=list(map(float, payload.get("temperatures", TEMPERATURES))),
                  targets={}, boundaries={}, identity=payload.get("identity", {}),
                  domain_fingerprint=payload.get("domain_fingerprint"),
                  updated_unix_s=payload.get("updated_unix_s"))
    for key, row in payload.get("targets", {}).items():
        saved = {name: row[name] for name in ("level", "T_0minus", "kind", "status") if name in row}
        root = row.get("root")
        saved["root"] = None
        if root:
            saved["root"] = {name: root[name] for name in
                             ("nB_0minus_over_n0", "T_0minus", "velocity_m_s") if name in root}
            saved["root"]["result"] = {"validated": (root.get("result") or {}).get("validated", False)}
        result["targets"][str(key)] = saved
    for key, row in payload.get("boundaries", {}).items():
        result["boundaries"][str(key)] = {name: row[name] for name in
                                         ("T_0minus", "lower", "upper", "status") if name in row}
    return result


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported plot metadata type: {type(value).__name__}")


def export_plot_archive(boundary_geometry, payload, path):
    """Atomically write a boundary-only portable NPZ with no pickle objects."""
    arrays, _, _, diagnostics = _geometry(boundary_geometry, payload)
    arrays["payload_json"] = np.asarray(json.dumps(_plot_payload(payload), sort_keys=True, default=_json_default))
    arrays["metadata_json"] = np.asarray(json.dumps(diagnostics, sort_keys=True, default=_json_default))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_plot_archive(path):
    """Return (boundary_geometry, payload), rejecting predecessor model archives."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if (metadata.get("schema_version") != ARCHIVE_SCHEMA_VERSION
                or metadata.get("model") != MODEL or metadata.get("xi") != XI):
            raise ValueError("Unsupported analytical isothermal logT plot archive schema/model")
        payload = json.loads(str(archive["payload_json"]))
        boundaries = {key: archive[key].copy() for key in ("inner_boundary", "outer_boundary")}
    return boundaries, payload


def _save_figure_atomic(fig, path, style):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    try:
        fig.savefig(temporary, format=path.suffix.lstrip("."), dpi=style["export_dpi"],
                    facecolor=style["background_color"])
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _shade(ax, boundary_geometry, payload, style):
    for side, field, edge in (("inner", "lower", style["x_limits"][0]),
                              ("outer", "upper", style["x_limits"][1])):
        vertices = boundary_geometry[f"{side}_boundary"]
        if len(vertices) < 2:
            continue
        vertices = vertices[np.argsort(vertices[:, 1], kind="stable")]
        saved_min, saved_max = vertices[0, 1], vertices[-1, 1]
        for row in payload.get("boundaries", {}).values():
            try:
                temperature, density = float(row["T_0minus"]), float(row[field])
            except (KeyError, TypeError, ValueError):
                continue
            if (row.get("status") != "success" or not np.isfinite(density)
                    or not saved_min <= temperature <= saved_max):
                continue
            # Solved boundaries replace coincident display points, but never
            # remove other saved geometry or the upper window intersection.
            vertices = vertices[~np.isclose(vertices[:, 1], temperature, rtol=1e-12, atol=1e-12)]
            vertices = np.vstack((vertices, [density, temperature]))
        vertices = vertices[np.argsort(vertices[:, 1], kind="stable")]
        ymin, ymax = style["y_limits"]
        temperatures = np.unique(np.concatenate(([ymin, ymax], vertices[:, 1])))
        temperatures = temperatures[(temperatures >= ymin) & (temperatures <= ymax)]
        densities = np.interp(temperatures, vertices[:, 1], vertices[:, 0])
        densities = np.clip(densities, *style["x_limits"])
        # The outer boundary ends at x=1, T~94.674 MeV. Extending that endpoint
        # to ymax covers the entire excluded upper region, not a wedge.
        polygon = np.vstack((np.column_stack((densities, temperatures)), [edge, ymax], [edge, ymin]))
        patch = ax.fill(polygon[:, 0], polygon[:, 1], color=style["excluded_color"],
                        alpha=style["excluded_alpha"], zorder=style["excluded_zorder"])[0]
        patch.set_gid(f"excluded:{side}")


def plot_contours(boundary_geometry, payload, output_dir=None, stem="isothermal-analytic-logT", **style):
    """Plot analytical roots and attach saved-data diagnostics to the figure."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    unknown = set(style) - set(DEFAULT_STYLE)
    if unknown:
        raise TypeError(f"Unknown plotting style keys: {sorted(unknown)}")
    style = dict(DEFAULT_STYLE, **style)
    if not 0 < style["y_limits"][0] < style["y_limits"][1]:
        raise ValueError("The logarithmic temperature axis requires increasing positive limits")
    boundaries, curves, accepted, diagnostics = _geometry(boundary_geometry, payload)
    rc = {"mathtext.fontset": style["mathtext_fontset"], "mathtext.rm": style["mathtext_roman"],
          "mathtext.it": style["mathtext_italic"], "mathtext.bf": style["mathtext_bold"]}
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=style["figure_size"], dpi=style["figure_dpi"],
                               constrained_layout=style["constrained_layout"],
                               facecolor=style["background_color"])
        ax.set_facecolor(style["background_color"])
        _shade(ax, boundaries, payload, style)
        for i, (level, runs) in enumerate(curves.items()):
            color_index = LEVELS.index(level) if level in LEVELS else i
            color = style["colors"][color_index % len(style["colors"])]
            for index, run in enumerate(runs):
                line, = ax.plot(run[:, 0], run[:, 1], color=color, linewidth=style["line_width"],
                                linestyle=style["line_style"], zorder=style["contour_zorder"])
                line.set_gid(f"contour:{level:g}:{index}")
            roots = accepted[level]
            if len(roots):
                collection = ax.scatter(roots[:, 0], roots[:, 1], s=style["marker_size"], color=color,
                                        zorder=style["contour_zorder"])
                collection.set_gid(f"analytical-roots:{level:g}")
            label_temperature = style["label_temperatures"].get(level)
            if label_temperature is not None:
                candidate = next((run for run in runs
                                  if run[0, 1] <= label_temperature <= run[-1, 1]), None)
                if candidate is not None:
                    x = np.interp(np.log(label_temperature), np.log(candidate[:, 1]), candidate[:, 0])
                    ax.text(x, label_temperature, style["label_format"] % level, color=color,
                            fontsize=style["label_font_size"], ha="center", va="center",
                            backgroundcolor=style["background_color"], zorder=style["region_text_zorder"])
        for side in ("inner", "outer"):
            ax.text(*style[f"{side}_text_position"], style[f"{side}_text"],
                    color=style["region_text_color"], fontsize=style["region_font_size"],
                    va="top" if side == "outer" else "baseline", zorder=style["region_text_zorder"])
        ax.set(xlim=style["x_limits"], ylim=style["y_limits"], xscale="linear", yscale="log")
        ax.set_box_aspect(style["box_aspect"])
        ax.set_xlabel(style["x_label"], fontsize=style["axis_label_font_size"])
        ax.set_ylabel(style["y_label"], fontsize=style["axis_label_font_size"])
        ax.set_title(style["title"], fontsize=style["title_font_size"], pad=style["title_pad"])
        ax.tick_params(which="both", direction=style["tick_direction"], top=style["ticks_top"],
                       right=style["ticks_right"], length=style["tick_length"],
                       width=style["tick_width"], labelsize=style["tick_label_font_size"])
        for spine in ax.spines.values():
            spine.set_linewidth(style["spine_line_width"])
        fig.logT_diagnostics = diagnostics
        if output_dir is not None:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            for extension in ("png", "pdf"):
                _save_figure_atomic(fig, directory / f"{stem}.{extension}", style)
    return fig
