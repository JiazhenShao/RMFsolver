"""Portable saved-data plotting; importing this module never imports the solver.

Warm curves are archival smoothed display geometry, not solver samples.  Cold
geometry uses only consecutive successful scheduled targets, including inserted
refinements.  The old zero-temperature anchors are never retained.
"""
from pathlib import Path
import json
import os
import tempfile

import numpy as np


LEVELS = (50., 100., 200., 500., 1000., 2000., 5000.)
TEMPERATURES = np.geomspace(.01, 20., 31)
JOIN_TEMPERATURE = 20.
JOIN_DENSITY_TOLERANCE = .01
WARM_PROVENANCE = (
    "Archival smoothed display curves retained strictly above 20 MeV; their "
    "vertices are not numerical solver points. curve_solved flags describe the "
    "old pre-resampling knots and are not a mask for the smoothed vertices. "
    "Only separately archived roots are identified as accepted warm roots. "
    "An unjoined warm curve includes its interpolated 20-MeV display endpoint, "
    "which is not a newly solved root."
)

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
    box_aspect=1., title=r"Numerical $u(0^-)$ [$\mathrm{m\,s^{-1}}$], isothermal",
    title_font_size=20, title_pad=8, x_label=r"$n_B(0^-)/n_0$",
    y_label=r"$T(0^-)$ [MeV]", axis_label_font_size=18,
    tick_label_font_size=18, tick_direction="in", ticks_top=True,
    ticks_right=True, tick_length=4, tick_width=.9, spine_line_width=.8,
    diagnostic_font_size=9, diagnostic_color="#a00000", diagnostic_position=(.03, .025),
)


def _xy(value):
    """Normalize saved geometry without creating interpolated solver samples."""
    array = np.asarray(value, dtype=float).reshape(-1, 2)
    return array[np.all(np.isfinite(array), axis=1)]


def _seam_density(vertices):
    if len(vertices) < 2:
        return None
    order = np.argsort(vertices[:, 1], kind="stable")
    vertices = vertices[order]
    if not vertices[0, 1] <= JOIN_TEMPERATURE <= vertices[-1, 1]:
        return None
    return float(np.interp(JOIN_TEMPERATURE, vertices[:, 1], vertices[:, 0]))


def _warm_snapshot(warm):
    """Discard all legacy cold geometry but retain seam comparison metadata."""
    levels = np.asarray(warm.get("levels_m_s", LEVELS), dtype=float)
    result = {"levels_m_s": levels.copy()}
    for key in ("inner_boundary", "outer_boundary"):
        vertices = _xy(warm.get(key, []))
        result[key] = vertices[vertices[:, 1] > JOIN_TEMPERATURE].copy()
        seam = warm.get(key + "_seam_density", _seam_density(vertices))
        if seam is not None:
            result[key + "_seam_density"] = np.asarray(seam, dtype=float)
    for i, _ in enumerate(levels):
        vertices = _xy(warm.get(f"curve_vertices_{i}", []))
        result[f"curve_vertices_{i}"] = vertices[vertices[:, 1] > JOIN_TEMPERATURE].copy()
        seam = warm.get(f"seam_density_{i}", _seam_density(vertices))
        if seam is not None:
            result[f"seam_density_{i}"] = np.asarray(seam, dtype=float)
        # These old knot flags cannot be indexed against resampled vertices.
        result[f"curve_solved_{i}"] = np.asarray(warm.get(f"curve_solved_{i}", []), dtype=bool)
        roots = _xy(warm.get(f"roots_{i}", []))
        result[f"roots_{i}"] = roots[roots[:, 1] > JOIN_TEMPERATURE].copy()
    return result


def _accepted(row):
    if row is None or row.get("status") != "success":
        return None
    root = row.get("root") or {}
    try:
        point = np.array([root["nB_0minus_over_n0"], root["T_0minus"]], dtype=float)
        velocity = float(root["velocity_m_s"])
        T = float(row["T_0minus"])
    except (KeyError, TypeError, ValueError):
        return None
    if (not np.all(np.isfinite(point)) or not np.isfinite(velocity) or velocity <= 0
            or point[0] <= 0 or not .01 <= T <= JOIN_TEMPERATURE
            or point[1] != T):
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


def _geometry(warm, payload):
    warm = _warm_snapshot(warm)
    levels = list(map(float, payload.get("levels", warm["levels_m_s"])))
    temperatures = sorted(set(map(float, payload.get("temperatures", TEMPERATURES))))
    diagnostics = dict(warm_provenance=WARM_PROVENANCE, warm_vertices_are_solver_points=False,
                       join_tolerance_n0=JOIN_DENSITY_TOLERANCE, joins={},
                       missing_base_targets=0, connectivity={}, schema_version=1,
                       source_inputs=payload.get("identity", {}).get("inputs", {}),
                       domain_fingerprint=payload.get("domain_fingerprint"),
                       checkpoint_updated_unix_s=payload.get("updated_unix_s"))
    diagnostics['accuracy_checks'] = {
        key: {field: row[field] for field in ('status','relative_speed_change','message') if field in row}
        for key,row in payload.get('validation',{}).items()}
    curves, accepted = {}, {}
    for level in levels:
        records = {}
        for row in payload.get("targets", {}).values():
            if row.get("kind") not in ("base", "refinement") or float(row["level"]) != level:
                continue
            T = float(row["T_0minus"])
            if .01 <= T <= JOIN_TEMPERATURE:
                records.setdefault(T, []).append(row)
        diagnostics["missing_base_targets"] += sum(
            not any(row.get("kind") == "base" for row in records.get(T, [])) for T in temperatures)
        schedule = sorted(set(temperatures) | set(records))
        points = [_accepted(records[T][0]) if len(records.get(T, [])) == 1 else None for T in schedule]
        runs = _runs(points)
        accepted[level] = np.asarray([p for p in points if p is not None]).reshape(-1, 2)
        key = f"{level:g}"
        diagnostics["connectivity"][key] = dict(scheduled=len(schedule), accepted=len(accepted[level]),
                                                 unresolved=len(schedule) - len(accepted[level]))
        warm_indices = np.flatnonzero(warm["levels_m_s"] == level)
        warm_curve = np.empty((0, 2))
        join = {"status": "no_warm_curve"}
        if len(warm_indices):
            index = int(warm_indices[0])
            warm_curve = warm[f"curve_vertices_{index}"]
            seam = warm.get(f"seam_density_{index}")
            cold_seam = next((p for T, p in zip(schedule, points) if T == JOIN_TEMPERATURE), None)
            if len(warm_curve):
                join = {"status": "missing_cold_root"}
                if cold_seam is not None:
                    join = {"status": "missing_warm_seam"}
                    if seam is not None and np.isfinite(seam):
                        difference = abs(float(seam) - float(cold_seam[0]))
                        join = dict(status="joined" if difference <= JOIN_DENSITY_TOLERANCE else "inconsistent",
                                    warm_interpolated_density=float(seam),
                                    cold_solved_density=float(cold_seam[0]), difference_n0=difference)
                if join["status"] == "joined":
                    # The final cold run already contains exactly one newly solved 20-MeV root.
                    runs[-1] = np.vstack((runs[-1], warm_curve))
                else:
                    if seam is not None and np.isfinite(seam):
                        # Retain the entire archival warm interval without claiming a
                        # numerical seam root or bridging to an inconsistent cold run.
                        warm_curve = np.vstack(([float(seam), JOIN_TEMPERATURE], warm_curve))
                        join["warm_seam_is_display_interpolation"] = True
                    runs.append(warm_curve)
        diagnostics["joins"][key] = join
        curves[level] = [run for run in runs if len(run) >= 2]
    return warm, curves, accepted, diagnostics


def _plot_payload(payload):
    """Keep scalar plotting evidence; never serialize profiles or pickle objects."""
    result = dict(levels=list(map(float, payload.get("levels", LEVELS))),
                  temperatures=list(map(float, payload.get("temperatures", TEMPERATURES))),
                  targets={}, boundaries={},
                  identity={"inputs": payload.get("identity", {}).get("inputs", {})},
                  domain_fingerprint=payload.get("domain_fingerprint"),
                  updated_unix_s=payload.get("updated_unix_s"))
    result['validation'] = {
        key: {field: row[field] for field in ('status','relative_speed_change','message') if field in row}
        for key,row in payload.get('validation',{}).items()}
    for key, row in payload.get("targets", {}).items():
        result["targets"][str(key)] = {name: row[name] for name in
                                      ("level", "T_0minus", "kind", "status") if name in row}
        root = row.get("root")
        if root:
            result["targets"][str(key)]["root"] = {name: float(root[name]) for name in
                ("nB_0minus_over_n0", "T_0minus", "velocity_m_s") if name in root}
    for key, row in payload.get("boundaries", {}).items():
        result["boundaries"][str(key)] = {name: row[name] for name in
                                         ("T_0minus", "lower", "upper", "status") if name in row}
    return result


def export_plot_archive(warm, payload, path):
    """Atomically write a portable NPZ readable with allow_pickle=False."""
    saved_warm, _, _, diagnostics = _geometry(warm, payload)
    arrays = dict(saved_warm)
    arrays["payload_json"] = np.asarray(json.dumps(_plot_payload(payload), sort_keys=True))
    arrays["metadata_json"] = np.asarray(json.dumps(diagnostics, sort_keys=True))
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
    """Return (warm, payload) with no cluster, solver, or EOS dependencies."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported isothermal logT plot archive schema")
        payload = json.loads(str(archive["payload_json"]))
        warm = {key: archive[key].copy() for key in archive.files
                if key not in ("metadata_json", "payload_json")}
    return warm, payload


def _save_figure_atomic(fig, path, style):
    """Publish only a fully rendered image on the destination filesystem."""
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    try:
        fig.savefig(temporary, format=path.suffix.lstrip("."), dpi=style["export_dpi"],
                    facecolor=style["background_color"])
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _shade(ax, warm, payload, style):
    temperatures = sorted(set(map(float, payload.get("temperatures", TEMPERATURES))) |
                          {float(row.get("T_0minus", float.fromhex(key)))
                           for key, row in payload.get("boundaries", {}).items()})
    for side, field, edge in (("inner", "lower", style["x_limits"][0]),
                              ("outer", "upper", style["x_limits"][1])):
        points = []
        for T in temperatures:
            row = payload.get("boundaries", {}).get(float(T).hex(), {})
            value = row.get(field)
            valid = (row.get("status") == "success" and value is not None and
                     np.isfinite(value) and .01 <= T <= JOIN_TEMPERATURE)
            points.append(np.array([value, T]) if valid else None)
        runs = _runs(points)
        warm_vertices = warm[f"{side}_boundary"]
        if len(warm_vertices):
            seam = warm.get(f"{side}_boundary_seam_density")
            if points and points[-1] is not None and points[-1][1] == JOIN_TEMPERATURE:
                warm_vertices = np.vstack((points[-1], warm_vertices))
            elif seam is not None:
                warm_vertices = np.vstack(([float(seam), JOIN_TEMPERATURE], warm_vertices))
            runs.append(warm_vertices)
        for index, run in enumerate(runs):
            if len(run) < 2:
                continue
            is_warm = bool(len(warm_vertices) and index == len(runs) - 1)
            end_temperature = style["y_limits"][1] if is_warm else run[-1, 1]
            # The outer polygon traverses the full upper edge, including x=1.
            extras = [(run[-1, 0], end_temperature), (edge, end_temperature), (edge, run[0, 1])]
            polygon = np.vstack((run, extras))
            patch = ax.fill(polygon[:, 0], polygon[:, 1], color=style["excluded_color"],
                            alpha=style["excluded_alpha"], zorder=style["excluded_zorder"])[0]
            patch.set_gid(f"excluded:{side}:" + ("warm" if is_warm else str(index)))


def plot_contours(warm, payload, output_dir=None, stem="isothermal-logT", **style):
    """Return a figure and attach plot diagnostics as ``fig.logT_diagnostics``."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    unknown = set(style) - set(DEFAULT_STYLE)
    if unknown:
        raise TypeError(f"Unknown plotting style keys: {sorted(unknown)}")
    style = dict(DEFAULT_STYLE, **style)
    if style["y_limits"][0] <= 0:
        raise ValueError("The logarithmic temperature axis requires a positive lower limit")
    warm, curves, accepted, diagnostics = _geometry(warm, payload)
    rc = {"mathtext.fontset": style["mathtext_fontset"], "mathtext.rm": style["mathtext_roman"],
          "mathtext.it": style["mathtext_italic"], "mathtext.bf": style["mathtext_bold"]}
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=style["figure_size"], dpi=style["figure_dpi"],
                               constrained_layout=style["constrained_layout"],
                               facecolor=style["background_color"])
        ax.set_facecolor(style["background_color"])
        _shade(ax, warm, payload, style)
        for i, (level, runs) in enumerate(curves.items()):
            color = style["colors"][LEVELS.index(level) if level in LEVELS else i % len(style["colors"])]
            for index, run in enumerate(runs):
                line, = ax.plot(run[:, 0], run[:, 1], color=color, linewidth=style["line_width"],
                                linestyle=style["line_style"], zorder=style["contour_zorder"])
                line.set_gid(f"contour:{level:g}:{index}")
            roots = accepted[level]
            if len(roots):
                collection = ax.scatter(roots[:, 0], roots[:, 1], s=style["marker_size"], color=color,
                                        zorder=style["contour_zorder"])
                collection.set_gid(f"cold-solver-roots:{level:g}")
            label_T = style["label_temperatures"].get(level)
            if label_T is not None:
                candidate = next((run for run in runs if run[0, 1] <= label_T <= run[-1, 1]), None)
                if candidate is not None:
                    x = np.interp(label_T, candidate[:, 1], candidate[:, 0])
                    ax.text(x, label_T, style["label_format"] % level, color=color,
                            fontsize=style["label_font_size"], ha="center", va="center",
                            backgroundcolor=style["background_color"], zorder=style["region_text_zorder"])
        for side in ("inner", "outer"):
            ax.text(*style[f"{side}_text_position"], style[f"{side}_text"],
                    color=style["region_text_color"], fontsize=style["region_font_size"],
                    va="top" if side == "outer" else "baseline",
                    zorder=style["region_text_zorder"])
        inconsistent = [key for key, value in diagnostics["joins"].items() if value["status"] == "inconsistent"]
        notices = []
        if inconsistent:
            notices.append('Unjoined at 20 MeV: '+', '.join(inconsistent)+' m/s')
        failed_checks = [v for v in diagnostics['accuracy_checks'].values() if v['status'] not in ('success','running')]
        if failed_checks:
            notices.append(f'Accuracy checks unresolved: {len(failed_checks)}')
        if notices:
            ax.text(*style["diagnostic_position"], '\n'.join(notices),
                    transform=ax.transAxes, color=style["diagnostic_color"],
                    fontsize=style["diagnostic_font_size"], zorder=style["region_text_zorder"])
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
