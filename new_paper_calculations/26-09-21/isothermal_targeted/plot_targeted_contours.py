"""Read-only plotting of preserved contours and certified targeted intervals.

New lines follow accepted numerical points only.  The interpolation used to
trim an overlapping old polyline is geometric clipping, not a velocity model.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPAIR_FOLDER = Path(__file__).resolve().parents[2] / "26-09-16/isothermal_numeric_repair"
if str(REPAIR_FOLDER) not in sys.path:
    sys.path.insert(0, str(REPAIR_FOLDER))

from _repair_common import DEFAULT_CONTROLS as DEFAULT_REPAIR_CONTROLS, LEVELS
from plot_isothermal_contours import (
    boundary_curve, continuous_level_curves, interpolated_bridges, merged_contour_segments,
    ray_coordinates)


COLORS = ("#1f77b4", "#17becf", "#ff7f0e", "#8c6d31", "#2ca02c", "#1f77b4", "#9467bd")
DENSITY_REPLACEMENT_TOLERANCE = .10
ROOT_MARKER_AREA = 12.


def _valid_point(point):
    if not isinstance(point, dict):
        return False
    try:
        return (all(np.isfinite(float(point[key])) for key in
                    ("T_0minus", "nB_0minus_over_n0", "velocity_m_s"))
                and float(point["velocity_m_s"]) > 0. and point.get("success", True))
    except (KeyError, TypeError, ValueError):
        return False


def _certified_tasks(targeted):
    return [task for task in (targeted or {}).get("targets", {}).values()
            if task.get("status") == "success" and task.get("certified") is True
            and not task.get("children")
            and all(_valid_point(task.get(name)) for name in ("left", "root", "right"))]


def _linear_interval(start, change, lower, upper):
    """Parameter interval on [0, 1] satisfying one linear range constraint."""
    if change == 0.:
        return (0., 1.) if lower <= start <= upper else None
    first, second = sorted(((lower - start) / change, (upper - start) / change))
    first, second = max(0., first), min(1., second)
    return (first, second) if second > first else None


def _covered_interval(left, right, target_left, target_right, density_tolerance):
    change = right - left
    target_change = target_right - target_left
    low_temperature, high_temperature = sorted((target_left[1], target_right[1]))
    temperature_interval = _linear_interval(left[1], change[1], low_temperature, high_temperature)
    if temperature_interval is None:
        return None
    if target_change[1] == 0.:
        low_density, high_density = sorted((target_left[0], target_right[0]))
        density_interval = _linear_interval(left[0], change[0], low_density - density_tolerance,
                                            high_density + density_tolerance)
    else:
        slope = target_change[0] / target_change[1]
        offset = left[0] - target_left[0] - (left[1] - target_left[1]) * slope
        density_interval = _linear_interval(offset, change[0] - change[1] * slope,
                                            -density_tolerance, density_tolerance)
    if density_interval is None:
        return None
    first = max(temperature_interval[0], density_interval[0])
    last = min(temperature_interval[1], density_interval[1])
    return (first, last) if last > first else None


def _trim_covered_segments(segments, target_lines, *, density_tolerance=DENSITY_REPLACEMENT_TOLERANCE):
    """Subtract only close, same-level interval masks from old geometry.

    Split exactly where the overlap starts or ends, retaining every exterior
    piece and unrelated density branch.  Task endpoints set the temperature
    coverage; it is not subject to the older global point-spacing heuristic.
    """
    if not target_lines:
        return [np.asarray(line).copy() for line in segments]
    target_edges = [(left, right) for line in target_lines for left, right in zip(line[:-1], line[1:])]
    output = []
    for line in segments:
        active = []

        def flush():
            if len(active) >= 2:
                output.append(np.asarray(active))
            active.clear()

        for left, right in zip(line[:-1], line[1:]):
            covered = sorted(interval for target_left, target_right in target_edges
                             if (interval := _covered_interval(left, right, target_left, target_right,
                                                               density_tolerance)) is not None)
            keep, cursor = [], 0.
            for first, last in covered:
                if first > cursor:
                    keep.append((cursor, first))
                cursor = max(cursor, last)
            if cursor < 1.:
                keep.append((cursor, 1.))
            if not keep:
                flush()
            for first, last in keep:
                start = left + first * (right - left)
                end = left + last * (right - left)
                if active and not np.allclose(active[-1], start, rtol=0., atol=1.e-12):
                    flush()
                if not active:
                    active.append(start)
                active.append(end)
                if last < 1.:
                    flush()
        flush()
    return output


def _root_groups(repair, targeted):
    """Include isolated accepted roots even when their interval is unresolved."""
    candidates = []
    for ray in (repair or {}).get("rays", {}).values():
        if ray.get("status") == "success":
            candidates.extend((root, ray["level"]) for root in ray.get("roots", []))
    for root in (targeted or {}).get("source_roots", []):
        candidates.append((root, root.get("level")))
    for task in (targeted or {}).get("targets", {}).values():
        candidates.append((task.get("root"), task.get("level")))
    groups = {level: [] for level in LEVELS}
    seen = set()
    for root, level in candidates:
        if not _valid_point(root):
            continue
        if level is None:
            # Some saved point records carry the accepted speed but not its
            # target level.  This labels a root; it does not interpolate it.
            matched = [value for value in LEVELS if abs(root["velocity_m_s"] / value - 1.) <= .005]
            level = matched[0] if len(matched) == 1 else None
        if level not in groups:
            continue
        coordinates = (float(root["nB_0minus_over_n0"]), float(root["T_0minus"]))
        key = (level, *coordinates)
        if key not in seen:
            groups[level].append(coordinates)
            seen.add(key)
    return groups


def accepted_root_groups(repair, targeted=None):
    """Return accepted (density/n0, temperature) marker coordinates by level.

    Includes isolated source and new roots, including uncertified intervals;
    repeated coordinates at the same level are returned only once.
    """
    return _root_groups(repair, targeted)


def _contour_parts(domain, numerical, repair, targeted, *, levels=LEVELS):
    if targeted and targeted.get("domain_fingerprint", domain["domain_fingerprint"]) != domain["domain_fingerprint"]:
        raise ValueError("Targeted and baseline domain fingerprints differ")
    tasks = _certified_tasks(targeted)
    target_lines = {level: [] for level in levels}
    for task in tasks:
        if task["level"] in LEVELS and task["level"] in target_lines:
            target_lines[task["level"]].append(np.asarray([
                (task[name]["nB_0minus_over_n0"], task[name]["T_0minus"]) for name in ("left", "root", "right")]))
    masks = {level: list(lines) for level, lines in target_lines.items()}
    for task in (targeted or {}).get("targets", {}).values():
        # A failed/deferred/refining interval must not borrow the inherited
        # line as apparent evidence of success.  Pending tasks leave it alone.
        if (task.get("status", "pending") in ("pending", "success") or task.get("level") not in masks
                or not all(_valid_point(task.get(name)) for name in ("left", "right"))):
            continue
        names = ("left", "root", "right") if _valid_point(task.get("root")) else ("left", "right")
        masks[task["level"]].append(np.asarray([
            (task[name]["nB_0minus_over_n0"], task[name]["T_0minus"]) for name in names]))
    original = merged_contour_segments(domain, numerical, repair, levels=levels)
    clipped = [_trim_covered_segments(lines, masks[level]) for level, lines in zip(levels, original)]
    return tasks, clipped, target_lines


def targeted_bridges(domain, repair, targeted=None, *, levels=LEVELS):
    """Interpolated gap fillers, dropped wherever a certified interval covers them.

    Bridges are fabricated geometry, so a real targeted result always wins and
    callers must draw whatever survives here differently from solved arcs.
    """
    bridges = interpolated_bridges(domain, repair, levels=levels)
    covered = {level: [np.asarray([(task[name]["nB_0minus_over_n0"], task[name]["T_0minus"])
                                   for name in ("left", "root", "right")])
                       for task in _certified_tasks(targeted) if task["level"] == level]
               for level in levels}
    return [_trim_covered_segments(lines, covered[level]) for level, lines in zip(levels, bridges)]


def merged_targeted_segments(domain, numerical, repair, targeted=None, *, levels=LEVELS):
    """Return per-level contour polylines in requested level order, without plotting.

    Each level contains the geometrically clipped original/repair segments,
    followed by certified successful leaf intervals.  Unresolved attempted
    intervals mask inherited crossings, and certified children restore only
    their own pieces.  With no targeted payload, original geometry is unchanged.
    Level selection affects display only; targeted edges use predefined levels.
    """
    levels = tuple(levels)
    _, clipped, target_lines = _contour_parts(domain, numerical, repair, targeted, levels=levels)
    return [lines + target_lines[level] for level, lines in zip(levels, clipped)]


def _atomic_save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    try:
        fig.savefig(temporary, format=path.suffix[1:], dpi=220, facecolor="white")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def plot_targeted(domain, numerical, repair, targeted=None, *, output_dir=None):
    """Plot immutable source data plus the optional targeted checkpoint.

    A missing targeted payload is normal before the first evaluation.  Pending,
    failed, uncertified and internal refinement tasks never create new edges.
    Attempted unresolved intervals also mask matching inherited crossings.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.contour import ContourSet

    tasks, clipped, _ = _contour_parts(domain, numerical, repair, targeted)
    with plt.rc_context({"mathtext.fontset": "cm"}):
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=160, constrained_layout=True)
        inner_x, inner_y = boundary_curve(domain, "delta_muB_equals_zero")
        outer_x, outer_y = boundary_curve(domain, "a_0plus_equals_one")
        ax.fill(np.r_[inner_x, 1., inner_x[0]], np.r_[inner_y, 0., 0.], color="0.72", zorder=0)
        ax.fill(np.r_[outer_x, 1., 5.5, 5.5, outer_x[0]], np.r_[outer_y, 115., 115., 0., 0.], color="0.72", zorder=0)
        contour = ContourSet(ax, LEVELS, clipped, colors=COLORS, linewidths=1.8, zorder=3)
        contour.set_gid("baseline-repair-contours")
        ax.clabel(contour, inline=True, inline_spacing=4, fmt="%g", fontsize=11)
        for level, bridges in zip(LEVELS, targeted_bridges(domain, repair, targeted)):
            for index, bridge in enumerate(bridges):
                ax.plot(bridge[:, 0], bridge[:, 1], color=COLORS[LEVELS.index(level)],
                        linewidth=1.8, linestyle=(0, (5, 3)), zorder=2,
                        gid=f"interpolated-bridge:{level:g}:{index}")
        for task in tasks:
            if task["level"] not in LEVELS:
                continue
            color = COLORS[LEVELS.index(task["level"])]
            ax.plot([task[name]["nB_0minus_over_n0"] for name in ("left", "root", "right")],
                    [task[name]["T_0minus"] for name in ("left", "root", "right")],
                    color=color, linewidth=1.8, zorder=4, gid=f"targeted:{task['id']}")
        for level, coordinates in accepted_root_groups(repair, targeted).items():
            if coordinates:
                density, temperature = np.asarray(coordinates).T
                ax.scatter(density, temperature, s=ROOT_MARKER_AREA, color=COLORS[LEVELS.index(level)],
                           edgecolors="none", zorder=5, gid=f"accepted-contour-roots:{level:g}")
        ax.text(4.3, 102., r"$a(0^+)\geq1$", fontsize=18, color="0.10")
        ax.text(1.18, 20., "Stable Neutron\nMatter", fontsize=18, color="0.10")
        ax.set(xlim=(1., 5.5), ylim=(0., 115.), box_aspect=1.)
        ax.set_xlabel(r"$n_B(0^-)/n_0$", fontsize=18)
        ax.set_ylabel(r"$T(0^-)$ [MeV]", fontsize=18)
        ax.set_title(r"Numerical $u(0^-)$ [$\mathrm{m\,s^{-1}}$], isothermal", fontsize=20, pad=8)
        ax.tick_params(direction="in", top=True, right=True, length=4, width=.9, labelsize=18)
        for spine in ax.spines.values():
            spine.set_linewidth(.8)
        if output_dir is not None:
            for extension in ("png", "pdf"):
                _atomic_save_figure(fig, Path(output_dir) / f"isothermal-targeted-contours.{extension}")
    return fig, ax
