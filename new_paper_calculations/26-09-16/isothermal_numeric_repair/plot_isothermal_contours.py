"""Plot original numerical contours plus certified numerical continuation segments."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import contourpy
from scipy.interpolate import PchipInterpolator

from _repair_common import LEVELS
from _repair_search import connected_segments


def boundary_curve(domain, name, samples=401):
    records = sorted(domain["boundary_ray_records"], key=lambda row: row["theta_rad"])
    angles = np.asarray([row["theta_rad"] for row in records], float)
    radii = np.asarray([row[name]["rho"] for row in records], float)
    theta = np.linspace(angles[0], angles[-1], samples)
    rho = np.interp(theta, angles, radii)
    transform = domain["ray_transform"]
    return (transform["density_origin_over_n0"] + transform["density_scale_over_n0"] * rho * np.cos(theta),
            transform["temperature_scale_MeV"] * rho * np.sin(theta))


def ray_coordinates(domain, theta, fraction):
    """Map ray coordinates back to the (density/n0, temperature) plane.

    Exact inverse of the projection used below, so a bridge built this way
    lands on its two accepted end roots to machine precision.
    """
    theta = np.asarray(theta, float)
    inner = np.interp(theta, domain["theta_axis_rad"], domain["rho_inner_boundary"])
    outer = np.interp(theta, domain["theta_axis_rad"], domain["rho_outer_boundary"])
    rho = inner + np.asarray(fraction, float) * (outer - inner)
    transform = domain["ray_transform"]
    return np.column_stack((
        transform["density_origin_over_n0"] + transform["density_scale_over_n0"] * rho * np.cos(theta),
        transform["temperature_scale_MeV"] * rho * np.sin(theta)))


def interpolated_bridges(domain, repair, *, levels=LEVELS):
    """Interpolated polylines across scheduled angles holding no accepted root.

    These are the only fabricated geometry in the figure and callers must draw
    them differently from solved arcs.  The radial fraction is interpolated in
    angle, so a bridge follows the domain geometry instead of cutting the chord
    between its endpoints.  Only the interior of a level's sweep is bridged: an
    open end is never extrapolated, and a ray carrying more than one branch is
    left alone because the correspondence across the gap is ambiguous.
    """
    if repair is None:
        return [[] for _ in levels]
    axis = np.asarray(domain["theta_axis_rad"], float)
    rays = list(repair["rays"].values())
    output = []
    for level in levels:
        accepted = sorted((ray for ray in rays if ray["level"] == level
                           and ray["status"] == "success" and len(ray.get("roots") or []) == 1),
                          key=lambda ray: ray["theta_rad"])
        bridges = []
        for left, right in zip(accepted[:-1], accepted[1:]):
            first = int(np.argmin(np.abs(axis - left["theta_rad"])))
            last = int(np.argmin(np.abs(axis - right["theta_rad"])))
            if last - first < 2:
                continue
            theta = axis[first:last + 1]
            fraction = np.interp(theta, (left["theta_rad"], right["theta_rad"]),
                                 (left["roots"][0]["radial_fraction"],
                                  right["roots"][0]["radial_fraction"]))
            bridges.append(ray_coordinates(domain, theta, fraction))
        output.append(bridges)
    return output


def _accepted_rays(repair, extra):
    """Every single-branch accepted ray from the repair payload and its addenda."""
    rays = []
    for payload in (repair, *extra):
        rays += [ray for ray in (payload or {}).get("rays", {}).values()
                 if ray["status"] == "success" and len(ray.get("roots") or []) == 1]
    return rays


def _radial_fraction_table(domain, repair, levels, extra=()):
    """Radial fraction per (level, angular node), NaN where nothing is known.

    Solved single-branch roots first, then interior interpolation in angle --
    the same rule `interpolated_bridges` uses -- and finally the open ends of a
    sweep, continued only where the level is bracketed by a lower and a higher
    level that both have a value at that angle.  The bracketing interpolation is
    linear in log(1 - fraction) against log(level), because the contours crowd
    against the outer boundary; on every angle where 500, 1000 and 2000 m/s are
    all solved it reproduces the solved 1000 m/s root to within 0.34 MeV.
    """
    axis = np.asarray(domain["theta_axis_rad"], float)
    rays = _accepted_rays(repair, extra)
    table = {level: np.full(axis.size, np.nan) for level in levels}
    for ray in rays:
        if ray["level"] in table:
            node = int(np.argmin(np.abs(axis - ray["theta_rad"])))
            table[ray["level"]][node] = ray["roots"][0]["radial_fraction"]
    for fraction in table.values():
        known = np.flatnonzero(np.isfinite(fraction))
        if known.size >= 2:
            interior = np.arange(known[0], known[-1] + 1)
            fraction[interior] = np.interp(axis[interior], axis[known], fraction[known])
    ordered = sorted(table)
    for index, level in enumerate(ordered):
        lower = ordered[:index]
        upper = ordered[index + 1:]
        fraction = table[level]
        known = np.flatnonzero(np.isfinite(fraction))
        if known.size < 2:
            continue
        for node in list(range(known[0] - 1, -1, -1)) + list(range(known[-1] + 1, axis.size)):
            below = [other for other in lower if np.isfinite(table[other][node])]
            above = [other for other in upper if np.isfinite(table[other][node])]
            if not below or not above:
                continue
            first, second = below[-1], above[0]
            weight = np.log(level / first) / np.log(second / first)
            gap = np.log1p(-table[first][node]) * (1. - weight) + np.log1p(-table[second][node]) * weight
            fraction[node] = -np.expm1(gap)
    return axis, table


def continuous_level_curves(domain, repair, *, levels=LEVELS, samples=0, extra=(),
                            cold_anchor=False):
    """One solid polyline per level, plus the mask of angles actually solved.

    Interior gaps and bracketed open ends are filled, so each level is a single
    curve instead of a set of arcs.  The mask is the provenance record, one entry
    per scheduled angle; a caller that must distinguish solved from filled
    geometry should use `merged_contour_segments` with `interpolated_bridges`.

    `samples` resamples the radial fraction against angle with a monotone cubic,
    which removes the polygonal look of the 5-degree angular axis without moving
    any accepted root: the interpolant passes through every node exactly and its
    shape preservation rules out overshoot between them.

    `extra` merges accepted rays from later checkpoints, such as the 26-09-23
    cold-endpoint solves, without touching the 26-09-16 payload.

    `cold_anchor` adds the exact point (3 n0, 0) at angle zero.  That is where
    the delta-muB = 0 boundary meets T(0-) = 0, the front speed vanishes on that
    boundary, and the radial fraction of every level's crossing goes to zero as
    the angle does -- the solved 50--500 m/s roots follow fraction ~ angle^0.78
    and the analytic grid shows the same for 1000--5000 m/s.  Between the anchor
    and the coldest accepted root the curve is one monotone cubic segment; no
    intermediate node is invented there.
    """
    if repair is None:
        return [(np.zeros((0, 2), float), np.zeros(0, bool)) for _ in levels]
    extra = tuple(extra)
    axis, table = _radial_fraction_table(domain, repair, tuple(levels), extra)
    rays = _accepted_rays(repair, extra)
    output = []
    for level in levels:
        fraction = table[level]
        known = np.flatnonzero(np.isfinite(fraction))
        if known.size < 2:
            output.append((np.zeros((0, 2), float), np.zeros(0, bool)))
            continue
        if cold_anchor and known[0] > 0 and axis[0] == 0.:
            fraction[0] = 0.
            known = np.concatenate(([0], known))
        # One mask entry per data node, so an anchored cold tail -- a single
        # cubic segment with no nodes inside it -- does not dilute the count.
        position = {node: index for index, node in enumerate(known)}
        solved = np.zeros(known.size, bool)
        for ray in rays:
            if ray["level"] == level:
                node = int(np.argmin(np.abs(axis - ray["theta_rad"])))
                if node in position:
                    solved[position[node]] = True
        # Only the known nodes carry data; an anchored cold tail therefore stays
        # a single cubic segment instead of inheriting an invented node.
        theta, values = axis[known], fraction[known]
        if samples > known.size:
            # Equal resolution per node interval, not per radian: the angular
            # axis is geometric near zero, so a uniform grid would leave the
            # cold tail -- where the curve moves fastest -- visibly polygonal.
            step = max(2, -(-samples // max(1, known.size - 1)))
            dense = np.unique(np.concatenate([np.linspace(first, second, step + 1)
                                              for first, second in zip(theta[:-1], theta[1:])]))
            values = PchipInterpolator(theta, values)(dense)
            theta = dense
        output.append((ray_coordinates(domain, theta, values), solved))
    return output


def merged_contour_segments(domain, numerical, repair=None, *, levels=LEVELS, replacement_radial_tolerance=.15):
    """Remove duplicate baseline arcs only where a validated repair arc covers them."""
    if repair is not None and repair["domain_fingerprint"] != domain["domain_fingerprint"]:
        raise ValueError("Repair and baseline domain fingerprints differ")
    velocity = np.asarray(numerical["velocity_m_s_grid"], float)
    good = (numerical["task_status"] == "success") & np.isfinite(velocity) & (velocity > 0)
    generator = contourpy.contour_generator(x=domain["nB_0minus_over_n0_grid"], y=domain["T_0minus_grid_MeV"],
                                            z=np.ma.masked_where(~good, velocity), corner_mask=True)
    transform = domain["ray_transform"]
    output = []
    for level in levels:
        repaired = connected_segments(list(repair["rays"].values()), level, repair["controls"],
                                      theta_axis=domain["theta_axis_rad"]) if repair else []
        repaired = [line for line in repaired if all(np.isfinite(p["velocity_m_s"]) and
                    abs(p["velocity_m_s"] / level - 1.) <= repair["controls"]["speed_rtol"] for p in line)]
        lines = []
        for baseline_line in generator.lines(level):
            active = []
            for left, right in zip(baseline_line[:-1], baseline_line[1:]):
                middle = .5 * (left + right)
                x = (middle[0] - transform["density_origin_over_n0"]) / transform["density_scale_over_n0"]
                y = middle[1] / transform["temperature_scale_MeV"]
                theta = float(np.arctan2(y, x))
                inner = np.interp(theta, domain["theta_axis_rad"], domain["rho_inner_boundary"])
                outer = np.interp(theta, domain["theta_axis_rad"], domain["rho_outer_boundary"])
                fraction = (np.hypot(x, y) - inner) / (outer - inner)
                covered = any(line[0]["theta_rad"] <= theta <= line[-1]["theta_rad"] and
                    abs(fraction - np.interp(theta, [p["theta_rad"] for p in line],
                                             [p["radial_fraction"] for p in line])) < replacement_radial_tolerance
                    for line in repaired)
                if covered:
                    if len(active) >= 2:
                        lines.append(np.asarray(active))
                    active = []
                else:
                    if not active:
                        active.append(left)
                    active.append(right)
            if len(active) >= 2:
                lines.append(np.asarray(active))
        lines.extend(np.asarray([(p["nB_0minus_over_n0"], p["T_0minus"]) for p in line]) for line in repaired)
        output.append(lines)
    return output


def plot_contours(domain, numerical, *, repair=None, output_dir=None, stem="isothermal-numerical-repaired"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.contour import ContourSet
    colors = ("#1f77b4", "#17becf", "#ff7f0e", "#8c6d31", "#2ca02c", "#1f77b4", "#9467bd")
    with plt.rc_context({"mathtext.fontset": "cm"}):
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=160, constrained_layout=True)
        inner_x, inner_y = boundary_curve(domain, "delta_muB_equals_zero")
        outer_x, outer_y = boundary_curve(domain, "a_0plus_equals_one")
        ax.fill(np.r_[inner_x, 1., inner_x[0]], np.r_[inner_y, 0., 0.], color="0.72", zorder=0)
        ax.fill(np.r_[outer_x, 1., 5.5, 5.5, outer_x[0]], np.r_[outer_y, 115., 115., 0., 0.], color="0.72", zorder=0)
        segments = merged_contour_segments(domain, numerical, repair)
        contour = ContourSet(ax, LEVELS, segments, colors=colors, linewidths=1.8, zorder=3)
        ax.clabel(contour, inline=True, inline_spacing=4, fmt="%g", fontsize=11)
        for level, bridges in zip(LEVELS, interpolated_bridges(domain, repair)):
            for bridge in bridges:
                ax.plot(bridge[:, 0], bridge[:, 1], color=colors[LEVELS.index(level)],
                        linewidth=1.8, linestyle=(0, (5, 3)), zorder=2)
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
                path = Path(output_dir) / f"{stem}.{extension}"
                fig.savefig(path, dpi=220, facecolor="white")
                print(f"Saved {path}")
    return fig, ax
