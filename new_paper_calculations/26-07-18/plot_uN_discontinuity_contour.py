#!/usr/bin/env python3
"""
Render the safe-domain extreme-endothermic velocity contour.

Reads the adaptive 2D mesh written by run_uN_discontinuity_contour.py and draws
the finite front-speed field u_N(0-) in m/s. The independently solved
A_boundary = 1 and Delta h = 0 limits are overlaid; no solver interpolation is
performed outside their intersection.

Usage:
    python plot_uN_discontinuity_contour.py [input.npy] [-o output.pdf]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = "uN-safe-adaptive-contour.npy"

# ---- Tunable plotting parameters -------------------------------------------
FIGURE_SIZE = (7.2, 5.4)
DPI = 200
TITLE_FONT_SIZE = 13
AXIS_LABEL_FONT_SIZE = 12
TICK_LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 9
CONTOUR_LABEL_FONT_SIZE = 8
COLORBAR_LABEL_FONT_SIZE = 11
LINE_WIDTH = 2.0
BOUNDARY_LINE_WIDTH = 2.4

VELOCITY_CMAP = "viridis"
VELOCITY_LEVELS_PER_DECADE = 6            # filled-field resolution
CONTOUR_LINE_LEVELS_M_S = [1.0, 1e1, 1e2, 1e3, 1e4]   # labelled iso-speed lines
FORBIDDEN_GRAY = "0.82"
A_BOUNDARY_COLOR = "0.25"
GAP_BOUNDARY_COLOR = "crimson"
A_BOUNDARY_LINE_STYLE = "--"
GAP_BOUNDARY_LINE_STYLE = "-"

X_LABEL = r"metastability  $(n_B(0^-)-n_B^{\rm crit})/n_0$"
Y_LABEL = r"temperature  $T$  [MeV]"
COLORBAR_LABEL = r"$u_N(0^-)$  [m s$^{-1}$]"
TITLE = r"Extreme-endothermic front speed in the safe hydro domain"


def load_payload(path: Path) -> dict:
    return np.load(path, allow_pickle=True).tolist()


def make_figure(payload: dict):
    m = np.asarray(payload["metastability_grid"], dtype=float)
    T = np.asarray(payload["temperature_grid"], dtype=float)
    V = np.asarray(payload["velocity_grid_m_s"], dtype=float)
    if m.ndim == 1 and T.ndim == 1 and V.shape == (m.size, T.size):
        # Compatibility with the original rectangular payload.
        m, T = np.meshgrid(m, T)
        V = V.T
    if not (m.shape == T.shape == V.shape):
        raise ValueError("metastability, temperature, and velocity grids must have matching shapes")

    A_boundary_m = np.asarray(
        payload.get("A_boundary_metastability_values", []), dtype=float
    )
    A_boundary_T = np.asarray(
        payload.get("A_boundary_temperature_values", []), dtype=float
    )
    gap_boundary_T = np.asarray(
        payload.get("gap_boundary_temperature_values", []), dtype=float
    )
    gap_boundary_m = np.asarray(
        payload.get("gap_boundary_metastability_values", []), dtype=float
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)

    finite = np.isfinite(V) & (V > 0.0)
    if np.any(finite):
        vmin = float(np.nanmin(V[finite]))
        vmax = float(np.nanmax(V[finite]))
        lo = np.floor(np.log10(vmin))
        hi = np.ceil(np.log10(vmax))
        levels = np.logspace(lo, hi, int((hi - lo) * VELOCITY_LEVELS_PER_DECADE) + 1)
        Vmasked = np.ma.masked_invalid(V)
        if min(V.shape) >= 2:
            mesh = ax.contourf(
                m, T, Vmasked, levels=levels,
                norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
                cmap=VELOCITY_CMAP, extend="both",
            )
        else:
            mesh = ax.scatter(
                m[finite], T[finite], c=V[finite], s=30,
                norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
                cmap=VELOCITY_CMAP,
            )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(COLORBAR_LABEL, fontsize=COLORBAR_LABEL_FONT_SIZE)
        cbar.ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)

        line_levels = [lv for lv in CONTOUR_LINE_LEVELS_M_S if levels[0] <= lv <= levels[-1]]
        if line_levels and min(V.shape) >= 2:
            lines = ax.contour(m, T, Vmasked, levels=line_levels,
                               colors="white", linewidths=0.8, alpha=0.7)
            ax.clabel(lines, fmt="%g", fontsize=CONTOUR_LABEL_FONT_SIZE)

    # The adaptive grid contains safe points only. Shade the area above the
    # combined per-column cap so missing values cannot look like data holes.
    safe_m = np.asarray(payload.get("A_boundary_metastability_values", []), dtype=float)
    safe_T = np.asarray(payload.get("safe_temperature_cut_values", []), dtype=float)
    safe = np.isfinite(safe_m) & np.isfinite(safe_T)
    if np.any(safe):
        order = np.argsort(safe_m[safe])
        ax.fill_between(
            safe_m[safe][order],
            safe_T[safe][order],
            float(np.nanmax(T)),
            color=FORBIDDEN_GRAY,
            zorder=0,
        )

    good_A = np.isfinite(A_boundary_m) & np.isfinite(A_boundary_T)
    if np.any(good_A):
        ax.plot(
            A_boundary_m[good_A],
            A_boundary_T[good_A],
            color=A_BOUNDARY_COLOR,
            ls=A_BOUNDARY_LINE_STYLE,
            lw=BOUNDARY_LINE_WIDTH,
            zorder=4,
            label=r"$A_{\rm boundary}=1$",
        )

    good_gap = np.isfinite(gap_boundary_m) & np.isfinite(gap_boundary_T)
    if np.any(good_gap):
        ax.plot(
            gap_boundary_m[good_gap],
            gap_boundary_T[good_gap],
            color=GAP_BOUNDARY_COLOR,
            ls=GAP_BOUNDARY_LINE_STYLE,
            lw=BOUNDARY_LINE_WIDTH,
            zorder=4,
            label=r"$\Delta h=0$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    x_min = float(np.nanmin(m))
    x_max = float(np.nanmax(m))
    if x_min == x_max:
        x_min *= 0.95
        x_max *= 1.05
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(np.nanmin(T), np.nanmax(T))
    ax.set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(TITLE, fontsize=TITLE_FONT_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)

    handles = [Patch(facecolor=FORBIDDEN_GRAY, edgecolor="0.4", label="outside safe scan domain")]
    if np.any(good_A):
        handles.append(
            plt.Line2D(
                [0], [0], color=A_BOUNDARY_COLOR, ls=A_BOUNDARY_LINE_STYLE,
                lw=BOUNDARY_LINE_WIDTH, label=r"$A_{\rm boundary}=1$",
            )
        )
    if np.any(good_gap):
        handles.append(
            plt.Line2D(
                [0], [0], color=GAP_BOUNDARY_COLOR, ls=GAP_BOUNDARY_LINE_STYLE,
                lw=BOUNDARY_LINE_WIDTH, label=r"$\Delta h=0$",
            )
        )
    ax.legend(handles=handles, fontsize=LEGEND_FONT_SIZE, loc="upper left",
              framealpha=0.9)

    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=str(SCRIPT_DIR / DEFAULT_INPUT))
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    payload = load_payload(input_path)
    fig = make_figure(payload)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".pdf")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    print(f"Wrote {output_path} and {output_path.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
