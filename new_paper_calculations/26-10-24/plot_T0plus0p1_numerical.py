#!/usr/bin/env python3
"""Plot only accepted new-solver results; never fill gaps with old velocities."""
from pathlib import Path
import argparse
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'nsm-contour-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# Tunable plotting parameters
FIGURE_SIZE = (8.0, 6.0)
EXPORT_DPI = 220
TITLE_FONT_SIZE = 13
AXIS_LABEL_FONT_SIZE = 13
TICK_LABEL_FONT_SIZE = 11
CONTOUR_LABEL_FONT_SIZE = 9
CONTOUR_LINE_WIDTH = 0.65
CONTOUR_COLOR = 'black'
COLORMAP = 'viridis'
FILL_LEVEL_COUNT = 35
LINE_LEVELS_M_S = (30, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000)
GRID_OPACITY = 0.16
GRID_LINE_WIDTH = 0.5

SCRIPT_DIR = Path(__file__).resolve().parent


def plot_payload(path, output, allow_partial=False):
    d = np.load(path, allow_pickle=True).item()
    if d.get('hydrodynamic_model') != 'nonrelativistic_isobaric' or d.get('closure') != 'adiabatic':
        raise ValueError('Input is not a nonrelativistic adiabatic result')
    if not np.array_equal(d['T_0plus_values_MeV'], [.1]):
        raise ValueError('Input must have T(0+)=0.1 MeV')
    ok = np.asarray(d['numerical_outward_success_grid'])[..., 0]
    if not np.all(ok) and not allow_partial:
        raise ValueError('Some cells are missing or failed; use --allow-partial to display masked gaps')
    values = np.asarray(d['numerical_outward_u_0minus_m_s_grid'])[..., 0]
    ok = ok & np.isfinite(values) & (values > 0)
    if np.count_nonzero(ok) < 4 or min(ok.shape) < 2:
        raise ValueError('Not enough valid grid points for a contour')
    density = np.asarray(d['nB_0minus_over_n0_grid'])[..., 0]
    temperature = np.asarray(d['T_0minus_MeV_grid'])[..., 0]
    speeds = np.ma.array(values, mask=~ok)
    low, high = float(speeds.min()), float(speeds.max())
    levels = np.geomspace(low, high, FILL_LEVEL_COUNT)
    line_levels = [level for level in LINE_LEVELS_M_S if low < level < high]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout='constrained')
    fill = ax.contourf(density, temperature, speeds, levels=levels,
                       norm=LogNorm(low, high), cmap=COLORMAP, corner_mask=False)
    if line_levels:
        lines = ax.contour(density, temperature, speeds, levels=line_levels,
                           colors=CONTOUR_COLOR, linewidths=CONTOUR_LINE_WIDTH, corner_mask=False)
        ax.clabel(lines, fmt='%g', fontsize=CONTOUR_LABEL_FONT_SIZE)
    bar = fig.colorbar(fill, ax=ax, ticks=line_levels, format='%g')
    bar.set_label(r'$u(0^-)\ [\mathrm{m\,s^{-1}}]$', fontsize=AXIS_LABEL_FONT_SIZE)
    bar.ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    ax.set_xlabel(r'$n_B(0^-)/n_0$', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(r'$T(0^-)\ [\mathrm{MeV}]$', fontsize=AXIS_LABEL_FONT_SIZE)
    title = r'Nonrelativistic adiabatic front, $T(0^+)=0.1\,\mathrm{MeV}$'
    if not np.all(ok):
        title += f'\nPartial grid: {np.count_nonzero(ok)}/{ok.size} accepted'
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(alpha=GRID_OPACITY, linewidth=GRID_LINE_WIDTH)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf'):
        fig.savefig(output.with_suffix(suffix), dpi=EXPORT_DPI)
    plt.close(fig)
    print(f'Saved {output.with_suffix(".png")} and PDF; {np.count_nonzero(ok)}/{ok.size} accepted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=SCRIPT_DIR / 'T0plus0p1-adiabatic-slow.npy')
    parser.add_argument('--output', type=Path, default=SCRIPT_DIR / 'T0plus0p1-adiabatic-slow-contour')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    plot_payload(args.input, args.output, args.allow_partial)
