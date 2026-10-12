#!/usr/bin/env python3
"""Map the adiabatic cold-interface speed ratio over the upstream state plane.

With T(0+)=0 the interface composition is an output of the upstream state:
F(a) = 1 + (1+a)**(4/3) + (1-a)**(4/3) equals
[(h/nB)**4 / (4*pi**2*(P + B))]**(1/3) evaluated at 0-.  Because the speed
ratio depends only on a(0+) (massless scaling), iso-ratio lines in the
(nB(0-), T(0-)) plane are iso-F lines.  This script evaluates F on a uniform
upstream grid with the same read-only QMC-RMF3 validator used by
adiabatic_upstream.py, then contours it.  No rate integral is recomputed: the
ratio-to-composition map is the saved universal quadrature.

    python3 adiabatic_map.py            # compute missing grid states, then plot
    python3 adiabatic_map.py --plot     # plot from the saved grid only
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '1')

import adiabatic_upstream as upstream_eos  # inserts the working SciPy sidecar first

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
CACHE_DIR = RESULTS / 'adiabatic-map-cache'
GRID_FILE = RESULTS / 'adiabatic-map.json'
TEMPERATURES = tuple(round(2.5 * i, 10) for i in range(49))           # 0..120 MeV
DENSITIES = tuple(round(1 + 0.05 * i, 10) for i in range(101))         # 1..6 n0
F_EQUILIBRIUM = 3.0                                                    # a(0+) = 0
F_FLAVOR = 1 + 2**(4 / 3)                                              # a(0+) = 1
EQ23_LEVELS = (1.05, 1.1, 1.2, 1.3, 1.5, 1.75)


def composition_function(a):
    a = np.asarray(a, dtype=float)
    return 1 + (1 + a)**(4 / 3) + (1 - a)**(4 / 3)


def slice_states(temperature: float) -> list[dict]:
    """Evaluate one temperature row; each worker owns its own checkpoint file."""
    eos = upstream_eos.CachedUpstreamEOS(CACHE_DIR / f'T{temperature:g}.json')
    rows = []
    for density in DENSITIES:
        state = eos(temperature, density)
        row = dict(T_0minus=temperature, nB_0minus_over_n0=density, status=state['status'])
        if state['status'] != 'eos_failure':
            F = (state['h_over_nB_0minus']**4
                 / (4 * math.pi**2 * state['P_kinetic']))**(1 / 3)
            row.update(F=F, a_0plus=state['a_0plus'], P_0minus=state['P_0minus'],
                       h_over_nB_0minus=state['h_over_nB_0minus'],
                       upstream_seed_recovery_used=state['upstream_seed_recovery_used'])
        else:
            row['error'] = state.get('error')
        rows.append(row)
    return rows


def compute_grid(workers: int) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for block in pool.map(slice_states, TEMPERATURES) for row in block]
    failures = [r for r in rows if r['status'] == 'eos_failure']
    # Consistency of the classification with the F thresholds.
    for r in rows:
        if r['status'] == 'physical' and not F_EQUILIBRIUM < r['F'] < F_FLAVOR:
            raise ArithmeticError(f'Classification disagrees with F at {r}')
        if r['status'] == 'physical':
            residual = composition_function(r['a_0plus']) / r['F'] - 1
            if abs(residual) > 1e-12:
                raise ArithmeticError(f'F(a) residual {residual:.3g} at {r}')
    payload = dict(
        metadata=dict(
            description='F = [(h/nB)^4/(4 pi^2 (P+B))]^(1/3) of upstream QMC-RMF3 PNM; '
                        'F(a)=1+(1+a)^(4/3)+(1-a)^(4/3) fixes a(0+) for T(0+)=0',
            upstream_EOS='QMC-RMF3 pure neutron matter', B14_MeV=upstream_eos.B14,
            temperatures_MeV=list(TEMPERATURES), densities_over_n0=list(DENSITIES),
            eos_source_sha256=upstream_eos.source_hashes(),
            eos_failures=len(failures)),
        rows=rows)
    GRID_FILE.write_text(json.dumps(payload, indent=1, allow_nan=False) + '\n')
    return payload


def universal_ratio_curves():
    """Ratio(a) for each model from the verified universal quadrature."""
    from adiabatic_report import load_results
    from adiabatic_sensitivity import MODEL_NAMES
    _, _, best, _ = load_results(RESULTS)
    compositions = np.array(sorted(best))
    ratios = {name: np.array([best[a]['ratios'][i] for a in compositions])
              for i, name in enumerate(MODEL_NAMES)}
    return compositions, ratios


def make_plot(payload: dict) -> dict:
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/massless-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.interpolate import PchipInterpolator
    from plot_style import (FIGURE_WIDTH, TEXT_SIZE, GRID_ALPHA, LINE_WIDTH,
                            paper_style, save_figure)
    from validate_rate import RATE_MODEL_STYLES

    compositions, ratios = universal_ratio_curves()
    eq23 = ratios['eq23']
    if np.any(np.diff(eq23) <= 0):
        raise ValueError('Eq. (23) ratio is not monotone in a(0+); levels are ambiguous')
    inverse = PchipInterpolator(eq23, compositions)
    level_compositions = {level: float(inverse(level)) for level in EQ23_LEVELS
                          if eq23[0] < level < eq23[-1]}
    level_F = {level: float(composition_function(a)) for level, a in level_compositions.items()}

    temperatures = np.array(payload['metadata']['temperatures_MeV'])
    densities = np.array(payload['metadata']['densities_over_n0'])
    F = np.full((len(temperatures), len(densities)), np.nan)
    index_T = {t: i for i, t in enumerate(temperatures)}
    index_n = {n: j for j, n in enumerate(densities)}
    for row in payload['rows']:
        if 'F' in row:
            F[index_T[row['T_0minus']], index_n[row['nB_0minus_over_n0']]] = row['F']
    if np.isnan(F).any():
        raise ValueError('Map contains unresolved EOS states')

    eq23_style = next(s for s in RATE_MODEL_STYLES if s[0] == 'eq23_rate')
    exact_style = next(s for s in RATE_MODEL_STYLES if s[0] == 'exact_rate')
    t2_style = next(s for s in RATE_MODEL_STYLES if s[0] == 'sommerfeld_T2_rate')
    t2_max = float(ratios['sommerfeld_T2'].max())

    @paper_style
    def draw():
        fig, axis = plt.subplots(figsize=(FIGURE_WIDTH, 6.3))
        N, T = np.meshgrid(densities, temperatures)
        # Same light shading and grey text as the former "Unavailable" panels.
        axis.contourf(N, T, F, levels=[-np.inf, F_EQUILIBRIUM], colors=['#eeeeee'], zorder=0)
        axis.contourf(N, T, F, levels=[F_FLAVOR, np.inf], colors=['#eeeeee'], zorder=0)
        axis.contour(N, T, F, levels=[F_EQUILIBRIUM, F_FLAVOR], colors=exact_style[2],
                     linestyles='-', linewidths=LINE_WIDTH, zorder=3)
        contours = axis.contour(N, T, F, levels=sorted(level_F.values()), colors=eq23_style[2],
                                linestyles=eq23_style[3], linewidths=LINE_WIDTH, zorder=2)
        labels = {level_F[level]: f'{level:g}' for level in level_F}
        axis.clabel(contours, fmt=labels, fontsize=TEXT_SIZE, inline=True, inline_spacing=6,
                    colors=eq23_style[2])
        axis.text(1.7, 15, 'Neutron matter\nstable', ha='center', va='center',
                  fontsize=TEXT_SIZE, color='#666666')
        axis.text(5.45, 70, r'$a(0^+)\geq1$', ha='center', va='center',
                  fontsize=TEXT_SIZE, color='#666666')
        axis.set(xlim=(densities[0], densities[-1]), ylim=(temperatures[0], temperatures[-1]),
                 xticks=(1, 2, 3, 4, 5, 6), yticks=(0, 20, 40, 60, 80, 100, 120))
        axis.set_xlabel(r'Upstream density $n_B(0^-)/n_0$')
        axis.set_ylabel(r'$T(0^-)\ [\mathrm{MeV}]$')
        axis.grid(alpha=GRID_ALPHA)
        fig.suptitle('Adiabatic EOS trajectories\n' + r'$T(0^+)=0$, constant $P$ and $h/n_B$')
        handles = [
            Line2D([], [], color=exact_style[2], linestyle='-', linewidth=LINE_WIDTH,
                   label=r'Front boundaries $a(0^+)=0$ and $1$'),
            Line2D([], [], color=eq23_style[2], linestyle=eq23_style[3], linewidth=LINE_WIDTH,
                   label='Madsen Eq. (23) speed estimate / exact estimate'),
            Line2D([], [], color=t2_style[2], linestyle=t2_style[3], linewidth=LINE_WIDTH,
                   label=rf'Derived $T^0+T^2$: at most {t2_max:.4f} (not contoured)'),
        ]
        fig.legend(handles=handles, loc='lower center', ncol=1, fontsize=TEXT_SIZE,
                   frameon=False, handlelength=2.8, bbox_to_anchor=(.5, .006), labelspacing=.35)
        fig.subplots_adjust(left=.13, right=.95, bottom=.27, top=.87)
        save_figure(fig, RESULTS, 'adiabatic-speed-map')
        plt.close(fig)
    draw()
    summary = dict(eq23_levels={f'{k:g}': dict(a_0plus=level_compositions[k], F=level_F[k])
                                for k in level_F},
                   sommerfeld_T2_max_ratio=t2_max,
                   grid_shape=[len(temperatures), len(densities)],
                   eos_failures=payload['metadata']['eos_failures'])
    (RESULTS / 'adiabatic-speed-map-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plot', action='store_true', help='plot from the saved grid only')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    payload = (json.loads(GRID_FILE.read_text()) if args.plot
               else compute_grid(args.workers))
    print(json.dumps(make_plot(payload), indent=2))


if __name__ == '__main__':
    main()
