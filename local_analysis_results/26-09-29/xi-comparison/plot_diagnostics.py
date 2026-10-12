"""Visualize saved-profile diagnostics; no cluster files are changed."""
from analyze_xi import OUTPUT, load_samples, summaries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIGURE_SIZE = (11, 4.7)
PROFILE_FIGURE_SIZE = (6.5, 5)
DPI = 180
LABEL_SIZE = 13
TICK_SIZE = 11
LEGEND_SIZE = 10
LINE_WIDTH = 1.6
SCATTER_SIZE = 7
SCATTER_ALPHA = .12
GRID_ALPHA = .2
COLORS = {'semi_analytic': 'tab:orange', 'analytic': 'tab:blue', 'profile': 'black'}
SPEED_LIMITS = (30, 6500)
XI_LIMITS = (-1.3, 1.4)


if __name__ == '__main__':
    _, _, frame, profiles = load_samples()
    frame = frame[frame.matched_plot_grid]
    binned = summaries(frame)
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, dpi=DPI, layout='constrained')
    for method, label in [('semi_analytic', 'Semi-analytical'), ('analytic', 'Full analytical')]:
        color = COLORS[method]
        axes[0].scatter(frame.numerical_u_m_s, frame[method+'_ratio'], s=SCATTER_SIZE,
                        color=color, alpha=SCATTER_ALPHA, rasterized=True)
        axes[0].plot(binned.numerical_u_m_s, binned[method+'_ratio'], color=color,
                     lw=LINE_WIDTH, marker='o', label=label)
        axes[1].scatter(frame.numerical_u_m_s, frame[method+'_xi_fit_frozen'], s=SCATTER_SIZE,
                        color=color, alpha=SCATTER_ALPHA, rasterized=True)
        axes[1].plot(binned.numerical_u_m_s, binned[method+'_xi_fit_frozen'], color=color,
                     lw=LINE_WIDTH, marker='o', label=label+' fitted')
    axes[0].axhline(1, color='black', ls=':', lw=LINE_WIDTH)
    axes[0].set_ylabel(r'$u_{\rm approximation}(0^-)/u_{\rm numerical}(0^-)$', fontsize=LABEL_SIZE)
    axes[1].plot(binned.numerical_u_m_s, binned.xi_profile, color=COLORS['profile'],
                 ls='--', lw=LINE_WIDTH, label=r'$\xi$ from numerical profiles')
    axes[1].axhline(-.5, color='gray', ls=':', lw=LINE_WIDTH, label=r'Current $\xi=-0.5$')
    axes[1].axhline(-1, color='gray', ls='-.', lw=LINE_WIDTH, label=r'Profile lower bound $-1$')
    axes[1].set_ylim(XI_LIMITS)
    axes[1].set_ylabel(r'Effective $\xi$', fontsize=LABEL_SIZE)
    axes[0].set_title('Same upstream points, not contour offsets', fontsize=LABEL_SIZE)
    axes[1].set_title('Fitted value vs. measured profile shape', fontsize=LABEL_SIZE)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlim(SPEED_LIMITS)
        ax.set_xlabel(r'Numerical $u(0^-)$ [m s$^{-1}$]', fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE, direction='in', top=True, right=True)
        ax.grid(alpha=GRID_ALPHA)
        ax.legend(fontsize=LEGEND_SIZE)
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT/f'xi-speed-diagnostics.{ext}', dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=PROFILE_FIGURE_SIZE, dpi=DPI, layout='constrained')
    chosen = frame[(frame.theta_index == 15) & frame.radial_fraction.isin([.02, .1, .2, .4, .6, .78, .84])]
    for color, (_, row) in zip(plt.cm.viridis(np.linspace(0, .95, len(chosen))), chosen.iterrows()):
        p = profiles[row.key]
        g = (p['jK'] - p['u'] * p['nK']) / p['nB']
        ax.plot(p['a']/p['a'][0], g/g[0], color=color, lw=LINE_WIDTH,
                label=rf'{row.numerical_u_m_s:.0f} m/s, $\xi={row.xi_profile:.3f}$')
    ax.plot([0, 1], [0, 1], color='gray', ls=':', lw=LINE_WIDTH, label=r'Linear ramp: $\xi=0$')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_xlabel(r'$a/a(0^+)$', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$g(a)/g(a(0^+))$', fontsize=LABEL_SIZE)
    ax.set_title(r'Diffusive-flux shape at $\theta=46.55^\circ$', fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE, direction='in', top=True, right=True)
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper left')
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT/f'diffusive-flux-shapes.{ext}', dpi=DPI)
    plt.close(fig)
