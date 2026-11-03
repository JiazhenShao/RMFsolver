"""Why the isothermal fitted xi exceeds 1: the diffusive flux g = -D_K a' of the saved high-T profiles.

For every accepted high-T point, xi_fitted is the xi at which the analytical speed equals the numerical one
(_iso_xi.effective_xi), and xi_actual = 2 I1/(G a(0+)) - 1 is the value the solved profile actually has, with
the manuscript's definitions taken literally on the saved profile: g = -D_K a' (a and D_K as saved, a' by
finite differences), I1 = int D_K a'^2 dx and G = g(0+). A monotone g(a) rising from 0 to G gives -1 < xi_actual < 1.
The solver's diffusive current is D_K nK' (J_K = u nK - D_K nK'), so g differs from (J_K - jB a)/nB, the flux
behind the record field xi_profile (legacy name), wherever nB varies; near a(0+) -> 1 the two differ by up to 0.6.
Three figures:
1. xi_fitted and xi_actual versus T;
2. g/G versus a/a(0+), the parametric curve (a(x), g(x)) for 0+ <= x < inf, at points with xi_fitted > 1:
   (a) xi_actual near 0, (b) a(0+) -> 1, where g peaks inside the layer;
3. a, a', n_B, mu_B and mu_K versus x [nm] for the hottest point of 2(b).

    PYTHONPATH=local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:. \
    ./bin/python3 -B new_paper_calculations/26-10-27/isothermal/xi_fit/xi_gt1_profiles.py
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _iso_common as common  # noqa: E402
import _iso_xi as iso_xi  # noqa: E402
from RMFsolver.constants import MeV_fm  # noqa: E402

PANEL = HERE.parent / "highT"
OUT = common.ROOT / "local_analysis_results/26-11-01/isothermal-xi-report"

# Tunable plotting parameters
TEXT_FONT_SIZE = 22            # manuscript standard at 7.2 in
LABEL_FONT_SIZE = 24
SCATTER_FIGURE_SIZE = (7.2, 5.6)
PROFILE_FIGURE_SIZE = (7.2, 11.0)
SCATTER_MARKER_SIZE = 10
SCATTER_ALPHA = 0.55
FITTED_COLOR = "#c62828"
ACTUAL_COLOR = "#1f4e79"
SCATTER_Y_LIMITS = (-1.0, 4.5)
PROFILE_Y_LIMITS = ((0.0, 1.25), (-0.1, 2.3))   # bottom: legend above the peaks, 0 tick clear of the corner
SHAPE_LIMIT_FILL_COLOR = "0.90"
REFERENCE_COLOR = "0.35"
PROFILE_LINE_WIDTH = 2.6
PROFILE_COLORS = ("#1f77b4", "#2ca02c", "#d62728")
TARGET_T_IN_RANGE = (50.0, 60.0, 70.0)      # panel (a): xi_fitted > 1, |xi_actual| < 0.5
TARGET_T_PEAKED = (76.0, 84.0, 92.0)        # panel (b): xi_fitted > 1, a(0+) >= PEAKED_A0, g peaks inside
PEAKED_A0 = 0.99
TARGET_WINDOW_MEV = 3.0
FIELDS_FIGURE_SIZE = (7.2, 17.0)
FIELDS_X_LIMITS_NM = (0.0, 3000.0)          # a has fallen to ~a(0+)/100 at 3000 nm (91 MeV)
FIELDS_COLOR = PROFILE_COLORS[-1]           # the hottest curve's color in the flux figure
STEEPEST_MARKER_SIZE = 9
EXPORT_DPI = 220


def flux(index):
    """The saved profile as the curve (a, g/G), g = -D_K a', G = g(0+), and its xi_actual."""
    profile = np.load(PANEL / "isothermal-highT-numerical-profiles" / "{}_{}.npz".format(*index))
    x, a, D_K = profile["x"], profile["a"], profile["D_K"]
    slope = np.gradient(a, x)
    g = -D_K * slope
    xi_actual = 2.0 * np.trapezoid(D_K * slope**2, x) / (g[0] * a[0]) - 1.0
    return a, g / g[0], float(xi_actual)


def points():
    numerical, analytic = (common.load(PANEL / f"isothermal-highT-{m}.npy") for m in ("numerical", "analytic"))
    xi_fitted = iso_xi.effective_xi(iso_xi.build_cache(analytic), numerical)
    out = []
    for (i, j), record in np.ndenumerate(numerical["records"]):
        if record and record.get("accepted") and np.isfinite(xi_fitted[i, j]):
            _, g, xi_actual = flux((i, j))
            out.append(dict(index=(i, j), T=record["T"], a0=record["a_0plus"], speed=numerical["speed_m_s"][i, j],
                            xi_fitted=float(xi_fitted[i, j]), xi_actual=xi_actual, peak=float(g.max())))
    return out


def pick(candidates, targets):
    chosen = []
    for target in targets:
        near = [p for p in candidates if abs(p["T"] - target) <= TARGET_WINDOW_MEV]
        if near:
            middle = np.median([p["xi_fitted"] for p in near])
            chosen.append(min(near, key=lambda p: abs(p["xi_fitted"] - middle)))
    return chosen


def scatter_figure(all_points):
    T = np.array([p["T"] for p in all_points])
    with plt.rc_context({"mathtext.fontset": "cm"}):
        figure, axis = plt.subplots(figsize=SCATTER_FIGURE_SIZE, dpi=160, constrained_layout=True)
        axis.axhspan(1.0, SCATTER_Y_LIMITS[1], color=SHAPE_LIMIT_FILL_COLOR, zorder=0, linewidth=0)
        axis.scatter(T, [p["xi_fitted"] for p in all_points], s=SCATTER_MARKER_SIZE, color=FITTED_COLOR,
                     alpha=SCATTER_ALPHA, linewidths=0, label=r"$\xi_{\rm fitted}$ (analytical = numerical)")
        axis.scatter(T, [p["xi_actual"] for p in all_points], s=SCATTER_MARKER_SIZE, color=ACTUAL_COLOR,
                     alpha=SCATTER_ALPHA, linewidths=0, label=r"$\xi_{\rm actual}$ (from $I_1$)")
        axis.set(xlim=(T.min() - 2, 96), ylim=SCATTER_Y_LIMITS)
        axis.set_xlabel("Nuclear phase temperature [MeV]", fontsize=LABEL_FONT_SIZE)
        axis.set_ylabel(r"$\xi$", fontsize=LABEL_FONT_SIZE)
        axis.legend(loc="upper left", fontsize=TEXT_FONT_SIZE, frameon=False, markerscale=3, handletextpad=0.3,
                    borderaxespad=0.2)
        axis.tick_params(direction="in", top=True, right=True, labelsize=TEXT_FONT_SIZE)
    return figure


def profile_figure(all_points):
    groups = (pick([p for p in all_points if p["xi_fitted"] > 1 and abs(p["xi_actual"]) < 0.5], TARGET_T_IN_RANGE),
              pick([p for p in all_points if p["xi_fitted"] > 1 and p["a0"] >= PEAKED_A0 and p["peak"] > 1.001],
                   TARGET_T_PEAKED))
    with plt.rc_context({"mathtext.fontset": "cm"}):
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=PROFILE_FIGURE_SIZE, dpi=160, constrained_layout=True)
        for axis, group, limits, peaked in zip(axes, groups, PROFILE_Y_LIMITS, (False, True)):
            axis.axhspan(1.0, limits[1], color=SHAPE_LIMIT_FILL_COLOR, zorder=0, linewidth=0)
            axis.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linestyle="--", linewidth=1.4, zorder=1)
            for point, color in zip(group, PROFILE_COLORS):
                a, g, _ = flux(point["index"])
                # the peaked panel shares a(0+) ~ 1, so one line per curve keeps the legend above the peaks
                head = f"{point['T']:.0f} MeV: " if peaked else f"{point['T']:.0f} MeV, $a(0^+)={point['a0']:.3f}$\n"
                axis.plot(a / point["a0"], g, color=color, linewidth=PROFILE_LINE_WIDTH, zorder=3,
                          label=(head + f"$\\xi_{{\\rm fitted}}={point['xi_fitted']:+.2f}$, "
                                 f"$\\xi_{{\\rm actual}}={point['xi_actual']:+.2f}$"))
            axis.set_ylim(*limits)
            axis.set_ylabel(r"$-D_K a'\,/\,G$", fontsize=LABEL_FONT_SIZE)
            title = f"$a(0^+)\\geq{min(p['a0'] for p in group):.3f}$" if peaked else None
            axis.legend(loc="upper left", fontsize=TEXT_FONT_SIZE - 4, title=title, title_fontsize=TEXT_FONT_SIZE - 4,
                        alignment="left", frameon=False, handlelength=1.2, labelspacing=0.5, borderaxespad=0.2)
            axis.tick_params(direction="in", top=True, right=True, labelsize=TEXT_FONT_SIZE)
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_xlabel(r"$a/a(0^+)$", fontsize=LABEL_FONT_SIZE)
    return figure, groups


def fields_figure(point):
    """a, a', n_B/n_0, mu_B, mu_K versus x [nm] for one profile; a dot marks the steepest point of a."""
    profile = np.load(PANEL / "isothermal-highT-numerical-profiles" / "{}_{}.npz".format(*point["index"]))
    x = profile["x"] * MeV_fm * 1e-6                       # MeV^-1 -> nm
    slope = np.gradient(profile["a"], x) * 1e3
    steepest = np.argmin(slope)
    fields = (profile["a"], slope, profile["nB"] / common.N0_MEV3, profile["muB"], profile["muK"])
    labels = (r"$a$", r"$a'$ [$10^{-3}\,$nm$^{-1}$]", r"$n_B/n_0$", r"$\mu_B$ [MeV]", r"$\mu_K$ [MeV]")
    with plt.rc_context({"mathtext.fontset": "cm"}):
        figure, axes = plt.subplots(5, 1, sharex=True, figsize=FIELDS_FIGURE_SIZE, dpi=160, constrained_layout=True)
        for axis, field, label in zip(axes, fields, labels):
            axis.plot(x, field, color=FIELDS_COLOR, linewidth=PROFILE_LINE_WIDTH)
            axis.set_ylabel(label, fontsize=LABEL_FONT_SIZE)
            axis.tick_params(direction="in", top=True, right=True, labelsize=TEXT_FONT_SIZE)
        axes[1].plot(x[steepest], slope[steepest], "o", color=FIELDS_COLOR, markersize=STEEPEST_MARKER_SIZE)
        axes[0].set_title(f"$T={point['T']:.0f}$ MeV, $a(0^+)={point['a0']:.4f}$", fontsize=LABEL_FONT_SIZE)
        axes[-1].set_xlim(*FIELDS_X_LIMITS_NM)
        axes[-1].set_xlabel(r"$x$ [nm]", fontsize=LABEL_FONT_SIZE)
    print(f"fields: T {point['T']:.1f} MeV, steepest a at {x[steepest]:.0f} nm ({slope[steepest] / slope[0]:.3f} "
          f"x its 0+ value), D_K x{profile['D_K'][-1] / profile['D_K'][0]:.3f}, nB x{profile['nB'][-1] / profile['nB'][0]:.3f}, "
          f"muB {profile['muB'][0]:.0f} -> {profile['muB'][-1]:.0f} MeV, muK {profile['muK'][0]:.0f} -> 0 MeV")
    return figure


def main():
    all_points = points()
    above = [p for p in all_points if p["xi_actual"] > 1]
    print(f"{len(all_points)} accepted high-T points; xi_fitted > 1 at {sum(p['xi_fitted'] > 1 for p in all_points)}, "
          f"g peaks inside the layer at {sum(p['peak'] > 1.001 for p in all_points)}, xi_actual > 1 at {len(above)}"
          + (f" (T {min(p['T'] for p in above):.1f}-{max(p['T'] for p in above):.1f} MeV, "
             f"max {max(p['xi_actual'] for p in above):+.2f})" if above else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    figure = scatter_figure(all_points)
    figure.savefig(OUT / "isothermal-xi-fitted-vs-actual.png", dpi=EXPORT_DPI, facecolor="white")
    figure.savefig(OUT / "isothermal-xi-fitted-vs-actual.pdf", facecolor="white")
    figure, groups = profile_figure(all_points)
    figure.savefig(OUT / "isothermal-xi-gt1-profiles.png", dpi=EXPORT_DPI, facecolor="white")
    figure.savefig(OUT / "isothermal-xi-gt1-profiles.pdf", facecolor="white")
    figure = fields_figure(groups[1][-1])
    figure.savefig(OUT / "isothermal-xi-gt1-profile-fields.png", dpi=EXPORT_DPI, facecolor="white")
    figure.savefig(OUT / "isothermal-xi-gt1-profile-fields.pdf", facecolor="white")
    for name, group in zip(("xi_actual near 0", "a(0+) -> 1"), groups):
        for p in group:
            print(f"{name}: point {p['index']} T {p['T']:.1f} MeV a0 {p['a0']:.4f} u {p['speed']:.4g} m/s "
                  f"xi_fitted {p['xi_fitted']:+.2f} xi_actual {p['xi_actual']:+.2f} max g/G {p['peak']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
