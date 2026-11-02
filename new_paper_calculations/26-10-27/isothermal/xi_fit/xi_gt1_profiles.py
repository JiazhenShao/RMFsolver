"""Why the isothermal speed-fit xi exceeds 1: the diffusive flux g = -D_K a' of the saved high-T profiles.

For every accepted high-T point, xi_speed is the xi at which the analytical speed equals the numerical one
(_iso_xi.effective_xi), and xi_profile = 2 I1/(G a(0+)) - 1 is the profile's own value, from I1 on the
numerical profile and G = lambda u(0-) [1 - a(0+)] with lambda = nB(0-)/nB(0+) (_iso_evaluate.py).
A monotone g(a) rising from 0 to G gives -1 < xi_profile < 1. Two figures:
1. xi_speed and xi_profile versus T;
2. g/G versus a/a(0+) for points with xi_speed > 1: (a) xi_profile in range, (b) xi_profile > 1.

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

PANEL = HERE.parent / "highT"
OUT = common.ROOT / "local_analysis_results/26-11-01/isothermal-xi-report"

# Tunable plotting parameters
TEXT_FONT_SIZE = 22            # manuscript standard at 7.2 in
LABEL_FONT_SIZE = 24
SCATTER_FIGURE_SIZE = (7.2, 5.6)
PROFILE_FIGURE_SIZE = (7.2, 11.0)
SCATTER_MARKER_SIZE = 10
SCATTER_ALPHA = 0.55
SPEED_COLOR = "#c62828"
PROFILE_COLOR = "#1f4e79"
SCATTER_Y_LIMITS = (-1.0, 4.5)
PROFILE_Y_LIMITS = ((0.0, 1.25), (0.0, 2.3))
SHAPE_LIMIT_FILL_COLOR = "0.90"
REFERENCE_COLOR = "0.35"
PROFILE_LINE_WIDTH = 2.6
PROFILE_COLORS = ("#1f77b4", "#2ca02c", "#d62728")
TARGET_T_IN_RANGE = (50.0, 60.0, 70.0)      # panel (a): xi_speed > 1, |xi_profile| < 0.5
TARGET_T_ABOVE_ONE = (76.0, 84.0, 92.0)     # panel (b): xi_profile > 1
TARGET_WINDOW_MEV = 3.0
EXPORT_DPI = 220


def points():
    numerical, analytic = (common.load(PANEL / f"isothermal-highT-{m}.npy") for m in ("numerical", "analytic"))
    xi_speed = iso_xi.effective_xi(iso_xi.build_cache(analytic), numerical)
    out = []
    for (i, j), record in np.ndenumerate(numerical["records"]):
        if record and record.get("accepted") and np.isfinite(xi_speed[i, j]):
            out.append(dict(index=(i, j), T=record["T"], a0=record["a_0plus"], speed=numerical["speed_m_s"][i, j],
                            xi_speed=float(xi_speed[i, j]), xi_profile=record["xi_profile"], record=record))
    return out


def flux(point):
    """a and g/G along the profile, g = -D_K a' = (J_K - jB a)/nB, G = g(0+) from the jump condition."""
    profile = np.load(PANEL / "isothermal-highT-numerical-profiles" / "{}_{}.npz".format(*point["index"]))
    record = point["record"]
    g = (profile["jK"] - record["jB"] * profile["a"]) / profile["nB"]
    G = record["nB_0minus"] / record["nB_0plus"] * record["u_0minus"] * (1.0 - record["a_0plus"])
    return profile["a"], g / G


def pick(candidates, targets):
    chosen = []
    for target in targets:
        near = [p for p in candidates if abs(p["T"] - target) <= TARGET_WINDOW_MEV]
        if near:
            middle = np.median([p["xi_speed"] for p in near])
            chosen.append(min(near, key=lambda p: abs(p["xi_speed"] - middle)))
    return chosen


def scatter_figure(all_points):
    T = np.array([p["T"] for p in all_points])
    with plt.rc_context({"mathtext.fontset": "cm"}):
        figure, axis = plt.subplots(figsize=SCATTER_FIGURE_SIZE, dpi=160, constrained_layout=True)
        axis.axhspan(1.0, SCATTER_Y_LIMITS[1], color=SHAPE_LIMIT_FILL_COLOR, zorder=0, linewidth=0)
        axis.scatter(T, [p["xi_speed"] for p in all_points], s=SCATTER_MARKER_SIZE, color=SPEED_COLOR,
                     alpha=SCATTER_ALPHA, linewidths=0, label=r"$\xi_{\rm speed}$ (analytical = numerical)")
        axis.scatter(T, [p["xi_profile"] for p in all_points], s=SCATTER_MARKER_SIZE, color=PROFILE_COLOR,
                     alpha=SCATTER_ALPHA, linewidths=0, label=r"$\xi_{\rm profile}$ (from $I_1$)")
        axis.set(xlim=(T.min() - 2, 96), ylim=SCATTER_Y_LIMITS)
        axis.set_xlabel("Nuclear phase temperature [MeV]", fontsize=LABEL_FONT_SIZE)
        axis.set_ylabel(r"$\xi$", fontsize=LABEL_FONT_SIZE)
        axis.legend(loc="upper left", fontsize=TEXT_FONT_SIZE, frameon=False, markerscale=3, handletextpad=0.3,
                    borderaxespad=0.2)
        axis.tick_params(direction="in", top=True, right=True, labelsize=TEXT_FONT_SIZE)
    return figure


def profile_figure(all_points):
    groups = (pick([p for p in all_points if p["xi_speed"] > 1 and abs(p["xi_profile"]) < 0.5], TARGET_T_IN_RANGE),
              pick([p for p in all_points if p["xi_profile"] > 1], TARGET_T_ABOVE_ONE))
    with plt.rc_context({"mathtext.fontset": "cm"}):
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=PROFILE_FIGURE_SIZE, dpi=160, constrained_layout=True)
        for axis, group, limits in zip(axes, groups, PROFILE_Y_LIMITS):
            axis.axhspan(1.0, limits[1], color=SHAPE_LIMIT_FILL_COLOR, zorder=0, linewidth=0)
            axis.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linestyle="--", linewidth=1.4, zorder=1)
            for point, color in zip(group, PROFILE_COLORS):
                a, g = flux(point)
                axis.plot(a / point["a0"], g, color=color, linewidth=PROFILE_LINE_WIDTH, zorder=3,
                          label=(f"{point['T']:.0f} MeV, $a(0^+)={point['a0']:.3f}$\n"
                                 f"$\\xi_{{\\rm speed}}={point['xi_speed']:+.2f}$, "
                                 f"$\\xi_{{\\rm profile}}={point['xi_profile']:+.2f}$"))
            axis.set_ylim(*limits)
            axis.set_ylabel(r"$-D_K a'/G$", fontsize=LABEL_FONT_SIZE)
            axis.legend(loc="upper left", fontsize=TEXT_FONT_SIZE - 4, frameon=False, handlelength=1.2,
                        labelspacing=0.5, borderaxespad=0.2)
            axis.tick_params(direction="in", top=True, right=True, labelsize=TEXT_FONT_SIZE)
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_xlabel(r"$a/a(0^+)$", fontsize=LABEL_FONT_SIZE)
    return figure, groups


def main():
    all_points = points()
    above = [p for p in all_points if p["xi_profile"] > 1]
    print(f"{len(all_points)} accepted high-T points; xi_speed > 1 at {sum(p['xi_speed'] > 1 for p in all_points)}, "
          f"xi_profile > 1 at {len(above)} (T {min(p['T'] for p in above):.1f}-{max(p['T'] for p in above):.1f} MeV, "
          f"a(0+) >= {min(p['a0'] for p in above):.4f})")
    OUT.mkdir(parents=True, exist_ok=True)
    figure = scatter_figure(all_points)
    figure.savefig(OUT / "isothermal-xi-speed-vs-profile.png", dpi=EXPORT_DPI, facecolor="white")
    figure.savefig(OUT / "isothermal-xi-speed-vs-profile.pdf", facecolor="white")
    figure, groups = profile_figure(all_points)
    figure.savefig(OUT / "isothermal-xi-gt1-profiles.png", dpi=EXPORT_DPI, facecolor="white")
    figure.savefig(OUT / "isothermal-xi-gt1-profiles.pdf", facecolor="white")
    for name, group in zip(("xi_profile in range", "xi_profile > 1"), groups):
        for p in group:
            a, g = flux(p)
            print(f"{name}: point {p['index']} T {p['T']:.1f} MeV a0 {p['a0']:.4f} u {p['speed']:.4g} m/s "
                  f"xi_speed {p['xi_speed']:+.2f} xi_profile {p['xi_profile']:+.2f} max g/G {g.max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
