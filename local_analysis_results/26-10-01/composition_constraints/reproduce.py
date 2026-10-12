"""Compare exact massless composition curves with the PI's screenshots.

Run from the project root using Python with numpy, scipy, matplotlib and Pillow.
All densities are net densities, including color, in natural MeV units.
"""

from pathlib import Path
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

# Tunable plotting parameters
FIGURE_SIZE = (12.0, 5.2)
EXTENDED_FIGURE_SIZE = (9.2, 5.6)
DPI = 180
TITLE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 12
TICK_LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
LINE_WIDTH = 2.3
MARKER_SIZE = 5.2
NEUTRAL_COLOR = "#1669a6"
SIMPLE_COLOR = "#bc492b"
REFERENCE_COLOR = "#777777"
SCREENSHOT_COLOR = "#252525"
GRID_ALPHA = 0.17
LEGEND_LOCATION = "lower right"
MU_K_VIEW_MAX = 500.0
MU_K_EXTENDED_MAX = 800.0
MU_K_TICK_STEP = 100.0
FRACTION_LIMITS = ((0.0, 1.06), (0.0, 1.53))
EXTENDED_FRACTION_LIMITS = (0.0, 1.43)
SUBPLOT_SPACING = 0.23
PANEL_LABEL_POSITIONS = ((0.045, 0.90), (0.045, 0.90))
PANEL_LABEL_FONT_SIZE = 10
ENDPOINT_TEXT_POSITIONS = ((355.0, 0.72), (470.0, 1.08))
OUTPUT_FORMATS = ("png", "pdf")

MU_B = 1200.0
TEMPERATURE = 0.0
GRID_POINTS = 1001
OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from RMFsolver import SQMsolver as sqm


def density(mu, temperature):
    return np.asarray(mu)**3 / np.pi**2 + np.asarray(mu) * temperature**2


def triplet(mu_k, mu_q_charge=0.0):
    return np.array([
        MU_B / 3 + 2 * mu_q_charge / 3,
        MU_B / 3 + mu_k / 2 - mu_q_charge / 3,
        MU_B / 3 - mu_k / 2 - mu_q_charge / 3,
    ])


def state(mu_k, temperature, neutral):
    def neutrality(mu_q_charge):
        nu, nd, ns = density(triplet(mu_k, mu_q_charge), temperature)
        return (2 * nu - nd - ns) / MU_B**3

    mu_q_charge = brentq(neutrality, -MU_B, MU_B, xtol=1e-11) if neutral else 0.0
    mu = triplet(mu_k, mu_q_charge)
    nu, nd, ns = density(mu, temperature)
    n_b = (nu + nd + ns) / 3
    n_k = (nd - ns) / 2
    return dict(muK=float(mu_k), muQ=float(mu_q_charge), a=float(n_k/n_b),
                nB=float(n_b), nK=float(n_k), mu_u=float(mu[0]),
                mu_d=float(mu[1]), mu_s=float(mu[2]),
                n_u=float(nu), n_d=float(nd), n_s=float(ns))


def neutral_mu_k_from_a(a):
    return MU_B * (np.cbrt(1+a)-np.cbrt(1-a)) / (
        1+np.cbrt(1+a)+np.cbrt(1-a)
    )


def simple_a(mu_k, temperature):
    mu_q = MU_B / 3
    return mu_k * (1.5*mu_q**2 + mu_k**2/8 + np.pi**2*temperature**2/2) / (
        mu_q**3 + mu_q*mu_k**2/2 + np.pi**2*mu_q*temperature**2
    )


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_axis(ax, xmax, ylim):
    ax.set(xlim=(0, xmax), ylim=ylim)
    ax.set_xlabel(r"$\mu_K=\mu_d-\mu_s$ [MeV]", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(r"$a=n_K/n_B$", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(np.arange(0, xmax+1, MU_K_TICK_STEP))
    ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    ax.axhline(1, color=REFERENCE_COLOR, linestyle="--", linewidth=1)
    ax.grid(alpha=GRID_ALPHA)
    ax.spines[["top", "right"]].set_visible(False)


def export(fig, basename):
    for extension in OUTPUT_FORMATS:
        fig.savefig(OUTPUT_DIR / f"{basename}.{extension}", dpi=DPI)
    plt.close(fig)


def main():
    neutral_endpoint = MU_B * np.cbrt(2)/(1+np.cbrt(2))
    simple_endpoint = brentq(lambda k: simple_a(k, 0)-1, 0, 2*MU_B/3)
    mu_k_grid = np.linspace(0, MU_K_EXTENDED_MAX, GRID_POINTS)
    rows = []
    max_live_error = 0.0
    for k in mu_k_grid:
        neutral = state(k, TEMPERATURE, True)
        simple = state(k, TEMPERATURE, False)
        assert abs(simple["a"] - simple_a(k, TEMPERATURE)) < 1e-12
        if k <= neutral_endpoint:
            assert abs(neutral_mu_k_from_a(neutral["a"])-k) < 1e-5
        rows.append(dict(muK=float(k), a_neutral=neutral["a"],
                         a_simple=simple["a"], muQ_neutral=neutral["muQ"],
                         mu_s_neutral=neutral["mu_s"], mu_s_simple=simple["mu_s"]))

    temperature_rows = []
    for temperature in (0.0, 0.5, 10.0, 50.0):
        for k in (0.0, 100.0, 200.0, 300.0, 400.0, 500.0, neutral_endpoint):
            neutral = state(k, temperature, True)
            live = sqm._quark_uds_state(MU_B, k, temperature, ms=0)
            err = abs(neutral["a"]-live["nK"]/live["nB"])
            max_live_error = max(max_live_error, err)
            assert err < 1e-10
            charge = 2*neutral["n_u"]-neutral["n_d"]-neutral["n_s"]
            assert abs(charge)/neutral["nB"] < 1e-11
            temperature_rows.append(dict(T=temperature, muK=k,
                a_neutral=neutral["a"], a_simple=simple_a(k, temperature)))

    save_csv(OUTPUT_DIR / "curves.csv", rows)
    save_csv(OUTPUT_DIR / "temperature_checks.csv", temperature_rows)
    with (OUTPUT_DIR / "screenshot_samples.csv").open() as stream:
        screenshot = list(csv.DictReader(stream))

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, layout="constrained")
    fig.get_layout_engine().set(wspace=SUBPLOT_SPACING)
    for i, (ax, name, key, color, title) in enumerate(zip(
        axes, ("neutral", "simple"), ("a_neutral", "a_simple"),
        (NEUTRAL_COLOR, SIMPLE_COLOR),
        (r"Neutral: $n_d+n_s=2n_u$", r"Simple: $\mu_d+\mu_s=2\mu_u$"),
    )):
        ax.plot(mu_k_grid, [r[key] for r in rows], color=color,
                linewidth=LINE_WIDTH, label="Exact massless EOS")
        samples = [r for r in screenshot if r["model"] == name]
        ax.plot([float(r["muK"]) for r in samples],
                [float(r["a_screenshot"]) for r in samples], "o", fillstyle="none",
                markersize=MARKER_SIZE, color=SCREENSHOT_COLOR,
                label="PI screenshot (approx. digitized)")
        format_axis(ax, MU_K_VIEW_MAX, FRACTION_LIMITS[i])
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        description = (
            r"$a(500)=0.9888$; $a=1$ at $669.01$ MeV" if i == 0 else
            r"$a(500)=1.1897$; $a=1$ at $344.13$ MeV"
        )
        ax.text(*PANEL_LABEL_POSITIONS[i], description, transform=ax.transAxes,
                fontsize=PANEL_LABEL_FONT_SIZE)
        if i == 1:
            ax.plot(simple_endpoint, 1, "o", color=color, markersize=MARKER_SIZE)
        ax.legend(loc=LEGEND_LOCATION, fontsize=LEGEND_FONT_SIZE)
    fig.suptitle(r"Local composition at $\mu_B=1200$ MeV, $T=0$, $m_u=m_d=m_s=0$",
                 fontsize=TITLE_FONT_SIZE+1)
    export(fig, "comparison_with_screenshots")

    fig, ax = plt.subplots(figsize=EXTENDED_FIGURE_SIZE, layout="constrained")
    physical_k = np.append(mu_k_grid[mu_k_grid < neutral_endpoint], neutral_endpoint)
    ax.plot(physical_k, [state(k, 0, True)["a"] for k in physical_k],
            color=NEUTRAL_COLOR, linewidth=LINE_WIDTH, label="Neutral, nonnegative net densities")
    ax.plot(mu_k_grid, [r["a_simple"] for r in rows], color=SIMPLE_COLOR,
            linewidth=LINE_WIDTH, label=r"$\mu_Q=0$")
    format_axis(ax, MU_K_EXTENDED_MAX, EXTENDED_FRACTION_LIMITS)
    for endpoint, color, text, position in zip(
        (simple_endpoint, neutral_endpoint), (SIMPLE_COLOR, NEUTRAL_COLOR),
        ("a=1 at 344.13 MeV\nstrange quarks remain", "a=1 at 669.01 MeV\nnet density ratio 1:2:0"),
        ENDPOINT_TEXT_POSITIONS,
    ):
        ax.plot(endpoint, 1, "o", color=color, markersize=MARKER_SIZE)
        ax.annotate(text, xy=(endpoint, 1), xytext=position,
                    arrowprops=dict(arrowstyle="->", color=color),
                    fontsize=PANEL_LABEL_FONT_SIZE, color=color)
    ax.set_title(r"The neutral $a=1$ endpoint is finite ($\mu_B=1200$ MeV, $T=0$)",
                 fontsize=TITLE_FONT_SIZE)
    ax.legend(loc=LEGEND_LOCATION, fontsize=LEGEND_FONT_SIZE)
    export(fig, "extended_endpoints")

    summary = dict(muB=MU_B, T=TEMPERATURE, ms=0.0,
        neutral_a1=state(neutral_endpoint, 0, True),
        simple_a1=state(simple_endpoint, 0, False),
        neutral_targets=[dict(a=a, muK=float(neutral_mu_k_from_a(a)))
                         for a in (0.5, 0.9, 0.99, 0.999, 1.0)],
        max_fraction_difference_from_SQMsolver=max_live_error,
        validation="Exact old-basis formula, inverse neutral formula, 28 live SQMsolver states, charge residuals")
    with (OUTPUT_DIR / "verification.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
