"""Plot the exact simple EOS against digitized samples of the revised PI image."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Tunable plotting parameters
FIGURE_SIZE = (9.6, 8.8)
DPI = 180
TITLE_FONT_SIZE = 22
AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
ANNOTATION_FONT_SIZE = 16
LINE_WIDTH = 3.0
MARKER_SIZE = 8.0
MARKER_EDGE_WIDTH = 1.7
CURVE_COLOR = "#bc492b"
MARKER_COLOR = "#252525"
REFERENCE_COLOR = "#777777"
GRID_ALPHA = 0.20
REFERENCE_LINE_WIDTH = 1.5
LEGEND_LOCATION = "lower right"
ANNOTATION_POSITION = (0.045, 0.90)
X_LIMITS = (0, 500)
Y_LIMITS = (0, 1.53)
X_TICKS = np.arange(0, 501, 100)
Y_TICKS = np.arange(0, 1.41, 0.2)
OUTPUT_FORMATS = ("png", "pdf")

OUTPUT_DIR = Path(__file__).resolve().parent
SAMPLES_PATH = OUTPUT_DIR / "revised_simple_screenshot_samples.csv"
MU_B = 1200.0
MU_K_AT_A_ONE = 344.1277672015574

# Axis calibration measured from the revised 486 x 327 pixel screenshot.
X_ZERO, X_FIVE_HUNDRED = 28, 432
Y_ZERO, Y_ONE = 300, 104
SAMPLE_MU_K = (50, 100, 150, 200, 250, 300, 350, 400, 450, 495)


def simple_fraction(mu_k):
    mu_q = MU_B / 3
    return (1.5 * mu_q**2 * mu_k + mu_k**3 / 8) / (
        mu_q**3 + mu_q * mu_k**2 / 2
    )


def digitize(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    if rgb.shape[:2] != (327, 486):
        raise ValueError("Axis calibration requires the revised 486 x 327 screenshot")
    blue = ((rgb[:, :, 2] > rgb[:, :, 0] + 50)
            & (rgb[:, :, 1] > rgb[:, :, 0] + 35)
            & (rgb[:, :, 2] > rgb[:, :, 1] + 15))
    blue[:50, :] = False
    samples = []
    for target in SAMPLE_MU_K:
        x = round(X_ZERO + (X_FIVE_HUNDRED-X_ZERO)*target/500)
        ys = np.flatnonzero(blue[:, x])
        if not len(ys):
            raise ValueError(f"No curve pixels at x={x}")
        y = float(np.median(ys))
        mu_k = (x-X_ZERO)/(X_FIVE_HUNDRED-X_ZERO)*500
        fraction = (Y_ZERO-y)/(Y_ZERO-Y_ONE)
        samples.append(dict(muK=mu_k, a_screenshot=fraction, pixel_x=x, pixel_y=y))
    with SAMPLES_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path,
                        help="Digitize the revised image; otherwise reuse the saved CSV")
    args = parser.parse_args()
    if args.screenshot:
        digitize(args.screenshot)
    with SAMPLES_PATH.open() as stream:
        samples = list(csv.DictReader(stream))
    x = np.array([float(row["muK"]) for row in samples])
    y = np.array([float(row["a_screenshot"]) for row in samples])
    error = float(np.max(np.abs(y-simple_fraction(x))))
    if error > 0.01:
        raise ValueError(f"Screenshot agreement exceeds expected resolution: {error}")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    curve_x = np.linspace(*X_LIMITS, 1001)
    ax.plot(curve_x, simple_fraction(curve_x), color=CURVE_COLOR,
            linewidth=LINE_WIDTH, label="Exact massless EOS")
    ax.plot(x, y, "o", fillstyle="none", color=MARKER_COLOR,
            markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
            label="Revised PI screenshot (approx. digitized)")
    ax.axhline(1, color=REFERENCE_COLOR, linestyle="--",
               linewidth=REFERENCE_LINE_WIDTH)
    ax.plot(MU_K_AT_A_ONE, 1, "o", color=CURVE_COLOR, markersize=MARKER_SIZE)
    ax.set(xlim=X_LIMITS, ylim=Y_LIMITS, xticks=X_TICKS, yticks=Y_TICKS)
    ax.set_title(r"Simple: $\mu_d+\mu_s=2\mu_u$", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel(r"$\mu_K=\mu_d-\mu_s$ [MeV]", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(r"$a=n_K/n_B$", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(alpha=GRID_ALPHA)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(*ANNOTATION_POSITION, r"$a(500)=1.1897$; $a=1$ at $344.13$ MeV",
            transform=ax.transAxes, fontsize=ANNOTATION_FONT_SIZE)
    ax.legend(loc=LEGEND_LOCATION, fontsize=LEGEND_FONT_SIZE)
    for extension in OUTPUT_FORMATS:
        fig.savefig(OUTPUT_DIR / f"simple_revised_comparison.{extension}", dpi=DPI)
    plt.close(fig)
    print(f"Digitized {len(x)} points; maximum absolute difference in a: {error:.6f}")


if __name__ == "__main__":
    main()
