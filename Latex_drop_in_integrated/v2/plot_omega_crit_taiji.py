"""Schematic of the critical-pressure region Omega_crit (two-lobed taiji shape).

Produces omega_crit_taiji.pdf/.png for the GW drop-in section.

Pre-existing-core picture. Equatorial cut of the early merger remnant. The
outer circle is the p = p_crit contour bounding Omega_crit. The m=2 double-core
deformation splits Omega_crit into two congruent lobes (taiji / yin-yang), each
of effective radius R_lobe = 3 km. Each lobe already contains a quark core of
radius R_0 = 1 km, inherited from the pre-merger star; the two tones only mark
the two lobes, which are otherwise the same metastable nuclear phase (P>p_crit).
Over the emission window the front advances the core boundary by only
Delta R = u(0^-)*Delta_t_pm ~ 10 m, about 1% of R_0, so the mantle stays nuclear.

All text >= 11 pt at figsize width 3.4 in (= \\columnwidth), so it stays >= the
10 pt body text when the PDF is inserted at width=\\columnwidth.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyArrowPatch

# Tunable plotting parameters
FIGURE_SIZE = (3.4, 3.8)
DPI = 300
FONT_FAMILY = "STIXGeneral"
MATH_FONTSET = "stix"
LABEL_FONT_SIZE = 11
ANNOT_FONT_SIZE = 11
R = 1.0                                 # taiji (Omega_crit equatorial) radius, schematic
R0_FRAC = 1.0 / 3.0                     # R_0 / R_lobe = 1 km / 3 km
RLOBE_FIG = 0.5 * R                     # visual lobe radius (= head circle radius)
R0_FIG = R0_FRAC * RLOBE_FIG            # visual pre-existing-core radius
FRONT_EXAG = 1.18                       # exaggeration of the ~1% front growth ring
LOBE_LIGHT = "#BBD4E8"
LOBE_DARK = "#5B8CB8"
OUTLINE_COLOR = "#1F2D3A"
OUTLINE_WIDTH = 1.8
CORE_COLOR = "#D55E00"                  # pre-existing quark core
CORE_EDGE = "#8A3D00"
FRONT_COLOR = "#8A3D00"
RLOBE_RING_COLOR = "#1F2D3A"
SEED_FACE = "#FFFFFF"
SEED_EDGE = "#1F2D3A"
SEED_SIZE = 85
SEED_TOP = (0.0, 0.5 * R)
SEED_BOT = (0.0, -0.5 * R)
ROT_COLOR = "0.35"
ROT_WIDTH = 1.5
DIM_COLOR = "#1F2D3A"
DIM_WIDTH = 1.3
LEADER_COLOR = "0.4"
LEADER_WIDTH = 0.8
AXIS_LIMIT = 1.72
OUTPUT_DIRS = ["Plots", "../../Latex_writting/Hydrodynamical combustion/Plots"]
OUTPUT_NAME = "omega_crit_taiji"

plt.rcParams.update({"font.family": FONT_FAMILY, "mathtext.fontset": MATH_FONTSET})

fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)

# Taiji: base disk light, left half dark, bottom small disk dark, top small disk light.
ax.add_patch(Circle((0, 0), R, facecolor=LOBE_LIGHT, edgecolor="none", zorder=1))
ax.add_patch(Wedge((0, 0), R, 90, 270, facecolor=LOBE_DARK, edgecolor="none", zorder=2))
ax.add_patch(Circle((0, -0.5 * R), 0.5 * R, facecolor=LOBE_DARK, edgecolor="none", zorder=3))
ax.add_patch(Circle((0, 0.5 * R), 0.5 * R, facecolor=LOBE_LIGHT, edgecolor="none", zorder=3))
theta = np.linspace(0, np.pi, 100)
ax.plot(0.5 * R * np.sin(theta), 0.5 * R + 0.5 * R * np.cos(theta),
        color=OUTLINE_COLOR, lw=1.0, zorder=4)
ax.plot(-0.5 * R * np.sin(theta), -0.5 * R - 0.5 * R * np.cos(theta),
        color=OUTLINE_COLOR, lw=1.0, zorder=4)
ax.add_patch(Circle((0, 0), R, facecolor="none", edgecolor=OUTLINE_COLOR,
                    lw=OUTLINE_WIDTH, zorder=5))

# R_lobe indicator ring around the bottom lobe
ax.add_patch(Circle(SEED_BOT, RLOBE_FIG, facecolor="none", edgecolor=RLOBE_RING_COLOR,
                    lw=DIM_WIDTH, ls=(0, (5, 3)), zorder=5))

# Exaggerated front-growth ring around the top core
ax.add_patch(Circle(SEED_TOP, R0_FIG * FRONT_EXAG, facecolor="none", edgecolor=FRONT_COLOR,
                    lw=1.1, ls=(0, (2, 2)), zorder=6))

# Pre-existing quark cores
for c in (SEED_TOP, SEED_BOT):
    ax.add_patch(Circle(c, R0_FIG, facecolor=CORE_COLOR, edgecolor=CORE_EDGE, lw=1.2, zorder=7))
    ax.scatter(*c, marker="*", s=SEED_SIZE, facecolor=SEED_FACE, edgecolor=SEED_EDGE,
               linewidth=0.9, zorder=8)

# R_0 dimension arrow on the top core
ax.add_patch(FancyArrowPatch(SEED_TOP, (R0_FIG, SEED_TOP[1]), arrowstyle="<->",
                             mutation_scale=8, lw=DIM_WIDTH, color=DIM_COLOR, zorder=9))
ax.text(R0_FIG * 0.5, SEED_TOP[1] + 0.055, r"$R_0$", fontsize=ANNOT_FONT_SIZE,
        color=DIM_COLOR, ha="center", va="bottom")
# R_lobe dimension arrow on the bottom lobe
ax.add_patch(FancyArrowPatch(SEED_BOT, (RLOBE_FIG, SEED_BOT[1]), arrowstyle="<->",
                             mutation_scale=8, lw=DIM_WIDTH, color=DIM_COLOR, zorder=9))
ax.text(RLOBE_FIG * 0.58, SEED_BOT[1] - 0.09, r"$R_{\rm lobe}$", fontsize=ANNOT_FONT_SIZE,
        color=DIM_COLOR, ha="center", va="top")

# Corotation arrow
ax.add_patch(FancyArrowPatch(
    (1.30 * R * np.cos(np.deg2rad(62)), 1.30 * R * np.sin(np.deg2rad(62))),
    (1.30 * R * np.cos(np.deg2rad(118)), 1.30 * R * np.sin(np.deg2rad(118))),
    connectionstyle="arc3,rad=0.32", arrowstyle="-|>", mutation_scale=12,
    lw=ROT_WIDTH, color=ROT_COLOR, zorder=6))
ax.text(0, 1.52 * R, r"$\Omega$ (corotating)", fontsize=ANNOT_FONT_SIZE,
        color=ROT_COLOR, ha="center", va="center")

# Labels with leader lines
ax.text(-1.66 * R, 1.28 * R, r"$\Omega_{\rm crit}$: metastable" + "\n" + r"nuclear matter",
        fontsize=LABEL_FONT_SIZE, color=OUTLINE_COLOR, ha="left", va="center")
ax.annotate(r"$P=p_{\rm crit}$", xy=(R * np.cos(np.deg2rad(-32)), R * np.sin(np.deg2rad(-32))),
            xytext=(1.06 * R, 0.70 * R), fontsize=ANNOT_FONT_SIZE, color=OUTLINE_COLOR,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=LEADER_COLOR, lw=LEADER_WIDTH))
ax.annotate("pre-existing\nquark core", xy=SEED_TOP,
            xytext=(0.62 * R, 0.95 * R), fontsize=ANNOT_FONT_SIZE, color=CORE_EDGE,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=LEADER_COLOR, lw=LEADER_WIDTH))
ax.annotate(r"front: $+\Delta R\!\approx\!10$ m", xy=(R0_FIG * FRONT_EXAG * 0.72, SEED_TOP[1] + R0_FIG * FRONT_EXAG * 0.72),
            xytext=(0.70 * R, 0.42 * R), fontsize=ANNOT_FONT_SIZE, color=FRONT_COLOR,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=LEADER_COLOR, lw=LEADER_WIDTH))
ax.text(-1.66 * R, -1.28 * R, r"$R_0=1$ km,  $R_{\rm lobe}=3$ km",
        fontsize=ANNOT_FONT_SIZE, color=DIM_COLOR, ha="left", va="center")

ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
ax.set_aspect("equal")
ax.axis("off")

fig.tight_layout(pad=0.2)
for d in OUTPUT_DIRS:
    fig.savefig(f"{d}/{OUTPUT_NAME}.pdf"); fig.savefig(f"{d}/{OUTPUT_NAME}.png")
print("wrote " + OUTPUT_NAME + " to " + ", ".join(OUTPUT_DIRS))
