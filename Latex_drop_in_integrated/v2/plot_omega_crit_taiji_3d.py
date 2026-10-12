"""3D schematic of the critical-pressure region Omega_crit.

Produces omega_crit_taiji_3d.pdf/.png for the GW drop-in section.

Omega_crit is a 3D volume, drawn here as an oblate body (rotating remnants are
flattened). The wireframe shell is its p = p_crit boundary. The equatorial cut
carries the m=2 double-core taiji (yin-yang) pattern: two congruent lobes, one
pressure maximum each. From each seed the converted (quark) region grows as a
genuine sphere of radius u(0^-)*Delta_t_pm (shaded, poking through the equator,
exaggerated in size). Both lobes are the same metastable nuclear phase; the two
tones only mark the two cores.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# Tunable plotting parameters
FIGURE_SIZE = (3.4, 3.3)
DPI = 300
FONT_FAMILY = "STIXGeneral"
MATH_FONTSET = "stix"
LABEL_FONT_SIZE = 11.5
ANNOT_FONT_SIZE = 11
VIEW_ELEV = 24
VIEW_AZIM = -58
R = 1.0                     # equatorial radius of Omega_crit
C = 0.5 * R                 # polar semi-axis (oblate: C < R)
LOBE_LIGHT = "#BBD4E8"
LOBE_DARK = "#5B8CB8"
SHELL_COLOR = "#4A6070"
SHELL_ALPHA = 0.5
SHELL_WIDTH = 0.8
QUARK_COLOR = "#D55E00"
SEED_COLOR = "#1F2D3A"
R_CONV = 0.5 * R / 3.0      # pre-existing quark-core radius, true scale:
                            # R_0/R_lobe = 1/3 with lobe radius 0.5*R
SEED_TOP = (0.0, 0.5 * R)
SEED_BOT = (0.0, -0.5 * R)
AXIS_COLOR = "0.35"
N_DISK_R = 60
N_DISK_PHI = 240
N_SPHERE = 40
N_SHELL = 22
OUTPUT_DIRS = ["Plots", "../../Latex_writting/Hydrodynamical combustion/Plots"]
OUTPUT_NAME = "omega_crit_taiji_3d"

plt.rcParams.update({"font.family": FONT_FAMILY, "mathtext.fontset": MATH_FONTSET})


def lobe_is_light(x, y):
    """Taiji rule: which lobe (True = light) a point in the equatorial plane is in."""
    top = x**2 + (y - 0.5 * R) ** 2 < (0.5 * R) ** 2
    bot = x**2 + (y + 0.5 * R) ** 2 < (0.5 * R) ** 2
    light = np.where(x >= 0, True, False)   # right half light, left half dark
    light = np.where(top, True, light)      # top small disk -> light
    light = np.where(bot, False, light)     # bottom small disk -> dark
    return light


fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
ax = fig.add_subplot(111, projection="3d")
ax.set_proj_type("ortho")
ax.computed_zorder = False   # honor manual zorder so domes draw over the disk

# Equatorial taiji disk at z = 0
rr = np.linspace(0, R, N_DISK_R)
pp = np.linspace(0, 2 * np.pi, N_DISK_PHI)
Rr, Pp = np.meshgrid(rr, pp)
Xd = Rr * np.cos(Pp)
Yd = Rr * np.sin(Pp)
Zd = np.zeros_like(Xd)
light = lobe_is_light(Xd, Yd)
facecolors = np.where(light[..., None],
                      np.array(to_rgba(LOBE_LIGHT)),
                      np.array(to_rgba(LOBE_DARK)))
ax.plot_surface(Xd, Yd, Zd, facecolors=facecolors, shade=False,
                rstride=1, cstride=1, linewidth=0, antialiased=True, zorder=2)

# Oblate p_crit shell as a sparse cage of rings (boundary without muddying interior)
th = np.linspace(0, np.pi, 120)
for phi0 in np.deg2rad([0, 45, 90, 135]):          # meridians
    ax.plot(R * np.sin(th) * np.cos(phi0), R * np.sin(th) * np.sin(phi0),
            C * np.cos(th), color=SHELL_COLOR, alpha=SHELL_ALPHA,
            lw=SHELL_WIDTH, zorder=1)
phc = np.linspace(0, 2 * np.pi, 160)
for th0 in np.deg2rad([40, 90, 140]):              # latitude rings
    ax.plot(R * np.sin(th0) * np.cos(phc), R * np.sin(th0) * np.sin(phc),
            C * np.cos(th0) * np.ones_like(phc), color=SHELL_COLOR,
            alpha=SHELL_ALPHA, lw=SHELL_WIDTH, zorder=1)

# Converted quark region: red equatorial slice + shaded upper dome. The dome
# sits entirely above z=0, so it never intersects the disk (avoids mplot3d's
# occlusion bug) yet reads as a sphere half-embedded in the equatorial cut.
uu = np.linspace(0, 2 * np.pi, N_SPHERE)
vv = np.linspace(0, np.pi / 2, N_SPHERE // 2)     # upper hemisphere only
nx = np.outer(np.cos(uu), np.sin(vv))
ny = np.outer(np.sin(uu), np.sin(vv))
nz = np.outer(np.ones_like(uu), np.cos(vv))
light = np.array([-0.5, -0.4, 0.77])
light = light / np.linalg.norm(light)
intensity = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0, 1)
shade = 0.45 + 0.55 * intensity
base = np.array(to_rgba(QUARK_COLOR))
dome_colors = np.empty(nx.shape + (4,))
dome_colors[..., :3] = base[:3] * shade[..., None]
dome_colors[..., 3] = 1.0
rs = np.linspace(0, R_CONV, 8)
ps = np.linspace(0, 2 * np.pi, N_SPHERE)
Rs, Ps = np.meshgrid(rs, ps)
Xsl, Ysl = Rs * np.cos(Ps), Rs * np.sin(Ps)
for (cx, cy) in (SEED_TOP, SEED_BOT):
    ax.plot_surface(Xsl + cx, Ysl + cy, np.full_like(Xsl, 0.02), color=QUARK_COLOR,
                    shade=False, rstride=1, cstride=1, linewidth=0, zorder=4)
    ax.plot_surface(R_CONV * nx + cx, R_CONV * ny + cy, R_CONV * nz,
                    facecolors=dome_colors, shade=False, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, zorder=6)
ax.scatter([SEED_TOP[0]], [SEED_TOP[1]], [R_CONV], color=SEED_COLOR, marker="*",
           s=70, depthshade=False, zorder=7)

# Rotation axis and Omega arrow
ax.plot([0, 0], [0, 0], [-1.35 * C, 1.5 * C], color=AXIS_COLOR, lw=1.2, zorder=4)
arc = np.linspace(-0.2 * np.pi, 1.05 * np.pi, 60)
ax.plot(0.42 * R * np.cos(arc), 0.42 * R * np.sin(arc),
        1.42 * C * np.ones_like(arc), color=AXIS_COLOR, lw=1.4, zorder=6)
ax.quiver(0.42 * R * np.cos(arc[-1]), 0.42 * R * np.sin(arc[-1]), 1.42 * C,
          -np.sin(arc[-1]), np.cos(arc[-1]), 0, length=0.28, color=AXIS_COLOR,
          arrow_length_ratio=0.6, lw=1.4, zorder=6)
ax.text(0, 0, 1.72 * C, r"$\Omega$", fontsize=LABEL_FONT_SIZE, color=AXIS_COLOR,
        ha="center", va="center")

# Labels
ax.text(-1.15 * R, 0.9 * R, 0.55 * C, r"$\Omega_{\rm crit}$", fontsize=LABEL_FONT_SIZE,
        color=SHELL_COLOR, ha="center")
ax.text(1.02 * R, -0.75 * R, 0.0, r"$P=p_{\rm crit}$", fontsize=ANNOT_FONT_SIZE,
        color=SHELL_COLOR, ha="left")
ax.text(SEED_BOT[0] + 0.15, SEED_BOT[1] - 0.05, -1.08 * C,
        "pre-existing quark core\n($R_0=1$ km, $R_{\\rm lobe}=3$ km)",
        fontsize=ANNOT_FONT_SIZE, color="#8A3D00", ha="center")
ax.text(SEED_TOP[0] - 0.1, SEED_TOP[1] + 0.34, 0.55 * C, "seed:\npressure max",
        fontsize=ANNOT_FONT_SIZE, color=SEED_COLOR, ha="center")

ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
ax.set_box_aspect((1, 1, 0.62))
ax.set_xlim(-1.15 * R, 1.15 * R)
ax.set_ylim(-1.15 * R, 1.15 * R)
ax.set_zlim(-1.2 * C, 1.2 * C)
ax.set_axis_off()

fig.tight_layout(pad=0.1)
for d in OUTPUT_DIRS:
    fig.savefig(f"{d}/{OUTPUT_NAME}.pdf", bbox_inches="tight")
    fig.savefig(f"{d}/{OUTPUT_NAME}.png", bbox_inches="tight")
print("wrote " + OUTPUT_NAME + " to " + ", ".join(OUTPUT_DIRS))
