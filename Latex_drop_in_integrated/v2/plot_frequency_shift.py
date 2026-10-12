"""Suppression of the phase-transition frequency shift by finite front speed.

Produces frequency_shift_suppression.pdf for the GW drop-in section.

Physics (see the section text):
    X_q(u)          = min(1, (u*DT_PM/R_LOBE)^3)   converted fraction, two-lobed
                      (taiji) geometry, spherical growth from a seed at each
                      pressure maximum; R_LOBE = (3*V_crit/(8*pi))^(1/3)
    Df_rate/Df_PT   = 1 - X_q(u)                    suppression factor

How to extend this into the full paper pipeline:
  1. Replace the generic speed axis with the calculated maximum boundary speed
     u_max(n_B - n_B_crit, T) from the adaptive contour scan
     (run_hydro_analytic_contour.py); each merger scenario then maps to a point
     on these curves instead of a band.
  2. Multiply the suppression factor by a calibrated Df_PT(M_chirp) from
     instantaneous-conversion simulations (e.g. Bauswein:2018bma tables) to get
     the absolute shift Df_rate in Hz as a function of chirp mass.
  3. R_CRIT and DT_PM can be replaced by Omega_crit geometries and postmerger
     emission windows extracted from published simulations.
"""

import numpy as np
import matplotlib.pyplot as plt

# Tunable plotting parameters
FIGURE_SIZE = (3.4, 2.7)        # inches, single revtex column
DPI = 300
FONT_FAMILY = "STIXGeneral"
MATH_FONTSET = "stix"
AXIS_LABEL_FONT_SIZE = 9
TICK_LABEL_FONT_SIZE = 8
LEGEND_FONT_SIZE = 7.2
ANNOTATION_FONT_SIZE = 7.2
LINE_WIDTH = 1.6
CURVE_COLORS = ["#0072B2", "#D55E00", "#009E73"]   # validated CVD-safe triad
CURVE_STYLES = ["-", "--", "-."]                   # secondary encoding
BAND_COLOR = "0.55"
BAND_ALPHA = 0.30
GRID_COLOR = "0.85"
GRID_LINE_WIDTH = 0.5
X_LIMITS = (1.0e1, 1.0e7)       # front speed in m/s
Y_LIMITS = (-0.04, 1.09)
U_MAX_BAND = (1.0e2, 1.0e3)     # calculated weak-limited u(0^-) range, m/s
LEGEND_LOCATION = "lower left"
OUTPUT_NAME = "frequency_shift_suppression.pdf"

# Scenario parameters: (R_lobe [m], Delta t_pm [s], label)
SCENARIOS = [
    (0.7e3, 20.0e-3, r"$R_{\rm lobe}=0.7\,{\rm km},\ \Delta t_{\rm pm}=20\,{\rm ms}$"),
    (1.0e3, 10.0e-3, r"$R_{\rm lobe}=1\,{\rm km},\ \Delta t_{\rm pm}=10\,{\rm ms}$"),
    (2.0e3, 5.0e-3, r"$R_{\rm lobe}=2\,{\rm km},\ \Delta t_{\rm pm}=5\,{\rm ms}$"),
]

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "mathtext.fontset": MATH_FONTSET,
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_LABEL_FONT_SIZE,
    "ytick.labelsize": TICK_LABEL_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
})

u = np.logspace(np.log10(X_LIMITS[0]), np.log10(X_LIMITS[1]), 2000)

fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)

for (r_lobe, dt_pm, label), color, style in zip(SCENARIOS, CURVE_COLORS, CURVE_STYLES):
    X_q = np.minimum(1.0, (u * dt_pm / r_lobe) ** 3)
    ax.plot(u, 1.0 - X_q, style, color=color, lw=LINE_WIDTH, label=label)

ax.axvspan(*U_MAX_BAND, color=BAND_COLOR, alpha=BAND_ALPHA, lw=0)
ax.annotate(
    "calculated\n$u_{\\rm max}(0^-)$",
    xy=(np.sqrt(U_MAX_BAND[0] * U_MAX_BAND[1]), 0.50),
    ha="center", va="center", fontsize=ANNOTATION_FONT_SIZE, color="0.25",
)

ax.set_xscale("log")
ax.set_xlim(*X_LIMITS)
ax.set_ylim(*Y_LIMITS)
ax.set_xlabel(r"front speed $u(0^-)$  [m s$^{-1}$]")
ax.set_ylabel(r"$\Delta f_{\rm rate}/\Delta f_{\rm PT} = 1-X_q$")
ax.grid(True, which="major", color=GRID_COLOR, lw=GRID_LINE_WIDTH)
ax.legend(loc=LEGEND_LOCATION, frameon=False, handlelength=2.2)

fig.tight_layout(pad=0.3)
fig.savefig(OUTPUT_NAME)
print(f"wrote {OUTPUT_NAME}")
