#!/usr/bin/env python3
"""Visualize success, current, and velocity along the hidden-rho pilot."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PAYLOAD_PATH = SCRIPT_DIR / "T0plus0p1-rho0p9-hidden-rho-pilot.npy"
OUTPUT_PNG = SCRIPT_DIR / "T0plus0p1-rho0p9-hidden-rho-pilot-diagnostic.png"
OUTPUT_PDF = SCRIPT_DIR / "T0plus0p1-rho0p9-hidden-rho-pilot-diagnostic.pdf"

# Tunable plotting parameters
FIGURE_SIZE = (18.0, 6.2)
FIGURE_DPI = 150
EXPORT_DPI = 260
FIGURE_FACE_COLOR = "white"
AXES_FACE_COLOR = "white"
THETA_COLORMAP = "viridis"
SUCCESS_COLOR = "#2ca02c"
TIMEOUT_COLOR = "#d62728"
BLOCKED_COLOR = "#9e9e9e"
LINE_WIDTH = 1.6
MARKER_SIZE = 7.0
SUCCESS_MARKER = "o"
TIMEOUT_MARKER = "x"
BLOCKED_MARKER = "s"
GRID_ENABLED = True
GRID_ALPHA = 0.22
TITLE_FONT_SIZE = 17
SUPTITLE_FONT_SIZE = 19
AXIS_LABEL_FONT_SIZE = 17
TICK_LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 13
STATUS_LEGEND_LOCATION = "best"
DATA_LEGEND_LOCATION = "best"
TICK_DIRECTION = "in"
TICKS_TOP = True
TICKS_RIGHT = True
SUBPLOT_HORIZONTAL_SPACE = 0.28
RHO_X_LIMITS = (0.837, 0.903)
VELOCITY_SCALE = 1.0e-3
SPEED_OF_LIGHT_M_S = 299_792_458.0
VELOCITY_UNIT_LABEL = r"$u(0^-)$ [km s$^{-1}$]"
JB_Y_SCALE = "linear"
VELOCITY_Y_SCALE = "linear"


payload = np.load(PAYLOAD_PATH, allow_pickle=True).item()
theta_indices = np.asarray(payload["theta_indices"], dtype=int)
rho_ladder = np.asarray(payload["rho_ladder"], dtype=float)
entries = payload["entries"]
cmap = plt.get_cmap(THETA_COLORMAP)
theta_colors = {
    int(theta_index): cmap(position / max(1, len(theta_indices) - 1))
    for position, theta_index in enumerate(theta_indices)
}

fig, axes = plt.subplots(
    1,
    3,
    figsize=FIGURE_SIZE,
    dpi=FIGURE_DPI,
    facecolor=FIGURE_FACE_COLOR,
)
status_axis, jB_axis, velocity_axis = axes
for axis in axes:
    axis.set_facecolor(AXES_FACE_COLOR)
    axis.grid(GRID_ENABLED, alpha=GRID_ALPHA)
    axis.tick_params(
        labelsize=TICK_LABEL_FONT_SIZE,
        direction=TICK_DIRECTION,
        top=TICKS_TOP,
        right=TICKS_RIGHT,
    )

status_labels_used = set()
for theta_position, theta_index in enumerate(theta_indices):
    successful_rho = []
    successful_jB = []
    successful_velocity = []
    for rho_position, rho_value in enumerate(rho_ladder):
        key = f"theta={int(theta_index):02d}|rho={float(rho_value):.8f}"
        record = entries[key]["record"]
        status = str(record.get("status", "failure"))
        if bool(record.get("success")):
            marker = SUCCESS_MARKER
            color = SUCCESS_COLOR
            label = "success"
            successful_rho.append(float(rho_value))
            successful_jB.append(float(record["jB"]))
            successful_velocity.append(
                float(record["u_0minus"]) * SPEED_OF_LIGHT_M_S * VELOCITY_SCALE
            )
        elif "timeout" in status:
            marker = TIMEOUT_MARKER
            color = TIMEOUT_COLOR
            label = "trial timeout"
        else:
            marker = BLOCKED_MARKER
            color = BLOCKED_COLOR
            label = "blocked by prior step"
        status_axis.plot(
            rho_value,
            theta_index,
            marker=marker,
            linestyle="none",
            color=color,
            markersize=MARKER_SIZE,
            label=label if label not in status_labels_used else None,
        )
        status_labels_used.add(label)
    if successful_rho:
        color = theta_colors[int(theta_index)]
        label = rf"$\theta_{{\rm index}}={int(theta_index)}$"
        jB_axis.plot(
            successful_rho,
            successful_jB,
            color=color,
            linewidth=LINE_WIDTH,
            marker=SUCCESS_MARKER,
            markersize=MARKER_SIZE,
            label=label,
        )
        velocity_axis.plot(
            successful_rho,
            successful_velocity,
            color=color,
            linewidth=LINE_WIDTH,
            marker=SUCCESS_MARKER,
            markersize=MARKER_SIZE,
            label=label,
        )

status_axis.set_title("Pilot status by hidden point", fontsize=TITLE_FONT_SIZE)
status_axis.set_xlabel(r"hidden radial fraction $\rho$", fontsize=AXIS_LABEL_FONT_SIZE)
status_axis.set_ylabel(r"$\theta$ index", fontsize=AXIS_LABEL_FONT_SIZE)
status_axis.set_yticks(theta_indices)
status_axis.set_xlim(RHO_X_LIMITS)
status_axis.legend(fontsize=LEGEND_FONT_SIZE, loc=STATUS_LEGEND_LOCATION)

jB_axis.set_title(r"Converged $j_B(\rho)$", fontsize=TITLE_FONT_SIZE)
jB_axis.set_xlabel(r"hidden radial fraction $\rho$", fontsize=AXIS_LABEL_FONT_SIZE)
jB_axis.set_ylabel(r"$j_B$", fontsize=AXIS_LABEL_FONT_SIZE)
jB_axis.set_yscale(JB_Y_SCALE)
jB_axis.set_xlim(RHO_X_LIMITS)
jB_axis.legend(fontsize=LEGEND_FONT_SIZE, loc=DATA_LEGEND_LOCATION)

velocity_axis.set_title(r"Converged velocity", fontsize=TITLE_FONT_SIZE)
velocity_axis.set_xlabel(r"hidden radial fraction $\rho$", fontsize=AXIS_LABEL_FONT_SIZE)
velocity_axis.set_ylabel(VELOCITY_UNIT_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
velocity_axis.set_yscale(VELOCITY_Y_SCALE)
velocity_axis.set_xlim(RHO_X_LIMITS)
velocity_axis.legend(fontsize=LEGEND_FONT_SIZE, loc=DATA_LEGEND_LOCATION)

fig.suptitle(
    r"$T(0^+)=0.1$ MeV hidden-$\rho$ continuation pilot",
    fontsize=SUPTITLE_FONT_SIZE,
)
fig.subplots_adjust(wspace=SUBPLOT_HORIZONTAL_SPACE)
fig.savefig(OUTPUT_PNG, dpi=EXPORT_DPI, facecolor=FIGURE_FACE_COLOR, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, facecolor=FIGURE_FACE_COLOR, bbox_inches="tight")
print(f"Saved {OUTPUT_PNG}")
print(f"Saved {OUTPUT_PDF}")
plt.show()
