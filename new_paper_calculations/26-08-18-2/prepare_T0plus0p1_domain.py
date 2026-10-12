#!/usr/bin/env python3
"""Prepare the endpoint-inclusive safe domain for T(0+)=0.1 MeV."""

from __future__ import annotations

import sys

import numpy as np

from _T0plus0p1_runner import prepare_domain_main


# Tunable run parameters
DEFAULT_OUTPUT = "T0plus0p1-domain.npy"
DEFAULT_RUN_TAG = "T0plus0p1_domain_radial_shell_v1"
DEFAULT_T0PLUS_MEV = 0.1
DEFAULT_BOUNDARY_THETA_COUNT = 12
DEFAULT_CONTOUR_RAY_COUNT = 30
DEFAULT_RADIAL_FRACTIONS = (
    0.02, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.66, 0.72, 0.78, 0.84, 0.90, 0.95, 0.99,
)
DEFAULT_BRANCH_CHECK_RADIAL_FRACTION = 0.55
DEFAULT_ANGULAR_BUFFER_FRACTION = 0.0
DEFAULT_RADIAL_BUFFER_FRACTION = 0.0
DEFAULT_DENSITY_MIN_OVER_N0 = 1.0
DEFAULT_DENSITY_MAX_OVER_N0 = 5.5
DEFAULT_TEMPERATURE_MIN_MEV = 1.0e-2
DEFAULT_TEMPERATURE_MAX_MEV = 120.0
DEFAULT_RAY_PROBE_COUNT = 15
DEFAULT_BISECTION_STEPS = 10
DEFAULT_XI = -0.5


def contour_theta_axis(count: int) -> np.ndarray:
    """Return an endpoint-inclusive theta axis."""
    return np.linspace(0.0, 0.5 * np.pi, int(count))


def default_arguments() -> list[str]:
    return [
        "--output", DEFAULT_OUTPUT,
        "--run-tag", DEFAULT_RUN_TAG,
        "--T0plus-values", f"{DEFAULT_T0PLUS_MEV:g}",
        "--boundary-theta-count", str(DEFAULT_BOUNDARY_THETA_COUNT),
        "--ray-count", str(DEFAULT_CONTOUR_RAY_COUNT),
        "--radial-fractions", ",".join(f"{value:g}" for value in DEFAULT_RADIAL_FRACTIONS),
        "--branch-check-radial-fraction", f"{DEFAULT_BRANCH_CHECK_RADIAL_FRACTION:g}",
        "--angular-buffer-fraction", f"{DEFAULT_ANGULAR_BUFFER_FRACTION:g}",
        "--radial-buffer-fraction", f"{DEFAULT_RADIAL_BUFFER_FRACTION:g}",
        "--density-min-over-n0", f"{DEFAULT_DENSITY_MIN_OVER_N0:g}",
        "--density-max-over-n0", f"{DEFAULT_DENSITY_MAX_OVER_N0:g}",
        "--temperature-min", f"{DEFAULT_TEMPERATURE_MIN_MEV:g}",
        "--temperature-max", f"{DEFAULT_TEMPERATURE_MAX_MEV:g}",
        "--ray-probe-count", str(DEFAULT_RAY_PROBE_COUNT),
        "--bisection-steps", str(DEFAULT_BISECTION_STEPS),
        "--xi", f"{DEFAULT_XI:g}",
    ]


if __name__ == "__main__":
    raise SystemExit(prepare_domain_main([*default_arguments(), *sys.argv[1:]]))
