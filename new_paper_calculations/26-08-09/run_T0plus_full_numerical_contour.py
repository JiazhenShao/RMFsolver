#!/usr/bin/env python3
"""Run the full numerical uNmax contour on the prepared domain."""

from __future__ import annotations

import sys

from _T0plus_contour_runner import numerical_main


# Tunable run parameters
DEFAULT_DOMAIN_FILE = "T0plus-contour-domain.npy"
DEFAULT_OUTPUT = "T0plus-full-numerical-contour.npy"
DEFAULT_RUN_TAG = "T0plus_numerical_contour_v1"
DEFAULT_SAVE_EVERY = 1
DEFAULT_TAIL_EPS = 1.0e-8
DEFAULT_N_MESH = 200
DEFAULT_TOL_BVP = 1.0e-4
DEFAULT_MAX_NODES = 10000
DEFAULT_JB_LOWER_BOUND = 1.0e-12
DEFAULT_JB_UPPER_FACTOR = 1.0e-4
DEFAULT_POINT_TIMEOUT_S = 1800.0
DEFAULT_BRANCH_CHECK_RADIAL_FRACTION = 0.55


def default_arguments() -> list[str]:
    return [
        "--domain-file",
        DEFAULT_DOMAIN_FILE,
        "--output",
        DEFAULT_OUTPUT,
        "--run-tag",
        DEFAULT_RUN_TAG,
        "--save-every",
        str(DEFAULT_SAVE_EVERY),
        "--tail-eps",
        f"{DEFAULT_TAIL_EPS:g}",
        "--n-mesh",
        str(DEFAULT_N_MESH),
        "--tol-bvp",
        f"{DEFAULT_TOL_BVP:g}",
        "--max-nodes",
        str(DEFAULT_MAX_NODES),
        "--jB-lower-bound",
        f"{DEFAULT_JB_LOWER_BOUND:g}",
        "--jB-upper-factor",
        f"{DEFAULT_JB_UPPER_FACTOR:g}",
        "--point-timeout",
        f"{DEFAULT_POINT_TIMEOUT_S:g}",
        "--branch-check-radial-fraction",
        f"{DEFAULT_BRANCH_CHECK_RADIAL_FRACTION:g}",
    ]


if __name__ == "__main__":
    raise SystemExit(numerical_main([*default_arguments(), *sys.argv[1:]]))
