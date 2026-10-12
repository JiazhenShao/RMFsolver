#!/usr/bin/env python3
"""Run fixed-T(0+)=0.1 MeV uNmax with angular-wavefront continuation."""

from __future__ import annotations

import sys

from _T0plus0p1_runner import numerical_main


# Tunable run parameters
DEFAULT_DOMAIN_FILE = "T0plus0p1-domain.npy"
DEFAULT_OUTPUT = "T0plus0p1-uNmax-angular-wavefront.npy"
DEFAULT_RUN_TAG = "T0plus0p1_uNmax_angular_wavefront_v2"
DEFAULT_SAVE_EVERY = 1
DEFAULT_TAIL_EPS = 1.0e-8
DEFAULT_N_MESH = 200
DEFAULT_TOL_BVP = 1.0e-4
DEFAULT_MAX_NODES = 10000
DEFAULT_JB_LOWER_BOUND = 1.0e-12
DEFAULT_JB_UPPER_FACTOR = 1.0e-4
DEFAULT_TRIAL_TIMEOUT_S = 300.0
DEFAULT_CELL_TIMEOUT_S = 1200.0
DEFAULT_REFERENCE_T0PLUS_MEV = 1.0
DEFAULT_REFERENCE_JB_SCALE = 1.18
DEFAULT_MAX_BRIDGE_DEPTH = 2
DEFAULT_MIN_BRIDGE_FRACTION_STEP = 1.0e-3


def default_arguments() -> list[str]:
    return [
        "--domain-file", DEFAULT_DOMAIN_FILE,
        "--output", DEFAULT_OUTPUT,
        "--run-tag", DEFAULT_RUN_TAG,
        "--save-every", str(DEFAULT_SAVE_EVERY),
        "--tail-eps", f"{DEFAULT_TAIL_EPS:g}",
        "--n-mesh", str(DEFAULT_N_MESH),
        "--tol-bvp", f"{DEFAULT_TOL_BVP:g}",
        "--max-nodes", str(DEFAULT_MAX_NODES),
        "--jB-lower-bound", f"{DEFAULT_JB_LOWER_BOUND:g}",
        "--jB-upper-factor", f"{DEFAULT_JB_UPPER_FACTOR:g}",
        "--point-timeout", f"{DEFAULT_TRIAL_TIMEOUT_S:g}",
        "--target-timeout", f"{DEFAULT_CELL_TIMEOUT_S:g}",
        "--reference-T0plus", f"{DEFAULT_REFERENCE_T0PLUS_MEV:g}",
        "--reference-jB-scale", f"{DEFAULT_REFERENCE_JB_SCALE:g}",
        "--max-bridge-depth", str(DEFAULT_MAX_BRIDGE_DEPTH),
        "--min-bridge-fraction-step", f"{DEFAULT_MIN_BRIDGE_FRACTION_STEP:g}",
        "--skip-branch-check",
    ]


if __name__ == "__main__":
    raise SystemExit(numerical_main([*default_arguments(), *sys.argv[1:]]))
