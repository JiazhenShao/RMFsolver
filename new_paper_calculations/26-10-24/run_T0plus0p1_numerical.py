#!/usr/bin/env python3
"""Run fixed-T(0+)=0.1 MeV nonrelativistic adiabatic BVP with angular-wavefront continuation."""

from __future__ import annotations

import sys

from _T0plus0p1_runner import numerical_main


# Tunable run parameters
DEFAULT_DOMAIN_FILE = "T0plus0p1-domain.npy"
DEFAULT_OUTPUT = "T0plus0p1-adiabatic-slow.npy"
DEFAULT_RUN_TAG = "T0plus0p1_adiabatic_slow_v1"
DEFAULT_SAVE_EVERY = 1
DEFAULT_TAIL_EPS = 1.0e-8
DEFAULT_N_MESH = 300
DEFAULT_TOL_BVP = 1.0e-4
DEFAULT_MAX_NODES = 50000
DEFAULT_TRIAL_TIMEOUT_S = 300.0
DEFAULT_CELL_TIMEOUT_S = 1200.0
DEFAULT_REFERENCE_JB_SCALE = 1.0
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
        "--point-timeout", f"{DEFAULT_TRIAL_TIMEOUT_S:g}",
        "--target-timeout", f"{DEFAULT_CELL_TIMEOUT_S:g}",
        "--reference-jB-scale", f"{DEFAULT_REFERENCE_JB_SCALE:g}",
        "--max-bridge-depth", str(DEFAULT_MAX_BRIDGE_DEPTH),
        "--min-bridge-fraction-step", f"{DEFAULT_MIN_BRIDGE_FRACTION_STEP:g}",
    ]


if __name__ == "__main__":
    raise SystemExit(numerical_main([*default_arguments(), *sys.argv[1:]]))
