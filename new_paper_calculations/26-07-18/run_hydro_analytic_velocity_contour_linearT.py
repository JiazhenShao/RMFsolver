#!/usr/bin/env python3
"""Parallel hydro-analytic contour on adaptive linear temperature columns."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hydro_analytic_contour_common import RunConfig, main


# Tunable run parameters
DEFAULT_OUTPUT_TEMPLATE = "hydro-analytic-LTE-adaptive-linearT-contour__{run_tag}.npy"
DEFAULT_RUN_TAG = "analytic_velocity_linearT_finiteT_ncrit_v1"
DEFAULT_TEMPERATURE_MESH = "adaptive_linear"
DEFAULT_TEMPERATURE_AXIS_SCALE = "linear"
DEFAULT_TEMPERATURE_MIN_MEV = 1.0e-2
DEFAULT_TEMPERATURE_MAX_MEV = 100.0
DEFAULT_TEMPERATURE_COUNT = 30
DEFAULT_METASTABILITY_MIN = 0.01
DEFAULT_METASTABILITY_MAX = 0.9
DEFAULT_METASTABILITY_COUNT = 20
DEFAULT_NCRIT_OVER_N0 = 3.0
DEFAULT_B14_MEV = np.nan
DEFAULT_XI = 0.0
DEFAULT_MS_MEV = 0.0
DEFAULT_NM_TYPE = "PNM"
DEFAULT_UPB = 5000
DEFAULT_INTERFACE_FRACTION_MODE = "LTE"
DEFAULT_BOUNDARY_PROBE_COUNT = 8
DEFAULT_BOUNDARY_BISECT_STEPS = 10
DEFAULT_BOUNDARY_EPS_MEV = 0.005
DEFAULT_SAVE_EVERY = 10
DEFAULT_SEED_TEMPERATURE_MEV = 10.0
DEFAULT_NEIGHBOR_RETRY = True

RUN_CONFIG = RunConfig(
    output_template=DEFAULT_OUTPUT_TEMPLATE,
    run_tag=DEFAULT_RUN_TAG,
    temperature_mesh=DEFAULT_TEMPERATURE_MESH,
    temperature_axis_scale=DEFAULT_TEMPERATURE_AXIS_SCALE,
    temperature_min=DEFAULT_TEMPERATURE_MIN_MEV,
    temperature_max=DEFAULT_TEMPERATURE_MAX_MEV,
    temperature_count=DEFAULT_TEMPERATURE_COUNT,
    metastability_min=DEFAULT_METASTABILITY_MIN,
    metastability_max=DEFAULT_METASTABILITY_MAX,
    metastability_count=DEFAULT_METASTABILITY_COUNT,
    ncrit_over_n0=DEFAULT_NCRIT_OVER_N0,
    B14=DEFAULT_B14_MEV,
    xi=DEFAULT_XI,
    ms=DEFAULT_MS_MEV,
    nm_type=DEFAULT_NM_TYPE,
    upB=DEFAULT_UPB,
    interface_fraction_mode=DEFAULT_INTERFACE_FRACTION_MODE,
    boundary_probe_count=DEFAULT_BOUNDARY_PROBE_COUNT,
    boundary_bisect_steps=DEFAULT_BOUNDARY_BISECT_STEPS,
    boundary_eps=DEFAULT_BOUNDARY_EPS_MEV,
    save_every=DEFAULT_SAVE_EVERY,
    seed_temperature=DEFAULT_SEED_TEMPERATURE_MEV,
    neighbor_retry=DEFAULT_NEIGHBOR_RETRY,
)


if __name__ == "__main__":
    main(RUN_CONFIG)
