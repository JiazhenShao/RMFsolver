#!/usr/bin/env python3
"""Run the 5 MeV semi-analytical scan and save numerical shooting seeds."""

from __future__ import annotations

import sys

from _T0plus5_runner import analytical_main


# Tunable run parameters
DEFAULT_DOMAIN_FILE = "T0plus5-domain.npy"
DEFAULT_OUTPUT = "T0plus5-semi-analytic.npy"
DEFAULT_RUN_TAG = "T0plus5_semi_analytic_v1"
DEFAULT_SAVE_EVERY = 10


def default_arguments() -> list[str]:
    return [
        "--domain-file", DEFAULT_DOMAIN_FILE,
        "--output", DEFAULT_OUTPUT,
        "--run-tag", DEFAULT_RUN_TAG,
        "--save-every", str(DEFAULT_SAVE_EVERY),
    ]


if __name__ == "__main__":
    raise SystemExit(analytical_main("semi_analytic", [*default_arguments(), *sys.argv[1:]]))
