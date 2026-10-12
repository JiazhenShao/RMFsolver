#!/usr/bin/env python3
"""Run the full analytical velocity contour on the prepared domain."""

from __future__ import annotations

import sys

from _T0plus_contour_runner import analytical_main


# Tunable run parameters
DEFAULT_DOMAIN_FILE = "T0plus-contour-domain.npy"
DEFAULT_OUTPUT = "T0plus-full-analytic-contour.npy"
DEFAULT_RUN_TAG = "T0plus_analytic_contour_v1"
DEFAULT_SAVE_EVERY = 1


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
    ]


if __name__ == "__main__":
    raise SystemExit(
        analytical_main("analytic", [*default_arguments(), *sys.argv[1:]])
    )
