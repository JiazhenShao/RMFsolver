#!/usr/bin/env python3
import sys

from _T0plus_split_runner import prepare_domain_main


DEFAULT_T0PLUS_VALUES_MEV = (50.0, 10.0, 1.0)
DEFAULT_BOUNDARY_THETA_COUNT = 10


def build_argv(cli_args=None):
    """Apply this launcher's defaults before any user CLI overrides."""
    values = ",".join(f"{value:g}" for value in DEFAULT_T0PLUS_VALUES_MEV)
    return [
        "--T0plus-values",
        values,
        "--boundary-theta-count",
        str(DEFAULT_BOUNDARY_THETA_COUNT),
        *(sys.argv[1:] if cli_args is None else cli_args),
    ]


if __name__ == "__main__":
    raise SystemExit(prepare_domain_main(build_argv()))
