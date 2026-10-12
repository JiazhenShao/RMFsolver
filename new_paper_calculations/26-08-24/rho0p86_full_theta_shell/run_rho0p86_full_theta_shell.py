#!/usr/bin/env python3
"""Run the complete theta shell at rho=0.86 for T(0+)=0.1 MeV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PILOT_ENGINE_DIR = SCRIPT_DIR.parent / "rho0p9_hidden_rho_pilot"
if str(PILOT_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_ENGINE_DIR))

import run_hidden_rho_pilot as engine


# Tunable run parameters
DEFAULT_THETA_INDICES = tuple(range(30))
DEFAULT_TARGET_RHO = 0.86
DEFAULT_WORKERS = 30
DEFAULT_JB_DIVERGENCE_EXPONENT = 1.26
DEFAULT_JB_PERTURBATIONS = (1.0, 0.95, 1.05)
DEFAULT_TRIAL_TIMEOUT_S = 300.0
DEFAULT_CELL_TIMEOUT_S = 900.0
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "T0plus0p1-rho0p86-full-theta-shell.npy"


def comma_join(values) -> str:
    return ",".join(str(value) for value in values)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Solve all 30 angular rays at rho=0.86 using each ray's "
            "successful rho=0.84 full numerical profile."
        )
    )
    result.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    result.add_argument("--trial-timeout", type=float, default=DEFAULT_TRIAL_TIMEOUT_S)
    result.add_argument("--cell-timeout", type=float, default=DEFAULT_CELL_TIMEOUT_S)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    result.add_argument("--restart", action="store_true")
    result.add_argument("--retry-failed", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    forwarded = [
        "--theta-indices",
        comma_join(DEFAULT_THETA_INDICES),
        "--rho-ladder",
        str(DEFAULT_TARGET_RHO),
        "--jB-exponent",
        str(DEFAULT_JB_DIVERGENCE_EXPONENT),
        "--jB-perturbations",
        comma_join(DEFAULT_JB_PERTURBATIONS),
        "--workers",
        str(args.workers),
        "--trial-timeout",
        str(args.trial_timeout),
        "--cell-timeout",
        str(args.cell_timeout),
        "--output",
        str(args.output),
    ]
    if args.restart:
        forwarded.append("--restart")
    if args.retry_failed:
        forwarded.append("--retry-failed")
    if args.dry_run:
        forwarded.append("--dry-run")
    return engine.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
