#!/usr/bin/env python3
"""Run the analytical isothermal contour stage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-isothermal-contour")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from tqdm.auto import tqdm

import _isothermal_domain as domain
import _isothermal_execution as execution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="isothermal-domain.npy")
    parser.add_argument("--output", default="isothermal-analytic.npy")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--timeout", type=float, default=execution.DEFAULT_ANALYTIC_TIMEOUT_S
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    domain_payload = domain.load_payload(args.domain, expected_kind="isothermal_domain")
    domain.assert_domain_fingerprint(domain_payload)
    domain.validate_cluster_physics(
        ms=domain_payload["ms_MeV"],
        NM_type=domain_payload["NM_type"],
        upB=domain_payload["upB"],
    )
    payload = execution.run_analytic_stage(
        domain_payload=domain_payload,
        output_path=Path(args.output),
        workers=args.workers,
        resume=args.resume,
        timeout_s=args.timeout,
        progress_factory=tqdm,
    )
    success = int(np.count_nonzero(payload["task_status"] == "success"))
    print(f"Saved {args.output}: analytical successes {success}/{payload['task_status'].size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
