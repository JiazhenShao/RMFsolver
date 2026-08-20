#!/usr/bin/env python3
"""Run the exact numerical isothermal contour stage."""

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
    parser.add_argument("--analytic", default="isothermal-analytic.npy")
    parser.add_argument("--output", default="isothermal-numerical.npy")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--trial-timeout", type=float, default=execution.DEFAULT_TRIAL_TIMEOUT_S
    )
    parser.add_argument(
        "--cell-timeout", type=float, default=execution.DEFAULT_CELL_TIMEOUT_S
    )
    parser.add_argument("--tail-eps", type=float, default=execution.DEFAULT_TAIL_EPS)
    parser.add_argument("--n-mesh", type=int, default=execution.DEFAULT_N_MESH)
    parser.add_argument("--tol-bvp", type=float, default=execution.DEFAULT_BVP_TOL)
    parser.add_argument("--max-nodes", type=int, default=execution.DEFAULT_MAX_NODES)
    parser.add_argument(
        "--kappa-factor", type=float, default=execution.DEFAULT_KAPPA_FACTOR
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
    analytic_payload = domain.load_payload(
        args.analytic, expected_kind="isothermal_analytic"
    )
    payload = execution.run_numerical_stage(
        domain_payload=domain_payload,
        analytic_payload=analytic_payload,
        output_path=Path(args.output),
        workers=args.workers,
        resume=args.resume,
        trial_timeout_s=args.trial_timeout,
        cell_timeout_s=args.cell_timeout,
        tail_eps=args.tail_eps,
        n_mesh=args.n_mesh,
        tol_bvp=args.tol_bvp,
        max_nodes=args.max_nodes,
        kappa_factor=args.kappa_factor,
        progress_factory=tqdm,
    )
    success = int(np.count_nonzero(payload["task_status"] == "success"))
    total = int(payload["task_status"].size)
    print(f"Saved {args.output}: numerical successes {success}/{total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
