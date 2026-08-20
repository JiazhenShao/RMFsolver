#!/usr/bin/env python3
"""Prepare the boundary-fitted isothermal contour domain."""

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


DEFAULT_OUTPUT = "isothermal-domain.npy"


def _worker_default() -> int:
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=_worker_default())
    parser.add_argument(
        "--B-one-forth", type=float, default=domain.DEFAULT_B_ONE_FORTH_MEV
    )
    parser.add_argument("--xi", type=float, default=domain.DEFAULT_XI)
    parser.add_argument("--ms", type=float, default=domain.DEFAULT_MS_MEV)
    parser.add_argument("--NM-type", default=domain.DEFAULT_NM_TYPE)
    parser.add_argument("--upB", type=int, default=domain.DEFAULT_UPB)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    domain.validate_cluster_physics(ms=args.ms, NM_type=args.NM_type, upB=args.upB)
    if args.smoke:
        temperatures = np.asarray((1.0, 20.0, 80.0))
        targets = np.asarray((0.1, 0.5, 0.9))
    else:
        temperatures = domain.default_temperature_axis()
        targets = domain.default_a_0plus_target_axis()
    output_path = Path(args.output)
    reference = domain.base_payload(
        kind="isothermal_domain",
        temperature_axis=temperatures,
        a_0plus_target_axis=targets,
        B_one_forth=args.B_one_forth,
        xi=args.xi,
        ms=args.ms,
        NM_type=args.NM_type,
        upB=args.upB,
    )
    if args.resume and output_path.exists():
        payload = domain.load_payload(output_path, expected_kind="isothermal_domain")
        domain.assert_compatible_payload(reference, payload)
        if bool(payload.get("run_complete")):
            domain.assert_domain_fingerprint(payload)
            output = output_path
        else:
            payload = domain.build_domain_payload(
                temperature_axis=temperatures,
                a_0plus_target_axis=targets,
                B_one_forth=args.B_one_forth,
                xi=args.xi,
                ms=args.ms,
                NM_type=args.NM_type,
                upB=args.upB,
                workers=args.workers,
                output_path=output_path,
                progress_factory=tqdm,
                resume_payload=payload,
            )
            output = output_path
    else:
        payload = domain.build_domain_payload(
            temperature_axis=temperatures,
            a_0plus_target_axis=targets,
            B_one_forth=args.B_one_forth,
            xi=args.xi,
            ms=args.ms,
            NM_type=args.NM_type,
            upB=args.upB,
            workers=args.workers,
            output_path=output_path,
            progress_factory=tqdm,
        )
        output = output_path
    success = int(np.count_nonzero(payload["cell_status"] == "success"))
    total = int(payload["cell_status"].size)
    print(f"Saved {output}: valid cells {success}/{total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
