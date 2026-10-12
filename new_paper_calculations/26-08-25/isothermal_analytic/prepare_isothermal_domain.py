#!/usr/bin/env python3
"""Prepare the boundary-fitted isothermal contour domain."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
        theta_axis = np.linspace(0.0, 0.5 * np.pi, 3)
        radial_axis = np.asarray((0.2, 0.5, 0.8))
        boundary_theta_axis = np.linspace(0.0, 0.5 * np.pi, 3)
        probe_count = 6
    else:
        theta_axis = domain.default_theta_axis_rad()
        radial_axis = domain.default_radial_fraction_axis()
        boundary_theta_axis = domain.default_boundary_theta_axis_rad()
        probe_count = domain.DEFAULT_BOUNDARY_PROBE_COUNT
    output_path = Path(args.output)
    reference = domain.polar_base_payload(
        kind="isothermal_domain",
        theta_axis_rad=theta_axis,
        radial_fraction_axis=radial_axis,
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
            payload = domain.build_polar_domain_payload(
                theta_axis_rad=theta_axis,
                radial_fraction_axis=radial_axis,
                boundary_theta_axis_rad=boundary_theta_axis,
                B_one_forth=args.B_one_forth,
                xi=args.xi,
                ms=args.ms,
                NM_type=args.NM_type,
                upB=args.upB,
                workers=args.workers,
                output_path=output_path,
                progress_factory=tqdm,
                resume_payload=payload,
                probe_count=probe_count,
            )
            output = output_path
    else:
        payload = domain.build_polar_domain_payload(
            theta_axis_rad=theta_axis,
            radial_fraction_axis=radial_axis,
            boundary_theta_axis_rad=boundary_theta_axis,
            B_one_forth=args.B_one_forth,
            xi=args.xi,
            ms=args.ms,
            NM_type=args.NM_type,
            upB=args.upB,
            workers=args.workers,
            output_path=output_path,
            progress_factory=tqdm,
            probe_count=probe_count,
        )
        output = output_path
    success = int(np.count_nonzero(payload["cell_status"] == "success"))
    total = int(payload["cell_status"].size)
    if success != total:
        raise RuntimeError(
            f"Polar domain is incomplete: {success}/{total} valid physical cells"
        )
    print(f"Saved {output}: valid cells {success}/{total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
