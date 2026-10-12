#!/usr/bin/env python3
"""Diagnose one representative semi-analytical boundary root."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import _T0plus_split_runner as split

import RMFsolver.constants as const
from RMFsolver import phase_velocity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    ray_grid = split.benchmark.ray_grid
    n0 = float(0.16 * const.MeV_fm**3)
    B14, _, _ = ray_grid.calibrate_b14(3.0 * n0, 0.0, "PNM")
    paths = (
        Path(phase_velocity.__file__).resolve(),
        Path(ray_grid.__file__).resolve(),
        Path(split.__file__).resolve(),
    )
    print("Imported source files:")
    for path in paths:
        print(f"  {_sha256(path)}  {path}")

    transform = ray_grid.RayTransform(
        density_origin_over_n0=1.0,
        density_scale_over_n0=9.0,
        temperature_scale_MeV=200.0,
    )
    theta = float(np.deg2rad(54.0))
    rho_min, rho_max = ray_grid._ray_rho_bounds(
        transform=transform,
        theta_rad=theta,
        density_max_over_n0=10.0,
        temperature_min_MeV=0.01,
        temperature_max_MeV=200.0,
    )
    task = {
        "index": 0,
        "theta_index": 0,
        "theta_rad": theta,
        "path_mode": "elliptical_ray",
        "transform": {
            "density_origin_over_n0": 1.0,
            "density_scale_over_n0": 9.0,
            "temperature_scale_MeV": 200.0,
        },
        "coordinates": np.linspace(rho_min, rho_max, 5),
        "n0": n0,
        "B14": float(B14),
        "nm_type": "PNM",
        "task_kind": "A_ray",
        "label": "diagnostic_theta_54_T0plus_0p5",
        "xi": -0.5,
        "ms": 0.0,
        "upB": 5000,
        "interface_fraction_mode": "fixed_T_0plus",
        "velocity_closure": "numerical_I2",
        "T_0plus_MeV": 0.5,
        "bisection_steps": 4,
        "allow_missing": True,
        "endpoint_hint": None,
        "muB_hint": 1100.0,
    }
    print(f"B^(1/4)={B14:.12f} MeV")
    print("Physical probes:")
    for rho in task["coordinates"]:
        density, temperature = transform.to_physical(float(rho), theta)
        print(
            f"  rho={rho:.9g}, nB(0-)/n0={density:.9g}, "
            f"T(0-)={temperature:.9g} MeV"
        )
    result = split._solve_direct_boundary_task(task)
    print(f"success={result['success']}")
    if result["success"]:
        print(result["completed"]["record"])
        return 0
    print(result["message"])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
