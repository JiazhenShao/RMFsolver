#!/usr/bin/env python3
"""Parallel numerical-I2 analytical velocity scan in the T versus nB(0-) plane.

The calculation mesh is constructed in dimensionless elliptical coordinates.
Its inner edge follows Delta h = 0; its outer edge follows A = 1 where that
boundary exists and otherwise clips to the requested physical window. Physics
is evaluated only after every point is transformed back to absolute nB(0-)/n0
and T. The saved primary arrays are two-dimensional physical Cartesian
coordinates suitable for direct use with matplotlib.contour. The solver is
imported from the live repository, so each run uses the current analytical
velocity implementation. This runner explicitly selects the numerical-I2
closure, which traces the conversion layer at conserved P and h/nB and
integrates the transport term with the exact rate and full diffusion model.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import RMFsolver.constants as const
from _analytic_ray_grid import (
    STATUS_FINITE,
    STATUS_NUMERICAL_FAILURE,
    STATUS_UNSCANNED,
    RayTransform,
    atomic_save,
    build_payload,
    build_safe_ray_mesh,
    calculate_absolute_density_point,
    calibrate_b14,
    chebyshev_open_axis,
    radial_solve_order,
    solve_boundary_curves,
)


# Tunable run parameters
DEFAULT_OUTPUT = "analytic-uN-T-nB-ray-grid-numerical-I2.npy"
DEFAULT_RUN_TAG = "analytic_uN_T_nB_numerical_I2_20260802_v3"
DEFAULT_DENSITY_MIN_OVER_N0 = 1.0
DEFAULT_DENSITY_MAX_OVER_N0 = 10.0
DEFAULT_TEMPERATURE_MIN_MEV = 1.0e-2
DEFAULT_TEMPERATURE_MAX_MEV = 200.0
DEFAULT_THETA_COUNT = 45
DEFAULT_RADIAL_COUNT = 30
DEFAULT_ANGULAR_BUFFER_FRACTION = 1.0e-4
DEFAULT_BOUNDARY_BUFFER_FRACTION = 1.0e-3
DEFAULT_ENDPOINT_PROBE_COUNT = 25
DEFAULT_RAY_PROBE_COUNT = 15
DEFAULT_BOUNDARY_BISECT_STEPS = 10
DEFAULT_SEED_TEMPERATURE_MEV = 10.0
DEFAULT_NEIGHBOR_RETRY = True
DEFAULT_SAVE_EVERY = 10
DEFAULT_ENDPOINT_GUESS = None
DEFAULT_INTERFACE_FRACTION_MODE = "extreme_endothermic"
DEFAULT_VELOCITY_CLOSURE = "numerical_I2"
DEFAULT_NCRIT_OVER_N0 = 3.0
DEFAULT_XI = 0.0
DEFAULT_MS_MEV = 0.0
DEFAULT_NM_TYPE = "PNM"
DEFAULT_UPB = 5000


def _default_workers() -> int:
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


def _finite_pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        pair = (float(value[0]), float(value[1]))
    except Exception:
        return None
    if not np.all(np.isfinite(pair)) or pair[0] <= 0.0 or pair[1] <= 0.0:
        return None
    return pair


def _parse_endpoint(value: str | None) -> tuple[float, float] | None:
    if value is None or not str(value).strip():
        return None
    fields = [field.strip() for field in str(value).split(",")]
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("endpoint guess must be MU_B_Q,T_inf")
    endpoint = _finite_pair((float(fields[0]), float(fields[1])))
    if endpoint is None:
        raise argparse.ArgumentTypeError(
            "endpoint guess values must be positive and finite"
        )
    return endpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-tag",
        default=os.getenv("RUN_TAG", DEFAULT_RUN_TAG),
    )
    parser.add_argument(
        "--density-min-over-n0",
        type=float,
        default=DEFAULT_DENSITY_MIN_OVER_N0,
    )
    parser.add_argument(
        "--density-max-over-n0",
        type=float,
        default=DEFAULT_DENSITY_MAX_OVER_N0,
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=DEFAULT_TEMPERATURE_MIN_MEV,
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=DEFAULT_TEMPERATURE_MAX_MEV,
    )
    parser.add_argument(
        "--theta-count",
        type=int,
        default=DEFAULT_THETA_COUNT,
    )
    parser.add_argument(
        "--radial-count",
        type=int,
        default=DEFAULT_RADIAL_COUNT,
    )
    parser.add_argument(
        "--angular-buffer-fraction",
        type=float,
        default=DEFAULT_ANGULAR_BUFFER_FRACTION,
    )
    parser.add_argument(
        "--boundary-buffer-fraction",
        type=float,
        default=DEFAULT_BOUNDARY_BUFFER_FRACTION,
    )
    parser.add_argument(
        "--endpoint-probe-count",
        type=int,
        default=DEFAULT_ENDPOINT_PROBE_COUNT,
    )
    parser.add_argument(
        "--ray-probe-count",
        type=int,
        default=DEFAULT_RAY_PROBE_COUNT,
    )
    parser.add_argument(
        "--boundary-bisect-steps",
        type=int,
        default=DEFAULT_BOUNDARY_BISECT_STEPS,
    )
    parser.add_argument(
        "--seed-temperature",
        type=float,
        default=DEFAULT_SEED_TEMPERATURE_MEV,
    )
    parser.add_argument(
        "--endpoint-guess",
        type=_parse_endpoint,
        default=DEFAULT_ENDPOINT_GUESS,
    )
    parser.add_argument(
        "--no-neighbor-retry",
        action="store_false",
        dest="neighbor_retry",
        default=DEFAULT_NEIGHBOR_RETRY,
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
    )
    parser.add_argument(
        "--interface-fraction-mode",
        choices=("extreme_endothermic",),
        default=DEFAULT_INTERFACE_FRACTION_MODE,
    )
    parser.add_argument(
        "--velocity-closure",
        choices=("numerical_I2", "closed_form"),
        default=DEFAULT_VELOCITY_CLOSURE,
        help=(
            "transport closure for A and I2; the 26-08-02 runner defaults to "
            "the conserved-trajectory numerical integral"
        ),
    )
    parser.add_argument(
        "--ncrit-over-n0",
        type=float,
        default=DEFAULT_NCRIT_OVER_N0,
    )
    parser.add_argument("--xi", type=float, default=DEFAULT_XI)
    parser.add_argument("--ms", type=float, default=DEFAULT_MS_MEV)
    parser.add_argument("--nm-type", default=DEFAULT_NM_TYPE)
    parser.add_argument("--upB", type=int, default=DEFAULT_UPB)
    parser.add_argument("--B14", "--b14", dest="B14", type=float, default=np.nan)
    parser.add_argument("--workers", type=int, default=_default_workers())
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not (
        0.0
        < args.density_min_over_n0
        < args.density_max_over_n0
    ):
        raise ValueError(
            "require 0 < density-min-over-n0 < density-max-over-n0"
        )
    if not 0.0 < args.t_min < args.t_max:
        raise ValueError("require 0 < t-min < t-max")
    if args.theta_count < 3 or args.radial_count < 2:
        raise ValueError("theta-count >= 3 and radial-count >= 2 are required")
    if not 0.0 < args.angular_buffer_fraction < 0.1:
        raise ValueError(
            "angular-buffer-fraction must satisfy 0 < value < 0.1"
        )
    if not 0.0 < args.boundary_buffer_fraction < 0.5:
        raise ValueError(
            "boundary-buffer-fraction must satisfy 0 < value < 0.5"
        )
    if args.endpoint_probe_count < 5 or args.ray_probe_count < 5:
        raise ValueError("boundary probe counts must be at least 5")
    if args.boundary_bisect_steps < 1:
        raise ValueError("boundary-bisect-steps must be positive")
    if args.seed_temperature <= 0.0:
        raise ValueError("seed-temperature must be positive")
    if args.save_every < 0 or args.workers < 1:
        raise ValueError(
            "save-every must be nonnegative and workers must be positive"
        )


def _output_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


def _empty_result_arrays(
    mesh: dict[str, np.ndarray],
    n0: float,
) -> dict[str, np.ndarray]:
    shape = mesh["nB_grid_over_n0"].shape
    arrays = {name: np.asarray(value) for name, value in mesh.items()}
    arrays["nB_grid_MeV3"] = arrays["nB_grid_over_n0"] * float(n0)
    arrays.update(
        {
            "velocity_grid_m_s": np.full(shape, np.nan),
            "u_0minus_grid": np.full(shape, np.nan),
            "jB_grid": np.full(shape, np.nan),
            "lambda_n_grid": np.full(shape, np.nan),
            "A_boundary_grid": np.full(shape, np.nan),
            "a_0plus_grid": np.full(shape, np.nan),
            "a_0plus_LTE_grid": np.full(shape, np.nan),
            "entropy_jump_grid": np.full(shape, np.nan),
            "I2_numerical_grid": np.full(shape, np.nan),
            "layer_trajectory_points_grid": np.zeros(shape, dtype=np.int32),
            "layer_trajectory_saturated_grid": np.zeros(shape, dtype=bool),
            "status_grid": np.full(
                shape,
                STATUS_UNSCANNED,
                dtype=np.int8,
            ),
        }
    )
    return arrays


def _grid_point_task(
    *,
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    row_index: int,
    col_index: int,
    direction: str,
    n0: float,
    B14: float,
    neighbor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    use_neighbor = bool(
        neighbor
        and neighbor.get("success")
        and _finite_pair(neighbor.get("endpoint_solution"))
    )
    endpoint = (
        _finite_pair(neighbor.get("endpoint_solution"))
        if use_neighbor
        else _finite_pair(args.endpoint_guess)
    )
    neighbor_muB = (
        float(neighbor.get("muB_0minus_MeV", np.nan))
        if use_neighbor
        else np.nan
    )
    muB_guess = neighbor_muB if np.isfinite(neighbor_muB) else 1100.0
    return {
        "row_index": int(row_index),
        "col_index": int(col_index),
        "theta_rad": float(arrays["theta_grid_rad"][row_index, col_index]),
        "theta_deg": float(
            np.degrees(arrays["theta_grid_rad"][row_index, col_index])
        ),
        "rho": float(arrays["rho_grid"][row_index, col_index]),
        "radial_fraction": float(
            arrays["radial_fraction_grid"][row_index, col_index]
        ),
        "nB_over_n0": float(
            arrays["nB_grid_over_n0"][row_index, col_index]
        ),
        "T_MeV": float(
            arrays["temperature_grid_MeV"][row_index, col_index]
        ),
        "n0_MeV3": float(n0),
        "B14_MeV": float(B14),
        "xi": float(args.xi),
        "ms_MeV": float(args.ms),
        "nm_type": str(args.nm_type),
        "upB": int(args.upB),
        "interface_fraction_mode": str(args.interface_fraction_mode),
        "velocity_closure": str(args.velocity_closure),
        "endpoint_guess": endpoint,
        "endpoint_guess_source": (
            f"same_ray_{direction}_neighbor" if use_neighbor else "default"
        ),
        "continuation_direction": str(direction),
        "muB_guess_MeV": muB_guess,
        "retry_attempt": 0,
        "retry_of": None,
    }


def _build_seed_point_tasks(
    *,
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    n0: float,
    B14: float,
) -> list[dict[str, Any]]:
    tasks = []
    for row_index in range(arrays["temperature_grid_MeV"].shape[0]):
        order = radial_solve_order(
            arrays["temperature_grid_MeV"][row_index],
            float(args.seed_temperature),
        )
        seed_index, _direction = order[0]
        tasks.append(
            _grid_point_task(
                args=args,
                arrays=arrays,
                row_index=row_index,
                col_index=seed_index,
                direction="seed",
                n0=n0,
                B14=B14,
            )
        )
    return tasks


def _continuation_successor_tasks(
    *,
    completed_task: dict[str, Any],
    completed_row: dict[str, Any],
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    n0: float,
    B14: float,
) -> list[dict[str, Any]]:
    row_index = int(completed_task["row_index"])
    col_index = int(completed_task["col_index"])
    direction = str(completed_task["continuation_direction"])
    column_count = arrays["temperature_grid_MeV"].shape[1]
    if direction == "seed":
        candidates = ((col_index + 1, "up"), (col_index - 1, "down"))
    elif direction == "up":
        candidates = ((col_index + 1, "up"),)
    elif direction == "down":
        candidates = ((col_index - 1, "down"),)
    else:
        return []
    return [
        _grid_point_task(
            args=args,
            arrays=arrays,
            row_index=row_index,
            col_index=next_index,
            direction=next_direction,
            n0=n0,
            B14=B14,
            neighbor=completed_row,
        )
        for next_index, next_direction in candidates
        if 0 <= next_index < column_count
    ]


def _retry_task(
    *,
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    records_by_cell: dict[tuple[int, int], dict[str, Any]],
    row_index: int,
    col_index: int,
    n0: float,
    B14: float,
) -> dict[str, Any] | None:
    candidates = (
        ((row_index, col_index - 1), "same_ray_lower_neighbor"),
        ((row_index, col_index + 1), "same_ray_upper_neighbor"),
        ((row_index - 1, col_index), "lower_theta_neighbor"),
        ((row_index + 1, col_index), "upper_theta_neighbor"),
    )
    neighbor = None
    source = ""
    for key, candidate_source in candidates:
        row = records_by_cell.get(key)
        if row and row.get("success") and _finite_pair(
            row.get("endpoint_solution")
        ):
            neighbor = row
            source = candidate_source
            break
    if neighbor is None:
        return None
    return {
        "row_index": int(row_index),
        "col_index": int(col_index),
        "theta_rad": float(arrays["theta_grid_rad"][row_index, col_index]),
        "theta_deg": float(
            np.degrees(arrays["theta_grid_rad"][row_index, col_index])
        ),
        "rho": float(arrays["rho_grid"][row_index, col_index]),
        "radial_fraction": float(
            arrays["radial_fraction_grid"][row_index, col_index]
        ),
        "nB_over_n0": float(
            arrays["nB_grid_over_n0"][row_index, col_index]
        ),
        "T_MeV": float(
            arrays["temperature_grid_MeV"][row_index, col_index]
        ),
        "n0_MeV3": float(n0),
        "B14_MeV": float(B14),
        "xi": float(args.xi),
        "ms_MeV": float(args.ms),
        "nm_type": str(args.nm_type),
        "upB": int(args.upB),
        "interface_fraction_mode": str(args.interface_fraction_mode),
        "velocity_closure": str(args.velocity_closure),
        "endpoint_guess": _finite_pair(neighbor["endpoint_solution"]),
        "endpoint_guess_source": f"retry_{source}",
        "continuation_direction": "retry",
        "muB_guess_MeV": float(neighbor.get("muB_0minus_MeV", 1100.0)),
        "retry_attempt": 1,
        "retry_of": (int(row_index), int(col_index)),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    output_path = _output_path(args.output)

    n0 = float(0.16 * const.MeV_fm**3)
    nB_crit0 = float(args.ncrit_over_n0 * n0)
    if np.isfinite(args.B14) and args.B14 > 0.0:
        B14 = float(args.B14)
        _, muB_crit0, pressure_crit0 = calibrate_b14(
            nB_crit0,
            args.ms,
            args.nm_type,
        )
    else:
        B14, muB_crit0, pressure_crit0 = calibrate_b14(
            nB_crit0,
            args.ms,
            args.nm_type,
        )
    calibration = {
        "B14_calibrated_MeV": float(B14),
        "muB_crit0_MeV": float(muB_crit0),
        "P_crit0_MeV4": float(pressure_crit0),
        "n0_MeV3": float(n0),
        "nB_crit0_MeV3": float(nB_crit0),
        "nB_crit0_over_n0": float(args.ncrit_over_n0),
    }

    transform = RayTransform(
        density_origin_over_n0=args.density_min_over_n0,
        density_scale_over_n0=(
            args.density_max_over_n0 - args.density_min_over_n0
        ),
        temperature_scale_MeV=args.t_max,
    )
    print(
        f"Calibrated B^(1/4) = {B14:.9f} MeV\n"
        f"Output: {output_path}\n"
        f"Run tag: {args.run_tag}\n"
        f"Hidden grid: {args.theta_count} theta rays x "
        f"{args.radial_count} safe radial fractions\n"
        f"Physical window: nB(0-)/n0={args.density_min_over_n0:g}-"
        f"{args.density_max_over_n0:g}, T={args.t_min:g}-{args.t_max:g} MeV\n"
        f"Velocity closure: {args.velocity_closure}\n"
        f"Workers: {args.workers}"
    )

    executor = (
        None
        if args.workers == 1
        else ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("spawn"),
        )
    )
    try:
        boundary_payload = solve_boundary_curves(
            transform=transform,
            theta_count=args.theta_count,
            density_min_over_n0=args.density_min_over_n0,
            density_max_over_n0=args.density_max_over_n0,
            temperature_min_MeV=args.t_min,
            temperature_max_MeV=args.t_max,
            n0=n0,
            B14=B14,
            xi=args.xi,
            ms=args.ms,
            nm_type=args.nm_type,
            upB=args.upB,
            interface_fraction_mode=args.interface_fraction_mode,
            velocity_closure=args.velocity_closure,
            endpoint_probe_count=args.endpoint_probe_count,
            ray_probe_count=args.ray_probe_count,
            bisection_steps=args.boundary_bisect_steps,
            angular_buffer_fraction=args.angular_buffer_fraction,
            progress_factory=tqdm,
            executor=executor,
        )
    except BaseException:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        raise
    eta_axis = chebyshev_open_axis(
        args.radial_count,
        args.boundary_buffer_fraction,
    )
    mesh = build_safe_ray_mesh(
        transform=transform,
        theta_axis_rad=boundary_payload["theta_axis_rad"],
        radial_fraction_axis=eta_axis,
        rho_inner=boundary_payload["rho_inner"],
        rho_outer=boundary_payload["rho_outer"],
    )
    arrays = _empty_result_arrays(mesh, n0)
    records: list[dict[str, Any]] = []
    records_by_cell: dict[tuple[int, int], dict[str, Any]] = {}
    failures: set[tuple[int, int]] = set()
    completed_since_save = 0

    def save() -> None:
        payload = build_payload(
            args=args,
            transform=transform,
            arrays=arrays,
            records=records,
            boundary_payload=boundary_payload,
            calibration=calibration,
        )
        atomic_save(output_path, payload)

    def accept(row: dict[str, Any], first_pass: bool) -> None:
        nonlocal completed_since_save
        i = int(row["row_index"])
        j = int(row["col_index"])
        records.append(row)
        records_by_cell[(i, j)] = row
        if row.get("success"):
            failures.discard((i, j))
            arrays["status_grid"][i, j] = STATUS_FINITE
            arrays["velocity_grid_m_s"][i, j] = float(row["u_0minus_m_s"])
            arrays["u_0minus_grid"][i, j] = float(row["u_0minus"])
            arrays["jB_grid"][i, j] = float(row["jB"])
            arrays["lambda_n_grid"][i, j] = float(row["lambda_n"])
            arrays["A_boundary_grid"][i, j] = float(row["A_boundary"])
            arrays["a_0plus_grid"][i, j] = float(row["a_0plus"])
            arrays["a_0plus_LTE_grid"][i, j] = float(
                row["a_0plus_LTE"]
            )
            arrays["entropy_jump_grid"][i, j] = float(
                row["entropy_jump_per_baryon"]
            )
            arrays["I2_numerical_grid"][i, j] = float(row["I2_numerical"])
            arrays["layer_trajectory_points_grid"][i, j] = int(
                row["layer_trajectory_points"]
            )
            arrays["layer_trajectory_saturated_grid"][i, j] = bool(
                row["layer_trajectory_saturated"]
            )
        else:
            arrays["status_grid"][i, j] = STATUS_NUMERICAL_FAILURE
            if first_pass:
                failures.add((i, j))
        completed_since_save += 1
        if args.save_every and completed_since_save >= args.save_every:
            save()
            completed_since_save = 0

    save()
    seed_tasks = _build_seed_point_tasks(
        args=args,
        arrays=arrays,
        n0=n0,
        B14=B14,
    )
    total = args.theta_count * args.radial_count
    progress = tqdm(
        total=total,
        desc="Analytical elliptical-ray contour",
        unit="pt",
        dynamic_ncols=True,
    )
    try:
        if executor is None:
            pending_tasks = list(seed_tasks)
            while pending_tasks:
                task = pending_tasks.pop(0)
                row = calculate_absolute_density_point(task)
                accept(row, first_pass=True)
                pending_tasks.extend(
                    _continuation_successor_tasks(
                        completed_task=task,
                        completed_row=row,
                        args=args,
                        arrays=arrays,
                        n0=n0,
                        B14=B14,
                    )
                )
                progress.update(1)
                progress.set_postfix_str(
                    f"theta={task['theta_deg']:.3g} deg, T={task['T_MeV']:.3g}"
                )
        else:
            pending = {
                executor.submit(calculate_absolute_density_point, task): task
                for task in seed_tasks
            }
            while pending:
                completed, _waiting = wait(
                    tuple(pending),
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    task = pending.pop(future)
                    row = future.result()
                    accept(row, first_pass=True)
                    for successor in _continuation_successor_tasks(
                        completed_task=task,
                        completed_row=row,
                        args=args,
                        arrays=arrays,
                        n0=n0,
                        B14=B14,
                    ):
                        pending[
                            executor.submit(
                                calculate_absolute_density_point,
                                successor,
                            )
                        ] = successor
                    progress.update(1)
                    progress.set_postfix_str(
                        f"theta={task['theta_deg']:.3g} deg, "
                        f"T={task['T_MeV']:.3g}"
                    )

        if args.neighbor_retry and failures:
            retry_tasks = []
            for i, j in sorted(failures):
                task = _retry_task(
                    args=args,
                    arrays=arrays,
                    records_by_cell=records_by_cell,
                    row_index=i,
                    col_index=j,
                    n0=n0,
                    B14=B14,
                )
                if task is not None:
                    retry_tasks.append(task)
            if retry_tasks:
                progress.total += len(retry_tasks)
                progress.refresh()
                if executor is None:
                    retry_results = map(
                        calculate_absolute_density_point,
                        retry_tasks,
                    )
                    for row in retry_results:
                        accept(row, first_pass=False)
                        progress.update(1)
                else:
                    retry_futures = {
                        executor.submit(
                            calculate_absolute_density_point,
                            task,
                        )
                        for task in retry_tasks
                    }
                    while retry_futures:
                        completed, retry_futures = wait(
                            retry_futures,
                            return_when=FIRST_COMPLETED,
                        )
                        for future in completed:
                            accept(future.result(), first_pass=False)
                            progress.update(1)
    finally:
        progress.close()
        save()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    finite = int(np.count_nonzero(arrays["status_grid"] == STATUS_FINITE))
    failed = int(
        np.count_nonzero(
            arrays["status_grid"] == STATUS_NUMERICAL_FAILURE
        )
    )
    print(
        f"Done. finite={finite}, failure={failed}, "
        f"saved records={len(records)}\nSaved {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
