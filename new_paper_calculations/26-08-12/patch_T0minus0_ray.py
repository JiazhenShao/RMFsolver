#!/usr/bin/env python3
"""Patch the missing exact-T(0-)=0 ray into copied 26-08-10 payloads."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import _T0plus5_runner as runner
import _method_engine as benchmark
import _radial_continuation as radial


# Tunable local patch parameters
DEFAULT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "26-08-10"
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_DIR
DEFAULT_DOMAIN_FILE = "T0plus5-domain.npy"
DEFAULT_FULL_ANALYTIC_FILE = "T0plus5-full-analytic.npy"
DEFAULT_SEMI_ANALYTIC_FILE = "T0plus5-semi-analytic.npy"
DEFAULT_NUMERICAL_FILE = "T0plus5-uNmax-radial-continuation__run01.npy"
DEFAULT_OUTPUT_SUFFIX = "-T0minus0-local-patched"
DEFAULT_WORKERS = min(8, max(1, (mp.cpu_count() or 2) - 1))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a dictionary payload")
    return payload


def _patched_path(directory: Path, source_name: str, suffix: str) -> Path:
    source = Path(source_name)
    return directory / f"{source.stem}{suffix}{source.suffix}"


def _replace_point_records(
    original: list[dict[str, Any]],
    replacement: list[dict[str, Any]],
    point_indices: set[int],
    *,
    sort_key,
) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in original
        if int(row.get("point_index", -1)) not in point_indices
    ]
    rows.extend(dict(row) for row in replacement)
    rows.sort(key=sort_key)
    return rows


def _theta_zero_points(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    points = [
        dict(point)
        for point in manifest["points"]
        if int(point["theta_index"]) == 0
    ]
    points.sort(key=lambda point: int(point["radial_index"]))
    if not points:
        raise RuntimeError("the domain has no theta=0 ray")
    if not all(np.isclose(float(point["theta_rad"]), 0.0) for point in points):
        raise RuntimeError("theta_index=0 is not the exact theta=0 ray")
    T_values = tuple(float(value) for value in manifest["T_0plus_values_MeV"])
    if len(T_values) != 1 or not np.isclose(T_values[0], 5.0):
        raise RuntimeError("the local patch expects exactly T(0+)=5 MeV")
    for point in points:
        state = runner._states_for_point(point, T_values)[0]
        if not np.isclose(float(state["T_0minus_MeV"]), 0.0):
            raise RuntimeError("theta=0 does not map to exact T(0-)=0")
    return points


def _analytical_tasks(
    manifest: dict[str, Any],
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    calibration = manifest["calibration"]
    common = runner._task_common(manifest)
    T_values = tuple(float(value) for value in manifest["T_0plus_values_MeV"])
    tasks: list[dict[str, Any]] = []
    for method in ("analytic", "semi_analytic"):
        for point in points:
            state = runner._states_for_point(point, T_values)[0]
            tasks.append(
                {
                    **common,
                    "method": method,
                    "point_index": int(point["point_index"]),
                    "T_0minus_MeV": 0.0,
                    "nB_0minus_MeV3": float(
                        state["nB_0minus_over_n0"] * calibration["n0_MeV3"]
                    ),
                    "T_0plus_values": T_values,
                }
            )
    return tasks


def _solve_analytical_rows(
    manifest: dict[str, Any],
    points: list[dict[str, Any]],
    workers: int,
) -> dict[str, list[dict[str, Any]]]:
    tasks = _analytical_tasks(manifest, points)
    rows = {"analytic": [], "semi_analytic": []}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max(1, min(int(workers), len(tasks))),
        mp_context=context,
    ) as executor, tqdm(
        total=len(tasks),
        desc="Local theta=0 analytical cells",
        unit="cell",
        dynamic_ncols=True,
    ) as progress:
        futures = {
            executor.submit(benchmark.evaluate_selected_analytical_point, task): task
            for task in tasks
        }
        for future in as_completed(futures):
            output = future.result()
            rows[str(output["method"])].extend(output["rows"])
            progress.update(1)
            row = output["rows"][0]
            progress.set_postfix_str(
                f"{output['method']}, p={row['point_index']}, "
                f"{'ok' if row['success'] else 'failed'}"
            )
    for method_rows in rows.values():
        method_rows.sort(key=lambda row: int(row["point_index"]))
    return rows


def _patch_metadata(
    payload: dict[str, Any],
    *,
    source_path: Path,
    point_indices: set[int],
) -> None:
    payload["local_T0minus0_patch"] = {
        "created_at_utc": _utc_now(),
        "source_cluster_payload": str(source_path.resolve()),
        "theta_index": 0,
        "T_0minus_MeV": 0.0,
        "point_indices": np.asarray(sorted(point_indices), dtype=int),
        "point_count": int(len(point_indices)),
        "phase_velocity_supports_exact_zero_T_0minus": True,
    }


def _patch_analytical_payload(
    *,
    method: str,
    source_path: Path,
    output_path: Path,
    source_payload: dict[str, Any],
    manifest: dict[str, Any],
    replacement_rows: list[dict[str, Any]],
    point_indices: set[int],
) -> dict[str, Any]:
    if not all(bool(row.get("success", False)) for row in replacement_rows):
        failures = [
            (row["point_index"], row.get("message", ""))
            for row in replacement_rows
            if not bool(row.get("success", False))
        ]
        raise RuntimeError(f"{method} theta=0 patch contains failures: {failures}")
    rows = _replace_point_records(
        list(source_payload.get("records", [])),
        replacement_rows,
        point_indices,
        sort_key=lambda row: (
            int(row["point_index"]),
            -float(row["T_0plus_MeV"]),
        ),
    )
    payload = runner.build_analytical_payload(
        method=method,
        args=Namespace(**source_payload["arguments"]),
        manifest=manifest,
        rows=rows,
        started_at=str(source_payload.get("started_at_utc", _utc_now())),
        run_complete=bool(source_payload.get("run_complete", True)),
    )
    _patch_metadata(
        payload,
        source_path=source_path,
        point_indices=point_indices,
    )
    benchmark.atomic_save(output_path, payload)
    return payload


class _LocalProgress:
    def __init__(self, total: int):
        self.progress = tqdm(
            total=int(total),
            desc="Local theta=0 numerical ray",
            unit="cell",
            dynamic_ncols=True,
        )

    def put(self, event: dict[str, Any]) -> None:
        if event.get("event") != "target_completed":
            return
        self.progress.update(1)
        self.progress.set_postfix_str(
            f"p={event['point_index']}, "
            f"{'ok' if event['success'] else 'failed'}, "
            f"attempts={event['attempt_count']}"
        )

    def close(self) -> None:
        self.progress.close()


def _solve_numerical_ray(
    manifest: dict[str, Any],
    points: list[dict[str, Any]],
    semi_rows: list[dict[str, Any]],
    numerical_args: Namespace,
) -> dict[str, Any]:
    T_values = tuple(float(value) for value in manifest["T_0plus_values_MeV"])
    T_0plus = T_values[0]
    calibration = manifest["calibration"]
    common = runner._task_common(manifest)
    seeds = runner.semi_seed_map(semi_rows)
    enriched = []
    for point in points:
        state = runner._states_for_point(point, T_values)[0]
        enriched.append(
            {
                **point,
                **state,
                "T_0minus_MeV": 0.0,
                "nB_0minus_MeV3": float(
                    state["nB_0minus_over_n0"] * calibration["n0_MeV3"]
                ),
                "semi_jB": seeds.get(int(point["point_index"]), {}).get(
                    T_0plus, np.nan
                ),
            }
        )
    progress = _LocalProgress(len(enriched))
    try:
        return radial.run_radial_trajectory(
            {
                **common,
                "theta_index": 0,
                "theta_rad": 0.0,
                "points": enriched,
                "T_0plus_MeV": T_0plus,
                "tail_eps": float(numerical_args.tail_eps),
                "n_mesh": int(numerical_args.n_mesh),
                "tol_bvp": float(numerical_args.tol_bvp),
                "max_nodes": int(numerical_args.max_nodes),
                "jB_lower_bound": float(numerical_args.jB_lower_bound),
                "jB_upper_factor": float(numerical_args.jB_upper_factor),
                "point_timeout_s": float(numerical_args.point_timeout),
                "target_timeout_s": float(numerical_args.target_timeout),
                "max_bridge_depth": int(numerical_args.max_bridge_depth),
                "min_bridge_fraction_step": float(
                    numerical_args.min_bridge_fraction_step
                ),
                "progress_queue": progress,
            }
        )
    finally:
        progress.close()


def _patch_numerical_payload(
    *,
    source_path: Path,
    output_path: Path,
    source_payload: dict[str, Any],
    manifest: dict[str, Any],
    semi_rows: list[dict[str, Any]],
    ray_output: dict[str, Any],
    point_indices: set[int],
) -> dict[str, Any]:
    target_records = _replace_point_records(
        list(source_payload.get("numerical_outward_records", [])),
        list(ray_output["target_records"]),
        point_indices,
        sort_key=lambda row: int(row["point_index"]),
    )
    attempt_records = [
        dict(row)
        for row in source_payload.get("radial_attempt_records", [])
        if int(row.get("theta_index", -1)) != 0
    ]
    attempt_records.extend(dict(row) for row in ray_output["attempt_records"])
    bridge_records = [
        dict(row)
        for row in source_payload.get("radial_bridge_records", [])
        if int(row.get("theta_index", -1)) != 0
    ]
    bridge_records.extend(dict(row) for row in ray_output["bridge_records"])
    profiles = dict(source_payload.get("radial_continuation_profiles", {}))
    for point_index in point_indices:
        profiles.pop(point_index, None)
    profiles.update(ray_output["profiles"])
    seed_rows = _replace_point_records(
        list(source_payload.get("semi_analytic_seed_records", [])),
        semi_rows,
        point_indices,
        sort_key=lambda row: int(row["point_index"]),
    )
    completed_rays = {
        int(value)
        for value in np.asarray(
            source_payload.get("completed_ray_indices", []), dtype=int
        )
    }
    completed_rays.add(0)
    payload = runner.build_radial_numerical_payload(
        args=Namespace(**source_payload["arguments"]),
        manifest=manifest,
        seed_rows=seed_rows,
        target_records=target_records,
        attempt_records=attempt_records,
        bridge_records=bridge_records,
        profiles=profiles,
        completed_ray_indices=completed_rays,
        started_at=str(source_payload.get("started_at_utc", _utc_now())),
        run_complete=bool(source_payload.get("run_complete", True)),
    )
    _patch_metadata(
        payload,
        source_path=source_path,
        point_indices=point_indices,
    )
    benchmark.atomic_save(output_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve the exact-T(0-)=0 ray locally and patch copied payloads."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--domain-file", default=DEFAULT_DOMAIN_FILE)
    parser.add_argument("--full-analytic-file", default=DEFAULT_FULL_ANALYTIC_FILE)
    parser.add_argument("--semi-analytic-file", default=DEFAULT_SEMI_ANALYTIC_FILE)
    parser.add_argument("--numerical-file", default=DEFAULT_NUMERICAL_FILE)
    parser.add_argument("--output-suffix", default=DEFAULT_OUTPUT_SUFFIX)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--point-timeout",
        type=float,
        help="Override the source payload's hard wall-clock limit per BVP trial.",
    )
    parser.add_argument(
        "--target-timeout",
        type=float,
        help="Override the source payload's total wall-clock budget per cell.",
    )
    parser.add_argument(
        "--max-bridge-depth",
        type=int,
        help="Override recursive midpoint bridging for the local repair ray.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_path = source_dir / args.domain_file
    full_path = source_dir / args.full_analytic_file
    semi_path = source_dir / args.semi_analytic_file
    numerical_path = source_dir / args.numerical_file
    manifest = _load_payload(domain_path)
    runner.benchmark.validate_domain_manifest(manifest)
    points = _theta_zero_points(manifest)
    point_indices = {int(point["point_index"]) for point in points}

    print(
        f"Patching {len(points)} exact-T(0-)=0 cells from {source_dir}",
        flush=True,
    )
    analytical_rows = _solve_analytical_rows(manifest, points, args.workers)
    full_source = _load_payload(full_path)
    semi_source = _load_payload(semi_path)
    full_output_path = _patched_path(
        output_dir, args.full_analytic_file, args.output_suffix
    )
    semi_output_path = _patched_path(
        output_dir, args.semi_analytic_file, args.output_suffix
    )
    full_payload = _patch_analytical_payload(
        method="analytic",
        source_path=full_path,
        output_path=full_output_path,
        source_payload=full_source,
        manifest=manifest,
        replacement_rows=analytical_rows["analytic"],
        point_indices=point_indices,
    )
    semi_payload = _patch_analytical_payload(
        method="semi_analytic",
        source_path=semi_path,
        output_path=semi_output_path,
        source_payload=semi_source,
        manifest=manifest,
        replacement_rows=analytical_rows["semi_analytic"],
        point_indices=point_indices,
    )
    print(f"Saved patched analytical payloads:\n  {full_output_path}\n  {semi_output_path}")

    numerical_source = _load_payload(numerical_path)
    numerical_args = Namespace(**numerical_source["arguments"])
    if args.point_timeout is not None:
        numerical_args.point_timeout = float(args.point_timeout)
    if args.target_timeout is not None:
        numerical_args.target_timeout = float(args.target_timeout)
    if args.max_bridge_depth is not None:
        numerical_args.max_bridge_depth = int(args.max_bridge_depth)
    ray_output = _solve_numerical_ray(
        manifest,
        points,
        list(semi_payload["records"]),
        numerical_args,
    )
    numerical_output_path = _patched_path(
        output_dir, args.numerical_file, args.output_suffix
    )
    numerical_payload = _patch_numerical_payload(
        source_path=numerical_path,
        output_path=numerical_output_path,
        source_payload=numerical_source,
        manifest=manifest,
        semi_rows=analytical_rows["semi_analytic"],
        ray_output=ray_output,
        point_indices=point_indices,
    )
    numerical_success = np.asarray(
        numerical_payload["numerical_outward_success_grid"], dtype=bool
    )[0, :, 0]
    print(
        f"Saved patched numerical payload:\n  {numerical_output_path}\n"
        f"Exact-T(0-)=0 numerical successes: "
        f"{int(np.count_nonzero(numerical_success))}/{numerical_success.size}",
        flush=True,
    )
    if not np.all(np.asarray(full_payload["success_grid"])[0, :, 0]):
        raise RuntimeError("full analytical theta=0 row is incomplete after patching")
    if not np.all(np.asarray(semi_payload["success_grid"])[0, :, 0]):
        raise RuntimeError("semi-analytical theta=0 row is incomplete after patching")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
