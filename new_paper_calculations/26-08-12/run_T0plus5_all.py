#!/usr/bin/env python3
"""Run the complete fixed-T(0+)=5 MeV contour workflow sequentially."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from _T0plus5_runner import analytical_main, numerical_main, prepare_domain_main


# Tunable run parameters
DEFAULT_OUTPUT_PREFIX = "T0plus5-all-zero-upstream-v1"
DEFAULT_WORKERS = 64
DEFAULT_T0PLUS_MEV = 5.0
DEFAULT_NCRIT_OVER_N0 = 3.0
DEFAULT_B14_MEV = float("nan")
DEFAULT_BOUNDARY_THETA_COUNT = 12
DEFAULT_CONTOUR_RAY_COUNT = 30
DEFAULT_RADIAL_FRACTIONS = (
    0.02, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.66, 0.72, 0.78, 0.84, 0.90, 0.95, 0.99,
)
DEFAULT_BRANCH_CHECK_RADIAL_FRACTION = 0.55
DEFAULT_ANGULAR_BUFFER_FRACTION = 0.0
DEFAULT_RADIAL_BUFFER_FRACTION = 0.0
DEFAULT_DENSITY_MIN_OVER_N0 = 1.0
DEFAULT_DENSITY_MAX_OVER_N0 = 5.5
DEFAULT_TEMPERATURE_MIN_MEV = 1.0e-2
DEFAULT_TEMPERATURE_MAX_MEV = 120.0
DEFAULT_RAY_PROBE_COUNT = 15
DEFAULT_BISECTION_STEPS = 10
DEFAULT_XI = -0.5
DEFAULT_MS_MEV = 0.0
DEFAULT_NM_TYPE = "PNM"
DEFAULT_UPB = 5000
DEFAULT_SAVE_EVERY_ANALYTICAL = 10
DEFAULT_SAVE_EVERY_NUMERICAL = 1
DEFAULT_TAIL_EPS = 1.0e-8
DEFAULT_N_MESH = 200
DEFAULT_TOL_BVP = 1.0e-4
DEFAULT_MAX_NODES = 10000
DEFAULT_JB_LOWER_BOUND = 1.0e-12
DEFAULT_JB_UPPER_FACTOR = 1.0e-4
DEFAULT_POINT_TIMEOUT_S = 600.0
DEFAULT_TARGET_TIMEOUT_S = 1800.0
DEFAULT_MAX_BRIDGE_DEPTH = 5
DEFAULT_MIN_BRIDGE_FRACTION_STEP = 1.0e-3


SCRIPT_DIR = Path(__file__).resolve().parent


def output_bundle(directory: str | Path, prefix: str) -> dict[str, Path]:
    """Choose one unused suffix shared by every output in this workflow."""
    directory = Path(directory).expanduser().resolve()
    prefix = str(prefix).strip()
    if not prefix:
        raise ValueError("output prefix cannot be empty")
    suffixes = {
        "domain": "domain.npy",
        "full": "full-analytic.npy",
        "semi": "semi-analytic.npy",
        "numerical": "uNmax-radial-continuation.npy",
    }
    run_index = 0
    while True:
        run_suffix = "" if run_index == 0 else f"__run{run_index:02d}"
        paths = {
            key: directory / f"{prefix}{run_suffix}-{suffix}"
            for key, suffix in suffixes.items()
        }
        if not any(path.exists() for path in paths.values()):
            return paths
        run_index += 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially prepare the T(0+)=5 MeV domain and run the full "
            "analytical, semi-analytical, and numerical contours."
        )
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output-directory", default=str(SCRIPT_DIR))
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--point-timeout", type=float, default=DEFAULT_POINT_TIMEOUT_S)
    parser.add_argument("--target-timeout", type=float, default=DEFAULT_TARGET_TIMEOUT_S)
    return parser


def _run_stage(label: str, function, *args) -> None:
    started = time.monotonic()
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    status = function(*args)
    if int(status) != 0:
        raise RuntimeError(f"{label} returned status {status}")
    print(f"{label} completed in {(time.monotonic() - started) / 60.0:.2f} min", flush=True)


def _require_output(label: str, path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} did not create expected output {path}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if int(args.workers) < 1:
        raise ValueError("workers must be positive")
    if float(args.point_timeout) <= 0.0 or float(args.target_timeout) <= 0.0:
        raise ValueError("timeouts must be positive")

    output_directory = Path(args.output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = output_bundle(output_directory, args.output_prefix)
    workers = str(int(args.workers))

    print("Fixed interface temperature: T(0+) = 5 MeV", flush=True)
    print("Execution: four sequential stages using one worker pool at a time", flush=True)
    print(f"Workers per stage: {workers}", flush=True)
    for name, path in outputs.items():
        print(f"  {name}: {path}", flush=True)

    prepare_args = [
        "--output", str(outputs["domain"]),
        "--run-tag", "T0plus5_integrated_domain_zero_upstream_v1",
        "--workers", workers,
        "--T0plus-values", f"{DEFAULT_T0PLUS_MEV:g}",
        "--boundary-theta-count", str(DEFAULT_BOUNDARY_THETA_COUNT),
        "--ray-count", str(DEFAULT_CONTOUR_RAY_COUNT),
        "--radial-fractions", ",".join(f"{value:g}" for value in DEFAULT_RADIAL_FRACTIONS),
        "--branch-check-radial-fraction", f"{DEFAULT_BRANCH_CHECK_RADIAL_FRACTION:g}",
        "--angular-buffer-fraction", f"{DEFAULT_ANGULAR_BUFFER_FRACTION:g}",
        "--radial-buffer-fraction", f"{DEFAULT_RADIAL_BUFFER_FRACTION:g}",
        "--density-min-over-n0", f"{DEFAULT_DENSITY_MIN_OVER_N0:g}",
        "--density-max-over-n0", f"{DEFAULT_DENSITY_MAX_OVER_N0:g}",
        "--temperature-min", f"{DEFAULT_TEMPERATURE_MIN_MEV:g}",
        "--temperature-max", f"{DEFAULT_TEMPERATURE_MAX_MEV:g}",
        "--ray-probe-count", str(DEFAULT_RAY_PROBE_COUNT),
        "--bisection-steps", str(DEFAULT_BISECTION_STEPS),
        "--xi", f"{DEFAULT_XI:g}",
        "--ms", f"{DEFAULT_MS_MEV:g}",
        "--nm-type", DEFAULT_NM_TYPE,
        "--upB", str(DEFAULT_UPB),
        "--ncrit-over-n0", f"{DEFAULT_NCRIT_OVER_N0:g}",
        "--b14", f"{DEFAULT_B14_MEV:g}",
    ]
    _run_stage("Stage 1/4: shared safe-domain boundaries", prepare_domain_main, prepare_args)
    _require_output("domain stage", outputs["domain"])

    analytical_common = [
        "--domain-file", str(outputs["domain"]),
        "--workers", workers,
        "--save-every", str(DEFAULT_SAVE_EVERY_ANALYTICAL),
    ]
    _run_stage(
        "Stage 2/4: full analytical contour",
        analytical_main,
        "analytic",
        [
            *analytical_common,
            "--output", str(outputs["full"]),
            "--run-tag", "T0plus5_integrated_full_analytic_v1",
        ],
    )
    _require_output("full analytical stage", outputs["full"])

    _run_stage(
        "Stage 3/4: semi-analytical contour and numerical seeds",
        analytical_main,
        "semi_analytic",
        [
            *analytical_common,
            "--output", str(outputs["semi"]),
            "--run-tag", "T0plus5_integrated_semi_analytic_v1",
        ],
    )
    _require_output("semi-analytical stage", outputs["semi"])

    numerical_args = [
        "--domain-file", str(outputs["domain"]),
        "--output", str(outputs["numerical"]),
        "--run-tag", "T0plus5_integrated_uNmax_radial_v1",
        "--workers", workers,
        "--save-every", str(DEFAULT_SAVE_EVERY_NUMERICAL),
        "--semi-results", str(outputs["semi"]),
        "--tail-eps", f"{DEFAULT_TAIL_EPS:g}",
        "--n-mesh", str(DEFAULT_N_MESH),
        "--tol-bvp", f"{DEFAULT_TOL_BVP:g}",
        "--max-nodes", str(DEFAULT_MAX_NODES),
        "--jB-lower-bound", f"{DEFAULT_JB_LOWER_BOUND:g}",
        "--jB-upper-factor", f"{DEFAULT_JB_UPPER_FACTOR:g}",
        "--point-timeout", f"{float(args.point_timeout):g}",
        "--target-timeout", f"{float(args.target_timeout):g}",
        "--max-bridge-depth", str(DEFAULT_MAX_BRIDGE_DEPTH),
        "--min-bridge-fraction-step", f"{DEFAULT_MIN_BRIDGE_FRACTION_STEP:g}",
        "--skip-branch-check",
    ]
    _run_stage("Stage 4/4: numerical uNmax contour", numerical_main, numerical_args)
    _require_output("numerical stage", outputs["numerical"])

    print("\nAll four stages completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
