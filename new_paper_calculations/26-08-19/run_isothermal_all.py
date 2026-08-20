#!/usr/bin/env python3
"""Run the complete isothermal contour workflow stage by stage."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def _ensure_project_runtime() -> None:
    required = ("numpy", "scipy", "tqdm", "matplotlib", "RMFsolver")
    project_root = Path(__file__).resolve().parents[2]
    project_python = project_root / "bin" / "python3"
    if not project_python.exists():
        raise RuntimeError(f"The project Python runtime is unavailable: {project_python}")
    if Path(sys.prefix).resolve() != project_root.resolve():
        os.execv(
            str(project_python),
            [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        )
    missing = [
        package for package in required if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise RuntimeError(
            "The project Python runtime is missing packages: " + ", ".join(missing)
        )


_ensure_project_runtime()

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import _isothermal_domain as domain
import _isothermal_execution as execution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--workers", type=int, default=execution.discover_workers())
    parser.add_argument("--B-one-forth", type=float, default=domain.DEFAULT_B_ONE_FORTH_MEV)
    parser.add_argument("--xi", type=float, default=domain.DEFAULT_XI)
    parser.add_argument("--ms", type=float, default=domain.DEFAULT_MS_MEV)
    parser.add_argument("--NM-type", default=domain.DEFAULT_NM_TYPE)
    parser.add_argument("--upB", type=int, default=domain.DEFAULT_UPB)
    parser.add_argument("--analytic-timeout", type=float, default=execution.DEFAULT_ANALYTIC_TIMEOUT_S)
    parser.add_argument("--trial-timeout", type=float, default=execution.DEFAULT_TRIAL_TIMEOUT_S)
    parser.add_argument("--cell-timeout", type=float, default=execution.DEFAULT_CELL_TIMEOUT_S)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def build_stage_commands(args: argparse.Namespace) -> list[list[str]]:
    domain.validate_cluster_physics(
        ms=args.ms, NM_type=args.NM_type, upB=args.upB
    )
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    domain_path = output_dir / "isothermal-domain.npy"
    analytic_path = output_dir / "isothermal-analytic.npy"
    numerical_path = output_dir / "isothermal-numerical.npy"
    figure_path = output_dir / "isothermal-contours.png"
    prepare = [
        sys.executable,
        str(script_dir / "prepare_isothermal_domain.py"),
        "--output", str(domain_path),
        "--workers", str(args.workers),
        "--B-one-forth", str(args.B_one_forth),
        "--xi", str(args.xi),
        "--ms", str(args.ms),
        "--NM-type", str(args.NM_type),
        "--upB", str(args.upB),
    ]
    analytic = [
        sys.executable,
        str(script_dir / "run_isothermal_analytic.py"),
        "--domain", str(domain_path),
        "--output", str(analytic_path),
        "--workers", str(args.workers),
        "--timeout", str(args.analytic_timeout),
    ]
    numerical = [
        sys.executable,
        str(script_dir / "run_isothermal_numerical.py"),
        "--domain", str(domain_path),
        "--analytic", str(analytic_path),
        "--output", str(numerical_path),
        "--workers", str(args.workers),
        "--trial-timeout", str(args.trial_timeout),
        "--cell-timeout", str(args.cell_timeout),
    ]
    plot = [
        sys.executable,
        str(script_dir / "plot_isothermal_contours.py"),
        "--domain", str(domain_path),
        "--analytic", str(analytic_path),
        "--numerical", str(numerical_path),
        "--output", str(figure_path),
    ]
    if args.smoke:
        prepare.append("--smoke")
    if args.resume:
        prepare.append("--resume")
        analytic.append("--resume")
        numerical.append("--resume")
    return [prepare, analytic, numerical, plot]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Using {args.workers} CPU workers", flush=True)
    for command in build_stage_commands(args):
        print(f"Running {Path(command[1]).name}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
