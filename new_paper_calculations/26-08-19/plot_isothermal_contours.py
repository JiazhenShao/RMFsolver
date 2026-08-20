#!/usr/bin/env python3
"""Plot analytical and numerical isothermal front-speed contours."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-isothermal-contour")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from RMFsolver import constants as const

import _isothermal_domain as domain


DEFAULT_LEVELS_M_S = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)


def masked_velocity(payload: dict[str, Any]) -> np.ma.MaskedArray:
    values = np.asarray(payload["velocity_m_s_grid"], dtype=float)
    statuses = np.asarray(payload["task_status"], dtype=object)
    return np.ma.array(
        values,
        mask=(statuses != "success") | ~np.isfinite(values) | (values <= 0.0),
    )


def _assert_plot_compatible(
    domain_payload: dict[str, Any], method_payload: dict[str, Any]
) -> None:
    domain.assert_compatible_payload(domain_payload, method_payload)
    if not np.array_equal(
        np.asarray(domain_payload["nB_0minus_grid"]),
        np.asarray(method_payload["nB_0minus_grid"]),
        equal_nan=True,
    ):
        raise RuntimeError("Method payload density grid differs from the domain")


def _contiguous_valid_slices(mask: np.ndarray) -> list[slice]:
    indexes = np.flatnonzero(mask)
    if indexes.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indexes) > 1) + 1
    groups = np.split(indexes, breaks)
    return [slice(int(group[0]), int(group[-1]) + 1) for group in groups]


def plot_comparison(
    domain_payload: dict[str, Any],
    analytic_payload: dict[str, Any],
    numerical_payload: dict[str, Any],
    *,
    output_path: str | Path,
    levels_m_s: tuple[float, ...] = DEFAULT_LEVELS_M_S,
) -> tuple[plt.Figure, np.ndarray]:
    _assert_plot_compatible(domain_payload, analytic_payload)
    _assert_plot_compatible(domain_payload, numerical_payload)
    temperatures = np.asarray(domain_payload["temperature_axis_MeV"], dtype=float)
    n0 = float(const.NuclearDensity_nucleons_MeV3)
    density_grid = np.asarray(domain_payload["nB_0minus_grid"], dtype=float) / n0
    temperature_grid = np.broadcast_to(temperatures[:, None], density_grid.shape)
    phase_temperature = np.asarray(
        domain_payload["phase_temperature_axis_MeV"], dtype=float
    )
    lower = np.asarray(domain_payload["lower_phase_nB_0minus"], dtype=float) / n0
    upper = np.asarray(domain_payload["upper_phase_nB_0minus"], dtype=float) / n0
    finite_phase = np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
    if not np.any(finite_phase):
        raise RuntimeError("No ordered finite phase-boundary points are available")
    x_max = 1.04 * float(np.nanmax(upper[finite_phase]))
    x_min = max(0.0, 0.96 * float(np.nanmin(lower[finite_phase])))

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharex=True, sharey=True)
    panels = (
        (analytic_payload, "Analytical isothermal velocity"),
        (numerical_payload, "Numerical isothermal velocity"),
    )
    for axis, (payload, title) in zip(axes, panels):
        velocity = masked_velocity(payload)
        for segment_index, segment in enumerate(_contiguous_valid_slices(finite_phase)):
            labels = segment_index == 0
            axis.fill_betweenx(
                phase_temperature[segment],
                x_min,
                lower[segment],
                color="0.78",
                label="Stable PNM" if labels else None,
                zorder=0,
            )
            axis.fill_betweenx(
                phase_temperature[segment],
                upper[segment],
                x_max,
                color="0.68",
                label="a(0+) >= 1" if labels else None,
                zorder=0,
            )
            axis.plot(
                lower[segment],
                phase_temperature[segment],
                color="tab:purple",
                linestyle="-.",
                linewidth=1.8,
                label="PNM--equilibrated quark boundary" if labels else None,
            )
            axis.plot(
                upper[segment],
                phase_temperature[segment],
                color="tab:red",
                linestyle="--",
                linewidth=1.8,
                label="Formal analytical divergence boundary" if labels else None,
            )
        if velocity.count() >= 4:
            contours = axis.contour(
                density_grid,
                temperature_grid,
                velocity,
                levels=levels_m_s,
                linewidths=1.25,
            )
            axis.clabel(contours, inline=True, fontsize=8, fmt="%g")
        axis.set_title(title)
        axis.set_xlabel(r"$n_B(0^-)/n_0$")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(0.0, float(np.nanmax(phase_temperature)))
        axis.grid(alpha=0.18)
    axes[0].set_ylabel(r"$T\;[\mathrm{MeV}]$")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    figure.suptitle(r"Isothermal front speed $[\mathrm{m\,s^{-1}}]$")
    figure.subplots_adjust(bottom=0.22, top=0.88, wspace=0.08)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    return figure, axes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="isothermal-domain.npy")
    parser.add_argument("--analytic", default="isothermal-analytic.npy")
    parser.add_argument("--numerical", default="isothermal-numerical.npy")
    parser.add_argument("--output", default="isothermal-contours.png")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    domain_payload = domain.load_payload(args.domain, expected_kind="isothermal_domain")
    domain.assert_domain_fingerprint(domain_payload)
    analytic_payload = domain.load_payload(
        args.analytic, expected_kind="isothermal_analytic"
    )
    numerical_payload = domain.load_payload(
        args.numerical, expected_kind="isothermal_numerical"
    )
    figure, _axes = plot_comparison(
        domain_payload,
        analytic_payload,
        numerical_payload,
        output_path=args.output,
    )
    plt.close(figure)
    print(f"Saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
