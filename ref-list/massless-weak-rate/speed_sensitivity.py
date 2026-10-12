#!/usr/bin/env python3
"""Compare rate-integral speed estimates with a fixed neutral background.

This is not a boundary-value solution. The same constant diffusion coefficient,
baryon-density normalization, and interface/profile factors are held fixed and
cancel from the ratios. Only the integral of the reaction rate is changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from cold_sympy import cold_phase_space
from thermal_sympy import phase_space_t2
from validate_rate import HERE, PREF, eq23_rate, exact_rate, gauss, neutral_fixed_a

TEMPERATURES = (0, 20, 50, 100)
BENCHMARK_COMPOSITIONS = (0.146, 0.5, 0.9)
RATIO_KEYS = ("exact_speed_ratio", "eq23_speed_ratio", "sommerfeld_T2_speed_ratio")
# Green endpoint labels sit in the empty band beside the green curve (points).
EQ23_MUB_LABEL_OFFSETS = {0: (-8, -20), 20: (-8, -19), 50: (-8, -14), 100: (-8, 16)}


def rates_at_composition(muB, composition, temperature, phase_order=24, tail=48):
    """All three rates at exactly the same finite-temperature EOS state."""
    chemical = neutral_fixed_a(muB, composition, temperature)
    return np.array([
        exact_rate(*chemical, temperature, n=phase_order, tail=tail),
        eq23_rate(*chemical, temperature),
        PREF * (cold_phase_space(*chemical) + temperature**2 * phase_space_t2(*chemical)),
    ])


def integrated_estimates(muB, temperature, compositions, outer_order=12, phase_order=24, tail=48):
    """Cumulative quadrature of rates on a fixed-(muB,T) neutral path.

    The transformation a=1-(1-x)^3 resolves the cold EOS cube-root endpoint.
    Exact rates always come from the phase-space quadrature, including at T=0.
    """
    compositions = np.asarray(compositions, dtype=float)
    if (compositions.ndim != 1 or len(compositions) == 0
            or np.any(compositions <= 0) or np.any(compositions > 1)
            or np.any(np.diff(compositions) <= 0)):
        raise ValueError("Composition endpoints must increase strictly within (0,1]")
    nodes, weights = gauss(outer_order)
    totals = np.zeros(3)
    previous = 0.0
    rows = []
    for composition in compositions:
        upper = 1 - np.cbrt(1 - composition)
        half, midpoint = (upper - previous) / 2, (upper + previous) / 2
        for x, weight in zip(midpoint + half * nodes, weights):
            a = 1 - (1 - x)**3
            jacobian = 3 * (1 - x)**2
            totals += (half * weight * jacobian
                       * rates_at_composition(muB, a, temperature, phase_order, tail))
        ratios = np.sqrt(totals / totals[0])
        rows.append(dict(muB=muB, T=temperature, a0plus=float(composition),
                         exact_integral=float(totals[0]), eq23_integral=float(totals[1]),
                         sommerfeld_T2_integral=float(totals[2]),
                         exact_speed_ratio=float(ratios[0]), eq23_speed_ratio=float(ratios[1]),
                         sommerfeld_T2_speed_ratio=float(ratios[2])))
        previous = upper
    return rows


def closed_form_integrals(rate, muB, temperature, compositions, outer_order=16):
    """Cumulative integrals of a closed-form rate on the nodes of integrated_estimates.

    outer_order=16 is the finest saved setting, so rate=eq23_rate reproduces the
    saved eq23_integral column.
    """
    nodes, weights = gauss(outer_order)
    total, previous, totals = 0.0, 0.0, []
    for composition in compositions:
        upper = 1 - np.cbrt(1 - composition)
        half, midpoint = (upper - previous) / 2, (upper + previous) / 2
        for x, weight in zip(midpoint + half * nodes, weights):
            a = 1 - (1 - x)**3
            total += (half * weight * 3 * (1 - x)**2
                      * rate(*neutral_fixed_a(muB, a, temperature), temperature))
        totals.append(total)
        previous = upper
    return totals


def with_eq23_muB(output, rows):
    """Add the plotting-only (mu_B/3)^5 model to saved rows; saved columns are unchanged."""
    from rate_figures import eq23_muB_rate
    rows = [dict(row) for row in rows]
    for temperature in sorted({row["T"] for row in rows}):
        selected = sorted((row for row in rows if row["T"] == temperature),
                          key=lambda row: row["a0plus"])
        totals = closed_form_integrals(eq23_muB_rate, selected[0]["muB"], temperature,
                                       [row["a0plus"] for row in selected])
        for row, total in zip(selected, totals):
            row["eq23_muB_integral"] = total
            row["eq23_muB_speed_ratio"] = math.sqrt(total / row["exact_integral"])
    _write_csv(output / "speed-sensitivity-eq23-muB.csv",
               [{key: row[key] for key in ("muB", "T", "a0plus", "exact_integral",
                                           "eq23_muB_integral", "eq23_muB_speed_ratio")}
                for row in rows])
    return rows


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def independent_quadrature_checks(muB, temperature, reference_rows):
    """Refine each numerical control separately at the hot benchmarks.

    The benchmark integration uses fewer, wider outer panels than the plotted
    curve, so its baseline is also compared with the saved production values.
    """
    settings = (("baseline", 16, 32, 64), ("outer", 24, 32, 64),
                ("phase", 16, 40, 64), ("tail", 16, 32, 80))
    calculated = {}
    checks = []
    reference = {row["a0plus"]: row for row in reference_rows}
    for control, outer, phase, tail in settings:
        calculated[control] = integrated_estimates(
            muB, temperature, BENCHMARK_COMPOSITIONS,
            outer_order=outer, phase_order=phase, tail=tail)
        print(f"T={temperature}: independent {control} check complete", flush=True)
        for index, row in enumerate(calculated[control]):
            baseline = (reference[row["a0plus"]] if control == "baseline"
                        else calculated["baseline"][index])
            check = dict(T=temperature, a0plus=row["a0plus"], control=control,
                         outer_order=outer, phase_order=phase, tail=tail)
            for key in ("exact_integral", "eq23_integral", "sommerfeld_T2_integral", *RATIO_KEYS):
                check[key + "_relative_change"] = row[key] / baseline[key] - 1
            checks.append(check)
    maximum = max(abs(row[key + "_relative_change"]) for row in checks for key in RATIO_KEYS)
    if maximum >= 1e-3:
        raise RuntimeError(f"Independent speed-ratio convergence failed: {maximum:.6g} >= 1e-3")
    return checks


def write_tex_summary(rows, summary):
    def scientific(value):
        coefficient, exponent = f"{value:.2e}".split("e")
        return rf"{coefficient}\times10^{{{int(exponent)}}}"

    lines = [
        r"\par",
        r"Increasing the outer Gauss--Legendre order from 12 to 16 and the phase-space order from 24 to 32, while extending the thermal tail from \(48T\) to \(64T\), changes the integrated exact rate by at most \("
        + scientific(summary["max_exact_integral_medium_to_fine"]) + r"\) relative across the sampled compositions.",
        r"The corresponding maximum changes for Eq.~(23) and the derived \(T^0+T^2\) expression are \("
        + scientific(summary["max_eq23_integral_medium_to_fine"]) + r"\) and \("
        + scientific(summary["max_T2_integral_medium_to_fine"]) + r"\), respectively.",
        r"Independent checks at \(T=50,100\,\mathrm{MeV}\) and \(a(0^+)=0.146,0.5,0.9\) increase only the outer order from 16 to 24, only the phase-space order from 32 to 40, or only the tail from \(64T\) to \(80T\).",
        r"Including a comparison of the benchmark and plotted composition partitions, the largest relative change of any speed ratio is \("
        + scientific(summary["max_independent_speed_ratio_relative_change"]) + r"\), below the \(10^{-3}\) acceptance tolerance.",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Percentage change of the rate-integral speed estimate relative to the exact phase-space result, whose change is identically zero, at \(\mu_B=1200\,\mathrm{MeV}\).}",
        r"\label{tab:speed-sensitivity}",
        r"\begin{tabular}{rr rr}",
        r"\toprule",
        r"\(T\,[\mathrm{MeV}]\) & \(a(0^+)\) & Eq.~(23) & Derived \(T^0+T^2\) \\",
        r"\midrule",
    ]
    lookup = {(row["T"], row["a0plus"]): row for row in rows}
    for temperature in TEMPERATURES:
        if temperature:
            lines.append(r"\addlinespace")
        for composition in BENCHMARK_COMPOSITIONS:
            percentages = [100 * (lookup[temperature, composition][key] - 1)
                           for key in ("eq23_speed_ratio", "sommerfeld_T2_speed_ratio")]
            lines.append(f"{temperature:g} & {composition:g} & " + " & ".join(
                rf"\({value:+.2f}\%\)" if abs(value) >= 0.005 else r"\(0.00\%\)"
                for value in percentages) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    for temperature in TEMPERATURES[1:]:
        warm_low, warm_high = (lookup[temperature, composition] for composition in (0.146, 0.9))
        lines.append(
            rf"At \(T={temperature}\,\mathrm{{MeV}}\) and \(a(0^+)=0.146\), Eq.~(23) and the derived \(T^0+T^2\) rate give speed-estimate changes relative to the exact result of \("
            + f'{100 * (warm_low["eq23_speed_ratio"] - 1):+.2f}' + r"\%\) and \("
            + f'{100 * (warm_low["sommerfeld_T2_speed_ratio"] - 1):+.2f}'
            + r"\%\); at \(a(0^+)=0.9\), the changes are \("
            + f'{100 * (warm_high["eq23_speed_ratio"] - 1):+.2f}' + r"\%\) and \("
            + f'{100 * (warm_high["sommerfeld_T2_speed_ratio"] - 1):+.2f}' + r"\%\), respectively.")
    lines.append(r"The hot points test the finite-temperature rate prescriptions within the same four-quark channel; agreement there does not establish a uniformly valid low-temperature expansion or a re-solved front speed.")
    (HERE / "speed-sensitivity-results.tex").write_text("\n".join(lines) + "\n")


def make_plot(output, rows):
    from plot_style import (FIGURE_WIDTH, TEXT_SIZE, GRID_ALPHA,
                            model_line, endpoint_label, model_legend, save_figure)
    from rate_figures import EQ23_MUB_STYLE, LOCAL_RATE_STYLES

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "massless-weak-rate-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratio_key = {"exact_rate": "exact_speed_ratio", "eq23_rate": "eq23_speed_ratio",
                 "sommerfeld_T2_rate": "sommerfeld_T2_speed_ratio",
                 "eq23_muB_rate": "eq23_muB_speed_ratio"}
    with matplotlib.rc_context({'font.size':TEXT_SIZE,'axes.labelsize':TEXT_SIZE,
                               'axes.titlesize':TEXT_SIZE,'xtick.labelsize':TEXT_SIZE,
                               'ytick.labelsize':TEXT_SIZE}):
        figure, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 6.7))
    axes = axes.ravel()
    for axis, temperature in zip(axes, TEMPERATURES):
        selected = [row for row in rows if row["T"] == temperature]
        for style in LOCAL_RATE_STYLES:
            model_line(axis,[row["a0plus"] for row in selected],
                       [row[ratio_key[style[0]]] for row in selected],style)
        all_ratios = [row[key] for row in selected for key in ratio_key.values()]
        padding = max(0.025, 0.08 * (max(all_ratios) - min(all_ratios)))
        axis.set(title=(r"$T=0$ (cold-rate limit)" if temperature == 0
                        else rf"$T={temperature}\,\mathrm{{MeV}}$"),
                 xlabel=r"Interface composition $a(0^+)$", xlim=(0, 1),
                 ylim=(min(all_ratios) - padding, max(all_ratios) + padding))
        endpoint = next(row for row in selected if row["a0plus"] == 1)
        endpoint_ratio = endpoint["eq23_speed_ratio"]
        endpoint_label(axis,1,endpoint_ratio)
        endpoint_label(axis,1,endpoint["eq23_muB_speed_ratio"],color=EQ23_MUB_STYLE[2],
                       marker=EQ23_MUB_STYLE[4],offset=EQ23_MUB_LABEL_OFFSETS[temperature])
        axis.grid(alpha=GRID_ALPHA)
        axis.tick_params(labelsize=TEXT_SIZE)
        axis.set_xlabel(r"Interface composition $a(0^+)$",fontsize=TEXT_SIZE)
        axis.set_title(axis.get_title(),fontsize=TEXT_SIZE)
    figure.supylabel("Speed estimate / exact estimate",fontsize=TEXT_SIZE,x=.015)
    axes[0].text(0.05, 0.73, "Exact and "+r"$T^0+T^2$"+"\noverlap",
                 transform=axes[0].transAxes,fontsize=TEXT_SIZE)
    figure.suptitle(r"Rate-integral sensitivity at $\mu_B=1200\,\mathrm{MeV}$",fontsize=13)
    model_legend(figure,LOCAL_RATE_STYLES)
    figure.tight_layout(rect=(.035,.18,1,.96),h_pad=1.5,w_pad=1.1)
    save_figure(figure,output,'speed-sensitivity')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        with (args.output / "speed-sensitivity.csv").open() as stream:
            rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        summary = json.loads((args.output / "speed-sensitivity-summary.json").read_text())
        for row in rows:
            for model in ("exact", "eq23", "sommerfeld_T2"):
                row[f"{model}_speed_ratio"] = math.sqrt(row[f"{model}_integral"] / row["exact_integral"])
        summary["normalization"] = "sqrt(integral(model)/integral(exact phase space))"
        summary["benchmarks"] = [row for row in rows if row["a0plus"] in (0.146, 0.5, 0.9, 0.99, 1)]
        _write_csv(args.output / "speed-sensitivity.csv", rows)
        (args.output / "speed-sensitivity-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_tex_summary(rows, summary)
        make_plot(args.output, with_eq23_muB(args.output, rows))
        return

    started = time.monotonic()
    compositions = np.unique(np.concatenate(([0.005, 0.146, 0.5, 0.9, 0.99, 1],
                                             np.linspace(0.025, 0.975, 39))))
    rows, convergence, independent = [], [], []
    settings = [(8, 12, 48), (12, 24, 48), (16, 32, 64)]
    for temperature in TEMPERATURES:
        calculations = []
        for outer, phase, tail in settings:
            calculated = integrated_estimates(1200, temperature, compositions,
                                               outer_order=outer, phase_order=phase, tail=tail)
            calculations.append(calculated)
            print(f"T={temperature}: outer={outer}, phase={phase}, tail={tail} complete", flush=True)
        rows.extend(calculations[-1])
        if temperature >= 50:
            independent.extend(independent_quadrature_checks(1200, temperature, calculations[-1]))
        for coarse, medium, fine in zip(*calculations):
            convergence.append(dict(
                T=temperature, a0plus=fine["a0plus"],
                exact_integral_coarse_to_medium=medium["exact_integral"] / coarse["exact_integral"] - 1,
                exact_integral_medium_to_fine=fine["exact_integral"] / medium["exact_integral"] - 1,
                eq23_integral_medium_to_fine=fine["eq23_integral"] / medium["eq23_integral"] - 1,
                T2_integral_medium_to_fine=fine["sommerfeld_T2_integral"] / medium["sommerfeld_T2_integral"] - 1))
    _write_csv(args.output / "speed-sensitivity.csv", rows)
    _write_csv(args.output / "speed-sensitivity-convergence.csv", convergence)
    _write_csv(args.output / "speed-sensitivity-independent-convergence.csv", independent)
    benchmarks = [row for row in rows if row["a0plus"] in (0.146, 0.5, 0.9, 0.99, 1)]
    summary = dict(
        muB_MeV=1200, temperatures_MeV=list(TEMPERATURES),
        interpretation="Rate-integral speed estimates with background, diffusion and profile factors fixed; not re-solved fronts",
        integral="integral_0^a0plus Gamma_K(muB,a,T) da",
        normalization="sqrt(integral(model)/integral(exact phase space))",
        T2_prescription="Intrinsic cold+T2 coefficients evaluated at the same instantaneous finite-T neutral flavor potentials as the other models",
        settings=[dict(outer_order=o, phase_order=p, tail=t) for o, p, t in settings],
        max_exact_integral_medium_to_fine=max(abs(row["exact_integral_medium_to_fine"]) for row in convergence),
        max_eq23_integral_medium_to_fine=max(abs(row["eq23_integral_medium_to_fine"]) for row in convergence),
        max_T2_integral_medium_to_fine=max(abs(row["T2_integral_medium_to_fine"]) for row in convergence),
        independent_check_baseline=dict(outer_order=16, phase_order=32, tail=64),
        independent_check_settings=[dict(control="outer", outer_order=24, phase_order=32, tail=64),
                                    dict(control="phase", outer_order=16, phase_order=40, tail=64),
                                    dict(control="tail", outer_order=16, phase_order=32, tail=80)],
        independent_check_tolerance=1e-3,
        max_independent_speed_ratio_relative_change=max(
            abs(row[key + "_relative_change"]) for row in independent for key in RATIO_KEYS),
        independent_checks=independent,
        benchmarks=benchmarks, elapsed_seconds=time.monotonic() - started)
    (args.output / "speed-sensitivity-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_tex_summary(rows, summary)
    make_plot(args.output, with_eq23_muB(args.output, rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
