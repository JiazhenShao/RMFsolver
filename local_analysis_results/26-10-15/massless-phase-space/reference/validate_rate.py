#!/usr/bin/env python3
"""Standalone massless four-quark phase-space reference (MeV units).

No production solver imports are used. The EOS includes thermal antiparticles
in net densities, while the collision integral contains the four-quark channel
only. Install numpy, sympy, and matplotlib; scipy is deliberately unnecessary.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

GF = 1.16637e-11
THETA = math.radians(13.02)
PREF = 9 * GF**2 * math.sin(THETA)**2 * math.cos(THETA)**2 / (8 * math.pi**5)
HERE = Path(__file__).resolve().parent
RATE_MODEL_STYLES = (
    ("exact_rate", "Exact phase-space integral", "#202020", "-", "o"),
    ("eq23_rate", "Madsen Eq. (23)", "#0072B2", "--", "s"),
    ("sommerfeld_T2_rate", r"Derived $T^0+T^2$", "#D55E00", ":", "^"),
)


@lru_cache(maxsize=None)
def gauss(n):
    return np.polynomial.legendre.leggauss(n)


def fermi(energy, chemical, temperature):
    return np.exp(-np.logaddexp(0, (np.asarray(energy) - chemical) / temperature))


def pair_integrals(energy, momentum, ma, mb, temperature):
    """Analytic occupied and hole pairs, stable even across E=ma+mb.

    Evaluate the larger pair first using positive log-domain expressions; the
    smaller follows by detailed balance. No subtraction of softplus values or
    multiplication of an overflowing exponential by an underflowing pair occurs.
    """
    energy, momentum = np.broadcast_arrays(energy, momentum)
    lo = (energy - momentum) / 2
    hi = (energy + momentum) / 2
    if temperature == 0:
        occupied = np.maximum(0, np.minimum(hi, ma) - np.maximum(lo, energy - mb))
        holes = np.maximum(0, np.minimum(hi, energy - mb) - np.maximum(lo, ma))
        return occupied, holes
    x = (energy - ma - mb) / temperature
    z = np.abs(x)
    ulo = (lo - ma) / temperature
    uhi = (hi - ma) / temperature
    q = -np.expm1(-z)
    with np.errstate(divide="ignore", invalid="ignore"):
        logdiff = (-np.logaddexp(0, ulo) - np.logaddexp(0, -uhi)
                   + np.log(-np.expm1(-momentum / temperature)))
        logq = np.log(q)
        logfhi = np.where(x >= 0, -np.logaddexp(0, uhi), -np.logaddexp(0, -ulo))
        logden = np.logaddexp(-z, logq + logfhi)
        logratio = np.logaddexp(0, logq + logdiff - logden)
        larger = temperature * np.divide(logratio, q, out=np.array(np.exp(logdiff)), where=q != 0)
    smaller = larger * np.exp(-z)
    return np.where(x >= 0, smaller, larger), np.where(x >= 0, larger, smaller)


def _unique_edges(values, lower, upper):
    values = np.asarray(values, dtype=float)
    values = np.clip(values, lower, upper)
    return np.unique(np.concatenate(([lower], values.ravel(), [upper])))


def _nodes(edges, n):
    node, weight = gauss(n)
    half = np.diff(edges) / 2
    center = (edges[:-1] + edges[1:]) / 2
    return (center[:, None] + half[:, None] * node).ravel(), (half[:, None] * weight).ravel()


def _energy_edges(u, d, s, temperature, tail):
    upper = u + d + tail * temperature if temperature else u + d
    lower = 0 if temperature else u + s
    chemical = np.array([u, d, s])
    base = np.concatenate((chemical, (chemical[:, None] + chemical).ravel(), [0, upper]))
    offsets = temperature * np.array([-32, -12, -4, -1, 0, 1, 4, 12, 32]) if temperature else np.array([0])
    return _unique_edges(base[:, None] + offsets, lower, upper)


def phase_space_forward(mu_u, mu_d, mu_s, temperature, n=20, tail=48):
    """Forward integral without PREF, including exact T=0 interval lengths."""
    if temperature < 0 or min(mu_u, mu_d, mu_s) < 0:
        raise ValueError("Reference supports nonnegative chemical potentials and temperature")
    scale = max(mu_u, mu_d, mu_s, temperature, 1e-100)
    u, d, s, t = np.array([mu_u, mu_d, mu_s, temperature]) / scale
    if t == 0 and d <= s:
        return 0.0
    en, ew = _nodes(_energy_edges(u, d, s, t, tail), n)
    node, weight = gauss(n)
    offsets = 2 * t * np.array([-32, -12, -4, -1, 0, 1, 4, 12, 32]) if t else np.array([0])
    total = 0.0
    # Vectorized blocks bound memory even at high quadrature order.
    for start in range(0, len(en), 128):
        energy = en[start:start + 128]
        centers = np.abs(energy[:, None] - 2 * np.array([u, d, s]))
        boundaries = (centers[:, :, None] + offsets).reshape(len(energy), -1)
        boundaries = np.clip(boundaries, 0, energy[:, None])
        boundaries = np.sort(np.concatenate((np.zeros((len(energy), 1)), boundaries, energy[:, None]), axis=1), axis=1)
        half = np.diff(boundaries, axis=1) / 2
        mid = (boundaries[:, 1:] + boundaries[:, :-1]) / 2
        momentum = mid[:, :, None] + half[:, :, None] * node
        energy3 = energy[:, None, None]
        occupied, _ = pair_integrals(energy3, momentum, u, d, t)
        _, holes = pair_integrals(energy3, momentum, u, s, t)
        integrand = (energy3**2 - momentum**2)**2 * occupied * holes
        inner = np.sum(integrand * half[:, :, None] * weight, axis=(1, 2))
        total += np.dot(ew[start:start + 128], inner)
    return float(total * scale**8)


def phase_space_zero(mu_u, mu_d, mu_s, n=16):
    return phase_space_forward(mu_u, mu_d, mu_s, 0, n=n)


def exact_rate(mu_u, mu_d, mu_s, temperature, n=20, tail=48):
    """Net rate in MeV^4, evaluated in the favorable forward direction."""
    if mu_d == mu_s:
        return 0.0
    if mu_d < mu_s:
        return -exact_rate(mu_u, mu_s, mu_d, temperature, n=n, tail=tail)
    balance = -math.expm1(-(mu_d - mu_s) / temperature) if temperature else 1
    return PREF * balance * phase_space_forward(mu_u, mu_d, mu_s, temperature, n=n, tail=tail)


def eq23_rate(mu_u, mu_d, mu_s, temperature):
    delta = mu_d - mu_s
    return (16 * GF**2 * math.sin(THETA)**2 * math.cos(THETA)**2 / (5 * math.pi**5)
            * mu_u**5 * delta * (delta**2 + 4 * math.pi**2 * temperature**2))


def eq24_factor(mu_u, mu_d, mu_s):
    # Madsen's original s -> d labels are exchanged for d -> s here.
    z, v = mu_s / mu_u, (mu_d - mu_s) / mu_u
    return (-v**5 / 128 - 5 * z * v**4 / 64 + 5 * (2 - z**2) * v**3 / 64
            + 5 * (1 + 3 * z) * v**2 / 16 + 5 * (4 * z + 3 * z**2) * v / 16
            + 5 * z**2 / 4 - 1 / 4)


def number_density(chemical, temperature):
    chemical = np.asarray(chemical)
    return chemical**3 / math.pi**2 + chemical * temperature**2


def _bisect(function, lo, hi, iterations=80):
    flo, fhi = function(lo), function(hi)
    if flo * fhi > 0:
        raise ValueError("Root is not bracketed")
    for _ in range(iterations):
        mid = (lo + hi) / 2
        fm = function(mid)
        if fm == 0:
            return mid
        if fm * flo > 0:
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


def _chemical_from_density(density, temperature):
    if density == 0:
        return 0.0
    if temperature == 0:
        return float(np.cbrt(density * math.pi**2))
    # Stable real root of mu^3 + pi^2*T^2*mu - pi^2*n = 0.
    p = math.pi**2 * temperature**2 / 3
    return 2 * math.sqrt(p) * math.sinh(math.asinh(math.pi**2 * density / (2 * p**1.5)) / 3)


def neutral_fixed_a(muB, a, temperature):
    if not 0 <= a <= 1:
        raise ValueError("Composition must be in [0,1]")
    def chemical(nu):
        return tuple(_chemical_from_density(nu * factor, temperature) for factor in (1, 1 + a, 1 - a))
    density = _bisect(lambda nu: sum(chemical(nu)) - muB, 0, float(number_density(muB, temperature)))
    return chemical(density)


def neutral_fixed_muK(muB, muK, temperature):
    if muK == 0:
        return (muB / 3,) * 3
    def chemical(u):
        return u, (muB - u + muK) / 2, (muB - u - muK) / 2
    def neutrality(u):
        nu, nd, ns = number_density(np.array(chemical(u)), temperature)
        return 2 * nu - nd - ns
    u = _bisect(neutrality, 0, muB)
    return chemical(u)


def eos_quadratic_shifts(mu_u, mu_d, mu_s, path):
    """Coefficients of T**2 in the neutral chemical potentials at T=0."""
    if path == "fixed_muB_muK":
        shift = math.pi**2 * (2*mu_u-mu_d-mu_s) / (3*(4*mu_u**2+mu_d**2+mu_s**2))
        return -2*shift, shift, shift
    if path == "fixed_muB_a":
        chemical = np.array([mu_u, mu_d, mu_s])
        positive = chemical > 0
        common = math.pi**2 * np.sum(1/chemical[positive]) / (3*np.sum(chemical))
        shifts = np.zeros(3)
        shifts[positive] = common*chemical[positive] - math.pi**2/(3*chemical[positive])
        return tuple(shifts)
    raise ValueError(f"Unknown EOS path: {path}")


def cold_phase_space_3d(mu_u, mu_d, mu_s, n=64):
    """Independent excitation-simplex quadrature of the angular kernel."""
    delta = mu_d - mu_s
    if delta <= 0:
        return 0.0
    node, weight = gauss(n)
    q, w = (node + 1) / 2, weight / 2
    xmax = min(1, mu_u / delta)
    total = 0.0
    for x, wx in zip(q * xmax, w * xmax):
        y, z = np.meshgrid(q, q, indexing="ij")
        p1 = mu_u - delta * x
        pd = mu_d - delta * (1 - x) * y
        p2 = mu_u + delta * (1 - x) * (1 - y) * z
        energy = p1 + pd
        kmin = np.maximum(np.abs(p1 - pd), np.abs(2 * p2 - energy))
        kernel = (8 * energy**5 / 15 - energy**4 * kmin
                  + 2 * energy**2 * kmin**3 / 3 - kmin**5 / 5)
        jacobian = delta**3 * (1 - x)**2 * (1 - y)
        total += wx * np.sum(kernel * jacobian * w[:, None] * w[None, :])
    return float(total)


def finite_phase_space_3d(mu_u, mu_d, mu_s, temperature, n=12, tail=40):
    """Independent radial-momentum integral with the k integral done first."""
    def momentum_nodes(chemical):
        offsets = temperature * np.array([-32, -12, -4, -1, 0, 1, 4, 12, 32])
        return _nodes(_unique_edges(chemical + offsets, 0, chemical + tail * temperature), n)
    p1s, w1s = momentum_nodes(mu_u)
    pds, wds = momentum_nodes(mu_d)
    node, weight = gauss(n)
    total = 0.0
    for p1, w1 in zip(p1s, w1s):
        energy = p1 + pds
        offsets = temperature * np.array([-32, -12, -4, -1, 0, 1, 4, 12, 32])
        thermal = np.concatenate((mu_u + offsets + np.zeros((len(energy), 1)), energy[:, None] - mu_s + offsets), axis=1)
        edges = np.concatenate((np.zeros((len(energy), 1)), energy[:, None], thermal,
                                np.full((len(energy), 1), p1), pds[:, None], energy[:, None] / 2), axis=1)
        edges = np.sort(np.clip(edges, 0, energy[:, None]), axis=1)
        half, mid = np.diff(edges, axis=1) / 2, (edges[:, 1:] + edges[:, :-1]) / 2
        p2 = mid[:, :, None] + half[:, :, None] * node
        en = energy[:, None, None]
        kmin = np.maximum(np.abs(p1 - pds)[:, None, None], np.abs(2 * p2 - en))
        kernel = 8 * en**5 / 15 - en**4 * kmin + 2 * en**2 * kmin**3 / 3 - kmin**5 / 5
        holes = fermi(-p2, -mu_u, temperature) * fermi(p2 - en, -mu_s, temperature)
        inner = np.sum(kernel * holes * half[:, :, None] * weight, axis=(1, 2))
        total += w1 * fermi(p1, mu_u, temperature) * np.dot(wds, inner * fermi(pds, mu_d, temperature))
    return float(total)


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verification_checks(output):
    from cold_sympy import cold_phase_space
    from thermal_sympy import phase_space_t2, phase_space_t3

    checks = {}
    cold_rows = []
    for a in (0.001, 0.146, 0.5, 0.9, 0.99, 1):
        chemical = neutral_fixed_a(900, a, 0)
        analytic = cold_phase_space(*chemical)
        reference = phase_space_zero(*chemical, n=16)
        low, high = (cold_phase_space_3d(*chemical, n=n) for n in (96, 192))
        cold_rows.append(dict(a=a, analytic_I0=analytic, quadrature_2d=reference,
                              quadrature_3d_n96=low, quadrature_3d_n192=high,
                              relative_3d_error=high / analytic - 1,
                              relative_3d_change=high / low - 1,
                              exact_to_eq23=PREF * analytic / eq23_rate(*chemical, 0)))
    _write_csv(output / "cold-benchmarks.csv", cold_rows)
    checks["cold_max_2d_analytic_relative_error"] = max(abs(r["quadrature_2d"] / r["analytic_I0"] - 1) for r in cold_rows)
    checks["cold_max_3d_analytic_relative_error"] = max(abs(r["relative_3d_error"]) for r in cold_rows)
    finite_rows = []
    for chemical, temperature in [((1, 1.2, 0.8), 0.12), ((1, 2**(1/3), 0), 0.05)]:
        reference = phase_space_forward(*chemical, temperature, n=32)
        for n in (12, 24):
            independent = finite_phase_space_3d(*chemical, temperature, n=n)
            finite_rows.append(dict(mu_u=chemical[0], mu_d=chemical[1], mu_s=chemical[2],
                                    T=temperature, n=n, quadrature_2d=reference,
                                    quadrature_3d=independent, relative_error=independent / reference - 1))
    _write_csv(output / "independent-finite-benchmarks.csv", finite_rows)
    checks["finite_3d_max_relative_error_n24"] = max(abs(r["relative_error"]) for r in finite_rows if r["n"] == 24)
    forward = phase_space_forward(1, 1.2, 0.8, 0.12, n=24)
    reverse = phase_space_forward(1, 0.8, 1.2, 0.12, n=24)
    checks["detailed_balance_relative_error"] = reverse / forward / math.exp(-0.4 / 0.12) - 1
    checks["equilibrium_rate"] = exact_rate(300, 300, 300, 10)
    checks["sign_reversal_relative_error"] = exact_rate(300, 250, 350, 10) / exact_rate(300, 350, 250, 10) + 1
    checks["dimensional_scaling_relative_error"] = exact_rate(600, 700, 500, 20) / exact_rate(300, 350, 250, 10) / 256 - 1
    residual_rows = []
    for a in (0.001, 0.146, 0.5, 0.9, 0.99, 1):
        chemical = (1, float(np.cbrt(1 + a)), float(np.cbrt(1 - a)))
        i0, c2, c3 = cold_phase_space(*chemical), phase_space_t2(*chemical), phase_space_t3(*chemical)
        # Fixed imbalance asymptotics require T smaller than the splitting too.
        base_t = min(0.003, (chemical[1] - chemical[2]) / 30)
        for temperature in (base_t, base_t / 2, base_t / 4, base_t / 8):
            exact = phase_space_forward(*chemical, temperature, n=40)
            residual_rows.append(dict(a=a, T_over_mu_u=temperature, I_exact=exact,
                                      I0=i0, C2=c2, C3=c3,
                                      residual_over_T3=(exact-i0-c2*temperature**2) / temperature**3,
                                      residual_after_T3_over_T4=(exact-i0-c2*temperature**2-c3*temperature**3) / temperature**4))
    _write_csv(output / "low-temperature-residuals.csv", residual_rows)
    return checks


def path_coefficients(output):
    from cold_sympy import cold_phase_space, cold_phase_space_gradient
    from thermal_sympy import phase_space_t2, phase_space_t3

    rows = []
    for muB in (900, 1200):
        for a in (0.001, 0.146, 0.5, 0.9, 0.99, 1):
            chemical = neutral_fixed_a(muB, a, 0)
            gradient = cold_phase_space_gradient(*chemical)
            for path in ("fixed_muB_a", "fixed_muB_muK"):
                shifts = eos_quadratic_shifts(*chemical, path)
                intrinsic = phase_space_t2(*chemical)
                correction = float(np.dot(gradient, shifts))
                temperature = min(muB * 1e-5, (chemical[1]-chemical[2]) / 200)
                warm = (neutral_fixed_a(muB, a, temperature) if path == "fixed_muB_a"
                        else neutral_fixed_muK(muB, chemical[1]-chemical[2], temperature))
                numeric = (exact_rate(*warm, temperature, n=40)/PREF-cold_phase_space(*chemical)) / temperature**2
                rows.append(dict(path=path, muB=muB, a_cold=a, mu_u=chemical[0],
                                 mu_d=chemical[1], mu_s=chemical[2],
                                 mu_u_T2=shifts[0], mu_d_T2=shifts[1], mu_s_T2=shifts[2],
                                 intrinsic_C2=intrinsic, eos_C2=correction,
                                 total_C2=intrinsic+correction, numeric_T=temperature,
                                 numeric_C2=numeric,
                                 numeric_C2_minus_predicted_C3_T=numeric-phase_space_t3(*chemical)*temperature,
                                 corrected_relative_error=(numeric-phase_space_t3(*chemical)*temperature)/(intrinsic+correction)-1))
    _write_csv(output / "eos-path-coefficients.csv", rows)
    return rows


def build_grid(output):
    from cold_sympy import cold_phase_space
    from thermal_sympy import phase_space_t2, phase_space_t3

    rows = []
    for muB in (900, 1200):
        for a in (0.001, 0.146, 0.5, 0.9, 0.99, 1):
            cold_chemical = neutral_fixed_a(muB, a, 0)
            cold_muK = cold_chemical[1] - cold_chemical[2]
            for path in ("fixed_muB_a", "fixed_muB_muK"):
                for temperature in (0.01, 0.1, 1, 5, 10, 25, 50, 100):
                    chemical = (neutral_fixed_a(muB, a, temperature) if path == "fixed_muB_a"
                                else neutral_fixed_muK(muB, cold_muK, temperature))
                    low, middle, reference = (exact_rate(*chemical, temperature, n=n) for n in (12, 24, 40))
                    extended = exact_rate(*chemical, temperature, n=40, tail=64)
                    i0 = cold_phase_space(*chemical)
                    c2, c3 = phase_space_t2(*chemical), phase_space_t3(*chemical)
                    cold, quadratic, cubic = (PREF * value for value in (i0, i0 + c2 * temperature**2,
                                               i0 + c2 * temperature**2 + c3 * temperature**3))
                    eq23 = eq23_rate(*chemical, temperature)
                    corrected = cold + (eq23-eq23_rate(*chemical, 0)) * eq24_factor(*chemical)
                    density = number_density(np.array(chemical), temperature)
                    row = dict(path=path, muB=muB, a_cold=a, T=temperature,
                               a_actual=(density[1]-density[2]) / (2*density[0]),
                               mu_u=chemical[0], mu_d=chemical[1], mu_s=chemical[2],
                               muK=chemical[1]-chemical[2], exact_rate=reference,
                               n12_rate=low, n24_rate=middle, n40_rate=reference,
                               convergence_relative=abs(reference/middle-1),
                               tail_relative=abs(extended/reference-1),
                               cold_rate=cold, sommerfeld_T2_rate=quadratic,
                               through_T3_rate=cubic, eq23_rate=eq23,
                               cold_plus_eq24_rate=corrected,
                               cold_relative_error=cold/reference-1,
                               sommerfeld_T2_relative_error=quadratic/reference-1,
                               through_T3_relative_error=cubic/reference-1,
                               eq23_relative_error=eq23/reference-1,
                               cold_plus_eq24_relative_error=corrected/reference-1)
                    rows.append(row)
            print(f"Completed muB={muB}, cold a={a}", flush=True)
    _write_csv(output / "rate-comparison-grid.csv", rows)
    return rows


def read_grid(path):
    """Read saved numerical rows without repeating the phase-space integrations."""
    with Path(path).open(newline="") as stream:
        return [{key: value if key == "path" else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def make_plots(output, rows):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "massless-weak-rate-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    a_values = (0.001, 0.146, 0.5, 0.9, 0.99, 1)
    for muB in (900, 1200):
        fig, axes = plt.subplots(2, 3, figsize=(7.3, 6), sharex=True)
        for axis, a in zip(axes.flat, a_values):
            selected = sorted([row for row in rows if row["path"] == "fixed_muB_a"
                               and row["muB"] == muB and row["a_cold"] == a],
                              key=lambda row: row["T"])
            for key, label, color, style, marker in RATE_MODEL_STYLES:
                axis.plot([row["T"] for row in selected],
                          [100*(row[key]/row["exact_rate"]-1) for row in selected],
                          color=color, linestyle=style, marker=marker, markersize=3,
                          linewidth=1.2, label=label)
            axis.set_xscale("log")
            axis.set_yscale("symlog", linthresh=1)
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}%"))
            axis.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
            axis.margins(y=0.12)
            axis.set_title(f"a = {a:g}", fontsize=10)
            axis.tick_params(axis="both", labelsize=9)
            axis.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.945),
                   ncol=1, fontsize=9, frameon=False, handlelength=2.8)
        fig.supxlabel("Temperature [MeV]", fontsize=11, y=0.025)
        fig.supylabel("Relative rate error [%] (symlog)", fontsize=11, x=0.01)
        fig.suptitle(f"Fixed neutral composition, $\\mu_B={muB}$ MeV", fontsize=11, y=0.995)
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.115, top=0.79,
                            hspace=0.34, wspace=0.44)
        fig.savefig(output / f"approximation-errors-muB-{muB}.pdf")
        fig.savefig(output / f"approximation-errors-muB-{muB}.png", dpi=220)
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(7.3, 7.7), sharex=True)
    for row_axes, a in zip(axes, (0.146, 0.5, 0.9)):
        for axis, muB in zip(row_axes, (900, 1200)):
            selected = sorted([row for row in rows if row["path"] == "fixed_muB_a"
                               and row["muB"] == muB and row["a_cold"] == a],
                              key=lambda row: row["T"])
            for key, label, color, style, marker in RATE_MODEL_STYLES:
                axis.loglog([row["T"] for row in selected], [row[key] for row in selected],
                            color=color, linestyle=style, marker=marker, markersize=3,
                            linewidth=1.2, label=label)
            axis.set_title(f"$\\mu_B={muB}$ MeV, a = {a:g}", fontsize=10)
            axis.tick_params(axis="both", labelsize=9)
            axis.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.995),
               ncol=1, fontsize=9, frameon=False, handlelength=2.8)
    fig.supxlabel("Temperature [MeV]", fontsize=11, y=0.025)
    fig.supylabel("Net rate [MeV$^4$]", fontsize=11, x=0.01)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.09, top=0.85,
                        hspace=0.34, wspace=0.25)
    fig.savefig(output / "exact-rates.pdf")
    fig.savefig(output / "exact-rates.png", dpi=220)
    plt.close(fig)


def write_summary(output, checks, rows):
    lines = ["# Numerical validation", "",
             "The reference is the massless four-quark channel only, with GF = 1.16637e-11 MeV^-2 and theta_C = 13.02 degrees.",
             "The neutral EOS uses net massless-quark densities n_i = mu_i^3/pi^2 + mu_i*T^2; this does not add antiquark reaction channels.",
             "The fixed-muB,a grid holds the local composition exactly. The fixed-muB,muK grid fixes muK to its cold value at the listed a_cold.",
             "All approximation formulas in the grid are evaluated at the same finite-temperature EOS chemical potentials as the reference.", "",
             "## Verification", ""]
    lines += [f"- {name}: {value:.6g}" for name, value in checks.items()]
    if rows:
        lines += [f"- Largest relative change from 24 to 40 nodes per subinterval: {max(row['convergence_relative'] for row in rows):.6g}.",
                  f"- Largest relative tail change from 48 T to 64 T: {max(row['tail_relative'] for row in rows):.6g}.", "",
                  "## Representative errors at fixed composition", "",
                  "Errors are (approximation / exact phase-space integral - 1) * 100 percent.",
                  "Every figure compares only the exact phase-space integral, Madsen Eq. (23), and the derived T0+T2 rate.",
                  "The exact reference has zero error; extra T3 and Eq. (24) diagnostics remain in the CSV but are not plotted.", "",
                  "| muB | a | T | Exact phase space | Madsen Eq. (23) | Derived T0+T2 |",
                  "|---|---|---|---|---|---|---|"]
        for row in rows:
            if row["path"] == "fixed_muB_a" and row["a_cold"] in (0.146, 0.9, 1) and row["T"] in (1, 10, 50, 100):
                lines.append(f"| {row['muB']:g} | {row['a_cold']:g} | {row['T']:g} | 0% | {100*row['eq23_relative_error']:.3g}% | {100*row['sommerfeld_T2_relative_error']:.3g}% |")
    lines += ["", "## Scope and limitations", "",
              "The T2 coefficient is the fixed-chemical-potential coefficient. Evaluating I0(mu(T)) + C2(mu(T))*T^2 automatically includes the EOS displacement but is not identical to re-expanding at T=0.",
              "Coincident incoming/outgoing up-quark surfaces produce a T3 correction. The saved residual sweep tests this directly; a T4-only remainder claim would be incorrect.",
              "The a=1 endpoint has mu_s=0. The coefficient formulas have finite limits there, but ordinary strange-surface degeneracy assumptions do not hold.",
              "These are rate errors for local thermodynamic states, not errors in a solved front velocity. Large-temperature extrapolations of truncated series are diagnostics only.",
              "The quadrature reports convergence differences, not rigorous interval-arithmetic error bounds.", ""]
    (output / "numerical-summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run the checks without the full grid")
    mode.add_argument("--plots-only", action="store_true", help="Replot the saved grid without rerunning quadrature")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        rows = read_grid(args.output / "rate-comparison-grid.csv")
        checks = json.loads((args.output / "verification.json").read_text())
        make_plots(args.output, rows)
        write_summary(args.output, checks, rows)
        return
    started = time.time()
    checks = verification_checks(args.output)
    path_rows = path_coefficients(args.output)
    checks["eos_path_C2_max_corrected_relative_error"] = max(abs(row["corrected_relative_error"]) for row in path_rows)
    rows = [] if args.quick else build_grid(args.output)
    if rows:
        checks["grid_max_relative_n24_to_n40"] = max(row["convergence_relative"] for row in rows)
        checks["grid_max_relative_tail48_to_tail64"] = max(row["tail_relative"] for row in rows)
        make_plots(args.output, rows)
    checks["wall_seconds"] = time.time() - started
    with (args.output / "verification.json").open("w") as stream:
        json.dump(checks, stream, indent=2)
    write_summary(args.output, checks, rows)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
