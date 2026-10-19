"""Check the manuscript's Taylor ceiling against saved, exact EOS ceilings.

Run with ./bin/python3 local_analysis_results/26-10-16/aiso-accuracy/audit.py.
No RMFsolver imports, profile solves, or changes to saved inputs are needed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "new_paper_calculations/26-10-02/isothermal_analytic_logT/isothermal-analytic-logT.npy"
FIGURE = ROOT / "local_analysis_results/All_Plots_new/isothermal_mu_plane_plot_data.npz"


def equilibrium_muB(pressure, temperature, bag_root):
    kinetic_pressure = pressure + bag_root**4 - 19 * np.pi**2 * temperature**4 / 36
    assert np.all(kinetic_pressure > 0)
    coefficient = temperature**2 / 6
    mu_squared = 2 * kinetic_pressure / (
        coefficient + np.sqrt(coefficient**2 + kinetic_pressure / (27 * np.pi**2))
    )
    return np.sqrt(mu_squared), kinetic_pressure


def flavor_mu(density, temperature):
    """Positive root of mu**3/pi**2 + mu*T**2 = n, for T > 0."""
    return 2 * np.pi * temperature / np.sqrt(3) * np.sinh(
        np.arcsinh(3 * np.sqrt(3) * density / (2 * np.pi * temperature**3)) / 3
    )


def isobar_state(a, temperature, muB_inf, kinetic_pressure, iterations=60):
    # Neutral densities are n_u=nB, n_d=nB*(1+a), n_s=nB*(1-a).
    fractions = np.stack([np.ones_like(a), 1 + a, 1 - a], axis=-1)
    lower = np.zeros_like(a)
    upper = (muB_inf / 3)**3 / np.pi**2 + muB_inf * temperature**2 / 3
    for _ in range(iterations):
        density = (lower + upper) / 2
        mu = flavor_mu(density[..., None] * fractions, temperature[..., None])
        pressure = np.sum(mu**4 / (4 * np.pi**2) + mu**2 * temperature[..., None]**2 / 2, axis=-1)
        below = pressure < kinetic_pressure
        lower = np.where(below, density, lower)
        upper = np.where(below, upper, density)
    density = (lower + upper) / 2
    mu = flavor_mu(density[..., None] * fractions, temperature[..., None])
    average_mu = np.sum(fractions * mu, axis=-1)
    pressure = np.sum(mu**4 / (4 * np.pi**2) + mu**2 * temperature[..., None]**2 / 2, axis=-1)
    density_residual = mu**3 / np.pi**2 + mu * temperature[..., None]**2 - density[..., None] * fractions
    return average_mu, pressure, density_residual / density[..., None]


def exact_ceiling(muB_0minus, temperature, muB_inf, kinetic_pressure, iterations=60):
    lower, upper = np.zeros_like(temperature), np.ones_like(temperature)
    for _ in range(iterations):
        middle = (lower + upper) / 2
        average_mu, _, _ = isobar_state(middle, temperature, muB_inf, kinetic_pressure, iterations)
        below = average_mu < muB_0minus
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return (lower + upper) / 2


def taylor_ceiling(muB_0minus, muB_inf, temperature):
    return 3 * np.sqrt(
        (muB_0minus / muB_inf - 1)
        * (muB_inf**2 + 3 * np.pi**2 * temperature**2)
        / (muB_inf**2 + 9 * np.pi**2 * temperature**2)
    )


def main():
    saved = np.load(SOURCE, allow_pickle=True).item()
    selected = [target for target in saved["targets"].values() if target["status"] == "success"]
    results = [target["root"]["result"] for target in selected]
    temperature = np.array([r["T_0minus"] for r in results])
    pressure = np.array([r["P_0minus"] for r in results])
    muB_0minus = np.array([r["muB_0minus"] for r in results])
    saved_ceiling = np.array([r["a_0plus_max"] for r in results])
    bag_root = saved["identity"]["physics"]["B_one_forth"]
    muB_inf, kinetic_pressure = equilibrium_muB(pressure, temperature, bag_root)
    exact = exact_ceiling(muB_0minus, temperature, muB_inf, kinetic_pressure)
    refined_difference = np.max(np.abs(exact - exact_ceiling(
        muB_0minus, temperature, muB_inf, kinetic_pressure, iterations=50
    )))
    average_mu, reconstructed_pressure, density_residual = isobar_state(exact, temperature, muB_inf, kinetic_pressure)
    approximate = taylor_ceiling(muB_0minus, muB_inf, temperature)
    error = 100 * (approximate / exact - 1)
    assert np.max(np.abs(exact - saved_ceiling)) < 1e-9
    assert refined_difference < 1e-9
    assert np.max(np.abs(reconstructed_pressure / kinetic_pressure - 1)) < 1e-12
    assert np.max(np.abs(average_mu / muB_0minus - 1)) < 1e-12
    assert np.max(np.abs(density_residual)) < 1e-12

    rows = []
    for i, target in enumerate(selected):
        rows.append(dict(
            target_id=target["id"], T_MeV=float(temperature[i]),
            nB_0minus_over_n0=target["root"]["nB_0minus_over_n0"],
            speed_level_m_s=target["level"], A_exact=float(exact[i]),
            A_taylor=float(approximate[i]), relative_error_percent=float(error[i]),
        ))

    cold_a = np.array([0.146, 0.3, 0.5, 0.7, 0.9, 1.0])
    cold_approximate = 3 * np.sqrt(((1 + (1 + cold_a)**(4/3) + (1 - cold_a)**(4/3)) / 3)**(3/4) - 1)
    cold_rows = [dict(A_exact=float(a), A_taylor=float(b), relative_error_percent=float(100 * (b/a - 1)))
                 for a, b in zip(cold_a, cold_approximate)]

    with np.load(FIGURE) as figure:
        figure_a = float((figure["muB_0minus"] - figure["muB_0plus"]) / figure["muK_0plus"])
        figure_approximate = float(taylor_ceiling(figure["muB_0minus"], figure["muB_inf"], figure["T_0minus"]))

    summary = dict(
        interpretation="Static fixed-pressure ceiling; muB in Taylor coefficients is muB(infinity).",
        source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        scope="Successful targets only from the original seven-level log-temperature checkpoint; not all project states.",
        independent_method="Neutral density cubics and two nested bisections; NumPy only, no solver imports.",
        count=len(rows), count_above_one_percent=int(np.sum(np.abs(error) > 1)),
        temperature_range_MeV=[float(np.min(temperature)), float(np.max(temperature))],
        maximum_error_row=max(rows, key=lambda row: abs(row["relative_error_percent"])),
        maximum_saved_ceiling_difference=float(np.max(np.abs(exact - saved_ceiling))),
        maximum_50_to_60_iteration_difference=float(refined_difference),
        maximum_relative_pressure_residual=float(np.max(np.abs(reconstructed_pressure / kinetic_pressure - 1))),
        maximum_relative_gibbs_residual=float(np.max(np.abs(average_mu / muB_0minus - 1))),
        maximum_density_residual_over_nB=float(np.max(np.abs(density_residual))),
        example_figure=dict(A_exact=figure_a, A_taylor=figure_approximate,
                            relative_error_percent=100 * (figure_approximate / figure_a - 1)),
        cold_examples=cold_rows,
        rows_at_20_MeV=[row for row in rows if abs(row["T_MeV"] - 20) < 1e-8],
    )
    destination = Path(__file__).resolve().parent / "results.json"
    destination.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
