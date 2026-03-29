"""Helpers for solving TOV sequences from tabulated EOS data."""

from pathlib import Path

import numpy as np

from .tov import TOV

__all__ = ["load_eos_table", "solve_mass_radius_curve", "solve_mass_radius_curve_from_file"]


def load_eos_table(path, energy_column="energy_density", pressure_column="pressure", skip_header=1):
    """Load a tabulated EOS file.

    Inputs:
    - path: text file readable by ``numpy.genfromtxt``
    - energy_column: column name for energy density in MeV/fm^3
    - pressure_column: column name for pressure in MeV/fm^3
    - skip_header: number of header rows to skip

    Outputs:
    - energy_density: 1D numpy array in MeV/fm^3
    - pressure: 1D numpy array in MeV/fm^3
    """
    table = np.genfromtxt(path, names=True, skip_header=skip_header)
    return np.asarray(table[energy_column], dtype=float), np.asarray(table[pressure_column], dtype=float)


def solve_mass_radius_curve(energy_density, pressure, central_densities, add_crust=False, **solve_kwargs):
    """Solve a TOV mass-radius sequence for a set of central densities.

    Inputs:
    - energy_density: EOS energy density array in MeV/fm^3
    - pressure: EOS pressure array in MeV/fm^3
    - central_densities: iterable of central densities in MeV/fm^3
    - add_crust: whether to merge the bundled crust table
    - solve_kwargs: forwarded to ``TOV.solve()``

    Outputs:
    - radii_km: radii in km
    - masses_msun: masses in solar masses
    - central_densities: numpy array of the sampled central densities
    """
    solver = TOV(np.asarray(energy_density, dtype=float), np.asarray(pressure, dtype=float), add_crust=add_crust)

    radii_km = []
    masses_msun = []
    sampled_densities = []

    for dens_c in np.asarray(central_densities, dtype=float):
        radius_km, mass_msun, *_ = solver.solve(dens_c, **solve_kwargs)
        radii_km.append(radius_km)
        masses_msun.append(mass_msun)
        sampled_densities.append(dens_c)

    return np.asarray(radii_km), np.asarray(masses_msun), np.asarray(sampled_densities)


def solve_mass_radius_curve_from_file(path, central_densities, add_crust=False, **solve_kwargs):
    """Load a tabulated EOS file and solve its TOV mass-radius curve."""
    energy_density, pressure = load_eos_table(Path(path))
    return solve_mass_radius_curve(energy_density, pressure, central_densities, add_crust=add_crust, **solve_kwargs)
