"""Minimal usage example for the public RMFsolver package."""

import numpy as np

from RMFsolver import RMFparameter, Solver, constants, tov


def main():
    params = RMFparameter.paraQMCRMF3
    n0 = params[10]

    # Example 1: evaluate a single RMF pressure point for pure neutron matter.
    pnm_pressure = Solver.RMFpressurePNM(n0, "nB", 0.0, params) / constants.MeV_fm**3
    print(f"PNM pressure at n0: {pnm_pressure:.3f} MeV/fm^3")

    # Example 2: feed a small tabulated core EOS into the TOV solver.
    # Replace this toy table with a production EOS before doing physics runs.
    energy_density = np.array([80.0, 120.0, 180.0, 260.0])
    pressure = np.array([0.3, 1.2, 4.0, 10.0])

    star = tov.TOV(energy_density, pressure, add_crust=True)
    radius_km, mass_msun, *_ = star.solve(180.0, rmax=20e5, dr=500)

    print(f"Toy-EOS radius: {radius_km:.3f} km")
    print(f"Toy-EOS mass: {mass_msun:.3f} Msun")


if __name__ == "__main__":
    main()
