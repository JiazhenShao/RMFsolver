# RMFsolver

RMFsolver is a scientific Python package for relativistic mean-field (RMF) calculations of dense matter relevant to neutron stars. It provides tools to solve RMF equations for hadronic matter, compute thermodynamic quantities and equations of state, evaluate neutron-star structure from tabulated EOS input, and analyze selected hadron-to-quark phase-transition quantities, including phase-boundary and conversion-velocity observables.

## Key Features

- Solve RMF equations for beta-equilibrated matter, pure neutron matter, and symmetric nuclear matter.
- Compute thermodynamic quantities such as pressure, energy density, baryon density, entropy, and binding energy.
- Analyze phase-transition and phase-boundary properties with utilities in `phase_velocity.py`.
- Solve Tolman-Oppenheimer-Volkoff (TOV) equations for tabulated equations of state.
- Designed for research calculations, parameter studies, and rapid prototyping.

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install the package locally:

```bash
pip install -e .
```

## Quick Example

```python
from RMFsolver import RMFparameter, Solver, constants

# set parameters
params = RMFparameter.paraQMCRMF3
n0 = params[10]

# call solver
pressure = Solver.RMFpressurePNM(n0, "nB", 0.0, params) / constants.MeV_fm**3
energy_density = Solver.RMFedensPNM(n0, "nB", 0.0, params) / constants.MeV_fm**3

# print result
print(f"P(n0) = {pressure:.3f} MeV/fm^3")
print(f"epsilon(n0) = {energy_density:.3f} MeV/fm^3")
```

## Package Structure

- `Solver.py`: core RMF solvers and thermodynamic routines.
- `phase_velocity.py`: tools for phase-transition, phase-boundary, and conversion-front calculations.
- `tov.py`: TOV solver for neutron-star structure from tabulated EOS input.
- `tov_solve.py`: convenience helpers for running TOV mass-radius sequences from EOS tables.
- `constants.py`: physical constants and unit-conversion factors.
- `RMFparameter.py`: built-in RMF parameter sets.
- `SQMsolver.py`: simple bag-model quark-matter helper functions.
- `TM1ecrust+BPS.table`: bundled crust EOS table used by the TOV utilities.

## Physics Notes

- The package is based on the relativistic mean-field approximation for dense hadronic matter.
- Typical applications are neutron-star equation-of-state studies and related dense-matter thermodynamics.
- Several routines assume equilibrium conditions such as beta equilibrium or isospin symmetry, depending on the solver used.
- Phase-transition utilities are model-dependent and should be interpreted within the assumptions of the chosen RMF and quark-matter descriptions.

## Limitations

- This is not a full astrophysical simulation framework.
- The microphysics is simplified and model-dependent relative to full production EOS pipelines.
- The code is intended for research exploration, benchmarking, and prototyping rather than turnkey inference or large-scale simulation workflows.

## Citation / Usage

If you use this code, please cite the relevant literature for the RMF model or method you apply, or contact the author for project-specific citation guidance.
