# RMFsolver

RMFsolver is a scientific Python package for relativistic mean-field (RMF) calculations of dense matter relevant to neutron stars. It solves RMF equations for hadronic matter, computes thermodynamic quantities and equations of state, evaluates neutron-star structure from tabulated EOS input, and analyzes hadron-to-quark phase-transition quantities — in particular phase-boundary and conversion-front velocity observables.

This repository holds two things: the package itself under `RMFsolver/`, and the research knowledge base that documents it under `wiki/`.

## Installation

Requires Python 3.12.

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick example

```python
from RMFsolver import RMFparameter, Solver, constants

params = RMFparameter.paraQMCRMF3
n0 = params[10]

pressure = Solver.RMFpressurePNM(n0, "nB", 0.0, params) / constants.MeV_fm**3
energy_density = Solver.RMFedensPNM(n0, "nB", 0.0, params) / constants.MeV_fm**3

print(f"P(n0) = {pressure:.3f} MeV/fm^3")
print(f"epsilon(n0) = {energy_density:.3f} MeV/fm^3")
```

## Package structure

| File | Role |
|------|------|
| `Solver.py` | Core RMF solvers and thermodynamic routines: beta-equilibrated matter, pure neutron matter (PNM), symmetric nuclear matter (SYM). |
| `phase_velocity.py` | Phase-transition front calculations: analytic velocity bounds, isothermal / adiabatic / energy-conserving front solvers, BVP steady-front profiles. |
| `SQMsolver.py` | Strange- and quark-matter thermodynamics, including finite-temperature quark-matter EOS. |
| `tov.py` | TOV solver for neutron-star structure from a tabulated EOS. |
| `tov_solve.py` | Convenience wrappers for TOV mass-radius sequences. |
| `constants.py` | Physical constants and unit conversions. |
| `RMFparameter.py` | Built-in RMF parameter sets (`paraQMCRMF3` is the default used throughout). |
| `TM1ecrust+BPS.table` | Bundled crust EOS table used by the TOV utilities. |

## Conventions

- All thermodynamic quantities are in natural MeV/fm units unless stated otherwise; `constants.MeV_fm = 197.327 MeV·fm` is the key conversion.
- Endpoints across the conversion front use location-based names: `_0minus` (nuclear side), `_0plus` (just inside the quark side), `_inf` (downstream asymptote). In formulas these are written `X(0^-)`, `X(0^+)`, `X(\infty)`.
- Quark densities: `P_f`, `E_f`, `n_B` are single-species helpers with spin included and color excluded, while `n_u`, `n_d`, `n_s` include the color factor.

## `wiki/` — research knowledge base

`wiki/` is a standalone [Obsidian](https://obsidian.md) vault recording the durable conclusions behind the code: the physics, solver behavior, numerical methodology, and known issues. It is written for retrieval rather than reading front to back — start at [`wiki/index.md`](wiki/index.md), which lists every page with a one-line summary. `wiki/SCHEMA.md` defines the page format.

Live code is authoritative for current behavior; the wiki is authoritative for conclusions, reversals, and history.

## Scope and limitations

- This is a research code, not a full astrophysical simulation framework, and not a turnkey inference pipeline.
- The microphysics is simplified and model-dependent relative to production EOS pipelines.
- Phase-transition results should be interpreted within the assumptions of the chosen RMF and quark-matter descriptions.

## Citation

If you use this code, please cite the relevant literature for the RMF model or method you apply, or contact the author for project-specific citation guidance.
