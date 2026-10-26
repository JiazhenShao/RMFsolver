# Notebook contour extension, 2026-10-26

The executed notebook is `../../26-10-26.ipynb`. It contains the nonrelativistic
contour in the supplied reference style and a solid/dashed overlay against the
saved historical relativistic contour. Each figure cell exposes its plotting
parameters and writes PNG/PDF files to `../../local_analysis_results/26-10-26/`.

The notebook loads saved data; it does not launch a numerical scan.

`nonrel-contour-extension.npy` supplements the unchanged October 24 result with
209 accepted numerical solves. These supply an inner shell at radial fraction
0.005 and narrow 5000 m/s brackets on all 30 angular rays. The widest bracket
is 0.0025 in radial fraction, and every upper bracket exceeds 5000 m/s.
The full profiles and trial provenance are retained.

To reproduce or resume the extension from the repository root:

```bash
python3 new_paper_calculations/26-10-26/extend_nonrel_contour.py --workers 4
```

Use a working SciPy installation. On this Mac, prefix the command with the
isolated runtime documented in `../26-10-24/README.md` and use `./bin/python3`.
The script checks that solver/EOS/helper hashes still match the original run.
It retains the strict EOS, conservation, and BVP acceptance gates and uses only
successful nonrelativistic numerical continuation states.

The largest supplementary pressure and enthalpy-flux errors are `6.238e-9` and
`1.665e-10`. All saved rates reproduce the prescribed formula to `4.45e-16`
relative accuracy. `verification.json` records these checks and an audit of the
reference shading at its saved boundary nodes.

The overlay compares the historical August relativistic calculation with the
current nonrelativistic calculation. Changes in transport/rate prescriptions
are included, so it is not a controlled test of relativity alone. The gray
regions reuse the reference figure's thermodynamic boundaries and are visual
context, not the mask used to accept numerical solutions.

## Exact-cold analytical contours

Both analytical methods at `T_0plus=0`, `xi=-0.85` are now computed locally.
The separate [local](local/README.md) and [cluster](cluster/README.md) entry
points share the engine in `shared/`. The completed manifest and permanent
point records are in `local/adiabatic-T0plus0-xi-minus0p85.npy` and its sibling
`-records/` directory. There are 101 full-analytical and 61 semi-analytical
rays, with no unresolved contours or boundaries. The full analytical 2000
and 5000 m/s levels lie outside the fixed plotting window.

`26-10-26.ipynb` contains three additional executed figures, exported to
`local_analysis_results/26-10-26/`: `adiabatic-full-analytical-cold`,
`adiabatic-semi-analytical-cold`, and
`adiabatic-cold-analytical-vs-numerical`, each as PNG/PDF. The comparison
explicitly retains the numerical interface temperature of 0.1 MeV.
All 36 targeted tests and the saved-data audit pass; see the local README
for exact checks and the independently verified cold-limit physics.
