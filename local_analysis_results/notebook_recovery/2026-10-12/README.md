# Notebook recovery

Restored `26-09-03.ipynb`, formerly `FL26-3.ipynb`, from this chat's actual
historical cell-source reads and recorded editing scripts through September 9,
2026, 15:23 UTC. The October 2 rename is documented in `wiki/notebook-renames.md`.

The damaged file had been reduced to five placeholder cells, including a
fabricated linear bag-calibration formula and a root solver unrelated to the
physical residuals. It is preserved unchanged as
`26-09-03.damaged-before-restore.ipynb`.

The recovered eight cells include the physical calibration and boundary
functions, cached boundary loading, the 2n0/3n0/4n0 numerical contour panels,
and the 3n0 isothermal enthalpy-versus-Gibbs comparison through 120 MeV.
Both saved PNG exports were restored as notebook outputs, explicitly labeled
as recovered figures rather than freshly calculated results.

Verification: notebook schema, all code-cell syntax, per-cell source hashes,
three saved boundary sets including angular endpoints, 52 cached isothermal
records, and both PNG files. The live notebook is byte-identical to
`26-09-03.recovered.ipynb` after restoration.

Full execution was not verified: the existing project SciPy binary fails to
import `_spropack`. The plot-only rerender also hits this dependency through
SciPy interpolation. No environment, solver, cluster result, or other notebook
was modified. Existing saved data and exports were not overwritten.

`recover_notebook.py` reconstructs the source; `verify_cached_plots.py` validates
the caches and reinstates saved figures. `provenance.json` records exact history
locations, timestamps, and source hashes. Historical code was recovered, not
redesigned to satisfy old tests that predate the eight-cell notebook.
