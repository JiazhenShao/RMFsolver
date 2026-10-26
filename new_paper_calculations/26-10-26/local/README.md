# Exact-cold adiabatic analytical contours: local runner

`run_adiabatic_contours.py` evaluates the two slow analytical APIs independently
at exactly `T_0plus=0` and `xi=-0.85`. It uses massless quarks, QMC-RMF3 PNM,
`B_one_forth=189.1565957288247` MeV, and the original cold `3*n0` calibration.
No spatial BVP is solved. The full analytical method uses the quadratic EOS,
leading diffusion and closed integral; the semi-analytical method uses the full
EOS path, full diffusion and numerical integral.

The exact cold endpoint has `D_K_0plus=+inf` and `invD_K_0plus=0`, with finite
positive `I2`, baryon flux and speed. A positive temperature floor is rejected.
Each method computes its own stable-matter and composition-ceiling exclusions;
the warm numerical mask is not reused.

## Run and resume

Run from the repository root with a working Python installation containing
NumPy, SciPy, numdifftools and matplotlib (the latter two are imported by the
existing `RMFsolver` package). The local default is four concurrent calls:

```bash
python3 -B new_paper_calculations/26-10-26/local/run_adiabatic_contours.py
```

The production invocation in this session uses eight workers. On this Mac,
the working isolated SciPy runtime is selected by:

```bash
export PYTHONPATH="$PWD/local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:$PWD"
./bin/python3 -B new_paper_calculations/26-10-26/local/run_adiabatic_contours.py --workers 8
```

That path selects a local dependency installation; the contour engine has no
runtime dependency on another dated calculation. The scan needs the sibling
`local/`, `cluster/`, `shared/` and `inputs/` folders at their current depth below
the repository root, plus the existing `RMFsolver/` directory. The frozen
`inputs/cold-scan-seeds.json` contains 749 upstream coordinates and calibration
values. Its historical source paths and hashes are provenance only: the engine
does not open those warm payloads or import their speeds or masks.

The default output is `local/adiabatic-T0plus0-xi-minus0p85.npy`, relative to the
calculation folder. Run the identical command again to resume; no separate
resume switch is needed. Successful and physically excluded point records are
reused. A previously failed point is evaluated once in each new invocation,
and its earlier failure is retained. The two methods have separate records,
so one method's failure does not invalidate the other's success.

Useful command options:

| Option | Behavior |
|---|---|
| `--method both`, `analytic`, or `semi_analytic` | Select both methods (default) or one API |
| `--workers N` | Set simultaneous bounded calls; BLAS uses one thread per call |
| `--smoke` | Evaluate both selected methods at two representative upstream states; append `-smoke.npy` to the output stem |
| `--dry-run` | Print settings, source hashes and output path; perform no solves or writes |
| `--output PATH.npy` | Choose a separate result; a relative path is relative to the current working directory |
| `--xi VALUE` | Override the default `-0.85`; requires a new output for different physics |
| `--path-rtol VALUE` | Set the semi-analytical path tolerance, default `1e-6` |
| `--angular-rounds N` | Set the angular refinement-round limit, default 6 |
| `--method-timeout SECONDS` | Set the hard per-method deadline, default 120 seconds |

For example:

```bash
python3 -B new_paper_calculations/26-10-26/local/run_adiabatic_contours.py --dry-run
python3 -B new_paper_calculations/26-10-26/local/run_adiabatic_contours.py --smoke --workers 4
```

Each call runs in a disposable spawned process. The 120-second default covers
that method call and its internal recovery attempts. Timeout or interruption
terminates the child. Only a path-table convergence failure triggers stricter
semi-analytical tolerances, using factors `0.5`, `0.2` and `0.1`; only the known
PNM seed-agreement failure enables the two additional nuclear starting guesses.
These retries preserve the original state and acceptance thresholds.

Physical-boundary bisection uses direct cold EOS classification, without
constructing a diffusion table or evaluating a speed integral. The classifier
checks pressure and enthalpy per baryon against each method's cold physical
limits. Additional near-boundary speed calls are made only as needed to bracket
the requested 50–5000 m/s levels. Classification calls have the same hard
deadline as speed calls.

## Checkpoints and completion

Each terminal point call is saved atomically in
`adiabatic-T0plus0-xi-minus0p85-records/`, and the `.npy` manifest is saved after
each finished ray and angular round. Keep both the manifest and its sibling
record directory when moving or archiving a run. Speed records retain endpoint
quantities, the integral, attempts, source identity and failure history.
Boundary records have a separate `boundary-` prefix, an
`evaluation="thermodynamics_only"` label, `interior` or `excluded` status, and
the thermodynamic margins. Boundary failures remain distinct failed records
and are included in resume retries and completion checks.

Resume requires identical exact-cold physics, seed-manifest bytes, timeout,
path tolerance, angular settings and smoke mode. Source identity hashes
`shared/cold_scan.py`, `shared/cold_physics.py`, `shared/bounded_call.py`, and
every `RMFsolver/*.py` file. A cold/warm or source/settings mismatch is rejected;
choose a new `--output` instead. Worker count can change. Plotting and independent
verification scripts are outside the scan's physics identity.

The initial production workflow made five 120-second calls extremely close
to the stable-matter boundary, where the vanishing-integral diffusion table
stalled in `_tabulate`. The boundary-classification change removes those
unnecessary tables without changing the speed APIs or physics. A one-time
verified migration reused 2,424 valid records (1,898 accepted and 526 excluded):
the original physics module was an identical AST prefix of the updated module,
and the EOS and input hashes matched. Reused rows preserve
`calculation_identity` and a `reuse_verification` explanation; the five obsolete
boundary failures are archived separately. The audit is
`local_analysis_results/26-10-26/cold-boundary-workflow-recovery.json`.
This explicit audited migration is not a general relaxation of resume hash
checks. Production resumed with eight workers; its final coverage remains a
manifest-level result.

Production completion is not established by the representative verification
report. Inspect the saved manifest, rather than inferring coverage from a
finished subprocess or the presence of a file:

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np
p = Path('new_paper_calculations/26-10-26/local/adiabatic-T0plus0-xi-minus0p85.npy')
d = np.load(p, allow_pickle=True).item()
print('complete:', d['complete'])
print('methods present:', sorted(d['methods']))
print('coverage:', d.get('coverage', 'scan still in progress'))
print('unresolved angular intervals:', d.get('unresolved_angular_intervals', 'not finalized'))
print('record directory:', p.parent / d['record_directory'])
PY
```

For the delivered two-method comparison, require both `analytic` and
`semi_analytic` in `coverage`, `complete=True`, and inspect each method's
`coverage`. A completed invocation exits with status 0; unresolved coverage
exits with status 2. `supported_levels` means a level has a resolved crossing
somewhere in the window, not on every ray. `outside_window` is distinct from an
unresolved or failed point.

The common window is `1 <= nB_0minus/n0 <= 5.5` and
`0 <= T_0minus <= 115` MeV. Requested levels are 50, 100, 200, 500, 1000, 2000
and 5000 m/s. A resolved radial root requires relative speed error below
`1e-3` and radial bracket width at most `0.001`. Physical boundary brackets
must be at most `1e-4` wide. Angular midpoint geometry is checked at `0.0025`
in coordinates normalized to the display window, with further refinement at
status transitions. Failed evaluations never establish a physical exclusion
or a contour connection.

## Verification and figures

Verification includes 25 orchestration tests, run with
`COLD_CONTOUR_LIVE=1` and the isolated SciPy runtime, and 11 analytical
slow-front physics tests. The orchestration suite includes ten live
method/state classifier checks against direct thermodynamic domains, with the
full-EOS diffusion-table step bypassed, plus a regression showing that boundary
bisection does not request nearly zero-speed integrals. It also checks that
boundary records are separate and boundary failures are retried.
The independent report
`local_analysis_results/26-10-26/cold-analytic-physics-verification.json` has
`passed=true` at all three upstream states `(T_0minus, nB_0minus/n0)`:
`(0, 3.5)`, `(30, 3.5)` and `(70, 2.5)`.
It checks direct conservation, rate and diffusion formulas, cold integrals,
warm-to-cold approach, and path refinement. The largest semi-analytical
conservation residual is `2.93e-11`, the independent integral difference is
`3.01e-8` relative or less, and halving the path tolerance changes speed by at
most `4.67e-9` relative. Its integral quadrature bypasses the stored rate spline
but reuses the diffusion primitive; this is not an entirely independent path
construction. These checks validate representative states, not every contour
point or the final coverage.

Warm-to-cold convergence is slow and follows fractional powers: lowering the
interface temperature from `0.01` to `1e-5` MeV reduces the speed difference
from about 3.4–4.0% to about 0.34–0.40%. A tiny positive temperature must not be
treated as numerically identical to exact zero.

The independent report can be regenerated from the repository root:

```bash
python3 -B new_paper_calculations/26-10-26/shared/verify_cold_physics.py
```

To repeat the full orchestration suite, including the live classifier checks,
use the working runtime above and run:

```bash
COLD_CONTOUR_LIVE=1 ./bin/python3 -B -m unittest discover -s tests -p test_cold_adiabatic_contours.py -v
```

With the completed production manifest,
`shared/add_cold_figures.py` appends/replaces its tagged cells in
`26-10-26.ipynb`: individual full analytical and semi-analytical figures, then
an overlay with the existing numerical reference. Execute the notebook cells
in order to load the existing numerical data and export PNG/PDF figures.
The figure cells expose their plotting parameters and perform no solves.
Only supported, adjacent resolved crossings are connected; individual gray
regions use each method's recomputed cold boundaries. The combined overlay
does not impose one method's shading on the others. Its numerical curves are
explicitly at **`T_0plus=0.1` MeV**, so this is a comparison at different
interface temperatures, as well as different approximations.

## Completed local result (2026-10-26)

The saved two-method manifest is complete: 101 full-analytical rays and 61
semi-analytical rays, with zero failed samples, unresolved contours or
unresolved boundaries. It retains 3,357 accepted full-analytical speed records
and 2,923 accepted semi-analytical speed records, plus physical exclusions and
separate thermodynamic boundary checks.

The full analytical method supports 50, 100, 200, 500 and 1000 m/s in the
chosen window; its 2000 and 5000 m/s levels lie outside the window. The
semi-analytical method supports all seven levels through 5000 m/s.
The maximum pressure, enthalpy-per-baryon and constant-density errors are
`1.818e-9`, `2.149e-10` and `7.742e-10`, below the unchanged `1e-8` gate.
The largest level residual is `9.993e-4`, radial bracket width `9.994e-4`,
and accepted angular midpoint displacement `0.001598`, satisfying their
respective `1e-3`, `0.001` and `0.0025` criteria.

All 36 targeted tests pass (25 orchestration/classifier checks and 11
analytical physics regressions). Local and cluster smoke results agree
bit-for-bit. The final saved-data audit is
`local_analysis_results/26-10-26/cold-contour-verification.json`.
The three new notebook figures are executed and exported as PNG/PDF; the
original seven notebook cells and numerical artifacts are preserved.

The initial five-round angular budget needed a sixth round to certify
where curves cross the plot window. Six is now the default. The budget
extension reused 8,533 records after verifying that the only source change
was the default round count, with unchanged physics and inputs; records
retain their calculation identity and sampling-budget provenance.
`local_analysis_results/26-10-26/cold-angular-budget-extension.json`
records that check. The final resumed invocation took 262.4 seconds; this
is only the last refinement stage, not the total calculation time.
