# Analytical isothermal log-temperature contours: xi = -0.85

From this directory on the cluster:

```bash
python3 run_isothermal_analytic.py
```

This runs the existing hydro-consistent `analytic_velocity_isothermal` method,
with automatic maximum interface composition and **xi = -0.85**. There is no
numerical BVP, profile transfer, numerical current search, or nested analytical
starting-density scan. Every accepted velocity is newly calculated with this xi;
old warm numerical curves are not spliced into the analytical result.

## Transfer and preflight

Upload this folder's six Python files, this README, and **both files in `inputs/`**:
`manifest.json` and `static-boundaries.npy`. Retain the repository layout
`new_paper_calculations/26-10-02/isothermal_analytic_logT/` and the current
`RMFsolver/` at the repository root. No other dated folder is needed at runtime.
Use the working cluster Python environment; no laptop interpreter is hardcoded.
No RMFsolver files were changed for this workflow.

```bash
python3 run_isothermal_analytic.py --dry-run
python3 run_isothermal_analytic.py --smoke --workers 4
```

Dry-run checks input hashes, the analytical API, resume compatibility and the
schedule without importing the EOS or solving anything. Smoke is an optional
**real cluster calculation**, with separate output in `smoke/`: 50 and 5000 m/s
at 0.01 MeV, 500 m/s at an intermediate temperature, and 1000 m/s at 80 MeV.
The warm smoke target may legitimately lie outside the physical density window.

The default uses the first positive SLURM/PBS/NSLOTS CPU allocation, then CPU
affinity or machine CPU count. Use `--workers 4` to override it. One BLAS thread
is used per child. Independent temperatures run concurrently; the seven speed
levels at one temperature share previous evaluations and run in order, so at
most 46 temperature chains can be active at once.

## Physics and grid

- Bag parameter: `B_one_forth=189.1565957288247` MeV; PNM; zero strange mass;
  `upB=5000`; `xi=-0.85` explicitly passed to every speed evaluation.
- Window: density ratio 1–5.5, temperature 0.01–120 MeV; neither exactly zero
  temperature nor an exactly unit interface fraction is evaluated as a front.
- Temperatures: the existing 31 values `geomspace(0.01,20,31)`, followed by
  `geomspace(20,120,16)[1:]`. Thus 46 temperatures with 20 MeV included once.
- Seven levels: 50, 100, 200, 500, 1000, 2000 and 5000 m/s.
- **322 initial contour targets**, not a rectangular density–temperature grid.
  Multiple analytical evaluations may be required per contour root; they are
  counted separately from completed targets. No automatic midpoint campaign is
  scheduled by this workflow.

The input bundle retains only xi-independent thermodynamic boundaries and saved
boundary geometry. It has 61 validated cold boundary-temperature records from
the downloaded numerical run; all 31 scheduled cold temperatures are reused.
Roundoff-level differences in `geomspace` across NumPy builds are matched without
interpolating a boundary root. New warm boundaries use cached direct fixed-density
PNM recovery and bracketed density roots, never broad chemical-potential scans.
The upper static boundary uses `a_0plus=1-1e-9` as before.

At warm temperatures a boundary can leave the density window. The interval is
then clipped to 1–5.5; a whole excluded interval is recorded as `outside_window`.
Validated analytical speeds at clipped edges distinguish an out-of-window target
from a numerical failure. This tracks the established increasing-speed branch;
it is not a search for additional disconnected/non-monotone branches.

Density correction uses cached analytical values and safeguarded secants. Initial
distances and recovery steps scale with the distance from coexistence. A failed
evaluation never becomes a speed bracket. The exact zero at coexistence is only
a bracket anchor, not a positive-speed contour root. Each accepted target must
match its speed within 0.5% and pass the analytical formula, composition and
momentum-flux residual checks and the public solver's slow-front validity gate.
The public API itself retains EOS and endpoint validation.

## Checkpointing, progress and bounds

- 180-second killable limit per analytical evaluation or new boundary task.
- At most 24 analytical evaluations and 600 accumulated evaluation seconds per
  target; shared successful evaluations cost no additional solve.
- Reservations are saved before launch; every completed evaluation is atomically
  saved before its progress report. Off-target speeds are retained for subsequent
  levels at the same temperature.
- Full bars separately show static-boundary temperatures and contour targets.
  The target bar also reports the analytical-evaluation count and accepted roots.
- Successful and terminal unresolved targets are retained on default resume.
  Interrupted reservations remain spent; pending targets resume within their
  existing allowances. An interrupted boundary can be attempted once more.
- Resume rejects changed xi/physics, axes, controls, source hashes or input hashes.
  Use `--output-dir DIFFERENT_DIRECTORY` for deliberately changed settings.
- One writer owns the checkpoint. Main-thread figure refreshes coalesce bursts
  of newly accepted roots and run outside the checkpoint lock; all pointwise data
  saves remain immediate. A final refresh always runs.

The single command resumes and produces:

- `isothermal-analytic-logT.npy`: static boundaries, analytical evaluations,
  contour roots, diagnostics and completion flags.
- `isothermal-analytic-logT-plot.npz`: portable plotting data without pickle.
- `isothermal-analytic-logT.png` and `.pdf`: analytical log-temperature figure.

Only consecutive accepted contour targets are connected. Failures remain gaps;
there is no interpolated velocity substitution. `work_complete` distinguishes
terminal work from `contours_complete`; the command returns exit code 2 if
unresolved targets remain. Out-of-window targets are physical exclusions, not
solver failures. Static boundary geometry alone supplies the gray regions.

Download partial results at any time. To redraw without any solver call:

```bash
python3 run_isothermal_analytic.py --plot-only
```

The plotting module exposes the established window, fonts, colors and line
settings through `DEFAULT_STYLE`; the title identifies analytical xi=-0.85.
No notebook or previous figure was overwritten by this preparation.

## Cleanup and verification

Copied numerical outputs, input velocity datasets, profiles, entry point and
superseded helpers were moved recoverably to the sibling folder
`legacy_backup_isothermal_analytic_logT/`. Do not upload that backup. The original
26-10-01 workflow/data remain untouched; their hashes are recorded in the new
static-only manifest. The preparation helper `_logT_inputs.py --source PATH`
exists only to recreate a fresh bundle from that original download; it is not a
cluster startup step and refuses to overwrite a prepared bundle.

Local verification is limited to mocked orchestration, analytical-call routing,
residual rejection, cache/resume persistence, edge classification, input hashes,
gap-preserving plotting, and dry runs. No expensive production calculation or
live analytical smoke was run locally. Cluster convergence is not guaranteed by
these orchestration checks.
