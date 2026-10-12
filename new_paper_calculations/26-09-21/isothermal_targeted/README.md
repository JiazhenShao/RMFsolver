# Designated numerical isothermal shots

Run **on the cluster**, from this folder:

```bash
python3 run_isothermal_numerical.py
```

This is not a new 600-cell scan. It starts with 20 named targets, reuses the
successful 26-09-16 numerical solutions and cached BVP profiles, and only adds
local midpoint checks where the saved curves are under-resolved.
The numerical solver and physical defaults are unchanged.

## What to upload / keep

Upload the Python sources and this README to the same repository-relative path,
`new_paper_calculations/26-09-21/isothermal_targeted/`.
Keep `26-09-16/isothermal_numeric_repair/` beside it, including its Python helpers,
three baseline `.npy` inputs, `isothermal-contour-repair.npy`, and
`isothermal-contour-repair-profiles/`. The latter contains the actual adapted
BVP seeds, not plotting data. The new workflow reads that cache without altering it.
No new `RMFsolver/phase_velocity.py` change is required by this workflow; preflight
requires the same solver SHA-256 as the saved repair checkpoint.

The default source is resolved relative to the script, not the working directory
or a laptop virtual environment. Python uses the active environment, with
NumPy, SciPy, Matplotlib/contourpy, tqdm, and the existing RMFsolver dependencies.
Workers obey allocation variables or available CPUs, with one BLAS thread per
worker. At most one target per speed level runs simultaneously to reuse fresh
profiles. Warm work has two independent chains; more CPU workers do not create
unnecessary targets.

Alternative input/cache locations can be passed explicitly:

```bash
python3 run_isothermal_numerical.py --source /path/to/isothermal_numeric_repair --profile-cache /path/to/isothermal-contour-repair-profiles
```

Missing caches are reported before calculation. There is no automatic rebuilding
of hundreds of anchors. Plotting and read-only preflight do not require profiles.

## Targets and order

1. Warm extension/connection: four midpoint checks on the 500 m/s curve (about
   84.37, 87.02, 89.69, 92.57 MeV) and four on the 1000 m/s curve (65.27, 69.87,
   73.74, 77.04 MeV). Temperatures are calculated exactly from saved endpoint roots.
2. Bends: one midpoint each for 50, 100, 200, 500, and 1000 m/s (about 15.48,
   17.00, 19.60, 24.11, 27.08 MeV), and two for 2000 m/s (16.64, 28.93 MeV).
3. Cold extension: 1000 m/s at 1.25 and 1 MeV; 2000 m/s at 2.3, 1.55 and 1 MeV.
   Saved colder roots on the other levels are retained. Local interval checks
   between saved roots in the 1--10 MeV region follow the five extension targets,
   subject to the same cold allowance. No new point below 1 MeV is submitted.

No 5000 m/s shots, phase boundary reconstruction, broad chemical-potential scan,
or global angular refinement are scheduled.

At fixed temperature, density is predicted from nearby numerical contour roots.
The first actual BVP is accepted when its speed is within 0.5% of the target and
the inherited physical/BVP/endpoint-flux validation passes. Otherwise a bounded
local density secant/bracket correction is tried. Every accepted speed is numerical.
Density search stays within 0.15 n0 of the prediction and within the physical
density window. Current guesses use the requested speed and nearby continued
current, with local quarter-to-four-times bounds; a near-bound solution requires
one expanded-current confirmation, counted within the same allowance.

## Hard cost limits

| Group | Maximum BVP calls |
|---|---:|
| Warm targets and their local curvature checks | 40 |
| Bend targets and their local curvature checks | 44 |
| Cold targets and their local connection checks | 32 |
| Four independent accuracy checks | 4 |
| **Total across all restarts** | **120** |

Each target has at most four BVP calls, including current-bound confirmations;
each trial has a hard 180-second timeout and each target a cumulative 600-second
trial-time allowance. There are no recursive auxiliary BVP chains. Curvature
subdivision creates separately recorded midpoint targets, up to three local
levels and only within the group/global limits. Failed targets are terminal for
this pass, not automatically retried on every restart.

Calls are reserved and atomically saved **before** a child process is launched.
An interrupted reservation remains spent. Re-running the same command resumes
pending work without resetting either successes or allowances. A single-writer
lock prevents two runners spending the same checkpoint's budget concurrently.
The spawned child also arms its own kernel timer before importing the solver,
so its lifetime is bounded even if the parent runner is abruptly terminated.

The initial-target bar has a fixed total of 20; extra local checks appear in its
`refinements` count. A separate full bar reports consumed calls out of 120. That
budget bar need not reach 100% when the requested work completes early.

## Accuracy and honest plotting

The old blanket spacing thresholds are **not** the new connectivity rule.
An interval is certified when its actual numerical midpoint matches the target
speed within 0.5% and differs from the endpoint density chord by at most 0.005 n0.
This is a local resolution check, not a proof of the entire continuous curve.
Larger deviations trigger local midpoint checks, not cosmetic spline smoothing.

Only successful, certified leaf intervals draw new lines through numerical
endpoints and midpoints. Isolated accepted roots remain visible as small dots.
Attempted-but-unresolved intervals mask matching inherited lines; certified
children restore only their validated subintervals. Other branches and original
data records are preserved. Pending work does not falsely claim a new connection.

Four representative roots (warm 500, warm 1000, bend 2000, cold 1000) are checked
with twice the initial mesh, one-tenth the tail gap, and one-fifth BVP tolerance.
Speed agreement must be within 1%. Check results are recorded separately; a
failed check is an explicit accuracy warning, never a claim of validated physics.

Every finished trial is saved before its counter advances. Every finished target
is saved before target progress and a PNG/PDF refresh. The calculation does not
wait for the entire run to produce a new plot.

## Outputs / resume

Generated files are ignored by Git:

- `inputs/`: byte-identical frozen source payloads and their hash manifest.
- `isothermal-targeted.npy`: target states, accepted shots, call ledger,
  provenance, certification and accuracy-check outcomes.
- `profiles/`: new continuation states; keep these when resuming.
- `isothermal-targeted-contours.png` and `.pdf`: refreshed after each target.

The first calculation snapshots inputs; subsequent runs use that snapshot even
if the old cluster run produces a newer checkpoint. Hash changes in snapshots,
the solver, numerical helper, core workflow or controls reject resume. Display-only
changes need not invalidate data. Original source payloads are never rewritten.

Read-only preflight and plot refresh:

```bash
python3 run_isothermal_numerical.py --dry-run
python3 run_isothermal_numerical.py --plot-only
```

For local visualization, download `isothermal-targeted.npy` and `inputs/` into
this folder, then rerun numerical cells 4--5 of `26-08-20.ipynb`. The notebook
verifies input hashes, retains its styling, and shows accepted repair roots even
before any new shots have run. Profiles are unnecessary for plotting.

## Local verification only

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./bin/python3 tests/test_isothermal_targeted.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./bin/python3 tests/test_isothermal_targeted_plot.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./bin/python3 tests/test_isothermal_targeted_trial.py
```

These use synthetic numerical results and read-only saved-data checks, not live
BVP recovery. Actual target convergence and the four accuracy checks must be
established on the cluster. A completed pass can contain failures or deferred
intervals and must not be called a fully repaired contour.
