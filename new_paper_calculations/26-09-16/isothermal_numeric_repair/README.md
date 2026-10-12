# Targeted numerical isothermal contour repair

From this directory on the cluster:

```bash
python3 run_isothermal_numerical.py
```

Copy this complete directory, including its three `.npy` baseline inputs and
`baseline-manifest.json`, and the updated repository `RMFsolver/phase_velocity.py`
to the cluster first. Keep this folder at its existing depth under
`new_paper_calculations/26-09-16/`. The script uses the active Python interpreter,
finds the repository relative to itself, and does not assume a laptop virtualenv.
It requires NumPy, SciPy, Matplotlib (including contourpy), tqdm, and the normal
RMFsolver dependencies. Numerical work uses spawned processes and respects
`SLURM_CPUS_PER_TASK`, `PBS_NP`, or `NSLOTS`; otherwise it uses available CPUs.
BLAS/OpenMP threads are limited to one per process. Independent targets run in
parallel, while cold-angle continuation advances from warmer to colder rays.

## What is preserved

The domain, analytical seeds, and numerical checkpoint were copied byte for byte
from `26-08-27/`. SHA-256 hashes are recorded in `baseline-manifest.json` and
checked at startup and after calculation. The original numerical payload has
362 successes, 194 cell timeouts, and 44 failures. None of its records is edited.
The domain and analytical contour are not recomputed.

The production grid and physics remain: 30 piecewise angular rays, 20 radial
fractions, 24 boundary rays, density from 1 to 5.5 times saturation density,
temperature from 0.01 to 120 MeV, `B_one_forth=189.1565957288247`, `ms=0`,
`NM_type="PNM"`, and `upB=5000`. The stored analytical predictor uses `xi=-0.5`;
the exact numerical BVP has no xi parameter. Exact zero temperature and the
singular composition boundary are never submitted to the transport solver.

## What the run computes

The targets are 50, 100, 200, 500, 1000, 2000, and 5000 m/s, using the same
ordinary-speed conversion as the original contour. The first targets extend
the warm 500 curve and reconnect the warm 1000 curve. Cold rays then advance
downward in temperature, followed by other warm levels and angular refinement.

At fixed hidden polar angle, accepted numerical speeds bracket a contour level
in radial position. A safeguarded logarithmic interpolation proposes the next
position, which is actually solved by the numerical BVP. Acceptance requires a
speed error no greater than 0.5%, in addition to the physical/BVP validation.
Analytical currents and velocities propose seeds and locations only.
If no bracket exists, sampling halves the remaining distance toward the relevant
physical boundary, beyond the original radial range 0.02--0.99 when necessary.
The static phase gate and automatic composition ceiling check each new state
before a moving BVP is attempted. Failed evaluations never become zero-speed
or interpolated numerical records. All observed numerical sign-change branches
are retained.

New angular points are added when adjacent accepted points differ by more than
0.05 in density divided by saturation density, or 2 MeV in temperature. Below
1 MeV their temperature ratio may not exceed 1.5. An unresolved angular target
or ambiguous branch correspondence splits the plotted line. Refinement is bounded
at eight angular rounds and eight physical continuation subdivisions; remaining
gaps are reported rather than connected artificially.

## Continuation and cost controls

The live isothermal solver accepts `continuation_guess` and
`return_continuation=True`. Its lightweight `continuation_state` carries the
adapted compact mesh, physical `nK` and `jK`, current, and endpoints. It does not
perform the full profile EOS reconstruction. The next solve transfers that
mesh and rescales both fields to the new endpoints. The exact rate and local
diffusion are still evaluated at the BVP nodes.

Only needed baseline anchors are recomputed to obtain missing profiles. Their
speeds must agree with the preserved value within 1%; a discrepancy is flagged
and that profile is not used. Nearby cached profiles are shared across targets.
Each physical point has at most two trial guesses in a pass. After a failure,
the runner tries a smaller physical continuation step. There is one bounded
second opportunity for unresolved targets after the new profiles are available.

Current brackets span one quarter of the smallest to four times the largest
usable continued/analytical estimate. This removes the old universal
`1e-4*nB_0minus` clip. A converged solution within 5% of a logarithmic bracket
edge requires one expanded-bracket confirmation before becoming a usable point.

Trials retain a disposable-process hard limit of 180 seconds; each contour
target has a 900-second budget and at most 24 root/bracket evaluations. Numerical
defaults remain `n_mesh=300`, `tol_bvp=1e-4`, `max_nodes=50000`, `tail_eps=1e-3`,
and `kappa_factor=1`. These are accuracy-preserving defaults, not a claim that
every cold or near-boundary target will converge.

## Checkpoints and plots

Generated files are ignored by Git:

- `isothermal-contour-repair.npy`: numerical evaluations, accepted added points,
  ray roots, connectivity, failure diagnostics, controls, hashes and validation.
- `isothermal-contour-repair-profiles/`: individual lightweight seed payloads.
- `isothermal-numerical-repaired.png` and `.pdf`: original contours combined
  with validated repair segments, using the established square plotting style.

The full contour-target progress bar starts at 210 targets (4 in smoke mode).
Adaptive refinement and retry batches increase its total as work is scheduled.
Each completed target advances the bar after its result is saved; failed targets
also count as processed, not as successful repairs. The separate `evals` counter
counts atomically saved numerical evaluations in this invocation. There is a
second full bar for the four convergence checks. This is not another 600-cell
scan. The same command resumes accepted points and ray
roots, reuses profiles, and retries unresolved targets. Keep the profile directory
with the checkpoint when resuming. Changed baseline hashes, physics, solver
source, workflow source, or controls are rejected; use a new `--output` filename
for a deliberate new configuration.

The display-only `_repair_progress.py` adapter does not change checkpoint-bound
solver/search hashes, so this progress update can resume existing repair files.
Startup prints the resolved solver path and SHA-256 hash. Passing the startup
API check confirms continuation support; matching the hash to the uploaded
source additionally verifies the exact version.

`run_complete` means the scheduled pass reached a terminal state.
`contours_complete` and `level_complete` additionally require validated,
sufficiently spaced roots across the full angular window. `validation_complete`
records the independent tighter-mesh/tail checks. Inspect these fields rather
than interpreting a completed progress bar as a fully repaired figure.

`26-08-20.ipynb` cells 4--5 now read this overlay when present and fall back to the
preserved numerical contour when absent. They perform no BVP calculation. Its
supplementary panels still describe the original 600-cell grid; curve tracing
does not claim to fill every missing grid value.

## Checks and useful commands

Read-only input/API verification:

```bash
python3 run_isothermal_numerical.py --dry-run
```

Four-target smoke run **on the cluster** (warm 500, warm 1000, cold 200, and
50 m/s at the temperature floor):

```bash
python3 run_isothermal_numerical.py --smoke
```

Smoke uses `isothermal-contour-smoke.npy` and a separate profile cache. For each
recovered category it doubles the initial mesh, reduces the tail gap tenfold,
tightens BVP tolerance fivefold, and requires speed agreement within 1%.
The production run performs the same representative validation. A failed or
missing check is retained explicitly and does not erase an original success.

Replot without calculations:

```bash
python3 run_isothermal_numerical.py --plot-only
```

Worker/time overrides:

```bash
python3 run_isothermal_numerical.py --workers 64 --trial-timeout 180 --target-timeout 900
```

Local development verification uses fast synthetic orchestration tests and two
opt-in, time-bounded warm checks. The full cold recovery and four-target smoke
are cluster work; no production contour was computed during implementation.
