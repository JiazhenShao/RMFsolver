# Fixed T(0+) = 0.1 MeV analytical comparison

Run from this folder in the cluster's activated `.NSM` environment:

```bash
python3 run_T0plus0p1_analytical.py --workers 64
```

Choose the worker count to match your allocation. Run just this one command;
it computes both the full analytical and semi-analytical methods. It never
runs the full numerical BVP or recalculates either boundary.

## What Is Reused

- `_method_engine.py` and `_analytic_ray_grid.py` are exact copies from
  `26-08-21/`. In particular, `evaluate_analytical_point` is unchanged: it
  recovers the upstream chemical potential once, then calls the two existing
  `RMFsolver.phase_velocity` analytical APIs.
- The subprocess watchdog in `_bounded_call.py` is adapted from
  `26-08-21/_radial_shell_continuation.py`; cancellation was added.
- `inputs/T0plus0p1-domain.npy` is the unchanged `26-08-21` domain.
- `inputs/T0plus0p1-uNmax-state-of-art-complete-5km.npy` is the unchanged
  consolidated `26-08-24` numerical result. The filename retains its legacy
  observable name for compatibility.
- Physics is unchanged: T(0+) = 0.1 MeV, cold critical density = 3 n0,
  B^(1/4) = 189.1565957288247 MeV, xi = -0.5, ms = 0, QMC-RMF3 PNM.

Upload this entire folder, including `inputs/`, under the cluster repository's
`new_paper_calculations/`. It uses that repository's existing `RMFsolver/`;
the older dated run folders are not runtime dependencies. Use the same current
solver source on local machine and cluster. The checkpoint records source hashes.

## Grid And Acceptance

The initial union contains **630 unique points**:

1. All 540 exact points of the finished numerical plotting grid (30 x 18).
2. The original 600-point domain (30 x 20), adding the outer radial fractions
   0.90, 0.95 and 0.99 not present in the plotting grid.

Shared points are evaluated only once. Both theta = 0 and theta = 90 degrees
are present; theta = 0 uses T(0-) = 0 exactly. Here `radial_fraction` is the
fraction of the interval between the saved inner and outer boundaries, not
the absolute elliptical radius stored as `rho` in the domain manifest.

After the initial grid, the runner finds an adjacent pair of successful
semi-analytical points straddling 5000 m/s on every ray. It bisects wide
brackets until their radial-fraction width is at most 0.01. If no upper
bracket exists, it searches outward, bounded by radial fraction 0.999.
A failed sample is a gap, not a value to interpolate across: recovery samples
the interval before that failure. Recovery is capped at 12 rounds by default.

Success means **30/30 resolved brackets**, including both angular endpoints.
The saved 30 x 2 `semi_analytic_bracket_grid` gives a connected contour strip
made only from converged solver samples. Ordinary contour interpolation
between those samples is allowed; failed solver values are never filled.
This is a sampling/coverage criterion, not a proof of continuum accuracy.
There is deliberately no 5000 m/s requirement for the full analytical method.

The existing numerical grid alone is insufficient to guarantee a
semi-analytical 5000 m/s contour: its speed reaches that level farther outward.
This is why the original domain's outer samples are retained.

## Progress, Checkpoints And Resume

- One tqdm increment per completed physical point, after both methods finish.
  The initial total is 630; it grows when adaptive points are scheduled.
- Atomic save after **every** terminal point, including failures. The NPY can
  be downloaded during the calculation without reading a half-written file.
- Hard 600-second wall-clock cap per paired point, including upstream recovery
  and both analytical calls. An over-budget child is terminated and the point
  is recorded as failed. No failed point supplies a plotted velocity.
- Resume by running the same command again. Accepted and failed terminal
  points are retained; unfinished points resume. Adaptive refinement seeks
  fresh nearby points rather than repeatedly rerunning identical failures.
- Ctrl-C cancels active children; restart from the last atomic checkpoint.
- Changed source files, input data, or physics/refinement settings require a
  new `--output`. Worker count may change on resume.

Default output: `T0plus0p1-analytical-comparison.npy`. It contains:

- `records`: raw per-point full/semi results, failure messages and timings.
- `matched_numerical_grid`: both methods on the exact 30 x 18 numerical grid.
- `original_domain_grid`: both methods on the original 30 x 20 domain.
- `semi_analytic_bracket_grid`: accepted narrow 5000 m/s bracket pairs.
- `numerical_reference_grid`: the existing numerical plotting values.
- Saved boundaries, calibration, all point coordinates, and provenance hashes.
- `coverage`, including unresolved theta indices, and `complete_5000_contour`.

`run_complete=True` means the bounded scheduling finished; it does **not**
mean every solver call succeeded. Exit code 0 requires the target contour
coverage; exit code 2 flags unresolved rays. Inspect the saved diagnostics
before claiming a complete curve.

## Checks And Tunables

```bash
python3 run_T0plus0p1_analytical.py --dry-run
python3 -m unittest discover -s . -p 'test_*.py'
python3 run_T0plus0p1_analytical.py --smoke --workers 3
```

The real smoke run evaluates three points: the cold inner endpoint, an
interior near-outer point, and the hot theta = 90 degree outer endpoint.
It writes a separate `*-smoke.npy`, never the production checkpoint.

Defaults are grouped at the top of the runner. CLI options include
`--workers`, `--point-timeout`, `--bracket-width`, `--max-rho`,
`--adaptive-rounds`, and `--output`. For a tighter run use a new output:

```bash
python3 run_T0plus0p1_analytical.py --workers 64 --bracket-width 0.005 --output T0plus0p1-analytical-fine.npy
```

## Local Verification (2026-09-28)

Syntax checks and 15 tests passed, including hard timeout, atomic checkpoint,
resume, endpoint preservation, failed-neighbor rejection, and a complete
mocked adaptive run (690 points, 30/30 target brackets).
All six real analytical evaluations in the three-point smoke test succeeded:

| Theta index | Radial fraction | Full [m/s] | Semi [m/s] |
| --- | --- | --- | --- |
| 0 | 0.02 | 43.418 | 38.661 |
| 15 | 0.84 | 862.802 | 1766.183 |
| 29 | 0.99 | 1793.810 | 15799.715 |

Smoke evidence is in
`local_analysis_results/26-09-28/analytical-verification-smoke.npy` at the
repository root. These tests do not substitute for the production scan.
The local SciPy 1.15.2 wheel failed to load on macOS; testing used SciPy 1.18.1
in an isolated temporary directory, without modifying the project environment.
No SciPy upgrade is required by the new runner itself.

## Local Recovery

`repair_analytical_local.py` repairs the downloaded comparison in place:

```bash
python3 repair_analytical_local.py --workers 4 --timeout 120
```

It first creates `T0plus0p1-analytical-comparison.pre-local-repair.npy`, without
overwriting an existing backup. It retries only failed method results using
the existing validated direct-density PNM helper; the analytical formulas and
xi=-0.5 are unchanged. Each method has its own hard timeout, so a semi-analytical
timeout does not discard a successful full-analytical result. Every completed
physical point triggers one progress increment and an atomic checkpoint.

It also samples radial fraction 0.005 on every ray and refines any remaining
adjacent successful bracket of 5000 m/s to width 0.01. It does not bridge failed
points or extrapolate solver values. The new `repaired_plot_grid` combines
this inner row with the original domain grid; existing grid fields are refreshed.
Successful cluster values and the numerical reference remain unchanged.
`local_repairs` records local versions, hashes, backup and completion status;
per-point `local_repair_history` preserves superseded failure records, and local
method results include upstream chemical potential and density residual.

The original cluster contract/version metadata is retained as provenance.
Use the repair script, not the original cluster runner, to resume this mixed
cluster/local artifact. `All_Plots_new.ipynb` uses `repaired_plot_grid` when present.
