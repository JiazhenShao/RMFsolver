# Cold numerical isothermal contours

From this directory **on the cluster**, run:

```bash
python3 run_isothermal_numerical.py
```

The command resumes its checkpoint, uses allocated CPUs, and never submits a new
numerical solution above 20 MeV. This is **217 contour targets**, not a rectangular
density/temperature sweep. Existing warm plotting data and accepted numerical
records are immutable. A complete contour is not guaranteed by a completed run:
timeouts and exhausted allowances remain explicit gaps.

## Transfer and preflight

Upload this entire directory, **including `inputs/`**, and the current
`RMFsolver/phase_velocity.py` in the project-relative layout. Keep the remaining
RMFsolver modules and the active cluster Python environment. No laptop Python path
is hardcoded. The new optional `initial_guess_mode`, `continuation_guess`, and
`return_continuation` API arguments are checked before work starts.

The local `inputs/` bundle contains byte-for-byte copies of the warm domain,
numerical checkpoint, repair checkpoint, cold-endpoint checkpoint, and plotting
archive, with SHA-256 hashes in `inputs/manifest.json`. Three previously recovered
0.01-MeV profiles are optional starting seeds; the missing historical cluster
profile cache is **not** required. These profiles seed guesses, not new solutions.
The prepared local bundle also includes 25 compatible saved cold profiles from
26-09-23 (28 profiles total). Their original directories are not runtime dependencies.
If a source-only checkout needs its first bundle, on the machine containing the
historical `local_analysis_results/` sources run:

```bash
python3 _logT_inputs.py --bundle-project
python3 run_isothermal_numerical.py --dry-run
```

Bundling never overwrites a snapshot. Generated inputs/results are ignored by Git;
copying only tracked source files is insufficient for cluster transfer.
Dependencies are the existing RMFsolver dependencies plus NumPy, SciPy,
Matplotlib and tqdm. One BLAS thread is used per worker. Worker discovery follows
SLURM/PBS/NSLOTS or CPU affinity; `--workers 4` overrides it.

## Calculation and cost

- Temperatures: `np.geomspace(0.01, 20.0, 31)`.
- Levels: 50, 100, 200, 500, 1000, 2000, 5000 m/s.
- Density window: 1–5.5 times saturation density.
- Physics: bag parameter 189.1565957288247 MeV, PNM, zero strange mass,
  automatic maximum interface composition. Analytical proposals use `xi=-0.5`.
- Direct-density boundary roots at every scheduled temperature use saved
  boundary seeds, cached thermodynamics and bracket expansion in **density**.
  The upper boundary uses the established static `a_0plus=1-1e-9` convention;
  no moving BVP is run at composition one.
- Seven shared anchor temperatures (indices 0,5,…,30) precede the six
  five-interval blocks. Independent level/block chains run concurrently;
  continuation dependencies can leave some CPUs idle.
- Safeguarded numerical secants correct density; steps scale with distance to
  coexistence. Analytical velocities only propose density/current guesses.
  Every new accepted contour root is a validated numerical result within 0.5%
  of its target speed. Off-target successful solutions/profiles are cached.
- Continuation transfers the adapted mesh and physical density/current profiles.
  Without a profile, the opt-in layer-resolved initializer is used; invalid layer
  scales fall back explicitly to the unchanged default initializer.
- Current bounds span one quarter to four times available estimates; one
  near-bound expansion consumes an ordinary attempt.
- Maximum **180 seconds per trial**, **600 accumulated seconds per target**,
  **four numerical attempts per target**, **two attempts at an identical state**.
  The persistent global limit is **900 numerical attempts**, with 14 reserved
  for tighter checks. Analytical proposal time also consumes the target budget.
  Boundary workers have separate 180-second limits and do not consume BVP slots.
- At most one geometric midpoint per original temperature interval and level
  (maximum 210 extra targets) is scheduled for failures, density jumps above
  0.05, or log-temperature chord curvature above 0.01 in density ratio.
- Fourteen tighter checks cover all levels at 0.01 and 20 MeV: 450 initial nodes,
  BVP tolerance `1e-5`, tail cutoff `1e-4`; require speed agreement within 1%.

This is a ceiling, not a prediction of runtime: 900 full 180-second attempts
would cost 45 aggregate worker-hours, plus boundary/proposal overhead. There is
no automatic enlargement of budgets and no reconstruction of all old anchors.

## Checkpoints and progress

Full bars report boundary temperatures, base contour targets, scheduled
refinements, and accuracy checks. The base bar separately displays numerical
attempts used out of 900; an off-target evaluation is not a completed contour.
Each reservation is atomically saved **before launch**, and every completed
evaluation is saved **before progress advances**. Profiles are saved before the
checkpoint references them. An interrupted reservation remains spent; pending
work resumes without resetting allowances. An interrupted boundary can resume
once (at most two recorded boundary attempts); failed/timed-out boundaries stay
terminal. Spent interrupted accuracy checks remain explicit; a completed saved
check is recovered without rerunning, and a never-launched placeholder is released.

Resume rejects changed physics, axes, controls, warm hashes, solver sources or
workflow versions. Keep a separate output directory for a deliberate changed
experiment. A file lock prevents two numerical writers using the same checkpoint.

Outputs:

- `isothermal-logT.npy`: boundaries, roots, off-target evaluations, spent attempts,
  failures and accuracy-check records.
- `profiles/`: lightweight continuation states from new accepted evaluations.
- `isothermal-logT-plot.npz`: portable saved-data plotting archive and provenance.
- `isothermal-logT.png` and `.pdf`: combined log-temperature figure.

The figure refreshes after each accepted contour target. It uses the newly solved
20-MeV point once, connecting to the immutable warm archive only when the seam
agrees within 0.01 in density ratio. Inconsistent joins are flagged, not forced.
Warm smoothed archive vertices keep their plotting provenance and never become
numerical solver records. Cold lines connect consecutive accepted scheduled
targets only; there is no zero-temperature anchor or analytical replacement.

Download the checkpoint, profiles if continuing elsewhere, and plotting archive
at any time. To redraw partial results without a solver call:

```bash
python3 run_isothermal_numerical.py --plot-only
```

`All_Plots_new.ipynb` has separate saved-data loading and tunable plotting cells
for this archive. The existing linear-temperature plot is unchanged.

## Cluster smoke and verification

```bash
python3 run_isothermal_numerical.py --smoke --workers 4
```

This uses a separate `smoke/` checkpoint: low/high levels at 0.01 MeV, an
intermediate-temperature target, and the 20-MeV join. Endpoint smoke targets get
tighter checks. It is a real bounded cluster calculation, not the local mocked
test suite. Local verification covers orchestration, initializer selection,
continuation transfer, snapshot preservation, persistent budgets, validation,
checkpoint ordering and honest plotting; it does not establish that every cold
contour converges. Exact zero temperature is never evaluated.
