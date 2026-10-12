# Numerical isothermal failed-cell repair run

This directory reruns only the non-successful cells from the completed
piecewise-angular 26-08-25 numerical contour. It does not rebuild either
physical boundary, rerun the analytical contour, or recompute cells already
stored with `task_status="success"`.

The copied checkpoint contains 358 accepted cells and 242 non-successful cells.
At startup the repair workflow preserves every accepted record and associated
grid value, clears only eligible non-success records, marks those cells
`pending`, and processes them through the current `solve_front_isothermal`.
That live solver evolves separately normalized internal representations of
physical `nK` and `jK`, while all returned endpoint and profile quantities
remain in physical units.

## Required inputs

The three compatible inputs are copied into this directory:

```text
26-08-27/
├── isothermal-domain.npy
├── isothermal-analytic.npy
├── isothermal-numerical.npy
└── run_isothermal_numerical.py
```

- `isothermal-domain.npy` is the completed 30-by-20, 600-cell domain.
- `isothermal-analytic.npy` supplies only per-cell baryon-current seeds.
- `isothermal-numerical.npy` is both the copied starting checkpoint and the
  pointwise repair output.

The three payloads must have the same schema, axes, physics, live-API
signature, and domain fingerprint. No domain-preparation command is part of
this workflow.

## Production command

From this directory, run:

```bash
python3 run_isothermal_numerical.py
```

The worker count comes from `SLURM_CPUS_PER_TASK`, `PBS_NP`, or `NSLOTS`, then
falls back to all CPUs visible to Python. Numerical work proceeds through the
20 radial shells in order so each shell can seed the next one; the 30 angular
rays within one shell run concurrently.

The progress bar still contains all 600 cells and starts with the saved
successes already complete. Each non-successful `(ray_index, radial_index)` is
rerun once in radial-shell order. Before every progress increment, the parent
atomically replaces `isothermal-numerical.npy`, so every displayed completed
point is already checkpointed. Workers never write the shared payload.

The same command resumes an interrupted repair. Saved successes—including new
successes from an earlier repair attempt—are skipped again, while the remaining
non-success cells are retried.

Useful overrides are:

```bash
python3 run_isothermal_numerical.py \
  --workers 64 \
  --trial-timeout 180 \
  --cell-timeout 900
```

Alternative compatible inputs can be selected with `--domain` and
`--analytic-seed`; `--output` changes the numerical checkpoint path.

## Output and failure behavior

`isothermal-numerical.npy` stores the combined old and repaired velocity grid,
maximum composition, baryon current, compact-coordinate and BVP diagnostics,
every seed attempt, structured failure status, and full per-cell record. The
ordinary speed is converted from the proper velocity using
`v = c*u(0-)/sqrt(1 + u(0-)^2)`.

Each BVP attempt has a disposable-process hard timeout of 180 seconds by
default, each cell has a total 900-second budget, and `tail_eps=1e-3` matches
the current public numerical solver default. A failure, exception, trial
timeout, or cell timeout remains a masked terminal record; it is never replaced
by zero or infinity. Failed cells receive one second-pass opportunity using
successful neighboring rays from the same radial shell.

The fixed physics is `B_one_forth=189.1565957288247 MeV`, `ms=0`,
`NM_type="PNM"`, and `upB=5000`. The inherited domain metadata also records
`xi=-0.5`, although the exact numerical BVP does not use the analytical profile
shape parameter.
