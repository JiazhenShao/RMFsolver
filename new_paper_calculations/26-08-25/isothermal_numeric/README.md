# Numerical isothermal contour cluster run

This directory runs only the exact numerical isothermal BVP on the completed
piecewise-angular 26-08-25 domain. It does not rebuild either physical boundary
or rerun the analytical contour.

The numerical stage calls `solve_front_isothermal` without prescribing
`a_0plus`, so the solver selects the thermodynamic maximum local fraction
`a(0+) = nK(0+)/nB(0+)` at every physical mesh point. Its BVP fields remain the
physical `nK` and `jK`, with the exact nonleptonic rate and the local
`D_K[muB(x), T]` evaluated at every node.

## Required inputs

The default paths are:

```text
26-08-25/
├── isothermal_analytic/
│   └── isothermal-analytic.npy
└── isothermal_numeric/
    ├── isothermal-domain.npy
    └── run_isothermal_numerical.py
```

- `isothermal-domain.npy` is the already completed 30-by-20, 600-cell domain
  copied into this directory.
- `../isothermal_analytic/isothermal-analytic.npy` supplies only the per-cell
  analytical baryon-current seeds. The numerical solver independently resolves
  every endpoint state, composition ceiling, and BVP.

The domain and analytical seed must have the same schema, axes, physics,
live-API signature, and domain fingerprint. No domain-preparation command is
part of this workflow.

## Production command

From this directory, run:

```bash
python3 run_isothermal_numerical.py
```

The worker count comes from `SLURM_CPUS_PER_TASK`, `PBS_NP`, or `NSLOTS`, then
falls back to all CPUs visible to Python. Numerical work proceeds through the
20 radial shells in order so each shell can seed the next one; the 30 angular
rays within one shell run concurrently.

The progress bar contains all 600 cells and advances after every completed
`(ray_index, radial_index)` point. Before each increment, the parent atomically
replaces `isothermal-numerical.npy`, so every displayed completed point is
already checkpointed. Workers never write the shared payload.

Resume a stopped calculation with:

```bash
python3 run_isothermal_numerical.py --resume
```

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

`isothermal-numerical.npy` stores the numerical velocity grid, maximum
composition, baryon current, compact-coordinate and BVP diagnostics, every seed
attempt, structured failure status, and full per-cell record. The ordinary
speed is converted from the proper velocity using
`v = c*u(0-)/sqrt(1 + u(0-)^2)`.

Each BVP attempt has a disposable-process hard timeout of 180 seconds by
default, each cell has a total 900-second budget, and `tail_eps=1e-3` matches
the current public numerical solver default. A failure, exception,
trial timeout, or cell timeout remains a masked terminal record; it is never
replaced by zero or infinity. Failed cells receive one second-pass opportunity
using successful neighboring rays from the same radial shell.

The fixed physics is `B_one_forth=189.1565957288247 MeV`, `ms=0`,
`NM_type="PNM"`, and `upB=5000`. The inherited domain metadata also records
`xi=-0.5`, although the exact numerical BVP does not use the analytical profile
shape parameter.
