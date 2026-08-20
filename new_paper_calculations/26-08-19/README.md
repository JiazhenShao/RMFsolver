# Isothermal contour cluster workflow

This directory builds matched analytical and exact numerical isothermal front-speed contours on the current thermodynamic interface-composition ceiling. At each positive temperature, the domain stage inverts the live solver relation

```text
P(0+) = P(0-)
a(0+) = nK(0+) / nB(0+)
muB(0-) = muB(0+) + a(0+)*muK(0+)
```

for uniformly spaced targets strictly inside `0 < a(0+) < 1`. Both velocity stages omit the optional `a_0plus` argument, so the public solvers independently recover and report the same maximum composition.

This is the static-isobar thermodynamic ceiling implemented by the current solver. It is not a separate finite-flux optimization in which the maximum composition and velocity are solved as two coupled eigenparameters.

## One-command production run

From this directory, the complete production calculation is:

```bash
python3 run_isothermal_all.py
```

No worker flag is needed. The driver uses the scheduler CPU allocation when
available, otherwise every CPU reported by the machine, and runs the stages in
this order:

1. `Stable neutron matter boundary` (31 phase-curve points, including `T=0`)
2. `a(0+)=1 boundary` (31 phase-curve points, including `T=0`)
3. `Domain grid` (600 individual `(i,j)` points)
4. `Analytical contour scan` (600 individual `(i,j)` points)
5. `Numerical contour scan` (600 individual `(i,j)` points)
6. Plotting

Each progress bar advances after one point, not after a row or column. Before
each increment, the parent process stores that result and atomically replaces
the corresponding `.npy` checkpoint. Worker processes never write shared
payloads.

For a short end-to-end check from this directory:

```bash
python3 run_isothermal_all.py --smoke
```

To place results elsewhere and resume completed domain, analytical, and numerical
points:

```bash
python3 run_isothermal_all.py --output-dir /path/to/results --resume

python3 run_isothermal_numerical.py --resume
```

The worker count defaults to the first available scheduler setting among `SLURM_CPUS_PER_TASK`, `PBS_NP`, and `NSLOTS`, then to the local CPU count. Every analytical cell runs in a disposable spawned process with a 300 s hard timeout. Numerical work advances in increasing-composition shells. Temperatures within one shell run concurrently, while each BVP attempt runs in its own disposable process with a 180 s default trial timeout and a 900 s default cell budget.

## Outputs

- `isothermal-domain.npy` stores the phase boundaries, target grid, inverted upstream states, ceiling residuals, and masks. The phase curves include `T=0`; moving-front cells do not.
- `isothermal-analytic.npy` stores the analytical velocity grid and complete per-cell diagnostics.
- `isothermal-numerical.npy` stores the exact physical-`nK`/`jK` BVP grid, candidate history, timeout records, local-diffusion and exact-rate diagnostics, and compact-coordinate metadata.
- `isothermal-contours.png` compares ordinary speeds in metres per second. The conversion from proper velocity is `v = c*u(0-)/sqrt(1 + u(0-)^2)`.

Payloads are NumPy-saved dictionaries with schema version, run tag, axes, physical inputs, Git revision, live API signatures, status arrays, scalar grids, and full diagnostic records. Writes are atomic and pointwise; a stopped run retains every point shown as completed by its progress bar. The domain carries a deterministic fingerprint over every coordinate-defining array. Resume continues partial domain construction, reuses terminal analytical and numerical cells, and rejects a different fingerprint, physical input, axis, analytical timeout, or numerical control set.

Only finite cells with `task_status == "success"` enter a contour. Stable-PNM, coexistence disagreement, analytical validity-gate failures, exact-model mismatch, non-finite numerical residuals, BVP failure, exceptions, trial timeout, cell timeout, and invalid-domain cells remain structured masks. No failure is replaced by zero or infinity. The exact `a(0+) = 1` boundary is plotted as the formal analytical divergence boundary but never passed to either moving solver.

## Separate stages

```bash
../../bin/python3 prepare_isothermal_domain.py
../../bin/python3 run_isothermal_analytic.py
../../bin/python3 run_isothermal_numerical.py
../../bin/python3 plot_isothermal_contours.py
```

The physical defaults are `B_one_forth=189.1565957288247 MeV`, `xi=-0.5`, `ms=0`, `NM_type="PNM"`, and `upB=5000`. The matched workflow currently requires the last three values exactly because the analytical automatic ceiling is PNM/massless and the numerical public API fixes `upB=5000`. The production grid has 30 positive temperatures from 0.01 to 120 MeV and 20 target compositions from 0.01 to 0.99. `--smoke` selects three temperatures and three interior compositions without changing the EOS or solver contracts.
