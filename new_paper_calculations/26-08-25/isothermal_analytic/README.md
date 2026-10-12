# Isothermal contour cluster workflow

This directory builds matched analytical and exact numerical isothermal front-speed contours on a boundary-fitted elliptical-polar mesh in the physical $\bigl(n_B(0^-),T(0^-)\bigr)$ plane. The physical window and mesh are exactly those of the previous energy-conserving cluster workflow:

```text
1 <= nB(0-)/n0 <= 5.5
0.01 MeV <= T(0-) <= 120 MeV
```

Internally, `X=(nB(0-)/n0-1)/4.5`, `Y=T(0-)/(120 MeV)`, `rho=sqrt(X^2+Y^2)`, and `theta=atan2(Y,X)`. Twenty-four endpoint-inclusive Chebyshev rays trace the `delta_muB=0` and `a(0+)=1` boundaries, twice the 12-ray density of the `26-08-20` baseline. The production contour keeps 30 angular rays: the exact zero-degree ray, 15 logarithmic positive rays from 0.01 through 20 degrees, and 14 linear rays above 20 through 90 degrees. Together with the same 20 outward radial fractions as `26-08-18` (`0.02, 0.06, 0.10, 0.15, ..., 0.90, 0.95, 0.99`), this gives 600 physical cells directly; the domain stage does not invert an artificial $a(0^+)$ target grid.

Both velocity stages omit the optional `a_0plus` argument, so each public solver independently recovers and reports the maximum local composition $a(0^+)=n_K(0^+)/n_B(0^+)$ at the physical mesh point. This remains the static-isobar thermodynamic ceiling implemented by the current solver, not a separate finite-flux optimization in which composition and velocity are two coupled eigenparameters.

## One-command production run

From this directory, the complete production calculation is:

```bash
python3 run_isothermal_all.py
```

No worker flag is needed. The driver uses the scheduler CPU allocation when
available, otherwise every CPU reported by the machine, and runs the stages in
this order:

1. `Stable neutron matter boundary rays` (24 angular rays)
2. `a(0+)=1 boundary rays` (the same 24 angular rays)
3. `Polar domain mesh` (600 individual `(i,j)` points)
4. `Analytical contour scan` (600 individual `(i,j)` points)
5. `Numerical contour scan` (600 individual `(i,j)` points)
6. Plotting

The launcher uses the currently activated Python environment when it provides the required scientific packages. A repository-local `bin/python3` is only an optional fallback and is not required on the cluster. Every stage and spawned child prepends this checkout to `PYTHONPATH`; payload metadata records the resolved `phase_velocity.py` path, preventing a stale installed `RMFsolver` from silently replacing the live source.
The boundary residuals and ray-root solver live inside this directory's `_isothermal_domain.py`; cluster workers do not require a workspace-level helper module.

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

The worker count defaults to the first available scheduler setting among `SLURM_CPUS_PER_TASK`, `PBS_NP`, and `NSLOTS`, then to the local CPU count. Boundary rays and analytical cells are independent. Every analytical cell runs in a disposable spawned process with a 300 s hard timeout. Numerical work advances through increasing radial-fraction shells; angular rays within one shell run concurrently. Each BVP attempt runs in its own disposable process with a 180 s default trial timeout and a 900 s default cell budget.

## Outputs

- `isothermal-domain.npy` stores the two ray-traced boundaries, coordinate transform, physical $n_B(0^-)$ and $T(0^-)$ grids, and per-cell status records.
- `isothermal-analytic.npy` stores the analytical velocity grid and complete per-cell diagnostics.
- `isothermal-numerical.npy` stores the exact physical-`nK`/`jK` BVP grid, candidate history, timeout records, local-diffusion and exact-rate diagnostics, and compact-coordinate metadata.
- `isothermal-contours.png` compares ordinary speeds in metres per second. The conversion from proper velocity is `v = c*u(0-)/sqrt(1 + u(0-)^2)`.

Payload schema version 4 is intentionally incompatible with both the superseded fixed-temperature/fixed-composition payload and the uniform-angular schema-3 payload. Payloads are NumPy-saved dictionaries with the polar axes, physical inputs, Git revision, live API signatures and source path, status arrays, scalar grids, and full diagnostic records. Writes are atomic and pointwise; a stopped run retains every point shown as completed by its progress bar. The domain carries a deterministic fingerprint over every coordinate-defining array. Resume continues partial boundary or mesh construction, reuses terminal analytical and numerical cells, and rejects a different fingerprint, physical input, axis, analytical timeout, or numerical control set.

Domain construction is fail-fast: every requested ray must contain exactly one root for each boundary, and the outer radius must exceed the coexistence radius. An incomplete boundary aborts the run before either 600-point velocity bar starts. Only finite cells with `task_status == "success"` enter a contour. Stable-PNM disagreement, analytical validity-gate failures, exact-model mismatch, non-finite numerical residuals, BVP failure, exceptions, trial timeout, and cell timeout remain structured masks. No failure is replaced by zero or infinity. The exact `a(0+) = 1` boundary is plotted as the formal analytical divergence boundary but never passed to either moving solver.

## Separate stages

```bash
python3 prepare_isothermal_domain.py
python3 run_isothermal_analytic.py
python3 run_isothermal_numerical.py
python3 plot_isothermal_contours.py
```

The physical defaults are `B_one_forth=189.1565957288247 MeV`, `xi=-0.5`, `ms=0`, `NM_type="PNM"`, and `upB=5000`. The matched workflow currently requires the last three values exactly because the analytical automatic ceiling is PNM/massless and the numerical public API fixes `upB=5000`. `--smoke` selects three angular rays and three radial fractions without changing the EOS or solver contracts.
