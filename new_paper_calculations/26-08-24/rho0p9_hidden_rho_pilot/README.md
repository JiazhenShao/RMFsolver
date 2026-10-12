# rho=0.90 hidden-rho pilot

This pilot targets the first five angular rays (`theta_index = 0, 1, 2, 3, 4`) at `T(0+) = 0.1 MeV`.
It starts from each ray's successful `rho = 0.84` full numerical profile and advances through hidden radial points before attempting `rho = 0.90`.

The default hidden ladder is:

```text
0.84 -> 0.86 -> 0.875 -> 0.885 -> 0.8925 -> 0.90
```

For each step, the initial current is predicted by:

```text
jB_new = jB_previous * ((1-rho_previous)/(1-rho_new))**1.26
```

The solver then tries narrow perturbations around that prediction while reusing the immediately previous full profile.
Every completed hidden point updates both the integrated tqdm bar and the atomic `.npy` checkpoint.

## Cluster command

Activate the same `.NSM` environment used for the earlier contour runs, then run:

```bash
cd new_paper_calculations/26-08-24/rho0p9_hidden_rho_pilot
python3 run_hidden_rho_pilot.py
```

The output is `T0plus0p1-rho0p9-hidden-rho-pilot.npy` in this directory.

Use `--dry-run` to inspect the first predicted current for every ray without launching a BVP.
Use `--retry-failed` to retry failed checkpoint entries while preserving successful hidden profiles.
Use `--restart` to discard checkpoint progress and start the pilot again.
