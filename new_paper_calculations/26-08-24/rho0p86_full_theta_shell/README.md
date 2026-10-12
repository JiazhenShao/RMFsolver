# Complete rho=0.86 theta shell

This job solves all 30 angular rays at `rho = 0.86` and `T(0+) = 0.1 MeV`.
Each point starts from the successful full numerical profile at the same theta and `rho = 0.84`.

The run has exactly 30 physical tasks.
The integrated tqdm bar advances once per completed theta point, and the output checkpoint is atomically updated after every point.

## Cluster command

Activate the `.NSM` environment and run:

```bash
cd new_paper_calculations/26-08-24/rho0p86_full_theta_shell
python3 run_rho0p86_full_theta_shell.py
```

The output is:

```text
T0plus0p1-rho0p86-full-theta-shell.npy
```

Useful options:

```bash
python3 run_rho0p86_full_theta_shell.py --dry-run
python3 run_rho0p86_full_theta_shell.py --retry-failed
python3 run_rho0p86_full_theta_shell.py --restart
```

This shell is a bracketing diagnostic for the `5000 m/s` contour.
Rays that remain below `5000 m/s` at `rho = 0.86` will subsequently continue to `rho = 0.875` and, only if necessary, `rho = 0.885`.
