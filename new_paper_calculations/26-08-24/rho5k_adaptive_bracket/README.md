# Adaptive numerical bracket for the 5000 m/s contour

This targeted cluster job continues only theta indices 7 through 26 beyond the complete numerical `rho=0.86` shell.
It predicts a separate radial target for every theta, then advances using hidden steps no larger than `0.005` while retaining the immediately previous full numerical profile.

The predicted target aims for `5200 m/s`.
A ray terminates successfully once its numerical velocity reaches `5050 m/s`.
If the predicted target undershoots, the job extends that ray by `0.0025`, never beyond `rho=0.885`.

The current prediction is

```text
rho_target = 1 - (1 - 0.86) * (u_at_0.86 / 5200)**(1 / 1.26)
```

The shooting-current candidates use the previous hidden-rho solution first.
Only the successful immediate angular neighbors at the `rho=0.86` base shell are appended as fallback current seeds; the runner never jumps across a missing adjacent theta.

## Cluster command

Activate the `.NSM` environment and run:

```bash
cd new_paper_calculations/26-08-24/rho5k_adaptive_bracket
python3 run_rho5k_adaptive_bracket.py
```

Inspect the exact planned targets without launching BVP solves:

```bash
python3 run_rho5k_adaptive_bracket.py --dry-run
```

Resume is automatic.
Use `--retry-failed` to retry failed checkpoint entries and `--restart` to discard the existing checkpoint.

The output is:

```text
T0plus0p1-rho5k-adaptive-bracket.npy
```

The integrated tqdm bar advances once for every completed hidden numerical point.
The `.npy` checkpoint is atomically replaced after every such point.
