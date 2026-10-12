# Fixed T(0+)=5 MeV Contours Including T(0-)=0

This folder runs all three velocity methods on one shared 600-cell domain.
The contour grid contains 30 theta rays, including exactly 0 and 90 degrees,
and 20 radial targets from the Delta-h=0 boundary toward the cold A=1
boundary.

The theta=0 ray is evaluated at exactly T(0-)=0 by all three methods. The
nuclear EOS uses its zero-temperature branch directly; this workflow does not
replace the endpoint with a small positive temperature.

The numerical runner owns one complete theta ray per worker. It starts near
the Delta-h boundary with the point's semi-analytical jB seed, then passes the
full successful nK/jK profile to the immediately adjacent outward point. A
failed step is retried with the current semi-analytical jB and, if necessary,
with recursively inserted midpoint bridge solves. A=1 itself is never solved;
the final regular target is at radial fraction 0.99.

Every BVP trial runs in a disposable subprocess with a hard 600-second wall-
clock limit. All trials and bridge solves for one requested contour cell share
a hard 1800-second budget. The partial numerical `.npy` payload is atomically
updated after every completed target, while full profiles are added whenever a
complete theta ray returns.

## Normal run

Activate the project environment, then run one command:

```bash
python3 run_T0plus5_all.py
```

The launcher executes domain preparation, full analytical, semi-analytical,
and numerical stages sequentially. Each stage may use all configured workers,
but only one worker pool exists at a time. The numerical stage automatically
uses the semi-analytical output produced by the preceding stage.

All tunable physics, grid, and BVP values are listed near the top of
`run_T0plus5_all.py`. Command-line overrides are available for the worker
count, output location and numerical timeouts. A rerun receives one shared
`__runNN` suffix across all four outputs, so files from different runs cannot
be mixed accidentally.

The four smaller launchers remain available only for recovery or isolated
diagnostics when a completed stage should not be repeated.

## Patch only the missing zero-temperature ray

The downloaded 26-08-10 cluster payloads were generated before the solvers
accepted exact `T(0-)=0`, so their theta=0 rows contain failed cells. To solve
only those 20 cells locally and merge them into drop-in-compatible copies, run:

```bash
python3 patch_T0minus0_ray.py
```

The source cluster files are never overwritten. Patched outputs are saved next
to them with the suffix `-T0minus0-local-patched.npy`. The analytical cells are
parallelized locally; the numerical cells are walked outward as one ray so each
successful full profile initializes its immediately adjacent point.

## Outputs

- `T0plus5-all-zero-upstream-v1-domain.npy`
- `T0plus5-all-zero-upstream-v1-full-analytic.npy`
- `T0plus5-all-zero-upstream-v1-semi-analytic.npy`
- `T0plus5-all-zero-upstream-v1-uNmax-radial-continuation.npy`

Existing files are not overwritten; a `__runNN` suffix is added instead.
