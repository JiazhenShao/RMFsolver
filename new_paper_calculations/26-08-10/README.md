# Fixed T(0+)=5 MeV Radial-Continuation Contours

This folder runs all three velocity methods on one shared 600-cell domain.
The contour grid contains 30 theta rays, including exactly 0 and 90 degrees,
and 20 radial targets from the Delta-h=0 boundary toward the cold A=1
boundary.

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

## Run order

Activate the project environment, then run:

```bash
python3 prepare_T0plus5_domain.py
python3 run_T0plus5_full_analytic.py
python3 run_T0plus5_semi_analytic.py
python3 run_T0plus5_numerical.py --semi-results T0plus5-semi-analytic.npy
```

The two analytical scans may run in parallel after domain preparation. The
numerical runner can also calculate its own semi-analytical seeds when
`--semi-results` is omitted.

## Outputs

- `T0plus5-domain.npy`
- `T0plus5-full-analytic.npy`
- `T0plus5-semi-analytic.npy`
- `T0plus5-uNmax-radial-continuation.npy`

Existing files are not overwritten; a `__runNN` suffix is added instead.
