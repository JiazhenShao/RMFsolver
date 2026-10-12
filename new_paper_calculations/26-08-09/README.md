# T(0+) Three-Method Contour Runs

This directory contains one domain-preparation command and three independent
velocity runners. It intentionally contains no diagnostic or debug launchers.

## Grid

- Boundary curves: 12 angles from 0 through 90 degrees, including the
  `T(0-) = 0.01 MeV` and `nB(0-)/n0 = 1` axis anchors.
- Contour grid: 20 interior angles and 10 radial fractions.
- Interface temperatures: 50, 10, 1, and 0.1 MeV.
- Total cells per method: `20 * 10 * 4 = 800`.

The shared manifest solves the cold `A = 1` curve once and solves the
`Delta h = 0` curve separately for every interface temperature. Every method
therefore evaluates the same physical cells.

## Run Order

Activate the project environment first, then prepare the shared domain:

```bash
python3 prepare_T0plus_contour_domain.py
```

After that completes, the analytical and semi-analytical runners can be
started in parallel in separate terminals:

```bash
python3 run_T0plus_full_analytic_contour.py
python3 run_T0plus_semi_analytic_contour.py
```

The numerical runner can also start immediately; it calculates its own
semi-analytical shooting seeds. To reuse the completed semi-analytical output
and skip that seed calculation, run:

```bash
python3 run_T0plus_full_numerical_contour.py \
  --semi-results T0plus-semi-analytic-contour.npy
```

## Outputs

- `T0plus-contour-domain.npy`
- `T0plus-full-analytic-contour.npy`
- `T0plus-semi-analytic-contour.npy`
- `T0plus-full-numerical-contour.npy`

Each method payload stores flat records and arrays reshaped as
`(theta, radial_fraction, T_0plus)` for direct contour reconstruction.
