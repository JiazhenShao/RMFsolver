# LTE Interface Temperature-Jump Contour Design

## Status

Approved for implementation on 2026-07-23.

## Goal

Create an isolated cluster scan in `new_paper_calculations/26-07-23/` that maps the signed interface temperature difference

`delta_T_interface = T_N - T_Qstar_LTE`

over the same two-block, linear-linear parameter domain used by the older 26-07-19 and 26-07-20 scans.

## Physical closure

The LTE interface composition is defined by

`A_boundary^2 - a_LTE^2 = beta_LTE*a_LTE*(1 - a_LTE)`

with

`beta_LTE = 5*lambda_n*u_N`.

Thus the coefficient is not the constant 5. The downstream bulk endpoint temperature `T_Q` and the interface LTE composition imply

`T_Qstar_LTE = T_Q*sqrt(max(0, 1 - (a_LTE/A_boundary)^2))`.

The primary scan field is

`delta_T_interface = T_N - T_Qstar_LTE`.

Positive values mean the nuclear side is hotter than the inferred LTE quark-side interface state. Negative values mean the LTE quark-side interface state is hotter.

The analytic jump solver still determines `u_N`, because `u_N` enters `beta_LTE`. Velocity is retained as a diagnostic and is not the plotted observable.

## Domain and mesh

The scan preserves the older two-block topology:

- Compute the `Delta h = 0` and `A_boundary = 1` gray boundaries first.
- Split the domain at `T_N = 80 MeV`.
- Low-temperature block: common linearly spaced temperature rows, with an adaptive linear metastability mesh to the allowed side of `Delta h = 0`.
- High-temperature block: common linearly spaced metastability columns, with an adaptive linear temperature mesh below `A_boundary = 1`.
- Copy the low-block seam into the first high-block row so both blocks agree exactly at `T_N = 80 MeV`.

Default scan values remain those of the older linear-linear runner: `T_N` from `1e-2` to `100 MeV`, 30 low-temperature rows, 12 high-temperature rows, metastability from 0 to 0.9 with 20 columns, 160 samples for the enthalpy boundary, 8 probes and 10 bisection steps for the composition boundary, and serial or scheduler-provided worker counts.

## Implementation layout

- `run_lte_temperature_jump_contour.py`: cluster runner, boundary construction, two-block continuation, checkpoint payload, and LTE temperature conversion.
- `_hydro_analytic_contour_common.py`: isolated copy of the existing analytic continuation engine so the dated calculation folder remains cluster-portable.
- `plot_lte_temperature_jump_contour.py`: signed diverging contour plot with the two gray boundaries overlaid.
- `README.md`: physical definition, run examples, payload fields, and interpretation.
- `tests/test_cluster_lte_temperature_jump_26_07_23.py`: focused regression tests.

## Output contract

The saved payload identifies `delta_T_interface_MeV` as the primary observable and stores, for both mesh blocks:

- upstream temperature;
- metastability;
- signed temperature difference;
- inferred LTE quark-side interface temperature;
- analytic `u_N`;
- `lambda_n`;
- `beta_LTE`;
- `A_boundary`;
- `a_interface_LTE`;
- status codes and continuation records.

The payload also stores the sampled gray boundaries, closure labels, formula strings, mesh metadata, calibrated bag parameter, and command-line arguments.

## Validation

Tests must establish:

- the LTE temperature conversion and signed-difference convention;
- endpoint limits `a_LTE = 0` and `a_LTE = A_boundary`;
- rejection of unphysical `a_LTE/A_boundary`;
- the exact `beta_LTE = 5*lambda_n*u_N` metadata;
- linear-linear defaults and the 80 MeV seam;
- presence of primary and diagnostic payload fields;
- no invocation of the numerical second-order front/BVP solver;
- successful loading and plotting of a synthetic two-block payload.

