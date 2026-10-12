# Two-Block Analytical Contour Design

## Goal

Replace the incorrect single-valued temperature-cut inversion in `new_paper_calculations/26-07-18/run_uN_discontinuity_contour.py` with two independently contoured adaptive mesh blocks that respect the actual forbidden-domain topology.

## Physical Boundaries

The enthalpy-gap boundary is single-valued in metastability at fixed temperature:

```text
Delta h(T, m_gap(T)) = 0
```

In the current parameter domain it begins at `(m, T) = (0, 0)`, reaches approximately `m = 0.312` near `T = 50 MeV`, and returns to the metastability axis at `(m, T) = (0, 60.1586 MeV)`. The forbidden region is bounded on the left by `m = 0` and on the right by `m_gap(T)`. There is no second orange branch.

The high-temperature boundary is single-valued in temperature at fixed metastability:

```text
A_boundary(T_A(m), m) = 1
```

The known boundary lies above approximately `93.9 MeV`. A split temperature of `80 MeV` is therefore safely above the enthalpy-gap lobe and below the high-temperature boundary.

## Mesh Blocks

The low block uses uniform temperature rows from the configured minimum through `80 MeV`. For each row, solve `m_gap(T)` on `[0, m_max]`. If the complete row has `Delta h <= 0`, use `m_gap = 0`. Otherwise construct the row from `m_gap(T) + epsilon_m` through `m_max` with the configured metastability count.

The high block uses uniform metastability columns from `0` through `m_max`. For each column, solve `T_A(m)` above `80 MeV`, then construct its temperature coordinates from the shared `80 MeV` seam through `T_A(m) - epsilon_T` with the configured high-block temperature count.

The blocks remain separate arrays in the saved payload. Plotters contour them separately on the same axes. They must not be concatenated into one logically rectangular array.

## Continuation

For the low block, solve temperature rows beginning with the row nearest `10 MeV`, then alternate upward and downward. Within a row, start at `m_max` and propagate left through immediately adjacent metastability points. The endpoint seed for a new row may come only from the previously solved adjacent row's `m_max` endpoint.

For the high block, use uniform metastability columns. Start each column at the shared `80 MeV` seam and propagate upward using the immediately preceding same-column endpoint. The seam value is solved once per high-block column.

Retries may use only immediately adjacent successful cells within the same block and may never cross a forbidden boundary.

## Output Contract

Use a new default filename, `uN-two-block-adaptive-contour.npy`, so existing safe-adaptive results are not overwritten.

Save `low_*_grid` and `high_*_grid` arrays for temperature, metastability, velocity, `u_N`, `lambda_n`, `A_boundary`, and interface fractions. Save low-row gap roots, high-column A-boundary temperatures, both boundary-record collections, continuation records, retry records, and explicit block metadata. Retain no misleading single `safe_temperature_cut_values` field in the new schema.

## Validation

Tests must establish that the gap root search includes `m = 0`, the forbidden lobe has one right boundary, the low mesh begins to the right of that boundary, the high mesh ends below `A = 1`, the seam is exactly `80 MeV`, continuation never crosses block boundaries, and the default output name differs from all existing payloads.
