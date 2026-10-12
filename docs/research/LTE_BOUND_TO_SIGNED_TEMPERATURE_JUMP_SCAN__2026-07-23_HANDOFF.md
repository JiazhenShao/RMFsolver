# LTE BOUND TO SIGNED INTERFACE TEMPERATURE JUMP — 2026-07-23 SCAN HANDOFF

**Status:** Requirements captured; LTE beta definition confirmed by the user; implementation plan pending.

**Planned calculation directory:** `new_paper_calculations/26-07-23/`

## 1. Requested observable

At each valid `(T_N, metastability)` point, calculate

```text
delta_T_interface_MeV = T_N - T_Qstar_LTE.
```

The sign must be preserved.
A positive value means the LTE-limited quark-side interface is colder than the incoming nuclear matter.
A negative value means the LTE-limited quark-side interface is hotter.

Use the existing LTE composition bound with coefficient `5`:

```text
beta_LTE = 5*u_N*lambda_n.
```

Important terminology:
The current code does not set `beta_LTE = 5`.
It sets the coefficient multiplying `u_N*lambda_n` to `5`.
Any new code and metadata must spell this out to avoid ambiguity.

**User confirmation on 2026-07-23:**
The intended definition is exactly

```text
beta_LTE = 5*lambda_n*u(0-),
```

as obtained by rewriting

```text
A^2 - a_LTE^2 = 5*lambda_n*u(0-)*a_LTE*(1 - a_LTE).
```

Do not replace `beta_LTE` by the constant `5`.

Use

```text
T_Qstar_LTE
  = T_Q*sqrt(max(0, 1 - (a_interface_LTE/A_boundary)^2)).
```

The required quantities are returned by

```text
phase_velocity.analytic_velocity_bound(
    ...,
    interface_fraction_mode="LTE",
)
```

except that `T_Qstar_LTE` and `delta_T_interface_MeV` must be derived by the scan runner.

## 2. Correct structural template

The requested “linear–linear” two-block runner is:

```text
new_paper_calculations/26-07-19/uNmax_contour.py
```

It has:

- a split at `T_N = 80 MeV`;
- a low-temperature block with common linear temperature rows and an adaptive linear metastability interval;
- a high-temperature block with common linear metastability columns and an adaptive linear temperature interval;
- the `Delta h = 0` lower admissibility boundary;
- the `A_boundary = 1` upper admissibility boundary;
- independent low/high block storage so a folded boundary is not inverted into a false single-valued cutoff;
- atomic checkpoint saves, multiprocessing, continuation ordering, and neighbor retries.

`new_paper_calculations/26-07-20/run_uNmax_logTm_entropy_safe.py` is a logarithmic low-temperature entropy-safe runner, not the requested linear–linear two-block template.
The copied `_hydro_analytic_contour_common.py` files remain useful for EOS calibration and analytic point helpers.

## 3. Boundary interpretation

The two gray boundaries requested from the linear–linear template are:

```text
Delta h = 0
A_boundary = 1
```

They must be calculated before filling the adaptive blocks.
The observable scan must use exactly the same valid-domain construction as the older linear–linear runner unless the user explicitly changes the domain.

## 4. Computational distinction from the older runner

The older runner evaluates a numerical `uNmax` BVP at fixed `TQstar` in every mesh cell.
The new runner should not submit that numerical BVP.
Instead it should call the analytic LTE-bound solver, derive `T_Qstar_LTE`, and save the signed temperature difference as the primary field.

The analytic LTE solver still solves internally for `u_N` because `beta_LTE = 5*u_N*lambda_n`.
Therefore `u_N` remains a diagnostic dependency, even though velocity is no longer the plotted or primary mesh quantity.

## 5. Minimum payload fields

Primary grids:

```text
delta_T_interface_MeV
T_Qstar_LTE_MeV
T_N_grid_MeV
metastability_grid
```

Required diagnostics:

```text
T_Q_MeV
A_boundary_grid
a_interface_LTE_grid
beta_LTE_grid
u_N_grid
lambda_n_grid
nB_crit_T_grid
muB_crit_T_grid
status_grid
```

Boundary payloads:

```text
gap_boundary_records
A_boundary_records
low_block
high_block
```

Metadata must state:

```text
observable = "T_N_minus_T_Qstar_LTE"
observable_sign = "positive_means_quark_interface_colder"
lte_beta_definition = "beta_LTE_equals_5_times_u_N_times_lambda_n"
lte_coefficient = 5.0
interface_fraction_mode = "LTE"
mesh_layout = "two_block_adaptive_linear_linear_split_at_80_MeV"
```

## 6. Verification requirements

- Unit-test the signed observable with hand-constructed values.
- Unit-test the limiting cases `a_interface_LTE = 0` and `a_interface_LTE = A_boundary`.
- Reject or mark points where `A_boundary <= 0`, `a_interface_LTE <= 0`, or the ratio lies outside `[0, 1]` beyond numerical tolerance.
- Confirm that the two blocks share the `80 MeV` edge consistently.
- Confirm that no numerical `solve_front_energy_conserving_uNmax` call occurs in a scan point.
- Smoke-test a tiny serial scan and inspect the saved payload.
- Preserve atomic checkpoint behavior and cluster worker detection.
