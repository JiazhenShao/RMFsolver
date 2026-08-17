---
summary: The signed interface-temperature observable compares the upstream temperature with the LTE-limited value at 0+, with a fixed sign convention and payload contract.
status: current
updated: 2026-08-17
tags: [physics, observable, scan]
---

# Signed interface temperature jump

The primary scanned observable of the 26-07-23 campaign. Built on [[lte-composition-bound]], meshed by [[two-block-contour-scans]], visualised in `26SU-9.ipynb`.

## Definition

$$\Delta T_{\rm interface} = T(0^-) - T_{\rm LTE}(0^+)$$

with

$$T_{\rm LTE}(0^+) = T(\infty)\sqrt{\max\!\left[0,\ 1 - \left(\frac{a_{\rm LTE}(0^+)}{A}\right)^2\right]}.$$

## Sign convention — preserve it

- **Positive** → the LTE-limited quark-side interface is **colder** than the incoming nuclear matter.
- **Negative** → the LTE-limited quark-side interface is **hotter**.

Payload metadata carries this literally as `observable_sign = "positive_means_quark_interface_colder"`. The negative range is much smaller than the positive range, which is why the zero contour is emphasised in the figure and why default levels are asymmetric: $-0.04, 0, 20, 40, 60, 80, 90$ MeV.

## What computes it

`analytic_velocity_bound_lte(...)` supplies the downstream temperature, composition boundary, LTE-limited interface composition, LTE coefficient, incoming speed, and density ratio — as `T_inf`, `A_boundary`, `a_0plus_LTE`, `beta_LTE`, `u_0minus_max`, `lambda_n`. The scan runner derives $T_{\rm LTE}(0^+)$ and the signed difference itself — they are **not** in the returned dictionary.

⚠ This was previously written as `analytic_velocity_bound(..., interface_fraction_mode="LTE")`. That call signature no longer exists; the LTE path is a separate function and `interface_fraction_mode` is output-only. See [[phase-velocity-overview]].

Velocity is retained as a **diagnostic**, not the plotted observable. But $u(0^-)$ cannot be skipped: it enters $\beta_{\rm LTE} = 5\,u(0^-)\lambda_n$, so the analytic solver still solves for it internally.

## What must NOT run

No numerical `solve_front_energy_conserving_uNmax` BVP call at a scan point. The older runner evaluated that per mesh cell; this one calls only the analytic LTE-bound solver. A test asserts the absence of that call — see [[unmax-degeneracy]] for why the numerical path is untrustworthy at low $T(0^+)$ anyway.

## Payload contract

Primary: the signed interface-temperature difference, the LTE-limited $T(0^+)$, the upstream-temperature grid, and the metastability grid. Existing payloads retain legacy field names for compatibility; do not reuse those names as physical notation.

Diagnostics: `T_Q_MeV`, `A_boundary_grid`, `A_boundary_exact_local_grid`, `a_interface_LTE_grid`, `beta_LTE_grid`, `u_N_grid`, `lambda_n_grid`, `nB_crit_T_grid`, `muB_crit_T_grid`, `status_grid`, plus separate `lte_isobar_closure_status_grid` and `lte_jump_closure_status_grid` (isobar failure and three-jump failure are distinct causes, both yielding NaN).

Boundaries: `gap_boundary_records`, `A_boundary_records`, `low_block`, `high_block`.

Run tag as of v3: `lte_temperature_jump_two_block_linear_beta5_exact_local_a_v3`.

## Validation requirements

Endpoint limits $a_{\rm LTE} = 0$ and $a_{\rm LTE} = A_{\rm boundary}$; rejection of ratios outside $[0,1]$ beyond tolerance; rejection where $A_{\rm boundary} \le 0$ or $a_{\rm interface,LTE} \le 0$; exact agreement of the two blocks across the 80 MeV seam; the `beta_LTE = 5*lambda_n*u_N` identity held in metadata and in the saved arrays.
