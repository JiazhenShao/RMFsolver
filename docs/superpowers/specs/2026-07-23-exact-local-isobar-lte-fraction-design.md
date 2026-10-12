# Exact Local-Isobar LTE Fraction Design

## Status

Approved for implementation on 2026-07-23.

## Goal

Correct the 26-07-23 LTE temperature-jump cluster scan by calculating the composition ceiling as the exact local EOS fraction `nK/nB` on the downstream endpoint's zero-temperature, fixed-`muB_Q`, constant-pressure isobar.

## Physical definition

For each successful analytic velocity point, keep the downstream equilibrium endpoint `muB_Q` and `P_Q` fixed and solve

```text
PQM(muB_Q, muK_A, B14, T=0) = P_Q.
```

The exact local ceiling is

```text
A_boundary_exact_local =
    nK_QM(muB_Q, muK_A, B14, T=0)
    / nB_QM(muB_Q, muK_A, B14, T=0).
```

The LTE interface fraction is then recomputed from

```text
beta_LTE = 5*lambda_n*u_N
A_boundary_exact_local^2 - a_interface_LTE^2
    = beta_LTE*a_interface_LTE*(1 - a_interface_LTE).
```

This corrected `a_interface_LTE` is passed to the existing exact interface helper, which solves baryon, energy, and momentum conservation for `muB(0+)`, `muK(0+)`, `T(0+)`, and the implied `u(0+)`.

## Scope

- Modify only `new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py` and its focused tests.
- Do not modify `RMFsolver/phase_velocity.py`.
- Preserve the existing analytic velocity, gray boundaries, adaptive two-block mesh, and 80 MeV seam so the scan remains comparable with earlier results.
- Preserve the analytic `A_boundary` under its existing payload field because it still defines the analytic velocity closure and gray boundary.
- Add a separate `A_boundary_exact_local` field for the exact EOS fraction used by the LTE jump target.
- Keep plotting and README files out of the dated cluster folder.

## Numerical method

At fixed `muB_Q`, define

```text
f(muK) = PQM(muB_Q, muK, B14, T=0) - P_Q.
```

Start at `muK=0`, use the old analytic estimate only as the first positive bracket probe, expand the positive upper bracket geometrically until `f` changes sign, then solve with `scipy.optimize.brentq`. Evaluate `nK_QM/nB_QM` at the root and require a finite fraction strictly between zero and one.

An exact-isobar failure is distinct from an exact three-jump failure. Both produce `T_Qstar_LTE_MeV = NaN` and `delta_T_interface_MeV = NaN`, with separate status codes and messages.

## Output contract

The payload run tag becomes `lte_temperature_jump_two_block_linear_beta5_exact_local_a_v3`.

Each block stores:

```text
A_boundary_grid
A_boundary_exact_local_grid
a_interface_LTE_grid
beta_LTE_grid
lte_isobar_closure_status_grid
lte_jump_closure_status_grid
T_Qstar_LTE_MeV_grid
delta_T_interface_MeV_grid
```

Metadata explicitly distinguishes the analytic boundary from the exact local isobaric ceiling.

## Validation

Tests must prove:

- the isobar helper solves pressure and returns the EOS ratio `nK/nB`;
- the LTE quadratic uses the exact local ceiling and `beta_LTE = 5*lambda_n*u_N`;
- no global helper is added to or required from `phase_velocity.py`;
- isobar failures and jump failures are distinguished and return `NaN`;
- the new arrays cross the 80 MeV seam correctly;
- representative real-EOS points that previously failed recover finite positive `T(0+)`.
