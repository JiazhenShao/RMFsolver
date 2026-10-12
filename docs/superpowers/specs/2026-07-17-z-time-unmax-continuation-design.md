# z(t) uNmax Continuation Repair Design

## Objective

Make the `26SU-8.ipynb` catch-up calculation run with the physically selected interface temperature `TQSTAR_MEV = 0.0`, while keeping that value directly tunable in the notebook.

The calculation must retain the corrected finite-temperature metastability definition:

`nB_N(z) = nB_crit(T) + density_slope_n0_per_km * n0 * z`

## Confirmed Root Causes

The notebook currently requests eight logarithmically spaced density states and passes only the previous `jB` and `T_Q` to the next hydrodynamic solve.

The fixed-`TQstar` uNmax solver does not accept the previous downstream chemical potential, interface state, or BVP profile.
It rebuilds the endpoint seed as `(1100 MeV, TQ_guess)` at every density.
Near equilibrium the endpoint and BVP equations are ill-conditioned, so a large density step can leave the local nonlinear convergence basin.

Changing only the residual tolerance is not a valid repair because it does not preserve the physical solution branch and merely moves the failure to a later point.

## Chosen Design

### Notebook control

Set the existing notebook constant to:

`TQSTAR_MEV = 0.0`

It remains in the physical-input control cell so the user can change it without editing solver code.
The cache signature already includes `TQstar_MeV`, so changing this value invalidates incompatible cached data.

### Full continuation state

Extend the fixed-`TQstar` uNmax solver with an optional `continuation_guess=None` input.
The state contains the previous successful downstream endpoint, interface state, BVP profile, and baryon current.

At the next density:

1. Seed the downstream endpoint with the previous `(muB_Q, T_Q)` pair.
2. Seed the interface root with the previous `(muB_Qstar, muK_Qstar, T_Qstar)` state.
3. Interpolate and rescale the previous absolute-`nK` and `jK` profiles onto the new BVP mesh.
4. Continue using the previous `jB` as the BVP parameter seed.

This follows the existing `solve_front_energy_conserving_nK` profile-continuation pattern rather than creating a separate numerical method.

### Adaptive z stepping

Treat `sample_count` as the nominal logarithmic grid.
Add `adaptive_continuation=True` and `max_continuation_subdivisions=12` keyword controls to `z_time_evolution`.
For each requested next point, attempt a solve from the previous full continuation state.

If it fails, insert the logarithmic midpoint between the last successful distance and the requested distance, solve that midpoint, and retry the original target from the new state.
Repeat until the target succeeds or a finite subdivision limit is reached.

All successful inserted points participate in the velocity interpolation and travel-time integral.
The returned sample arrays therefore include both nominal and inserted continuation points and remain strictly increasing after reversal.

### Public behavior and diagnostics

Preserve existing arguments and return fields.
The new keyword arguments have defaults that keep existing callers valid.
`z_time_evolution` requests profiles internally for continuation but omits them from the public result unless `return_solver_results=True`.
Record `adaptive_continuation_used`, `continuation_inserted_point_count`, and `nominal_sample_count` in the result.

When continuation is exhausted, raise an error that reports the last successful `z`, failed target `z`, subdivision count, and nested solver message.

## Testing

Follow test-driven development:

1. Add a regression test that verifies `z_time_evolution` propagates a complete previous solver result, not only scalar guesses.
2. Add a regression test using a controlled solver double that fails on a large step and succeeds after logarithmic midpoint insertion.
3. Verify the returned adaptive sample grid is ordered and includes inserted points in the travel-time surrogate.
4. Run the existing phase-velocity and cluster-runner tests.
5. Execute `26SU-8.ipynb` with `TQSTAR_MEV = 0.0` and confirm all four temperatures complete, numerical validations pass, and the plot is regenerated from the corrected cache.

## Non-goals

- Do not relax the hydrodynamic endpoint residual tolerance as the primary repair.
- Do not impose an assumed near-equilibrium velocity power law.
- Do not change the finite-temperature critical-density calculation.
- Do not alter the visual style of the notebook plot.
