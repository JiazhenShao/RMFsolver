---
summary: Planned cold fix for solve_front_isothermal_slow — backward shooting from the equilibrium tail gives the flux in ~2 s and matches the BVP to 6 digits; it seeds a layer-clustered collocation mesh, and shooting itself is the fallback where collocation stays singular (0.01 MeV, >=3.5 n0).
status: current
updated: 2026-10-27
tags: [solver, isothermal, numerical, low-temperature, plan, slow-front]
---

# Fix plan for the cold slow isothermal BVP

The failure mechanism is in [[isothermal-slow-cold-failure]]: the tail-scaled coordinate hides a conversion layer $10^3$–$10^7$ times thinner than the tail.
The plan is not implemented. It is backed by prototypes that change only arrays passed to the unchanged solver, not its source.
Scripts: `new_paper_calculations/26-10-27/isothermal_cold_diagnosis/prototype_layer_mesh.py` and `prototype_shooting.py`.

## Prototype evidence

| State | Default solver | Layer mesh + advection-free guess | Backward shooting | Collocation polish from shooting |
|---|---|---|---|---|
| 0.01 MeV, $3.05n_0$ | fails | 3249.93 m/s, 1330 nodes | 3249.93 m/s | 3249.93 m/s, 1465 nodes |
| 0.1 MeV, $3.5n_0$ | fails | fails | 6678.57 m/s | 6678.57 m/s, 1042 nodes |
| 1 MeV, $4.8n_0$ | fails | fails | 10454.1 m/s | 10454.1 m/s, 480 nodes |
| 3 MeV, $4.8n_0$ | fails | 4059.32 m/s | — | — |
| 0.01 MeV, $3.5n_0$ | fails | fails | 46578.8 m/s | singular Jacobian |
| 0.01 MeV, $4.0n_0$ | fails | fails | 133356 m/s | singular Jacobian |
| 0.01 MeV, $4.8n_0$ | fails | fails | 520113 m/s | singular Jacobian |

The brute-force default guess with `n_mesh=3000` gives 3249.94 m/s at the first state, using 201,775 nodes.
The advection-free guess works only up to about 4000 m/s; above roughly 7000 m/s advection is comparable to diffusion inside the layer.
Each shooting solve takes 1.5–2.2 s, and each polish about 1 s.

## Plan, all inside `RMFsolver/phase_velocity.py`

1. **Exact equilibrium slope (isothermal only).** Replace the table's `drate_dr` with $d\Gamma_K/dr$ from two tiny-imbalance isobar states (Richardson). This is needed for the shooting start direction; its effect on the speed is negligible. Adiabatic tabulation stays unchanged, so validated adiabatic results cannot move.
2. **Backward-shooting initializer.**
   - Start at $r=10^{-10}r_{\rm scale}$ on the decaying eigenvector, $J_K-j_Ba(\infty)=(j_B\,da/dr+\lambda_\infty)\,r$.
   - Integrate $dr/dx=j_Ba-J_K$, $dJ_K/dx=-\Gamma_K$ toward the interface (decreasing $x$) with LSODA at `rtol=1e-10`, on the existing path splines, until $r=r_{\rm scale}$.
   - Solve one scalar root in $\ln j_B$ for $J_K(0^+)=j_Ba(0^-)$: scan 0.2–3 times the analytic flux to bracket, then refine with `brentq`.
   - This direction is stable because components off the decaying solution shrink as the integration runs toward the interface. Advection is included exactly.
3. **Layer-clustered mesh and guess.**
   - Mesh: the existing compact mesh, plus 40 log-spaced nodes per decade from $10^{-3}s_{1/2}$ to $\max(30s_{1/2},10^{-2})$, where $s_{1/2}$ is the shooting profile's half-point.
   - Guess: the shooting profile mapped with the solver's own $\lambda_\infty(j_B)$.
   - Continuation guesses keep their own nodes (union with the mesh) instead of being interpolated away.
4. **API.**
   - Add `initial_guess_mode="default" | "shooting" | "auto"` to `solve_front_isothermal_slow`, keeping `"default"` as the default.
   - `"auto"` uses shooting when the predicted $s_{1/2}$ is below 10 first mesh intervals or the default attempt fails.
   - Record the initializer, the shooting mismatch and timings in the result.
5. **Fallback where collocation stays singular (recommended).**
   - Accept the shooting trajectory itself, marked `solution_method="shooting"`, when it passes checks equivalent to the BVP acceptance:
     - direct-EOS closure at sampled nodes at most `1e-8`;
     - constitutive and reaction residuals;
     - mismatch at most `1e-10`;
     - speed changes at most `1e-6` when `rtol`, the start depth $r/r_{\rm scale}$ and `path_rtol` are tightened.
   - It solves the same ODEs on the same EOS tables, so no physics changes.
   - The alternative is a two-scale collocation coordinate. Defer it unless BVP-only provenance is required, because the singular Jacobian's cause is not yet established (likely conditioning at scale ratios of $10^6$ or more).
6. **Leave unchanged:** the equations, boundary conditions, microphysics, acceptance tolerances, analytic APIs, massive EOS and `solve_front_adiabatic_slow`.

## Verification before any contour run

- **Coverage:** all 35 states of the failure map converge, and the 16 current failures are kept as a regression list.
- **Agreement:** on the 19 states the default already solves, `"shooting"`/`"auto"` speeds match the default to at most `1e-6` relative. Where both collocation and shooting finish, they also agree to at most `1e-6`.
- **Independence:** halving `rtol`, `path_rtol`, `tol_bvp` and the start depth, and doubling `n_mesh`, changes no speed by more than `1e-4`.
- **Tests:** add unit tests to `tests/test_phase_velocity_slow.py`; 0.01 MeV, $3.05n_0$ must reproduce $3249.93\pm0.01$ m/s. Rerun the slow, analytic-slow and microphysics suites through the isolated SciPy runtime, plus adiabatic spot checks against saved contour points.
- **Timing:** measure per-solve time and nodes, then size `max_nodes`, timeouts and cluster budgets from the measurements rather than from the earlier 0.5–1.2 s estimate.

The cold layers stay in local equilibrium (mean free path at most $8\times10^{-4}$ of the layer width), so a working solver gives physically usable speeds; see [[isothermal-slow-cold-failure]].
