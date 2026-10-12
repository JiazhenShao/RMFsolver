# T(0+) Benchmark Boundary Redesign

**Status (2026-08-08):** Implemented and verified with unit tests, a real-EOS temperature-indexed boundary smoke run, the four-ray 26-08-02 A-boundary regression, and an end-to-end semi-analytical runner smoke test.

> **Implementation note:** Execute this plan without modifying any file under `RMFsolver/`. The 26-08-02 numerical-I2 boundary calculation is the regression reference.

**Goal:** Make the 26-08-07 three-method benchmark use one reproducible cold `A = 1` outer boundary and a separate finite-`T(0+)` enthalpy boundary for each requested interface temperature.

**Architecture:** Reuse the boundary construction that made the 26-08-02 payload: cold extreme-endothermic endpoint states, exact local composition, numerical `I2`, dense radial bracketing, and ten bisection steps. Compute that `A = 1` curve once. Independently solve the fixed-`T(0+)` enthalpy residual for every interface temperature, then build one adaptive ray mesh per interface temperature. All three velocity runners consume the same saved manifest.

**Scope:** Modify only `new_paper_calculations/26-08-07/` and its two test modules. Do not edit `RMFsolver/phase_velocity.py` or any other solver file.

---

## Task 1: Lock down the 26-08-02 `A = 1` behavior

**Files:**
- Modify: `tests/test_T0plus_split_runners.py`
- Modify: `tests/test_T0plus_method_benchmark.py`

- [ ] Add a regression test proving the boundary residual acceptance remains `abs(1 - A) <= 5e-3`; do not loosen it.
- [ ] Assert the boundary defaults match the successful 26-08-02 calculation: 15 radial probes and 10 bisection steps.
- [ ] Add four reference anchors from the reproduced cold numerical-`I2` calculation:

```text
theta = 18 deg: nB(0-)/n0 = 5.165661035, T = 30.077895958 MeV
theta = 36 deg: nB(0-)/n0 = 4.739009542, T = 60.367765449 MeV
theta = 54 deg: nB(0-)/n0 = 3.914422068, T = 89.141285400 MeV
theta = 72 deg: nB(0-)/n0 = 2.601779030, T = 109.550421143 MeV
```

- [ ] Require the reconstructed boundary to agree with these anchors within an explicit numerical tolerance.
- [ ] Add a contract test that changing the velocity-model `xi` does not alter the saved cold `A = 1` geometry.
- [ ] Run these tests first and confirm they fail against the current four-bisection implementation.

## Task 2: Restore the textbook cold `A = 1` solver

**Files:**
- Modify: `new_paper_calculations/26-08-07/_analytic_ray_grid.py`
- Modify: `new_paper_calculations/26-08-07/_T0plus_split_runner.py`
- Modify: `new_paper_calculations/26-08-07/run_T0plus_method_benchmark.py`

- [ ] Introduce one clearly named cluster-local evaluator for the cold 26-08-02 boundary path. It must use:

```text
T(0+) = 0
interface mode = extreme_endothermic
velocity closure = numerical_I2
exact local a = nK/nB
A_boundary = a(0+) on the cold endpoint
```

- [ ] Keep this boundary evaluator independent of the full-analytic and semi-analytic benchmark velocity evaluators. It is boundary geometry, not a fourth velocity method.
- [ ] Restore 26-08-02 bracketing/refinement defaults: 15 radial probes and 10 bisection steps. Use the old endpoint and ray transition ordering rather than the simplified five-probe/four-bisection shortcut.
- [ ] Compute the cold `A = 1` curve once, outside the loop over `T(0+)`.
- [ ] Keep `xi = 0` as explicit metadata for this reference calculation, while documenting that `xi` is not used in defining `A` or locating `A = 1`.
- [ ] Reject a ray only after the restored refinement fails to achieve `abs(1 - A) <= 5e-3`.
- [ ] Save per-ray diagnostic fields: bracket endpoints, probe statuses, bisection count, `A_inside`, `1 - A_inside`, and failure message.

## Task 3: Compute a distinct enthalpy boundary for each `T(0+)`

**Files:**
- Modify: `new_paper_calculations/26-08-07/_analytic_ray_grid.py`
- Modify: `new_paper_calculations/26-08-07/_T0plus_split_runner.py`

- [ ] Add a fixed-interface-temperature enthalpy residual evaluated on the local isobar:

```text
Delta_h(T0plus) = h_Q(P_Q = P_N, muK = 0, T = T0plus) / nB_Q
                  - h_N(T_N, nB_N) / nB_N
```

- [ ] For each requested `T(0+)` in `(0.01, 0.5, 1.0, 3.0) MeV`, solve `Delta_h(T0plus) = 0` along every boundary ray using bracketed one-dimensional refinement.
- [ ] Keep the cold `A = 1` radius as the shared outer radius and the finite-`T(0+)` enthalpy root as that temperature's inner radius.
- [ ] Fail early with a compact per-ray diagnostic when the ordering `rho_inner(T0plus) < rho_A` is absent.
- [ ] Save both physical coordinates and residuals for every finite-temperature inner-boundary point.

## Task 4: Build per-temperature adaptive meshes

**Files:**
- Modify: `new_paper_calculations/26-08-07/_T0plus_split_runner.py`
- Modify: `new_paper_calculations/26-08-07/run_T0plus_method_benchmark.py`

- [ ] Replace the current single shared point list with geometry indexed by `(T0plus_index, theta_index, radial_index)`.
- [ ] For each `T(0+)`, distribute radial points between its own enthalpy boundary and the shared cold `A = 1` boundary, applying the existing small inner and outer safety buffers.
- [ ] Store a flattened task list for cluster execution, but retain the three indices and physical `(T_N, nB(0-))` coordinates in every row.
- [ ] Preserve a common manifest for all three methods so full analytical, semi-analytical, and numerical results are compared at identical physical points for a given `T(0+)`.

## Task 5: Update the three independent runners and payload schema

**Files:**
- Modify: `new_paper_calculations/26-08-07/run_T0plus_full_analytic.py`
- Modify: `new_paper_calculations/26-08-07/run_T0plus_semi_analytic.py`
- Modify: `new_paper_calculations/26-08-07/run_T0plus_full_numerical.py`
- Modify: `new_paper_calculations/26-08-07/_T0plus_split_runner.py`

- [ ] Make each runner load the new manifest and solve only its assigned method.
- [ ] Preserve independent execution so the three scripts can run in three cluster terminals after one successful domain-preparation run.
- [ ] Save the single cold `A = 1` curve once in the manifest and the `Delta_h = 0` curves as an array indexed by `T(0+)`.
- [ ] Record boundary provenance:

```text
A_boundary_source = 26-08-02_cold_extreme_endothermic_numerical_I2
A_boundary_T0plus_MeV = 0.0
A_boundary_radial_probes = 15
A_boundary_bisection_steps = 10
A_boundary_residual_tolerance = 5e-3
enthalpy_boundary_definition = fixed_T0plus_local_isobar
```

- [ ] Keep failure records; do not silently replace failed values with initial guesses.

## Task 6: Verify before launching expensive runs

**Files:**
- Test: `tests/test_T0plus_split_runners.py`
- Test: `tests/test_T0plus_method_benchmark.py`

- [ ] Run unit tests:

```bash
./bin/python -m unittest tests.test_T0plus_split_runners tests.test_T0plus_method_benchmark -v
```

- [ ] Compile the cluster scripts:

```bash
./bin/python -m py_compile new_paper_calculations/26-08-07/*.py
```

- [ ] Run a local four-ray boundary smoke test with the production refinement settings and confirm all four cold `A = 1` rays satisfy the strict residual criterion.
- [ ] Compare the four smoke-test points with the hard-coded 26-08-02 anchors.
- [ ] Verify that the four `T(0+)` enthalpy curves are separately present and not numerically copied from one another.
- [ ] Verify every mesh point lies strictly between its finite-temperature enthalpy boundary and the cold `A = 1` boundary.
- [ ] Only then run `prepare_T0plus_common_domain.py` on the cluster, followed by the three method scripts in parallel.
