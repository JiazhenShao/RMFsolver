# Exact Local Composition In Analytical Velocity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `analytic_velocity_bound` and every active consumer obtain `a`, `A_boundary`, and the `A=1` exclusion boundary from the full quark EOS definition `a = nK_QM/nB_QM`, so `A=1` is exactly the `n_s=0`, `n_d=2*n_u` limit.

**Architecture:** Move the already-proven exact zero-temperature isobar closure out of the dated 26-07-23 runner and into `RMFsolver/phase_velocity.py` as the canonical composition helper. The analytical velocity eigenvalue solve will use that exact EOS result everywhere it currently uses the quadratic `a ~= 9*muK/(2*muB)` mapping; the closed-form transport kernel remains an explicitly named piecewise/quadratic approximation rather than being mislabeled as an exact composition closure. Active cluster runners must consume the exact returned boundary and must not reconstruct a physical boundary from the old leading-order formula.

**Tech Stack:** Python, NumPy, SciPy `root_scalar`/`brentq`, the existing SQM EOS functions, `unittest`, cluster runner smoke tests.

---

## Scope And Non-Goals

- Change physical composition evaluation in the three current solver paths: `analytic_velocity_bound`, `solve_front_energy_conserving_nK`, and `solve_front_energy_conserving_uNmax`.
- Preserve the public `analytic_velocity_bound(...)` signature and the `LTE`/`extreme_endothermic` mode names.
- Preserve legacy return aliases, but populate them from the exact EOS composition.
- Do not rewrite existing `.npy`, pickle, CSV, or rendered notebook artifacts.
- Do not remove the quadratic transport approximation from the manuscript or pretend the remaining closed-form speed kernel is exact. Its rate/isobar integral approximation must be reported separately from the exact composition closure.
- The leading-order `muK ~= 2*muB*a/9` relation may remain only as an explicitly named initial bracket/seed. It must never determine a saved composition, a domain mask, or an `A=1` boundary.

### Task 1: Lock Down The Exact Composition Contract

**Files:**
- Modify: `tests/test_phase_velocity.py`

- [ ] **Step 1: Add a failing real-EOS zero-temperature isobar test**

Construct a downstream endpoint with finite `muB_Q`, `P_Q`, and bag constant, call the new canonical helper, and assert:

```python
closure = pv._solve_exact_zero_temperature_isobar_composition(
    muB_Q=muB_Q,
    P_Q=P_Q,
    B_one_forth=B14,
    ms=0.0,
    upB=5000,
)
assert abs(closure["pressure_residual_norm"]) < 1.0e-9
assert np.isclose(
    closure["A_boundary"],
    closure["nK_A"] / closure["nB_A"],
    rtol=1.0e-12,
)
assert np.isclose(
    1.0 - closure["A_boundary"],
    closure["n_s_A"] / closure["nB_A"],
    rtol=1.0e-10,
    atol=1.0e-12,
)
```

- [ ] **Step 2: Add a failing `A=1` species-identity test**

Create a pressure target from a charge-neutral zero-temperature state at the onset of `n_s=0`, solve the isobar, and require:

```python
assert abs(closure["A_boundary"] - 1.0) < 1.0e-9
assert closure["n_s_A"] / closure["nB_A"] < 1.0e-9
assert abs(closure["n_d_over_2n_u_A"] - 1.0) < 1.0e-9
```

- [ ] **Step 3: Add the regression that exposes the current bug**

Use the representative saved boundary endpoint `muB_Q = 1253.5558402248055 MeV`, `T_Q = 60.53274641481824 MeV`, and the calibrated `B14`. Recompute `P_Q` from the full EOS and assert the exact zero-temperature isobar root is `muK_A ~= 280.91444 MeV` with `A_boundary ~= 0.797141`, while the old nominal quadratic value is approximately one. The production result must equal the full-EOS result and must not equal the quadratic value.

- [ ] **Step 4: Run the tests and confirm they fail for the missing core helper**

Run:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp ./bin/python -m unittest tests.test_phase_velocity -v
```

Expected: the new exact-composition tests fail before production changes.

### Task 2: Add One Canonical Full-EOS Isobar Closure

**Files:**
- Modify: `RMFsolver/phase_velocity.py`

- [ ] **Step 1: Implement the scalar pressure closure**

Add a private helper with this contract:

```python
def _solve_exact_zero_temperature_isobar_composition(
    *,
    muB_Q,
    P_Q,
    B_one_forth,
    ms=0.0,
    upB=5000,
    muK_seed=None,
):
    """Solve P_QM(muB_Q, muK_A, T=0) = P_Q and return exact local a."""
```

Use `muK_seed` only as the first positive bracket probe. Expand the upper bracket geometrically, solve with `root_scalar(..., method="brentq")`, then evaluate the full `_quark_uds_state`, `nB_QM`, and `nK_QM` at the root.

- [ ] **Step 2: Return physical and numerical diagnostics**

Return at least:

```python
{
    "muK_A": muK_A,
    "A_boundary": nK_A / nB_A,
    "nB_A": nB_A,
    "nK_A": nK_A,
    "n_u_A": n_u_A,
    "n_d_A": n_d_A,
    "n_s_A": n_s_A,
    "n_d_over_2n_u_A": n_d_A / (2.0*n_u_A),
    "pressure_residual_norm": (P_at_root - P_Q) / max(abs(P_Q), 1.0),
    "neutrality_residual_norm": (2.0*n_u_A - n_d_A - n_s_A) / nB_A,
    "composition_identity_residual": 1.0 - nK_A/nB_A - n_s_A/nB_A,
}
```

Reject non-finite roots, non-positive `nB_A`, negative composition outside numerical tolerance, pressure residuals above `1e-9`, or failures of neutrality/composition identities.

- [ ] **Step 3: Treat `A=1` as a limit, not a sign-crossing quantity**

Do not expect the exact fraction to exceed one. At zero temperature it saturates at one once `n_s=0`. Domain callers must classify `A_boundary >= 1 - A_TOL` as the excluded endpoint and bracket it between the last safe `A<1` evaluation and the first saturated evaluation.

- [ ] **Step 4: Run the helper tests**

Run the new exact-isobar tests by name and expect `OK`.

### Task 3: Replace The Analytical Composition Closure

**Files:**
- Modify: `RMFsolver/phase_velocity.py:605-774`
- Modify: `RMFsolver/phase_velocity.py:777-1020`
- Modify: `tests/test_phase_velocity.py`

- [ ] **Step 1: Pass EOS inputs into the private formula evaluator**

Change the private signature to accept `B_one_forth`, `ms`, and `upB`, and pass them from every `evaluate_log_u` call. Keep the public function signature unchanged.

- [ ] **Step 2: Delete the quadratic physical `A_boundary` construction**

Remove both production branches:

```python
A_boundary = 1.5*np.sqrt(...)/(muB_Q*muB_Q)
A_boundary = 9.0*np.pi/np.sqrt(2.0)*T_Q/muB_Q
```

Call `_solve_exact_zero_temperature_isobar_composition(...)` instead. Set:

```python
A_boundary = exact_isobar["A_boundary"]
muKstar_max = exact_isobar["muK_A"]
```

The old leading-order `muK` estimate may be supplied only as `muK_seed` for bracket speed.

- [ ] **Step 3: Use exact fractions in both interface modes**

For `extreme_endothermic`, keep `a_interface = A_boundary`, where both now mean exact `nK/nB`. For `LTE`, call `_analytic_aqstar_lte` with the exact `A_boundary`; rename its docstring from “exact LTE-limited” to “closed-form LTE-limited” because the LTE equation remains part of the approximate transport model.

- [ ] **Step 4: Preserve aliases without preserving approximate semantics**

Keep `aQstar_max`, `A_extreme_endothermic`, `aQstar_used`, and `muKstar_max`, but populate every alias from the exact closure. Add the exact species and residual diagnostics to the public result.

- [ ] **Step 5: Report the approximation boundary honestly**

Return:

```python
"composition_definition": "a_local_equals_nK_over_nB",
"composition_closure": "full_quark_eos_zero_temperature_isobar",
"analytic_formula_variant": "piecewise_constant_lambda_n_exact_local_a",
"transport_integral_model": "quadratic_piecewise_constant_closed_form",
```

This explicitly distinguishes exact composition from the still-approximate analytical transport integral.

- [ ] **Step 6: Update formula tests**

Retain tests for the algebraic speed denominator and LTE correction, but source `A_boundary` from the mocked exact closure. Add assertions that no call path can fall back to the leading-order composition formula.

- [ ] **Step 7: Run the analytical solver tests**

Run:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp ./bin/python -m unittest tests.test_phase_velocity -v
```

Expected: all analytical and numerical phase-velocity tests pass.

### Task 4: Remove The Remaining Nonlocal `aQstar` Alias In Current Solvers

**Files:**
- Modify: `RMFsolver/phase_velocity.py:1784-1915`
- Modify: `tests/test_phase_velocity.py`

- [ ] **Step 1: Correct the fixed-temperature interface helper**

Replace:

```python
thermo["aQstar"] = (thermo["nK"] - nK_Q) / nB_Q
```

with:

```python
thermo["aQstar"] = thermo["nK"] / thermo["nB"]
thermo["delta_nK_over_nB_Q"] = (thermo["nK"] - nK_Q) / nB_Q
```

The second quantity may remain as a separately named transport diagnostic; it must never be returned under an `a` name.

- [ ] **Step 2: Confirm the two numerical solvers remain exact-local**

Assert for both `solve_front_energy_conserving_nK` and `solve_front_energy_conserving_uNmax` that all returned/profile fractions satisfy `a == nK/nB`, including `aQstar`, `aQstar_derived`, and `a_end`.

- [ ] **Step 3: Run focused numerical solver tests**

Run the existing nK and uNmax profile/zero-temperature tests and expect `OK`.

### Task 5: Make Cluster Boundary Decisions Consume The Core Exact Closure

**Files:**
- Modify: `new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py`
- Modify: `new_paper_calculations/26-07-24/_analytic_ray_grid.py`
- Modify: `new_paper_calculations/26-07-24-2/_analytic_ray_grid.py`
- Modify: `new_paper_calculations/26-07-25/_analytic_ray_grid.py`
- Modify: `new_paper_calculations/26-07-26/_analytic_ray_grid.py`
- Modify: `new_paper_calculations/26-07-29/_analytic_ray_grid.py`
- Modify: `new_paper_calculations/26-07-19/uNmax_contour.py`
- Modify: `new_paper_calculations/26-07-19/uNmax_contour-log.py`
- Modify: `new_paper_calculations/26-07-20/run_uNmax_logTm_entropy_safe.py`
- Modify: `new_paper_calculations/26-07-25/run_uNmax_ray_grid.py`
- Modify: `new_paper_calculations/26-07-26/run_uNmax_ray_grid.py`
- Modify: `new_paper_calculations/26-07-29/run_uNmax_ray_grid.py`
- Modify: relevant focused tests under `tests/`

- [ ] **Step 1: Make the 26-07-23 helper delegate to the core**

Remove its duplicate EOS root implementation. Keep a thin wrapper only if its dated payload needs its old status-code shape, and populate that wrapper from `phase_velocity._solve_exact_zero_temperature_isobar_composition`.

- [ ] **Step 2: Update analytical ray-grid metadata**

Change the formula variant and composition-closure labels to the exact-local schema. Their boundary logic may continue to call `analytic_velocity_bound`; it will now receive the exact `A_boundary` automatically.

- [ ] **Step 3: Delete numerical-runner physical fallbacks**

Remove every fallback of the form:

```python
A_boundary = 9.0*np.pi/np.sqrt(2.0)*T_Q/muB_Q
```

If a row truly needs `A_boundary` and the solver result lacks it, call the canonical exact helper using `muB_Q`, `P_Q`, `B14`, `ms`, and `upB`. Otherwise store `NaN`; never manufacture a physical boundary from an approximation.

- [ ] **Step 4: Keep approximate seeds visibly nonphysical**

Rename any retained `2*muB*a/9` local variable to `muK_quadratic_seed` or `muK_bracket_seed`, and require an exact pressure/composition postcheck before its result can enter a saved row.

- [ ] **Step 5: Update cluster contract tests**

Assert exact composition metadata, exact `A_boundary` propagation, saturation-based `A=1` handling, absence of physical fallbacks, and preservation of existing adaptive mesh/progress behavior.

### Task 6: Align The Manuscript And Prevent Future Semantic Regression

**Files:**
- Modify: `Latex_writting/Main.tex:523-555`
- Modify: `tests/test_phase_velocity.py`

- [ ] **Step 1: Keep the derivation labeled as leading-order**

Preserve the equations showing `a ~= 9*muK/(2*muB)` and the quarter ellipse as derivational approximations. Add complete-sentence source lines stating that production calculations evaluate endpoint/interface composition with the full EOS ratio `nK/nB`, while the closed-form transport integral retains the quadratic path approximation.

- [ ] **Step 2: Add a repository guard test**

Add a focused source-contract test or explicit audit assertion that leading-order `A` formulas cannot appear in physical-result or boundary branches. Allow them only in names/comments containing `approx`, `quadratic`, `seed`, or manuscript derivations.

- [ ] **Step 3: Do not rewrite historical artifacts**

Leave prior `.npy`, CSV, pickle, and rendered plot files unchanged. Newly generated payloads carry the exact composition-closure metadata so old and new data cannot be silently mixed.

### Task 7: End-To-End Verification

**Files:**
- No additional production files.

- [ ] **Step 1: Run phase-velocity regression tests**

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp ./bin/python -m unittest tests.test_phase_velocity -v
```

- [ ] **Step 2: Run affected cluster tests**

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp ./bin/python -m unittest tests.test_cluster_lte_temperature_jump_26_07_23 tests.test_cluster_analytic_26_07_18 tests.test_cluster_analytic_ray_grid tests.test_cluster_numerical_ray_grid_26_07_25 tests.test_cluster_numerical_ray_grid_26_07_29 -v
```

Use the actual discovered module names if the 26-07-29 runner has its own focused module by implementation time.

- [ ] **Step 3: Compile every modified Python file**

```bash
./bin/python -m py_compile RMFsolver/phase_velocity.py new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py new_paper_calculations/26-07-29/run_uNmax_ray_grid.py
```

- [ ] **Step 4: Run a real-EOS boundary smoke scan**

Evaluate representative low-, middle-, and near-boundary states. Require finite exact pressure residuals, monotonic nondecreasing `A_boundary`, and at the terminal boundary:

```python
assert abs(A_boundary - 1.0) < boundary_tolerance
assert n_s_A / nB_A < species_tolerance
assert abs(n_d_over_2n_u_A - 1.0) < species_tolerance
```

- [ ] **Step 5: Record the numerical change**

For the known old nominal boundary point, report both the previous quadratic value and the new exact value in the verification output. This is an intentional physics correction, not a tolerance-level regression.

- [ ] **Step 6: Time a small analytical contour sample**

Record wall time and EOS-root counts for a small ray/grid sample. The exact scalar isobar solve may cost more than the closed form, but it must remain far cheaper than the numerical BVP. If needed, add a bounded cache keyed by rounded `(muB_Q, P_Q, B14, ms, upB)` without changing results.

## Acceptance Criteria

- `analytic_velocity_bound` never uses `a ~= 9*muK/(2*muB)` to determine a physical composition or boundary.
- Every returned `a`, `a_interface`, and `A_boundary` is an exact full-EOS `nK/nB` value.
- `A_boundary=1` numerically implies `n_s=0` and `n_d=2*n_u`.
- `energy_conserving_nK` and `uNmax` continue to return `a=nK/nB` everywhere.
- Active cluster code contains no leading-order physical `A` fallback.
- Remaining quadratic assumptions are explicitly labeled as transport-integral approximations rather than exact composition closures.
- Existing generated artifacts remain untouched; new outputs are schema-distinguishable.
