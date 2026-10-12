# Exact Local-Isobar LTE Fraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the approximate analytic composition passed to the exact interface jump with the exact local `nK/nB` fraction on the fixed-`muB_Q`, zero-temperature isobar.

**Architecture:** Keep `RMFsolver/phase_velocity.py`, the analytic velocity closure, and both adaptive gray boundaries unchanged. Add a private exact-isobar solver to the dated cluster runner, recompute `a_interface_LTE` from its local fraction, and export separate isobar and jump diagnostics.

**Tech Stack:** Python, NumPy, SciPy `brentq`, existing SQM EOS functions, `unittest`.

---

### Task 1: Specify the exact-isobar behavior

**Files:**
- Modify: `tests/test_cluster_lte_temperature_jump_26_07_23.py`

- [ ] **Step 1: Write a failing exact-isobar unit test**

Add a test that patches the runner's EOS pressure and density functions, calls `_exact_local_A_on_zero_temperature_isobar`, and asserts that pressure is solved at `T=0` and the result equals `nK/nB`.

- [ ] **Step 2: Write a failing enrichment test**

Create a synthetic analytic row with distinct `A_boundary` and exact isobar fraction. Assert that `add_lte_temperature_fields` passes the LTE fraction recomputed from the exact ceiling—not the analytic `A_boundary`—to `_solve_interface_state_from_local_a_E_and_Pi`.

- [ ] **Step 3: Write a failing isobar-failure test**

Make the pressure residual fail to bracket and assert a dedicated isobar failure status, `NaN` exact ceiling, `NaN` downstream temperature, and a preserved analytic velocity result.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp ./bin/python -m unittest tests.test_cluster_lte_temperature_jump_26_07_23 -v
```

Expected: FAIL because the private exact-isobar helper and fields do not exist.

### Task 2: Implement the exact local isobaric ceiling

**Files:**
- Modify: `new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py`

- [ ] **Step 1: Import exact EOS functions**

Import `PQM`, `nB_QM`, and `nK_QM` from `RMFsolver.SQMsolver`.

- [ ] **Step 2: Add the private scalar solve**

Implement:

```python
def _exact_local_A_on_zero_temperature_isobar(
    *,
    muB_Q,
    P_Q,
    B14,
    ms,
    upB,
    muK_probe,
):
    ...
```

Use `brentq` after geometric positive-bracket expansion. Return `muK_A`, `A_boundary_exact_local`, and a normalized pressure residual. Reject non-finite EOS values, non-positive density, or a fraction outside `(0, 1)`.

- [ ] **Step 3: Verify the focused unit test passes**

Run the exact-isobar test by name and expect `OK`.

### Task 3: Recompute the LTE target and distinguish failures

**Files:**
- Modify: `new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py`
- Modify: `tests/test_cluster_lte_temperature_jump_26_07_23.py`

- [ ] **Step 1: Add isobar status codes**

Define solved, not-attempted, root-failure, and postcheck-failure codes separate from the existing jump status codes.

- [ ] **Step 2: Recompute the point composition**

Inside `add_lte_temperature_fields`, solve the exact isobar first and calculate:

```python
lte_data = phase_velocity._analytic_aqstar_lte(
    A_boundary_exact_local,
    row["u_N"],
    row["lambda_n"],
)
row["a_interface_LTE"] = lte_data["a_interface_LTE"]
row["beta_LTE"] = lte_data["beta_LTE"]
```

Then call the existing exact local-`a` jump wrapper.

- [ ] **Step 3: Store diagnostics**

Store `A_boundary_analytic`, `A_boundary_exact_local`, `muK_A_exact_local_MeV`, the isobar pressure residual, isobar status/message, corrected `a_interface_LTE`, and unchanged analytic velocity fields.

- [ ] **Step 4: Handle failures**

On isobar failure, clear the exact isobar and jump-derived fields, mark the isobar status, leave the jump status as not attempted, and return `NaN` temperatures. Do not fall back to the approximate fraction.

- [ ] **Step 5: Run all focused tests**

Run the complete 26-07-23 test module and expect `OK`.

### Task 4: Update the payload and seam

**Files:**
- Modify: `new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py`
- Modify: `tests/test_cluster_lte_temperature_jump_26_07_23.py`

- [ ] **Step 1: Add failing payload assertions**

Require exact-local ceiling grids, isobar status grids, and explicit exact-isobar metadata.

- [ ] **Step 2: Allocate and fill the arrays**

Add both low- and high-block arrays for the exact ceiling, isobar chemical potential, pressure residual, and isobar status.

- [ ] **Step 3: Copy the new fields across the 80 MeV seam**

Include the new float fields and integer isobar status in the existing seam-copy lists.

- [ ] **Step 4: Version the result**

Change the default run tag to `lte_temperature_jump_two_block_linear_beta5_exact_local_a_v3` and describe both boundary definitions in `_payload`.

- [ ] **Step 5: Run the focused tests**

Run the complete cluster test module and expect `OK`.

### Task 5: Verify with the real EOS

**Files:**
- No production files modified.

- [ ] **Step 1: Run representative real-EOS closures**

Using records from the current v2 payload, recompute the exact local ceiling and run the existing jump helper for representative low- and high-temperature failures. Require finite positive `T(0+)`.

- [ ] **Step 2: Run regression tests**

Run:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp ./bin/python -m unittest tests.test_cluster_lte_temperature_jump_26_07_23 tests.test_26SU_9_notebook -v
```

Expected: all tests pass.

- [ ] **Step 3: Confirm protected-file scope**

Verify that `RMFsolver/phase_velocity.py` is byte-identical to its pre-implementation checksum and that the dated cluster folder contains no plotting code or README.
