# LTE Local-Fraction Interface Temperature Jump Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the analytic ellipse estimate of `T(0+)` in the 26-07-23 cluster scan with an exact three-jump-condition solve at the LTE local fraction `a(0+) = nK(0+)/nB(0+)`, returning a marked `NaN` temperature when no physical interface root exists.

**Architecture:** Keep `RMFsolver/phase_velocity.py` unchanged. The cluster runner will directly call the existing private helper `phase_velocity._solve_interface_state_from_local_a_E_and_Pi`, which is importable despite its leading underscore. `analytic_velocity_bound` will still determine the trial velocity, downstream endpoint, LTE fraction, and closure of `u(0-)`; the newer helper will then independently reconstruct `muB(0+)`, `muK(0+)`, and `T(0+)` from the conserved fluxes and the local-fraction condition.

**Tech Stack:** Python, NumPy, SciPy roots inside `RMFsolver.phase_velocity`, `unittest`, existing two-block adaptive cluster runner.

---

## File structure and scope

- Modify `new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py`
  - Call the newer local-fraction interface helper.
  - Store exact closure diagnostics and `NaN` on failure.
  - Preserve the existing two gray bounds, 80 MeV split, adaptive linear-linear grids, and velocity closure.
- Modify `tests/test_cluster_lte_temperature_jump_26_07_23.py`
  - Replace ellipse-temperature tests with local-jump-helper tests.
  - Verify exact helper inputs, local composition, failure marking, payload metadata, and seam copying.
- Delete `new_paper_calculations/26-07-23/plot_lte_temperature_jump_contour.py`
  - Plotting stays in notebook `26SU-9`, outside the cluster folder.
- Delete `new_paper_calculations/26-07-23/README.md`
  - Keep the cluster folder limited to the runner, its required execution helper, and result payloads.
- Do not modify `RMFsolver/phase_velocity.py`.
- Do not call the obsolete `_solve_local_quark_state_from_a_E_and_Pi`.

### Required physical closure

For every analytically successful mesh point, retain

```text
jB = nB_N*u_N
Pi = h_N*u_N^2 + P_N
E = h_N*u_N*sqrt(1 + u_N^2)
a_target = a_interface_LTE
```

and solve for `muB_Qstar`, `muK_Qstar`, and `T_Qstar` using

```text
Pi_Qstar = Pi
E_Qstar = E
nK_Qstar/nB_Qstar = a_target
```

The primary observable remains

```text
delta_T_interface_MeV = T_N_MeV - T_Qstar_LTE_MeV
```

where `T_Qstar_LTE_MeV` now comes from the exact jump solve, not from `T_Q*sqrt(1 - (a/A)^2)`.

### Failure policy

- Preserve the analytic velocity result when it succeeds.
- If the local jump helper raises, returns non-finite values, or fails post-validation:
  - set `T_Qstar_LTE_MeV = NaN`;
  - set `delta_T_interface_MeV = NaN`;
  - keep `beta_LTE = 5*lambda_n*u_N`;
  - set a dedicated nonzero local-jump status code;
  - store the exception or validation message in the point record.
- Do not silently replace a failed exact solve with the old ellipse temperature.

---

### Task 1: Replace the old temperature formula tests with exact local-jump tests

**Files:**
- Modify: `tests/test_cluster_lte_temperature_jump_26_07_23.py`

- [x] **Step 1: Remove the three tests that make `lte_interface_temperature` part of the required public behavior**

Remove:

```text
test_lte_interface_temperature_uses_composition_ratio
test_lte_interface_temperature_has_correct_endpoint_limits
test_lte_interface_temperature_rejects_unphysical_ratio
```

- [x] **Step 2: Add a test proving that the runner calls the newer private helper**

Monkeypatch:

```python
phase_velocity._solve_interface_state_from_local_a_E_and_Pi
```

Use a successful synthetic row containing:

```python
row = {
    "success": True,
    "u_N": 0.02,
    "jB": 3.0,
    "P_N_MeV4": 40.0,
    "h_N_MeV4": 1000.0,
    "muB_Q_MeV": 1100.0,
    "T_Q_MeV": 20.0,
    "a_interface_LTE": 0.4,
    "lambda_n": 0.75,
    "beta_LTE": 0.075,
}
```

Have the fake helper return a state with `nK/nB = 0.4` and `T = 12.0`. Assert that it received:

```text
a_target = 0.4
E = 1000.0*0.02*sqrt(1 + 0.02^2)
Pi = 1000.0*0.02^2 + 40.0
jB = 3.0
B_one_forth = supplied calibrated B14
```

Assert that the enriched row contains:

```text
T_Qstar_LTE_MeV = 12.0
delta_T_interface_MeV = T_N_MeV - 12.0
lte_jump_closure_success = True
lte_jump_closure_status = 0
```

- [x] **Step 3: Add a test proving the obsolete helper is never called**

Monkeypatch `_solve_local_quark_state_from_a_E_and_Pi` to raise `AssertionError`, then run the new enrichment function with a successful fake response from `_solve_interface_state_from_local_a_E_and_Pi`.

- [x] **Step 4: Add a test for root failure**

Make the newer helper raise `RuntimeError("no physical local-a interface root")`. Assert:

```text
T_Qstar_LTE_MeV is NaN
delta_T_interface_MeV is NaN
lte_jump_closure_success is False
lte_jump_closure_status is nonzero
lte_jump_closure_message contains "no physical local-a interface root"
beta_LTE = 5*lambda_n*u_N
```

- [x] **Step 5: Add a post-validation test**

Return a nominal helper result whose `nK/nB` differs from `a_interface_LTE`. Assert that the point is marked invalid and the temperature is `NaN`, rather than accepting a numerically inconsistent root.

- [x] **Step 6: Run the focused tests and verify that they fail before implementation**

Run:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp ./bin/python -m unittest tests.test_cluster_lte_temperature_jump_26_07_23 -v
```

Expected: the new exact-closure tests fail because the runner still uses the ellipse conversion and has no local-jump diagnostics.

---

### Task 2: Add the cluster-local exact interface closure

**Files:**
- Modify: `new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py`

- [x] **Step 1: Define dedicated local-jump status codes**

Add integer codes near the existing status constants:

```python
LTE_JUMP_SOLVED = 0
LTE_JUMP_NOT_ATTEMPTED = 1
LTE_JUMP_ROOT_FAILURE = 2
LTE_JUMP_POSTCHECK_FAILURE = 3
```

- [x] **Step 2: Rename the ellipse expression so it can only be used as an initial guess**

Replace `lte_interface_temperature` with:

```python
def _ellipse_temperature_seed(*, T_Q_MeV, A_boundary, a_interface_LTE):
    ...
```

Its return value may seed the nonlinear root, but it must never be written directly to `T_Qstar_LTE_MeV`.

- [x] **Step 3: Implement the exact cluster-local wrapper**

Add:

```python
def solve_lte_interface_state(
    row,
    *,
    B14,
    ms,
    upB,
):
    ...
```

The wrapper must:

1. Read `u_N`, `jB`, `P_N_MeV4`, `h_N_MeV4`, and `a_interface_LTE`.
2. Compute `E` and `Pi` from the nuclear side.
3. Build an initial guess using the downstream endpoint and the ellipse temperature only as a seed.
4. Call exactly:

```python
phase_velocity._solve_interface_state_from_local_a_E_and_Pi(
    a_target,
    E,
    Pi,
    jB,
    B14,
    ms=ms,
    upB=upB,
    initial_guess=(muB_seed, muK_seed, T_seed),
)
```

5. Validate finite positive `muB`, finite `muK`, finite positive `T`, finite positive `nB`, and finite `nK`.
6. Validate:

```text
abs(nK/nB - a_target) <= 1e-7
abs(E_Qstar - E)/max(abs(E), 1) <= 1e-7
abs(Pi_Qstar - Pi)/max(abs(Pi), 1) <= 1e-7
```

7. Return the solved state and residual diagnostics.

- [x] **Step 4: Replace `add_lte_temperature_fields`**

Change its signature to include the closure inputs:

```python
def add_lte_temperature_fields(
    row,
    *,
    T_N_MeV,
    B14,
    ms,
    upB,
):
    ...
```

On success, store:

```text
T_Qstar_LTE_MeV
delta_T_interface_MeV
muB_Qstar_LTE_MeV
muK_Qstar_LTE_MeV
nB_Qstar_LTE_MeV3
nK_Qstar_LTE_MeV3
u_Qstar_LTE
a_Qstar_LTE_derived
lte_jump_a_residual
lte_jump_energy_residual_norm
lte_jump_momentum_residual_norm
lte_jump_closure_success
lte_jump_closure_status
lte_jump_closure_message
```

On failure, set all solved interface quantities and both temperature observables to `NaN`, while preserving `beta_LTE`.

- [x] **Step 5: Pass the calibrated EOS and solver parameters at each accepted point**

Replace the current call with:

```python
add_lte_temperature_fields(
    row,
    T_N_MeV=float(result["T_MeV"]),
    B14=B14,
    ms=args.ms,
    upB=args.upB,
)
```

- [x] **Step 6: Run the focused tests**

Run the command from Task 1.

Expected: the exact helper-call, obsolete-helper guard, success, and failure tests pass.

---

### Task 3: Export closure diagnostics across both adaptive blocks

**Files:**
- Modify: `new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py`
- Modify: `tests/test_cluster_lte_temperature_jump_26_07_23.py`

- [x] **Step 1: Add failing payload and seam-copy tests**

Require low- and high-block arrays for:

```text
lte_jump_closure_status_grid
lte_jump_a_residual_grid
lte_jump_energy_residual_norm_grid
lte_jump_momentum_residual_norm_grid
muB_Qstar_LTE_MeV_grid
muK_Qstar_LTE_MeV_grid
```

Verify that the shared 80 MeV seam copies these fields exactly.

- [x] **Step 2: Allocate the diagnostic arrays**

Add floating diagnostic fields to `field_names`. Allocate the closure-status grids separately with integer dtype and default `LTE_JUMP_NOT_ATTEMPTED`.

- [x] **Step 3: Fill diagnostics in `accept`**

Write exact-closure fields whenever the analytic velocity succeeds. If the jump closure fails, retain `NaN` in the temperature and residual grids and write the nonzero closure status.

- [x] **Step 4: Preserve diagnostics at the 80 MeV seam**

Extend `copy_low_seam_to_high` so the integer closure-status grid and all new floating fields are copied with the existing point fields.

- [x] **Step 5: Update payload metadata**

Replace the old definition with:

```text
T_Qstar_LTE_definition = exact local-a interface solve of baryon, energy, and momentum jumps
interface_state_helper = phase_velocity._solve_interface_state_from_local_a_E_and_Pi
composition_definition = a_local_equals_nK_over_nB
beta_LTE_definition = beta_LTE = 5*lambda_n*u_N
jump_failure_policy = mark closure status and return NaN temperature
```

Remove metadata that claims the ellipse formula is the reported temperature.

- [x] **Step 6: Run the focused cluster tests**

Expected: all cluster-runner tests pass.

---

### Task 4: Keep the cluster directory execution-only

**Files:**
- Delete: `new_paper_calculations/26-07-23/plot_lte_temperature_jump_contour.py`
- Delete: `new_paper_calculations/26-07-23/README.md`
- Modify: `tests/test_cluster_lte_temperature_jump_26_07_23.py`

- [x] **Step 1: Remove the standalone plotter test**

Delete `test_plotter_renders_synthetic_two_block_payload`; visualization remains the responsibility of notebook `26SU-9`.

- [x] **Step 2: Delete the plotting script and README**

Keep:

```text
_hydro_analytic_contour_common.py
run_lte_temperature_jump_contour.py
cluster result payloads
```

- [x] **Step 3: Verify no cluster runner imports plotting code**

Run:

```bash
rg -n "matplotlib|plot_lte|make_figure" new_paper_calculations/26-07-23
```

Expected: no matches in Python source.

---

### Task 5: End-to-end verification

**Files:**
- Verify: `new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py`
- Verify: `tests/test_cluster_lte_temperature_jump_26_07_23.py`

- [x] **Step 1: Run the focused unit tests**

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp ./bin/python -m unittest tests.test_cluster_lte_temperature_jump_26_07_23 -v
```

Expected: all tests pass.

- [x] **Step 2: Run a tiny serial smoke scan**

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp ./bin/python new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py --workers 1 --t-count 3 --high-t-count 3 --m-count 3 --boundary-t-count 8 --a-boundary-probe-count 4 --a-boundary-bisect-steps 2 --save-every 0 --output /private/tmp/lte-local-jump-smoke.npy
```

Expected:

- the runner finishes;
- every finite `T_Qstar_LTE_MeV` has closure status `LTE_JUMP_SOLVED`;
- failed interface roots have `NaN` temperature and a nonzero closure status;
- `beta_LTE` equals `5*lambda_n*u_N`;
- the low and high blocks share identical values at the 80 MeV seam.

- [x] **Step 3: Inspect the smoke payload numerically**

Use `./bin/python` to load `/private/tmp/lte-local-jump-smoke.npy` and assert:

```text
finite(T_Qstar_LTE) implies finite(delta_T_interface)
finite(T_Qstar_LTE) implies abs(a_Qstar_LTE_derived - a_interface_LTE) <= 1e-7
nonzero closure status implies T_Qstar_LTE is NaN
low seam arrays equal high seam arrays
```

- [x] **Step 4: Confirm `RMFsolver/phase_velocity.py` was not modified**

Inspect the working changes and verify the RMF solver file is absent from the modification list.

- [x] **Step 5: Record the exact cluster rerun command**

After the smoke scan passes, report the production command using the existing default mesh and requested worker count. Do not regenerate plots inside `26-07-23`; the downloaded production payload will be visualized in `26SU-9`.

---

## Self-review

- The plan preserves the two gray bounds, adaptive linear-linear construction, and 80 MeV split.
- The analytic velocity bound remains responsible for closing the trial `u(0-)`.
- The reported `T(0+)` comes only from the three exact conservation equations plus local `a = nK/nB`.
- The obsolete reference-density helper is explicitly forbidden.
- Root failures are marked and produce `NaN`, as requested.
- No new global helper is added to `RMFsolver/phase_velocity.py`.
- Plotting code and README are removed from the cluster folder.
