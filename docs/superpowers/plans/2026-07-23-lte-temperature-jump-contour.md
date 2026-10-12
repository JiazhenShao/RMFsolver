# LTE Interface Temperature Jump Contour Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task by task.

**Goal:** Add an isolated 26-07-23 cluster calculation that scans the LTE-implied signed interface temperature difference on the established two-block linear-linear mesh.

**Architecture:** Reuse the proven analytic continuation and boundary algorithms from 26-07-18 in a self-contained dated folder. The analytic jump solver supplies the downstream bulk state, `u_N`, `lambda_n`, `A_boundary`, and `a_LTE`; a small pure conversion layer derives `T_Qstar_LTE` and `delta_T_interface`. Save the temperature field as the primary observable while preserving velocity and LTE quantities as diagnostics.

**Tech Stack:** Python, NumPy, SciPy, Matplotlib, `unittest`, existing `RMFsolver.phase_velocity` analytic solver.

---

### Task 1: Specify the pure LTE temperature conversion

**Files:**
- Create: `tests/test_cluster_lte_temperature_jump_26_07_23.py`
- Create: `new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py`

**Step 1: Write the failing test**

Add tests that load the runner by path and require:

- `T_Qstar_LTE = T_Q*sqrt(1 - (a_LTE/A_boundary)^2)`;
- `delta_T_interface = T_N - T_Qstar_LTE`;
- the two endpoint limits;
- a `ValueError` for an unphysical ratio.

**Step 2: Run test to verify it fails**

Run: `./bin/python -m unittest tests.test_cluster_lte_temperature_jump_26_07_23`

Expected: failure because the new runner does not yet exist.

**Step 3: Write minimal implementation**

Create the dated directory and add pure helpers with finite-value checks and a small roundoff tolerance.

**Step 4: Run test to verify it passes**

Run the same focused unittest command.

### Task 2: Build the two-block cluster runner

**Files:**
- Create: `new_paper_calculations/26-07-23/_hydro_analytic_contour_common.py`
- Modify: `new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py`
- Modify: `tests/test_cluster_lte_temperature_jump_26_07_23.py`

**Step 1: Write failing structural tests**

Require:

- `adaptive_linear` as the default temperature mesh;
- `LTE` as the only interface-fraction mode;
- the exact 80 MeV split;
- low-row and high-column mesh constructors;
- exact seam copying;
- calls to `analytic_velocity_bound`;
- no reference or call to `solve_front_energy_conserving_uNmax`.

**Step 2: Run tests to verify failure**

Run the focused unittest module.

**Step 3: Implement the runner**

Port the proven two-block analytic runner and common continuation engine into the dated folder. Change the observable and closure defaults, preserve the gray-boundary-first workflow, and derive LTE temperature fields for every successful point.

**Step 4: Run tests to verify pass**

Run the focused unittest module and the older analytic-runner regression suite.

### Task 3: Define and test the checkpoint payload

**Files:**
- Modify: `new_paper_calculations/26-07-23/run_lte_temperature_jump_contour.py`
- Modify: `tests/test_cluster_lte_temperature_jump_26_07_23.py`

**Step 1: Write failing payload tests**

Require primary-observable metadata, exact beta formula metadata, both block fields, gray boundaries, status codes, and diagnostic velocity fields.

**Step 2: Run tests to verify failure**

Run the focused unittest module.

**Step 3: Implement payload fields**

Add `low_` and `high_` arrays for `delta_T_interface_MeV`, `T_Qstar_LTE_MeV`, and `beta_LTE`, plus explicit formula and sign-convention strings.

**Step 4: Run tests to verify pass**

Run the focused unittest module.

### Task 4: Add the signed contour plot and usage notes

**Files:**
- Create: `new_paper_calculations/26-07-23/plot_lte_temperature_jump_contour.py`
- Create: `new_paper_calculations/26-07-23/README.md`
- Modify: `tests/test_cluster_lte_temperature_jump_26_07_23.py`

**Step 1: Write a failing plotting smoke test**

Build a synthetic two-block payload, run the plotting entry point with a noninteractive backend, and require a nonempty image.

**Step 2: Run test to verify failure**

Run the focused unittest module.

**Step 3: Implement plotting and documentation**

Use a diverging normalization centered at zero, render both mesh blocks, draw `Delta h = 0` and `A_boundary = 1` in gray, and document cluster invocation and interpretation.

**Step 4: Run test to verify pass**

Run the focused unittest module.

### Task 5: Verify the integrated calculation

**Files:**
- Verify: `new_paper_calculations/26-07-23/*.py`
- Verify: `tests/test_cluster_lte_temperature_jump_26_07_23.py`

**Step 1: Compile**

Run: `./bin/python -m py_compile new_paper_calculations/26-07-23/*.py tests/test_cluster_lte_temperature_jump_26_07_23.py`

**Step 2: Run focused and regression tests**

Run: `MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp ./bin/python -m unittest tests.test_cluster_lte_temperature_jump_26_07_23 tests.test_cluster_analytic_26_07_18`

Run: `./bin/python tests/test_phase_velocity.py`

**Step 3: Run command-line smoke checks**

Run the runner and plotter with `--help`, then execute a tiny serial scan if the physical solver runtime permits.

**Step 4: Inspect saved metadata**

Load the smoke payload and confirm the sign convention, beta formula, two-block array shapes, finite-point consistency, and exact seam values.

