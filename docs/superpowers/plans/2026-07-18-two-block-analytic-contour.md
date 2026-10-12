# Two-Block Analytical Contour Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and save two separately contoured adaptive analytical-velocity mesh blocks that exclude the left `Delta h > 0` lobe and the top `A_boundary >= 1` region without deleting valid cells.

**Architecture:** The runner first solves `m_gap(T)` on every low-block temperature row and `T_A(m)` on every high-block metastability column. It then runs row-wise continuation in the low block and column-wise continuation in the high block, storing independent curvilinear arrays and a shared seam at `80 MeV`.

**Tech Stack:** Python, NumPy, SciPy, `concurrent.futures`, tqdm, `unittest`.

---

### Task 1: Boundary and Mesh Contracts

**Files:**
- Modify: `tests/test_cluster_analytic_26_07_18.py`
- Modify: `new_paper_calculations/26-07-18/run_uN_discontinuity_contour.py`

- [ ] Write failing tests asserting the new output filename, `80 MeV` split, gap-root bracket starting at `m = 0`, closed-lobe endpoint classification, low-row coordinates strictly right of `m_gap(T)`, and high-column temperatures strictly below `T_A(m)`.
- [ ] Run `./bin/python -m unittest tests.test_cluster_analytic_26_07_18 -v` and confirm failures identify the missing two-block API.
- [ ] Add focused helpers for low-row gap boundaries, low-row coordinate construction, high-column A-boundary solving, and high-column coordinate construction.
- [ ] Re-run the focused tests and confirm the boundary/coordinate tests pass.

### Task 2: Two-Direction Continuation

**Files:**
- Modify: `tests/test_cluster_analytic_26_07_18.py`
- Modify: `new_paper_calculations/26-07-18/run_uN_discontinuity_contour.py`

- [ ] Write failing tests for low-row solve order from `m_max` leftward, temperature-row order anchored near `10 MeV`, high-column order from the `80 MeV` seam upward, and immediate-neighbor-only seeds.
- [ ] Run the focused tests and confirm the continuation helpers are missing.
- [ ] Implement block-specific work iterators that stream one completed cell at a time to the integrated tqdm bar.
- [ ] Add one-pass immediate-neighbor retries confined to each block.
- [ ] Re-run the focused tests and confirm all continuation tests pass.

### Task 3: Payload and Checkpoint Schema

**Files:**
- Modify: `tests/test_cluster_analytic_26_07_18.py`
- Modify: `new_paper_calculations/26-07-18/run_uN_discontinuity_contour.py`

- [ ] Write a failing payload test requiring independent `low_*_grid` and `high_*_grid` fields, gap and A boundary arrays, block metadata, and absence of `safe_temperature_cut_values`.
- [ ] Replace the single-grid allocation and save path with independent low/high block arrays and atomic periodic checkpoints.
- [ ] Keep the existing extreme-endothermic analytical result schema inside every successful row.
- [ ] Run the focused tests and confirm the payload contract passes.

### Task 4: Verification

**Files:**
- Verify: `new_paper_calculations/26-07-18/run_uN_discontinuity_contour.py`
- Verify: `tests/test_cluster_analytic_26_07_18.py`

- [ ] Run `./bin/python -m unittest tests.test_cluster_analytic_26_07_18 -v`.
- [ ] Run `./bin/python -m py_compile new_paper_calculations/26-07-18/run_uN_discontinuity_contour.py tests/test_cluster_analytic_26_07_18.py`.
- [ ] Run the runner's `--help` command and inspect the new split/buffer controls and output default.
- [ ] Run a tiny real-EOS smoke scan to `/private/tmp`, verify both blocks share the `80 MeV` seam, every low point is right of its gap boundary, every high point is below its A boundary, and the original `.npy` files remain unchanged.
