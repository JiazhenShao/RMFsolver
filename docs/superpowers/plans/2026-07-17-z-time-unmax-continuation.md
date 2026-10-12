# z(t) uNmax Continuation Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `26SU-8.ipynb` complete its finite-temperature-critical-density catch-up curves with tunable `TQSTAR_MEV = 0.0`.

**Architecture:** Propagate the complete previous uNmax solution into the next fixed-`TQstar` solve and adaptively bisect failed logarithmic distance steps. Reuse the existing absolute-`nK` profile interpolation pattern and preserve current public defaults.

**Tech Stack:** Python, NumPy, SciPy `solve_bvp`, pytest, Jupyter nbconvert.

---

### Task 1: Specify adaptive continuation behavior

**Files:**
- Modify: `tests/test_phase_velocity.py`

- [ ] Add a failing test whose synthetic solver requires `continuation_guess` after the first point.
- [ ] Add a failing test whose synthetic solver rejects a large step and succeeds after logarithmic midpoint insertion.
- [ ] Run the two tests and confirm they fail because `z_time_evolution` neither propagates profiles nor inserts points.

### Task 2: Implement complete uNmax continuation state

**Files:**
- Modify: `RMFsolver/phase_velocity.py`
- Test: `tests/test_phase_velocity.py`

- [ ] Add `continuation_guess=None` to the private and public fixed-`TQstar` uNmax solvers.
- [ ] Seed the endpoint and interface roots from the previous result.
- [ ] Interpolate and rescale previous `nK` and `jK` profiles onto the new BVP mesh.
- [ ] Run the continuation-state test and existing uNmax tests.

### Task 3: Implement adaptive logarithmic distance stepping

**Files:**
- Modify: `RMFsolver/phase_velocity.py`
- Test: `tests/test_phase_velocity.py`

- [ ] Add `adaptive_continuation=True` and `max_continuation_subdivisions=12` to `z_time_evolution`.
- [ ] On failure, solve the logarithmic midpoint from the last successful full result and retry the target.
- [ ] Include inserted points in the velocity surrogate and add continuation diagnostics to the result.
- [ ] Run all `z_time_evolution` tests.

### Task 4: Update and execute the notebook

**Files:**
- Modify: `26SU-8.ipynb`

- [ ] Set the tunable `TQSTAR_MEV` default to `0.0` and bump the cache signature.
- [ ] Execute the notebook from a clean cache-compatible state.
- [ ] Confirm all four curves, numerical checks, and plot cells complete.

### Task 5: Full verification

**Files:**
- Verify: `RMFsolver/phase_velocity.py`
- Verify: `tests/test_phase_velocity.py`
- Verify: `tests/test_cluster_uNmax_runners.py`
- Verify: `26SU-8.ipynb`

- [ ] Compile modified Python files.
- [ ] Run phase-velocity and cluster-runner regression tests.
- [ ] Inspect notebook outputs and generated plot for successful corrected execution.
