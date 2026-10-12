# T(0+)=0.1 MeV Numerical Contour Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a robust, resumable full-numerical uNmax contour runner for fixed `T_0plus=0.1 MeV` that uses numerical propagation rather than semi-analytical velocity seeds and cannot hang indefinitely on one bad `jB` guess.

**Architecture:** Construct the physical domain between the fixed-`T_0plus` zero-speed boundary and the cold `A=1` boundary, then solve it one radial shell at a time. Each target first continues the full successful profile from the immediately smaller radial coordinate at the same theta; failed targets are retried with robust `jB` estimates formed from nearby successful theta rays. Every solver trial runs in a disposable child process with a hard wall-clock limit, every cell has a total time budget, and every completed cell is atomically checkpointed with its profile.

**Tech Stack:** Python 3.12, NumPy, SciPy, `concurrent.futures`, `multiprocessing` with the `spawn` context, tqdm, unittest, existing `RMFsolver.phase_velocity.solve_front_energy_conserving_uNmax`.

---

## File Structure

- Create `new_paper_calculations/26-08-18/prepare_T0plus0p1_domain.py`: visible domain parameters and domain-generation CLI.
- Create `new_paper_calculations/26-08-18/run_T0plus0p1_numerical.py`: visible numerical parameters and the single user-facing numerical command.
- Create `new_paper_calculations/26-08-18/_T0plus0p1_runner.py`: domain loading, shell scheduling, checkpoints, resume behavior, payload assembly, and tqdm integration.
- Create `new_paper_calculations/26-08-18/_radial_shell_continuation.py`: numerical `jB` candidate generation, hard child-process timeout, target budget, profile continuation, and retry decisions.
- Create `new_paper_calculations/26-08-18/_method_engine.py`: local copy of the current analytical/domain support required by the runner.
- Create `new_paper_calculations/26-08-18/_analytic_ray_grid.py`: local copy of the current ray-boundary support required by the runner.
- Create `new_paper_calculations/26-08-18/README.md`: two-command run and resume instructions.
- Create `tests/test_T0plus0p1_numerical_26_08_18.py`: fast mocked tests for the new scheduler, guesses, timeouts, checkpoints, and schema.

No file under `RMFsolver/` will be modified.

### Task 1: Scaffold the Fixed-Temperature Domain

**Files:**
- Create: `new_paper_calculations/26-08-18/prepare_T0plus0p1_domain.py`
- Create: `new_paper_calculations/26-08-18/_T0plus0p1_runner.py`
- Create: `new_paper_calculations/26-08-18/_method_engine.py`
- Create: `new_paper_calculations/26-08-18/_analytic_ray_grid.py`
- Test: `tests/test_T0plus0p1_numerical_26_08_18.py`

- [ ] **Step 1: Add a failing domain-contract test**

Assert that the prepare module exposes these fixed defaults:

```python
assert prepare.DEFAULT_T0PLUS_MEV == 0.1
assert prepare.DEFAULT_CONTOUR_RAY_COUNT == 30
assert prepare.DEFAULT_RADIAL_FRACTIONS[0] == 0.02
assert prepare.DEFAULT_RADIAL_FRACTIONS[-1] == 0.99
assert len(prepare.DEFAULT_RADIAL_FRACTIONS) == 20
```

Also assert that the generated theta axis contains both exact endpoints:

```python
np.testing.assert_allclose(theta_axis[[0, -1]], [0.0, 0.5 * np.pi])
```

- [ ] **Step 2: Run the domain-contract test and verify that it fails**

Run:

```bash
./bin/python -m unittest tests.test_T0plus0p1_numerical_26_08_18.DomainContractTests -v
```

Expected: failure because the `26-08-18` modules do not exist.

- [ ] **Step 3: Implement the fixed-temperature domain defaults**

Use this tunable block in `prepare_T0plus0p1_domain.py`:

```python
# Tunable run parameters
DEFAULT_OUTPUT = "T0plus0p1-domain.npy"
DEFAULT_RUN_TAG = "T0plus0p1_domain_radial_shell_v1"
DEFAULT_T0PLUS_MEV = 0.1
DEFAULT_BOUNDARY_THETA_COUNT = 12
DEFAULT_CONTOUR_RAY_COUNT = 30
DEFAULT_RADIAL_FRACTIONS = (
    0.02, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.66, 0.72, 0.78, 0.84, 0.90, 0.95, 0.99,
)
DEFAULT_DENSITY_MIN_OVER_N0 = 1.0
DEFAULT_DENSITY_MAX_OVER_N0 = 5.5
DEFAULT_TEMPERATURE_MIN_MEV = 0.0
DEFAULT_TEMPERATURE_MAX_MEV = 120.0
DEFAULT_XI = -0.5
```

Generate the lower boundary at `T_0plus=0.1 MeV`, generate the cold `A=1` outer boundary once, and interpolate both onto an endpoint-inclusive theta axis made with `np.linspace(0.0, 0.5*np.pi, 30)`.

- [ ] **Step 4: Run the domain-contract test and verify that it passes**

Run the command from Step 2. Expected: all domain-contract tests pass.

### Task 2: Implement Numerically Derived `jB` Candidates

**Files:**
- Create: `new_paper_calculations/26-08-18/_radial_shell_continuation.py`
- Test: `tests/test_T0plus0p1_numerical_26_08_18.py`

- [ ] **Step 1: Add failing tests for the candidate order**

Test these cases:

```python
assert first_shell_candidates() == (0.8, 0.6, 1.0, 1.2, 1.4)
assert previous_shell_multipliers() == (1.15, 1.25, 1.35, 1.50)
assert same_shell_multipliers() == (1.00, 0.90, 1.10, 1.20)
```

Test that `nearby_theta_median` uses at most the two nearest successful rays on either side, ignores nonfinite values, and returns the median rather than the mean.

- [ ] **Step 2: Run the candidate tests and verify that they fail**

Run:

```bash
./bin/python -m unittest tests.test_T0plus0p1_numerical_26_08_18.CandidateTests -v
```

Expected: failures because the candidate helpers do not exist.

- [ ] **Step 3: Implement the candidate helpers**

The primary candidate must come from the same-theta numerical history:

```python
def extrapolated_same_ray_jB(previous_jB, earlier_jB, previous_rho, earlier_rho, target_rho):
    if earlier_jB is None:
        return previous_jB
    log_slope = (np.log(previous_jB) - np.log(earlier_jB)) / (previous_rho - earlier_rho)
    predicted = previous_jB * np.exp(log_slope * (target_rho - previous_rho))
    return float(np.clip(predicted, 1.05 * previous_jB, 1.55 * previous_jB))
```

Construct the retry order as follows:

1. Same-theta previous full profile plus extrapolated `jB`.
2. Same-theta previous full profile plus previous `jB` multiplied by `1.15`, `1.25`, `1.35`, then `1.50`, stopping when the cell budget expires.
3. Same-theta previous profile plus the median same-shell neighboring-theta `jB`, perturbed by `1.00`, `0.90`, `1.10`, then `1.20`.
4. No profile plus the same-shell neighboring-theta median if profile continuation itself appears pathological.

Deduplicate candidates whose relative difference is below 3 percent. Do not generate or consume any semi-analytical `jB` seed.

- [ ] **Step 4: Run the candidate tests and verify that they pass**

Run the command from Step 2. Expected: all candidate tests pass.

### Task 3: Add Hard Trial and Cell Time Limits

**Files:**
- Modify: `new_paper_calculations/26-08-18/_radial_shell_continuation.py`
- Create: `new_paper_calculations/26-08-18/run_T0plus0p1_numerical.py`
- Test: `tests/test_T0plus0p1_numerical_26_08_18.py`

- [ ] **Step 1: Add failing timeout tests**

Use a mock solver that sleeps indefinitely and assert:

```python
with self.assertRaises(BVPTrialTimeoutError):
    run_module_call_with_hard_timeout(TEST_MODULE, "sleep_forever", {}, 0.2)
```

Add a second test in which the first `jB` sleeps past the trial cap and the second candidate succeeds. Assert that the second candidate is attempted and returned.

Add a third test asserting that no candidate begins after the total cell deadline.

- [ ] **Step 2: Run the timeout tests and verify that they fail**

Run:

```bash
./bin/python -m unittest tests.test_T0plus0p1_numerical_26_08_18.TimeoutTests -v
```

Expected: failures because hard timeout handling is not implemented.

- [ ] **Step 3: Implement killable subprocess trials**

Use the `spawn` multiprocessing context. Run one solver call in a disposable child, wait for at most `min(trial_timeout, cell_deadline-now)`, then call `terminate()`, join briefly, and call `kill()` if necessary.

Expose these defaults at the top of `run_T0plus0p1_numerical.py`:

```python
# Tunable run parameters
DEFAULT_DOMAIN_FILE = "T0plus0p1-domain.npy"
DEFAULT_OUTPUT = "T0plus0p1-uNmax-radial-shell.npy"
DEFAULT_RUN_TAG = "T0plus0p1_uNmax_radial_shell_v1"
DEFAULT_TRIAL_TIMEOUT_S = 300.0
DEFAULT_CELL_TIMEOUT_S = 1200.0
DEFAULT_MAX_BRIDGE_DEPTH = 2
DEFAULT_MIN_BRIDGE_FRACTION_STEP = 1.0e-3
DEFAULT_SAVE_EVERY = 1
DEFAULT_N_MESH = 200
DEFAULT_TOL_BVP = 1.0e-4
DEFAULT_MAX_NODES = 10000
DEFAULT_JB_LOWER_BOUND = 1.0e-12
DEFAULT_JB_UPPER_FACTOR = 1.0e-4
```

The timeout exception must be raised in the parent process, outside `phase_velocity.py`, so solver-level broad exception handlers cannot swallow it.

- [ ] **Step 4: Record timeout diagnostics**

Every attempt record must include:

```python
{
    "jB_guess": float,
    "jB_guess_source": str,
    "trial_timeout_s": float,
    "cell_timeout_s": float,
    "runtime_s": float,
    "status": "success" | "trial_timeout" | "cell_timeout" | "failure",
    "child_terminated": bool,
}
```

- [ ] **Step 5: Run the timeout tests and verify that they pass**

Run the command from Step 2. Expected: all timeout tests pass and no child process remains alive.

### Task 4: Implement Radial-Shell Scheduling and Neighbor Retries

**Files:**
- Modify: `new_paper_calculations/26-08-18/_T0plus0p1_runner.py`
- Modify: `new_paper_calculations/26-08-18/_radial_shell_continuation.py`
- Test: `tests/test_T0plus0p1_numerical_26_08_18.py`

- [ ] **Step 1: Add failing scheduler tests**

Mock the numerical solver and assert this ordering:

```text
all theta cells at rho[0], then retries at rho[0],
all theta cells at rho[1], then retries at rho[1],
...
```

Assert that a target receives the successful full profile from the immediately preceding radial point at the same theta. Assert that a failed immediate predecessor prevents use of an older profile.

- [ ] **Step 2: Run the scheduler tests and verify that they fail**

Run:

```bash
./bin/python -m unittest tests.test_T0plus0p1_numerical_26_08_18.SchedulerTests -v
```

Expected: failures because shell scheduling is not implemented.

- [ ] **Step 3: Implement shell-by-shell parallelism**

For each radial fraction:

1. Submit one primary task per theta ray.
2. Collect the complete shell.
3. Build nearby-theta medians from successful shell results.
4. Retry failed cells with the numerical neighbor candidates.
5. Attempt midpoint bridges only after the candidate list fails and only while the same 1200-second cell deadline has remaining time.
6. Advance to the next radial fraction.

Use all available workers up to the theta count. Keep the integrated tqdm total fixed at 600 physical cells. Update it once when each physical cell reaches a terminal success or failure; show attempts, retries, bridges, trial timeouts, and cell timeouts in the postfix.

- [ ] **Step 4: Run the scheduler tests and verify that they pass**

Run the command from Step 2. Expected: all scheduler tests pass.

### Task 5: Add Per-Cell Checkpoints and Resume

**Files:**
- Modify: `new_paper_calculations/26-08-18/_T0plus0p1_runner.py`
- Test: `tests/test_T0plus0p1_numerical_26_08_18.py`

- [ ] **Step 1: Add failing checkpoint tests**

Run three mocked cells, interrupt the fourth, reload the saved payload, and assert that the three terminal records and successful profiles are present. Resume and assert that those cells are skipped while the fourth is attempted.

- [ ] **Step 2: Run the checkpoint tests and verify that they fail**

Run:

```bash
./bin/python -m unittest tests.test_T0plus0p1_numerical_26_08_18.CheckpointTests -v
```

Expected: failures because per-cell profile checkpointing and resume do not exist.

- [ ] **Step 3: Implement atomic per-cell saves**

After every terminal cell:

1. Update its velocity, `jB`, status, message, diagnostics, and attempt records.
2. Save its full profile when successful.
3. Write to a temporary sibling file and atomically replace the output `.npy`.

Set:

```python
payload["checkpoint_granularity"] = "completed_cell_with_profile"
payload["run_complete"] = False
```

On successful completion set `run_complete=True`. On resume, preserve successful cells, resume the first incomplete radial shell, and use the last contiguous same-theta profile only.

- [ ] **Step 4: Run the checkpoint tests and verify that they pass**

Run the command from Step 2. Expected: all checkpoint tests pass.

### Task 6: Verify the Known Hard Region and Full CLI Contract

**Files:**
- Create: `new_paper_calculations/26-08-18/README.md`
- Test: `tests/test_T0plus0p1_numerical_26_08_18.py`

- [ ] **Step 1: Run the complete mocked test module**

Run:

```bash
./bin/python -m unittest tests.test_T0plus0p1_numerical_26_08_18 -v
```

Expected: all tests pass.

- [ ] **Step 2: Compile and smoke-test both CLIs**

Run:

```bash
./bin/python -m py_compile \
  new_paper_calculations/26-08-18/*.py \
  tests/test_T0plus0p1_numerical_26_08_18.py
./bin/python new_paper_calculations/26-08-18/prepare_T0plus0p1_domain.py --help
./bin/python new_paper_calculations/26-08-18/run_T0plus0p1_numerical.py --help
```

Expected: compilation succeeds and both help commands exit zero.

- [ ] **Step 3: Run a bounded physical smoke test**

Use one theta ray near 65 degrees and radial fractions `0.72,0.78,0.84,0.90,0.95`. This targets the previously failed neighborhood around `T_0minus=100.22 MeV`, `nB_0minus/n0=3.059` without launching the complete contour.

Expected:

- Every target finishes with success or an explicit bounded failure.
- No individual attempt exceeds 300 seconds plus process-shutdown tolerance.
- No cell exceeds 1200 seconds plus process-shutdown tolerance.
- The output `.npy` is readable while the run is in progress.

- [ ] **Step 4: Document the cluster workflow**

Write these commands in `README.md`:

```bash
python3 prepare_T0plus0p1_domain.py
python3 run_T0plus0p1_numerical.py
```

Document that rerunning the second command resumes its existing output, that `--restart` explicitly discards checkpoint state, and that the historical `26-08-09` artifact is never modified.

## Acceptance Criteria

- The physical grid contains 30 theta values by 20 radial values and includes theta 0 and 90 degrees.
- Numerical velocity guesses do not use semi-analytical `jB` values.
- The first shell uses the observed numerical bootstrap ladder.
- Later shells prioritize the immediately smaller same-theta full numerical profile.
- Failed cells use robust nearby-theta numerical medians and bounded multiplier retries.
- Every solver trial is forcibly terminated after 300 seconds.
- Every physical cell reaches a terminal record within 1200 seconds.
- Midpoint bridges share the cell budget and cannot extend it.
- The payload is atomically updated after every cell and includes successful profiles.
- An interrupted run resumes without recomputing successful cells.
- The old `26-08-09` data remains untouched.
