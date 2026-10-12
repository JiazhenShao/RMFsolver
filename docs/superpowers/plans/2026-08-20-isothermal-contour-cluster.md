# Isothermal Analytical and Numerical Contour Cluster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable cluster workflow in `new_paper_calculations/26-08-19/` that evaluates the analytical and exact numerical isothermal velocities on the thermodynamic-maximum interface-composition surface between the stable-PNM and strangeness-free boundaries.

**Architecture:** Use the live `_solve_a_0plus_max` contract as the single source of composition physics. Construct a boundary-fitted grid by inverting `a_0plus_max(T, nB_0minus)` for uniformly spaced target fractions, then call both public solvers with `a_0plus` omitted so each solver independently resolves and reports the same maximum. Execute analytical points independently and numerical points in increasing-composition shells with hard subprocess timeouts, scalar-`jB` continuation, atomic checkpoints, and explicit structured failure masks.

**Tech Stack:** Python 3.12, NumPy, SciPy, Matplotlib, `concurrent.futures`, `multiprocessing` with the `spawn` context, unittest, and the live `RMFsolver.phase_velocity` isothermal APIs.

---

## Updated physics and API contract

The implementation must use the current thermodynamic ceiling rather than the earlier unweighted chemical-potential condition:

```text
P(0+) = P(0-)
a(0+) = nK(0+) / nB(0+)
muB(0-) = muB(0+) + a(0+)*muK(0+)
```

At `a_0plus_max -> 0`, the ceiling meets the PNM--equilibrated three-flavor transition. At `a_0plus_max -> 1`, it meets the strangeness-free transition. The moving solvers must never be called at the exact endpoints.

Both public calls must omit `a_0plus`:

```python
analytic = pv.analytic_velocity_isothermal(
    T_0minus,
    nB_0minus,
    B_one_forth,
    xi=xi,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    upB=5000,
)

numerical = pv.solve_front_isothermal(
    T,
    nB_0minus,
    B_one_forth,
    ms=0.0,
    param=para.paraQMCRMF3,
    NM_type="PNM",
    tail_eps=tail_eps,
    n_mesh=n_mesh,
    tol_bvp=tol_bvp,
    max_nodes=max_nodes,
    jB_guess=jB_guess,
    jB_bounds=jB_bounds,
)
```

The returned `u_0minus` is proper velocity. Convert it to ordinary speed only with:

```python
SPEED_OF_LIGHT_M_S = 299_792_458.0
velocity_m_s = SPEED_OF_LIGHT_M_S * u_0minus / np.sqrt(1.0 + u_0minus**2)
```

Do not introduce an additional Lorentz factor into baryon or momentum flux. The live `Pi = P + h*u**2` closure is already exact under this convention.

## File structure

- Create `new_paper_calculations/26-08-19/_isothermal_domain.py`: axes, phase curves, ceiling inversion, payload schemas, proper-speed conversion, and atomic save/load.
- Create `new_paper_calculations/26-08-19/_isothermal_execution.py`: worker discovery, analytical tasks, numerical seed generation, hard timeouts, shell scheduling, and checkpoint assembly.
- Create `new_paper_calculations/26-08-19/prepare_isothermal_domain.py`: domain CLI and visible production defaults.
- Create `new_paper_calculations/26-08-19/run_isothermal_analytic.py`: analytical-stage CLI.
- Create `new_paper_calculations/26-08-19/run_isothermal_numerical.py`: numerical-stage CLI.
- Create `new_paper_calculations/26-08-19/run_isothermal_all.py`: sequential prepare/analytical/numerical/plot driver.
- Create `new_paper_calculations/26-08-19/plot_isothermal_contours.py`: comparison figure and mask rendering.
- Create `new_paper_calculations/26-08-19/README.md`: cluster, resume, smoke-test, and payload instructions.
- Create `tests/test_isothermal_contour_cluster_26_08_19.py`: fast mocked contract tests plus a small opt-in live-EOS smoke test.
- Modify `.gitignore` only enough to track the new dated source directory and its test; keep generated payloads and figures ignored.
- Do not modify any file under `RMFsolver/`.

### Task 1: Lock the axes, schemas, and proper-velocity convention

**Files:**
- Create: `new_paper_calculations/26-08-19/_isothermal_domain.py`
- Create: `new_paper_calculations/26-08-19/prepare_isothermal_domain.py`
- Create: `tests/test_isothermal_contour_cluster_26_08_19.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing tests for production defaults**

Assert these exact defaults:

```python
assert domain.DEFAULT_B_ONE_FORTH_MEV == 189.1565957288247
assert domain.DEFAULT_XI == -0.5
assert domain.DEFAULT_NM_TYPE == "PNM"
assert domain.DEFAULT_MS_MEV == 0.0
assert domain.DEFAULT_UPB == 5000

temperature = domain.default_temperature_axis()
composition = domain.default_a_0plus_target_axis()
assert temperature.shape == (30,)
assert composition.shape == (20,)
assert temperature[0] == 1.0e-2
assert temperature[-1] == 120.0
assert composition[0] == 0.01
assert composition[-1] == 0.99
```

Also test `proper_velocity_to_m_s(0) == 0`, finite output above `u=1`, odd symmetry, and agreement with `c*u/sqrt(1+u**2)`.

- [ ] **Step 2: Run the tests and verify the expected import failure**

Run:

```bash
./bin/python -m unittest tests.test_isothermal_contour_cluster_26_08_19.AxisAndUnitsTests -v
```

Expected: failure because the dated modules do not exist.

- [ ] **Step 3: Implement exact axes and common schema helpers**

Use:

```python
def default_temperature_axis():
    low = np.geomspace(1.0e-2, 1.0, 10)
    high = np.linspace(1.0, 120.0, 21)[1:]
    return np.concatenate((low, high))


def default_a_0plus_target_axis():
    return np.linspace(0.01, 0.99, 20)


def phase_temperature_axis():
    return np.concatenate(([0.0], default_temperature_axis()))
```

Define `SCHEMA_VERSION = 2` and require every payload to record the axes, physical defaults, run tag, creation/update timestamps, current Git commit when available, and the live isothermal API signatures.

- [ ] **Step 4: Add atomic dictionary persistence**

Write `np.save` to a temporary sibling, flush and `os.fsync`, then replace the requested path with `os.replace`. Loading must require a dictionary and the expected `schema_version`; resume must reject mismatched axes or physical parameters.

- [ ] **Step 5: Run the focused tests and commit**

Run the command from Step 2. Expected: pass.

```bash
git add .gitignore tests/test_isothermal_contour_cluster_26_08_19.py new_paper_calculations/26-08-19
git commit -m "feat: scaffold isothermal contour cluster workflow"
```

### Task 2: Build the two phase boundaries and invert the live ceiling

**Files:**
- Modify: `new_paper_calculations/26-08-19/_isothermal_domain.py`
- Modify: `new_paper_calculations/26-08-19/prepare_isothermal_domain.py`
- Modify: `tests/test_isothermal_contour_cluster_26_08_19.py`

- [ ] **Step 1: Write failing domain-physics tests**

For a live EOS point, assert that the helper result obeys:

```python
ceiling = domain.ceiling_at_upstream(T=1.0, nB_0minus=3.5*N0, B_one_forth=B14)
assert ceiling["status"] == "interior"
assert abs(
    ceiling["muB_0plus"]
    + ceiling["a_0plus_max"]*ceiling["muK_0plus"]
    - ceiling["muB_0minus"]
) < 1.0e-6
```

Test that the inverse solve returns a density whose recomputed ceiling differs from the target by at most `1e-8`. Test ordered lower/interior/upper densities and a `no_allowed_band` row where the phase curves do not bracket a physical domain.

- [ ] **Step 2: Implement the lower phase curve**

At each temperature, solve common pressure and baryon chemical potential between PNM and equilibrated quark matter with `muK_0plus=0`. Save `T`, `nB_0minus`, `muB_0minus`, pressure residual, and status. Use continuation from the preceding temperature and a bounded fallback scan over the validated nuclear chemical-potential range.

- [ ] **Step 3: Implement the upper phase curve**

Solve the strangeness-free state using unknown `muB_0minus` and `muK_0plus`, with:

```text
muB_0plus = muB_0minus - muK_0plus
P(0+) - P(0-) = 0
n_s(0+) = 0
```

Save the same diagnostics plus `muB_0plus` and `muK_0plus`. A missing or unordered root marks the temperature row `no_allowed_band`; it must not abort other rows.

- [ ] **Step 4: Wrap the live thermodynamic-ceiling helper**

For a candidate `nB_0minus`, recover the branch-validated PNM state, then call `pv._solve_a_0plus_max`. Do not copy its Gibbs residual or substitute an independently derived closure. Echo `status`, `a_0plus_max`, endpoint chemical potentials, and the weighted residual.

- [ ] **Step 5: Invert the ceiling for every interior grid cell**

For each valid temperature and each target from `default_a_0plus_target_axis`, solve:

```text
a_0plus_max(T, nB_0minus) - a_0plus_target = 0
```

Bracket strictly between the two phase densities and use `brentq`. Recompute the ceiling at the root and accept only `status="interior"` with absolute composition residual at most `1e-8`. Save two-dimensional `nB_0minus`, `muB_0minus`, `P_0minus`, resolved `a_0plus_max`, residual, and cell-status arrays.

- [ ] **Step 6: Save the domain payload and run tests**

Use `isothermal-domain.npy` by default. Include the (T=0) phase-boundary row, but never create (T=0) moving-front cells.

```bash
./bin/python -m unittest tests.test_isothermal_contour_cluster_26_08_19.DomainPhysicsTests -v
```

Expected: all domain tests pass.

```bash
git add tests/test_isothermal_contour_cluster_26_08_19.py new_paper_calculations/26-08-19
git commit -m "feat: construct thermodynamic isothermal contour domain"
```

### Task 3: Implement the analytical cluster stage with current validity gates

**Files:**
- Create: `new_paper_calculations/26-08-19/_isothermal_execution.py`
- Create: `new_paper_calculations/26-08-19/run_isothermal_analytic.py`
- Modify: `tests/test_isothermal_contour_cluster_26_08_19.py`

- [ ] **Step 1: Write failing call-contract and status tests**

Mock `analytic_velocity_isothermal` and assert that `a_0plus` is absent from both positional and keyword arguments. Assert `xi=-0.5`, `ms=0`, `NM_type="PNM"`, and `upB=5000` are passed explicitly.

Test classification of `moving_front`, `slow_front_approximation_invalid`, `momentum_flux_ratio_above_tolerance`, `composition_ceiling_saturated`, stable/coexistence disagreement, exception, and timeout records. Only a successful `moving_front` may populate the plotted speed grid.

- [ ] **Step 2: Implement one analytical task**

Call the public function through its automatic-maximum path. Require:

```python
result["a_0plus_source"] == "maximum"
abs(result["a_0plus_max"] - task["a_0plus_target"]) <= 1.0e-7
```

Keep the full diagnostic dictionary. Add `task_status`, elapsed time, grid indices, target composition, ordinary velocity, and a `domain_solver_disagreement` status when the solver's phase classification contradicts an interior domain cell.

- [ ] **Step 3: Implement independent parallel scheduling and resume**

Use a `spawn`-context `ProcessPoolExecutor`. Determine workers from `--workers`, then `SLURM_CPUS_PER_TASK`, `PBS_NP`, or `NSLOTS`, then `os.cpu_count()`. Set BLAS/OpenMP thread counts to one before importing numerical libraries in workers. Atomically checkpoint every terminal cell and skip terminal records on resume.

- [ ] **Step 4: Preserve all new analytical diagnostics**

The scalar arrays and records must retain `momentum_flux_ratio`, `gamma_minus_1`, `slow_front_consistent`, `a_0plus_max_status`, `a_0plus_max_residual`, `delta_muB`, formula residual, both momentum residuals, and composition residual. Do not create any separate relativistic-flux field.

- [ ] **Step 5: Run tests and commit**

```bash
./bin/python -m unittest tests.test_isothermal_contour_cluster_26_08_19.AnalyticalStageTests -v
```

Expected: pass.

```bash
git add tests/test_isothermal_contour_cluster_26_08_19.py new_paper_calculations/26-08-19
git commit -m "feat: add analytical isothermal contour stage"
```

### Task 4: Implement numerical shell continuation and killable timeouts

**Files:**
- Modify: `new_paper_calculations/26-08-19/_isothermal_execution.py`
- Create: `new_paper_calculations/26-08-19/run_isothermal_numerical.py`
- Modify: `tests/test_isothermal_contour_cluster_26_08_19.py`

- [ ] **Step 1: Write failing numerical call-contract tests**

Mock `solve_front_isothermal` and assert `a_0plus` is omitted. Assert `ms=0`, `NM_type="PNM"`, `tail_eps=1e-8`, `n_mesh=300`, `tol_bvp=1e-4`, `max_nodes=50000`, and bounded positive `jB_guess` are passed.

Test that a successful record reports `a_0plus_source="maximum"`, `a_0plus_max_status="interior"`, and composition agreement within `1e-7`.

- [ ] **Step 2: Add hard subprocess timeouts**

Each BVP attempt must run in a disposable `spawn` child. The parent waits for at most `min(trial_timeout, cell_deadline-now)`, terminates and joins the child, then kills it if needed. Defaults are:

```python
DEFAULT_TRIAL_TIMEOUT_S = 180.0
DEFAULT_CELL_TIMEOUT_S = 900.0
DEFAULT_JB_LOWER_BOUND = 1.0e-12
DEFAULT_JB_UPPER_FACTOR = 1.0e-4
```

Every attempt must record guess, source, runtime, timeout limits, status, exception text, and whether termination was required. Tests must prove no child survives a timeout.

- [ ] **Step 3: Implement increasing-composition shell scheduling**

Process the 20 composition columns from `0.01` to `0.99`. Run valid temperatures in parallel within one shell, then checkpoint before advancing. The numerical API accepts no profile-continuation input, so continue only scalar `jB`; do not invent a profile keyword.

- [ ] **Step 4: Implement deterministic `jB` candidate order**

Use the first finite candidate from each source, deduplicating guesses within 3 percent:

1. Previous successful numerical shell at the same temperature.
2. Logarithmic extrapolation from the previous two successful shells at that temperature.
3. Current-cell analytical `jB`.
4. Each available base multiplied by `0.5`, `0.8`, `1.25`, and `2.0`.
5. Median successful `jB` from the two nearest temperatures on the current shell, multiplied by `0.8`, `1.0`, and `1.25`.
6. `1e-6*nB_0minus` as the final fallback.

Clip every guess strictly inside `(1e-12, 1e-4*nB_0minus)`. Failure of one temperature must not block the rest of its shell or later shells.

- [ ] **Step 5: Validate numerical results and checkpoint**

Accept only `success=True`, finite positive `jB` and `u_0minus`, automatic interior composition matching the domain target, finite endpoint momentum data, and finite BVP residuals. Retain the exact-model tags `physical nK/jK`, local diffusion, exact nonleptonic rate, compact-coordinate metadata, tail residual, boundary residuals, closure error, and momentum diagnostics. Use `return_profile=False` for the production contour.

- [ ] **Step 6: Run timeout, continuation, and resume tests**

```bash
./bin/python -m unittest tests.test_isothermal_contour_cluster_26_08_19.NumericalStageTests -v
```

Expected: all tests pass, including forced timeout cleanup and a resume from a partially completed shell.

```bash
git add tests/test_isothermal_contour_cluster_26_08_19.py new_paper_calculations/26-08-19
git commit -m "feat: add numerical isothermal contour stage"
```

### Task 5: Add comparison plotting and the all-stage driver

**Files:**
- Create: `new_paper_calculations/26-08-19/plot_isothermal_contours.py`
- Create: `new_paper_calculations/26-08-19/run_isothermal_all.py`
- Create: `new_paper_calculations/26-08-19/README.md`
- Modify: `tests/test_isothermal_contour_cluster_26_08_19.py`

- [ ] **Step 1: Write failing plot-mask and orchestration tests**

Use synthetic payloads to assert that only successful moving cells enter contours, the stable-PNM region is shaded below the lower curve, and the `a_0plus_max=1` region is shaded above the upper curve. Assert that no infinite value is inserted into either velocity array.

- [ ] **Step 2: Implement the comparison plot**

Use the two-dimensional `nB_0minus/n0` coordinates with the shared temperature grid. Draw analytical and numerical panels with default levels `(50, 100, 200, 500, 1000, 2000)` m/s, the lower curve in purple dash-dot, and the upper curve in red dashed style. Label the upper curve as the formal analytical divergence boundary while leaving its velocity masked.

- [ ] **Step 3: Implement the all-stage driver**

Run prepare, analytical, numerical, then plot as subprocess stages. Stop after a nonzero return code, pass common physical arguments through unchanged, and preserve completed stage payloads. Add `--smoke`, which selects three temperatures and three interior compositions while retaining the same physics and schemas.

- [ ] **Step 4: Document exact commands and outputs**

The README must show:

```bash
./bin/python new_paper_calculations/26-08-19/run_isothermal_all.py --workers 64
./bin/python new_paper_calculations/26-08-19/run_isothermal_all.py --workers 4 --smoke
./bin/python new_paper_calculations/26-08-19/run_isothermal_numerical.py --resume
```

Document `isothermal-domain.npy`, `isothermal-analytic.npy`, `isothermal-numerical.npy`, and `isothermal-contours.png`, including which statuses are masked.

- [ ] **Step 5: Run tests and commit**

```bash
./bin/python -m unittest tests.test_isothermal_contour_cluster_26_08_19 -v
```

Expected: all fast cluster tests pass.

```bash
git add tests/test_isothermal_contour_cluster_26_08_19.py new_paper_calculations/26-08-19
git commit -m "feat: complete isothermal contour cluster workflow"
```

### Task 6: Live verification and wiki refresh

**Files:**
- Modify: `wiki/physics/isothermal-analytic-front-speed.md`
- Modify: `wiki/solvers/phase-velocity-overview.md`
- Modify: `wiki/index.md`
- Modify: `wiki/log.md`

- [ ] **Step 1: Run the focused live solver suite**

```bash
./bin/python -m unittest \
  tests.test_phase_velocity_isothermal_analytic \
  tests.test_phase_velocity_isothermal_numerical \
  tests.test_phase_velocity_isothermal_a_0plus_max \
  tests.test_phase_velocity_isothermal_relativistic -v
```

Expected: all focused isothermal tests pass.

- [ ] **Step 2: Run the cluster smoke workflow**

```bash
./bin/python new_paper_calculations/26-08-19/run_isothermal_all.py \
  --workers 3 --smoke --output-dir /tmp/isothermal-contour-smoke
```

Expected: domain and analytical stages complete; every numerical cell ends in a recorded success or structured terminal failure; plotting completes without using infinity.

- [ ] **Step 3: Check smoke invariants**

Verify ordered phase boundaries, interior `a_0plus_max` agreement below `1e-7`, negative `delta_muB` for analytical moving cells, exact proper-speed conversion, finite successful numerical residuals, and identical grid coordinates in both velocity payloads.

- [ ] **Step 4: Refresh the wiki**

Read `wiki/SCHEMA.md` completely before editing. Update the analytical page and solver overview to document optional automatic `a_0plus`, the weighted chemical-potential balance, `composition_ceiling_saturated`, the momentum-flux validity gate, and the proper-velocity convention. Add the cluster workflow to the index and append exactly one dated session entry to `wiki/log.md`. Use location-based endpoint notation exclusively.

- [ ] **Step 5: Run final repository checks and commit**

```bash
git diff --check
./bin/python -m unittest tests.test_isothermal_contour_cluster_26_08_19 -v
git status --short
```

Expected: no whitespace errors, all cluster tests pass, and status contains only the intended files plus any pre-existing user changes.

```bash
git add wiki tests/test_isothermal_contour_cluster_26_08_19.py new_paper_calculations/26-08-19 .gitignore
git commit -m "docs: record isothermal contour workflow"
```

## Acceptance criteria

- The production grid contains 30 positive-temperature rows and 20 uniformly spaced thermodynamic-maximum compositions, with (T=0) stored only for phase curves.
- Interior densities are obtained by inverting the live weighted ceiling; no unweighted chemical-potential closure remains.
- Both public solvers are called with `a_0plus` omitted and independently report `a_0plus_source="maximum"`.
- The lower and upper boundaries are stored explicitly, while exact `a_0plus_max` values 0 and 1 are never passed to moving solvers.
- Proper velocity is converted to ordinary speed without an extra Lorentz factor.
- Analytical validity failures and numerical BVP failures are preserved as structured masks rather than converted to zero or infinity.
- The numerical stage is killable, bounded in wall time per attempt and cell, atomically resumable, and parallel only across independent temperatures within one composition shell.
- No source file under `RMFsolver/` is modified.
