# Analytical Log-T Log-m Cluster Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel analytical velocity scan over `T = 1e-2...50 MeV` and `m = 1e-5...1`, with logarithmic coordinates, a mesh adapted only to the `Delta h = 0` boundary, and a saved entropy-per-baryon zero curve that does not constrain the mesh.

**Architecture:** Create a single-block runner around `phase_velocity.analytic_velocity_bound(..., interface_fraction_mode="extreme_endothermic")`. At every logarithmic temperature, solve the finite-temperature `Delta h = 0` metastability boundary and build a logarithmic row on its allowed side. Start near `10 MeV`, propagate downstream endpoint guesses through neighboring successful analytical solutions, and parallelize each completed row. Derive the entropy-jump grid and zero curve after the velocity scan without changing the mesh. Do not run an `A = 1` boundary search because the scan ends at `50 MeV`.

**Tech Stack:** Python, NumPy, SciPy, `ProcessPoolExecutor`, tqdm, `unittest`

---

## File Structure

- Create `new_paper_calculations/26-07-19/scan_analytic_logTm.py`: standalone analytical cluster runner.
- Create `tests/test_cluster_analytic_logTm_26_07_19.py`: mocked tests for defaults, mesh topology, analytical closure, continuation, entropy extraction, and payload schema.
- Reuse `new_paper_calculations/26-07-19/_hydro_analytic_contour_common.py`: `AnalyticPointTask`, `calculate_point`, bag calibration, finite-temperature critical density, and EOS helpers.
- Leave `uNmax_contour.py`, `uNmax_contour-log.py`, and all existing result artifacts unchanged.

The concise runner name is `scan_analytic_logTm.py`; the default payload is `analytic-uN-logTm.npy`.

### Task 1: Establish the analytical runner contract

**Files:**
- Create: `new_paper_calculations/26-07-19/scan_analytic_logTm.py`
- Create: `tests/test_cluster_analytic_logTm_26_07_19.py`

- [ ] **Step 1: Write failing default-contract tests**

```python
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "new_paper_calculations"
    / "26-07-19"
    / "scan_analytic_logTm.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("scan_analytic_logTm", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnalyticLogTLogMRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_defaults_define_requested_domain(self):
        args = self.runner.build_parser().parse_args([])
        self.assertEqual(args.t_min, 1.0e-2)
        self.assertEqual(args.t_max, 50.0)
        self.assertEqual(args.m_min, 1.0e-5)
        self.assertEqual(args.m_max, 1.0)
        self.assertEqual(args.t_count, 30)
        self.assertEqual(args.m_count, 20)
        self.assertEqual(args.seed_temperature, 10.0)
        self.assertEqual(args.interface_fraction_mode, "extreme_endothermic")
        self.assertEqual(args.output, "analytic-uN-logTm.npy")

    def test_runner_contains_no_numerical_uNmax_contract(self):
        self.assertFalse(hasattr(self.runner, "NumericalPointTask"))
        self.assertFalse(hasattr(self.runner, "resolve_jB_bounds"))

    def test_runner_contains_no_A_boundary_search(self):
        self.assertFalse(hasattr(self.runner, "ABoundaryTask"))
        self.assertFalse(hasattr(self.runner, "solve_A_boundary_for_metastability"))
```

- [ ] **Step 2: Run the tests and verify the runner import fails**

```bash
./bin/python -m unittest tests.test_cluster_analytic_logTm_26_07_19 -v
```

Expected: import failure because `scan_analytic_logTm.py` does not exist.

- [ ] **Step 3: Add the runner header and tuning block**

```python
# Tunable run parameters
DEFAULT_OUTPUT = "analytic-uN-logTm.npy"
DEFAULT_RUN_TAG = "analytic_uN_logT_logm_extreme_v1"
DEFAULT_T_MIN_MEV = 1.0e-2
DEFAULT_T_MAX_MEV = 50.0
DEFAULT_T_COUNT = 30
DEFAULT_M_MIN = 1.0e-5
DEFAULT_M_MAX = 1.0
DEFAULT_M_COUNT = 20
DEFAULT_GAP_BUFFER_FRACTION = 1.0e-3
DEFAULT_SEED_TEMPERATURE_MEV = 10.0
DEFAULT_NEIGHBOR_RETRY = True
DEFAULT_SAVE_EVERY = 10
DEFAULT_INTERFACE_FRACTION_MODE = "extreme_endothermic"
DEFAULT_ENDPOINT_GUESS = None
DEFAULT_NCRIT_OVER_N0 = 3.0
DEFAULT_XI = 0.0
DEFAULT_MS_MEV = 0.0
DEFAULT_NM_TYPE = "PNM"
DEFAULT_UPB = 5000
```

Expose matching CLI flags, including `--gap-buffer-fraction`, `--seed-temperature`, `--endpoint-guess MU_B_Q,T_Q`, `--no-neighbor-retry`, `--save-every`, `--B14`, and `--workers`. Preserve the `RUN_TAG` environment override.

- [ ] **Step 4: Validate the log domain**

```python
def _validate_args(args):
    if not 0.0 < args.t_min < args.t_max <= 50.0:
        raise ValueError("require 0 < t-min < t-max <= 50 MeV")
    if not 0.0 < args.m_min < args.m_max:
        raise ValueError("log metastability bounds require 0 < m-min < m-max")
    if args.t_count < 2 or args.m_count < 2:
        raise ValueError("t-count and m-count must be at least 2")
    if args.gap_buffer_fraction < 0.0:
        raise ValueError("gap-buffer-fraction must be nonnegative")
```

- [ ] **Step 5: Run the default-contract tests**

Expected: all Task 1 tests pass.

### Task 2: Construct the Delta-h-only adaptive logarithmic mesh

**Files:**
- Modify: `new_paper_calculations/26-07-19/scan_analytic_logTm.py`
- Test: `tests/test_cluster_analytic_logTm_26_07_19.py`

- [ ] **Step 1: Add failing logarithmic-mesh tests**

```python
import numpy as np


def test_temperature_axis_is_logarithmic(self):
    values = self.runner.build_temperature_axis(1.0e-2, 50.0, 30)
    self.assertEqual(values[0], 1.0e-2)
    self.assertEqual(values[-1], 50.0)
    np.testing.assert_allclose(np.diff(np.log(values)), np.diff(np.log(values))[0])


def test_gap_root_sets_log_metastability_lower_bound(self):
    values = self.runner.build_metastability_row(
        boundary=2.0e-2,
        boundary_status=self.runner.GAP_BOUNDARY_ROOT,
        allowed_above_boundary=True,
        m_min=1.0e-5,
        m_max=1.0,
        count=20,
        gap_buffer_fraction=1.0e-3,
    )
    self.assertAlmostEqual(values[0], 2.0e-2 * 1.001)
    self.assertEqual(values[-1], 1.0)
    np.testing.assert_allclose(np.diff(np.log(values)), np.diff(np.log(values))[0])


def test_all_allowed_row_starts_at_global_minimum(self):
    values = self.runner.build_metastability_row(
        boundary=np.nan,
        boundary_status=self.runner.GAP_BOUNDARY_ALL_ALLOWED,
        allowed_above_boundary=True,
        m_min=1.0e-5,
        m_max=1.0,
        count=20,
        gap_buffer_fraction=1.0e-3,
    )
    self.assertEqual(values[0], 1.0e-5)


def test_all_forbidden_row_is_skipped(self):
    values = self.runner.build_metastability_row(
        boundary=np.nan,
        boundary_status=self.runner.GAP_BOUNDARY_ALL_FORBIDDEN,
        allowed_above_boundary=True,
        m_min=1.0e-5,
        m_max=1.0,
        count=20,
        gap_buffer_fraction=1.0e-3,
    )
    self.assertIsNone(values)
```

- [ ] **Step 2: Implement the axis builders**

```python
def build_temperature_axis(t_min, t_max, count):
    return np.geomspace(float(t_min), float(t_max), int(count))


def build_metastability_row(
    *, boundary, boundary_status, allowed_above_boundary,
    m_min, m_max, count, gap_buffer_fraction,
):
    if boundary_status == GAP_BOUNDARY_ALL_FORBIDDEN:
        return None
    if boundary_status == GAP_BOUNDARY_ALL_ALLOWED:
        lower = float(m_min)
    elif boundary_status == GAP_BOUNDARY_ROOT and allowed_above_boundary:
        lower = max(
            float(m_min),
            float(boundary) * (1.0 + float(gap_buffer_fraction)),
        )
    else:
        return None
    if not 0.0 < lower < float(m_max):
        return None
    return np.geomspace(lower, float(m_max), int(count))
```

- [ ] **Step 3: Solve Delta h at every actual temperature**

Use the manuscript metastability definition:

```text
nB_N(T,m) = nB_crit(T) + m*n0
```

For every exact temperature row, solve `Delta h = 0` in `m_min...m_max`. Save the root, status, endpoint signs, and `allowed_above_boundary`. Do not interpolate a coarse boundary onto the scan temperatures.

- [ ] **Step 4: Build each adaptive row only from Delta h**

Use `np.geomspace(lower_allowed_m, 1.0, m_count)`. The entropy sign and analytical `A_boundary` value must not enter `build_metastability_row`.

- [ ] **Step 5: Run the mesh tests**

Expected: all Task 2 tests pass.

### Task 3: Evaluate the extreme-endothermic analytical velocity with endpoint continuation

**Files:**
- Modify: `new_paper_calculations/26-07-19/scan_analytic_logTm.py`
- Test: `tests/test_cluster_analytic_logTm_26_07_19.py`

- [ ] **Step 1: Add a failing analytical-call test**

Mock `_hydro_analytic_contour_common.calculate_point` and assert the runner creates:

```python
AnalyticPointTask(
    T_N=temperature,
    metastability=metastability,
    n0=n0,
    nB_crit0=nB_crit0,
    B14=B14,
    xi=xi,
    ms=ms,
    nm_type=nm_type,
    upB=upB,
    interface_fraction_mode="extreme_endothermic",
    endpoint_initial_guess=endpoint_guess,
)
```

Assert that no call is made to `solve_front_energy_conserving_uNmax`.

- [ ] **Step 2: Add failing endpoint-neighbor tests**

```python
def test_adjacent_temperature_uses_nearest_log_m_endpoint(self):
    adjacent = {
        0: {"success": True, "metastability": 1.0e-3,
            "endpoint_solution": (1000.0, 20.0)},
        1: {"success": True, "metastability": 1.0e-2,
            "endpoint_solution": (1050.0, 25.0)},
    }
    guess, source = self.runner.adjacent_row_endpoint_guess(
        metastability=8.0e-3,
        adjacent_row=adjacent,
        default=None,
    )
    self.assertEqual(guess, (1050.0, 25.0))
    self.assertEqual(source, "adjacent_temperature_nearest_log_m")


def test_failed_nearest_endpoint_does_not_use_farther_success(self):
    adjacent = {
        0: {"success": True, "metastability": 1.0e-3,
            "endpoint_solution": (1000.0, 20.0)},
        1: {"success": False, "metastability": 1.0e-2,
            "endpoint_solution": None},
    }
    guess, source = self.runner.adjacent_row_endpoint_guess(
        metastability=8.0e-3,
        adjacent_row=adjacent,
        default=(1100.0, 30.0),
    )
    self.assertEqual(guess, (1100.0, 30.0))
    self.assertEqual(source, "fallback_after_failed_nearest_log_m")
```

- [ ] **Step 3: Implement analytical point evaluation**

Call `_hydro_analytic_contour_common.calculate_point(task)`, which delegates to:

```python
phase_velocity.analytic_velocity_bound(
    muB_N,
    T_N,
    B14,
    xi=xi,
    ms=ms,
    initial_guess=endpoint_initial_guess,
    interface_fraction_mode="extreme_endothermic",
)
```

Retain the current analytical schema, including `u_N_m_s`, `lambda_n`, `A_boundary`, `a_interface`, `a_interface_LTE`, `analytic_velocity_residual`, `h_over_nB_N_MeV`, `h_over_nB_Q_MeV`, and `endpoint_solution = (muB_Q, T_Q)`.

- [ ] **Step 4: Implement temperature-first continuation**

1. Order temperature rows from the row nearest `10 MeV`, then upward to `50 MeV`, then downward to `1e-2 MeV`.
2. Solve the seed row from `m = 1` toward its Delta-h boundary, using only the immediately adjacent higher-m successful endpoint.
3. For each later row, use only the immediately adjacent completed temperature row.
4. Select its nearest point in `abs(log(m_previous) - log(m_current))`.
5. If that nearest point failed, use the user/default endpoint seed; do not search a farther point.
6. Once seeds are chosen, submit all points in the current row to the process pool.

- [ ] **Step 5: Keep one integrated tqdm bar and one retry pass**

The bar total equals the actual number of safe mesh points and increases before retries. Retry a failure once from an immediate same-row successful neighbor, then from the nearest point in the immediately adjacent temperature row. Preserve original failure records and append retry records.

- [ ] **Step 6: Run analytical-call and continuation tests**

Expected: tests pass without running EOS-heavy calculations.

### Task 4: Derive the entropy-per-baryon diagnostic and zero curve

**Files:**
- Modify: `new_paper_calculations/26-07-19/scan_analytic_logTm.py`
- Test: `tests/test_cluster_analytic_logTm_26_07_19.py`

- [ ] **Step 1: Add failing entropy tests**

```python
def test_entropy_jump_is_downstream_minus_upstream(self):
    row = {
        "T_MeV": 10.0,
        "T_Q_MeV": 20.0,
        "h_over_nB_N_MeV": 1100.0,
        "muB_N_MeV": 1000.0,
        "h_over_nB_Q_MeV": 1050.0,
        "muB_Q_MeV": 1010.0,
    }
    value = self.runner.entropy_jump_per_baryon(row)
    self.assertEqual(value, 2.0 - 10.0)


def test_entropy_zero_uses_log_m_interpolation(self):
    crossing = self.runner.entropy_zero_crossing(
        metastability=np.array([1.0e-3, 1.0e-2]),
        entropy_jump=np.array([-1.0, 1.0]),
    )
    self.assertAlmostEqual(crossing, np.sqrt(1.0e-3 * 1.0e-2))
```

- [ ] **Step 2: Implement the entropy jump**

```python
def entropy_jump_per_baryon(row):
    values = np.asarray(
        [
            row.get("T_MeV", np.nan),
            row.get("T_Q_MeV", np.nan),
            row.get("h_over_nB_N_MeV", np.nan),
            row.get("muB_N_MeV", np.nan),
            row.get("h_over_nB_Q_MeV", np.nan),
            row.get("muB_Q_MeV", np.nan),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or values[0] <= 0.0 or values[1] <= 0.0:
        return np.nan
    T_N, T_Q, h_N_over_nB, muB_N, h_Q_over_nB, muB_Q = values
    return (h_Q_over_nB - muB_Q) / T_Q - (h_N_over_nB - muB_N) / T_N
```

Use:

```text
entropy_jump = s_Q/nB_Q - s_N/nB_N
entropy-forbidden region: entropy_jump <= 0
```

- [ ] **Step 3: Extract the curve only after velocity retries finish**

For each temperature row, sort successful finite points by metastability, find adjacent sign changes, and interpolate linearly in `log(m)`:

```python
fraction = -jump_left / (jump_right - jump_left)
log_m_zero = np.log(m_left) + fraction * (np.log(m_right) - np.log(m_left))
m_zero = float(np.exp(log_m_zero))
```

Save all crossings and these statuses:

```python
ENTROPY_BOUNDARY_ROOT = "root"
ENTROPY_ALL_ALLOWED = "all_positive"
ENTROPY_ALL_FORBIDDEN = "all_nonpositive"
ENTROPY_MULTIPLE_ROOTS = "multiple_roots"
ENTROPY_INSUFFICIENT = "insufficient_data"
```

Do not add analytical solves to refine this curve. Do not remove points, alter metastability rows, or change solve ordering based on entropy.

- [ ] **Step 4: Run entropy tests**

Expected: all Task 4 tests pass.

### Task 5: Save a single-block analytical payload with atomic checkpoints

**Files:**
- Modify: `new_paper_calculations/26-07-19/scan_analytic_logTm.py`
- Test: `tests/test_cluster_analytic_logTm_26_07_19.py`

- [ ] **Step 1: Add a failing payload-schema test**

Assert:

```python
expected_metadata = {
    "temperature_axis_scale": "log",
    "metastability_axis_scale": "log",
    "mesh_layout": "single_block_logT_adaptive_logm",
    "analytic_closure": "hydro_jump_eigenvalue",
    "interface_fraction_mode": "extreme_endothermic",
    "analytic_formula_variant": "piecewise_constant_lambda_n",
    "safe_region_definition": "Delta_h <= 0",
    "A_boundary_handling": "not_scanned_T_max_50_MeV",
    "entropy_jump_definition": "sQ_over_nB_Q_minus_sN_over_nB_N",
    "entropy_forbidden_condition": "entropy_jump <= 0",
    "entropy_affects_mesh": False,
}
```

Assert that the payload has no `A_boundary_records`, no BVP/root-method fields, and no two-block seam arrays.

- [ ] **Step 2: Save the analytical arrays and records**

```text
temperature_values
temperature_grid
metastability_grid
status_grid
velocity_grid_m_s
u_N_grid
jB_grid
lambda_n_grid
A_boundary_grid
a_interface_grid
a_interface_LTE_grid
entropy_jump_grid
gap_boundary_metastability_values
gap_boundary_status
entropy_boundary_metastability_values
entropy_boundary_status
entropy_boundary_crossings
records
gap_boundary_records
```

`A_boundary_grid` is audit-only. Save its finite maximum and print a warning if any successful point has `A_boundary >= 1`; do not perform a separate A-boundary search or reshape the mesh.

- [ ] **Step 3: Preserve incremental atomic saves**

Save after every `save_every` boundary tasks and analytical point attempts using temporary-file replacement. Always save once after entropy postprocessing and at normal completion.

- [ ] **Step 4: Run payload tests**

Expected: all Task 5 tests pass.

### Task 6: Verify the runner and a four-edge analytical smoke scan

**Files:**
- Verify: `new_paper_calculations/26-07-19/scan_analytic_logTm.py`
- Verify: `tests/test_cluster_analytic_logTm_26_07_19.py`

- [ ] **Step 1: Compile and inspect CLI help**

```bash
./bin/python -m py_compile \
  new_paper_calculations/26-07-19/scan_analytic_logTm.py \
  tests/test_cluster_analytic_logTm_26_07_19.py
MPLCONFIGDIR=/private/tmp/mpl-cache ./bin/python \
  new_paper_calculations/26-07-19/scan_analytic_logTm.py --help
```

Expected: commands exit zero; help describes analytical extreme-endothermic velocity, log axes, and endpoint seeding.

- [ ] **Step 2: Run focused tests**

```bash
./bin/python -m unittest \
  tests.test_cluster_analytic_logTm_26_07_19 \
  tests.test_cluster_analytic_26_07_18 -v
```

Expected: all tests pass.

- [ ] **Step 3: Run a four-edge scan over the requested domain**

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache ./bin/python \
  new_paper_calculations/26-07-19/scan_analytic_logTm.py \
  --workers 1 \
  --t-count 2 \
  --m-count 2 \
  --save-every 1 \
  --no-neighbor-retry \
  --output /private/tmp/analytic-uN-logTm-smoke.npy
```

Expected: the integrated bar reaches every submitted safe cell and the payload is saved even if an edge point fails.

- [ ] **Step 4: Inspect the smoke payload**

Verify:

```text
temperature_values = [1e-2, 50]
every metastability row is positive and logarithmically spaced
no submitted point lies on the Delta h > 0 side
entropy_jump_grid matches velocity_grid_m_s shape
entropy boundary fields exist but did not modify the mesh
no A-boundary search records exist
every successful point reports interface_fraction_mode = extreme_endothermic
```

- [ ] **Step 5: Record the verified-file checkpoint**

Confirm the intended new files are exactly `scan_analytic_logTm.py`, `tests/test_cluster_analytic_logTm_26_07_19.py`, and this plan. This workspace is not currently detected as a Git working tree, so no commit command is included.

## Assumptions

- The scan uses the analytical piecewise-constant `lambda_n` velocity formula, not uNmax or the numerical nK solver.
- The interface fraction mode is `extreme_endothermic`.
- `T <= 50 MeV` stays below the `A_boundary = 1` limit; A is retained only as an audit field.
- The only mesh-defining forbidden region is `Delta h > 0`, on the low-metastability side.
- The entropy condition is `s_Q/nB_Q - s_N/nB_N <= 0`; its zero curve is saved for plotting but never changes the mesh.
- The default grid remains 30 temperatures by 20 metastability points, preserving the current 600-point cost.

## Self-Review

- All numerical BVP, nK, fixed-`TQstar`, and `jB`-shooting requirements were removed.
- The analytical closure, endpoint continuation, log mesh, Delta-h boundary, entropy curve, parallelism, and saved schema each have explicit tests.
- No A-boundary search remains, and entropy cannot feed back into mesh generation.
- Existing runners and datasets remain untouched.
