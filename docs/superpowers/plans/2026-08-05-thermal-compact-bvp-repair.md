# Thermal-Conducting Compact BVP Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the thermal solver's finite-physical-`x` truncation with the established compact-coordinate eigenparameter BVP, so a successful solution demonstrably reaches the equilibrated quark-matter tail selected by `tail_eps`.

**Architecture:** Solve the existing three physical ODEs for `[nK, jK, T]` on the fixed compact interval `s in [0, 1-tail_eps]`, with `s = 1-exp(-lambda_stable*x)` and `jB` retained as the single `solve_bvp` parameter. Keep the two growing-mode projection conditions, but impose them deep in the asymptotic tail and reject solutions whose endpoint is not independently close to `(muK, q_th, nK, jK) = (0, 0, 0, 0)`. Use temperature continuation to reach `T_interface=0` and tail-depth continuation to reach the requested `tail_eps` without attacking the stiffest problem from a cold start.

**Tech Stack:** Python, NumPy, SciPy 1.15 `solve_bvp`, the existing RMF/quark EOS helpers, and the function-style tests in `tests/test_phase_velocity.py`.

---

## Scope and non-goals

- Modify `/Users/janshao/NSM_related/RMFsolver/phase_velocity.py` only under `RMFsolver/`.
- Modify `/Users/janshao/NSM_related/tests/test_phase_velocity.py` for tests.
- Preserve the public signature and existing return keys of `solve_front_thermal_conducting` wherever possible.
- Leave `solve_front_energy_conserving_uNmax`, `solve_front_energy_conserving_nK`, and `solve_front_isothermal` unchanged.
- Do not change the EOS, weak rate, diffusion coefficient, or thermal-conductivity quadrature in this repair. The separate high-`y` `I_kappa` constant discrepancy remains a follow-up issue.
- Do not claim that Fourier conduction is physically valid at exact `T=0`; retain an LTE warning, but repair its thermal mean-free-path definition separately from this compactification change.
- `/Users/janshao/NSM_related` is not currently recognized as a Git repository, so this plan intentionally omits executable commit steps. If implementation is moved into a Git worktree, commit after each completed task.

## Target BVP

The physical equations remain:

```text
dnK/dx = (u*nK - jK)*invD
djK/dx = -Gamma_K
dT/dx  = (h*gamma*u - F_E)/kappa_th
```

Use:

```text
s       = 1 - exp(-lambda_compact*x)
x       = -log(1-s)/lambda_compact
dx/ds   = 1/(lambda_compact*(1-s))
s_end   = 1 - tail_eps
lambda_compact = abs(real(lambda_stable))
```

Therefore:

```text
dnK/ds = (dnK/dx)*dx/ds
djK/ds = (djK/dx)*dx/ds
dT/ds  = (dT/dx)*dx/ds
```

The four BVP residuals remain:

```text
T(0) = T_interface
jK(0) = jB
W_grow_1 . (y(s_end) - y_Q) = 0
W_grow_2 . (y(s_end) - y_Q) = 0
```

The crucial repair is that `s_end=1-tail_eps` is the actual numerical endpoint. No physical-`x` growth-budget cap may replace it.

### Task 1: Add regression tests that expose the current false infinity

**Files:**
- Modify: `/Users/janshao/NSM_related/tests/test_phase_velocity.py:2168-2287`
- Test: `/Users/janshao/NSM_related/tests/test_phase_velocity.py`

- [ ] **Step 1: Extend the cached reference solution with one compact-tail solve**

Keep `_tc_reference_solution()` as the shared expensive solve. Require its final result to expose `s_end`, `compact_tail_e_folds`, `tail_state_residual_norm`, `tail_muK_residual_norm`, `tail_q_residual_norm`, and independent ODE residuals.

- [ ] **Step 2: Add a failing compact-coordinate endpoint test**

Add this test below `_tc_reference_solution()`:

```python
def test_thermal_conducting_uses_requested_compact_endpoint():
    r = _tc_reference_solution()
    if not r.get("success"):
        raise AssertionError(f"reference thermal solve failed: {r.get('message')!r}")
    expected_s_end = 1.0 - float(r["tail_eps"])
    if not np.isclose(float(r["s_end"]), expected_s_end, rtol=0.0, atol=1.0e-14):
        raise AssertionError(f"s_end={r['s_end']!r}, expected {expected_s_end!r}")
    if not np.isclose(float(r["s"][-1]), expected_s_end, rtol=0.0, atol=1.0e-14):
        raise AssertionError("returned profile does not reach the solved compact endpoint")
    expected_efolds = -np.log(float(r["tail_eps"]))
    if abs(float(r["compact_tail_e_folds"]) - expected_efolds) > 1.0e-10:
        raise AssertionError(
            f"compact tail has {r['compact_tail_e_folds']!r} e-folds, "
            f"expected {expected_efolds!r}"
        )
    if r.get("coordinate") != "s=1-exp(-lambda_compact*x)":
        raise AssertionError(f"unexpected coordinate metadata: {r.get('coordinate')!r}")
```

- [ ] **Step 3: Add a failing downstream-equilibrium test**

```python
def test_thermal_conducting_reaches_downstream_equilibrium():
    r = _tc_reference_solution()
    if not r.get("success"):
        raise AssertionError(f"reference thermal solve failed: {r.get('message')!r}")
    limit = max(10.0 * float(r["tol_bvp"]), 100.0 * float(r["tail_eps"]))
    for key in (
        "tail_state_residual_norm",
        "tail_muK_residual_norm",
        "tail_q_residual_norm",
    ):
        value = float(r[key])
        if not np.isfinite(value) or value > limit:
            raise AssertionError(f"{key}={value!r} exceeds compact-tail limit {limit!r}")
    if abs(float(r["muK_profile"][-1])) > limit * max(abs(float(r["muB_Q"])), 1.0):
        raise AssertionError("downstream muK is not close to zero")
    if abs(float(r["q_th_profile"][-1])) > limit * max(abs(float(r["F_E"])), 1.0):
        raise AssertionError("downstream conductive heat flux is not close to zero")
```

- [ ] **Step 4: Add a failing independent-ODE-residual test**

```python
def test_thermal_conducting_validates_spline_derivatives_against_rhs():
    r = _tc_reference_solution()
    if not r.get("success"):
        raise AssertionError(f"reference thermal solve failed: {r.get('message')!r}")
    limit = 10.0 * float(r["tol_bvp"])
    for key in (
        "nK_ode_residual_norm",
        "jK_ode_residual_norm",
        "T_ode_residual_norm",
    ):
        value = float(r[key])
        if not np.isfinite(value) or value > limit:
            raise AssertionError(f"{key}={value!r} exceeds {limit!r}")
```

- [ ] **Step 5: Run the new tests and confirm that the old implementation fails for the intended reason**

Run:

```text
MPLCONFIGDIR=/private/tmp/mpl-cache ./bin/python -c "import runpy; d=runpy.run_path('tests/test_phase_velocity.py'); d['test_thermal_conducting_uses_requested_compact_endpoint']()"
```

Expected before implementation: failure because the current result either lacks `s_end` or ends near `s=0.65`, not `1-tail_eps`.

Run:

```text
MPLCONFIGDIR=/private/tmp/mpl-cache ./bin/python -c "import runpy; d=runpy.run_path('tests/test_phase_velocity.py'); d['test_thermal_conducting_reaches_downstream_equilibrium']()"
```

Expected before implementation: failure because the reference endpoint has `muK` near `100 MeV` and a large chemical-state deviation.

### Task 2: Refactor the downstream state so scaling is fixed and compactification can depend on `jB`

**Files:**
- Modify: `/Users/janshao/NSM_related/RMFsolver/phase_velocity.py:5550-5658`
- Test: `/Users/janshao/NSM_related/tests/test_phase_velocity.py`

- [ ] **Step 1: Remove the finite-domain growth-budget state**

Delete `_THERMAL_GROWTH_BUDGET`, `L_tail`, `L_growth`, `L_domain`, `domain_limited_by`, and `x_end` from the thermal solve path. Retain `lambda_growing_max` only as a stiffness diagnostic.

- [ ] **Step 2: Split physical state construction from BVP scaling**

Refactor the nested state builder into this sequence:

```python
physical_state_cache = {}

def build_physical_state(theta):
    key = round(float(theta), 12)
    if key in physical_state_cache:
        return physical_state_cache[key]
    stats["global_state_builds"] += 1
    jB = jB_from_param(theta)
    info = _thermal_downstream_analysis(
        jB,
        nuclear_state,
        B_one_forth,
        ms=ms,
        upB=upB,
        kappa_model=kappa_model,
        endpoint_guess=endpoint_guess_cache["value"],
    )
    endpoint_guess_cache["value"] = info["endpoint"].get(
        "endpoint_initial_guess", (info["muB_Q"], info["T_Q"])
    )
    if int(info["stable_dimension"]) != 1:
        raise RuntimeError(
            "Thermal downstream stable manifold has dimension "
            f"{info['stable_dimension']}, expected 1"
        )
    lambda_compact = float(info["lambda_stable"])
    if not np.isfinite(lambda_compact) or lambda_compact <= 0.0:
        raise RuntimeError("Thermal compactification rate must be positive")
    info = dict(info)
    info["lambda_compact"] = lambda_compact
    info["lambda_growing_max"] = float(np.max(info["eigenvalues"].real))
    physical_state_cache[key] = info
    return info
```

- [ ] **Step 3: Freeze the BVP field scales at the starting parameter**

After constructing `theta0` and `physical_state0`, define scales once per BVP solve:

```python
nB_Q0 = float(physical_state0["thermo_Q"]["nB"])
bvp_scales = np.array(
    [
        max(abs(float(physical_state0["nK_Q"])), nB_Q0, 1.0),
        max(abs(float(physical_state0["jB"])), abs(float(physical_state0["jK_Q"])), 1.0),
        max(float(physical_state0["T_Q"]), 1.0),
    ],
    dtype=float,
)
```

Do not change the meaning of a scaled collocation variable when Newton changes `p`. This follows the fixed `bvp_nK_scale`/`bvp_jK_scale` design in the established solver.

- [ ] **Step 4: Complete each trial state using the fixed scales**

```python
state_cache = {}

def build_state(theta):
    key = round(float(theta), 12)
    if key in state_cache:
        return state_cache[key]
    info = dict(build_physical_state(theta))
    projections, lambda_stable = _thermal_scaled_growing_projections(
        info["jacobian"], bvp_scales
    )
    info["scales"] = bvp_scales
    info["growing_projections"] = projections
    info["lambda_compact"] = float(lambda_stable)
    state_cache[key] = info
    return info
```

- [ ] **Step 5: Add a fixed-scaling regression test**

Add `bvp_scales` to the result and add:

```python
def test_thermal_conducting_reports_fixed_positive_bvp_scales():
    r = _tc_reference_solution()
    if not r.get("success"):
        raise AssertionError(f"reference thermal solve failed: {r.get('message')!r}")
    scales = np.asarray(r["bvp_scales"], dtype=float)
    if scales.shape != (3,):
        raise AssertionError(f"expected three BVP scales, got shape {scales.shape!r}")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise AssertionError(f"BVP scales must be positive and finite: {scales!r}")
```

The implementation audit for this step must also confirm that `bvp_scales` is assigned once after `physical_state0` and is never reassigned inside `build_state`, `ode`, or `bc`.

### Task 3: Replace the finite-`x` BVP with the compact-`s` BVP

**Files:**
- Modify: `/Users/janshao/NSM_related/RMFsolver/phase_velocity.py:5703-5808`
- Test: `/Users/janshao/NSM_related/tests/test_phase_velocity.py`

- [ ] **Step 1: Add a reusable compact mesh helper**

Add near the thermal constants:

```python
def _thermal_compact_mesh(n_mesh, tail_eps):
    n_mesh = int(n_mesh)
    tail_eps = float(tail_eps)
    if n_mesh < 5:
        raise RuntimeError("thermal compact mesh requires at least five points")
    if not (0.0 < tail_eps < 1.0):
        raise RuntimeError("thermal compact mesh requires 0 < tail_eps < 1")
    s_end = 1.0 - tail_eps
    if s_end <= 0.9:
        return np.linspace(0.0, s_end, n_mesh)
    n_tail = max(12, n_mesh // 3)
    n_core = n_mesh - n_tail + 1
    core_t = np.linspace(0.0, 1.0, n_core)
    core = 0.9 * np.expm1(4.0 * core_t) / np.expm1(4.0)
    tail = 1.0 - np.geomspace(0.1, tail_eps, n_tail)
    mesh = np.unique(np.concatenate((core, tail)))
    mesh[0] = 0.0
    mesh[-1] = s_end
    if np.any(np.diff(mesh) <= 0.0):
        raise RuntimeError("thermal compact mesh is not strictly increasing")
    return mesh
```

- [ ] **Step 2: Test the compact mesh independently**

```python
def test_thermal_compact_mesh_has_exact_endpoints_and_monotone_tail():
    for tail_eps in (1.0e-3, 1.0e-6, 1.0e-8):
        mesh = pv._thermal_compact_mesh(200, tail_eps)
        if mesh[0] != 0.0 or mesh[-1] != 1.0 - tail_eps:
            raise AssertionError(f"wrong compact endpoints for tail_eps={tail_eps:g}")
        if np.any(np.diff(mesh) <= 0.0):
            raise AssertionError("compact mesh is not strictly increasing")
        if not np.any(mesh > 1.0 - 10.0 * tail_eps):
            raise AssertionError("compact mesh does not resolve the requested tail")
```

- [ ] **Step 3: Transform the thermal ODE to `s`**

Replace the physical-`x` callback with:

```python
def ode(s_coord, y, p):
    stats["bvp_ode_calls"] += 1
    state = state_or_none(float(p[0]))
    if state is None:
        return np.full_like(y, 1.0e12)
    scales = state["scales"]
    dyds = np.empty_like(y)
    guess = (state["muB_Q"], 1.0e-3)
    for i in range(y.shape[1]):
        try:
            nK_value = float(y[0, i]) * scales[0]
            jK_value = float(y[1, i]) * scales[1]
            T_value = float(y[2, i]) * scales[2]
            thermo = pointwise(nK_value, T_value, state, guess)
            guess = (thermo["muB"], thermo["muK"])
            one_minus_s = max(1.0 - float(s_coord[i]), np.finfo(float).tiny)
            dx_ds = 1.0 / (state["lambda_compact"] * one_minus_s)
            dyds[0, i] = (
                (thermo["u"] * nK_value - jK_value)
                * thermo["invD"]
                * dx_ds
                / scales[0]
            )
            dyds[1, i] = -thermo["Gamma_K"] * dx_ds / scales[1]
            dyds[2, i] = thermo["T_prime"] * dx_ds / scales[2]
        except Exception as exc:
            stats["local_state_rejections"] += 1
            last_failure["message"] = str(exc)
            dyds[:, i] = 1.0e12
    return dyds
```

- [ ] **Step 4: Keep the four boundary residuals but evaluate them at `s_end`**

Use the fixed scales and the trial-`jB` equilibrium target:

```python
def bc(ya, yb, p):
    stats["bvp_bc_calls"] += 1
    state = state_or_none(float(p[0]))
    if state is None:
        return np.full(4, 1.0e12, dtype=float)
    scales = state["scales"]
    delta_scaled = np.array(
        [
            float(yb[0]) - state["nK_Q"] / scales[0],
            float(yb[1]) - state["jK_Q"] / scales[1],
            float(yb[2]) - state["T_Q"] / scales[2],
        ],
        dtype=float,
    )
    projections = state["growing_projections"]
    return np.array(
        [
            float(ya[2]) - T_interface / scales[2],
            float(ya[1]) - state["jB"] / scales[1],
            float(projections[0] @ delta_scaled),
            float(projections[1] @ delta_scaled),
        ],
        dtype=float,
    )
```

- [ ] **Step 5: Pass the compact mesh to `solve_bvp`**

```python
s_end = 1.0 - float(tail_eps)
s_mesh = _thermal_compact_mesh(n_mesh, tail_eps)
sol = solve_bvp(
    ode,
    bc,
    s_mesh,
    y_guess,
    p=np.array([theta0], dtype=float),
    tol=float(tol_bvp),
    max_nodes=int(max_nodes),
    verbose=0,
)
```

The independent variable passed to `solve_bvp` must be named and interpreted as `s`. Do not reconstruct a capped physical `L` before this call.

- [ ] **Step 6: Run the compact-coordinate test**

Expected after this task: `test_thermal_conducting_uses_requested_compact_endpoint` passes, although endpoint and independent-residual tests may still fail until Tasks 4 and 5 are complete.

### Task 4: Make continuation work in both interface temperature and tail depth

**Files:**
- Modify: `/Users/janshao/NSM_related/RMFsolver/phase_velocity.py:5778-5795`
- Modify: `/Users/janshao/NSM_related/RMFsolver/phase_velocity.py:6065-6248`
- Test: `/Users/janshao/NSM_related/tests/test_phase_velocity.py`

- [ ] **Step 1: Store continuation profiles in physical variables on `s`**

Replace the scaled physical-`x` continuation payload with:

```python
"_continuation_state": {
    "s": np.asarray(sol.x, dtype=float),
    "physical_y": np.vstack(
        (
            np.asarray(sol.sol(sol.x)[0], dtype=float) * scales[0],
            np.asarray(sol.sol(sol.x)[1], dtype=float) * scales[1],
            np.asarray(sol.sol(sol.x)[2], dtype=float) * scales[2],
        )
    ),
    "equilibrium_y": np.array(
        [state_final["nK_Q"], state_final["jK_Q"], state_final["T_Q"]],
        dtype=float,
    ),
    "theta": float(sol.p[0]),
    "tail_eps": float(tail_eps),
}
```

- [ ] **Step 2: Add a tail-aware interpolation helper**

Add:

```python
def _thermal_interpolate_profile_guess(profile_guess, s_mesh, equilibrium_y, scales):
    previous_s = np.asarray(profile_guess["s"], dtype=float)
    previous_y = np.asarray(profile_guess["physical_y"], dtype=float)
    previous_eq = np.asarray(profile_guess["equilibrium_y"], dtype=float)
    if previous_y.shape != (3, previous_s.size):
        raise RuntimeError("thermal continuation profile has invalid shape")
    old_end = float(previous_s[-1])
    old_gap = max(1.0 - old_end, np.finfo(float).tiny)
    physical_guess = np.empty((3, s_mesh.size), dtype=float)
    for row in range(3):
        old_delta = previous_y[row] - previous_eq[row]
        inside = s_mesh <= old_end
        physical_guess[row, inside] = equilibrium_y[row] + np.interp(
            s_mesh[inside], previous_s, old_delta
        )
        outside = ~inside
        physical_guess[row, outside] = equilibrium_y[row] + old_delta[-1] * (
            (1.0 - s_mesh[outside]) / old_gap
        )
    return physical_guess / scales[:, None]
```

This extends a coarse-tail solution using the known stable asymptotic dependence `delta_y proportional to 1-s`; it must not hold the old endpoint value constant across the newly added tail.

- [ ] **Step 3: Enforce the new left boundary in the interpolated initial guess**

After interpolation, smoothly correct the two left-controlled fields:

```python
left_shape = (1.0 - s_mesh) / max(1.0 - s_mesh[0], np.finfo(float).tiny)
y_guess[1] += (state0["jB"] / scales[1] - y_guess[1, 0]) * left_shape
y_guess[2] += (T_interface / scales[2] - y_guess[2, 0]) * left_shape
y_guess[1, 0] = state0["jB"] / scales[1]
y_guess[2, 0] = T_interface / scales[2]
```

- [ ] **Step 4: Add a deterministic tail-depth schedule**

```python
def _thermal_tail_schedule(tail_eps):
    tail_eps = float(tail_eps)
    if not (0.0 < tail_eps < 1.0):
        raise RuntimeError("thermal tail schedule requires 0 < tail_eps < 1")
    first = max(tail_eps, 1.0e-4)
    schedule = [first]
    while schedule[-1] > tail_eps:
        next_eps = max(tail_eps, schedule[-1] * 1.0e-2)
        if next_eps == schedule[-1]:
            break
        schedule.append(float(next_eps))
    return schedule
```

For the default target, this produces `[1e-4, 1e-6, 1e-8]`.

- [ ] **Step 5: Use two-stage continuation in the public driver**

Implement this order:

1. At the first requested `T_interface` waypoint, solve the compact BVP successively over `_thermal_tail_schedule(tail_eps)`.
2. Use the deepest converged profile for all subsequent temperature-continuation waypoints.
3. Keep the existing adaptive midpoint insertion for failed temperature decrements.
4. If a tail-depth step fails, insert the geometric midpoint `sqrt(last_good_eps*target_eps)` and retry, capped by `_THERMAL_MAX_CONTINUATION_ATTEMPTS`.
5. Return success only if both the exact requested `T_interface` and exact requested `tail_eps` were reached.

Record every attempt:

```python
tail_history.append(
    {
        "tail_eps": float(eps_step),
        "success": bool(result.get("success", False)),
        "uN": result.get("uN", np.nan),
        "jB": result.get("jB", np.nan),
        "aQstar": result.get("aQstar", np.nan),
        "tail_state_residual_norm": result.get("tail_state_residual_norm", np.nan),
        "message": result.get("message", ""),
    }
)
```

- [ ] **Step 6: Add a one-run tail-convergence test using the recorded history**

```python
def test_thermal_tail_continuation_converges_physical_outputs():
    r = _tc_reference_solution()
    successful = [entry for entry in r["tail_continuation_history"] if entry["success"]]
    if len(successful) < 2:
        raise AssertionError("expected at least two successful compact-tail depths")
    coarse, fine = successful[-2], successful[-1]
    if abs(float(fine["uN"]) / float(coarse["uN"]) - 1.0) > 1.0e-3:
        raise AssertionError(f"uN is not tail-converged: {coarse!r} -> {fine!r}")
    if abs(float(fine["aQstar"]) - float(coarse["aQstar"])) > 1.0e-3:
        raise AssertionError(f"aQstar is not tail-converged: {coarse!r} -> {fine!r}")
```

### Task 5: Reconstruct from the compact spline and make validation independent

**Files:**
- Modify: `/Users/janshao/NSM_related/RMFsolver/phase_velocity.py:5839-6009`
- Test: `/Users/janshao/NSM_related/tests/test_phase_velocity.py`

- [ ] **Step 1: Reconstruct on a dense compact grid**

Use a dense grid that includes both the interface and logarithmic tail:

```python
s_dense = _thermal_compact_mesh(max(600, 3 * int(n_mesh)), tail_eps)
y_dense = sol.sol(s_dense)
lambda_compact = float(state_final["lambda_compact"])
x_dense = -np.log1p(-s_dense) / lambda_compact
```

Return:

```python
coordinate = "s=1-exp(-lambda_compact*x)"
tail_eps = float(tail_eps)
tol_bvp = float(tol_bvp)
s_end = 1.0 - tail_eps
compact_tail_e_folds = -np.log(tail_eps)
x_end = -np.log(tail_eps) / lambda_compact
L_domain = x_end
domain_limited_by = "tail_eps"
bvp_scales = np.asarray(scales, dtype=float)
nK_Q = float(state_final["nK_Q"])
jK_Q = float(state_final["jK_Q"])
```

Keep `L_domain` as a backward-compatible alias, but it must now mean the physical image of the requested compact endpoint.
Include `tail_eps`, `tol_bvp`, `s_end`, and `bvp_scales` in both successful and failed result dictionaries so diagnostic callers never have to infer the numerical problem that was attempted.

- [ ] **Step 2: Remove temperature clipping during reconstruction**

Replace:

```python
pointwise(nK_value, max(T_value, 0.0), state_final, guess)
```

with:

```python
if T_value < 0.0:
    raise RuntimeError(f"negative reconstructed temperature T={T_value:.6e}")
pointwise(nK_value, T_value, state_final, guess)
```

Any negative dense-output temperature must cause reconstruction failure and `success=False`.

- [ ] **Step 3: Evaluate physical spline derivatives**

```python
dy_ds_scaled = _bvp_dense_derivative(sol, s_dense)
dy_ds_physical = dy_ds_scaled * scales[:, None]
dx_ds = 1.0 / (
    lambda_compact * np.maximum(1.0 - s_dense, np.finfo(float).tiny)
)
dy_dx_physical = dy_ds_physical / dx_ds[None, :]
dnK_dx_spline = dy_dx_physical[0]
djK_dx_spline = dy_dx_physical[1]
dT_dx_spline = dy_dx_physical[2]
```

- [ ] **Step 4: Build independent ODE residuals**

During pointwise reconstruction, store the physical RHS arrays:

```python
rhs_nK[i] = (th["u"] * nK_value - jK_value) * th["invD"]
rhs_jK[i] = -th["Gamma_K"]
rhs_T[i] = (th["E_flux"] - state_final["F_E"]) / th["kappa_th"]
```

After reconstruction:

```python
def mean_relative_residual(lhs, rhs):
    finite = np.isfinite(lhs) & np.isfinite(rhs)
    if not np.any(finite):
        return np.nan
    scale = max(float(np.mean(np.abs(rhs[finite]))), _FLOAT_TINY)
    return float(np.mean(np.abs(lhs[finite] - rhs[finite])) / scale)

nK_ode_residual_norm = mean_relative_residual(dnK_dx_spline, rhs_nK)
jK_ode_residual_norm = mean_relative_residual(djK_dx_spline, rhs_jK)
T_ode_residual_norm = mean_relative_residual(dT_dx_spline, rhs_T)
```

- [ ] **Step 5: Recompute heat flux from the spline derivative**

Do not use the ODE-defined `T_prime` to validate energy conservation. Use:

```python
q_th_spline = -fields["kappa_th"] * dT_dx_spline
flux_E_spline = fields["E_advective"] + q_th_spline
```

where `E_advective=h*gamma*u` is stored during reconstruction. Report the previous algebraic `q_th=F_E-E_advective` separately as `q_th_closure_profile`, and add:

```python
q_th_consistency_norm = mean_relative_residual(q_th_spline, fields["q_th_closure"])
```

The public `q_th_profile` and `T_prime_profile` should refer to spline-derived quantities because those are what the numerical solution actually differentiates.

- [ ] **Step 6: Add explicit downstream-distance diagnostics**

```python
equilibrium_y = np.array(
    [state_final["nK_Q"], state_final["jK_Q"], state_final["T_Q"]],
    dtype=float,
)
tail_delta_scaled = (
    np.array([fields["nK"][-1], fields["jK"][-1], fields["T"][-1]])
    - equilibrium_y
) / scales
tail_state_residual_norm = float(np.max(np.abs(tail_delta_scaled)))
tail_muK_residual_norm = float(
    abs(fields["muK"][-1]) / max(abs(state_final["muB_Q"]), 1.0)
)
tail_q_residual_norm = float(
    abs(q_th_spline[-1]) / max(abs(state_final["F_E"]), 1.0)
)
```

- [ ] **Step 7: Strengthen the success gate**

Use:

```python
ode_limit = 10.0 * float(tol_bvp)
tail_limit = max(10.0 * float(tol_bvp), 100.0 * float(tail_eps))
success = bool(
    sol.status == 0
    and reconstruction_failures == 0
    and np.max(np.abs(bc_residual)) < 1.0e-4
    and closure_max < 1.0e-7
    and max(nK_ode_residual_norm, jK_ode_residual_norm, T_ode_residual_norm) <= ode_limit
    and q_th_consistency_norm <= ode_limit
    and tail_state_residual_norm <= tail_limit
    and tail_muK_residual_norm <= tail_limit
    and tail_q_residual_norm <= tail_limit
)
```

If any tail check fails, the message must say that the compact BVP reached `s_end` but did not enter the downstream asymptotic regime. Never return `success=True` solely because the two eigenvector projections are small.

- [ ] **Step 8: Run the three Task 1 regression tests**

Expected: all pass, and the reference endpoint has `muK`, `q_th`, `nK`, and `jK` close to equilibrium within the stated normalized tolerance.

### Task 6: Verify convergence, branch selection, and regression safety

**Files:**
- Modify: `/Users/janshao/NSM_related/tests/test_phase_velocity.py:2269-2301`
- Test: `/Users/janshao/NSM_related/tests/test_phase_velocity.py`

- [ ] **Step 1: Retain and run interface-seed independence**

Keep `test_thermal_conducting_is_independent_of_the_interface_composition_seed`. Extend it to compare:

```python
for key in ("tail_state_residual_norm", "tail_muK_residual_norm", "tail_q_residual_norm"):
    if float(other[key]) > max(10.0 * float(other["tol_bvp"]), 100.0 * float(other["tail_eps"])):
        raise AssertionError(f"seed-variant solve has unresolved tail: {key}={other[key]!r}")
```

- [ ] **Step 2: Add mesh robustness**

```python
def test_thermal_conducting_is_mesh_converged():
    base = _tc_reference_solution()
    refined = pv.solve_front_thermal_conducting(
        10.0,
        3.0 * N0,
        180.0,
        T_interface=0.0,
        tail_eps=float(base["tail_eps"]),
        n_mesh=300,
        tol_bvp=5.0e-5,
    )
    if not refined.get("success"):
        raise AssertionError(f"refined thermal solve failed: {refined.get('message')!r}")
    if abs(float(refined["uN"]) / float(base["uN"]) - 1.0) > 2.0e-3:
        raise AssertionError(f"uN is not mesh-converged: {base['uN']!r} vs {refined['uN']!r}")
    if abs(float(refined["aQstar"]) - float(base["aQstar"])) > 2.0e-3:
        raise AssertionError(
            f"aQstar is not mesh-converged: {base['aQstar']!r} vs {refined['aQstar']!r}"
        )
```

- [ ] **Step 3: Retain the exact-zero local power-law test**

Run `test_thermal_conducting_near_interface_power_laws` after compactification. The physical `x` used for fitting must be reconstructed from `s`; do not fit against `s` directly.

- [ ] **Step 4: Run the focused thermal checks**

Run:

```text
MPLCONFIGDIR=/private/tmp/mpl-cache ./bin/python -c "import runpy; d=runpy.run_path('tests/test_phase_velocity.py'); names=[n for n in d if n.startswith('test_thermal_')]; [(d[n](), print(n+': ok')) for n in sorted(names)]"
```

Expected: every thermal test prints `: ok`. Record total runtime because the compact tail and continuation may make the reference solve substantially slower.

- [ ] **Step 5: Run the complete function-style test suite**

Run:

```text
MPLCONFIGDIR=/private/tmp/mpl-cache ./bin/python tests/test_phase_velocity.py
```

Expected: every listed test prints `: ok`; no pre-existing solver regression.

- [ ] **Step 6: Produce a numerical completion table**

Report, for the reference case and the refined-mesh case:

```text
success
uN
jB
aQstar
Tprime_Qstar
q_th_Qstar
tail_eps
s_end
compact_tail_e_folds
x_end
muK_end
nK_end/nB_Q
jK_end/jB
q_th_end/F_E
tail_state_residual_norm
tail_muK_residual_norm
tail_q_residual_norm
nK_ode_residual_norm
jK_ode_residual_norm
T_ode_residual_norm
q_th_consistency_norm
bvp_status
bvp_iterations
bvp_nodes
```

Do not state that the physical front is solved unless the endpoint, independent ODE residual, seed, mesh, temperature-continuation, and tail-depth-continuation checks all pass.

## Completion criteria

The repair is complete only when all of the following are true:

1. `solve_bvp` receives `s_mesh`, not a finite physical-`x` mesh.
2. Every RHS contains the exact `dx/ds = 1/(lambda_compact*(1-s))` factor.
3. The final solved endpoint is exactly `s_end=1-tail_eps`.
4. No growth-budget cap shortens the requested compact tail.
5. `jB` remains the single BVP parameter and all four boundary residuals remain independent.
6. The endpoint is independently close to `muK=0`, `q_th=0`, `nK=nK_Q`, and `jK=jK_Q`.
7. Dense spline derivatives satisfy all three physical ODEs.
8. No negative-temperature clipping occurs in solve or reconstruction.
9. `uN` and `aQstar` are stable under tail refinement, mesh refinement, and interface-composition seed changes.
10. Existing non-thermal solver tests remain unchanged and pass.
