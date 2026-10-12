# Isothermal Relativistic Junction Conditions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the momentum-flux approximation in the isothermal front solvers measurable and guarded, then optionally exact, without perturbing the energy-conserving or thermal-conducting solvers.

**Architecture:** The isothermal path currently uses the `gamma -> 1` junction conditions `jB = nB*u` and `Pi = P + w*u^2`. Tasks 1-2 expose the size of that approximation as reported diagnostics and refuse states where it exceeds a declared tolerance. Task 3 adds the exact relativistic forms `jB = nB*gamma*u` and `Pi = P + w*gamma^2*u^2` behind an opt-in flag threaded only through the isothermal call chain, leaving the shared default behaviour byte-identical.

**Tech Stack:** Python 3.12 venv at the project root (`./bin/python3`), numpy, scipy, `unittest` (pytest is not installed).

**Spec:** No separate spec file — the design was settled in session and is restated in "Design Basis" below. Physics context: `wiki/physics/isothermal-analytic-front-speed.md`, `wiki/physics/interface-closure.md`.

## Design Basis

For a planar steady front in its rest frame, with `v` the fluid 3-velocity,
`gamma = 1/sqrt(1 - v^2)`, `w = e + P`, the exact junction conditions are:

```text
baryon    : nB(0-)*gamma_-*v_-           = nB(0+)*gamma_+*v_+        = jB
momentum  : w(0-)*gamma_-^2*v_-^2 + P(0-) = w(0+)*gamma_+^2*v_+^2 + P(0+) = Pi
```

Energy flux is deliberately not imposed; isothermality replaces it. The code
currently implements the `gamma -> 1` limit of both.

Measured 2026-08-19 across three states, sweeping `1 - a_0plus` over eight
decades:

| T (MeV) | nB/n0 | w/P | `1-a` where `(Pi-P)/P = 10%` | `gamma-1` there |
|---:|---:|---:|---:|---:|
| 1 | 3.5 | 5.74 | 1e-9 | 0.019 |
| 40 | 3.0 | 6.80 | 1e-11 | 0.0075 |
| 80 | 2.0 | 8.74 | 3e-12 | 0.039 |

`(Pi-P)/P` scales as `1/(1-a_0plus)` — one decade per decade, clean over the
whole sweep. Two consequences drive this plan:

1. At realized compositions (`a_0plus_max` spans 0.01-0.99 across the domain)
   the correction is below `1e-6`, i.e. under every tolerance in the code
   (`tol_bvp = 1e-4`, ray `rho` tolerance `1e-4`, momentum-flux residual gate
   `1e-8`). Task 3 will not move any published number.
2. The flux term `w*u^2` dominates the `gamma` factors by roughly a decade, so
   a partial fix that adds `gamma` without the flux term would be backwards.

## Global Constraints

- **Only `RMFsolver/phase_velocity.py` may be modified inside the package.** `CLAUDE.md` forbids editing `Solver.py`, `SQMsolver.py`, `constants.py`, `RMFparameter.py`, `tov.py`, `tov_solve.py` without explicit instruction. Reading and importing them is fine.
- **Default behaviour must not change.** `_Pi_QM_state` is shared by the energy-conserving, isothermal, and thermal-conducting solvers (15+ `u = jB/nB` sites). Every new relativistic path is opt-in and defaults off.
- **Endpoint notation is location-based:** `_0minus`, `_0plus`, `_inf`. Never introduce `_N`, `_Qstar`, `_Q`.
- **Interpreter:** `./bin/python3` from the project root. Tests: `./bin/python3 -m unittest ...`.
- **Never read `.ipynb` files directly** — use `.claude/tools/nbsrc.py`.
- **Do not touch** `tests/test_phase_velocity_isothermal_numerical.py` — it is another session's untracked work.
- Isothermal solvers require `NM_type="PNM"` and `ms=0.0`.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `RMFsolver/phase_velocity.py` | All solver code | Modify — add `_momentum_flux_diagnostics`, `_relativistic_flux_pair`, thread `relativistic` flag |
| `tests/test_phase_velocity_isothermal_relativistic.py` | Tests for this feature only | Create |
| `wiki/physics/isothermal-analytic-front-speed.md` | Physics conclusions | Modify — record the validity bound |

---

### Task 1: Momentum-flux diagnostics on the isothermal results

Report how large the approximation is, at every solve, so it is never silently applied.

**Files:**
- Modify: `RMFsolver/phase_velocity.py` (add `_momentum_flux_diagnostics`; call it in `analytic_velocity_isothermal` and `solve_front_isothermal` result dicts)
- Test: `tests/test_phase_velocity_isothermal_relativistic.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_momentum_flux_diagnostics(P_0minus, w_0minus, u_0minus) -> dict` with float keys `momentum_flux_ratio`, `gamma_minus_1`, `relativistic_flux_ratio`. Result dicts of both isothermal solvers gain those three keys.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phase_velocity_isothermal_relativistic.py
import unittest

import numpy as np

from RMFsolver import phase_velocity as pv


class TestMomentumFluxDiagnostics(unittest.TestCase):
    def test_ratios_match_closed_form(self):
        P, w, u = 1.0e9, 5.74e9, 5.8e-3
        d = pv._momentum_flux_diagnostics(P, w, u)
        gamma_squared = 1.0 / (1.0 - u * u)
        self.assertAlmostEqual(d["momentum_flux_ratio"], w * u * u / P, places=12)
        self.assertAlmostEqual(
            d["relativistic_flux_ratio"], w * gamma_squared * u * u / P, places=12
        )
        self.assertAlmostEqual(
            d["gamma_minus_1"], np.sqrt(gamma_squared) - 1.0, places=12
        )

    def test_static_limit_is_zero(self):
        d = pv._momentum_flux_diagnostics(1.0e9, 5.0e9, 0.0)
        self.assertEqual(d["momentum_flux_ratio"], 0.0)
        self.assertEqual(d["gamma_minus_1"], 0.0)

    def test_non_finite_velocity_gives_nan_not_raise(self):
        d = pv._momentum_flux_diagnostics(1.0e9, 5.0e9, np.nan)
        self.assertTrue(np.isnan(d["momentum_flux_ratio"]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./bin/python3 -m unittest tests.test_phase_velocity_isothermal_relativistic.TestMomentumFluxDiagnostics -v`
Expected: FAIL with `AttributeError: module 'RMFsolver.phase_velocity' has no attribute '_momentum_flux_diagnostics'`

- [ ] **Step 3: Write minimal implementation**

Insert immediately above `def _solve_a_0plus_max(` in `RMFsolver/phase_velocity.py`:

```python
def _momentum_flux_diagnostics(P_0minus, w_0minus, u_0minus):
    """Report how far the front departs from the static-pressure limit.

    The isothermal solvers impose the gamma -> 1 junction conditions,
    jB = nB*u and Pi = P + w*u**2.  ``momentum_flux_ratio`` is the size of
    the flux term relative to the pressure, (Pi - P)/P, and is the leading
    error of treating the interface as a static isobar.
    ``relativistic_flux_ratio`` is the same quantity with the exact
    gamma**2 factor, so their difference isolates the relativistic part.
    """
    P_0minus = float(P_0minus)
    w_0minus = float(w_0minus)
    u_0minus = float(u_0minus)
    if (
        (not np.isfinite(P_0minus))
        or (not np.isfinite(w_0minus))
        or (not np.isfinite(u_0minus))
        or P_0minus <= 0.0
        or not (0.0 <= u_0minus < 1.0)
    ):
        return {
            "momentum_flux_ratio": np.nan,
            "relativistic_flux_ratio": np.nan,
            "gamma_minus_1": np.nan,
        }
    gamma_squared = 1.0 / (1.0 - u_0minus * u_0minus)
    flux = w_0minus * u_0minus * u_0minus / P_0minus
    return {
        "momentum_flux_ratio": float(flux),
        "relativistic_flux_ratio": float(flux * gamma_squared),
        "gamma_minus_1": float(np.sqrt(gamma_squared) - 1.0),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./bin/python3 -m unittest tests.test_phase_velocity_isothermal_relativistic.TestMomentumFluxDiagnostics -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Add the keys to `analytic_velocity_isothermal`**

In the `result.update({...})` block guarded by `"status": "moving_front"` (search for `"slow_front_consistent": bool(u_0minus < 1.0),`), add immediately before that line:

```python
            **_momentum_flux_diagnostics(
                P_0minus,
                float(nuclear_state["h_0minus"]),
                u_0minus,
            ),
```

- [ ] **Step 6: Add the keys to `solve_front_isothermal`**

In the main `result = {` dict (the one containing `"a_0plus_derived":`), add immediately after the `"a_0plus_derived"` line:

```python
        **_momentum_flux_diagnostics(
            float(nuclear_state["P_0minus"]),
            float(nuclear_state["h_0minus"]),
            float(jB / nB_0minus),
        ),
```

- [ ] **Step 7: Write the integration test**

Append to `tests/test_phase_velocity_isothermal_relativistic.py`:

```python
from RMFsolver.constants import MeV_fm

N0 = 0.16 * MeV_fm**3
B14 = 189.1566


class TestDiagnosticsReachTheResults(unittest.TestCase):
    def test_analytic_reports_a_small_ratio_at_realistic_composition(self):
        r = pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, 0.5, xi=-0.5)
        self.assertEqual(r["status"], "moving_front")
        self.assertLess(r["momentum_flux_ratio"], 1.0e-6)
        self.assertLess(r["gamma_minus_1"], 1.0e-6)

    def test_ratio_grows_as_composition_approaches_one(self):
        near = pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, 1.0 - 1e-8, xi=-0.5)
        far = pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, 1.0 - 1e-4, xi=-0.5)
        self.assertGreater(near["momentum_flux_ratio"], far["momentum_flux_ratio"])
        # (Pi-P)/P scales as 1/(1-a): four decades of a give four of ratio.
        self.assertAlmostEqual(
            np.log10(near["momentum_flux_ratio"] / far["momentum_flux_ratio"]),
            4.0,
            delta=0.3,
        )
```

- [ ] **Step 8: Run the integration test**

Run: `./bin/python3 -m unittest tests.test_phase_velocity_isothermal_relativistic -v`
Expected: PASS, 5 tests. Runtime ~4 minutes (each analytic solve does a branch-validated density inversion).

- [ ] **Step 9: Commit**

```bash
git add RMFsolver/phase_velocity.py tests/test_phase_velocity_isothermal_relativistic.py
git commit -m "[v4.5] report isothermal momentum-flux approximation size"
```

---

### Task 2: Validity guard on the static-isobar approximation

Refuse, rather than silently return, when the approximation is worse than a declared tolerance.

**Files:**
- Modify: `RMFsolver/phase_velocity.py` (`_solve_a_0plus_max`, both isothermal solvers)
- Test: `tests/test_phase_velocity_isothermal_relativistic.py`

**Interfaces:**
- Consumes: `_momentum_flux_diagnostics` from Task 1.
- Produces: module constant `MOMENTUM_FLUX_RATIO_TOLERANCE = 1.0e-3`. Both isothermal solvers accept `momentum_flux_tol=MOMENTUM_FLUX_RATIO_TOLERANCE`; exceeding it sets `status="static_isobar_approximation_invalid"` (analytic) or raises `RuntimeError` (numerical).

- [ ] **Step 1: Write the failing test**

```python
class TestValidityGuard(unittest.TestCase):
    def test_tolerance_constant_exists_and_is_conservative(self):
        self.assertLessEqual(pv.MOMENTUM_FLUX_RATIO_TOLERANCE, 1.0e-3)

    def test_analytic_flags_when_the_flux_term_is_large(self):
        # 1-a = 1e-9 measured at (Pi-P)/P ~ 0.2 for this state.
        r = pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, 1.0 - 1e-9, xi=-0.5)
        self.assertFalse(r["success"])
        self.assertEqual(r["status"], "static_isobar_approximation_invalid")
        self.assertGreater(r["momentum_flux_ratio"], pv.MOMENTUM_FLUX_RATIO_TOLERANCE)

    def test_realistic_composition_is_unaffected(self):
        r = pv.analytic_velocity_isothermal(1.0, 3.5 * N0, B14, 0.5, xi=-0.5)
        self.assertTrue(r["success"])
        self.assertEqual(r["status"], "moving_front")

    def test_guard_can_be_relaxed_by_the_caller(self):
        r = pv.analytic_velocity_isothermal(
            1.0, 3.5 * N0, B14, 1.0 - 1e-9, xi=-0.5, momentum_flux_tol=1.0
        )
        self.assertEqual(r["status"], "moving_front")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./bin/python3 -m unittest tests.test_phase_velocity_isothermal_relativistic.TestValidityGuard -v`
Expected: FAIL with `AttributeError: ... 'MOMENTUM_FLUX_RATIO_TOLERANCE'`

- [ ] **Step 3: Add the constant**

Near the top of `RMFsolver/phase_velocity.py`, after the imports:

```python
# Above this fraction the static-isobar reduction used by the isothermal
# interface construction is no longer trustworthy: (Pi - P)/P = w*u**2/P is the
# leading error of treating the interface as a static isobar.  Measured
# 2026-08-19, it reaches 10% only at 1 - a_0plus ~ 1e-9 to 1e-12, far outside
# the realized range a_0plus_max in (0.01, 0.99), so this bound never binds in
# practice; it exists so the failure is loud if it ever does.
MOMENTUM_FLUX_RATIO_TOLERANCE = 1.0e-3
```

- [ ] **Step 4: Add the keyword and guard to `analytic_velocity_isothermal`**

Add `momentum_flux_tol=MOMENTUM_FLUX_RATIO_TOLERANCE` to the keyword-only parameters. Then, immediately after the `result.update({...})` that sets `"status": "moving_front"`, append:

```python
    if result["momentum_flux_ratio"] > float(momentum_flux_tol):
        result.update(
            {
                "success": False,
                "status": "static_isobar_approximation_invalid",
                "message": (
                    "The momentum-flux term is "
                    f"{result['momentum_flux_ratio']:.3e} of the upstream "
                    "pressure, above momentum_flux_tol="
                    f"{float(momentum_flux_tol):.3e}; the static-isobar "
                    "interface reduction is not trustworthy here"
                ),
                "front_exists": False,
            }
        )
    return result
```

Delete the now-duplicated bare `return result` that previously ended the function.

- [ ] **Step 5: Run test to verify it passes**

Run: `./bin/python3 -m unittest tests.test_phase_velocity_isothermal_relativistic.TestValidityGuard -v`
Expected: PASS, 4 tests

- [ ] **Step 6: Confirm no regression in the a_0plus_max tests**

Run: `./bin/python3 -m unittest tests.test_phase_velocity_isothermal_a_0plus_max -v`
Expected: PASS, 11 tests. If `test_auto_matches_explicitly_passing_the_ceiling` now fails, the guard is firing at realized `a_0plus_max` — stop and report; that would contradict the 2026-08-19 measurement.

- [ ] **Step 7: Commit**

```bash
git add RMFsolver/phase_velocity.py tests/test_phase_velocity_isothermal_relativistic.py
git commit -m "[v4.6] guard the isothermal static-isobar approximation"
```

---

### Task 3: Exact relativistic junction conditions, opt-in

**Only start this task if Task 2's measurement is judged insufficient.** It is invasive and, by the 2026-08-19 measurement, changes no published number.

**Files:**
- Modify: `RMFsolver/phase_velocity.py` (`_Pi_QM_state`, `_quark_state_residual`, `_solve_muB_inf_at_muK0_for_given_Pi`, `_solve_muB_inf_at_muK0_for_given_Pi_ms`, `solve_front_isothermal`)
- Test: `tests/test_phase_velocity_isothermal_relativistic.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces: `_relativistic_flux_pair(nB, w, P, u) -> (jB, Pi)`. `_Pi_QM_state` gains `relativistic=False`; `solve_front_isothermal` gains `relativistic=False` and threads it to every helper it calls.

- [ ] **Step 1: Write the failing test**

```python
class TestRelativisticFluxPair(unittest.TestCase):
    def test_reduces_to_the_newtonian_pair_at_small_velocity(self):
        nB, w, P, u = 1.0e7, 5.0e9, 1.0e9, 1.0e-6
        jB, Pi = pv._relativistic_flux_pair(nB, w, P, u)
        self.assertAlmostEqual(jB / (nB * u), 1.0, places=10)
        self.assertAlmostEqual(Pi / (P + w * u * u), 1.0, places=10)

    def test_carries_the_gamma_factors_at_large_velocity(self):
        nB, w, P, u = 1.0e7, 5.0e9, 1.0e9, 0.6
        gamma = 1.0 / np.sqrt(1.0 - u * u)
        jB, Pi = pv._relativistic_flux_pair(nB, w, P, u)
        self.assertAlmostEqual(jB, nB * gamma * u, places=4)
        self.assertAlmostEqual(Pi, P + w * gamma**2 * u * u, delta=1.0)

    def test_rejects_superluminal_velocity(self):
        with self.assertRaises(RuntimeError):
            pv._relativistic_flux_pair(1.0e7, 5.0e9, 1.0e9, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./bin/python3 -m unittest tests.test_phase_velocity_isothermal_relativistic.TestRelativisticFluxPair -v`
Expected: FAIL with `AttributeError: ... '_relativistic_flux_pair'`

- [ ] **Step 3: Write the implementation**

Insert immediately below `_momentum_flux_diagnostics`:

```python
def _relativistic_flux_pair(nB, w, P, u):
    """Return the exact (jB, Pi) junction fluxes for 3-velocity ``u``.

    jB = nB*gamma*u and Pi = P + w*gamma**2*u**2, the planar steady-front
    baryon and momentum fluxes.  The gamma -> 1 limit is the pair the
    isothermal solvers use by default.
    """
    nB = float(nB)
    w = float(w)
    P = float(P)
    u = float(u)
    if (not np.isfinite(u)) or not (0.0 <= u < 1.0):
        raise RuntimeError("Front-frame velocity must satisfy 0 <= u < 1")
    gamma_squared = 1.0 / (1.0 - u * u)
    gamma = float(np.sqrt(gamma_squared))
    return float(nB * gamma * u), float(P + w * gamma_squared * u * u)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./bin/python3 -m unittest tests.test_phase_velocity_isothermal_relativistic.TestRelativisticFluxPair -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Commit the standalone helper**

```bash
git add RMFsolver/phase_velocity.py tests/test_phase_velocity_isothermal_relativistic.py
git commit -m "[v4.7] add exact relativistic junction-flux helper"
```

- [ ] **Step 6: Write the equivalence test that must hold after threading**

```python
class TestRelativisticThreading(unittest.TestCase):
    def test_default_path_is_unchanged(self):
        """relativistic=False must reproduce the pre-existing result exactly."""
        base = pv.solve_front_isothermal(
            1.0, 3.5 * N0, B14, 0.5, n_mesh=120, tol_bvp=1.0e-3
        )
        same = pv.solve_front_isothermal(
            1.0, 3.5 * N0, B14, 0.5, n_mesh=120, tol_bvp=1.0e-3, relativistic=False
        )
        self.assertEqual(base["jB"], same["jB"])

    def test_relativistic_agrees_to_the_measured_correction_size(self):
        newton = pv.solve_front_isothermal(
            1.0, 3.5 * N0, B14, 0.5, n_mesh=120, tol_bvp=1.0e-3
        )
        exact = pv.solve_front_isothermal(
            1.0, 3.5 * N0, B14, 0.5, n_mesh=120, tol_bvp=1.0e-3, relativistic=True
        )
        self.assertTrue(exact["success"])
        self.assertLess(abs(exact["jB"] / newton["jB"] - 1.0), 1.0e-5)
```

- [ ] **Step 7: Thread the flag**

Add `relativistic=False` to `_Pi_QM_state`; when true, compute `u` from `jB` by inverting `jB = nB*gamma*u` (solve `u = jB/(nB*sqrt(1 + (jB/nB)**2))`, the closed-form inverse) and return `P + w*gamma**2*u**2`. Add `relativistic=False` to `_quark_state_residual`, `_solve_muB_inf_at_muK0_for_given_Pi`, `_solve_muB_inf_at_muK0_for_given_Pi_ms`, and `solve_front_isothermal`, passing it through unchanged at every call site **inside `solve_front_isothermal` only**. Do not change any call site reached from `analytic_velocity_bound`, `semi_analytic_velocity_bound`, `analytic_velocity_bound_lte`, or `solve_front_thermal_conducting`.

- [ ] **Step 8: Run the full isothermal test suite**

Run: `./bin/python3 -m unittest tests.test_phase_velocity_isothermal_relativistic tests.test_phase_velocity_isothermal_a_0plus_max -v`
Expected: PASS. `test_default_path_is_unchanged` must pass with exact equality — any difference means the flag leaked into the default path.

- [ ] **Step 9: Confirm the other solver families are untouched**

Run: `./bin/python3 -m unittest discover tests/ -v`
Expected: same pass/fail set as before this plan. Two failures in `solve_front_isothermal` strict-retry are known pre-existing (see the auto-memory note on the isothermal strict-retry bug) — they must not change.

- [ ] **Step 10: Commit**

```bash
git add RMFsolver/phase_velocity.py tests/test_phase_velocity_isothermal_relativistic.py
git commit -m "[v4.8] opt-in relativistic junction conditions for the isothermal BVP"
```

---

### Task 4: Record the validity bound in the wiki

**Files:**
- Modify: `wiki/physics/isothermal-analytic-front-speed.md`, `wiki/index.md`, `wiki/log.md`

- [ ] **Step 1: Read the schema first**

Run: `cat wiki/SCHEMA.md`
Frontmatter, size caps, and supersession rules are defined there and are not optional.

- [ ] **Step 2: Add the section**

Append to `wiki/physics/isothermal-analytic-front-speed.md`, before `## Sources`:

```markdown
## Static-isobar validity

The interface construction reduces the momentum junction condition to a static
isobar, $P(0^+)=P(0^-)$, dropping the flux term $w\,u^2(0^-)$. Its relative
size is $(\Pi-P)/P = w\,u^2/P$, which scales as $1/[1-a(0^+)]$ because
$u(0^-)\propto[1-a(0^+)]^{-1/2}$.

| $T$ (MeV) | $n_B(0^-)/n_0$ | $w/P$ | $1-a(0^+)$ at 10% | $\gamma-1$ there |
|---:|---:|---:|---:|---:|
| 1 | 3.5 | 5.74 | $10^{-9}$ | 0.019 |
| 40 | 3.0 | 6.80 | $10^{-11}$ | 0.0075 |
| 80 | 2.0 | 8.74 | $3\times10^{-12}$ | 0.039 |

Realized compositions span $a_{\max}(0^+)\in(0.01,0.99)$, where the correction
is below $10^{-6}$ — under `tol_bvp`, the ray solver's $\rho$ tolerance, and
the momentum-flux residual gate alike. The flux term dominates the $\gamma$
factors by about a decade, so a partial fix adding $\gamma$ without $w\,u^2$
would be backwards. `MOMENTUM_FLUX_RATIO_TOLERANCE` makes the failure loud
rather than silent; it does not bind anywhere in the present domain.
```

- [ ] **Step 3: Update the index and log**

Add one line to `wiki/log.md` dated 2026-08-19 recording the measurement and the guard. `wiki/index.md` needs no new row — the page already exists — unless Task 3 was executed, in which case note the opt-in flag in the solvers table.

- [ ] **Step 4: Commit**

```bash
git add wiki/
git commit -m "[v4.9] record the isothermal static-isobar validity bound"
```

---

## Self-Review

**Spec coverage.** The design basis has three claims: the exact junction conditions (Task 3), the measured validity bound (Tasks 1-2, 4), and the constraint that shared solvers must not shift (Task 3 Steps 7-9). All covered.

**Placeholders.** None — every code step carries literal code, every run step a literal command and expected outcome.

**Type consistency.** `_momentum_flux_diagnostics` returns the same three keys used by Task 2's guard and Task 1's tests. `_relativistic_flux_pair` returns `(jB, Pi)` in that order, matching Task 3's tests and the threading step. `MOMENTUM_FLUX_RATIO_TOLERANCE` is defined in Task 2 Step 3 and referenced in Task 2 Steps 1 and 4 and in the wiki text.

**Known risk.** Task 3 Step 7 is the only step whose blast radius exceeds one function. If `test_default_path_is_unchanged` fails at Step 8, revert to the Task 2 commit — Tasks 1-2 stand alone and deliver the validity guard without it.
