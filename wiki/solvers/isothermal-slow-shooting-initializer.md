---
summary: solve_front_isothermal_slow(initial_guess_mode="auto") converges on all 35 mapped states down to 0.01 MeV (16 previously failing) in 0.5–15 s, agreeing with the old default to 8e-7 where both work; the default mode is bit-identical to before.
status: current
updated: 2026-10-27
tags: [solver, isothermal, numerical, low-temperature, slow-front, api]
---

# Shooting initializer for the cold slow isothermal BVP

Implemented 2026-10-27 in `RMFsolver/phase_velocity.py` from [[isothermal-slow-cold-fix-plan]]; the failure mechanism is in [[isothermal-slow-cold-failure]].
Use `initial_guess_mode="auto"` for any new isothermal slow-front work below about 10 MeV; the default still fails there.
Pre-change state: local git tag `pre-cold-isothermal-fix` (commit `6b02b74d`).

## API

```python
solve_front_isothermal_slow(T, nB_0minus, B_one_forth, ..., initial_guess_mode="auto")
```

- `"default"`: the old tail-shaped guess and tabulated tail slope. Outputs are bit-identical to the pre-change code at five reference states (three isothermal, two adiabatic, including one adiabatic failure).
- `"shooting"`: shoot from the equilibrium tail. A supplied `continuation_guess` is tried first, with its own nodes merged into the mesh; shooting follows if that collocation fails.
- `"auto"`: shoot when the advection-free layer half-point is predicted within 10 first-mesh intervals, or when the default attempt fails.
- Both non-default modes put the direct equilibrium slope `_SlowFrontPath.equilibrium_rate_slope()` into the tail condition (`tail_rate_slope="direct"`). `solve_front_adiabatic_slow` is unchanged, and non-default modes raise for other closures.
- New result fields: `initial_guess_mode`, `initial_guess_used` (`default`, `continuation`, `shooting`), `solution_method` (`collocation`, `shooting`) and `tail_rate_slope`. Shooting runs add `shooting_flux_change`, `shooting_mismatch`, `shooting_time_s`, `shooting_layer_s_half`, and `collocation_message` on fallback.

## How it works

1. `_slow_front_shoot` starts at $r=10^{-10}r_{\rm scale}$ on the decaying eigenvector and integrates $dr/dx=j_Ba-J_K$, $dJ_K/dx=-\Gamma_K$ toward the interface with LSODA (`rtol=1e-10`) on the existing path splines. Off-solution components decay in this direction.
2. `_slow_front_shooting_flux` brackets $\ln j_B$ within 0.2–3 (then 0.01–100) times the advection-free flux $\sqrt{2r_{\rm scale}\int\Gamma_K\,dz}/(a(0^-)-a(0^+))$, then refines with `brentq`. A second root at `rtol=1e-12` and start depth $10^{-12}$ gives `shooting_flux_change`.
3. The mesh is the existing compact mesh plus 40 logarithmic nodes per decade from $10^{-3}s_{1/2}$ to $\max(30s_{1/2},10^{-2})$. The guess is the shot profile.
4. Collocation uses the unchanged equations and acceptance.
5. If collocation fails, the shot trajectory itself is checked. It is sampled at four points per LSODA step with nodes at least $10^{-4}$ layer half-widths apart, and interpolated by a cubic Hermite spline in $s$. It must pass the same acceptance (direct-EOS closure and conservation at most $10^{-8}$, residual norms at most $\max(5\,\mathrm{tol_{bvp}},10^{-7})$, boundary residuals at most $10^{-7}$) and also `shooting_flux_change` at most $10^{-6}$.

⚠ Without the $10^{-4}$ node-spacing floor, dense-output roundoff dominated the Hermite derivative near the interface, and fallback residuals reached $10^{-2}$–$10^{-1}$.

## Verification

- **Coverage:** on the 35-state map ($T=10,3,1,0.3,0.1,0.03,0.01$ MeV × $n_B/n_0=3.05,3.2,3.5,4.0,4.8$) with `"auto"`, all 35 converge. 34 converge by collocation and one ($4.8n_0$, 0.03 MeV) by validated shooting. Each solve takes 0.5–15 s; the longest is $4.8n_0$ at 0.1 MeV, with 36,590 nodes.
- **Agreement:** on the 19 states the default already solved, speeds differ by at most $8.1\times10^{-7}$ relative.
- **Independence:** at four cold states, `n_mesh=600`, `tol_bvp=2e-5`, `tail_eps=1e-9`, `path_rtol=2e-7` and `kappa_factor=2` each change the flux by at most $1.8\times10^{-9}$.
- **Forced fallback:** with collocation patched to fail at six cold states, all fallbacks are accepted with residuals $10^{-7}$–$2\times10^{-4}$ and the collocation speeds.
- **Speeds at 0.01 MeV:** 3249.93, 14990.1, 46578.8, 133356 and 520113 m/s at $3.05$, $3.2$, $3.5$, $4.0$, $4.8n_0$. Before the fix, brute force gave 3249.94 m/s at $3.05n_0$ on 201,775 nodes.
- **Tests:** `ColdIsothermalSlowTests` (8 tests) in `tests/test_phase_velocity_slow.py`. The slow and analytic-slow suites pass 38/38 (30/30 before the change); both runs take 15 min.
- **Legacy:** the bare-function `tests/test_phase_velocity.py` (pytest-style; run with `run_plain_tests.py` because the venv has no pytest) gives the same 11 pre-existing failures before and after, among 87 tests. These include the known strict-retry pair; see [[known-issues]].
- **Massive quarks:** at $m_s=100$ MeV (10 MeV, $a(0^+)=0.6$), `"shooting"` matches the default flux to $4\times10^{-9}$.

Logs, maps and scripts: `new_paper_calculations/26-10-27/isothermal_cold_diagnosis/` (`verify_fix.py`, `verify_fix.log`, `default_baseline_*.json`).
