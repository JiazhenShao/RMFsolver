# Isothermal Contours, High-Temperature Phase (10–100 MeV, polar mesh) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompute the isothermal speed contours 50–5000 m/s for $T(0^-)$ = 10–100 MeV with the current non-relativistic solvers, on a polar mesh confined to the physical quarter ring.
- Save every intermediate quantity needed to retune the analytic $\xi$ later without new EOS or BVP solves.
- Render figures that reproduce the attached reference figures' style.
- The low-temperature phase (0.01–10 MeV, log axis, Cartesian mesh) follows in a separate plan and completes the $T<10$ MeV band of the linear figure and the log-axis figure.

**Architecture:** Use the established hidden polar coordinates of [[isothermal-contour-cluster]] (wiki):

$$X=\frac{n_B(0^-)/n_0-1}{4.5},\qquad Y=\frac{T(0^-)}{120\ \mathrm{MeV}},\qquad X=\rho\cos\theta,\ Y=\rho\sin\theta .$$

On each ray, solve two static boundaries afresh:
- the inner radius, where PNM coexists with $\mu_K=0$ uds quark matter ($\Delta\mu_B=0$);
- the outer radius, where PNM is in Gibbs balance with udd quark matter ($a(0^+)=1-10^{-9}$).

Mesh points lie strictly between them, at radial fractions clustered toward both boundaries. Solve both methods at every interior point and keep per-point records, including the $\xi$-independent analytic pieces and the numerical profile integrals.

Contours are drawn from the curvilinear $(n_B,T)$ grid of $\log v$. Retuning $\xi$ is an exact per-point rescaling of the analytic grid followed by recontouring, in under a second. The gray regions are never evaluated; they are drawn from static boundary curves computed over the whole 0.01–115 MeV range.

**Tech Stack:** Python 3.12 venv, NumPy, isolated SciPy 1.18.1 runtime (`local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64` first on `PYTHONPATH`), `RMFsolver.phase_velocity` `[v4.9.5]`, `multiprocessing` (spawn), `unittest`, `nbformat`, matplotlib.

**Spec:**
1. The user's design pasted 2026-10-27.
2. The user's corrections of 2026-10-27:
   - split high T and low T;
   - the polar quarter-ring mesh for high T;
   - a log-T Cartesian mesh for low T;
   - never calculate inside the gray regions;
   - the gray boundaries are PNM–uds coexistence (zero speed below) and PNM–udd balance (speed diverges above);
   - save intermediate results for later $\xi$ tuning;
   - figures closely reproducing the two attached references, with the log figure's upper limit now 10 MeV.
3. The method-aware acceptance from the independent solver review in `wiki/solvers/isothermal-slow-shooting-initializer.md`.

This plan supersedes the contour-target design in `wiki/methods/nonrelativistic-isothermal-contours.md` for the high-T panel; that page is updated in Task 7.

**Reference figures and code (consult, do not import):**
- **Linear reference:** `All_Plots_new.ipynb` cell 25, "Isothermal numerical $u(0^-)$", exported as `local_analysis_results/All_Plots_new/isothermal-numerical-contour.{png,pdf}`.
- **Log reference:** `26-10-02.ipynb` cell 8, "Isothermal $u(0^-)$", exported as `isothermal-analytic-numerical-logT`.
- **Domain code:** `new_paper_calculations/26-08-20/_isothermal_domain.py` (polar transform and boundary roots); its $\mu_B$ scans are obsolete.
- **Saved boundaries:** `new_paper_calculations/26-09-16/isothermal_numeric_repair/isothermal-domain.npy`, 24 rays, used as a cross-check.
- **Adiabatic polar workflow:** `new_paper_calculations/26-10-26/extend_nonrel_contour.py` (inner 0.005 shell, outer bracket strip).
- **Prior $\xi$ cache:** `new_paper_calculations/26-10-26/xi_fit/` (reusable-cache precedent).

## Global Constraints

- **Solvers:** `solve_front_isothermal_slow(..., initial_guess_mode="auto", return_profile=True)` and `analytic_velocity_isothermal_slow(..., xi=-0.85)`. No `RMFsolver/` edits.
- **Physics:** massless, PNM, `paraQMCRMF3`, `B_one_forth=189.1565957288247` MeV, $n_0$ = `1229360.5098323745` MeV³, automatic exact $A_{\rm iso}$, revised Madsen rate, $\alpha_s=0.4$. The adiabatic fitted $\xi$ values are not transferred.
- **Polar transform:** density origin `1.0` $n_0$, density scale `4.5`, temperature scale `120.0` MeV.
- **Upper boundary convention:** $a(0^+)=1-10^{-9}$.
- **No gray evaluations:** fractions lie strictly in (0, 1). A near-boundary point that the solver classifies as stable or saturated produces a `boundary_roundoff` record (masked), never a speed and never a failure.
- **Mesh band and panel:** compute only $8\le T(0^-)\le100$ MeV in this phase; points below are `outside_panel`. The 8–10 MeV band is the seam overlap with the low-T phase.
- **Numerical settings:** `tail_eps=1e-8`, `n_mesh=300`, `tol_bvp=1e-4`, `max_nodes=50000`, `kappa_factor=1`, `path_rtol=1e-6`; at most two attempts per physical state.
- **Speed and levels:** $v=c\,u(0^-)$ with $c$ = `299792458.0` m/s; linear-figure levels `(50, 100, 200, 500, 1000, 2000, 5000)` m/s.
- **Contours:** never filled, extrapolated or connected through a non-accepted point.
- **$\xi$ retuning:** the analytic speed is $v(\xi)=c\sqrt{2I_2/\left[\lambda_n^2(1-a)(1+\xi a)\right]}$, with $I_2$, $\lambda_n$ and $a=a(0^+)$ saved at every point. Retuned grids must equal direct solver calls to `1e-12` relative.
- **Figure fidelity:** reproduce the reference cells' tunable values exactly, except data-dependent label placement and the font-rule question in Task 6.
- **Notebooks and figures:**
  - Build cells with `nbformat` and never read a `.ipynb` directly.
  - Every plot cell starts with a `# Tunable plotting parameters` block.
  - Inspect every exported PNG for overlap.
- **Notation:** location-based endpoints (`_0minus`, `_0plus`, `_inf`).
- **Git:** stage only the paths this plan creates, and only your own hunks of `wiki/index.md` and `wiki/log.md`. Data outputs are not committed unless the user asks.

## Review Focus

1. **Near-boundary roundoff** at fractions 0.005 and $1-10^{-5}$ becomes `boundary_roundoff`, and nothing probes further into the gray region. Test: `test_roundoff_inside_mesh_is_masked_not_failed` (Task 3).
2. **NaN collocation RMS with `success=True`** gets one 600-node retry. A shooting fallback is accepted only through its own branch. Tests: `test_nan_rms_collocation_retries_then_accepts` and `test_shooting_branch_requires_flux_change_and_mismatch` (Task 3).
3. **$\xi$ retuning silently drifting from the solver** after code changes. The cache records the analytic source hash; a retune against a mismatched hash raises unless forced. Tests: `test_retuned_grid_matches_direct_analytic` and `test_retune_rejects_changed_source` (Task 5).
4. **Top level unreachable on a ray** (still below 5000 m/s at fraction $1-10^{-5}$). That ray's 5000 contour is absent and reported, never extrapolated. Test: `test_coverage_reports_rays_missing_top_level` (Task 6).
5. **Resume after interruption or a source change.** Completed points are reused; a changed identity refuses to resume without `--restart`. Tests: `test_resume_skips_completed_points` and `test_resume_rejects_changed_identity` (Task 4).

---

## File Structure

| Path | Responsibility |
|---|---|
| `new_paper_calculations/26-10-27/isothermal/_iso_common.py` | Constants, polar transform, atomic save/load, identity hashes |
| `new_paper_calculations/26-10-27/isothermal/_iso_boundaries.py` | Static boundary residuals, ray roots, fixed-T roots, dense boundary curves |
| `new_paper_calculations/26-10-27/isothermal/_iso_evaluate.py` | One evaluation, acceptance, $\xi$-independent and profile-integral fields |
| `new_paper_calculations/26-10-27/isothermal/_iso_engine.py` | Point scheduling, timeouts, checkpoints, resume, profile sidecars (mesh-agnostic) |
| `new_paper_calculations/26-10-27/isothermal/_iso_xi.py` | $\xi$ cache build, `retune`, `effective_xi`, provenance checks (no EOS imports) |
| `new_paper_calculations/26-10-27/isothermal/_iso_plot.py` | Masked contour arrays, gray polygons, coverage, reference-style figure functions |
| `new_paper_calculations/26-10-27/isothermal/highT/polar_mesh.py` | High-T rays, fraction columns, panel mask |
| `new_paper_calculations/26-10-27/isothermal/highT/run_highT.py` | CLI and configuration; outputs in `highT/` |
| `new_paper_calculations/26-10-27/isothermal/highT/README.md` | Commands, settings, coverage, timings, retune usage |
| `new_paper_calculations/26-10-27/isothermal/lowT/README.md` | Placeholder pointing to the later low-T plan |
| `new_paper_calculations/26-10-27/isothermal/build_isothermal_notebook.py` | Builds and executes `26-10-27-isothermal.ipynb` |
| `tests/test_isothermal_contours_26_10_27.py` (+ `tests/_iso_fake_evaluator.py`) | Tests for Tasks 1–6 |

---

### Task 1: Static boundaries

**Files:** create `_iso_common.py` and `_iso_boundaries.py`; test in `tests/test_isothermal_contours_26_10_27.py`.

**Interfaces (produces):**
- `to_physical(rho, theta) -> (x, T)` and `to_polar(x, T) -> (rho, theta)`, where `x` is $n_B(0^-)/n_0$.
- `coexistence_residual(x, T) -> float`: $\mu_B(\infty)-\mu_B(0^-)$ at common $P,T$ with $\mu_K=0$, positive in the lower gray region. Built from `_validated_pnm_state_from_nB` and `_slow_isobar_state`.
- `udd_residual(x, T) -> float`: $\mu_B(0^+)+a\mu_K(0^+)-\mu_B(0^-)$ at $a=1-10^{-9}$ and $P(0^-)$, from `_solve_interface_0plus_from_local_a_and_Pi`, with its sign chosen to be positive in the upper gray region.
- `ray_boundaries(theta) -> dict(theta, rho_inner, rho_outer, x_inner, T_inner, x_outer, T_outer)`, solved by `brentq` to `xtol=1e-12`. It raises `RuntimeError` if a root is missing or duplicated, or if `rho_outer <= rho_inner`.
- `fixed_T_window(T) -> dict(x_inner, x_outer, status)`, for the low-T rows later.
- `boundary_curves(T_min=0.01, T_max=115.0, n=121) -> dict(inner=(x,T), outer=(x,T))`. It combines Chebyshev-clustered rays with fixed-T rows on a log grid below 10 MeV, so both the linear and the log figures get smooth gray outlines.

- [ ] **Step 1: Write failing tests**
  - `test_polar_round_trip`.
  - `test_boundary_status_on_each_side`: on θ = 10°, 45°, 80°, the slow analytic API reports `stable_neutron_matter` just inside $\rho_{\rm in}$, `moving_front` just outside $\rho_{\rm in}$ and just inside $\rho_{\rm out}$, and `composition_ceiling_saturated` just outside $\rho_{\rm out}$. Offsets are ±$10^{-4}$ relative.
  - `test_boundaries_match_probe`: at T = 10 MeV the window lies in (2.75, 3.0) to (5.25, 5.5) $n_0$.
  - `test_boundaries_cross_check_saved_domain`: at the 24 saved boundary rays, $|\Delta\rho|\le10^{-3}$. If this fails, stop and report, because the EOS may have changed since August.
  - `test_boundary_curves_cover_both_figures`: the curves span 0.01 to at least 94 MeV and are monotone along the curve.
- [ ] **Step 2: Run; expect FAIL (no module).**
  - Command: `PYTHONPATH=<runtime>:. ./bin/python3 -B -m unittest tests.test_isothermal_contours_26_10_27 -k boundar`
- [ ] **Step 3: Implement.** Bracket each root by stepping outward (0.02 in $\rho$, or 0.05 $n_0$ at fixed T) until the sign changes, then refine.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit** `isothermal contours: static boundaries`.

### Task 2: High-T polar mesh

**Files:** create `highT/polar_mesh.py`; same test file.

**Interfaces (produces):** `build_mesh(n_rays=41, fractions=FRACTIONS, T_min=8.0, T_max=100.0) -> dict` with keys `theta (n_rays,)`, `fractions (n_f,)`, `rho/x/T (n_rays, n_f)`, `point_status` in {`pending`, `outside_panel`}, and `boundaries`.
- `FRACTIONS` = `(0.005, 0.02, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.66, 0.72, 0.78, 0.84, 0.90, 0.95, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 0.99995, 0.99999)`.
- The θ rays are uniform on $[\theta_8, 90^\circ]$, where $\theta_8$ solves $T_{\rm outer}(\theta)=8$ MeV. The 90° ray is the $n_B=n_0$ edge.

- [ ] **Step 1: Write failing tests**
  - `test_mesh_is_strictly_inside_ring`.
  - `test_mesh_panel_mask`.
  - `test_mesh_counts`: shape (41, 29), with at least 900 pending points.
  - `test_ninety_degree_ray_is_n0_edge`.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit** `isothermal contours: high-T polar mesh`.

### Task 3: Evaluation, acceptance and saved intermediates

**Files:** create `_iso_evaluate.py`; same test file.

**Interfaces (produces):** `evaluate(method, x, T) -> (record: dict, profile: dict | None)`.
- `record` always has `method, x, T, status (accepted|boundary_roundoff|unresolved|error), accepted, speed_m_s, message, runtime_s, attempts, a_0plus, nB_0minus, nB_0plus, nB_inf, muB_0minus, muB_0plus, muK_0plus, muB_inf, P_0minus`.
- **Analytic intermediates ($\xi$-independent, saved for retuning):** `I2, lambda_n, eta, D_K, invD_K, gamma_K, qD, delta_muB, xi_used`.
- **Numerical diagnostics:** `solution_method, initial_guess_used, bvp_status, bvp_max_rms_residual, shooting_flux_change, shooting_mismatch, closure_error_max, conservation_max, n_mesh_used, path_rtol_used, lambda_inf, jB`.
- **Numerical profile integrals** in the manuscript's definitions, for a later effective-$\xi$ comparison:
  - `I1_num` $=\int_0^\infty D_K(a')^2\,dx$;
  - `I2_num` $=n_B(0^+)^{-1}\int_0^{a(0^+)}D_K\Gamma_K\,da$;
  - `xi_profile` $=2I_1/[\lambda u(0^-)(1-a)a]-1$ with $\lambda=n_B(0^-)/n_B(0^+)$, plus the same with $n_B(\infty)$ in `xi_profile_inf`.
- `profile` (numerical only) is resampled to 256 nodes in $s$: `x, a, nK, jK, D_K, Gamma_K, muB, muK`, kept for later diagnostics without re-solving.

Acceptance rules:
- **Analytic:** `success` and `front_exists`.
- **Numerical, either method:**
  - solver `success`;
  - finite positive speed;
  - $0<a(0^+)<1$;
  - closure and conservation at most `1e-8`;
  - boundary residuals at most `1e-7`;
  - physical residual norms at most `max(5*tol_bvp, 1e-7)`.
- **Collocation:** additionally `bvp_status == 0` and a finite RMS at most `tol_bvp`. A NaN or Inf RMS triggers one retry with `n_mesh=600`.
- **Shooting:** additionally `shooting_flux_change <= 1e-6` and `abs(shooting_mismatch) <= 1e-8`.
- **Table retry:** the exact table-convergence message triggers one retry at `path_rtol=5e-7`.
- **Attempts:** never more than two per state. Exceptions and IVP event failures become `unresolved`, with the message kept.

- [ ] **Step 1: Write failing tests**
  - `test_reference_points`: at (3.5, 10 MeV), analytic gives 221.7 ± 0.1 m/s and numerical 154.5 ± 0.1 m/s, both accepted.
  - `test_analytic_intermediates_reproduce_speed`: $c\sqrt{2I_2/[\lambda_n^2(1-a)(1+\xi a)]}$ equals `speed_m_s` to `1e-12`.
  - `test_numerical_profile_integrals`: `I2_num` agrees with `I2` from the analytic call at the same state to within 30%, a loose sanity bound because the approximations differ; `xi_profile` lies in (−1, 1).
  - `test_roundoff_inside_mesh_is_masked_not_failed`.
  - `test_nan_rms_collocation_retries_then_accepts`.
  - `test_shooting_branch_requires_flux_change_and_mismatch`.
  - `test_two_attempt_cap`.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.** The integrals use the trapezoid rule on the solver's returned profile nodes in $x$.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit** `isothermal contours: evaluation, acceptance and intermediates`.

### Task 4: Engine and high-T entry point

**Files:** create `_iso_engine.py`, `highT/run_highT.py`, `lowT/README.md` and `tests/_iso_fake_evaluator.py`; same test file.

**Interfaces:**
- Consumes: Tasks 2–3.
- Produces: `run(mesh, out_dir, method, workers, evaluator="…_iso_evaluate:evaluate", timeout_s, restart=False) -> dict`.
  - It writes `highT/isothermal-highT-<method>.npy`: `identity`, `mesh`, `records (n_rays, n_f)`, `speed_m_s` grid (NaN unless accepted), `status` grid, `timings`.
  - For the numerical method it also writes `highT/profiles-numerical.npz`, one entry per accepted point keyed `"i_j"`.
- CLI: `run_highT.py --method {analytic,numerical,both} --workers 4 [--smoke] [--dry-run] [--restart] [--plot-only]`.

- [ ] **Step 1: Write failing tests** with the fake evaluator:
  - `test_run_fills_only_pending_points`;
  - `test_resume_skips_completed_points`;
  - `test_resume_rejects_changed_identity`;
  - `test_point_timeout_is_recorded_and_others_finish`;
  - `test_profiles_sidecar_matches_accepted_points`.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.**
  - Each point is a bounded child process: spawn, one numerical-library thread, a timeout of 120 s numerical or 90 s analytic, at most `workers` concurrently.
  - The parent alone writes atomically after every completed point, then updates progress.
  - Identity covers the hashes of `phase_velocity.py` and the `_iso_*.py` files, physics, settings, the mesh fingerprint and the levels.
  - `--smoke` uses rays θ = {15°, 45°, 75°} and all 29 fractions, written to `highT/smoke/`.
- [ ] **Step 4: Run the tests, then the real smoke.**
  - Command: `run_highT.py --method both --smoke --workers 4`
  - Record per-point timings and the projected full-run time in `highT/README.md`. The outermost column should exceed 5000 m/s numerically on the 15° and 45° rays.
- [ ] **Step 5: Commit** `isothermal contours: engine and high-T entry point`.

### Task 5: Reusable $\xi$ cache and effective $\xi$

**Files:** create `_iso_xi.py`; same test file.

**Interfaces:**
- Consumes: the analytic payload, plus the numerical payload for `effective_xi`.
- Produces:
  - `build_cache(analytic_payload) -> dict` with grids `I2, lambda_n, a_0plus, accepted_mask, x, T` and provenance (source hashes, physics, mesh fingerprint). Written to `highT/xi-cache-highT.npz`.
  - `retune(cache, xi) -> speed_m_s grid`, which needs no EOS or solver imports.
  - `effective_xi(cache, numerical_payload) -> grid`: the $\xi$ at which the analytic speed equals the numerical one at each point, $\xi_{\rm eff}=\left[2I_2/(\lambda_n^2(1-a)(v_{\rm num}/c)^2)-1\right]/a$; NaN where either method is not accepted.
  - `fit_xi(cache, numerical_payload, levels, objective="log_speed"|"contour") -> float`, a scalar fit for later use. The fit itself is run by the user later; here only the function and its test.

- [ ] **Step 1: Write failing tests**
  - `test_retuned_grid_matches_direct_analytic`: at 5 sample points and $\xi\in\{-0.85,-0.4,0,0.5\}$, the retuned speed equals a fresh `analytic_velocity_isothermal_slow` call to `1e-12`.
  - `test_retune_rejects_changed_source`.
  - `test_effective_xi_recovers_injected_value`: a synthetic numerical grid built from the cache at $\xi=0.3$ gives $\xi_{\rm eff}=0.3$ everywhere.
  - `test_xi_module_imports_no_solver`.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit** `isothermal contours: reusable xi cache`.

### Task 6: Contours, reference-style figures and notebook

**Files:** create `_iso_plot.py`, `build_isothermal_notebook.py` and `26-10-27-isothermal.ipynb`; same test file.

**Interfaces:**
- Consumes: Tasks 1, 4 and 5.
- Produces:
  - `contour_arrays(payload_or_grid) -> (x, T, log10_v_masked)`;
  - `gray_polygons(curves, x_limits, T_limits) -> dict(lower, upper)`;
  - `coverage(payload, levels) -> dict`;
  - `plot_linear_reference(ax, sources, style) -> None`. `sources` is a list of `(x, T, log10_v, T_range, line_style)`. In this phase it holds only the polar grid for T ≥ 10 MeV; phase 2 adds the Cartesian low-T grid for T < 10, and the polar contours are clipped at 10 MeV so the two meet.
  - `plot_logT_reference(...)` is implemented in phase 2.
- Linear style is copied exactly from `All_Plots_new.ipynb` cell 25:
  - figure 7.2×7.2 in, box aspect 1, DPI 160/220, white background, constrained layout, mathtext `cm`;
  - limits x (1.0, 5.5), y (0.0, 115.0) MeV, linear axes;
  - title "Isothermal numerical $u(0^-)$ [$\mathrm{m\,s^{-1}}$]" at 20 pt with pad 8; axis labels and ticks 18 pt; ticks in, on all sides, length 4, width 0.9; spine width 0.8;
  - level colours `("#1f77b4","#17becf","#ff7f0e","#8c6d31","#2ca02c","#d62728","#9467bd")`, solid lines of width 1.8;
  - inline `clabel` labels in `%g` format at the temperatures {50: 50, 100: 55, 200: 60, 500: 55, 1000: 45, 2000: 30, 5000: 15} MeV;
  - excluded regions gray `0.72`;
  - region texts "Quark matter\nnot favored" at (1.18, 20) and "$udd$ quark\nmatter favored" at (3.85, 94), colour `0.10`, 18 pt.
- Comparison figure (linear): identical style. Numerical contours are solid and analytic ($\xi=-0.85$) contours dashed `--`, in matching colours. A lower-left method legend uses the labels "Analytical, $\xi=-0.85$" and "Numerical", following the log reference's legend style.
- Exports:
  - `local_analysis_results/26-10-27/isothermal-highT-numerical.{png,pdf}`;
  - `isothermal-highT-comparison.{png,pdf}`;
  - `isothermal-highT-old-vs-new.{png,pdf}`, an overlay of the August relativistic numerical archive (dotted) on the new curves, to show the physics shift.

- [ ] **Step 1: Write failing tests**
  - `test_outside_panel_points_masked_in_contour_arrays`.
  - `test_coverage_reports_rays_missing_top_level`.
  - `test_gray_polygons_lie_outside_mesh_points`.
  - `test_linear_style_matches_reference_values`: parse the reference cell's tunable block with `nbsrc.py -c 25` and assert the copied constants are equal, except the documented font decision.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.**
  - Notebook cells:
    1. markdown;
    2. load-and-check (identity, coverage);
    3. linear numerical figure;
    4. linear comparison figure, with an `XI` tunable that calls `retune`, so later changes to $\xi$ redraw instantly;
    5. old-vs-new overlay.
  - **Font decision.** The reference uses 11 pt contour labels, while the project rule requires at least 16 pt for annotations in figures up to 8 in wide. Default to 16 pt as the documented single deviation, unless the user chooses exact reference sizes.
- [ ] **Step 4: Run the tests, build and execute the notebook, and inspect the PNGs** next to `local_analysis_results/All_Plots_new/isothermal-numerical-contour.png`. Adjust only the data-dependent label temperatures if a label falls off its curve.
- [ ] **Step 5: Commit** `isothermal contours: reference-style figures, xi retune cell and notebook`.

### Task 7: Production, verification and documentation

**Files:** modify `highT/README.md`; create `wiki/methods/isothermal-highT-polar-contours.md`; update `wiki/methods/nonrelativistic-isothermal-contours.md`; add one index line and one log entry.

- [ ] **Step 1: Run production.**
  - Command: `PYTHONPATH=<runtime>:. ./bin/python3 -B new_paper_calculations/26-10-27/isothermal/highT/run_highT.py --method both --workers 4`
  - Expected wall time: about 10 min numerical and 3 min analytic, from about 1100 points at 0.4–4 s each.
  - If the smoke projection exceeds 30 min, stop and prepare the cluster command instead.
- [ ] **Step 2: Coverage.** Report counts per method and the per-level coverage table, and list every unresolved point.
- [ ] **Step 3: Check contour placement against direct solves.** At about 12 crossings (levels 100, 1000 and 5000 on 4 rays), require the grid contour within $0.01\,n_0$ and 0.5 MeV of a direct Illinois root with $|v/v_{\rm level}-1|\le10^{-3}$. Refine columns or rays only where this fails.
- [ ] **Step 4: Check resolution independence.** At 6 accepted points, re-solve with `n_mesh=600`, `tol_bvp=2e-5`, `path_rtol=2e-7` and `tail_eps=1e-9`. Require a speed change of at most $10^{-3}$, saved to `highT/verify_highT.json`.
- [ ] **Step 5: Build the $\xi$ cache, run the retune checks and report the $\xi_{\rm eff}$ statistics** by ray and fraction, without choosing a new $\xi$. That choice is the user's later step.
- [ ] **Step 6: Re-execute the notebook on the production data and inspect the figures. Then update the wiki pages, index and log, and commit and push** the code, tests, notebook builder, notebook, README and wiki.

---

## Phase 2 (low T, 0.01–10 MeV): outline for the later plan

- **Mesh:** Cartesian in ($n_B(0^-)/n_0$, $\log_{10}T$), with 31 rows `np.geomspace(0.01, 10, 31)`. Each row uses `fixed_T_window` and evaluates only interior points.
- **Clustering is mandatory:** at 0.01 MeV, 50–3250 m/s lie within about $0.05\,n_0$ of coexistence, and $10^6$ m/s lies about $0.2\,n_0$ below the udd boundary. Either use geometric offsets from both boundaries or use per-row fraction columns, which is still Cartesian in (fraction, $\log T$). Choose with the user.
- **Levels:** all 14, through $10^6$ m/s. Same intermediates and $\xi$ cache.
- **Seam:** high-T points between 8 and 10 MeV overlap the top rows as a consistency check.
- **Figures:**
  - The log figure reproduces `26-10-02.ipynb` cell 8, with `Y_LIMITS=(0.01, 10.0)` and ticks (0.01, 0.1, 1, 10). Numerical results are drawn as solid lines everywhere; the old $T<20$ MeV dot markers and split legend are dropped because the cold numerics now converge.
  - The linear figure gains its $T<10$ MeV band from these data.
- **Cost:** 2–15 s per cold solve, with one known unresolved state (0.01 MeV, $5.2n_0$). A pilot decides between local and cluster runs.

---

## Phase 2 addendum (2026-10-27, started while the user reviews the high-T figure)

**Mesh:** Cartesian in (radial fraction, $\log_{10}T$). The rows are `np.geomspace(0.01, 10.0, 31)`. On each row, $n_B=x_{\rm in}(T)+f\,[x_{\rm out}(T)-x_{\rm in}(T)]$ between that row's coexistence and udd roots (`fixed_T_window`, cached per T), so no gray point is evaluated.

**Low-T fractions:** `(1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.006, 0.01, 0.02, 0.04, 0.06, 0.1, 0.15, …, 0.95, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999)`.
- The inner decades are needed because at 0.01 MeV, 50–3000 m/s lie within about $5\times10^{-4}$ of the window above coexistence.
- The outer columns stop at $1-10^{-4}$: $10^6$ m/s is reached at $f\approx0.9$ at 0.01 MeV, and near-udd cold solves are the known failure region.

**Display:** per-row PCHIP of $\log_{10}v$ in $\mathrm{logit}\,f$, which resolves both ends. Contours are computed in $(n_B,\log_{10}T)$ and mapped back, so the log axis carries no linear-in-$T$ interpolation artifact.

**Figures:**
- The log figure follows `26-10-02.ipynb` cell 8: 14 levels, solid numerical and dashed analytic, two legends, $T$ from 0.01 to 10 MeV. The `a(0^+)>=1` text is rotated into the narrow udd strip, as in `All_Plots_new` cell 28.
- The linear figure combines high T (polar, $T\ge10$) with low T (rows, $T\le10$).

**Checks:**
- Seam: contour positions at 10 MeV from both meshes agree within $0.01\,n_0$.
- Placement and resolution checks as in Task 7.

**Entry point:** `lowT/run_lowT.py`, with the same flags as `run_highT.py`.

---

## Execution record (2026-10-28)

Copied verbatim from the execution ledger before its deletion. Commits 0e557b89..35a5a1d2 on `codex/slow-front-bvp`; final review by a fresh reviewer, four Important findings fixed (full module 46/46).

### Rulings

- Setup: Ruling: no git worktree; work in the main workspace on feature branch codex/slow-front-bvp — the workspace root is the venv and holds the data/notebooks the run reads and writes — cost if wrong: concurrent sessions' edits could collide; mitigated by staging only own paths.
- Setup: Ruling: contour-label and legend fonts >= 16 pt (user instruction 2026-10-27) — overrides reference 11-12 pt — cost: small visual difference from references.
- Setup: Ruling: analytic prescription pending user (A keep code + manuscript sentences / B reader recipe); implement with current code (A), rerun analytic (~3 min) if user picks B — cost: one rerun.
- Task 1: Ruling: saved-domain cross-check test uses 6 of the 24 rays (indices 3,8,12,16,20,23; theta=0 excluded since T=0 is not a valid state); full 24-ray comparison moves to Task 7 verification on cached boundaries — each root costs ~15 s of PNM solves — cost if wrong: a mismatch on an untested ray surfaces later, at Task 7.
- Task 1: Ruling: udd_residual = -(gibbs residual of _solve_a_0plus_max at a=1-1e-9), coexistence_residual = muB_eq(P,T,muK=0) - muB(0-); both reuse one cached PNM state per (x,T) — same equations the solvers use for their statuses.
- Task 1: Ruling: boundary_curves signature is (n_rows, n_rays, T_rows_max, workers) instead of (T_min, T_max, n) — rows below 10 MeV and rays above need separate counts — cost: none (plan internal).
- Task 2: Ruling: theta_8 from the fixed-T udd root at 8 MeV (exact angle of that boundary point) instead of a root search over theta; ray boundaries cached in highT/ray-boundaries.npy keyed by solver/boundary source hashes and angles — saves ~2.5 min per mesh build — cost if wrong: none (cache invalidates on source change).
- Task 3: Ruling: any non-accepted, non-roundoff first numerical attempt gets exactly one retry (path_rtol x0.5 for the table message, else n_mesh=600); exceptions become 'unresolved' (no separate 'error' status) — matches the two-attempt cap — cost if wrong: one extra solve per failing point.
- Task 3: Ruling: numerical roundoff is detected by the solver message "Automatic interface ceiling is not interior" — the slow numerical API's only thermodynamic gray signal — cost if wrong: a near-boundary failure with another message counts as unresolved instead of masked.
- Task 3: Ruling: profile I1/I2 use D_K a' = (jB a - J_K)/n_B (manuscript a' = n_K'/n_B), avoiding numerical differentiation; xi_profile reported for both n_B(0+) and n_B(inf) — cost: none.
- Task 4: Ruling: removed the Task 2 single-key boundary cache file (format changed to a dict keyed by angle set); the next mesh build recomputes the 41 rays once (~4 min) — cost: one recomputation.
- Task 4: Ruling: numerical profiles saved as one small .npz per point in <payload>-profiles/ (atomic per file) instead of one aggregate npz rewritten after every point — avoids rewriting ~20 MB per point — cost: a folder of ~1000 small files.
- Task 4: Ruling: --plot-only dropped from run_highT.py; plotting lives in _iso_plot/notebook (Task 6) — the CLI stays solver-only — cost: none.
- Task 4: Ruling: Tasks 4 and 5 committed together (shared test file) — cost: coarser history.
- Task 4: Ruling: solver fix in RMFsolver/phase_velocity.py ([v4.9.6]): shooting centre and auto layer prediction use jB_estimate = current_scale/sqrt((a0- - a0+)(a0- - a_inf)); SciPy event-root ValueError inside a shot counts as a missing value; the tight self-check retries at 1e-11 and is non-fatal (NaN flux change, which the fallback now rejects — previously NaN > 1e-6 would have passed). Tests: test_shooting_brackets_flux_next_to_udd_boundary, test_failed_tight_check_never_validates_a_shooting_fallback (RED->GREEN). Cost if wrong: default mode untouched; only auto/shooting paths change.
- Task 5: Ruling: retune/fit_xi default check_source=True (refuse a cache made with a different phase_velocity.py); synthetic-cache tests opt out explicitly — cost: callers must pass check_source=False knowingly.
- Task 6: Ruling: the exact table-refinement message walks the adiabatic wrapper's path_rtol ladder (x0.5, 0.2, 0.1) instead of a single retry; all other failures keep the two-attempt cap — the 5 interior table failures converge at x0.2/x0.1 in 0.2 s — cost if wrong: up to 3 extra fast attempts on table failures.
- Task 6: Ruling: added _iso_engine.repair (re-evaluate unresolved/timeout points in place, records marked repaired=True, pass logged with source hashes) instead of a full --restart rerun — cost: payload mixes two evaluator versions for 9 points, logged explicitly.
- Task 6: Ruling: boundary_curves(extra_rays=mesh boundaries) so polygons have exact vertices on every mesh ray — chords of a coarse curve put f=0.99999 points on the gray side — cost: none.
- Task 7: Ruling: angular refinement 41 -> 81 rays after the first figures (kinks); seeded the 81-ray run from the 41-ray payloads (engine.run seed_from; identical (x,T) points copied with records/profiles) — 1146 numerical / 1150 analytic points reused — cost: none.
- Task 7: Ruling: run identity narrowed to result-affecting sources (phase_velocity.py, _iso_common, _iso_boundaries, _iso_evaluate); seeding requires matching solver hash + physics/settings/method — engine/plot edits no longer invalidate data — cost if wrong: an unlisted module that changes results would not force a rerun.
- Task 7: Ruling: ray-boundary cache per angle, keyed on phase_velocity.py + _iso_boundaries.py hashes and physical constants — cost: none.
- Task 6: Ruling: kinks persisted at 81 rays => radial, not angular; added display-only radial refinement (PCHIP of log10 v in u=-ln(1-f) within runs of accepted points, upsample 8) — failures stay holes, nothing extrapolated; placement checked against direct roots in verify_highT.py — cost if wrong: interpolation error in drawn contours, bounded by the verification.
- Task 6: Ruling: method legends moved to 'upper left' (empty udd-favored strip above ~95 MeV) — lower-left would cover the low-T contours in phase 2 — cost: none. Inline contour labels 16 pt (user instruction).
- Task 6: Ruling: draw_contours clip box limited to the axis limits — an unbounded clip box made the inline (tight bbox) renderer request a 7.7e9-pixel canvas.
- Phase 2: Ruling: low-T rows = the 40 cached boundary rows geomspace(0.01,10,40) instead of 31 — their exact coexistence/udd roots already exist (no new boundary solves) and the T resolution is finer — cost: ~30% more points.
- Phase 2: Ruling: per-row fraction columns (Cartesian in (f, log10 T)) with inner decades down to 1e-5 and outer to 1-1e-4 — boundaries move with T, fixed n_B columns would land in the gray region on some rows — cost: x columns not shared between rows (contouring handles a curvilinear grid).
- Phase 2: Ruling: the 2 interior holes stay holes (no relaxed acceptance, no unapproved solver change) — they may interrupt the 5e5/1e6 m/s numerical contours on a few rows — cost if wrong: two missing points at the top levels.
- Final: Ruling: boundary cache key widened to all RMFsolver/*.py + library versions (consistency with identity) — forces a deterministic rebuild of ray-boundaries.npy/boundary-curves.npz (old copies in session scratchpad) — cost if wrong: ~10 min recompute.
- Final: Ruling: reviewer #5 (solver edit [v4.9.6] contradicts plan constraint 'No RMFsolver/ edits'; jB_estimate also moves the auto-mode shoot-first decision) — not reverted; needs explicit user sign-off — cost if wrong: revert 5b5ae549 and rerun near-udd points.
- Final: Ruling: reviewer 'Ruling 7 not done' — added highT/verify_saved_domain.py (all 24 saved rays, criterion 1e-3): worst |drho| 2.5e-5, passed.

### Deferred minors (final review)

- two-attempt cap wording — one 600-node retry then table failures allows up to 5 attempts.
- verify_highT/verify_lowT 'worst' uses max/nanmax, hiding NaN entries.
- coverage() reports above_outer_column when outer columns are unresolved holes.
- timeout records drop x/T/method and earlier diagnostics; not marked repaired.
- no payload lock against concurrent run + repair.
- polar mesh fingerprint omits T_max.
- _iso_xi effective_xi/fit_xi skip mesh-fingerprint check; _ray_crossings compresses holes; contour objective needs mesh['rho'] (fails on row mesh).
- notebook builder checks identity only for high T, ignores -restartN payloads, COMPARISON cell crashes with numerical-only low-T.

### Fixed after the final review

- NaN acceptance hole (max()/'>' let NaN through) — test_acceptance_rejects_nonfinite_metrics RED→GREEN; shipped payloads: 0 accepted points with non-finite stored metrics.
- stale boundary-curves.npz read without key check (row mesh, gray polygons) — test_stale_boundary_curves_are_refused RED→GREEN.
- repair mixing solver versions + identity too narrow — engine._compatible (all RMFsolver/*.py recorded in identity, numpy/scipy versions; legacy payloads compared on what they recorded), repair(force=False) refuses — tests test_repair_refuses_a_payload_from_another_solver, test_identity_covers_eos_modules_and_library_versions, test_seed_refuses_a_changed_eos_module RED→GREEN.
- worker crash aborting the run (EOFError on recv) — dead worker's point recorded unresolved with exit code, slot respawned — test_worker_crash_is_recorded_and_run_continues RED→GREEN. Full module 46/46.
