---
summary: The notebook and combined plotting dataset now use the corrected second-order analytical contours; the semi-analytical and numerical results remain unchanged.
status: current
updated: 2026-10-26
tags: [adiabatic, analytic, cold-limit, contour, verification]
---

# Exact-cold adiabatic analytical contours

⚠ **Second-order correction, 2026-10-26:** the full-analytical endpoint and its original contour/figure results below are superseded by the consistently second-order heat matching in [[analytic-slow-front-solvers]].
The old endpoint omitted a density term of the same order as the heat balance, suppressing the temperature and ceiling by approximately `1/sqrt(3)`.
The closed integral and diffusion prescription are unchanged; the semi-analytical branch was not modified.
The separate corrected output is `local/adiabatic-T0plus0-xi-minus0p85-second-order.npy`, with its own sibling records directory.
At the user's subsequent request, the full-analytical branch in the combined plotting manifest was overwritten with the corrected scan, and notebook cells 10 and 14 were executed to replace the full-analytical figure and comparison overlay, including PNG/PDF exports.
The semi-analytical branch, its notebook cell/exports, and the original numerical cells/exports remain unchanged.
The pre-refresh notebook and combined manifest are archived under `local_analysis_results/26-10-26-second-order/notebook-refresh/`.
The canonical `local/adiabatic-T0plus0-xi-minus0p85.npy` is now a plotting collection with separate `method_identities`, `record_directories`, and `source_manifests`; it must not be treated as one resumable scan identity.

The corrected runner's analytical domain classifier uses the same quartic endpoint and `A_adb<1` criterion as the API.
The cold state at `nB_0minus=5.4*n0` is now excluded by the analytical composition ceiling, agreeing with the semi-analytical exclusion; the former leading-density mask admitted it.
Use the explicit new output path below: trying to resume the original default artifact with changed source hashes is intentionally rejected.

```sh
PYTHONPATH=local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:.:tests \
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/nsm-rate-mpl \
./bin/python3 -B new_paper_calculations/26-10-26/local/run_adiabatic_contours.py \
  --method analytic --workers 8 \
  --output new_paper_calculations/26-10-26/local/adiabatic-T0plus0-xi-minus0p85-second-order.npy
```

The correction passes 14 analytical tests and 25 contour-workflow tests.
Independent closed-integral quadrature remains accurate within `3.6e-15`; its formula was not altered.
At the four validation states the revised full-analytical speeds are 39.6547, 96.3749, 304.4747 and 1372.3870 m/s, compared with the superseded 19.5901, 44.5890, 126.1055 and 289.9968 m/s.
The first three exact-cold spatial BVP speeds were reproduced as 34.8871, 81.6343 and 250.4187 m/s.
These matched-temperature comparisons, full endpoint values, test expectation changes and manuscript checks are in `local_analysis_results/26-10-26-second-order/REPORT.md`.

## Completed corrected contour

The separately saved second-order run is complete: 61 rays support all seven requested levels through 5000 m/s, with zero failed samples, unresolved contours, unresolved boundaries, or pending angular intervals.
Its 4,602 fresh records comprise 2,732 accepted speeds, 787 physically excluded speed calls, 509 interior boundary checks, and 574 excluded boundary checks.
Runtime with eight local workers was 1,074.6 seconds (17.9 minutes).
The solver hash changes from `8bfdd9c23538...` to `3c22ccbb039bc...`; every new record matches the corrected identity, with no old analytical record reused.

The maximum level residual is `9.98125e-4`, radial bracket width `9.86279e-4`, and accepted angular displacement `0.00100922`, below their unchanged limits.
Maximum pressure error is `2.61e-15`; the largest explicit endpoint residual is the relative ceiling check at `9.11e-14`, below `1e-10`.
Accepted speeds of at least 50 m/s have equilibrium temperatures at least 6.0099 MeV, well outside the demonstrated near-coexistence cancellation limit of the strict relative ceiling check.
The independent audit is `local_analysis_results/26-10-26-second-order/contour-audit.json`.
Before the authorized notebook refresh, all 10,015 protected historical-file hashes were unchanged.
The subsequent refresh replaces only the canonical plotting manifest, notebook, full-analytical PNG/PDF and overlay PNG/PDF among those protected files; the original point records remain unchanged.

`new_paper_calculations/26-10-26/` adds a shared local/cluster contour engine for
`analytic_velocity_adiabatic_slow` and
`semi_analytic_velocity_adiabatic_slow` at exactly $T(0^+)=0$ and $\xi=-0.85$.
It uses massless quarks, QMC-RMF3 PNM and `B_one_forth=189.1565957288247` MeV
from the original cold $3n_0$ calibration.
Each method evaluates an independent speed field without a spatial BVP and
recomputes its own physical exclusions.
The full analytical method retains the quadratic EOS, leading diffusion and
closed integral; the semi-analytical method uses the full constrained EOS,
full diffusion and numerical integral; see [[analytic-slow-front-solvers]].

The endpoint requires $D_K(0^+)=\infty$ and $D_K^{-1}(0^+)=0$, while $I_2$,
baryon flux and speed remain finite and positive.
The validator rejects a temperature floor and retains the relative
conservation threshold `1e-8`.
Both methods use the modified-Madsen weak-rate approximation with
`alpha_s=0.4`; the historical metadata label does not imply an exact
phase-space rate calculation.

## Domain and contour acceptance

The common display window is $1\le n_B(0^-)/n_0\le5.5$ and
$0\le T(0^-)\le115$ MeV, with requested speeds 50, 100, 200, 500, 1000, 2000
and 5000 m/s.
The frozen `inputs/cold-scan-seeds.json` supplies 749 coordinates and
calibration from the warm numerical calculation and extension in
[[nonrelativistic-adiabatic-contour]].
These coordinates seed sampling only: no warm speed, physical mask or
successful-solve flag becomes a new cold result.
The engine reads the bundled JSON, not the historical source payloads.

Each method traces radial crossings independently, adding radial samples,
refining physical transitions and checking angular midpoints.
Boundary bisection uses direct cold EOS classification from pressure and
enthalpy per baryon, without constructing a diffusion table or computing a
speed integral.
Near-boundary speed evaluations are added only as needed to bracket the
requested 50–5000 m/s levels.
A resolved root has relative speed error below `1e-3` and radial bracket width
at most `0.001`.
Physical stable-matter and composition-ceiling boundary brackets must be at
most `1e-4` wide.
Angular refinement checks midpoint displacement against the average of its
neighbors in window-normalized coordinates, with threshold `0.0025`, and
also resolves changes in contour/boundary status.
The default allows six angular refinement rounds; exhausted refinement is
reported explicitly rather than silently accepted.

Physical exclusions, numerical failures and levels outside the display window
are separate states.
Only adjacent accepted samples can bracket a root; a failed sample never
establishes a physical boundary or a contour connection.
Each individual plot's gray regions use that method's resolved cold
boundaries, not the historical numerical shading.

## Execution and durable records

The local and cluster entry points are `local/run_adiabatic_contours.py` and
`cluster/run_adiabatic_contours.py`.
Local default concurrency is four; the current production invocation uses
eight.
Cluster concurrency follows `SLURM_CPUS_PER_TASK`, `PBS_NP`, `NSLOTS`, CPU
affinity or CPU count, and `--workers` overrides either runner.
One BLAS thread is used per spawned method call.
The hard default deadline is 120 seconds per call, including its recovery
attempts; cancellation/timeout terminates the child.
Path-table failure alone triggers stricter semi-analytical tolerances;
the known PNM seed-agreement failure alone enables additional nuclear starts.

The four sibling `local/`, `cluster/`, `shared/` and `inputs/` folders must
remain at their current repository depth beside the existing `RMFsolver/`
package.
No other dated calculation is an engine runtime dependency.
The Mac-specific isolated SciPy installation used locally is an environment
choice, not a cluster dependency; use a cluster-native runtime.
Full commands and option defaults are in the two runner READMEs.

Every terminal method-point call is saved atomically under
`<output-stem>-records/`, and the `.npy` manifest is saved after each completed
ray and angular round.
The **historical** two-method scan manifest is now archived at
`local_analysis_results/26-10-26-second-order/notebook-refresh/adiabatic-T0plus0-xi-minus0p85-leading-density.npy`.
Its `record_directory` is still relative to the original `new_paper_calculations/26-10-26/local/` directory.
Its point records retain endpoints, integrals, attempts, source identity and
failure history and are permanent parts of the result.
Thermodynamic boundary records have a separate `boundary-` prefix,
`evaluation="thermodynamics_only"`, `interior` or `excluded` status, and
thermodynamic margins.
They use the same hard timeout as speed records; boundary failures enter the
retry and completion checks.
Rerunning the same command resumes; successful and excluded points are
reused, while a previously failed point is retried once per invocation.
`--smoke` uses a separate output; `--dry-run` makes no writes.

Resume compatibility includes exact physics, frozen input bytes, numerical
controls and hashes of the scan engine, physics adapter, bounded-call helper
and every `RMFsolver/*.py` file.
Cold/warm or source/settings mismatches are rejected; changing the worker
count is allowed.
Independent verification and plotting scripts are outside the scan physics
identity.

## Historical boundary-workflow recovery

Five preliminary calls hit the 120-second deadline extremely close to the
stable-matter boundary while the semi-analytical diffusion table stalled in
`_tabulate` at vanishing $I_2$.
The direct EOS classifier replaces that unnecessary table construction for
boundary bisection; the speed APIs and physical acceptance thresholds remain
unchanged.
Ten live method/state checks confirm the classifier against the direct
thermodynamic domains without building the full-EOS diffusion table.

The one-time migration reused 2,424 valid point records: 1,898 accepted and
526 physically excluded.
Before reuse, the original physics module was verified as an identical AST
prefix of the updated module, with matching EOS and frozen-input hashes.
Reused records retain their original `calculation_identity` and an explicit
`reuse_verification` explanation; five obsolete boundary failures are archived
separately.
The audit is
`local_analysis_results/26-10-26/cold-boundary-workflow-recovery.json`.
This audited migration does not weaken normal source-mismatch rejection.
Production completed with eight workers; both method coverage checks pass.

Production coverage must be inspected in the saved manifest: require both
method keys in `coverage`, `complete=True`, no pending angular intervals, and the per-method
`coverage` report for the two-method deliverable.
`supported_levels` denotes at least one resolved crossing, not coverage on
every ray; `outside_window_levels` is distinct from unresolved computation.
Exit status is 0 for complete selected-method coverage and 2 otherwise.
The verified representative states below do not by themselves establish
final production coverage.

## Historical physics verification, before the endpoint correction

`shared/verify_cold_physics.py` runs independently of the contour engine.
Its report is
`local_analysis_results/26-10-26/cold-analytic-physics-verification.json`, with
`passed=true` for all three states and the 11 analytical slow-front regression
tests.
The separate 25-test orchestration suite passes with `COLD_CONTOUR_LIVE=1`
and the isolated SciPy runtime.
It covers bounded calls, cancellation,
atomic checkpoints, compatibility, independent methods, root/mask handling
and dry-run behavior, including ten live classifier cases, separate boundary
records, boundary-failure retries, and a regression proving that boundary
bisection avoids nearly zero-speed integral calls.

| $T(0^-)$ [MeV] | $n_B(0^-)/n_0$ | Full analytical speed [m/s] | Semi-analytical speed [m/s] |
|---:|---:|---:|---:|
| 0 | 3.5 | 99.69158696 | 217.66626786 |
| 30 | 3.5 | 128.40217106 | 301.44998513 |
| 70 | 2.5 | 150.53995318 | 371.91044265 |

Direct quadratic integral quadrature agrees to `1.01e-15` relative or better.
The full-EOS checks recompute pressure, enthalpy per baryon, constant density,
the local rate and inverse diffusion at 65 path samples.
The maximum semi-analytical conservation error is `2.93e-11`;
the direct-rate integral differs by at most `3.01e-8` relative.
Halving `path_rtol` from `1e-6` to `5e-7` changes speed by at most `4.67e-9`
relative.
This direct-rate quadrature bypasses the stored rate spline but reuses the
existing diffusion primitive; it is not an independent EOS-path construction.

⚠ Warm-to-cold convergence is fractional-power and slow.
Reducing $T(0^+)$ from 0.01 to $10^{-5}$ MeV reduces the speed discrepancy
from about 3.4–4.0% to about 0.34–0.40% at these states, approximately a factor
of ten rather than a factor of one thousand.
No tiny warm/cold equality tolerance or positive temperature floor is used.
These checks establish representative-state consistency, not a universal
accuracy bound or completed contour coverage.

## Notebook comparison (corrected full-analytical data)

`shared/add_cold_figures.py` adds its tagged section to `26-10-26.ipynb`,
preserving the pre-existing cells.
It supplies three figures: full analytical, semi-analytical, and both methods
overlaid with the existing numerical reference.
The loading cell requires a complete cold manifest; the figure cells load
saved calculations and never solve an EOS or front.
The loader checks that the analytical branch uses `quadratic_pressure_second_order_density` and that both method-specific identities have the same physical settings.
The authorized refresh script and preservation checks are in `local_analysis_results/26-10-26-second-order/notebook-refresh/refresh.py` and `verification.json`.
Every figure cell starts with its tunable plotting-parameter block, and the
exports go to `local_analysis_results/26-10-26/` as PNG and PDF.
Unsupported or unresolved contour portions are not extrapolated, and the
combined overlay does not select one analytical method's gray mask.

⚠ The numerical reference remains at $T(0^+)=0.1$ MeV, whereas both analytical
curves are at exactly zero.
Their differences include interface temperature and analytical approximation;
the overlay is not a comparison at identical interface conditions.
See [[nonrelativistic-adiabatic-contour]] for the numerical reference and
[[adiabatic-xi-diagnostics]] for the origin of the chosen shape factor.

## Historical completed local result (before endpoint correction)

The saved two-method manifest is complete: 101 full-analytical rays and 61
semi-analytical rays, with zero failed samples, unresolved contours or
unresolved boundaries. It retains 3,357 accepted full-analytical speed records
and 2,923 accepted semi-analytical speed records, plus physical exclusions and
separate thermodynamic boundary checks.

The full analytical method supports 50, 100, 200, 500 and 1000 m/s in the
chosen window; its 2000 and 5000 m/s levels lie outside the window. The
semi-analytical method supports all seven levels through 5000 m/s.
The maximum pressure, enthalpy-per-baryon and constant-density errors are
`1.818e-9`, `2.149e-10` and `7.742e-10`, below the unchanged `1e-8` gate.
The largest level residual is `9.993e-4`, radial bracket width `9.994e-4`,
and accepted angular midpoint displacement `0.001598`, satisfying their
respective `1e-3`, `0.001` and `0.0025` criteria.

All 36 targeted tests pass (25 orchestration/classifier checks and 11
analytical physics regressions). Local and cluster smoke results agree
bit-for-bit. The final saved-data audit is
`local_analysis_results/26-10-26/cold-contour-verification.json`.
The three new notebook figures are executed and exported as PNG/PDF; the
original seven notebook cells and numerical artifacts are preserved.

The initial five-round angular budget needed a sixth round to certify
where curves cross the plot window. Six is now the default. The budget
extension reused 8,533 records after verifying that the only source change
was the default round count, with unchanged physics and inputs; records
retain their calculation identity and sampling-budget provenance.
`local_analysis_results/26-10-26/cold-angular-budget-extension.json`
records that check. The final resumed invocation took 262.4 seconds; this
is only the last refinement stage, not the total calculation time.
