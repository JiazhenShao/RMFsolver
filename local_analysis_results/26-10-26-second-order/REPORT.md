# Consistent second-order adiabatic endpoint

The full analytical endpoint now matches quadratic pressure and equilibrium heat at the same expansion order.
Only `analytic_velocity_adiabatic_slow` changed inside the live `RMFsolver/phase_velocity.py`; all 115 other top-level definitions match the pre-task snapshot.
The exact patch is `phase_velocity.diff` beside this report.
No commit or push was made.

## Implementation

The solver selects the unique scaled quartic root below the cold chemical potential, rejecting `C**0.25>=H` with the existing `no_adiabatic_equilibrium` status.
The corrected temperature relation has `9*pi**2` in its denominator and the corrected ceiling is `3/sqrt(2)*sqrt((H-muB)/muB)`.
The conserved density remains `4*(P+bag**4)/H`.
The original interface ellipse, composition/temperature restrictions, closed integral, weak coefficient, diffusion, and speed closure remain unchanged.
The semi-analytical, isothermal, spatial BVP, and older APIs are untouched.

`quadratic_endpoint_residuals` contains signed relative quartic, pressure, heat, and Eq. (a-max) agreement errors, each gated at `1e-10`.
Both quadratic endpoint pressures retain the `1e-8` check.
The enthalpy conservation error now checks the expanded heat relation rather than the identity produced by dividing enthalpy by the conserved density.
The derivative density and unexpanded enthalpy-ratio error are diagnostics only.
Both EOS metadata values now read `quadratic_pressure_second_order_density`.

The contour adapter's analytical domain classifier was necessarily updated to the same quartic endpoint and corrected composition ceiling; its semi-analytical branch is unchanged.
This change is outside `RMFsolver/` and prevents the new scan from retaining the old analytical mask.

## Reproduced validation table

All rows use massless QMC-RMF3 PNM, `B_one_forth=189.1565957288247`, `xi=-0.85`, and exactly `T_0plus=0`.
The independent corrected references were computed and saved before the source was changed.
The old and semi-analytical results were reproduced before the patch; the corrected and first three BVP results were reproduced afterward.

| Upstream temperature [MeV] | Density / saturation | New chemical potential [MeV] | New equilibrium temperature [MeV] | New ceiling | Old speed | New speed | Semi speed | Matched cold BVP speed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8.860992751744538 | 3.0268624251597 | 1190.798571090 | 9.306583545 | 0.156253358 | 19.590121 | 39.654712 | 35.961409 | 34.887051 |
| 27.83373842080244 | 2.96875 | 1181.401508786 | 17.327404487 | 0.293233360 | 44.589022 | 96.374871 | 85.989856 | 81.634275 |
| 70.6735943750199 | 1.434488038438507 | 1089.243288810 | 25.555560332 | 0.469070261 | 126.105514 | 304.474721 | 267.958807 | 250.418688 |
| 10.15113868209788 | 4.500181704610503 | 1265.311487838 | 54.774539690 | 0.865483255 | 289.996845 | 1372.386997 | 1054.417203 | Previously reported unavailable; not rerun |

Speeds are in metres per second.
The new-to-semi speed ratios are approximately 1.103, 1.121, 1.136, and 1.302.
The first three ceiling differences from the full EOS are 0.0243%, 0.1002%, and 0.3370%; the fourth is 2.2875%.
The fourth unexpanded enthalpy-ratio diagnostic is -2.0364%, larger than the request's approximate “few parts in a thousand” estimate, and is correctly excluded from acceptance checks.
The old notebook numerical overlay uses `T_0plus=0.1`, so it was not used as a matched-temperature validation target.

Detailed numerical inputs, outputs and diagnostics are in `endpoint-before-and-independent-targets.json` and `endpoint-after-validation.json`.
`validate_corrected_endpoints.py` reproduces the latter without touching old artifacts.

## Tests and changed expectations

The original 11 analytical tests passed before the patch.
The expanded 14-test suite first failed against the old source, then passed against the correction.
The original 11 frozen test identifiers also pass afterward with the documented corrected expectations.
The 25 contour orchestration/classifier tests pass, including local/cluster behavior, timeout, resume, masks and live domain classification.
Independent temperature-coordinate quadrature agrees with closed `I2` within `3.6e-15` at interface temperatures 0, 1 and 5 MeV, exceeding the requested `1e-12` tolerance.
No acceptance tolerance was relaxed.

The historical 60-test selection has identical before/after results: 55 pass, one failure and four errors.
Four pre-existing synthetic analytical cases raise `SlowFrontNoSolution` with the same 42.0272 MeV enthalpy gap; the fifth is the existing missing expected exception in `test_analytic_velocity_validates_xi_endpoint_and_density_domain`.
Exact test IDs, tracebacks, source hashes and execution times are in `baseline_historical_60.json` and `after_historical_60.json`.

The unchanged 16-test spatial BVP selection was run before and after with a 180-second limit per test.
Before: 14 pass and two cold-test timeouts.
After: 13 pass and the same two cold-test timeouts plus a warm massive-branch timeout under concurrent load.
That warm test passes on its 600-second retry (238.4 seconds), resolving the only additional short-budget timeout.
There are no new assertion failures; the cold timeout cases are recorded separately from physics failures and do not support a claim that the complete BVP suite passed.
The massive cold table/mesh/tail test passes its 600-second retry both before (350.2 seconds) and after (338.8 seconds).
The cold seed-independence test still reaches the 600-second limit in both versions.
Combining the original selection and bounded retries gives 15 passing BVP test methods and one shared unresolved runtime limit, with no new assertion failure.
`regression-comparison.json` records the exact frozen selections, source identities, status differences and retry outcomes.

Every changed analytical assertion and input, including exact old/new values, appears in `test-expectation-changes.md`.
The main warm reference changes from equilibrium temperature `18.4171319423` to `30.9370395272`, ceiling `0.3228643376` to `0.5536383788`, and flux `0.674248479176` to `1.781723291935`.
The only additional changed assertion is in `test_cold_adiabatic_contours.py`: the cold state at density `5.4*n0` was previously classified as analytical `interior`; it is now `excluded` with `composition_ceiling`, agreeing with the corrected direct API.
The semi-analytical classification there remains excluded.

## Manuscript and preservation

The insertion immediately after Eq. (a-max) contains the two requested matching relations and explains why the leading density is insufficient for heat matching.
All inserted material is wrapped in `\JS{...}`; the prose was reviewed with the humanizer skill.
Both isolated manuscript builds pass with 17 pages and no new warnings.
All 64 existing numbered equation environments, all 100 label values, the speed formula and both 82 m/s prefactors are unchanged.
The inserted page was rendered and visually inspected.
See `manuscript-build-humanizer-report.md` and `manuscript-build-verification.json` for commands, warning comparison and exact insertion.

All 120 tracked manuscript build artifacts match their pre-task hashes.
An initial BibTeX output-directory side effect rewrote `Main.bbl`, which was restored byte-for-byte after confirming its baseline hash; the untracked `Main.blg` log was regenerated.
Subsequent builds used fully isolated source copies.
The detailed manuscript report records this side effect.
`git diff --check` reports an existing trailing space at `Latex_writting/Main.tex:734`, outside the insertion; it was left untouched.

Existing notebook, calculation manifests, record directories, figures and other live RMFsolver sources are protected by the 10,015-file pre-task hash manifest.
No figure regeneration or notebook execution was performed.

## Completed contour and final status

The new output is `new_paper_calculations/26-10-26/local/adiabatic-T0plus0-xi-minus0p85-second-order.npy`.
Its sibling `adiabatic-T0plus0-xi-minus0p85-second-order-records/` directory is part of the result.
The scan completes 61 rays and all seven levels, 50, 100, 200, 500, 1000, 2000 and 5000 m/s, with no failed samples, unresolved crossings, unresolved boundaries or pending angular intervals.
The 4,602 fresh records comprise 2,732 accepted speeds, 787 physically excluded speed evaluations, 509 interior boundary checks and 574 excluded boundary checks.
Runtime was 1,074.5958 seconds with eight workers, approximately 17.9 minutes.

The full independent audit passes (`contour-audit.json`).
The largest level residual is `0.0009981248214`, radial bracket width `0.0009862786326`, and accepted angular error `0.001009212923`, satisfying all original thresholds.
The largest endpoint residual is `9.10383e-14`, and the largest pressure conservation error is `2.60725e-15`.
Every new record has the corrected identity; no original analytical record was reused.
The solver hashes are:

```text
before: 8bfdd9c23538e432e43f6b9898375a0c51b1f29c814eb8e945994ec56136c9d1
after:  3c22ccbb039bcc0b7812daeed4c2e233cfdf09c8d5f650a2160c415bc494140b
```

The final protected-file audit is `protected-artifacts-after.json`; the final working-tree listing is `git-status-final.txt`.
That listing includes pre-existing changes from other sessions and the already-dirty nested repository; neither was cleaned, reverted, committed or pushed.

## Numerical resolution limitation

The requested relative Eq. (a-max) cross-check uses `C-muB**4`, which loses significant digits arbitrarily close to coexistence.
An independent review reproduced a synthetic endpoint at equilibrium temperature 0.0402634 MeV where the relative ceiling disagreement is `7.24e-10`, so the solver rejects it under the required `1e-10` gate.
This is an explicitly failed numerical check, not an accepted inaccurate state; the gate was not weakened.
The four validation states and resolved production contours are far from that cancellation regime.
The final contour audit reports the smallest equilibrium temperature and relative heat gap among accepted speeds at least 50 m/s.

## Reproduction

```sh
export PYTHONPATH="$PWD/local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:$PWD:$PWD/tests"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/private/tmp/nsm-rate-mpl
./bin/python3 -B -m unittest -v test_phase_velocity_analytic_slow
COLD_CONTOUR_LIVE=1 ./bin/python3 -B -m unittest -v test_cold_adiabatic_contours
./bin/python3 -B local_analysis_results/26-10-26-second-order/validate_corrected_endpoints.py
./bin/python3 -B new_paper_calculations/26-10-26/local/run_adiabatic_contours.py \
  --method analytic --workers 8 \
  --output new_paper_calculations/26-10-26/local/adiabatic-T0plus0-xi-minus0p85-second-order.npy
./bin/python3 -B local_analysis_results/26-10-26-second-order/audit_corrected_contour.py
```

The baseline reproduction script writes the frozen before-change reference file and must not be rerun against the corrected source.
The new contour invocation resumes only compatible second-order records; it does not reuse the original analytical records.

## Subsequent authorized notebook refresh

The user subsequently requested replacing the old full-analytical results in `26-10-26.ipynb`. The canonical combined plotting manifest now contains the corrected analytical branch and unchanged semi-analytical branch, with separate per-method provenance. Notebook cells 8, 10 and 14 were executed, and the full-analytical and comparison-overlay PNG/PDF exports were overwritten. Numerical and semi-analytical cells/exports were preserved.
The earlier preservation report above describes the state before this separately authorized refresh. Its archive, execution script and verification report are in `notebook-refresh/`. The standalone corrected scan and original point-record directories remain unchanged.
