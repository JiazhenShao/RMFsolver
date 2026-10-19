# Isothermal mean-free-path overlay

Open `26-10-14.ipynb` at the project root. Its first three code cells implement the tail comparison below; its final two code cells implement the separate core criterion documented at the end.
The first loads the existing analytical curves, warm numerical curves, and accepted cold numerical dots; it does not calculate velocities.
The second calculates static downstream thermodynamics and searches for the chosen length ratio.
The third only plots, with all style controls at its top. Changing the figure does not rerun physics.

## Definition and controls

The condition is `lambda_mfp = MFP_FRACTION * L_conv`, with:

```text
lambda_mfp = 3*D_K
L_conv = muB_inf/(9*pi*T)*sqrt(D_K/gamma_K)
lambda_mfp/L_conv = 27*pi*T/muB_inf*sqrt(D_K*gamma_K)
```

Both lengths use the same static, pressure-matched equilibrated quark endpoint:
`T_inf = T_0minus`, `P_inf = P_0minus`, `muK_inf = 0`, `jB = 0`.
The upstream state is obtained directly at fixed density with the validated PNM helper, never a nuclear chemical-potential scan.
The full diffusion coefficient is the default; `DIFFUSION_APPROXIMATION = 'landau'` drops its screened longitudinal contribution consistently from both lengths.
Constants match the manuscript and current solver; worker-side comparisons reject a coefficient mismatch.
The thickness uses its coefficient formula, not the rounded 0.2-micrometer normalization.

`MFP_FRACTION = 1.0` means 100%; use `0.1` for 10%.
Neither is expected to cross in the current domain. This is reported explicitly rather than drawing a fictitious line.
Only fractions of order one millionth can be comparable to this estimate.
At fixed chemical potential in the Landau limit, the ratio is proportional to `T**(1/6)` and goes to zero, even though both lengths individually diverge.
This tail comparison is **not** the appendix's local-gradient LTE condition `lambda_mfp < D_K/abs(dD_K/dx)`.

The default schedule is the existing 46 temperatures from 0.01 to 120 MeV.
At each slice, use the saved allowed density interval clipped to `1 <= nB_0minus/n0 <= 5.5`.
There are no EOS evaluations at temperatures where that interval is absent.
`DENSITY_SAMPLES = 2` tests the interval endpoints; increase it to sample the interior before choosing a bracket.
The static equilibrated endpoint is nonsingular at the composition boundary: no moving BVP is attempted there.
Safeguarded secants use only evaluated ratios inside a sign-changing bracket; acceptance requires relative ratio error at most `1e-6` and validated EOS states.
Never extrapolate a crossing or connect through an unsuccessful temperature slice.

Execution is deliberately serial with one BLAS thread per bounded subprocess.
Controls expose a 60-second EOS-call timeout, 180 seconds per temperature, and at most 40 evaluations per slice (including cache accesses).
An ordinary no-crossing sweep needs just two endpoint evaluations per nonempty temperature slice, not a density-temperature grid or any front solves.
The worker reuses the existing isolated SciPy runtime when present; it does not install dependencies or change the environment.
A runtime import failure stops preflight before reserving physical states.

## Checkpoints and plotting partial results

Only new files under `local_analysis_results/26-10-14/` are written:

- `mfp-static-tail/thermodynamics.npy`: fraction-independent endpoint states, both diffusion choices, EOS residuals, warnings, and runtime provenance.
- `mfp-static-tail/overlay-<model>-f<fraction>-<controls-hash>.npy`: separate root/no-crossing results for each fraction/model/schedule/control combination.
- `mfp-static-tail/latest-overlay.npy`: convenient pointer payload for the last selected calculation, not an independent dataset.
- `isothermal-contours-mfp-tail.png` and `.pdf`: figure exports.
- `verification.npy`: bounded local-check record and preservation hashes.

Each EOS reservation and completed evaluation is saved atomically. A completed evaluation is saved **before** the evaluation progress bar increments; the root result is saved before the temperature progress bar increments.
Changing only the fraction or diffusion model reuses thermodynamics. A source, physics, or domain mismatch rejects the cache rather than mixing results.
Finished slices, including failed and interrupted ones, are retained on resume; failures are not silently retried. Choose a new `MFP_RUN_DIRECTORY` for an intentional retry.
`success`, `no_crossing`, `outside_domain`, failed EOS/boundary, interruption, and budget exhaustion remain distinct diagnostics.
Only `success` rows satisfying the ratio tolerance are plotted. Isolated roots get a marker; a failed or missing row always breaks a line.
With no validated roots, neither a black overlay nor its legend is added.

For plotting an existing result without further calculation, load its specific saved payload as `mfp_result` after the loading cell, then run the visualization cell:

```python
mfp_result = np.load('/absolute/path/to/overlay-result.npy', allow_pickle=True).item()
isothermal_mfp.report(mfp_result)
```

The original `26-10-02.ipynb`, all earlier data, and every `RMFsolver` source file remain unchanged.

## Verification scope

Local verification used mocked root-search/cache tests, exact artist-array comparisons against the original plot (including numerical dots and shading), and six live EOS states: both density endpoints at 0.01, 20, and 83.85925425258951 MeV.
All six passed validation; maximum relative pressure mismatch was `7.0e-13`.
The full-diffusion ratio ranged from `2.72603e-7` to `1.11485e-6`; leading-Landau ratios ranged from `2.78781e-7` to `1.38497e-6`.
Those three slices had no crossing at fractions 1 and 0.1; switching fractions/models required no additional EOS evaluations.
The complete 46-temperature assessment is **not pre-run**. The notebook starts with a saved-data-only figure preview; run its calculation cell to finish that assessment, reusing the six cached states.
No numerical BVP was run.

Run the lightweight tests with:

```bash
./bin/python3 -m unittest discover -s tests -p 'test_isothermal_mfp_26_10_14.py'
```

## Standalone core boundary at fraction 1e-4

The notebook section **Standalone core boundary** has an independent calculation/resume cell followed by a solver-free visualization cell. Run those two cells alone; the earlier tail calculation is not required.

```text
L_core = sqrt(D_K(0+)/gamma_K(0+))/a(0+)
lambda_mfp/L_core = 3*a(0+)*sqrt(D_K(0+)*gamma_K(0+)) = 1e-4
```

This is the approximate nonlinear reaction-core length, not the equilibrated downstream tail. All coefficients are evaluated at the maximum-composition static interface obtained from pressure matching and `muB_0plus + a_0plus*muK_0plus = muB_0minus`. Nuclear states are recovered directly at fixed density. No moving-front or numerical BVP calculation is performed.

`isothermal_core_mfp.py` searches temperature at each density. The cold power law only proposes a starting temperature; actual EOS/transport evaluations and a safeguarded log-temperature root correction determine accepted roots. Full diffusion is the default. Exposed controls include fraction, density schedule, temperature limits, diffusion approximation, tolerances, 60-second point timeout, maximum 16 EOS evaluations per point, and two workers with one BLAS thread each.

The completed first run has 33/33 successful density points, using 226 EOS evaluations. Sampling is denser near coexistence. Temperatures range from `0.0005150551` MeV at density ratio `3.000099948` to a sampled maximum of `0.1715029` MeV. Maximum relative criterion error is `5.63e-7`; maximum pressure and composition residuals are below `3e-13`, and the Gibbs residual is below `4.1e-11` MeV. The largest `eta/a_0plus**2` is `1.93e-5`, consistent with cubic-core dominance. One root retains a quadrature warning; its validated upstream residuals remain small and the warning is saved rather than suppressed.

New outputs, separate from the unchanged tail cache:

- `local_analysis_results/26-10-14/mfp-core-f1e-4.npy`: roots, all completed root-search evaluations, EOS residuals, warnings, controls, and source identity.
- `local_analysis_results/26-10-14/isothermal-core-mfp-f1e-4.png` and `.pdf`: standalone curve with logarithmic temperature axis from `1e-4` to `1` MeV.

The standalone figure now also shades stable neutron matter on the low-density side and the unit-composition excluded region on the high-density side. `isothermal_core_regions.py` reuses 19 saved boundary slices and directly solves five missing slices (four below 0.01 MeV and the 1-MeV endpoint), giving coverage over the complete plotted window without extrapolation. All 24 slices succeeded; the maximum new chemical-potential residual is `1.01e-9` MeV, with no new quadrature warnings. The upper boundary keeps the established `a_0plus=1-1e-9` convention.

`core-static-regions.npy` stores these separate boundary data. Their calculation is appended to the core calculation cell, with tqdm and save-before-progress; plotting uses only saved values. The 33 original core roots are unchanged. Shading colors, boundary lines, and both region labels are tunable at the top of the visualization cell. If changing the plotted temperature window, also extend the region calculation schedule; failed boundary slices are not bridged.

Each completed density point and its trial states are atomically saved before tqdm advances. Matching checkpoints resume without recomputing finished points; incompatible physics, schedules, controls, or source hashes are rejected. Choose a new output filename after changing those inputs. Failed points break plotted lines; no artificial zero-temperature endpoint is added.

The cold side exceeds the chosen `1e-4` tolerance. This is a deliberately stringent scale-separation diagnostic, not a demonstrated sharp LTE breakdown boundary or a universal 1-keV cutoff.

Run both lightweight suites with:

```bash
./bin/python3 -m unittest discover -s tests -p 'test_isothermal*mfp*.py'
```

## Historical momentum-flux profile figure

Cells 9 and 10 of `26-10-14.ipynb` reproduce the figure originally produced by cell 9 of `26-04-09.ipynb` (zero-based indices).
The old notebook's stored figure uses interface temperature 0.10 MeV and downstream temperature 17.34 MeV, but its upstream calculation cell was subsequently changed to another run.
No matching raw profile was located, so the original curves and sampled points are recovered from `jargon/entropy_tqguess_uagrad_pi_qm_contour.pdf` using its vector paths, not raster digitization.
This is a historical EOS/solver result, not a new run with the current EOS.

`recover_profile_momentum_figure.py` uses the global Python's PyMuPDF to recreate the portable coordinate cache:

```bash
/usr/local/bin/python3 new_paper_calculations/26-10-14/recover_profile_momentum_figure.py
```

`profile_momentum_plot.py` is the source copied into the notebook plotting cell; `profile_momentum_plot.md` supplies its provenance cell.
The cell needs only NumPy, Matplotlib, and `local_analysis_results/26-10-14/constant-momentum-profile-data.json`.
It has local tuning controls and exports PNG/PDF under `local_analysis_results/26-10-14/`, plus the PNG embedded in `26-10-14.md` under `26-10-14.assets/`.
For a noninteractive replot from the repository root:

```bash
MPLBACKEND=Agg ./bin/python3 new_paper_calculations/26-10-14/profile_momentum_plot.py
```

The recovered endpoint change is about 0.073 MeV (0.0064%); these are vector-figure precision values for one example, not a uniform error bound or a new full-profile numerical verification.
Endpoint labels use only location notation.

```bash
./bin/python3 -m unittest discover -s tests -p 'test_profile_momentum_plot_26_10_14.py' -v
```

## Isothermal ceiling accuracy

Cells 11 and 12 compare Eq. (35) with the exact neutral-EOS ceiling using the verified 292-state audit under `local_analysis_results/26-10-16/aiso-accuracy/`.
`aiso_accuracy_plot.md` supplies the explanation cell and `aiso_accuracy_plot.py` is copied into the plotting cell, including its local tuning block.
The plot uses exact composition on the horizontal axis, with three panels for values, signed difference, and percentage difference relative to exact.
Temperature colors distinguish saved states; the continuous dashed curve is the scale-independent cold limit, not a finite-temperature interpolation.
No EOS or BVP solve is run by the plotting cell.

```bash
MPLBACKEND=Agg ./bin/python3 new_paper_calculations/26-10-14/aiso_accuracy_plot.py
```

This exports `aiso-accuracy.png` and `.pdf` under `local_analysis_results/26-10-14/` and copies the PNG to `26-10-14.assets/` for the report.
The source audit can be regenerated separately with `./bin/python3 local_analysis_results/26-10-16/aiso-accuracy/audit.py`.
The report identifies the exact denominator and limits the 113/292 count and 8.48% maximum to the checked checkpoint.
