# Massless Nonleptonic Weak Rate

This standalone calculation derives the unequal-flavor rate for `d + u -> s + u`, including exact cold phase space, the thermal coefficient through `T^2`, and the leading `T^3` remainder.
It does not modify the combustion manuscript or production solver.
The adiabatic upstream module imports the existing branch-validated QMC-RMF3 EOS read-only; the phase-space and conserved-quark-trajectory calculations remain standalone.

## Files

- `Main.tex`, `Main.pdf`: single-column, 11-point derivation manuscript.
- `cold-result.tex`, `thermal-result.tex`, `collected-results.tex`, `numerical-results.tex`, `speed-sensitivity-results.tex`, `algebra-appendix.tex`: manuscript inputs.
- `references.bib`: local bibliography, checked against the original Madsen PDF at `../Rate-PhysRevD.47.325.pdf` (Phys. Rev. D 47, 325, DOI 10.1103/PhysRevD.47.325).
- `verification.wl`, `spin-trace.wl`: reproducible Wolfram algebra and explicit Dirac-trace checks.
- `wolfram-transcript.md`: complete successful Wolfram MCP input and returned output, dated 2026-10-07.
- `cold_sympy.py`, `thermal_sympy.py`: independently reproduced polynomial integrals and numerically stable coefficient functions.
- `validate_rate.py`, `test_validate_rate.py`: finite-temperature reference, independent energy-space quadrature, neutral EOS mapping, and tests.
- `speed_sensitivity.py`, `test_speed_sensitivity.py`: three-model rate-integral speed estimates, with independent quadrature refinements and benchmark tests.
- `cold-derivation-notes.md`, `thermal-derivation-notes.md`: supporting analytical working notes.
- `adiabatic-section.tex`, `adiabatic-results.tex`, `results/adiabatic-verification.tex`: Section 8.6 and generated results.
- `run_adiabatic.py`: complete adiabatic calculation driver; component scripts are documented below.
- `plot_manuscript.py`, `plot_style.py`, `rate_figures.py`: redraw all seven publication figures from saved data with shared typography and model styling.
- `audit_adiabatic_physical.py`: independently reconstruct and integrate six dimensional states to check the repeated adiabatic ratios without rescaling cached integrals; saves `results/adiabatic-physical-audit.json`.
- `wolfram-adiabatic-transcript.md`: symbolic conservation, Jacobian, cold-integrand and scaling verification.
- `results/`: numerical states, trajectories, integrals, convergence checks, and seven comparison figures in PDF and PNG formats.

The repository's existing `.gitignore` ignores `ref-list/`; these files exist on disk but are not automatically included by an ordinary `git add .`.
No ignore rules were changed and no files were staged.

## Calculation Assumptions

Use natural units and metric `+---`.
All quark masses are exactly zero; the weak interaction is tree level, with no QCD corrections, pairing, or additional antiquark reaction channels.
Each flavor has a Fermi-Dirac distribution at a common temperature.
The positive net relaxation rate is forward minus reverse, with `mu_K = mu_d - mu_s` and `n_K = (n_d - n_s)/2`.
The analytical branches require `0 <= mu_s < mu_u < mu_d`; the finite-temperature integral itself has a broader domain.
The cold branch switches at `mu_K = mu_u`, corresponding to neutral cold `a = 0.983149408702098`.
The net EOS densities are `n_f = mu_f^3/pi^2 + mu_f*T^2`; their antiparticle contribution does not supply the omitted antiquark collision channels.
Neutrality without electrons is `2*n_u = n_d + n_s`, with `mu_B = mu_u + mu_d + mu_s`.

The fixed-flavor expansion is not the same as expansion at fixed `mu_B,mu_K`.
The latter includes temperature-induced shifts of all three flavor potentials and the corresponding cold-rate chain rule.
The incoming and outgoing up thresholds must be differentiated independently before setting them equal.
Their coincident physical values produce a nonanalytic cubic thermal correction: truncation after `T^2` has an `O(T^3)` remainder, not an `O(T^4)` remainder.
The strict positive-imbalance series is not uniform as other flavor gaps vanish; the manuscript separately derives the leading joint near-equilibrium formula.
At zero strange density, endpoint power counting preserves the displayed quadratic and cubic coefficients, but not an unrestricted higher-order Sommerfeld series.

## Build

From this directory, with a TeX installation containing `latexmk` and BibTeX:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error Main.tex
```

The build uses only the local bibliography and the saved PDF figures in `results/`.
On this machine, `latexmk` is `/Library/TeX/texbin/latexmk`.
The manuscript includes four isothermal temperature panels in Figure 4 and the conserved adiabatic trajectory calculation in Section 8.6.
Rebuild after regenerating the result inputs and figures.
The revised 2026-10-11 build has 29 pages, no LaTeX warnings or overfull boxes, and visually inspected rendered pages.
Figure canvases are 6.5 inches wide, matching the manuscript text width, with ordinary text at least 12 PDF points (the manuscript body is 11 TeX points).
The six-panel figures use two columns to retain readable labels; model curves share circle/square/triangle markers, and blue endpoint labels identify the last computed Eq. (23) speed ratio.
`results/plot-style-verification.json` records the seven figure widths, minimum text sizes, and unchanged numerical-data hashes.

## Reproduce the Calculations

The local phase-space and isothermal calculations need Python 3 with NumPy, SymPy, and Matplotlib.
The adiabatic upstream scan additionally needs the repository RMF package and SciPy, and independent verification uses mpmath.
On this machine, `adiabatic_upstream.py` automatically selects the working SciPy sidecar under `local_analysis_results/26-10-02/runtime/`; it does not change the environment or solver files.
The recorded run used Python 3.12.8, NumPy 2.2.3, SymPy 1.14.0, and Matplotlib 3.10.1.
From this directory, the repository interpreter is `../../bin/python3`.
Equivalent commands with an appropriately configured `python3` are:

```bash
python3 -m unittest -v test_validate_rate.py test_speed_sensitivity.py
python3 cold_sympy.py
python3 thermal_sympy.py
python3 validate_rate.py
python3 speed_sensitivity.py
```

The local-rate run regenerates the 192-state grid and three figures in `results/` and took about 200 seconds on the development machine.
The speed-sensitivity script separately integrates all three rate models over composition, checks convergence, and writes its figure, CSV files, JSON summary, and LaTeX table.
To redraw the figures from saved data without repeating the phase-space integrations:

```bash
python3 plot_manuscript.py
```

Use this redraw command after regenerating numerical grids as well.
The original local-rate plotting routine remains in `validate_rate.py` for numerical-source provenance; `rate_figures.py` supplies the manuscript's current typography without changing the hashed collision-integral source.
The three local-rate figures (`exact-rates`, `approximation-errors-muB-{900,1200}`) add a fourth, plotting-only model defined in `rate_figures.py`: Madsen Eq. (23) with `mu_u**5` replaced by `(mu_B/3)**5`, evaluated from the saved chemical potentials. `validate_rate.py` and its three shared styles are unchanged, so the quadrature source hashes still match.
The same model is added to `speed-sensitivity.pdf` and `adiabatic-trajectory-diagnostics.pdf`. For the isothermal figure, `speed_sensitivity.closed_form_integrals` integrates it on the saved outer quadrature (it reproduces the saved Eq. (23) integrals to 2e-16) and writes `results/speed-sensitivity-eq23-muB.csv`; `speed-sensitivity.csv` is unchanged. For the trajectory diagnostic it rescales the saved per-node Eq. (23) integrand by `((mu_u+mu_d+mu_s)/(3 mu_u))**5`, and the lower-left panel now shows each integrand relative to the exact one in percent (symlog), i.e. the local rate error along the path.

For a shorter diagnostic run without overwriting the full grid summary, use a separate output directory:

```bash
python3 validate_rate.py --quick --output /tmp/massless-rate-quick
```

With a licensed local Wolfram kernel:

```bash
wolframscript -file verification.wl
```

The supplied transcript was executed through the connected Wolfram evaluator, not a local `wolframscript` executable.
For a remote kernel, evaluate `spinCheck = Get[...]` only if the file has been uploaded; otherwise evaluate the complete embedded input in `wolfram-transcript.md`.

## Numerical Evidence

`results/rate-comparison-grid.csv` contains 192 states over both EOS paths, `mu_B = 900,1200 MeV`, six compositions, and `T = 0.01,0.1,1,5,10,25,50,100 MeV`.
All grid approximations use the exact finite-temperature EOS chemical potentials at the same state; this is a local prescription rather than a strict EOS Taylor series beyond `T^2`.
`results/eos-path-coefficients.csv` separately tests the strict constrained `T^2` expansion.
Rates are in `MeV^4`; unprefactored phase-space integrals are in `MeV^8`.
The low-temperature residual file sets `mu_u = 1` and reports dimensionless scaled quantities.
Relative error columns are fractions, not percentages.

## Reading the Plots

Every comparison figure uses the same three models:

- Black solid: **Exact phase-space integral**, the numerically integrated tree-level quark-channel rate without a temperature expansion.
- Blue dashed: **Madsen Eq. (23)**, the rate used by the current combustion calculation, evaluated at the same neutral flavor potentials.
- Orange dotted: **Derived T0+T2**, the exact cold term plus the derived quadratic thermal term evaluated at those same finite-temperature flavor potentials.

The third prescription includes the EOS shift through order T2, but is not the strict fixed-muB,muK Taylor polynomial beyond that order.
The T3 extension and Eq. (24) diagnostic are retained in the verification data, not in the plot legends.
The existing filename `exact-rates.pdf` now contains absolute rates for all four local-rate models, not just the exact rate.

`speed-sensitivity.pdf` compares the dimensionless estimate `sqrt(integral Gamma_model da / integral Gamma_exact da)` at fixed `muB=1200 MeV`, for the cold-rate limit and `T=20,50,100 MeV`.
The integration runs from equilibrium composition to `a(0+)`, recomputing the neutral chemical potentials at each composition.
The diffusion, baryon-density normalization, interface, and profile-shape factors are held fixed and cancel.
This is not a constant-pressure thermodynamic trajectory or a re-solved front; no absolute speed is inferred at T=0.
The exact and derived T0+T2 curves coincide in the cold panel.
The saved speed-ratio data are in `speed-sensitivity.csv`; separate convergence data and settings are in `speed-sensitivity-convergence.csv` and `speed-sensitivity-summary.json`.

## Verification Results

The saved checks give:

- All 34 unit tests pass; the suite covers the printed Eq. (24) coefficient, all three rate prescriptions, the old isothermal benchmarks, hot-temperature coverage, conserved adiabatic EOS states, scaling, cold endpoint conditioning, upstream root validation, and branch recovery.
- Full isothermal convergence differences and independent hot-temperature outer/phase/tail refinements are saved in `results/speed-sensitivity-summary.json` and its convergence CSV.
- Largest grid change from 24 to 40 Gauss-Legendre nodes per subinterval: `1.71e-13`.
- Largest change after extending the energy tail from `48*T` to `64*T`: `5.56e-15`.
- Independent finite-temperature 3D quadrature agreement: `3.11e-15` at the two tested points.
- Cold analytic versus independent 3D quadrature: `1.72e-8`; the separate 2D cold check agrees within `5.15e-14`.
- Maximum corrected constrained thermal-coefficient discrepancy: `9.71e-8`.
- All final Wolfram algebraic residuals vanished; the cold branch matches through five derivatives and the quadratic coefficient through three.

These convergence comparisons are not rigorous numerical error bounds.
Table 2 of the manuscript gives the highest consecutively passing sampled temperature at 5% accuracy; it is not an interpolated continuous boundary.
The cold-plus-quadratic approximation passes all sampled points up to only `1 MeV` at `a=0.001`, while several larger compositions pass through `25` or `50 MeV`.
An isolated agreement at `100 MeV` can be an error crossing and is not evidence for a controlled expansion.
The finite-temperature integral remains the reference for the stated quark channel; the added speed figure tests a rate-integral estimate, not a solved front or the omission of other channels at high temperature.

## Reference Corrections

The 2026-10-07 reference audit restored the factor `z` in Madsen's Eq. (24) term `-5*z*v^4/64`, in both the manuscript and `eq24_factor`.
The full validation run and comparison figures were regenerated; only the two Eq. (24) comparison columns changed in the 192-row grid.
At fixed `mu_B=900 MeV,a=1`, the corrected diagnostic errors are `+15.4916%`, `+32.9033%`, and `+11.2145%` at `T=25,50,100 MeV`.
All exact rates, derived coefficients, other approximation columns, and the sampled 5% table remain unchanged.
A footnote now identifies the normalization misprint in Madsen's printed Eq. (4); the standard invariant normalization used here agrees with his later equations and was not changed.


## Adiabatic calculation (Section 8.6)

Run the complete reproducible chain from this directory:

```bash
../../bin/python3 run_adiabatic.py
../../bin/python3 -m unittest -v test_validate_rate.py test_speed_sensitivity.py test_adiabatic_sensitivity.py test_adiabatic_upstream.py
```

The driver runs these components in order:

1. `adiabatic_upstream.py`: branch-validated QMC-RMF3 neutron matter at upstream T=0,0.1,5,20,50,100 MeV; scan nB(0-)/n0=1..10, refine admissibility boundaries and requested compositions, save the complete EOS query cache including recovered failures.
2. `adiabatic_sensitivity.py`: cold-interface conserved quark trajectories, all three rates, two-term local diffusion, the constant-diffusion diagnostic and outer/phase/tail refinements. The normalization energy scale is numerical; muB is not frozen.
3. `complete_adiabatic_quadrature.py`: extra compositions required by clipped or nonmonotone upstream slices, with independent outer refinement.
4. `adiabatic_verify.py`: independent high-precision EOS/integration coordinate, root refinement, decreasing-cutoff endpoint checks, and saved convergence summary.
5. `adiabatic_report.py`: dimensional physical trajectories, benchmark tables, the former six-panel speed figure (still generated, no longer included in the manuscript), four-panel trajectory diagnostic, and constant-diffusion comparison.
6. `adiabatic_map.py`: upstream-plane map used as Figure 5. Evaluates [(h/nB)^4/(4 pi^2 (P+B))]^(1/3) of QMC-RMF3 neutron matter on 49 temperatures (0-120 MeV) x 101 densities (1-6 n0), which fixes a(0+) for T(0+)=0 (the a(0+)=1 boundary reaches n0 at T(0-)=112.28 MeV, so the grid encloses both boundaries), and contours the Eq. (23)/exact speed ratio through the universal ratio-versus-composition quadrature. Run after `adiabatic_report.py`; `--plot` redraws from `results/adiabatic-map.json`. Per-temperature EOS checkpoints are in `results/adiabatic-map-cache/`; interpolated compositions agree with the 128 precisely solved states inside the grid to 2.6e-3.

The calibrated bag scale is 189.1566 MeV and alpha_s=0.3. T(0+)=0 is imposed, whereas T(infinity) is solved at muK=0. Constant P and h/nB imply constant quark nB because h=4(P+U_bag); thermal gluons remain in P. Integration uses z=(T/T_inf)^(1/3), the EOS Jacobian and its analytic finite cold limit. No temperature floor, spatial BVP, absolute speed or rate-dependent profile factor is used.

At fixed interface composition, rescaling all energies leaves the ratios invariant. With the weak prefactor fixed, Gamma scales as energy^8, D as energy^(-1), nB as energy^3, and I2 as energy^4. I2 is dimensionless in natural units because the weak prefactor carries energy^(-4). This permits one normalized quadrature to be reused across upstream states. The physical density and the residual between target and solved interface composition are retained for every row; `adiabatic-trajectories.csv` checks the resulting pressure and neutrality after rescaling.

Primary outputs are:

- `adiabatic-upstream.json`, `adiabatic-upstream-cache.json`: intervals, inaccessible targets, upstream states, root validation and recovery provenance.
- `adiabatic-quadrature.json`, `adiabatic-quadrature-provenance.json`: quadrature nodes, local thermodynamics and rates, weighted integrands, integrated estimates, refinements and source hashes.
- `adiabatic-universal.csv`: all computed composition-dependent ratios and dimensional reference integrals.
- `adiabatic-speed-sensitivity.csv`, `adiabatic-trajectories.csv`, `adiabatic-summary.json`: upstream-matched results and dimensional trajectories.
- `adiabatic-independent-verification.json`: independent integral, endpoint, EOS and isolated quadrature checks.
- `adiabatic-convergence.csv`, `adiabatic-verification.tex`: tabular convergence diagnostics and the generated manuscript input, both in `results/`.
- `adiabatic-speed-sensitivity.pdf`, `adiabatic-trajectory-diagnostics.pdf`, `adiabatic-diffusion-weighting.pdf`, with PNG previews.

The first five temperature slices cover the full interior composition interval. The 100 MeV upstream slice has a nonmonotone composition-density relation with a minimum near a(0+)=0.879; it cannot reach the 0.146 or 0.5 benchmarks within the density search range. Its 0.9 benchmark is reachable. Both density roots are saved where a composition is attained twice. No curve is extrapolated into the unavailable interval.

For a(0+)=0.146,0.5,0.9, local-diffusion Eq. (23)/exact speed ratios are 1.03698503,1.16660735,1.56107230; the derived T0+T2 ratios are 1.00350287,1.00814924,1.00462560. These are sensitivities of I2 on the same trajectory, not measured changes in a solved front velocity. Small weighted quadratic errors do not establish uniform local validity, particularly near equilibrium. The microscopic reference still omits additional antiquark collision channels and QCD corrections, and the cold diffusion endpoint is a formal improper-integral limit.

Quadrature checkpoints reject changed numerical-source hashes; use `run_adiabatic.py --force-quadrature` to regenerate them after changing the rate or trajectory implementation. The upstream EOS cache separately validates the hashes of the read-only solver sources. To redraw the adiabatic tables and figures only, run `adiabatic_report.py` after completing the saved calculations.
