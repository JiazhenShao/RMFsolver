---
summary: Catalog of every wiki page. Read this first; drill down from here.
status: current
updated: 2026-10-08
tags: [meta, index]
---

# Index

The nuclear→quark combustion front project. **This is the only page loaded by default** — everything else is fetched on demand from here. Conventions live in [[SCHEMA]]; chronology in [[log]].

## Start here

- [[diffusion-limited-front]] — the baseline physical model: a slow planar front selected by quark-side reaction and diffusion. **Read first for project physics.**
- [[interface-closure]] — the central problem: four unknowns, three conservation laws, and the closure hierarchy. **Read first.**
- [[combustion-paper]] — the paper, its files, house conventions, and what still blocks submission.
- [[notebook-renames]] — season-named notebooks became date-named on 2026-10-02; read it to decode old names in [[log]] and `docs/`.

## Physics

| Page | What it settles |
|---|---|
| [[diffusion-limited-front]] | Physical regime, assumptions, fluxes, variables and boundary conditions |
| [[analytic-front-speed]] | Adiabatic composition ceiling A_ad, constant-background speed estimate, and shared layer thickness now introduced under isothermal flow |
| [[isothermal-analytic-front-speed]] | Fixed-$T$ speed, local-fraction current jump, and separately verified chemical-potential/density drifts along the saved isobar |
| [[isothermal-maximum-speed]] | Static-isobar ceiling with the neutral EOS substituted explicitly in Eq. (35), maximum-speed formula and low-$T$ scaling |
| [[front-speed-phenomenology]] | Fixed-$T(0^+)$ numerical contour, zero-speed boundary, and tail-dominated $z(t)$ evolution |
| [[quark-matter-eos]] | Neutral bag-model EOS, verified exact free-gas citation, former non-neutral basis and quadratic analytic limit |
| [[bag-model-thermal-gluons]] | Appendix A cites Madsen and Buballa reviews for thermal gluons; combustion precedents, effective-model exceptions and distinction from interaction corrections |
| [[charge-neutral-quark-eos]] | Appendix A density representation, neutral quartic expansion and consistent density derivatives |
| [[composition-constraint-plot-audit]] | Neutral and revised simple PI plots match; exact a=1 endpoints are 669.01 and 344.13 MeV at mu_B=1200 MeV |
| [[strangeness-reaction-diffusion]] | Madsen rate is approximate even in the numerical solver; exact-rate speed impact ≤20% for a(0+)≤0.7, NLO (2609.35165) ≈ +19% speed; diffusion and coupling choice |
| [[unequal-flavor-weak-rate]] | General massless collision integral and neutral T=0 checks: full/Eq. (23) rate is 0.932 at a=0.146 and 0.359 at a=0.9 |
| [[massless-finite-temperature-weak-rate]] | Exact cold/T-squared derivation, 192-state grid, and isothermal speed sensitivity extended to 50 and 100 MeV |
| [[adiabatic-weak-rate-sensitivity]] | Conserved cold-interface trajectories: universal ratios audited with six independent dimensional integrations, 153 upstream states, diffusion weighting and hot-slice limits |
| [[interface-closure]] | Why a fourth relation is needed and which one is authoritative |
| [[thermal-transparent-closure]] | The baseline $T(0^+)=T(0^-)$ — and why "adiabatic" is the wrong word |
| [[lte-composition-bound]] | $\beta_{\rm LTE}=5u(0^-)\lambda_n$, the exact-local-isobar correction, the coefficient trap |
| [[interface-temperature-jump]] | The signed observable $\Delta T=T(0^-)-T_{\rm LTE}(0^+)$ and its payload contract |
| [[quark-transport]] | HP93 coefficients — and why $\kappa_{\rm th}$ and $D_K$ do **not** share a relaxation time |
| [[thermal-limits]] | Current (+---) convention, invariant physical fluxes, and Eckart coefficient 5/2 versus Fourier's 3/2 |
| [[thermal-limit-ordering]] | Isothermal subsection now isolates pressure and expands in velocity; former Appendix D proof is commented out; finite heat flow survives infinite conductivity |
| [[kapitza-resistance]] | The honest gap: $G_{\rm int}$ has no dense-matter literature value |

## Solvers

| Page | What it settles |
|---|---|
| [[phase-velocity-overview]] | Analytical and numerical solver APIs, including the local-fraction isothermal BVP, units, and the editing restriction |
| [[pnm-density-state-recovery]] | Direct fixed-density upstream PNM recovery replaces the analytical isothermal solver's broad chemical-potential scan |
| [[unmax-degeneracy]] | At $T(0^+)=0$, `jB` is a non-Lipschitz continuum — `success=True` proves nothing |
| [[unmax-low-temperature]] | The `logT`→$T^2$ fix (landed) and the $n_K$-resolution limit (open) |
| [[thermal-conducting]] | $T$ as a propagated field; $a(0^+)$ becomes an output; six debugging facts |
| [[known-issues]] | Completed numerical-isothermal failure audit, edge recovery and current bounds, cold quadrature, finite-$m_s$ cost, GIL-bound cells, and bisection asymmetry |

## Methods

- [[fixed-interface-analytical-contours]] — Fixed 0.1 MeV comparison, xi=-0.5; 734/734 per method and all 30 semi-analytical 5000 m/s brackets; unshaded through 5000 and shaded through 1000 m/s, with enlarged locally backed labels in the shaded figure.
- [[analytical-xi-minus0p85-cluster]] — original `26-10-02.ipynb` comparison and local recovery: connected adiabatic analytical contours and supported 2000 m/s band; seven-level isothermal baseline has 297 accepted/25 outside-window targets, warm numerical curves and 58 cold numerical dots.
- [[isothermal-high-speed-contours]] — `26-10-02.ipynb` calculates all 14 analytical log-temperature levels locally (644 targets), compares saved numerical curves/dots, and resumes pointwise checkpoints; persistent isolated SciPy fixes the macOS import failure, with separate target and per-evaluation progress.
- [[isothermal-notebook-cold-points]] — Six cold 500/2000 m/s roots plus one 10000 m/s root: 65 numerical dots total; bounded 50000/500000 m/s attempts and a tighter cold check remain explicitly unresolved, with reproducible notebook code and checkpoints.
- [[adiabatic-xi-diagnostics]] — Measured profile xi approaches -1 at high speeds; it fixes semi-analytical speeds to within 0.1%, while full analytical retains composition and integral errors. Best single constant over 50-2000 m/s is xi≈-0.86 (basin -0.845 to -0.89), so -0.85 is already optimal.
- [[steady-front-bvp]] — the physical-$n_K,J_K$ reaction-diffusion BVPs, flux eigenvalues, EOS reconstruction and compactification.
- [[isothermal-contour-cluster]] — Piecewise-angular 600-cell isothermal domain, analytical workflow, numerical shell continuation, and failed-cell-only normalized-BVP repair.
- [[isothermal-targeted-shots]] — Latest bounded follow-up: 20 designated targets, a persistent 120-call ceiling, numerical midpoint connectivity checks, and per-target plots.
- [[isothermal-cold-numerical-probes]] — Full BVP solutions at 0.01 MeV recovered at three near-coexistence densities with a layer-resolved initial guess; one tighter check passes, higher-density robustness remains unresolved.
- [[isothermal-log-temperature-contours]] — Downloaded cold run: only 13/427 targets accepted; analytical-proposal timeouts, unconditional continuation and two-density retry termination identified. Includes analytical-envelope comparison and recommendation against an unchanged rerun.
- [[isothermal-contour-repair]] — Predecessor adaptive repair: preserves 362 baseline successes, supplies numerical roots/profiles for the bounded follow-up, and defines contour connectivity by angular adjacency.
- [[two-block-contour-scans]] — the adaptive mesh every contour uses, the 80 MeV seam, and why blocks are never concatenated.

## Paper

- [[combustion-paper]] — manuscript hub and appendix order; Heat Transport follows Strangeness diffusion as Appendix D.
- [[gw-observables-section]] — settled notation, taiji geometry, the six calibrated cases, recurring citation error.
- [[supernova-quark-oscillations]] — Mathews/Rupp/Boccioli DNP 2026 supernova core oscillations need rapid conversion; laminar fronts move only ~m per ms oscillation.

## Sources

- [[heiselberg-pethick-1993]] — the transport paper everything rests on. Paywalled; local PDF only.

## Open questions

1. $G_{\rm int}$ for a sharp nuclear–quark interface — no literature value ([[kapitza-resistance]]).
2. The ~8% BVP-vs-reference systematic, seen plateau-free, unexplained ([[unmax-degeneracy]]).
3. Whether the conduction result licenses the "$a(0^+)$ is determined" claim — gated on seed/mesh/schedule independence tests ([[thermal-conducting]]).
4. Reconciling the $3\,\mathrm{Le}$ coefficient generalisation with the corrected HP93 reading ([[lte-composition-bound]]).
5. Whether the relativistic $\gamma_v^2$ factor belongs in every K-diffusion BVP and jump equation ([[diffusion-limited-front]]).
6. Propagating the validated local weak-rate corrections into finite-temperature front solutions; I2-level speed estimates now exist (−3% to −40% for the numerical rate, partly offset by NLO), but no BVP re-solve has measured them ([[massless-finite-temperature-weak-rate]], [[strangeness-reaction-diffusion]]).
7. Predicting the state-dependent diffusive-flux shape parameter $\xi$ without an existing BVP profile; measured values explain the semi-analytical error ([[adiabatic-xi-diagnostics]]).
8. Replacing the prescribed $T(0^+)=5$ MeV slice with a physical thermal interface closure ([[front-speed-phenomenology]]).
9. Which change retired the 196/202/228 m/s reference triple, and what $X_q$ should be recomputed with — the three analytic entry points now differ by up to 30% ([[phase-velocity-overview]], [[gw-observables-section]]).

## Not in this vault

Live code (`RMFsolver/`), notebooks, cluster payloads and LaTeX sources stay in `NSM_related/`. This wiki records *conclusions*; the repo holds the artifacts. Design-history specs remain at `docs/superpowers/specs/` — treat them as archaeology, and prefer this wiki where the two disagree.
