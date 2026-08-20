---
summary: Catalog of every wiki page. Read this first; drill down from here.
status: current
updated: 2026-08-18
tags: [meta, index]
---

# Index

The nuclear→quark combustion front project. **This is the only page loaded by default** — everything else is fetched on demand from here. Conventions live in [[SCHEMA]]; chronology in [[log]].

## Start here

- [[diffusion-limited-front]] — the baseline physical model: a slow planar front selected by quark-side reaction and diffusion. **Read first for project physics.**
- [[interface-closure]] — the central problem: four unknowns, three conservation laws, and the closure hierarchy. **Read first.**
- [[combustion-paper]] — the paper, its files, house conventions, and what still blocks submission.

## Physics

| Page | What it settles |
|---|---|
| [[diffusion-limited-front]] | Physical regime, assumptions, fluxes, variables and boundary conditions |
| [[analytic-front-speed]] | Constant-background speed estimate, scaling, layer thickness and formal endpoint limits |
| [[isothermal-analytic-front-speed]] | Fixed-$T$ speed, the density-ratio-correct local-fraction current jump, the $\Delta\mu_B$ gate, and the exact physical-$n_K,J_K$ BVP |
| [[front-speed-phenomenology]] | Fixed-$T(0^+)$ numerical contour, zero-speed boundary, and tail-dominated $z(t)$ evolution |
| [[quark-matter-eos]] | Bag-model EOS, chemical-potential basis and quadratic analytic expansion |
| [[strangeness-reaction-diffusion]] | Weak source, flavor diffusion, and the phenomenological status of $\alpha_s=0.3$ |
| [[interface-closure]] | Why a fourth relation is needed and which one is authoritative |
| [[thermal-transparent-closure]] | The baseline $T(0^+)=T(0^-)$ — and why "adiabatic" is the wrong word |
| [[lte-composition-bound]] | $\beta_{\rm LTE}=5u(0^-)\lambda_n$, the exact-local-isobar correction, the coefficient trap |
| [[interface-temperature-jump]] | The signed observable $\Delta T=T(0^-)-T_{\rm LTE}(0^+)$ and its payload contract |
| [[quark-transport]] | HP93 coefficients — and why $\kappa_{\rm th}$ and $D_K$ do **not** share a relaxation time |
| [[kapitza-resistance]] | The honest gap: $G_{\rm int}$ has no dense-matter literature value |

## Solvers

| Page | What it settles |
|---|---|
| [[phase-velocity-overview]] | Analytical and numerical solver APIs, including the local-fraction isothermal BVP, units, and the editing restriction |
| [[unmax-degeneracy]] | At $T(0^+)=0$, `jB` is a non-Lipschitz continuum — `success=True` proves nothing |
| [[unmax-low-temperature]] | The `logT`→$T^2$ fix (landed) and the $n_K$-resolution limit (open) |
| [[thermal-conducting]] | $T$ as a propagated field; $a(0^+)$ becomes an output; six debugging facts |
| [[known-issues]] | Finite-$m_s$ isothermal cost, superseded strict-retry history, GIL-bound "parallel" cells, and bisection asymmetry |

## Methods

- [[steady-front-bvp]] — the physical-$n_K,J_K$ reaction-diffusion BVPs, flux eigenvalues, EOS reconstruction and compactification.
- [[two-block-contour-scans]] — the adaptive mesh every contour uses, the 80 MeV seam, and why blocks are never concatenated.

## Paper

- [[combustion-paper]] — hub.
- [[gw-observables-section]] — settled notation, taiji geometry, the six calibrated cases, recurring citation error.

## Sources

- [[heiselberg-pethick-1993]] — the transport paper everything rests on. Paywalled; local PDF only.

## Open questions

1. $G_{\rm int}$ for a sharp nuclear–quark interface — no literature value ([[kapitza-resistance]]).
2. The ~8% BVP-vs-reference systematic, seen plateau-free, unexplained ([[unmax-degeneracy]]).
3. Whether the conduction result licenses the "$a(0^+)$ is determined" claim — gated on seed/mesh/schedule independence tests ([[thermal-conducting]]).
4. Reconciling the $3\,\mathrm{Le}$ coefficient generalisation with the corrected HP93 reading ([[lte-composition-bound]]).
5. Whether the relativistic $\gamma_v^2$ factor belongs in every K-diffusion BVP and jump equation ([[diffusion-limited-front]]).
6. Reconciling the electron-free quark EOS with the numerical exact-rate charge-neutrality condition ([[quark-matter-eos]]).
7. Determining or bounding the diffusive-flux shape parameter $\xi$; the manuscript's prose and diagnostic disagree ([[analytic-front-speed]]).
8. Replacing the prescribed $T(0^+)=5$ MeV slice with a physical thermal interface closure ([[front-speed-phenomenology]]).
9. Which change retired the 196/202/228 m/s reference triple, and what $X_q$ should be recomputed with — the three analytic entry points now differ by up to 30% ([[phase-velocity-overview]], [[gw-observables-section]]).

## Not in this vault

Live code (`RMFsolver/`), notebooks, cluster payloads and LaTeX sources stay in `NSM_related/`. This wiki records *conclusions*; the repo holds the artifacts. Design-history specs remain at `docs/superpowers/specs/` — treat them as archaeology, and prefer this wiki where the two disagree.
