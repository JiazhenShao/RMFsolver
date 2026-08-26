---
summary: Append-only chronology of wiki operations. Grep the "## [" prefix for a timeline.
status: current
updated: 2026-08-26
tags: [meta, log]
---

# Log

Append-only. Every entry starts `## [YYYY-MM-DD] verb | subject` so the file is greppable:

```bash
grep "^## \[" log.md | tail -5
```

Newest last.

---

## [2026-08-11] seed | Initial wiki from docs/research, specs, and memory

Built `wiki/` as a standalone Obsidian vault. Seeded from `docs/research/` (3 files), `docs/superpowers/specs/` (11), and the auto-memory directory (9). Design-history plans were deliberately excluded as archaeology.

15 content pages created across `physics/`, `solvers/`, `methods/`, `paper/`, `sources/`. Conventions recorded in [[SCHEMA]].

Two conclusions were reconciled during the seed rather than copied forward:

- [[unmax-low-temperature]] concluded the exact-zero result "looks sound". [[unmax-degeneracy]] (2026-08-04) shows that point sits on a continuum and the agreement was one lucky seed. The older page now carries the correction inline.
- The 2026-07-22 interface-jump spec's shared-$\tau$ premise was already marked superseded in `docs/research/README.md`; that reversal is now recorded on [[quark-transport]] and [[heiselberg-pethick-1993]] where it will actually be seen.

Not yet ingested: the 72 notebooks, `Latex_writting/`, cluster payloads under `new_paper_calculations/`, and `docs/superpowers/plans/`.

## [2026-08-11] tooling | Notebook read path and project-map retirement

Preceded the wiki. `.claude/tools/nbsrc.py` extracts notebook cell source, text outputs and figures on demand; a `PreToolUse` hook blocks raw `.ipynb` reads. Measured 551,360 → 641 tokens for a table of contents on `26SU-9.ipynb`.

`PROJECT_MAP.md` (14 MB) and `.project-map/index.json` (75 MB) moved to `backup/project-map-retired-2026-08-11/` — the project-context-map skill had indexed the venv.

## [2026-08-11] ingest | Active hydrodynamical-combustion manuscript and notes

Ingested the first, compiled document body of `Latex_writting/Hydrodynamical combustion/Main.tex` (through its first `\end{document}`) and used `Notes.tex` as older derivational context. The second complete document embedded after the first `\end{document}` was excluded as an ignored duplicate.

Added six retrieval-sized pages: [[diffusion-limited-front]], [[analytic-front-speed]], [[front-speed-phenomenology]], [[quark-matter-eos]], [[strangeness-reaction-diffusion]], and [[steady-front-bvp]]. Updated [[combustion-paper]], [[gw-observables-section]], [[interface-closure]], [[phase-velocity-overview]], and [[quark-transport]] to route into them.

Preserved rather than papered over five unresolved manuscript inconsistencies: the definition of $a$, the dropped $\gamma_v^2$ diffusion factor, the missing $\gamma_v$ in one energy equation, charge neutrality in an electron-free model, and the contradictory assessment of $\xi=0$. Older entropy-flux formulations and the coefficient-3 LTE derivation in `Notes.tex` were not promoted because newer energy-flux and [[lte-composition-bound]] conclusions supersede them.

## [2026-08-11] clarify | Equilibrium-isobar candidate versus actual downstream endpoint

Recorded in [[interface-closure]] that the fixed-$T(0^+)$, $\mu_K=0$ pressure root is only an auxiliary equilibrium-isobar candidate away from the zero-speed boundary.
At the boundary the conversion layer collapses, so $0^+$ and $\infty$ coincide and the candidate becomes the actual downstream endpoint.
Also established the endpoint-notation rule in `AGENTS.md`: new work uses only `0minus`, `0plus`, and `inf` labels.

## [2026-08-11] clarify | Why the enthalpy-gap root is the zero-speed endpoint

Expanded [[interface-closure]] with the physical reason $\Delta h=0$ terminates the moving branch.
At the root, the prescribed interface state is already the equilibrated downstream endpoint: $T(0^+)=T(\infty)$ and $\mu_K(0^+)=\mu_K(\infty)=0$.
The full $x>0$ reaction--diffusion profile and its integrated source therefore collapse, giving $I_2=0$ and $u(0^-)=0$.

## [2026-08-11] fix | Hard numerical-contour time budgets and point checkpoints

The fixed-$T(0^+)=5$ MeV runner in `new_paper_calculations/26-08-10/` now executes every uNmax BVP trial in a disposable subprocess, so a parent-enforced wall-clock timeout cannot be swallowed by solver-side `except Exception` handlers.
Defaults are 600 seconds per BVP trial and 1800 seconds total per requested contour cell, including retries and midpoint bridges.
Each completed target now triggers an atomic partial `.npy` update containing its scalar result and attempt diagnostics; full profiles are added when the corresponding theta ray completes.

## [2026-08-12] fix | Exact zero upstream-temperature contour ray

`analytic_velocity_bound`, `semi_analytic_velocity_bound`, and `solve_front_energy_conserving_uNmax` now accept finite non-negative $T(0^-)$ rather than rejecting zero before EOS evaluation.
The nuclear EOS zero-temperature branch was already consistent at the affected densities; a representative $T(0^-)=0$, $T(0^+)=5$ MeV point converged in all three methods.
Created the source-only `new_paper_calculations/26-08-12/` workflow, whose boundary construction and 600-cell endpoint-inclusive mesh both use exactly $T(0^-)=0$ on the $0^\circ$ ray.

## [2026-08-12] workflow | One-command fixed-temperature contour run

Added `new_paper_calculations/26-08-12/run_T0plus5_all.py` as the normal cluster entry point.
It runs domain construction, full analytical, semi-analytical, and numerical stages sequentially, so the allocation is never oversubscribed by competing worker pools.
The numerical stage consumes the semi-analytical output from the same run, and all four output files share one collision-safe rerun suffix.

## [2026-08-13] result | Fixed-interface-temperature contour and finite catch-up times

Promoted the full numerical $T(0^+)=5$ MeV contour and `26SU-12.ipynb` propagation analysis into [[front-speed-phenomenology]].
The accepted speeds span approximately 25--3415 m/s, while temperature shifts the zero-speed density strongly enough that the contour cannot be interpreted through $D_K\propto T^{-5/3}$ alone.
For $T(0^-)=0$, 20, 40 and 60 MeV, the fitted 1-km catch-up times are 38.08, 38.19, 43.19 and 107.19 s; all are finite, and the 60 MeV result is dominated by the extrapolated low-speed tail.

## [2026-08-14] clarify | Status of the $\alpha_s=0.3$ input

Recorded in [[strangeness-reaction-diffusion]] that $\alpha_s=0.3$ is supported as a conventional phenomenological input in bag-model quark-matter studies, not as a controlled vacuum-running value at 400 MeV.
The manuscript now cites the classic Farhi--Jaffe calculation and Madsen's Lecture Notes review at both the diffusion comment and the Results setup.

## [2026-08-17] derive | Isothermal analytical front speed

Derived the fixed-$T$ speed relation by multiplying the K-ness equation by $D_Ka'$ and integrating.
With frozen $D_K$, $\gamma_K$, and $\eta$, the source integral is exactly $I_2=D_K\gamma_Ka^2(0^+)[a^2(0^+)+2\eta]/4$, leaving only the $I_1$ profile area to be represented by $\xi$.
Recorded both the piecewise-constant formula and the endpoint-ratio generalization matching `solve_front_isothermal`, and verified the integral and algebra in Wolfram Language.

## [2026-08-17] clarify | Zero-temperature limit and endpoint notation

Removed starred interface notation throughout the wiki in favor of location-based endpoint notation.
For fixed $0<a(0^+)<1$, the isothermal formula has the formal scaling $D_K\propto T^{-5/3}$ and $u(0^-)\propto T^{-5/6}$ as $T\to0^+$.
This divergence is not physical: the mean free path outgrows the conversion length, so local diffusion and LTE fail, while a joint limit with $a(0^+)\to0$ is path dependent and requires an interface closure.

## [2026-08-17] derive | Stable-neutron-matter mask for the isothermal contour

Established that the fixed-$T$ velocity formula has no finite-temperature zero except $a(0^+)=0$ and therefore cannot locate bulk phase stability by itself.
Because energy conservation has been removed, the old $\Delta h=0$ boundary is inapplicable; the correct static isothermal boundary is common $T$, $\mu_B$, and $P$ with equilibrated quark matter.
A live EOS scan confirmed a lower-density stable-neutron-matter region and located representative coexistence densities for both $B^{1/4}=180$ MeV and the $189.1566$ MeV cold-$3n_0$ calibration.

## [2026-08-17] lint | phase_velocity API drift: three analytic entry points, not one

Audited the wiki against live `phase_velocity.py` (now 7353 lines, was recorded as ~5255).
The analytic velocity bound is **three** public functions, not one: `analytic_velocity_bound` (closed-form $I_2$, fixed $T(0^+)$), `semi_analytic_velocity_bound` (numerical $I_2$, fixed $T(0^+)$), and `analytic_velocity_bound_lte` (closed-form, LTE interface) — all thin wrappers over `_solve_analytic_velocity_bound` switched by `velocity_closure` and `interface_control`.
`interface_fraction_mode` is no longer a parameter, so the documented `analytic_velocity_bound(..., interface_fraction_mode="LTE")` call would now raise `TypeError`.
Endpoint-notation renames had propagated into the code but not the wiki: `uN`→`u_0minus`, `_analytic_aqstar_lte`→`_analytic_a_0plus_lte`, `_solve_analytic_downstream_endpoint_for_uN`→`_solve_analytic_inf_endpoint_for_u_0minus`, `_uNmax_collocation_status_is_acceptable`→`_u0minus_max_collocation_status_is_acceptable`, and result keys `T_Q`/`a_interface_LTE`/`u_N`→`T_inf`/`a_0plus_LTE`/`u_0minus_max`.
`__all__` exports only six names, omitting `semi_analytic_velocity_bound`, `analytic_velocity_bound_lte` and `u_0minus`.
Re-measured the 196/202/228 m/s reference triple: it reproduces under **no** current code path (full analytic at $T(0^+)=5$ MeV gives 143.93/147.82/163.23; LTE, the closest, 189.22/193.84/212.06), and the quoted $\mu_B\approx1181$ MeV is really 1190.86 at $3n_{\rm sat}$.
The $X_q$ estimate in [[gw-observables-section]] rests on the retired triple and is flagged for recomputation rather than silently rescaled.
Substance of [[unmax-degeneracy]] and [[unmax-low-temperature]] re-verified against live code and found correct; only names were stale.

## [2026-08-17] implement | Hydro-consistent isothermal analytical velocity

Added `analytic_velocity_isothermal`, which accepts $T(0^-)$, $n_B(0^-)$ and the local interface fraction $a(0^+)=n_K(0^+)/n_B(0^+)$.
It classifies the stable-neutron-matter region with $\Delta\mu_B=\mu_{B,\rm QM}^{(P)}-\mu_B(0^-)$ at common $(P,T)$, returning zero speed when $\Delta\mu_B\geq0$, and couples the negative-$\Delta\mu_B$ branch to baryon and momentum conservation through a scalar finite-flux eigenvalue.
The massless analytical result was verified on both sides of the $T=20$ MeV coexistence curve; the pre-existing numerical isothermal regression still passes and remains on its older normalized-$\widetilde a$ convention pending a separate migration.
A live $T=10^{-8}$ MeV check reaches the $u(0^-)=1$ slow-front boundary without an eigenvalue and now returns `slow_front_approximation_invalid` rather than jumping to a disconnected high-velocity EOS root.

## [2026-08-17] tooling | Wiki is now version-controlled and public

The workspace root became a git repository so `wiki/` and `RMFsolver/` are tracked together; it pushes to `https://github.com/JiazhenShao/RMFsolver`.
The repository history was moved up from `RMFsolver/` rather than restarted, so `git log --follow` on the package still reaches the 2025 baseline, and the four-commit public snapshot previously on `main` was merged in with `-s ours` rather than force-pushed away.
`.gitignore` is an allowlist because the workspace root is also the venv: everything at top level is ignored and only the two published trees are re-included.
Consequence for wiki practice: this vault is **public**, including [[combustion-paper]] and [[gw-observables-section]]. Write pages knowing they are readable by anyone, and treat every commit as publication.

## [2026-08-18] implement | Numerical isothermal local-fraction BVP

Rewrote `solve_front_isothermal` so its fundamental fields are the physical K density and current, $n_K$ and $J_K$, while every public composition value is the local fraction $a=n_K/n_B$.
The fixed-temperature EOS closure now recomputes $D_K$ from the local $\mu_B$ and evaluates the exact $\mu_K$-dependent nonleptonic rate at every node; the cubic reduced source and frozen transport remain analytical approximations only.
The solver retains the compactified `solve_bvp` construction with $j_B$ as its scalar eigenvalue, supports PNM, SYM, and beta-equilibrated upstream matter through $a(0^-)=1-Y_p/2$, and supports finite $m_s$ with a shifted tail about nonzero equilibrated $n_K(\infty)$ and $J_K(\infty)$.
Added tracked regression coverage for the local-fraction contract, exact-rate and local-diffusion profiles, upstream composition, domains, finite-flux residuals, endpoint closure, and a finite-$m_s$ live EOS case.
The old normalized-composition strict-retry reference is superseded; its formerly failing case now converges directly in the physical-field formulation.

## [2026-08-18] verify | Local-fraction form of the isothermal eigenvalue

Translated an independently derived current-jump equation into location-based notation and the local definition $a=n_K/n_B$.
Baryon conservation requires the upstream local fraction in the downstream-background jump to appear as $\lambda_na(0^-)$, or equivalently requires both composition terms to be multiplied by $u(\infty)$.
With the profile-area coefficient $(1+\xi)/2$, Wolfram verified the general result $u^2(0^-)=2I_2/\{\lambda_n^2[a(0^-)-a(0^+)][a(0^-)+\xi a(0^+)]\}$, whose pure-neutron-matter specialization is exactly the implemented formula.
Omitting the density-ratio conversion produces a different denominator and is correct only in the special case $\lambda_n=1$.

## [2026-08-19] verify | Proper-velocity convention makes the isothermal momentum flux exact

$u=j_B/n_B$ throughout `phase_velocity.py` is the proper velocity $\gamma v$, confirmed by the pre-existing `_relativistic_gamma_from_u` ($\gamma=\sqrt{1+u^2}$, 15 call sites); $\Pi=P+hu^2$ is therefore already the exact relativistic momentum flux, not a $\gamma\to1$ reduction of it.
An opt-in "relativistic" junction-flux branch was built to test this and shifted results by $0$ to $2.4\times10^{-15}$ — an algebraic identity — so it was reverted rather than kept.
`_momentum_flux_diagnostics` had briefly shipped with $\gamma$ computed as the 3-velocity form $1/\sqrt{1-u^2}$ and a `relativistic_flux_ratio` key that multiplied in a second, double-counted $\gamma^2$; both were corrected or dropped.
Recorded on [[isothermal-analytic-front-speed]], along with the residual static-isobar bound from `_solve_a_0plus_max` and the `slow_front_consistent` label mismatch (it tests $u(0^-)<1$, i.e. $v<0.707$, not $v<1$).

## [2026-08-20] implement | Boundary-fitted isothermal contour cluster workflow

Added `new_paper_calculations/26-08-19/` to invert the live weighted static-isobar ceiling between the PNM--equilibrated-quark and strangeness-free phase boundaries, then run matched analytical and physical-$n_K,J_K$ contours with `a_0plus` omitted from both public calls.
Analytical cells have a 300-second killable limit; the numerical stage advances in composition shells with scalar-$j_B$ continuation, disposable spawned BVP attempts, 180-second trial and 900-second cell limits, and atomic per-cell checkpoints.
The one-command driver now auto-uses the scheduler allocation or all local CPUs and displays ordered boundary, 600-point domain, 600-point analytical, and 600-point numerical `tqdm` bars; the parent atomically saves each point before incrementing its bar, and domain resume can continue partial checkpoints.
Its runtime selection validates the active cluster environment first and treats a repository-local `bin/python3` only as an optional fallback.
The first production payload was audited after both 600-point bars completed instantly: all cells were masked, both velocity files contained zero records, and the domain contained 460 failures from a stale subprocess `RMFsolver` without `_solve_a_0plus_max` plus 140 cells masked by seven missing boundary points.
The successful boundary records themselves satisfy the intended PNM--equilibrated-quark and $a(0^+)=1$ PNM--ud equations to about $10^{-10}$ relative pressure accuracy, but the upper curve has six low-temperature `no_root` gaps and the lower curve lacks its $120$ MeV endpoint.
The domain fingerprint covers every phase boundary and curvilinear coordinate, while resume reuses that domain and rejects changed physics or solver controls; every failure remains a structured mask rather than zero or infinity.
A $3\times3$ live smoke run gave 9 valid domain cells, 9 analytical successes, a maximum composition residual of $4.6\times10^{-12}$, and an exact proper-speed conversion; deliberately reduced numerical budgets produced 9 clean `cell_timeout` masks without blocking later shells or plotting.
The workflow and the distinction between this static-isobar ceiling and a future finite-flux maximum are recorded in [[isothermal-contour-cluster]].

## [2026-08-20] fix | Isothermal contours moved to the energy-conserving polar mesh

Replaced the superseded fixed-temperature by fixed-$a(0^+)$ domain in `new_paper_calculations/26-08-19/` with the exact 26-08-18 elliptical-polar window: $1\leq n_B(0^-)/n_0\leq5.5$, $0.01\leq T(0^-)\leq120$ MeV, 30 endpoint-inclusive contour rays, and the same 20 outward radial fractions.
Twelve endpoint-inclusive Chebyshev rays now trace the $\Delta\mu_B=0$ and $a(0^+)=1$ radii from shared upstream probes; incomplete or reversed boundaries stop the workflow before either velocity scan.
The analytical and numerical solvers receive the resulting physical $n_B(0^-),T(0^-)$ points with `a_0plus` omitted, and the numerical continuation advances outward in radial-fraction shells rather than composition shells.
Every stage now forces the repository checkout ahead of installed packages, records the resolved solver path, atomically checkpoints before each point-bar increment, and uses an incompatible schema version so the invalid first production payload cannot resume.

## [2026-08-20] fix | Dated isothermal workflow is self-contained

Moved the active polar contour workflow to `new_paper_calculations/26-08-20/` and embedded its boundary residual and ray-root logic directly in `_isothermal_domain.py`.
Spawned boundary workers no longer import the workspace-level `isothermal_domain_rays` prototype, which was absent on the cluster and caused all twelve rays to fail before their first EOS probe.
The 26-08-20 run tag and regression path now match the active directory, and the repository allowlist includes its Python files and README.

## [2026-08-21] fix | Analytical isothermal upstream PNM recovery no longer scans chemical potential

Replaced the broad `muB_from_nB_physical` scan inside `analytic_velocity_isothermal` with a direct fixed-$n_B(0^-)$ `RMFsolvePNM` validator.
The new path rejects bad scaled RMF residuals, requires two seeds to agree on both $\sigma(0^-)$ and $\mu_B(0^-)$, stops once agreement is found, and reuses the accepted solution table for the forward-density check, pressure, and energy density.
The former cold and hot-low-density timeout examples now complete in $0.78$ s and $0.14$ s instead of $102.8$ s and $65.9$ s locally, while reproducing the scan-based velocities to better than $4\times10^{-8}$ relatively.
Added public upstream diagnostics and regression coverage for scan removal, residual rejection, branch agreement, forward-density validation, existing analytical behavior, numerical isolation, and the 26-08-20 cluster workflow.
The remaining low-temperature quadrature and disabled root-success checks in `Solver.py` are documented but were not changed.

## [2026-08-24] revise | Isothermal analytical density-drift approximation

Replaced the manuscript's explicit $\mathcal C(a)$ correction with a direct numerical error estimate for the representative $T=10$ MeV, $n_B(0^-)=3n_0$, $B^{1/4}=189.2$ MeV profile.
Along its fixed-$(P,T)$ isobar, $\mu_B$ and $n_B$ each increase by about $0.24\%$, and $a'\simeq n_K'/n_B$ differs from the exact local-fraction derivative by at most $0.47\%$.

## [2026-08-25] implement | Piecewise-angular analytical isothermal cluster workflow

Added the source-only `new_paper_calculations/26-08-25/isothermal_analytic/` workflow with a one-command domain, analytical scan, and plotting sequence and no numerical BVP stage.
The production contour retains 600 cells but replaces the uniform angular axis by an exact zero-degree ray, 15 logarithmic points from $0.01^\circ$ through $20^\circ$, and 14 linear points above the seam through $90^\circ$.
Both the $\Delta\mu_B=0$ and $a(0^+)=1$ boundaries are now solved directly on 61 piecewise rays, and schema 4 fingerprints the angular-grid metadata while preserving atomic pointwise checkpoints and resume rejection.

## [2026-08-25] fix | Analytical cluster removed obsolete chemical-potential scans

Replaced the 26-08-25 boundary residual's copied `muB_from_nB_physical` scan with the same direct fixed-$n_B(0^-)$ branch-validated PNM recovery used by `analytic_velocity_isothermal`, reusing its accepted $\mu_B(0^-)$ and $P(0^-)$ without a duplicate upstream solve.
Removed the narrow and broad scan machinery that could probe forbidden RMF states and changed both boundary rays and analytical cells from thread-plus-disposable-child execution to persistent process pools.
Schema 5 and the v2 run tag invalidate partial payloads from the expensive implementation; atomic parent-side checkpoint-before-progress behavior is unchanged.

## [2026-08-25] fix | Fixed-density PNM agreement now rejects collapsed thermodynamic branches

Reproduced the cluster boundary crash at $T(0^-)=62.20097$ MeV and $n_B(0^-)=1.38240n_0$ and traced it to seed ordering inside `_validated_pnm_state_from_nB`.
Seeds two and three agreed on a collapsed $\sigma(0^-)=178.26$ MeV branch with negative pressure, causing an early stop before seed four corroborated the physical $\sigma(0^-)=36.42$ MeV positive-pressure branch found by seed one.
Branch agreement now validates forward density, pressure, and enthalpy before terminating the seed search; the exact former failure returns a moving front in 0.44 s with four seeds.
The 26-08-25 cluster schema is now 6 so partial results from the faulty branch-selection rule cannot resume.

## [2026-08-25] revise | Analytical cluster restored to the successful 26-08-20 baseline

The experimental 26-08-25 domain and analytical execution code was deleted and replaced by the successful 26-08-20 implementation.
The restored workflow changes only angular sampling: both Chebyshev boundary axes increase from 12 to 24 directly solved rays, while the 30-ray analytical mesh uses an exact zero-degree ray, 15 logarithmic positive angles from $0.01^\circ$ through $20^\circ$, and 14 linear angles above the unique seam through $90^\circ$.
The 20 radial fractions and 600-cell total are unchanged; schema 4 rejects the previous uniform-angular checkpoints.

## [2026-08-25] derive | Isothermal static-isobar maximum-speed formula

Substituted the Taylor static-isobar ceiling for $a_{\max}(0^+)$ into the positive analytical isothermal speed branch and verified the algebra with Wolfram Language.
The resulting expression retains the full $D_K(T,\mu_q)$ and thermal weak-rate term; its Eq.-(31)-style Landau-damped form has a $98.3\ {\rm m\,s^{-1}}$ coefficient when $\alpha_s^{-5/6}$ is explicit, equivalently $268.1\ {\rm m\,s^{-1}}$ at $\alpha_s=0.3$.
For the massless bag EOS, the ceiling reduces to $a_{\max}^2(0^+)\simeq9[\mu_B(0^-)-\mu_B(\infty)]\,[\mu_B^2(\infty)+3\pi^2T^2]/\{\mu_B(\infty)[\mu_B^2(\infty)+9\pi^2T^2]\}$.

## [2026-08-26] implement | Numerical-only piecewise-angular isothermal cluster rerun

Converted the copied `new_paper_calculations/26-08-25/isothermal_numeric/` analytical runner into `run_isothermal_numerical.py` and fixed its repository-root discovery for the extra subdirectory depth.
The runner reuses the completed local domain and the sibling analytical payload only for baryon-current seeds, then calls the existing radial-shell numerical stage backed by `solve_front_isothermal`.
It preserves scheduler-wide angular parallelism, disposable trial timeouts, save-before-progress pointwise checkpoints, structured terminal failures, and resume validation while aligning `tail_eps=1e-3` with the current public solver default.

## [2026-08-26] diagnose | Numerical isothermal corner smoke exposed obsolete upstream scan

Ran one production-configured numerical trial at each corner of the 30-by-20 allowed-domain mesh.
The high-angle inner corner converged in 71.9 s, the low-temperature outer corner failed after 154.9 s, and the other two corners reached the 180 s hard trial limit.
Traced the initial stall and `Solver.py:223/341` warning flood to `solve_front_isothermal` repeating two fixed-density upstream RMF solves plus the obsolete 48-point branch-validated chemical-potential scan for every baryon-current candidate; this work occurs before the BVP and is independent of the trial current.
Also found that the low-temperature outer analytical seed is clipped from $6.38\times10^4$ to $6.48\times10^2$ by the wrapper's global current bound.
No solver behavior was changed in this diagnostic pass.
