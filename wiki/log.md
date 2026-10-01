---
summary: Append-only chronology of wiki operations. Grep the "## [" prefix for a timeline.
status: current
updated: 2026-09-09
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

## [2026-08-26] fix | Numerical isothermal upstream recovery now uses one direct PNM state

Replaced `solve_front_isothermal`'s separate pressure and energy RMF solves plus its 48-point `muB_from_nB_physical` scan with one `_validated_pnm_state_from_nB` result.
The automatic composition ceiling now reuses the accepted $\mu_B(0^-)$ and upstream thermodynamics, matching `analytic_velocity_isothermal`.
The high-angle inner corner's first trial fell from 71.9 s to 4.18 s with the same $23.7242\ \mathrm{m\,s^{-1}}$ result and no warnings; the low-temperature outer failure fell from 154.9 s to 54.9 s.
Three corner trials still fail or time out after upstream recovery, so dynamic quark-state recovery and the near-ceiling current bound remain separate open issues.

## [2026-08-26] diagnose | Numerical isothermal BVP residual overflow is a low-temperature scaling failure

Traced the cluster's `scipy.integrate._bvp.py:569--573` overflow warnings to runaway collocation residuals, not to the RMF momentum quadratures.
On outward shell 13, low-temperature states combine physical $n_K(0^+)\sim7\times10^6$, currents of only $O(10)$--$O(10^2)$, and compact-coordinate factors as large as $dx/ds\sim2\times10^{13}$, while the isothermal BVP passes unscaled physical fields to SciPy.
Fallback trials at $T(0^-)=2.17$ and $3.73$ MeV drove $j_B$ to the artificial $10^{-12}$ lower bound, grew to 8074 and 2692 nodes, and terminated with singular Jacobians; colder trials timed out, whereas representative 31--85 MeV cells on the same shell converged cleanly in 5.8--9.2 s.
The low-angle analytical seed is also clipped by the wrapper's current upper bound, from $6580$ to $555$ at the first shell-13 cell.
No code was changed; the completed numerical payload still needs a status/attempt audit because warnings from rejected disposable trials are not themselves evidence of corrupted saved results.

## [2026-08-27] fix | Numerical isothermal BVP now uses separately scaled physical fields

Transferred the internal field-scaling pattern from `solve_front_energy_conserving_uNmax` to `solve_front_isothermal` without changing the physical $n_K,J_K$ formulation or public profile units.
SciPy now receives $n_K/n_{K,\mathrm{scale}}$ and $J_K/J_{K,\mathrm{scale}}$; EOS and rate calls reconstruct physical values, ODE derivatives use the inverse scales, and endpoint/profile reconstruction restores physical units.
A regression test captured the pre-fix initial collocation density at $2.63\times10^6$ and now verifies both initial fields are order unity; 11 focused numerical-isothermal tests pass.
The very large low-temperature compact-coordinate factor and the cluster's clipping current bounds remain open and require a production rerun before declaring the earlier overflow/timeouts resolved.

## [2026-08-27] diagnose | Completed numerical isothermal payload separates three failure mechanisms

Audited all 600 records in the completed pre-field-scaling `26-08-25/isothermal_numeric/isothermal-numerical.npy`: 358 succeeded, 218 exhausted the 900 s cell budget, and 24 failed, with 23 retained singular-Jacobian BVP results.
All terminal failures lie below $T(0^-)=0.89$ MeV, but high-temperature timeouts persist on the outermost $a(0^+)\simeq0.993$--$0.997$ shell.
The attempt history contains 1172 trial timeouts and about 66.7 aggregate trial CPU-hours; 321 successes occurred on candidate one, 35 on candidate two, only 2 on candidate three, and none later.
All 93 cells whose analytical current exceeded the global `1e-4*nB_0minus` bound were non-successes, while another 149 below-bound cells timed out, proving that field scaling, current-bracket repair, and retry-policy reduction must be tested as separate changes.
For the 357 successes with finite analytical seeds, the accepted numerical current was 0.739--1.594 times the analytical current, providing a measured basis for an analytical-centered per-cell bracket.
The shell scheduler also caps live cell supervisors at the 30 angular rays, so the advertised 64-CPU allocation cannot run more than 30 BVP trials concurrently.

## [2026-08-27] implement | Failed-cell-only normalized isothermal repair workflow

Created `new_paper_calculations/26-08-27/` from the validated 26-08-25 numerical workflow and copied its compatible domain, analytical seeds, and completed numerical checkpoint.
The one-command runner preserves all 358 saved successes and resets only the 242 non-success cells before calling the current separately normalized `solve_front_isothermal`; repeated commands preserve newly accepted cells and retry only those still unresolved.
The original bounds, candidates, timeouts, mesh, and shell ordering remain unchanged so the rerun isolates the field-scaling update.
A portability regression now ignores only the absolute `phase_velocity_file` location while still rejecting any functional API-signature difference, and an in-memory real-payload check confirmed exact 358/242 selection with unchanged accepted grids.

## [2026-09-09] revise | Thermal transport and limits in the hydrodynamic manuscript

Added heat-current terms to the stress-energy tensor, a neutrino loss four-vector, and the slow-flow Fourier model before the general steady-front equations.
Specified that infinite conductivity gives the isothermal limit for finite heat flux and thermal contact, while the adiabatic limit requires both zero conductivity and zero neutrino cooling.
Reused the existing isothermal figure and derivation in the earlier isothermal section, made the three systems' equation labels distinct, and retained all PI text and author comments.
Added INSPIRE-HEP exports for Eckart and Lander--Andersson; recorded the limiting assumptions and Wolfram component check in [[thermal-limits]].

## [2026-09-10] revise | State the thermal model's transport approximations

Named the baryon Eckart frame and explicitly stated neglected K-interdiffusion energy transport, Soret/Dufour cross-effects, and the stationary scope of the Fourier closure.
Clarified the finite-flux isothermal limit and the external-reservoir energy balance while preserving all equations and author comments.
Did not adopt the review's universal per-mille estimate: interface-current matching gives a different conditional chemical-energy scale, checked with Wolfram and recorded in [[thermal-limits]].
Added four verbatim INSPIRE exports, including the three previously unresolved references; latexmk succeeds with no undefined citations or references, and the edited pages were visually inspected.

## [2026-09-11] reorganize | Introduce shared front equations in the isothermal section

Moved the common baryon/momentum flux definitions, reduced K transport system, boundary conditions, and analytical derivation through $I_1$, $I_2$, $\xi$, and the general speed relation into the earlier isothermal section.
Separated conserved adiabatic energy flux from the shared flux display; the adiabatic section now refers back before deriving its variable-temperature trajectory and integral.
Reused existing prose, removed anticipatory references to undefined adiabatic quantities, updated shared appendix references, and preserved all author comments and physical formulas.
Independent ordering review, equation/comment preservation checks, isolated and regular latexmk builds, and visual inspection passed; see [[thermal-limits]] for the section and label map.

## [2026-09-11] revise | Derive Fourier conduction from the Eckart heat law

Replaced the thermal-model presentation-choice comment with a cited covariant Eckart law, its exact flat steady-front component, and the slow-flow reduction to Fourier conduction.
Stated short heat-current relaxation and negligible acceleration as separate assumptions; stationarity alone does not imply either, and the idealized thermal limits follow the reduction.
Wolfram verified the components and orthogonality; latexmk and visual inspection of the revised two-column page passed without undefined references or overfull boxes.
Recorded the limitations and references in [[thermal-limits]].

## [2026-09-11] verify | Separate isothermal chemical-potential and density drifts

Recovered five actual samples from the saved T=10 MeV, nB(0-)=3n0 isothermal profile and reconstructed densities with the live charge-neutral EOS; pressure constancy and interface composition agree with the saved solution.
Wolfram gives endpoint fractional increases of 0.2399168% for muB and 0.2365807% for nB, with ratio 0.9860947; both previously rounded to 0.24%.
Updated the manuscript and [[isothermal-analytic-front-speed]] with separate values and the role of the simultaneous composition change.
The isolated latexmk build resolves all references and citations with no overfull boxes.

## [2026-09-16] implement | Targeted numerical isothermal contour repair

Created `new_paper_calculations/26-09-16/isothermal_numeric_repair/` with hash-verified copies of the completed 26-08-27 inputs and immutable preservation of all 362 successful records.
Added actual numerical contour tracing, local current brackets, profile-seeded continuation, bounded subdivision, per-evaluation atomic checkpoints, explicit gaps and separate completion/convergence states.
Extended only the isothermal public solver with optional lightweight continuation input/output; the adapted mesh and physical fields are transferred without a plotting-profile EOS pass.
Updated `26SU-14.ipynb` to consume the repair overlay while preserving the current contour style, and documented the method in [[isothermal-contour-repair]].
Verification passed 23 fast repair tests, 20 existing non-production isothermal regressions, one bounded warm continuation comparison and one bounded spawned-worker anchor check; both visualization paths and the read-only production preflight passed.
The two live warm checks took approximately nine seconds combined on one core; no production cold scan or four-target cluster smoke was run locally.
Recorded the completed scaling-repair outcome (362 successes, 194 timeouts, 44 failures) and the remaining cluster verification in [[known-issues]].

## [2026-09-16] improve | Repair progress display and solver identification

Added a display-only adapter with a full target progress bar, adaptive totals, separate saved-evaluation counts and a four-check validation bar.
Kept checkpoint-bound search and common modules unchanged so existing cluster repair checkpoints remain compatible.
Startup now reports the resolved solver path and SHA-256; the successful second cluster launch establishes continuation API compatibility, not remote file identity.
Three display tests and 23 existing lightweight tests passed; two opt-in live tests were skipped, and the read-only startup preflight passed. No BVP run was launched.

## [2026-09-17] verify | Conductive momentum stress and slow-flow accuracy

Wolfram verified the exact steady Eckart heat-current components, energy and momentum fluxes, and algebraic momentum closure after eliminating the heat current; five residual checks returned zero.
Recorded in [[thermal-limits]] that conductive stress is of the same velocity order as ideal ram pressure when heat and advective energy fluxes are comparable, although both can be negligible on the thermodynamic pressure scale.
On one saved 26SU-10 thermal profile, the omitted stress reaches 1.13% of ram pressure but only 3.05e-14 of pressure; this is a diagnostic on the reduced solution, with its existing large-Knudsen warning, not a covariant BVP validation.
Updated the context index; no solver or manuscript equations were modified.

## [2026-09-17] derive | Separate conductivity endpoints from the slow-velocity limit

Checked the active manuscript's Eq. (11), Eq. (12), and complete heat-transport appendix.
Wolfram verified the exact identity q_th^x = -kappa_th gamma (gamma T)' and the inverse-Lorentz-factor expansion.
Recorded in [[thermal-limit-ordering]] that infinite conductivity first gives constant gamma T at finite heat flux, then constant T and P at leading slow velocity; the adiabatic endpoint gives constant P and h/nB at the same velocity order.
Clarified finite heat flux, boundary energy compatibility, and the nonuniformity of dropping acceleration before the infinite-conductivity limit; linked the result from [[thermal-limits]] and the index.
No manuscript or solver equations were changed.

## [2026-09-17] revise | Substitute the heat law into second-order fluxes

Updated the main manuscript system by substituting the acceleration-neglected heat law into the exact energy and momentum fluxes before expanding at fixed kappa_th*T'.
Wolfram verified the energy conduction factor 1+5*u^2/2 and momentum conduction term -2*u*kappa_th*T'.
Clarified that steady flow does not imply u'=0, and that neglecting acceleration is a separate approximation.
Recorded the difference from the earlier conversion-driven heat-flux ordering in [[thermal-limits]] and updated the index.
No solver changes were made.

## [2026-09-17] reorganize | Move heat transport after strangeness diffusion

Moved the complete Heat Transport appendix after Strangeness diffusion without changing its content.
The compiled appendix sequence is bag model A, strangeness creation B, strangeness diffusion C, heat transport D, flux shape E, numerical method F, and local thermal equilibrium G.
Converted the two hard-coded appendix references in author comments to labels; all section and equation references resolve after rebuilding.
Verified that the manuscript changes consist only of the block move and those reference substitutions; the rendered transition has no layout defects.
Recorded the new order in [[combustion-paper]] and the index.

## [2026-09-17] revise | Align the main system with direct Fourier conduction

Changed the conductive energy factor in Eq. (16) from 1+5*u^2/2 to 1+3*u^2/2 after the user switched the main constitutive law to Fourier conduction.
Wolfram verified the stress-tensor prefactor expansion; the momentum equation is unchanged.
Reduced the introduction to one sentence stating the Fourier law, slow-velocity expansion through second order, and zero neutrino source.
Updated [[thermal-limits]] and the index to distinguish the current Fourier model from the superseded acceleration-neglected Eckart substitution.

## [2026-09-17] implement | Add the heat-flux elimination proof to Appendix D

Added exact energy and momentum flux integrals, eliminated the heat current, and derived pressure departures of order enthalpy density times squared velocity under the conversion-driven heat-flux ordering.
Explained that infinite Fourier conductivity permits finite heat current: total energy remains conserved while the advective contribution can vary.
Added a short Appendix D reference after the main-text exact balances (9)–(10); the reduced system (16) is unchanged.
Wolfram verified the elimination and series expansion, and the manuscript compiled with resolved references.
Updated [[thermal-limit-ordering]] and the index; no solver changes were made.

## [2026-09-17] simplify | Move the isobaric derivation into the isothermal subsection

Replaced the isothermal subsection's opening with a direct elimination of the heat flux, pressure alone on the left, and the velocity expansion on the right.
Kept one sentence specifying conversion-driven energy-flux scaling, then took infinite conductivity at finite heat current in the user's current main heat law to obtain constant temperature.
Commented out the former Appendix D proof and made the following leading-order momentum-flux definition consistent with constant pressure.
Wolfram verified the isolated-pressure identity and Taylor series; the manuscript compiles with resolved references.
Preserved all user edits in the general hydrodynamic-equations subsection, including the newly restored acceleration-neglected Eckart law; recorded its remaining coefficient mismatch with the reduced thermal energy equation in [[thermal-limit-ordering]].

## [2026-09-21] implement | Bounded designated isothermal shots

Added `new_paper_calculations/26-09-21/isothermal_targeted/`: twenty initial targets, fixed-temperature density correction, reuse of the existing profile cache, and a persistent 120-call ceiling including four tighter checks.
Warm extension/connection precedes local bend checks and a 1 MeV cold endpoint; no full angular scan, recursive auxiliary BVP chain, or new solver physics is introduced.
Midpoint speed/curvature checks now govern new line connectivity; unresolved attempted intervals remain visible gaps and accepted roots are displayed immediately.
Input snapshots, save-before-progress reservations, child-owned timeouts, and interruption recovery preserve work and prevent restarted runs from resetting budgets or repeating completed checks.
Updated `26SU-14.ipynb` to consume the new optional overlay and refreshed its numerical outputs from the existing 738-trial checkpoint, showing its 119 accepted contour roots.
Verified 43 new synthetic/read-only tests plus 26 existing lightweight regressions; two opt-in live tests remained skipped.
No real EOS/BVP recovery was run locally, and `RMFsolver/phase_velocity.py` and the old cluster sources/data were left unchanged.
Documented the cluster command, retained-cache requirement, exact targets, cost bounds, and remaining cluster verification in [[isothermal-targeted-shots]].

## [2026-09-21] fix | Isothermal contour lines follow the angular schedule

Diagnosed the broken numerical isothermal curves in `26SU-14.ipynb`: markers and lines came from two different code paths, and only the marker path worked.
`connected_segments` gated ray-to-ray connection on `within_spacing`, whose `density_step` = 0.05 and `temperature_step` = 2 MeV are continuation step sizes, while the production 30-node angular sweep places neighbouring roots 0.11--0.24 and 3--8 MeV apart; all 109 links were cut and the repair layer contributed no line geometry above 100 m/s.
Connectivity now follows adjacency on `domain["theta_axis_rad"]`, breaking only at a failed ray, an unscheduled node or a branch-count change; the new `connectable` vetoes only a gross jump.
Result: single continuous 50/100/200/500 m/s curves and a near-complete 1000 m/s curve, with the repair arcs superseding every baseline grid arc.
Residual 1000/2000/5000 m/s gaps are unscheduled or failed rays — missing data, not plotting — and still need [[isothermal-targeted-shots]] to run.
Added `local_analysis_results/All_Plots_new/isothermal_contour_data/export_isothermal_contour_archive.py` because `All_Plots_new.ipynb` reads a frozen hash-checked `.npz` rather than recomputing the merge, and gave its cell 16 the tuning block it had been inheriting from leaked kernel state.
Details in [[isothermal-contour-repair]].

## [2026-09-21] fix | Interpolated bridges across unscheduled contour angles

Added `interpolated_bridges`/`targeted_bridges`: an interior gap between two accepted roots is filled by interpolating the radial fraction in angle and mapping back through `ray_coordinates`, which follows the domain geometry instead of cutting the chord (worth ~0.17 in density divided by saturation density on the 2000 m/s gap).
Bridges are fabricated geometry, so they are returned separately, drawn dashed by every caller, trimmed wherever a certified targeted interval covers them, and never extrapolated past an open end.
2000 m/s is now continuous to the density axis and 5000 m/s is a single arc; 1000 m/s still stops at its open end above 86 MeV.
Also made `26SU-14.ipynb` cell 4 drop the workflow helpers from `sys.modules`, after a live kernel silently kept serving pre-fix modules through repeated Run All.
Details in [[isothermal-contour-repair]].

## [2026-09-22] fix | One solid, smooth curve per level for the paper figure

`All_Plots_new.ipynb` needed continuous solid contours with no markers, so added `continuous_level_curves`: interior gaps filled as before, open ends continued where the level is bracketed by a lower and a higher level, and the whole curve resampled onto 361 angles with a monotone cubic that passes through every scheduled node exactly.
The end continuation interpolates linearly in log(1 - radial fraction) against log(speed), which reproduces solved 1000 m/s roots to within 0.34 MeV wherever 500/1000/2000 are all solved; 1000 and 2000 m/s now reach the density axis at 94.54 and 94.64 MeV, bracketed by the solved 500 m/s endpoint at 94.10.
5000 m/s has nothing above it to bracket against and stays a four-node fragment.
Also fixed two defects in that figure: 50 and 2000 m/s shared the colour `#1f77b4`, and `clabel` stacked the 1000/2000 labels where those curves converge, so labels are now placed manually per level.
`26SU-14.ipynb` keeps markers and dashed interpolation as the diagnostic view; the split is documented in [[isothermal-contour-repair]].

## [2026-09-23] fix | Explicit weak-rate thermal factor in baryon notation

Updated the hydrodynamical manuscript to expand the thermal factor in the weak source, isothermal integral, and speed formula, and express the common quark chemical potential through the baryon chemical potential throughout.
Wolfram verified the rate prefactor, the factors 81 and 162 in the rate and integrated speed, and the corrected cubic-dominance condition `a(0+) >> 9*pi*T/mu_B`.
Updated the bag-model and Debye-scale expressions consistently; the distinct electric-charge chemical potential and the unresolved neutrality-model discussion were left unchanged.
Preserved resolved PI requests as source comments, checked the rendered equations, and verified an isolated `latexmk` build with no undefined references or overfull boxes.
Recorded the formulas and the factor-of-three crossover correction in [[strangeness-reaction-diffusion]].

## [2026-09-23] fix | Consistent heat-current factors

Corrected only two equation factors in the hydrodynamical manuscript: `1/gamma` became `(1+2*u^2)/gamma` in `eq:energy_balance_exact`, and `3/2` became `5/2` in `eq:energy_conservation_thermal`.
Wolfram verified the stress-tensor components and their expansion under the existing acceleration-neglected Eckart closure.
A consistent change of metric signature does not produce the erroneous exact factor.
An isolated `latexmk` build passed without undefined references or overfull boxes.
Updated [[thermal-limits]] and [[thermal-limit-ordering]] to mark the earlier Fourier-based manuscript status as superseded.

## [2026-09-23] calc | Cold endpoints close the 1000/2000/5000 m/s contours

Added `new_paper_calculations/26-09-23/isothermal_cold_endpoints/`, which solves the two scheduled angles below the coldest accepted root of each of 1000, 2000 and 5000 m/s.
All six rays converged in 140--410 s and matched their target speeds to better than 0.25 m/s, although the cluster had reported every one of them as `target_budget_exhausted` -- the difference is that their neighbours now exist to anchor the continuation.
`continuous_level_curves` gained `extra=` to merge later checkpoints without touching the 26-09-16 payload, and `cold_anchor=True` to end every contour at (3 n0, 0), which the analytic grid and the solved 50--500 m/s roots both confirm is their common endpoint.
Resampling is now per node interval rather than uniform in angle, since the angular axis is geometric near zero and the cold tail is where the curve moves fastest.
Interpolated tails are now under 1.3 MeV long and agree with the independent analytic contour to 0.018/0.051/0.260 in density divided by saturation density.
Details in [[isothermal-contour-repair]].

## [2026-09-24] notation | Mostly-minus metric in the current manuscript

Converted `Latex_writting/Main.tex` to signature (+---), including the velocity normalization, stress-tensor pressure term, K-diffusion projector, covariant Eckart law, and explanatory comments.
The older hydrodynamical-subfolder copy and all solver files were left unchanged.
Wolfram verified orthogonality and equality of the physical currents and stress-tensor components before and after conversion; the one-dimensional equations were preserved.
An isolated `latexmk` build passed without undefined references, citations, or overfull boxes, and the affected equation pages were inspected.
Recorded the convention and the newer manuscript's exact-flux form in [[thermal-limits]].

## [2026-09-28] cluster | Fixed-interface full/semi-analytical comparison

Added `new_paper_calculations/26-09-28/` with unchanged copies of the 26-08-21 analytical engine and ray helper, plus the saved domain and consolidated 26-08-24 numerical input.
The 630-point union preserves the 540 displayed numerical coordinates and both exact angular endpoints; paired analytical evaluations share upstream recovery.
Bounded adaptive refinement resolves a semi-analytical 5000 m/s bracket per ray, with atomic point-level checkpoints, resumable provenance checks, explicit failures, and no full-analytical 5000 m/s requirement.
Syntax checks and 15 tests passed, including a mocked 690-point run with 30/30 target brackets; all six real evaluations at three representative points succeeded.
Verification used an isolated SciPy 1.18.1 wheel because the local 1.15.2 binary failed the macOS loader; the project environment and RMFsolver files were unchanged.
No production cluster scan was launched. Usage and output contracts are in [[fixed-interface-analytical-contours]].

## [2026-09-28] visualization | Downloaded fixed-interface comparison

Added and executed the 0.1 MeV three-method comparison in `All_Plots_new.ipynb`, immediately after the existing numerical contour, using the 5 MeV reference's plotting sizes and style.
The downloaded result has 634/702 accepted points for each analytical method; 67 paired timeouts and one upstream inversion failure remain masked.
The semi-analytical 5000 m/s bracket covers 29/30 rays, with theta index 1 unresolved; the plot does not bridge that gap.
Exported PNG and PDF to `local_analysis_results/All_Plots_new/T0plus0p1-three-method-comparison.*`, validated notebook structure and masks, and visually inspected the rendered figure.
Updated [[fixed-interface-analytical-contours]]; numerical data and existing notebook cells were preserved.

## [2026-09-28] diagnosis | Analytical endpoint gaps

Verified xi=-0.5 for both methods and confirmed that both endpoint rays were attempted: original-grid successes are 20/20 at 0 degrees and 9/20 at 90 degrees, with only 5/20 on the neighboring 3.103-degree ray.
Diagnosed legacy upstream inversion cost, a shared watchdog discarding paired results, a reproduced false density rejection caused by an unvalidated forward branch, and adaptive probes trapped below a failed coordinate.
Four failed points succeed locally using the existing direct-density upstream helper and unchanged analytical methods; additional probes verify both a 5000 m/s upper anchor and the need for smaller inner fractions to bracket 50 m/s near 90 degrees.
Filed evidence paths and environment caveats in [[fixed-interface-analytical-contours]]. Production code and contour data were not changed by this diagnosis.

## [2026-09-29] recovery | Local analytical retries and refreshed comparison

At the user's request, recovered all 68 failed paired cluster samples using the existing direct-density PNM helper and unchanged analytical formulas, with independent per-method time caps and point-level atomic checkpoints.
Added 30 inner samples and two bracket-refinement points: both methods now accept 734/734 points, and all 30 semi-analytical 5000 m/s brackets are complete.
Preserved the original NPY in a byte-identical pre-local-repair backup; verified exact retention of 634 previous successful records per method, the numerical reference and gray boundaries.
Updated and executed only the fixed-0.1-MeV comparison cell in All_Plots_new, retaining its style and exports. Verified connected analytical contours from theta=0 to theta=90 and filed provenance/evidence in [[fixed-interface-analytical-contours]].

## [2026-09-29] visualization | Matching-velocity comparison bands

Added translucent level-colored bands between numerical and full-analytical contour paths in All_Plots_new, clipped to their common angular coverage without extrapolation.
Bands cover 50 through 1000 m/s; 2000 and 5000 m/s remain unshaded because the full-analytical contours are absent in the calculated domain.
Regenerated notebook and LaTeX PDFs, synchronized both manuscript captions, and verified four geometry tests plus all five real band polygons. Solver data were unchanged; see [[fixed-interface-analytical-contours]].

## [2026-09-29] visualization | High-speed band preview

Added the requested 2000 and 5000 m/s bands using numerical-to-semi-analytical contours because corresponding full-analytical contours are unavailable, keeping alpha=0.16.
Notebook notes and both LaTeX captions identify this distinction explicitly. Six tests and execution checks verify all seven bands; plot exports were regenerated without changing solver data. See [[fixed-interface-analytical-contours]].

## [2026-09-29] visualization | Separate unshaded and shaded comparisons

Replaced the crowded seven-band preview with two figures in All_Plots_new: unshaded contours through 5000 m/s, and contours with numerical-to-full-analytical bands only through 1000 m/s.
Exported both PDFs to the manuscript Plots folder, retaining the unshaded version at the existing manuscript figure path and correcting its caption in both Main.tex copies.
Seven tests and executed-figure checks verify the requested limits, band counts and connected contours. Solver data were unchanged; see [[fixed-interface-analytical-contours]].

## [2026-09-29] diagnosis | State-dependent adiabatic shape factor

Compared 540 exactly matched plotted states and extracted diffusive-flux areas from 614 saved numerical profiles.
Measured xi becomes more negative at high speed; replacing -0.5 with the measured profile value reduces semi-analytical speed errors to below 0.1% on the matched plot grid.
Eight live endpoint checks distinguish exact speed-matching xi from the frozen-endpoint shortcut; full analytical requires xi below -1 at high speeds because additional interface-composition and integral approximations remain.
Saved CSV diagnostics, two diagnostic figures, reproducible scripts and explanation under local_analysis_results/26-09-29/xi-comparison/; filed conclusions in [[adiabatic-xi-diagnostics]] and superseded the old xi=0 warning.
No production solver, cluster payload, notebook, or manuscript was changed by this diagnosis.

## [2026-09-30] visualization | Clearer banded-contour labels

Enlarged the five banded-comparison contour labels from 11 to 14 points and added compact white backgrounds at alpha=0.85 to fade crossing curves locally.
Kept the unshaded variant's labels and both variants' data, bands and curves unchanged.
Refreshed All_Plots_new comparison outputs and local/LaTeX exports; eight tests pass, and rendered-artist checks verify both variants' label settings. See [[fixed-interface-analytical-contours]].

## [2026-09-30] diagnosis | Full numerical isothermal probes at 0.01 MeV

Ran 14 bounded single-worker physical trials over seven densities, including initialization comparisons and two tighter checks.
The default BVP initialization found no accepted result within 40 seconds; a layer-resolved initial mesh/profile recovered 74.535, 771.979 and 4139.965 m/s at density ratios 3.001, 3.01 and 3.05.
Verified positive monotone density profiles, local EOS and endpoint flux residuals; the tighter 3.001 check agrees to 1.08e-10 relatively, while the tighter 3.05 check and higher-density retries remain time-limited.
Saved raw point diagnostics, continuation states, audited JSON/CSV and reproduction scripts in local_analysis_results/isothermal_low_temperature_probe/.
No production solver, cluster checkpoint or notebook was changed. See [[isothermal-cold-numerical-probes]] for the evidence, separate SciPy runtime used, and limited log-temperature contour feasibility conclusion.

## [2026-10-01] verification | Neutral versus symmetric-chemical-potential composition plots

Derived the exact massless local-fraction curves at muB=1200 MeV and reproduced both constraints with approximate screenshot samples overlaid.
The neutral screenshot matches; the simple screenshot is inconsistent with the established definitions, predicting about 1.42 near muK=500 MeV against the exact 1.18969.
Verified finite a=1 endpoints at 669.007999 MeV (neutral, flavor ratio 1:2:0) and 344.127767 MeV (simple, residual strange density).
Checked Wolfram algebra, inverse composition mapping, charge residuals, and 28 live SQMsolver states; agreement in a is within 1.23e-15. Saved reproducible code, derivation, PNG/PDF figures and CSV data under local_analysis_results/26-10-01/composition_constraints/ and filed [[composition-constraint-plot-audit]].

## [2026-10-01] visualization | Revised PI simple composition comparison

The revised PI simple screenshot resolves the earlier mismatch with the exact massless EOS at muB=1200 MeV and T=0.
Independently digitized ten revised screenshot points, with maximum absolute difference in a of 0.00845, consistent with screenshot resolution and calibration.
Saved the standalone plotting script, separate revised samples, and visually checked PNG/PDF comparison under local_analysis_results/26-10-01/composition_constraints/; preserved the original audit artifacts.
Updated [[composition-constraint-plot-audit]] and its index entry to distinguish the resolved discrepancy from the original screenshot. No solver code changed.
