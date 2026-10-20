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

## [2026-10-01] implementation | Bounded cold numerical log-temperature contours

Added the 26-10-01/isothermal_numeric_logT cluster workflow: 31 logarithmic temperatures, seven speed levels, direct-density boundary roots, numerical density targeting, shared continuation profiles, persistent attempt reservations and pointwise atomic checkpoints.
Ported the recovered layer-resolved initializer into an opt-in phase_velocity API with explicit fallback and maximum BVP RMS diagnostics; existing defaults and physical closures are retained.
Bundled hashed immutable warm inputs and three optional cold profiles, added solver-free saved-data notebook cells without replacing the linear-temperature plot, and kept unresolved cold intervals disconnected.
Verified with lightweight orchestration, API/profile, preservation and plotting tests plus read-only preflight; stopped a legacy regression when it entered live EOS calculations rather than continuing heavy local work. Cold smoke/recovery and tighter accuracy checks remain for the cluster.
See [[isothermal-log-temperature-contours]] for commands, source compatibility, budgets, transfer requirements and completion criteria.

## [2026-10-01] manuscript | Shared layer thickness and adiabatic composition notation

Moved the conversion-layer thickness analysis in `Latex_writting/Main.tex` to the end of the isothermal analytical subsection, adding only the clarification that its asymptotic temperature equals the fixed temperature there and joining a split sentence.
Renamed the adiabatic composition ceiling from A to A_ad throughout the manuscript source, including result captions and the LTE appendix; retained all equation labels and left figure artwork unchanged.
Verified that equation contents match the previous source after undoing the rename and bracket sizing, and inspected the rendered thickness and LTE equations.
The isolated `latexmk` build completed without overfull boxes or undefined references; the pre-existing missing bibliography entry `Andersson:2006nr` and package/float warnings remain.
Recorded the notation and section location in [[analytic-front-speed]].

## [2026-10-01] diagnosis | Downloaded cold log-temperature run and analytical fallback

Audited the downloaded checkpoint without new EOS/BVP solves: 61 successful boundary temperatures, 13/427 accepted contour targets, 27/385 converged numerical trials, 225/225 timed-out analytical proposals and 14 blocked accuracy checks.
Verified matching solver/workflow hashes and traced unconditional continuation overriding the cold initializer, expensive nested analytical proposals, and retry termination after two failed density guesses.
Recommended against another unchanged production sweep; the failures do not establish absence of cold fronts.
Wolfram verified the xi -> -1+ frozen-formula maximum, with the caveat that exact xi=-1 sets the positive profile area to zero and is not a rigorous bound on the full numerical BVP.
Saved read-only audit code, JSON, CSV and report under local_analysis_results/26-10-01/isothermal-logT-audit/; frozen-endpoint comparisons show 0.76–5.34% excess at three earlier 0.01-MeV probes but 11.4–76.3% at the new converged points.
Updated [[isothermal-log-temperature-contours]], [[isothermal-analytic-front-speed]] and the index; production code, datasets and notebooks were unchanged by this audit.

## [2026-10-02] implementation | Analytical cluster workflows at xi=-0.85

Prepared the user-copied adiabatic and isothermal folders under 26-10-02 for fresh xi=-0.85 calculations.
Adiabatic uses both existing analytical methods at fixed T(0+)=0.1 MeV, validated direct-density upstream recovery, separate method timeouts, 660 initial paired coordinates and bounded high-speed refinement.
Isothermal replaces the BVP campaign with the public analytical solver, 322 contour targets across 46 logarithmic temperatures spanning 0.01–120 MeV, a static-only boundary bundle, cached analytical secants and immediate atomic point checkpoints.
Removed numerical profile/velocity reuse and nested proposal scans; fixed clipped-window wasted searches and separated figure rendering from checkpoint locking.
Moved copied legacy artifacts into recoverable sibling backups; original datasets, notebooks and RMFsolver files remain unchanged.
Verified 40 lightweight tests covering mocked orchestration, API/residual/edge handling, checkpoint/resume, immutable inputs, timeout termination and gap-preserving plotting, plus both solver-free dry runs; no production or live EOS/BVP smoke was launched locally.
Documented commands, transfer requirements and unmeasured cluster convergence in [[analytical-xi-minus0p85-cluster]], refreshed related pages/index, and unignored the new source/README files only.

## [2026-10-02] visualization | Downloaded xi=-0.85 comparisons

Created and executed `26-10-02.ipynb` with separate loading and plotting cells, reusing the banded adiabatic comparison style and the new 30-by-21 plotting grid.
The downloaded adiabatic data contain 748 full-analytical and 619 semi-analytical successes out of 749 evaluations; the isothermal analytical data have 292 accepted targets, 25 outside the window and five unresolved targets.
Added both linear- and log-temperature isothermal overlays: saved numerical warm polylines at/above 20 MeV and 58 distinct accepted numerical roots as colored dots below 20 MeV, with no cold numerical lines or interpolated pseudo-points.
Preserved analytical gaps, warm archival smoothing provenance, original datasets, old notebooks and all solver files.
Exported three PNG/PDF pairs under `local_analysis_results/26-10-02/`; ran eight saved-data tests, executed all notebook source cells and inspected all figures without EOS/BVP calls.
Updated [[analytical-xi-minus0p85-cluster]] and the index with the downloaded outcomes, data sources and visualization rules.

## [2026-10-02] recovery | Analytical gaps and 2000 m/s band

Diagnosed 129 adiabatic semi-analytical cluster timeouts, one cold upstream branch-corroboration failure, and clipped-edge isothermal density searches that exhausted their point allowances.
Ran targeted local analytical evaluations with two workers and pointwise atomic saving: recovered one full-analytical and 85 semi-analytical samples, plus all five missing isothermal targets.
Preserved every prior success and the baseline file hashes; repairs reside separately in `local_analysis_results/26-10-02/analytical-local-repair.npy`.
Confirmed single connected paths for all six displayed adiabatic levels and added the numerical-to-full-analytical 2000 m/s band over supported common angular coverage only.
Near-boundary checks show that the full-analytical 2000 m/s curve is not supported on the cold/intermediate rays within the shared domain; no band or speed was extrapolated there.
Re-executed `26-10-02.ipynb`, exported all comparison figures, and verified maximum recovered adiabatic relative closure error 1.18e-11 and isothermal target error 0.3421%.
The original numerical data, downloaded analytical checkpoints and RMFsolver sources were unchanged; updated [[analytical-xi-minus0p85-cluster]] and the index.

## [2026-10-02] maintenance | Notebook files renamed to creation dates

Renamed all 36 `YYSeason-N.ipynb` root notebooks to `YY-MM-DD.ipynb` using filesystem creation times; `All_Plots*`, `qcd_phase_diagram_paraQMCRMF3.ipynb` and `save_profiles.ipynb` were left unchanged, and `FL26-1/2/3` were included despite writing the season first.
Six files share two creation dates, so `26SP-3`, `26SP-4`, `26SU-2` and `26SU-5` became `26-07-17-1`..`-4`, and `26SU-11` and `26SU-13` became `26-08-18-1` and `-2`; the `2026-07-17` stamp is a bulk copy event rather than authorship.
⚠ Creation time is the only date signal available: no notebook carries date metadata, the notebooks are untracked by git, and five of them share the `2026-07-17 11:50` copy stamp.
Updated the five notebook-reading test modules, six wiki pages, three README files, one cluster-script docstring, and the `nbsrc.py` examples in `CLAUDE.md`.
Left data-artifact names intact — `local_analysis_results/26SU-11/`-style directories, `26SU-2-hydro-*` output filenames, and the `FL26-3_coupled_interface_roots` label — because renaming a reference without renaming the artifact would break the notebooks that write and load them.
Verified no stale `.ipynb` reference remains outside [[log]] and `docs/superpowers/`; the five affected test modules report 13 pre-existing failures from a broken `scipy` binary and drifted content assertions, with no missing-file errors.
Recorded the map in [[notebook-renames]] and linked it from the index.

## [2026-10-04] analysis | Extend the isothermal log-temperature contours through 1000000 m/s

Added analytical levels 10000, 20000, 50000, 100000, 200000, 500000, and 1000000 m/s to the logarithmic-temperature panel of `26-10-02.ipynb` at unchanged xi=-0.85 and physics.
Two bounded local workers reused the original fixed-temperature boundaries and saved speed brackets, producing 308 accepted contour roots and 14 outside-window classifications with no unresolved targets; 1175 total analytical evaluations include off-target bracket samples.
All new levels have one connected 44-point path, with maximum speed error 0.4971% and momentum-flux ratio 1.183e-4, below the unchanged acceptance thresholds.
Saved the two extensions and seed cache separately in `local_analysis_results/26-10-02/`, verified original input/solver hashes, and recorded residuals in `isothermal-high-speed-validation.json`.
Retained all seven original numerical warm curves and 58 cold numerical dots, with no new numerical data assigned to the added analytical speeds; moved the two-column speed legend into the gray region.
Refreshed notebook outputs and PNG/PDF exports, visually inspected the result, and passed 60 focused tests; the plotting cells import no EOS solver.
Used temporary SciPy 1.18.1 to avoid the existing macOS binary-loader error, without modifying the project environment or RMFsolver sources.
Updated [[isothermal-high-speed-contours]], linked it from [[analytical-xi-minus0p85-cluster]], and refreshed the index.

## [2026-10-04] implementation | Reproducible local isothermal calculation in the notebook

Replaced the isothermal loading cell of `26-10-02.ipynb` with a standalone numerical-data import and local analytical calculation/resume cell, leaving both visualization cells solver-free and preserving their appearance.
Added `notebook_isothermal.py` and a stdlib-only worker entry point under `new_paper_calculations/26-10-02/`; the fresh path reuses only static boundaries and calculates all 14 analytical levels on 46 temperatures, with no old analytical-speed samples or numerical BVP calls.
Exposed worker count, xi, time/evaluation limits, run-folder name, worker import paths, and an explicit plot-only switch; new run-folder names start fresh, existing names resume without overwriting prior datasets.
Reused the established atomic checkpoint and hard-timeout implementation, improved density proposals at clipped edges, and fixed resumption after an interrupted reserved evaluation without resetting its spent budget.
Passed 66 focused tests. A separate nine-target live check at 0.01, 20, and 83.8593 MeV produced eight accepted roots and one correct outside-window classification in 90 analytical evaluations; the notebook calculation cell then resumed it with zero additional evaluations.
The full 644-target new workflow was not rerun locally. Original cluster/recovery payloads and RMFsolver source hashes are unchanged, and the completed reference figures remain embedded.
The current project SciPy import issue is surfaced by a preflight, with a working kernel or explicit compatible worker-library path required; no environment package was changed while awaiting the user's runtime preference.
Updated [[isothermal-high-speed-contours]], [[analytical-xi-minus0p85-cluster]], and the index with the executable notebook workflow.

## [2026-10-04] debugging | Notebook SciPy runtime and per-evaluation progress

Reproduced the reported macOS loader rejection of the installed SciPy 1.15.2 PROPACK binary before any physics calculation.
Installed verified SciPy 1.18.1 separately under `local_analysis_results/26-10-02/runtime/`, leaving the Python environment and RMFsolver sources unchanged; the notebook now automatically selects the matching persistent runtime, prints its provenance, and reloads its helper on rerun.
Added explicit runtime preflight progress, a full contour-target bar, and a distinct counter refreshed after every atomically saved evaluation; resume restores counts without repeating successful work.
Passed 70 focused tests, including runtime overrides, preflight failures without reservations, and save-before-progress ordering.
Executed the actual calculation cell in a real Jupyter kernel for two 500 m/s targets at 0.01 and 20 MeV: both converged within 0.15%, requiring 24 evaluations; a second execution added zero evaluations.
Saved this bounded check separately in `local_analysis_results/26-10-02/notebook-runtime-fix-check/`; did not launch the full 644-target run or alter previous datasets/figures.
Updated [[isothermal-high-speed-contours]] and the index with the runtime fix and progress semantics.

## [2026-10-05] manuscript | Neutral density-variable EOS in Appendix A

Rewrote Appendix A of `Latex_writting/Main.tex` with the exact neutral flavor densities and their finite-temperature pressure, including the gluon term already present in the old fourth-order expansion.
Changed the quartic chemical-potential pressure term to its neutral negative sign and expanded both conjugate densities consistently; the quadratic pressure and leading composition relation are unchanged.
Wolfram returned zero residuals for the pressure and both density expansions, with numerical density-inversion checks at two finite-temperature points.
Commented out the purple non-neutral-EOS discussion verbatim, retained A1--A6 labels and numbering, and left all manuscript text outside Appendix A and all solver sources unchanged.
Built with latexmk and inspected the rendered appendix; no new undefined references or overfull boxes, with the existing missing `Andersson:2006nr` citation unchanged.
Updated [[quark-matter-eos]], [[charge-neutral-quark-eos]] and the index; Appendix B collaborator notes about the old EOS remain for separate review.

## [2026-10-05] manuscript | Highlight Appendix A and substitute the isothermal ceiling

Wrapped the revised Appendix A prose and formulas in `\JS{}`, leaving the unchanged quadratic EOS and PI comments as they were.
Substituted the neutral EOS susceptibility and equilibrium baryon density into Eq. (35), retaining the finite-temperature ratio and replacing the equilibrium chemical-potential endpoint with the requested constant quark-side superscript notation.
Wolfram returned zero for the difference between the old and new radicands, and the cold limit recovers the existing leading estimate; no new numerical ceiling comparison was performed.
Latexmk completed successfully, Eq. (35) and A1--A6 kept their numbering, and both rendered pages were inspected; only the pre-existing `Andersson:2006nr` citation remains unresolved.
Updated [[isothermal-maximum-speed]], [[charge-neutral-quark-eos]] and the index; no solver changes.

## [2026-10-05] literature | Supernova quark-coexistence oscillations

Reviewed the Mathews/Rupp/Boccioli APS DNP abstract forwarded by Liam Brodie, with Consensus and web searches for the supernova phase-transition and STIR literature.
Filed [[supernova-quark-oscillations]]: their Gamma<4/3 coexistence-region driver assumes equilibrium conversion, while the laminar front advances only ~0.03-3 m per 1 ms period.
No solver or manuscript changes.

## [2026-10-07] literature | Madsen weak-rate reference and approximation audit

Confirmed the bibliography metadata and traced the numerical helper to the same cubic-plus-thermal-linear rate as Appendix B.
The original paper's indexed text identifies Eqs. (20)--(21) as exact at zero temperature and strange mass, Eq. (22) as their small-imbalance limit, and Eq. (23) as the finite-temperature net-rate approximation; its full PDF remained inaccessible for visual inspection.
Recorded that exact neutral-EOS chemical potentials do not make this approximate rate an exact phase-space calculation, superseding the old full-rate wording in [[strangeness-reaction-diffusion]].
Updated that page and the index, with a separate open task to quantify rate errors across the front; no manuscript, bibliography or solver edits.

## [2026-10-07] physics | Unequal-flavor weak-rate benchmarks

Distinguished Madsen's general collision integral from Eq. (23) and the improved low-temperature prescription using Eqs. (20)/(21) and (24).
Derived the massless angular reduction independently from the invariant rate and checked its normalization and detailed-balance sign with Wolfram.
Neutral T=0 quadrature gives full/Eq. (23) ratios of 0.93249 at a=0.146, 0.72499 at a=0.5 and 0.35942 at a=0.9; three quadrature orders agree, and Wolfram independently integrated the a=1 endpoint exactly.
Filed [[unequal-flavor-weak-rate]] and updated [[strangeness-reaction-diffusion]] and the index; the inaccessible original PDF's Eq. (24) was not transcribed or guessed.
Finite-temperature profile errors remain unmeasured; no manuscript, bibliography or solver changes.

## [2026-10-07] physics | Massless finite-temperature rate and original-PDF correction

Read and rendered the user-supplied `ref-list/Rate-PhysRevD.47.325.pdf`, resolving the source-access limitation.
Both printed cold branches require an ordering that fails for our neutral d -> s conversion after exchanging flavor labels; corrected the earlier overly broad recommendation to use Eq. (20) or (21).
Recorded Eq. (24) explicitly and distinguished its improved thermal approximation from a systematic expansion for our branch.
Derived an exact massless reduction with two analytically integrated energy variables, leaving two numerical dimensions, and verified the antiderivatives with Wolfram.
Filed [[massless-finite-temperature-weak-rate]], including a near-equilibrium check and a converged prototype quadrature; no manuscript or solver changes.

## [2026-10-07] derivation | Standalone massless weak-rate manuscript

Created `ref-list/massless-weak-rate/` with a complete LaTeX derivation, local bibliography, Wolfram inputs/output, independent symbolic helpers, and a finite-temperature numerical reference.
Derived the correct cold branch for neutral ordering, its exact T-squared coefficient from four independent occupation thresholds, and the fixed-neutral-EOS chain rule.
Identified and verified the genuine T-cubed remainder from coincident up thresholds; a generic T-fourth remainder would be incorrect.
The 192-state grid covers both neutral EOS paths, muB900/1200, six compositions, and temperatures .01--100 MeV, with quadrature refinement changes below 1.71e-13 and explicit sampled 5% ranges.
Updated [[massless-finite-temperature-weak-rate]] and the index; the main combustion manuscript, shared bibliography, and production solver were left unchanged.

## [2026-10-07] verification | Independent check of the massless weak-rate manuscript

Re-derived `ref-list/massless-weak-rate/Main.pdf` without its helper scripts: four-threshold cold integral, both cold branches, J2 and every Appendix C boundary sum, the exact cusp polynomial, EOS shifts, spin trace, and normalization all agree.
An independent finite-temperature integrator reproduces the quoted cubic residuals, all 192 grid rates (5e-12), Table 2, and the neutral-EOS chain rule.
Found one transcription error: Madsen's Eq. (24) has `-(5/64) z v^4`; manuscript Eq. (8.2), `validate_rate.py:eq24_factor`, and [[massless-finite-temperature-weak-rate]] dropped the z, so the Figs. 1--2 "Cold + Eq. (24)" diagnostic is not his printed formula. Corrected the wiki page only; manuscript and code unchanged.

## [2026-10-07] correction | Apply Madsen reference-audit fixes

Verified the missing z against the original p. 329 scan, corrected the standalone manuscript and `eq24_factor`, and added a regression test that failed before the fix and passed afterward.
Reran all 192 states and regenerated the comparison figures; only the Eq. (24) rate/error columns changed, reproducing +15.4916%, +32.9033%, and +11.2145% at muB900,a1,T25/50/100.
Confirmed the printed Eq. (4) normalization discrepancy against Madsen's later equations and added a manuscript footnote without changing any derived normalization.
The 11-test suite passes and latexmk builds cleanly; preserved the independent verification notes and left the main combustion manuscript, shared bibliography, production solver, and SciPy installation unchanged.

## [2026-10-07] assessment | Exact-rate and NLO impact on front speeds

Integrated the exact tree-level rate over a(0+) at fixed muB=900/1200 MeV and T=0--80 MeV.
Since speed ~ sqrt(I2), replacing the solver's Madsen-at-exact-EOS rate by the exact rate lowers speeds by 3--16% for a(0+)<=0.5, 10--20% at 0.7, 19--30% at 0.9, and 27--40% at 0.99.
The linearized analytic rate stays within -9%/+3% of exact speeds for a(0+)<=0.7 and T<=0.035 muB, by cancellation.
Ingested arXiv:2609.35165: the O(alpha_s) correction is +41% at alpha_s=0.3 (about +19% speed), computed only near equilibrium with mu_u=mu_d.
Recorded in [[strangeness-reaction-diffusion]]. Recommendation given to the user: revise the rate-appendix text only, with no code or analytic rederivation. No manuscript or RMFsolver changes.

## [2026-10-07] visualization | Three weak-rate models and speed sensitivity

Updated only the standalone derivation and its numerical helpers to show the exact phase-space integral, Madsen Eq. (23), and the derived T0+T2 rate consistently in every comparison figure.
Added reproducible fixed-background rate-integral speed estimates at muB=1200 MeV, T=0 and 20 MeV, with convergence CSV/JSON and a generated comparison table.
The PDF explicitly distinguishes the sensitivity estimate from a solved front, the finite-temperature coefficient prescription from the strict constrained Taylor polynomial, and the cold-rate ratio from an absolute zero-temperature speed.
All 17 tests pass, the 20-page PDF builds cleanly, and every page was visually inspected.
Preserved combustion manuscript and production solver hashes; recorded the results in [[massless-finite-temperature-weak-rate]].

## [2026-10-07] computation | Six additional cold full-numerical comparison dots

Added and executed a bounded numerical computation cell immediately before the logarithmic-temperature figure in `26-10-02.ipynb`.
Recovered 500 and 2000 m/s roots at 0.01, 0.027550997842357025, and 0.09779327685429287 MeV, with maximum target error 0.3146% and passing standard BVP, composition, endpoint-flux, and seven-node profile EOS checks.
The first pilot exposed an unnecessarily doubled coexistence-distance step near an already close speed; preserved its checkpoint, changed to a small numerical-speed correction, and completed the final six targets in 13 evaluations on two workers with no final-batch failures.
Saved roots, off-target trials and physical continuation profiles separately in `local_analysis_results/26-10-02/isothermal-cold-numerical-points.npy`; tqdm updates only after atomic saving, and checkpoint resume repeated no numerical solves.
Retained one quadrature roundoff warning on the cold 2000 m/s root; a separate tighter mesh/tail check of the 0.01-MeV 500 m/s point timed out at 90 seconds and remains explicitly unresolved, not a mesh-independence claim.
Preserved all 58 original cold dots and added six matching-color dots, refreshed the embedded figure and PNG/PDF exports, and visually inspected the 64-dot plot without changing its window or styling.
Passed 78 focused tests and a real Jupyter resume/render check; original analytical/numerical archive hashes and all RMFsolver sources are unchanged.
Documented the code, results and limits in [[isothermal-notebook-cold-points]], linked it from [[isothermal-high-speed-contours]], and refreshed the index.

## [2026-10-07] computation | Bounded high-speed cold numerical comparisons

Extended the calculation cell before the log-temperature plot in `26-10-02.ipynb` to target 10000, 50000 and 500000 m/s at 0.01 and 0.027550997842357025 MeV, with independent checkpoints and tqdm progress.
Recovered one 10000 m/s point at 0.027550997842357025 MeV: density ratio 3.2359548050553277, numerical speed 10003.19563277806 m/s, target error 0.03196%, passing standard BVP/composition/EOS/flux checks and no captured warnings.
The other five primary targets timed out at 90 seconds; one warmer fallback each for 50000 and 500000 m/s at 0.09779327685429287 MeV also timed out.
Eight targets used ten new numerical evaluations (three converged density trials and seven timeouts); only the accepted contour root was plotted, with unresolved higher-speed points explicitly absent.
Enabled high-speed cold dots independently of the seven warm numerical curves, retained all previous points and plotting controls, and refreshed the embedded figure plus PNG/PDF exports to show 65 cold dots.
Passed 78 focused tests and a real Jupyter resume/render check with zero extra BVP evaluations; visually inspected the plot and verified original archive, six-root checkpoint and solver hashes unchanged.
Updated [[isothermal-notebook-cold-points]], [[isothermal-high-speed-contours]] and the index; no tighter mesh/tail claim or numerical substitute was introduced.

## [2026-10-08] literature | Fermi-surface nonleptonic rate formula

Traced the 2<->2 Fermi-surface energy integral (4 pi^2 xi + xi^3)/(6(1-e^{-xi})) to Ghosh, Haber, Andersson and Counsell, arXiv:2608.07311 (Eqs. A8-A10); Alford and Haber, PRC 103, 045810 (Eq. 14); and Shapiro-Teukolsky, Yakovlev et al. 2001, and Kaminker et al. 2016.
Confirmed that Madsen credits Eq. (23) to Heiselberg-Baym-Pethick and Koch (1991).
Applied to our quark rate with unequal Fermi momenta, the approximation is within 1.5% of the exact cold rate through a=0.5 but falls to 0.68, 0.26 and 0 at a=0.9, 0.99 and 1. Recorded in [[strangeness-reaction-diffusion]].

## [2026-10-08] literature | Thermal gluons in bag-model EOSs

Used Consensus discovery and fetched paper records, then checked primary EOS equations in finite-temperature combustion, hot-star bag extensions, the user's Entropy 26, 183 paper, and the cold ABPR parameterization.
Niebergal et al. (2010) and Herzog-Ropke (2011) include thermal gluons through the 19*pi^2*T^4/36 pressure coefficient; Lopes et al. and vBag examples show that a separate ideal-gluon gas is not universal across effective extensions.
Separated thermal gluons, gluon-mediated interaction corrections and vacuum bag energy; avoided claiming a measured majority from a targeted survey.
Confirmed current SQMsolver pressure, energy and entropy all include free thermal gluons even with include_em=False; Wolfram checked coefficients, derivatives and thermal-pressure ratios.
Filed [[bag-model-thermal-gluons]], corrected the wiki's missing explicit gluon term in the former-basis pressure line with a dated note, and updated the index. No solver, notebook or manuscript files changed.

## [2026-10-08] manuscript | Authoritative bag-EOS review citations in Appendix A

Verified Madsen's review Sec. 3.4, Eqs. (62)-(65) and the explicit bulk-limit statement after Eq. (72), plus Buballa's Physics Reports review Sec. 2.1.2, Eqs. (2.4)-(2.7).
INSPIRE returned database-specific citation counts of 246 and 1752, respectively; imported Buballa:2003qv verbatim and reused Madsen:1998uh.
Added both citations at the end of the requested opening pressure paragraph in Latex_writting/Main.tex, preserving the physics and collaborative comments.
Rebuilt the manuscript and verified references [45, 50] in the rendered Appendix A; the existing unrelated Andersson:2006nr citation and eq:energy_flux_adiabatic reference remain unresolved.
Recorded exact source locations and the distinction between Buballa's general formula and two-flavor illustration in [[bag-model-thermal-gluons]], then refreshed the index. No solver or notebook changes.

## [2026-10-11] calculation | Isothermal extension and conserved adiabatic weak-rate sensitivity

Implemented the approved standalone extension in ref-list/massless-weak-rate/: Section 8.5 now explicitly labels the fixed-background isothermal comparison and Figure 4 includes 0,20,50,100 MeV; all 88 prior cold/20 rows are unchanged.
Added Section 8.6 with QMC-RMF3 upstream inputs, bag scale 189.1566 MeV, cold quark interface, constant pressure and enthalpy per baryon, full thermal-gluon EOS, local two-term diffusion, and three same-state weak rates.
Saved 153 upstream states, 45 compositions, 143 quadrature records and 11,016 physical trajectory samples, plus reproducible drivers, figures, tables, source provenance and independent convergence checks.
The 100 MeV slice reaches only a(0+) >= 0.879368 within 1..10 n0; both density branches are retained where present. Fixed-composition speed ratios are universal under massless scaling: Eq.23/exact = 1.036985, 1.166607, 1.561072 at a(0+) = 0.146, 0.5, 0.9.
Verified the improper cold endpoint without a temperature floor using Wolfram, independent high-precision EOS/a-coordinate integration and decreasing-cutoff checks; 34 tests pass and all accepted speed-ratio changes stay below 1e-3.
Rebuilt and inspected the standalone manuscript; no production solver or combustion-manuscript files were changed. Filed [[adiabatic-weak-rate-sensitivity]], marked the older two-temperature scope historical, and updated the index.

## [2026-10-11] presentation | Consistent weak-rate figure styling

Matched adiabatic and isothermal model markers, line styles and blue endpoint labels in the standalone weak-rate manuscript.
Redrew all seven figures at the actual 6.5-inch manuscript text width with ordinary plot text at least 12 points; six-panel figures now use two columns.
Added plot_manuscript.py and shared plotting modules, preserving the hashed collision-integral source and every saved numerical-data hash. Typography/source audit passes, 21 relevant tests pass, and the 29-page PDF rebuild has no warnings or overfull boxes.
Visually inspected every figure page and corrected a clipped endpoint tick; documented reproduction in the README and [[adiabatic-weak-rate-sensitivity]], then refreshed the index.

## [2026-10-11] audit | Repeated adiabatic temperature-panel ratios

Traced the identical plotted ratios to explicit dimensionless-quadrature reuse at fixed interface composition. Verified the scale cancellation symbolically and added audit_adiabatic_physical.py to avoid relying on that reuse as evidence.
Independently reconstructed six dimensional quark EOS trajectories from saved upstream pressure/enthalpy, then integrated in composition with 40-digit EOS inversion and no production trajectory class or cached integrals. At upstream temperatures 0,20,100 MeV and compositions 0.9,0.995, published ratios agree within 2.10e-6; temperature-to-temperature spread is below 5.16e-13. Absolute temperatures and integrals differ.
Saved results/adiabatic-physical-audit.json and documented the fixed-composition scaling, differing upstream densities and distinction from isothermal comparisons in [[adiabatic-weak-rate-sensitivity]], README and the Wolfram transcript. No plotted data or manuscript PDF were changed.

## [2026-10-11] check | Adiabatic rate-ratio universality

The identical adiabatic Eq. (23)/exact speed ratios across T(0-) in Sec. 8.6 of `ref-list/massless-weak-rate/` come from exact massless-bag scaling with T(0+)=0, not from a bug.
The saved data and an independent EOS solve agree to 10 digits. Recorded in [[strangeness-reaction-diffusion]].

## [2026-10-11] presentation | Adiabatic upstream-plane speed map

Replaced the six-panel ratio-versus-a(0+) figure in the standalone Section 8.6 with one (nB(0-)/n0, T(0-)) map of Eq.23/exact iso-ratio contours, since a(0+) is fixed by the upstream state when T(0+)=0.
Grid extended to 49 x 101 (0-120 MeV, 1-6 n0), no EOS failures; the a(0+)=1 boundary reaches n0 at T(0-)=112.28 MeV, so both boundaries close inside the plot.
Section text, caption and README updated; 29-page PDF rebuilds without warnings. See [[adiabatic-weak-rate-sensitivity]].

## [2026-10-11] presentation | (mu_B/3)^5 rate variant in local-rate figures

Added Madsen Eq. (23) with mu_u^5 -> (mu_B/3)^5 as a fourth, plotting-only model in standalone Figures 1-3 (rate_figures.py), with model-list text, captions and README updated; validate_rate.py untouched, 34 tests pass, 29-page PDF builds without warnings.
At T=1 MeV it overestimates the exact rate by 6.5/24.4/65.1/121 percent at a=0.146/0.5/0.9/1. See [[massless-finite-temperature-weak-rate]].

## [2026-10-11] presentation | (mu_B/3)^5 variant in Figures 4 and 6

Added the (mu_B/3)^5 Eq. (23) variant to the isothermal speed-sensitivity figure (closed-form quadrature on the saved nodes, reproducing saved Eq. 23 integrals to 2e-16; new speed-sensitivity-eq23-muB.csv) and to the trajectory diagnostic, whose integrand panel now shows percent error relative to exact. Speed ratios at a(0+)=1: 1.29/1.27/1.22/1.03 at T=0/20/50/100 MeV; 1.252 on the adiabatic a(0+)=0.9 trajectory. 35 tests pass; PDF builds without warnings. See [[massless-finite-temperature-weak-rate]].

## [2026-10-12] manuscript | Exact massless EOS citation and LaTeX repair

Verified Laine and Vuorinen, Sec. 7, Appendix A, Eq. (7.42), as the exact free massless Dirac-gas result; three colors reproduce the manuscript pressure and its derivative reproduces the net density, including antiquarks, without a low-temperature truncation.
Added the INSPIRE entry and highlighted statement in `Latex_writting/Main.tex`, removed conflicting `notag` declarations, supplied the missing Andersson–Comer citation, and repaired a stale energy-conservation reference.
Removed unused REVTeX-incompatible float packages, wrapped the overwide speed equation, and placed the two numerical comparison panels side by side.
The 15-page PDF builds without undefined citations, labels, or overfull boxes; a deferred-float warning remains, with the affected figures visually verified in the output.
No solver or standalone rate-derivation code changed.
See [[quark-matter-eos]].

## [2026-10-12] manuscript | Explicit arXiv version for the exact EOS citation

Changed the equation citation to `Laine:2022thermalnotes`, explicitly linking arXiv:1701.01554v2, with Eq. (7.42) as the only locator.
The rebuilt bibliography maps it to [52], with Buballa's hep-ph/0402234 separately at [51]; no undefined references remain.
See [[quark-matter-eos]].

## [2026-10-12] analysis | Best constant xi for the adiabatic analytical comparison

Evaluated both public analytical methods at xi=-0.85 on all 618 successful T(0+)=0.1 MeV numerical upstream states (617 succeed) and rescaled exactly to other xi through the (1+xi a(0+)) factor, verified against live solves to 4e-10.
Equal-weight, contour-band-balanced RMS of ln(u_analytic/u_numerical) over 50-2000 m/s is minimized at xi=-0.865 (1% basin -0.89 to -0.845); -0.85 is within 0.5% of that minimum.
Semi-analytical alone prefers -0.83; full analytical prefers the -1 limit at high speed, which compensates its composition and integral errors rather than measuring a shape factor.
Evidence in `local_analysis_results/26-10-12/xi-optimum/`. See [[adiabatic-xi-diagnostics]].

## [2026-10-14] implementation | Tunable isothermal mean-free-path / tail-length overlay

Created `26-10-14.ipynb` with saved isothermal data loading, a separate bounded/cached static EOS calculation, and solver-free plotting.
Default fraction is 1.0, full diffusion; optional Landau diffusion uses the same coefficient in both lengths. Roots require a real density bracket and relative ratio error at most 1e-6; no roots means no overlay or legend.
Six live endpoint states across three temperatures passed EOS validation, with sampled full-diffusion ratios 2.72603e-7 to 1.11485e-6 and no crossings at fractions 1 or 0.1; fraction/model changes reused the cache. The complete 46-slice assessment was not pre-run, and no BVP was executed.
Tests cover root search, failure/resume/checkpoint ordering and exact preservation of plotted data. Original notebook, datasets and RMFsolver sources remain unchanged.
Documented the distinction from the appendix's local-gradient LTE criterion in [[isothermal-mfp-overlay]], added workflow README, and refreshed the index.

## [2026-10-14] physics | Cold core versus downstream-tail validity

Clarified that the tail ratio scales as T^(1/6) with a small dimensional prefactor and cannot justify a sub-keV exclusion or an extrapolated high-temperature cutoff.
Wolfram verified the cubic-core scale ratio 3*a*sqrt(D_K*gamma_K), its T^(-5/6) behavior at fixed finite composition, and the slow-front interface concentration-gradient ratio 3*u(0+)*(1-a(0+))/a(0+).
At the illustrative fixed coefficients muB=1200 MeV and a=0.5, the core ratio remains about 0.00404 at 1 keV; no universal 1-keV breakdown was established.
Recorded the proposed local-gradient diagnostic and its limitations in [[isothermal-mfp-overlay]]. No notebook, solver, or dataset changed, and no EOS/BVP run was performed.

## [2026-10-14] calculation | Standalone core mean-free-path curve at fraction 1e-4

Added independent calculation/resume and visualization cells to the end of `26-10-14.ipynb`, using the static maximum-composition interface and transport coefficients at 0plus rather than the downstream tail.
Found 33/33 roots with 226 direct-density EOS evaluations, two single-threaded workers, bounded calls, pointwise atomic saves, and tqdm; no BVP or solver-source change.
Sampled roots range from 0.515 keV near coexistence to 0.172 MeV, with maximum relative criterion error 5.63e-7. One retained quadrature warning passed upstream and interface residual checks.
Saved separate data and standalone log-temperature PNG/PDF under `local_analysis_results/26-10-14/`; cached notebook reexecution adds no EOS evaluations.
Documented that exceeding the chosen 1e-4 core tolerance is not a sharp LTE breakdown proof in [[isothermal-mfp-overlay]], and preserved the earlier tail calculation and original inputs.

## [2026-10-14] visualization | Static excluded regions on the core-ratio curve

Added gray stable-NM and unit-composition regions to the standalone core plot in `26-10-14.ipynb`, preserving all 33 core points.
Reused 19 saved static boundary slices and solved five missing slices directly at fixed density, covering 1e-4 through 1 MeV without extrapolation or BVP runs.
All 24 boundary slices passed validation; new residuals are below 1.01e-9 MeV with no warnings. Separate boundary checkpoints save before tqdm advances and reuse terminal results.
Verified region coverage and original core-line coordinates in plotting tests; refreshed the notebook output and PNG/PDF. See [[isothermal-mfp-overlay]].

## [2026-10-15] export | Collaborator phase-space benchmarks

Freshly calculated 60 local massless weak-rate states for muB=900,1200 MeV and a=0.146,0.5,0.9 across ten temperatures from 0 to 100 MeV. Saved combined and six panel CSVs under local_analysis_results/26-10-15/massless-phase-space/, plus a shareable ZIP.
Exported dimensional flavor potentials, raw forward/reverse/net phase space, the weak prefactor, absolute rates and all four plotted comparison prescriptions. README specifies the exact integral normalization, units, detailed balance and fixed-muB,a EOS path.
Order/tail refinements pass 1e-6; 12 independent radial 3D checks, six exact cold-polynomial checks and all 48 overlapping manuscript rows agree. Verified CSV round-trip precision, normalization, source snapshots and archive integrity; included reproducible calculation code and metadata. Updated [[massless-finite-temperature-weak-rate]] and the index. No production solver or manuscript files were modified.

## [2026-10-15] visualization | Matched manuscript trajectory figures

Updated the adiabatic legend and A_adb intercept, added a matching-style isothermal cell directly below the retained original, and changed figure input labels to U_bag^(1/4).
Saved notebook outputs and overwrote T_a_constant_pressure.pdf and isothermal_muB_muK_plane.pdf under the original manuscript paths; the new isothermal cell supports cached replotting after restart.
Numerical profiles are unchanged; the reconstructed isobar matches their pressure to 1e-15. All 23 unrelated cells are preserved; three new style tests and eight contour-band tests pass.
The manuscript PDF was rebuilt with the existing citation/reference warnings; only its two trajectory captions were updated to match the new labels/markers and the saved adiabatic interface fraction 0.538, with no body-text changes. See [[notebook-trajectory-figures]].

## [2026-10-15] presentation | October checklist meeting notes

Created root-level `26-10-14.md`, following all eight October 8 checklist items with calculations, saved benchmark tables, source links, and short speaking notes.
Embedded 11 portable PNGs under `26-10-14.assets/`: ten unchanged existing figures and one saved-data replot of the band-balanced xi fit; source hashes and scan CSV are retained.
The notes distinguish fixed-profile rate sensitivities from BVP solutions, bounded tail checks from the separate chosen core tolerance, and user-reported Box/Overleaf uploads from locally verified exports.
No production solver, notebook, manuscript, or underlying result file was modified, and no EOS/BVP calculation was run. See [[group-meeting-2026-10-16]].

## [2026-10-15] visualization | Recovered momentum-flux profile figure

Located the original figure in `26-04-09.ipynb`, cell 9, and its matching vector PDF under `jargon/`.
The old calculation inputs have changed, and the matching raw profile was not found; recovered six original vector contours, four sampled positions and three endpoints rather than silently substituting another run.
Appended provenance and cached plotting cells to `26-10-14.ipynb`, preserving all nine original cells. Enlarged fonts, replaced endpoint labels with location notation, exported PNG/PDF, and embedded the PNG in `26-10-14.md`.
The historical plotted endpoint change is about 0.073 MeV (0.0064%) at figure precision, not a current-EOS verification or general constant-potential error bound.
Four focused tests pass. No production solver, original notebook or manuscript changed, and no EOS/BVP calculation was run. See [[notebook-trajectory-figures]] and [[group-meeting-2026-10-16]].

## [2026-10-16] physics | Checked the one-percent isothermal ceiling claim

Compared the manuscript Taylor estimate with the exact static neutral-EOS ceiling for 292 successful saved isothermal targets; 113 exceed 1% relative error and the largest error is 8.48%.
An independent density-cubic/bisection calculation reproduces the saved ceilings within 1.1e-11, with converged iteration counts and pressure/Gibbs residuals below 5e-15.
Wolfram verifies the cold series, the 1% crossing at a=0.495544, and two finite-temperature counterexamples. The plotted T=10 MeV, 3n0 example itself has only 0.0802% error.
Saved reproducible code, all target comparisons, and Wolfram inputs/outputs under local_analysis_results/26-10-16/aiso-accuracy/. Updated [[isothermal-maximum-speed]] and the index.
No manuscript, notebook, or production solver was changed; the solver already obtains its ceiling from the exact static EOS rather than this Taylor estimate.

## [2026-10-16] visualization | Exact-baseline composition-ceiling comparison

Appended provenance and an executed plotting cell to `26-10-14.ipynb`, preserving all 11 pre-existing cells and outputs.
Added an "Additional Check: Accuracy of Eq. (35)" section to `26-10-14.md`, with the figure, exact-denominator definitions, representative comparisons, and the checked accuracy range.
The three-panel figure shows values, signed differences, and percentage differences versus exact composition; saved points are colored by temperature and the cold-limit curve is separate.
Exported PNG/PDF under local_analysis_results/26-10-14/ and a report PNG under 26-10-14.assets/; saved a reproducible source cell with local font/style controls and updated asset provenance.
Verified all 292 plotted points against the audit, exact zero-error baselines, notebook preservation, and PDF/PNG legibility. No new EOS/BVP solve or manuscript/solver edit was made. See [[isothermal-maximum-speed]] and [[group-meeting-2026-10-16]].

## [2026-10-19] build | Main manuscript LaTeX repair

Restored the missing isothermal log-temperature PDF from the existing generated artifact, separated the duplicate isothermal speed label, and redirected two stale cross-references.
Added four exact INSPIRE-HEP bibliography exports and regenerated a malformed Main.bbl.
The active `Latex_writting/Main.tex` now builds successfully to 17 pages with no undefined citations/references; checked the restored figure and affected pages visually.
Existing package/float warnings and a small equation overfull box remain; no physics formulas, notebooks, or solver files were changed. See [[combustion-paper]].

## [2026-10-20] physics | Deconfinement quench temperature

Energy conservation fixes $T$ of freshly deconfined matter: at $3n_0$, $T(0^-)=10$ MeV, pure n->udd deconfinement lacks 149.96 MeV/baryon of enthalpy, and full conversion cools to $T(\infty)=6.868$ MeV; udd is allowed above $5.2836\,n_0$ ($T_q=45.58$ MeV at $6n_0$).
Wolfram-verified $T^2/T^4$ series and production-EOS checks pass; files in local_analysis_results/26-10-20/deconfinement-quench/. Not signed off. See [[deconfinement-quench-temperature]].
