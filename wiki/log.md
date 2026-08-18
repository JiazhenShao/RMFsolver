---
summary: Append-only chronology of wiki operations. Grep the "## [" prefix for a timeline.
status: current
updated: 2026-08-17
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
