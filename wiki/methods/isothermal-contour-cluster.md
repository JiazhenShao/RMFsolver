---
summary: One run_isothermal_all.py command uses the allocated CPUs for pointwise-checkpointed domain, analytical, and physical-nK/jK contour stages.
status: current
updated: 2026-08-20
tags: [method, cluster, contour, isothermal]
---

# Isothermal contour cluster workflow

`new_paper_calculations/26-08-19/` is the cluster-ready workflow for comparing `analytic_velocity_isothermal` with the exact `solve_front_isothermal` BVP on identical thermodynamic-maximum coordinates.
From that directory, its production entry point is simply `python3 run_isothermal_all.py`.
It takes the scheduler CPU allocation, or all locally available CPUs, and runs domain preparation, analytical cells, numerical composition shells, and plotting sequentially.

## Boundary-fitted domain

The production moving grid has 30 positive temperatures from $0.01$ to $120$ MeV and 20 targets uniformly spaced over $0.01\leq a(0^+)\leq0.99$.
The separate phase-curve axis also includes exactly $T=0$; no moving solver is called there.

The lower boundary is PNM--equilibrated-quark coexistence at common $(P,T,\mu_B)$ and $\mu_K=0$.
The upper boundary is the strangeness-free interface state with $n_s(0^+)=0$.
For each interior target, the domain stage inverts the live `_solve_a_0plus_max` result between those boundaries rather than copying its closure:

$$
P(0^+)=P(0^-),\qquad
a(0^+)=\frac{n_K(0^+)}{n_B(0^+)},\qquad
\mu_B(0^-)=\mu_B(0^+)+a(0^+)\mu_K(0^+).
$$

Both public velocity calls omit `a_0plus`, independently recover `a_0plus_source="maximum"`, and must agree with the domain target within $10^{-7}$.
This is the current **static-isobar thermodynamic ceiling**, not a two-eigenparameter finite-flux maximization.
Near $a(0^+)=1$, the analytical speed and momentum-flux correction grow; the exact endpoint is a shaded boundary, not a velocity datum.

## Execution and failure isolation

Analytical cells are independent, run in disposable spawned processes with a default 300-second hard limit, and checkpoint after every terminal result.
The numerical stage advances in increasing-$a(0^+)$ shells, parallelizing only temperatures within a shell.
Its deterministic baryon-current seeds use the preceding one or two composition shells, the same-cell analytical current, bounded multiplicative variants, nearby successful temperatures, and finally a density-scaled fallback.

Every numerical attempt runs in a disposable spawned process.
The default hard limits are 180 seconds per attempt and 900 seconds per cell; timeout children are terminated, joined, and killed if necessary.
Payload writes use temporary siblings plus `fsync` and atomic replacement.
The parent process is the only payload writer: for each completed $(i,j)$ result it updates the in-memory record, atomically replaces the `.npy` file, and then advances `tqdm` by one.
Consequently, the visible count never gets ahead of the recoverable checkpoint, and completion is counted per point rather than per temperature row or composition shell.
The command displays five calculation bars in order: the stable-neutron-matter boundary, the $a(0^+)=1$ boundary, the 600-point domain grid, the 600-point analytical scan, and the 600-point numerical scan.
Domain boundary continuation remains serial, while every independent grid stage uses the available CPU allocation; numerical temperature points are concurrent within each ordered composition shell.
The domain stores a deterministic fingerprint over all phase boundaries, curvilinear coordinates, masks, axes, physical inputs, and live API signatures.
Resume continues a partially checkpointed domain, rejects a mismatched fingerprint or solver-control set, and skips every terminal cell.
The domain grid avoids process-pool semaphore limits by using coordinator threads that each launch one spawned inversion child, so the requested CPU allocation remains usable on restricted cluster nodes.

Only finite `task_status="success"` cells enter contours.
Stable-PNM classifications, coexistence disagreement, analytical validity gates, ceiling saturation, exact-model mismatch, non-finite BVP diagnostics, exceptions, solver failures, and timeouts remain explicit masks; no failure becomes zero or infinity.
The returned proper velocity is converted for plotting as

$$
v=c\frac{u(0^-)}{\sqrt{1+u^2(0^-)}}.
$$

## Verified smoke result

The 2026-08-20 three-temperature by three-composition smoke run produced 9 ordered interior domain cells and 9 successful analytical cells.
The maximum composition inversion residual was $4.6\times10^{-12}$, every moving analytical record had $\Delta\mu_B<0$, and the speed conversion agreed exactly to printed precision.
With deliberately reduced 20-second attempt and 40-second cell limits, all 9 exact BVP cells ended as structured `cell_timeout` masks; later shells still ran and the comparison figure completed without infinity.

See [[isothermal-analytic-front-speed]] for the physics and [[phase-velocity-overview]] for the public API contracts.
