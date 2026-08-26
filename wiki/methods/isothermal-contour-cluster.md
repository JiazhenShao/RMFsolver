---
summary: The restored analytical isothermal contour keeps 600 cells, uses a 0.01-to-20-degree logarithmic angular block, and solves each boundary on 24 Chebyshev rays.
status: current
updated: 2026-08-26
tags: [method, cluster, contour, isothermal]
---

# Isothermal contour cluster workflow

`new_paper_calculations/26-08-20/` compares `analytic_velocity_isothermal` with the physical-$n_K,J_K$ `solve_front_isothermal` BVP at identical physical points.
From that directory the production entry point is `python3 run_isothermal_all.py`.
The launcher uses the scheduler CPU allocation, or all locally available CPUs, and executes domain preparation, the analytical contour, outward-shell numerical continuation, and plotting in order.

## Current 26-08-25 analytical workflow

`new_paper_calculations/26-08-25/isothermal_analytic/` now uses the successful 26-08-20 domain and analytical execution code as its baseline.
Only the angular sampling changed; its EOS calls, worker execution, root logic, checkpointing, physical window, and radial fractions remain on that baseline.
It retains the 20 established radial fractions and 30 angular rays, hence 600 physical cells.

The angular axis is now piecewise: an exact $0^\circ$ ray, 15 positive logarithmic points from $0.01^\circ$ through $20^\circ$, and 14 linear points above $20^\circ$ through $90^\circ$.
The exact $0^\circ$ ray remains clamped to $T(0^-)=0.01$ MeV.
The $20^\circ$ seam appears exactly once.

Both the $\Delta\mu_B=0$ and $a(0^+)=1$ boundaries retain the endpoint-clustered Chebyshev construction but increase from 12 to 24 directly solved rays.
Their radii are interpolated onto the 30 contour rays for mesh construction.
Schema 4 and the domain fingerprint reject the earlier uniform 30-ray contour axis and 12-ray boundary checkpoints.
The parent atomically checkpoints each boundary ray, domain cell, and analytical cell before advancing its pointwise progress bar.
The default physics remains $B^{1/4}=189.1565957288247$ MeV, $\xi=-0.5$, $m_s=0$, `NM_type="PNM"`, and `upB=5000`.

⚠ The deleted experimental 26-08-25 implementation used 61 piecewise boundary rays and a different persistent-worker/direct-density path.
It is superseded by the restored 26-08-20 baseline described above; do not infer its execution behavior from the retained historical log entries.

## 26-08-25 numerical-only rerun

`new_paper_calculations/26-08-25/isothermal_numeric/` reuses the completed 600-cell `isothermal-domain.npy` directly and does not execute domain construction.
Its single production command is `python3 run_isothermal_numerical.py`.
The sibling `../isothermal_analytic/isothermal-analytic.npy` is used only for per-cell baryon-current seeds; every numerical endpoint state, automatic composition ceiling, and BVP is independently recomputed by `solve_front_isothermal`.

The numerical stage advances through increasing radial shells so a ray's accepted baryon current seeds the next shell, while angular rays within a shell run concurrently on the scheduler allocation.
Each BVP attempt remains isolated behind a hard timeout, and the parent atomically writes `isothermal-numerical.npy` before incrementing the 600-point progress bar.
The numerical-only rerun uses the public solver's current `tail_eps=1e-3` default rather than the copied historical `1e-8` cluster override.
Resume reuses all terminal cells and rejects changed controls, axes, physics, live API signatures, or domain fingerprints.

⚠ A four-corner production-configuration smoke test on 2026-08-26 found that this numerical rerun is not yet cluster-ready.
Only the high-angle inner corner completed its first trial, in 71.9 s; two corners hit the 180 s trial limit and the low-temperature outer corner failed after 154.9 s.
The root cause is in the live numerical solver rather than domain construction: every trial repeats two upstream fixed-density RMF solves and the obsolete 48-point branch-validated $\mu_B$ scan before starting the BVP.
At $T(0^-)=0.01$ MeV those repeated RMF iterates exercise the difficult finite-temperature quadratures at `Solver.py:223` and `Solver.py:341`, producing the observed warning flood.
See [[known-issues]] for the measurements, candidate amplification, and the separate outer-corner current-bound mismatch.

## 26-08-20 matched analytical--numerical domain

The production calculation uses the same coordinate window and mesh structure as the 26-08-18 energy-conserving cluster workflow:

$$
1\leq \frac{n_B(0^-)}{n_0}\leq5.5,
\qquad
0.01\ {\rm MeV}\leq T(0^-)\leq120\ {\rm MeV}.
$$

The hidden dimensionless coordinates are

$$
X=\frac{n_B(0^-)/n_0-1}{4.5},
\qquad
Y=\frac{T(0^-)}{120\ {\rm MeV}},
$$

with $X=\rho\cos\theta$ and $Y=\rho\sin\theta$.
The $\theta=0$ ray is clamped to $T(0^-)=0.01$ MeV because the transport microphysics does not accept exact zero temperature.

Twelve endpoint-inclusive Chebyshev angular rays trace both boundaries.
The inner radius is the PNM--equilibrated-quark coexistence root $\Delta\mu_B=0$ at common pressure and temperature with $\mu_K(\infty)=0$.
The outer radius is the $a(0^+)=1$ Gibbs-balance root.
Both residuals reuse the same upstream nuclear state at each radial probe.

The contour interpolates those boundary radii onto 30 endpoint-inclusive uniformly spaced rays.
Its 20 outward radial fractions are exactly

```text
0.02, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
0.50, 0.55, 0.60, 0.66, 0.72, 0.78, 0.84, 0.90, 0.95, 0.99
```

For ray $i$ and fraction $s_j$,

$$
\rho_{ij}=\rho_{\rm inner}(\theta_i)
+s_j\left[\rho_{\rm outer}(\theta_i)-\rho_{\rm inner}(\theta_i)\right].
$$

Mapping $\rho_{ij},\theta_i$ back to physical variables gives 600 points directly.
The domain no longer constructs a fixed-$T$ by fixed-$a(0^+)$ grid and no longer inverts `_solve_a_0plus_max` to place a point.
Both public velocity calls omit `a_0plus`; each solver determines and records its thermodynamic maximum composition at the physical mesh point.

## Boundary and source fail-fast rules

Every requested ray must have exactly one finite root for each boundary, and the $a(0^+)=1$ radius must lie outside the coexistence radius.
Any missing, duplicate, reversed, or non-finite boundary aborts domain construction before the analytical or numerical contour begins.
The calculation window itself enforces $n_B(0^-)/n_0\geq1$; neither solver calls nor the plot include lower density.

⚠ The first production payload used the superseded fixed-temperature/fixed-composition domain and was invalid.
Its stage subprocesses imported a stale installed `RMFsolver`, producing 460 `ceiling_inversion_failed` cells and 140 `no_allowed_band` cells; both 600-point velocity stages then pre-masked every cell and finished immediately.
Schema version 3 rejects those payloads.
Every stage now prepends the repository checkout to its import path, the launcher propagates that path through `PYTHONPATH`, and payload metadata records the resolved `phase_velocity.py` file.
The boundary residual and ray-root implementation is contained in `26-08-20/_isothermal_domain.py`; spawned workers do not import the workspace-level `isothermal_domain_rays` prototype.

## Execution and checkpointing

The displayed calculation bars are the stable-neutron-matter boundary rays, the $a(0^+)=1$ boundary rays, the 600-point polar mesh, the 600-point analytical scan, and the 600-point numerical scan.
Each boundary worker returns both residual roots for one ray; the parent checkpoints that ray before advancing both boundary bars.
For every mesh or velocity cell the parent updates the in-memory payload, atomically replaces the `.npy` file, and only then advances `tqdm` by one.
Worker processes never write the shared payload.

In the 26-08-20 matched analytical--numerical workflow, analytical cells remain independent spawned processes with a default 300-second hard limit.
The numerical stage advances through the 20 radial-fraction shells from the inner boundary outward and parallelizes the 30 angular rays inside each shell.
Its deterministic $j_B$ seeds use the preceding one or two radial shells on the same ray, the same-cell analytical current, bounded multiplicative variants, nearby successful rays in the current shell, and a density-scaled fallback.
Each BVP attempt has a 180-second default hard limit and each physical cell a 900-second total budget.

The domain fingerprint covers the polar axes, both boundaries, physical coordinate grids, physics inputs, and live API signatures.
Resume continues partial boundary or mesh construction and skips terminal analytical and numerical cells, while rejecting changed coordinates, source signatures, physics, or solver controls.
Only finite `task_status="success"` velocities enter a contour; no failure becomes zero or infinity.

The plotted ordinary velocity is

$$
v=c\frac{u(0^-)}{\sqrt{1+u^2(0^-)}}.
$$

See [[isothermal-analytic-front-speed]] for the physics and [[phase-velocity-overview]] for the public APIs.
