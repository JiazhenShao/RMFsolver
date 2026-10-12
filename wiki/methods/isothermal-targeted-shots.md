---
summary: Twenty designated fixed-temperature targets replace broad contour repair; persistent budgets cap all BVP work at 120 calls and midpoint checks govern connectivity.
status: current
updated: 2026-10-02
tags: [method, cluster, contour, isothermal, continuation]
---

# Designated isothermal contour shots

The new workflow is `new_paper_calculations/26-09-21/isothermal_targeted/`, run on the cluster with `python3 run_isothermal_numerical.py`.
It reads the 26-09-16 repair and profiles without changing them or `RMFsolver/phase_velocity.py`.
See [[isothermal-contour-repair]] for the predecessor and [[isothermal-analytic-front-speed]] for the transport limit.

## Evidence motivating the smaller workflow

The downloaded 738-evaluation snapshot contains 396 accepted solutions/anchors, 341 timeouts and one exception; 119 contour targets succeeded.
Warm 500 m/s roots already reach 94.10 MeV and warm 1000 m/s roots occupy the visible gap.
The old spacing gates nevertheless produce no new connected segments for either contour.
The 500 m/s roots jump from 17.18 to 31.03 MeV, and the 1000 m/s roots from 19.60 to 34.55 MeV, so the bends near 20 MeV remain under-resolved.
Local read-only comparisons establish positional agreement, not that the displayed polygonal kinks are physical.

The local checkpoint download has none of its 396 referenced profile sidecars.
Plotting needs only scalar data; new continuation solves must reuse the profile cache already on the cluster.
Missing profiles do not trigger mass anchor rebuilding.

## Targets and finite budget

Eight warm midpoint targets connect/extend 500 and 1000 m/s, seven midpoint targets check the bends across levels 50--2000 m/s, and five cold targets extend 1000/2000 m/s to 1 MeV.
The exact list is printed by `--dry-run` and documented in the folder README.
At fixed temperature, each target corrects only upstream density using actual BVP speeds and the nearest cached profile.
The inherited exact reaction rate, local diffusion, normalized BVP, maximum interface composition and direct density recovery remain unchanged.
No new phase boundaries, global scan, 5000 m/s targets, or sub-1-MeV shoots are scheduled.

Persistent call quotas are warm 40, bends 44, cold 32 and tighter checks 4, totaling 120 across restarts.
Each target gets at most four calls and 600 seconds cumulative trial time; each trial is hard-limited to 180 seconds.
Calls are reserved atomically before spawning, including confirmations and interrupted work.
Failed targets do not recursively spawn auxiliary continuation solves or automatically retry on every restart.
Local curvature checks may add targets, with three subdivision levels and the same quotas.

Warm work completes or exhausts its allowance before bends, and bends before cold continuation.
The cold group also checks missing connections among saved 1--10 MeV roots within its allowance.
The formal zero-temperature divergence assumes a fixed positive composition; fixed-speed contours change their density/composition, so it is not a prescription to force these contours through exact zero.

## Connectivity and accuracy

The previous fixed density/temperature spacing rule is replaced for the new intervals by a numerical midpoint check: speed error at most 0.5% and density displacement from the endpoint chord at most 0.005 saturation density.
Only certified leaf intervals add lines, while every accepted root remains visible as a marker.
An attempted but unresolved interval masks matching inherited geometry; certified children restore only their own portions.
No inferred velocities or smoothing splines substitute for solves.
This is a sampled local geometry criterion, not a rigorous error bound over an entire curve.

One warm 500, warm 1000, bend 2000 and cold 1000 root receive independent tighter mesh/tail checks, requiring speed agreement within 1%.
Their outcomes remain separate from root matching and interval certification.
An unresolved/failed representative is explicitly reported, not silently counted as an accuracy pass.

## Preservation, progress and visualization

The first calculation freezes the three baseline files and repair checkpoint into `inputs/` with hashes.
Later source-run updates cannot silently alter a resumed target manifest.
The source and new profile directories remain separate; completed root/check events survive interruption without losing child scheduling or repeating saved checks.
A single-writer lock prevents simultaneous runs on one output.

Each trial is checkpointed before evaluation progress; each target is saved before target progress and figure regeneration.
The fixed initial-target bar counts 20, with local refinement counts separate; a second bar displays calls consumed out of the hard 120 limit.
Its completion fraction is a budget, not a promised number of numerical successes.

`26-08-20.ipynb` numerical cells 4--5 load the new optional overlay, validate the frozen input hashes, retain established styling, and display isolated accepted roots.
Local verification uses synthetic tests and saved-data rendering only.
Cluster convergence and complete curve recovery have not yet been established for this workflow.
