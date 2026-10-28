# Isothermal contours, high temperature (polar mesh, $T(0^-)\ge 8$ MeV)

The numerical speeds come from `solve_front_isothermal_slow(initial_guess_mode="auto")` and the analytical speeds from `analytic_velocity_isothermal_slow(xi=-0.85)`, both at `RMFsolver` `[v4.9.6]`. The microphysics is massless quarks, QMC-RMF3 pure neutron matter, $B^{1/4}=189.1565957288247$ MeV, $\alpha_s=0.4$ and the revised Madsen rate.

## Mesh

- **Polar coordinates:** $X=(n_B/n_0-1)/4.5=\rho\cos\theta$ and $Y=T/120\ \mathrm{MeV}=\rho\sin\theta$, as in the 26-08-20 isothermal domain.
- **Rays:** 81, uniform in $\theta$ from the angle whose udd-boundary point lies at 8 MeV up to $90^\circ$ (the $n_B=n_0$ edge).
- **Boundaries:** each ray solves two static boundaries afresh. The inner one is PNM–uds coexistence ($\Delta\mu_B=0$); the outer one is PNM–udd balance ($a(0^+)=1-10^{-9}$).
- **Points:** 29 radial fractions strictly between the two boundaries, from 0.005 to 0.99999. The 8 outer columns follow $v\propto(1-A)^{-1/2}$. Points below 8 MeV are `outside_panel`. Nothing is evaluated inside a gray region.

## Commands (from the repository root)

```sh
export PYTHONPATH=local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:.
./bin/python3 -B new_paper_calculations/26-10-27/isothermal/highT/run_highT.py --method both --workers 4
./bin/python3 -B new_paper_calculations/26-10-27/isothermal/highT/run_highT.py --method numerical --repair
./bin/python3 -B new_paper_calculations/26-10-27/isothermal/highT/build_boundary_curves.py 4
./bin/python3 -B new_paper_calculations/26-10-27/isothermal/build_isothermal_notebook.py
```

- **Resume:** rerunning skips finished points when the payload identity still matches: the hashes of every `RMFsolver/*.py` and of the result modules, the NumPy/SciPy versions, the physics and the settings. Otherwise `--restart` writes a suffixed payload.
- **Shipped payloads:** they predate two later wrapper fixes (rejecting non-finite validation metrics, recording all `RMFsolver/` hashes and library versions), so a rerun needs `--restart`. Adding `--seed-from` then reuses their points, because the solver bytes, physics and settings match. No shipped accepted point has a non-finite stored metric.
- **Repair:** `--repair` re-evaluates only unresolved or timed-out points, in place, and logs the pass. It refuses a payload made with another solver, library or physics.
- **Seeding:** `--seed-from 'highT/isothermal-highT-{method}-41rays.npy'` copies finished points with identical $(n_B,T)$ from a compatible earlier payload.
- **Worker crash:** a worker that dies (segfault, OOM kill) leaves its point `unresolved` with the exit code, and the run continues.

## Outputs

| File | Content |
|---|---|
| `isothermal-highT-{analytic,numerical}.npy` | Payload: identity, mesh, per-point records, speed and status grids |
| `isothermal-highT-numerical-profiles/i_j.npz` | 256-node numerical profile per accepted point |
| `isothermal-highT-*-41rays.npy` | First 41-ray production; it seeds the 81-ray run |
| `ray-boundaries.npy`, `boundary-curves.npz` | Cached static boundaries |

**Saved intermediates for $\xi$ tuning:**
- **Analytical records** keep $I_2$, $\lambda_n$, $a(0^+)$, $D_K$, $\gamma_K$ and the endpoints. `_iso_xi.retune(cache, xi)` rescales the grid exactly, with no EOS calls; `effective_xi` and `fit_xi` compare against the numerical grid.
- **Numerical records** keep the manuscript integrals $I_1$ and $I_2$ and the $\xi$ they imply (`xi_profile`, `xi_profile_inf`).

## Results

Numerical 2282/2291 accepted (9 unresolved, all at $f\ge0.9999$); analytical 2291/2291. Verification in `verify_highT.json`. Details: wiki page `methods/isothermal-contours-polar-and-row-meshes`.
