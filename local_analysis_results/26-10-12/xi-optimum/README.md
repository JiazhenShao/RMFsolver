# Best constant xi for the adiabatic T(0+)=0.1 MeV analytical comparison

Date: 2026-10-12. Wiki: `wiki/physics/adiabatic-xi-diagnostics.md`, section "Best single constant xi".

## Result

Over the displayed 50-2000 m/s range of the `26-10-02.ipynb` adiabatic figure, the single xi that
makes the full-analytical and semi-analytical speeds agree best with the full numerical speeds is

    xi = -0.865   (1% basin: -0.89 to -0.845)

The current xi = -0.85 is within 0.5% of the minimum error, so it is already effectively optimal.
The optimum depends mainly on the speed range included (table below), not on the error measure.

| Speed range [m/s] | Full analytical best | Semi-analytical best | Joint best (1% basin) |
|---|---|---|---|
| 50-1000 | -0.82 | -0.755 | -0.765 (-0.785, -0.745) |
| 50-2000 (figure) | -0.99 (scan edge) | -0.83 | -0.865 (-0.89, -0.845) |
| 50-5000 | -0.99 (scan edge) | -0.91 | -0.945 (-0.96, -0.925) |

Joint error for 50-2000 m/s: 0.1443 at the optimum, 0.1449 at xi=-0.85, 0.276 at xi=-0.5.

Optimum by other measures, 50-2000 m/s unless stated (`report-output.txt`, `contour-bracketed-output.txt`):

| Measure | Full | Semi | Joint |
|---|---|---|---|
| Pointwise RMS of ln(u_an/u_num) | -0.99 | -0.825 | -0.86 |
| Pointwise mean abs ln(u_an/u_num) | -0.99 | -0.805 | -0.845 |
| Pointwise RMS, all 38-7800 m/s | -0.99 | -0.925 | -0.95 |
| Contour displacement, bracketed rays, 50-1000 m/s | -0.99 | -0.815 | -0.855 |

Per-speed-band optimum (`band-output.txt`) shows the trade-off seen in the figures:

| Band [m/s] | n | mean numerical a(0+) | Full best | Semi best |
|---|---|---|---|---|
| 50-100 | 28 | 0.25 | >= -0.30 (scan edge) | -0.485 |
| 100-200 | 59 | 0.36 | >= -0.30 (scan edge) | -0.56 |
| 200-500 | 132 | 0.55 | -0.67 | -0.695 |
| 500-1000 | 108 | 0.72 | -0.97 | -0.80 |
| 1000-2000 | 85 | 0.83 | -0.99 | -0.875 |
| 2000-5000 | 141 | 0.93 | -0.99 | -0.94 |

More negative xi hurts near the stable-neutron-matter boundary (low a(0+)) and helps toward a(0+)=1.
The full-analytical high-speed deficit is not a xi error: at xi=-0.99 its RMS log error in 1000-2000 m/s
is still 0.24. It comes from the fixed-muB composition and closed-form I2 approximations, so pushing xi
toward -1 compensates for them rather than measuring a physical shape factor.

## Workflow

1. **Benchmark states** (`common.benchmark_points`). Union of successful numerical records, keyed by
   (theta_index, radial fraction):
   - `new_paper_calculations/26-08-21/T0plus0p1-uNmax-angular-wavefront.npy`, `radial_shell_records`: 514
   - `new_paper_calculations/26-08-24/rho0p86_full_theta_shell/T0plus0p1-rho0p86-full-theta-shell.npy`: 30
   - `new_paper_calculations/26-08-24/rho5k_adaptive_bracket/T0plus0p1-rho5k-adaptive-bracket.npy`: 74
   - Geometry and calibration: `new_paper_calculations/26-08-21/T0plus0p1-domain.npy`.
   Total 618 states, numerical speeds 38-7800 m/s.

2. **Analytical evaluation at xi=-0.85** (`run_batch.py`, `common.evaluate`). For each state, direct
   fixed-density upstream recovery (`phase_velocity._validated_pnm_state_from_nB`) followed by the unchanged
   public `analytic_velocity_bound` and `semi_analytic_velocity_bound`, with T(0+)=0.1 MeV, PNM, ms=0,
   QMC-RMF3 and the 3 n0 bag calibration, as in
   `new_paper_calculations/26-10-02/adiabatic_xi_minus0p85/_adiabatic_physics.py`. Each result stores
   u(0-), a(0+), A_boundary and lambda_n. Output: `analytical_xi-0.8500.jsonl` (8 workers, ~22 min).
   617/618 succeed; the failure is the known cold upstream-recovery point (theta 0, fraction 0.72,
   nB = 4.6546 n0) diagnosed in the 26-10-02 local repair, and it is excluded.

3. **Exact rescaling to any xi** (`analyze.rescale`). xi enters only through
   u^2 = 2*I2 / (lambda_n^2 * (1 - a0p) * (1 + xi*a0p)); a0p = a(0+), I2 and lambda_n are xi-independent
   (the endpoint depends on u(0-) only at relativistic order), so

       u(xi) = u(-0.85) * sqrt((1 - 0.85*a0p) / (1 + xi*a0p))

   with each method's own a0p.

4. **Scan and error measures** (`report.py`, `band_report.py`, `contour_bracketed.py`). xi in [-0.99, -0.30],
   step 0.005. Error e = ln(u_analytic / u_numerical) at the same upstream state.
   - Band-balanced RMS (primary): RMS within each contour interval, averaged with equal weight per interval
     and per method, matching the log-spaced levels 50, 100, 200, 500, 1000, 2000 m/s.
   - Pointwise RMS and mean |e| within 50-2000 m/s (412 states), and over all states.
   - Contour displacement: along each of the 30 rays, the radial fraction where log u crosses each level,
     linearly interpolated, times the ray's (rho_outer - rho_inner). This is the in-figure distance
     (axes scaled 4.5 n0 and 120 MeV). `report.py` extrapolates unbracketed analytical crossings and counts
     them; `contour_bracketed.py` keeps only rays where all three methods are bracketed. The 2000 m/s
     full-analytical contour is bracketed on only a few rays, so the 50-2000 m/s contour result (-0.79)
     is a small-sample selection effect and is not used.

## Verification

- Recomputed xi=-0.85 speeds reproduce the cluster checkpoint
  `new_paper_calculations/26-10-02/adiabatic_xi_minus0p85/T0plus0p1-analytical-xi-minus0p85.npy`
  to 1.4e-8 relative on 543 (full) and 423 (semi) shared states (`checkpoint-check-output.txt`).
- Rescaled speeds match live solves at xi=-0.5 and xi=-0.95 on six states spanning the domain to
  4e-10 relative (`analytical_xi-0.5000.jsonl`, `analytical_xi-0.9500.jsonl`; first lines of `report-output.txt`).
- Semi-analytical a(0+) equals the numerical a(0+) to ~1e-9 (spot check at theta 15, fraction 0.5).

## Files

| File | Role |
|---|---|
| `common.py` | benchmark assembly from the raw numerical files; one-state analytical evaluation |
| `run_batch.py` | resumable parallel evaluation: `run_batch.py XI WORKERS [comma-separated keys]` |
| `analyze.py` | loading, xi rescaling, crossing and metric functions |
| `report.py` | full scan, optimum per measure, summary tables; writes `scan.npy` |
| `band_report.py` | per-speed-band optima and band-balanced joint RMS |
| `contour_bracketed.py` | contour displacement restricted to bracketed crossings |
| `checkpoint_check.py` | reproduction check against the cluster checkpoint |
| `analytical_xi-0.8500.jsonl` | 618 per-state analytical results at xi=-0.85 |
| `analytical_xi-0.5000.jsonl`, `analytical_xi-0.9500.jsonl` | live-solve rescaling checks (6 states each) |
| `*-output.txt`, `scan.npy` | saved outputs |

## Reproduce

SciPy 1.15.2 in the project venv fails to import on macOS 27; use the isolated SciPy 1.18.1 runtime for
evaluations. The analysis scripts need only NumPy.

    cd local_analysis_results/26-10-12/xi-optimum
    PYTHONPATH=../../26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64 ../../../bin/python3 run_batch.py -0.85 8
    ../../../bin/python3 report.py > report-output.txt
    ../../../bin/python3 band_report.py > band-output.txt
    ../../../bin/python3 contour_bracketed.py > contour-bracketed-output.txt
    ../../../bin/python3 checkpoint_check.py > checkpoint-check-output.txt

`run_batch.py` appends to the JSONL and skips states already present; delete the JSONL to recompute.
No `RMFsolver/` source, cluster checkpoint, or notebook was modified.
