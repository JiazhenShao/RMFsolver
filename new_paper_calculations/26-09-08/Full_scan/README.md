# nB_crit(0)=4n0, Fixed T(0+)=0.1 MeV Numerical Contour

This folder reruns the full numerical uNmax contour at fixed
`T(0+)=0.1 MeV`, with the bag constant calibrated by
`nB_crit(T=0)=4n0`, giving `B^(1/4)=210.057427337 MeV`.
The physical domain is constructed with the direct coupled
interface roots used in `FL26-3`; no velocity formula or numerical-I2 scan is
called while locating either boundary. It is not used as a numerical `jB` seed.

By default preparation reads the bundled `T0plus0p1-ncrit4-boundaries.npy`,
copied unchanged from `local_analysis_results/FL26-3/boundaries-ncrit4-T0plus0.1-cold0.001.npy`.
It validates calibration, temperatures, residuals, and geometry, without resolving
either boundary. The saved curves contain 18 endpoint-inclusive angles;
their radial coordinates are interpolated onto 30 contour rays (not 30 new roots).
The window is `nB(0-)/n0=1..7`, `T(0-)=0..170 MeV`, matching FL26-3
and enclosing the 4n0 outer boundary. Theta=0 evaluates exactly zero upstream temperature.

To explicitly recalculate 30 boundary rays, use `--recompute-boundaries` with
the prepare command. This includes FL26-3's corrected cold-boundary solve:
muB(0-) is an auxiliary inversion variable constrained to the requested density,
avoiding the spurious fixed-density RMF branch. The upstream state is not free.
At each angle, the inner boundary solves pressure and enthalpy-per-baryon
matching with `muK(0+)=0` at `T(0+)=0.1 MeV`. The outer boundary solves the
same matching equations together with `n_s(0+)=0` at the cold numerical
temperature `T(0+)=1e-3 MeV`, which is the exact local-fraction `A=1`
condition for the massless-strange-quark model used here.

The contour has 30 endpoint-inclusive theta rays and 20 outward radial
targets, for 600 requested cells. Every radial shell is solved in parallel
across theta. Before the target scan, the runner solves the same 30 first-shell
coordinates at `T(0+)=1 MeV`. The target first-shell candidates begin with
`1.18*jB(T(0+)=1 MeV)`, followed by the unscaled matched current and the fixed
numerical fallback ladder. Successful target profiles then seed the immediately
larger radial shell. Each radial primary stage tries only the extrapolated
same-theta current, with a 300-second cap. Failed cells are then recovered in
angular wavefronts: only successful immediate theta neighbors on the same
radial shell may provide a `jB` seed. A two-sided seed linearly interpolates the
adjacent currents. New successes become anchors for the next angular wave.
Small angular perturbations, the old radial multiplier ladder, and midpoint
bridges run only after direct angular propagation stalls.

Every individual BVP trial runs in a disposable child process with a hard
300-second wall-clock limit. All trials and bridges for one requested cell
share a 1200-second budget. A timed-out child is terminated before the next
`jB` candidate is tried.

## Run

Upload this entire folder, including both bundled 4n0 input files, and activate
the project environment. A 600-point domain is already prepared, so you can
run the numerical command directly. To rebuild the domain from the saved roots:

```bash
python3 prepare_T0plus0p1_domain.py
```

Run the numerical contour:

```bash
python3 run_T0plus0p1_numerical.py
```

The numerical payload is atomically updated after every completed reference or
target cell, including each successful target profile. Rerunning the same
command resumes an incomplete payload and does not repeat terminal cells. If
the payload is already complete, the runner exits without changing it. Use
`--restart` to start a new run under an automatically suffixed output name.

The integrated tqdm bar also advances and visibly refreshes once per terminal
physical cell. Successful primary solves are reported immediately; only cells
that still require same-shell retries remain pending while neighbor results are
collected.

## Outputs

- `T0plus0p1-ncrit4-boundaries.npy` (saved FL26-3 input)
- `T0plus0p1-ncrit4-domain.npy` (600-point input manifest)
- `T0plus0p1-ncrit4-uNmax-angular-wavefront.npy` (new numerical output)

The numerical payload records the chosen `jB`, every tried candidate and its
source, hard-timeout termination, BVP diagnostics, bridge attempts, completed
radial shells, and the full successful `nK`/`jK` profile for each solved cell.
Angular attempts record the wave index, seed theta indices, seed current,
one- or two-sided seed mode, and the reserved radial/angular budgets.
It stores the matched 1 MeV bootstrap results separately under
`matched_reference_records` and `matched_reference_attempt_records`.
