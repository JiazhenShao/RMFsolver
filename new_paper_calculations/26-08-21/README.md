# Fixed T(0+)=0.1 MeV Angular-Wavefront Numerical Contour

This folder reruns the full numerical uNmax contour at fixed
`T(0+)=0.1 MeV`. The shared semi-analytical construction is used only to
define the physical domain. It is not used as a numerical `jB` seed.

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

Activate the project environment, then prepare the domain once:

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

- `T0plus0p1-domain.npy`
- `T0plus0p1-uNmax-angular-wavefront.npy`

The numerical payload records the chosen `jB`, every tried candidate and its
source, hard-timeout termination, BVP diagnostics, bridge attempts, completed
radial shells, and the full successful `nK`/`jK` profile for each solved cell.
Angular attempts record the wave index, seed theta indices, seed current,
one- or two-sided seed mode, and the reserved radial/angular budgets.
It stores the matched 1 MeV bootstrap results separately under
`matched_reference_records` and `matched_reference_attempt_records`.
