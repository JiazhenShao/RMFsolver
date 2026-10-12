# Downloaded cold-contour run: feasibility audit

Date: 2026-10-01. Recommendation: do not repeat this production workflow unchanged. The download demonstrates poor targeting/continuation economics, not absence of finite-temperature numerical fronts. Analytical contours are useful as an approximation; the xi -> -1+ limit is an optimistic envelope of the frozen-coefficient formula, not a proven bound on the full numerical problem.

No EOS roots, BVP solves, or live analytical eigenvalue solves were performed for this audit. Production scripts, checkpoints and notebooks were not modified. `audit.py` reads saved records and evaluates only the pure microphysics coefficient helper at saved numerical endpoints.

## Checkpoint evidence

Source: `new_paper_calculations/26-10-01/isothermal_numeric_logT/isothermal-logT.npy`.
SHA256: `f6122f975a5828c61e816d44a9c3ff9febbb1727cdbe34f684cb0deb44a017f6`.
All nine saved RMFsolver source hashes and all six workflow source hashes match the local files used for inspection.

| Work | Result |
|---|---|
| Boundary temperatures | 61/61 successful; both boundaries at each temperature |
| Original contour targets | 3/217 accepted |
| Added midpoint targets | 10/210 accepted |
| Numerical evaluations | 27 converged, 358 killed by the 180-second limit |
| Analytical density proposals | 225/225 killed by the 60-second limit |
| Tight endpoint accuracy checks | 14 blocked because their base targets were missing |

The 13 accepted targets are fewer than the 27 successful BVP evaluations because off-target numerical speeds are retained but are not accepted contour roots.
Successful new evaluations span 0.84256–4.96411 MeV; none establishes the 0.01-MeV floor or 20-MeV join.
The flags are correctly `work_complete=True`, `contours_complete=False`.
Summed worker time is 18.85 hours for numerical trials, 3.75 hours for analytical proposals, and 1.09 hours for boundaries; this is **not wall-clock elapsed time**.

The download lacks all 27 newly referenced `profiles/evaluation-*.npy` files. Endpoint diagnostics are available, but independent new-profile reconstruction and profile-area measurements are not possible from this download alone. Bundled input profiles remain available; this local omission does not explain the cluster failures.

## Root-cause evidence

1. **Starting-density search fails before the BVP.** `_logT_trial.proposal` invokes the entire hydro-coupled analytical solver at up to 18 logarithmically spaced densities, followed by a bracketed root. `_logT_engine.prediction` gives the entire operation only 60 seconds. All 225 such calls timed out, aborting their targets without a numerical trial. This does not show that the closed-form velocity expression itself is invalid.
2. **The advertised cold initializer is bypassed.** `_logT_engine.seed` always chooses a profile if any exists, without a proximity threshold. `shoot` supplies it to `solve_front_isothermal`; that solver gives continuation precedence over `initial_guess_mode='layer_resolved'`. Every successful record reports `initial_guess_mode_used='continuation'`. Source inspection implies the same route for trials with valid supplied profiles, but timeouts contain no internal stage trace. Transfer rescales endpoint amplitudes and retains shape in compact s; it does not adjust the physical core-to-tail width. There is no continued-seed versus fresh-layer-seed comparison after failure.
3. **Recovery can stop after two failed densities.** `propose_density([], [], 3.01, 500, (3, 5.3))` returns 3.01. With that failure it returns 3.005. With both failures it returns None: it recomputes the same prediction and halves only once. This pure-function reproduction launches no solver. Of the 194 targets with numerical attempts, 176 have exactly two attempts; the checkpoint spends only 385 of its 900 available numerical attempts.
4. **Even successful solves are expensive.** The 27 successes take about 72–177 seconds inside the child, with only 1–2 BVP iterations but 17,799–45,845 local thermodynamic-state calls and 931–2,202 nodes. The wrapper limit is close to successful runtimes. SciPy uses forward finite differences when no Jacobian is supplied, adding repeated right-hand-side evaluations ([official solve_bvp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html)). These observations identify repeated EOS work as an important cost, but the timeout records do not isolate the internal failure stage or establish hardware contention.

All unsuccessful numerical calls are timeouts, not returned singular-Jacobian or residual-rejection reports. We cannot attribute every timeout to a bad initial guess. Failure-driven refinement expanded nearly the same difficulty into another 210 targets without first changing the seed strategy.

## Analytical fallback: saved-endpoint comparison

`frozen-endpoint-comparison.csv` compares the frozen analytical expression with the numerical proper velocity, holding each saved numerical endpoint fixed. It is **not** a fresh solution of the implicit finite-flux analytical eigenvalue problem. New successes have momentum-flux ratios below 1.2e-9, but we do not use that observation to claim an independently validated analytical solution.

| Saved T=0.01 MeV probe density / n0 | Numerical speed (m/s) | xi=-0.5 speed excess | xi -> -1+ speed excess |
|---|---:|---:|---:|
| 3.001 | 74.5351 | 0.176% | 0.763% |
| 3.01 | 771.9789 | 0.455% | 2.400% |
| 3.05 | 4139.9648 | 0.595% | 5.343% |

For the 27 new successful evaluations (interface fractions 0.337–0.931), the analytical/numerical speed ratio is 0.632–0.994 at xi=-0.5 and 1.114–1.763 in the xi -> -1+ limit. The latter has a median excess of 32.7%. These are sparse, convergence-selected data, not an error guarantee throughout the contour domain. At the three very cold, near-coexistence probes, xi=-0.5 is actually closer.

Speed-fitted xi values in the CSV merely absorb the other frozen-coefficient/rate approximations; they are not measurements of the positive profile integral I1.

For fixed endpoints and positive transport coefficients, speed decreases with xi. The xi -> -1+ limit is therefore the maximum of the closure family. However, the nontrivial profile has I1 = integral D_K*(da/dx)^2 dx > 0, while the shape-factor definition assigns I1=0 at exactly xi=-1. The limit is an envelope, not an ordinary attained profile. The full numerical model has local diffusion and the exact reaction rate, so no global upper-bound theorem for it follows.

The public analytical API currently requires -1 < xi < 1 and rejects exactly -1. An explicit limiting-formula path or a documented one-sided numerical limit would be required for production plotting. No such change was made here. Exactly T=0 and a(0+)=1 remain excluded; the formal divergence does not establish physical transport validity there.

### Wolfram verification

Input (positive DK, gammaK, lambdaN; eta >= 0; 0 < a0plus < 1; -1 < xi < 1):

```wolfram
u2 = DK gammaK a0plus^2 (a0plus^2 + 2 eta)/
     (2 lambdaN^2 (1-a0plus) (1+xi a0plus));
FullSimplify[D[Log[Sqrt[u2]], xi], Assumptions -> ass]
FullSimplify[Limit[u2, xi -> -1, Direction -> "FromAbove"], Assumptions -> ass]
```

Returned derivative `-a0plus/(2*(1+xi*a0plus))` and squared limit `DK*gammaK*a0plus^2*(a0plus^2+2*eta)/(2*lambdaN^2*(1-a0plus)^2)`. An explicit equality check of the limit with that expression returned True. This verifies algebra at fixed endpoints, not the full implicit hydrodynamic branch.

## Decision

For making the requested scientific figure, an analytical contour with its approximation and xi limit stated, plus existing numerical checks, currently has better value than another broad numerical campaign. Keep the validated warm numerical figure and all successful records.

If full numerical cold contours remain scientifically necessary, first use a very small controlled initializer comparison at an already recovered cold state and one adjacent continuation state. Compare transferred and fresh layer-resolved seeds at identical density/current bounds, with internal timing and stage diagnostics. Only expand after this demonstrates reliable, affordable continuation. This is a possible follow-up, not a calculation performed in this audit.

Do not increase all timeouts or resubmit all failed targets as the first response. Do not fill gaps with analytical speeds while labeling the result numerical.

Reproduction in the existing local verification environment:

```bash
PYTHONPATH=/private/tmp/nsm-scipy-verification MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/nsm-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./bin/python3 local_analysis_results/26-10-01/isothermal-logT-audit/audit.py
```
