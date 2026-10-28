# Isothermal contours, low temperature (row mesh, $0.01\le T(0^-)\le10$ MeV)

Same solvers, physics and saved intermediates as `../highT/README.md`.

## Mesh

- **Rows:** 40 temperatures `np.geomspace(0.01, 10, 40)`, the rows of the cached boundary curves.
- **Points:** 35 fractions from $10^{-5}$ to 0.9999 between the PNM–uds inner boundary and the PNM–udd outer boundary of each row, $x=x_{\rm in}+f\,(x_{\rm out}-x_{\rm in})$. Nothing is evaluated inside a gray region.
- **Contours:** computed in $(n_B/n_0,\log_{10}T)$, with display-only PCHIP refinement along rows in ${\rm logit}\,f$. Points are verified against levels from 50 to $10^6$ m/s; the figure draws the half-decades 100, 300, …, $10^6$ m/s, labelled on the contours (`CONTOUR_LEVELS` and `LABEL_TEMPERATURES` in the notebook's tuning block).

## Commands (from the repository root)

```sh
export PYTHONPATH=local_analysis_results/26-10-02/runtime/scipy-1.18.1-cpython-312-darwin-arm64:.
./bin/python3 -B new_paper_calculations/26-10-27/isothermal/lowT/run_lowT.py --method both --workers 8
./bin/python3 -B new_paper_calculations/26-10-27/isothermal/lowT/run_lowT.py --method numerical --repair --workers 8
./bin/python3 -B new_paper_calculations/26-10-27/isothermal/lowT/verify_lowT.py
./bin/python3 -B new_paper_calculations/26-10-27/isothermal/build_isothermal_notebook.py
```

`--smoke` runs the rows at 0.01, 0.1 and 1 MeV into `smoke/`. Resume and `--restart` behave as in `../highT/`.

## Results (2026-10-28)

| | Accepted | Not accepted | Wall time, 8 workers |
|---|---|---|---|
| Analytical | 1400/1400 | — | 247 s |
| Numerical | 1310/1400 | 54 unresolved, 36 timeouts | 1505 s production + 1224 s repair |

- **Failures:** 88 of 90 are at $f\ge0.99$, where the analytical speed is at least $2.3\times10^5$ m/s. The other 2 are at $f=0.9$, $T=0.024$ and 0.029 MeV: the shooting fallback fails its monotone-path check. They may break the $5\times10^5$ and $10^6$ m/s numerical contours on a few rows.
- **Verification** (`verify_lowT.json`): drawn contours match direct roots within $3.8\times10^{-4}\,n_0$; tighter tolerances change speeds by at most $2.4\times10^{-6}$; the 10 MeV seam with the polar mesh agrees within $1.9\times10^{-3}\,n_0$.
- **Figure:** `local_analysis_results/26-10-27/isothermal-lowT-logT.{png,pdf}`.

Details and the analytic-vs-numerical comparison: wiki page `methods/isothermal-contours-polar-and-row-meshes`.
