---
summary: On 2026-10-02 all 36 season-named notebooks became date-named from their creation time; `26SU-14` is now `26-08-20`, and the six files sharing a creation date carry `-1`/`-2`/`-3`/`-4` suffixes.
status: current
updated: 2026-10-02
tags: [meta, notebooks, naming]
---

# Notebook rename map

Every `YYSeason-N.ipynb` root notebook was renamed to `YY-MM-DD.ipynb` on 2026-10-02, using each file's filesystem creation time.
Semantically named notebooks were left alone: `All_Plots.ipynb`, `All_Plots_new.ipynb`, `All_Plots_new-backup.ipynb`, `qcd_phase_diagram_paraQMCRMF3.ipynb`, `save_profiles.ipynb`, and `26-10-02.ipynb`.
`FL26-1`/`FL26-2`/`FL26-3` were the same family with the season written first, and were renamed along with the rest.

## Why suffixes appear

Six files have two creation dates between them, so a bare `YY-MM-DD` name could not hold them.
`26SP-3`, `26SP-4`, `26SU-2` and `26SU-5` all carry creation time `2026-07-17 11:50`, and `26SU-11` and `26SU-13` both carry `2026-08-18`.
⚠ The `2026-07-17` stamp is a bulk copy/sync event, not authorship, so those four names are administrative rather than physical: the `-1`..`-4` order follows the old index order and carries no chronological meaning.

## Map

| Now | Was | | Now | Was |
|---|---|---|---|---|
| `25-05-15` | `25SU-1` | | `26-07-17-1` | `26SP-3` |
| `25-06-04` | `25SU-2` | | `26-07-17-2` | `26SP-4` |
| `25-06-11` | `25SU-3` | | `26-07-17-3` | `26SU-2` |
| `25-06-19` | `25SU-3.5` | | `26-07-17-4` | `26SU-5` |
| `25-07-14` | `25SU-4` | | `26-07-19` | `26SU-8` |
| `25-08-07` | `25SU-5` | | `26-07-23` | `26SU-9` |
| `25-08-27` | `25FL-1` | | `26-08-04` | `26SU-10` |
| `25-09-16` | `25FL-2` | | `26-08-13` | `26SU-12` |
| `25-10-22` | `25FL-3` | | `26-08-18-1` | `26SU-11` |
| `25-11-23` | `25FL-4` | | `26-08-18-2` | `26SU-13` |
| `25-11-30` | `25FL-5` | | `26-08-20` | `26SU-14` |
| `25-12-11` | `25FL-6` | | `26-08-23` | `FL26-1` |
| `26-01-01` | `26SP-1` | | `26-08-31` | `FL26-2` |
| `26-01-22` | `26SP-2` | | `26-09-03` | `FL26-3` |
| `26-04-09` | `26SP-5` | | | |
| `26-04-19` | `26SP-6` | | | |
| `26-04-30` | `26SP-7` | | | |
| `26-05-01` | `26SU-1` | | | |
| `26-05-14` | `26SU-3` | | | |
| `26-05-31` | `26SU-4` | | | |
| `26-06-07` | `26SU-6` | | | |
| `26-06-26` | `26SU-7` | | | |

A machine-readable copy of the same map is at `tmp/notebook-rename-2026-10-02.csv`.

## ⚠ Old names still appear, and that is deliberate

Data artifacts keep the season names, because renaming a reference without renaming the artifact on disk would break the notebooks that write and load them.

- Directory names under `local_analysis_results/`, e.g. `local_analysis_results/26SU-11/`, `local_analysis_results/FL26-3/`.
- Output filenames, e.g. `26SU-2-hydro-analytic-velocity-contour-linear_v1.npy`, and the `DEFAULT_OUTPUT_TEMPLATE` strings in `new_paper_calculations/` that generate them.
- The label `FL26-3_coupled_interface_roots`, which is written into saved boundary records and compared against in `tests/test_T0plus0p1_ncrit2_26_09_03.py`. Do not rewrite it.
- [[log]] entries and `docs/superpowers/` plans and specs, which are historical records of what was done at the time.

So an old-style season name in this wiki, in `docs/`, or in a saved payload is expected. Resolve it through the map above rather than assuming the notebook still exists under that name.

## Known consequence

`tests/test_FL26_2_notebook.py`, `test_FL26_3_notebook.py`, `test_26su8_contour_plotting.py`, `test_26SU_9_notebook.py` and `test_26su9_elliptical_boundaries.py` were updated to the new paths.
Their remaining failures are unrelated to the rename: five come from a broken `scipy` binary in the venv (`_spropack` fails to `dlopen` because a ROOT install shadows the library), and eight are content assertions that had already drifted from the notebooks.
