# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Start from the wiki, not the repo

`wiki/` is a compiled knowledge base — its own Obsidian vault — holding this project's durable conclusions: physics, solver behavior, numerical methodology, manuscript state. It exists so research context is *retrieved* rather than re-derived from notebooks and specs every session.

- **For any physics, research, or "why is it done this way" question, read `wiki/index.md` first.** Do not sweep the repository, and do not preload the wiki wholesale — `index.md` is ~800 tokens and lists every page with a one-line summary.
- **Retrieval ladder, never skipped:** `wiki/index.md` → targeted `rg` over `wiki/` → the relevant page body → raw source only if the wiki is insufficient. `rg` returns lines; `Read` returns whole files.
- **For code changes,** take established constraints from the relevant wiki page, then read only the live source you actually need. **Live code is authoritative for current behavior; the wiki is authoritative for conclusions, reversals, and history.**
- `docs/superpowers/specs/` is design archaeology and is stale in places. Where it disagrees with the wiki, **the wiki wins**.

### Maintaining it

- **Read `wiki/SCHEMA.md` before creating or revising any wiki page.** It defines frontmatter, page size caps, how supersession is recorded, and the folder layout.
- File durable conclusions back into the wiki as pages. That is the mechanism that stops the same work being repeated — a good answer left only in chat history is lost.
- Update `wiki/index.md` and append one `wiki/log.md` entry **at the end of the session, batched.** `index.md` sits in the cached prompt prefix; editing it mid-session invalidates the cache and re-bills everything before it.
- The auto-memory directory is the **hot tier** — loaded automatically every session, so it must stay small. The wiki is the **cold tier**, loaded on demand. Memory entries should point *into* the wiki, never duplicate it.

## Never read a `.ipynb` directly

Notebooks in this workspace are ~95% base64 PNG by weight — a raw read costs 20–100× the code it contains (a typical notebook is 5k tokens of source wrapped in 200k+ tokens of embedded figures). Never use `Read`, `cat`, or `grep` on a `.ipynb`. Use `.claude/tools/nbsrc.py`:

```bash
./bin/python3 .claude/tools/nbsrc.py 26-08-18-1.ipynb                    # table of contents, ~150 tokens
./bin/python3 .claude/tools/nbsrc.py 26-08-18-1.ipynb -c 3               # source of cell 3
./bin/python3 .claude/tools/nbsrc.py 26-08-18-1.ipynb -c 2-5,9           # a range
./bin/python3 .claude/tools/nbsrc.py 26-08-18-1.ipynb -o 3               # text outputs only, never images
./bin/python3 .claude/tools/nbsrc.py 26-08-18-1.ipynb -f 3               # write figures to .claude/cache/figs/
./bin/python3 .claude/tools/nbsrc.py 26-08-18-1.ipynb -g 'uNmax'         # regex over cell sources
```

Always start with the table of contents, then fetch only the cells you need. To *see* a figure, extract it with `-f` and `Read` the resulting `.png` — that costs ~3k tokens instead of ~65k for the same image as base64. `NotebookEdit` remains the correct tool for writing to notebooks.

## Editing `RMFsolver/` — scope restriction

When making changes inside the `RMFsolver` package, modify **only `RMFsolver/phase_velocity.py`**. Do not edit any other file in the package (`Solver.py`, `SQMsolver.py`, `constants.py`, `RMFparameter.py`, `tov.py`, `tov_solve.py`, …) unless the user has specifically and clearly instructed you to change that file.

Reading from and importing those modules is fine; only writing to them is restricted. If a task appears to require changing another module, stop and ask first rather than editing it.

## Version control

This workspace is a git repository rooted at `/Users/janshao/NSM_related`, pushing to
`https://github.com/JiazhenShao/RMFsolver` (**public**). Only two trees are tracked:
`RMFsolver/` and `wiki/`, plus `README.md`, `requirements.txt`, and `setup.py`.

`.gitignore` is an **allowlist** — `/*` ignores everything at the top level and the tracked
paths are re-included. This is deliberate: the workspace root is also the venv and holds ~30
notebooks, LaTeX sources, and large `.npz` outputs. Do not convert it to a denylist, and never
`git add -f` a path outside the allowlist.

Maintenance rule: at the end of any session that changed `RMFsolver/` or `wiki/`, commit and
push. Match the existing subject style, `[vX.Y] short summary`, for package changes. The
batched `wiki/index.md` and `wiki/log.md` updates belong in that same commit.

Before staging, run `git status` and read it. Other sessions edit this workspace concurrently;
`git add -A` will happily sweep up someone else's half-finished work. Stage the paths you
actually changed when anything unexpected appears.

`RMFsolver_Pub/` is a retired standalone copy, ~10k lines stale, and its remote has been renamed
to `origin-stale` so it cannot clobber the live repo. Do not push from it.

The public repo carries the wiki, which includes unpublished manuscript material. That was a
deliberate choice by the user, not an oversight.

## Environment

This directory is a Python 3.12 virtual environment rooted at the project. Always use the venv interpreter:

```bash
./bin/python3          # or python3 if venv is activated
source bin/activate    # to activate
```

Core dependencies: `numpy`, `scipy`, `matplotlib`, `numdifftools`.

Install the `RMFsolver` package in editable mode from the root:

```bash
pip install -e .
```

## Running Tests

Tests live in `tests/` and use the standard `unittest` framework (pytest is not installed in the venv):

```bash
python3 -m unittest discover tests/        # run all tests
python3 -m unittest tests.test_phase_velocity  # run a single test module
```

## Architecture

This repository is a scientific research workspace for neutron-star matter physics (NSM = neutron-star matter). The core is the `RMFsolver` Python package.

### `RMFsolver/` — main package

| Module | Role |
|--------|------|
| `Solver.py` | Core RMF solvers: hadronic EOS in beta equilibrium, pure neutron matter (PNM), symmetric nuclear matter (SYM). Entry points: `RMFsolve`, `RMFpressurePNM`, `RMFedensPNM`, `RMFsolve_mu`, etc. |
| `SQMsolver.py` | Strange/quark matter thermodynamics: free-fermion pressure/energy, finite-T quark-matter EOS (`PQM`, `edensQM`, `entropyQM`), quark chemical-potential solver. |
| `phase_velocity.py` | Phase-transition front calculations: `solve_front_isothermal`, `solve_front_adiabatic`, `solve_front_energy_conserving`; analytic velocity bounds; BVP-based steady-front profiles. |
| `tov.py` | TOV solver (`class TOV`) — takes `(en_arr, p_arr)` in MeV/fm³ and computes neutron-star mass-radius sequences. Merges in a BPS crust EOS automatically. |
| `tov_solve.py` | Convenience wrappers around `TOV`. |
| `constants.py` | All physical constants and unit conversions (CGS, SI, MeV units). Canonical unit in this codebase is **MeV** and **fm**; `MeV_fm = 197.327 MeV·fm` is the key conversion. |
| `RMFparameter.py` | Named parameter sets: `paraTMA`, `paraSFHo`, `paraGM1`, `paraIUF`, `paraTM1e`, `paraQMCRMF1/2/3/4`. The default throughout most notebooks and scripts is **`paraQMCRMF3`**. |

Parameter list format (used by all solvers):
```
para = [[mn, mp], [ms, mw, mr], [I3n, I3p], [gsn, gsp], [gwn, gwp],
        [grn, grp], [b, c, Mn_scale], [zeta], [xi],
        [b1..a6], nsat, bar, bag_const]
```

### Notebooks

Named by semester: `{YY}{SEM}-{N}.ipynb` where `SU`=Summer, `FL`=Fall, `SP`=Spring. The most recent active semester is `26SU`. `All_Plots.ipynb` / `All_Plots_new.ipynb` aggregate figures for the paper.

### Calculation scripts

- **`run_hydro_analytic_contour.py`** (root and `new_paper_calculations/26-06-07/`): parallel, adaptive velocity-contour scan over the `(aQstar, T)` parameter space. Spawns workers via `ProcessPoolExecutor`; suppresses thread oversubscription with env vars (`OMP_NUM_THREADS=1`, etc.) for cluster use.
- **`new_paper_calculations/26-03-31/steady_front_scan.py`**: cluster-ready driver for shooting-method steady-front profiles. Reads a grid of initial guesses, writes per-trial `.npz` outputs and a `run_config.json`.
- **`new_paper_calculations/steady_front_summary.py`**: loads `.npz` results from a `steady_front_run_*` directory and produces summary plots.

### Other directories

- `wiki/` — the compiled research knowledge base, and a standalone Obsidian vault. Read `wiki/index.md` before searching elsewhere for project context; see the routing rules at the top of this file.
- `new_paper_calculations/` — dated subdirectories (`20Mar`, `31Mar`, `7Jun`, …) contain scripts and results for the current paper.
- `RMFsolver_Pub/` — standalone public-facing copy of the package with its own `pyproject.toml`; built separately with `pip install -e RMFsolver_Pub/`.
- `RMFsolver_Cluster_backup/` — snapshot of the version deployed on the compute cluster.
- `jargon/` — scratch PDFs and `.npz` files from exploratory calculations.
- `csv_share/`, `local_analysis_results/` — exported data for sharing or local reanalysis.

## Units convention

All thermodynamic quantities inside `RMFsolver` are in natural MeV/fm units unless otherwise stated. Use `constants.MeV_fm**3` (= ħc³ in MeV³·fm³) to convert between number density in fm⁻³ and MeV³.

Quark density convention used in phase-transition scripts:
- `P_f`, `E_f`, `n_B` are single-species helpers with spin included and color **excluded**.
- `n_u`, `n_d`, `n_s` are physical quark densities including color factor.
- `nB_QM = (n_u + n_d + n_s) / 3`, `nK_QM = (n_d - n_s) / 2`.

## Endpoint Notation

- Use location-based endpoint notation exclusively in all new or edited code, Markdown, prose, and formulas.
- In code and Markdown identifiers, use the suffix mapping `_N` -> `_0minus`, `_Qstar` -> `_0plus`, and `_Q` -> `_inf`. Do not introduce new endpoint identifiers with `_N`, `_Qstar`, or `_Q`.
- In mathematical formulas, use the mapping `X^N` -> `X(0^-)`, `X^{Q^*}` -> `X(0^+)`, and `X^Q` -> `X(\infty)`.
- Preserve legacy endpoint names only when required to read existing saved artifacts or maintain a public compatibility contract; mark them explicitly as legacy and do not use them in new physical reasoning.