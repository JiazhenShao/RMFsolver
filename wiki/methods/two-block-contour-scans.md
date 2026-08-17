---
summary: The two-block adaptive mesh used by every contour scan — why the domain is not rectangular, and the 80 MeV seam rule.
status: current
updated: 2026-08-11
tags: [method, mesh, cluster]
---

# Two-block adaptive contour scans

The standard mesh for every $(T_N,\ \text{metastability})$ contour in this project. Established 2026-07-18, reused by every scan since.

## Why not a rectangular grid

The admissible domain is bounded by two *different kinds* of curve, and a single-valued temperature cut inverts one of them into a false cutoff.

**Enthalpy-gap boundary** — single-valued in metastability at fixed temperature:

$$\Delta h(T,\ m_{\rm gap}(T)) = 0.$$

In the current parameter domain it begins at $(m,T) = (0,0)$, reaches about $m = 0.312$ near $T = 50$ MeV, and returns to the metastability axis at $(m,T) = (0,\ 60.1586\ \mathrm{MeV})$. The forbidden region is bounded left by $m=0$ and right by $m_{\rm gap}(T)$. **There is no second orange branch** — an earlier reading that saw one was wrong.

**High-temperature boundary** — single-valued in temperature at fixed metastability:

$$A_{\rm boundary}(T_A(m),\ m) = 1,$$

lying above roughly 93.9 MeV.

## The 80 MeV seam

A split at $T_N = 80$ MeV is safely above the enthalpy-gap lobe and below the high-temperature boundary.

- **Low block:** uniform temperature rows from the configured minimum through 80 MeV. Per row, solve $m_{\rm gap}(T)$ on $[0, m_{\max}]$; if the whole row has $\Delta h \le 0$, use $m_{\rm gap}=0$; otherwise run from $m_{\rm gap}(T)+\epsilon_m$ to $m_{\max}$.
- **High block:** uniform metastability columns from 0 to $m_{\max}$. Per column, solve $T_A(m)$ above 80 MeV, then build temperature coordinates from the shared seam through $T_A(m)-\epsilon_T$.

**Both boundaries must be computed before the adaptive blocks are filled.**

## ⚠ Never concatenate the blocks

They remain separate arrays in the payload. Plotters contour them separately on the same axes. They must **not** be merged into one logically rectangular array — that is exactly the mistake the design replaced.

The low-block seam row is copied into the first high-block row so both agree exactly at 80 MeV. When plotting, **omit the duplicated first high row from the second contour call** or the seam is drawn twice.

## Continuation rules

Low block: begin with the row nearest 10 MeV, then alternate upward and downward. Within a row start at $m_{\max}$ and propagate left through immediately adjacent points. A new row's endpoint seed may come only from the previously solved adjacent row's $m_{\max}$ endpoint.

High block: start each column at the seam and propagate upward from the preceding same-column endpoint. The seam is solved once per column.

**Retries may use only immediately adjacent successful cells within the same block, and may never cross a forbidden boundary.**

## Defaults

$T_N$ from $10^{-2}$ to 100 MeV; 30 low rows, 12 high rows; metastability 0 to 0.9 in 20 columns; 160 samples for the enthalpy boundary; 8 probes and 10 bisection steps for the composition boundary. Serial or scheduler-provided worker counts.

For bisection depth on ray grids specifically, see the asymmetry note in [[known-issues]].

## The runner lineage

- `26-07-19/uNmax_contour.py` — the canonical **linear–linear** two-block template
- `26-07-20/run_uNmax_logTm_entropy_safe.py` — logarithmic low-$T$ entropy-safe variant, *not* the linear–linear template
- `26-07-23/run_lte_temperature_jump_contour.py` — the [[interface-temperature-jump]] scan
- `_hydro_analytic_contour_common.py` — copied into each dated folder so the folder stays cluster-portable

Each dated folder under `new_paper_calculations/` carries its own isolated copy by design. Do not refactor them into a shared import.

## Output naming

New observables get a **new default filename** so existing payloads are never overwritten. Retain no misleading single `safe_temperature_cut_values` field in any new schema.
