# 26SU-9 LTE Temperature-Jump Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and execute `26SU-9.ipynb` with diagnostics and a publication-style native-grid contour of the LTE interface-temperature bound.

**Architecture:** The notebook loads the dated NumPy payload, validates the two mesh blocks and saved physical identities, then plots both adaptive blocks directly with shared contour levels. Saved boundary arrays define the two gray forbidden regions, avoiding interpolation outside the computed domain.

**Tech Stack:** Jupyter Notebook, Python 3.12, NumPy, pandas, Matplotlib, nbconvert.

---

### Task 1: Create the notebook and diagnostics

**Files:**
- Create: `26SU-9.ipynb`

- [ ] **Step 1: Add the notebook introduction and loader**

Use `Path.cwd()` as the repository root and load:

```python
DATA_PATH = ROOT / "new_paper_calculations" / "26-07-23" / "lte-temperature-jump-two-block-linear.npy"
payload = np.load(DATA_PATH, allow_pickle=True).item()
```

- [ ] **Step 2: Add diagnostics and assertions**

Calculate the low/high finite counts and ranges, status counts, seam mismatch, temperature-identity residual, and beta-identity residual.
Assert successful statuses, finite fields, matching shapes, exact seam agreement, and residuals below `1e-12`.

- [ ] **Step 3: Execute to verify diagnostics**

Run:

```text
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp ./bin/jupyter nbconvert --to notebook --execute 26SU-9.ipynb --output 26SU-9.executed.ipynb
```

Expected: execution succeeds and the diagnostics table is persisted.

### Task 2: Add the native two-block publication contour

**Files:**
- Modify: `26SU-9.ipynb`

- [ ] **Step 1: Add the tunable plotting block**

Define every used plotting control as an uppercase constant at the top of the visualization cell, including figure geometry, fonts, contour levels and colors, mask colors, boundary styles, labels, limits, ticks, annotations, and export paths.

- [ ] **Step 2: Draw masks and contours**

Shade above the saved `A_boundary = 1` curve and the low-metastability side of the saved `Delta h = 0` curve.
Draw the low block and the high block excluding its duplicated seam row with shared levels `(-0.04, 0, 20, 40, 60, 80, 90)`.
Emphasize zero with a thicker line and label all levels inline.

- [ ] **Step 3: Export and execute**

Save:

```text
local_analysis_results/26SU-9/lte-temperature-jump-contour.pdf
local_analysis_results/26SU-9/lte-temperature-jump-contour.png
```

Execute the notebook with nbconvert and require nonempty exports.

### Task 3: Verify the rendered result

**Files:**
- Verify: `26SU-9.executed.ipynb`
- Verify: `local_analysis_results/26SU-9/lte-temperature-jump-contour.png`
- Verify: `local_analysis_results/26SU-9/lte-temperature-jump-contour.pdf`

- [ ] **Step 1: Inspect notebook outputs**

Require no error outputs, one diagnostics table, and one inline Matplotlib image.

- [ ] **Step 2: Inspect the exported PNG**

Check visually for correct gray masks, readable labels, no duplicated seam, no clipping, and style agreement with the supplied reference.

- [ ] **Step 3: Re-run after any corrections**

Repeat nbconvert execution until diagnostics, notebook outputs, and both exports pass verification.

