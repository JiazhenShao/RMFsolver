# Split Log and Linear Velocity Contours Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the combined logarithmic- and linear-temperature contour figure in `26SU-8.ipynb` into two independent figures while fixing the disappearing low-temperature log contours.

**Architecture:** Replace the current two-panel visualization cell with two self-contained visualization cells. Each cell loads only its own payload, owns a complete `# Tunable plotting parameters` block, creates one square figure, and writes uniquely named PNG/PDF files. The log cell configures the logarithmic transform before `contour()` and `clabel()`; the linear cell keeps its current linear behavior.

**Tech Stack:** Python, Jupyter notebook JSON, NumPy, Matplotlib, `unittest`

---

## Root Cause

The saved logarithmic grid is complete: all 600 low-block cells are finite. At `T = 0.01 MeV`, the `50`, `100`, and `200 m/s` levels are bracketed and cross near metastabilities `0.0515`, `0.1480`, and `0.3806`.

The current combined cell creates and labels the log contours while the axis still uses a linear transform, then calls `set_yscale("log")`. Because `clabel(inline=True)` removes line sections in display coordinates, the later scale change stretches those gaps across several decades. The missing branches are therefore a rendering-order defect, not missing cluster data.

### Task 1: Add regression coverage for separate cells and log operation order

**Files:**
- Create: `tests/test_26su8_contour_plotting.py`
- Inspect: `26SU-8.ipynb` current cell 13

- [ ] **Step 1: Write the failing notebook-structure test**

```python
from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "26SU-8.ipynb"
LOG_DATA_MARKER = (
    'LOG_T_DATA_PATH = "new_paper_calculations/26-07-18/'
    'uN-two-block-adaptive-contour-log.npy"'
)
LINEAR_DATA_MARKER = (
    'LINEAR_T_DATA_PATH = "new_paper_calculations/26-07-18/'
    'uN-two-block-adaptive-contour-linear-v2.npy"'
)
LOG_PNG_MARKER = (
    'CLUSTER_CONTOUR_PNG_FILENAME = '
    '"analytic-two-block-contour-log-entropy-boundary.png"'
)
LINEAR_PNG_MARKER = (
    'CLUSTER_CONTOUR_PNG_FILENAME = '
    '"analytic-two-block-contour-linear-v2-entropy-boundary.png"'
)


class ContourNotebookStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]

    def _single_cell_with(self, marker):
        matching = [
            (index, source)
            for index, source in enumerate(self.code_cells)
            if marker in source
        ]
        self.assertEqual(len(matching), 1)
        return matching[0]

    def test_log_and_linear_figures_are_separate_cells(self):
        log_index, log_source = self._single_cell_with(LOG_DATA_MARKER)
        linear_index, linear_source = self._single_cell_with(LINEAR_DATA_MARKER)

        self.assertNotEqual(log_index, linear_index)
        self.assertNotIn(LINEAR_DATA_MARKER, log_source)
        self.assertNotIn(LOG_DATA_MARKER, linear_source)
        self.assertIn(LOG_PNG_MARKER, log_source)
        self.assertIn(LINEAR_PNG_MARKER, linear_source)

    def test_each_visualization_has_a_tuning_block(self):
        self.assertTrue(
            self._single_cell_with(LOG_DATA_MARKER)[1].startswith(
                "# Tunable plotting parameters\n"
            )
        )
        self.assertTrue(
            self._single_cell_with(LINEAR_DATA_MARKER)[1].startswith(
                "# Tunable plotting parameters\n"
            )
        )

    def test_log_transform_precedes_contours_and_inline_labels(self):
        _, source = self._single_cell_with(LOG_DATA_MARKER)
        scale_index = source.index(
            "ax.set_yscale(CLUSTER_CONTOUR_Y_SCALE)"
        )
        limits_index = source.index(
            "ax.set_ylim(*CLUSTER_CONTOUR_Y_LIMITS_MEV)"
        )
        contour_index = source.index("contour = ax.contour(")
        label_index = source.index("ax.clabel(")

        self.assertLess(scale_index, contour_index)
        self.assertLess(limits_index, contour_index)
        self.assertLess(contour_index, label_index)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails on the combined cell**

Run:

```bash
./bin/python -m unittest tests.test_26su8_contour_plotting -v
```

Expected: failures because both data markers currently occur in one cell, the output filename is shared, and the log scale is applied after contours and labels.

- [ ] **Step 3: Preserve the failing test before editing the notebook**

Do not weaken the assertions to accommodate the current combined structure. This workspace is not currently detected as a Git working tree, so no commit command is included.

### Task 2: Replace the combined cell with a standalone logarithmic figure

**Files:**
- Modify: `26SU-8.ipynb` current cell 13
- Test: `tests/test_26su8_contour_plotting.py`

- [ ] **Step 1: Retain only log-specific inputs and outputs in the first visualization cell**

The cell must begin with its own tuning block and use these identifiers:

```python
# Tunable plotting parameters
LOG_T_DATA_PATH = "new_paper_calculations/26-07-18/uN-two-block-adaptive-contour-log.npy"
CLUSTER_CONTOUR_OUTPUT_DIR = "local_analysis_results/26SU-8"
CLUSTER_CONTOUR_PNG_FILENAME = "analytic-two-block-contour-log-entropy-boundary.png"
CLUSTER_CONTOUR_PDF_FILENAME = "analytic-two-block-contour-log-entropy-boundary.pdf"
CLUSTER_CONTOUR_FIGURE_SIZE = (7.2, 7.2)
CLUSTER_CONTOUR_FIGURE_DPI = 160
CLUSTER_CONTOUR_EXPORT_DPI = 200
CLUSTER_CONTOUR_Y_LIMITS_MEV = (1.0e-2, 100.0)
CLUSTER_CONTOUR_Y_SCALE = "log"
CLUSTER_CONTOUR_PANEL_TITLE = r"Two-block adaptive logarithmic $T$ mesh"
```

Retain the existing adjustable colors, contour levels, line widths, label settings, shading, legend, axis fonts, tick settings, box aspect, and export background as uppercase cell-local constants. Remove `LINEAR_T_DATA_PATH`, linear limits, linear scale, subplot row/column counts, and the linear panel title from this cell.

- [ ] **Step 2: Create one square logarithmic axis**

Replace the two-panel figure construction with:

```python
fig, ax = plt.subplots(
    figsize=CLUSTER_CONTOUR_FIGURE_SIZE,
    dpi=CLUSTER_CONTOUR_FIGURE_DPI,
    constrained_layout=CLUSTER_CONTOUR_CONSTRAINED_LAYOUT,
    facecolor=CLUSTER_CONTOUR_BACKGROUND_COLOR,
)
ax.set_facecolor(CLUSTER_CONTOUR_BACKGROUND_COLOR)
ax.set_yscale(CLUSTER_CONTOUR_Y_SCALE)
ax.set_xlim(*CLUSTER_CONTOUR_X_LIMITS)
ax.set_ylim(*CLUSTER_CONTOUR_Y_LIMITS_MEV)
```

These transform and limit calls must occur before the first `ax.fill_between(...)`, `ax.contour(...)`, `ax.contourf(...)`, or `ax.clabel(...)` call.

- [ ] **Step 3: Keep only the existing log-payload plotting path**

Load `LOG_T_DATA_PATH`, require `temperature_mesh == "adaptive_log"`, and build `block_data` for the `low` and `high` blocks. Move the existing log implementations of the following operations into this cell, changing `log_ax` to `ax`:

```python
ax.fill_between(...)  # A_boundary >= 1 region
ax.plot(...)          # A_boundary = 1
ax.fill(...)          # Delta h > 0 region
ax.plot(...)          # Delta h = 0
ax.contour(...)       # velocity levels
ax.clabel(...)        # velocity labels
ax.contourf(...)      # negative entropy-jump region
ax.contour(...)       # zero entropy-jump boundary
```

Use the existing `entropy_per_baryon_jump_grid` calculation unchanged. Add the complete four-entry legend to this standalone figure rather than leaving it only on the linear figure.

- [ ] **Step 4: Save and display only the log figure**

Use:

```python
fig.suptitle(
    rf"Extreme-endothermic $u(0^-)$ for $n_B^{{\rm crit}}(T=0)={float(payload['nB_crit0_over_n0']):.1f}\,n_0$ in $m\cdot s^{{-1}}$",
    fontsize=CLUSTER_CONTOUR_SUPTITLE_FONT_SIZE,
)
fig.savefig(
    CLUSTER_CONTOUR_OUTPUT_DIR / CLUSTER_CONTOUR_PNG_FILENAME,
    dpi=CLUSTER_CONTOUR_EXPORT_DPI,
    facecolor=CLUSTER_CONTOUR_BACKGROUND_COLOR,
)
fig.savefig(
    CLUSTER_CONTOUR_OUTPUT_DIR / CLUSTER_CONTOUR_PDF_FILENAME,
    facecolor=CLUSTER_CONTOUR_BACKGROUND_COLOR,
)
plt.show()
```

### Task 3: Add a standalone linear-temperature figure cell

**Files:**
- Modify: `26SU-8.ipynb` by inserting one code cell immediately after the log figure cell
- Test: `tests/test_26su8_contour_plotting.py`

- [ ] **Step 1: Add an independent linear tuning block**

Start the new cell with:

```python
# Tunable plotting parameters
LINEAR_T_DATA_PATH = "new_paper_calculations/26-07-18/uN-two-block-adaptive-contour-linear-v2.npy"
CLUSTER_CONTOUR_OUTPUT_DIR = "local_analysis_results/26SU-8"
CLUSTER_CONTOUR_PNG_FILENAME = "analytic-two-block-contour-linear-v2-entropy-boundary.png"
CLUSTER_CONTOUR_PDF_FILENAME = "analytic-two-block-contour-linear-v2-entropy-boundary.pdf"
CLUSTER_CONTOUR_FIGURE_SIZE = (7.2, 7.2)
CLUSTER_CONTOUR_FIGURE_DPI = 160
CLUSTER_CONTOUR_EXPORT_DPI = 200
CLUSTER_CONTOUR_Y_LIMITS_MEV = (0.0, 100.0)
CLUSTER_CONTOUR_Y_SCALE = "linear"
CLUSTER_CONTOUR_PANEL_TITLE = r"Two-block adaptive linear $T$ mesh (v2)"
```

Repeat the currently used plotting constants in this cell so it remains independently tunable. Do not read or reference `LOG_T_DATA_PATH` or any variables created by the preceding log cell.

- [ ] **Step 2: Build one square linear figure**

Use the same one-axis construction as the log cell, then configure:

```python
ax.set_facecolor(CLUSTER_CONTOUR_BACKGROUND_COLOR)
ax.set_yscale(CLUSTER_CONTOUR_Y_SCALE)
ax.set_xlim(*CLUSTER_CONTOUR_X_LIMITS)
ax.set_ylim(*CLUSTER_CONTOUR_Y_LIMITS_MEV)
ax.set_box_aspect(CLUSTER_CONTOUR_BOX_ASPECT)
```

- [ ] **Step 3: Move the existing linear plotting path unchanged**

Load only `LINEAR_T_DATA_PATH`, require `temperature_mesh == "adaptive_linear"`, and retain the existing safe-domain shading, velocity contours, entropy overlay, labels, ticks, spines, title, and four-entry legend. Use `ax` rather than `axes[1]`.

- [ ] **Step 4: Save the independent linear outputs**

Save and display:

```text
local_analysis_results/26SU-8/analytic-two-block-contour-linear-v2-entropy-boundary.png
local_analysis_results/26SU-8/analytic-two-block-contour-linear-v2-entropy-boundary.pdf
```

Do not overwrite the new log outputs or the old combined figure artifacts.

### Task 4: Verify both figures independently

**Files:**
- Verify: `26SU-8.ipynb`
- Test: `tests/test_26su8_contour_plotting.py`

- [ ] **Step 1: Run the structure and transform-order tests**

Run:

```bash
./bin/python -m unittest tests.test_26su8_contour_plotting -v
```

Expected: all three tests pass.

- [ ] **Step 2: Execute the log cell by itself**

Expected: it succeeds without the linear cell having run, produces a square log figure, and shows the `50`, `100`, and `200 m/s` branches approaching `T = 0.01 MeV` except for small local gaps directly beneath inline labels.

- [ ] **Step 3: Execute the linear cell by itself**

Expected: it succeeds without the log cell having run and reproduces the current linear-temperature contours in its own square figure.

- [ ] **Step 4: Inspect the output identities and dimensions**

Confirm that all four new files exist, that the log and linear PNGs are distinct files, and that neither is the old combined two-panel image.

- [ ] **Step 5: Run the existing analytic cluster regressions**

Run:

```bash
./bin/python -m unittest \
  tests.test_26su8_contour_plotting \
  tests.test_cluster_analytic_26_07_18 -v
```

Expected: all tests pass and no `.npy` artifact changes.

- [ ] **Step 6: Record the verified-file checkpoint**

Confirm that the intended changed files are exactly `26SU-8.ipynb`, `tests/test_26su8_contour_plotting.py`, and this plan document. This workspace is not currently detected as a Git working tree, so no commit command is included.

## Self-Review

- The combined visualization becomes exactly two independent figures and two visualization cells.
- Both cells satisfy the notebook requirement for a top-of-cell `# Tunable plotting parameters` block.
- The log transform is installed before contours and inline labels, addressing the evidenced root cause.
- The linear rendering path remains physically and visually unchanged apart from becoming a standalone square figure.
- Cluster `.npy` data, adaptive meshes, solver code, and safe-domain definitions remain untouched.
