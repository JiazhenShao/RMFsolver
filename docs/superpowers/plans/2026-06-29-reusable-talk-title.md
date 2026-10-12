# Reusable Talk Title Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-use LaTeX title with a reusable full-width `\talktitle{...}` bubble.

**Architecture:** Keep the title implementation inside `Notes.tex` as a single one-argument command because the document is self-contained. The command reuses the current nested `tcolorbox` styling, while callers remain responsible for inserting `\newpage`.

**Tech Stack:** LaTeX, `tcolorbox`, `pdflatex`, Poppler

---

### Task 1: Establish a failing compile test

**Files:**
- Test: `NNPSS 2026/Notes/Notes.tex` through a command-line LaTeX hook
- Create during testing: `tmp/pdfs/reusable-title-test/`

- [ ] **Step 1: Create the test-output directory**

Run:

```bash
mkdir -p tmp/pdfs/reusable-title-test
```

Expected: the directory exists and the source tree is unchanged.

- [ ] **Step 2: Compile with a second title injected on a new final page**

Run from `NNPSS 2026/Notes`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Notes-title-test -output-directory="../../tmp/pdfs/reusable-title-test" '\AtEndDocument{\newpage\talktitle{Reusable Title Test}}\input{Notes.tex}'
```

Expected: FAIL with an undefined-control-sequence error for `\talktitle`.

### Task 2: Implement the reusable command

**Files:**
- Modify: `NNPSS 2026/Notes/Notes.tex:35-45`
- Modify: `NNPSS 2026/Notes/Notes.tex:166-167`

- [ ] **Step 1: Replace the one-use title definition with the reusable command**

Replace the current `\title{...}` and `\date{}` block with:

```latex
\newcommand{\talktitle}[1]{%
\vspace{-3em}
\begin{tcolorbox}[colframe=white,opacityback=0]
\begin{tcolorbox}
\Huge\sffamily #1
\end{tcolorbox}
\end{tcolorbox}
\vspace{-3em}
}
```

- [ ] **Step 2: Replace the initial `\maketitle` call**

Immediately after `\begin{document}`, use:

```latex
\talktitle{Lecture Notes NNPSS 2026}
```

- [ ] **Step 3: Re-run the injected-title compile test**

Run from `NNPSS 2026/Notes`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Notes-title-test -output-directory="../../tmp/pdfs/reusable-title-test" '\AtEndDocument{\newpage\talktitle{Reusable Title Test}}\input{Notes.tex}'
```

Expected: PASS and produce `tmp/pdfs/reusable-title-test/Notes-title-test.pdf` containing both title strings.

- [ ] **Step 4: Compile the ordinary document**

Run from `NNPSS 2026/Notes`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error Notes.tex
```

Expected: PASS and update `Notes.pdf`.

### Task 3: Verify rendered layout

**Files:**
- Read: `NNPSS 2026/Notes/Notes.pdf`
- Create during verification: `tmp/pdfs/notes-page`

- [ ] **Step 1: Render the ordinary PDF to PNG**

Run:

```bash
pdftoppm -png -r 150 "NNPSS 2026/Notes/Notes.pdf" "tmp/pdfs/notes-page"
```

Expected: one PNG per PDF page.

- [ ] **Step 2: Inspect the first page**

Confirm that the title bubble is legible, is not clipped, retains its original styling, and spans the full text width above the two-sided `\topic` layout.

- [ ] **Step 3: Verify the reusable-title test PDF**

Render `tmp/pdfs/reusable-title-test/Notes-title-test.pdf` and inspect the page containing `Reusable Title Test`.
Confirm that the injected title uses the same full-width styling and does not overlap adjacent content.

No commit step is included because `/Users/janshao/NSM_related` is not a Git repository.
