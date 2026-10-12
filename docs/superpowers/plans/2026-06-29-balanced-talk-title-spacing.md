# Balanced Talk Title Spacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a balanced `\maketitle`-like gap between each reusable talk title and the note bubbles below it.

**Architecture:** Keep the reusable `\talktitle` command and all visual styling unchanged. Adjust only its trailing vertical spacing from `-3em` to `-2em`, then verify the ordinary and repeated-title renderings.

**Tech Stack:** LaTeX, `tcolorbox`, `pdflatex`, Ghostscript

---

### Task 1: Establish the spacing regression check

**Files:**
- Test: `NNPSS 2026/Notes/Notes.tex:35-43`

- [ ] **Step 1: Confirm that the selected balanced spacing is absent**

Run:

```bash
rg -F '\vspace{-2em}' 'NNPSS 2026/Notes/Notes.tex'
```

Expected: FAIL with exit code `1` because the command still ends with `\vspace{-3em}`.

### Task 2: Apply balanced spacing

**Files:**
- Modify: `NNPSS 2026/Notes/Notes.tex:42`

- [ ] **Step 1: Change only the trailing title spacing**

The completed command must be:

```latex
\newcommand{\talktitle}[1]{%
\vspace{-3em}
\begin{tcolorbox}[colframe=white,opacityback=0]
\begin{tcolorbox}
\Huge\sffamily #1
\end{tcolorbox}
\end{tcolorbox}
\vspace{-2em}
}
```

- [ ] **Step 2: Confirm the selected spacing is present**

Run:

```bash
rg -n -F '\vspace{-2em}' 'NNPSS 2026/Notes/Notes.tex'
```

Expected: PASS with one match inside `\talktitle`.

- [ ] **Step 3: Compile the ordinary document**

Run from `NNPSS 2026/Notes`:

```bash
pdflatex -interaction=batchmode -halt-on-error Notes.tex
```

Expected: PASS with exit code `0`.

- [ ] **Step 4: Compile a repeated-title regression document**

Run from `NNPSS 2026/Notes`:

```bash
pdflatex -interaction=batchmode -halt-on-error -jobname=Notes-balanced-test -output-directory='../../tmp/pdfs/balanced-title-test' '\AtEndDocument{\newpage\talktitle{Balanced Title Test}}\input{Notes.tex}'
```

Expected: PASS with exit code `0` and a two-page `Notes-balanced-test.pdf`.

### Task 3: Verify the rendered balance

**Files:**
- Read: `NNPSS 2026/Notes/Notes.pdf`
- Read: `tmp/pdfs/balanced-title-test/Notes-balanced-test.pdf`
- Create during verification: `tmp/pdfs/balanced-title-test/*.png`

- [ ] **Step 1: Render both title pages**

Run:

```bash
gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r150 -sOutputFile='tmp/pdfs/balanced-title-test/notes-%d.png' 'NNPSS 2026/Notes/Notes.pdf'
```

Run:

```bash
gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r150 -sOutputFile='tmp/pdfs/balanced-title-test/test-%d.png' 'tmp/pdfs/balanced-title-test/Notes-balanced-test.pdf'
```

Expected: PNG renderings for the ordinary page and both test pages.

- [ ] **Step 2: Inspect the first ordinary page and repeated title page**

Confirm that both title bubbles remain full width and unclipped, the note bubbles do not overlap the title, and the gap below the title matches the selected balanced treatment.

No commit step is included because `/Users/janshao/NSM_related` is not a Git repository.
