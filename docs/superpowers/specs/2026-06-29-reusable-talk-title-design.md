# Reusable Talk Title Design

## Goal

Make the existing title bubble in `NNPSS 2026/Notes/Notes.tex` reusable for multiple talks in the same document.

## Interface

Define a one-argument command:

```latex
\talktitle{Talk title}
```

The author controls pagination explicitly:

```latex
\newpage
\talktitle{Another Talk}
```

## Appearance and behavior

- The command renders the title in the same large sans-serif style as the current title.
- The bubble occupies the full text width, spanning both sides of the `\topic` layout.
- The command does not insert a page break.
- The current one-off `\title` and `\maketitle` setup is replaced with `\talktitle{Lecture Notes NNPSS 2026}`.
- No unrelated document styling or note content is changed.

## Verification

Create a temporary compile test containing two `\talktitle` calls separated by `\newpage`.
First confirm that this test fails before the command exists.
After implementation, compile the document and confirm that both full-width title bubbles render without LaTeX errors.
