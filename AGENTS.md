# Project Instructions

## Token-Efficient Context Routing

- Keep this file small: it is hot context loaded for every task. Domain conclusions belong in `wiki/` and are retrieved only when relevant.
- Do not preload the whole wiki. For physics or research questions, follow `wiki/index.md` -> targeted `rg` search -> relevant page body -> raw source only if the wiki is insufficient.
- For code changes, use the relevant wiki page for established constraints, then inspect the exact live source needed for implementation. Live code remains authoritative for current behavior.
- Before creating or revising wiki pages, read `wiki/SCHEMA.md`. File durable conclusions back into the wiki, then update `wiki/index.md` and append one `wiki/log.md` entry at the end of the session.
- Prefer `rg` and bounded line ranges over whole-file reads. Check the type and size of unfamiliar artifacts before opening them; avoid generated data, embedded base64, binary outputs, caches, and checkpoints unless the task requires them.

## Never Read a Notebook Raw

Notebooks are dominated by embedded base64 figures, so raw reads can cost 20--100 times more tokens than their source. Never use `cat`, `sed`, `grep`, `rg`, or a whole-file read directly on an `.ipynb`. Use the existing extractor and start with its table of contents:

```bash
./bin/python3 .claude/tools/nbsrc.py NOTEBOOK.ipynb
./bin/python3 .claude/tools/nbsrc.py NOTEBOOK.ipynb -g 'PATTERN'
./bin/python3 .claude/tools/nbsrc.py NOTEBOOK.ipynb -c 3
./bin/python3 .claude/tools/nbsrc.py NOTEBOOK.ipynb -c 2-5,9
./bin/python3 .claude/tools/nbsrc.py NOTEBOOK.ipynb -o 3
./bin/python3 .claude/tools/nbsrc.py NOTEBOOK.ipynb -f 3
```

Fetch only the cells or text outputs needed. To inspect a figure, extract it with `-f` and open the resulting image rather than reading its base64 payload.

## RMFsolver Protection

- Do not create, edit, delete, rename, format, or otherwise modify any file under `RMFsolver/` except `RMFsolver/phase_velocity.py` unless the user specifically and clearly instructs modification of another named file.
- A general request to debug, fix, update, or inspect a solver does not authorize changes outside `RMFsolver/phase_velocity.py`.
- Reading files, inspecting Git history, running diagnostics, and executing tests under `RMFsolver/` remain allowed.

## Endpoint Notation

- Use location-based endpoint notation exclusively in all new or edited code, Markdown, prose, and formulas.
- In code and Markdown identifiers, use the suffix mapping `_N` -> `_0minus`, `_Qstar` -> `_0plus`, and `_Q` -> `_inf`. Do not introduce new endpoint identifiers with `_N`, `_Qstar`, or `_Q`.
- In mathematical formulas, use the mapping `X^N` -> `X(0^-)`, `X^{Q^*}` -> `X(0^+)`, and `X^Q` -> `X(\infty)`.
- Preserve legacy endpoint names only when required to read existing saved artifacts or maintain a public compatibility contract; mark them explicitly as legacy and do not use them in new physical reasoning.
