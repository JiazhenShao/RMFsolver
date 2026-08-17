---
summary: How this wiki is structured and maintained. Any LLM agent reads this before writing to the wiki.
status: current
updated: 2026-08-11
tags: [meta, schema]
---

# Schema

Conventions for maintaining this wiki. **LLM agents write this vault; the user reads it in Obsidian.** The user is never expected to edit pages by hand.

## What this vault is

A *compiled* artifact, not a document collection. Raw sources — notebooks, solver code, cluster payloads, papers — are expensive to read. This wiki is the compiled form: written once, read many times, by both parties. Pages must therefore be cheap to retrieve from, not merely pleasant to read.

## Layout

```
wiki/
  index.md      catalog — the only page loaded by default
  log.md        append-only chronology
  SCHEMA.md     this file
  physics/      concepts, closures, bounds, observables
  solvers/      solver behavior, pathologies, known issues
  methods/      numerical methodology reused across scans
  paper/        manuscript state and section drafts
  sources/      one page per external source that matters
  assets/       images (created on demand)
```

## Page rules

- **One page per idea.** If a page needs two summary sentences, it is two pages.
- **Cap ~3k tokens.** Split when exceeded. A page is a unit of retrieval, not a chapter.
- **Frontmatter is mandatory:**
  ```yaml
  summary: one line, states the conclusion, not the topic
  status: current | superseded | open-question
  updated: YYYY-MM-DD
  tags: [...]
  ```
  `summary` is what gets scanned when deciding whether to open a page. "How the LTE bound works" is a bad summary; "β_LTE = 5·u_N·λ_n, and the coefficient is not 5" is a good one.
- **Link liberally** using Obsidian wikilink syntax. Links are the retrieval graph.
- **Mark superseded claims in place.** Set `status: superseded`, state what replaced it and when, and link forward. Never silently delete a conclusion — the reversal is often the most valuable content.
- **Lead with the conclusion.** Derivations go below it, trimmed to what changes a decision.
- **Flag traps with ⚠.** Reserve it for things that have actually caused rework.

## Math

Obsidian renders MathJax, so use `$inline$` and `$$display$$` here. Keep code identifiers, payload field names and shell commands in backticks — never as math. Formulas meant to be copied into Python stay ASCII in fenced blocks.

## Operations

**Ingest.** Identify the smallest complete authoritative source unit, then read that unit in full before writing. A unit may be a paper, the active body of a mixed draft, a self-contained section, or selected notebook cells discovered through the notebook table of contents. Do not load ignored duplicates, generated payloads or superseded append-only history unless provenance requires them. A single ingest may touch several pages. Append one `log.md` entry and update `index.md` — but see the caching rule below.

**Query.** `index.md` → frontmatter → page body → raw source. Never skip up the ladder. `rg` over `wiki/` returns lines; `Read` returns whole files — prefer the former.

**File the answer.** When a session produces a conclusion worth keeping, it becomes a page. That is the compounding mechanism; without it the wiki goes stale and the work is repeated.

**Lint.** Periodically: contradictions between pages, claims newer sources have superseded, orphans, concepts mentioned everywhere but lacking a page, missing cross-references. Obsidian's graph view shows orphans and hubs directly.

## Token rules

- **Batch index updates to end of session.** `index.md` sits in the stable prefix; editing it mid-session invalidates the prompt cache and re-bills everything before it.
- **Isolate large ingests when the agent supports it.** A paper ingest generates far more intermediate reasoning than conclusion; the primary thread needs only the source-backed conclusions and edit summary.
- **Never read a `.ipynb` raw.** Use `.claude/tools/nbsrc.py`. Claude's `PreToolUse` hook enforces this; other agents must follow their project instructions.

## Two memory tiers — do not merge them

| | Location | Loaded | Holds |
|---|---|---|---|
| Hot (Codex) | `AGENTS.md` | every task, automatically | invariant safety and retrieval rules only |
| Hot (Claude) | `~/.claude/projects/-Users-janshao-NSM-related/memory/` | every session, automatically | ~8 short facts; must stay small |
| Cold | this wiki | on demand | durable domain conclusions, unbounded |

Hot-context entries should point *into* the wiki rather than duplicating it. Keeping every automatically loaded layer small is what keeps the per-session floor cheap.

## Boundaries

This vault holds conclusions. It does not hold code, payloads, or figures — those stay in `NSM_related/` and are referenced by path. `docs/superpowers/specs/` is design archaeology: where it disagrees with this wiki, **this wiki wins**.
