# Manuscript insertion: build and humanizer review

Date: 2026-10-26.
Source: `Latex_writting/Main.tex`, immediately after `eq:a-max` in `sec:adiabatic-analytic`.
Only the insertion below was added to the pre-existing manuscript; the full pre-edit source was saved before editing.

## Build verification

Both the untouched snapshot and the edited source build successfully with exit status 0 and produce 17-page PDFs.
The final builds ran from independent temporary source directories, each with its own copy of `Main.tex`, `reflist.bib`, and `JHEP.bst`; the existing figure directory was referenced by symlink.
Command in each isolated source directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error Main.tex
```

- Baseline directory: `/private/tmp/nsm-second-order-manuscript-6c6r46_2/baseline-isolated/`.
- Edited directory: `/private/tmp/nsm-second-order-manuscript-6c6r46_2/edited-isolated/`.
- Original source snapshot: `/private/tmp/nsm-second-order-manuscript-6c6r46_2/Main.before.tex`.
- Baseline console log: `/private/tmp/nsm-second-order-manuscript-6c6r46_2/baseline-isolated.console.log`.
- Edited console log: `/private/tmp/nsm-second-order-manuscript-6c6r46_2/edited-isolated.console.log`.

Final LaTeX logs contain the same 16 warnings/box diagnostics after normalizing source-line and page locations, with no additions or removals and no errors.
The pre-existing unresolved references are `sec:XX`, `fig:speed-adiabatic`, and `fig:speed-adibatic`; `eq:isothermal_speed_scaling` remains multiply defined.
The existing `nameref`, `caption`, REVTeX/float warnings and the 5.09543-point equation overfull box are unchanged.
The new display introduces no overfull or underfull box.

Verification compared the source snapshots and generated auxiliary files:

- All 64 pre-existing numbered equation environments are byte-identical.
- All 100 auxiliary label assignments retain their original values; insertion-dependent float record ordering may differ.
- Both occurrences of the 82 m/s prefactor and the entire `eq:speed_modified` environment are unchanged.
- Removing the single inserted byte sequence reproduces the original `Main.tex` exactly.
- SHA-256 checks confirm all 120 tracked build artifacts are unchanged, including the user's already-modified `Main.pdf` and `Main.aux`.

An initial build using only `-outdir` exposed a latexmk/BibTeX behavior that wrote source-side `Main.bbl` and the untracked BibTeX log `Main.blg`.
The original `Main.bbl` was recovered from Git only after its SHA-256 was proven identical to the hash recorded before the build, then restored byte-for-byte.
The final build comparison uses the fully isolated source directories above, which keep both TeX and BibTeX output outside the manuscript tree.

Page 13 of the edited PDF was rendered with macOS PDFKit and visually inspected: the entire violet insertion fits the column, has no clipping or overlap, and the surrounding equation numbers remain E2 and E3.
The inspected rendering is `/private/tmp/nsm-second-order-manuscript-6c6r46_2/edited-page-13.png`.

## Humanizer review

Detected format: scientific manuscript prose; apply the universal humanizer checks and use the surrounding appendix as the voice sample.
The skill's blog/LinkedIn formatting and engagement advice do not apply to this technical insertion.
Editorial scores describe this passage only, not statistical AI-detection probabilities.

| Dimension | Score | Assessment |
|---|---|---|
| AI-likeness | 1/10 | The passage states the matching conditions directly, with no stock transitions, hype, or ornamental framing. |
| Authenticity | 8/10 | Two concise sentences match the appendix's explanatory register and established notation. |
| Reader value | 9/10 | The text identifies the endpoint equations, their consequence for the composition ceiling, and the density-truncation limitation. |
| Domain credibility | 9/10 | The assertions are tied to the existing EOS and density equations, with no added scientific claim beyond the authorized snippet. |

No AI-pattern or originality flags require a rewrite.
The final prose uses one complete sentence per source line, preserves all requested content, and keeps the leading-order density warning explicit.
No new reusable detection pattern was found; the humanizer skill file was not changed.

## Exact insertion diff

```diff
--- Main.tex before insertion
+++ Main.tex after insertion
@@ -1118,6 +1118,17 @@
 \label{eq:a-max}
 \end{equation}
 
+\JS{
+Matching $P$ and $h/n_B$ at the same order as Eq.~\eqref{eq:QM-EoS-2nd-order} fixes $\mu_B(0^+)$ and $T(\infty)$.
+\[
+\begin{aligned}
+P(0^-)+U_{\text{bag}} &= \frac{\mu_B(0^+)^4}{108\pi^2}+\frac{\mu_B(0^+)^2 T(\infty)^2}{6}\,,\\
+\frac{h(0^-)}{n_B(0^-)} &= \mu_B(0^+)+\frac{9\pi^2 T(\infty)^2}{\mu_B(0^+)}\,.
+\end{aligned}
+\]
+These conditions give $A_{\rm adb}=9\pi T(\infty)/(\sqrt{2}\,\mu_B(0^+))$; the leading-order density in Eq.~\eqref{eq:nB-approx} is insufficient for this heat balance.
+}
+
 We can now evaluate the integral $I_2$ \eqref{eq:I2_definition} by using the results from
 Appendices~\ref{sec:K-creation} and \ref{sec:K-diffusion})
 for $\Ga_K(a,T)$ and $D_K(a,T)$ (keeping only the first term), and using \eqref{eq:constant_pressure_path} to express $T$ as a function of $a$. 
```

## Source hashes

```text
Before: a6b284f40dfa2e3187ede4b2fa8ffa44ea08810f05928f8bf7c3179eb7c5b35b
After:  ccca3bd174785589048ba2a202a19596de1b0ae1fbbddc03811188767a2f82dc
```
