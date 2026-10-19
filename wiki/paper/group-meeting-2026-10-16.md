---
summary: Root-level 26-10-14.md collects the October 8 checklist and an added exact-baseline comparison of the Eq. (35) ceiling, with embedded figures and speaking notes.
status: current
updated: 2026-10-16
tags: [paper, meeting, presentation, provenance]
---

# Group Meeting: October 8 Checklist Follow-Up

[26-10-14.md](../../26-10-14.md) is the meeting document for the 2026-10-16 discussion.
It follows the eight checklist items in their original order: thermal gluons, chemical-potential substitution in Eq. (23), collaborator export, hot isothermal rate sensitivity, adiabatic sensitivity, mean-free-path diagnostics, the isothermal log-temperature figure, and the constant profile factor.

## Artifact Contract

- Local PNGs are available under `26-10-14.assets/`; the meeting document embeds the selected figures through relative paths.
- Ten plots are byte-identical copies of existing saved figures; `manifest.json` records their source paths and hashes.
- The new `xi-optimum.png` rescales saved predictions and reproduces the band-balanced 50–2000 m/s result: best constant -0.865, current -0.85, objective increase 0.435%, 1%-objective basin [-0.890,-0.845].
- `constant-momentum-profile.png` redraws the original vector curves and samples from the archived PDF associated with `26-04-09.ipynb`, cell 9. Its matching raw profile was not located; the old notebook's calculation inputs have changed. The added cells 9–10 of `26-10-14.ipynb` record this provenance and use location-only endpoint labels. The endpoint change is approximately 0.073 MeV (0.0064%), at figure precision for this historical example, not a new validation with the current EOS.
- `aiso-accuracy.png` uses the verified 292-state audit to compare Eq. (35) with the exact static EOS ceiling. Cells 11–12 give provenance and an independently executable plot with values, signed differences, and percentage differences against exact composition. The exact reference is black, saved states are colored by temperature, and the dashed cold curve crosses 1% at exact composition 0.495544. The plot and report state that 113/292 checked targets exceed 1%, with a maximum of 8.48%; these are composition errors, not speed errors. See [[isothermal-maximum-speed]].
- `prepare_assets.py` recreates the figures and scan CSV without new EOS/BVP solves.
- The meeting document links the reproducible 60-state collaborator ZIP; Box and Overleaf upload status comes from the user's completed checklist, not remote verification.

## Scope Preserved in the Notes

The rate-integral speed comparisons keep interface/profile factors fixed and are not new front BVP solutions.
The exact rate is the specified tree-level massless four-quark channel, not a QCD-complete rate.
The tail criterion has only bounded sampled checks; the separate 33-root core curve uses a chosen tolerance of 1e-4 and is not a sharp LTE cutoff or the original shorthand's 10%-composition/gradient definition.
The fitted profile factor is an adiabatic, speed-range-dependent calibration, not an independent validation of the isothermal choice.

Initial assembly left all notebooks unchanged; follow-ups append provenance and plotting cells for the momentum-flux figure and Eq. (35) accuracy to `26-10-14.ipynb`.
No production solver, main manuscript, standalone derivation source, or original notebook cell was changed.
See [[bag-model-thermal-gluons]], [[massless-finite-temperature-weak-rate]], [[adiabatic-weak-rate-sensitivity]], [[isothermal-mfp-overlay]], and [[adiabatic-xi-diagnostics]] for the underlying conclusions.
