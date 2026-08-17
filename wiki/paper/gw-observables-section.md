---
summary: The GW-observables drop-in section — settled notation, the taiji geometry, the six calibrated cases, and the citation corrections that keep recurring.
status: current
updated: 2026-08-11
tags: [paper, gravitational-waves, writing]
---

# GW observables section

Candidate section for the [[combustion-paper]]: "Semi-analytic gravitational-wave criticalities from finite-rate phase conversion". Working file is `gw_section_standalone.tex` — **`gw_section_body.tex` is stale.**

## Settled notation (do not re-litigate)

| Symbol | Meaning | Rejected alternative |
|---|---|---|
| $X_q$ | converted fraction | $x_q$ — lowercase $x$ is the coordinate |
| $Y_q$ | indicator field | $\chi_q$, $\xi$ — susceptibility / RMF-coupling connotations; $\xi$ is already the sharpness parameter |
| $\Delta f_{\rm peak}$ | finite-rate deficit | $\Delta f_{\rm rate}$ |

The tex label `eq:rate_shift` was kept despite the rename.

Structure: a single `\section` with three `\textbf{}` run-in paragraph leads (Converted fraction / Catch-up / Peak-frequency shift), no subsections. Prose passed through the humanizer skill.

## The geometry — pre-existing cores

Adopted 2026-07-15. Each inspiraling star **already has a quark core**; the front only advances it by $\Delta R = u\,\delta t_{\rm pm} \sim 10$ m over the postmerger window.

$$X_q = \min\left\{1,\ \left(\frac{R_0 + u\,\delta t_{\rm pm}}{R_{\rm lobe}}\right)^3\right\},\qquad R_0 = 1\ \mathrm{km},\ R_{\rm lobe} = 3\ \mathrm{km}.$$

Baseline $(R_0/R_{\rm lobe})^3 = 0.037$, deficit $\approx 0.96\,\Delta f_{\rm PT}$, front contributes $\sim10^{-3}$. For the fiducial $\delta t_{\rm pm}=10$ ms window, saturation is $u_{\rm sat} = (R_{\rm lobe}-R_0)/\delta t_{\rm pm} = 2\times10^5$ m/s.

**The physics punchline: conversion is frozen.** The 0.96 deficit reflects core *size*, not front speed.

Do not conflate two geometric thresholds. The displayed spherical-lobe saturation above asks when a radius reaches $R_{\rm lobe}$. Reaching the far tip of the full taiji-shaped critical region instead uses $L_{\max}\approx2R_{\rm lobe}-R_0$ and requires roughly $3\times10^5$--$10^6$ m/s for a 5--20 ms window. Both dwarf the calculated laminar speeds, but they answer different questions and need different labels.

## The honest-attribution result

The linear law $\Delta f_{\rm peak} = (1-X_q)\Delta f_{\rm PT}$ was investigated on 2026-07-16 and confirmed **algebraically correct** — $X_q$ is the total quark fraction, which handles pre-existing cores, and endpoints match hadronic-vs-hybrid simulations. What was wrong was the *justification*: "small converted volume changes the local EoS" was replaced by "replacing nuclear by quark matter alters the local pressure–density relation feeding $K_2/I_2$".

A mode-weighting caveat is stated: central cores weigh more in the $m=2$ bar, so $X_{q,0}\Delta f_{\rm PT}$ is a lower estimate.

## ⚠ Six cases, two sources — the recurring citation error

Originally seven. The Bauswein $q=0.8$ case ($\Delta f_{\rm PT} = 0.780$ kHz) was **dropped** 2026-07-16: Prakash Table IV has no asymmetric DD2F/DD2F-SF1 row to substitute, and the Bauswein table is not extractable via WebFetch or pdftotext.

**Of the six remaining, five are from Prakash:2023afe Table IV, but the sixth — DD2F/DD2F-SF1 at 1.35+1.35, $\Delta f_{\rm PT} = 0.442$ — is from Bauswein:2018bma and is not in Prakash's table at all.** This was gotten wrong once by assuming Bauswein was no longer a data source. Both citations belong on the data sentences. Bauswein is *additionally* kept in three general-literature sentences.

Use the **"Injected"** column of Table IV (the true model value), not "Recovered" (carries Bayesian uncertainty). Prakash confirms the onset density $\gtrsim 3\rho_{\rm nuc}$, matching $n_B^{\rm crit} = 3n_0$, but reports **no** quark-core radius or volume fraction — so per-case geometry is not sourceable and one fiducial geometry is applied to all six.

## Observability

$\delta f_{\min} \sim 0.1$ kHz adopted for a loud detection. Breschi:2022ens (ET): $f_2$ errors of order 1 kHz at postmerger SNR 7, below 100 Hz at SNR 10 — **that SNR statement is Breschi's alone**. Clark:2015zxa: Fisher estimate ~50 Hz at ~30 Mpc with aLIGO, **no SNR quoted** (an earlier Consensus-indexed abstract said 138 Hz Monte Carlo; the arXiv version differs — check the published CQG before submission).

All six deficits are resolvable at SNR $\gtrsim 10$; none near threshold. Resolution scaling near threshold is much steeper than $1/\rho$.

## ⚠ Where the figure actually lives

**The $f_{\rm peak}$ figure is tuned in cell 18 of `All_Plots_new.ipynb`, NOT in `plot_f_peak.py`.** The `.py` is stale and divergent (band, annotation and output paths all differ). Edit the notebook cell — via `nbsrc.py`, never a raw read. The cell saves PDF only; any `.png` in `Plots/` is stale.

All figure scripts write to **both** `Plots` directories (`v2/Plots` and `Latex_writting/Hydrodynamical combustion/Plots`).

## Figure design brief (user-approved)

All text ≥ 10 pt at 1:1 columnwidth. No boxed legend. Color = EOS family (blue BLh/BLQ, vermillion DD2F/DD2F-SF1), shades and linestyles within family. Tags restacked at $x = 1.3\times10^1$ in vertical gaps with leader lines — no text on curves. Dotted $\delta f_{\min} = 0.1$ kHz guide. Light geometry band spanning $u_{\rm sat}$ for $R_{\rm lobe} = 2$–4 km and $\delta t_{\rm pm} = 5$–20 ms. $y$-grid only. The fall-off is common **by construction** (geometry-only, EOS-independent) and the caption says so.

## Taiji illustrations

2D and 3D versions of the $\Omega_{\rm crit}$ schematic exist, cores at true 1:3 ratio. **Neither is referenced by the tex yet.**

3D lesson worth keeping: mplot3d ignores `zorder` by default and mis-sorts a sphere intersecting a large disk. Fix was `ax.computed_zorder=False` + manual zorder + drawing only the upper dome plus a flat equatorial slice, with manual Lambertian facecolors (`shade=False`) to keep the vermillion.

## Verified against the solver

⚠ **Stale — recompute before use.** This section rested on `analytic_velocity_bound` at $n_B = 3n_{\rm sat}$, $B^{1/4} = 180$ MeV, QMC-RMF3 giving $u_N \approx 196/202/228$ m/s at $T = 5/10/20$ MeV, hence realistic $X_q \sim 4\times10^{-6}$ under the old spherical model (the text's $10^{-4}$ being the conservative top-of-band corner).

Re-measured 2026-08-17, that triple does not reproduce under any current code path — the full analytical bound at $T(0^+)=5$ MeV gives 143.93/147.82/163.23 m/s, roughly 27% lower; see [[phase-velocity-overview]] for the full table. **$X_q$ has not been recomputed**, and which variant the section should quote is a physics choice (the three entry points now differ by up to 30%), so it is left for the author rather than silently rescaled.

## Merging

`\input` the body after `sec:results`; append the new-entry block of `gw_refs.bib` to `reflist.bib`. Recompile with `latexmk -pdf gw_section_standalone.tex`. Per project convention, bib entries come from INSPIRE-HEP verbatim.
