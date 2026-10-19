---
summary: The tail ratio has no order-unity crossing; a separate core criterion at f=1e-4 now has 33 validated roots from 0.515 keV to 0.172 MeV in 26-10-14.ipynb, not a sharp LTE boundary.
status: current
updated: 2026-10-14
tags: [method, isothermal, visualization, transport, validity]
---

# Isothermal mean-free-path / downstream-tail overlay

`26-10-14.ipynb` retains the isothermal saved-data figure from `26-10-02.ipynb` and adds a separate tunable calculation of $\lambda_{\mathrm{mfp}}=fL_{\mathrm{conv}}$.
`MFP_FRACTION=1.0` is the default; `0.1` tests ten percent.
Both are far above the estimated ratio in the plotted domain, so a missing overlay is the expected physical result, not a plotting failure.
The original notebook, contour/numerical datasets, and solver source are unchanged.

## Physical definition

Use the current manuscript `Latex_writting/Main.tex`, `eq:LTE_condition`, `eq:conversion-thickness`, `eq:diffusion_coefficient`, and `eq:cubic_rate`:

$$
\lambda_{\mathrm{mfp}}=3D_K,\qquad
L_{\mathrm{conv}}=\frac{\mu_B(\infty)}{9\pi T}\sqrt{\frac{D_K}{\gamma_K}}.
$$

Both lengths are evaluated at the **static pressure-matched equilibrated quark endpoint**, with common temperature, $P(\infty)=P(0^-)$, $\mu_K(\infty)=0$, and zero baryon flux.
Recover the nuclear state directly from its supplied density using `_validated_pnm_state_from_nB`, then solve the equilibrated quark pressure equation.
This is the constant-background tail estimate, not a finite-flux front reconstruction.
Full diffusion is the default; the optional leading-Landau approximation is applied consistently to both lengths.
The code uses the actual coefficient expression, not the rounded $0.2\,\mu\mathrm{m}$ normalization.

$$
\frac{\lambda_{\mathrm{mfp}}}{L_{\mathrm{conv}}}
=\frac{27\pi T}{\mu_B(\infty)}\sqrt{D_K\gamma_K}.
$$

For fixed $\mu_B(\infty)$, $D_K\propto T^{-5/3}$ in the Landau limit, so the ratio tends to zero as $T^{1/6}$, while $L_{\mathrm{conv}}\propto T^{-11/6}$ diverges faster than the mean free path.
Both lengths are initially in $\mathrm{MeV}^{-1}$ and convert to meters using $197.326960\times10^{-15}$.
The ratio identity and scaling were symbolically checked using Wolfram, with faithful input:

```text
FullSimplify[{3 dk/(muB/(9 Pi temp) Sqrt[dk/gammaK]),
 (3 dk/(muB/(9 Pi temp) Sqrt[dk/gammaK])) /. dk -> cD temp^(-5/3)},
 Assumptions -> {dk>0,muB>0,temp>0,gammaK>0,cD>0}]
```

⚠ This is **not** the appendix's local diffusion-gradient condition $\lambda_{\mathrm{mfp}}<D_K/|dD_K/dx|$ and must not be described as a general LTE-validity boundary.
See [[lte-composition-bound]] and [[quark-transport]].

## Why this cannot establish a sub-keV cutoff

The $T^{1/6}$ scaling is an asymptotic trend with a dimensional prefactor, not an order-unity temperature threshold or a justified high-temperature exclusion. It must not be extrapolated beyond the degenerate transport approximation.
The downstream thermal tail is not necessarily the shortest gradient scale in the front.
In the finite-composition nonlinear core, with $a^2\gg\eta$, balancing diffusion against the cubic source gives the separate **order-of-magnitude** scale

$$
L_{\mathrm{core}}\sim\frac{1}{a}\sqrt{\frac{D_K}{\gamma_K}},\qquad
\frac{\lambda_{\mathrm{mfp}}}{L_{\mathrm{core}}}\sim3a\sqrt{D_K\gamma_K}\propto T^{-5/6}.
$$

Here $a=n_K/n_B$ remains the established local fraction. The scaling holds at fixed nonzero composition and chemical potential with the assumed Landau transport law; advection and the actual profile change the numerical coefficient and composition dependence.
Unlike the tail ratio, this core diagnostic can grow toward breakdown as temperature decreases. It does **not** establish a universal boundary at $1$ keV.
For an illustrative frozen state at $\mu_B=1200$ MeV, $a=0.5$, and $T=0.001$ MeV, the full diffusion expression gives $\lambda_{\mathrm{mfp}}=2.09665\,\mu\mathrm{m}$ and $L_{\mathrm{core}}\sim0.518433$ mm, hence ratio $0.0040442$; this is coefficient evaluation, not a validated numerical front.

A useful proposed **diffusion-gradient diagnostic**, not a sufficient LTE proof, is $\lambda_{\mathrm{mfp}}|dn_K/dx|/|n_K|$, checked throughout the quark-side profile together with other relevant hydrodynamic gradients.
In the implemented slow-front Fick closure, current continuity for PNM gives $D_K n_K'(0^+)=j_B[a(0^+)-1]$, hence its interface value is $3u(0^+)[1-a(0^+)]/a(0^+)$ in natural units.
The full maximum need not occur at the interface, and extrapolating the slow-front formula beyond its speed regime is not valid.
The general need for separation of microscopic and gradient scales is discussed by [Niemi and Denicol](https://arxiv.org/abs/1404.7327); the choice above is a proposed application to this conversion model, not a bound derived in that paper.

Wolfram verified the algebra and cold scaling with:

```text
FullSimplify[{3 dk/Sqrt[dk/(gammaK a^2)],
 (3 dk/Sqrt[dk/(gammaK a^2)]) /. dk -> cD temp^(-5/3),
 3 dk Abs[u (a - 1)/dk]/a},
 Assumptions -> {dk>0,gammaK>0,cD>0,temp>0,0<a<1,u>0}]
```

The returned expressions were `3*a*Sqrt[dk*gammaK]`, `3*a*Sqrt[cD*gammaK]/temp^(5/6)`, and `3*u*(1-a)/a`.
The original tail-only implementation has now been extended with a **separate standalone core figure** below; a full profile-gradient diagnostic remains unimplemented.

## Implemented standalone core curve

The final two code cells of `26-10-14.ipynb` independently calculate/resume and plot the core condition $3a(0^+)\sqrt{D_K(0^+)\gamma_K(0^+)}=10^{-4}$.
All coefficients use the finite-composition interface, not the equilibrated endpoint.
The static maximum-composition closure matches upstream pressure and satisfies $\mu_B(0^+)+a(0^+)\mu_K(0^+)=\mu_B(0^-)$.
The fraction, density samples, temperature search limits, diffusion model, workers, and tolerances are exposed; no numerical BVP or moving-speed solve is used.

The first run found **33/33 roots** using 226 EOS evaluations and two single-threaded workers.
The sampled density ratios span $3.000099948$–$5.295928904$, with logarithmically denser samples near coexistence.
The lowest sampled root is $T=0.0005150551$ MeV ($0.515$ keV); the sampled maximum is $0.1715029$ MeV near density ratio $4.914$.
These are sampled extrema, not universal limiting temperatures.
Maximum relative criterion residual is $5.63\times10^{-7}$, pressure/composition residuals are below $3\times10^{-13}$, and the Gibbs residual is below $4.1\times10^{-11}$ MeV.
Maximum $\eta/a(0^+)^2$ is $1.93\times10^{-5}$, consistent with the assumed cubic-core regime.
One accepted root retained a quadrature warning; validated upstream residuals remained small and the warning is preserved in the payload.

`new_paper_calculations/26-10-14/isothermal_core_mfp.py` brackets/corrects in log temperature with live EOS checks; the cold power law only proposes a guess.
`local_analysis_results/26-10-14/mfp-core-f1e-4.npy` stores roots and their trial evaluations, with source/physics/control compatibility checks.
Every completed density point is atomically saved before tqdm advances; a matching resume performs no new EOS work.
The standalone PNG/PDF `isothermal-core-mfp-f1e-4` plots only this curve over $10^{-4}\leq T\leq1$ MeV, and unsuccessful samples would break connecting segments.
The earlier tail results and contour figure are retained.

The standalone core figure also shades stable neutron matter and $a(0^+)\geq1$ across its full window.
`isothermal_core_regions.py` reuses 19 saved boundary slices and directly solves five missing ones: four below $0.01$ MeV and the $1$-MeV endpoint.
All 24 slices passed; maximum new chemical-potential residual is $1.01\times10^{-9}$ MeV, with no new quadrature warnings.
`core-static-regions.npy` is separate from the unchanged 33 core roots, atomically checkpointed before progress, and does not extrapolate the original $0.01$-MeV boundary downwards.
The upper boundary uses the existing $a(0^+)=1-10^{-9}$ convention; failures would break shading connectivity.

⚠ Below this curve the core estimate exceeds the **chosen stringent tolerance** $10^{-4}$; it does not establish physical invalidity there, nor does the scale estimate incorporate actual profile/advection corrections.

## Notebook and resume contract

The first three code cells load saved contours/dots, calculate the tail-ratio crossings, and plot. The final two handle the separate core curve. Plotting makes no solver calls.
Helpers live in `new_paper_calculations/26-10-14/`; its README documents controls and saved-result loading.
Defaults preserve the existing PNM, massless-strange-quark, bag constant, 46-temperature schedule, and physical window $1\leq n_B(0^-)/n_0\leq5.5$, $0.01\leq T\leq120$ MeV.
The speed figure retains $\xi=-0.85$; the static length estimate itself has no $\xi$ dependence.
The search tests the saved allowed density interval endpoints at each temperature, with optional extra interior samples, and solves only an actual ratio bracket.
Roots must satisfy relative ratio error at most $10^{-6}$ and validated EOS pressure/density residuals.

`local_analysis_results/26-10-14/mfp-static-tail/thermodynamics.npy` caches both diffusion choices independently of the fraction.
Per-fraction/model/control result files retain no-crossing, outside-domain, failed, interrupted, and budget-exhausted slices separately.
Source, input, and domain fingerprints prevent incompatible resume.
Reservations and finished EOS evaluations are saved atomically before progress increments; the temperature bar advances only after its root/status is saved.
Terminal failures do not automatically retry; use a new run directory for that explicit operation.
One bounded subprocess and one BLAS thread keep local execution modest. No numerical BVP or nuclear chemical-potential scan is used.
The existing isolated SciPy runtime is reused; no dependency installation or solver-source change is needed.

Black dash-dot segments connect only consecutive successful temperature slices. A singleton is a marker; missing or failed temperatures break the overlay.
No validated roots means no new curve and no overlay legend.
All visual controls are exposed at the top of the plotting cell, and the original curves, gray regions, fonts, and log-temperature window are preserved.
See [[isothermal-high-speed-contours]] and [[isothermal-notebook-cold-points]] for saved-data provenance.

## Bounded verification

Six live EOS evaluations checked both allowed density endpoints at $T=0.01$, $20$, and $83.8592543$ MeV.
All passed; maximum relative pressure residual was $6.995\times10^{-13}$.
The sampled full-diffusion ratio range was $2.72603\times10^{-7}$–$1.11485\times10^{-6}$; the Landau-only range was $2.78781\times10^{-7}$–$1.38497\times10^{-6}$.
Neither $f=1$ nor $f=0.1$ crossed on these three slices. Changing the fraction/model reused all six states without new EOS work.
These are bounded sample checks, **not a completed 46-slice domain calculation**; that assessment remains reproducible through the notebook's calculation cell.
The six states seed its cache. The embedded preview is saved-data-only.
Tests cover formula/units/scaling, actual bracketed synthetic roots, cache/resume/failures, interrupted reservations, save-before-progress, and missing-root gaps.
Exact artist-array comparisons preserve the original analytical lines, numerical curves/dots, and shading; hashes verify unchanged original inputs and solver sources.
