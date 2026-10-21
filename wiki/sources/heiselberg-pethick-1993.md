---
summary: Heiselberg & Pethick 1993, PRD 48 2916 — the transport source for both kappa_th and D_K; its Eq. (25) h_D carries a /6 the manuscripts had dropped. Paywalled, no arXiv preprint.
status: current
updated: 2026-10-21
tags: [source, transport]
---

# Heiselberg & Pethick 1993

**"Transport and relaxation in degenerate quark plasmas"**, Physical Review D **48**, 2916 (1993). DOI `10.1103/PhysRevD.48.2916`. INSPIRE key `Heiselberg:1993cr`.

## Access

**No arXiv preprint. Paywalled at APS.** The PDF is held locally at `~/Documents/GRAD_STUDY/Research/Dense_Matter/ref[35]-D_Q.pdf` — this is the only reachable copy, and an earlier attempt to build the solver without it stalled on Eq. (62) being unrecoverable.
`pdftoppm` is not installed, so `Read` cannot render it; a short Swift script using `PDFKit` extracts page text and PNGs.

## What it supplies

Both the quark thermal conductivity $\kappa_T$ and the flavor diffusion coefficient $D_K$ used throughout [[quark-transport]].

Equations that matter: **Eq. (7)** susceptibilities · **Eq. (8)** fixes the log branch · **Eq. (21)** momentum-relaxation integrand · **Eq. (59)** the $I_\kappa$ integral · **Eq. (60)** its low-$y$ asymptote · **Eq. (62)** · **Appendix B** the analytic-vs-numerical constant gap.

## ⚠ Flavor diffusion: the $h_D$ prefactor has a $/6$

HP93 Eq. (25) defines the low-$T$ transverse coefficient (our $h_D$) as

$$h_D=\frac{\Gamma(\tfrac{8}{3})\,\zeta(\tfrac{5}{3})\,(2\pi)^{2/3}}{6}=1.8131745\ldots$$

Without the $/6$ the expression evaluates to $10.879$.
Until 2026-10-21 the manuscripts (`Latex_writting/Main.tex`, `Hydrodynamical combustion/Main.tex` and `Notes.tex`, `Latex_drop_in_integrated/section.tex`) printed it without the $/6$ beside "$\simeq1.81$"; the $/6$ has been restored there.
The code was never affected: `_TRANSPORT_H_CONST = 1.81317` is hardcoded.
The value was rederived from HP93 Eqs. (20) and (23), with transverse term $\frac{\pi}{6}(4\omega^2/\pi q_D^2)^{1/3}$ and the identity $\int_0^\infty s^{5/3}\,\frac{e^{-s}}{(1-e^{-s})^2}\,ds=\Gamma(\tfrac83)\zeta(\tfrac53)$; quadrature agrees to $10^{-20}$.
The longitudinal coefficient $\pi^3/12$ checks the same way.

The rest of the chain matches the code and manuscripts:
Eq. (28) $\tau_s^{-1}=\frac{8N_q}{3\pi}\alpha_s^2\left[h_D T^{5/3}/q_D^{2/3}+\pi^3T^2/(12q_D)\right]$ for $T\ll q_D$, Eq. (36) $D=\tau_s/3$, and $q_D^2=g^2N_q\mu_q^2/(2\pi^2)$ (p. 2918).
Together these give $D_K^{-1}=\frac{8N_f\alpha_s^2}{\pi}[\ldots]$, i.e. `_TRANSPORT_D_PREFACTOR = 24*alpha_s**2/pi` for $N_f=3$.

## ⚠ The misreading to avoid

It is tempting — and wrong — to assume $\kappa_T$ and $D_K$ share a single Landau-damped transport time $\tau \propto T^{-5/3}$. **They do not.** HP93 calculate different relaxation times for different processes, and in the low-$T$ degenerate regime the thermal-relaxation rate carries a different temperature dependence from the momentum/flavor rate.

The 2026-07-22 interface-jump spec was built on the shared-$\tau$ assumption and was **superseded on 2026-07-23** for exactly this reason.

## Numerical discrepancies found

- The quoted high-$y$ constant **0.30** appears to be an analytic estimate, not the numerical curve. Converged quadrature gives **0.388**, with the slope in $\ln y$ matching $1/3$ exactly. Our quadrature matches their Fig. 4.
- Appendix B documents the same kind of gap for $I_s$: 2.810 analytic vs 2.72 numerical.

Irrelevant for the solver ($y \lesssim 0.19$), relevant if the asymptote is quoted in [[combustion-paper]].

## Related sources

- **Shovkovy & Ellis 2002**, PRC **66**, 015802, DOI `10.1103/PhysRevC.66.015802` — assumes temperature continuity across a directly contacting nuclear–quark interface. Precedent for [[thermal-transparent-closure]].
- **Zhao & Patankar 2021**, Int. J. Heat Mass Transfer, DOI `10.1016/j.ijheatmasstransfer.2022.123389` — general phase-change reference on why an interface temperature can require an additional constitutive principle.
- See [[kapitza-resistance]] for the diffuse-mismatch and moving-front literature, none of which yields a number for this system.
