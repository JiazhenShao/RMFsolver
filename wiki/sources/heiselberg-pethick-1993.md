---
summary: Heiselberg & Pethick 1993, PRD 48 2916 — the transport source for both kappa_th and D_K. Paywalled, no arXiv preprint.
status: current
updated: 2026-08-11
tags: [source, transport]
---

# Heiselberg & Pethick 1993

**"Transport and relaxation in degenerate quark plasmas"**, Physical Review D **48**, 2916 (1993). DOI `10.1103/PhysRevD.48.2916`. INSPIRE key `Heiselberg:1993cr`.

## Access

**No arXiv preprint. Paywalled at APS.** The PDF is held locally at `~/Documents/GRAD_STUDY/Research/Dense_Matter/ref[35]-D_Q.pdf` — this is the only reachable copy, and an earlier attempt to build the solver without it stalled on Eq. (62) being unrecoverable.

## What it supplies

Both the quark thermal conductivity $\kappa_T$ and the flavor diffusion coefficient $D_K$ used throughout [[quark-transport]].

Equations that matter: **Eq. (7)** susceptibilities · **Eq. (8)** fixes the log branch · **Eq. (21)** momentum-relaxation integrand · **Eq. (59)** the $I_\kappa$ integral · **Eq. (60)** its low-$y$ asymptote · **Eq. (62)** · **Appendix B** the analytic-vs-numerical constant gap.

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
