---
summary: Free thermal gluons are included in directly relevant finite-temperature combustion bag EOSs, but not universally in bag-model extensions; the current solver includes them consistently.
status: current
updated: 2026-10-08
tags: [physics, equation-of-state, bag-model, literature, gluons]
---

# Thermal gluons in bag-model EOSs

Keeping the free thermal-gluon term is an established choice for the project's unpaired, massless, finite-temperature ideal bag EOS. Directly relevant combustion papers use exactly its combined thermal coefficient. This is not a universal convention for everything called a bag model: cold models have zero free thermal-gluon pressure, and some finite-temperature effective extensions have no separate ideal-gluon term. A targeted literature audit establishes these practices, not a statistically measured majority or consensus about quantitative QCD accuracy.

## Checked literature

Consensus was used for discovery and paper-record retrieval, followed by inspection of primary-paper EOS equations. Abstracts alone were not used to determine whether the term is present.

### Reviews cited in Appendix A

The opening pressure paragraph of `Latex_writting/Main.tex`, Appendix A, now cites `Madsen:1998uh,Buballa:2003qv` (2026-10-08). These reviews are stronger foundational references for that statement than the nucleon-pressure example or individual combustion calculations.

- Jes Madsen, *Physics and astrophysics of strange quark matter*, Lect. Notes Phys. **516**, 162-203 (1999), DOI [10.1007/BFb0107314](https://doi.org/10.1007/BFb0107314), [full text](https://arxiv.org/pdf/astro-ph/9809032). Section 3.4, Eqs. (62)-(64), gives the massless per-flavor quark potential, net density and explicit gluon potential. Eq. (65) includes the three-flavor $19\pi^2T^4/36$ term and bag energy. The text after Eq. (72) explicitly permits the bulk-SQM limit by setting the curvature term to zero. Thus this is a direct match, not merely a paper mentioning gluons. The same equations also support the exact massless finite-temperature formulas used in Appendix A.
- Michael Buballa, *NJL-model analysis of dense quark matter*, Phys. Rept. **407**, 205-376 (2005), DOI [10.1016/j.physrep.2004.11.004](https://doi.org/10.1016/j.physrep.2004.11.004), [full text](https://arxiv.org/pdf/hep-ph/0402234). Despite its NJL title, Sec. 2.1.2 reviews MIT-bag thermodynamics: Eqs. (2.4)-(2.6) explicitly separate bag energy, free quarks/antiquarks/gluons and perturbative corrections. Eq. (2.6) sums over arbitrary flavors and their chemical potentials; the evaluated illustration in Eq. (2.7) is two-flavor and therefore has a different combined $T^4$ coefficient. Do not cite that two-flavor coefficient as the three-flavor result.

INSPIRE returned 1752 citations for [Buballa](https://inspirehep.net/literature/637058) and 246 for [Madsen](https://inspirehep.net/literature/476574) on 2026-10-08. These are database-specific counts, not universal totals or proof of unique authority. Buballa's new BibTeX entry was imported verbatim from INSPIRE; its exported title uses the older wording "NJL model analysis of quark matter at large density". The existing Madsen entry was reused. The paragraph wording and author comments were preserved apart from joining its source line and adding the citation with punctuation.

### Application examples

| Paper | What its equations contain |
|---|---|
| Niebergal, Ouyed and Jaikumar (2010), *Numerical Simulation of the Hydrodynamical Combustion to Strange Quark Matter*, PRC 82, 062801; [full text](https://arxiv.org/pdf/1008.4806), DOI [10.1103/PhysRevC.82.062801](https://doi.org/10.1103/PhysRevC.82.062801) | Eq. (5): enthalpy has $19\pi^2T^4/9$ and $P=h/4-B$. The resulting pressure coefficient $19/36$ includes free thermal gluons. |
| Herzog and Ropke (2011), *Three-dimensional hydrodynamic simulations of the combustion of a neutron star into a quark star*, PRD 84, 083002; [full text](https://arxiv.org/pdf/1109.0539), DOI [10.1103/PhysRevD.84.083002](https://doi.org/10.1103/PhysRevD.84.083002) | Eqs. (1)-(3): pressure explicitly has $19\pi^2T^4/36$, energy has $19\pi^2T^4/12$. Includes thermal gluons despite introductory prose describing ideal quark gases. |
| Matias Astorga and Herrera-Corral (2024), *Pressure Distribution Inside Nucleons in a Tsallis-MIT Bag Model*, Entropy 26, 183; [article](https://www.mdpi.com/1099-4300/26/3/183), [full-text mirror](https://inspirehep.net/files/f888ae100a8f373aa7a0ebbe456fb634), DOI [10.3390/e26030183](https://doi.org/10.3390/e26030183) | Standard MIT baseline Eqs. (3), (5), (6) explicitly adds $P_G=16\pi^2T^4/90$ and subtracts $B$. The paper's main topic is internal nucleon pressure and a Tsallis extension, not bulk dense stellar matter. |
| Lopes, Biesdorf, Marquez and Menezes (2021), *Modified MIT Bag Models - part II: QCD phase diagram and hot quark stars*, Physica Scripta 96; [author preprint](https://arxiv.org/pdf/2009.13552), DOI [10.1088/1402-4896/abef35](https://doi.org/10.1088/1402-4896/abef35) | Preprint Eqs. (5)-(10): quark Fermi integrals plus bag and vector-field terms, no separate free thermal-gluon pressure. Classifies the displayed model, not an inferred justification for omission. |
| Klahn, Fischer, Cierniak and Hempel (2017), *Phase Diagram of (Proto)Neutron Star Matter in an Extended Bag Model*, JPCS 861, 012026; [full text](https://s3.cern.ch/inspire-prod-files-8/897fe95aead80683d3c91b897c4c7005), DOI [10.1088/1742-6596/861/1/012026](https://doi.org/10.1088/1742-6596/861/1/012026) | Eqs. (11)-(13): quark Fermi-gas pressure, vector term and chiral/deconfinement bag terms, without a separate ideal thermal-gluon gas. Effective gluonic interactions are not the same thing as thermal gluon particles. |
| Alford, Braby, Paris and Reddy (2005), *Hybrid stars that masquerade as neutron stars*, ApJ 629, 969; [full text](https://arxiv.org/pdf/nucl-th/0411016), DOI [10.1086/430902](https://doi.org/10.1086/430902) | Eqs. (1.1), (3.1)-(3.2): cold pressure includes the phenomenological QCD correction $a_4=1-c$ multiplying the free Fermi-sea term. This is not a thermal $T^4$ contribution. |

## Distinct meanings of gluon contributions

- Thermal free gluons: eight colors and two transverse polarizations, zero equilibrium gluon chemical potential. This gives $P_g=8\pi^2T^4/45$ and vanishes at zero temperature.
- Gluon-mediated interactions: corrections to quark thermodynamics, including effects at zero temperature, sometimes parameterized by $\alpha_s$, $a_4$ or effective vector interactions. Adding $P_g$ does not add these corrections.
- Vacuum contribution: the bag energy $U_{\rm bag}$. In the ideal bag construction it is distinct from the thermal gas, so including both is not double counting. A fitted, medium-dependent effective model requires its own accounting; do not indiscriminately add an ideal gas to it.

## Ideal model and scale estimate

In natural units, with massless quarks and no leptons,

$$P_{\rm QM}=-U_{\rm bag}+\sum_{i=u,d,s}\left[\frac{\mu_i^4}{4\pi^2}+\frac{\mu_i^2T^2}{2}+\frac{7\pi^2T^4}{60}\right]+\frac{8\pi^2T^4}{45}.$$

Thus the pure $T^4$ coefficient is $7/20$ for quarks alone and $19/36$ for quarks plus gluons. Independently,

$$s_g=\frac{32\pi^2T^3}{45},\qquad \epsilon_g=\frac{8\pi^2T^4}{15}=3P_g.$$

At $\mu_K=0$ in the massless neutral model, $\mu_i=\mu_B/3$. Comparing gluon pressure to the leading fixed-chemical-potential quark thermal correction gives

$$\frac{P_g}{\mu_B^2T^2/6}=\frac{16\pi^2}{15}\left(\frac{T}{\mu_B}\right)^2.$$

At $\mu_B=1200$ MeV this is $0.0731\%$ at $T=10$ MeV, $1.8277\%$ at 50 MeV, and $7.3108\%$ at 100 MeV. These are fractions of the leading thermal correction, not fractions of the net pressure or errors in front speed. The net pressure may be sensitive to cancellation against the bag term; a velocity error requires solving the front problem.

Wolfram verified the coefficient, derivatives and numerical ratios on 2026-10-08. Essential inputs were:

```text
3*7/60 + 8/45
D[8*Pi^2*T^4/45, T]
Simplify[T*D[8*Pi^2*T^4/45,T] - 8*Pi^2*T^4/45]
FullSimplify[(8*Pi^2*T^4/45)/(muB^2*T^2/6)]
```

## Current project implications

`RMFsolver/SQMsolver.py` defines `_gauge_pressure`, `_gauge_energy` and `_gauge_entropy` at lines 91-110 with exactly the ideal-gluon values. `PQM` includes the pressure at line 682; `edensQM` includes the energy at line 727 even with `include_em=False`; `entropyQM` includes entropy at line 765. The flag chooses the electron-inclusive branch, not whether gluons are present. The gauge-helper docstrings mention the electromagnetic branch but are narrower than their actual call sites. No source code changed during this audit.

At fixed $\mu_B,\mu_K,T$, adding this chemical-potential-independent term changes neither neutrality nor $n_B,n_K$, susceptibilities or $a(\mu_K)$. It does change pressure, energy and entropy, hence can indirectly change reconstructed thermodynamic profiles and numerical front speeds. The massless identity $\epsilon=3P+4U_{\rm bag}$ holds with or without the free-gluon term.

The quadratic small-$T$, small-$\mu_K$ EOS in [[quark-matter-eos]] and its corresponding leading analytical velocity formula do not acquire a gluon term: $P_g$ is fourth order in temperature. A consistent truncation may omit it along with other fourth-order terms. For the untruncated finite-temperature ideal EOS, keeping it matches the cited combustion papers and the current code. See [[charge-neutral-quark-eos]] for the neutral expansion.
