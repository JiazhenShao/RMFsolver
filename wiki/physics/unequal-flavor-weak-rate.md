---
summary: Madsen Eq. (23) is a near-equal-flavor approximation; massless zero-temperature phase space gives rates below it by 6.8% at a=0.146 and 64.1% at a=0.9.
status: current
updated: 2026-10-07
tags: [physics, weak-rates, phase-space, verification]
---

# Weak rates with unequal flavor chemical potentials

For the numerical front, the reference calculation should be the full nonleptonic collision integral with the local neutral-EOS chemical potentials, not the polynomial currently evaluated by `_exact_kaon_transport_rate`.
Equation (23) remains a possible leading-order analytical approximation near equal flavor chemical potentials; unequal potentials alone do not invalidate a controlled expansion, but inserting the exact value of mu_u does not supply the missing phase-space corrections.
See [[strangeness-reaction-diffusion]] and [[quark-matter-eos]].

## What was verified from the literature

The indexed original Madsen paper, Phys. Rev. D 47, 325 (1993), describes the general integral in Eq. (17) and the net rate in Eq. (8).
Its Section IV introduces Eq. (23) for low temperature, equal up/down potentials and small strange/down splitting.
For massless quarks it recommends the appropriate exact zero-temperature branch, Eq. (20) or (21), plus the thermal term of Eq. (23) multiplied by Eq. (24).
Those are improved low-temperature approximations, not the full finite-temperature integral.
Madsen's forward direction is s -> d; our positive source is d -> s, so massless branch formulas require exchanging the down and strange labels.
Correction after direct PDF inspection on 2026-10-07: both printed branches (20) and (21) also require mu_u < mu_d in his original labels.
After exchanging labels this fails for our neutral ordering mu_s < mu_u < mu_d; neither printed branch is directly usable for our conversion process.
The earlier general branch recommendation above is superseded for our matter by [[massless-finite-temperature-weak-rate]].

Former access limitation, now resolved: the original PDF was initially unavailable for visual inspection and its indexed text omitted the displayed general-rate and correction formulas.
The user supplied `ref-list/Rate-PhysRevD.47.325.pdf`; pages 326--329 and Eq. (24) have now been visually checked, with the corrected conclusions in [[massless-finite-temperature-weak-rate]].
The invariant collision integral and matrix element were independently read in Eqs. (8)--(11) of Zheng, Zhou and Liu, arXiv:astro-ph/0507390.
Their later magnetic-field approximations are not used here.

Sources: [Madsen article](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.47.325), [indexed original](https://electronicsandbooks.com/edt/manual/Magazine/P/Physical%20Review%20D/PhysRevD%201982-1994/root/data/PhysRevD%201982-1994/pdf/PRD/v47/i1/PRD_v47_i01_p0325_1.pdf), [Zheng et al. PDF](https://arxiv.org/pdf/astro-ph/0507390).

## Independently reduced massless integral

The following reduction was derived in this session, not transcribed from Madsen's Eq. (17).
It assumes massless on-shell quarks, no magnetic field, Fermi-Dirac occupations at one temperature, and the tree-level four-quark weak interaction without QCD corrections or additional antiquark channels.
Let p1 and p2 label the incoming and outgoing up-quark momentum magnitudes, and pd and ps the down and strange magnitudes.
Using the (+---) convention, the published spin convention gives

```text
R(d+u -> s+u) = (36/2) * integral[product_i d^3p_i/((2*pi)^3*2*E_i)]
               * (2*pi)^4*delta4(P1+Pd-P2-Ps)
               * |M|^2*f_u(E1)*f_d(Ed)*(1-f_u(E2))*(1-f_s(Es))
|M|^2 = 64*G_F^2*sin(theta_C)^2*cos(theta_C)^2*(P1.Pd)*(P2.Ps)
f_i(E) = 1/(1+exp((E-mu_i)/T))
Gamma_K = R(d+u -> s+u) - R(s+u -> d+u)
        = (1-exp(-mu_K/T))*R(d+u -> s+u), T>0
mu_K = mu_d-mu_s
```

Wolfram verified the detailed-balance sign directly from the Fermi factors and energy conservation.
There is one unit of K relaxation per reaction because n_K=(n_d-n_s)/2.

The common total spatial momentum magnitude k reduces the angular integral exactly:

```text
E = p1+pd = p2+ps
kmin = max(abs(p1-pd),abs(p2-ps))
integral[product_i dOmega_i]*delta3(p1+pd-p2-ps)
    = 16*pi^3/(p1*pd*p2*ps) * integral_{kmin}^E dk
(P1.Pd)*(P2.Ps) = (E^2-k^2)^2/4

R(d+u -> s+u) = 9*G_F^2*sin(theta_C)^2*cos(theta_C)^2/(8*pi^5)
    * integral_0^infinity dp1 integral_0^infinity dpd integral_0^(p1+pd) dp2
    * [8*E^5/15 - E^4*kmin + 2*E^2*kmin^3/3 - kmin^5/5]
    * f_u(p1)*f_d(pd)*(1-f_u(p2))*(1-f_s(E-p2))
```

The symbols E and k are integration variables needed for the angular reduction, not new thermodynamic parameters.
Setting the angular kernel to its common-Fermi-surface value gives 256*mu_u^5/15.
The zero-temperature excitation simplex has volume mu_K^3/6.
Together with the prefactor 9/8, this reproduces the manuscript's 16/5 coefficient.

Wolfram inputs for the kernel and normalization were

```wolfram
Integrate[(e^2-k^2)^2,{k,l,e}]
Simplify[(9/8) Integrate[((2mu)^2-k^2)^2,{k,0,2mu}] delta^3/6]
```

The outputs are the kernel above and 16*delta^3*mu^5/5.

## Neutral zero-temperature benchmark

Charge neutrality gives mu_d/mu_u=(1+a)^(1/3) and mu_s/mu_u=(1-a)^(1/3).
These results compare the full massless T=0 collision integral with Eq. (23) at T=0, both evaluated at the same chemical potentials.
The ratio is independent of the overall density scale.

| a | mu_d/mu_u | mu_s/mu_u | Full rate / Eq. (23) |
|---|---|---|---|
| 0.001 | 1.000333 | 0.999667 | 0.99957009 |
| 0.146 | 1.046473 | 0.948752 | 0.93249051 |
| 0.5 | 1.144714 | 0.793701 | 0.72499017 |
| 0.9 | 1.238562 | 0.464159 | 0.35942364 |
| 0.99 | 1.257818 | 0.215443 | 0.18968252 |
| 1 | 1.259921 | 0 | 0.10965830 |

Gauss-Legendre quadrature with 48, 96 and 192 nodes per coordinate agreed to better than 4e-7 in absolute rate ratio.
The integration used incoming holes h_u and h_d and outgoing excitations e_u and e_s, all nonnegative, satisfying h_u+h_d+e_u+e_s=mu_K.
With mu_u=1, the cube mapping was h_u=mu_K*x, h_d=mu_K*(1-x)*y, e_u=mu_K*(1-x)*(1-y)*z, e_s=mu_K*(1-x)*(1-y)*(1-z), with 0<=x<=min(1,1/mu_K), 0<=y,z<=1, and Jacobian mu_K^3*(1-x)^2*(1-y).
The upper bound on x is necessary when mu_K exceeds mu_u near a=1.

An independent exact integration at mu_s=0, with d=mu_d/mu_u between 1 and 2, gives

```text
Full rate / Eq. (23) = (65-32*d-84*d^2-112*d^3+210*d^4)/(896*d^3)
At d=2^(1/3): 0.1096583030099788
```

Here d is only a dimensionless integration-check variable.
Wolfram performed the exact polynomial integrations and confirmed this endpoint, independently of the numerical quadrature.
For reproduction, its input was

```wolfram
Clear[e,m,d];
inner0=Integrate[32m^2(e-m)^2(1-m)(e-1-m),{m,0,e-1}];
inner1=Integrate[32m^2(e-m)^2(1-e+d)(e-1-m),{m,0,e-d}]
       +Integrate[32m^2(e-m)^2(1-m)(e-1-m),{m,e-d,e-1}];
inner2=Integrate[32m^2(e-m)^2(1-e+d)(e-1-m),{m,0,e-d}]
       +Integrate[32m^2(e-m)^2(1-m)(e-1-m),{m,e-d,1}];
total=Factor[Integrate[inner0,{e,1,d}]+Integrate[inner1,{e,d,2}]
             +Integrate[inner2,{e,2,1+d}]];
{Factor[45total/(128d^3)],N[45total/(128d^3)/.d->2^(1/3),12]}
```

These are T=0 composition benchmarks, not errors measured along a finite-temperature front or predictions for the change in boundary speed.
Saved-profile finite-temperature validation remains open.
No manuscript, bibliography or solver changes were made.
