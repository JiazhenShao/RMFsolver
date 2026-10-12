# Cold-interface adiabatic verification

Executed through the connected Wolfram Language evaluator on 2026-10-11.
The computational symbols `muu,mud,mus` denote flavor chemical potentials; `tinf` denotes `T_inf`.
The input uses a single-flavor pressure for the density-constrained derivatives and the full quark-plus-gluon pressure for the conformal identity.

```wolfram
ClearAll[mu,t,nB,mud,mus,muu,tinf,z,cD,gamma0,bag,u,d,s];
p=mu^4/(4 Pi^2)+mu^2 t^2/2;
n=mu^3/Pi^2+mu t^2;
chi=D[n,mu];
pt=FullSimplify[D[p,t]-D[p,mu]*D[n,t]/chi];
coldPt=Limit[pt/t,t->0];
coldPa=FullSimplify[nB*((D[p,mu]/chi/.{mu->mud,t->0})
                     -(D[p,mu]/chi/.{mu->mus,t->0}))];
endpoint=FullSimplify[
  Limit[cD*(tinf*z^3)^(-5/3)*gamma0/nB
    *((tinf*z^3)*(muu^2+mud^2+mus^2)/(nB*(mud-mus)))
    *(3*tinf*z^2),z->0,Direction->"FromAbove"],
  Assumptions->{tinf>0,nB>0,mud>mus>0}];
pressure=Sum[q^4/(4 Pi^2)+q^2*t^2/2,{q,{u,d,s}}]
         +19 Pi^2*t^4/36-bag;
energy=-pressure+t*D[pressure,t]
       +Sum[q*D[pressure,q],{q,{u,d,s}}];
{pt,coldPt,coldPa,endpoint,
 FullSimplify[energy-3*pressure-4*bag],2^(-1)*2^8/2^3}
```

Returned result:

```text
{
  mu^2*t*(-1 + 4*mu^2/(3*mu^2 + Pi^2*t^2)),
  mu^2/3,
  (mud-mus)*nB/3,
  3*cD*gamma0*(mud^2+mus^2+muu^2)*tinf^(1/3)/((mud-mus)*nB^2),
  0,
  16
}
```

Thus the cold pressure derivative at fixed flavor densities is `P_T/T -> sum(mu_f^2)/3`, while `P_a -> nB*(mu_d-mu_s)/3`.
With `cD=pi*qD^(2/3)/(24*alpha_s^2*H_D)`, the fourth result is the exact finite limit of the transformed integrand at `z=0`, matching `cold_endpoint_integrand`.
The conformal residual is zero: `epsilon=3*P+4*bag`, so fixed pressure and fixed `h/nB` imply fixed baryon density for a constant bag pressure.
Under a common energy rescaling by two at fixed composition and coupling, diffusion scales by `2^(-1)`, rate by `2^8`, and baryon density by `2^3`; the integral scales by 16.
The weak prefactor is held fixed in this numerical homogeneity statement.

The temperature-dependent gluon contribution vanishes from the leading cold derivative but is retained in the full trajectory.
These identities establish the formal integral regularization, not the validity of a local diffusion description at exactly zero temperature.

The complementary numerical checks are reproduced by `adiabatic_verify.py` and saved in `results/adiabatic-independent-verification.json`.
Its independent coordinate is `a=a_0plus*(1-y^6)`, with `T(a)` obtained from a separately implemented 40-digit EOS and a 100-step temperature bisection.
The production Jacobian is not used in that integral.
The rate and diffusion routines are shared deliberately, so this is an independent EOS/coordinate check rather than an independent phase-space calculation.
The driver also records the 3,025-state cold-roundoff regression sweep, decreasing-temperature-cutoff integrals with and without their leading cold tail, EOS iteration refinement, and separate outer-order, phase-order, and tail-length comparisons from the production checkpoint.
The completed check covers all 45 composition targets and 143 quadrature records, with outer-order refinement at every composition and phase-order/tail checks at four representative compositions.
The compact machine-readable and manuscript summaries are `results/adiabatic-convergence.csv` and `results/adiabatic-verification.tex`.
After adding production compositions, `adiabatic_verify.py --refresh-production` refreshes these comparisons without repeating the expensive independent integrations, provided the physics-source and numerical-recipe signatures still agree.

## Audit of the repeated temperature panels

The additional symbolic check on 2026-10-11 evaluated the ratio of the two-term diffusion coefficients under a common positive energy rescaling.
The overall transport prefactor cancels; `c` is the positive first-term coefficient and `q` denotes the Debye momentum.
`PowerExpand` is legitimate here because `lam,q,t,c` are positive real quantities.

```wolfram
Clear[lam,q,t,c];
Simplify[PowerExpand[
 (c*t^(5/3)/q^(2/3)+Pi^3*t^2/(12*q)) /
 (c*(lam*t)^(5/3)/(lam*q)^(2/3)+Pi^3*(lam*t)^2/(12*lam*q))
]]
```

Returned `lam^(-1)`.
An earlier `FullSimplify` call left this ratio unevaluated; it did return density scaling `lambda^3`, integral scaling `lambda^4`, and cancellation of the common integral factor in the speed ratio.

The separate driver `audit_adiabatic_physical.py` starts from each saved upstream pressure and enthalpy per baryon, recomputes quark density, and independently solves the dimensional cold interface and finite-temperature EOS in 40-digit arithmetic.
It integrates all three microscopic rates in `a=a_0plus*(1-y^6)` at upstream temperatures 0, 20 and 100 MeV and interface compositions 0.9 and 0.995.
Neither the production trajectory class nor cached rate integrals are used in this reconstruction and quadrature; the rate and diffusion functions themselves remain shared.
The output `results/adiabatic-physical-audit.json` retains physical states, independently computed integrals, comparisons to the published values, and the spread across upstream temperatures.
